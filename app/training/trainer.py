import gc
import os
from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

import torch
from transformers import AutoTokenizer

from app.conf.model import MODEL_CFG, get_device
from app.core.architecture import build_model
from app.core.transformer import cross_entropy_loss_and_accuracy
from app.training.config import TrainingSettings
from app.training.dataset import create_chat_dataloader, dataset_summary
from app.training.lora import (
    apply_lora_to_model,
    count_lora_parameters,
    describe_lora_layers,
    export_lora_adapter,
    lora_parameters,
    merge_lora_weights,
)
from app.training.mlflow_registry import MLflowRegistry
from app.training.minio_storage import MinioStorage
from app.training.tokenizer_utils import ensure_chat_template


@dataclass
class LoRATrainConfig:
    dataset_path: str
    run_name: str = "lora-posttrain"
    job_id: str | None = None
    epochs: int = 3
    batch_size: int = 2
    learning_rate: float = 2e-4
    max_seq_len: int = 512
    lora_rank: int = 8
    lora_alpha: float = 16.0
    gradient_accumulation_steps: int = 1
    log_every_steps: int = 5
    seed: int = 42


@dataclass
class LoRATrainResult:
    run_id: str
    model_version: str
    registry_name: str
    minio_model_uri: str
    minio_adapter_uri: str | None
    final_loss: float
    trainable_parameters: int
    lora_layers: list[str]
    epochs_completed: int
    dataset: dict[str, Any]
    output_model_path: str
    output_adapter_path: str


class TrainingCancelledError(Exception):
    """Raised when a training job is stopped cooperatively."""

    def __init__(self, partial_result: dict[str, Any]) -> None:
        super().__init__("training cancelled")
        self.partial_result = partial_result


class LoRATrainer:
    def __init__(
        self,
        settings: TrainingSettings | None = None,
        registry: MLflowRegistry | None = None,
        storage: MinioStorage | None = None,
    ) -> None:
        self.settings = settings or TrainingSettings.from_env()
        self.storage = storage or MinioStorage(self.settings)
        self.registry = registry or MLflowRegistry(self.settings, self.storage)

    def train(
        self,
        config: LoRATrainConfig,
        *,
        progress_callback: Callable[[dict[str, Any]], None] | None = None,
        should_stop: Callable[[], bool] | None = None,
    ) -> LoRATrainResult:
        def _stop_requested() -> bool:
            return should_stop is not None and should_stop()

        torch.manual_seed(config.seed)
        device = get_device(self.settings.device)
        dataset_path = Path(config.dataset_path)
        if not dataset_path.exists():
            raise FileNotFoundError(f"dataset not found: {dataset_path}")

        summary = dataset_summary(dataset_path)
        hf_token = (os.environ.get("HF_TOKEN") or "").strip() or None
        tokenizer = AutoTokenizer.from_pretrained(
            self.settings.tokenizer_name,
            token=hf_token,
        )
        ensure_chat_template(tokenizer, tokenizer_name=self.settings.tokenizer_name)
        if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
            tokenizer.pad_token = tokenizer.eos_token

        model = build_model(
            device=device,
            checkpoint_path=self.settings.base_checkpoint_path,
        )
        for param in model.parameters():
            param.requires_grad_(False)
        apply_lora_to_model(
            model,
            rank=config.lora_rank,
            alpha=config.lora_alpha,
        )
        trainable = count_lora_parameters(model)
        lora_layers = describe_lora_layers(model)
        model.train()

        dataloader = create_chat_dataloader(
            dataset_path,
            tokenizer,
            max_seq_len=config.max_seq_len,
            batch_size=config.batch_size,
            device=device,
            shuffle=True,
        )

        optimizer = torch.optim.AdamW(
            lora_parameters(model),
            lr=config.learning_rate,
        )

        params = {
            **asdict(config),
            "trainable_parameters": trainable,
            "lora_layers": len(lora_layers),
            "base_checkpoint": self.settings.base_checkpoint_path,
        }

        output_dir = Path(self.settings.training_jobs_dir) / (
            config.job_id or config.run_name
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        adapter_path = output_dir / "lora_adapter.pt"
        merged_path = output_dir / "model.pt"
        checkpoints: list[dict[str, Any]] = []

        final_loss = 0.0
        global_step = 0
        epochs_completed = 0
        try:
            with self.registry.start_run(
                run_name=config.run_name,
                tags={"pipeline": "lora_posttrain"},
            ) as run:
                run_id = run.info.run_id
                self.registry.log_params(params)
                self.registry.log_metrics(
                    {"dataset_examples": float(summary["examples"])}
                )

                for epoch in range(config.epochs):
                    if _stop_requested():
                        if checkpoints:
                            latest = checkpoints[-1]
                            raise TrainingCancelledError(
                                self._build_partial_result(
                                    run_id=run_id,
                                    final_loss=final_loss,
                                    epochs_completed=epochs_completed,
                                    checkpoints=checkpoints,
                                    latest=latest,
                                    trainable=trainable,
                                    lora_layers=lora_layers,
                                    summary=summary,
                                )
                            )
                        raise TrainingCancelledError(
                            {
                                "run_id": run_id,
                                "epochs_completed": 0,
                                "checkpoints": [],
                                "cancelled": True,
                            }
                        )

                    epoch_loss = 0.0
                    epoch_steps = 0
                    optimizer.zero_grad(set_to_none=True)

                    for step_idx, batch in enumerate(dataloader, start=1):
                        if _stop_requested():
                            break

                        state = [None] * MODEL_CFG.num_hidden_layers
                        output = model(state=state, seq=batch)
                        loss, _ = cross_entropy_loss_and_accuracy(
                            output.logits,
                            batch.target_tokens,
                            batch.loss_masks,
                        )
                        scaled_loss = loss / config.gradient_accumulation_steps
                        scaled_loss.backward()
                        epoch_loss += float(loss.detach().item())
                        epoch_steps += 1
                        global_step += 1

                        if step_idx % config.gradient_accumulation_steps == 0:
                            optimizer.step()
                            optimizer.zero_grad(set_to_none=True)

                        if global_step % config.log_every_steps == 0:
                            metric_payload = {
                                "train_loss": float(loss.detach().item()),
                                "epoch": float(epoch + 1),
                            }
                            self.registry.log_metrics(metric_payload, step=global_step)
                            if progress_callback is not None:
                                progress_callback(
                                    {
                                        "epoch": epoch + 1,
                                        "step": global_step,
                                        "loss": float(loss.detach().item()),
                                        "checkpoints": checkpoints,
                                    }
                                )

                    if _stop_requested() and epoch_steps == 0:
                        if checkpoints:
                            latest = checkpoints[-1]
                            raise TrainingCancelledError(
                                self._build_partial_result(
                                    run_id=run_id,
                                    final_loss=final_loss,
                                    epochs_completed=epochs_completed,
                                    checkpoints=checkpoints,
                                    latest=latest,
                                    trainable=trainable,
                                    lora_layers=lora_layers,
                                    summary=summary,
                                )
                            )
                        raise TrainingCancelledError(
                            {
                                "run_id": run_id,
                                "epochs_completed": 0,
                                "checkpoints": [],
                                "cancelled": True,
                            }
                        )

                    if _stop_requested() and epoch_steps > 0:
                        if checkpoints:
                            latest = checkpoints[-1]
                            raise TrainingCancelledError(
                                self._build_partial_result(
                                    run_id=run_id,
                                    final_loss=final_loss,
                                    epochs_completed=epochs_completed,
                                    checkpoints=checkpoints,
                                    latest=latest,
                                    trainable=trainable,
                                    lora_layers=lora_layers,
                                    summary=summary,
                                )
                            )
                        raise TrainingCancelledError(
                            {
                                "run_id": run_id,
                                "epochs_completed": 0,
                                "checkpoints": [],
                                "cancelled": True,
                            }
                        )

                    final_loss = epoch_loss / max(epoch_steps, 1)
                    epochs_completed = epoch + 1
                    self.registry.log_metrics(
                        {"epoch_loss": final_loss, "epoch": float(epoch + 1)},
                        step=global_step,
                    )

                    checkpoint = self._save_epoch_checkpoint(
                        model=model,
                        output_dir=output_dir,
                        epoch=epoch,
                        config=config,
                        run_id=run_id,
                    )
                    checkpoints.append(checkpoint)
                    adapter_path = Path(checkpoint["adapter_path"])
                    merged_path = Path(checkpoint["model_path"])

                    if progress_callback is not None:
                        progress_callback(
                            {
                                "epoch": epoch + 1,
                                "step": global_step,
                                "epoch_loss": final_loss,
                                "checkpoints": checkpoints,
                                "latest_checkpoint": checkpoint,
                            }
                        )

                    if _stop_requested():
                        raise TrainingCancelledError(
                            self._build_partial_result(
                                run_id=run_id,
                                final_loss=final_loss,
                                epochs_completed=epochs_completed,
                                checkpoints=checkpoints,
                                latest=checkpoint,
                                trainable=trainable,
                                lora_layers=lora_layers,
                                summary=summary,
                            )
                        )

                adapter_payload = export_lora_adapter(model)
                torch.save(adapter_payload, adapter_path)

                merged_model = deepcopy(model)
                merge_lora_weights(merged_model)
                merged_model.eval()
                torch.save({"model_weights": merged_model.state_dict()}, merged_path)
                del merged_model

                publish_info = self.registry.save_and_log_model_bundle(
                    run_id=run_id,
                    model_path=merged_path,
                    adapter_path=adapter_path,
                    params=params,
                    final_metrics={"final_loss": final_loss},
                )

            return LoRATrainResult(
                run_id=run_id,
                model_version=publish_info["model_version"],
                registry_name=publish_info["registry_name"],
                minio_model_uri=publish_info["minio_model_uri"],
                minio_adapter_uri=publish_info.get("minio_adapter_uri"),
                final_loss=final_loss,
                trainable_parameters=trainable,
                lora_layers=lora_layers,
                epochs_completed=epochs_completed,
                dataset=summary,
                output_model_path=str(merged_path),
                output_adapter_path=str(adapter_path),
            )
        finally:
            del model, optimizer, dataloader
            gc.collect()
            if device.type == "cuda":
                torch.cuda.synchronize()
                torch.cuda.empty_cache()

    def _build_partial_result(
        self,
        *,
        run_id: str,
        final_loss: float,
        epochs_completed: int,
        checkpoints: list[dict[str, Any]],
        latest: dict[str, Any],
        trainable: int,
        lora_layers: list[str],
        summary: dict[str, Any],
    ) -> dict[str, Any]:
        return {
            "run_id": run_id,
            "final_loss": final_loss,
            "epochs_completed": epochs_completed,
            "trainable_parameters": trainable,
            "lora_layers": lora_layers,
            "dataset": summary,
            "checkpoints": checkpoints,
            "output_model_path": latest["model_path"],
            "output_adapter_path": latest["adapter_path"],
            "minio_model_uri": latest.get("minio_model_uri"),
            "minio_adapter_uri": latest.get("minio_adapter_uri"),
            "cancelled": True,
        }

    def _save_epoch_checkpoint(
        self,
        *,
        model: torch.nn.Module,
        output_dir: Path,
        epoch: int,
        config: LoRATrainConfig,
        run_id: str,
    ) -> dict[str, Any]:
        epoch_num = epoch + 1
        epoch_dir = output_dir / f"epoch_{epoch_num}"
        epoch_dir.mkdir(parents=True, exist_ok=True)
        epoch_adapter_path = epoch_dir / "lora_adapter.pt"
        epoch_model_path = epoch_dir / "model.pt"

        adapter_payload = export_lora_adapter(model)
        torch.save(adapter_payload, epoch_adapter_path)

        merged_model = deepcopy(model)
        merge_lora_weights(merged_model)
        merged_model.eval()
        torch.save({"model_weights": merged_model.state_dict()}, epoch_model_path)

        latest_adapter_path = output_dir / "lora_adapter.pt"
        latest_model_path = output_dir / "model.pt"
        torch.save(adapter_payload, latest_adapter_path)
        torch.save(
            {"model_weights": merged_model.state_dict()},
            latest_model_path,
        )

        checkpoint_key = config.job_id or config.run_name
        minio_prefix = f"checkpoints/{checkpoint_key}/epoch_{epoch_num}"
        self.storage.ensure_buckets()
        minio_adapter_uri = self.storage.upload_file(
            self.settings.minio_bucket_models,
            f"{minio_prefix}/lora_adapter.pt",
            epoch_adapter_path,
            content_type="application/octet-stream",
        )
        minio_model_uri = self.storage.upload_file(
            self.settings.minio_bucket_models,
            f"{minio_prefix}/model.pt",
            epoch_model_path,
            content_type="application/octet-stream",
        )

        self.registry.log_artifact(epoch_adapter_path)
        self.registry.log_artifact(epoch_model_path)

        return {
            "epoch": epoch_num,
            "adapter_path": str(epoch_adapter_path),
            "model_path": str(epoch_model_path),
            "latest_adapter_path": str(latest_adapter_path),
            "latest_model_path": str(latest_model_path),
            "minio_adapter_uri": minio_adapter_uri,
            "minio_model_uri": minio_model_uri,
            "run_id": run_id,
        }
