import json
from pathlib import Path

import pytest

from app.training.inference_model import (
    get_inference_model_status,
    read_deploy_manifest,
    register_inference_startup,
    write_deploy_manifest,
)


def test_write_and_read_deploy_manifest(tmp_path: Path) -> None:
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"weights")

    payload = write_deploy_manifest(
        checkpoint_path=checkpoint,
        registry_name="llm-lora",
        version="3",
        object_name="models/llm-lora/3/model.pt",
    )

    assert payload["version"] == "3"
    manifest = read_deploy_manifest(checkpoint)
    assert manifest is not None
    assert manifest["registry_name"] == "llm-lora"
    assert manifest["checkpoint"]["size_bytes"] == len(b"weights")


def test_inference_status_loaded_after_startup(tmp_path: Path) -> None:
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"weights-v1")

    write_deploy_manifest(
        checkpoint_path=checkpoint,
        registry_name="llm-lora",
        version="7",
        object_name="models/llm-lora/7/model.pt",
    )
    register_inference_startup(
        checkpoint_path=checkpoint,
        model_revision="model.pt",
    )

    status = get_inference_model_status(
        checkpoint_path=checkpoint,
        registry_name="llm-lora",
    )
    assert status["status"] == "loaded"
    assert status["loaded_version"] == "7"
    assert status["pending_reload"] is False


def test_inference_status_reload_required(tmp_path: Path) -> None:
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"weights-v1")

    write_deploy_manifest(
        checkpoint_path=checkpoint,
        registry_name="llm-lora",
        version="1",
        object_name="models/llm-lora/1/model.pt",
    )
    register_inference_startup(
        checkpoint_path=checkpoint,
        model_revision="model.pt",
    )

    checkpoint.write_bytes(b"weights-v2-updated")
    write_deploy_manifest(
        checkpoint_path=checkpoint,
        registry_name="llm-lora",
        version="2",
        object_name="models/llm-lora/2/model.pt",
    )

    status = get_inference_model_status(
        checkpoint_path=checkpoint,
        registry_name="llm-lora",
    )
    assert status["status"] == "reload_required"
    assert status["loaded_version"] == "1"
    assert status["deployed_version"] == "2"
    assert status["pending_reload"] is True
