"""Multilingual toxicity scoring via HuggingFace regression model."""

import os

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

DEFAULT_TOXICITY_MODEL = "unitary/multilingual-toxic-xlm-roberta"


class ToxicityScorer:
    """
    XLM-R regression head (Detoxify multilingual).

    Returns max sigmoid probability across toxicity dimensions in [0, 1].
    """

    def __init__(
        self,
        model_name: str | None = None,
        device: str | None = None,
        *,
        hf_token: str | None = None,
        max_length: int = 512,
    ) -> None:
        self.model_name = model_name or os.environ.get(
            "TOXICITY_MODEL_NAME", DEFAULT_TOXICITY_MODEL
        )
        device_str = device or os.environ.get("TOXICITY_DEVICE", "cpu")
        self.device = torch.device(device_str)
        self.max_length = max_length
        token = (hf_token or os.environ.get("HF_TOKEN") or "").strip() or None

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, token=token)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            token=token,
            use_safetensors=True,
        )
        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def score_many(self, texts: list[str]) -> float:
        cleaned = [text.strip() for text in texts if text and text.strip()]
        if not cleaned:
            return 0.0

        inputs = self.tokenizer(
            cleaned,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
            padding=True,
        )
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        logits = self.model(**inputs).logits
        probabilities = torch.sigmoid(logits)
        return float(probabilities.max().item())

    @torch.no_grad()
    def score(self, text: str) -> float:
        return self.score_many([text])

    def score_turn(self, *, prompt_text: str, response_text: str) -> float:
        return self.score_many([prompt_text, response_text])
