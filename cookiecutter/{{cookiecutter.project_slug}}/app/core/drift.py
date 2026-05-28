"""Online drift detection: data, concept, and target drift."""

import io
import math
import os
import time
from collections import Counter, deque
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from app.core.embeddings import PromptEmbedder

PSI_WARN = 0.2
PSI_CRITICAL = 0.5


def toxicity_tier(score: float) -> str:
    if score < 0.3:
        return "low"
    if score < 0.7:
        return "medium"
    return "high"


def length_bucket(length: int) -> str:
    if length < 50:
        return "short"
    if length < 150:
        return "medium"
    if length < 500:
        return "long"
    return "very_long"


def distribution(counter: Counter[str]) -> dict[str, float]:
    total = sum(counter.values())
    if total == 0:
        return {}
    return {key: value / total for key, value in counter.items()}


def population_stability_index(
    expected: dict[str, float],
    actual: dict[str, float],
    *,
    eps: float = 1e-6,
) -> float:
    keys = set(expected) | set(actual)
    if not keys:
        return 0.0
    score = 0.0
    for key in keys:
        baseline_ratio = max(expected.get(key, 0.0), eps)
        current_ratio = max(actual.get(key, 0.0), eps)
        score += (current_ratio - baseline_ratio) * math.log(
            current_ratio / baseline_ratio
        )
    return max(0.0, float(score))


def centroid(vectors: list[np.ndarray]) -> np.ndarray | None:
    if not vectors:
        return None
    return np.mean(np.stack(vectors, axis=0), axis=0)


def cosine_distance(left: np.ndarray, right: np.ndarray) -> float:
    denom = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denom <= 1e-9:
        return 0.0
    cosine = float(np.dot(left, right) / denom)
    return max(0.0, 1.0 - cosine)


@dataclass
class DriftSnapshot:
    data_embedding_distance: float = 0.0
    data_prompt_length_psi: float = 0.0
    data_language_psi: float = 0.0
    data_drift_score: float = 0.0
    concept_response_embedding_distance: float = 0.0
    concept_toxicity_psi: float = 0.0
    concept_response_length_psi: float = 0.0
    concept_drift_score: float = 0.0
    target_toxicity_tier_psi: float = 0.0
    target_json_valid_psi: float = 0.0
    target_response_language_psi: float = 0.0
    target_user_rating_psi: float = 0.0
    target_drift_score: float = 0.0
    baseline_samples: int = 0
    window_samples: int = 0
    baseline_locked: float = 0.0


@dataclass
class _DriftWindow:
    prompt_embeddings: deque[np.ndarray] = field(default_factory=deque)
    response_embeddings: deque[np.ndarray] = field(default_factory=deque)
    prompt_lengths: Counter[str] = field(default_factory=Counter)
    response_lengths: Counter[str] = field(default_factory=Counter)
    prompt_languages: Counter[str] = field(default_factory=Counter)
    response_languages: Counter[str] = field(default_factory=Counter)
    toxicity_tiers: Counter[str] = field(default_factory=Counter)
    json_validity: Counter[str] = field(default_factory=Counter)
    user_ratings: Counter[str] = field(default_factory=Counter)
    toxicity_values: deque[float] = field(default_factory=deque)
    response_length_values: deque[int] = field(default_factory=deque)

    def add(
        self,
        *,
        prompt_embedding: np.ndarray,
        response_embedding: np.ndarray,
        prompt_length: int,
        response_length: int,
        prompt_lang: str,
        response_lang: str,
        toxicity: float,
        json_valid: bool,
    ) -> None:
        self.prompt_embeddings.append(prompt_embedding)
        self.response_embeddings.append(response_embedding)
        self.prompt_lengths[length_bucket(prompt_length)] += 1
        self.response_lengths[length_bucket(response_length)] += 1
        self.prompt_languages[prompt_lang] += 1
        self.response_languages[response_lang] += 1
        self.toxicity_tiers[toxicity_tier(toxicity)] += 1
        self.json_validity["valid" if json_valid else "invalid"] += 1
        self.toxicity_values.append(toxicity)
        self.response_length_values.append(response_length)


class DriftMonitor:
    """
    Online drift monitor with frozen baseline + rolling current window.

    Data drift:
      prompt embedding centroid shift + PSI on prompt length/language.

    Concept drift (recommended for TTT LLM):
      response embedding centroid shift while prompt side is tracked separately.
      If prompts are stable but responses move in embedding space, P(Y|X) changed.
      Also PSI on toxicity and response length as output-characteristic drift.

    Target drift:
      PSI on user ratings (1-5) when present, otherwise proxy labels.
    """

    def __init__(
        self,
        embedder: PromptEmbedder,
        *,
        redis_client: Any | None = None,
        baseline_size: int | None = None,
        window_size: int | None = None,
        prefix: str = "drift:state",
    ) -> None:
        self.embedder = embedder
        self.redis_client = redis_client
        self.baseline_size = baseline_size or int(
            os.environ.get("DRIFT_BASELINE_SIZE", "200")
        )
        self.window_size = window_size or int(
            os.environ.get("DRIFT_WINDOW_SIZE", "100")
        )
        self.prefix = prefix
        self.state_version = 2
        self.baseline = _DriftWindow()
        self.current = _DriftWindow(
            prompt_embeddings=deque(maxlen=self.window_size),
            response_embeddings=deque(maxlen=self.window_size),
            toxicity_values=deque(maxlen=self.window_size),
            response_length_values=deque(maxlen=self.window_size),
        )
        self._baseline_locked = False
        self.snapshot = DriftSnapshot()
        self._load_state()

    def observe(
        self,
        *,
        prompt_text: str,
        response_text: str,
        prompt_lang: str,
        response_lang: str,
        toxicity: float,
        json_valid: bool,
    ) -> DriftSnapshot:
        prompt_embedding = self.embedder.embed(prompt_text)
        response_embedding = self.embedder.embed(response_text)
        prompt_length = len(prompt_text)
        response_length = len(response_text)

        sample_kwargs = {
            "prompt_embedding": prompt_embedding,
            "response_embedding": response_embedding,
            "prompt_length": prompt_length,
            "response_length": response_length,
            "prompt_lang": prompt_lang,
            "response_lang": response_lang,
            "toxicity": toxicity,
            "json_valid": json_valid,
        }

        if not self._baseline_locked:
            self.baseline.add(**sample_kwargs)
            if len(self.baseline.prompt_embeddings) >= self.baseline_size:
                self._baseline_locked = True

        self.current.add(**sample_kwargs)
        self.snapshot = self._compute_snapshot()
        self._save_state()
        return self.snapshot

    def record_user_rating(
        self, rating: int, *, include_baseline: bool
    ) -> DriftSnapshot:
        label = str(rating)
        self.current.user_ratings[label] += 1
        if include_baseline:
            self.baseline.user_ratings[label] += 1
        self.snapshot = self._compute_snapshot()
        self._save_state()
        return self.snapshot

    def baseline_is_open(self) -> bool:
        return not self._baseline_locked

    def _compute_snapshot(self) -> DriftSnapshot:
        baseline_count = len(self.baseline.prompt_embeddings)
        window_count = len(self.current.prompt_embeddings)
        if baseline_count == 0 or window_count == 0:
            return DriftSnapshot(
                baseline_samples=baseline_count,
                window_samples=window_count,
                baseline_locked=1.0 if self._baseline_locked else 0.0,
            )

        baseline_prompt_centroid = centroid(list(self.baseline.prompt_embeddings))
        current_prompt_centroid = centroid(list(self.current.prompt_embeddings))
        baseline_response_centroid = centroid(list(self.baseline.response_embeddings))
        current_response_centroid = centroid(list(self.current.response_embeddings))

        data_embedding_distance = 0.0
        concept_response_embedding_distance = 0.0
        if baseline_prompt_centroid is not None and current_prompt_centroid is not None:
            data_embedding_distance = cosine_distance(
                baseline_prompt_centroid,
                current_prompt_centroid,
            )
        if (
            baseline_response_centroid is not None
            and current_response_centroid is not None
        ):
            concept_response_embedding_distance = cosine_distance(
                baseline_response_centroid,
                current_response_centroid,
            )

        data_prompt_length_psi = population_stability_index(
            distribution(self.baseline.prompt_lengths),
            distribution(self.current.prompt_lengths),
        )
        data_language_psi = population_stability_index(
            distribution(self.baseline.prompt_languages),
            distribution(self.current.prompt_languages),
        )
        concept_toxicity_psi = population_stability_index(
            distribution(self.baseline.toxicity_tiers),
            distribution(self.current.toxicity_tiers),
        )
        concept_response_length_psi = population_stability_index(
            distribution(self.baseline.response_lengths),
            distribution(self.current.response_lengths),
        )
        target_toxicity_tier_psi = concept_toxicity_psi
        target_json_valid_psi = population_stability_index(
            distribution(self.baseline.json_validity),
            distribution(self.current.json_validity),
        )
        target_response_language_psi = population_stability_index(
            distribution(self.baseline.response_languages),
            distribution(self.current.response_languages),
        )
        baseline_rating_total = sum(self.baseline.user_ratings.values())
        current_rating_total = sum(self.current.user_ratings.values())
        has_user_ratings = baseline_rating_total > 0 and current_rating_total > 0
        target_user_rating_psi = (
            population_stability_index(
                distribution(self.baseline.user_ratings),
                distribution(self.current.user_ratings),
            )
            if has_user_ratings
            else 0.0
        )

        data_drift_score = self._normalize_data_score(
            data_embedding_distance,
            data_prompt_length_psi,
            data_language_psi,
        )
        concept_drift_score = self._normalize_concept_score(
            concept_response_embedding_distance,
            concept_toxicity_psi,
            concept_response_length_psi,
        )
        target_drift_score = self._normalize_target_score(
            target_user_rating_psi,
            target_toxicity_tier_psi,
            target_json_valid_psi,
            target_response_language_psi,
            has_user_ratings=has_user_ratings,
        )

        return DriftSnapshot(
            data_embedding_distance=data_embedding_distance,
            data_prompt_length_psi=data_prompt_length_psi,
            data_language_psi=data_language_psi,
            data_drift_score=data_drift_score,
            concept_response_embedding_distance=concept_response_embedding_distance,
            concept_toxicity_psi=concept_toxicity_psi,
            concept_response_length_psi=concept_response_length_psi,
            concept_drift_score=concept_drift_score,
            target_toxicity_tier_psi=target_toxicity_tier_psi,
            target_json_valid_psi=target_json_valid_psi,
            target_response_language_psi=target_response_language_psi,
            target_user_rating_psi=target_user_rating_psi,
            target_drift_score=target_drift_score,
            baseline_samples=baseline_count,
            window_samples=window_count,
            baseline_locked=1.0 if self._baseline_locked else 0.0,
        )

    @staticmethod
    def _normalize_data_score(
        embedding_distance: float,
        length_psi: float,
        language_psi: float,
    ) -> float:
        embedding_component = min(1.0, embedding_distance / 0.25)
        psi_component = min(1.0, max(length_psi, language_psi) / PSI_CRITICAL)
        return min(1.0, 0.6 * embedding_component + 0.4 * psi_component)

    @staticmethod
    def _normalize_concept_score(
        response_embedding_distance: float,
        toxicity_psi: float,
        response_length_psi: float,
    ) -> float:
        embedding_component = min(1.0, response_embedding_distance / 0.25)
        psi_component = min(
            1.0,
            max(toxicity_psi, response_length_psi) / PSI_CRITICAL,
        )
        return min(1.0, 0.55 * embedding_component + 0.45 * psi_component)

    @staticmethod
    def _normalize_target_score(
        user_rating_psi: float,
        toxicity_psi: float,
        json_valid_psi: float,
        response_language_psi: float,
        *,
        has_user_ratings: bool,
    ) -> float:
        if has_user_ratings:
            return min(1.0, user_rating_psi / PSI_CRITICAL)
        return min(
            1.0,
            max(toxicity_psi, json_valid_psi, response_language_psi) / PSI_CRITICAL,
        )

    def _state_key(self) -> str:
        return f"{self.prefix}:v{self.state_version}"

    def _save_state(self) -> None:
        if self.redis_client is None:
            return
        payload = {
            "baseline_locked": self._baseline_locked,
            "saved_at": time.time(),
            "baseline": self._window_to_dict(self.baseline),
            "current": self._window_to_dict(self.current),
        }
        buffer = io.BytesIO()
        np.savez_compressed(
            buffer,
            payload=np.array([payload], dtype=object),
        )
        self.redis_client.set(self._state_key(), buffer.getvalue())

    def _load_state(self) -> None:
        if self.redis_client is None:
            return
        blob = self.redis_client.get(self._state_key())
        if blob is None:
            return
        try:
            loaded = np.load(io.BytesIO(blob), allow_pickle=True)
            payload = loaded["payload"].item()
            self._baseline_locked = bool(payload.get("baseline_locked", False))
            self.baseline = self._dict_to_window(payload.get("baseline", {}))
            current = self._dict_to_window(payload.get("current", {}))
            self.current = self._trim_window(current)
            self.snapshot = self._compute_snapshot()
        except Exception:  # pylint: disable=broad-except
            return

    def _trim_window(self, window: _DriftWindow) -> _DriftWindow:
        window.prompt_embeddings = deque(
            list(window.prompt_embeddings)[-self.window_size :],
            maxlen=self.window_size,
        )
        window.response_embeddings = deque(
            list(window.response_embeddings)[-self.window_size :],
            maxlen=self.window_size,
        )
        window.toxicity_values = deque(
            list(window.toxicity_values)[-self.window_size :],
            maxlen=self.window_size,
        )
        window.response_length_values = deque(
            list(window.response_length_values)[-self.window_size :],
            maxlen=self.window_size,
        )
        return window

    @staticmethod
    def _window_to_dict(window: _DriftWindow) -> dict[str, Any]:
        return {
            "prompt_embeddings": [vec.tolist() for vec in window.prompt_embeddings],
            "response_embeddings": [vec.tolist() for vec in window.response_embeddings],
            "prompt_lengths": dict(window.prompt_lengths),
            "response_lengths": dict(window.response_lengths),
            "prompt_languages": dict(window.prompt_languages),
            "response_languages": dict(window.response_languages),
            "toxicity_tiers": dict(window.toxicity_tiers),
            "json_validity": dict(window.json_validity),
            "user_ratings": dict(window.user_ratings),
            "toxicity_values": list(window.toxicity_values),
            "response_length_values": list(window.response_length_values),
        }

    def _dict_to_window(self, payload: dict[str, Any]) -> _DriftWindow:
        window = _DriftWindow()
        window.prompt_embeddings = deque(
            np.asarray(row, dtype=np.float32)
            for row in payload.get("prompt_embeddings", [])
        )
        window.response_embeddings = deque(
            np.asarray(row, dtype=np.float32)
            for row in payload.get("response_embeddings", [])
        )
        window.prompt_lengths = Counter(payload.get("prompt_lengths", {}))
        window.response_lengths = Counter(payload.get("response_lengths", {}))
        window.prompt_languages = Counter(payload.get("prompt_languages", {}))
        window.response_languages = Counter(payload.get("response_languages", {}))
        window.toxicity_tiers = Counter(payload.get("toxicity_tiers", {}))
        window.json_validity = Counter(payload.get("json_validity", {}))
        window.user_ratings = Counter(payload.get("user_ratings", {}))
        window.toxicity_values = deque(payload.get("toxicity_values", []))
        window.response_length_values = deque(payload.get("response_length_values", []))
        return window
