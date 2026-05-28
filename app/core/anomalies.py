"""Heuristic anomaly flags for logged chat interactions."""

TOXICITY_WARN = 0.7
TOXICITY_CRITICAL = 0.85


def detect_anomaly_flags(
    *,
    prompt: str,
    response: str,
    prompt_lang: str,
    response_lang: str,
    toxicity: float,
    json_valid: bool,
    status: str,
    user_rating: int | None = None,
) -> list[str]:
    flags: list[str] = []
    prompt_stripped = prompt.strip()
    response_stripped = response.strip()

    if status != "success":
        flags.append("error_response")
    if not response_stripped:
        flags.append("empty_response")
    if toxicity >= TOXICITY_CRITICAL:
        flags.append("toxicity_critical")
    elif toxicity >= TOXICITY_WARN:
        flags.append("high_toxicity")
    if (
        prompt_lang not in {"", "unknown"}
        and response_lang not in {"", "unknown"}
        and prompt_lang != response_lang
    ):
        flags.append("language_mismatch")
    if not json_valid and ("{" in response_stripped or "[" in response_stripped):
        flags.append("invalid_json")
    if user_rating is not None and user_rating <= 2:
        flags.append("low_rating")
    if len(response_stripped) > 2500:
        flags.append("very_long_response")
    if len(prompt_stripped) < 3:
        flags.append("very_short_prompt")

    return flags
