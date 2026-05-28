"""Tokenizer helpers for chat-style LoRA training."""

from transformers import PreTrainedTokenizerBase

# Meta-Llama-3 default (from tokenizer_config.json on Hugging Face).
LLAMA3_CHAT_TEMPLATE = "{% set loop_messages = messages %}{% for message in loop_messages %}{% set content = '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n'+ message['content'] | trim + '<|eot_id|>' %}{% if loop.index0 == 0 %}{% set content = bos_token + content %}{% endif %}{{ content }}{% endfor %}{% if add_generation_prompt %}{{ '<|start_header_id|>assistant<|end_header_id|>\n\n' }}{% endif %}"


def has_chat_template(tokenizer: PreTrainedTokenizerBase) -> bool:
    template = getattr(tokenizer, "chat_template", None)
    if isinstance(template, str) and template.strip():
        return True
    getter = getattr(tokenizer, "get_chat_template", None)
    if callable(getter):
        try:
            return bool(getter())
        except Exception:  # pylint: disable=broad-except
            return False
    return False


def _looks_like_llama3(tokenizer_name: str) -> bool:
    normalized = tokenizer_name.lower()
    return "llama" in normalized and ("-3" in normalized or "llama3" in normalized)


def ensure_chat_template(
    tokenizer: PreTrainedTokenizerBase,
    *,
    tokenizer_name: str | None = None,
) -> PreTrainedTokenizerBase:
    if has_chat_template(tokenizer):
        return tokenizer

    name = tokenizer_name or getattr(tokenizer, "name_or_path", "") or ""
    if _looks_like_llama3(name):
        tokenizer.chat_template = LLAMA3_CHAT_TEMPLATE
        return tokenizer

    return tokenizer
