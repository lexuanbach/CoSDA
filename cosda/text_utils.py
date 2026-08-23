from __future__ import annotations


def compact_text(text: str | None, max_chars: int = 2200, head_ratio: float = 0.65) -> str:
    """Keep prompts bounded while preserving beginning and ending context."""
    value = (text or "").strip()
    if len(value) <= max_chars:
        return value
    marker = "\n[... excerpt truncated ...]\n"
    budget = max(0, max_chars - len(marker))
    head = max(1, int(budget * head_ratio))
    tail = max(1, budget - head)
    return value[:head].rstrip() + marker + value[-tail:].lstrip()


def text_was_compacted(text: str | None, max_chars: int) -> bool:
    return len((text or "").strip()) > max_chars
