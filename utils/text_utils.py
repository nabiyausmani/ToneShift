"""
utils/text_utils.py
Utilities for text analysis: word count, character count, validation, etc.
"""

import re


def count_words(text: str) -> int:
    """Count the number of words in a string."""
    if not text or not text.strip():
        return 0
    return len(text.split())


def count_characters(text: str) -> int:
    """Count the number of characters (excluding leading/trailing whitespace)."""
    if not text:
        return 0
    return len(text)


def count_sentences(text: str) -> int:
    """Estimate the number of sentences in text."""
    if not text or not text.strip():
        return 0
    sentences = re.split(r'[.!?]+', text)
    return len([s for s in sentences if s.strip()])


def is_empty(text: str) -> bool:
    """Return True if text is None or contains only whitespace."""
    return not text or not text.strip()


def is_too_short(text: str, min_words: int = 3) -> bool:
    """Return True if text has fewer than min_words words."""
    return count_words(text) < min_words


def is_too_long(text: str, max_chars: int = 8000) -> bool:
    """Return True if text exceeds max_chars characters."""
    return len(text) > max_chars


def word_count_diff(original: str, rewritten: str) -> dict:
    """
    Calculate word count difference between original and rewritten text.

    Returns a dict with:
        original_count, rewritten_count, difference, percentage_change
    """
    orig_count = count_words(original)
    rew_count = count_words(rewritten)
    difference = rew_count - orig_count
    if orig_count > 0:
        percentage = round((difference / orig_count) * 100, 1)
    else:
        percentage = 0.0
    return {
        "original_count": orig_count,
        "rewritten_count": rew_count,
        "difference": difference,
        "percentage_change": percentage,
    }


def truncate_text(text: str, max_chars: int = 8000) -> str:
    """Truncate text to max_chars if it exceeds the limit."""
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rsplit(" ", 1)[0] + "..."


def format_percentage(value: float) -> str:
    """Format a float as a percentage string with sign."""
    sign = "+" if value > 0 else ""
    return f"{sign}{value:.1f}%"


def get_text_stats(text: str) -> dict:
    """Return a dictionary of text statistics."""
    return {
        "word_count": count_words(text),
        "char_count": count_characters(text),
        "sentence_count": count_sentences(text),
    }


def formality_label(level: int) -> str:
    """Convert a numeric formality level (1-5) to a descriptive label."""
    labels = {
        1: "Very Casual",
        2: "Casual",
        3: "Neutral",
        4: "Formal",
        5: "Very Formal",
    }
    return labels.get(level, "Neutral")


def length_description(option: str) -> str:
    """Return a human-readable description for a length option."""
    descriptions = {
        "Much Shorter": "Keep only the essential information. Be very concise.",
        "Shorter": "Remove unnecessary words. Be more concise than the original.",
        "Same Length": "Maintain approximately the same length as the original.",
        "Longer": "Add useful explanations and context without inventing new facts.",
        "Much Longer": "Expand with detailed explanations and rich context while preserving the original meaning.",
    }
    return descriptions.get(option, "Maintain approximately the same length as the original.")
