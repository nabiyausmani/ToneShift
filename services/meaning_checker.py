"""
services/meaning_checker.py
Offline meaning-preservation analysis using TF-IDF cosine similarity + word overlap.
No API key required.
"""

import re
import math
import logging
from collections import Counter
from typing import List, Set

import nltk

for _pkg in ("punkt", "punkt_tab", "stopwords"):
    try:
        nltk.data.find(f"tokenizers/{_pkg}" if "punkt" in _pkg else f"corpora/{_pkg}")
    except LookupError:
        nltk.download(_pkg, quiet=True)

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

logger = logging.getLogger(__name__)

_STOPWORDS: Set[str] = set(stopwords.words("english"))

# ─────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────

def _tokenize(text: str) -> List[str]:
    """Lowercase, tokenize, remove punctuation tokens."""
    tokens = word_tokenize(text.lower())
    return [t for t in tokens if t.isalpha()]


def _content_words(text: str) -> List[str]:
    """Return content words (stopwords removed)."""
    return [t for t in _tokenize(text) if t not in _STOPWORDS]


def _extract_numbers(text: str) -> Set[str]:
    """Extract all numeric tokens (including percentages, years, decimals)."""
    return set(re.findall(r'\b\d+(?:\.\d+)?%?\b', text))


def _extract_named_entities(text: str) -> Set[str]:
    """
    Extract potential named entities using simple heuristic:
    capitalized words not at the start of a sentence.
    """
    sentences = text.split('.')
    entities = set()
    for s in sentences:
        words = s.split()
        for i, w in enumerate(words):
            cleaned = re.sub(r'[^a-zA-Z]', '', w)
            if cleaned and cleaned[0].isupper() and i > 0 and len(cleaned) > 1:
                entities.add(cleaned.lower())
    return entities


# ─────────────────────────────────────────────────────────────────────
# TF-IDF Cosine Similarity
# ─────────────────────────────────────────────────────────────────────

def _tf(tokens: List[str]) -> dict:
    count = Counter(tokens)
    total = len(tokens) or 1
    return {word: cnt / total for word, cnt in count.items()}


def _cosine_similarity(text_a: str, text_b: str) -> float:
    """
    Compute TF cosine similarity between two texts (no IDF needed for 2-doc comparison).
    Returns float in [0, 1].
    """
    tokens_a = _content_words(text_a)
    tokens_b = _content_words(text_b)

    if not tokens_a or not tokens_b:
        return 0.0

    tf_a = _tf(tokens_a)
    tf_b = _tf(tokens_b)

    vocab = set(tf_a.keys()) | set(tf_b.keys())
    dot_product = sum(tf_a.get(w, 0.0) * tf_b.get(w, 0.0) for w in vocab)
    mag_a = math.sqrt(sum(v ** 2 for v in tf_a.values()))
    mag_b = math.sqrt(sum(v ** 2 for v in tf_b.values()))

    if mag_a == 0 or mag_b == 0:
        return 0.0

    return dot_product / (mag_a * mag_b)


def _jaccard_similarity(text_a: str, text_b: str) -> float:
    """Jaccard similarity on content word sets."""
    set_a = set(_content_words(text_a))
    set_b = set(_content_words(text_b))
    if not set_a and not set_b:
        return 1.0
    if not set_a or not set_b:
        return 0.0
    return len(set_a & set_b) / len(set_a | set_b)


def _number_preservation_score(original: str, rewritten: str) -> float:
    """Score how well numbers/statistics are preserved. Returns 0-1."""
    orig_nums = _extract_numbers(original)
    if not orig_nums:
        return 1.0  # Nothing to preserve
    rew_nums  = _extract_numbers(rewritten)
    preserved = orig_nums & rew_nums
    return len(preserved) / len(orig_nums)


def _entity_preservation_score(original: str, rewritten: str) -> float:
    """Score how well named entities are preserved. Returns 0-1."""
    orig_ents = _extract_named_entities(original)
    if not orig_ents:
        return 1.0
    rew_ents  = _extract_named_entities(rewritten)
    preserved = orig_ents & rew_ents
    return len(preserved) / len(orig_ents)


# ─────────────────────────────────────────────────────────────────────
# Public: Meaning Preservation Check
# ─────────────────────────────────────────────────────────────────────

def check_meaning_preservation(original: str, rewritten: str) -> dict:
    """
    Analyse how well the rewritten text preserves the original meaning.

    Returns a dict with:
        score (int 0-100), status (str), explanation (str),
        facts_changed (bool), info_removed (bool), info_added (bool), error (None)
    """
    try:
        cosine  = _cosine_similarity(original, rewritten)
        jaccard = _jaccard_similarity(original, rewritten)
        num_score  = _number_preservation_score(original, rewritten)
        ent_score  = _entity_preservation_score(original, rewritten)

        # Word coverage: how many original content words appear in rewrite
        orig_words = set(_content_words(original))
        rew_words  = set(_content_words(rewritten))
        coverage   = len(orig_words & rew_words) / len(orig_words) if orig_words else 1.0

        # Bloat: how many unique new words in rewrite vs original
        new_words = rew_words - orig_words
        bloat_ratio = len(new_words) / (len(rew_words) or 1)

        # Composite score (weighted)
        raw_score = (
            cosine     * 45 +   # semantic content similarity
            jaccard    * 20 +   # vocabulary overlap
            coverage   * 20 +   # original word retention
            num_score  * 10 +   # number preservation
            ent_score  * 5      # entity preservation
        )
        # Bloat penalty: very high new-word ratio may indicate hallucination
        bloat_penalty = max(0, bloat_ratio - 0.5) * 10
        score = max(0, min(100, int(raw_score * 100 - bloat_penalty)))

        # Determine status
        if score >= 90:
            status = "Meaning Preserved"
        elif score >= 70:
            status = "Minor Meaning Drift"
        else:
            status = "Significant Meaning Drift"

        # Build explanation
        facts_changed = num_score < 0.8 or ent_score < 0.7
        info_removed  = coverage < 0.5
        info_added    = bloat_ratio > 0.6

        parts = []
        if cosine >= 0.85:
            parts.append("vocabulary is well-preserved")
        elif cosine >= 0.6:
            parts.append("core vocabulary partially preserved")
        else:
            parts.append("significant vocabulary change detected")

        if num_score < 1.0 and _extract_numbers(original):
            parts.append(f"{int(num_score*100)}% of numbers retained")
        if info_removed:
            parts.append("some original concepts may be missing")
        if info_added:
            parts.append("new content was introduced")

        explanation = (
            f"Semantic similarity: {int(cosine*100)}%. "
            + (", ".join(parts).capitalize() + "." if parts else "")
        )

        return {
            "score":         score,
            "status":        status,
            "explanation":   explanation,
            "facts_changed": facts_changed,
            "info_removed":  info_removed,
            "info_added":    info_added,
            "cosine":        round(cosine, 3),
            "jaccard":       round(jaccard, 3),
            "coverage":      round(coverage, 3),
            "num_score":     round(num_score, 3),
            "error":         None,
        }

    except Exception as exc:
        logger.error("Meaning check error: %s", exc)
        return _error_result(f"Meaning analysis failed: {exc}")


# ─────────────────────────────────────────────────────────────────────
# Public: Back-Translation Verification
# ─────────────────────────────────────────────────────────────────────

def run_back_translation(original: str, rewritten: str) -> dict:
    """
    Perform back-translation verification:
    1. Convert rewritten text back to neutral style (rule-based).
    2. Compare neutral reconstruction with original.

    Returns a dict with:
        neutral_version (str), score (int), differences (list),
        summary (str), warning (bool), error (None)
    """
    try:
        from services.rewriter import rewrite_neutral
        neutral_version = rewrite_neutral(rewritten)

        # Compare neutral reconstruction with original
        cosine   = _cosine_similarity(original, neutral_version)
        jaccard  = _jaccard_similarity(original, neutral_version)
        coverage = _word_coverage(original, neutral_version)
        num_score = _number_preservation_score(original, neutral_version)

        score = max(0, min(100, int(
            cosine   * 50 +
            jaccard  * 20 +
            coverage * 20 +
            num_score * 10
        ) * 100))

        # Find differences
        differences = _find_differences(original, neutral_version)

        if score >= 85:
            summary = "The rewritten text faithfully preserves the original meaning after back-translation."
        elif score >= 65:
            summary = "Minor differences detected between original and back-translated version."
        else:
            summary = "Notable differences found — the rewrite may have shifted the original meaning."

        return {
            "neutral_version": neutral_version,
            "score":           score,
            "differences":     differences,
            "summary":         summary,
            "warning":         score < 65,
            "error":           None,
        }

    except Exception as exc:
        logger.error("Back-translation error: %s", exc)
        return _bt_error_result(str(exc))


def _word_coverage(text_a: str, text_b: str) -> float:
    """Fraction of content words in text_a that also appear in text_b."""
    words_a = set(_content_words(text_a))
    words_b = set(_content_words(text_b))
    if not words_a:
        return 1.0
    return len(words_a & words_b) / len(words_a)


def _find_differences(original: str, reconstructed: str) -> List[str]:
    """Identify key content differences between two texts."""
    differences = []

    # Check number differences
    orig_nums = _extract_numbers(original)
    rec_nums  = _extract_numbers(reconstructed)
    missing_nums = orig_nums - rec_nums
    if missing_nums:
        differences.append(f"Numbers not retained: {', '.join(sorted(missing_nums))}")

    # Check entity differences
    orig_ents = _extract_named_entities(original)
    rec_ents  = _extract_named_entities(reconstructed)
    missing_ents = orig_ents - rec_ents
    if missing_ents and len(missing_ents) <= 5:
        differences.append(f"Possible missing entities: {', '.join(sorted(missing_ents))}")

    # Check significant unique content words in original not in reconstruction
    orig_content = set(_content_words(original))
    rec_content  = set(_content_words(reconstructed))
    lost_words = orig_content - rec_content
    # Filter to longer meaningful words
    meaningful_lost = {w for w in lost_words if len(w) > 5}
    if len(meaningful_lost) > 3:
        sample = sorted(meaningful_lost)[:5]
        differences.append(f"Key concepts may be missing: {', '.join(sample)}")

    return differences


# ─────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────

def _error_result(message: str) -> dict:
    return {
        "score": 0,
        "status": "Analysis Failed",
        "explanation": message,
        "facts_changed": False,
        "info_removed": False,
        "info_added": False,
        "cosine": 0.0,
        "jaccard": 0.0,
        "coverage": 0.0,
        "num_score": 0.0,
        "error": message,
    }


def _bt_error_result(message: str) -> dict:
    return {
        "neutral_version": "",
        "score": 0,
        "differences": [],
        "summary": message,
        "warning": False,
        "error": message,
    }


def status_emoji(status: str) -> str:
    mapping = {
        "Meaning Preserved": "🟢",
        "Minor Meaning Drift": "🟡",
        "Significant Meaning Drift": "🔴",
        "Analysis Failed": "⚠️",
    }
    return mapping.get(status, "⚪")


def score_to_color(score: int) -> str:
    if score >= 75:
        return "#22c55e"
    elif score >= 55:
        return "#f59e0b"
    else:
        return "#ef4444"
