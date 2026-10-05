"""
services/rewriter.py
Rule-based text rewriting engine — works entirely offline, no API key needed.

Pipeline:
  1. Tokenize into sentences (NLTK)
  2. Apply contraction expansion / contraction based on tone
  3. Apply vocabulary substitution map
  4. Apply audience-specific jargon replacement
  5. Apply length adjustment (shorten / expand sentences)
  6. Apply formality-level micro-adjustments
  7. Apply tone-specific structural touches (executive bullets, friendly openers, etc.)
  8. Reassemble text
"""

import re
import random
import textwrap
from typing import List

import nltk

# Download required NLTK data on first use (quiet mode)
for _pkg in ("punkt", "punkt_tab", "averaged_perceptron_tagger", "averaged_perceptron_tagger_eng"):
    try:
        nltk.data.find(f"tokenizers/{_pkg}" if "punkt" in _pkg else f"taggers/{_pkg}")
    except LookupError:
        nltk.download(_pkg, quiet=True)

from nltk.tokenize import sent_tokenize, word_tokenize

from services.tone_data import (
    CONTRACTIONS_EXPAND,
    CONTRACTIONS_CONTRACT,
    INFORMAL_TO_FORMAL,
    FORMAL_TO_CASUAL_EXTRA,
    CHILD_FRIENDLY,
    ACADEMIC_VOCAB,
    PERSUASIVE_CONNECTORS,
    TONE_SENTENCE_STARTERS,
    TONE_CONFIG,
    AUDIENCE_CONFIG,
)

# Seed for reproducibility within a session
random.seed(42)


# ─────────────────────────────────────────────────────────────────────
# Step 1 — Tokenisation helpers
# ─────────────────────────────────────────────────────────────────────

def _split_sentences(text: str) -> List[str]:
    """Split text into sentences using NLTK."""
    return [s.strip() for s in sent_tokenize(text) if s.strip()]


def _join_sentences(sentences: List[str]) -> str:
    """Join sentences into a paragraph."""
    return " ".join(s.strip() for s in sentences if s.strip())


# ─────────────────────────────────────────────────────────────────────
# Step 2 — Contraction handling
# ─────────────────────────────────────────────────────────────────────

def _expand_contractions(text: str) -> str:
    """Expand contractions: don't → do not"""
    result = text
    # Sort by length descending to avoid partial matches
    for contraction, expanded in sorted(CONTRACTIONS_EXPAND.items(), key=lambda x: -len(x[0])):
        pattern = re.compile(re.escape(contraction), re.IGNORECASE)
        def _replace(m, exp=expanded, orig=contraction):
            if m.group(0)[0].isupper():
                return exp[0].upper() + exp[1:]
            return exp
        result = pattern.sub(_replace, result)
    return result


def _contract_words(text: str) -> str:
    """Contract formal phrases: do not → don't"""
    result = text
    for formal, contraction in sorted(CONTRACTIONS_CONTRACT.items(), key=lambda x: -len(x[0])):
        pattern = re.compile(r'\b' + re.escape(formal) + r'\b', re.IGNORECASE)
        def _replace(m, con=contraction):
            return con
        result = pattern.sub(_replace, result)
    return result


# ─────────────────────────────────────────────────────────────────────
# Step 3 — Vocabulary substitution
# ─────────────────────────────────────────────────────────────────────

def _apply_vocab_map(text: str, vocab_map: dict, max_replacements: int = 20) -> str:
    """
    Replace words/phrases from vocab_map.
    Applies at most max_replacements to avoid over-transformation.
    """
    result = text
    applied = 0
    # Sort by phrase length descending (replace phrases before single words)
    for source, target in sorted(vocab_map.items(), key=lambda x: -len(x[0])):
        if applied >= max_replacements:
            break
        pattern = re.compile(r'\b' + re.escape(source) + r'\b', re.IGNORECASE)
        matches = list(pattern.finditer(result))
        if matches:
            def _replace(m, tgt=target, src=source):
                if m.group(0)[0].isupper() and tgt:
                    return tgt[0].upper() + tgt[1:]
                return tgt
            result = pattern.sub(_replace, result)
            applied += len(matches)
    return result


# ─────────────────────────────────────────────────────────────────────
# Step 4 — Audience jargon replacement
# ─────────────────────────────────────────────────────────────────────

def _apply_audience_vocab(text: str, audience: str) -> str:
    config = AUDIENCE_CONFIG.get(audience, {})
    jargon_map = config.get("jargon_replacement", {})
    if jargon_map:
        text = _apply_vocab_map(text, jargon_map, max_replacements=30)
    return text


# ─────────────────────────────────────────────────────────────────────
# Step 5 — Length adjustment
# ─────────────────────────────────────────────────────────────────────

def _split_long_sentence(sentence: str, max_words: int) -> List[str]:
    """Split a long sentence at a conjunction if possible."""
    words = sentence.split()
    if len(words) <= max_words:
        return [sentence]

    conjunctions = ["and", "but", "or", "so", "yet", "because", "which", "that",
                    "however", "therefore", "furthermore", "additionally", "moreover"]

    # Find a good split point around the midpoint
    mid = len(words) // 2
    best_split = -1
    best_dist = float("inf")
    for i, word in enumerate(words):
        if word.lower() in conjunctions and abs(i - mid) < best_dist:
            best_split = i
            best_dist = abs(i - mid)

    if best_split > 1:
        first = " ".join(words[:best_split]).rstrip(",").rstrip(";") + "."
        second = " ".join(words[best_split:]).capitalize()
        return [first, second]
    return [sentence]


def _merge_short_sentences(sentences: List[str], min_words: int = 5) -> List[str]:
    """Merge short consecutive sentences with commas."""
    result = []
    i = 0
    while i < len(sentences):
        s = sentences[i]
        if len(s.split()) < min_words and i + 1 < len(sentences):
            # Merge with next
            next_s = sentences[i + 1]
            merged = s.rstrip(".!?") + ", " + next_s[0].lower() + next_s[1:]
            result.append(merged)
            i += 2
        else:
            result.append(s)
            i += 1
    return result


def _expand_sentence(sentence: str) -> str:
    """Add an explanatory clause to expand a sentence."""
    # Add contextual phrases based on sentence content
    expansions = [
        " This is particularly significant in today's context.",
        " It is important to understand the broader implications of this.",
        " This aspect deserves careful consideration.",
        " The implications of this extend across multiple areas.",
        " Understanding this fully requires examining the underlying factors.",
    ]
    # Don't expand if already long
    if len(sentence.split()) > 20:
        return sentence
    return sentence.rstrip(".") + random.choice(expansions)


def _adjust_length(sentences: List[str], length_option: str, audience: str) -> List[str]:
    """Adjust number and size of sentences based on length option."""
    config = AUDIENCE_CONFIG.get(audience, {})
    max_sent_words = config.get("sentence_length_limit", 25)

    if length_option == "Much Shorter":
        # Keep only top ~40% sentences, split none
        keep = max(1, int(len(sentences) * 0.4))
        return sentences[:keep]

    elif length_option == "Shorter":
        # Keep ~65% of sentences
        keep = max(1, int(len(sentences) * 0.65))
        result = sentences[:keep]
        # Also split sentences over limit
        split_result = []
        for s in result:
            split_result.extend(_split_long_sentence(s, max_sent_words))
        return split_result

    elif length_option == "Same Length":
        # Split long sentences but keep count similar
        result = []
        for s in sentences:
            result.extend(_split_long_sentence(s, max_sent_words))
        return result

    elif length_option == "Longer":
        # Expand each sentence slightly + add split ones
        result = []
        for i, s in enumerate(sentences):
            if i % 2 == 0:
                result.append(_expand_sentence(s))
            else:
                result.append(s)
        split_result = []
        for s in result:
            split_result.extend(_split_long_sentence(s, max_sent_words + 5))
        return split_result

    elif length_option == "Much Longer":
        # Expand every sentence + add bridging sentences
        result = []
        for s in sentences:
            result.append(_expand_sentence(s))
            # Add a connector sentence every 2
        bridged = []
        for i, s in enumerate(result):
            bridged.append(s)
            if i < len(result) - 1 and i % 2 == 0:
                bridged.append(
                    "Furthermore, this is an important consideration that merits further examination."
                )
        return bridged

    return sentences


# ─────────────────────────────────────────────────────────────────────
# Step 6 — Formality micro-adjustments
# ─────────────────────────────────────────────────────────────────────

def _apply_formality(text: str, level: int) -> str:
    """
    Apply formality adjustments.
    level 1-2: casual markers; level 4-5: formal markers.
    """
    if level <= 2:
        # Remove stiff phrases
        text = text.replace("It is important to note that", "Just so you know,")
        text = text.replace("One must acknowledge that", "Keep in mind that")
        text = text.replace("It should be noted that", "Note that")
        text = text.replace("In accordance with the aforementioned,", "With that in mind,")
        text = text.replace("Furthermore,", "Also,")
        text = text.replace("Moreover,", "Plus,")
        text = text.replace("Nevertheless,", "Still,")
        text = text.replace("Subsequently,", "Then,")
        text = text.replace("Prior to", "Before")
        text = text.replace("Subsequent to", "After")
        text = text.replace("approximately", "about")
        text = text.replace("sufficient", "enough")
        text = text.replace("regarding", "about")
        text = text.replace("inquire", "ask")
        text = text.replace("obtain", "get")
        text = text.replace("commence", "start")
        text = text.replace("terminate", "end")
    elif level >= 4:
        # Add formality
        text = text.replace(" and also", ", and furthermore,")
        text = text.replace("Also,", "Furthermore,")
        text = text.replace("But ", "However, ")
        text = text.replace(" so ", " therefore ")
        text = text.replace("about", "approximately")
        text = text.replace(" get ", " obtain ")
        text = text.replace(" start ", " commence ")
        text = text.replace(" end ", " conclude ")
    return text


# ─────────────────────────────────────────────────────────────────────
# Step 7 — Tone-specific structural touches
# ─────────────────────────────────────────────────────────────────────

def _apply_tone_structure(sentences: List[str], tone: str, audience: str) -> List[str]:
    """Apply tone-specific structural changes to the sentence list."""

    if tone == "Executive Summary":
        # Prefix key sentences with executive labels
        labels = ["Key point:", "Action:", "Impact:", "Summary:", "Note:"]
        structured = []
        for i, s in enumerate(sentences):
            if i < len(labels):
                structured.append(f"• **{labels[i]}** {s}")
            else:
                structured.append(f"• {s}")
        return structured

    elif tone == "Child-Friendly" or audience == "Children":
        # Add encouraging interjections + question hooks
        result = []
        for i, s in enumerate(sentences):
            if i == 0:
                result.append("Did you know? " + s)
            elif i == len(sentences) - 1:
                result.append(s + " Pretty cool, right?")
            else:
                result.append(s)
        return result

    elif tone == "Friendly":
        # Add warm opener to first sentence
        openers = [
            "Here's something interesting — ",
            "Great to share this with you — ",
            "Let's explore this together — ",
        ]
        if sentences:
            sentences[0] = random.choice(openers) + sentences[0][0].lower() + sentences[0][1:]
        return sentences

    elif tone == "Persuasive":
        # Apply persuasive connectors
        result = []
        for i, s in enumerate(sentences):
            for trigger, replacement in PERSUASIVE_CONNECTORS.items():
                pattern = re.compile(r'\b' + re.escape(trigger) + r'\b', re.IGNORECASE)
                s = pattern.sub(replacement, s, count=1)
            result.append(s)
        # Add strong closing
        if result:
            result[-1] = result[-1].rstrip(".") + " — and the evidence speaks for itself."
        return result

    elif tone == "Academic":
        # Prepend a scholarly framing sentence, then keep originals
        result = ["This study examines the following proposition."] + sentences
        # Add concluding scholarly phrase to last sentence
        if result:
            result[-1] = result[-1].rstrip(".") + ", as the evidence presented herein demonstrates."
        return result

    elif tone == "Casual":
        # Keep it loose — add "Look," or "Basically," to start
        openers = ["Look, ", "So basically, ", "Here's the deal — ", "Alright, so "]
        if sentences:
            sentences[0] = random.choice(openers) + sentences[0][0].lower() + sentences[0][1:]
        return sentences

    return sentences


# ─────────────────────────────────────────────────────────────────────
# Step 8 — Sentence length cap (audience-based)
# ─────────────────────────────────────────────────────────────────────

def _enforce_sentence_length_cap(sentences: List[str], audience: str) -> List[str]:
    """Split sentences that exceed the audience word limit."""
    config = AUDIENCE_CONFIG.get(audience, {})
    limit = config.get("sentence_length_limit", 30)
    result = []
    for s in sentences:
        result.extend(_split_long_sentence(s, limit))
    return result


# ─────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────

def rewrite(
    text: str,
    tone: str,
    audience: str,
    length: str,
    formality: int,
) -> str:
    """
    Rewrite text using the rule-based engine.

    Args:
        text:       Original input text.
        tone:       Selected tone name.
        audience:   Target audience name.
        length:     Desired length option string.
        formality:  Integer 1-5 formality level.

    Returns:
        Rewritten text string.
    """
    tone_cfg = TONE_CONFIG.get(tone, TONE_CONFIG["Professional"])

    # 1 — Tokenise
    sentences = _split_sentences(text)
    if not sentences:
        return text

    # 2 — Contraction handling
    processed_sentences = []
    for s in sentences:
        if tone_cfg["expand_contractions"]:
            s = _expand_contractions(s)
        elif tone_cfg["contract_contractions"]:
            s = _contract_words(s)
        processed_sentences.append(s)

    # 3 — Vocabulary substitution (tone-based)
    vocab_map = tone_cfg.get("vocab_map", {})
    full_text = _join_sentences(processed_sentences)
    full_text = _apply_vocab_map(full_text, vocab_map)

    # 4 — Audience vocabulary
    full_text = _apply_audience_vocab(full_text, audience)

    # 5 — Formality micro-adjustments (applied to full text)
    full_text = _apply_formality(full_text, formality)

    # 6 — Re-tokenise for structural work
    sentences = _split_sentences(full_text)

    # 7 — Length adjustment
    sentences = _adjust_length(sentences, length, audience)

    # 8 — Audience sentence-length cap
    sentences = _enforce_sentence_length_cap(sentences, audience)

    # 9 — Tone-specific structure
    sentences = _apply_tone_structure(sentences, tone, audience)

    # 10 — Final assembly
    result = _join_sentences(sentences)

    # Clean up double spaces and awkward punctuation
    result = re.sub(r'\s+', ' ', result).strip()
    result = re.sub(r'\s([,.!?;:])', r'\1', result)
    result = re.sub(r'\.+', '.', result)

    # Ensure ends with punctuation
    if result and result[-1] not in ".!?":
        result += "."

    return result


def rewrite_neutral(text: str) -> str:
    """
    Convert text to a plain, neutral style (used for back-translation check).
    Applies moderate formality, Professional tone, Same Length.
    """
    return rewrite(
        text=text,
        tone="Professional",
        audience="General Audience",
        length="Same Length",
        formality=3,
    )
