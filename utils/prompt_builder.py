"""
utils/prompt_builder.py
Provides human-readable descriptions of tone and audience settings.
No LLM prompts needed — this is used for UI display only.
"""

TONE_DESCRIPTIONS = {
    "Formal": "Professional and structured language. Avoids slang and contractions.",
    "Casual": "Relaxed and conversational. Natural, everyday language.",
    "Friendly": "Warm, approachable, and positive. Encouraging and personable.",
    "Professional": "Clear, polished, workplace-appropriate. Balanced formality.",
    "Executive Summary": "Concise, direct, decision-focused. Leads with key information.",
    "Persuasive": "Convincing and engaging. Strong reasoning without changing facts.",
    "Child-Friendly": "Simple vocabulary, short sentences, easy explanations.",
    "Academic": "Structured, objective, scholarly vocabulary and logical flow.",
}

AUDIENCE_DESCRIPTIONS = {
    "General Audience":   "No assumed background. Accessible vocabulary.",
    "Children":           "Ages 7-10. Very simple words, short sentences, analogies.",
    "Students":           "Middle/high school. Clear and approachable.",
    "College Students":   "Assume solid general knowledge. Moderate complexity.",
    "Professionals":      "Domain competence assumed. Industry terminology OK.",
    "Executives":         "Concise, decision-focused. Highlight impacts.",
    "Technical Experts":  "Precise technical terminology. No simplification needed.",
    "Non-Technical Users":"No technical background. Plain language, analogies.",
}

TONE_TECHNIQUE_LABELS = {
    "Formal": [
        "Expands contractions (don't → do not)",
        "Replaces informal words with formal equivalents",
        "Adds formal transition phrases",
    ],
    "Casual": [
        "Contracts formal phrases (do not → don't)",
        "Replaces formal words with everyday equivalents",
        "Adds relaxed sentence openers",
    ],
    "Friendly": [
        "Uses warm, welcoming openers",
        "Replaces cold words with warmer alternatives",
        "Adds encouraging tone markers",
    ],
    "Professional": [
        "Expands contractions",
        "Replaces informal vocabulary",
        "Maintains measured, balanced language",
    ],
    "Executive Summary": [
        "Formats as bullet points with labels",
        "Keeps sentences concise and direct",
        "Leads with decisions and impacts",
    ],
    "Persuasive": [
        "Adds evidence-forward connectors",
        "Strengthens closing statements",
        "Uses convincing transition words",
    ],
    "Child-Friendly": [
        "Replaces complex words with simple alternatives",
        "Shortens sentences to child-friendly length",
        "Adds engaging hooks (Did you know?)",
    ],
    "Academic": [
        "Replaces common words with scholarly equivalents",
        "Adds formal academic framing",
        "Appends evidence-citing closing phrases",
    ],
}
