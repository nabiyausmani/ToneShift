# ToneShift — Offline AI Rewriter

> **Transform your writing for any audience and tone — 100% offline, no API key required.**

ToneShift is a fully-featured text rewriting application built with Python and Streamlit. It uses a **local rule-based NLP engine** (NLTK + scikit-learn) to intelligently rewrite text according to your chosen tone, audience, length, and formality — with no internet connection or API key needed.

---

## ✨ Features

| Feature | Description |
|---|---|
| **8 Tones** | Formal, Casual, Friendly, Professional, Executive Summary, Persuasive, Child-Friendly, Academic |
| **8 Audiences** | General, Children, Students, College Students, Professionals, Executives, Technical Experts, Non-Technical |
| **Length Control** | 5-level slider: Much Shorter → Much Longer |
| **Formality Slider** | 1 (Very Casual) to 5 (Very Formal) |
| **Meaning Preservation** | TF-IDF cosine similarity + Jaccard overlap + number/entity retention (0–100%) |
| **Back-Translation Check** | Independently verifies meaning via neutral rule-based reconstruction |
| **Side-by-Side Comparison** | Visual diff with word count stats |
| **Tone Comparison** | Generate up to 4 tone variants simultaneously |
| **Session History** | Review, restore, and download previous rewrites |
| **100% Offline** | No API key, no internet required after first install |

---

## 🛠 Technology Stack

| Layer | Technology |
|---|---|
| **Language** | Python 3.10+ |
| **Web Framework** | Streamlit |
| **NLP Engine** | NLTK (tokenisation, stopwords, POS tagging) |
| **Similarity** | TF-IDF Cosine, Jaccard word overlap (scikit-learn / built-in) |
| **Rewriting** | Rule-based engine (contraction handling, vocab maps, sentence transforms) |
| **Styling** | Custom CSS (dark theme, glassmorphism) |

---

## 📁 Project Structure

```
ToneShift/
│
├── app.py                   # Main Streamlit application
├── requirements.txt         # Dependencies (NLTK, Streamlit, scikit-learn)
├── .gitignore
├── README.md
│
├── services/
│   ├── __init__.py
│   ├── rewriter.py          # Rule-based rewriting engine
│   ├── meaning_checker.py   # Semantic similarity + back-translation
│   └── tone_data.py         # Vocabulary dictionaries & tone configs
│
├── utils/
│   ├── __init__.py
│   ├── prompt_builder.py    # Tone/audience descriptions for UI
│   └── text_utils.py        # Word/char count, validation helpers
│
├── assets/
│   └── styles.css           # Custom dark-theme Streamlit styling
│
└── .streamlit/
    └── config.toml          # Dark theme defaults
```

---

## ⚡ Quick Start

### 1. Prerequisites

- Python 3.10 or newer
- pip

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the Application

```bash
streamlit run app.py
```

The app opens automatically at **http://localhost:8501**.

> **No API key or `.env` file needed.** The app is completely self-contained.

---

## 📖 How It Works

### Rewriting Engine (`services/rewriter.py`)
The rule-based engine applies the following pipeline to each text:

1. **Sentence tokenisation** (NLTK punkt)
2. **Contraction expansion / contraction** based on tone (e.g. `don't → do not` for Formal)
3. **Vocabulary substitution** using tone-specific word maps (1000+ word pairs)
4. **Audience jargon replacement** (e.g. `algorithm → way the computer thinks` for Children)
5. **Length adjustment** — splits/merges/expands sentences as requested
6. **Formality micro-adjustments** (levels 1–5)
7. **Tone-specific structural transforms** (bullet points for Executive, hooks for Child-Friendly, etc.)
8. **Reassembly and clean-up**

### Meaning Checker (`services/meaning_checker.py`)
Uses four independent signals:
- **TF-IDF cosine similarity** — semantic content overlap (45% weight)
- **Jaccard similarity** — vocabulary set overlap (20% weight)
- **Content word coverage** — fraction of key original words retained (20% weight)
- **Number/entity preservation** — checks that all numbers and named entities survived (15% weight)

Combined score 0–100 → `Meaning Preserved / Minor Drift / Significant Drift`

---

## 📖 How to Use

### Rewrite Tab
1. Enter text in the left panel
2. Select a tone, audience, length, and formality
3. Click **✨ Rewrite Text**
4. View result, word counts, meaning score, and back-translation on the right

### Compare Tones Tab
- Select up to **4 tones** → click **🎨 Generate Tone Comparison**
- All 4 versions appear side by side

### Meaning Check Tab
- Paste any two texts and click **🔍 Run Meaning Analysis**

### History Tab
- Browse, restore, or download previous rewrites from this session

---

## 🔒 Privacy

Since ToneShift runs entirely offline:
- Your text **never leaves your device**
- No API calls, no data collection, no telemetry
- `.env` file is not needed

---

## 🐛 Troubleshooting

| Problem | Solution |
|---|---|
| App won't start | Ensure Python 3.10+ and run `pip install -r requirements.txt` |
| NLTK error on first run | The app auto-downloads NLTK data on first launch — wait a few seconds |
| Result looks similar to original | Try a more extreme tone (e.g. Child-Friendly or Academic) or longer input |
| Short input, short output | Add more content — rule-based engines work better with 2+ sentences |

---

## 📄 License

This project is provided for educational and personal use.
