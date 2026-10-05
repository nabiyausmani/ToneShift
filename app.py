"""
app.py — ToneShift: Audience-Aware AI Rewriter
Fully offline — no API key required. Uses a local rule-based NLP engine.
"""

import json
from datetime import datetime
from pathlib import Path

import streamlit as st
import nltk

# ── NLTK bootstrap (download required data if not present) ────────────────────
for _pkg in [("tokenizers", "punkt"), ("tokenizers", "punkt_tab"),
             ("corpora", "stopwords"),
             ("taggers", "averaged_perceptron_tagger"),
             ("taggers", "averaged_perceptron_tagger_eng")]:
    try:
        nltk.data.find(f"{_pkg[0]}/{_pkg[1]}")
    except LookupError:
        nltk.download(_pkg[1], quiet=True)

from services.rewriter import rewrite
from services.meaning_checker import (
    check_meaning_preservation,
    run_back_translation,
    status_emoji,
    score_to_color,
)
from utils.prompt_builder import (
    TONE_DESCRIPTIONS,
    AUDIENCE_DESCRIPTIONS,
    TONE_TECHNIQUE_LABELS,
)
from utils.text_utils import (
    count_words,
    count_characters,
    is_empty,
    is_too_short,
    is_too_long,
    word_count_diff,
    format_percentage,
    formality_label,
)

# ─────────────────────────────────────────────────────────────────────
# Page config (must be FIRST Streamlit call)
# ─────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ToneShift — Offline AI Rewriter",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────────────────────────────
_CSS_PATH = Path(__file__).parent / "assets" / "styles.css"
if _CSS_PATH.exists():
    st.markdown(
        f"<style>{_CSS_PATH.read_text(encoding='utf-8')}</style>",
        unsafe_allow_html=True,
    )

# ─────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────
TONES = [
    "Formal",
    "Casual",
    "Friendly",
    "Professional",
    "Executive Summary",
    "Persuasive",
    "Child-Friendly",
    "Academic",
]

AUDIENCES = [
    "General Audience",
    "Children",
    "Students",
    "College Students",
    "Professionals",
    "Executives",
    "Technical Experts",
    "Non-Technical Users",
]

LENGTHS = ["Much Shorter", "Shorter", "Same Length", "Longer", "Much Longer"]

DEFAULT_TONE      = "Professional"
DEFAULT_AUDIENCE  = "General Audience"
DEFAULT_LENGTH    = "Same Length"
DEFAULT_FORMALITY = 3
MAX_CHARS         = 8000


# ─────────────────────────────────────────────────────────────────────
# Session State Init
# ─────────────────────────────────────────────────────────────────────
def _init_state():
    defaults = {
        "rewrite_result":   None,
        "meaning_result":   None,
        "bt_result":        None,
        "tone_comparison":  None,
        "history":          [],
        "last_input":       "",
        "last_settings":    {},
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


_init_state()


# ─────────────────────────────────────────────────────────────────────
# History helper
# ─────────────────────────────────────────────────────────────────────
def _add_history(original, rewritten, tone, audience, length, formality, meaning_score):
    st.session_state.history.insert(0, {
        "id":             len(st.session_state.history),
        "timestamp":      datetime.now().strftime("%H:%M:%S"),
        "original":       original[:300] + ("…" if len(original) > 300 else ""),
        "rewritten":      rewritten[:300] + ("…" if len(rewritten) > 300 else ""),
        "original_full":  original,
        "rewritten_full": rewritten,
        "tone":           tone,
        "audience":       audience,
        "length":         length,
        "formality":      formality,
        "meaning_score":  meaning_score,
    })


# ─────────────────────────────────────────────────────────────────────
# Hero
# ─────────────────────────────────────────────────────────────────────
def _hero():
    st.markdown("""
    <div class="ts-hero">
      <div class="ts-logo">🎨 ToneShift</div>
      <div class="ts-tagline">Audience-Aware Rewriter &nbsp;·&nbsp; 100% Offline · No API Key Required</div>
      <div class="ts-desc">
        Transform your writing for any audience and tone while preserving your original meaning —
        powered entirely by local NLP. No internet or API key needed.
      </div>
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────
# Input Panel
# ─────────────────────────────────────────────────────────────────────
def _input_panel():
    """Render left panel. Returns (text, tone, audience, length, formality)."""
    st.markdown('<div class="ts-card-title">✍️ Original Text</div>', unsafe_allow_html=True)

    input_text = st.text_area(
        "Enter your text",
        placeholder="Paste or type the text you want to rewrite…",
        height=220,
        max_chars=MAX_CHARS,
        key="input_text_area",
        label_visibility="collapsed",
    )

    words = count_words(input_text)
    chars = count_characters(input_text)
    st.markdown(
        f'<div class="ts-badge-row">'
        f'<span class="ts-badge">📝 {words} words</span>'
        f'<span class="ts-badge ts-badge-neutral">⌨️ {chars}/{MAX_CHARS} chars</span>'
        f'</div>',
        unsafe_allow_html=True,
    )

    if not is_empty(input_text) and is_too_short(input_text, 5):
        st.warning("⚠️ Text is very short. Results may be limited.", icon=None)
    if is_too_long(input_text, MAX_CHARS):
        st.error(f"❌ Exceeds {MAX_CHARS} characters. Please shorten.", icon=None)

    st.markdown("---")
    st.markdown('<div class="ts-card-title">⚙️ Rewrite Settings</div>', unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<div class="ts-label">🎭 Tone</div>', unsafe_allow_html=True)
        tone = st.selectbox(
            "Tone", TONES,
            index=TONES.index(DEFAULT_TONE),
            label_visibility="collapsed", key="tone_select",
        )
        desc = TONE_DESCRIPTIONS.get(tone, "")
        st.caption(f"*{desc[:65]}{'…' if len(desc)>65 else ''}*")

    with c2:
        st.markdown('<div class="ts-label">👥 Target Audience</div>', unsafe_allow_html=True)
        audience = st.selectbox(
            "Audience", AUDIENCES,
            index=0,
            label_visibility="collapsed", key="audience_select",
        )
        adesc = AUDIENCE_DESCRIPTIONS.get(audience, "")
        st.caption(f"*{adesc[:65]}{'…' if len(adesc)>65 else ''}*")

    st.markdown('<div class="ts-label" style="margin-top:.75rem">📏 Desired Length</div>', unsafe_allow_html=True)
    length = st.select_slider(
        "Length", options=LENGTHS, value=DEFAULT_LENGTH,
        label_visibility="collapsed", key="length_slider",
    )

    st.markdown('<div class="ts-label" style="margin-top:.75rem">🎛️ Formality Level</div>', unsafe_allow_html=True)
    formality = st.slider(
        "Formality", min_value=1, max_value=5, value=DEFAULT_FORMALITY,
        label_visibility="collapsed", key="formality_slider",
    )
    fl = formality_label(formality)
    st.markdown(
        f'<div style="display:flex;justify-content:space-between;font-size:.75rem;'
        f'color:var(--text-muted);margin-top:-.5rem;">'
        f'<span>Very Casual</span>'
        f'<span style="color:var(--accent-secondary);font-weight:600">{fl}</span>'
        f'<span>Very Formal</span></div>',
        unsafe_allow_html=True,
    )

    # Show active techniques
    techniques = TONE_TECHNIQUE_LABELS.get(tone, [])
    if techniques:
        with st.expander("🔧 Active rewriting techniques"):
            for t in techniques:
                st.markdown(f"• {t}")

    return input_text, tone, audience, length, formality


# ─────────────────────────────────────────────────────────────────────
# Rewrite Logic
# ─────────────────────────────────────────────────────────────────────
def _do_rewrite(input_text, tone, audience, length, formality):
    if is_empty(input_text):
        st.error("❌ Please enter some text before rewriting.")
        return
    if is_too_long(input_text, MAX_CHARS):
        st.error(f"❌ Text exceeds {MAX_CHARS} characters.")
        return

    with st.spinner("🔄 Rewriting your text…"):
        try:
            rewritten = rewrite(input_text, tone, audience, length, formality)
        except Exception as exc:
            st.error(f"❌ Rewriting error: {exc}")
            return

    st.session_state.rewrite_result = rewritten
    st.session_state.last_input     = input_text
    st.session_state.last_settings  = {
        "tone": tone, "audience": audience,
        "length": length, "formality": formality,
    }

    with st.spinner("🔍 Analysing meaning preservation…"):
        meaning = check_meaning_preservation(input_text, rewritten)
    st.session_state.meaning_result = meaning

    with st.spinner("↩️ Running back-translation check…"):
        bt = run_back_translation(input_text, rewritten)
    st.session_state.bt_result = bt

    _add_history(input_text, rewritten, tone, audience, length, formality,
                 meaning.get("score", 0))
    st.success("✅ Done!")


# ─────────────────────────────────────────────────────────────────────
# Output Panel
# ─────────────────────────────────────────────────────────────────────
def _output_panel():
    if st.session_state.rewrite_result is None:
        st.markdown("""
        <div style="display:flex;flex-direction:column;align-items:center;justify-content:center;
          height:320px;border:2px dashed rgba(124,107,255,0.25);border-radius:16px;
          color:var(--text-muted);text-align:center;padding:2rem;">
          <div style="font-size:3rem;margin-bottom:.75rem">✨</div>
          <div style="font-size:1rem;font-weight:600;color:var(--text-secondary);margin-bottom:.35rem">
            Your rewritten text will appear here
          </div>
          <div style="font-size:.85rem">
            Configure your settings on the left and click <strong>Rewrite Text</strong>
          </div>
        </div>""", unsafe_allow_html=True)
        return

    rewritten = st.session_state.rewrite_result
    original  = st.session_state.last_input
    settings  = st.session_state.last_settings

    st.markdown('<div class="ts-card-title">✨ Rewritten Text</div>', unsafe_allow_html=True)

    diff = word_count_diff(original, rewritten)
    pct  = format_percentage(diff["percentage_change"])
    m1, m2, m3 = st.columns(3)
    m1.metric("Original Words",  diff["original_count"])
    m2.metric("Rewritten Words", diff["rewritten_count"],
              delta=f"{'+' if diff['difference']>0 else ''}{diff['difference']}")
    m3.metric("Length Change", pct)

    st.text_area(
        "Rewritten", value=rewritten, height=220,
        key="output_text_area", label_visibility="collapsed",
    )

    c_copy, c_regen, c_dl, c_clear = st.columns(4)
    with c_copy:
        if st.button("📋 Copy", key="copy_btn", use_container_width=True):
            st.write(
                f'<script>navigator.clipboard.writeText({json.dumps(rewritten)})</script>',
                unsafe_allow_html=True,
            )
            st.toast("📋 Copied!", icon="✅")
    with c_regen:
        if st.button("🔄 Regenerate", key="regen_btn", use_container_width=True):
            s = st.session_state.last_settings
            _do_rewrite(st.session_state.last_input,
                        s["tone"], s["audience"], s["length"], s["formality"])
            st.rerun()
    with c_dl:
        st.download_button(
            "⬇️ Download", data=rewritten,
            file_name="toneshift_rewrite.txt", mime="text/plain",
            key="dl_btn", use_container_width=True,
        )
    with c_clear:
        if st.button("🗑️ Clear", key="clear_btn", use_container_width=True):
            st.session_state.rewrite_result = None
            st.session_state.meaning_result = None
            st.session_state.bt_result      = None
            st.rerun()

    st.markdown(
        f'<div class="ts-badge-row" style="margin-top:.6rem">'
        f'<span class="ts-badge">🎭 {settings.get("tone","")}</span>'
        f'<span class="ts-badge">👥 {settings.get("audience","")}</span>'
        f'<span class="ts-badge ts-badge-neutral">📏 {settings.get("length","")}</span>'
        f'<span class="ts-badge ts-badge-neutral">🎛️ {formality_label(settings.get("formality",3))}</span>'
        f'</div>',
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────────────────────────────
# Comparison Block
# ─────────────────────────────────────────────────────────────────────
def _render_comparison():
    original  = st.session_state.last_input
    rewritten = st.session_state.rewrite_result

    st.markdown('<div class="ts-card-title">🔀 Side-by-Side Comparison</div>', unsafe_allow_html=True)
    left, right = st.columns(2, gap="medium")
    with left:
        st.markdown('<div class="ts-compare-label ts-compare-label-orig">📄 Original</div>',
                    unsafe_allow_html=True)
        st.markdown(f'<div class="ts-compare-box">{original}</div>', unsafe_allow_html=True)
    with right:
        st.markdown('<div class="ts-compare-label ts-compare-label-rew">✨ Rewritten</div>',
                    unsafe_allow_html=True)
        st.markdown(f'<div class="ts-compare-box">{rewritten}</div>', unsafe_allow_html=True)

    diff = word_count_diff(original, rewritten)
    pct  = format_percentage(diff["percentage_change"])
    c1, c2, c3 = st.columns(3)
    c1.metric("Original",  f"{diff['original_count']} words")
    c2.metric("Rewritten", f"{diff['rewritten_count']} words")
    c3.metric("Length Change", pct)


# ─────────────────────────────────────────────────────────────────────
# Meaning Analysis Block
# ─────────────────────────────────────────────────────────────────────
def _render_meaning_analysis():
    meaning = st.session_state.meaning_result
    bt      = st.session_state.bt_result
    if meaning is None:
        return

    st.markdown('<div class="ts-card-title">🔬 Meaning Preservation Analysis</div>',
                unsafe_allow_html=True)

    m_col, bt_col = st.columns(2, gap="medium")

    # ── Semantic Check ─────────────────────────────────────────────
    with m_col:
        st.markdown("**Semantic Meaning Check**")
        if meaning.get("error"):
            st.warning(f"⚠️ {meaning['explanation']}")
        else:
            score  = meaning["score"]
            status = meaning["status"]
            emoji  = status_emoji(status)
            color  = score_to_color(score)

            css = ("ts-status-green" if score >= 75 else
                   "ts-status-yellow" if score >= 55 else "ts-status-red")
            st.markdown(
                f'<div style="font-size:1.15rem;margin-bottom:.4rem">'
                f'{emoji} <span class="{css}">{status}</span></div>',
                unsafe_allow_html=True,
            )
            st.markdown(
                f'<div class="ts-score-bar-bg">'
                f'<div class="ts-score-bar-fill" style="width:{score}%;background:{color}"></div>'
                f'</div>'
                f'<div style="font-size:.85rem;color:var(--text-secondary);margin-bottom:.5rem">'
                f'Meaning Preservation Score: <strong style="color:{color}">{score}%</strong>'
                f'</div>',
                unsafe_allow_html=True,
            )
            st.caption(meaning["explanation"])

            # Detail metrics
            with st.expander("📊 Detail metrics"):
                d1, d2, d3, d4 = st.columns(4)
                d1.metric("Cosine", f"{int(meaning.get('cosine',0)*100)}%")
                d2.metric("Overlap", f"{int(meaning.get('jaccard',0)*100)}%")
                d3.metric("Coverage", f"{int(meaning.get('coverage',0)*100)}%")
                d4.metric("Numbers", f"{int(meaning.get('num_score',0)*100)}%")

            flags = []
            if meaning["facts_changed"]:
                flags.append("⚠️ Numbers or entities may have changed")
            if meaning["info_removed"]:
                flags.append("📭 Some key concepts may be missing")
            if meaning["info_added"]:
                flags.append("➕ New content was introduced")
            for f in flags:
                st.warning(f, icon=None)

    # ── Back-Translation ───────────────────────────────────────────
    with bt_col:
        st.markdown("**Back-Translation Verification**")
        if bt is None or bt.get("error"):
            st.warning(f"⚠️ {bt['summary'] if bt else 'Unavailable.'}")
        else:
            score = bt["score"]
            color = score_to_color(score)
            st.markdown(
                f'<div class="ts-score-bar-bg">'
                f'<div class="ts-score-bar-fill" style="width:{score}%;background:{color}"></div>'
                f'</div>'
                f'<div style="font-size:.85rem;color:var(--text-secondary);margin-bottom:.5rem">'
                f'Back-Translation Score: <strong style="color:{color}">{score}%</strong>'
                f'</div>',
                unsafe_allow_html=True,
            )
            if bt["warning"]:
                st.warning(
                    "⚠️ **Warning:** The rewrite may have shifted key parts of the original meaning.",
                    icon=None,
                )
            st.caption(bt["summary"])
            if bt["differences"]:
                with st.expander("🔍 Detected Differences"):
                    for d in bt["differences"]:
                        st.markdown(f"• {d}")
            with st.expander("📄 Neutral Reconstruction"):
                st.text(bt["neutral_version"] or "Not available.")


# ─────────────────────────────────────────────────────────────────────
# TAB 1 — Rewrite
# ─────────────────────────────────────────────────────────────────────
def _tab_rewrite():
    left_col, right_col = st.columns([1, 1], gap="large")

    with left_col:
        input_text, tone, audience, length, formality = _input_panel()

        st.markdown("<br>", unsafe_allow_html=True)
        disabled = is_empty(input_text) or is_too_long(input_text, MAX_CHARS)

        if st.button("✨ Rewrite Text", key="rewrite_btn",
                     use_container_width=True, disabled=disabled):
            _do_rewrite(input_text, tone, audience, length, formality)
            st.rerun()

        if disabled and not is_empty(input_text):
            st.error(f"❌ Text too long ({count_characters(input_text)}/{MAX_CHARS} chars).")
        elif disabled:
            st.info("💡 Enter some text above to get started.")

    with right_col:
        _output_panel()

    if st.session_state.rewrite_result:
        st.markdown("---")
        _render_comparison()
        st.markdown("---")
        _render_meaning_analysis()


# ─────────────────────────────────────────────────────────────────────
# TAB 2 — Compare Tones
# ─────────────────────────────────────────────────────────────────────
def _tab_compare_tones():
    st.markdown('<div class="ts-card-title">🎨 Compare Tones</div>', unsafe_allow_html=True)
    st.markdown(
        "Generate up to **4 versions** of your text in different tones and compare side by side."
    )

    original = st.session_state.last_input or ""

    if not original:
        st.info("💡 Go to the **Rewrite** tab, enter your text, then return here.")
        compare_input = st.text_area(
            "Or enter text here:",
            placeholder="Paste your text to compare across tones…",
            height=140, key="compare_input",
        )
        if compare_input.strip():
            original = compare_input.strip()
    else:
        st.markdown(
            f'<div class="ts-compare-box" style="margin-bottom:1rem">'
            f'{original[:400]}{"…" if len(original)>400 else ""}</div>',
            unsafe_allow_html=True,
        )

    if not original:
        return

    sel_col, aud_col = st.columns([3, 1])
    with sel_col:
        st.markdown('<div class="ts-label">Select up to 4 tones:</div>', unsafe_allow_html=True)
        selected_tones = st.multiselect(
            "Tones", TONES,
            default=["Formal", "Casual", "Professional", "Child-Friendly"],
            max_selections=4, label_visibility="collapsed", key="compare_tones_select",
        )
    with aud_col:
        st.markdown('<div class="ts-label">Audience:</div>', unsafe_allow_html=True)
        cmp_audience = st.selectbox(
            "CmpAud", AUDIENCES, index=0,
            label_visibility="collapsed", key="compare_audience",
        )

    if len(selected_tones) < 2:
        st.warning("Please select at least 2 tones.")
        return

    if st.button("🎨 Generate Tone Comparison", key="gen_compare_btn"):
        results = {}
        prog = st.progress(0, text="Generating comparisons…")
        for i, tone in enumerate(selected_tones):
            prog.progress(i / len(selected_tones), text=f"Rewriting in **{tone}** style…")
            try:
                results[tone] = rewrite(original, tone, cmp_audience, "Same Length", DEFAULT_FORMALITY)
            except Exception as exc:
                results[tone] = f"❌ Error: {exc}"
        prog.progress(1.0, text="✅ Done!")
        st.session_state.tone_comparison = results

    if st.session_state.tone_comparison:
        st.markdown("---")
        st.markdown('<div class="ts-card-title">🗂️ Results</div>', unsafe_allow_html=True)
        comparison = st.session_state.tone_comparison
        tones_list = list(comparison.keys())
        for i in range(0, len(tones_list), 2):
            cols = st.columns(2, gap="medium")
            for j, col in enumerate(cols):
                if i + j < len(tones_list):
                    t   = tones_list[i + j]
                    txt = comparison[t]
                    with col:
                        st.markdown(
                            f'<div class="ts-tone-card">'
                            f'<div class="ts-tone-badge">{t}</div>'
                            f'<div style="font-size:.92rem;line-height:1.7;'
                            f'color:var(--text-primary)">{txt}</div></div>',
                            unsafe_allow_html=True,
                        )
                        st.download_button(
                            label=f"⬇️ Download {t}",
                            data=txt,
                            file_name=f"toneshift_{t.lower().replace(' ','_')}.txt",
                            mime="text/plain",
                            key=f"dl_cmp_{t}",
                            use_container_width=True,
                        )


# ─────────────────────────────────────────────────────────────────────
# TAB 3 — Meaning Check
# ─────────────────────────────────────────────────────────────────────
def _tab_meaning_check():
    st.markdown('<div class="ts-card-title">🔬 Meaning Preservation Check</div>',
                unsafe_allow_html=True)
    st.markdown(
        "Compare any two texts to see how well meaning, facts, and intent are preserved. "
        "Uses **TF-IDF cosine similarity**, **Jaccard word overlap**, and **entity/number retention**."
    )

    mc_l, mc_r = st.columns(2, gap="medium")
    with mc_l:
        st.markdown('<div class="ts-label">Original Text</div>', unsafe_allow_html=True)
        mc_orig = st.text_area(
            "MC Orig",
            value=st.session_state.last_input,
            placeholder="Paste the original text here…",
            height=200, label_visibility="collapsed", key="mc_original",
        )
    with mc_r:
        st.markdown('<div class="ts-label">Rewritten / Comparison Text</div>',
                    unsafe_allow_html=True)
        mc_rew = st.text_area(
            "MC Rew",
            value=st.session_state.rewrite_result or "",
            placeholder="Paste the text you want to check against the original…",
            height=200, label_visibility="collapsed", key="mc_rewritten",
        )

    if st.button("🔍 Run Meaning Analysis", key="mc_run_btn"):
        if is_empty(mc_orig) or is_empty(mc_rew):
            st.error("❌ Please provide both original and rewritten text.")
        else:
            with st.spinner("🔍 Analysing meaning…"):
                mc_result = check_meaning_preservation(mc_orig, mc_rew)
            with st.spinner("↩️ Running back-translation…"):
                bt_result  = run_back_translation(mc_orig, mc_rew)
            st.session_state.meaning_result = mc_result
            st.session_state.bt_result      = bt_result
            st.session_state.last_input     = mc_orig
            st.session_state.rewrite_result = mc_rew
            st.rerun()

    if st.session_state.meaning_result:
        st.markdown("---")
        _render_meaning_analysis()


# ─────────────────────────────────────────────────────────────────────
# TAB 4 — History
# ─────────────────────────────────────────────────────────────────────
def _tab_history():
    st.markdown('<div class="ts-card-title">📜 Rewrite History</div>', unsafe_allow_html=True)
    history = st.session_state.history

    if not history:
        st.info("📭 No history yet. Start rewriting in the **Rewrite** tab.")
        return

    info_col, clear_col = st.columns([4, 1])
    with info_col:
        st.caption(f"{len(history)} rewrite(s) this session")
    with clear_col:
        if st.button("🗑️ Clear All", key="clear_hist"):
            st.session_state.history = []
            st.rerun()

    for entry in history:
        score = entry["meaning_score"]
        color = score_to_color(score)
        em    = "🟢" if score >= 75 else "🟡" if score >= 55 else "🔴"

        with st.expander(
            f"[{entry['timestamp']}]  {entry['tone']} → {entry['audience']}  "
            f"|  {em} {score}%  |  \"{entry['original'][:55]}…\""
        ):
            hl, hr = st.columns(2, gap="medium")
            with hl:
                st.markdown("**Original**")
                st.text(entry["original"])
            with hr:
                st.markdown("**Rewritten**")
                st.text(entry["rewritten"])

            st.markdown(
                f'<div class="ts-badge-row">'
                f'<span class="ts-badge">🎭 {entry["tone"]}</span>'
                f'<span class="ts-badge">👥 {entry["audience"]}</span>'
                f'<span class="ts-badge ts-badge-neutral">📏 {entry["length"]}</span>'
                f'<span class="ts-badge ts-badge-neutral">'
                f'🎛️ {formality_label(entry["formality"])}</span>'
                f'<span class="ts-badge" style="border-color:{color};color:{color}">'
                f'Meaning: {score}%</span></div>',
                unsafe_allow_html=True,
            )

            if st.button("♻️ Restore", key=f"restore_{entry['id']}"):
                st.session_state.rewrite_result = entry["rewritten_full"]
                st.session_state.last_input     = entry["original_full"]
                st.session_state.last_settings  = {
                    "tone":     entry["tone"],
                    "audience": entry["audience"],
                    "length":   entry["length"],
                    "formality": entry["formality"],
                }
                st.toast("♻️ Result restored!", icon="✅")
                st.rerun()


# ─────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────
def main():
    _hero()

    tab_rw, tab_cmp, tab_mc, tab_hist = st.tabs([
        "✨ Rewrite",
        "🎨 Compare Tones",
        "🔬 Meaning Check",
        "📜 History",
    ])

    with tab_rw:
        _tab_rewrite()
    with tab_cmp:
        _tab_compare_tones()
    with tab_mc:
        _tab_meaning_check()
    with tab_hist:
        _tab_history()


if __name__ == "__main__":
    main()
