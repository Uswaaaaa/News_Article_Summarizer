import streamlit as st
from model import (
    load_model,
    get_raw_article_trafilatura,
    process_direct_text,
    process_article_with_refinement,
    summarize_text,
    generate_final_summary,
    clean_generated_summary,
)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="News Article Summarizer", layout="wide")

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
html, body, [data-testid="stAppViewContainer"], [data-testid="stApp"] {
    background-color: #100D2E !important;
}
[data-testid="stHeader"] { background-color: #100D2E !important; }

h1 {
    color: #ffffff;
    text-align: center;
    font-size: 2.4rem;
    font-weight: 800;
    margin-bottom: 1.2rem;
}
.panel {
    background: #D9D9D9;
    border-radius: 8px;
    padding: 16px;
    height: 340px;
    overflow-y: auto;
    color: #111;
    font-size: 0.95rem;
    line-height: 1.6;
    white-space: pre-wrap;
    word-break: break-word;
}
.panel-title {
    font-weight: 700;
    font-size: 1.1rem;
    text-align: center;
    margin-bottom: 10px;
    color: #111;
}
.placeholder-text { color: #666; }
.info-box {
    background: #1e1a4a;
    border-left: 4px solid #8B7FD4;
    border-radius: 6px;
    padding: 10px 14px;
    color: #ccc;
    font-size: 0.85rem;
    margin-bottom: 10px;
}
div[data-testid="stButton"] > button {
    background-color: #8B7FD4;
    color: #ffffff;
    font-weight: 700;
    font-size: 1.05rem;
    border: none;
    border-radius: 8px;
    padding: 0.65rem 2.5rem;
    cursor: pointer;
    transition: background 0.2s;
}
div[data-testid="stButton"] > button:hover {
    background-color: #7063C0;
    color: #ffffff;
}
.clear-wrap div[data-testid="stButton"] > button {
    background-color: #D9D9D9 !important;
    color: #111 !important;
    border: 1px solid #aaa !important;
    border-radius: 4px !important;
    font-size: 0.85rem !important;
    padding: 0.28rem 1rem !important;
    font-weight: 400 !important;
}
.clear-wrap div[data-testid="stButton"] > button:hover {
    background-color: #c4c4c4 !important;
}
[data-testid="stSelectbox"] > div > div {
    background-color: #D9D9D9;
    border-radius: 4px;
    color: #111;
}
textarea, [data-baseweb="input"] input {
    background-color: #D9D9D9 !important;
    color: #111 !important;
}
label { color: #ccc !important; font-size: 0.85rem; }
</style>
""", unsafe_allow_html=True)

# ── Cached model loader ───────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading BART model… (first run only, ~1–2 min)")
def get_model():
    return load_model()

# ── Full pipeline ─────────────────────────────────────────────────────────────
def run_full_pipeline(input_mode, text_input, url_input, progress_placeholder):
    bart_tokenizer, summarizer_tok, summarizer_model = get_model()

    progress_placeholder.markdown('<div class="info-box">📡 Acquiring article…</div>', unsafe_allow_html=True)
    if input_mode == "News URL":
        article_data, err = get_raw_article_trafilatura(url_input.strip())
        if err:
            return None, f"❌ {err}"
    else:
        article_data, err = process_direct_text(text_input.strip())
        if err:
            return None, f"❌ {err}"

    progress_placeholder.markdown('<div class="info-box">🔬 Normalising & chunking text…</div>', unsafe_allow_html=True)
    article_data = process_article_with_refinement(article_data, bart_tokenizer)

    qf = article_data["refinement"]["quality_filter"]
    if not qf["suitable"]:
        return None, f"❌ Article not suitable: {qf['reason']}"

    token_count = article_data["refinement"]["token_analysis"]["token_count"]

    if token_count < 700:
        progress_placeholder.markdown('<div class="info-box">✍️ Summarising (short article)…</div>', unsafe_allow_html=True)
        final_summary = summarize_text(article_data["full_cleaned_text"], summarizer_tok, summarizer_model)
    else:
        all_summaries = []
        n = len(article_data["refined_chunks"])
        for i, chunk in enumerate(article_data["refined_chunks"]):
            progress_placeholder.markdown(
                f'<div class="info-box">✍️ Summarising chunk {i+1}/{n}…</div>',
                unsafe_allow_html=True)
            s = summarize_text(chunk, summarizer_tok, summarizer_model)
            all_summaries.append(clean_generated_summary(s))
        combined = "\n".join(list(dict.fromkeys(all_summaries)))
        progress_placeholder.markdown('<div class="info-box">🔗 Generating final combined summary…</div>', unsafe_allow_html=True)
        final_summary = generate_final_summary(combined, summarizer_tok, summarizer_model)

    final_summary = clean_generated_summary(final_summary)
    progress_placeholder.empty()
    return article_data["full_cleaned_text"], final_summary

# ── Session state ─────────────────────────────────────────────────────────────
for key, default in {
    "summary": None,
    "article_text": "",
    "url_input": "",
    "input_mode": "Raw Article Text",
    "display_text": "",
    "error_msg": None,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

# ── Title ─────────────────────────────────────────────────────────────────────
st.markdown("<h1>News Article Summarizer</h1>", unsafe_allow_html=True)

generated = st.session_state.summary is not None

# ── Dropdown (hidden after generation) ───────────────────────────────────────
if not generated:
    st.session_state.input_mode = st.selectbox(
        label="Input mode",
        options=["Raw Article Text", "News URL"],
        index=["Raw Article Text", "News URL"].index(st.session_state.input_mode),
        label_visibility="collapsed",
    )

# ── Two-column layout ─────────────────────────────────────────────────────────
left_col, right_col = st.columns(2, gap="medium")

with left_col:
    if not generated:
        if st.session_state.input_mode == "Raw Article Text":
            st.session_state.article_text = st.text_area(
                label="article", value=st.session_state.article_text,
                placeholder="Insert Raw Article Text",
                height=280, label_visibility="collapsed")
        else:
            st.session_state.url_input = st.text_input(
                label="url", value=st.session_state.url_input,
                placeholder="Paste News Article URL here…",
                label_visibility="collapsed")
    else:
        display = st.session_state.display_text or st.session_state.article_text or st.session_state.url_input
        safe = display.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace("\n", "<br>")
        st.markdown(f'<div class="panel">{safe}</div>', unsafe_allow_html=True)

    st.markdown('<div class="clear-wrap">', unsafe_allow_html=True)
    if st.button("Clear"):
        for k in ["summary", "article_text", "url_input", "display_text", "error_msg"]:
            st.session_state[k] = None if k in ("summary", "error_msg") else ""
        st.session_state.input_mode = "Raw Article Text"
        st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)

with right_col:
    if st.session_state.error_msg:
        st.markdown(
            f'<div class="panel"><div class="panel-title">Summary</div>'
            f'<span style="color:#e55;">{st.session_state.error_msg}</span></div>',
            unsafe_allow_html=True)
    elif st.session_state.summary:
        safe_sum = st.session_state.summary.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace("\n", "<br>")
        st.markdown(
            f'<div class="panel"><div class="panel-title">Summary</div>{safe_sum}</div>',
            unsafe_allow_html=True)
    else:
        st.markdown(
            '<div class="panel"><div class="panel-title">Summary</div>'
            '<span class="placeholder-text">Your summary will appear here.</span></div>',
            unsafe_allow_html=True)

# ── Generate button (hidden after generation) ─────────────────────────────────
if not generated:
    st.markdown("<br>", unsafe_allow_html=True)
    _, btn_col, _ = st.columns([1, 1, 1])
    with btn_col:
        if st.button("Generate Summary", use_container_width=True):
            raw = (st.session_state.url_input if st.session_state.input_mode == "News URL"
                   else st.session_state.article_text).strip()
            if not raw:
                st.warning("Please enter some article text or a valid URL first.")
                st.stop()

            progress_area = st.empty()
            cleaned_text, result = run_full_pipeline(
                st.session_state.input_mode,
                st.session_state.article_text,
                st.session_state.url_input,
                progress_area,
            )
            if cleaned_text is None:
                st.session_state.error_msg = result
                st.session_state.summary   = "ERROR"
            else:
                st.session_state.summary      = result
                st.session_state.display_text = cleaned_text
                st.session_state.error_msg    = None
            st.rerun()
