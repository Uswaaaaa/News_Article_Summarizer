import streamlit as st
from model import (
    load_model, load_pegasus,
    get_raw_article_trafilatura, process_direct_text,
    process_article_with_refinement,
    summarize_text, generate_final_summary, clean_generated_summary,
    pegasus_summarize,
    calculate_rouge_scores, compare_rouge,
)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="News Article Summarizer", layout="wide")

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
html, body, [data-testid="stAppViewContainer"], [data-testid="stApp"] {
    background-color: #100D2E !important;
}
[data-testid="stHeader"] { background-color: #100D2E !important; }

h1 {
    color: #ffffff; text-align: center;
    font-size: 2.4rem; font-weight: 800; margin-bottom: 1.2rem;
}
h3 { color: #ffffff !important; }

/* panels */
.panel {
    background: #D9D9D9; border-radius: 8px; padding: 16px;
    height: 300px; overflow-y: auto; color: #111;
    font-size: 0.92rem; line-height: 1.6;
    white-space: pre-wrap; word-break: break-word;
}
.panel-title {
    font-weight: 700; font-size: 1.05rem;
    text-align: center; margin-bottom: 10px; color: #111;
}
.placeholder-text { color: #666; }

/* info box */
.info-box {
    background: #1e1a4a; border-left: 4px solid #8B7FD4;
    border-radius: 6px; padding: 10px 14px;
    color: #ccc; font-size: 0.85rem; margin-bottom: 10px;
}

/* ROUGE table */
.rouge-table {
    width: 100%; border-collapse: collapse;
    color: #eee; font-size: 0.88rem; margin-top: 8px;
}
.rouge-table th {
    background: #2a2560; padding: 8px 12px;
    text-align: center; border: 1px solid #3a3580;
}
.rouge-table td {
    padding: 7px 12px; text-align: center;
    border: 1px solid #2a2560;
}
.rouge-table tr:nth-child(even) { background: #1a1740; }
.rouge-table tr:nth-child(odd)  { background: #141230; }
.winner-bart    { color: #6EE7B7; font-weight: 700; }
.winner-pegasus { color: #FCA5A5; font-weight: 700; }
.winner-tie     { color: #FCD34D; font-weight: 700; }

/* summary comparison boxes */
.sum-box {
    background: #1e1a4a; border-radius: 8px;
    padding: 14px; color: #e2e2f0;
    font-size: 0.9rem; line-height: 1.65;
    min-height: 160px; border: 1px solid #3a3580;
}
.sum-label {
    font-weight: 700; font-size: 0.8rem;
    text-transform: uppercase; letter-spacing: 1px;
    margin-bottom: 8px;
}
.bart-label    { color: #6EE7B7; }
.pegasus-label { color: #FCA5A5; }

/* buttons */
div[data-testid="stButton"] > button {
    background-color: #8B7FD4; color: #ffffff;
    font-weight: 700; font-size: 1.05rem;
    border: none; border-radius: 8px;
    padding: 0.65rem 2.5rem; transition: background 0.2s;
}
div[data-testid="stButton"] > button:hover { background-color: #7063C0; color: #ffffff; }

.clear-wrap div[data-testid="stButton"] > button {
    background-color: #D9D9D9 !important; color: #111 !important;
    border: 1px solid #aaa !important; border-radius: 4px !important;
    font-size: 0.85rem !important; padding: 0.28rem 1rem !important;
    font-weight: 400 !important;
}
.clear-wrap div[data-testid="stButton"] > button:hover { background-color: #c4c4c4 !important; }

[data-testid="stSelectbox"] > div > div { background-color: #D9D9D9; border-radius: 4px; color: #111; }
textarea, [data-baseweb="input"] input { background-color: #D9D9D9 !important; color: #111 !important; }
label { color: #ccc !important; font-size: 0.85rem; }
</style>
""", unsafe_allow_html=True)

# ── Cached loaders ────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading BART model… (first run only)")
def get_bart():
    return load_model()

@st.cache_resource(show_spinner="Loading PEGASUS model… (first run only)")
def get_pegasus():
    return load_pegasus()

# ── Full pipeline ─────────────────────────────────────────────────────────────
def run_pipeline(input_mode, article_text, url_input, progress):
    bart_tokenizer, summarizer_tok, summarizer_model = get_bart()
    pegasus_tokenizer, pegasus_model                = get_pegasus()

    # 1 — Acquire
    progress.markdown('<div class="info-box">📡 Acquiring article…</div>', unsafe_allow_html=True)
    if input_mode == "News URL":
        article_data, err = get_raw_article_trafilatura(url_input.strip())
        if err:
            return None, None, None, None, f"❌ {err}"
    else:
        article_data, err = process_direct_text(article_text.strip())
        if err:
            return None, None, None, None, f"❌ {err}"

    # 2 — Refine
    progress.markdown('<div class="info-box">🔬 Normalising & chunking…</div>', unsafe_allow_html=True)
    article_data = process_article_with_refinement(article_data, bart_tokenizer)

    qf = article_data["refinement"]["quality_filter"]
    if not qf["suitable"]:
        return None, None, None, None, f"❌ Article not suitable: {qf['reason']}"

    token_count = article_data["refinement"]["token_analysis"]["token_count"]
    full_text   = article_data["full_cleaned_text"]

    # 3 — BART summarise
    if token_count < 700:
        progress.markdown('<div class="info-box">✍️ BART: summarising (short article)…</div>', unsafe_allow_html=True)
        bart_summary = summarize_text(full_text, summarizer_tok, summarizer_model)
    else:
        all_sums = []
        n = len(article_data["refined_chunks"])
        for i, chunk in enumerate(article_data["refined_chunks"]):
            progress.markdown(
                f'<div class="info-box">✍️ BART: summarising chunk {i+1}/{n}…</div>',
                unsafe_allow_html=True)
            s = summarize_text(chunk, summarizer_tok, summarizer_model)
            all_sums.append(clean_generated_summary(s))
        combined     = "\n".join(list(dict.fromkeys(all_sums)))
        progress.markdown('<div class="info-box">🔗 BART: generating final summary…</div>', unsafe_allow_html=True)
        bart_summary = generate_final_summary(combined, summarizer_tok, summarizer_model)

    bart_summary = clean_generated_summary(bart_summary)

    # 4 — PEGASUS summarise
    progress.markdown('<div class="info-box">🤖 PEGASUS: generating summary…</div>', unsafe_allow_html=True)
    peg_summary = pegasus_summarize(full_text, pegasus_tokenizer, pegasus_model)

    # 5 — ROUGE
    progress.markdown('<div class="info-box">📊 Calculating ROUGE scores…</div>', unsafe_allow_html=True)
    bart_rouge   = calculate_rouge_scores(full_text, bart_summary)
    pegasus_rouge = calculate_rouge_scores(full_text, peg_summary)
    comparison   = compare_rouge(bart_rouge, pegasus_rouge)

    progress.empty()
    return full_text, bart_summary, peg_summary, comparison, None

# ── Session state ─────────────────────────────────────────────────────────────
for key, default in {
    "bart_summary": None, "peg_summary": None,
    "comparison": None,   "display_text": "",
    "article_text": "",   "url_input": "",
    "input_mode": "Raw Article Text",
    "error_msg": None,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

# ── Title ─────────────────────────────────────────────────────────────────────
st.markdown("<h1>News Article Summarizer</h1>", unsafe_allow_html=True)

generated = st.session_state.bart_summary is not None

# ── Dropdown ──────────────────────────────────────────────────────────────────
if not generated:
    st.session_state.input_mode = st.selectbox(
        label="Input mode",
        options=["Raw Article Text", "News URL"],
        index=["Raw Article Text","News URL"].index(st.session_state.input_mode),
        label_visibility="collapsed",
    )

# ── Input / Article panel + Summary panels ────────────────────────────────────
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
        safe = (st.session_state.display_text
                .replace("&","&amp;").replace("<","&lt;").replace(">","&gt;")
                .replace("\n","<br>"))
        st.markdown(f'<div class="panel">{safe}</div>', unsafe_allow_html=True)

    st.markdown('<div class="clear-wrap">', unsafe_allow_html=True)
    if st.button("Clear"):
        for k in ["bart_summary","peg_summary","comparison","display_text",
                  "article_text","url_input","error_msg"]:
            st.session_state[k] = None if k in ("bart_summary","peg_summary","comparison","error_msg") else ""
        st.session_state.input_mode = "Raw Article Text"
        st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)

with right_col:
    if st.session_state.error_msg:
        st.markdown(
            f'<div class="panel"><div class="panel-title">Summary</div>'
            f'<span style="color:#e55;">{st.session_state.error_msg}</span></div>',
            unsafe_allow_html=True)
    elif st.session_state.bart_summary:
        safe_b = (st.session_state.bart_summary
                  .replace("&","&amp;").replace("<","&lt;").replace(">","&gt;")
                  .replace("\n","<br>"))
        st.markdown(
            f'<div class="panel"><div class="panel-title">BART Summary</div>{safe_b}</div>',
            unsafe_allow_html=True)
    else:
        st.markdown(
            '<div class="panel"><div class="panel-title">Summary</div>'
            '<span class="placeholder-text">Your summary will appear here.</span></div>',
            unsafe_allow_html=True)

# ── Generate button ───────────────────────────────────────────────────────────
if not generated:
    st.markdown("<br>", unsafe_allow_html=True)
    _, btn_col, _ = st.columns([1,1,1])
    with btn_col:
        if st.button("Generate Summary", use_container_width=True):
            raw = (st.session_state.url_input if st.session_state.input_mode == "News URL"
                   else st.session_state.article_text).strip()
            if not raw:
                st.warning("Please enter some article text or a valid URL first.")
                st.stop()

            progress_area = st.empty()
            full_text, bart_sum, peg_sum, comparison, err = run_pipeline(
                st.session_state.input_mode,
                st.session_state.article_text,
                st.session_state.url_input,
                progress_area,
            )
            if err:
                st.session_state.error_msg    = err
                st.session_state.bart_summary = "ERROR"
            else:
                st.session_state.bart_summary = bart_sum
                st.session_state.peg_summary  = peg_sum
                st.session_state.comparison   = comparison
                st.session_state.display_text = full_text
                st.session_state.error_msg    = None
            st.rerun()

# ── Results section (shown after generation) ──────────────────────────────────
if generated and st.session_state.bart_summary != "ERROR":
    st.markdown("---")

    # ── Side-by-side summaries ────────────────────────────────────────────────
    st.markdown("### 📝 Summary Comparison")
    b_col, p_col = st.columns(2, gap="medium")

    with b_col:
        safe_b = (st.session_state.bart_summary or "")
        st.markdown(
            f'<div class="sum-box">'
            f'<div class="sum-label bart-label">🟢 BART (facebook/bart-large-cnn)</div>'
            f'{safe_b}</div>',
            unsafe_allow_html=True)

    with p_col:
        safe_p = (st.session_state.peg_summary or "")
        st.markdown(
            f'<div class="sum-box">'
            f'<div class="sum-label pegasus-label">🔴 PEGASUS (google/pegasus-xsum)</div>'
            f'{safe_p}</div>',
            unsafe_allow_html=True)

    # ── ROUGE comparison table ────────────────────────────────────────────────
    st.markdown("### 📊 ROUGE Score Comparison")

    if st.session_state.comparison:
        cmp = st.session_state.comparison
        metric_labels = {"rouge1":"ROUGE-1","rouge2":"ROUGE-2","rougeL":"ROUGE-L"}

        rows = ""
        for key, label in metric_labels.items():
            d = cmp[key]
            w = d["winner"]
            winner_class = (
                "winner-bart"    if w == "BART"    else
                "winner-pegasus" if w == "PEGASUS" else
                "winner-tie"
            )
            winner_icon = (
                "🟢 BART"    if w == "BART"    else
                "🔴 PEGASUS" if w == "PEGASUS" else
                "🟡 Tie"
            )
            rows += f"""
            <tr>
                <td><b>{label}</b></td>
                <td>{d['bart_precision']:.4f}</td>
                <td>{d['bart_recall']:.4f}</td>
                <td>{d['bart_f1']:.4f}</td>
                <td>{d['pegasus_precision']:.4f}</td>
                <td>{d['pegasus_recall']:.4f}</td>
                <td>{d['pegasus_f1']:.4f}</td>
                <td class="{winner_class}">{winner_icon}</td>
            </tr>"""

        table_html = f"""
        <table class="rouge-table">
            <thead>
                <tr>
                    <th>Metric</th>
                    <th>BART P</th>
                    <th>BART R</th>
                    <th>BART F1</th>
                    <th>PEGASUS P</th>
                    <th>PEGASUS R</th>
                    <th>PEGASUS F1</th>
                    <th>Winner</th>
                </tr>
            </thead>
            <tbody>{rows}</tbody>
        </table>"""
        st.markdown(table_html, unsafe_allow_html=True)

        # ── Interpretation ────────────────────────────────────────────────────
        st.markdown("<br>", unsafe_allow_html=True)
        bart_wins = sum(1 for k in cmp if cmp[k]["winner"] == "BART")
        peg_wins  = sum(1 for k in cmp if cmp[k]["winner"] == "PEGASUS")

        if bart_wins > peg_wins:
            verdict = "🟢 **BART** outperforms PEGASUS on this article based on F1-Score."
        elif peg_wins > bart_wins:
            verdict = "🔴 **PEGASUS** outperforms BART on this article based on F1-Score."
        else:
            verdict = "🟡 **Both models** performed equally on this article."

        st.markdown(
            f'<div class="info-box">'
            f'<b>Verdict:</b> {verdict}<br><br>'
            f'<b>Precision</b> measures how much of the summary is accurate and relevant to the original article.<br>'
            f'<b>Recall</b> measures how much of the original article content is captured in the summary.<br>'
            f'<b>F1-Score</b> is the harmonic mean of Precision and Recall — the overall quality score.'
            f'</div>',
            unsafe_allow_html=True)
