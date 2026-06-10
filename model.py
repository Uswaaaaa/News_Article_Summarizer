import re
import contractions
import nltk
from nltk.tokenize import sent_tokenize
from transformers import (
    AutoTokenizer,
    BartForConditionalGeneration,
    BartTokenizer,
    AutoModelForSeq2SeqLM,
)
import textwrap

nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)

# ── Load Models ───────────────────────────────────────────────────────────────
def load_model():
    """Load BART tokenizer and model."""
    bart_tokenizer   = AutoTokenizer.from_pretrained("facebook/bart-base")
    summarizer_tok   = BartTokenizer.from_pretrained("facebook/bart-large-cnn")
    summarizer_model = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn")
    return bart_tokenizer, summarizer_tok, summarizer_model

def load_pegasus():
    """Load PEGASUS tokenizer and model (google/pegasus-xsum — matches notebook)."""
    baseline_tokenizer = AutoTokenizer.from_pretrained("google/pegasus-xsum")
    baseline_model     = AutoModelForSeq2SeqLM.from_pretrained("google/pegasus-xsum")
    return baseline_tokenizer, baseline_model

# ── Section A: Text Chunking ──────────────────────────────────────────────────
def _clean_and_chunk_text(text, chunk_size=200):
    clean_text = " ".join(text.split())
    words  = clean_text.split()
    chunks = [" ".join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]
    return clean_text, chunks

def chunk_by_sentences_and_tokens(sentences, tokenizer, max_tokens=1024, overlap_sentences=3):
    if not sentences:
        return []
    chunks, current_sents, current_count = [], [], 0
    for sentence in sentences:
        stc = len(tokenizer.encode(sentence, add_special_tokens=False))
        if current_count + stc > max_tokens and current_sents:
            chunks.append(" ".join(current_sents))
            overlap_start = max(0, len(current_sents) - overlap_sentences)
            current_sents = list(current_sents[overlap_start:])
            current_count = len(tokenizer.encode(" ".join(current_sents), add_special_tokens=False))
        current_sents.append(sentence)
        current_count += stc
    if current_sents:
        chunks.append(" ".join(current_sents))
    return chunks

def get_raw_article_trafilatura(url, chunk_size=200):
    import trafilatura, requests
    from bs4 import BeautifulSoup
    downloaded = trafilatura.fetch_url(url)
    if downloaded is None:
        return None, "Could not access the website."
    extracted = trafilatura.bare_extraction(downloaded)
    if extracted is None:
        return None, "Downloaded page but could not extract content."
    raw_text      = extracted.text  if hasattr(extracted, "text")  else extracted.get("text")
    article_title = extracted.title if hasattr(extracted, "title") else extracted.get("title")
    if not raw_text:
        return None, "Downloaded page but found no text."
    if not article_title:
        try:
            resp = requests.get(url, timeout=5)
            soup = BeautifulSoup(resp.content, "html.parser")
            article_title = soup.title.string.strip() if soup.title and soup.title.string else "No Title Found"
        except Exception:
            article_title = "No Title Found"
    clean_text, chunks = _clean_and_chunk_text(raw_text, chunk_size)
    return {
        "title": article_title,
        "full_cleaned_text": clean_text,
        "text_chunks": chunks,
        "top_image": "N/A",
        "original_text_length": len(clean_text.split()),
        "num_chunks": len(chunks),
    }, None

def process_direct_text(text_input, chunk_size=200):
    clean_text, chunks = _clean_and_chunk_text(text_input, chunk_size)
    return {
        "title": "User Provided Text",
        "full_cleaned_text": clean_text,
        "text_chunks": chunks,
        "top_image": "N/A",
        "original_text_length": len(clean_text.split()),
        "num_chunks": len(chunks),
    }, None

# ── Section A: NLP Refinement ─────────────────────────────────────────────────
def normalize_text(text):
    replacements = {"\u2018":"'","\u2019":"'","\u201c":'"',"\u201d":'"',"—":"-","–":"-","…":"...","•":"*"}
    for bad, good in replacements.items():
        text = text.replace(bad, good)
    text = contractions.fix(text)
    text = re.sub(r"[^\x00-\x7F]+"," ", text)
    return text

def segment_sentences(text):
    return sent_tokenize(text)

def analyze_tokenization(text, tokenizer, model_max_length=1024):
    tokens      = tokenizer.encode(text, add_special_tokens=True)
    token_count = len(tokens)
    exceeds     = token_count > model_max_length
    return {"token_count": token_count, "exceeds_limit": exceeds,
            "warning": f"Text has {token_count} tokens, exceeding limit of {model_max_length}." if exceeds else None}

JUNK_PHRASES = ["cookie policy","privacy policy","terms of service","sign up",
                "subscribe now","advertisement","all rights reserved",
                "this site uses cookies","download our app","newsletter"]

def quality_filter(text, min_words=50):
    wc = len(text.split())
    if wc < min_words:
        return {"suitable": False, "reason": f"Too short: {wc} words (minimum {min_words})"}
    for junk in JUNK_PHRASES:
        if junk in text.lower():
            return {"suitable": False, "reason": f"Contains junk phrase: '{junk}'"}
    return {"suitable": True, "reason": None}

def process_article_with_refinement(article_data, bart_tokenizer, max_chunk_tokens=1024, overlap_sents=3):
    raw = article_data.get("full_cleaned_text","")
    if not raw:
        article_data["refinement"] = {"error":"No text to refine"}
        return article_data
    normalized   = normalize_text(raw)
    sentences    = segment_sentences(normalized)
    token_info   = analyze_tokenization(normalized, bart_tokenizer)
    quality      = quality_filter(normalized)
    refined_chunks = chunk_by_sentences_and_tokens(sentences, bart_tokenizer, max_chunk_tokens, overlap_sents)
    article_data["refinement"] = {
        "normalized_text": normalized,
        "sentences": sentences,
        "num_sentences": len(sentences),
        "token_analysis": token_info,
        "quality_filter": quality,
    }
    article_data["refined_chunks"]    = refined_chunks
    article_data["num_chunks"]        = len(refined_chunks)
    article_data["full_cleaned_text"] = normalized
    return article_data

# ── Section B: BART Summarization ─────────────────────────────────────────────
def summarize_text(text, summarizer_tok, summarizer_model):
    guided_prompt = f"""You are a professional news editor.

Write a concise but informative news summary.

The summary MUST include:
- the main event
- the people involved
- what caused the event
- important consequences
- legal or government actions if mentioned

Focus on the MOST important facts first.

Avoid:
- repetition
- unnecessary legal wording
- copying sentences directly
- unimportant side details

News Article:
{text}

Professional News Summary:
"""
    inputs = summarizer_tok(guided_prompt, max_length=1024, truncation=True, return_tensors="pt")
    ids    = summarizer_model.generate(
        inputs["input_ids"], num_beams=8, no_repeat_ngram_size=4,
        repetition_penalty=1.3, length_penalty=1.8,
        max_length=180, min_length=90, early_stopping=True)
    return summarizer_tok.decode(ids[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)

def generate_final_summary(text, summarizer_tok, summarizer_model):
    prompt = f"""Create a complete and coherent news report summary.

The summary should clearly explain:
- who was involved
- what happened
- where and when it happened
- why it matters
- important outcomes or legal consequences

Text:
{text}
"""
    inputs = summarizer_tok(prompt, max_length=1024, truncation=True, return_tensors="pt")
    ids    = summarizer_model.generate(
        inputs["input_ids"], num_beams=8, no_repeat_ngram_size=3,
        repetition_penalty=1.2, length_penalty=2.0,
        max_length=220, min_length=100, early_stopping=True)
    return summarizer_tok.decode(ids[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)

def clean_generated_summary(summary):
    for phrase in ["For confidential support","Samaritans","click here","call the hotline","suicide prevention"]:
        summary = summary.replace(phrase,"")
    summary   = re.sub(r"\s+"," ", summary)
    summary   = re.sub(r"\.\s*\.",".", summary)
    sentences = summary.split(". ")
    unique    = []
    for s in sentences:
        s = s.strip()
        if s not in unique:
            unique.append(s)
    return ". ".join(unique).strip()

# ── Section C: PEGASUS Summarization ─────────────────────────────────────────
def pegasus_summarize(text, baseline_tokenizer, baseline_model):
    inputs = baseline_tokenizer(text, return_tensors="pt", truncation=True, max_length=1024)
    ids    = baseline_model.generate(
        inputs["input_ids"], num_beams=6, length_penalty=1.0,
        max_length=120, min_length=30, early_stopping=True)
    return baseline_tokenizer.decode(ids[0], skip_special_tokens=True)

# ── Section D: ROUGE Evaluation ───────────────────────────────────────────────
def calculate_rouge_scores(reference_text, generated_summary):
    from rouge_score import rouge_scorer as rs
    scorer = rs.RougeScorer(["rouge1","rouge2","rougeL"], use_stemmer=True)
    return scorer.score(reference_text, generated_summary)

def compare_rouge(bart_scores, pegasus_scores):
    """Returns a dict with per-metric comparison data."""
    comparison = {}
    for metric in ["rouge1","rouge2","rougeL"]:
        b = bart_scores[metric]
        p = pegasus_scores[metric]
        comparison[metric] = {
            "bart_precision":    round(b.precision,  4),
            "bart_recall":       round(b.recall,     4),
            "bart_f1":           round(b.fmeasure,   4),
            "pegasus_precision": round(p.precision,  4),
            "pegasus_recall":    round(p.recall,     4),
            "pegasus_f1":        round(p.fmeasure,   4),
            "diff_f1":           round(b.fmeasure - p.fmeasure, 4),
            "winner":           ("BART" if b.fmeasure > p.fmeasure
                                 else "PEGASUS" if p.fmeasure > b.fmeasure
                                 else "Tie"),
        }
    return comparison
