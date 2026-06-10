# ── Imports & Setup ───────────────────────────────────────────────────────────
import re
import contractions
import nltk
from nltk.tokenize import sent_tokenize
from transformers import AutoTokenizer, BartForConditionalGeneration, BartTokenizer
import textwrap

nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)

# ── Load Models ───────────────────────────────────────────────────────────────
def load_model():
    """
    Loads and returns the BART tokenizer and model.
    Called once by Streamlit's @st.cache_resource.
    """
    bart_tokenizer   = AutoTokenizer.from_pretrained("facebook/bart-base")
    summarizer_tok   = BartTokenizer.from_pretrained("facebook/bart-large-cnn")
    summarizer_model = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn")
    return bart_tokenizer, summarizer_tok, summarizer_model

# ── Section A: Web Scraper & Text Chunking ────────────────────────────────────

def _clean_and_chunk_text(text, chunk_size=200):
    """Normalizes raw text spacing and splits it into manageable chunks."""
    clean_text = " ".join(text.split())
    words = clean_text.split()
    chunks = [" ".join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]
    return clean_text, chunks


def chunk_by_sentences_and_tokens(sentences: list, tokenizer, max_tokens: int = 1024, overlap_sentences: int = 3) -> list:
    """
    Chunks a list of sentences into larger segments, respecting a maximum token
    limit and providing overlap between chunks.
    """
    if not sentences:
        return []

    chunks = []
    current_chunk_sentences = []
    current_token_count = 0

    for sentence in sentences:
        sentence_tokens = tokenizer.encode(sentence, add_special_tokens=False)
        sentence_token_count = len(sentence_tokens)

        if current_token_count + sentence_token_count > max_tokens and current_chunk_sentences:
            chunks.append(" ".join(current_chunk_sentences))
            overlap_start_index = max(0, len(current_chunk_sentences) - overlap_sentences)
            current_chunk_sentences = list(current_chunk_sentences[overlap_start_index:])
            current_token_count = len(tokenizer.encode(" ".join(current_chunk_sentences), add_special_tokens=False))

        current_chunk_sentences.append(sentence)
        current_token_count += sentence_token_count

    if current_chunk_sentences:
        chunks.append(" ".join(current_chunk_sentences))

    return chunks


def get_raw_article_trafilatura(url, chunk_size=200):
    """
    Scrapes a news article from a given URL using Trafilatura.
    Uses BeautifulSoup as a fallback if the title is missing.
    Returns (article_data_dict, error_message). One of them will be None.
    """
    import trafilatura
    import requests
    from bs4 import BeautifulSoup

    downloaded = trafilatura.fetch_url(url)
    if downloaded is None:
        return None, "Could not access the website."

    extracted_data = trafilatura.bare_extraction(downloaded)
    if extracted_data is None:
        return None, "Downloaded page but could not extract content."

    if hasattr(extracted_data, "text"):
        raw_text      = extracted_data.text
        article_title = extracted_data.title
    elif isinstance(extracted_data, dict):
        raw_text      = extracted_data.get("text")
        article_title = extracted_data.get("title")
    else:
        raw_text      = None
        article_title = None

    if not raw_text:
        return None, "Downloaded page but found no text."

    if not article_title:
        try:
            response = requests.get(url, timeout=5)
            soup = BeautifulSoup(response.content, "html.parser")
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
    """Processes raw user-provided text through the cleaning and chunking pipeline."""
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

def normalize_text(text: str) -> str:
    """Replaces non-ASCII punctuation and expands contractions."""
    replacements = {
        "\u2018": "'", "\u2019": "'", "\u201c": '"', "\u201d": '"',
        "\u2014": "-", "\u2013": "-", "\u2026": "...", "\u2022": "*"
    }
    for bad, good in replacements.items():
        text = text.replace(bad, good)
    text = contractions.fix(text)
    text = re.sub(r"[^\x00-\x7F]+", " ", text)
    return text


def segment_sentences(text: str) -> list:
    """Splits normalized text into a list of sentences using NLTK."""
    return sent_tokenize(text)


def analyze_tokenization(text: str, tokenizer, model_max_length: int = 1024) -> dict:
    """Counts tokens with BART tokenizer and checks if text exceeds model limit."""
    tokens      = tokenizer.encode(text, add_special_tokens=True)
    token_count = len(tokens)
    exceeds     = token_count > model_max_length
    warning     = (f"Text has {token_count} tokens, exceeding model limit of {model_max_length}."
                   if exceeds else None)
    return {"token_count": token_count, "exceeds_limit": exceeds, "warning": warning}


JUNK_PHRASES = [
    "cookie policy", "privacy policy", "terms of service", "sign up",
    "subscribe now", "advertisement", "all rights reserved",
    "this site uses cookies", "download our app", "newsletter"
]

def quality_filter(text: str, min_words: int = 50) -> dict:
    """Flags text as unsuitable if too short or contains junk phrases."""
    word_count = len(text.split())
    if word_count < min_words:
        return {"suitable": False, "reason": f"Too short: {word_count} words (minimum {min_words})"}
    lower_text = text.lower()
    for junk in JUNK_PHRASES:
        if junk in lower_text:
            return {"suitable": False, "reason": f"Contains junk phrase: '{junk}'"}
    return {"suitable": True, "reason": None}


def process_article_with_refinement(article_data: dict, bart_tokenizer, max_chunk_tokens: int = 1024, overlap_sents: int = 3) -> dict:
    """
    Takes the output of get_raw_article_trafilatura or process_direct_text
    and adds NLP refinements including advanced chunking.
    """
    raw_cleaned_text = article_data.get("full_cleaned_text", "")
    if not raw_cleaned_text:
        article_data["refinement"] = {"error": "No text to refine"}
        return article_data

    normalized_text = normalize_text(raw_cleaned_text)
    sentences       = segment_sentences(normalized_text)
    token_info      = analyze_tokenization(normalized_text, bart_tokenizer)
    quality         = quality_filter(normalized_text)

    if bart_tokenizer:
        refined_chunks = chunk_by_sentences_and_tokens(
            sentences=sentences,
            tokenizer=bart_tokenizer,
            max_tokens=max_chunk_tokens,
            overlap_sentences=overlap_sents
        )
    else:
        _, refined_chunks = _clean_and_chunk_text(raw_cleaned_text, chunk_size=200)

    article_data["refinement"] = {
        "normalized_text": normalized_text,
        "sentences": sentences,
        "num_sentences": len(sentences),
        "token_analysis": token_info,
        "quality_filter": quality,
    }
    article_data["refined_chunks"]      = refined_chunks
    article_data["num_chunks"]          = len(refined_chunks)
    article_data["full_cleaned_text"]   = normalized_text

    return article_data


# ── Section B: Summarization ──────────────────────────────────────────────────

def summarize_text(text: str, summarizer_tok, summarizer_model) -> str:
    """First-stage abstractive summarization using BART with a guided prompt."""
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
    summary_ids = summarizer_model.generate(
        inputs["input_ids"],
        num_beams=8,
        no_repeat_ngram_size=4,
        repetition_penalty=1.3,
        length_penalty=1.8,
        max_length=180,
        min_length=90,
        early_stopping=True
    )
    return summarizer_tok.decode(summary_ids[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)


def generate_final_summary(text: str, summarizer_tok, summarizer_model) -> str:
    """Second-pass summarization that combines chunk summaries into one coherent report."""
    final_prompt = f"""Create a complete and coherent news report summary.

The summary should clearly explain:
- who was involved
- what happened
- where and when it happened
- why it matters
- important outcomes or legal consequences

Text:
{text}
"""
    inputs = summarizer_tok(final_prompt, max_length=1024, truncation=True, return_tensors="pt")
    summary_ids = summarizer_model.generate(
        inputs["input_ids"],
        num_beams=8,
        no_repeat_ngram_size=3,
        repetition_penalty=1.2,
        length_penalty=2.0,
        max_length=220,
        min_length=100,
        early_stopping=True
    )
    return summarizer_tok.decode(summary_ids[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)


def clean_generated_summary(summary: str) -> str:
    """Post-processes the summary to remove banned phrases, duplicates, and broken punctuation."""
    banned_phrases = [
        "For confidential support", "Samaritans", "click here",
        "call the hotline", "suicide prevention"
    ]
    for phrase in banned_phrases:
        summary = summary.replace(phrase, "")

    summary = re.sub(r"\s+", " ", summary)
    summary = re.sub(r"\.\s*\.", ".", summary)

    sentences        = summary.split(". ")
    unique_sentences = []
    for sent in sentences:
        sent = sent.strip()
        if sent not in unique_sentences:
            unique_sentences.append(sent)

    return ". ".join(unique_sentences).strip()
