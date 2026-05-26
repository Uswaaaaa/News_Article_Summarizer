# 📰 News Article Summarizer

An abstractive news summarization web app built with **Streamlit** and **Facebook's BART model** (`facebook/bart-large-cnn`). It accepts either a raw news article text or a URL, processes it through an NLP pipeline, and generates a concise professional summary.

---

## ✨ Features

- 🔗 Accepts **News URL** or **Raw Article Text** as input
- 🧹 Full NLP preprocessing — normalization, contraction expansion, noise filtering
- 🔪 Smart sentence-based chunking with token overlap for long articles
- 🤖 Abstractive summarization using `facebook/bart-large-cnn`
- 🧼 Post-processing to remove hallucinations and duplicate sentences
- 📊 ROUGE metric evaluation (in notebook)
- 🎨 Clean dark-themed Streamlit UI

---

## 📁 Project Structure

```
NLP_News_Summarizer/
├── app.py                        # Streamlit UI
├── model.py                      # Full NLP pipeline & BART model
├── NLP_GROUP_ASSIGNMENT.ipynb  # Original Colab notebook
├── requirements.txt              # All dependencies
├── .gitignore                    # Files excluded from Git
└── README.md                     # This file
```

---

## 🚀 How to Run

### Option 1 — VS Code / Local Machine

**Step 1 — Clone the repository**
```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
cd YOUR_REPO_NAME
```

**Step 2 — Create a virtual environment (recommended)**
```bash
python -m venv venv
```

Activate it:
- Windows:
```bash
venv\Scripts\activate
```
- Mac/Linux:
```bash
source venv/bin/activate
```

**Step 3 — Install dependencies**
```bash
pip install -r requirements.txt
```

**Step 4 — Run the app**
```bash
python -m streamlit run app.py
```

Then open your browser at `http://localhost:8501`

> ⚠️ The first run will automatically download the `facebook/bart-large-cnn` model (~1.6 GB). This only happens once — it gets cached locally after that.

---

### Option 2 — Google Colab

**Step 1 — Install dependencies**
```python
!pip install streamlit trafilatura requests beautifulsoup4 contractions nltk transformers torch rouge_score pyngrok -q
```

**Step 2 — Clone the repo**
```python
!git clone https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
%cd YOUR_REPO_NAME
```

**Step 3 — Launch with ngrok**
```python
from pyngrok import ngrok
import subprocess, time

ngrok.set_auth_token("PASTE_YOUR_NGROK_TOKEN_HERE")  # get free token at ngrok.com

proc = subprocess.Popen([
    "streamlit", "run", "app.py",
    "--server.port", "8501",
    "--server.headless", "true"
])
time.sleep(3)

public_url = ngrok.connect(8501)
print("✅ Open your app here:", public_url)
```

> 💡 For faster summarization on Colab, go to **Runtime → Change runtime type → T4 GPU**

---

## ⚙️ How It Works

```
Input (URL or Raw Text)
        ↓
  Web Scraping (Trafilatura + BeautifulSoup fallback)
        ↓
  NLP Preprocessing
  - Text normalization
  - Contraction expansion
  - Sentence segmentation (NLTK)
  - Token analysis (BART tokenizer)
  - Quality filtering
        ↓
  Smart Chunking (sentence-aware, token-limited, with overlap)
        ↓
  BART Summarization
  - Short article (<700 tokens) → direct summarization
  - Long article (≥700 tokens) → chunk-by-chunk → final combined summary
        ↓
  Post-processing (remove duplicates, clean punctuation)
        ↓
  Final Summary Output
```

---

## 📦 Dependencies

| Package | Purpose |
|---|---|
| `streamlit` | Web UI |
| `transformers` | BART model & tokenizer |
| `torch` | Deep learning backend |
| `trafilatura` | Web article scraping |
| `beautifulsoup4` | HTML parsing fallback |
| `nltk` | Sentence tokenization |
| `contractions` | Contraction expansion |
| `rouge_score` | Summary evaluation |
| `pyngrok` | Colab tunneling |

---

## ⏱️ Expected Runtime

| Environment | Per Summary |
|---|---|
| Local CPU | ~5–10 min |
| Local GPU | ~1–2 min |
| Colab CPU | ~5–10 min |
| Colab T4 GPU | ~30–60 sec |

---

## 👥 Contributors

- NLP Group 20 (University Malaya)
```
Laila
Iman
Serena
Damia
Uswatun
Mahbub
```
---

## 📄 License

This project is for educational purposes only.
