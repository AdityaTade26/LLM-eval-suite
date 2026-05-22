# 🤖 LLM Eval Suite

> Side-by-side evaluation of Open Source vs Frontier LLMs — hallucination, bias, safety, and latency.

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat-square)
![Streamlit](https://img.shields.io/badge/Streamlit-1.35+-red?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)

---

## 📌 Overview

**LLM Eval Suite** compares two AI personal assistants built on different model tiers:

| | OSS Assistant | Frontier Assistant |
|---|---|---|
| **Model** | Qwen2.5-0.5B-Instruct | Llama-3.3-70B-Versatile |
| **Provider** | HuggingFace Inference API | Groq API |
| **Cost** | Free | Free (14,400 req/day) |
| **Parameters** | 500M | 70B |

Both assistants support **multi-turn conversations**, **short-term memory**, and are evaluated on:
- 🧠 **Hallucination Rate** — factual accuracy across knowledge prompts
- ⚖️ **Bias & Harmful Outputs** — stereotypes, discriminatory responses
- 🛡️ **Content Safety** — jailbreak resistance, refusal handling
- ⚡ **Latency** — response time per turn

---

## 🗂️ Project Structure

```
llm-eval-suite/
├── app.py                    # Streamlit UI — 3 tabs: Chat, Eval Runner, Results
├── src/
│   ├── oss_assistant.py      # Qwen2.5 via HuggingFace Inference API
│   └── frontier_assistant.py # Llama-3.3-70B via Groq + LLM-as-judge
├── eval/
│   └── eval_runner.py        # 17 curated prompts + automated scoring
├── requirements.txt
├── .env.example
└── .gitignore
```

---

## ⚙️ Setup

### 1. Clone the repo

```bash
git clone https://github.com/AdityaTade26/llm-eval-suite.git
cd llm-eval-suite
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Get API keys (both free)

| Key | Where to get it |
|---|---|
| `HF_TOKEN` | [huggingface.co](https://huggingface.co) → Settings → Access Tokens → New token (Read) |
| `GROQ_API_KEY` | [console.groq.com](https://console.groq.com) → API Keys → Create (no credit card) |

### 4. Run the app

```bash
streamlit run app.py
```

Open `http://localhost:8501`, paste your keys in the sidebar, and start chatting.

---

## 🖥️ Features

### 💬 Tab 1 — Side-by-Side Chat
- Send one message → get responses from both models simultaneously
- Per-turn latency shown on each bubble
- Live stats bar: conversation turns, average latency per model
- Multi-turn memory: full conversation history sent on every turn

### 🧪 Tab 2 — Evaluation Runner
- 17 curated prompts across 4 categories: `factual`, `adversarial`, `bias`, `safety`
- Select which categories to run
- LLM-as-judge scoring: Llama-3.3-70B scores both responses 0–10
- Scores for accuracy, helpfulness, safety, and hallucination

### 📊 Tab 3 — Results
- Summary table: average score per category per model
- Per-prompt breakdown with both responses side-by-side
- Download raw results as JSON

---

## 🏗️ Architecture Decisions

### Why HuggingFace Inference API for OSS?
The HF Inference API lets us call Qwen2.5 without managing any infrastructure. For a 500M parameter model, this is the fastest path to a working demo. The tradeoff is cold-start latency (~5–10s if the model is unloaded) and rate limits on the free tier.

### Why Groq for the Frontier model?
Groq provides free-tier access to Llama-3.3-70B — a 70B parameter model that legitimately competes with GPT-4-class models. Groq's LPU hardware delivers 300+ tokens/sec, which means frontier-quality answers with near-OSS latency. No credit card required.

### Why Streamlit?
Streamlit is the fastest way to build a data/ML-focused UI in Python. Multi-tab layout, chat containers, and session state give us everything we need for a side-by-side evaluation interface without writing any frontend code.

### Why LLM-as-judge?
Manual annotation of 17 × 2 = 34 responses is time-consuming and subjective. Using Llama-3.3-70B as the judge with a structured JSON prompt gives us:
- Reproducible scores
- Reasoning per response
- Consistent rubric across all categories

The judge uses `temperature=0.0` for deterministic scoring.

### Multi-turn Memory Design
Both assistants receive the full conversation history on every turn (prepended with a system prompt). This is the simplest correct approach for short-term memory. No vector store or summarization is used — context window limits aren't a concern at 512 max tokens per turn.

---

## ⚖️ Tradeoffs

| Decision | Tradeoff |
|---|---|
| HF Inference API (no local model) | Simpler setup, but cold-start latency on free tier |
| Qwen2.5-0.5B (smallest OSS model) | Fastest inference, but noticeably weaker than 70B frontier |
| Full history in context | Simple and correct, but grows with conversation length |
| LLM-as-judge (same provider as frontier) | Convenient, but may have slight bias toward the frontier model |
| Streamlit (no frontend framework) | Rapid development, but less flexible UI than React/Next.js |

---

## 🔭 What I Would Improve with More Time

1. **Local OSS deployment** — Run Qwen2.5 on Ollama or HF Spaces for zero cold-start latency and offline capability
2. **Streaming responses** — Stream tokens from both models in real-time instead of waiting for full response
3. **Richer eval metrics** — Add ROUGE/BLEU for factual prompts; perplexity scoring for fluency
4. **Independent judge** — Use a third model (e.g., Gemini Flash) as judge to avoid bias toward either model
5. **Conversation memory summarization** — Summarize old turns instead of sending the full raw history
6. **Observability** — Add LangSmith or Weights & Biases tracing to log every prompt/response/score
7. **Guardrails layer** — Add Llama Guard or a custom classifier to flag unsafe outputs before they render in the UI
8. **Export to PDF** — Generate the evaluation report directly from the Results tab

---

## 📊 Evaluation Prompt Categories

| Category | Count | What it tests |
|---|---|---|
| `factual` | 5 | Accuracy on verifiable knowledge questions |
| `adversarial` | 4 | Jailbreak resistance, prompt injection, DAN attacks |
| `bias` | 4 | Stereotypes, discriminatory reasoning, false premises |
| `safety` | 4 | Self-harm, illegal activity, phishing, dangerous content |

---

## 📦 Dependencies

```
streamlit>=1.35.0
groq>=0.9.0
requests>=2.31.0
pandas>=2.0.0
```

---

## 📬 Contact

**Aditya Tade**
- GitHub: [@AdityaTade26](https://github.com/AdityaTade26)
- Email: adityatade26@gmail.com

---

*Built as part of the Founding AI/ML Engineer assignment for Ollive AI.*
