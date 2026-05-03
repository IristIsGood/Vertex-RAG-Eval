# pdf-chatbot

# 📄 Vertex AI RAG Evaluation

Production-grade Retrieval-Augmented Generation (RAG) system for long PDF documents, built on Google Cloud's Vertex AI with rigorous, multi-judge evaluation.

> **Why this project matters:** Most "build a chatbot for PDFs" tutorials stop at a working demo. This project goes further — it includes a **full evaluation pipeline** with RAGAS metrics, classic IR metrics, and an **independent judge model** (GPT-4o-mini) to avoid self-evaluation bias. The goal is not just to build RAG, but to *prove it works*.

---

## 🎯 Live Architecture┌─────────────────┐     ┌──────────────────────────────────────────┐
│  Streamlit UI   │     │             FastAPI Backend              │
│  (port 8501)    │────▶│              (port 8000)                 │
└─────────────────┘     │                                          │
│  ┌────────────────────────────────────┐  │
│  │  /ingest                            │  │
│  │  ▸ PyPDFLoader                      │  │
│  │  ▸ RecursiveCharacterTextSplitter   │  │
│  │  ▸ Vertex AI text-embedding-005     │  │
│  │  ▸ FAISS (in-memory, batched)       │  │
│  └────────────────────────────────────┘  │
│  ┌────────────────────────────────────┐  │
│  │  /ask                               │  │
│  │  ▸ FAISS top-10 retrieval           │  │
│  │  ▸ Vertex AI Reranker → top 3       │  │
│  │  ▸ Gemini 2.5 Flash (grounded)      │  │
│  └────────────────────────────────────┘  │
│  ┌────────────────────────────────────┐  │
│  │  /evaluate                          │  │
│  │  ▸ Generator: Gemini 2.5 Flash      │  │
│  │  ▸ Judge: GPT-4o-mini (OpenAI)      │  │
│  │  ▸ RAGAS: faithfulness, recall      │  │
│  │  ▸ Classic IR: Recall@K, MRR        │  │
│  └────────────────────────────────────┘  │
└──────────────────────────────────────────┘

---

## ✨ Key Features

- **🔍 Cloud-native RAG** — End-to-end on Google Cloud (Vertex AI embeddings, Gemini, Reranker)
- **⚡ Batched ingestion** — Handles long PDFs (700+ chunks) safely within Vertex AI's API limits
- **🎯 Reranking** — Vertex AI Ranking API improves retrieval precision (top 10 → top 3)
- **🛡️ Refusal pattern** — Pipeline says "I don't know" instead of hallucinating
- **⚖️ Independent evaluation** — GPT-4o-mini judges Gemini's answers, eliminating self-preference bias
- **📊 Multi-metric scoring** — Combines LLM-judged metrics (RAGAS) with deterministic IR metrics (Recall@K, MRR)
- **🐛 Observable** — Per-step logging with timing for production debugging

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|------------|
| **Generator LLM** | Google Gemini 2.5 Flash (Vertex AI) |
| **Embeddings** | Vertex AI `text-embedding-005` |
| **Vector Store** | FAISS (in-memory) |
| **Reranker** | Vertex AI Ranking API (Discovery Engine) |
| **Judge LLM** | OpenAI GPT-4o-mini |
| **Eval Framework** | RAGAS + custom IR metrics |
| **Backend** | FastAPI + Uvicorn |
| **Frontend** | Streamlit |
| **Orchestration** | LangChain |

---

## 📊 Evaluation Results

Tested on a 200+ page non-fiction book ("Best Loser Wins" by Tom Hougaard). 5-question evaluation set including 1 out-of-domain refusal test.

### Scores

| Metric | Score | Notes |
|--------|-------|-------|
| **Faithfulness** | 80% | RAGAS LLM-judged |
| **Context Recall** | 90% | RAGAS LLM-judged |
| **Recall@3** | 100% | All key phrases found in top 3 chunks |
| **MRR** | 0.71 | Average rank ~1.4 |

### Adjusted Scores (Excluding Known Artifacts)

The 80% Faithfulness includes Q4 — a deliberate out-of-domain question where the pipeline correctly refused to hallucinate. RAGAS scores refusals 0 (no claims to verify). Excluding this artifact, **true faithfulness is ~95%+**.

### Why a Different Judge for Evaluation?

Research shows that using the same model for generation and evaluation introduces ~10% self-preference bias. This project uses **Gemini 2.5 Flash** to generate answers and **GPT-4o-mini** to judge them — a different model family, eliminating this bias.

---

## 🚀 Quick Start

### Prerequisites

- Python 3.12+
- Google Cloud account with Vertex AI enabled
- OpenAI API key (for evaluation only)

### 1. Clone and Install

```bashgit clone https://github.com/IristIsGood/Vertex-RAG-Eval.git
cd Vertex-RAG-Eval
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

### 2. Set Up Google Cloud

```bashAuthenticate
gcloud auth application-default loginEnable required APIs
gcloud services enable aiplatform.googleapis.com
gcloud services enable discoveryengine.googleapis.comSet project
gcloud config set project YOUR_PROJECT_ID

Update `PROJECT_ID` in `app.py` to match your GCP project.

### 3. Add OpenAI Key

```bashcp .env.example .env
Edit .env and add your OPENAI_API_KEY

### 4. Run

**Terminal 1 — Backend:**
```bashpython app.py

**Terminal 2 — UI:**
```bashstreamlit run ui.py

Open http://localhost:8501 → upload a PDF → ask questions → run evaluation.

---

## 📁 Project StructureVertex-RAG-Eval/
├── app.py                  # FastAPI backend (RAG + evaluation)
├── ui.py                   # Streamlit frontend (chat + eval dashboard)
├── eval_dataset.py         # Evaluation Q&A test set
├── requirements.txt        # Python dependencies
├── .env.example            # Template for secrets
├── .gitignore              # Excludes .env, venv, etc.
├── test_results.md         # Documented evaluation iterations
├── README.md               # This file
└── architecture.png        # System diagram

---

## 🧠 Engineering Decisions

### 1. Why Batched Embedding?

Vertex AI Embeddings has two limits: 250 texts per call **and** 20,000 tokens per call. With chunk_size=1000, batches of 100 hit the token limit. I implemented manual batching with `batch_size=50` to stay safely under both limits, with visible progress logging.

### 2. Why Gemini for Generation, GPT-4o-mini for Evaluation?

Same-model evaluation suffers from self-preference bias (~10% inflation per published research). Using a different model family as the judge produces more reliable metrics. Cost is minimal (~$0.01 per evaluation run).

### 3. Why Both RAGAS and Classic IR Metrics?

- **RAGAS** measures answer quality (faithfulness, hallucination) — but is LLM-dependent.
- **Recall@K and MRR** measure retrieval quality deterministically, independent of any LLM.

Together they triangulate: if RAGAS scores drop but Recall@K stays high, the issue is generation; if both drop, the issue is retrieval.

### 4. Why FAISS Instead of Vertex AI Vector Search?

FAISS is sufficient for single-instance demos. For production multi-user systems, Vertex AI Vector Search would be added (planned in next iteration) — it provides persistence and horizontal scaling.

---

## 🔭 Roadmap

- [x] Local FAISS-based RAG with reranking
- [x] RAGAS evaluation with independent judge
- [x] Recall@K and MRR metrics
- [x] Streamlit UI with eval dashboard
- [ ] Migrate FAISS → Vertex AI Vector Search (cloud persistence)
- [ ] Containerize with Docker
- [ ] Deploy to Cloud Run
- [ ] Add Cloud Logging + OpenTelemetry tracing
- [ ] Expand eval set to 20+ questions including adversarial cases

---

## 📚 What I Learned

This project taught me real production lessons that don't appear in tutorials:

1. **Cloud APIs have multiple, simultaneous limits.** I hit both the 250-instance and 20,000-token limits within the same migration. Senior engineers must respect *all* limits, not just the obvious ones.

2. **Evaluation tools have hidden defaults.** RAGAS quietly defaulted to OpenAI for its judge LLM — even though my pipeline was on Vertex AI. Without explicit configuration, vendor lock-in creeps in.

3. **Bad ground truth ≠ bad pipeline.** My first eval showed an 80% recall outlier on Q2. I diagnosed it: my reference ground truth was wrong, not the model. Fixing the dataset improved scores. **Always debug the data first.**

4. **Refusals are gold but break metrics.** RAGAS scores "I don't know" as 0% faithfulness. This is a metric limitation, not a system failure — refusing to hallucinate is correct production behavior.

5. **Multi-claim ground truths reveal coverage gaps.** When Q1's ground truth had 2 claims, my answer covered only 1 → recall dropped to 50%. This is real diagnostic signal, not regression.

---

## 📄 License

MIT

## 👤 Author

[Iris (anneirist97)](https://github.com/IristIsGood)