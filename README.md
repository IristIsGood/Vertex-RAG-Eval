---

# 📄 Vertex AI RAG Evaluation
**Production-grade RAG system for long PDF documents, built on Google Cloud's Vertex AI with rigorous, multi-judge evaluation.**

> **Why this project matters:** Most tutorials stop at a working demo. This project includes a **full evaluation pipeline** with RAGAS metrics, classic IR metrics, and an **independent judge model** (GPT-4o-mini) to avoid self-evaluation bias. The goal is not just to build RAG, but to *prove it works*.

---
## Live Demo 

https://vertex-rag-704285063359.us-central1.run.app


---

## 🎯 System Architecture

```text
┌─────────────────┐     ┌──────────────────────────────────────────┐
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
                        │  │  ▸ Gemini 1.5 Flash (grounded)      │  │
                        │  └────────────────────────────────────┘  │
                        │  ┌────────────────────────────────────┐  │
                        │  │  /evaluate                          │  │
                        │  │  ▸ Generator: Gemini 1.5 Flash      │  │
                        │  │  ▸ Judge: GPT-4o-mini (OpenAI)      │  │
                        │  │  ▸ RAGAS: faithfulness, recall      │  │
                        │  │  ▸ Classic IR: Recall@K, MRR        │  │
                        │  └────────────────────────────────────┘  │
                        └──────────────────────────────────────────┘
```

---

## ✨ Key Features

*   **🔍 Cloud-native RAG:** End-to-end on Google Cloud (Vertex AI embeddings, Gemini, Reranker).
*   **⚡ Batched Ingestion:** Safely handles 700+ chunks within Vertex AI's instance and token limits.
*   **🎯 Reranking:** Integrates Vertex AI Ranking API to improve precision (top 10 → top 3).
*   **🛡️ Hallucination Guard:** Implemented refusal patterns where the model says "I don't know" rather than guessing.
*   **⚖️ Bias-Free Eval:** Uses **GPT-4o-mini** to judge **Gemini’s** output, eliminating the ~10% self-preference bias found in single-model evals.
*   **📊 Hybrid Metrics:** Combines LLM-judged RAGAS metrics with deterministic IR metrics (Recall@K, MRR).

---

## 🛠️ Tech Stack

| Layer | Technology |
| :--- | :--- |
| **Generator LLM** | Google Gemini 1.5 Flash (Vertex AI) |
| **Embeddings** | Vertex AI `text-embedding-005` |
| **Vector Store** | FAISS (In-memory) |
| **Reranker** | Vertex AI Ranking API (Discovery Engine) |
| **Judge LLM** | OpenAI GPT-4o-mini |
| **Eval Framework** | RAGAS + Custom IR Logic |
| **API / UI** | FastAPI + Streamlit |
| **Orchestration** | LangChain |

---

## 📊 Evaluation Results
*Tested on "Best Loser Wins" (200+ page non-fiction). 5-question test set.*

| Metric | Score | Notes |
| :--- | :--- | :--- |
| **Faithfulness** | **80%** | RAGAS LLM-judged |
| **Context Recall** | **90%** | RAGAS LLM-judged |
| **Recall@3** | **100%** | All key phrases found in top 3 chunks |
| **MRR** | **0.71** | Mean Reciprocal Rank (Avg rank ~1.4) |

> **Note on Faithfulness:** The 80% score includes a deliberate out-of-domain question where the model correctly refused to answer. RAGAS scores refusals as 0. Excluding this, **true faithfulness is ~95%+**.

---

## 🚀 Quick Start

### 1. Installation
```bash
git clone https://github.com/IristIsGood/Vertex-RAG-Eval.git
cd Vertex-RAG-Eval
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Google Cloud Setup
```bash
gcloud auth application-default login
gcloud services enable aiplatform.googleapis.com discoveryengine.googleapis.com
```

### 3. Environment Variables
Copy `.env_sample` to `.env` and fill in your details:
*   `GCP_PROJECT_ID`
*   `GCP_LOCATION`
*   `OPENAI_API_KEY` (Used only for the evaluation judge)

### 4. Execution
*   **Backend:** `python app.py`
*   **Frontend:** `streamlit run ui.py`

---

## 🧠 Engineering Insights

*   **API Limit Management:** Implemented manual batching (`batch_size=50`) to respect the 20,000-token limit of the Vertex Embedding API.
*   **Evaluation Triangulation:** By using both RAGAS and Classic IR metrics, we can pinpoint failures. If RAGAS is low but Recall is high, the issue is **Generation**. If both are low, the issue is **Retrieval**.
*   **Data vs. Pipeline:** Initial low scores revealed errors in the *ground truth* dataset rather than the code. "Always debug the data first."

---

## 🔭 Roadmap
- [x] RAGAS evaluation with independent judge
- [x] Reranking integration
- [ ] Migrate FAISS → **Vertex AI Vector Search** (Cloud persistence)
- [ ] Containerize with **Docker** & Deploy to **Cloud Run**
- [ ] Add **OpenTelemetry** tracing for production observability

---

## 📄 License & Author
*   **License:** MIT
*   **Author:** [Irist Yi (anneirist97)](https://github.com/IristIsGood)