import os
import uvicorn
import tempfile
import vertexai
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from typing import List

# Vertex AI & LangChain
from langchain_google_vertexai import VertexAIEmbeddings, ChatVertexAI
from langchain_google_community import VertexAIRank
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# RAGAS for Evaluation
from ragas import evaluate
from ragas.metrics import faithfulness, context_recall
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from datasets import Dataset

# ══════════════════════════════════════════════════════════════════════════════
# LOAD ENVIRONMENT VARIABLES
# ══════════════════════════════════════════════════════════════════════════════
# EN: Load OPENAI_API_KEY from .env file (used only for RAGAS evaluation,
#     not for the RAG pipeline itself).
# ZH: 从 .env 文件加载 OPENAI_API_KEY（仅用于 RAGAS 评估，不用于 RAG 流程）。
load_dotenv()

# ══════════════════════════════════════════════════════════════════════════════
# INITIALIZATION
# ══════════════════════════════════════════════════════════════════════════════
PROJECT_ID = os.getenv("GCP_PROJECT_ID")
LOCATION = os.getenv("GCP_LOCATION")
vertexai.init(project=PROJECT_ID, location=LOCATION)

app = FastAPI(title="Vertex AI RAG Backend")

# Initialize models (using latest stable versions for new GCP projects)
embeddings = VertexAIEmbeddings(model_name="text-embedding-005", project=PROJECT_ID,location=LOCATION)
llm = ChatVertexAI(model_name="gemini-2.5-flash", temperature=0, project=PROJECT_ID,location=LOCATION)

# Global vectorstore (in-memory for this demo)
# In production, replace with Vertex AI Vector Search (Managed Index)
vectorstore = None


class QuestionRequest(BaseModel):
    question: str


# ══════════════════════════════════════════════════════════════════════════════
# HELPER: BATCHED FAISS BUILDER
# ══════════════════════════════════════════════════════════════════════════════
# EN: Vertex AI Embeddings API has two limits:
#     1. Max 250 texts per call (instance limit)
#     2. Max 20,000 tokens per call (token limit)
#     batch_size=50 is a safe value that respects both limits.
# ZH: Vertex AI 嵌入 API 有两个限制：
#     1. 单次调用最多 250 条文本（实例限制）
#     2. 单次调用最多 20,000 tokens（token 限制）
#     batch_size=50 是同时满足两个限制的安全值。
def build_faiss_in_batches(chunks, embeddings, batch_size: int = 50):
    print(f"📦 Embedding {len(chunks)} chunks in batches of {batch_size}...")
    vs = None
    total_batches = (len(chunks) + batch_size - 1) // batch_size

    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        batch_num = (i // batch_size) + 1
        print(f"  → Batch {batch_num}/{total_batches} ({len(batch)} chunks)")

        if vs is None:
            vs = FAISS.from_documents(batch, embeddings)
        else:
            batch_store = FAISS.from_documents(batch, embeddings)
            vs.merge_from(batch_store)

    return vs


# ══════════════════════════════════════════════════════════════════════════════
# ENDPOINT: /ingest — upload and index a PDF
# ══════════════════════════════════════════════════════════════════════════════
@app.post("/ingest")
async def ingest_pdf(file: UploadFile = File(...)):
    global vectorstore
    try:
        print(f"📥 Received file: {file.filename}")

        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(await file.read())
            tmp_path = tmp.name

        print(f"📄 Loading PDF from: {tmp_path}")
        loader = PyPDFLoader(tmp_path)
        pages = loader.load()

        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        chunks = splitter.split_documents(pages)
        print(f"✂️ Created {len(chunks)} chunks")

        # Build FAISS in batches to respect Vertex AI's API limits
        vectorstore = build_faiss_in_batches(chunks, embeddings, batch_size=50)
        print("✅ Vectorstore built successfully!")

        return {"status": "success", "chunks": len(chunks)}

    except Exception as e:
        print(f"❌ ERROR DURING INGESTION: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


# ══════════════════════════════════════════════════════════════════════════════
# ENDPOINT: /ask — query the RAG pipeline
# ══════════════════════════════════════════════════════════════════════════════
@app.post("/ask")
async def ask_question(request: QuestionRequest):
    if not vectorstore:
        raise HTTPException(status_code=400, detail="Please upload a PDF first.")

    # 1. Retrieval (top 10 candidates)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 10})
    docs = retriever.invoke(request.question)

    # 2. Reranking with Vertex AI Ranking API (top 3)
    reranker = VertexAIRank(project_id=PROJECT_ID, location_id="global", top_n=3)
    reranked_docs = reranker.compress_documents(docs, request.question)

    # 3. Grounded generation
    context = "\n\n".join([d.page_content for d in reranked_docs])
    template = """Answer the question strictly using the provided context.
If the answer is not there, say "I do not have enough information."

Context: {context}
Question: {question}"""

    prompt = ChatPromptTemplate.from_template(template)
    chain = prompt | llm | StrOutputParser()

    response = chain.invoke({"context": context, "question": request.question})
    return {"answer": response}


# ══════════════════════════════════════════════════════════════════════════════
# ENDPOINT: /evaluate — run RAGAS + classic IR metrics on a Q&A test set
# ══════════════════════════════════════════════════════════════════════════════
@app.get("/evaluate")
async def run_evaluation():
    """
    Run RAGAS evaluation:
    - Generator: Gemini 2.5 Flash (Vertex AI)
    - Judge:     GPT-4o-mini (OpenAI) — independent for unbiased scoring

    Also computes deterministic IR metrics (Recall@K, MRR) using key phrases.
    """
    if not vectorstore:
        raise HTTPException(status_code=400, detail="Please upload a PDF first via /ingest.")

    from eval_dataset import EVAL_DATASET

    questions, answers, contexts, ground_truths = [], [], [], []

    print(f"🧪 Running evaluation on {len(EVAL_DATASET)} questions...")

    # ── Generate answers using your RAG pipeline (Gemini) ───────────────
    for i, item in enumerate(EVAL_DATASET):
        question = item["question"]
        ground_truth = item["ground_truth"]
        print(f"  → Q{i+1}: {question[:60]}...")

        retriever = vectorstore.as_retriever(search_kwargs={"k": 10})
        docs = retriever.invoke(question)

        reranker = VertexAIRank(project_id=PROJECT_ID, location_id="global", top_n=3)
        reranked_docs = reranker.compress_documents(docs, question)

        context = "\n\n".join([d.page_content for d in reranked_docs])
        template = """Answer the question strictly using the provided context.
If the answer is not there, say "I do not have enough information."

Context: {context}
Question: {question}"""
        prompt = ChatPromptTemplate.from_template(template)
        chain = prompt | llm | StrOutputParser()
        answer = chain.invoke({"context": context, "question": question})

        questions.append(question)
        answers.append(answer)
        contexts.append([d.page_content for d in reranked_docs])
        ground_truths.append(ground_truth)

    # ── Build RAGAS dataset ──────────────────────────────────────────────
    dataset = Dataset.from_dict({
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths,
    })

    # ── Independent judge (avoid self-preference bias) ───────────────────
    # EN: Using a different model family as judge avoids self-preference bias.
    #     Research shows same-model evaluation inflates scores by ~10%.
    # ZH: 使用不同模型家族作为评估者，避免自我偏好偏差。
    #     研究显示同模型自评会使分数虚高约 10%。
    judge_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    judge_embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    ragas_llm = LangchainLLMWrapper(judge_llm)
    ragas_embeddings = LangchainEmbeddingsWrapper(judge_embeddings)

    print("⚖️  Judge: GPT-4o-mini (OpenAI) — independent from Gemini generator")
    print("📊 Computing RAGAS metrics...")

    result = evaluate(
        dataset,
        metrics=[faithfulness, context_recall],
        llm=ragas_llm,
        embeddings=ragas_embeddings,
    )

    # ── Classic IR metrics: Recall@K and MRR ─────────────────────────────
    # EN: These complement RAGAS by measuring retrieval quality directly,
    #     independent of the LLM judge.
    # ZH: 这些指标补充 RAGAS，直接衡量检索质量，不依赖 LLM 评估者。
    recall_at_k_scores = []
    mrr_scores = []

    for item, retrieved_ctx in zip(EVAL_DATASET, contexts):
        key_phrase = item.get("key_phrase")
        if not key_phrase:
            continue  # skip "not in document" questions

        rank = None
        for rank_idx, ctx_text in enumerate(retrieved_ctx, start=1):
            if key_phrase.lower() in ctx_text.lower():
                rank = rank_idx
                break

        if rank is not None:
            recall_at_k_scores.append(1.0)
            mrr_scores.append(1.0 / rank)
        else:
            recall_at_k_scores.append(0.0)
            mrr_scores.append(0.0)

    avg_recall_at_k = (
        sum(recall_at_k_scores) / len(recall_at_k_scores)
        if recall_at_k_scores else 0
    )
    avg_mrr = sum(mrr_scores) / len(mrr_scores) if mrr_scores else 0

    # ── Combine RAGAS + classic IR metrics ───────────────────────────────
    ragas_summary = result.to_pandas().mean(numeric_only=True).to_dict()
    combined_summary = {
        **ragas_summary,
        "recall_at_3": avg_recall_at_k,
        "mrr": avg_mrr,
    }

    return {
        "summary": combined_summary,
        "details": result.to_pandas().to_dict(orient="records"),
        "config": {
            "generator": "gemini-2.5-flash (Vertex AI)",
            "judge": "gpt-4o-mini (OpenAI)",
            "judge_embeddings": "text-embedding-3-small (OpenAI)",
            "rationale": "Different model families avoid self-preference bias",
            "metrics_explained": {
                "faithfulness": "Is the answer grounded in retrieved context? (LLM-judged)",
                "context_recall": "Did context contain ground-truth info? (LLM-judged)",
                "recall_at_3": "Did the top 3 retrieved chunks contain the key phrase? (deterministic)",
                "mrr": "Mean Reciprocal Rank — 1/rank of first relevant chunk (deterministic)",
            }
        }
    }


# ══════════════════════════════════════════════════════════════════════════════
# RUN SERVER
# ══════════════════════════════════════════════════════════════════════════════
# if __name__ == "__main__":
#     uvicorn.run(app, host="0.0.0.0", port=8000)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)