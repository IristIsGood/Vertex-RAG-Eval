import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_classic.chains import RetrievalQA # ← fixed: not langchain_classic

# ══════════════════════════════════════════════════════════════════════════════
# STEP 1 — API KEY
# ══════════════════════════════════════════════════════════════════════════════
# EN: Never hardcode API keys in code. Use .env files instead (see previous lesson).
# ZH: 永远不要把 API 密钥写在代码里，应该使用 .env 文件管理密钥。
os.environ["OPENAI_API_KEY"] = "YOUR-NEW-KEY-HERE"


# 🚀 1. “Chat with Your PDF” (RAG Project)
# 💡 What you’ll learn
# * RAG (Retrieval-Augmented Generation)
# * Embeddings + vector databases
# * Basic LangChain pipeline
# 🛠 What to build
# Upload a PDF → ask questions → AI answers based on the document.
# ⚙️ Tech stack
# * Python
# * LangChain
# * OpenAI API (or local model)
# * FAISS (vector database)

# ══════════════════════════════════════════════════════════════════════════════
# STEP 2 — DOCUMENT LOADING
# ══════════════════════════════════════════════════════════════════════════════
#
# EN: A "Document Loader" reads a file and converts it into LangChain's
#     standard Document object: { page_content: "...", metadata: { page: 0 } }
#     LangChain has 100+ loaders: PDF, Word, websites, Notion, YouTube, etc.
#
# ZH: "文档加载器" 读取文件并转换为 LangChain 标准的 Document 对象：
#     { page_content: "文本内容", metadata: { page: 页码 } }
#     LangChain 支持 100+ 种加载器：PDF、Word、网页、Notion、YouTube 等。
#
# INTERVIEW QUESTION 面试题:
#   Q: What is a Document in LangChain?
#   A: A Document is a standard data object with two fields:
#      page_content (the text) and metadata (source, page number, etc.)
#
PDF_PATH = "Best Loser Wins Why Normal Thinking Never Wins the Trading Game.pdf"

loader = PyPDFLoader(PDF_PATH)
pages = loader.load()
print(f"✅ Loaded {len(pages)} pages from your PDF.")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 3 — TEXT SPLITTING (CHUNKING)
# ══════════════════════════════════════════════════════════════════════════════
#
# EN: WHY do we split? LLMs have a "context window" — a maximum number of
#     tokens they can process at once (e.g. gpt-4o-mini = 128,000 tokens).
#     More importantly, sending an entire book to the LLM every time a user
#     asks a question is extremely expensive and slow.
#     So instead, we split the document into small chunks and only send
#     the RELEVANT chunks to the LLM. This is the core idea of RAG.
#
# ZH: 为什么要分块？LLM 有"上下文窗口"限制——每次能处理的 token 数量有上限
#     （如 gpt-4o-mini 最多 128,000 tokens）。更重要的是，每次都把整本书
#     发给 LLM 既昂贵又慢。所以我们把文档切成小块，只把"相关的块"发给 LLM。
#     这就是 RAG 的核心思想。
#
# EN: chunk_size=500 → each chunk is ~500 characters
#     chunk_overlap=50 → consecutive chunks share 50 characters
#     Why overlap? So that a sentence split across two chunks isn't lost.
#
# ZH: chunk_size=500 → 每块约 500 个字符
#     chunk_overlap=50 → 相邻块共享 50 个字符的重叠
#     为什么要重叠？防止一句话被切断后语义丢失。
#
# EN: RecursiveCharacterTextSplitter is the most recommended splitter.
#     It tries to split on paragraphs first (\n\n), then sentences (\n),
#     then words (" "), then characters — in that priority order.
#     This preserves natural language boundaries as much as possible.
#
# ZH: RecursiveCharacterTextSplitter 是最推荐的分割器。
#     它按优先级依次尝试：段落(\n\n) → 句子(\n) → 单词(" ") → 字符
#     这样能最大程度保留自然语言的边界。
#
# INTERVIEW QUESTION 面试题:
#   Q: What chunking strategy do you use and why?
#   A: RecursiveCharacterTextSplitter because it respects natural language
#      boundaries. chunk_size and chunk_overlap are tuned based on the
#      document type — smaller chunks for precise retrieval, larger for
#      more context. Overlap prevents information loss at chunk boundaries.
#
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)
chunks = splitter.split_documents(pages)
print(f"✅ Split into {len(chunks)} chunks.")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 4 — EMBEDDINGS + VECTOR DATABASE (THE HEART OF RAG)
# ══════════════════════════════════════════════════════════════════════════════
#
# ── What are Embeddings? / 什么是 Embeddings（向量嵌入）? ──────────────────────
#
# EN: An embedding model converts text into a list of numbers (a "vector").
#     Example: "king" → [0.2, 0.8, -0.1, 0.5, ...]  (1536 numbers for OpenAI)
#     The key property: texts with SIMILAR MEANING produce SIMILAR vectors.
#     So "dog" and "puppy" will have vectors very close to each other,
#     while "dog" and "rocket" will be far apart.
#     This allows us to search by MEANING, not just by keyword matching.
#
# ZH: 嵌入模型把文本转换成一组数字（"向量"）。
#     例如："king" → [0.2, 0.8, -0.1, 0.5, ...]（OpenAI 生成 1536 个数字）
#     关键特性：语义相似的文本，其向量也相近。
#     所以 "dog" 和 "puppy" 的向量很接近，
#     而 "dog" 和 "rocket" 的向量相差很远。
#     这让我们可以按"语义"搜索，而不只是关键词匹配。
#
# ── What is a Vector Database? / 什么是向量数据库? ────────────────────────────
#
# EN: A vector database stores all those number lists (vectors) and lets you
#     search them FAST. When you ask a question, it converts your question
#     to a vector too, then finds the stored vectors closest to it.
#     "Closest" = most semantically similar = most relevant chunks.
#     FAISS (by Meta) is a local, in-memory vector database — fast and free.
#     Other options: Pinecone, Chroma, Weaviate, Qdrant (for production).
#
# ZH: 向量数据库存储所有的向量，并支持快速搜索。
#     当你提问时，问题也被转成向量，然后数据库找出与它最接近的向量。
#     "最接近" = 语义最相似 = 最相关的文本块。
#     FAISS（Meta 开发）是本地内存向量数据库——快速且免费。
#     生产环境还可以用：Pinecone、Chroma、Weaviate、Qdrant。
#
# ── How similarity is measured / 相似度如何计算 ────────────────────────────────
#
# EN: FAISS uses "cosine similarity" — it measures the angle between two
#     vectors. Angle = 0° means identical meaning (score = 1.0).
#     Angle = 90° means completely unrelated (score = 0.0).
#
# ZH: FAISS 使用"余弦相似度"——计算两个向量之间的夹角。
#     夹角 = 0° 表示语义完全相同（得分 = 1.0）。
#     夹角 = 90° 表示完全不相关（得分 = 0.0）。
#
# INTERVIEW QUESTION 面试题:
#   Q: What is the difference between a vector database and a regular database?
#   A: A regular database searches by exact match or range (WHERE name = "John").
#      A vector database searches by semantic similarity — it finds the most
#      "meaning-similar" entries using distance metrics like cosine similarity
#      or euclidean distance on high-dimensional vectors.
#
#   Q: Why use FAISS over Pinecone?
#   A: FAISS is local, free, and great for prototypes and small datasets.
#      Pinecone is a managed cloud service — better for production because
#      it supports persistence, scaling, and real-time updates.
#
embeddings = OpenAIEmbeddings()
# EN: This line does 3 things at once:
#     1. Takes every chunk's text
#     2. Sends it to OpenAI's embedding API → gets back a 1536-dim vector
#     3. Stores all vectors in FAISS in memory
# ZH: 这一行做了 3 件事：
#     1. 取出每个文本块的内容
#     2. 发送给 OpenAI 的嵌入 API → 返回 1536 维向量
#     3. 把所有向量存入内存中的 FAISS 索引
vectorstore = FAISS.from_documents(chunks, embeddings)
print("✅ Embeddings created and stored in FAISS.")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 5 — THE RAG CHAIN (PUTTING IT ALL TOGETHER)
# ══════════════════════════════════════════════════════════════════════════════
#
# EN: This is where RAG (Retrieval-Augmented Generation) is fully assembled.
#     RAG = Retrieval + Generation. Two separate systems working together:
#
#     RETRIEVAL (검색): Given a question → find relevant chunks from FAISS
#     GENERATION (생성): Given those chunks + question → LLM generates answer
#
#     Without RAG, you'd just ask GPT a question and it answers from its
#     training data — it doesn't know your specific PDF.
#     With RAG, GPT answers based on YOUR document's actual content.
#
# ZH: 这里是 RAG（检索增强生成）的完整组装。
#     RAG = 检索（Retrieval）+ 生成（Generation），两个系统协同工作：
#
#     检索：输入问题 → 从 FAISS 找出相关文本块
#     生成：将文本块 + 问题 → 交给 LLM 生成回答
#
#     没有 RAG：直接问 GPT，它只能用训练数据回答，不了解你的 PDF。
#     有了 RAG：GPT 基于你文档的实际内容来回答。
#
# ── The full RAG flow / 完整 RAG 流程 ─────────────────────────────────────────
#
#   User question "What is the main idea?"
#        ↓
#   Convert question → vector (via OpenAIEmbeddings)
#        ↓
#   FAISS finds top 3 most similar chunks (k=3)
#        ↓
#   Chunks + question assembled into a prompt:
#   "Answer using this context: [chunk1][chunk2][chunk3] Question: ..."
#        ↓
#   gpt-4o-mini generates the final answer
#        ↓
#   Answer returned to user
#
# EN: temperature=0 means the model gives deterministic, factual answers.
#     Higher temperature (e.g. 0.7) = more creative but less consistent.
#     For Q&A over documents, always use temperature=0.
#
# ZH: temperature=0 表示模型给出确定性的、事实性的回答。
#     较高的 temperature（如 0.7）= 更有创意但不稳定。
#     对文档问答，始终使用 temperature=0。
#
# EN: k=3 means retrieve the 3 most relevant chunks.
#     Too low (k=1): might miss context. Too high (k=10): adds noise and cost.
#     k=3 to k=5 is the standard range for most RAG applications.
#
# ZH: k=3 表示检索最相关的 3 个文本块。
#     太少（k=1）：可能遗漏上下文。太多（k=10）：引入噪音且增加成本。
#     k=3 到 k=5 是大多数 RAG 应用的标准范围。
#
# INTERVIEW QUESTION 面试题:
#   Q: Explain RAG in simple terms.
#   A: RAG grounds an LLM's response in a specific knowledge base.
#      Instead of relying on training data, the model first retrieves
#      relevant documents using semantic search, then generates an answer
#      conditioned on those documents. This reduces hallucination and
#      allows the model to answer questions about private or recent data.
#
#   Q: What are the limitations of RAG?
#   A: 1. Retrieval quality — if the wrong chunks are retrieved, the answer
#         will be wrong even if the LLM is perfect.
#      2. Chunk boundary issues — relevant info split across chunks may not
#         be retrieved together.
#      3. No reasoning across the full document — only sees k chunks at once.
#      Solutions: better chunking, re-ranking, larger k, hybrid search.
#
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
)


# ══════════════════════════════════════════════════════════════════════════════
# STEP 6 — CHAT LOOP
# ══════════════════════════════════════════════════════════════════════════════
#
# EN: qa_chain.invoke(question) triggers the full RAG pipeline in one call:
#     retrieve → format prompt → generate → return answer.
#     The result is a dict: { "query": "...", "result": "the answer" }
#
# ZH: qa_chain.invoke(question) 一次调用触发完整 RAG 流程：
#     检索 → 格式化提示词 → 生成 → 返回答案。
#     结果是一个字典：{ "query": "问题", "result": "回答" }
#
print("\n💬 PDF chatbot ready! Type 'quit' to exit.\n")
while True:
    question = input("You: ")
    if question.lower() == "quit":
        break
    answer = qa_chain.invoke(question)
    print(f"\nAI: {answer['result']}\n")