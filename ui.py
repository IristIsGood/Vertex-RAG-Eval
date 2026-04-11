import os
import tempfile
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_classic.chains import RetrievalQA  # ← fixed: langchain_classic does not exist
from dotenv import load_dotenv

# ══════════════════════════════════════════════════════════════════════════════
# ENVIRONMENT SETUP / 环境配置
# ══════════════════════════════════════════════════════════════════════════════
#
# EN: load_dotenv() reads your .env file and loads OPENAI_API_KEY into the
#     environment so OpenAI libraries can find it automatically.
#     This is safer than hardcoding the key in your source code.
#
# ZH: load_dotenv() 读取 .env 文件，把 OPENAI_API_KEY 加载到环境变量中，
#     OpenAI 库会自动读取它。比把密钥写在代码里更安全。
#
# INTERVIEW 面试:
#   Q: How do you manage secrets in a Python project?
#   A: Store them in a .env file, add .env to .gitignore, and use
#      python-dotenv to load them at runtime. Never hardcode secrets.
#
load_dotenv()


# ══════════════════════════════════════════════════════════════════════════════
# STREAMLIT PAGE CONFIG / Streamlit 页面配置
# ══════════════════════════════════════════════════════════════════════════════
#
# EN: st.set_page_config() must be the FIRST Streamlit call in your script.
#     It sets the browser tab title, icon, and layout.
#     layout="centered" puts content in a fixed-width column in the middle.
#     layout="wide" uses the full browser width.
#
# ZH: st.set_page_config() 必须是脚本中第一个 Streamlit 调用。
#     设置浏览器标签页的标题、图标和布局。
#     layout="centered" 把内容放在屏幕中间的固定宽度列中。
#     layout="wide" 使用浏览器全宽。
#
# EN: WHAT IS STREAMLIT?
#     Streamlit is a Python library that turns your script into a web app.
#     Every time a user interacts (clicks, uploads, types), Streamlit
#     re-runs your entire script from top to bottom.
#     This is called the "Streamlit execution model" — important for interviews.
#
# ZH: 什么是 Streamlit？
#     Streamlit 是一个 Python 库，把脚本变成网页应用。
#     每当用户交互（点击、上传、输入），Streamlit 会从头到尾重新执行整个脚本。
#     这叫做 "Streamlit 执行模型"——面试常考。
#
# INTERVIEW 面试:
#   Q: How does Streamlit work under the hood?
#   A: Streamlit re-runs the entire Python script on every user interaction.
#      State is preserved using st.session_state. Expensive computations are
#      cached using @st.cache_resource or @st.cache_data to avoid re-running
#      on every rerun.
#
st.set_page_config(
    page_title="PDF Chatbot",
    page_icon="📄",
    layout="centered"
)


# ══════════════════════════════════════════════════════════════════════════════
# BUILD RAG CHAIN / 构建 RAG 链
# ══════════════════════════════════════════════════════════════════════════════
#
# EN: @st.cache_resource is a Streamlit decorator that caches the return value
#     of this function. It only re-runs when the function arguments change.
#     Without it: every time the user types a message, Streamlit reruns the
#     script and would re-embed the entire PDF — extremely slow and costly.
#     With it: the chain is built ONCE per uploaded file and reused.
#
# ZH: @st.cache_resource 是 Streamlit 的缓存装饰器，缓存函数的返回值。
#     只有当函数参数变化时才重新执行。
#     没有它：每次用户发消息，Streamlit 重新运行脚本，重新嵌入整个 PDF——极慢且昂贵。
#     有了它：每个上传的文件只构建一次链，之后复用。
#
# EN: The difference between @st.cache_resource and @st.cache_data:
#     cache_resource → for shared objects like ML models, DB connections, chains
#     cache_data     → for data like dataframes, lists, dicts
#
# ZH: @st.cache_resource 和 @st.cache_data 的区别：
#     cache_resource → 用于共享对象，如 ML 模型、数据库连接、chain
#     cache_data     → 用于数据，如 dataframe、列表、字典
#
@st.cache_resource(show_spinner=False)
def build_chain(file_bytes, file_name):

    # ── 2a. Save PDF to temp file / 把 PDF 保存到临时文件 ──────────────────────
    #
    # EN: Streamlit gives us the uploaded file as bytes (raw binary data),
    #     but PyPDFLoader needs a file path on disk to read from.
    #     tempfile.NamedTemporaryFile creates a temporary file on your OS,
    #     we write the bytes into it, and pass its path to PyPDFLoader.
    #     delete=False means the file stays on disk after closing so the
    #     loader can still access it.
    #
    # ZH: Streamlit 把上传的文件给我们的是字节（原始二进制数据），
    #     但 PyPDFLoader 需要磁盘上的文件路径。
    #     tempfile.NamedTemporaryFile 在操作系统创建临时文件，
    #     我们把字节写入其中，把路径传给 PyPDFLoader。
    #     delete=False 表示关闭后文件仍保留在磁盘，加载器才能访问。
    #
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(file_bytes)   # write raw bytes to disk / 把原始字节写入磁盘
        tmp_path = tmp.name     # save the file path / 保存文件路径

    # ── 2b. Load PDF / 加载 PDF ────────────────────────────────────────────────
    #
    # EN: PyPDFLoader reads the PDF page by page and returns a list of
    #     Document objects: [Document(page_content="...", metadata={page: 0}), ...]
    #     Each Document has:
    #       .page_content → the text on that page
    #       .metadata     → dict with source file path and page number
    #
    # ZH: PyPDFLoader 逐页读取 PDF，返回 Document 对象列表：
    #     [Document(page_content="...", metadata={page: 0}), ...]
    #     每个 Document 有：
    #       .page_content → 该页的文本内容
    #       .metadata     → 包含源文件路径和页码的字典
    #
    loader = PyPDFLoader(tmp_path)
    pages = loader.load()   # returns List[Document] / 返回 Document 列表

    # ── 2c. Split into chunks / 分割成文本块 ───────────────────────────────────
    #
    # EN: WHY CHUNK? LLMs have a context window limit. More importantly,
    #     we don't want to send the whole document to the LLM every time —
    #     that's slow and expensive. Instead we split into chunks and only
    #     send the RELEVANT chunks. This is the core idea of RAG.
    #
    # ZH: 为什么分块？LLM 有上下文窗口限制。更重要的是，
    #     我们不想每次都把整个文档发给 LLM——那样既慢又贵。
    #     我们把文档切成块，只发送相关的块。这是 RAG 的核心思想。
    #
    # EN: RecursiveCharacterTextSplitter splits by priority:
    #     paragraph (\n\n) → sentence (\n) → word (" ") → character
    #     This preserves natural language boundaries as much as possible.
    #     chunk_overlap=50 → adjacent chunks share 50 chars to avoid
    #     cutting a sentence's meaning across two chunks.
    #
    # ZH: RecursiveCharacterTextSplitter 按优先级分割：
    #     段落(\n\n) → 句子(\n) → 单词(" ") → 字符
    #     尽可能保留自然语言边界。
    #     chunk_overlap=50 → 相邻块共享 50 个字符，避免句意被截断。
    #
    # INTERVIEW 面试:
    #   Q: Why use RecursiveCharacterTextSplitter over CharacterTextSplitter?
    #   A: Recursive tries multiple separators in order of priority to keep
    #      semantically meaningful boundaries. CharacterTextSplitter only
    #      uses one separator, which can cut sentences mid-way.
    #
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,     # each chunk ~500 characters / 每块约500字符
        chunk_overlap=50    # overlap between chunks / 块之间的重叠字符数
    )
    chunks = splitter.split_documents(pages)  # returns List[Document] / 返回分块后的 Document 列表

    # ── 2d. Embeddings + FAISS vector store / 嵌入 + FAISS 向量数据库 ───────────
    #
    # EN: WHAT ARE EMBEDDINGS?
    #     An embedding model converts text into a list of numbers (a vector).
    #     Example: "trading psychology" → [0.21, -0.45, 0.87, ...] (1536 numbers)
    #     Key property: texts with SIMILAR MEANING → SIMILAR vectors.
    #     "risk management" and "stop loss" will have similar vectors.
    #     "stop loss" and "chocolate cake" will have very different vectors.
    #     This lets us search by MEANING, not just exact keyword matching.
    #
    # ZH: 什么是 Embeddings（向量嵌入）？
    #     嵌入模型把文本转换成数字列表（向量）。
    #     例如："trading psychology" → [0.21, -0.45, 0.87, ...]（1536个数字）
    #     关键特性：语义相似的文本 → 相近的向量。
    #     "risk management" 和 "stop loss" 向量很接近。
    #     "stop loss" 和 "chocolate cake" 向量差异很大。
    #     这让我们可以按语义搜索，而不只是关键词匹配。
    #
    # EN: WHAT IS FAISS?
    #     FAISS (Facebook AI Similarity Search) is a vector database by Meta.
    #     It stores all the chunk vectors and lets you find the most similar
    #     ones to a query vector extremely fast (milliseconds even for millions).
    #     It runs locally in memory — no server needed. Free.
    #     Production alternatives: Pinecone, Chroma, Weaviate, Qdrant.
    #
    # ZH: 什么是 FAISS？
    #     FAISS（Facebook AI 相似度搜索）是 Meta 开发的向量数据库。
    #     存储所有文本块的向量，并能极快地找出与查询向量最相似的向量
    #     （即使数百万条数据也只需毫秒级）。
    #     在本地内存中运行——不需要服务器，免费。
    #     生产环境替代方案：Pinecone、Chroma、Weaviate、Qdrant。
    #
    # EN: FAISS.from_documents() does 3 things in one line:
    #     1. Takes each chunk's text
    #     2. Calls OpenAI Embeddings API → gets back a 1536-dim vector per chunk
    #     3. Stores all vectors in a FAISS index in memory
    #
    # ZH: FAISS.from_documents() 一行做了 3 件事：
    #     1. 取出每个文本块的文本
    #     2. 调用 OpenAI Embeddings API → 每块返回 1536 维向量
    #     3. 把所有向量存入内存中的 FAISS 索引
    #
    # INTERVIEW 面试:
    #   Q: What is the difference between a vector DB and a regular DB?
    #   A: Regular DB searches by exact match (WHERE name = 'John').
    #      Vector DB searches by semantic similarity using cosine similarity
    #      or euclidean distance on high-dimensional vectors. No exact match needed.
    #
    #   Q: What is cosine similarity?
    #   A: A metric that measures the angle between two vectors.
    #      Angle 0° = identical meaning (score 1.0).
    #      Angle 90° = completely unrelated (score 0.0).
    #      FAISS uses this to rank which chunks are most relevant to a question.
    #
    embeddings = OpenAIEmbeddings()                              # load embedding model / 加载嵌入模型
    vectorstore = FAISS.from_documents(chunks, embeddings)       # embed all chunks + store / 嵌入所有块并存储

    # ── 2e. Build the RetrievalQA chain / 构建 RetrievalQA 链 ──────────────────
    #
    # EN: THIS IS WHERE RAG IS FULLY ASSEMBLED.
    #     RAG = Retrieval-Augmented Generation. Two systems working together:
    #
    #     RETRIEVAL: question → embed → search FAISS → get top k chunks
    #     GENERATION: (chunks + question) → LLM → answer
    #
    #     The full flow when user asks a question:
    #     "What is the main lesson of chapter 3?"
    #          ↓
    #     Convert question to vector (OpenAIEmbeddings)
    #          ↓
    #     FAISS finds top 3 most similar chunks (k=3)
    #          ↓
    #     Prompt assembled: "Answer using this context:
    #                        [chunk1] [chunk2] [chunk3]
    #                        Question: What is the main lesson of chapter 3?"
    #          ↓
    #     gpt-4o-mini generates the answer
    #          ↓
    #     Returns { "query": "...", "result": "the answer" }
    #
    # ZH: 这里是 RAG 的完整组装。
    #     RAG = 检索增强生成。两个系统协同工作：
    #
    #     检索：问题 → 嵌入 → 搜索 FAISS → 获取 top k 文本块
    #     生成：（文本块 + 问题）→ LLM → 回答
    #
    #     用户提问时的完整流程：
    #     "第三章的主要教训是什么？"
    #          ↓
    #     把问题转成向量（OpenAIEmbeddings）
    #          ↓
    #     FAISS 找出最相似的 3 个文本块（k=3）
    #          ↓
    #     组装提示词："根据以下内容回答：
    #                  [块1] [块2] [块3]
    #                  问题：第三章的主要教训是什么？"
    #          ↓
    #     gpt-4o-mini 生成回答
    #          ↓
    #     返回 { "query": "...", "result": "回答内容" }
    #
    # EN: temperature=0 → deterministic, factual answers. No creativity.
    #     For document Q&A always use 0. Use 0.7+ for creative writing.
    #     k=3 → retrieve 3 chunks. Too few = missing context. Too many = noise.
    #     k=3 to k=5 is the standard for most RAG apps.
    #
    # ZH: temperature=0 → 确定性、事实性回答，无创意发挥。
    #     文档问答始终用 0。创意写作用 0.7+。
    #     k=3 → 检索 3 个块。太少 = 缺失上下文。太多 = 引入噪音。
    #     k=3 到 k=5 是大多数 RAG 应用的标准。
    #
    # INTERVIEW 面试:
    #   Q: Explain RAG in one sentence.
    #   A: RAG grounds an LLM's response in a specific knowledge base by
    #      retrieving relevant document chunks via semantic search and
    #      conditioning the generation on those chunks — reducing hallucination
    #      and enabling answers about private or recent data.
    #
    #   Q: What are RAG's limitations?
    #   A: 1. Retrieval quality — wrong chunks = wrong answer even with perfect LLM
    #      2. Chunk boundary issues — key info may be split across chunks
    #      3. No full-document reasoning — only sees k chunks at a time
    #      Solutions: reranking, larger k, better chunking, hybrid search.
    #
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
    )
    return chain, len(pages), len(chunks)  # return chain + stats for sidebar display / 返回链和统计信息用于侧边栏显示


# ══════════════════════════════════════════════════════════════════════════════
# STREAMLIT UI — SIDEBAR / 侧边栏
# ══════════════════════════════════════════════════════════════════════════════
#
# EN: `with st.sidebar:` is a context manager — everything indented inside
#     renders in the left sidebar panel, not the main page.
#     Streamlit has 3 layout containers: st.sidebar, st.columns, st.tabs.
#
# ZH: `with st.sidebar:` 是上下文管理器——里面缩进的内容渲染在左侧边栏，
#     而不是主页面。
#     Streamlit 有 3 种布局容器：st.sidebar、st.columns、st.tabs。
#
with st.sidebar:
    st.title("📄 PDF Chatbot")          # large title text / 大标题文字
    st.caption("Upload a PDF and ask anything about it.")  # small grey text / 小号灰色文字
    st.divider()                        # horizontal line / 水平分隔线

    # EN: st.file_uploader renders a drag-and-drop upload widget.
    #     type=["pdf"] restricts accepted file types.
    #     Returns a UploadedFile object (or None if nothing uploaded).
    #     UploadedFile has: .name (filename), .read() (bytes), .size (bytes)
    #
    # ZH: st.file_uploader 渲染一个拖放上传组件。
    #     type=["pdf"] 限制接受的文件类型。
    #     返回 UploadedFile 对象（未上传则返回 None）。
    #     UploadedFile 有：.name（文件名）、.read()（字节）、.size（大小）
    #
    uploaded_file = st.file_uploader("Choose a PDF", type="pdf")

    if uploaded_file:
        # EN: st.spinner() shows a loading animation while the indented code runs.
        #     show_spinner=False on cache_resource hides the default cache spinner,
        #     so we control the spinner message ourselves here.
        #
        # ZH: st.spinner() 在缩进代码运行时显示加载动画。
        #     cache_resource 上的 show_spinner=False 隐藏默认缓存加载提示，
        #     让我们自己在这里控制加载提示信息。
        #
        with st.spinner("Reading and indexing your PDF..."):
            chain, num_pages, num_chunks = build_chain(
                uploaded_file.read(),   # pass raw bytes / 传入原始字节
                uploaded_file.name      # used as cache key / 用作缓存键
            )

        # EN: st.success() shows a green success box.
        #     Other alert types: st.error() red, st.warning() yellow, st.info() blue
        #
        # ZH: st.success() 显示绿色成功提示框。
        #     其他提示类型：st.error() 红色、st.warning() 黄色、st.info() 蓝色
        #
        st.success(f"Ready! {num_pages} pages · {num_chunks} chunks indexed.")
        st.divider()

        # EN: st.button() renders a clickable button. Returns True only on the
        #     rerun triggered by clicking it, False on all other reruns.
        #
        # ZH: st.button() 渲染可点击按钮。只在点击触发的重新运行中返回 True，
        #     其他所有重新运行中返回 False。
        #
        if st.button("🗑️ Clear chat"):
            st.session_state.messages = []  # wipe chat history / 清空聊天记录
            st.rerun()                       # force immediate rerun / 强制立即重新运行


# ══════════════════════════════════════════════════════════════════════════════
# STREAMLIT UI — MAIN AREA / 主区域
# ══════════════════════════════════════════════════════════════════════════════
if not uploaded_file:
    # EN: Show empty state when no PDF is uploaded yet.
    #     st.columns([1,2,1]) creates 3 columns with width ratio 1:2:1.
    #     We use the middle column (col2) to center the message.
    #     unsafe_allow_html=True lets us use raw HTML like <br> for spacing.
    #
    # ZH: 未上传 PDF 时显示空状态。
    #     st.columns([1,2,1]) 创建宽度比例为 1:2:1 的三列。
    #     我们用中间列（col2）居中显示消息。
    #     unsafe_allow_html=True 允许使用 <br> 等原始 HTML 标签。
    #
    st.markdown("<br><br>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("## 👈 Upload a PDF to get started")
        st.caption("Your document stays local — nothing is stored.")
else:
    st.markdown(f"### 💬 Chatting with: `{uploaded_file.name}`")
    st.divider()

    # ── Session state — chat history / 会话状态——聊天记录 ──────────────────────
    #
    # EN: st.session_state is a dict that PERSISTS across Streamlit reruns.
    #     Normal Python variables reset to their initial value on every rerun.
    #     session_state survives reruns within the same browser session.
    #     We use it to store the full chat history so it doesn't disappear
    #     every time the user sends a new message.
    #
    # ZH: st.session_state 是一个在 Streamlit 重新运行之间持久保存的字典。
    #     普通 Python 变量每次重新运行都会重置为初始值。
    #     session_state 在同一浏览器会话的重新运行中保持不变。
    #     我们用它存储完整聊天记录，这样每次用户发新消息时记录不会消失。
    #
    # INTERVIEW 面试:
    #   Q: How do you persist state between reruns in Streamlit?
    #   A: Use st.session_state. It's a dict-like object that survives reruns
    #      within a browser session. For persistence across sessions (e.g. after
    #      refresh), you'd need a database or external storage.
    #
    if "messages" not in st.session_state:
        st.session_state.messages = []  # initialize empty chat history / 初始化空聊天记录

    # EN: Render all previous messages from history.
    #     st.chat_message(role) creates a chat bubble styled for "user" or "assistant".
    #     role="user" → right-aligned bubble with person icon
    #     role="assistant" → left-aligned bubble with bot icon
    #
    # ZH: 渲染历史中所有之前的消息。
    #     st.chat_message(role) 创建样式化的聊天气泡，"user" 或 "assistant"。
    #     role="user" → 右对齐气泡，人物图标
    #     role="assistant" → 左对齐气泡，机器人图标
    #
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])  # render message text as markdown / 以 markdown 渲染消息文本

    # ── Chat input + RAG response / 聊天输入 + RAG 响应 ───────────────────────
    #
    # EN: st.chat_input() renders the input bar pinned to the bottom of the page.
    #     It returns the user's typed text when they press Enter, else None.
    #     The walrus operator `:=` assigns AND checks in one line:
    #     `if question := st.chat_input(...)` means:
    #     "assign the return value to `question`, and if it's not None, run the block"
    #
    # ZH: st.chat_input() 渲染固定在页面底部的输入栏。
    #     用户按回车时返回输入的文本，否则返回 None。
    #     海象运算符 `:=` 在一行内赋值并判断：
    #     `if question := st.chat_input(...)` 意思是：
    #     "把返回值赋给 `question`，如果不是 None 就执行代码块"
    #
    if question := st.chat_input("Ask something about your PDF..."):

        # save user message to history / 把用户消息保存到历史记录
        st.session_state.messages.append({"role": "user", "content": question})

        # display user bubble immediately / 立即显示用户气泡
        with st.chat_message("user"):
            st.markdown(question)

        # EN: Generate the AI response using the RAG chain.
        #     chain.invoke(question) triggers the full pipeline:
        #     question → embed → FAISS search → top 3 chunks → GPT → answer
        #     Returns dict: { "query": "...", "result": "the answer text" }
        #
        # ZH: 使用 RAG 链生成 AI 回答。
        #     chain.invoke(question) 触发完整流程：
        #     问题 → 嵌入 → FAISS 搜索 → top 3 文本块 → GPT → 回答
        #     返回字典：{ "query": "...", "result": "回答文本" }
        #
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):        # show loading dots / 显示加载点动画
                result = chain.invoke(question)    # run the full RAG pipeline / 运行完整 RAG 流程
                answer = result["result"]          # extract answer text / 提取回答文本
            st.markdown(answer)                    # render answer as markdown / 以 markdown 渲染回答

        # save assistant message to history / 把 AI 回答保存到历史记录
        st.session_state.messages.append({"role": "assistant", "content": answer})