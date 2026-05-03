import streamlit as st
import requests
import pandas as pd
import os
# ══════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Vertex AI RAG",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)


BACKEND_URL = os.environ.get("BACKEND_URL", "http://localhost:8000")

# ══════════════════════════════════════════════════════════════════════════════
# CUSTOM CSS — visual polish without leaving Streamlit
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
    /* Tighter top spacing */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1400px;
    }

    /* Header banner */
    .header-banner {
        background: linear-gradient(135deg, #4285F4 0%, #34A853 100%);
        color: white;
        padding: 1.5rem 2rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 12px rgba(66, 133, 244, 0.15);
    }
    .header-banner h1 {
        color: white;
        margin: 0;
        font-size: 1.8rem;
        font-weight: 600;
    }
    .header-banner p {
        color: rgba(255, 255, 255, 0.9);
        margin: 0.3rem 0 0 0;
        font-size: 0.95rem;
    }

    /* Sidebar polish */
    section[data-testid="stSidebar"] {
        background-color: #F8F9FA;
        border-right: 1px solid #E8EAED;
    }

    /* Sidebar section spacing */
    .sidebar-section {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        margin-bottom: 1rem;
        border: 1px solid #E8EAED;
    }

    /* Metric cards — colored by score */
    [data-testid="stMetric"] {
        background: white;
        border: 1px solid #E8EAED;
        border-radius: 12px;
        padding: 1rem;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);
        transition: transform 0.15s ease, box-shadow 0.15s ease;
    }
    [data-testid="stMetric"]:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08);
    }
    [data-testid="stMetricValue"] {
        font-size: 2rem !important;
        font-weight: 700;
        color: #4285F4;
    }
    [data-testid="stMetricLabel"] {
        font-size: 0.85rem;
        color: #5F6368;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        font-weight: 600;
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        border-bottom: 1px solid #E8EAED;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 10px 20px;
        font-weight: 500;
    }
    .stTabs [aria-selected="true"] {
        background-color: #E8F0FE;
        border-radius: 8px 8px 0 0;
        color: #4285F4;
    }

    /* Expanders */
    .streamlit-expanderHeader {
        background-color: #F8F9FA;
        border-radius: 8px;
        font-weight: 500;
    }

    /* Buttons */
    .stButton > button {
        border-radius: 8px;
        font-weight: 500;
        transition: all 0.15s ease;
    }
    .stButton > button[kind="primary"] {
        background-color: #4285F4;
    }
    .stButton > button[kind="primary"]:hover {
        background-color: #357AE8;
        transform: translateY(-1px);
    }

    /* Status badges */
    .status-badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 12px;
        font-size: 0.8rem;
        font-weight: 500;
    }
    .status-success { background: #E6F4EA; color: #137333; }
    .status-warning { background: #FEF7E0; color: #B06000; }
    .status-error   { background: #FCE8E6; color: #C5221F; }

    /* Chat message bubble */
    [data-testid="stChatMessage"] {
        background-color: #F8F9FA;
        border-radius: 12px;
        padding: 1rem;
        margin-bottom: 0.5rem;
    }

    /* Reduce default spacing */
    div[data-testid="stVerticalBlock"] > div {
        gap: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# HEADER BANNER
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="header-banner">
    <h1>📄 Vertex AI RAG Evaluation</h1>
    <p>Production-grade RAG with multi-judge evaluation • Powered by Gemini 2.5 Flash</p>
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# SESSION STATE
# ══════════════════════════════════════════════════════════════════════════════
if "messages" not in st.session_state:
    st.session_state.messages = []
if "pdf_indexed" not in st.session_state:
    st.session_state.pdf_indexed = False
if "chunk_count" not in st.session_state:
    st.session_state.chunk_count = 0


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("### ⚙️ Pipeline Control")
    st.caption("Configure and run the RAG system")

    # ── System Status ────────────────────────────────────────────────
    st.markdown("#### 📊 Status")
    status_cols = st.columns(2)
    with status_cols[0]:
        if st.session_state.pdf_indexed:
            st.markdown(
                '<span class="status-badge status-success">✓ Indexed</span>',
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                '<span class="status-badge status-warning">⏳ No PDF</span>',
                unsafe_allow_html=True
            )
    with status_cols[1]:
        if "eval_results" in st.session_state:
            st.markdown(
                '<span class="status-badge status-success">✓ Evaluated</span>',
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                '<span class="status-badge status-warning">⏳ Not eval</span>',
                unsafe_allow_html=True
            )

    if st.session_state.chunk_count > 0:
        st.caption(f"📦 {st.session_state.chunk_count} chunks loaded")

    st.divider()

    # ── PDF Upload ───────────────────────────────────────────────────
    st.markdown("#### 1️⃣  Upload Document")
    uploaded_file = st.file_uploader(
        "Choose a PDF file",
        type="pdf",
        label_visibility="collapsed",
    )

    if uploaded_file:
        st.caption(f"📎 {uploaded_file.name}")
        if st.button("🚀 Index Document", type="primary", use_container_width=True):
            with st.spinner("Embedding chunks via Vertex AI..."):
                files = {
                    "file": (
                        uploaded_file.name,
                        uploaded_file.getvalue(),
                        "application/pdf",
                    )
                }
                response = requests.post(f"{BACKEND_URL}/ingest", files=files)
                if response.status_code == 200:
                    chunks = response.json().get("chunks", 0)
                    st.session_state.pdf_indexed = True
                    st.session_state.chunk_count = chunks
                    st.success(f"✅ Indexed ({chunks} chunks)")
                    st.rerun()
                else:
                    st.error("Failed to index PDF.")

    st.divider()

    # ── Evaluation ───────────────────────────────────────────────────
    st.markdown("#### 2️⃣  Run Evaluation")
    st.caption("RAGAS metrics with independent GPT judge")

    eval_disabled = not st.session_state.pdf_indexed
    if st.button(
        "🧪 Run Evaluation",
        use_container_width=True,
        disabled=eval_disabled,
        help="Upload a PDF first" if eval_disabled else None,
    ):
        with st.spinner("Running evaluation (1–3 min)..."):
            try:
                response = requests.get(f"{BACKEND_URL}/evaluate", timeout=300)
                if response.status_code == 200:
                    st.session_state.eval_results = response.json()
                    st.success("✅ Evaluation complete!")
                    st.rerun()
                else:
                    st.error(f"Eval failed: {response.text[:200]}")
            except requests.exceptions.Timeout:
                st.error("Evaluation timed out.")
            except Exception as e:
                st.error(f"Error: {str(e)[:100]}")

    st.divider()

    # ── About ─────────────────────────────────────────────────────────
    with st.expander("ℹ️  About this project"):
        st.markdown("""
        **Stack:**
        - 🧠 Gemini 2.5 Flash
        - 🔍 Vertex AI Embeddings
        - 📊 RAGAS + GPT-4o-mini judge
        - ⚡ FastAPI + Streamlit

        **Repo:** [GitHub](https://github.com/IristIsGood/Vertex-RAG-Eval)
        """)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN AREA — TABS
# ══════════════════════════════════════════════════════════════════════════════
tab_chat, tab_eval = st.tabs(["💬 Chat", "📊 Evaluation Results"])


# ──────────────────────────────────────────────────────────────────────
# CHAT TAB
# ──────────────────────────────────────────────────────────────────────
with tab_chat:
    if not st.session_state.pdf_indexed:
        st.info(
            "👈 **Upload and index a PDF** in the sidebar to start asking questions."
        )

    # Display chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Chat input
    if prompt := st.chat_input(
        "Ask a question about the document...",
        disabled=not st.session_state.pdf_indexed,
    ):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Searching document..."):
                response = requests.post(
                    f"{BACKEND_URL}/ask", json={"question": prompt}
                )
                if response.status_code == 200:
                    answer = response.json()["answer"]
                    st.markdown(answer)
                    st.session_state.messages.append(
                        {"role": "assistant", "content": answer}
                    )
                else:
                    st.error("Backend error. Is `app.py` running?")


# ──────────────────────────────────────────────────────────────────────
# EVALUATION TAB
# ──────────────────────────────────────────────────────────────────────
def score_color(score):
    """Return color for score visualization."""
    if score >= 0.85:
        return "#137333"  # green
    elif score >= 0.70:
        return "#B06000"  # amber
    else:
        return "#C5221F"  # red


def score_emoji(score):
    if score >= 0.85:
        return "🟢"
    elif score >= 0.70:
        return "🟡"
    else:
        return "🔴"


with tab_eval:
    if "eval_results" not in st.session_state:
        st.info("👈 Run **'🧪 Run Evaluation'** in the sidebar to see metrics.")
    else:
        results = st.session_state.eval_results

        # ── Summary cards ──────────────────────────────────────────
        st.markdown("### 📈 Overall Scores")

        if "summary" in results:
            summary = results["summary"]
            cols = st.columns(len(summary))
            for col, (metric, score) in zip(cols, summary.items()):
                with col:
                    score_val = float(score) if score is not None else 0
                    score_pct = round(score_val * 100, 1)
                    emoji = score_emoji(score_val)
                    col.metric(
                        label=f"{emoji} {metric.replace('_', ' ').title()}",
                        value=f"{score_pct}%",
                    )

            st.divider()

        # ── Configuration ──────────────────────────────────────────
        if "config" in results:
            with st.expander("⚙️  Evaluation Configuration", expanded=False):
                config = results["config"]
                cfg_cols = st.columns(2)
                with cfg_cols[0]:
                    st.markdown("**🤖 Generator**")
                    st.code(config.get("generator", "N/A"), language=None)
                    st.markdown("**⚖️ Judge LLM**")
                    st.code(config.get("judge", "N/A"), language=None)
                with cfg_cols[1]:
                    st.markdown("**🔢 Judge Embeddings**")
                    st.code(config.get("judge_embeddings", "N/A"), language=None)
                    st.markdown("**💡 Rationale**")
                    st.caption(config.get("rationale", "N/A"))

                if "metrics_explained" in config:
                    st.divider()
                    st.markdown("**📚 Metrics Explained**")
                    for metric, explanation in config["metrics_explained"].items():
                        st.markdown(f"- **`{metric}`** — {explanation}")

        # ── Per-question details ───────────────────────────────────
        if "details" in results:
            st.divider()
            st.markdown("### 🔍 Per-Question Details")

            details = results["details"]
            if isinstance(details, list) and len(details) > 0:
                df = pd.DataFrame(details)

                for i, row in df.iterrows():
                    question = (
                        row.get("user_input")
                        or row.get("question")
                        or "N/A"
                    )
                    answer = (
                        row.get("response")
                        or row.get("answer")
                        or "N/A"
                    )
                    ground_truth = (
                        row.get("reference")
                        or row.get("ground_truth")
                        or "N/A"
                    )
                    contexts = (
                        row.get("retrieved_contexts")
                        or row.get("contexts")
                        or []
                    )

                    # Calculate average score for this question
                    f_score = float(row["faithfulness"]) if pd.notna(row.get("faithfulness")) else None
                    cr_score = float(row["context_recall"]) if pd.notna(row.get("context_recall")) else None
                    avg_score = None
                    if f_score is not None and cr_score is not None:
                        avg_score = (f_score + cr_score) / 2

                    badge = (
                        score_emoji(avg_score) if avg_score is not None else "⚪"
                    )

                    with st.expander(
                        f"{badge}  Q{i+1}: {str(question)[:80]}",
                        expanded=False,
                    ):
                        # Q&A side-by-side
                        qa_cols = st.columns([1, 1])
                        with qa_cols[0]:
                            st.markdown("**🤖 Generated Answer**")
                            st.info(answer)
                        with qa_cols[1]:
                            st.markdown("**✅ Ground Truth**")
                            st.success(ground_truth)

                        # Metric scores
                        st.markdown("**📊 Scores**")
                        score_cols = st.columns(2)
                        if f_score is not None:
                            score_cols[0].metric(
                                "Faithfulness",
                                f"{round(f_score * 100, 1)}%",
                            )
                        if cr_score is not None:
                            score_cols[1].metric(
                                "Context Recall",
                                f"{round(cr_score * 100, 1)}%",
                            )

                        # Refusal note
                        if isinstance(answer, str) and (
                            "do not have enough information" in answer.lower()
                        ):
                            st.warning(
                                "💡 **Refusal artifact:** This question correctly "
                                "refused to hallucinate. RAGAS faithfulness=0 is a "
                                "known limitation when the answer is a refusal — "
                                "there are no claims to verify."
                            )

                        # Retrieved contexts
                        if isinstance(contexts, (list, tuple)) and len(contexts) > 0:
                            with st.expander(
                                f"📚 Retrieved Contexts ({len(contexts)} chunks)",
                                expanded=False,
                            ):
                                for j, ctx in enumerate(contexts):
                                    st.markdown(f"**Chunk {j+1}:**")
                                    st.code(
                                        ctx[:600] + "..."
                                        if len(str(ctx)) > 600
                                        else ctx,
                                        language=None,
                                    )
            else:
                st.json(details)