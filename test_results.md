# Evaluation Results — Vertex AI RAG

This document tracks evaluation iterations of the RAG pipeline. Each iteration records what was tested, what was learned, and what changed.

---

## Configuration (Constant Across Iterations)

- **Generator:** Gemini 2.5 Flash (Vertex AI)
- **Embeddings:** Vertex AI `text-embedding-005`
- **Reranker:** Vertex AI Ranking API (top 3 from top 10)
- **Judge LLM:** GPT-4o-mini (OpenAI) — *independent from generator to avoid self-preference bias*
- **Test Document:** "Best Loser Wins" by Tom Hougaard (~250 pages, 713 chunks)

---

## Iteration 1 — Baseline

### Eval Set
5 questions, including 1 deliberate out-of-domain question.

### Results

| Metric | Score |
|--------|-------|
| Faithfulness | 80% |
| Context Recall | 80% |
| Recall@3 | 100% |
| MRR | 70.8% |

### Anomalies Found

1. **Q2:** Faithfulness 100% but Context Recall 0%. On inspection, the pipeline's answer was *more accurate* than the reference ground truth. The dataset was wrong, not the system.
2. **Q4:** Faithfulness 0% on a refusal answer. Investigation revealed this is a known RAGAS limitation: refusals contain no factual claims to verify.

### Action

Rewrite ground truths for Q1–Q3 to match the actual book content. Keep Q4 as-is and document the limitation.

---

## Iteration 2 — Ground Truth Refinement

### Changes Made

- Q1: Simplified to match book's framing ("changing how we think about losing")
- Q2: Rewrote to "hard-coded to avoid pain and keep us alive"; key_phrase changed from "normal" to "pain"
- Q3: Refined for specificity around emotional desensitization

### Results

| Metric | Iter 1 | Iter 2 | Δ |
|--------|--------|--------|----|
| Faithfulness | 80% | 80% | — |
| Context Recall | 80% | **90%** | **+10** ✅ |
| Recall@3 | 100% | 100% | — |
| MRR | 70.8% | 70.8% | — |

### New Insight

Q1 dropped to 50% Context Recall after the rewrite. Investigation: the new ground truth contains 2 distinct claims, and the answer covered only 1. This is **not regression** — it's RAGAS correctly identifying a coverage gap.

### Lessons Learned

1. **Debug the data, not just the model.** Outlier metrics often signal dataset issues.
2. **Multi-claim ground truths surface real diagnostic signal.** A drop in score on a corrected dataset can be informative, not a regression.
3. **Refusal artifacts are acceptable trade-offs.** Better to keep refusal tests with known 0% scoring than to remove them and lose production-critical behavior.

---

## Future Iterations

- Expand to 20+ questions including adversarial / multi-hop / cross-section
- Add human review on a sample to validate the LLM judge
- Test alternative rerankers to push MRR > 0.85
- Track scores after migrating to Vertex AI Vector Search