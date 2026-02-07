# LLM Gating Bench

## Position

This repository is part of the **Judgment Boundary** work:
a set of experiments and specifications focused on
*when AI systems must stop or not execute*.

See the overarching map:
→ https://github.com/Nick-heo-eg/stop-first-rag/blob/master/JUDGMENT_BOUNDARY_MANIFEST.md

---

We benchmark a simple idea: deciding whether to call an LLM **before generation**.

---

This repository benchmarks decision gating and boundary placement rather than model capability or output quality.

## Scope and Assumptions

This benchmark evaluates gating decisions **before calling an LLM**.

- **Target environment**: CPU-only, local LLM setups
- **Workloads**: synthetic and controlled
- **Goal**: measure relative performance impact, not real-world task quality

---

## Why It Matters

- CPU-only local LLMs are slow (seconds to minutes per call)
- Most systems decide *after* generation (hallucination detection, output validation)
- We show that **pre-generation gating** changes the cost profile

---

## Key Results

| Metric           | Baseline (Naive) | Stop-First | Improvement |
|------------------|------------------|------------|-------------|
| Avg Latency      | 100.1s           | 19.4s      | **5.17×**   |
| LLM Call Rate    | 100%             | 60%        | **40% ↓**   |
| Token Usage      | 716              | 461        | **36% ↓**   |

> Measured on CPU-only Ollama (phi3:mini), 5 mixed queries (3 answerable, 2 unanswerable).

**Reported speedups apply only under the conditions described above. Results should not be generalized without validation in other environments.**

---

## Scope and Non-Goals

This repository benchmarks **pre-generation gating** as a cost-control pattern.

It does **not** claim:
- State-of-the-art RAG accuracy
- Improvements in answer quality or reasoning
- Superiority over advanced adaptive RAG methods

The goal is to measure how **deciding before generation** changes latency, token usage, and LLM call rate.

---

## Dataset Size and Interpretation

Current results are based on a **small-n measurement** (5 queries).

This benchmark is intended to show **effect magnitude** under a constrained environment (CPU-only local LLM), not statistical generalization.

Scripts and structure are provided to scale:
- number of queries
- corpus size
- gating policies

Larger-scale evaluation is a planned extension.

---

## When This Is Useful

This benchmark is most relevant when:
- LLM inference is expensive (CPU-only, large models, rate limits)
- Many queries are unanswerable or off-topic
- Auditability of "why we did not answer" matters

It is less relevant when:
- LLM calls are cheap (fast GPU inference)
- All queries are well-formed and answerable
- Generation cost is negligible compared to retrieval

---

## What is Stop-First

```
retrieve → gate → generate (or stop)
           ↓
      decision + reason
```

**Core idea**: Decide whether to call the LLM based on retrieval signals, **before** incurring generation cost.

**Key properties**:
- Typed stop reasons (`no_data`, `conflict`, `low_confidence`)
- Auditable decisions (every stop is logged with reason)
- Extensible gates (retrieval score, document conflict, confidence gap)

---

## Quick Start

```bash
# 1. Start Ollama
ollama serve
ollama pull phi3:mini

# 2. Run benchmark (3 variants: naive, threshold, stop-first)
PYTHONPATH=/path/to/llm-gating-bench python3 bench/run.py

# 3. View metrics
PYTHONPATH=/path/to/llm-gating-bench python3 bench/metrics.py
```

Expected runtime: ~15 minutes on CPU

---

## Benchmark Design

We compare **3 variants** on identical retrieval:

1. **baseline_naive**: Always generate (no gate)
2. **baseline_score_threshold**: Single signal gate (BM25 score > τ)
3. **stop_first**: Multi-signal gate (score + conflict + confidence gap)

**Fairness**:
- Same corpus, same queries, same LLM
- Threshold τ tuned on answerable set (Recall ≥ 0.95)
- Stop-First uses same τ as baseline

See [`docs/methodology.md`](docs/methodology.md) for details.

---

## Results Breakdown

**Queries that generated** (3/5):
- "What is your return policy?" → All 3 variants generated
- "How do I reset my password?" → All 3 variants generated
- "What payment methods do you accept?" → All 3 variants generated

**Queries that stopped** (2/5):
- "How do I cook pasta?" → Threshold + Stop-First stopped (BM25 score = 0.0)
- "What is the capital of France?" → Threshold + Stop-First stopped (BM25 score = 1.79 < 2.0)

**Key finding:** Stop-First achieved the same precision as threshold-based gating, while avoiding generation latency entirely for stopped queries (0ms vs 8–161s).

---

## When to Use This

Stop-First gating is most valuable when:

- **LLM calls are expensive** (CPU-only, large models, rate limits)
- **Many queries are unanswerable** (off-topic, insufficient data)
- **Audit trail matters** (why did we not answer?)

Not useful when:
- LLM calls are cheap (small models, GPU inference)
- All queries are well-formed and answerable

---

## What This Is Not

- This is not a new retrieval algorithm.
- This does not improve model intelligence or answer quality.
- This is not limited to RAG systems.

This repository focuses on **when** to call an LLM, not **how** the LLM reasons.

---

## Reference Implementation

This benchmark includes a Stop-First implementation originally derived from a RAG use case.

For RAG-specific patterns (document conflict detection, evidence scoring):
→ [stop-first-rag](https://github.com/Nick-heo-eg/stop-first-rag)

---

## Why This Exists

Pre-generation gating provides concrete operational benefits when LLM calls are expensive or unreliable:

- **Reduced cost and latency** by avoiding generation for unanswerable queries
  (40% fewer LLM calls, 5.17× latency speedup on CPU-only)

- **No hallucinated answers** when evidence is missing
  (queries with no supporting documents stop with explicit reasons)

- **Auditable non-answers** via typed stop reasons
  (`no_data`, `conflict`, `low_confidence`), enabling debugging and corpus improvement

- **Safer operation in high-risk domains**, where not answering is preferable to fabricating confidence

These effects are quantified in the benchmark results above.

---

## License

MIT
