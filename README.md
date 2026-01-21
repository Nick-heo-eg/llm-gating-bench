# LLM Gating Bench

We benchmark a simple idea: deciding whether to call an LLM **before generation**.

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
→ [stop-first-rag](https://github.com/nick-heo123/stop-first-rag)

---

## License

MIT
