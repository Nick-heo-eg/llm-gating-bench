# Scaling Plan (Planned Work)

Planned extensions include:

## 1. Larger query sets (n ≥ 100)
- Gate-only evaluation for stop rate and false-stop rate
- Sampled LLM calls for cost measurement
- Statistical validation of latency improvements

## 2. Additional gating policies
- Reranker-based gates
- Confidence-gap heuristics
- External verifier hooks
- Multi-stage gating (cheap → expensive checks)

## 3. Quality-side metrics
- False-stop rate on answerable queries
- Hallucination prevention rate on unanswerable queries
- Precision/recall of gating decisions

## 4. Additional LLM backends
- vLLM, llama.cpp comparisons
- GPU inference baseline
- API-based LLMs (OpenAI, Anthropic)

These are intentionally out of scope for the initial benchmark, which focuses on **pre-generation cost behavior**.
