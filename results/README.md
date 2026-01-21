# Results

This directory contains benchmark results from running the 3-way comparison.

## Files

- **`run_results.jsonl`**: Raw results (15 lines = 5 queries × 3 variants)
- **`metrics_summary.json`**: Aggregated metrics and comparison

## Key Metrics

From `metrics_summary.json`:

```json
{
  "summary": {
    "baseline_naive": {
      "total_queries": 5,
      "llm_calls": 5,
      "llm_call_rate": 1.0,
      "avg_total_latency_ms": 100143.6
    },
    "stop_first": {
      "total_queries": 5,
      "llm_calls": 3,
      "llm_call_rate": 0.6,
      "avg_total_latency_ms": 19370
    }
  },
  "comparison": {
    "llm_call_reduction": 0.4,
    "latency_speedup": 5.17
  }
}
```

## What This Shows

**Stop-First gating achieved:**
- **5.17× latency speedup** (100s → 19s avg)
- **40% fewer LLM calls** (5 → 3 calls)
- **Same precision** as threshold baseline (both stopped 2/5 queries)

**Key insight:** Pre-generation decisions avoid latency entirely for stopped queries (0ms vs 8-161s for generation).

## Reproduction

To reproduce these results:

```bash
PYTHONPATH=/path/to/llm-gating-bench python3 bench/run.py
PYTHONPATH=/path/to/llm-gating-bench python3 bench/metrics.py
```

Expected runtime: ~15 minutes on CPU with Ollama (phi3:mini)
