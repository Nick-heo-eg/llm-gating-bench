# Reproducibility Notes

## Environment

- OS: Linux
- LLM runtime: Ollama
- Model: phi3:mini
- Inference: CPU-only
- Python: 3.10+

## Versions

- Ollama version: 0.10.1
- Python dependencies: see requirements (rank-bm25, ollama)

## Hardware

- CPU-only machine
- No GPU acceleration used

## How to Reproduce

```bash
ollama serve
ollama pull phi3:mini

PYTHONPATH=/path/to/llm-gating-bench python3 bench/run.py
PYTHONPATH=/path/to/llm-gating-bench python3 bench/metrics.py
```

Expected runtime:
- ~15 minutes total on CPU-only machine

## Notes

Latency numbers are highly environment-dependent.
Absolute values may differ, but relative behavior between variants should remain consistent.
