import json
from pathlib import Path
from typing import List, Dict
import sys

from bench.retrieval import load_corpus_jsonl, BM25Retriever
from bench.rag_baseline_naive import RAGBaselineNaive
from bench.rag_baseline_threshold import RAGBaselineThreshold
from bench.rag_stop_first import RAGStopFirst

# Import Ollama LLM
try:
    from bench.llm_ollama import ollama_generate, check_model_available
    USE_OLLAMA = True
except ImportError:
    print("Warning: ollama not available, using mock LLM")
    USE_OLLAMA = False


# -------------------------
# Mock LLM (fallback)
# -------------------------
def mock_llm_generate(query, retrieval):
    # Deterministic + cheap
    return {
        "prompt_tokens": 120,
        "gen_tokens": 180,
        "latency_ms": 60,
    }


def load_queries(path: str) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    # paths
    corpus_path = "corpus/corpus.jsonl"
    queries_path = "datasets/test_queries.json"
    output_path = Path("results/run_results.jsonl")
    output_path.parent.mkdir(exist_ok=True)

    # Select LLM
    if USE_OLLAMA:
        if not check_model_available():
            print("\nFalling back to mock LLM")
            llm_generate_fn = mock_llm_generate
        else:
            print("\n✓ Using Ollama (phi3:mini)")
            llm_generate_fn = ollama_generate
    else:
        print("\n✓ Using mock LLM")
        llm_generate_fn = mock_llm_generate

    # load
    corpus = load_corpus_jsonl(corpus_path)
    retriever = BM25Retriever(corpus)

    queries = load_queries(queries_path)

    # tau calibrated for BM25 score meaningfulness
    # answerable min=2.58, all pass at τ=2.0
    TAU_THRESHOLD = 2.0      # BM25 score threshold (recall=1.0)
    TAU_STOP = 2.0           # stop-first uses same base threshold

    # variants
    rag_naive = RAGBaselineNaive(retriever, llm_generate_fn)
    rag_threshold = RAGBaselineThreshold(retriever, TAU_THRESHOLD, llm_generate_fn)
    rag_stop_first = RAGStopFirst(retriever, TAU_STOP, llm_generate_fn)

    variants = [
        rag_naive,
        rag_threshold,
        rag_stop_first,
    ]

    # run
    with open(output_path, "w", encoding="utf-8") as out:
        for q in queries:
            query_text = q["question"]

            for rag in variants:
                result = rag.run(query_text)
                out.write(json.dumps(result.__dict__, default=str) + "\n")

    print(f"Saved results to {output_path}")


if __name__ == "__main__":
    main()
