"""
Ollama LLM integration for benchmarking.

Optimized for CPU-only environments.
"""
import time

try:
    import ollama
except ImportError:
    raise ImportError(
        "ollama package not found. Install with: pip install ollama"
    )

# Optimized for CPU-only laptop environments
MODEL_NAME = "phi3:mini"


def ollama_generate(query: str, retrieval) -> dict:
    """
    Generate answer using Ollama.

    Args:
        query: User question
        retrieval: RetrievalResult object with retrieved_doc_ids

    Returns:
        dict with prompt_tokens, gen_tokens, latency_ms
    """
    # Extract doc text from corpus (simplified for benchmark)
    # In real implementation, would fetch full doc text
    retrieved_docs = retrieval.retrieved_doc_ids[:3]  # top 3

    # Build context from doc IDs (placeholder)
    context = "\n".join([f"Document: {doc_id}" for doc_id in retrieved_docs])

    prompt = f"""Context:
{context}

Question:
{query}

Answer:"""

    start = time.perf_counter()

    response = ollama.generate(
        model=MODEL_NAME,
        prompt=prompt,
        options={
            "temperature": 0,      # deterministic
            "num_predict": 128,    # shorter responses for CPU
        }
    )

    latency_ms = int((time.perf_counter() - start) * 1000)

    return {
        "prompt_tokens": response.get("prompt_eval_count", 0),
        "gen_tokens": response.get("eval_count", 0),
        "latency_ms": latency_ms
    }


def check_model_available():
    """Check if phi3:mini is available in Ollama."""
    try:
        response = ollama.list()
        models = response.get('models', []) if isinstance(response, dict) else response.models

        available = any(MODEL_NAME in m.model for m in models)

        if not available:
            print(f"\nModel '{MODEL_NAME}' not found.")
            print(f"Pull it with: ollama pull {MODEL_NAME}")
            return False

        return True
    except Exception as e:
        print(f"Error checking Ollama: {e}")
        print("Make sure Ollama is running.")
        return False


if __name__ == "__main__":
    # Quick test
    if check_model_available():
        print(f"\n✓ {MODEL_NAME} is available")

        # Test generation
        class MockRetrieval:
            retrieved_doc_ids = ["doc_001", "doc_002"]

        print("\nTesting generation...")
        result = ollama_generate("What is the return policy?", MockRetrieval())
        print(f"✓ Generation successful")
        print(f"  Latency: {result['latency_ms']}ms")
        print(f"  Tokens: {result['gen_tokens']}")
