"""
Threshold (τ) tuning for baseline_score_threshold.

Goal: Find τ that satisfies Recall >= 0.95 on answerable queries.
Strategy: Select the highest τ meeting this constraint.
"""
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from bench.retrieval import load_corpus_jsonl, BM25Retriever


def load_answerable(path: str):
    """Load answerable queries."""
    queries = []
    with open(path, 'r') as f:
        for line in f:
            queries.append(json.loads(line))
    return queries


def tune_tau(retriever, queries, tau_grid):
    """
    Tune τ on answerable set.

    Returns:
        dict with tau_results, best_tau, scores
    """
    # Get retrieval scores for all queries
    scores = []
    for q in queries:
        result = retriever.retrieve(q['question'])
        scores.append(result.max_score)

    scores = np.array(scores)
    n_total = len(scores)

    # Evaluate each τ
    tau_results = []
    for tau in tau_grid:
        # How many queries have score >= tau?
        n_answered = np.sum(scores >= tau)
        recall = n_answered / n_total

        tau_results.append({
            'tau': tau,
            'n_answered': int(n_answered),
            'n_total': n_total,
            'recall': round(recall, 3)
        })

    # Select: highest τ with recall >= 0.95
    valid = [r for r in tau_results if r['recall'] >= 0.95]

    if not valid:
        print("Warning: No τ satisfies recall >= 0.95")
        print("Using τ with highest recall instead")
        best = max(tau_results, key=lambda x: x['recall'])
    else:
        best = max(valid, key=lambda x: x['tau'])

    return {
        'tau_results': tau_results,
        'best_tau': best['tau'],
        'best_recall': best['recall'],
        'scores': scores.tolist(),
        'score_stats': {
            'min': float(scores.min()),
            'max': float(scores.max()),
            'mean': float(scores.mean()),
            'std': float(scores.std())
        }
    }


def plot_score_distribution(scores, best_tau, output_path):
    """Plot histogram of retrieval scores."""
    plt.figure(figsize=(8, 5))
    plt.hist(scores, bins=20, alpha=0.7, edgecolor='black')
    plt.axvline(best_tau, color='red', linestyle='--', linewidth=2,
                label=f'Best τ = {best_tau:.2f}')
    plt.xlabel('Retrieval Score')
    plt.ylabel('Frequency')
    plt.title('Retrieval Score Distribution (Answerable Set)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved: {output_path}")


def main():
    # Paths
    corpus_path = "corpus/corpus.jsonl"
    answerable_path = "datasets/answerable.jsonl"
    output_dir = Path("results/tau_tuning")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    corpus = load_corpus_jsonl(corpus_path)
    queries = load_answerable(answerable_path)
    retriever = BM25Retriever(corpus)

    print(f"\nTuning τ on {len(queries)} answerable queries")
    print(f"Corpus: {len(corpus)} documents\n")

    # Tune
    tau_grid = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]

    result = tune_tau(retriever, queries, tau_grid)

    # Display results
    print("=== τ Grid Results ===\n")
    print(f"{'τ':<8} {'Answered':<12} {'Recall':<10}")
    print("-" * 30)
    for r in result['tau_results']:
        marker = " ← BEST" if r['tau'] == result['best_tau'] else ""
        print(f"{r['tau']:<8.2f} {r['n_answered']}/{r['n_total']:<10} {r['recall']:<10.3f}{marker}")

    print(f"\n=== Selected τ ===")
    print(f"τ = {result['best_tau']:.2f}")
    print(f"Recall = {result['best_recall']:.3f}")
    print(f"\nScore stats: min={result['score_stats']['min']:.2f}, "
          f"max={result['score_stats']['max']:.2f}, "
          f"mean={result['score_stats']['mean']:.2f}")

    # Save results
    with open(output_dir / "tau_results.json", 'w') as f:
        json.dump(result, f, indent=2)

    with open(output_dir / "best_tau.json", 'w') as f:
        json.dump({
            'tau': result['best_tau'],
            'recall': result['best_recall'],
            'constraint': 'recall >= 0.95',
            'strategy': 'highest tau satisfying constraint'
        }, f, indent=2)

    # Plot
    plot_score_distribution(
        np.array(result['scores']),
        result['best_tau'],
        output_dir / "score_histogram.png"
    )

    print(f"\nResults saved to {output_dir}/")


if __name__ == "__main__":
    main()
