import json
from collections import defaultdict, Counter
from pathlib import Path
from statistics import mean


RESULT_PATH = Path("results/run_results.jsonl")


def load_results(path: Path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def aggregate_by_variant(rows):
    agg = defaultdict(list)
    for r in rows:
        agg[r["variant"]].append(r)
    return agg


def summarize_variant(rows):
    total = len(rows)

    llm_calls = sum(1 for r in rows if r["llm_called"])
    prompt_tokens = sum(r.get("prompt_tokens", 0) for r in rows)
    gen_tokens = sum(r.get("gen_tokens", 0) for r in rows)

    avg_latency = mean(r["total_latency_ms"] for r in rows)
    avg_gate_latency = mean(r.get("gate_latency_ms", 0) for r in rows)

    stop_reasons = Counter(
        r["stop_reason"] for r in rows if r.get("stop_reason") is not None
    )

    return {
        "total_queries": total,
        "llm_calls": llm_calls,
        "llm_call_rate": llm_calls / total,
        "prompt_tokens": prompt_tokens,
        "gen_tokens": gen_tokens,
        "avg_total_latency_ms": round(avg_latency, 2),
        "avg_gate_latency_ms": round(avg_gate_latency, 2),
        "stop_reason_breakdown": dict(stop_reasons),
    }


def compare_variants(summary):
    naive = summary["baseline_naive"]
    stop = summary["stop_first"]

    return {
        "llm_call_reduction": round(
            1 - (stop["llm_calls"] / naive["llm_calls"]), 3
        ) if naive["llm_calls"] > 0 else 0.0,
        "prompt_token_reduction": round(
            1 - (stop["prompt_tokens"] / naive["prompt_tokens"]), 3
        ) if naive["prompt_tokens"] > 0 else 0.0,
        "gen_token_reduction": round(
            1 - (stop["gen_tokens"] / naive["gen_tokens"]), 3
        ) if naive["gen_tokens"] > 0 else 0.0,
        "latency_speedup": round(
            naive["avg_total_latency_ms"] / stop["avg_total_latency_ms"], 2
        ) if stop["avg_total_latency_ms"] > 0 else 0.0,
    }


def main():
    rows = load_results(RESULT_PATH)
    grouped = aggregate_by_variant(rows)

    summary = {}
    for variant, rs in grouped.items():
        summary[variant] = summarize_variant(rs)

    comparison = compare_variants(summary)

    print("\n=== Variant Summary ===")
    for v, s in summary.items():
        print(f"\n[{v}]")
        for k, val in s.items():
            print(f"  {k}: {val}")

    print("\n=== Stop-First vs Baseline Naive ===")
    for k, v in comparison.items():
        print(f"  {k}: {v}")

    # Optional: save to json
    out = {
        "summary": summary,
        "comparison": comparison,
    }
    with open("results/metrics_summary.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    print("\nSaved results/metrics_summary.json")


if __name__ == "__main__":
    main()
