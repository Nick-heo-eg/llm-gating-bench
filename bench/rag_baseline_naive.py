from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional

from bench.retrieval import BM25Retriever, RetrievalResult


@dataclass
class RAGNaiveResult:
    variant: str  # "baseline_naive"
    query: str

    # retrieval
    retrieval: RetrievalResult

    # decision
    decision: str          # always "answer"
    stop_reason: Optional[str]  # always None

    # costs
    gate_latency_ms: int  # always 0 (no gate)
    llm_called: bool      # always True
    prompt_tokens: int
    gen_tokens: int
    gen_latency_ms: int

    # aggregate
    total_latency_ms: int


class RAGBaselineNaive:
    """
    RAG-A: Naive baseline (always generate).

    Logic:
      retrieve(query)
      ALWAYS call LLM (no gate, no threshold)

    This is the "generate-anyway" baseline.
    """

    def __init__(
        self,
        retriever: BM25Retriever,
        llm_generate_fn,
    ):
        """
        llm_generate_fn(query, retrieval_result) -> dict:
          {
            "prompt_tokens": int,
            "gen_tokens": int,
            "latency_ms": int
          }
        """
        self.retriever = retriever
        self.llm_generate_fn = llm_generate_fn

    def run(self, query: str) -> RAGNaiveResult:
        t_start = time.perf_counter()

        # --- retrieval ---
        retrieval = self.retriever.retrieve(query)

        # --- no gate (always generate) ---
        decision = "answer"
        stop_reason = None
        llm_called = True

        llm_out = self.llm_generate_fn(query, retrieval)
        prompt_tokens = llm_out.get("prompt_tokens", 0)
        gen_tokens = llm_out.get("gen_tokens", 0)
        gen_latency_ms = llm_out.get("latency_ms", 0)

        gate_latency_ms = 0  # no gate
        total_latency_ms = int((time.perf_counter() - t_start) * 1000)

        return RAGNaiveResult(
            variant="baseline_naive",
            query=query,
            retrieval=retrieval,
            decision=decision,
            stop_reason=stop_reason,
            gate_latency_ms=gate_latency_ms,
            llm_called=llm_called,
            prompt_tokens=prompt_tokens,
            gen_tokens=gen_tokens,
            gen_latency_ms=gen_latency_ms,
            total_latency_ms=total_latency_ms,
        )
