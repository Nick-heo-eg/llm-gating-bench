from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional, Dict, Any

from bench.retrieval import BM25Retriever, RetrievalResult


@dataclass
class RAGThresholdResult:
    variant: str  # "baseline_score_threshold"
    query: str

    # retrieval
    retrieval: RetrievalResult

    # decision
    tau: float
    decision: str          # "answer" | "stop"
    stop_reason: Optional[str]

    # costs
    gate_latency_ms: int
    llm_called: bool
    prompt_tokens: int
    gen_tokens: int
    gen_latency_ms: int

    # aggregate
    total_latency_ms: int


class RAGBaselineThreshold:
    """
    RAG-B: Retrieval score threshold baseline.

    Logic:
      retrieve(query)
      if max_score < tau:
          STOP (no LLM call)
      else:
          CALL LLM
    """

    def __init__(
        self,
        retriever: BM25Retriever,
        tau: float,
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
        self.tau = tau
        self.llm_generate_fn = llm_generate_fn

    def run(self, query: str) -> RAGThresholdResult:
        t_start = time.perf_counter()

        # --- retrieval ---
        retrieval = self.retriever.retrieve(query)

        # --- gate (threshold) ---
        t_gate_start = time.perf_counter()

        if retrieval.max_score < self.tau:
            decision = "stop"
            stop_reason = "score_below_threshold"
            llm_called = False
            prompt_tokens = 0
            gen_tokens = 0
            gen_latency_ms = 0
        else:
            decision = "answer"
            stop_reason = None
            llm_called = True

            llm_out = self.llm_generate_fn(query, retrieval)
            prompt_tokens = llm_out.get("prompt_tokens", 0)
            gen_tokens = llm_out.get("gen_tokens", 0)
            gen_latency_ms = llm_out.get("latency_ms", 0)

        gate_latency_ms = int((time.perf_counter() - t_gate_start) * 1000)
        total_latency_ms = int((time.perf_counter() - t_start) * 1000)

        return RAGThresholdResult(
            variant="baseline_score_threshold",
            query=query,
            retrieval=retrieval,
            tau=self.tau,
            decision=decision,
            stop_reason=stop_reason,
            gate_latency_ms=gate_latency_ms,
            llm_called=llm_called,
            prompt_tokens=prompt_tokens,
            gen_tokens=gen_tokens,
            gen_latency_ms=gen_latency_ms,
            total_latency_ms=total_latency_ms,
        )
