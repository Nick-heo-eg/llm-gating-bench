from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional, Dict

from bench.retrieval import BM25Retriever, RetrievalResult


@dataclass
class RAGStopFirstResult:
    variant: str  # "stop_first"
    query: str

    # retrieval
    retrieval: RetrievalResult

    # decision
    decision: str              # "answer" | "stop"
    stop_reason: Optional[str] # "no_data" | "conflict" | "low_confidence" | None

    # gate params (for auditability)
    tau_stop: float

    # costs
    gate_latency_ms: int
    llm_called: bool
    prompt_tokens: int
    gen_tokens: int
    gen_latency_ms: int

    # aggregate
    total_latency_ms: int


class RAGStopFirst:
    """
    Stop-First RAG:
      retrieve(query)
      cheap multi-signal gate
      if STOP -> typed stop_reason
      else -> CALL LLM
    """

    def __init__(
        self,
        retriever: BM25Retriever,
        tau_stop: float,
        llm_generate_fn,
    ):
        """
        tau_stop:
          - tuned or conservative threshold for 'no_data'
        llm_generate_fn(query, retrieval_result) -> dict:
          {
            "prompt_tokens": int,
            "gen_tokens": int,
            "latency_ms": int
          }
        """
        self.retriever = retriever
        self.tau_stop = tau_stop
        self.llm_generate_fn = llm_generate_fn

    def _gate(self, r: RetrievalResult) -> tuple[str, Optional[str]]:
        """
        Returns:
          decision: "stop" | "answer"
          stop_reason: typed reason or None
        """

        # 1) No data: retrieval confidence too low
        if r.max_score < self.tau_stop:
            return "stop", "no_data"

        # 2) Conflict candidate: ambiguous top evidence
        if r.conflict_candidate:
            return "stop", "conflict"

        # 3) Low confidence (optional cheap heuristic)
        # If top score exists but gap is tiny AND absolute score not strong
        if r.top1_score < (self.tau_stop * 1.2) and r.score_gap_12 < 0.01:
            return "stop", "low_confidence"

        return "answer", None

    def run(self, query: str) -> RAGStopFirstResult:
        t_start = time.perf_counter()

        # --- retrieval ---
        retrieval = self.retriever.retrieve(query)

        # --- gate ---
        t_gate_start = time.perf_counter()
        decision, stop_reason = self._gate(retrieval)
        gate_latency_ms = int((time.perf_counter() - t_gate_start) * 1000)

        # --- generation ---
        if decision == "stop":
            llm_called = False
            prompt_tokens = 0
            gen_tokens = 0
            gen_latency_ms = 0
        else:
            llm_called = True
            llm_out = self.llm_generate_fn(query, retrieval)
            prompt_tokens = llm_out.get("prompt_tokens", 0)
            gen_tokens = llm_out.get("gen_tokens", 0)
            gen_latency_ms = llm_out.get("latency_ms", 0)

        total_latency_ms = int((time.perf_counter() - t_start) * 1000)

        return RAGStopFirstResult(
            variant="stop_first",
            query=query,
            retrieval=retrieval,
            decision=decision,
            stop_reason=stop_reason,
            tau_stop=self.tau_stop,
            gate_latency_ms=gate_latency_ms,
            llm_called=llm_called,
            prompt_tokens=prompt_tokens,
            gen_tokens=gen_tokens,
            gen_latency_ms=gen_latency_ms,
            total_latency_ms=total_latency_ms,
        )
