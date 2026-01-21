from __future__ import annotations

import json
import time
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import numpy as np

# pip install rank-bm25
from rank_bm25 import BM25Okapi


@dataclass(frozen=True)
class RetrievalResult:
    query: str
    retrieved_doc_ids: List[str]
    retrieved_scores: List[float]
    max_score: float
    retrieval_latency_ms: int

    # For later diagnostics / conflict handling
    top1_doc_id: Optional[str]
    top2_doc_id: Optional[str]
    top1_score: float
    top2_score: float
    score_gap_12: float
    conflict_candidate: bool  # heuristic only


def load_corpus_jsonl(path: str | Path) -> List[Dict[str, str]]:
    """
    Expected JSONL:
      {"doc_id": "doc_0001", "text": "..."}
    """
    corpus: List[Dict[str, str]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                corpus.append(json.loads(line))
    # Minimal validation
    for d in corpus[:5]:
        if "doc_id" not in d or "text" not in d:
            raise ValueError("Corpus items must have 'doc_id' and 'text'")
    return corpus


class BM25Retriever:
    """
    Bench-friendly BM25 retriever.
    - deterministic
    - simple tokenizer
    - returns top-k doc ids and scores
    """

    def __init__(self, corpus: List[Dict[str, str]]):
        self.corpus = corpus
        self.doc_ids = [d["doc_id"] for d in corpus]
        self.texts = [d["text"] for d in corpus]

        self.tokenized = [self._tokenize(t) for t in self.texts]
        self.bm25 = BM25Okapi(self.tokenized)

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        # Simple + stable tokenizer (avoid overfitting debate)
        # - lowercase
        # - keep alnum tokens
        return re.findall(r"[a-z0-9]+", text.lower())

    def retrieve(self, query: str, top_k: int = 5) -> RetrievalResult:
        q_tokens = self._tokenize(query)

        t0 = time.perf_counter()
        scores = self.bm25.get_scores(q_tokens)
        latency_ms = int((time.perf_counter() - t0) * 1000)

        scores = np.asarray(scores, dtype=float)
        if scores.size == 0:
            return RetrievalResult(
                query=query,
                retrieved_doc_ids=[],
                retrieved_scores=[],
                max_score=0.0,
                retrieval_latency_ms=latency_ms,
                top1_doc_id=None,
                top2_doc_id=None,
                top1_score=0.0,
                top2_score=0.0,
                score_gap_12=0.0,
                conflict_candidate=False,
            )

        # Top-k
        k = min(top_k, scores.size)
        top_idx = scores.argsort()[::-1][:k]

        retrieved_doc_ids = [self.doc_ids[i] for i in top_idx]
        retrieved_scores = [float(scores[i]) for i in top_idx]
        max_score = float(retrieved_scores[0]) if retrieved_scores else 0.0

        # Heuristic conflict candidate:
        # If top1 and top2 are close, we might be in "ambiguous evidence" territory.
        top1_score = float(retrieved_scores[0]) if len(retrieved_scores) >= 1 else 0.0
        top2_score = float(retrieved_scores[1]) if len(retrieved_scores) >= 2 else 0.0
        score_gap_12 = float(top1_score - top2_score)

        # Default heuristic threshold; can be tuned later
        conflict_candidate = (len(retrieved_scores) >= 2) and (score_gap_12 <= 0.05 * max(1.0, top1_score))

        return RetrievalResult(
            query=query,
            retrieved_doc_ids=retrieved_doc_ids,
            retrieved_scores=retrieved_scores,
            max_score=max_score,
            retrieval_latency_ms=latency_ms,
            top1_doc_id=retrieved_doc_ids[0] if len(retrieved_doc_ids) >= 1 else None,
            top2_doc_id=retrieved_doc_ids[1] if len(retrieved_doc_ids) >= 2 else None,
            top1_score=top1_score,
            top2_score=top2_score,
            score_gap_12=score_gap_12,
            conflict_candidate=conflict_candidate,
        )


def make_retrieval_fn_max_score(retriever: BM25Retriever, top_k: int = 5):
    """
    Adapter for tune_threshold.py: returns only max_score.
    """
    def retrieval_fn(question: str) -> float:
        res = retriever.retrieve(question, top_k=top_k)
        return res.max_score
    return retrieval_fn
