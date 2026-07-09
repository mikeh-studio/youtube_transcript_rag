"""
Cross-encoder reranking stage for transcript retrieval.

Rescores fused candidate rows with a multilingual cross-encoder before
feedback tuning and diversity selection. Model loading is lazy and
failure-tolerant: when the model cannot be loaded (for example on offline
machines), reranking is skipped and rows pass through unchanged so the
retrieval pipeline keeps working.
"""

from __future__ import annotations

import math
import os
import threading
from typing import Callable, List, Optional, Sequence, Tuple

DEFAULT_RERANKER_MODEL = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
RERANKER_MODEL_ENV = "YT_RAG_RERANKER_MODEL"
DEFAULT_RERANK_CANDIDATES = 50


def _sigmoid(value: float) -> float:
    if value >= 0:
        return 1.0 / (1.0 + math.exp(-value))
    exp_value = math.exp(value)
    return exp_value / (1.0 + exp_value)


def _normalize_scores(raw_scores: List[float]) -> List[float]:
    """Map raw cross-encoder outputs to a 0..1 scale.

    Some cross-encoder checkpoints emit probabilities and others emit raw
    logits; only apply the sigmoid when scores fall outside [0, 1] so both
    kinds end up on the same scale as dense cosine scores.
    """
    if all(0.0 <= score <= 1.0 for score in raw_scores):
        return raw_scores
    return [_sigmoid(score) for score in raw_scores]


class CrossEncoderReranker:
    """Lazy cross-encoder wrapper with an injectable scorer for tests."""

    def __init__(
        self,
        model_name: Optional[str] = None,
        score_fn: Optional[Callable[[Sequence[Tuple[str, str]]], List[float]]] = None,
    ):
        """Initialize the reranker.

        Args:
            model_name: Cross-encoder model id. Defaults to the
                YT_RAG_RERANKER_MODEL environment variable, then
                DEFAULT_RERANKER_MODEL.
            score_fn: Optional scorer taking (query, text) pairs and returning
                one float per pair. Used by tests and offline fixtures.
        """
        self.model_name = (
            str(model_name or "").strip()
            or str(os.environ.get(RERANKER_MODEL_ENV) or "").strip()
            or DEFAULT_RERANKER_MODEL
        )
        self._score_fn = score_fn
        self._load_error: Optional[str] = None
        self._load_lock = threading.Lock()

    @property
    def load_error(self) -> Optional[str]:
        return self._load_error

    def _ensure_score_fn(self):
        if self._score_fn is not None:
            return self._score_fn
        # Double-checked lock so concurrent first requests load the model once.
        with self._load_lock:
            if self._score_fn is not None:
                return self._score_fn
            if self._load_error is not None:
                return None
            try:
                from sentence_transformers import CrossEncoder

                model = CrossEncoder(self.model_name)
                self._score_fn = lambda pairs: [
                    float(score) for score in model.predict(list(pairs))
                ]
            except Exception as exc:
                self._load_error = f"{type(exc).__name__}: {exc}"
                return None
            return self._score_fn

    def rerank(
        self,
        query: str,
        rows: List[dict],
        top_n: int = DEFAULT_RERANK_CANDIDATES,
    ) -> dict:
        """Rescore and reorder the top candidate rows for a query.

        Only the first `top_n` rows are scored to bound latency; remaining
        rows keep their original order after the reranked head.

        Returns:
            Dict with 'rows' (reranked copies, or the original rows when
            reranking was skipped), 'applied', 'model', 'scored_count',
            and 'error'.
        """
        outcome = {
            "rows": rows,
            "applied": False,
            "model": self.model_name,
            "scored_count": 0,
            "error": None,
        }
        if not rows or not str(query or "").strip():
            return outcome

        score_fn = self._ensure_score_fn()
        if score_fn is None:
            outcome["error"] = self._load_error
            return outcome

        limit = max(1, int(top_n))
        head = rows[:limit]
        tail = rows[limit:]
        pairs = [(str(query), str(row.get("text") or "")) for row in head]
        try:
            raw_scores = [float(score) for score in score_fn(pairs)]
        except Exception as exc:
            outcome["error"] = f"{type(exc).__name__}: {exc}"
            return outcome
        if len(raw_scores) != len(head):
            outcome["error"] = "reranker returned a mismatched score count"
            return outcome

        normalized_scores = _normalize_scores(raw_scores)
        reranked: List[dict] = []
        for row, rerank_score in zip(head, normalized_scores):
            item = dict(row)
            item["pre_rerank_rank"] = int(row.get("rank") or 0) or None
            item["pre_rerank_score"] = row.get("score")
            item["rerank_score"] = float(rerank_score)
            item["score"] = float(rerank_score)
            reranked.append(item)

        reranked.sort(
            key=lambda item: (
                float(item.get("rerank_score", 0.0)),
                -int(item.get("pre_rerank_rank") or 10**9),
            ),
            reverse=True,
        )
        merged = reranked + [dict(row) for row in tail]
        for idx, row in enumerate(merged, start=1):
            row["rank"] = idx

        outcome.update({"rows": merged, "applied": True, "scored_count": len(head)})
        return outcome
