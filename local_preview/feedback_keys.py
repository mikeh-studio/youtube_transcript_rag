"""Shared identity helpers for local search feedback records."""

from __future__ import annotations

import hashlib
import json
from typing import Optional


def feedback_chunk_key(
    video_id: str, chunk_index: Optional[int], start: float, end: float
) -> str:
    """Build the chunk-only key used for feedback grouping."""
    if chunk_index is not None:
        return f"{video_id}:{chunk_index}"
    start_ms = int(max(0.0, float(start)) * 1000)
    end_ms = int(max(0.0, float(end)) * 1000)
    return f"{video_id}:{start_ms}:{end_ms}"


def normalize_feedback_query(query: str) -> str:
    """Normalize query text before hashing feedback identity."""
    return " ".join(str(query or "").strip().lower().split())


def feedback_query_hash(query: str, retrieval_mode: str) -> str:
    """Build the query+retrieval-mode hash used in feedback identity."""
    payload = json.dumps(
        {
            "query": normalize_feedback_query(query),
            "retrieval_mode": str(retrieval_mode or "hybrid").strip().lower(),
        },
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def feedback_record_key(chunk_key: str, query_hash: str) -> str:
    """Build the query-aware feedback record key."""
    return f"{chunk_key}:{query_hash}"
