"""Local workflow helpers for agent-assisted search review labeling."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


DEFAULT_SERVER_URL = "http://127.0.0.1:8000"
DEFAULT_BATCH_DIR = Path("data/runtime/review_batches")
DEFAULT_RECOMMENDATION_DIR = Path("data/runtime/review_recommendations")
REVIEW_LABELS = {"relevant", "not_relevant"}
RECOMMENDATION_LABELS = REVIEW_LABELS | {"unclear"}
CONFIDENCE_LEVELS = {"high", "medium", "low"}
CONFIDENCE_RANK = {"low": 0, "medium": 1, "high": 2}
RETRIEVAL_MODES = {"hybrid", "dense", "lexical"}
DEFAULT_SHARD_SIZE = 50
DEFAULT_OVERLAP_RATIO = 0.15
DEFAULT_RANDOM_SEED = 7
FEEDBACK_LIST_LIMIT = 5000
DEFAULT_TOP_K = 5
MAX_TOP_K = 12
DEFAULT_FEEDBACK_MODEL = "agent_review"


def now_iso() -> str:
    """Return a UTC ISO timestamp."""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def format_timestamp_label(seconds: float) -> str:
    """Return a compact timestamp label."""
    total_seconds = max(0, int(float(seconds or 0.0)))
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    remaining = total_seconds % 60
    if hours > 0:
        return f"{hours}:{minutes:02d}:{remaining:02d}"
    return f"{minutes}:{remaining:02d}"


def _coerce_float(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _coerce_optional_float(value) -> Optional[float]:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _coerce_optional_int(value) -> Optional[int]:
    if value in (None, ""):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _normalize_text(value) -> str:
    return str(value or "").strip()


def _normalize_label(value) -> Optional[str]:
    scoped = _normalize_text(value).lower()
    return scoped if scoped in RECOMMENDATION_LABELS else None


def _normalize_confidence(value, default: str = "low") -> str:
    scoped = _normalize_text(value).lower()
    return scoped if scoped in CONFIDENCE_LEVELS else default


def _normalize_retrieval_mode(value, default: str = "hybrid") -> str:
    scoped = _normalize_text(value).lower()
    return scoped if scoped in RETRIEVAL_MODES else default


def _coerce_search_top_k(value, default: int = DEFAULT_TOP_K) -> int:
    try:
        scoped = int(value)
    except (TypeError, ValueError):
        scoped = int(default)
    return max(1, min(scoped, MAX_TOP_K))


def build_feedback_identity(
    video_id: str, chunk_index: Optional[int], start: float, end: float
) -> str:
    """Match the local preview chunk identity key format."""
    if chunk_index is not None:
        return f"{video_id}:{chunk_index}"
    start_ms = int(max(0.0, float(start)) * 1000)
    end_ms = int(max(0.0, float(end)) * 1000)
    return f"{video_id}:{start_ms}:{end_ms}"


def normalize_feedback_query(query: str) -> str:
    """Match the local preview feedback query normalization."""
    return " ".join(_normalize_text(query).lower().split())


def build_feedback_query_hash(query: str, retrieval_mode: str) -> str:
    """Match the local preview query-aware feedback hash."""
    payload = json.dumps(
        {
            "query": normalize_feedback_query(query),
            "retrieval_mode": _normalize_retrieval_mode(retrieval_mode),
        },
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def build_feedback_record_identity(
    chunk_key: str, query: str, retrieval_mode: str
) -> str:
    """Build the query+chunk feedback record identity."""
    return f"{chunk_key}:{build_feedback_query_hash(query, retrieval_mode)}"


def build_review_item(
    *,
    query: str,
    retrieval_mode: str,
    row: dict,
    current_label: Optional[str] = None,
) -> Optional[dict]:
    """Convert a search result row into a stable review item."""
    video_id = _normalize_text(row.get("video_id"))
    if not video_id:
        return None

    chunk_index = _coerce_optional_int(row.get("chunk_index"))
    start = _coerce_float(row.get("start", row.get("start_seconds", 0.0)))
    end = _coerce_float(row.get("end", row.get("end_seconds", start)))
    feedback_key = build_feedback_identity(video_id, chunk_index, start, end)
    query_hash = hashlib.sha1(_normalize_text(query).encode("utf-8")).hexdigest()[:10]
    review_item_id = f"{query_hash}:{feedback_key}"
    url = _normalize_text(
        row.get("url")
        or f"https://www.youtube.com/watch?v={video_id}&t={int(max(0.0, start))}s"
    )

    return {
        "review_item_id": review_item_id,
        "feedback_key": feedback_key,
        "query": _normalize_text(query),
        "retrieval_mode": _normalize_retrieval_mode(retrieval_mode),
        "current_label": _normalize_label(current_label),
        "video_id": video_id,
        "video_title": _normalize_text(row.get("video_title")) or video_id,
        "language": _normalize_text(row.get("language")),
        "chunk_index": chunk_index,
        "start": start,
        "end": end,
        "timestamp_label": format_timestamp_label(start),
        "url": url,
        "video_url": _normalize_text(row.get("video_url"))
        or f"https://www.youtube.com/watch?v={video_id}",
        "rank": _coerce_optional_int(row.get("rank")),
        "score": _coerce_optional_float(row.get("score")),
        "dense_score": _coerce_optional_float(row.get("dense_score")),
        "lexical_score": _coerce_optional_float(row.get("lexical_score")),
        "hybrid_score": _coerce_optional_float(row.get("hybrid_score")),
        "text": _normalize_text(row.get("text")),
    }


def _build_feedback_lookup(rows: Iterable[dict]) -> Dict[str, dict]:
    lookup: Dict[str, dict] = {}
    for row in rows or []:
        if not isinstance(row, dict):
            continue
        video_id = _normalize_text(row.get("video_id"))
        if not video_id:
            continue
        chunk_index = _coerce_optional_int(row.get("chunk_index"))
        start = _coerce_float(row.get("start", 0.0))
        end = _coerce_float(row.get("end", start))
        chunk_key = (
            _normalize_text(row.get("chunk_key"))
            or _normalize_text(row.get("feedback_key"))
            or build_feedback_identity(video_id, chunk_index, start, end)
        )
        query = _normalize_text(row.get("query"))
        retrieval_mode = _normalize_retrieval_mode(row.get("retrieval_mode"))
        key = _normalize_text(row.get("key"))
        if not key or key == chunk_key:
            key = build_feedback_record_identity(chunk_key, query, retrieval_mode)
        lookup[key] = row
    return lookup


def _feedback_label_for_row(
    feedback_lookup: Dict[str, dict], row: dict, query: str, retrieval_mode: str
) -> Optional[str]:
    video_id = _normalize_text(row.get("video_id"))
    if not video_id:
        return None
    chunk_key = build_feedback_identity(
        video_id,
        _coerce_optional_int(row.get("chunk_index")),
        _coerce_float(row.get("start", 0.0)),
        _coerce_float(row.get("end", row.get("start", 0.0))),
    )
    key = build_feedback_record_identity(chunk_key, query, retrieval_mode)
    return _normalize_label((feedback_lookup.get(key) or {}).get("label"))


def assign_review_shards(
    items: List[dict],
    *,
    shard_size: int = DEFAULT_SHARD_SIZE,
    overlap_ratio: float = DEFAULT_OVERLAP_RATIO,
    random_seed: int = DEFAULT_RANDOM_SEED,
) -> List[dict]:
    """Partition review items into deterministic shards with an overlap set."""
    safe_shard_size = max(1, int(shard_size or DEFAULT_SHARD_SIZE))
    sorted_items = list(items or [])
    if not sorted_items:
        return []

    shards: List[dict] = []
    for start_index in range(0, len(sorted_items), safe_shard_size):
        primary_items = sorted_items[start_index : start_index + safe_shard_size]
        primary_ids = [item["review_item_id"] for item in primary_items]
        shard_number = len(shards) + 1
        shards.append(
            {
                "shard_id": f"shard-{shard_number:03d}",
                "primary_item_ids": primary_ids,
                "overlap_item_ids": [],
                "all_item_ids": list(primary_ids),
            }
        )

    if len(shards) < 2 or overlap_ratio <= 0:
        for shard in shards:
            shard["item_count"] = len(shard["all_item_ids"])
        return shards

    overlap_count = min(
        len(sorted_items), max(0, int(round(len(sorted_items) * float(overlap_ratio))))
    )
    if overlap_count <= 0:
        for shard in shards:
            shard["item_count"] = len(shard["all_item_ids"])
        return shards

    rng = random.Random(int(random_seed))
    overlap_indexes = sorted(rng.sample(range(len(sorted_items)), overlap_count))
    for item_index in overlap_indexes:
        item = sorted_items[item_index]
        source_shard_index = item_index // safe_shard_size
        target_shard_index = (source_shard_index + 1) % len(shards)
        if target_shard_index == source_shard_index:
            continue
        review_item_id = item["review_item_id"]
        target_shard = shards[target_shard_index]
        if review_item_id not in target_shard["all_item_ids"]:
            target_shard["overlap_item_ids"].append(review_item_id)
            target_shard["all_item_ids"].append(review_item_id)

    for shard in shards:
        shard["all_item_ids"].sort()
        shard["overlap_item_ids"].sort()
        shard["item_count"] = len(shard["all_item_ids"])
    return shards


def build_review_batch(
    query_specs: List[dict],
    *,
    search_runner: Callable[[dict], dict],
    existing_feedback: Optional[Iterable[dict]] = None,
    shard_size: int = DEFAULT_SHARD_SIZE,
    overlap_ratio: float = DEFAULT_OVERLAP_RATIO,
    random_seed: int = DEFAULT_RANDOM_SEED,
    batch_id: Optional[str] = None,
    created_at: Optional[str] = None,
    server_url: Optional[str] = None,
) -> dict:
    """Build a review batch from live search results."""
    normalized_specs: List[dict] = []
    feedback_lookup = _build_feedback_lookup(existing_feedback or [])
    items: List[dict] = []

    for index, raw_spec in enumerate(query_specs or []):
        query = _normalize_text((raw_spec or {}).get("query"))
        if not query:
            raise ValueError("each query spec must include query")
        normalized_spec = {
            "query": query,
            "retrieval_mode": _normalize_retrieval_mode(
                (raw_spec or {}).get("retrieval_mode")
            ),
            "k": _coerce_search_top_k((raw_spec or {}).get("k", DEFAULT_TOP_K)),
            "language": _normalize_text((raw_spec or {}).get("language")) or None,
            "video_id": _normalize_text((raw_spec or {}).get("video_id")) or None,
        }
        normalized_specs.append(normalized_spec)

        response = search_runner(normalized_spec)
        for row in response.get("results", []) or []:
            response_mode = _normalize_retrieval_mode(
                response.get("retrieval_mode"), normalized_spec["retrieval_mode"]
            )
            item = build_review_item(
                query=normalized_spec["query"],
                retrieval_mode=response_mode,
                row=row,
                current_label=_feedback_label_for_row(
                    feedback_lookup,
                    row,
                    normalized_spec["query"],
                    response_mode,
                ),
            )
            if not item:
                continue
            item["query_index"] = index
            items.append(item)

    items.sort(
        key=lambda item: (
            item["query_index"],
            item["query"].lower(),
            int(item["rank"] or 0),
            item["review_item_id"],
        )
    )

    resolved_batch_id = batch_id or (
        f"review_batch_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    )
    resolved_created_at = created_at or now_iso()
    shards = assign_review_shards(
        items,
        shard_size=shard_size,
        overlap_ratio=overlap_ratio,
        random_seed=random_seed,
    )

    return {
        "batch_id": resolved_batch_id,
        "created_at": resolved_created_at,
        "server_url": _normalize_text(server_url),
        "query_specs": normalized_specs,
        "item_count": len(items),
        "overlap_item_ids": sorted(
            {
                review_item_id
                for shard in shards
                for review_item_id in shard["overlap_item_ids"]
            }
        ),
        "items": items,
        "shards": shards,
        "reviewer_output_contract": {
            "copy_input_fields": True,
            "required_added_fields": [
                "recommended_label",
                "decision",
                "confidence",
                "reason",
                "quoted_evidence",
                "reviewer_id",
            ],
            "allowed_recommended_labels": sorted(RECOMMENDATION_LABELS),
        },
    }


def get_shard_items(batch: dict, shard_id: str) -> List[dict]:
    """Return the items assigned to one shard."""
    items_by_id = {
        _normalize_text(item.get("review_item_id")): item
        for item in batch.get("items", [])
    }
    for shard in batch.get("shards", []):
        if _normalize_text(shard.get("shard_id")) != _normalize_text(shard_id):
            continue
        return [
            copy.deepcopy(items_by_id[item_id])
            for item_id in shard.get("all_item_ids", [])
            if item_id in items_by_id
        ]
    raise KeyError(f"shard not found: {shard_id}")


def build_reviewer_prompt(batch: dict, shard_id: str) -> str:
    """Render a reviewer prompt for one shard."""
    shard_items = get_shard_items(batch, shard_id)
    prompt_payload = {
        "batch_id": batch.get("batch_id"),
        "shard_id": shard_id,
        "items": shard_items,
    }
    return (
        "Review this search-result shard and label each result for retrieval relevance.\n\n"
        "Rubric:\n"
        "- relevant: the chunk materially helps answer the query.\n"
        "- not_relevant: the chunk is off-topic or not useful for answering the query.\n"
        "- unclear: the chunk is borderline, too vague, or lacks enough context.\n\n"
        "Rules:\n"
        "- Use only the provided chunk text and metadata.\n"
        "- Judge relevance against the query, not transcript quality in general.\n"
        "- Prefer unclear over an overconfident label.\n"
        "- Return JSON only.\n"
        "- For each item, copy all input fields and add: recommended_label, decision, confidence, reason, quoted_evidence, reviewer_id.\n"
        "- decision must be keep, change, or unclear.\n\n"
        f"{json.dumps(prompt_payload, ensure_ascii=False, indent=2)}\n"
    )


def build_adjudication_cases(recommendations: Iterable[dict]) -> dict:
    """Group reviewer recommendations into approval and adjudication buckets."""
    groups: Dict[str, List[dict]] = {}
    for row in recommendations or []:
        normalized = normalize_review_recommendation(row)
        if not normalized:
            continue
        groups.setdefault(normalized["review_item_id"], []).append(normalized)

    auto_approved: List[dict] = []
    single_reviewer: List[dict] = []
    needs_adjudication: List[dict] = []

    for review_item_id, rows in sorted(groups.items()):
        labels = {row["recommended_label"] for row in rows}
        confident_consensus = (
            len(rows) >= 2
            and len(labels) == 1
            and next(iter(labels)) in REVIEW_LABELS
            and all(
                CONFIDENCE_RANK[row["confidence"]] >= CONFIDENCE_RANK["medium"]
                for row in rows
            )
        )
        if confident_consensus:
            primary = copy.deepcopy(rows[0])
            primary["approved"] = True
            primary["final_label"] = primary["recommended_label"]
            primary["approval_reason"] = "consensus"
            primary["reviewers"] = sorted(
                {
                    _normalize_text(row.get("reviewer_id")) or f"reviewer_{index + 1}"
                    for index, row in enumerate(rows)
                }
            )
            auto_approved.append(primary)
            continue

        if len(rows) == 1:
            entry = copy.deepcopy(rows[0])
            entry["approved"] = False
            entry["final_label"] = None
            entry["approval_reason"] = "single_reviewer"
            single_reviewer.append(entry)
            continue

        needs_adjudication.append(
            {
                "review_item_id": review_item_id,
                "query": rows[0]["query"],
                "current_label": rows[0].get("current_label"),
                "recommendations": rows,
            }
        )

    return {
        "created_at": now_iso(),
        "recommendation_count": sum(len(rows) for rows in groups.values()),
        "item_count": len(groups),
        "auto_approved": auto_approved,
        "single_reviewer": single_reviewer,
        "needs_adjudication": needs_adjudication,
    }


def build_adjudicator_prompt(adjudication_payload: dict) -> str:
    """Render an adjudicator prompt for disagreement cases."""
    return (
        "Adjudicate the conflicting retrieval-review recommendations below.\n\n"
        "Rules:\n"
        "- Use only the provided query, chunk text, and reviewer evidence.\n"
        "- Choose final_label as relevant, not_relevant, or unclear.\n"
        "- Prefer unclear when support is weak.\n"
        "- Return JSON only.\n"
        "- For each case, output: review_item_id, final_label, confidence, approved, rationale.\n"
        "- approved must be true only for final_label relevant or not_relevant.\n\n"
        f"{json.dumps(adjudication_payload, ensure_ascii=False, indent=2)}\n"
    )


def normalize_review_recommendation(row: dict) -> Optional[dict]:
    """Normalize a reviewer or adjudicator recommendation record."""
    if not isinstance(row, dict):
        return None

    source = row.get("source_ref") if isinstance(row.get("source_ref"), dict) else {}
    recommended_label = _normalize_label(
        row.get("recommended_label") or row.get("final_label")
    )
    if not recommended_label:
        return None

    normalized = {
        "review_item_id": _normalize_text(
            row.get("review_item_id") or source.get("review_item_id")
        ),
        "feedback_key": _normalize_text(
            row.get("feedback_key") or source.get("feedback_key")
        ),
        "query": _normalize_text(row.get("query") or source.get("query")),
        "retrieval_mode": (
            _normalize_retrieval_mode(
                row.get("retrieval_mode") or source.get("retrieval_mode")
            )
        ),
        "current_label": _normalize_label(
            row.get("current_label") or source.get("current_label")
        ),
        "recommended_label": recommended_label,
        "decision": _normalize_text(row.get("decision")).lower() or None,
        "confidence": _normalize_confidence(row.get("confidence"), default="low"),
        "reason": _normalize_text(row.get("reason") or row.get("rationale")),
        "quoted_evidence": _normalize_text(
            row.get("quoted_evidence") or row.get("evidence")
        ),
        "reviewer_id": _normalize_text(row.get("reviewer_id")),
        "approved": bool(row.get("approved", False)),
        "final_label": _normalize_label(row.get("final_label")),
        "video_id": _normalize_text(row.get("video_id") or source.get("video_id")),
        "video_title": _normalize_text(
            row.get("video_title") or source.get("video_title")
        ),
        "language": _normalize_text(row.get("language") or source.get("language")),
        "chunk_index": _coerce_optional_int(
            row.get("chunk_index", source.get("chunk_index"))
        ),
        "start": _coerce_float(row.get("start", source.get("start", 0.0))),
        "end": _coerce_float(
            row.get(
                "end", source.get("end", row.get("start", source.get("start", 0.0)))
            )
        ),
        "url": _normalize_text(row.get("url") or source.get("url")),
        "video_url": _normalize_text(row.get("video_url") or source.get("video_url")),
        "rank": _coerce_optional_int(row.get("rank", source.get("rank"))),
        "score": _coerce_optional_float(row.get("score", source.get("score"))),
        "dense_score": _coerce_optional_float(
            row.get("dense_score", source.get("dense_score"))
        ),
        "lexical_score": _coerce_optional_float(
            row.get("lexical_score", source.get("lexical_score"))
        ),
        "hybrid_score": _coerce_optional_float(
            row.get("hybrid_score", source.get("hybrid_score"))
        ),
        "text": _normalize_text(row.get("text") or source.get("text")),
    }
    if (
        not normalized["review_item_id"]
        or not normalized["query"]
        or not normalized["video_id"]
    ):
        return None
    if not normalized["feedback_key"]:
        normalized["feedback_key"] = build_feedback_identity(
            normalized["video_id"],
            normalized["chunk_index"],
            normalized["start"],
            normalized["end"],
        )
    return normalized


def build_feedback_payload(recommendation: dict) -> Optional[dict]:
    """Convert an approved recommendation into the live feedback API shape."""
    row = normalize_review_recommendation(recommendation)
    if not row:
        return None

    resolved_label = row["final_label"] or (
        row["recommended_label"] if row["approved"] else None
    )
    if resolved_label not in REVIEW_LABELS:
        return None
    if not row["approved"] and not row["final_label"]:
        return None

    return {
        "query": row["query"],
        "retrieval_mode": row["retrieval_mode"] or "hybrid",
        "label": resolved_label,
        "video_id": row["video_id"],
        "chunk_index": row["chunk_index"],
        "start": row["start"],
        "end": row["end"],
        "url": row["url"],
        "video_title": row["video_title"],
        "language": row["language"],
        "score": row["score"],
        "dense_score": row["dense_score"],
        "lexical_score": row["lexical_score"],
        "hybrid_score": row["hybrid_score"],
        "rank": row["rank"],
        "model": DEFAULT_FEEDBACK_MODEL,
    }


def apply_review_recommendations(
    recommendations: Iterable[dict],
    *,
    submit_feedback: Callable[[dict], dict],
    dry_run: bool = False,
) -> dict:
    """Submit approved recommendations through the existing feedback endpoint."""
    applied_payloads: List[dict] = []
    responses: List[dict] = []
    skipped_unclear = 0
    skipped_unapproved = 0
    skipped_invalid = 0

    for row in recommendations or []:
        normalized = normalize_review_recommendation(row)
        if not normalized:
            skipped_invalid += 1
            continue
        if (
            normalized["recommended_label"] == "unclear"
            and normalized["final_label"] != "relevant"
            and normalized["final_label"] != "not_relevant"
        ):
            skipped_unclear += 1
            continue
        payload = build_feedback_payload(normalized)
        if not payload:
            skipped_unapproved += 1
            continue
        applied_payloads.append(payload)
        if not dry_run:
            responses.append(submit_feedback(payload))

    return {
        "applied_count": len(applied_payloads),
        "skipped_unclear": skipped_unclear,
        "skipped_unapproved": skipped_unapproved,
        "skipped_invalid": skipped_invalid,
        "payloads": applied_payloads,
        "responses": responses,
    }


def _request_json(
    server_url: str,
    path: str,
    *,
    method: str = "GET",
    body: Optional[dict] = None,
) -> dict:
    url = f"{server_url.rstrip('/')}{path}"
    headers = {"Accept": "application/json"}
    data = None
    if body is not None:
        data = json.dumps(body).encode("utf-8")
        headers["Content-Type"] = "application/json"
    request = Request(url, data=data, method=method.upper(), headers=headers)
    try:
        with urlopen(request) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        message = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code} for {path}: {message}") from exc
    except URLError as exc:
        raise RuntimeError(f"request failed for {path}: {exc.reason}") from exc

    if isinstance(payload, dict) and payload.get("ok") is False:
        error = payload.get("error") or {}
        raise RuntimeError(error.get("message") or f"request failed for {path}")
    return payload


def make_search_runner(server_url: str) -> Callable[[dict], dict]:
    """Create a localhost-backed search runner."""

    def _run(spec: dict) -> dict:
        body = {
            "query": spec["query"],
            "retrieval_mode": spec.get("retrieval_mode") or "hybrid",
            "k": spec.get("k", 5),
        }
        if spec.get("language"):
            body["language"] = spec["language"]
        if spec.get("video_id"):
            body["video_id"] = spec["video_id"]
        return _request_json(server_url, "/v1/search", method="POST", body=body)

    return _run


def fetch_search_feedback(
    server_url: str, limit: int = FEEDBACK_LIST_LIMIT
) -> List[dict]:
    """Fetch existing runtime feedback to backfill current labels."""
    payload = _request_json(
        server_url,
        f"/v1/feedback/search-review?limit={max(1, min(int(limit), FEEDBACK_LIST_LIMIT))}",
        method="GET",
    )
    return list(payload.get("reviews") or [])


def make_feedback_submitter(server_url: str) -> Callable[[dict], dict]:
    """Create a localhost-backed feedback submitter."""

    def _submit(payload: dict) -> dict:
        return _request_json(
            server_url, "/v1/feedback/search-review", method="POST", body=payload
        )

    return _submit


def load_json(path: Path):
    """Read JSON from disk."""
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload) -> None:
    """Write JSON to disk with stable formatting."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def _extract_recommendation_rows(payload) -> List[dict]:
    if isinstance(payload, list):
        return [row for row in payload if isinstance(row, dict)]
    if not isinstance(payload, dict):
        return []

    rows: List[dict] = []
    for key in (
        "recommendations",
        "approved",
        "auto_approved",
        "single_reviewer",
        "items",
    ):
        value = payload.get(key)
        if isinstance(value, list):
            rows.extend(row for row in value if isinstance(row, dict))
    return rows


def _cmd_build_batch(args: argparse.Namespace) -> int:
    query_specs = load_json(Path(args.query_file))
    if not isinstance(query_specs, list):
        raise ValueError("query file must contain a JSON array")
    server_url = args.server_url or DEFAULT_SERVER_URL
    existing_feedback = (
        fetch_search_feedback(server_url) if args.include_existing_feedback else []
    )
    batch = build_review_batch(
        query_specs,
        search_runner=make_search_runner(server_url),
        existing_feedback=existing_feedback,
        shard_size=args.shard_size,
        overlap_ratio=args.overlap_ratio,
        random_seed=args.random_seed,
        server_url=server_url,
    )
    output_path = (
        Path(args.out) if args.out else DEFAULT_BATCH_DIR / f"{batch['batch_id']}.json"
    )
    write_json(output_path, batch)
    print(str(output_path))
    return 0


def _cmd_render_prompt(args: argparse.Namespace) -> int:
    payload = load_json(Path(args.input))
    if args.kind == "reviewer":
        if not args.shard_id:
            raise ValueError("--shard-id is required for reviewer prompts")
        print(build_reviewer_prompt(payload, args.shard_id))
        return 0
    if args.kind == "adjudicator":
        print(build_adjudicator_prompt(payload))
        return 0
    raise ValueError(f"unsupported prompt kind: {args.kind}")


def _cmd_build_adjudication(args: argparse.Namespace) -> int:
    payload = load_json(Path(args.input))
    adjudication = build_adjudication_cases(_extract_recommendation_rows(payload))
    output_path = (
        Path(args.out)
        if args.out
        else DEFAULT_RECOMMENDATION_DIR
        / f"adjudication_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json"
    )
    write_json(output_path, adjudication)
    print(str(output_path))
    return 0


def _cmd_apply(args: argparse.Namespace) -> int:
    payload = load_json(Path(args.input))
    result = apply_review_recommendations(
        _extract_recommendation_rows(payload),
        submit_feedback=make_feedback_submitter(args.server_url or DEFAULT_SERVER_URL),
        dry_run=bool(args.dry_run),
    )
    output_path = Path(args.out) if args.out else None
    if output_path:
        write_json(output_path, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


def build_arg_parser() -> argparse.ArgumentParser:
    """Create the CLI parser."""
    parser = argparse.ArgumentParser(
        description="Agent workflow helpers for local search review labeling."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    build_batch = subparsers.add_parser(
        "build-batch", help="Build a review batch from live /v1/search queries."
    )
    build_batch.add_argument(
        "--query-file", required=True, help="JSON file of query specs."
    )
    build_batch.add_argument("--server-url", default=DEFAULT_SERVER_URL)
    build_batch.add_argument("--out", help="Output path for the batch JSON.")
    build_batch.add_argument("--shard-size", type=int, default=DEFAULT_SHARD_SIZE)
    build_batch.add_argument(
        "--overlap-ratio", type=float, default=DEFAULT_OVERLAP_RATIO
    )
    build_batch.add_argument("--random-seed", type=int, default=DEFAULT_RANDOM_SEED)
    build_batch.add_argument(
        "--include-existing-feedback",
        action="store_true",
        help="Backfill current_label from /v1/feedback/search-review.",
    )
    build_batch.set_defaults(func=_cmd_build_batch)

    render_prompt = subparsers.add_parser(
        "render-prompt", help="Render a reviewer or adjudicator prompt."
    )
    render_prompt.add_argument(
        "--kind", choices=("reviewer", "adjudicator"), required=True
    )
    render_prompt.add_argument(
        "--input", required=True, help="Batch or adjudication JSON."
    )
    render_prompt.add_argument(
        "--shard-id", help="Shard id for reviewer prompt rendering."
    )
    render_prompt.set_defaults(func=_cmd_render_prompt)

    build_adjudication = subparsers.add_parser(
        "build-adjudication",
        help="Group reviewer outputs into auto-approved and adjudication buckets.",
    )
    build_adjudication.add_argument(
        "--input", required=True, help="Recommendation JSON."
    )
    build_adjudication.add_argument("--out", help="Output path for adjudication JSON.")
    build_adjudication.set_defaults(func=_cmd_build_adjudication)

    apply_cmd = subparsers.add_parser(
        "apply",
        help="POST approved recommendations into /v1/feedback/search-review.",
    )
    apply_cmd.add_argument(
        "--input", required=True, help="Approved recommendation JSON."
    )
    apply_cmd.add_argument("--server-url", default=DEFAULT_SERVER_URL)
    apply_cmd.add_argument("--dry-run", action="store_true")
    apply_cmd.add_argument("--out", help="Optional output path for apply summary JSON.")
    apply_cmd.set_defaults(func=_cmd_apply)

    return parser


def main(argv: Optional[List[str]] = None) -> int:
    """CLI entrypoint."""
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(1)
