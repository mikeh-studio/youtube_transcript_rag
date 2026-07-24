"""Dataset loading and validation for video-routing evaluations.

The adapter-facing request is deliberately smaller than a query case. Gold
labels, categories, and distractor annotations must never be passed to the
router under test.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Mapping


SCHEMA_VERSION = "video-routing-eval-v1"
LABEL_METHOD = "independent_fixture_authoring"
FORBIDDEN_QUERY_KEYS = {
    "ablation",
    "prediction",
    "predictions",
    "ranked_chunks",
    "ranked_video_ids",
    "router_score",
    "router_scores",
    "routing_results",
}


class DatasetValidationError(ValueError):
    """Raised when a routing evaluation dataset violates its public schema."""


def _require_text(payload: Mapping[str, Any], field: str, *, location: str) -> str:
    value = payload.get(field)
    if not isinstance(value, str) or not value.strip():
        raise DatasetValidationError(f"{location}.{field} must be non-empty text")
    return value.strip()


def _require_string_list(
    payload: Mapping[str, Any],
    field: str,
    *,
    location: str,
    allow_empty: bool = True,
) -> list[str]:
    value = payload.get(field)
    if not isinstance(value, list) or any(
        not isinstance(item, str) or not item.strip() for item in value
    ):
        raise DatasetValidationError(f"{location}.{field} must be a list of strings")
    normalized = [item.strip() for item in value]
    if not allow_empty and not normalized:
        raise DatasetValidationError(f"{location}.{field} must not be empty")
    if len(normalized) != len(set(normalized)):
        raise DatasetValidationError(f"{location}.{field} must not contain duplicates")
    return normalized


def _validate_provenance(dataset: Mapping[str, Any]) -> None:
    provenance = dataset.get("label_provenance")
    if not isinstance(provenance, dict):
        raise DatasetValidationError("label_provenance must be an object")
    if provenance.get("method") != LABEL_METHOD:
        raise DatasetValidationError(
            f"label_provenance.method must be {LABEL_METHOD!r}"
        )
    if provenance.get("system_under_test_used") is not False:
        raise DatasetValidationError(
            "label_provenance.system_under_test_used must be false"
        )
    evidence_basis = provenance.get("evidence_basis")
    if not isinstance(evidence_basis, list) or not evidence_basis:
        raise DatasetValidationError(
            "label_provenance.evidence_basis must be a non-empty list"
        )


def _validate_source(
    source: Any,
    *,
    location: str,
    expected_video_id: str,
) -> None:
    if source is None:
        return
    if not isinstance(source, dict):
        raise DatasetValidationError(f"{location}.source must be an object or null")
    if source.get("platform") != "youtube":
        raise DatasetValidationError(
            f"{location}.source.platform must be 'youtube'"
        )
    source_video_id = _require_text(
        source, "video_id", location=f"{location}.source"
    )
    if source_video_id != expected_video_id:
        raise DatasetValidationError(
            f"{location}.source.video_id must match video_id"
        )
    _require_text(source, "url", location=f"{location}.source")
    channel = source.get("channel")
    if channel is None:
        return
    if not isinstance(channel, dict):
        raise DatasetValidationError(
            f"{location}.source.channel must be an object or null"
        )
    _require_text(channel, "id", location=f"{location}.source.channel")
    _require_text(channel, "name", location=f"{location}.source.channel")
    _require_text(channel, "url", location=f"{location}.source.channel")
    aliases = channel.get("aliases", [])
    if not isinstance(aliases, list) or any(
        not isinstance(alias, str) or not alias.strip() for alias in aliases
    ):
        raise DatasetValidationError(
            f"{location}.source.channel.aliases must be a list of strings"
        )


def _validate_video(video: Any, *, index: int) -> tuple[str, set[int]]:
    location = f"videos[{index}]"
    if not isinstance(video, dict):
        raise DatasetValidationError(f"{location} must be an object")
    video_id = _require_text(video, "video_id", location=location)
    _require_text(video, "title", location=location)
    _require_text(video, "transcript_excerpt", location=location)
    _validate_source(
        video.get("source"),
        location=location,
        expected_video_id=video_id,
    )

    chunks = video.get("chunks")
    if not isinstance(chunks, list) or not chunks:
        raise DatasetValidationError(f"{location}.chunks must be a non-empty list")
    seen_indices = set()
    for chunk_position, chunk in enumerate(chunks):
        chunk_location = f"{location}.chunks[{chunk_position}]"
        if not isinstance(chunk, dict):
            raise DatasetValidationError(f"{chunk_location} must be an object")
        chunk_index = chunk.get("chunk_index")
        if not isinstance(chunk_index, int) or chunk_index < 0:
            raise DatasetValidationError(
                f"{chunk_location}.chunk_index must be a non-negative integer"
            )
        if chunk_index in seen_indices:
            raise DatasetValidationError(
                f"{location}.chunks has duplicate chunk_index {chunk_index}"
            )
        seen_indices.add(chunk_index)
        _require_text(chunk, "text", location=chunk_location)
    return video_id, seen_indices


def _validate_query(
    query: Any,
    *,
    index: int,
    video_chunks: Mapping[str, set[int]],
) -> str:
    location = f"queries[{index}]"
    if not isinstance(query, dict):
        raise DatasetValidationError(f"{location} must be an object")
    forbidden = sorted(FORBIDDEN_QUERY_KEYS.intersection(query))
    if forbidden:
        raise DatasetValidationError(
            f"{location} contains router output fields: {', '.join(forbidden)}"
        )

    query_id = _require_text(query, "id", location=location)
    _require_text(query, "query", location=location)
    _require_text(query, "language", location=location)
    _require_string_list(query, "categories", location=location, allow_empty=False)
    relevant = _require_string_list(
        query, "relevant_video_ids", location=location, allow_empty=False
    )
    same_channel = _require_string_list(
        query, "same_channel_distractor_video_ids", location=location
    )
    cross_channel = _require_string_list(
        query, "cross_channel_distractor_video_ids", location=location
    )

    referenced = set(relevant + same_channel + cross_channel)
    unknown = sorted(referenced.difference(video_chunks))
    if unknown:
        raise DatasetValidationError(
            f"{location} references unknown video IDs: {', '.join(unknown)}"
        )
    overlap = set(relevant).intersection(same_channel + cross_channel)
    if overlap:
        raise DatasetValidationError(
            f"{location} labels relevant videos as distractors: "
            f"{', '.join(sorted(overlap))}"
        )
    duplicate_distractors = set(same_channel).intersection(cross_channel)
    if duplicate_distractors:
        raise DatasetValidationError(
            f"{location} duplicates distractors across channel groups: "
            f"{', '.join(sorted(duplicate_distractors))}"
        )

    gold_chunks = query.get("gold_chunks", [])
    if not isinstance(gold_chunks, list):
        raise DatasetValidationError(f"{location}.gold_chunks must be a list")
    for chunk_position, chunk in enumerate(gold_chunks):
        chunk_location = f"{location}.gold_chunks[{chunk_position}]"
        if not isinstance(chunk, dict):
            raise DatasetValidationError(f"{chunk_location} must be an object")
        chunk_video_id = _require_text(
            chunk, "video_id", location=chunk_location
        )
        chunk_index = chunk.get("chunk_index")
        if chunk_video_id not in relevant:
            raise DatasetValidationError(
                f"{chunk_location}.video_id must also be relevant"
            )
        if not isinstance(chunk_index, int) or chunk_index < 0:
            raise DatasetValidationError(
                f"{chunk_location}.chunk_index must be a non-negative integer"
            )
        if chunk_index not in video_chunks[chunk_video_id]:
            raise DatasetValidationError(
                f"{chunk_location} references an unknown chunk"
            )
    return query_id


def validate_dataset(dataset: Mapping[str, Any]) -> None:
    """Validate schema, label independence, and cross-reference integrity."""
    if not isinstance(dataset, Mapping):
        raise DatasetValidationError("dataset must be an object")
    if dataset.get("schema_version") != SCHEMA_VERSION:
        raise DatasetValidationError(
            f"schema_version must be {SCHEMA_VERSION!r}"
        )
    _require_text(dataset, "dataset_id", location="dataset")
    _validate_provenance(dataset)

    videos = dataset.get("videos")
    queries = dataset.get("queries")
    if not isinstance(videos, list) or not videos:
        raise DatasetValidationError("videos must be a non-empty list")
    if not isinstance(queries, list) or not queries:
        raise DatasetValidationError("queries must be a non-empty list")

    video_entries = [
        _validate_video(video, index=index)
        for index, video in enumerate(videos)
    ]
    video_ids = [video_id for video_id, _chunk_indices in video_entries]
    if len(video_ids) != len(set(video_ids)):
        raise DatasetValidationError("videos must have unique video_id values")
    video_chunks = dict(video_entries)

    query_ids = [
        _validate_query(query, index=index, video_chunks=video_chunks)
        for index, query in enumerate(queries)
    ]
    if len(query_ids) != len(set(query_ids)):
        raise DatasetValidationError("queries must have unique id values")


def load_dataset(path: Path | str) -> Dict[str, Any]:
    """Load and validate a JSON video-routing dataset."""
    dataset_path = Path(path)
    with dataset_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    validate_dataset(payload)
    return payload


def build_adapter_request(query_case: Mapping[str, Any], *, top_k: int = 5) -> dict:
    """Return the label-free request that may be given to a router adapter."""
    request = {
        "query_id": _require_text(query_case, "id", location="query"),
        "query": _require_text(query_case, "query", location="query"),
        "language": _require_text(query_case, "language", location="query"),
        "top_k": max(1, int(top_k)),
    }
    return request
