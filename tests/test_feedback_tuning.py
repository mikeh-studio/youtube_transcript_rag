"""Tests for feedback-adaptive retrieval tuning in local preview."""

import importlib
import json
import os
import sys
import tempfile
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
LOCAL_PREVIEW_DIR = ROOT_DIR / "local_preview"
if str(LOCAL_PREVIEW_DIR) not in sys.path:
    sys.path.insert(0, str(LOCAL_PREVIEW_DIR))

os.environ["YT_RAG_SKIP_GLOBAL_SERVICE"] = "1"
local_api = importlib.import_module("local_api")
LocalRAGService = local_api.LocalRAGService


def _make_service(enabled=True):
    service = LocalRAGService.__new__(LocalRAGService)
    service.feedback = {}
    service.feedback_index = {}
    service.feedback_lock = threading.Lock()
    service.feedback_tuning_enabled = bool(enabled)
    service._persist_feedback = lambda: None
    return service


def test_apply_feedback_rerank_no_feedback_keeps_order():
    service = _make_service(enabled=True)
    rows = [
        {
            "video_id": "vid1",
            "chunk_index": 0,
            "start": 0.0,
            "end": 10.0,
            "score": 0.8,
            "rank": 1,
        },
        {
            "video_id": "vid1",
            "chunk_index": 1,
            "start": 10.0,
            "end": 20.0,
            "score": 0.7,
            "rank": 2,
        },
    ]

    result = service._apply_feedback_rerank(query="python tips", rows=rows)
    ranked = result["results"]

    assert result["adjusted_count"] == 0
    assert [row["chunk_index"] for row in ranked] == [0, 1]
    assert ranked[0]["base_score"] == 0.8
    assert ranked[0]["feedback_adjustment"] == 0.0
    assert ranked[0]["feedback_signal"]["applied"] is False


def test_global_feedback_prior_can_flip_ranking():
    service = _make_service(enabled=True)
    service.save_search_feedback(
        {
            "query": "python list",
            "retrieval_mode": "dense",
            "label": "relevant",
            "video_id": "vid1",
            "chunk_index": 0,
            "start": 0.0,
            "end": 10.0,
        }
    )
    service.save_search_feedback(
        {
            "query": "python list",
            "retrieval_mode": "dense",
            "label": "not_relevant",
            "video_id": "vid1",
            "chunk_index": 1,
            "start": 10.0,
            "end": 20.0,
        }
    )

    rows = [
        {
            "video_id": "vid1",
            "chunk_index": 0,
            "start": 0.0,
            "end": 10.0,
            "score": 0.50,
            "rank": 2,
        },
        {
            "video_id": "vid1",
            "chunk_index": 1,
            "start": 10.0,
            "end": 20.0,
            "score": 0.53,
            "rank": 1,
        },
    ]
    result = service._apply_feedback_rerank(query="python list methods", rows=rows)
    ranked = result["results"]

    assert ranked[0]["chunk_index"] == 0
    assert ranked[0]["feedback_adjustment"] > 0
    assert ranked[1]["feedback_adjustment"] < 0


def test_feedback_is_stored_per_query_chunk_pair():
    service = _make_service(enabled=True)

    first = service.save_search_feedback(
        {
            "query": "python list",
            "retrieval_mode": "dense",
            "label": "relevant",
            "video_id": "vid1",
            "chunk_index": 0,
            "start": 0.0,
            "end": 10.0,
        }
    )
    second = service.save_search_feedback(
        {
            "query": "soccer world cup",
            "retrieval_mode": "dense",
            "label": "not_relevant",
            "video_id": "vid1",
            "chunk_index": 0,
            "start": 0.0,
            "end": 10.0,
        }
    )

    assert len(service.feedback) == 2
    assert first["chunk_key"] == "vid1:0"
    assert second["chunk_key"] == "vid1:0"
    assert first["key"] != second["key"]
    assert first["query_hash"] != second["query_hash"]
    assert service.feedback_index["vid1:0"]["relevant_count"] == 1
    assert service.feedback_index["vid1:0"]["not_relevant_count"] == 1
    assert len(service.feedback_index["vid1:0"]["entries"]) == 2


def test_feedback_update_preserves_single_normalized_query_pair():
    service = _make_service(enabled=True)

    first = service.save_search_feedback(
        {
            "query": "  Python   LIST  ",
            "retrieval_mode": "dense",
            "label": "relevant",
            "video_id": "vid1",
            "chunk_index": 0,
            "start": 0.0,
            "end": 10.0,
        }
    )
    second = service.save_search_feedback(
        {
            "query": "python list",
            "retrieval_mode": "dense",
            "label": "not_relevant",
            "video_id": "vid1",
            "chunk_index": 0,
            "start": 0.0,
            "end": 10.0,
        }
    )

    assert len(service.feedback) == 1
    assert second["id"] == first["id"]
    assert second["key"] == first["key"]
    assert second["created_at"] == first["created_at"]
    assert second["label"] == "not_relevant"
    assert service.feedback_index["vid1:0"]["relevant_count"] == 0
    assert service.feedback_index["vid1:0"]["not_relevant_count"] == 1


def test_query_similarity_gate_blocks_unrelated_query_boost():
    service = _make_service(enabled=True)
    service.save_search_feedback(
        {
            "query": "soccer world cup",
            "retrieval_mode": "dense",
            "label": "not_relevant",
            "video_id": "vid1",
            "chunk_index": 0,
            "start": 0.0,
            "end": 10.0,
        }
    )

    result = service._apply_feedback_rerank(
        query="python list comprehension",
        rows=[
            {
                "video_id": "vid1",
                "chunk_index": 0,
                "start": 0.0,
                "end": 10.0,
                "score": 0.6,
                "rank": 1,
            }
        ],
    )
    row = result["results"][0]

    assert row["feedback_signal"]["query_matches"] == 0
    assert row["feedback_adjustment"] < 0


def test_recent_feedback_outweighs_old_feedback():
    service = _make_service(enabled=True)
    now = datetime.now(timezone.utc)
    old = (now - timedelta(days=120)).isoformat()
    recent = (now - timedelta(days=1)).isoformat()

    service.feedback = {
        "vid1:0:old": {
            "id": "f1",
            "key": "vid1:0",
            "query": "python list sort",
            "query_language": "en",
            "query_tokens": ["python", "list", "sort"],
            "label": "relevant",
            "video_id": "vid1",
            "chunk_index": 0,
            "start": 0.0,
            "end": 10.0,
            "created_at": old,
            "updated_at": old,
        },
        "vid1:0:new": {
            "id": "f2",
            "key": "vid1:0",
            "query": "python list sort",
            "query_language": "en",
            "query_tokens": ["python", "list", "sort"],
            "label": "not_relevant",
            "video_id": "vid1",
            "chunk_index": 0,
            "start": 0.0,
            "end": 10.0,
            "created_at": recent,
            "updated_at": recent,
        },
    }
    service._rebuild_feedback_index()

    result = service._apply_feedback_rerank(
        query="python list sort",
        rows=[
            {
                "video_id": "vid1",
                "chunk_index": 0,
                "start": 0.0,
                "end": 10.0,
                "score": 0.6,
                "rank": 1,
            }
        ],
    )
    row = result["results"][0]
    assert row["feedback_adjustment"] < 0
    assert row["feedback_signal"]["query_matches"] == 2


def test_normalize_feedback_backfills_query_tokens():
    service = _make_service(enabled=True)
    row = service._normalize_feedback_record(
        {
            "video_id": "vid1",
            "chunk_index": 0,
            "start": 0.0,
            "end": 10.0,
            "label": "relevant",
            "query": "python list sort",
        }
    )
    assert isinstance(row["query_tokens"], list)
    assert row["query_tokens"]
    assert row["query_language"] == "en"
    assert row["chunk_key"] == "vid1:0"
    assert row["key"].startswith("vid1:0:")
    assert row["key"] != "vid1:0"


def test_load_feedback_migrates_legacy_rows_to_query_aware_keys():
    service = _make_service(enabled=True)
    runtime_dir = Path(tempfile.mkdtemp())
    service.feedback_path = runtime_dir / "search_feedback.json"
    service.legacy_feedback_path = runtime_dir / "legacy_feedback.json"
    service.feedback_path.write_text(
        json.dumps(
            [
                {
                    "id": "fb_old",
                    "key": "vid1:0",
                    "query": "python list",
                    "retrieval_mode": "dense",
                    "label": "relevant",
                    "video_id": "vid1",
                    "chunk_index": 0,
                    "start": 0.0,
                    "end": 10.0,
                    "created_at": "2026-01-01T00:00:00+00:00",
                    "updated_at": "2026-01-01T00:00:00+00:00",
                }
            ]
        ),
        encoding="utf-8",
    )

    LocalRAGService._load_feedback(service)

    assert len(service.feedback) == 1
    record = next(iter(service.feedback.values()))
    assert record["id"] == "fb_old"
    assert record["chunk_key"] == "vid1:0"
    assert record["key"].startswith("vid1:0:")
    assert record["key"] != "vid1:0"
    assert service.feedback_index["vid1:0"]["relevant_count"] == 1


def test_load_feedback_collapses_queryless_legacy_duplicates_to_latest():
    service = _make_service(enabled=True)
    runtime_dir = Path(tempfile.mkdtemp())
    service.feedback_path = runtime_dir / "search_feedback.json"
    service.legacy_feedback_path = runtime_dir / "legacy_feedback.json"
    service.feedback_path.write_text(
        json.dumps(
            [
                {
                    "id": "fb_old",
                    "key": "vid1:0",
                    "retrieval_mode": "dense",
                    "label": "relevant",
                    "video_id": "vid1",
                    "chunk_index": 0,
                    "start": 0.0,
                    "end": 10.0,
                    "created_at": "2026-01-01T00:00:00+00:00",
                    "updated_at": "2026-01-01T00:00:00+00:00",
                },
                {
                    "id": "fb_new",
                    "key": "vid1:0",
                    "retrieval_mode": "dense",
                    "label": "not_relevant",
                    "video_id": "vid1",
                    "chunk_index": 0,
                    "start": 0.0,
                    "end": 10.0,
                    "created_at": "2026-01-02T00:00:00+00:00",
                    "updated_at": "2026-01-02T00:00:00+00:00",
                },
            ]
        ),
        encoding="utf-8",
    )

    LocalRAGService._load_feedback(service)

    assert len(service.feedback) == 1
    record = next(iter(service.feedback.values()))
    assert record["id"] == "fb_new"
    assert record["query"] == ""
    assert record["chunk_key"] == "vid1:0"
    assert record["label"] == "not_relevant"


def test_retrieve_adds_feedback_metadata():
    service = _make_service(enabled=True)
    service._dense_search = lambda query, k, language: [
        {
            "video_id": "vid1",
            "video_title": "Video 1",
            "start": 0.0,
            "end": 10.0,
            "chunk_index": 0,
            "text": "python list",
            "score": 0.9,
            "rank": 1,
        }
    ]
    service._lexical_bm25_search = lambda query, k, language: []

    response = service.retrieve("python list", k=1, retrieval_mode="dense")

    assert response["details"]["feedback_tuning"]["enabled"] is True
    assert "base_score" in response["results"][0]
    assert "feedback_adjustment" in response["results"][0]
    assert "feedback_signal" in response["results"][0]
