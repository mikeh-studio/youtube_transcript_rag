"""Tests for the local review-agent workflow helpers."""

import importlib
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
LOCAL_PREVIEW_DIR = ROOT_DIR / "local_preview"
if str(LOCAL_PREVIEW_DIR) not in sys.path:
    sys.path.insert(0, str(LOCAL_PREVIEW_DIR))

workflow = importlib.import_module("review_agent_workflow")


def test_build_review_batch_shapes_items_and_backfills_current_label():
    def fake_search_runner(spec):
        assert spec["query"] == "python lists"
        assert spec["retrieval_mode"] == "dense"
        return {
            "ok": True,
            "retrieval_mode": "dense",
            "results": [
                {
                    "video_id": "vid1",
                    "video_title": "Video 1",
                    "language": "en",
                    "chunk_index": 2,
                    "start": 12.0,
                    "end": 25.0,
                    "url": "https://www.youtube.com/watch?v=vid1&t=12s",
                    "rank": 1,
                    "score": 0.88,
                    "dense_score": 0.88,
                    "text": "Python lists can be sorted in place with sort().",
                }
            ],
        }

    batch = workflow.build_review_batch(
        [{"query": "python lists", "retrieval_mode": "dense", "k": 3}],
        search_runner=fake_search_runner,
        existing_feedback=[
            {
                "query": "python lists",
                "retrieval_mode": "dense",
                "label": "relevant",
                "video_id": "vid1",
                "chunk_index": 2,
                "start": 12.0,
                "end": 25.0,
            }
        ],
        shard_size=10,
        overlap_ratio=0.0,
        batch_id="batch_demo",
        created_at="2026-04-01T00:00:00+00:00",
        server_url="http://127.0.0.1:8000",
    )

    assert batch["batch_id"] == "batch_demo"
    assert batch["item_count"] == 1
    assert batch["server_url"] == "http://127.0.0.1:8000"
    assert batch["shards"][0]["primary_item_ids"] == [
        batch["items"][0]["review_item_id"]
    ]
    assert batch["items"][0]["current_label"] == "relevant"
    assert batch["items"][0]["feedback_key"] == "vid1:2"
    assert batch["items"][0]["timestamp_label"] == "0:12"
    assert batch["items"][0]["text"].startswith("Python lists")


def test_build_review_batch_ignores_feedback_from_different_query():
    def fake_search_runner(spec):
        return {
            "ok": True,
            "retrieval_mode": "dense",
            "results": [
                {
                    "video_id": "vid1",
                    "video_title": "Video 1",
                    "language": "en",
                    "chunk_index": 2,
                    "start": 12.0,
                    "end": 25.0,
                    "rank": 1,
                    "score": 0.88,
                    "text": "Python lists can be sorted in place with sort().",
                }
            ],
        }

    batch = workflow.build_review_batch(
        [{"query": "python lists", "retrieval_mode": "dense", "k": 3}],
        search_runner=fake_search_runner,
        existing_feedback=[
            {
                "query": "soccer world cup",
                "retrieval_mode": "dense",
                "label": "not_relevant",
                "video_id": "vid1",
                "chunk_index": 2,
                "start": 12.0,
                "end": 25.0,
            }
        ],
        shard_size=10,
        overlap_ratio=0.0,
    )

    assert batch["items"][0]["current_label"] is None


def test_assign_review_shards_adds_deterministic_overlap():
    items = [
        {
            "review_item_id": f"item-{index:02d}",
            "query_index": 0,
            "query": "q",
            "rank": index + 1,
        }
        for index in range(10)
    ]

    shards = workflow.assign_review_shards(
        items,
        shard_size=4,
        overlap_ratio=0.2,
        random_seed=7,
    )

    assert [shard["shard_id"] for shard in shards] == [
        "shard-001",
        "shard-002",
        "shard-003",
    ]
    overlap_ids = sorted(
        {
            review_item_id
            for shard in shards
            for review_item_id in shard["overlap_item_ids"]
        }
    )
    assert len(overlap_ids) == 2
    duplicated_counts = {
        review_item_id: sum(
            1 for shard in shards if review_item_id in shard["all_item_ids"]
        )
        for review_item_id in overlap_ids
    }
    assert duplicated_counts == {review_item_id: 2 for review_item_id in overlap_ids}


def test_build_adjudication_cases_splits_consensus_single_and_disagreement():
    recommendations = [
        {
            "review_item_id": "item-1",
            "query": "python lists",
            "retrieval_mode": "dense",
            "video_id": "vid1",
            "chunk_index": 1,
            "start": 0.0,
            "end": 10.0,
            "recommended_label": "relevant",
            "confidence": "high",
            "reviewer_id": "agent_a",
        },
        {
            "review_item_id": "item-1",
            "query": "python lists",
            "retrieval_mode": "dense",
            "video_id": "vid1",
            "chunk_index": 1,
            "start": 0.0,
            "end": 10.0,
            "recommended_label": "relevant",
            "confidence": "medium",
            "reviewer_id": "agent_b",
        },
        {
            "review_item_id": "item-2",
            "query": "python dicts",
            "retrieval_mode": "hybrid",
            "video_id": "vid2",
            "chunk_index": 2,
            "start": 10.0,
            "end": 20.0,
            "recommended_label": "not_relevant",
            "confidence": "medium",
            "reviewer_id": "agent_a",
        },
        {
            "review_item_id": "item-3",
            "query": "python tuples",
            "retrieval_mode": "hybrid",
            "video_id": "vid3",
            "chunk_index": 3,
            "start": 20.0,
            "end": 30.0,
            "recommended_label": "relevant",
            "confidence": "medium",
            "reviewer_id": "agent_a",
        },
        {
            "review_item_id": "item-3",
            "query": "python tuples",
            "retrieval_mode": "hybrid",
            "video_id": "vid3",
            "chunk_index": 3,
            "start": 20.0,
            "end": 30.0,
            "recommended_label": "unclear",
            "confidence": "low",
            "reviewer_id": "agent_b",
        },
    ]

    result = workflow.build_adjudication_cases(recommendations)

    assert result["item_count"] == 3
    assert result["auto_approved"][0]["review_item_id"] == "item-1"
    assert result["auto_approved"][0]["approved"] is True
    assert result["auto_approved"][0]["final_label"] == "relevant"
    assert result["single_reviewer"][0]["review_item_id"] == "item-2"
    assert result["single_reviewer"][0]["approved"] is False
    assert result["needs_adjudication"][0]["review_item_id"] == "item-3"
    assert len(result["needs_adjudication"][0]["recommendations"]) == 2


def test_build_feedback_payload_requires_approved_recommendation():
    approved_payload = workflow.build_feedback_payload(
        {
            "review_item_id": "item-1",
            "query": "python list sort",
            "retrieval_mode": "dense",
            "video_id": "vid1",
            "video_title": "Video 1",
            "language": "en",
            "chunk_index": 5,
            "start": 42.0,
            "end": 58.0,
            "url": "https://www.youtube.com/watch?v=vid1&t=42s",
            "recommended_label": "relevant",
            "approved": True,
            "confidence": "high",
        }
    )
    skipped_payload = workflow.build_feedback_payload(
        {
            "review_item_id": "item-2",
            "query": "python list sort",
            "retrieval_mode": "dense",
            "video_id": "vid1",
            "chunk_index": 6,
            "start": 60.0,
            "end": 75.0,
            "recommended_label": "relevant",
            "approved": False,
            "confidence": "medium",
        }
    )

    assert approved_payload["label"] == "relevant"
    assert approved_payload["model"] == "agent_review"
    assert skipped_payload is None


def test_apply_review_recommendations_posts_only_approved_labels():
    posted = []

    def fake_submit(payload):
        posted.append(payload)
        return {"ok": True, "feedback": payload}

    result = workflow.apply_review_recommendations(
        [
            {
                "review_item_id": "item-1",
                "query": "python lists",
                "retrieval_mode": "dense",
                "video_id": "vid1",
                "video_title": "Video 1",
                "language": "en",
                "chunk_index": 1,
                "start": 0.0,
                "end": 10.0,
                "url": "https://www.youtube.com/watch?v=vid1&t=0s",
                "recommended_label": "relevant",
                "approved": True,
                "confidence": "high",
            },
            {
                "review_item_id": "item-2",
                "query": "python dicts",
                "retrieval_mode": "hybrid",
                "video_id": "vid2",
                "chunk_index": 2,
                "start": 10.0,
                "end": 20.0,
                "recommended_label": "unclear",
                "approved": False,
                "confidence": "low",
            },
            {
                "review_item_id": "item-3",
                "query": "python tuples",
                "retrieval_mode": "hybrid",
                "video_id": "vid3",
                "chunk_index": 3,
                "start": 20.0,
                "end": 30.0,
                "recommended_label": "not_relevant",
                "approved": False,
                "confidence": "medium",
            },
        ],
        submit_feedback=fake_submit,
    )

    assert result["applied_count"] == 1
    assert result["skipped_unclear"] == 1
    assert result["skipped_unapproved"] == 1
    assert posted[0]["label"] == "relevant"
    assert posted[0]["video_id"] == "vid1"


def test_build_reviewer_prompt_includes_copy_fields_instruction():
    batch = {
        "batch_id": "batch_demo",
        "items": [
            {
                "review_item_id": "item-1",
                "query": "python list sort",
                "retrieval_mode": "dense",
                "video_id": "vid1",
                "chunk_index": 0,
                "start": 0.0,
                "end": 10.0,
                "text": "Use list.sort() to sort in place.",
            }
        ],
        "shards": [
            {
                "shard_id": "shard-001",
                "primary_item_ids": ["item-1"],
                "overlap_item_ids": [],
                "all_item_ids": ["item-1"],
            }
        ],
    }

    prompt = workflow.build_reviewer_prompt(batch, "shard-001")

    assert "relevant" in prompt
    assert "not_relevant" in prompt
    assert "unclear" in prompt
    assert "copy all input fields" in prompt.lower()
    assert '"review_item_id": "item-1"' in prompt
