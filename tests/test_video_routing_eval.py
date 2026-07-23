"""Focused tests for the video-first routing evaluation package."""

from __future__ import annotations

import copy
from pathlib import Path

import pytest

from evals.video_routing.dataset import (
    DatasetValidationError,
    build_adapter_request,
    load_dataset,
    validate_dataset,
)
from evals.video_routing.runner import run_adapter
from evals.video_routing.scoring import evaluate_predictions


ROOT_DIR = Path(__file__).resolve().parents[1]
FIXTURE_PATH = ROOT_DIR / "evals" / "datasets" / "video_routing_v1.json"


def _video(video_id: str, channel_id: str | None = "channel-1") -> dict:
    channel = None
    if channel_id is not None:
        channel = {
            "id": channel_id,
            "name": f"Channel {channel_id}",
            "url": f"https://www.youtube.com/channel/{channel_id}",
            "aliases": [],
        }
    return {
        "video_id": video_id,
        "title": f"Title {video_id}",
        "transcript_excerpt": f"Independent transcript evidence for {video_id}.",
        "source": {
            "platform": "youtube",
            "video_id": video_id,
            "url": f"https://www.youtube.com/watch?v={video_id}",
            "channel": channel,
        },
        "chunks": [{"chunk_index": 0, "text": f"Evidence chunk for {video_id}."}],
    }


def _dataset() -> dict:
    return {
        "schema_version": "video-routing-eval-v1",
        "dataset_id": "test-video-routing",
        "label_provenance": {
            "method": "independent_fixture_authoring",
            "system_under_test_used": False,
            "evidence_basis": ["transcript_excerpt"],
        },
        "videos": [
            _video("v1", "channel-a"),
            _video("v2", "channel-a"),
            _video("v3", "channel-b"),
            _video("v4", None),
        ],
        "queries": [
            {
                "id": "q1",
                "query": "Find the first independently labeled video.",
                "language": "en",
                "categories": ["channel_targeted"],
                "relevant_video_ids": ["v1"],
                "same_channel_distractor_video_ids": ["v2"],
                "cross_channel_distractor_video_ids": ["v3"],
                "gold_chunks": [{"video_id": "v1", "chunk_index": 0}],
            },
            {
                "id": "q2",
                "query": "Compare the third and fourth videos.",
                "language": "en",
                "categories": ["ordinary", "missing_metadata"],
                "relevant_video_ids": ["v3", "v4"],
                "same_channel_distractor_video_ids": [],
                "cross_channel_distractor_video_ids": [],
                "gold_chunks": [
                    {"video_id": "v3", "chunk_index": 0},
                    {"video_id": "v4", "chunk_index": 0},
                ],
            },
        ],
    }


def _complete_predictions() -> list[dict]:
    return [
        {
            "query_id": "q1",
            "ranked_video_ids": ["v2", "v1", "v3"],
            "ranked_chunks": [{"video_id": "v1", "chunk_index": 0}],
            "fallback_used": False,
            "latency_ms": 10.0,
            "ablation": {
                "without_channel": {
                    "ranked_video_ids": ["v2", "v3", "v1"]
                }
            },
        },
        {
            "query_id": "q2",
            "ranked_video_ids": ["v3", "v4"],
            "ranked_chunks": [
                {"video_id": "v3", "chunk_index": 0},
                {"video_id": "v4", "chunk_index": 0},
            ],
            "fallback_used": True,
            "latency_ms": 30.0,
            "ablation": {
                "without_channel": {
                    "ranked_video_ids": ["v4", "v3"]
                }
            },
        },
    ]


def test_metric_math_and_distractor_accounting():
    report = evaluate_predictions(_dataset(), _complete_predictions())

    assert report["status"] == "complete"
    assert report["metrics"]["video_recall@1"] == pytest.approx(0.25)
    assert report["metrics"]["video_recall@3"] == pytest.approx(1.0)
    assert report["metrics"]["video_recall@5"] == pytest.approx(1.0)
    assert report["metrics"]["video_mrr"] == pytest.approx(0.75)
    assert report["metrics"]["final_chunk_recall@5"] == pytest.approx(1.0)
    assert report["metrics"]["fallback_rate"] == pytest.approx(0.5)
    assert report["metrics"]["mean_routing_latency_ms"] == pytest.approx(20.0)
    assert report["metrics"]["p95_routing_latency_ms"] == pytest.approx(29.0)
    assert report["metrics"]["channel_specific_video_recall@3"] == 1.0
    assert report["metrics"]["ordinary_video_recall@3"] == 1.0
    assert report["metrics"]["same_channel_distractor_error_rate"] == 1.0
    assert report["metrics"]["cross_channel_distractor_error_rate"] == 0.0
    assert report["metrics"]["missing_metadata_fallback_rate"] == 1.0


def test_partial_and_empty_inputs_are_explicit():
    partial = evaluate_predictions(
        _dataset(),
        [{"query_id": "q1", "ranked_video_ids": ["v1"]}],
    )

    assert partial["status"] == "partial"
    assert partial["counts"]["valid_prediction_count"] == 1
    assert partial["counts"]["missing_prediction_count"] == 1
    assert partial["metrics"]["video_recall@3"] == pytest.approx(0.5)
    assert partial["metrics"]["fallback_rate"] is None
    assert partial["metrics"]["mean_routing_latency_ms"] is None
    assert partial["metrics"]["final_chunk_recall@5"] is None
    assert partial["channel_ablation"]["status"] == "not_available"

    empty = evaluate_predictions(_dataset(), [])
    assert empty["status"] == "empty"
    assert empty["metrics"]["video_recall@1"] == 0.0
    assert empty["counts"]["missing_prediction_count"] == 2


def test_invalid_optional_fields_score_as_empty_and_report_error():
    report = evaluate_predictions(
        _dataset(),
        [
            {
                "query_id": "q1",
                "ranked_video_ids": ["v1"],
                "fallback_used": "no",
            }
        ],
    )

    assert report["status"] == "empty"
    assert report["counts"]["invalid_prediction_count"] == 1
    assert report["counts"]["missing_prediction_count"] == 1
    assert report["metrics"]["video_recall@3"] == 0.0
    assert report["errors"][0]["query_id"] == "q1"
    assert "fallback_used" in report["errors"][0]["message"]


def test_channel_ablation_reports_covered_queries_and_deltas():
    report = evaluate_predictions(_dataset(), _complete_predictions())
    ablation = report["channel_ablation"]

    assert ablation["status"] == "available"
    assert ablation["query_count"] == 2
    assert ablation["with_channel"]["video_mrr"] == pytest.approx(0.75)
    assert ablation["without_channel"]["video_mrr"] == pytest.approx(2 / 3)
    assert ablation["delta"]["video_mrr"] == pytest.approx(1 / 12)
    assert (
        ablation["delta"]["channel_specific_video_recall@3"]
        == pytest.approx(0.0)
    )


def test_fixture_is_valid_and_covers_approved_scenarios():
    dataset = load_dataset(FIXTURE_PATH)
    categories = {
        category
        for query in dataset["queries"]
        for category in query["categories"]
    }

    assert len(dataset["videos"]) >= 10
    assert len(dataset["queries"]) >= 8
    assert {
        "channel_targeted",
        "same_channel_distractor",
        "cross_channel_similar_title",
        "weak_title_strong_transcript",
        "missing_metadata",
        "renamed_channel",
        "cross_channel_comparison",
        "ordinary",
    }.issubset(categories)


def test_adapter_request_excludes_gold_labels_and_eval_categories():
    query_case = load_dataset(FIXTURE_PATH)["queries"][0]

    request = build_adapter_request(query_case, top_k=3)

    assert request == {
        "query_id": query_case["id"],
        "query": query_case["query"],
        "language": query_case["language"],
        "top_k": 3,
    }
    assert not {
        "relevant_video_ids",
        "gold_chunks",
        "categories",
        "same_channel_distractor_video_ids",
        "cross_channel_distractor_video_ids",
    }.intersection(request)


def test_non_circularity_guardrails_reject_outputs_and_sut_labels():
    dataset = _dataset()
    dataset["queries"][0]["ranked_video_ids"] = ["v1"]
    with pytest.raises(DatasetValidationError, match="router output fields"):
        validate_dataset(dataset)

    dataset = _dataset()
    dataset["label_provenance"]["system_under_test_used"] = True
    with pytest.raises(DatasetValidationError, match="must be false"):
        validate_dataset(dataset)


def test_validation_rejects_unknown_and_overlapping_distractor_labels():
    dataset = _dataset()
    dataset["queries"][0]["same_channel_distractor_video_ids"] = ["v1"]
    with pytest.raises(DatasetValidationError, match="relevant videos"):
        validate_dataset(dataset)

    dataset = _dataset()
    dataset["queries"][0]["relevant_video_ids"] = ["does-not-exist"]
    with pytest.raises(DatasetValidationError, match="unknown video IDs"):
        validate_dataset(dataset)

    dataset = _dataset()
    dataset["queries"][0]["gold_chunks"][0]["chunk_index"] = 99
    with pytest.raises(DatasetValidationError, match="unknown chunk"):
        validate_dataset(dataset)


def test_runner_uses_label_blind_request_and_records_adapter_failures():
    dataset = _dataset()
    requests = []

    def adapter(request):
        requests.append(dict(request))
        if request["query_id"] == "q2":
            raise RuntimeError("synthetic adapter failure")
        return {
            "ranked_video_ids": ["v1"],
            "fallback_used": False,
            "latency_ms": 1.5,
        }

    report = run_adapter(dataset, adapter)

    assert requests[0] == {
        "query_id": "q1",
        "query": dataset["queries"][0]["query"],
        "language": "en",
        "top_k": 5,
    }
    assert "relevant_video_ids" not in requests[0]
    assert report["status"] == "partial"
    assert report["counts"]["adapter_error_count"] == 1
    assert any(error["type"] == "adapter_error" for error in report["errors"])


def test_duplicate_rankings_and_duplicate_predictions_are_reported():
    report = evaluate_predictions(
        _dataset(),
        [
            {"query_id": "q1", "ranked_video_ids": ["v1", "v1"]},
            {"query_id": "q1", "ranked_video_ids": ["v1"]},
        ],
    )

    assert report["status"] == "empty"
    assert report["counts"]["invalid_prediction_count"] == 1
    assert any(error["type"] == "duplicate_prediction" for error in report["errors"])


def test_fixture_mutation_does_not_hide_label_overlap():
    dataset = load_dataset(FIXTURE_PATH)
    mutated = copy.deepcopy(dataset)
    first_query = mutated["queries"][0]
    first_query["cross_channel_distractor_video_ids"].append(
        first_query["relevant_video_ids"][0]
    )

    with pytest.raises(DatasetValidationError, match="relevant videos"):
        validate_dataset(mutated)
