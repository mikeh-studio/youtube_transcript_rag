"""Deterministic tests for offline retrieval benchmark scoring."""

import math

from evals.scoring import (
    evaluate_thresholds,
    gold_matches_result,
    mrr_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    select_optimized_run,
)


def test_gold_evidence_matches_by_chunk_index():
    gold = {"video_id": "vid1", "chunk_index": 3}
    result = {"video_id": "vid1", "chunk_index": 3, "start": 40, "end": 60}

    assert gold_matches_result(gold, result) is True


def test_gold_evidence_matches_by_timestamp_overlap():
    gold = {"video_id": "vid1", "start": 30.0, "end": 50.0}
    result = {"video_id": "vid1", "chunk_index": 1, "start": 45.0, "end": 70.0}

    assert gold_matches_result(gold, result) is True


def test_recall_at_k_counts_unique_gold_hits():
    gold = [
        {"video_id": "vid1", "chunk_index": 1},
        {"video_id": "vid1", "chunk_index": 3},
    ]
    results = [
        {"video_id": "vid1", "chunk_index": 0},
        {"video_id": "vid1", "chunk_index": 1},
        {"video_id": "vid1", "chunk_index": 3},
    ]

    assert recall_at_k(gold, results, 1) == 0.0
    assert recall_at_k(gold, results, 2) == 0.5
    assert recall_at_k(gold, results, 3) == 1.0


def test_precision_at_k_counts_relevant_ranked_rows():
    gold = [{"video_id": "vid1", "chunk_index": 1}]
    results = [
        {"video_id": "vid1", "chunk_index": 1},
        {"video_id": "vid1", "chunk_index": 0},
        {"video_id": "vid1", "chunk_index": 1},
    ]

    assert precision_at_k(gold, results, 1) == 1.0
    assert precision_at_k(gold, results, 3) == 2 / 3
    assert precision_at_k(gold, results, 5) == 2 / 5


def test_mrr_at_k_uses_first_gold_rank():
    gold = [{"video_id": "vid1", "chunk_index": 3}]
    results = [
        {"video_id": "vid1", "chunk_index": 0},
        {"video_id": "vid1", "chunk_index": 2},
        {"video_id": "vid1", "chunk_index": 3},
    ]

    assert mrr_at_k(gold, results, 10) == 1 / 3


def test_ndcg_at_k_de_duplicates_repeated_gold_hits():
    gold = [
        {"video_id": "vid1", "chunk_index": 1},
        {"video_id": "vid1", "chunk_index": 2},
    ]
    results = [
        {"video_id": "vid1", "chunk_index": 0},
        {"video_id": "vid1", "chunk_index": 1},
        {"video_id": "vid1", "chunk_index": 1},
        {"video_id": "vid1", "chunk_index": 2},
    ]
    expected_dcg = (1 / math.log2(2 + 1)) + (1 / math.log2(4 + 1))
    expected_idcg = 1.0 + (1 / math.log2(3))

    assert ndcg_at_k(gold, results, 10) == expected_dcg / expected_idcg


def test_thresholds_use_absolute_checks_for_small_fixture_sets():
    baseline = {
        "gold_recall@5": 0.5,
        "MRR@10": 0.5,
        "nDCG@10": 0.5,
        "p95_latency_ms": 10.0,
        "failed_query_count": 0,
    }
    candidate = {
        "gold_recall@5": 1.0,
        "MRR@10": 1.0,
        "nDCG@10": 1.0,
        "p95_latency_ms": 11.0,
        "failed_query_count": 0,
    }
    criteria = {
        "min_queries_for_relative": 20,
        "sample_absolute": {
            "gold_recall@5": 1.0,
            "MRR@10": 1.0,
            "nDCG@10": 1.0,
        },
        "max_p95_latency_regression": 0.2,
        "failed_query_count": 0,
    }

    result = evaluate_thresholds(baseline, candidate, criteria, query_count=3)

    assert result["small_sample"] is True
    assert result["passed"] is True
    assert all(check["mode"] != "relative_or_absolute" for check in result["checks"][:3])


def test_thresholds_fail_when_latency_regresses_too_far():
    baseline = {
        "gold_recall@5": 1.0,
        "MRR@10": 1.0,
        "nDCG@10": 1.0,
        "p95_latency_ms": 10.0,
        "failed_query_count": 0,
    }
    candidate = {
        "gold_recall@5": 1.0,
        "MRR@10": 1.0,
        "nDCG@10": 1.0,
        "p95_latency_ms": 13.0,
        "failed_query_count": 0,
    }
    criteria = {
        "min_queries_for_relative": 20,
        "sample_absolute": {
            "gold_recall@5": 1.0,
            "MRR@10": 1.0,
            "nDCG@10": 1.0,
        },
        "max_p95_latency_regression": 0.2,
        "failed_query_count": 0,
    }

    result = evaluate_thresholds(baseline, candidate, criteria, query_count=3)

    assert result["passed"] is False
    assert any(
        check["metric"] == "p95_latency_ms" and check["passed"] is False
        for check in result["checks"]
    )


def test_select_optimized_run_prefers_quality_then_latency():
    runs = [
        {
            "name": "hybrid_baseline",
            "metrics": {
                "gold_recall@5": 0.8,
                "MRR@10": 0.7,
                "nDCG@10": 0.75,
                "p95_latency_ms": 5.0,
                "failed_query_count": 0,
            },
        },
        {
            "name": "optimized_hybrid",
            "metrics": {
                "gold_recall@5": 1.0,
                "MRR@10": 0.95,
                "nDCG@10": 0.97,
                "p95_latency_ms": 6.0,
                "failed_query_count": 0,
            },
        },
        {
            "name": "broken_fast",
            "metrics": {
                "gold_recall@5": 1.0,
                "MRR@10": 1.0,
                "nDCG@10": 1.0,
                "p95_latency_ms": 1.0,
                "failed_query_count": 1,
            },
        },
    ]

    assert select_optimized_run(runs)["name"] == "optimized_hybrid"
