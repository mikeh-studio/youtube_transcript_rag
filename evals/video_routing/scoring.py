"""Metrics for deterministic, offline video-routing evaluation."""

from __future__ import annotations

import math
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence

from evals.video_routing.dataset import validate_dataset


def _mean(values: Iterable[float]) -> Optional[float]:
    normalized = [float(value) for value in values]
    if not normalized:
        return None
    return sum(normalized) / len(normalized)


def _percentile(values: Sequence[float], percentile: float) -> Optional[float]:
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * percentile / 100.0
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def _recall_at_k(
    relevant_video_ids: Sequence[str], ranked_video_ids: Sequence[str], k: int
) -> float:
    relevant = set(relevant_video_ids)
    if not relevant:
        return 0.0
    hits = relevant.intersection(ranked_video_ids[: max(0, int(k))])
    return len(hits) / len(relevant)


def _video_mrr(
    relevant_video_ids: Sequence[str], ranked_video_ids: Sequence[str]
) -> float:
    relevant = set(relevant_video_ids)
    for rank, video_id in enumerate(ranked_video_ids, start=1):
        if video_id in relevant:
            return 1.0 / rank
    return 0.0


def _chunk_recall_at_5(
    gold_chunks: Sequence[Mapping[str, Any]],
    ranked_chunks: Sequence[Mapping[str, Any]],
) -> float:
    if not gold_chunks:
        return 0.0
    gold = {
        (str(chunk["video_id"]), int(chunk["chunk_index"]))
        for chunk in gold_chunks
    }
    retrieved = {
        (str(chunk["video_id"]), int(chunk["chunk_index"]))
        for chunk in ranked_chunks[:5]
    }
    return len(gold.intersection(retrieved)) / len(gold)


def _distractor_error(
    distractor_ids: Sequence[str],
    relevant_ids: Sequence[str],
    ranked_video_ids: Sequence[str],
) -> Optional[bool]:
    if not distractor_ids:
        return None
    ranks = {video_id: rank for rank, video_id in enumerate(ranked_video_ids)}
    relevant_ranks = [ranks[item] for item in relevant_ids if item in ranks]
    first_relevant_rank = min(relevant_ranks) if relevant_ranks else math.inf
    return any(
        distractor_id in ranks and ranks[distractor_id] < first_relevant_rank
        for distractor_id in distractor_ids
    )


def _validate_ranked_ids(value: Any, *, location: str) -> list[str]:
    if not isinstance(value, list):
        raise ValueError(f"{location} must be a list")
    if any(not isinstance(item, str) or not item.strip() for item in value):
        raise ValueError(f"{location} must contain non-empty strings")
    normalized = [item.strip() for item in value]
    if len(normalized) != len(set(normalized)):
        raise ValueError(f"{location} must not contain duplicates")
    return normalized


def _validate_ranked_chunks(value: Any, *, location: str) -> list[dict]:
    if not isinstance(value, list):
        raise ValueError(f"{location} must be a list")
    normalized = []
    for index, chunk in enumerate(value):
        chunk_location = f"{location}[{index}]"
        if not isinstance(chunk, Mapping):
            raise ValueError(f"{chunk_location} must be an object")
        video_id = chunk.get("video_id")
        chunk_index = chunk.get("chunk_index")
        if not isinstance(video_id, str) or not video_id.strip():
            raise ValueError(f"{chunk_location}.video_id must be non-empty text")
        if not isinstance(chunk_index, int) or chunk_index < 0:
            raise ValueError(
                f"{chunk_location}.chunk_index must be a non-negative integer"
            )
        normalized.append({"video_id": video_id.strip(), "chunk_index": chunk_index})
    return normalized


def _normalize_prediction(prediction: Any, *, query_id: str) -> dict:
    if not isinstance(prediction, Mapping):
        raise ValueError("prediction must be an object")
    ranked_video_ids = _validate_ranked_ids(
        prediction.get("ranked_video_ids"),
        location="ranked_video_ids",
    )
    normalized: Dict[str, Any] = {
        "query_id": query_id,
        "ranked_video_ids": ranked_video_ids,
    }
    if "ranked_chunks" in prediction:
        normalized["ranked_chunks"] = _validate_ranked_chunks(
            prediction["ranked_chunks"], location="ranked_chunks"
        )
    if "fallback_used" in prediction:
        if not isinstance(prediction["fallback_used"], bool):
            raise ValueError("fallback_used must be a boolean")
        normalized["fallback_used"] = prediction["fallback_used"]
    if "latency_ms" in prediction:
        latency = prediction["latency_ms"]
        if (
            isinstance(latency, bool)
            or not isinstance(latency, (int, float))
            or not math.isfinite(float(latency))
            or float(latency) < 0
        ):
            raise ValueError("latency_ms must be a finite non-negative number")
        normalized["latency_ms"] = float(latency)
    if "ablation" in prediction:
        ablation = prediction["ablation"]
        if not isinstance(ablation, Mapping):
            raise ValueError("ablation must be an object")
        without_channel = ablation.get("without_channel")
        if not isinstance(without_channel, Mapping):
            raise ValueError("ablation.without_channel must be an object")
        normalized["ablation"] = {
            "without_channel": {
                "ranked_video_ids": _validate_ranked_ids(
                    without_channel.get("ranked_video_ids"),
                    location="ablation.without_channel.ranked_video_ids",
                )
            }
        }
    return normalized


def _index_predictions(predictions: Any) -> tuple[dict[str, Any], list[dict]]:
    if isinstance(predictions, Mapping):
        indexed = {str(key): value for key, value in predictions.items()}
        return indexed, []
    if not isinstance(predictions, Sequence) or isinstance(predictions, (str, bytes)):
        raise ValueError("predictions must be a mapping or a list")

    indexed: dict[str, Any] = {}
    errors: list[dict] = []
    for index, prediction in enumerate(predictions):
        if not isinstance(prediction, Mapping):
            errors.append(
                {
                    "type": "invalid_prediction",
                    "location": f"predictions[{index}]",
                    "message": "prediction must be an object",
                }
            )
            continue
        query_id = prediction.get("query_id")
        if not isinstance(query_id, str) or not query_id.strip():
            errors.append(
                {
                    "type": "invalid_prediction",
                    "location": f"predictions[{index}]",
                    "message": "query_id must be non-empty text",
                }
            )
            continue
        query_id = query_id.strip()
        if query_id in indexed:
            errors.append(
                {
                    "type": "duplicate_prediction",
                    "query_id": query_id,
                    "message": "only the first prediction was scored",
                }
            )
            continue
        indexed[query_id] = prediction
    return indexed, errors


def _ablation_report(per_query: Sequence[dict]) -> dict:
    covered = [row for row in per_query if row.get("without_channel") is not None]
    if not covered:
        return {
            "status": "not_available",
            "query_count": 0,
            "with_channel": None,
            "without_channel": None,
            "delta": None,
        }

    def summary(side: str) -> dict:
        return {
            "video_recall@1": _mean(
                row[side]["video_recall@1"] for row in covered
            ),
            "video_recall@3": _mean(
                row[side]["video_recall@3"] for row in covered
            ),
            "video_recall@5": _mean(
                row[side]["video_recall@5"] for row in covered
            ),
            "video_mrr": _mean(row[side]["video_mrr"] for row in covered),
            "channel_specific_video_recall@3": _mean(
                row[side]["video_recall@3"]
                for row in covered
                if "channel_targeted" in row["categories"]
            ),
            "ordinary_video_recall@3": _mean(
                row[side]["video_recall@3"]
                for row in covered
                if "ordinary" in row["categories"]
            ),
        }

    with_channel = summary("with_channel")
    without_channel = summary("without_channel")
    delta = {
        key: (
            None
            if with_channel[key] is None or without_channel[key] is None
            else with_channel[key] - without_channel[key]
        )
        for key in with_channel
    }
    return {
        "status": "available",
        "query_count": len(covered),
        "with_channel": with_channel,
        "without_channel": without_channel,
        "delta": delta,
    }


def evaluate_predictions(dataset: Mapping[str, Any], predictions: Any) -> dict:
    """Score precomputed router outputs against an independently labeled dataset.

    Missing or invalid predictions count as zero video recall while coverage
    counters and errors make the partial state explicit. Optional metrics use
    only observations that actually provide the corresponding field.
    """
    validate_dataset(dataset)
    indexed, errors = _index_predictions(predictions)
    query_ids = {str(query["id"]) for query in dataset["queries"]}
    for unknown_id in sorted(set(indexed).difference(query_ids)):
        errors.append(
            {
                "type": "unknown_query",
                "query_id": unknown_id,
                "message": "prediction does not match a dataset query",
            }
        )

    per_query = []
    valid_prediction_count = 0
    missing_prediction_count = 0
    invalid_prediction_count = 0
    fallback_values: list[bool] = []
    latencies: list[float] = []
    chunk_scores: list[float] = []
    missing_metadata_fallbacks: list[bool] = []

    for query in dataset["queries"]:
        query_id = str(query["id"])
        raw_prediction = indexed.get(query_id)
        prediction_status = "valid"
        if raw_prediction is None:
            prediction_status = "missing"
            missing_prediction_count += 1
            prediction = {"query_id": query_id, "ranked_video_ids": []}
        else:
            try:
                prediction = _normalize_prediction(
                    raw_prediction, query_id=query_id
                )
                valid_prediction_count += 1
            except ValueError as exc:
                prediction_status = "invalid"
                invalid_prediction_count += 1
                prediction = {"query_id": query_id, "ranked_video_ids": []}
                errors.append(
                    {
                        "type": "invalid_prediction",
                        "query_id": query_id,
                        "message": str(exc),
                    }
                )

        ranked_video_ids = prediction["ranked_video_ids"]
        relevant_video_ids = query["relevant_video_ids"]
        with_channel = {
            "video_recall@1": _recall_at_k(
                relevant_video_ids, ranked_video_ids, 1
            ),
            "video_recall@3": _recall_at_k(
                relevant_video_ids, ranked_video_ids, 3
            ),
            "video_recall@5": _recall_at_k(
                relevant_video_ids, ranked_video_ids, 5
            ),
            "video_mrr": _video_mrr(relevant_video_ids, ranked_video_ids),
        }

        final_chunk_recall = None
        if query.get("gold_chunks") and "ranked_chunks" in prediction:
            final_chunk_recall = _chunk_recall_at_5(
                query["gold_chunks"], prediction["ranked_chunks"]
            )
            chunk_scores.append(final_chunk_recall)
        if "fallback_used" in prediction:
            fallback_values.append(prediction["fallback_used"])
            if "missing_metadata" in query["categories"]:
                missing_metadata_fallbacks.append(prediction["fallback_used"])
        if "latency_ms" in prediction:
            latencies.append(prediction["latency_ms"])

        without_channel = None
        if "ablation" in prediction:
            ablation_ids = prediction["ablation"]["without_channel"][
                "ranked_video_ids"
            ]
            without_channel = {
                "video_recall@1": _recall_at_k(
                    relevant_video_ids, ablation_ids, 1
                ),
                "video_recall@3": _recall_at_k(
                    relevant_video_ids, ablation_ids, 3
                ),
                "video_recall@5": _recall_at_k(
                    relevant_video_ids, ablation_ids, 5
                ),
                "video_mrr": _video_mrr(relevant_video_ids, ablation_ids),
            }

        per_query.append(
            {
                "query_id": query_id,
                "prediction_status": prediction_status,
                "categories": list(query["categories"]),
                "ranked_video_ids": ranked_video_ids,
                "with_channel": with_channel,
                "without_channel": without_channel,
                "final_chunk_recall@5": final_chunk_recall,
                "fallback_used": prediction.get("fallback_used"),
                "latency_ms": prediction.get("latency_ms"),
                "same_channel_distractor_error": _distractor_error(
                    query["same_channel_distractor_video_ids"],
                    relevant_video_ids,
                    ranked_video_ids,
                ),
                "cross_channel_distractor_error": _distractor_error(
                    query["cross_channel_distractor_video_ids"],
                    relevant_video_ids,
                    ranked_video_ids,
                ),
            }
        )

    query_count = len(dataset["queries"])
    status = "complete"
    if valid_prediction_count == 0:
        status = "empty"
    elif valid_prediction_count < query_count:
        status = "partial"

    same_channel_errors = [
        row["same_channel_distractor_error"]
        for row in per_query
        if row["same_channel_distractor_error"] is not None
    ]
    cross_channel_errors = [
        row["cross_channel_distractor_error"]
        for row in per_query
        if row["cross_channel_distractor_error"] is not None
    ]
    metrics = {
        "video_recall@1": _mean(
            row["with_channel"]["video_recall@1"] for row in per_query
        ),
        "video_recall@3": _mean(
            row["with_channel"]["video_recall@3"] for row in per_query
        ),
        "video_recall@5": _mean(
            row["with_channel"]["video_recall@5"] for row in per_query
        ),
        "video_mrr": _mean(
            row["with_channel"]["video_mrr"] for row in per_query
        ),
        "final_chunk_recall@5": _mean(chunk_scores),
        "fallback_rate": _mean(float(value) for value in fallback_values),
        "mean_routing_latency_ms": _mean(latencies),
        "p95_routing_latency_ms": _percentile(latencies, 95),
        "channel_specific_video_recall@3": _mean(
            row["with_channel"]["video_recall@3"]
            for row in per_query
            if "channel_targeted" in row["categories"]
        ),
        "ordinary_video_recall@3": _mean(
            row["with_channel"]["video_recall@3"]
            for row in per_query
            if "ordinary" in row["categories"]
        ),
        "same_channel_distractor_error_rate": _mean(
            float(value) for value in same_channel_errors
        ),
        "cross_channel_distractor_error_rate": _mean(
            float(value) for value in cross_channel_errors
        ),
        "missing_metadata_fallback_rate": _mean(
            float(value) for value in missing_metadata_fallbacks
        ),
    }
    return {
        "dataset_id": dataset["dataset_id"],
        "schema_version": dataset["schema_version"],
        "status": status,
        "counts": {
            "query_count": query_count,
            "valid_prediction_count": valid_prediction_count,
            "missing_prediction_count": missing_prediction_count,
            "invalid_prediction_count": invalid_prediction_count,
            "fallback_observation_count": len(fallback_values),
            "latency_observation_count": len(latencies),
            "chunk_metric_query_count": len(chunk_scores),
            "same_channel_distractor_query_count": len(same_channel_errors),
            "cross_channel_distractor_query_count": len(cross_channel_errors),
            "missing_metadata_fallback_observation_count": len(
                missing_metadata_fallbacks
            ),
        },
        "metrics": metrics,
        "channel_ablation": _ablation_report(per_query),
        "per_query": per_query,
        "errors": errors,
    }
