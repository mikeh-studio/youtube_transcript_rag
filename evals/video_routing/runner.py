"""Runner and adapter contract for video-first routing evaluations."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Mapping, Protocol, runtime_checkable

from evals.video_routing.dataset import build_adapter_request, load_dataset
from evals.video_routing.scoring import evaluate_predictions


@runtime_checkable
class RoutingAdapter(Protocol):
    """Minimal integration boundary for the production video router."""

    def route(self, request: Mapping[str, Any]) -> Mapping[str, Any]:
        """Return ranked IDs plus optional chunks, fallback, latency, and ablation."""


def _call_adapter(adapter: Any, request: Mapping[str, Any]) -> Mapping[str, Any]:
    if hasattr(adapter, "route"):
        response = adapter.route(request)
    elif callable(adapter):
        response = adapter(request)
    else:
        raise TypeError("adapter must be callable or implement route(request)")
    if not isinstance(response, Mapping):
        raise TypeError("adapter response must be an object")
    return response


def run_adapter(
    dataset: Mapping[str, Any],
    adapter: RoutingAdapter | Any,
    *,
    top_k: int = 5,
) -> dict:
    """Execute a router adapter and score its label-blind outputs.

    Adapter failures are converted to empty rankings so every dataset query is
    represented in recall metrics. When the adapter does not provide its own
    routing latency, the runner supplies wall-clock latency.
    """
    predictions = []
    adapter_errors = []
    for query_case in dataset["queries"]:
        request = build_adapter_request(query_case, top_k=top_k)
        started = time.perf_counter()
        try:
            response = dict(_call_adapter(adapter, request))
            response["query_id"] = request["query_id"]
            response.setdefault(
                "latency_ms", (time.perf_counter() - started) * 1000.0
            )
        except Exception as exc:  # adapter errors are evaluation data
            response = {
                "query_id": request["query_id"],
                "ranked_video_ids": [],
                "latency_ms": (time.perf_counter() - started) * 1000.0,
            }
            adapter_errors.append(
                {
                    "type": "adapter_error",
                    "query_id": request["query_id"],
                    "message": f"{type(exc).__name__}: {exc}",
                }
            )
        predictions.append(response)

    report = evaluate_predictions(dataset, predictions)
    report["errors"].extend(adapter_errors)
    report["counts"]["adapter_error_count"] = len(adapter_errors)
    if adapter_errors and report["status"] == "complete":
        report["status"] = "partial"
    return report


def load_predictions(path: Path | str) -> list[dict]:
    """Load predictions from a JSON array/object or JSONL file."""
    prediction_path = Path(path)
    text = prediction_path.read_text(encoding="utf-8")
    if prediction_path.suffix.lower() == ".jsonl":
        rows = []
        for line_number, line in enumerate(text.splitlines(), start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(
                    f"prediction line {line_number} must contain an object"
                )
            rows.append(row)
        return rows
    payload = json.loads(text)
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        return [
            dict(value, query_id=str(query_id))
            for query_id, value in payload.items()
        ]
    raise ValueError("prediction file must contain a JSON array or object")


def evaluate_prediction_file(
    dataset_path: Path | str,
    prediction_path: Path | str,
    *,
    output_path: Path | str | None = None,
) -> dict:
    """Score a precomputed prediction file and optionally write a JSON report."""
    dataset = load_dataset(dataset_path)
    predictions = load_predictions(prediction_path)
    report = evaluate_predictions(dataset, predictions)
    if output_path is not None:
        destination = Path(output_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Score precomputed video-routing predictions"
    )
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    report = evaluate_prediction_file(
        args.dataset, args.predictions, output_path=args.out
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
