"""Offline evaluation helpers for video-first retrieval routing."""

from evals.video_routing.dataset import (
    DatasetValidationError,
    build_adapter_request,
    load_dataset,
    validate_dataset,
)
from evals.video_routing.adapters import (
    MultiVectorRouterAdapter,
    build_library_videos,
)
from evals.video_routing.scoring import evaluate_predictions

__all__ = [
    "DatasetValidationError",
    "build_adapter_request",
    "build_library_videos",
    "evaluate_predictions",
    "load_dataset",
    "MultiVectorRouterAdapter",
    "validate_dataset",
]
