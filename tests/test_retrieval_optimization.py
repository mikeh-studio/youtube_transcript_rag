"""Tests for the optimized hybrid retrieval profile."""

import importlib
import os
import sys
import threading
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
LOCAL_PREVIEW_DIR = ROOT_DIR / "local_preview"
if str(LOCAL_PREVIEW_DIR) not in sys.path:
    sys.path.insert(0, str(LOCAL_PREVIEW_DIR))

os.environ["YT_RAG_SKIP_GLOBAL_SERVICE"] = "1"
local_api = importlib.import_module("local_api")
LocalRAGService = local_api.LocalRAGService


def _make_service():
    service = LocalRAGService.__new__(LocalRAGService)
    service.feedback = {}
    service.feedback_index = {}
    service.feedback_lock = threading.Lock()
    service.feedback_tuning_enabled = False
    service._persist_feedback = lambda: None
    service.engine = type(
        "DummyEngine",
        (),
        {
            "library": type(
                "DummyLibrary",
                (),
                {
                    "videos": {
                        "vid1": {
                            "chunks": [
                                {"raw_text": "AI 推論 の一般論", "start": 0, "end": 10},
                                {"raw_text": "別の話題", "start": 10, "end": 20},
                                {"raw_text": "さらに別の話題", "start": 20, "end": 30},
                                {
                                    "raw_text": "Jetson Orin は エッジAI 推論 に使う",
                                    "start": 30,
                                    "end": 40,
                                },
                            ]
                        }
                    }
                },
            )()
        },
    )()
    dense_rows = [
        {
            "video_id": "vid1",
            "chunk_index": 0,
            "text": "AI 推論 の一般論",
            "score": 0.95,
            "dense_score": 0.95,
            "start": 0,
            "end": 10,
            "rank": 1,
        },
        {
            "video_id": "vid1",
            "chunk_index": 1,
            "text": "別の話題",
            "score": 0.80,
            "dense_score": 0.80,
            "start": 10,
            "end": 20,
            "rank": 2,
        },
        {
            "video_id": "vid1",
            "chunk_index": 2,
            "text": "さらに別の話題",
            "score": 0.75,
            "dense_score": 0.75,
            "start": 20,
            "end": 30,
            "rank": 3,
        },
        {
            "video_id": "vid1",
            "chunk_index": 3,
            "text": "Jetson Orin は エッジAI 推論 に使う",
            "score": 0.60,
            "dense_score": 0.60,
            "start": 30,
            "end": 40,
            "rank": 4,
        },
    ]
    lexical_rows = [
        {
            "video_id": "vid1",
            "chunk_index": 3,
            "text": "Jetson Orin は エッジAI 推論 に使う",
            "score": 8.0,
            "lexical_score": 8.0,
            "start": 30,
            "end": 40,
            "rank": 1,
        },
        {
            "video_id": "vid1",
            "chunk_index": 0,
            "text": "AI 推論 の一般論",
            "score": 4.0,
            "lexical_score": 4.0,
            "start": 0,
            "end": 10,
            "rank": 2,
        },
    ]
    service._dense_search = lambda query, k, language, video_id=None: dense_rows
    service._lexical_bm25_search = (
        lambda query, k, language, video_id=None: lexical_rows
    )
    return service


def test_optimized_hybrid_profile_fixes_rrf_dense_tie_bias():
    service = _make_service()

    baseline = service.retrieve(
        "Jetson Orin エッジAI 推論",
        k=1,
        retrieval_mode="hybrid",
        retrieval_profile="baseline_rrf",
    )
    optimized = service.retrieve(
        "Jetson Orin エッジAI 推論",
        k=1,
        retrieval_mode="hybrid",
        retrieval_profile="optimized_v1",
    )

    assert baseline["results"][0]["chunk_index"] == 0
    assert optimized["results"][0]["chunk_index"] == 3
    assert optimized["details"]["fusion_profile"] == "optimized_v1"
    assert optimized["details"]["fusion_weights"]["lexical"] > 0
