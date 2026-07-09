"""Tests for the cross-encoder reranking stage."""

import importlib
import os
import sys
import threading
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
LOCAL_PREVIEW_DIR = ROOT_DIR / "local_preview"
if str(LOCAL_PREVIEW_DIR) not in sys.path:
    sys.path.insert(0, str(LOCAL_PREVIEW_DIR))

os.environ["YT_RAG_SKIP_GLOBAL_SERVICE"] = "1"
local_api = importlib.import_module("local_api")
LocalRAGService = local_api.LocalRAGService

from multilingual.reranker import CrossEncoderReranker  # noqa: E402


def _rows():
    return [
        {"video_id": "vid1", "chunk_index": 0, "text": "general AI talk", "score": 0.9, "rank": 1, "start": 0, "end": 10},
        {"video_id": "vid1", "chunk_index": 1, "text": "unrelated topic", "score": 0.8, "rank": 2, "start": 10, "end": 20},
        {"video_id": "vid1", "chunk_index": 2, "text": "Jetson Orin edge inference", "score": 0.7, "rank": 3, "start": 20, "end": 30},
    ]


def test_rerank_reorders_rows_and_annotates_scores():
    reranker = CrossEncoderReranker(
        model_name="fake-model",
        score_fn=lambda pairs: [0.9 if "Jetson" in text else 0.1 for _q, text in pairs],
    )

    outcome = reranker.rerank("jetson orin", _rows())

    assert outcome["applied"] is True
    assert outcome["error"] is None
    assert outcome["scored_count"] == 3
    assert outcome["model"] == "fake-model"
    top = outcome["rows"][0]
    assert top["chunk_index"] == 2
    assert top["rank"] == 1
    assert top["pre_rerank_rank"] == 3
    assert top["pre_rerank_score"] == 0.7
    assert top["rerank_score"] == 0.9
    assert top["score"] == 0.9
    assert [row["rank"] for row in outcome["rows"]] == [1, 2, 3]


def test_rerank_probability_scores_are_used_as_is():
    reranker = CrossEncoderReranker(
        model_name="fake-model", score_fn=lambda pairs: [0.2, 0.4, 0.6]
    )
    outcome = reranker.rerank("query", _rows())
    assert [row["rerank_score"] for row in outcome["rows"]] == [0.6, 0.4, 0.2]


def test_rerank_logit_scores_are_sigmoid_normalized():
    reranker = CrossEncoderReranker(
        model_name="fake-model", score_fn=lambda pairs: [-4.0, 0.0, 4.0]
    )
    outcome = reranker.rerank("query", _rows())
    scores = [row["rerank_score"] for row in outcome["rows"]]
    assert all(0.0 <= score <= 1.0 for score in scores)
    assert scores == sorted(scores, reverse=True)
    assert outcome["rows"][0]["chunk_index"] == 2


def test_rerank_only_scores_top_n_and_keeps_tail_order():
    reranker = CrossEncoderReranker(
        model_name="fake-model", score_fn=lambda pairs: [0.1 for _ in pairs]
    )
    outcome = reranker.rerank("query", _rows(), top_n=2)
    assert outcome["scored_count"] == 2
    assert outcome["rows"][2]["chunk_index"] == 2
    assert "rerank_score" not in outcome["rows"][2]
    assert [row["rank"] for row in outcome["rows"]] == [1, 2, 3]


def test_rerank_passes_through_when_model_unavailable():
    reranker = CrossEncoderReranker(model_name="fake-model")
    reranker._load_error = "OSError: offline"

    rows = _rows()
    outcome = reranker.rerank("query", rows)

    assert outcome["applied"] is False
    assert outcome["rows"] is rows
    assert outcome["error"] == "OSError: offline"


def test_rerank_passes_through_when_scorer_fails():
    def _boom(pairs):
        raise RuntimeError("scorer exploded")

    reranker = CrossEncoderReranker(model_name="fake-model", score_fn=_boom)
    rows = _rows()
    outcome = reranker.rerank("query", rows)

    assert outcome["applied"] is False
    assert outcome["rows"] is rows
    assert "scorer exploded" in outcome["error"]


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
                                {
                                    "raw_text": "Jetson Orin は エッジAI 推論 に使う",
                                    "start": 20,
                                    "end": 30,
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
            "chunk_index": 2,
            "text": "Jetson Orin は エッジAI 推論 に使う",
            "score": 0.60,
            "dense_score": 0.60,
            "start": 20,
            "end": 30,
            "rank": 2,
        },
    ]
    service._dense_search = lambda query, k, language, video_id=None: dense_rows
    service._lexical_bm25_search = lambda query, k, language, video_id=None: []
    return service


def test_retrieve_applies_cross_encoder_reranker():
    service = _make_service()
    service._reranker = CrossEncoderReranker(
        model_name="fake-model",
        score_fn=lambda pairs: [0.9 if "Jetson" in text else 0.1 for _q, text in pairs],
    )

    result = service.retrieve(
        "Jetson Orin エッジAI 推論",
        k=2,
        retrieval_mode="dense",
        reranker="cross_encoder",
    )

    reranker_details = result["details"]["reranker"]
    assert reranker_details["requested"] == "cross_encoder"
    assert reranker_details["applied"] is True
    assert reranker_details["scored_count"] == 2
    assert result["results"][0]["chunk_index"] == 2


def test_retrieve_defaults_to_no_reranker(monkeypatch):
    monkeypatch.delenv(local_api.RERANKER_ENV, raising=False)
    service = _make_service()

    result = service.retrieve("Jetson Orin", k=2, retrieval_mode="dense")

    reranker_details = result["details"]["reranker"]
    assert reranker_details["requested"] == "none"
    assert reranker_details["applied"] is False
    assert result["results"][0]["chunk_index"] == 0


def test_retrieve_reranker_env_default(monkeypatch):
    monkeypatch.setenv(local_api.RERANKER_ENV, "1")
    service = _make_service()
    service._reranker = CrossEncoderReranker(
        model_name="fake-model",
        score_fn=lambda pairs: [0.9 if "Jetson" in text else 0.1 for _q, text in pairs],
    )

    result = service.retrieve("Jetson Orin エッジAI 推論", k=2, retrieval_mode="dense")

    assert result["details"]["reranker"]["requested"] == "cross_encoder"
    assert result["results"][0]["chunk_index"] == 2


def test_retrieve_rejects_unknown_reranker():
    service = _make_service()
    try:
        service.retrieve("query", k=2, retrieval_mode="dense", reranker="bogus")
    except ValueError as exc:
        assert "reranker" in str(exc)
    else:
        raise AssertionError("expected ValueError for unknown reranker")


def test_retrieve_reports_reranker_load_failure_and_falls_back():
    service = _make_service()
    service._reranker = CrossEncoderReranker(model_name="fake-model")
    service._reranker._load_error = "OSError: offline"

    result = service.retrieve(
        "Jetson Orin エッジAI 推論",
        k=2,
        retrieval_mode="dense",
        reranker="cross_encoder",
    )

    reranker_details = result["details"]["reranker"]
    assert reranker_details["applied"] is False
    assert reranker_details["error"] == "OSError: offline"
    assert result["results"][0]["chunk_index"] == 0


def test_rerank_passes_through_on_mismatched_score_count():
    reranker = CrossEncoderReranker(model_name="fake-model", score_fn=lambda pairs: [0.5])
    rows = _rows()
    outcome = reranker.rerank("query", rows)

    assert outcome["applied"] is False
    assert outcome["rows"] is rows
    assert "mismatched" in outcome["error"]
