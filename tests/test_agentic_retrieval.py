"""Tests for the agentic retrieval loop."""

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

from agentic_retrieval import (  # noqa: E402
    STOPPED_MAX_ATTEMPTS,
    STOPPED_NO_NEW_STRATEGY,
    STOPPED_SUFFICIENT,
    STRATEGY_BROADEN_TOP_K,
    STRATEGY_INITIAL,
    STRATEGY_REWRITE_QUERY,
    STRATEGY_SWITCH_MODE,
    heuristic_query_rewrite,
    run_agentic_retrieval,
)


def _retrieval(rows):
    return {"retrieval_mode": "hybrid", "details": {}, "results": rows}


def _row(chunk_index, text="row text"):
    return {
        "video_id": "vid1",
        "chunk_index": chunk_index,
        "text": text,
        "score": 0.9,
        "start": 0,
        "end": 10,
        "rank": 1,
    }


def _assessment(sufficient, reason_code="thin_support", confidence_cap="low"):
    return {
        "sufficient": sufficient,
        "reason_code": reason_code,
        "confidence_cap": confidence_cap,
        "warnings": [],
    }


def test_heuristic_query_rewrite_strips_english_question_phrasing():
    assert heuristic_query_rewrite("What is Jetson Orin used for?") == "jetson orin used"


def test_heuristic_query_rewrite_strips_japanese_question_phrasing():
    assert heuristic_query_rewrite("Jetson Orin とは何ですか？") == "Jetson Orin"


def test_heuristic_query_rewrite_returns_none_when_unchanged_or_empty():
    assert heuristic_query_rewrite("jetson orin") is None
    assert heuristic_query_rewrite("   ") is None
    assert heuristic_query_rewrite("what is the?") is None


def test_sufficient_first_attempt_stops_immediately():
    calls = []

    def retrieve_fn(*, query, retrieval_mode, k):
        calls.append((query, retrieval_mode, k))
        return _retrieval([_row(0)])

    outcome = run_agentic_retrieval(
        question="strong question",
        retrieve_fn=retrieve_fn,
        assess_fn=lambda *, rows, retrieval_mode: _assessment(True, "multi_chunk_support", "high"),
        rewrite_fn=lambda *, query, attempted_queries: "should not be called",
    )

    assert len(calls) == 1
    assert outcome["sufficient"] is True
    assert outcome["agentic_applied"] is False
    assert outcome["stopped_reason"] == STOPPED_SUFFICIENT
    assert outcome["attempts"][0]["strategy"] == STRATEGY_INITIAL


def test_thin_support_retries_with_rewritten_query():
    def retrieve_fn(*, query, retrieval_mode, k):
        if query == "rewritten query":
            return _retrieval([_row(1), _row(2)])
        return _retrieval([_row(0)])

    def assess_fn(*, rows, retrieval_mode):
        return _assessment(len(rows) == 2, "thin_support")

    outcome = run_agentic_retrieval(
        question="original question",
        retrieve_fn=retrieve_fn,
        assess_fn=assess_fn,
        rewrite_fn=lambda *, query, attempted_queries: "rewritten query",
    )

    assert outcome["sufficient"] is True
    assert outcome["agentic_applied"] is True
    assert outcome["final_query"] == "rewritten query"
    assert len(outcome["attempts"]) == 2
    assert outcome["attempts"][1]["strategy"] == STRATEGY_REWRITE_QUERY


def test_mixed_signals_switches_retrieval_mode_first():
    modes_seen = []

    def retrieve_fn(*, query, retrieval_mode, k):
        modes_seen.append(retrieval_mode)
        return _retrieval([_row(0)])

    def assess_fn(*, rows, retrieval_mode):
        return _assessment(retrieval_mode == "lexical", "mixed_signals")

    outcome = run_agentic_retrieval(
        question="question",
        retrieve_fn=retrieve_fn,
        assess_fn=assess_fn,
        rewrite_fn=lambda *, query, attempted_queries: "unused rewrite",
        retrieval_mode="hybrid",
    )

    assert modes_seen == ["hybrid", "lexical"]
    assert outcome["sufficient"] is True
    assert outcome["final_mode"] == "lexical"
    assert outcome["attempts"][1]["strategy"] == STRATEGY_SWITCH_MODE


def test_single_weak_chunk_broadens_top_k():
    ks_seen = []

    def retrieve_fn(*, query, retrieval_mode, k):
        ks_seen.append(k)
        return _retrieval([_row(0)])

    def assess_fn(*, rows, retrieval_mode):
        return _assessment(len(ks_seen) > 1, "single_weak_chunk")

    outcome = run_agentic_retrieval(
        question="question",
        retrieve_fn=retrieve_fn,
        assess_fn=assess_fn,
        k=5,
    )

    assert ks_seen == [5, 10]
    assert outcome["sufficient"] is True
    assert outcome["final_k"] == 10
    assert outcome["attempts"][1]["strategy"] == STRATEGY_BROADEN_TOP_K


def test_never_sufficient_falls_back_to_first_attempt():
    first_rows = [_row(0, "first attempt row")]

    def retrieve_fn(*, query, retrieval_mode, k):
        if query == "question":
            return _retrieval(first_rows)
        return _retrieval([_row(9, "later attempt row")])

    rewrites = iter(["second query", "third query"])

    outcome = run_agentic_retrieval(
        question="question",
        retrieve_fn=retrieve_fn,
        assess_fn=lambda *, rows, retrieval_mode: _assessment(False, "thin_support"),
        rewrite_fn=lambda *, query, attempted_queries: next(rewrites),
    )

    assert outcome["sufficient"] is False
    assert outcome["stopped_reason"] == STOPPED_MAX_ATTEMPTS
    assert len(outcome["attempts"]) == 3
    assert outcome["rows"] == first_rows
    assert outcome["final_query"] == "question"
    assert outcome["final_mode"] == "hybrid"


def test_stops_when_no_new_strategy_is_available():
    outcome = run_agentic_retrieval(
        question="question",
        retrieve_fn=lambda *, query, retrieval_mode, k: _retrieval([_row(0)]),
        assess_fn=lambda *, rows, retrieval_mode: _assessment(False, "no_results"),
        rewrite_fn=lambda *, query, attempted_queries: None,
        k=12,
        max_attempts=5,
    )

    # hybrid -> lexical -> dense, then no rewrite, no untried mode, k at max.
    assert outcome["stopped_reason"] == STOPPED_NO_NEW_STRATEGY
    assert len(outcome["attempts"]) == 3
    assert [attempt["retrieval_mode"] for attempt in outcome["attempts"]] == [
        "hybrid",
        "lexical",
        "dense",
    ]


def test_rewrite_matching_attempted_query_is_skipped():
    outcome = run_agentic_retrieval(
        question="question",
        retrieve_fn=lambda *, query, retrieval_mode, k: _retrieval([_row(0)]),
        assess_fn=lambda *, rows, retrieval_mode: _assessment(False, "thin_support"),
        rewrite_fn=lambda *, query, attempted_queries: "QUESTION",
        max_attempts=2,
    )

    # The rewrite normalizes to the already-attempted query, so the loop
    # falls back to a mode switch instead of retrying the same query.
    assert outcome["attempts"][1]["strategy"] == STRATEGY_SWITCH_MODE


def _make_service():
    service = LocalRAGService.__new__(LocalRAGService)
    service.feedback = {}
    service.feedback_index = {}
    service.feedback_lock = threading.Lock()
    service.feedback_tuning_enabled = False
    service._persist_feedback = lambda: None
    return service


def test_retrieve_agentic_uses_heuristic_rewrite_without_provider():
    service = _make_service()
    strong_rows = [
        {
            "video_id": "vid1",
            "chunk_index": 0,
            "text": "what is jetson orin used for in edge ai inference",
            "score": 0.9,
            "dense_score": 0.9,
            "lexical_score": 5.0,
            "start": 0,
            "end": 10,
            "rank": 1,
        },
        {
            "video_id": "vid1",
            "chunk_index": 1,
            "text": "jetson orin is used for robotics and edge ai",
            "score": 0.88,
            "dense_score": 0.88,
            "lexical_score": 5.0,
            "start": 60,
            "end": 70,
            "rank": 2,
        },
    ]
    retrieve_calls = []

    def fake_retrieve(query, **kwargs):
        retrieve_calls.append(query)
        rows = strong_rows if query == "jetson orin used" else []
        return {"retrieval_mode": kwargs.get("retrieval_mode", "hybrid"), "details": {}, "results": rows}

    def fail_llm(**kwargs):
        raise ValueError("no provider key configured")

    service.retrieve = fake_retrieve
    service._llm_text_response = fail_llm

    outcome = service.retrieve_agentic("What is Jetson Orin used for?", k=5)

    assert retrieve_calls == ["What is Jetson Orin used for?", "jetson orin used"]
    assert outcome["sufficient"] is True
    assert outcome["agentic_applied"] is True
    assert outcome["final_query"] == "jetson orin used"
    trace = outcome["retrieval"]["details"]["agentic_retrieval"]
    assert trace["enabled"] is True
    assert trace["applied"] is True
    assert trace["stopped_reason"] == STOPPED_SUFFICIENT
    assert [attempt["strategy"] for attempt in trace["attempts"]] == [
        STRATEGY_INITIAL,
        STRATEGY_REWRITE_QUERY,
    ]


def test_retrieve_agentic_prefers_llm_rewrite_when_available():
    service = _make_service()
    retrieve_calls = []

    def fake_retrieve(query, **kwargs):
        retrieve_calls.append(query)
        return {"retrieval_mode": "hybrid", "details": {}, "results": []}

    service.retrieve = fake_retrieve
    service._llm_text_response = lambda **kwargs: {
        "provider": "chatgpt",
        "model": "test-model",
        "text": '{"query": "llm rewritten query"}',
    }

    outcome = service.retrieve_agentic("What is Jetson Orin used for?", k=5)

    assert retrieve_calls[1] == "llm rewritten query"
    assert outcome["sufficient"] is False


def test_coerce_request_flag_parses_common_values():
    assert local_api._coerce_request_flag(True) is True
    assert local_api._coerce_request_flag("true") is True
    assert local_api._coerce_request_flag("1") is True
    assert local_api._coerce_request_flag("off") is False
    assert local_api._coerce_request_flag(None) is False
    assert local_api._coerce_request_flag(None, default=True) is True
    assert local_api._coerce_request_flag("garbage", default=True) is True


def test_env_flag_reads_environment(monkeypatch):
    monkeypatch.delenv(local_api.AGENTIC_RETRIEVAL_ENV, raising=False)
    assert local_api._env_flag(local_api.AGENTIC_RETRIEVAL_ENV) is False
    monkeypatch.setenv(local_api.AGENTIC_RETRIEVAL_ENV, "1")
    assert local_api._env_flag(local_api.AGENTIC_RETRIEVAL_ENV) is True
    monkeypatch.setenv(local_api.AGENTIC_RETRIEVAL_ENV, "off")
    assert local_api._env_flag(local_api.AGENTIC_RETRIEVAL_ENV) is False


class _FakeHandler:
    def __init__(self, path, body):
        self.path = path
        self._body = body
        self.response_status = None
        self.response_payload = None

    def _read_json_body(self):
        return self._body

    def _json(self, payload, status=200):
        self.response_status = status
        self.response_payload = payload


def test_ask_route_uses_agentic_retrieval_when_flag_set(monkeypatch):
    calls = {"agentic": 0, "plain": 0}

    class StubService:
        def retrieve_agentic(self, question, **kwargs):
            calls["agentic"] += 1
            assert kwargs.get("provider") == "chatgpt"
            return {
                "retrieval": {
                    "retrieval_mode": "hybrid",
                    "details": {"agentic_retrieval": {"applied": False}},
                    "results": [],
                },
                "final_mode": "hybrid",
            }

        def retrieve(self, question, **kwargs):
            calls["plain"] += 1
            return {"retrieval_mode": "hybrid", "details": {}, "results": []}

        def ask_with_sources(self, question, sources, **kwargs):
            return {
                "status": "insufficient_evidence",
                "answer": "n/a",
                "confidence": "low",
                "citations": [],
                "retrieved_chunks": [],
                "warnings": [],
                "sources": [],
                "provider": "chatgpt",
                "model": "test-model",
            }

    monkeypatch.delenv(local_api.AGENTIC_RETRIEVAL_ENV, raising=False)
    monkeypatch.setattr(local_api, "SERVICE", StubService())

    handler = _FakeHandler("/v1/ask", {"question": "q?", "agentic": True, "provider": "chatgpt"})
    local_api.Handler.do_POST(handler)
    assert handler.response_status == 200
    assert calls == {"agentic": 1, "plain": 0}
    assert "agentic_retrieval" in handler.response_payload["retrieval_details"]

    handler = _FakeHandler("/v1/ask", {"question": "q?", "provider": "chatgpt"})
    local_api.Handler.do_POST(handler)
    assert handler.response_status == 200
    assert calls == {"agentic": 1, "plain": 1}


def test_heuristic_query_rewrite_strips_longest_ja_phrase_first():
    assert heuristic_query_rewrite("これは何ですか") == "これは"


def test_never_sufficient_prefers_attempt_with_most_evidence():
    def retrieve_fn(*, query, retrieval_mode, k):
        if query == "question":
            return _retrieval([])
        return _retrieval([_row(5, "retry row a"), _row(6, "retry row b")])

    rewrites = iter(["second query", "third query"])

    outcome = run_agentic_retrieval(
        question="question",
        retrieve_fn=retrieve_fn,
        assess_fn=lambda *, rows, retrieval_mode: _assessment(False, "no_results" if not rows else "thin_support"),
        rewrite_fn=lambda *, query, attempted_queries: next(rewrites),
    )

    assert outcome["sufficient"] is False
    assert len(outcome["rows"]) == 2
    assert outcome["final_query"] == "second query"
