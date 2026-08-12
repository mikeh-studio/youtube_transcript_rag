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
    STRATEGY_READ_CONTEXT,
    STRATEGY_SWITCH_MODE,
    TOOL_KEYWORD_SEARCH,
    TOOL_SEMANTIC_SEARCH,
    choose_initial_search_tool,
    heuristic_query_rewrite,
    run_agentic_retrieval,
    run_agentic_tool_search,
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


def test_agentic_tool_policy_prefers_keyword_for_japanese_and_exact_queries():
    assert choose_initial_search_tool("代理店施策の勝ち筋") == TOOL_KEYWORD_SEARCH
    assert choose_initial_search_tool('find "Jetson Orin"') == TOOL_KEYWORD_SEARCH
    assert choose_initial_search_tool("What supports robot inference?") == TOOL_SEMANTIC_SEARCH


def test_agentic_tool_search_reads_raw_context_around_strong_anchor():
    calls = []

    def semantic_search(*, query, k):
        calls.append((TOOL_SEMANTIC_SEARCH, query))
        return {
            "retrieval_mode": "dense",
            "details": {},
            "results": [_row(2, "robot inference anchor")],
        }

    def keyword_search(*, query, k):
        raise AssertionError("sufficient semantic evidence should not switch search tools")

    def read_context(*, video_id, timestamp, window):
        calls.append(("read_context", video_id))
        return {
            "start": 0.0,
            "end": 20.0,
            "text": "before\nrobot inference anchor\nafter",
            "segments": [
                {"start": 0.0, "end": 5.0, "text": "before"},
                {"start": 5.0, "end": 10.0, "text": "robot inference anchor"},
                {"start": 10.0, "end": 20.0, "text": "after"},
            ],
            "segment_count": 3,
            "source_basis": "full_transcript.segments",
        }

    outcome = run_agentic_tool_search(
        question="What supports robot inference?",
        semantic_search_fn=semantic_search,
        keyword_search_fn=keyword_search,
        read_context_fn=read_context,
        assess_fn=lambda *, rows, retrieval_mode: _assessment(
            True, "multi_chunk_support", "high"
        ),
        k=3,
    )

    assert calls == [(TOOL_SEMANTIC_SEARCH, "What supports robot inference?"), ("read_context", "vid1")]
    assert outcome["final_tool"] == "read_context"
    assert outcome["rows"][0]["chunk_index"] == 2
    assert outcome["rows"][0]["text"].startswith("before")
    assert outcome["rows"][0]["source_basis"] == "full_transcript.segments"
    assert outcome["attempts"][-1]["strategy"] == STRATEGY_READ_CONTEXT


def test_agentic_tool_search_switches_to_semantic_after_weak_japanese_keyword_hit():
    calls = []

    def keyword_search(*, query, k):
        calls.append((TOOL_KEYWORD_SEARCH, query))
        return {"retrieval_mode": "lexical", "details": {}, "results": []}

    def semantic_search(*, query, k):
        calls.append((TOOL_SEMANTIC_SEARCH, query))
        return {
            "retrieval_mode": "dense",
            "details": {},
            "results": [_row(3, "semantic hit"), _row(4, "supporting hit")],
        }

    outcome = run_agentic_tool_search(
        question="これは何ですか",
        keyword_search_fn=keyword_search,
        semantic_search_fn=semantic_search,
        read_context_fn=lambda **kwargs: (_ for _ in ()).throw(ValueError("legacy")),
        assess_fn=lambda *, rows, retrieval_mode: _assessment(
            len(rows) >= 2, "no_results" if not rows else "multi_chunk_support"
        ),
        rewrite_fn=lambda *, query, attempted_queries: "これは",
    )

    assert calls == [
        (TOOL_KEYWORD_SEARCH, "これは何ですか"),
        (TOOL_SEMANTIC_SEARCH, "これは"),
    ]
    assert outcome["sufficient"] is True
    assert outcome["final_mode"] == "dense"
    assert outcome["attempts"][1]["strategy"] == STRATEGY_REWRITE_QUERY


def test_agentic_context_expansion_preserves_legacy_and_lower_ranked_results():
    rows = [_row(index, f"row {index}") for index in range(5)]
    for index, row in enumerate(rows):
        row["start"] = index * 10

    def read_context(*, video_id, timestamp, window):
        if timestamp == 0:
            return {
                "start": 0,
                "end": 20,
                "text": "expanded row 0",
                "segments": [{"start": 0, "end": 20, "text": "expanded row 0"}],
                "segment_count": 1,
                "source_basis": "full_transcript.segments",
            }
        raise ValueError("legacy transcript")

    outcome = run_agentic_tool_search(
        question="semantic question",
        semantic_search_fn=lambda **kwargs: {
            "retrieval_mode": "dense",
            "details": {},
            "results": rows,
        },
        keyword_search_fn=lambda **kwargs: {"results": []},
        read_context_fn=read_context,
        assess_fn=lambda *, rows, retrieval_mode: _assessment(True),
        k=5,
    )

    assert len(outcome["rows"]) == 5
    assert outcome["rows"][0]["text"] == "expanded row 0"
    assert [row["chunk_index"] for row in outcome["rows"]] == [0, 1, 2, 3, 4]
    assert len(outcome["attempts"][-1]["calls"]) == 1


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


def test_retrieve_agentic_uses_deterministic_rewrite_without_llm_call():
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
        raise AssertionError("deterministic retrieval rewriting must not call an LLM")

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
    assert trace["rewrite_method"] == "deterministic_heuristic"
    assert [attempt["strategy"] for attempt in trace["attempts"]] == [
        STRATEGY_INITIAL,
        STRATEGY_REWRITE_QUERY,
    ]


def test_retrieve_agentic_does_not_use_available_llm_for_rewrite():
    service = _make_service()
    retrieve_calls = []
    llm_calls = []

    def fake_retrieve(query, **kwargs):
        retrieve_calls.append(query)
        return {"retrieval_mode": "hybrid", "details": {}, "results": []}

    service.retrieve = fake_retrieve
    service._llm_text_response = lambda **kwargs: llm_calls.append(kwargs)

    outcome = service.retrieve_agentic("What is Jetson Orin used for?", k=5)

    assert retrieve_calls[1] == "jetson orin used"
    assert llm_calls == []
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
            assert "provider" not in kwargs
            assert "model" not in kwargs
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


def test_search_route_uses_agentic_retrieval_only_when_requested(monkeypatch):
    calls = {"agentic": 0, "plain": 0}

    class StubService:
        def retrieve_agentic(self, question, **kwargs):
            calls["agentic"] += 1
            return {
                "retrieval": {
                    "retrieval_mode": "lexical",
                    "details": {
                        "agentic_retrieval": {
                            "final_tool": "read_context",
                            "applied": True,
                        }
                    },
                    "results": [_row(1)],
                }
            }

        def retrieve(self, question, **kwargs):
            calls["plain"] += 1
            return {"retrieval_mode": "hybrid", "details": {}, "results": []}

    monkeypatch.delenv(local_api.AGENTIC_RETRIEVAL_ENV, raising=False)
    monkeypatch.setattr(local_api, "SERVICE", StubService())

    handler = _FakeHandler("/v1/search", {"query": "検索", "agentic": True})
    local_api.Handler.do_POST(handler)
    assert handler.response_status == 200
    assert handler.response_payload["agentic"] is True
    assert handler.response_payload["retrieval_mode"] == "lexical"
    assert calls == {"agentic": 1, "plain": 0}

    handler = _FakeHandler("/v1/search", {"query": "検索"})
    local_api.Handler.do_POST(handler)
    assert handler.response_status == 200
    assert handler.response_payload["agentic"] is False
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
