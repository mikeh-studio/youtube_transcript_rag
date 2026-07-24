"""Ask-layer tests for video-first routing and YouTube provenance."""

import importlib
import os
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
LOCAL_PREVIEW_DIR = ROOT_DIR / "local_preview"
if str(LOCAL_PREVIEW_DIR) not in sys.path:
    sys.path.insert(0, str(LOCAL_PREVIEW_DIR))

os.environ["YT_RAG_SKIP_GLOBAL_SERVICE"] = "1"
local_api = importlib.import_module("local_api")
grounded_answer = importlib.import_module("grounded_answer")


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


def _retrieval(video_ids, *, sufficient):
    rows = [
        {
            "video_id": video_id,
            "chunk_index": index,
            "text": f"evidence from {video_id}",
        }
        for index, video_id in enumerate(video_ids or [])
    ]
    return {
        "retrieval": {
            "retrieval_mode": "hybrid",
            "details": {},
            "results": rows,
        },
        "sufficient": sufficient,
    }


def test_video_first_agentic_expands_shortlist_before_global_fallback():
    service = local_api.LocalRAGService.__new__(local_api.LocalRAGService)
    service.engine = type(
        "Engine",
        (),
        {
            "library": type(
                "Library",
                (),
                {"videos": {video_id: {} for video_id in ("v1", "v2", "v3", "v4")}},
            )()
        },
    )()
    route_calls = []
    retrieval_scopes = []

    def route_videos(query, *, top_k, language, video_ids=None):
        del video_ids
        route_calls.append((query, top_k, language))
        video_ids = ["v1", "v2"] if top_k == 2 else ["v1", "v2", "v3", "v4"]
        return {
            "video_ids": video_ids,
            "results": [{"video_id": value} for value in video_ids],
            "used_fallback": False,
            "fallback_reason": None,
            "dense_available": True,
            "lexical_available": True,
            "fusion": "rrf",
            "latency_ms": 1.5,
        }

    def retrieve_agentic(question, **kwargs):
        scope = kwargs["video_ids"]
        retrieval_scopes.append(scope)
        return _retrieval(scope, sufficient=len(scope) == 4)

    service.route_videos = route_videos
    service.retrieve_agentic = retrieve_agentic

    outcome = service.retrieve_video_first(
        "starter timing",
        video_top_k=2,
        agentic=True,
    )
    details = outcome["retrieval"]["details"]["video_routing"]
    assert [call[1] for call in route_calls] == [2, 4]
    assert retrieval_scopes == [["v1", "v2"], ["v1", "v2", "v3", "v4"]]
    assert [stage["scope"] for stage in details["stages"]] == [
        "routed_top_k",
        "routed_expanded",
    ]
    assert details["selected_video_ids"] == ["v1", "v2"]
    assert details["used_fallback"] is False


def test_video_first_uses_global_retrieval_when_router_is_unavailable():
    service = local_api.LocalRAGService.__new__(local_api.LocalRAGService)
    service.engine = type(
        "Engine",
        (),
        {"library": type("Library", (), {"videos": {"v1": {}, "v2": {}}})()},
    )()
    service.route_videos = lambda *args, **kwargs: {
        "video_ids": [],
        "results": [],
        "used_fallback": True,
        "fallback_reason": "router_unavailable",
        "dense_available": False,
        "lexical_available": False,
        "fusion": "rrf",
        "latency_ms": 0.5,
    }
    scopes = []

    def retrieve(question, **kwargs):
        scopes.append(kwargs["video_ids"])
        return {
            "retrieval_mode": "hybrid",
            "details": {},
            "results": [],
        }

    service.retrieve = retrieve
    outcome = service.retrieve_video_first("question", agentic=False)
    details = outcome["retrieval"]["details"]["video_routing"]

    assert scopes == [None]
    assert details["used_fallback"] is True
    assert details["fallback_reason"] == "router_unavailable"
    assert details["stages"][0]["scope"] == "global_fallback"


def test_english_frieren_query_routes_to_japanese_video_after_translation():
    service = local_api.LocalRAGService.__new__(local_api.LocalRAGService)
    service.engine = type(
        "Engine",
        (),
        {
            "library": type(
                "Library",
                (),
                {
                    "videos": {
                        "frieren1234": {"language": "ja"},
                        "nba12345678": {"language": "en"},
                    }
                },
            )()
        },
    )()
    service._llm_text_response = lambda **kwargs: {
        "provider": "chatgpt",
        "model": "test-model",
        "text": (
            '{"translations":{"ja":'
            '"フリーレンのポッドキャストについて詳しく教えて。何が話されましたか"}}'
        ),
    }

    def route_videos(query, *, top_k, language, video_ids=None):
        del query, top_k
        if language == "ja":
            video_id = "frieren1234"
            lexical_score = 0.7
        else:
            video_id = "nba12345678"
            lexical_score = None
        assert video_id in video_ids
        return {
            "video_ids": [video_id],
            "results": [
                {
                    "video_id": video_id,
                    "title": video_id,
                    "language": language,
                    "source": {"platform": "youtube"},
                    "dense_score": 0.5,
                    "lexical_score": lexical_score,
                }
            ],
            "used_fallback": False,
            "fallback_reason": None,
            "latency_ms": 1.0,
        }

    retrieval_calls = []

    def retrieve(query, **kwargs):
        retrieval_calls.append((query, kwargs["language"], kwargs["video_ids"]))
        return {
            "retrieval_mode": "hybrid",
            "details": {},
            "results": [
                {
                    "video_id": kwargs["video_ids"][0],
                    "chunk_index": 0,
                    "text": "フリーレンの番組で出演者が作品について話しました",
                }
            ],
        }

    service.route_videos = route_videos
    service.retrieve = retrieve
    question = "Tell me more about the Frieren podcasts? What was discussed"

    outcome = service.retrieve_video_first(
        question,
        video_top_k=1,
        provider="chatgpt",
    )
    details = outcome["retrieval"]["details"]["video_routing"]
    grounding = outcome["retrieval"]["details"]["answer_grounding"]

    assert details["selected_video_ids"] == ["frieren1234"]
    assert details["query_expansion"]["translation_applied"] is True
    assert details["query_expansion"]["variants"][1]["language"] == "ja"
    assert grounding == {
        "question": "フリーレンのポッドキャストについて詳しく教えて。何が話されましたか",
        "query_language": "ja",
        "answer_language": "en",
    }
    assert retrieval_calls == [
        (
            "フリーレンのポッドキャストについて詳しく教えて。何が話されましたか",
            "ja",
            ["frieren1234"],
        )
    ]


def test_ask_route_uses_video_first_only_for_all_videos(monkeypatch):
    calls = []

    class StubService:
        def retrieve_video_first(self, question, **kwargs):
            calls.append((question, kwargs))
            return {
                "retrieval": {
                    "retrieval_mode": "hybrid",
                    "details": {
                        "video_routing": {
                            "enabled": True,
                            "selected_video_ids": ["v1", "v2"],
                        }
                    },
                    "results": [],
                },
                "agentic_outcome": None,
            }

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

    monkeypatch.setattr(local_api, "SERVICE", StubService())
    handler = _FakeHandler(
        "/v1/ask",
        {
            "question": "Which video explains starters?",
            "video_routing": "multi_vector",
            "video_top_k": 4,
            "agentic": True,
            "provider": "chatgpt",
        },
    )

    local_api.Handler.do_POST(handler)

    assert handler.response_status == 200
    assert len(calls) == 1
    assert calls[0][1]["video_top_k"] == 4
    assert calls[0][1]["agentic"] is True
    assert calls[0][1]["provider"] == "chatgpt"
    assert handler.response_payload["retrieval_details"]["video_routing"]["enabled"]


def test_citations_keep_youtube_source_separate_from_source_type():
    source = {
        "platform": "youtube",
        "video_id": "video123456",
        "url": "https://www.youtube.com/watch?v=video123456",
        "channel": {
            "id": "UC-source",
            "name": "Source Channel",
            "url": "https://www.youtube.com/channel/UC-source",
        },
        "metadata_provider": "youtube_data_api",
        "fetched_at": "2026-07-23T00:00:00+00:00",
    }
    rows = [
        {
            "source_type": "transcript",
            "source": source,
            "video_id": "video123456",
            "video_title": "Starter Guide",
            "chunk_index": 2,
            "text": "Feed the starter after peak fermentation.",
            "start": 65.0,
            "end": 80.0,
        }
    ]

    citation = grounded_answer.build_citation_catalog(rows)[0]
    chunk = grounded_answer.build_retrieved_chunks_payload(rows)[0]
    _system, prompt = grounded_answer.build_grounded_answer_messages(
        question="When should I feed it?",
        citations=[citation],
        answer_language="en",
    )

    assert citation["source_type"] == "transcript"
    assert citation["source"]["platform"] == "youtube"
    assert citation["source"]["channel"]["id"] == "UC-source"
    assert chunk["source"]["channel"]["name"] == "Source Channel"
    assert "YouTube channel: Source Channel" in prompt
