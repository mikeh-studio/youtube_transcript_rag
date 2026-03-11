"""Tests for local preview summary generation and retrieval upgrades."""

import importlib
import os
import sys
import tempfile
import threading
from pathlib import Path

import pytest


ROOT_DIR = Path(__file__).resolve().parents[1]
LOCAL_PREVIEW_DIR = ROOT_DIR / "local_preview"
if str(LOCAL_PREVIEW_DIR) not in sys.path:
    sys.path.insert(0, str(LOCAL_PREVIEW_DIR))

os.environ["YT_RAG_SKIP_GLOBAL_SERVICE"] = "1"
local_api = importlib.import_module("local_api")
LocalRAGService = local_api.LocalRAGService


def _make_service(enabled=True):
    service = LocalRAGService.__new__(LocalRAGService)
    service.feedback = {}
    service.feedback_index = {}
    service.feedback_lock = threading.Lock()
    service.feedback_tuning_enabled = bool(enabled)
    service._persist_feedback = lambda: None
    runtime_dir = Path(tempfile.mkdtemp())
    cache_dir = Path(tempfile.mkdtemp())
    legacy_dir = Path(tempfile.mkdtemp())
    service.runtime_data_dir = runtime_dir
    service.cache_data_dir = cache_dir
    service.summary_cache_dir = cache_dir / "summaries"
    service.legacy_data_dir = legacy_dir
    service.feedback_path = runtime_dir / "search_feedback.json"
    service.legacy_feedback_path = legacy_dir / "search_feedback.json"
    service.openai_model = "gpt-4o-mini"
    service.ask_history_lock = threading.Lock()
    service.ask_history = {}
    service.ask_history_path = runtime_dir / "ask_history.json"
    service.legacy_ask_history_path = legacy_dir / "ask_history.json"
    service.ingest_log_path = runtime_dir / "ingest_jobs.log"
    service.legacy_ingest_log_path = legacy_dir / "ingest_jobs.log"
    service.log_lock = threading.Lock()
    return service


def test_allowed_local_origin_accepts_only_loopback_hosts():
    assert (
        local_api._allowed_local_origin("http://127.0.0.1:8000")
        == "http://127.0.0.1:8000"
    )
    assert (
        local_api._allowed_local_origin("https://localhost:3000")
        == "https://localhost:3000"
    )
    assert local_api._allowed_local_origin("https://example.com") is None
    assert local_api._allowed_local_origin("file:///tmp/index.html") is None


def test_persist_ask_history_writes_private_file_mode():
    service = _make_service(enabled=False)
    service.ask_history = {"vid1": [{"question": "q1"}]}

    LocalRAGService._persist_ask_history(service)

    assert service.ask_history_path.exists()
    assert (service.ask_history_path.stat().st_mode & 0o777) == 0o600


def test_persist_feedback_writes_private_file_mode():
    service = _make_service(enabled=False)
    service.feedback = {"vid1:0": {"video_id": "vid1", "label": "relevant"}}

    LocalRAGService._persist_feedback(service)

    assert service.feedback_path.exists()
    assert (service.feedback_path.stat().st_mode & 0o777) == 0o600


def test_list_videos_includes_chunking_metadata():
    class DummyLibrary:
        def __init__(self):
            self.videos = {
                "vid1": {
                    "title": "Video vid1",
                    "url": "https://www.youtube.com/watch?v=vid1",
                    "language": "en",
                    "chunks": [{"raw_text": "hello", "start": 0.0, "end": 8.0}],
                }
            }

        def get_video_chunking_metadata(self, video_id):
            assert video_id == "vid1"
            return {"version": "time_v1_45s_15s"}

        def video_chunking_is_stale(self, video_id):
            assert video_id == "vid1"
            return True

    class DummyEngine:
        def __init__(self):
            self.library = DummyLibrary()

    service = _make_service(enabled=False)
    service.engine = DummyEngine()
    service.title_cache = {}
    service._hydrate_video_title = lambda video_id: None

    videos = service.list_videos()

    assert videos == [
        {
            "video_id": "vid1",
            "title": "Video vid1",
            "url": "https://www.youtube.com/watch?v=vid1",
            "language": "en",
            "num_chunks": 1,
            "chunking_version": "time_v1_45s_15s",
            "chunking_stale": True,
            "ask_history_count": 0,
            "last_ask_at": None,
        }
    ]


def test_retrieve_rerank_happens_before_topk_cut():
    service = _make_service(enabled=True)

    service._dense_search = lambda query, k, language, video_id=None: [
        {
            "video_id": "vid1",
            "video_title": "Video 1",
            "start": float(i * 10),
            "end": float(i * 10 + 9),
            "chunk_index": i,
            "text": f"chunk {i} python list sort",
            "score": 0.70 - (0.01 * i),
            "rank": i + 1,
        }
        for i in range(10)
    ]
    service._lexical_bm25_search = lambda query, k, language, video_id=None: []

    service.save_search_feedback(
        {
            "query": "python list sort",
            "retrieval_mode": "dense",
            "label": "relevant",
            "video_id": "vid1",
            "chunk_index": 9,
            "start": 90.0,
            "end": 99.0,
        }
    )

    result = service.retrieve("python list sort", k=3, retrieval_mode="dense")
    chunk_ids = [row["chunk_index"] for row in result["results"]]

    assert 9 in chunk_ids
    assert result["details"]["pre_rerank_candidate_count"] == 10
    assert result["details"]["post_feedback_candidate_count"] == 10


def test_retrieve_applies_diversity_to_mix_videos():
    service = _make_service(enabled=False)

    dense_rows = [
        {
            "video_id": "vidA",
            "video_title": "Video A",
            "start": 0.0,
            "end": 10.0,
            "chunk_index": 0,
            "text": "python lists",
            "score": 0.95,
            "rank": 1,
        },
        {
            "video_id": "vidA",
            "video_title": "Video A",
            "start": 50.0,
            "end": 60.0,
            "chunk_index": 1,
            "text": "python tuples",
            "score": 0.94,
            "rank": 2,
        },
        {
            "video_id": "vidA",
            "video_title": "Video A",
            "start": 120.0,
            "end": 130.0,
            "chunk_index": 2,
            "text": "python dicts",
            "score": 0.93,
            "rank": 3,
        },
        {
            "video_id": "vidB",
            "video_title": "Video B",
            "start": 25.0,
            "end": 35.0,
            "chunk_index": 0,
            "text": "python generators",
            "score": 0.70,
            "rank": 4,
        },
    ]
    service._dense_search = lambda query, k, language, video_id=None: dense_rows
    service._lexical_bm25_search = lambda query, k, language, video_id=None: []

    result = service.retrieve("python", k=3, retrieval_mode="dense")
    video_ids = [row["video_id"] for row in result["results"]]

    assert "vidB" in video_ids
    assert result["details"]["diversity_applied"] is True
    assert result["details"]["selected_per_video_cap"] == 2


def test_retrieve_passes_video_id_filter_to_search_backends():
    service = _make_service(enabled=False)
    calls = {"dense": [], "lexical": []}

    def fake_dense(query, k, language, video_id=None):
        calls["dense"].append(
            {
                "query": query,
                "k": k,
                "language": language,
                "video_id": video_id,
            }
        )
        return [
            {
                "video_id": "vid1",
                "video_title": "Video 1",
                "start": 0.0,
                "end": 10.0,
                "chunk_index": 0,
                "text": "python lists",
                "score": 0.8,
                "rank": 1,
            }
        ]

    def fake_lexical(query, k, language, video_id=None):
        calls["lexical"].append(
            {
                "query": query,
                "k": k,
                "language": language,
                "video_id": video_id,
            }
        )
        return []

    service._dense_search = fake_dense
    service._lexical_bm25_search = fake_lexical
    service.engine = type(
        "DummyEngine",
        (),
        {"library": type("DummyLibrary", (), {"videos": {"vid1": {}}})()},
    )()

    result = service.retrieve(
        "python",
        k=2,
        retrieval_mode="dense",
        video_id="vid1",
    )

    assert result["details"]["video_id_filter"] == "vid1"
    assert calls["dense"][0]["video_id"] == "vid1"
    assert calls["lexical"] == []


def test_save_ask_history_persists_and_lists_records():
    class DummyLibrary:
        def __init__(self):
            self.videos = {
                "vid1": {
                    "title": "Video vid1",
                    "url": "https://www.youtube.com/watch?v=vid1",
                    "language": "en",
                }
            }

        def get_video_chunking_metadata(self, video_id):
            assert video_id == "vid1"
            return {"version": "time_v2_60s_15s"}

    service = _make_service(enabled=False)
    service.engine = type("DummyEngine", (), {"library": DummyLibrary()})()

    record = service.save_ask_history(
        {
            "video_id": "vid1",
            "question": "What is this video about?",
            "k": 5,
            "language": "en",
            "retrieval_mode": "hybrid",
            "provider": "chatgpt",
            "model": "gpt-4o-mini",
            "answer": "It explains Python basics.",
            "sources": [
                {
                    "video_id": "vid1",
                    "video_title": "Video vid1",
                    "video_url": "https://www.youtube.com/watch?v=vid1",
                    "language": "en",
                    "chunk_index": 0,
                    "text": "Python basics",
                    "start": 0.0,
                    "end": 30.0,
                    "url": "https://www.youtube.com/watch?v=vid1&t=0s",
                }
            ],
            "retrieval_details": {"fusion": "rrf"},
        }
    )

    listed = service.list_ask_history(video_id="vid1", limit=5)

    assert record["video_id"] == "vid1"
    assert record["chunking_version"] == "time_v2_60s_15s"
    assert record["source_fingerprint"]
    assert listed[0]["question"] == "What is this video about?"
    assert listed[0]["answer"] == "It explains Python basics."
    assert listed[0]["retrieval_details"]["fusion"] == "rrf"


def test_save_ask_history_trims_to_recent_limit():
    class DummyLibrary:
        def __init__(self):
            self.videos = {
                "vid1": {
                    "title": "Video vid1",
                    "url": "https://www.youtube.com/watch?v=vid1",
                    "language": "en",
                }
            }

        def get_video_chunking_metadata(self, video_id):
            assert video_id == "vid1"
            return {"version": "time_v2_60s_15s"}

    service = _make_service(enabled=False)
    service.engine = type("DummyEngine", (), {"library": DummyLibrary()})()

    for idx in range(25):
        service.save_ask_history(
            {
                "video_id": "vid1",
                "question": f"Question {idx}",
                "k": 5,
                "language": "en",
                "retrieval_mode": "hybrid",
                "provider": "chatgpt",
                "model": "gpt-4o-mini",
                "answer": f"Answer {idx}",
                "sources": [],
                "created_at": f"2026-01-{idx + 1:02d}T00:00:00+00:00",
            }
        )

    rows = service.list_ask_history(video_id="vid1", limit=50)

    assert len(rows) == local_api.ASK_HISTORY_LIMIT_PER_VIDEO
    assert rows[0]["question"] == "Question 24"
    assert rows[-1]["question"] == "Question 5"


def test_summarize_video_transcript_returns_five_ranked_items():
    class DummyLibrary:
        def __init__(self):
            self.videos = {
                "vid1": {
                    "title": "Demo Video",
                    "full_transcript": {
                        "segments": [
                            {"text": "Intro", "start": 0.0, "end": 20.0},
                            {"text": "Main topic", "start": 20.0, "end": 60.0},
                            {"text": "Details", "start": 60.0, "end": 120.0},
                        ],
                        "text": "Intro\nMain topic\nDetails",
                    },
                    "chunks": [
                        {"raw_text": "Intro", "start": 0.0, "end": 20.0},
                        {"raw_text": "Main topic", "start": 20.0, "end": 60.0},
                        {"raw_text": "Details", "start": 60.0, "end": 120.0},
                    ],
                }
            }

    class DummyEngine:
        def __init__(self):
            self.library = DummyLibrary()
            self.model = "claude-sonnet-4-5-20250929"

    service = _make_service(enabled=False)
    service.engine = DummyEngine()
    service._summarize_transcript_single_pass = lambda **kwargs: {
        "provider": "chatgpt",
        "model": "gpt-4o-mini",
        "items": [
            {
                "title": "Point 1",
                "tldr": "A. B. C. D.",
                "anchor_text": "Intro",
                "start": 0.0,
                "end": 10.0,
            },
            {
                "title": "Point 2",
                "tldr": "A. B. C. D.",
                "anchor_text": "Main topic",
                "start": 15.0,
                "end": 25.0,
            },
            {
                "title": "Point 3",
                "tldr": "A. B. C. D.",
                "anchor_text": "Details",
                "start": 30.0,
                "end": 40.0,
            },
            {
                "title": "Point 4",
                "tldr": "A. B. C. D.",
                "anchor_text": "Intro",
                "start": 45.0,
                "end": 55.0,
            },
            {
                "title": "Point 5",
                "tldr": "A. B. C. D.",
                "anchor_text": "Main topic",
                "start": 60.0,
                "end": 70.0,
            },
        ],
        "strategy": "single_pass",
    }

    response = service.summarize_video_transcript(
        video_id="vid1",
        language="en",
        provider="chatgpt",
        max_points=5,
    )

    assert response["video_id"] == "vid1"
    assert response["language"] == "en"
    assert response["provider"] == "chatgpt"
    assert response["model"] == "gpt-4o-mini"
    assert len(response["summary"]) == 5
    assert response["summary"][0]["rank"] == 1
    assert response["summary"][4]["rank"] == 5
    assert response["generation_details"]["strategy"] == "single_pass"
    assert "youtube.com/watch?v=vid1&t=" in response["summary"][0]["url"]


def test_summarize_video_transcript_rejects_ten_points():
    class DummyLibrary:
        def __init__(self):
            self.videos = {
                "vid1": {
                    "title": "Demo Video",
                    "full_transcript": {
                        "segments": [
                            {"text": "Intro", "start": 0.0, "end": 20.0},
                            {"text": "Main topic", "start": 20.0, "end": 60.0},
                        ],
                        "text": "Intro\nMain topic",
                    },
                    "chunks": [
                        {"raw_text": "Intro", "start": 0.0, "end": 20.0},
                        {"raw_text": "Main topic", "start": 20.0, "end": 60.0},
                    ],
                }
            }

    class DummyEngine:
        def __init__(self):
            self.library = DummyLibrary()
            self.model = "claude-sonnet-4-5-20250929"

    service = _make_service(enabled=False)
    service.engine = DummyEngine()
    with pytest.raises(ValueError, match="max_points must be 5"):
        service.summarize_video_transcript(
            video_id="vid1",
            language="en",
            provider="chatgpt",
            max_points=10,
        )


def test_summarize_video_transcript_rejects_invalid_max_points():
    class DummyLibrary:
        def __init__(self):
            self.videos = {
                "vid1": {
                    "title": "Demo Video",
                    "full_transcript": {
                        "segments": [
                            {"text": "Intro", "start": 0.0, "end": 10.0},
                        ],
                        "text": "Intro",
                    },
                    "chunks": [
                        {"raw_text": "Intro", "start": 0.0, "end": 10.0},
                    ],
                }
            }

    class DummyEngine:
        def __init__(self):
            self.library = DummyLibrary()
            self.model = "claude-sonnet-4-5-20250929"

    service = _make_service(enabled=False)
    service.engine = DummyEngine()

    with pytest.raises(ValueError, match="max_points must be 5"):
        service.summarize_video_transcript(
            video_id="vid1",
            language="en",
            provider="chatgpt",
            max_points=7,
        )


def test_summarize_video_transcript_rejects_invalid_language():
    class DummyLibrary:
        def __init__(self):
            self.videos = {
                "vid1": {
                    "title": "Demo Video",
                    "full_transcript": {
                        "segments": [
                            {"text": "Hello", "start": 0.0, "end": 10.0},
                        ],
                        "text": "Hello",
                    },
                    "chunks": [
                        {"raw_text": "Hello", "start": 0.0, "end": 10.0},
                    ],
                }
            }

    class DummyEngine:
        def __init__(self):
            self.library = DummyLibrary()
            self.model = "claude-sonnet-4-5-20250929"

    service = _make_service(enabled=False)
    service.engine = DummyEngine()

    with pytest.raises(ValueError, match="language must be one of: en, ja"):
        service.summarize_video_transcript(
            video_id="vid1",
            language="fr",
            provider="chatgpt",
            max_points=5,
        )


def test_summarize_long_transcript_uses_compact_single_pass():
    class DummyLibrary:
        def __init__(self):
            self.videos = {
                "vid1": {
                    "title": "Long Video",
                    "full_transcript": {
                        "segments": [
                            {
                                "text": ("long transcript block " * 40).strip(),
                                "start": float(idx * 10),
                                "end": float(idx * 10 + 9),
                            }
                            for idx in range(120)
                        ],
                        "text": "\n".join(
                            ("long transcript block " * 40).strip() for _ in range(120)
                        ),
                    },
                    "chunks": [
                        {
                            "raw_text": ("long transcript block " * 40).strip(),
                            "start": float(idx * 10),
                            "end": float(idx * 10 + 9),
                        }
                        for idx in range(120)
                    ],
                }
            }

    class DummyEngine:
        def __init__(self):
            self.library = DummyLibrary()
            self.model = "claude-sonnet-4-5-20250929"

    service = _make_service(enabled=False)
    service.engine = DummyEngine()

    called = {"compact": False}

    def fake_compact_single_pass(**kwargs):
        called["compact"] = True
        return {
            "provider": "chatgpt",
            "model": "gpt-4o-mini",
            "items": [
                {
                    "title": "Theme 1",
                    "tldr": "A. B. C. D.",
                    "anchor_text": "long transcript block",
                    "start": 0.0,
                    "end": 0.0,
                },
                {
                    "title": "Theme 2",
                    "tldr": "A. B. C. D.",
                    "anchor_text": "long transcript block",
                    "start": 20.0,
                    "end": 20.0,
                },
                {
                    "title": "Theme 3",
                    "tldr": "A. B. C. D.",
                    "anchor_text": "long transcript block",
                    "start": 40.0,
                    "end": 40.0,
                },
                {
                    "title": "Theme 4",
                    "tldr": "A. B. C. D.",
                    "anchor_text": "long transcript block",
                    "start": 60.0,
                    "end": 60.0,
                },
                {
                    "title": "Theme 5",
                    "tldr": "A. B. C. D.",
                    "anchor_text": "long transcript block",
                    "start": 80.0,
                    "end": 80.0,
                },
            ],
            "strategy": "compact_single_pass",
            "total_windows": 1,
            "processed_windows": 1,
            "retry_count": 0,
            "attempt_count": 1,
        }

    service._summarize_transcript_compact_single_pass = fake_compact_single_pass
    response = service.summarize_video_transcript(
        video_id="vid1",
        language="en",
        provider="chatgpt",
        max_points=5,
    )

    assert called["compact"] is True
    assert response["generation_details"]["strategy"] == "compact_single_pass"
    assert len(response["summary"]) == 5


def test_summarize_long_transcript_relaxes_after_compact_and_reduce_failures():
    class DummyLibrary:
        def __init__(self):
            self.videos = {
                "vid1": {
                    "title": "Long Video",
                    "full_transcript": {
                        "segments": [
                            {
                                "text": ("long transcript block " * 40).strip(),
                                "start": float(idx * 10),
                                "end": float(idx * 10 + 9),
                            }
                            for idx in range(120)
                        ],
                        "text": "\n".join(
                            ("long transcript block " * 40).strip() for _ in range(120)
                        ),
                    },
                    "chunks": [
                        {
                            "raw_text": ("long transcript block " * 40).strip(),
                            "start": float(idx * 10),
                            "end": float(idx * 10 + 9),
                        }
                        for idx in range(120)
                    ],
                }
            }

    class DummyEngine:
        def __init__(self):
            self.library = DummyLibrary()
            self.model = "claude-sonnet-4-5-20250929"

    service = _make_service(enabled=False)
    service.engine = DummyEngine()
    call_counts = {"compact": 0, "reduce": 0}

    def fake_compact_single_pass(**kwargs):
        del kwargs
        call_counts["compact"] += 1
        raise local_api.SummaryGenerationError(
            "compact single-pass theme generation failed after retries"
        )

    def fake_map_reduce(**kwargs):
        call_counts["reduce"] += 1
        min_sentences = kwargs.get("min_sentences", local_api.SUMMARY_MIN_SENTENCES)
        max_sentences = kwargs.get("max_sentences", local_api.SUMMARY_MAX_SENTENCES)

        if (
            min_sentences == local_api.SUMMARY_MIN_SENTENCES
            and max_sentences == local_api.SUMMARY_MAX_SENTENCES
        ):
            raise local_api.SummaryGenerationError(
                "reduce stage summary generation failed after retries"
            )

        return {
            "provider": "chatgpt",
            "model": "gpt-4o-mini",
            "items": [
                {
                    "title": f"Theme {idx}",
                    "tldr": "One. Two. Three.",
                    "anchor_text": "long transcript block",
                    "start": float((idx - 1) * 20),
                    "end": float((idx - 1) * 20 + 10),
                }
                for idx in range(1, 6)
            ],
            "strategy": "map_reduce",
            "total_windows": 3,
            "processed_windows": 3,
            "retry_count": 2,
        }

    service._summarize_transcript_compact_single_pass = fake_compact_single_pass
    service._summarize_transcript_map_reduce = fake_map_reduce

    response = service.summarize_video_transcript(
        video_id="vid1",
        language="en",
        provider="chatgpt",
        max_points=5,
    )

    assert call_counts["compact"] == 1
    assert call_counts["reduce"] == 2
    assert len(response["summary"]) == 5
    assert response["generation_details"]["primary_strategy"] == "compact_single_pass"
    assert response["generation_details"]["strategy"] == "map_reduce"
    assert response["generation_details"]["fallback_applied"] is True
    assert response["generation_details"]["validation_relaxed"] is True
    assert "reduce stage failed" in str(
        response["generation_details"]["fallback_reason"]
    )


def test_summarize_single_pass_retries_then_fails():
    class DummyLibrary:
        def __init__(self):
            self.videos = {}

    class DummyEngine:
        def __init__(self):
            self.library = DummyLibrary()
            self.model = "claude-sonnet-4-5-20250929"

    service = _make_service(enabled=False)
    service.engine = DummyEngine()

    attempt_counter = {"count": 0}

    def failing_llm(**kwargs):
        attempt_counter["count"] += 1
        return {
            "provider": "chatgpt",
            "model": "gpt-4o-mini",
            "text": "not-json-output",
        }

    service._llm_text_response = failing_llm
    chunks = [{"raw_text": "hello", "start": 0.0, "end": 8.0}]
    lines = service._summary_transcript_lines(chunks)

    with pytest.raises(
        local_api.SummaryGenerationError, match="single-pass summary generation failed"
    ):
        service._summarize_transcript_single_pass(
            transcript_lines=lines,
            segments=chunks,
            language="en",
            provider="chatgpt",
            max_points=5,
        )

    assert attempt_counter["count"] == local_api.SUMMARY_RETRY_ATTEMPTS


def test_normalize_summary_items_does_not_fallback_to_chunks():
    service = _make_service(enabled=False)

    items = service._normalize_summary_items(
        items={"items": []},
        segments=[
            {"raw_text": "chunk A", "start": 0.0, "end": 10.0},
            {"raw_text": "chunk B", "start": 12.0, "end": 22.0},
        ],
        language="en",
        max_points=5,
    )

    assert items == []


def test_normalize_summary_items_preserves_importance_order():
    service = _make_service(enabled=False)
    segments = [
        {"raw_text": "first", "start": 0.0, "end": 10.0},
        {"raw_text": "second", "start": 10.0, "end": 300.0},
    ]

    items = service._normalize_summary_items(
        items={
            "items": [
                {"title": "Most Important", "tldr": "theme A", "start": 120.0},
                {"title": "Second", "tldr": "theme B", "start": 10.0},
                {"title": "Third", "tldr": "theme C", "start": 30.0},
                {"title": "Fourth", "tldr": "theme D", "start": 40.0},
                {"title": "Fifth", "tldr": "theme E", "start": 50.0},
            ]
        },
        segments=segments,
        language="en",
        max_points=5,
    )

    assert [row["title"] for row in items] == [
        "Most Important",
        "Second",
        "Third",
        "Fourth",
        "Fifth",
    ]


def test_summary_items_match_language_detects_non_english_output():
    service = _make_service(enabled=False)
    assert (
        service._summary_items_match_language(
            [
                {"title": "Intro", "tldr": "Overview of the video"},
                {"title": "Topic 1", "tldr": "Key argument and examples"},
            ],
            "en",
        )
        is True
    )
    assert (
        service._summary_items_match_language(
            [
                {"title": "イントロ", "tldr": "これは日本語です"},
                {"title": "Topic 1", "tldr": "Still mixed"},
            ],
            "en",
        )
        is False
    )


def test_summary_copy_detector_flags_verbatim_chunk_copy():
    service = _make_service(enabled=False)
    chunks = [
        {
            "raw_text": "This is a long transcript line with very specific wording that should not be copied directly."
        },
    ]
    assert (
        service._summary_looks_like_transcript_copy(
            "This is a long transcript line with very specific wording that should not be copied directly.",
            chunks,
        )
        is True
    )


def test_summary_sentence_count_validator_requires_four_to_five_sentences():
    service = _make_service(enabled=False)
    assert (
        service._summary_items_have_required_sentence_count(
            [{"tldr": "One. Two. Three. Four."}],
            "en",
        )
        is True
    )
    assert (
        service._summary_items_have_required_sentence_count(
            [{"tldr": "One. Two. Three."}],
            "en",
        )
        is False
    )


def test_resolve_theme_anchor_timestamp_prefers_anchor_match():
    service = _make_service(enabled=False)
    segments = [
        {"text": "intro setup", "start": 0.0, "end": 8.0},
        {"text": "important anchor appears here", "start": 90.0, "end": 108.0},
    ]
    resolved = service._resolve_theme_anchor_timestamp(
        item={"anchor_text": "anchor appears here", "start": 2.0},
        segments=segments,
    )
    assert resolved["source"] == "anchor_match"
    assert resolved["start"] == 90.0


def test_resolve_theme_anchor_timestamp_uses_model_start_for_ambiguous_anchor():
    service = _make_service(enabled=False)
    segments = [
        {"text": "repeated anchor text", "start": 10.0, "end": 20.0},
        {"text": "some middle section", "start": 40.0, "end": 50.0},
        {"text": "repeated anchor text again", "start": 90.0, "end": 100.0},
    ]

    resolved = service._resolve_theme_anchor_timestamp(
        item={"anchor_text": "repeated anchor text", "start": 95.0},
        segments=segments,
    )

    assert resolved["source"] == "anchor_match"
    assert resolved["start"] == 90.0


def test_resolve_theme_anchor_timestamp_penalizes_reused_segment():
    service = _make_service(enabled=False)
    segments = [
        {"text": "anchor phrase", "start": 10.0, "end": 20.0},
        {"text": "anchor phrase", "start": 80.0, "end": 90.0},
    ]

    resolved = service._resolve_theme_anchor_timestamp(
        item={"anchor_text": "anchor phrase", "start": 0.0},
        segments=segments,
        used_segment_counts={0: 2},
    )

    assert resolved["source"] == "anchor_match"
    assert resolved["start"] == 80.0


def test_summarize_video_transcript_backfills_full_transcript_once():
    class DummyLibrary:
        def __init__(self):
            self.videos = {
                "vid1": {
                    "title": "Legacy Video",
                    "chunks": [
                        {"raw_text": "Intro section", "start": 0.0, "end": 12.0},
                        {"raw_text": "Core explanation", "start": 30.0, "end": 42.0},
                    ],
                }
            }
            self.save_calls = 0

        def save(self):
            self.save_calls += 1

    class DummyEngine:
        def __init__(self):
            self.library = DummyLibrary()
            self.model = "claude-sonnet-4-5-20250929"

    service = _make_service(enabled=False)
    service.engine = DummyEngine()
    service._summarize_transcript_single_pass = lambda **kwargs: {
        "provider": "chatgpt",
        "model": "gpt-4o-mini",
        "items": [
            {
                "title": "Theme 1",
                "tldr": "One. Two. Three. Four.",
                "anchor_text": "Intro section",
            },
            {
                "title": "Theme 2",
                "tldr": "One. Two. Three. Four.",
                "anchor_text": "Core explanation",
            },
            {
                "title": "Theme 3",
                "tldr": "One. Two. Three. Four.",
                "anchor_text": "Intro section",
            },
            {
                "title": "Theme 4",
                "tldr": "One. Two. Three. Four.",
                "anchor_text": "Core explanation",
            },
            {
                "title": "Theme 5",
                "tldr": "One. Two. Three. Four.",
                "anchor_text": "Intro section",
            },
        ],
        "strategy": "single_pass",
    }

    first = service.summarize_video_transcript(
        video_id="vid1", language="en", provider="chatgpt"
    )
    second = service.summarize_video_transcript(
        video_id="vid1", language="en", provider="chatgpt"
    )

    assert first["generation_details"]["full_transcript_backfilled"] is True
    assert second["generation_details"]["full_transcript_backfilled"] is False
    assert service.engine.library.save_calls == 1
    assert "full_transcript" in service.engine.library.videos["vid1"]
    assert service._summary_cache_path("vid1").exists()


def test_summarize_video_transcript_uses_summary_cache_for_repeat_request():
    class DummyLibrary:
        def __init__(self):
            self.videos = {
                "vid1": {
                    "title": "Cached Video",
                    "full_transcript": {
                        "segments": [
                            {"text": "Intro section", "start": 0.0, "end": 12.0},
                            {"text": "Core explanation", "start": 30.0, "end": 42.0},
                        ],
                        "text": "Intro section\nCore explanation",
                        "segment_count": 2,
                    },
                    "chunks": [
                        {"raw_text": "Intro section", "start": 0.0, "end": 12.0},
                        {"raw_text": "Core explanation", "start": 30.0, "end": 42.0},
                    ],
                }
            }
            self.save_calls = 0

        def save(self):
            self.save_calls += 1

    class DummyEngine:
        def __init__(self):
            self.library = DummyLibrary()
            self.model = "claude-sonnet-4-5-20250929"

    service = _make_service(enabled=False)
    service.engine = DummyEngine()
    counter = {"count": 0}

    def fake_single_pass(**kwargs):
        counter["count"] += 1
        return {
            "provider": "chatgpt",
            "model": "gpt-4o-mini",
            "items": [
                {
                    "title": "Theme 1",
                    "tldr": "One. Two. Three. Four.",
                    "anchor_text": "Intro section",
                },
                {
                    "title": "Theme 2",
                    "tldr": "One. Two. Three. Four.",
                    "anchor_text": "Core explanation",
                },
                {
                    "title": "Theme 3",
                    "tldr": "One. Two. Three. Four.",
                    "anchor_text": "Intro section",
                },
                {
                    "title": "Theme 4",
                    "tldr": "One. Two. Three. Four.",
                    "anchor_text": "Core explanation",
                },
                {
                    "title": "Theme 5",
                    "tldr": "One. Two. Three. Four.",
                    "anchor_text": "Intro section",
                },
            ],
            "strategy": "single_pass",
        }

    service._summarize_transcript_single_pass = fake_single_pass

    first = service.summarize_video_transcript(
        video_id="vid1", language="en", provider="chatgpt"
    )
    second = service.summarize_video_transcript(
        video_id="vid1", language="en", provider="chatgpt"
    )

    assert first["cached"] is False
    assert second["cached"] is True
    assert first["generation_details"]["cache_hit"] is False
    assert second["generation_details"]["cache_hit"] is True
    assert second["generation_details"]["cache_generated_at"] is not None
    assert counter["count"] == 1
    assert service.engine.library.save_calls == 0
    assert service._summary_cache_path("vid1").exists()


def test_summarize_video_transcript_invalidates_cache_when_transcript_changes():
    class DummyLibrary:
        def __init__(self):
            self.videos = {
                "vid1": {
                    "title": "Cache Invalidation Video",
                    "full_transcript": {
                        "segments": [
                            {"text": "Intro section", "start": 0.0, "end": 12.0},
                            {"text": "Core explanation", "start": 30.0, "end": 42.0},
                        ],
                        "text": "Intro section\nCore explanation",
                        "segment_count": 2,
                    },
                    "chunks": [
                        {"raw_text": "Intro section", "start": 0.0, "end": 12.0},
                        {"raw_text": "Core explanation", "start": 30.0, "end": 42.0},
                    ],
                }
            }
            self.save_calls = 0

        def save(self):
            self.save_calls += 1

    class DummyEngine:
        def __init__(self):
            self.library = DummyLibrary()
            self.model = "claude-sonnet-4-5-20250929"

    service = _make_service(enabled=False)
    service.engine = DummyEngine()
    counter = {"count": 0}

    def fake_single_pass(**kwargs):
        counter["count"] += 1
        return {
            "provider": "chatgpt",
            "model": "gpt-4o-mini",
            "items": [
                {
                    "title": "Theme 1",
                    "tldr": "One. Two. Three. Four.",
                    "anchor_text": "Intro section",
                },
                {
                    "title": "Theme 2",
                    "tldr": "One. Two. Three. Four.",
                    "anchor_text": "Core explanation",
                },
                {
                    "title": "Theme 3",
                    "tldr": "One. Two. Three. Four.",
                    "anchor_text": "Intro section",
                },
                {
                    "title": "Theme 4",
                    "tldr": "One. Two. Three. Four.",
                    "anchor_text": "Core explanation",
                },
                {
                    "title": "Theme 5",
                    "tldr": "One. Two. Three. Four.",
                    "anchor_text": "Intro section",
                },
            ],
            "strategy": "single_pass",
        }

    service._summarize_transcript_single_pass = fake_single_pass

    first = service.summarize_video_transcript(
        video_id="vid1", language="en", provider="chatgpt"
    )
    service.engine.library.videos["vid1"]["full_transcript"]["segments"][0][
        "text"
    ] = "Updated intro section"
    second = service.summarize_video_transcript(
        video_id="vid1", language="en", provider="chatgpt"
    )

    assert first["cached"] is False
    assert second["cached"] is False
    assert second["generation_details"]["cache_hit"] is False
    assert counter["count"] == 2
    assert service.engine.library.save_calls == 0


def test_summarize_video_transcript_cache_key_partitions_requests():
    class DummyLibrary:
        def __init__(self):
            self.videos = {
                "vid1": {
                    "title": "Cache Key Video",
                    "full_transcript": {
                        "segments": [
                            {"text": "Intro section", "start": 0.0, "end": 12.0},
                            {"text": "Core explanation", "start": 30.0, "end": 42.0},
                        ],
                        "text": "Intro section\nCore explanation",
                        "segment_count": 2,
                    },
                    "chunks": [
                        {"raw_text": "Intro section", "start": 0.0, "end": 12.0},
                        {"raw_text": "Core explanation", "start": 30.0, "end": 42.0},
                    ],
                }
            }
            self.save_calls = 0

        def save(self):
            self.save_calls += 1

    class DummyEngine:
        def __init__(self):
            self.library = DummyLibrary()
            self.model = "claude-sonnet-4-5-20250929"

    service = _make_service(enabled=False)
    service.engine = DummyEngine()
    counter = {"count": 0}

    def fake_single_pass(**kwargs):
        counter["count"] += 1
        provider = kwargs.get("provider", "chatgpt")
        return {
            "provider": provider,
            "model": "gpt-4o-mini",
            "items": [
                {
                    "title": "Theme 1",
                    "tldr": "One. Two. Three. Four.",
                    "anchor_text": "Intro section",
                },
                {
                    "title": "Theme 2",
                    "tldr": "One. Two. Three. Four.",
                    "anchor_text": "Core explanation",
                },
                {
                    "title": "Theme 3",
                    "tldr": "One. Two. Three. Four.",
                    "anchor_text": "Intro section",
                },
                {
                    "title": "Theme 4",
                    "tldr": "One. Two. Three. Four.",
                    "anchor_text": "Core explanation",
                },
                {
                    "title": "Theme 5",
                    "tldr": "One. Two. Three. Four.",
                    "anchor_text": "Intro section",
                },
            ],
            "strategy": "single_pass",
        }

    service._summarize_transcript_single_pass = fake_single_pass

    first = service.summarize_video_transcript(
        video_id="vid1", language="en", provider="chatgpt"
    )
    second = service.summarize_video_transcript(
        video_id="vid1", language="en", provider="chatgpt"
    )
    third = service.summarize_video_transcript(
        video_id="vid1", language="ja", provider="chatgpt"
    )
    fourth = service.summarize_video_transcript(
        video_id="vid1", language="en", provider="claude"
    )
    assert first["cached"] is False
    assert second["cached"] is True
    assert third["cached"] is False
    assert fourth["cached"] is False
    assert counter["count"] == 3


def test_summary_prompt_requests_descriptive_speaker_aware_themes():
    service = _make_service(enabled=False)

    prompt = service._summary_system_prompt(language="en", max_points=5)

    assert "Make each theme title concrete and descriptive" in prompt
    assert "who is speaking" in prompt
    assert "never guess" in prompt


def test_ingest_persists_library_and_initializes_summary_cache():
    class DummyLibrary:
        def __init__(self):
            self.videos = {}
            self.save_calls = 0

        def add_video(self, video_id, language="ja"):
            self.videos[video_id] = {
                "title": f"Video {video_id}",
                "language": language,
                "chunks": [{"raw_text": "hello", "start": 0.0, "end": 8.0}],
                "full_transcript": {
                    "segments": [{"text": "hello", "start": 0.0, "end": 8.0}],
                    "text": "hello",
                    "segment_count": 1,
                },
            }
            return video_id

        def get_video_chunking_metadata(self, video_id):
            del video_id
            return {"version": "time_v2_60s_15s"}

        def video_chunking_is_stale(self, video_id):
            del video_id
            return False

        def save(self):
            self.save_calls += 1

    class DummyEngine:
        def __init__(self):
            self.library = DummyLibrary()
            self.model = "claude-sonnet-4-5-20250929"

    service = _make_service(enabled=False)
    service.engine = DummyEngine()
    service.lock = threading.Lock()
    service.jobs = {}
    service.title_cache = {}
    service._append_ingest_log = lambda **kwargs: None
    service._hydrate_video_title = lambda video_id: None

    result = service.ingest(
        url="https://www.youtube.com/watch?v=abc12345678",
        mode="single",
        language="en",
        force=False,
    )

    assert result["queued_count"] == 1
    assert service.engine.library.save_calls == 1
    assert service.engine.library.videos["abc12345678"].get("summary_cache") is None


def test_ingest_reingests_stale_video_without_force():
    class DummyLibrary:
        def __init__(self):
            self.videos = {
                "abc12345678": {
                    "title": "Old Video",
                    "language": "en",
                    "chunks": [{"raw_text": "old", "start": 0.0, "end": 45.0}],
                    "chunking": {"version": "time_v1_45s_15s"},
                }
            }
            self.save_calls = 0
            self.add_calls = []

        def add_video(self, video_id, language="ja"):
            self.add_calls.append((video_id, language))
            self.videos[video_id] = {
                "title": f"Video {video_id}",
                "language": language,
                "chunks": [{"raw_text": "new", "start": 0.0, "end": 60.0}],
                "full_transcript": {
                    "segments": [{"text": "new", "start": 0.0, "end": 60.0}],
                    "text": "new",
                    "segment_count": 1,
                },
                "chunking": {"version": "time_v2_60s_15s"},
            }
            return video_id

        def get_video_chunking_metadata(self, video_id):
            return self.videos[video_id].get("chunking", {"version": "unknown"})

        def video_chunking_is_stale(self, video_id):
            return (
                self.get_video_chunking_metadata(video_id).get("version")
                != "time_v2_60s_15s"
            )

        def save(self):
            self.save_calls += 1

    class DummyEngine:
        def __init__(self):
            self.library = DummyLibrary()
            self.model = "claude-sonnet-4-5-20250929"

    service = _make_service(enabled=False)
    service.engine = DummyEngine()
    service.lock = threading.Lock()
    service.jobs = {}
    service.title_cache = {}
    service._append_ingest_log = lambda **kwargs: None
    service._hydrate_video_title = lambda video_id: None

    result = service.ingest(
        url="https://www.youtube.com/watch?v=abc12345678",
        mode="single",
        language="en",
        force=False,
    )

    assert result["queued_count"] == 1
    assert result["skipped_count"] == 0
    assert service.engine.library.add_calls == [("abc12345678", "en")]
    assert service.engine.library.save_calls == 1
    assert (
        service.engine.library.videos["abc12345678"]["chunking"]["version"]
        == "time_v2_60s_15s"
    )
