"""Tests for local preview summary generation and retrieval upgrades."""

import importlib
import os
import sys
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
    service.openai_model = "gpt-4o-mini"
    return service


def test_retrieve_rerank_happens_before_topk_cut():
    service = _make_service(enabled=True)

    service._dense_search = lambda query, k, language: [
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
    service._lexical_bm25_search = lambda query, k, language: []

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
    service._dense_search = lambda query, k, language: dense_rows
    service._lexical_bm25_search = lambda query, k, language: []

    result = service.retrieve("python", k=3, retrieval_mode="dense")
    video_ids = [row["video_id"] for row in result["results"]]

    assert "vidB" in video_ids
    assert result["details"]["diversity_applied"] is True
    assert result["details"]["selected_per_video_cap"] == 2


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
