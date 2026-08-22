"""Tests for semantic, keyword, and raw transcript retrieval tools."""

from types import SimpleNamespace

import pytest

from local_preview.retrieval_tools import TranscriptRetrievalTools


def _library():
    return SimpleNamespace(
        videos={
            "vid1": {
                "title": "Raw transcript fixture",
                "url": "https://www.youtube.com/watch?v=vid1",
                "language": "ja",
                "full_transcript": {
                    "segments": [
                        {"start": 0.0, "end": 4.0, "text": "最初の行"},
                        {"start": 5.0, "end": 9.0, "text": "中央の行"},
                        {"start": 10.0, "end": 14.0, "text": "最後の行"},
                    ]
                },
            },
            "legacy": {
                "title": "Legacy chunks only",
                "chunks": [{"start": 0, "end": 60, "raw_text": "derived"}],
            },
        }
    )


def test_search_tools_delegate_to_existing_retrieval_modes():
    calls = []

    def retrieve(query, **kwargs):
        calls.append((query, kwargs))
        return {"retrieval_mode": kwargs["retrieval_mode"], "results": []}

    tools = TranscriptRetrievalTools(library=_library(), retrieve_fn=retrieve)

    tools.semantic_search("意味検索", k=3, video_id="vid1")
    tools.keyword_search("完全一致", k=4, language="ja")

    assert calls[0][1]["retrieval_mode"] == "dense"
    assert calls[0][1]["video_id"] == "vid1"
    assert calls[1][1]["retrieval_mode"] == "lexical"
    assert calls[1][1]["language"] == "ja"


def test_read_context_uses_raw_segments_and_clamps_bounds():
    tools = TranscriptRetrievalTools(library=_library(), retrieve_fn=lambda *a, **k: {})

    context = tools.read_context("vid1", timestamp=2, window=8)

    assert context["source_basis"] == "full_transcript.segments"
    assert context["start"] == 0.0
    assert context["end"] == 14.0
    assert context["requested_start"] == 0.0
    assert context["requested_end"] == 10.0
    assert [row["text"] for row in context["segments"]] == [
        "最初の行",
        "中央の行",
        "最後の行",
    ]
    assert context["text"] == "最初の行\n中央の行\n最後の行"


def test_read_context_selects_overlapping_nearby_segments():
    tools = TranscriptRetrievalTools(library=_library(), retrieve_fn=lambda *a, **k: {})

    context = tools.read_context("vid1", timestamp=8, window=1)

    assert context["start"] == 5.0
    assert context["end"] == 9.0
    assert context["requested_start"] == 7.0
    assert context["requested_end"] == 9.0
    assert context["segment_count"] == 1
    assert context["segments"][0]["text"] == "中央の行"


@pytest.mark.parametrize(
    ("timestamp", "window", "message"),
    [
        (-1, 5, "timestamp"),
        (99, 5, "timestamp"),
        (1, 0, "window"),
        (float("nan"), 5, "timestamp"),
    ],
)
def test_read_context_rejects_invalid_bounds(timestamp, window, message):
    tools = TranscriptRetrievalTools(library=_library(), retrieve_fn=lambda *a, **k: {})

    with pytest.raises(ValueError, match=message):
        tools.read_context("vid1", timestamp=timestamp, window=window)


def test_read_context_requires_genuine_raw_segments():
    tools = TranscriptRetrievalTools(library=_library(), retrieve_fn=lambda *a, **k: {})

    with pytest.raises(ValueError, match="Re-ingest"):
        tools.read_context("legacy", timestamp=5, window=5)

    with pytest.raises(KeyError, match="missing"):
        tools.read_context("missing", timestamp=5, window=5)
