"""Tests for new chunking strategies and reconstruct_lines_from_transcript."""

import re

import pytest
from multilingual.text_processing import EnglishTokenizer, TextProcessor


class _PassthroughTokenizer:
    def tokenize(self, text):
        return text


@pytest.fixture
def processor():
    proc = TextProcessor.__new__(TextProcessor)
    proc._ws = re.compile(r"\s+")
    proc.tokenizers = {
        "en": EnglishTokenizer(),
        "ja": _PassthroughTokenizer(),
    }
    return proc


def _make_lines(count, duration=5.0, text_template="Line {i} of the transcript."):
    """Create fake processed lines for testing."""
    lines = []
    for i in range(count):
        raw_text = text_template.format(i=i)
        lines.append({
            "start": i * duration,
            "duration": duration,
            "raw_text": raw_text,
            "embed_text": raw_text,
        })
    return lines


def _make_full_transcript(count, duration=5.0, text_template="Line {i} of the transcript."):
    """Create a fake full_transcript payload."""
    segments = []
    for i in range(count):
        start = i * duration
        segments.append({
            "start": start,
            "end": start + duration,
            "text": text_template.format(i=i),
        })
    return {
        "version": 1,
        "segments": segments,
        "segment_count": count,
    }


class TestReconstructLinesFromTranscript:
    def test_basic_reconstruction(self, processor):
        ft = _make_full_transcript(3, text_template="Hello world {i}")
        lines = processor.reconstruct_lines_from_transcript(ft, "en")
        assert len(lines) == 3
        assert lines[0]["start"] == 0.0
        assert lines[0]["duration"] == 5.0
        assert "Hello world 0" in lines[0]["raw_text"]
        assert lines[0]["embed_text"]  # should have embed text

    def test_empty_transcript(self, processor):
        ft = {"version": 1, "segments": []}
        lines = processor.reconstruct_lines_from_transcript(ft, "en")
        assert lines == []

    def test_missing_segments_key(self, processor):
        ft = {"version": 1}
        lines = processor.reconstruct_lines_from_transcript(ft, "en")
        assert lines == []

    def test_skips_empty_text(self, processor):
        ft = {
            "version": 1,
            "segments": [
                {"start": 0, "end": 5, "text": "Hello"},
                {"start": 5, "end": 10, "text": ""},
                {"start": 10, "end": 15, "text": "  "},
                {"start": 15, "end": 20, "text": "World"},
            ],
        }
        lines = processor.reconstruct_lines_from_transcript(ft, "en")
        assert len(lines) == 2


class TestChunkBySentenceBoundary:
    def test_basic_chunking(self, processor):
        lines = _make_lines(10, text_template="Sentence number {i}.")
        chunks = processor.chunk_by_sentence_boundary(lines, max_chars=100)
        assert len(chunks) > 0
        for chunk in chunks:
            assert "start" in chunk
            assert "end" in chunk
            assert "raw_text" in chunk
            assert "embed_text" in chunk

    def test_empty_lines(self, processor):
        chunks = processor.chunk_by_sentence_boundary([], max_chars=1000)
        assert chunks == []

    def test_single_line(self, processor):
        lines = _make_lines(1, text_template="Only one sentence.")
        chunks = processor.chunk_by_sentence_boundary(lines, max_chars=1000)
        assert len(chunks) == 1
        assert "Only one sentence." in chunks[0]["raw_text"]

    def test_respects_max_chars(self, processor):
        lines = _make_lines(20, text_template="This is sentence number {i}.")
        chunks = processor.chunk_by_sentence_boundary(lines, max_chars=100)
        # Each chunk's raw_text should be close to or under max_chars
        for chunk in chunks:
            # Allow some overshoot since we split at sentence boundaries
            assert len(chunk["raw_text"]) < 300

    def test_cjk_sentence_boundaries(self, processor):
        lines = [
            {"start": 0, "duration": 5, "raw_text": "これはテストです。", "embed_text": "これはテストです。"},
            {"start": 5, "duration": 5, "raw_text": "二番目の文です。", "embed_text": "二番目の文です。"},
            {"start": 10, "duration": 5, "raw_text": "三番目の文です。", "embed_text": "三番目の文です。"},
            {"start": 15, "duration": 5, "raw_text": "四番目の文です。", "embed_text": "四番目の文です。"},
        ]
        chunks = processor.chunk_by_sentence_boundary(lines, max_chars=30)
        assert len(chunks) > 1

    def test_no_sentence_boundary_forces_split(self, processor):
        lines = [
            {"start": 0, "duration": 5, "raw_text": "A" * 50, "embed_text": "A" * 50},
            {"start": 5, "duration": 5, "raw_text": "B" * 50, "embed_text": "B" * 50},
            {"start": 10, "duration": 5, "raw_text": "C" * 50, "embed_text": "C" * 50},
        ]
        chunks = processor.chunk_by_sentence_boundary(lines, max_chars=60)
        assert len(chunks) > 1

    def test_remainder_timestamps_preserved_after_split(self, processor):
        lines = [
            {"start": 0, "duration": 5, "raw_text": "A.", "embed_text": "A."},
            {"start": 5, "duration": 5, "raw_text": "continuation", "embed_text": "continuation"},
            {"start": 10, "duration": 5, "raw_text": "more", "embed_text": "more"},
        ]
        chunks = processor.chunk_by_sentence_boundary(lines, max_chars=14)
        assert chunks == [
            {"start": 0, "end": 5, "raw_text": "A.", "embed_text": "A."},
            {
                "start": 5,
                "end": 15,
                "raw_text": "continuation more",
                "embed_text": "continuation\nmore",
            },
        ]


class TestChunkByTokenCount:
    def test_basic_chunking(self, processor):
        lines = _make_lines(10, text_template="Word number {i} in the transcript")
        chunks = processor.chunk_by_token_count(
            lines, token_count=10, overlap_fraction=0.25, language="en"
        )
        assert len(chunks) > 0
        for chunk in chunks:
            assert "start" in chunk
            assert "end" in chunk
            assert "raw_text" in chunk
            assert "embed_text" in chunk

    def test_empty_lines(self, processor):
        chunks = processor.chunk_by_token_count(
            [], token_count=256, language="en"
        )
        assert chunks == []

    def test_single_line(self, processor):
        lines = _make_lines(1, text_template="Short line")
        chunks = processor.chunk_by_token_count(
            lines, token_count=256, language="en"
        )
        assert len(chunks) == 1

    def test_overlap(self, processor):
        lines = _make_lines(20, text_template="Token {i} value data info")
        no_overlap = processor.chunk_by_token_count(
            lines, token_count=10, overlap_fraction=0.0, language="en"
        )
        with_overlap = processor.chunk_by_token_count(
            lines, token_count=10, overlap_fraction=0.25, language="en"
        )
        # With overlap should produce more (or equal) chunks
        assert len(with_overlap) >= len(no_overlap)

    def test_timestamps_preserved(self, processor):
        lines = _make_lines(5, duration=10.0, text_template="Some text {i}")
        chunks = processor.chunk_by_token_count(
            lines, token_count=5, language="en"
        )
        for chunk in chunks:
            assert chunk["start"] >= 0
            assert chunk["end"] > chunk["start"]
