"""Tests for chunking comparison API endpoints and service methods.

Run with: pytest tests/test_chunking_api.py -v
"""

import importlib
import os
import sys
import tempfile
import threading
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
LOCAL_PREVIEW_DIR = ROOT_DIR / "local_preview"
if str(LOCAL_PREVIEW_DIR) not in sys.path:
    sys.path.insert(0, str(LOCAL_PREVIEW_DIR))

os.environ["YT_RAG_SKIP_GLOBAL_SERVICE"] = "1"
local_api = importlib.import_module("local_api")
LocalRAGService = local_api.LocalRAGService


# ---------------------------------------------------------------------------
# Fixtures & helpers
# ---------------------------------------------------------------------------

FAKE_EMBED_DIM = 64


class FakeProcessor:
    """Minimal TextProcessor stand-in with deterministic embeddings."""

    embed_dim = FAKE_EMBED_DIM

    def __init__(self):
        self.tokenizers = {"ja": True, "en": True}
        self.last_time_call = None
        self.last_sentence_call = None
        self.last_token_call = None

    def clean_text(self, s):
        return s.replace("\n", " ").strip()

    def tokenize(self, text, language):
        return text.lower()

    def make_embed_text(self, raw_text, language):
        clean = self.clean_text(raw_text)
        if not clean:
            return ""
        return clean + "\n" + self.tokenize(clean, language)

    def reconstruct_lines_from_transcript(self, full_transcript, language):
        segments = full_transcript.get("segments", [])
        lines = []
        for seg in segments:
            raw_text = str(seg.get("text", "")).strip()
            if not raw_text:
                continue
            start = float(seg.get("start", 0.0))
            end = float(seg.get("end", start))
            duration = max(0.0, end - start)
            embed_text = self.make_embed_text(raw_text, language)
            if embed_text.strip():
                lines.append({
                    "start": start,
                    "duration": duration,
                    "raw_text": self.clean_text(raw_text),
                    "embed_text": embed_text,
                })
        return lines

    def chunk_by_time_with_overlap(self, lines, window=60, overlap=15):
        self.last_time_call = {
            "lines": lines,
            "window": window,
            "overlap": overlap,
        }
        chunks = []
        for i, line in enumerate(lines):
            chunks.append({
                "start": line["start"],
                "end": line["start"] + line["duration"],
                "raw_text": line["raw_text"],
                "embed_text": line["embed_text"],
            })
        return chunks

    def chunk_by_sentence_boundary(self, lines, max_chars=1000):
        self.last_sentence_call = {
            "lines": lines,
            "max_chars": max_chars,
        }
        raw = " ".join(l["raw_text"] for l in lines)
        embed = "\n".join(l["embed_text"] for l in lines)
        if not lines:
            return []
        return [{
            "start": lines[0]["start"],
            "end": lines[-1]["start"] + lines[-1]["duration"],
            "raw_text": raw,
            "embed_text": embed,
        }]

    def chunk_by_token_count(self, lines, token_count=256, overlap_fraction=0.25, language="ja"):
        self.last_token_call = {
            "lines": lines,
            "token_count": token_count,
            "overlap_fraction": overlap_fraction,
            "language": language,
        }
        return self.chunk_by_time_with_overlap(lines)

    def generate_embeddings(self, chunks):
        n = len(chunks)
        rng = np.random.RandomState(42)
        emb = rng.randn(n, self.embed_dim).astype("float32")
        norms = np.linalg.norm(emb, axis=1, keepdims=True)
        return emb / norms

    def encode_query(self, query, language=None):
        rng = np.random.RandomState(hash(query) % 2**31)
        emb = rng.randn(1, self.embed_dim).astype("float32")
        norms = np.linalg.norm(emb, axis=1, keepdims=True)
        return emb / norms


def _make_full_transcript(segment_count=5):
    segments = []
    for i in range(segment_count):
        segments.append({
            "start": float(i * 10),
            "end": float(i * 10 + 9),
            "text": f"Segment {i} has some interesting content about topic {i}.",
        })
    return {
        "version": 1,
        "text": "\n".join(s["text"] for s in segments),
        "segments": segments,
        "segment_count": segment_count,
    }


def _make_service_with_video(video_id="testvid12345", language="en", segment_count=5):
    """Create a LocalRAGService with a fake video that has a full_transcript."""
    service = LocalRAGService.__new__(LocalRAGService)
    service.lock = threading.Lock()
    service.feedback = {}
    service.feedback_index = {}
    service.feedback_lock = threading.Lock()
    service.feedback_tuning_enabled = False
    service._persist_feedback = lambda: None
    runtime_dir = Path(tempfile.mkdtemp())
    service.runtime_data_dir = runtime_dir
    service.cache_data_dir = Path(tempfile.mkdtemp())
    service.summary_cache_dir = service.cache_data_dir / "summaries"
    service.legacy_data_dir = Path(tempfile.mkdtemp())
    service.feedback_path = runtime_dir / "search_feedback.json"
    service.legacy_feedback_path = service.legacy_data_dir / "search_feedback.json"
    service.openai_model = "gpt-4o-mini"
    service.ask_history_lock = threading.Lock()
    service.ask_history = {}
    service.ask_history_path = runtime_dir / "ask_history.json"
    service.legacy_ask_history_path = service.legacy_data_dir / "ask_history.json"
    service.ingest_log_path = runtime_dir / "ingest_jobs.log"
    service.legacy_ingest_log_path = service.legacy_data_dir / "ingest_jobs.log"
    service.log_lock = threading.Lock()
    service.title_cache = {}
    service.jobs = {}

    processor = FakeProcessor()

    class FakeLibrary:
        def __init__(self):
            self.processor = processor
            self.videos = {
                video_id: {
                    "title": f"Test Video {video_id}",
                    "url": f"https://www.youtube.com/watch?v={video_id}",
                    "language": language,
                    "chunks": [],
                    "full_transcript": _make_full_transcript(segment_count),
                    "summary_cache": {},
                }
            }

    class FakeEngine:
        def __init__(self):
            self.library = FakeLibrary()

    service.engine = FakeEngine()
    return service


# ---------------------------------------------------------------------------
# preview_chunking tests
# ---------------------------------------------------------------------------


class TestPreviewChunking:
    """Tests for the /v1/chunking/preview service method."""

    def test_preview_time_strategy(self):
        service = _make_service_with_video()
        result = service.preview_chunking(
            "testvid12345", "time", {"window": 60, "overlap": 15}
        )
        assert result["chunk_count"] > 0
        assert len(result["chunks"]) == result["chunk_count"]
        assert "stats" in result
        assert result["stats"]["total_chars"] > 0
        assert result["stats"]["avg_chunk_chars"] > 0

    def test_preview_sentence_strategy(self):
        service = _make_service_with_video()
        result = service.preview_chunking(
            "testvid12345", "sentence", {"max_chars": 1000}
        )
        assert result["chunk_count"] > 0
        for chunk in result["chunks"]:
            assert "index" in chunk
            assert "start" in chunk
            assert "end" in chunk
            assert "raw_text" in chunk
            assert "char_count" in chunk

    def test_preview_token_strategy(self):
        service = _make_service_with_video()
        result = service.preview_chunking(
            "testvid12345", "token", {"token_count": 256, "overlap_fraction": 0.25}
        )
        assert result["chunk_count"] > 0

    def test_preview_chunk_fields(self):
        service = _make_service_with_video()
        result = service.preview_chunking(
            "testvid12345", "time", {"window": 60, "overlap": 0}
        )
        chunk = result["chunks"][0]
        assert chunk["index"] == 0
        assert isinstance(chunk["start"], float)
        assert isinstance(chunk["end"], float)
        assert isinstance(chunk["raw_text"], str)
        assert isinstance(chunk["char_count"], int)
        assert chunk["char_count"] == len(chunk["raw_text"])

    def test_preview_stats_accuracy(self):
        service = _make_service_with_video(segment_count=10)
        result = service.preview_chunking(
            "testvid12345", "time", {"window": 60, "overlap": 0}
        )
        stats = result["stats"]
        char_counts = [c["char_count"] for c in result["chunks"]]
        assert stats["total_chars"] == sum(char_counts)
        assert stats["min_chunk_chars"] == min(char_counts)
        assert stats["max_chunk_chars"] == max(char_counts)

    def test_preview_missing_video_raises(self):
        service = _make_service_with_video()
        with pytest.raises(KeyError, match="not found"):
            service.preview_chunking("nonexistent11", "time", {})

    def test_preview_empty_video_id_raises(self):
        service = _make_service_with_video()
        with pytest.raises(ValueError, match="video_id is required"):
            service.preview_chunking("", "time", {})

    def test_preview_unknown_strategy_raises(self):
        service = _make_service_with_video()
        with pytest.raises(ValueError, match="Unknown strategy"):
            service.preview_chunking("testvid12345", "random_split", {})

    def test_preview_missing_full_transcript_raises(self):
        service = _make_service_with_video()
        service.engine.library.videos["testvid12345"]["full_transcript"] = None
        with pytest.raises(ValueError, match="no full_transcript"):
            service.preview_chunking("testvid12345", "time", {})

    def test_preview_empty_segments_raises(self):
        service = _make_service_with_video()
        service.engine.library.videos["testvid12345"]["full_transcript"] = {
            "version": 1,
            "segments": [],
        }
        with pytest.raises(ValueError, match="no full_transcript"):
            service.preview_chunking("testvid12345", "time", {})

    def test_preview_param_clamping_window(self):
        """Window is clamped to [10, 300]."""
        service = _make_service_with_video()
        # Extremely small window should be clamped to 10
        result = service.preview_chunking(
            "testvid12345", "time", {"window": 1, "overlap": 0}
        )
        assert result["chunk_count"] > 0

    def test_preview_param_clamping_max_chars(self):
        """max_chars is clamped to [100, 5000]."""
        service = _make_service_with_video()
        result = service.preview_chunking(
            "testvid12345", "sentence", {"max_chars": 1}
        )
        assert result["chunk_count"] > 0


# ---------------------------------------------------------------------------
# search_with_chunking tests
# ---------------------------------------------------------------------------


class TestSearchWithChunking:
    """Tests for the /v1/chunking/search service method."""

    def test_search_returns_results(self):
        service = _make_service_with_video(segment_count=10)
        result = service.search_with_chunking(
            "testvid12345", "time", {"window": 60, "overlap": 0},
            query="interesting content",
            k=3,
            language_override=None,
        )
        assert "chunk_count" in result
        assert "results" in result
        assert "embedding_time_ms" in result
        assert "search_time_ms" in result
        assert len(result["results"]) <= 3

    def test_search_result_fields(self):
        service = _make_service_with_video(segment_count=10)
        result = service.search_with_chunking(
            "testvid12345", "time", {},
            query="topic 3",
            k=5,
            language_override=None,
        )
        assert len(result["results"]) > 0
        hit = result["results"][0]
        assert "rank" in hit
        assert "score" in hit
        assert "chunk_index" in hit
        assert "start" in hit
        assert "end" in hit
        assert "text" in hit
        assert hit["rank"] == 1

    def test_search_ranks_are_sequential(self):
        service = _make_service_with_video(segment_count=10)
        result = service.search_with_chunking(
            "testvid12345", "time", {},
            query="some query",
            k=5,
            language_override=None,
        )
        ranks = [r["rank"] for r in result["results"]]
        assert ranks == list(range(1, len(ranks) + 1))

    def test_search_scores_are_descending(self):
        service = _make_service_with_video(segment_count=10)
        result = service.search_with_chunking(
            "testvid12345", "time", {},
            query="another query",
            k=5,
            language_override=None,
        )
        scores = [r["score"] for r in result["results"]]
        for i in range(len(scores) - 1):
            assert scores[i] >= scores[i + 1]

    def test_search_k_limits_results(self):
        service = _make_service_with_video(segment_count=20)
        result = service.search_with_chunking(
            "testvid12345", "time", {},
            query="topic",
            k=2,
            language_override=None,
        )
        assert len(result["results"]) <= 2

    def test_search_with_language_override(self):
        service = _make_service_with_video(language="ja")
        result = service.search_with_chunking(
            "testvid12345", "time", {},
            query="interesting",
            k=3,
            language_override="en",
        )
        assert len(result["results"]) > 0

    def test_search_sentence_strategy(self):
        service = _make_service_with_video(segment_count=10)
        result = service.search_with_chunking(
            "testvid12345", "sentence", {"max_chars": 500},
            query="content",
            k=3,
            language_override=None,
        )
        assert "results" in result

    def test_search_token_strategy(self):
        service = _make_service_with_video(segment_count=10)
        result = service.search_with_chunking(
            "testvid12345", "token", {"token_count": 128, "overlap_fraction": 0.25},
            query="content",
            k=3,
            language_override=None,
        )
        assert "results" in result

    def test_search_empty_transcript_raises(self):
        service = _make_service_with_video()
        service.engine.library.videos["testvid12345"]["full_transcript"] = {
            "version": 1,
            "segments": [],
        }
        with pytest.raises(ValueError, match="no full_transcript"):
            service.search_with_chunking(
                "testvid12345", "time", {},
                query="something",
                k=5,
                language_override=None,
            )

    def test_search_missing_video_raises(self):
        service = _make_service_with_video()
        with pytest.raises(KeyError):
            service.search_with_chunking(
                "nonexistent11", "time", {},
                query="test", k=3, language_override=None,
            )

    def test_search_timing_fields_are_nonnegative(self):
        service = _make_service_with_video(segment_count=10)
        result = service.search_with_chunking(
            "testvid12345", "time", {},
            query="topic",
            k=3,
            language_override=None,
        )
        assert result["embedding_time_ms"] >= 0
        assert result["search_time_ms"] >= 0


# ---------------------------------------------------------------------------
# _apply_chunking_strategy dispatch tests
# ---------------------------------------------------------------------------


class TestApplyChunkingStrategy:
    """Test that _apply_chunking_strategy dispatches correctly and clamps params."""

    def test_unknown_strategy_raises(self):
        processor = FakeProcessor()
        with pytest.raises(ValueError, match="Unknown strategy"):
            LocalRAGService._apply_chunking_strategy(
                processor, [], "nonexistent", {}, "en"
            )

    def test_time_strategy_clamps_window(self):
        processor = FakeProcessor()
        lines = [{"start": 0, "duration": 5, "raw_text": "a", "embed_text": "a"}]
        # window=5000 should be clamped to 300
        LocalRAGService._apply_chunking_strategy(
            processor, lines, "time", {"window": 5000, "overlap": 0}, "en"
        )
        assert processor.last_time_call == {
            "lines": lines,
            "window": 300,
            "overlap": 0,
        }

    def test_sentence_strategy_clamps_max_chars(self):
        processor = FakeProcessor()
        lines = [{"start": 0, "duration": 5, "raw_text": "a", "embed_text": "a"}]
        # max_chars=50 should be clamped to 100
        LocalRAGService._apply_chunking_strategy(
            processor, lines, "sentence", {"max_chars": 50}, "en"
        )
        assert processor.last_sentence_call == {
            "lines": lines,
            "max_chars": 100,
        }

    def test_token_strategy_clamps_token_count(self):
        processor = FakeProcessor()
        lines = [{"start": 0, "duration": 5, "raw_text": "a", "embed_text": "a"}]
        LocalRAGService._apply_chunking_strategy(
            processor, lines, "token", {"token_count": 9999}, "en"
        )
        assert processor.last_token_call == {
            "lines": lines,
            "token_count": 1024,
            "overlap_fraction": 0.25,
            "language": "en",
        }

    def test_overlap_fraction_clamped(self):
        processor = FakeProcessor()
        lines = [{"start": 0, "duration": 5, "raw_text": "a", "embed_text": "a"}]
        LocalRAGService._apply_chunking_strategy(
            processor, lines, "token",
            {"token_count": 128, "overlap_fraction": 0.99}, "en"
        )
        assert processor.last_token_call == {
            "lines": lines,
            "token_count": 128,
            "overlap_fraction": 0.5,
            "language": "en",
        }


# ---------------------------------------------------------------------------
# _get_video_with_transcript guard tests
# ---------------------------------------------------------------------------


class TestGetVideoWithTranscript:
    """Test input validation for video + transcript access."""

    def test_empty_video_id(self):
        service = _make_service_with_video()
        with pytest.raises(ValueError, match="video_id is required"):
            service._get_video_with_transcript("")

    def test_whitespace_video_id(self):
        service = _make_service_with_video()
        with pytest.raises(ValueError, match="video_id is required"):
            service._get_video_with_transcript("   ")

    def test_nonexistent_video(self):
        service = _make_service_with_video()
        with pytest.raises(KeyError, match="not found"):
            service._get_video_with_transcript("doesntexist1")

    def test_video_without_full_transcript(self):
        service = _make_service_with_video()
        service.engine.library.videos["testvid12345"]["full_transcript"] = None
        with pytest.raises(ValueError, match="no full_transcript"):
            service._get_video_with_transcript("testvid12345")

    def test_video_with_empty_segments(self):
        service = _make_service_with_video()
        service.engine.library.videos["testvid12345"]["full_transcript"] = {
            "version": 1,
            "segments": [],
        }
        with pytest.raises(ValueError, match="no full_transcript"):
            service._get_video_with_transcript("testvid12345")

    def test_valid_video_returns_data(self):
        service = _make_service_with_video()
        video, ft = service._get_video_with_transcript("testvid12345")
        assert video["language"] == "en"
        assert len(ft["segments"]) == 5
