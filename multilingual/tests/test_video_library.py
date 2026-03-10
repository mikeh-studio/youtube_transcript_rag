"""Tests for the multilingual VideoLibrary class."""

import os
import json
import tempfile
import numpy as np
import pytest
from unittest.mock import patch, MagicMock

import faiss


class FakeProcessor:
    """Minimal fake processor for testing without loading real models."""

    def __init__(self):
        self.embed_dim = 768
        self.tokenizers = {"ja": True, "en": True}

    def extract_video_id(self, url):
        import re

        patterns = [
            r"(?:youtube\.com\/watch\?v=)([a-zA-Z0-9_-]+)",
            r"(?:youtu\.be\/)([a-zA-Z0-9_-]+)",
        ]
        for pattern in patterns:
            match = re.search(pattern, url)
            if match:
                return match.group(1)
        if re.match(r"^[a-zA-Z0-9_-]{11}$", url):
            return url
        return None

    def process_transcript(self, transcript, language):
        lines = []
        for seg in transcript:
            text = (
                seg.get("text", "")
                if isinstance(seg, dict)
                else getattr(seg, "text", "")
            )
            start = float(
                seg.get("start", 0.0)
                if isinstance(seg, dict)
                else getattr(seg, "start", 0.0)
            )
            duration = float(
                seg.get("duration", 0.0)
                if isinstance(seg, dict)
                else getattr(seg, "duration", 0.0)
            )
            if text.strip():
                lines.append(
                    {
                        "start": start,
                        "duration": duration,
                        "raw_text": text.strip(),
                        "embed_text": text.strip(),
                    }
                )
        return lines

    def chunk_by_time_with_overlap(self, lines, window=60, overlap=15):
        chunks = []
        for line in lines:
            chunks.append(
                {
                    "start": line["start"],
                    "end": line["start"] + line["duration"],
                    "raw_text": line["raw_text"],
                    "embed_text": line["embed_text"],
                }
            )
        return chunks

    def generate_embeddings(self, chunks):
        n = len(chunks)
        rng = np.random.RandomState(42)
        emb = rng.randn(n, self.embed_dim).astype("float32")
        norms = np.linalg.norm(emb, axis=1, keepdims=True)
        return emb / norms

    def encode_query(self, query, language=None):
        self.last_query_language = language
        rng = np.random.RandomState(hash(query) % 2**31)
        emb = rng.randn(1, self.embed_dim).astype("float32")
        norms = np.linalg.norm(emb, axis=1, keepdims=True)
        return emb / norms


def _make_transcript(n_segments=5, start_offset=0.0, language="ja"):
    """Create a fake transcript with n segments."""
    segments = []
    for i in range(n_segments):
        if language == "ja":
            text = f"テスト文{i + 1}"
        else:
            text = f"Test sentence {i + 1}"
        segments.append(
            {
                "text": text,
                "start": start_offset + i * 10.0,
                "duration": 8.0,
            }
        )
    return segments


@pytest.fixture
def tmp_data_dir():
    with tempfile.TemporaryDirectory() as d:
        yield d


@pytest.fixture
def library(tmp_data_dir):
    from multilingual.video_library import VideoLibrary

    return VideoLibrary(data_dir=tmp_data_dir, processor=FakeProcessor())


class TestVideoLibraryBasics:
    """Test basic library operations."""

    def test_empty_library(self, library):
        assert library.list_videos() == []
        assert library.total_chunks == 0
        assert library.index is None

    @patch("multilingual.video_library.YouTubeTranscriptApi")
    def test_add_video(self, mock_api_cls, library):
        from multilingual.video_library import CHUNKING_VERSION

        mock_api = MagicMock()
        mock_api.fetch.return_value = _make_transcript(5)
        mock_api_cls.return_value = mock_api

        vid = library.add_video("https://www.youtube.com/watch?v=abc12345678")
        assert vid == "abc12345678"
        assert len(library.videos) == 1
        assert library.total_chunks == 5
        assert library.index is not None
        assert library.index.ntotal == 5
        # Default language is "ja"
        assert library.videos["abc12345678"]["language"] == "ja"
        assert library.videos["abc12345678"]["full_transcript"]["segment_count"] == 5
        assert library.videos["abc12345678"]["summary_cache"] == {}
        assert library.videos["abc12345678"]["chunking"]["version"] == CHUNKING_VERSION
        assert library.video_chunking_is_stale("abc12345678") is False

    @patch("multilingual.video_library.YouTubeTranscriptApi")
    def test_add_english_video(self, mock_api_cls, library):
        mock_api = MagicMock()
        mock_api.fetch.return_value = _make_transcript(4, language="en")
        mock_api_cls.return_value = mock_api

        vid = library.add_video(
            "https://www.youtube.com/watch?v=eng12345678", language="en"
        )
        assert vid == "eng12345678"
        assert library.videos["eng12345678"]["language"] == "en"
        assert library.total_chunks == 4
        # Verify transcript was fetched with English codes
        mock_api.fetch.assert_called_once_with("eng12345678", languages=["en", "en-US"])

    @patch("multilingual.video_library.YouTubeTranscriptApi")
    def test_add_duplicate_video(self, mock_api_cls, library):
        mock_api = MagicMock()
        mock_api.fetch.return_value = _make_transcript(3)
        mock_api_cls.return_value = mock_api

        library.add_video("https://www.youtube.com/watch?v=abc12345678")
        library.add_video("https://www.youtube.com/watch?v=abc12345678")
        assert len(library.videos) == 1

    @patch("multilingual.video_library.YouTubeTranscriptApi")
    def test_add_multiple_videos(self, mock_api_cls, library):
        mock_api = MagicMock()
        mock_api.fetch.side_effect = [_make_transcript(3), _make_transcript(4)]
        mock_api_cls.return_value = mock_api

        library.add_video("https://www.youtube.com/watch?v=abc12345678")
        library.add_video("https://www.youtube.com/watch?v=xyz98765432")
        assert len(library.videos) == 2
        assert library.total_chunks == 7
        assert library.index.ntotal == 7

    @patch("multilingual.video_library.YouTubeTranscriptApi")
    def test_add_mixed_language_videos(self, mock_api_cls, library):
        mock_api = MagicMock()
        mock_api.fetch.side_effect = [
            _make_transcript(3, language="ja"),
            _make_transcript(4, language="en"),
        ]
        mock_api_cls.return_value = mock_api

        library.add_video("https://www.youtube.com/watch?v=jpn12345678", language="ja")
        library.add_video("https://www.youtube.com/watch?v=eng12345678", language="en")
        assert len(library.videos) == 2
        assert library.videos["jpn12345678"]["language"] == "ja"
        assert library.videos["eng12345678"]["language"] == "en"
        assert library.total_chunks == 7

    @patch("multilingual.video_library.YouTubeTranscriptApi")
    def test_remove_video(self, mock_api_cls, library):
        mock_api = MagicMock()
        mock_api.fetch.return_value = _make_transcript(3)
        mock_api_cls.return_value = mock_api

        library.add_video("https://www.youtube.com/watch?v=abc12345678")
        library.remove_video("abc12345678")
        assert len(library.videos) == 0
        assert library.total_chunks == 0
        assert library.index is None

    def test_remove_nonexistent_video(self, library):
        with pytest.raises(KeyError):
            library.remove_video("nonexistent_id")

    def test_invalid_url(self, library):
        with pytest.raises(ValueError):
            library.add_video("not-a-url")

    def test_unsupported_language(self, library):
        with pytest.raises(ValueError, match="Unsupported language"):
            library.add_video(
                "https://www.youtube.com/watch?v=abc12345678", language="xx"
            )

    @patch("multilingual.video_library.YouTubeTranscriptApi")
    def test_list_videos_shows_language(self, mock_api_cls, library):
        from multilingual.video_library import CHUNKING_VERSION

        mock_api = MagicMock()
        mock_api.fetch.side_effect = [
            _make_transcript(3, language="ja"),
            _make_transcript(2, language="en"),
        ]
        mock_api_cls.return_value = mock_api

        library.add_video("https://www.youtube.com/watch?v=jpn12345678", language="ja")
        library.add_video("https://www.youtube.com/watch?v=eng12345678", language="en")
        videos = library.list_videos()
        assert len(videos) == 2
        langs = {v["video_id"]: v["language"] for v in videos}
        assert langs["jpn12345678"] == "ja"
        assert langs["eng12345678"] == "en"
        chunking_versions = {v["video_id"]: v["chunking_version"] for v in videos}
        assert chunking_versions["jpn12345678"] == CHUNKING_VERSION
        assert chunking_versions["eng12345678"] == CHUNKING_VERSION
        assert all(v["chunking_stale"] is False for v in videos)


class TestVideoLibrarySearch:
    """Test search functionality."""

    @patch("multilingual.video_library.YouTubeTranscriptApi")
    def test_search_returns_results(self, mock_api_cls, library):
        mock_api = MagicMock()
        mock_api.fetch.return_value = _make_transcript(5)
        mock_api_cls.return_value = mock_api

        library.add_video("https://www.youtube.com/watch?v=abc12345678")
        results = library.search("テスト", k=3)
        assert len(results) == 3
        assert all("video_id" in r for r in results)
        assert all("language" in r for r in results)
        assert all("text" in r for r in results)

    @patch("multilingual.video_library.YouTubeTranscriptApi")
    def test_search_can_scope_to_single_video(self, mock_api_cls, library):
        mock_api = MagicMock()
        mock_api.fetch.side_effect = [_make_transcript(3), _make_transcript(3)]
        mock_api_cls.return_value = mock_api

        library.add_video("https://www.youtube.com/watch?v=abc12345678")
        library.add_video("https://www.youtube.com/watch?v=xyz98765432")
        results = library.search("テスト", k=3, video_id="abc12345678")

        assert results
        assert all(row["video_id"] == "abc12345678" for row in results)

    def test_search_empty_library(self, library):
        results = library.search("テスト", k=3)
        assert results == []

    @patch("multilingual.video_library.YouTubeTranscriptApi")
    def test_search_k_capped(self, mock_api_cls, library):
        mock_api = MagicMock()
        mock_api.fetch.return_value = _make_transcript(2)
        mock_api_cls.return_value = mock_api

        library.add_video("https://www.youtube.com/watch?v=abc12345678")
        results = library.search("テスト", k=100)
        assert len(results) <= 2

    @patch("multilingual.video_library.YouTubeTranscriptApi")
    def test_search_mixed_language(self, mock_api_cls, library):
        """Search across a library with both Japanese and English videos."""
        mock_api = MagicMock()
        mock_api.fetch.side_effect = [
            _make_transcript(3, language="ja"),
            _make_transcript(3, language="en"),
        ]
        mock_api_cls.return_value = mock_api

        library.add_video("https://www.youtube.com/watch?v=jpn12345678", language="ja")
        library.add_video("https://www.youtube.com/watch?v=eng12345678", language="en")
        results = library.search("test", k=6)
        assert len(results) == 6
        # Results include language info
        languages = {r["language"] for r in results}
        assert len(languages) >= 1

    @patch("multilingual.video_library.YouTubeTranscriptApi")
    def test_search_results_include_language(self, mock_api_cls, library):
        mock_api = MagicMock()
        mock_api.fetch.return_value = _make_transcript(3, language="en")
        mock_api_cls.return_value = mock_api

        library.add_video("https://www.youtube.com/watch?v=eng12345678", language="en")
        results = library.search("test", k=3)
        assert all(r["language"] == "en" for r in results)


class TestVideoLibraryPersistence:
    """Test save/load functionality."""

    @patch("multilingual.video_library.YouTubeTranscriptApi")
    def test_save_and_load(self, mock_api_cls, tmp_data_dir):
        from multilingual.video_library import CHUNKING_VERSION, VideoLibrary

        mock_api = MagicMock()
        mock_api.fetch.return_value = _make_transcript(5)
        mock_api_cls.return_value = mock_api

        lib1 = VideoLibrary(data_dir=tmp_data_dir, processor=FakeProcessor())
        lib1.add_video("https://www.youtube.com/watch?v=abc12345678")
        lib1.videos["abc12345678"]["summary_cache"] = {
            "en:chatgpt:5": {
                "version": 1,
                "generated_at": "2026-01-01T00:00:00+00:00",
                "source_fingerprint": "abc123",
                "result": {"summary": []},
            }
        }
        lib1.save()

        lib2 = VideoLibrary(data_dir=tmp_data_dir, processor=FakeProcessor())
        assert len(lib2.videos) == 1
        assert lib2.index is not None
        assert lib2.index.ntotal == 5
        assert len(lib2.chunk_map) == 5
        assert lib2.videos["abc12345678"]["full_transcript"]["segment_count"] == 5
        assert "en:chatgpt:5" in lib2.videos["abc12345678"]["summary_cache"]
        assert lib2.videos["abc12345678"]["chunking"]["version"] == CHUNKING_VERSION
        assert lib2.video_chunking_is_stale("abc12345678") is False

    @patch("multilingual.video_library.YouTubeTranscriptApi")
    def test_save_preserves_language(self, mock_api_cls, tmp_data_dir):
        """Verify language metadata survives save/load cycle."""
        from multilingual.video_library import VideoLibrary

        mock_api = MagicMock()
        mock_api.fetch.side_effect = [
            _make_transcript(3, language="ja"),
            _make_transcript(2, language="en"),
        ]
        mock_api_cls.return_value = mock_api

        lib1 = VideoLibrary(data_dir=tmp_data_dir, processor=FakeProcessor())
        lib1.add_video("https://www.youtube.com/watch?v=jpn12345678", language="ja")
        lib1.add_video("https://www.youtube.com/watch?v=eng12345678", language="en")
        lib1.save()

        lib2 = VideoLibrary(data_dir=tmp_data_dir, processor=FakeProcessor())
        assert lib2.videos["jpn12345678"]["language"] == "ja"
        assert lib2.videos["eng12345678"]["language"] == "en"

    @patch("multilingual.video_library.YouTubeTranscriptApi")
    def test_save_creates_files(self, mock_api_cls, tmp_data_dir):
        from multilingual.video_library import VideoLibrary

        mock_api = MagicMock()
        mock_api.fetch.return_value = _make_transcript(3)
        mock_api_cls.return_value = mock_api

        lib = VideoLibrary(data_dir=tmp_data_dir, processor=FakeProcessor())
        lib.add_video("https://www.youtube.com/watch?v=abc12345678")
        lib.save()

        assert os.path.exists(os.path.join(tmp_data_dir, "library_meta.json"))
        assert os.path.exists(os.path.join(tmp_data_dir, "library.faiss"))

    @patch("multilingual.video_library.YouTubeTranscriptApi")
    def test_load_backfills_missing_summary_cache(self, mock_api_cls, tmp_data_dir):
        from multilingual.video_library import VideoLibrary

        mock_api = MagicMock()
        mock_api.fetch.return_value = _make_transcript(3)
        mock_api_cls.return_value = mock_api

        lib = VideoLibrary(data_dir=tmp_data_dir, processor=FakeProcessor())
        lib.add_video("https://www.youtube.com/watch?v=abc12345678")
        lib.save()

        meta_path = os.path.join(tmp_data_dir, "library_meta.json")
        with open(meta_path, "r", encoding="utf-8") as fh:
            meta = json.load(fh)
        meta["videos"]["abc12345678"].pop("summary_cache", None)
        with open(meta_path, "w", encoding="utf-8") as fh:
            json.dump(meta, fh, ensure_ascii=False, indent=2)

        loaded = VideoLibrary(data_dir=tmp_data_dir, processor=FakeProcessor())
        assert loaded.videos["abc12345678"]["summary_cache"] == {}

    @patch("multilingual.video_library.YouTubeTranscriptApi")
    def test_load_legacy_chunking_metadata_marks_video_stale(
        self, mock_api_cls, tmp_data_dir
    ):
        from multilingual.video_library import LEGACY_CHUNKING_VERSION, VideoLibrary

        mock_api = MagicMock()
        mock_api.fetch.return_value = _make_transcript(3)
        mock_api_cls.return_value = mock_api

        lib = VideoLibrary(data_dir=tmp_data_dir, processor=FakeProcessor())
        lib.add_video("https://www.youtube.com/watch?v=abc12345678")
        lib.save()

        meta_path = os.path.join(tmp_data_dir, "library_meta.json")
        with open(meta_path, "r", encoding="utf-8") as fh:
            meta = json.load(fh)
        meta.pop("library_metadata", None)
        meta["videos"]["abc12345678"].pop("chunking", None)
        with open(meta_path, "w", encoding="utf-8") as fh:
            json.dump(meta, fh, ensure_ascii=False, indent=2)

        loaded = VideoLibrary(data_dir=tmp_data_dir, processor=FakeProcessor())
        assert (
            loaded.videos["abc12345678"]["chunking"]["version"]
            == LEGACY_CHUNKING_VERSION
        )
        assert loaded.video_chunking_is_stale("abc12345678") is True

    def test_load_nonexistent(self, tmp_data_dir):
        from multilingual.video_library import VideoLibrary

        lib = VideoLibrary(data_dir=tmp_data_dir, processor=FakeProcessor())
        assert len(lib.videos) == 0

    @patch("multilingual.video_library.YouTubeTranscriptApi")
    def test_save_empty_library(self, mock_api_cls, tmp_data_dir):
        from multilingual.video_library import VideoLibrary

        lib = VideoLibrary(data_dir=tmp_data_dir, processor=FakeProcessor())
        lib.save()
        assert os.path.exists(os.path.join(tmp_data_dir, "library_meta.json"))


class TestDominantLanguage:
    """Test _dominant_language() heuristic."""

    def test_empty_library(self, library):
        assert library._dominant_language() is None

    @patch("multilingual.video_library.YouTubeTranscriptApi")
    def test_single_language(self, mock_api_cls, library):
        mock_api = MagicMock()
        mock_api.fetch.return_value = _make_transcript(3, language="en")
        mock_api_cls.return_value = mock_api

        library.add_video("https://www.youtube.com/watch?v=eng12345678", language="en")
        assert library._dominant_language() == "en"

    @patch("multilingual.video_library.YouTubeTranscriptApi")
    def test_mixed_language_majority_wins(self, mock_api_cls, library):
        mock_api = MagicMock()
        mock_api.fetch.side_effect = [
            _make_transcript(3, language="ja"),
            _make_transcript(2, language="ja"),
            _make_transcript(2, language="en"),
        ]
        mock_api_cls.return_value = mock_api

        library.add_video("https://www.youtube.com/watch?v=jpn12345678", language="ja")
        library.add_video("https://www.youtube.com/watch?v=jpn22345678", language="ja")
        library.add_video("https://www.youtube.com/watch?v=eng12345678", language="en")
        assert library._dominant_language() == "ja"


class TestSearchLanguageThreading:
    """Test that search() passes language to encode_query."""

    @patch("multilingual.video_library.YouTubeTranscriptApi")
    def test_search_passes_dominant_language(self, mock_api_cls, library):
        mock_api = MagicMock()
        mock_api.fetch.return_value = _make_transcript(3, language="en")
        mock_api_cls.return_value = mock_api

        library.add_video("https://www.youtube.com/watch?v=eng12345678", language="en")
        library.search("test query", k=2)
        assert library.processor.last_query_language == "en"

    @patch("multilingual.video_library.YouTubeTranscriptApi")
    def test_search_with_explicit_language(self, mock_api_cls, library):
        mock_api = MagicMock()
        mock_api.fetch.return_value = _make_transcript(3, language="ja")
        mock_api_cls.return_value = mock_api

        library.add_video("https://www.youtube.com/watch?v=jpn12345678", language="ja")
        library.search("test query", k=2, language="en")
        assert library.processor.last_query_language == "en"

    def test_search_empty_library_no_crash(self, library):
        results = library.search("test", k=3, language="en")
        assert results == []
