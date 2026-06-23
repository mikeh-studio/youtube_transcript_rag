"""Tests for the multilingual RAGEngine class."""

import os
import pytest
from unittest.mock import patch, MagicMock


class FakeLibrary:
    """Fake VideoLibrary for testing RAGEngine without real models."""

    def __init__(self, videos=None):
        self.videos = videos or {}

    def search(self, query, k=5, language=None):
        self.last_search_language = language
        if not self.videos:
            return []
        results = []
        rank = 1
        for vid, data in self.videos.items():
            for i, chunk in enumerate(data["chunks"][:k]):
                results.append({
                    "rank": rank,
                    "score": 0.9 - rank * 0.1,
                    "video_id": vid,
                    "video_title": data["title"],
                    "video_url": f"https://www.youtube.com/watch?v={vid}",
                    "language": data.get("language", "ja"),
                    "chunk_index": i,
                    "text": chunk["raw_text"],
                    "start": chunk["start"],
                    "end": chunk["end"],
                    "url": f"https://www.youtube.com/watch?v={vid}&t={int(chunk['start'])}s",
                })
                rank += 1
                if rank > k:
                    break
            if rank > k:
                break
        return results


def _fake_library_ja():
    return FakeLibrary(videos={
        "vid1": {
            "title": "テスト動画1",
            "language": "ja",
            "chunks": [
                {"raw_text": "これはテストです", "start": 0.0, "end": 45.0},
                {"raw_text": "二番目のチャンク", "start": 30.0, "end": 75.0},
            ],
        },
        "vid2": {
            "title": "テスト動画2",
            "language": "ja",
            "chunks": [
                {"raw_text": "別の動画のテスト", "start": 0.0, "end": 45.0},
            ],
        },
    })


def _fake_library_en():
    return FakeLibrary(videos={
        "vid_en1": {
            "title": "Test Video 1",
            "language": "en",
            "chunks": [
                {"raw_text": "This is a test", "start": 0.0, "end": 45.0},
                {"raw_text": "Second chunk here", "start": 30.0, "end": 75.0},
            ],
        },
    })


def _fake_library_mixed():
    return FakeLibrary(videos={
        "vid_ja": {
            "title": "日本語動画",
            "language": "ja",
            "chunks": [
                {"raw_text": "日本語のテスト", "start": 0.0, "end": 45.0},
            ],
        },
        "vid_en": {
            "title": "English Video",
            "language": "en",
            "chunks": [
                {"raw_text": "English test content", "start": 0.0, "end": 45.0},
                {"raw_text": "More English content", "start": 30.0, "end": 75.0},
            ],
        },
    })


class TestRAGEngineSearch:
    """Test the search passthrough."""

    def test_search_delegates_to_library(self):
        from multilingual.rag_engine import RAGEngine
        lib = _fake_library_ja()
        engine = RAGEngine(library=lib)
        results = engine.search("テスト", k=3)
        assert len(results) == 3
        assert results[0]["video_id"] == "vid1"

    def test_search_empty_library(self):
        from multilingual.rag_engine import RAGEngine
        engine = RAGEngine(library=FakeLibrary())
        results = engine.search("テスト")
        assert results == []


class TestRAGEngineAsk:
    """Test the RAG ask pipeline."""

    @patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"})
    @patch("multilingual.rag_engine.anthropic.Anthropic")
    def test_ask_calls_claude_japanese(self, mock_anthropic_cls):
        from multilingual.rag_engine import RAGEngine

        mock_client = MagicMock()
        mock_anthropic_cls.return_value = mock_client
        mock_response = MagicMock()
        mock_response.content = [MagicMock()]
        mock_response.content[0].type = "text"
        mock_response.content[0].text = "テストの回答です。"
        mock_client.messages.create.return_value = mock_response

        lib = _fake_library_ja()
        engine = RAGEngine(library=lib, model="claude-sonnet-4-5-20250929")
        result = engine.ask("これは何ですか？", k=3)

        assert result["answer"] == "テストの回答です。"
        assert len(result["sources"]) == 3
        assert result["model"] == "claude-sonnet-4-5-20250929"

        # Verify Japanese context labels were used
        call_args = mock_client.messages.create.call_args
        messages = call_args.kwargs["messages"]
        assert "参考情報" in messages[0]["content"]
        assert "質問" in messages[0]["content"]

    @patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"})
    @patch("multilingual.rag_engine.anthropic.Anthropic")
    def test_ask_calls_claude_english(self, mock_anthropic_cls):
        from multilingual.rag_engine import RAGEngine

        mock_client = MagicMock()
        mock_anthropic_cls.return_value = mock_client
        mock_response = MagicMock()
        mock_response.content = [MagicMock()]
        mock_response.content[0].type = "text"
        mock_response.content[0].text = "This is a test answer."
        mock_client.messages.create.return_value = mock_response

        lib = _fake_library_en()
        engine = RAGEngine(library=lib, model="claude-sonnet-4-5-20250929")
        result = engine.ask("What is this?", k=2)

        assert result["answer"] == "This is a test answer."

        # Verify English context labels were used
        call_args = mock_client.messages.create.call_args
        messages = call_args.kwargs["messages"]
        assert "Context" in messages[0]["content"]
        assert "Question" in messages[0]["content"]

    @patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"})
    @patch("multilingual.rag_engine.anthropic.Anthropic")
    def test_ask_mixed_language_picks_dominant(self, mock_anthropic_cls):
        """When results are mixed, the dominant language's prompt is used."""
        from multilingual.rag_engine import RAGEngine

        mock_client = MagicMock()
        mock_anthropic_cls.return_value = mock_client
        mock_response = MagicMock()
        mock_response.content = [MagicMock()]
        mock_response.content[0].type = "text"
        mock_response.content[0].text = "Mixed answer."
        mock_client.messages.create.return_value = mock_response

        # Mixed library: 2 en chunks, 1 ja chunk → en dominates with k=3
        lib = _fake_library_mixed()
        engine = RAGEngine(library=lib, model="claude-sonnet-4-5-20250929")
        result = engine.ask("What?", k=3)

        call_args = mock_client.messages.create.call_args
        messages = call_args.kwargs["messages"]
        # English dominates (2 en vs 1 ja), so English labels should be used
        assert "Context" in messages[0]["content"]
        assert "Question" in messages[0]["content"]

    def test_ask_empty_library(self):
        from multilingual.rag_engine import RAGEngine
        engine = RAGEngine(library=FakeLibrary())
        result = engine.ask("test question")
        assert "No videos" in result["answer"]
        assert result["sources"] == []

    def test_ask_no_api_key(self):
        from multilingual.rag_engine import RAGEngine
        lib = _fake_library_ja()
        engine = RAGEngine(library=lib)

        env = os.environ.copy()
        env.pop("ANTHROPIC_API_KEY", None)
        with patch.dict(os.environ, env, clear=True):
            with pytest.raises(ValueError, match="ANTHROPIC_API_KEY"):
                engine.ask("テスト質問")


class TestFormatContext:
    """Test context formatting."""

    def test_format_context(self):
        from multilingual.rag_engine import _format_context

        results = [
            {
                "video_title": "動画A",
                "start": 65.0,
                "end": 110.0,
                "text": "テスト文1",
            },
            {
                "video_title": "Video B",
                "start": 0.0,
                "end": 45.0,
                "text": "Test text 2",
            },
        ]

        context = _format_context(results)
        assert "動画A" in context
        assert "Video B" in context
        assert "1:05" in context
        assert "テスト文1" in context
        assert "Test text 2" in context

    def test_format_context_empty(self):
        from multilingual.rag_engine import _format_context
        assert _format_context([]) == ""


class TestDetectDominantLanguage:
    """Test language detection from search results."""

    def test_detect_japanese_dominant(self):
        from multilingual.rag_engine import _detect_dominant_language
        results = [
            {"language": "ja"},
            {"language": "ja"},
            {"language": "en"},
        ]
        assert _detect_dominant_language(results) == "ja"

    def test_detect_english_dominant(self):
        from multilingual.rag_engine import _detect_dominant_language
        results = [
            {"language": "en"},
            {"language": "en"},
            {"language": "ja"},
        ]
        assert _detect_dominant_language(results) == "en"

    def test_detect_empty_results(self):
        from multilingual.rag_engine import _detect_dominant_language
        assert _detect_dominant_language([]) == "en"


class TestRAGEngineLanguageForwarding:
    """Test that RAGEngine forwards language parameter to library."""

    def test_search_forwards_language(self):
        from multilingual.rag_engine import RAGEngine
        lib = _fake_library_ja()
        engine = RAGEngine(library=lib)
        engine.search("テスト", k=3, language="ja")
        assert lib.last_search_language == "ja"

    def test_search_forwards_none_by_default(self):
        from multilingual.rag_engine import RAGEngine
        lib = _fake_library_ja()
        engine = RAGEngine(library=lib)
        engine.search("テスト", k=3)
        assert lib.last_search_language is None

    @patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"})
    @patch("multilingual.rag_engine.anthropic.Anthropic")
    def test_ask_forwards_language(self, mock_anthropic_cls):
        from multilingual.rag_engine import RAGEngine

        mock_client = MagicMock()
        mock_anthropic_cls.return_value = mock_client
        mock_response = MagicMock()
        mock_response.content = [MagicMock()]
        mock_response.content[0].type = "text"
        mock_response.content[0].text = "Answer."
        mock_client.messages.create.return_value = mock_response

        lib = _fake_library_en()
        engine = RAGEngine(library=lib, model="claude-sonnet-4-5-20250929")
        engine.ask("What?", k=2, language="en")
        assert lib.last_search_language == "en"
