"""Tests for the RAGEngine class."""

import os
import pytest
from unittest.mock import patch, MagicMock


class FakeLibrary:
    """Fake VideoLibrary for testing RAGEngine without real models."""

    def __init__(self, videos=None):
        self.videos = videos or {}

    def search(self, query, k=5, language=None):
        if not self.videos:
            return []
        results = []
        rank = 1
        for vid, data in self.videos.items():
            for i, chunk in enumerate(data["chunks"][:k]):
                results.append(
                    {
                        "rank": rank,
                        "score": 0.9 - rank * 0.1,
                        "video_id": vid,
                        "video_title": data["title"],
                        "video_url": f"https://www.youtube.com/watch?v={vid}",
                        "chunk_index": i,
                        "text": chunk["raw_text"],
                        "start": chunk["start"],
                        "end": chunk["end"],
                        "url": f"https://www.youtube.com/watch?v={vid}&t={int(chunk['start'])}s",
                    }
                )
                rank += 1
                if rank > k:
                    break
            if rank > k:
                break
        return results


def _fake_library_with_data():
    return FakeLibrary(
        videos={
            "vid1": {
                "title": "テスト動画1",
                "chunks": [
                    {"raw_text": "これはテストです", "start": 0.0, "end": 45.0},
                    {"raw_text": "二番目のチャンク", "start": 30.0, "end": 75.0},
                ],
            },
            "vid2": {
                "title": "テスト動画2",
                "chunks": [
                    {"raw_text": "別の動画のテスト", "start": 0.0, "end": 45.0},
                ],
            },
        }
    )


class TestRAGEngineSearch:
    """Test the search passthrough."""

    def test_search_delegates_to_library(self):
        from multilingual.rag_engine import RAGEngine

        lib = _fake_library_with_data()
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
    def test_ask_calls_claude(self, mock_anthropic_cls):
        from multilingual.rag_engine import RAGEngine

        # Set up mock response
        mock_client = MagicMock()
        mock_anthropic_cls.return_value = mock_client
        mock_response = MagicMock()
        mock_response.content = [MagicMock()]
        mock_response.content[0].type = "text"
        mock_response.content[0].text = "テストの回答です。"
        mock_client.messages.create.return_value = mock_response

        lib = _fake_library_with_data()
        engine = RAGEngine(library=lib, model="claude-sonnet-4-5-20250929")
        result = engine.ask("これは何ですか？", k=3)

        assert result["answer"] == "テストの回答です。"
        assert len(result["sources"]) == 3
        assert result["model"] == "claude-sonnet-4-5-20250929"

        # Verify Anthropic was called correctly
        call_args = mock_client.messages.create.call_args
        assert call_args.kwargs["model"] == "claude-sonnet-4-5-20250929"
        messages = call_args.kwargs["messages"]
        assert len(messages) == 1
        assert messages[0]["role"] == "user"
        assert "質問" in messages[0]["content"]

    def test_ask_empty_library(self):
        from multilingual.rag_engine import RAGEngine

        engine = RAGEngine(library=FakeLibrary())
        result = engine.ask("テスト質問")
        assert "No videos in the library" in result["answer"]
        assert result["sources"] == []

    def test_ask_no_api_key(self):
        from multilingual.rag_engine import RAGEngine

        lib = _fake_library_with_data()
        engine = RAGEngine(library=lib)

        # Remove API key if set
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
                "video_title": "動画B",
                "start": 0.0,
                "end": 45.0,
                "text": "テスト文2",
            },
        ]

        context = _format_context(results)
        assert "動画A" in context
        assert "動画B" in context
        assert "1:05" in context
        assert "テスト文1" in context
        assert "テスト文2" in context

    def test_format_context_empty(self):
        from multilingual.rag_engine import _format_context

        assert _format_context([]) == ""
