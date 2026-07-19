"""Tests for morphological Japanese lexical tokenization with bigram fallback."""

import importlib
import os
import sys
from pathlib import Path

import pytest


ROOT_DIR = Path(__file__).resolve().parents[1]
LOCAL_PREVIEW_DIR = ROOT_DIR / "local_preview"
if str(LOCAL_PREVIEW_DIR) not in sys.path:
    sys.path.insert(0, str(LOCAL_PREVIEW_DIR))

os.environ["YT_RAG_SKIP_GLOBAL_SERVICE"] = "1"
local_api = importlib.import_module("local_api")


@pytest.fixture
def bigram_fallback(monkeypatch):
    """Force the bigram fallback path as if fugashi were unavailable."""
    monkeypatch.setattr(local_api, "_ja_lexical_tokenizer", lambda: None)


def test_japanese_morphological_tokens():
    tokens = local_api.tokenize_for_lexical("東京都で機械学習を勉強する", "ja")
    assert "機械" in tokens
    assert "学習" in tokens
    assert "勉強" in tokens
    # No cross-word character bigrams.
    assert "を勉" not in tokens
    assert "械学" not in tokens


def test_japanese_punctuation_morphemes_dropped():
    tokens = local_api.tokenize_for_lexical("機械学習を。", "ja")
    assert "。" not in tokens
    assert "機械" in tokens


def test_mixed_script_keeps_latin_and_numeric_tokens():
    tokens = local_api.tokenize_for_lexical("GPT-4で機械学習", "ja")
    assert "gpt" in tokens
    assert "4" in tokens
    assert "機械" in tokens
    # Latin tokens are not double-counted.
    assert tokens.count("gpt") == 1


def test_english_tokenization_unchanged():
    tokens = local_api.tokenize_for_lexical("How to study machine learning?", "en")
    assert tokens == ["how", "to", "study", "machine", "learning"]


def test_token_version_reports_morph_when_available():
    assert local_api._ja_lexical_tokenizer() is not None
    assert local_api.lexical_token_version() == local_api.LEXICAL_TOKENIZER_MORPH_VERSION


def test_bigram_fallback_matches_legacy_behavior(bigram_fallback):
    tokens = local_api.tokenize_for_lexical("機械学習", "ja")
    assert tokens == ["機械", "械学", "学習", "機械学習"]
    assert (
        local_api.lexical_token_version()
        == local_api.LEXICAL_TOKENIZER_BIGRAM_VERSION
    )


def test_staticmethod_delegates_to_module_function():
    service_tokens = local_api.LocalRAGService._tokenize_for_lexical(
        "機械学習を勉強する", "ja"
    )
    module_tokens = local_api.tokenize_for_lexical("機械学習を勉強する", "ja")
    assert service_tokens == module_tokens


def test_empty_and_whitespace_input():
    assert local_api.tokenize_for_lexical("", "ja") == []
    assert local_api.tokenize_for_lexical("   ", None) == []
