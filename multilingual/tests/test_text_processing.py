"""Tests for EnglishTokenizer improvements."""

from multilingual.text_processing import EnglishTokenizer


class TestEnglishTokenizer:
    """Test punctuation removal and stop word filtering."""

    def setup_method(self):
        self.tok = EnglishTokenizer()

    def test_removes_punctuation(self):
        result = self.tok.tokenize("Hello, world! How's it going?")
        assert "," not in result
        assert "!" not in result
        assert "'" not in result
        assert "?" not in result

    def test_filters_stop_words(self):
        result = self.tok.tokenize("This is a test of the system")
        tokens = result.split()
        assert "this" not in tokens
        assert "is" not in tokens
        assert "a" not in tokens
        assert "the" not in tokens
        assert "of" not in tokens

    def test_preserves_content_words(self):
        result = self.tok.tokenize("The quick brown fox jumps over the lazy dog")
        tokens = result.split()
        assert "quick" in tokens
        assert "brown" in tokens
        assert "fox" in tokens
        assert "jumps" in tokens
        assert "lazy" in tokens
        assert "dog" in tokens

    def test_lowercases(self):
        result = self.tok.tokenize("HELLO WORLD")
        assert "hello" in result
        assert "world" in result

    def test_empty_string(self):
        assert self.tok.tokenize("") == ""

    def test_only_stop_words(self):
        result = self.tok.tokenize("the a an is are was were")
        assert result.strip() == ""

    def test_content_rich_sentence(self):
        result = self.tok.tokenize("Machine learning algorithms process data efficiently")
        tokens = result.split()
        assert "machine" in tokens
        assert "learning" in tokens
        assert "algorithms" in tokens
        assert "process" in tokens
        assert "data" in tokens
        assert "efficiently" in tokens


class TestEmbeddingModelConfig:
    """Tests for configurable embedding model and prefix handling."""

    def test_e5_models_use_e5_prefixes(self):
        from multilingual.text_processing import embedding_prefixes

        assert embedding_prefixes("intfloat/multilingual-e5-large") == (
            "query: ",
            "passage: ",
        )
        assert embedding_prefixes("intfloat/multilingual-e5-base") == (
            "query: ",
            "passage: ",
        )

    def test_non_e5_models_use_no_prefixes(self):
        from multilingual.text_processing import embedding_prefixes

        assert embedding_prefixes("BAAI/bge-m3") == ("", "")
        assert embedding_prefixes("some/unknown-model") == ("", "")
        assert embedding_prefixes("") == ("", "")

    def test_env_var_overrides_model_name(self, monkeypatch):
        monkeypatch.setenv("YT_RAG_FORCE_HASH_EMBEDDINGS", "1")
        monkeypatch.setenv("YT_RAG_EMBED_MODEL", "BAAI/bge-m3")
        from multilingual.text_processing import TextProcessor

        processor = TextProcessor()
        assert processor.embed_model_name == "BAAI/bge-m3"
        assert processor.query_prefix == ""
        assert processor.passage_prefix == ""

    def test_default_model_and_hashing_metadata(self, monkeypatch):
        monkeypatch.setenv("YT_RAG_FORCE_HASH_EMBEDDINGS", "1")
        monkeypatch.delenv("YT_RAG_EMBED_MODEL", raising=False)
        monkeypatch.setenv("RAG_LOCAL_EMBED_DIM", "256")
        from multilingual.text_processing import DEFAULT_EMBED_MODEL, TextProcessor

        processor = TextProcessor()
        assert processor.embed_model_name == DEFAULT_EMBED_MODEL
        assert DEFAULT_EMBED_MODEL == "intfloat/multilingual-e5-large"
        assert processor.query_prefix == ""
        assert processor.passage_prefix == ""
        assert processor.embedding_metadata() == {
            "backend": "hashing",
            "model": "local_hash",
            "dim": 256,
        }

    def test_hashing_space_does_not_depend_on_configured_model(
        self, monkeypatch
    ):
        import numpy as np

        from multilingual.text_processing import TextProcessor

        monkeypatch.setenv("YT_RAG_FORCE_HASH_EMBEDDINGS", "1")
        monkeypatch.setenv("YT_RAG_EMBED_MODEL", "intfloat/multilingual-e5-large")
        e5_processor = TextProcessor()

        monkeypatch.setenv("YT_RAG_EMBED_MODEL", "BAAI/bge-m3")
        bge_processor = TextProcessor()

        chunks = [{"embed_text": "shared fallback text"}]
        np.testing.assert_array_equal(
            e5_processor.generate_embeddings(chunks),
            bge_processor.generate_embeddings(chunks),
        )
        np.testing.assert_array_equal(
            e5_processor.encode_query("shared fallback text"),
            bge_processor.encode_query("shared fallback text"),
        )
