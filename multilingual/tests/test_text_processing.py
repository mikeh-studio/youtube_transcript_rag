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
