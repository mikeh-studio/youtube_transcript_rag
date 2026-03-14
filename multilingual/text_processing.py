"""
Multilingual text processing utilities for YouTube transcript search.

Supports Japanese and English, extensible to Chinese and Korean.
Uses a tokenizer registry pattern: each language provides a Tokenizer subclass,
and LANGUAGE_CONFIG maps language codes to their tokenizer + transcript codes.
"""

import os
import hashlib

# Set threading environment variables BEFORE importing numpy/torch/faiss
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import re
import numpy as np
from abc import ABC, abstractmethod
from dataclasses import asdict
from sentence_transformers import SentenceTransformer


class Tokenizer(ABC):
    """Base class for language-specific tokenizers."""

    @abstractmethod
    def tokenize(self, text: str) -> str:
        """Tokenize text into a space-separated string of tokens."""


class JapaneseTokenizer(Tokenizer):
    """Japanese tokenizer using Fugashi morphological analysis."""

    def __init__(self):
        from fugashi import Tagger
        self.tagger = Tagger()

    def tokenize(self, text: str) -> str:
        return " ".join(w.surface for w in self.tagger(text))


class EnglishTokenizer(Tokenizer):
    """English tokenizer with punctuation removal and stop word filtering.

    Lowercases, strips punctuation, removes common stop words, and collapses
    whitespace. No extra dependencies needed.
    """

    _punct = re.compile(r"[^\w\s]")
    _stop_words = frozenset({
        "a", "an", "the", "and", "or", "but", "not", "no", "nor",
        "is", "am", "are", "was", "were", "be", "been", "being",
        "have", "has", "had", "having",
        "do", "does", "did", "doing",
        "will", "would", "shall", "should", "may", "might", "must", "can", "could",
        "i", "me", "my", "mine", "myself",
        "you", "your", "yours", "yourself", "yourselves",
        "he", "him", "his", "himself",
        "she", "her", "hers", "herself",
        "it", "its", "itself",
        "we", "us", "our", "ours", "ourselves",
        "they", "them", "their", "theirs", "themselves",
        "this", "that", "these", "those",
        "what", "which", "who", "whom", "whose",
        "where", "when", "why", "how",
        "in", "on", "at", "to", "for", "of", "with", "by", "from",
        "up", "out", "off", "over", "under", "into", "through",
        "about", "between", "after", "before", "during", "without",
        "again", "further", "then", "once",
        "here", "there", "all", "each", "every", "both", "few",
        "more", "most", "other", "some", "such",
        "only", "own", "same", "so", "than", "too", "very",
        "just", "because", "as", "until", "while",
        "if", "else", "also", "any", "many", "much",
    })

    def tokenize(self, text: str) -> str:
        text = text.lower()
        text = self._punct.sub(" ", text)
        words = text.split()
        words = [w for w in words if w not in self._stop_words]
        return " ".join(words)


# Registry: add a new language by adding one entry here + one Tokenizer subclass.
LANGUAGE_CONFIG = {
    "ja": {
        "transcript_codes": ["ja", "ja-JP"],
        "tokenizer": JapaneseTokenizer,
    },
    "en": {
        "transcript_codes": ["en", "en-US"],
        "tokenizer": EnglishTokenizer,
    },
    # Future:
    # "zh": {"transcript_codes": ["zh", "zh-CN", "zh-TW"], "tokenizer": ChineseTokenizer},
    # "ko": {"transcript_codes": ["ko", "ko-KR"], "tokenizer": KoreanTokenizer},
}


class TextProcessor:
    """Handles multilingual text processing, embedding, and chunking.

    Loads the shared embedding model once and initializes tokenizers
    for all configured languages.
    """

    def __init__(self):
        """Initialize embedding model and all language tokenizers."""
        self._ws = re.compile(r"\s+")
        self.embed_dim = max(128, int(os.environ.get("RAG_LOCAL_EMBED_DIM", "768")))
        self.embed_model = None
        self.embedding_backend = "hashing"

        print("Initializing tokenizers...")
        self.tokenizers = {}
        for lang, cfg in LANGUAGE_CONFIG.items():
            try:
                self.tokenizers[lang] = cfg["tokenizer"]()
            except Exception as e:
                print(f"  Warning: could not load {lang} tokenizer: {e}")

        print("Loading embedding model (this may take a moment)...")
        try:
            self.embed_model = SentenceTransformer("intfloat/multilingual-e5-base")
            self.embedding_backend = "sentence_transformers"
        except Exception as e:
            print(f"  Warning: embedding model unavailable ({e}). Falling back to local hashing embeddings.")
            self.embed_model = None
            self.embedding_backend = "hashing"

    def _hash_embedding(self, text: str) -> np.ndarray:
        """Deterministic local embedding fallback used when remote model is unavailable."""
        cleaned = self.clean_text(text)
        vec = np.zeros(self.embed_dim, dtype="float32")
        if not cleaned:
            return vec

        for token in cleaned:
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
            idx = int.from_bytes(digest[:4], "little") % self.embed_dim
            sign = 1.0 if (digest[4] & 1) == 0 else -1.0
            vec[idx] += sign

        norm = np.linalg.norm(vec)
        if norm > 0:
            vec /= norm
        return vec

    def extract_video_id(self, url):
        """Extract video ID from YouTube URL."""
        patterns = [
            r'(?:youtube\.com\/watch\?v=)([a-zA-Z0-9_-]+)',
            r'(?:youtu\.be\/)([a-zA-Z0-9_-]+)',
            r'(?:youtube\.com\/embed\/)([a-zA-Z0-9_-]+)',
        ]
        for pattern in patterns:
            match = re.search(pattern, url)
            if match:
                return match.group(1)

        if re.match(r'^[a-zA-Z0-9_-]{11}$', url):
            return url

        return None

    def clean_text(self, s: str) -> str:
        """Clean text by normalizing whitespace."""
        s = s.replace("\n", " ").replace("\r", " ")
        s = self._ws.sub(" ", s).strip()
        return s

    def tokenize(self, text: str, language: str) -> str:
        """Tokenize text using the language-specific tokenizer.

        Args:
            text: Text to tokenize.
            language: Language code (e.g. "ja", "en").

        Returns:
            Space-separated tokenized string.

        Raises:
            ValueError: If language is not configured.
        """
        if language not in self.tokenizers:
            raise ValueError(
                f"No tokenizer for language '{language}'. "
                f"Available: {list(self.tokenizers.keys())}"
            )
        return self.tokenizers[language].tokenize(text)

    def make_embed_text(self, raw_text: str, language: str) -> str:
        """Create embedding text with both raw and tokenized versions.

        Args:
            raw_text: Original text.
            language: Language code for tokenization.

        Returns:
            Dual-representation string (raw + tokenized).
        """
        raw_clean = self.clean_text(raw_text)
        if not raw_clean:
            return ""
        tok = self.tokenize(raw_clean, language)
        return raw_clean + "\n" + tok

    def to_dict_line(self, x):
        """Convert transcript segment to dictionary."""
        if isinstance(x, dict):
            return x
        try:
            return asdict(x)
        except (TypeError, AttributeError):
            pass
        return {
            "text": getattr(x, "text", ""),
            "start": getattr(x, "start", 0.0),
            "duration": getattr(x, "duration", 0.0),
        }

    def chunk_by_time_with_overlap(self, lines, window=45, overlap=15):
        """Chunk transcript into time windows with overlap."""
        chunks = []
        i = 0
        n = len(lines)

        if n == 0:
            return chunks

        if overlap >= window:
            raise ValueError(f"overlap ({overlap}) must be less than window ({window})")

        while i < n:
            start_time = lines[i]["start"]
            end_time_target = start_time + window

            raw_parts = []
            embed_parts = []
            j = i
            while j < n and lines[j]["start"] < end_time_target:
                raw_parts.append(lines[j]["raw_text"])
                embed_parts.append(lines[j]["embed_text"])
                j += 1

            if j == i:
                i += 1
                continue

            last = lines[j - 1]
            end_time = last["start"] + last["duration"]

            raw_text = " ".join(raw_parts).strip()
            embed_text = "\n".join(embed_parts).strip()

            if embed_text:
                chunks.append({
                    "start": start_time,
                    "end": end_time,
                    "raw_text": raw_text,
                    "embed_text": embed_text,
                })

            next_start = end_time_target - overlap
            initial_i = i
            while i < n and lines[i]["start"] < next_start:
                i += 1
            if i == initial_i:
                i += 1

        return chunks

    def reconstruct_lines_from_transcript(self, full_transcript, language):
        """Convert stored full_transcript.segments back to processed-line format.

        Args:
            full_transcript: Dict with 'segments' list of {start, end, text}.
            language: Language code for tokenization.

        Returns:
            List of processed line dicts with start, duration, raw_text, embed_text.
        """
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

    def chunk_by_sentence_boundary(self, lines, max_chars=1000):
        """Chunk lines by accumulating up to max_chars, splitting at sentence boundaries.

        Args:
            lines: List of processed line dicts with start, duration, raw_text, embed_text.
            max_chars: Maximum characters per chunk before forcing a split.

        Returns:
            List of chunk dicts with start, end, raw_text, embed_text.
        """
        if not lines:
            return []

        sentence_end_re = re.compile(r'[.!?。！？]\s*$')
        chunks = []
        buf_lines = []
        buf_chars = 0

        def build_chunk(chunk_lines):
            raw_text = " ".join(item["raw_text"] for item in chunk_lines).strip()
            embed_text = "\n".join(item["embed_text"] for item in chunk_lines).strip()
            if not embed_text:
                return None
            last_line = chunk_lines[-1]
            return {
                "start": chunk_lines[0]["start"],
                "end": last_line["start"] + last_line["duration"],
                "raw_text": raw_text,
                "embed_text": embed_text,
            }

        for line in lines:
            line_chars = len(line["raw_text"])

            if buf_chars > 0 and buf_chars + line_chars > max_chars:
                # Try to split at last sentence boundary
                split_idx = len(buf_lines)
                for k in range(len(buf_lines) - 1, -1, -1):
                    if sentence_end_re.search(buf_lines[k]["raw_text"]):
                        split_idx = k + 1
                        break

                if split_idx == 0:
                    split_idx = len(buf_lines)

                chunk = build_chunk(buf_lines[:split_idx])
                if chunk:
                    chunks.append(chunk)

                buf_lines = buf_lines[split_idx:]
                if buf_lines:
                    buf_chars = sum(len(item["raw_text"]) for item in buf_lines)
                else:
                    buf_chars = 0

            buf_lines.append(line)
            buf_chars += line_chars

        if buf_lines:
            chunk = build_chunk(buf_lines)
            if chunk:
                chunks.append(chunk)

        return chunks

    def chunk_by_token_count(self, lines, token_count=256, overlap_fraction=0.25, language="ja"):
        """Chunk lines by fixed token windows with overlap.

        Tokenizes all lines using the language-specific tokenizer, creates
        fixed-size token windows, and maps back to timestamps.

        Args:
            lines: List of processed line dicts with start, duration, raw_text, embed_text.
            token_count: Number of tokens per chunk window.
            overlap_fraction: Fraction of token_count to overlap between windows.
            language: Language code for tokenization.

        Returns:
            List of chunk dicts with start, end, raw_text, embed_text.
        """
        if not lines:
            return []

        # Build flat token list with line index tracking
        all_tokens = []
        line_boundaries = []  # (token_start_idx, token_end_idx) per line
        for i, line in enumerate(lines):
            try:
                tokenized = self.tokenize(line["raw_text"], language)
            except ValueError:
                tokenized = line["raw_text"]
            tokens = tokenized.split()
            start_idx = len(all_tokens)
            all_tokens.extend(tokens)
            line_boundaries.append((start_idx, len(all_tokens), i))

        if not all_tokens:
            return []

        overlap_tokens = max(0, int(token_count * overlap_fraction))
        step = max(1, token_count - overlap_tokens)
        chunks = []
        pos = 0

        while pos < len(all_tokens):
            end_pos = min(pos + token_count, len(all_tokens))

            # Find which lines are covered by this token window
            first_line_idx = None
            last_line_idx = None
            for tok_start, tok_end, line_idx in line_boundaries:
                if tok_end > pos and tok_start < end_pos:
                    if first_line_idx is None:
                        first_line_idx = line_idx
                    last_line_idx = line_idx

            if first_line_idx is not None:
                chunk_start = lines[first_line_idx]["start"]
                last = lines[last_line_idx]
                chunk_end = last["start"] + last["duration"]

                raw_parts = []
                embed_parts = []
                for li in range(first_line_idx, last_line_idx + 1):
                    raw_parts.append(lines[li]["raw_text"])
                    embed_parts.append(lines[li]["embed_text"])

                raw_text = " ".join(raw_parts).strip()
                embed_text = "\n".join(embed_parts).strip()
                if embed_text:
                    chunks.append({
                        "start": chunk_start,
                        "end": chunk_end,
                        "raw_text": raw_text,
                        "embed_text": embed_text,
                    })

            if end_pos >= len(all_tokens):
                break
            pos += step

        return chunks

    def process_transcript(self, transcript, language):
        """Process raw transcript segments into lines ready for chunking.

        Args:
            transcript: List of transcript segments (dicts or objects with text/start/duration).
            language: Language code for tokenization.

        Returns:
            List of processed line dicts with start, duration, raw_text, embed_text.
        """
        transcript_dicts = [self.to_dict_line(x) for x in transcript]
        lines = []
        for line in transcript_dicts:
            raw_text = line.get("text", "")
            start = float(line.get("start", 0.0))
            duration = float(line.get("duration", 0.0))
            embed_text = self.make_embed_text(raw_text, language)

            if embed_text.strip():
                lines.append({
                    "start": start,
                    "duration": duration,
                    "raw_text": self.clean_text(raw_text),
                    "embed_text": embed_text,
                })
        return lines

    def generate_embeddings(self, chunks):
        """Generate normalized embeddings for a list of chunks.

        Args:
            chunks: List of chunk dicts with 'embed_text' key.

        Returns:
            numpy array of shape (n_chunks, embedding_dim), dtype float32.
        """
        texts = ["passage: " + c["embed_text"] for c in chunks]
        if self.embed_model is not None:
            emb = self.embed_model.encode(
                texts,
                normalize_embeddings=True,
                show_progress_bar=len(texts) > 10,
            )
            return np.asarray(emb, dtype="float32")

        return np.asarray([self._hash_embedding(text) for text in texts], dtype="float32")

    def encode_query(self, query, language=None):
        """Encode a search query using E5 query prefix convention.

        Args:
            query: Raw query string.
            language: Language code for tokenization. If None, uses raw query only.

        Returns:
            numpy array of shape (1, embedding_dim), dtype float32.
        """
        clean = self.clean_text(query)
        if language and language in self.tokenizers:
            query_tok = self.tokenize(clean, language)
            query_text = f"query: {clean}\n{query_tok}"
        else:
            # No language specified: use raw query (works well with multilingual-e5)
            query_text = f"query: {clean}"
        if self.embed_model is not None:
            emb = self.embed_model.encode([query_text], normalize_embeddings=True)
            return np.asarray(emb, dtype="float32")

        return np.asarray([self._hash_embedding(query_text)], dtype="float32")
