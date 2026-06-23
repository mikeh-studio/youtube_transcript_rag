"""
RAG (Retrieval-Augmented Generation) engine with multilingual support.

Combines semantic search from VideoLibrary with LLM-based answer generation
using the Anthropic Claude API. Automatically selects the appropriate system
prompt and context labels based on the language of retrieved chunks.
"""

import os
from collections import Counter

import anthropic

from multilingual.video_library import VideoLibrary


SYSTEM_PROMPTS = {
    "ja": """あなたは日本語のYouTube動画の内容について質問に答えるアシスタントです。
以下の動画の書き起こしの抜粋を参考にして、質問に日本語で回答してください。

ルール:
- 提供された抜粋の情報のみに基づいて回答してください。
- 回答には、情報の出典（動画タイトルとタイムスタンプ）を含めてください。
- 抜粋に答えが見つからない場合は、「提供された情報からはお答えできません」と回答してください。
- 簡潔で分かりやすい回答を心がけてください。""",

    "en": """You are an assistant that answers questions about YouTube video content.
Use the following transcript excerpts as reference to answer the question.

Rules:
- Base your answer only on the provided excerpts.
- Include the source (video title and timestamp) in your answer.
- If the answer is not found in the excerpts, say "I cannot answer based on the provided information."
- Keep your answer concise and clear.""",
}

# Context labels per language
CONTEXT_LABELS = {
    "ja": {"context": "参考情報", "question": "質問"},
    "en": {"context": "Context", "question": "Question"},
}


def _format_context(results):
    """Format search results into context for the LLM prompt.

    Args:
        results: List of search result dicts from VideoLibrary.search().

    Returns:
        Formatted context string.
    """
    parts = []
    for r in results:
        start_min = int(r["start"] // 60)
        start_sec = int(r["start"] % 60)
        end_min = int(r["end"] // 60)
        end_sec = int(r["end"] % 60)
        timestamp = f"{start_min}:{start_sec:02d}-{end_min}:{end_sec:02d}"

        parts.append(
            f"【{r['video_title']}】({timestamp})\n{r['text']}"
        )
    return "\n\n---\n\n".join(parts)


def _detect_dominant_language(results):
    """Detect the dominant language from search results.

    Args:
        results: List of search result dicts (each has a 'language' key).

    Returns:
        Language code string (e.g. "ja", "en"). Defaults to "en".
    """
    if not results:
        return "en"
    counts = Counter(r.get("language", "ja") for r in results)
    return counts.most_common(1)[0][0]


class RAGEngine:
    """RAG pipeline: retrieve relevant chunks then generate an answer with an LLM."""

    DEFAULT_MODEL = "claude-sonnet-4-5-20250929"

    def __init__(self, library=None, model=None):
        """Initialize the RAG engine.

        Args:
            library: VideoLibrary instance (created if not provided).
            model: Anthropic model to use for generation. Defaults to the
                ANTHROPIC_MODEL environment variable, then DEFAULT_MODEL.
        """
        self.library = library or VideoLibrary()
        self.model = (
            model
            or os.environ.get("ANTHROPIC_MODEL", "").strip()
            or self.DEFAULT_MODEL
        )
        self._client = None

    @property
    def client(self):
        """Lazy-init Anthropic client (only needed for `ask`)."""
        if self._client is None:
            api_key = os.environ.get("ANTHROPIC_API_KEY")
            if not api_key:
                raise ValueError(
                    "ANTHROPIC_API_KEY environment variable is not set. "
                    "Set it to use the 'ask' command."
                )
            self._client = anthropic.Anthropic(api_key=api_key)
        return self._client

    def search(self, query, k=5, language=None, video_id=None):
        """Semantic search across all videos.

        Args:
            query: Search query string.
            k: Number of results to return.
            language: Language code for query tokenization. Forwarded to library.
            video_id: Optional video ID to restrict retrieval to one video.

        Returns:
            List of result dicts with score, text, video info, and timestamps.
        """
        if video_id is None:
            return self.library.search(query, k=k, language=language)
        return self.library.search(query, k=k, language=language, video_id=video_id)

    def ask(self, question, k=5, language=None):
        """Full RAG: retrieve chunks then generate an answer with citations.

        Automatically detects the dominant language of retrieved chunks and
        uses the matching system prompt and context labels.

        Args:
            question: The question to answer.
            k: Number of chunks to retrieve as context.
            language: Language code for query tokenization. Forwarded to library.

        Returns:
            Dict with 'answer' (str), 'sources' (list of result dicts),
            and 'model' (str).
        """
        # Retrieve relevant chunks
        results = self.library.search(question, k=k, language=language)

        if not results:
            return {
                "answer": "No videos in the library. Please add videos first.",
                "sources": [],
                "model": self.model,
            }

        # Detect dominant language and pick prompt/labels
        lang = _detect_dominant_language(results)
        system_prompt = SYSTEM_PROMPTS.get(lang, SYSTEM_PROMPTS["en"])
        labels = CONTEXT_LABELS.get(lang, CONTEXT_LABELS["en"])

        # Build context from retrieved chunks
        context = _format_context(results)

        # Call LLM
        user_message = f"{labels['context']}:\n\n{context}\n\n{labels['question']}: {question}"

        response = self.client.messages.create(
            model=self.model,
            max_tokens=1024,
            system=system_prompt,
            messages=[
                {"role": "user", "content": user_message},
            ],
            temperature=0.3,
        )

        answer = "".join(
            block.text
            for block in (response.content or [])
            if getattr(block, "type", "") == "text"
        )
        if not answer.strip():
            raise ValueError("Claude response contained no text content.")

        return {
            "answer": answer,
            "sources": results,
            "model": self.model,
        }
