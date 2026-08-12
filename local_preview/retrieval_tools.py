"""Retrieval tools backed by the persisted timestamped transcript."""

from __future__ import annotations

import math
from typing import Callable, Optional


DEFAULT_CONTEXT_WINDOW_SECONDS = 30.0


class TranscriptRetrievalTools:
    """Expose semantic, keyword, and raw-transcript context retrieval."""

    def __init__(self, *, library, retrieve_fn: Callable[..., dict]):
        self.library = library
        self._retrieve_fn = retrieve_fn

    def semantic_search(
        self,
        query: str,
        *,
        k: int = 5,
        language: Optional[str] = None,
        video_id: Optional[str] = None,
        video_ids: Optional[list[str]] = None,
        retrieval_profile: Optional[str] = None,
        reranker: Optional[str] = None,
    ) -> dict:
        """Delegate to the existing E5 + FAISS dense-search path."""
        return self._retrieve_fn(
            query,
            k=k,
            language=language,
            retrieval_mode="dense",
            video_id=video_id,
            video_ids=video_ids,
            retrieval_profile=retrieval_profile,
            reranker=reranker,
        )

    def keyword_search(
        self,
        query: str,
        *,
        k: int = 5,
        language: Optional[str] = None,
        video_id: Optional[str] = None,
        video_ids: Optional[list[str]] = None,
        retrieval_profile: Optional[str] = None,
        reranker: Optional[str] = None,
    ) -> dict:
        """Delegate to the existing Japanese-aware BM25 search path."""
        return self._retrieve_fn(
            query,
            k=k,
            language=language,
            retrieval_mode="lexical",
            video_id=video_id,
            video_ids=video_ids,
            retrieval_profile=retrieval_profile,
            reranker=reranker,
        )

    def read_context(
        self,
        video_id: str,
        timestamp: float,
        window: float = DEFAULT_CONTEXT_WINDOW_SECONDS,
    ) -> dict:
        """Read raw transcript segments around ``timestamp``.

        ``window`` is a symmetric radius in seconds. The returned bounds are
        clamped to the transcript while the stored segment timestamps and text
        remain unchanged.
        """
        scoped_video_id = str(video_id or "").strip()
        if not scoped_video_id:
            raise ValueError("video_id is required")
        if self.library is None:
            raise ValueError("raw timestamped transcript storage is unavailable")
        if scoped_video_id not in self.library.videos:
            raise KeyError(f"video_id not found: {scoped_video_id}")

        try:
            requested_timestamp = float(timestamp)
            requested_window = float(window)
        except (TypeError, ValueError) as exc:
            raise ValueError("timestamp and window must be numbers") from exc
        if not math.isfinite(requested_timestamp) or requested_timestamp < 0:
            raise ValueError("timestamp must be a finite non-negative number")
        if not math.isfinite(requested_window) or requested_window <= 0:
            raise ValueError("window must be a finite positive number")

        video = self.library.videos[scoped_video_id]
        full_transcript = video.get("full_transcript")
        segments = (
            full_transcript.get("segments")
            if isinstance(full_transcript, dict)
            else None
        )
        if not isinstance(segments, list) or not segments:
            raise ValueError(
                f"Video {scoped_video_id} has no raw timestamped transcript. "
                "Re-ingest it before using read_context."
            )

        normalized_segments = []
        for segment in segments:
            if not isinstance(segment, dict):
                continue
            text = str(segment.get("text") or "").strip()
            if not text:
                continue
            try:
                start = float(segment.get("start", 0.0))
                end = float(segment.get("end", start))
            except (TypeError, ValueError):
                continue
            if not math.isfinite(start) or not math.isfinite(end):
                continue
            if end < start:
                start, end = end, start
            normalized_segments.append({"start": start, "end": end, "text": text})

        if not normalized_segments:
            raise ValueError(
                f"Video {scoped_video_id} has no valid raw transcript segments."
            )

        transcript_start = min(row["start"] for row in normalized_segments)
        transcript_end = max(row["end"] for row in normalized_segments)
        if requested_timestamp < transcript_start or requested_timestamp > transcript_end:
            raise ValueError(
                "timestamp must fall within the raw transcript bounds "
                f"({transcript_start:g}-{transcript_end:g})"
            )
        context_start = max(transcript_start, requested_timestamp - requested_window)
        context_end = min(transcript_end, requested_timestamp + requested_window)
        selected = [
            row
            for row in normalized_segments
            if row["end"] >= context_start and row["start"] <= context_end
        ]
        text = "\n".join(row["text"] for row in selected)
        video_url = str(
            video.get("url")
            or f"https://www.youtube.com/watch?v={scoped_video_id}"
        )
        separator = "&" if "?" in video_url else "?"

        return {
            "tool": "read_context",
            "video_id": scoped_video_id,
            "video_title": str(video.get("title") or f"Video {scoped_video_id}"),
            "language": str(video.get("language") or "ja"),
            "timestamp": requested_timestamp,
            "window": requested_window,
            "start": context_start,
            "end": context_end,
            "segments": selected,
            "segment_count": len(selected),
            "text": text,
            "url": f"{video_url}{separator}t={int(context_start)}s",
            "source_basis": "full_transcript.segments",
        }
