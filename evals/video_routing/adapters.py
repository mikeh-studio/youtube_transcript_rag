"""Adapters that connect production video routers to the label-blind evaluator."""

from __future__ import annotations

import copy
import tempfile
import time
from pathlib import Path
from typing import Any, Iterable, Mapping

from multilingual.video_library import VideoLibrary
from multilingual.video_routing import MultiVectorVideoRouter


def _library_video(record: Mapping[str, Any]) -> dict:
    video_id = str(record.get("video_id") or "").strip()
    chunks = []
    for index, chunk in enumerate(record.get("chunks") or []):
        if not isinstance(chunk, Mapping):
            continue
        text = str(chunk.get("text") or "").strip()
        if not text:
            continue
        start = float(chunk.get("start", index * 60.0))
        end = float(chunk.get("end", start + 60.0))
        chunks.append(
            {
                "raw_text": text,
                "embed_text": text,
                "start": start,
                "end": end,
            }
        )
    transcript_excerpt = str(record.get("transcript_excerpt") or "").strip()
    if not chunks and transcript_excerpt:
        chunks.append(
            {
                "raw_text": transcript_excerpt,
                "embed_text": transcript_excerpt,
                "start": 0.0,
                "end": 60.0,
            }
        )
    return {
        "url": str(
            record.get("url")
            or (record.get("source") or {}).get("url")
            or f"https://www.youtube.com/watch?v={video_id}"
        ),
        "title": str(record.get("title") or f"Video {video_id}"),
        "language": str(record.get("language") or "en"),
        "source": copy.deepcopy(record.get("source") or {}),
        "chunks": chunks,
        "chunking": {"version": "video_routing_eval_v1"},
    }


def build_library_videos(video_records: Iterable[Mapping[str, Any]]) -> dict:
    """Convert label-free eval video records into stored-library shapes."""
    videos = {}
    for record in video_records:
        video_id = str(record.get("video_id") or "").strip()
        if video_id:
            videos[video_id] = _library_video(record)
    return videos


class MultiVectorRouterAdapter:
    """Run the production multi-vector router against an eval video corpus."""

    def __init__(
        self,
        video_records: Iterable[Mapping[str, Any]],
        processor,
        *,
        artifact_dir: Path | str | None = None,
        include_channel_ablation: bool = True,
        include_chunk_retrieval: bool = True,
        chunk_top_k: int = 5,
    ):
        self.videos = build_library_videos(video_records)
        self._temp_dir = None
        if artifact_dir is None:
            self._temp_dir = tempfile.TemporaryDirectory(
                prefix="video-routing-eval-"
            )
            artifact_dir = self._temp_dir.name
        self.router = MultiVectorVideoRouter(
            processor,
            artifact_dir=str(artifact_dir),
        )
        self.router.build(self.videos, persist=False)
        self.chunk_top_k = max(1, int(chunk_top_k))
        self.library = None
        if include_chunk_retrieval:
            self.library = VideoLibrary(
                data_dir=str(Path(artifact_dir) / "chunk_library"),
                processor=processor,
            )
            self.library.videos = copy.deepcopy(self.videos)
            self.library._rebuild_index()
        self.without_channel_router = None
        if include_channel_ablation:
            ablated_videos = copy.deepcopy(self.videos)
            for video in ablated_videos.values():
                source = video.get("source")
                channel = source.get("channel") if isinstance(source, dict) else None
                if isinstance(channel, dict):
                    channel["name"] = None
            self.without_channel_router = MultiVectorVideoRouter(
                processor,
                artifact_dir=str(Path(artifact_dir) / "without_channel"),
            )
            self.without_channel_router.build(ablated_videos, persist=False)

    def route(self, request: Mapping[str, Any]) -> dict:
        """Return evaluator-compatible rankings without reading gold labels."""
        started_at = time.perf_counter()
        query = str(request.get("query") or "")
        language = str(request.get("language") or "").strip() or None
        top_k = max(1, int(request.get("top_k") or 5))
        result = self.router.search(query, top_k=top_k, language=language)
        response = {
            "ranked_video_ids": list(result.get("video_ids") or []),
            "fallback_used": bool(result.get("used_fallback")),
            "latency_ms": (time.perf_counter() - started_at) * 1000.0,
        }
        if self.library is not None and response["ranked_video_ids"]:
            chunk_rows = self.library.search(
                query,
                k=self.chunk_top_k,
                language=language,
                video_ids=response["ranked_video_ids"],
            )
            response["ranked_chunks"] = [
                {
                    "video_id": row["video_id"],
                    "chunk_index": row["chunk_index"],
                }
                for row in chunk_rows
            ]
        if self.without_channel_router is not None:
            ablated = self.without_channel_router.search(
                query,
                top_k=top_k,
                language=language,
            )
            response["ablation"] = {
                "without_channel": {
                    "ranked_video_ids": list(ablated.get("video_ids") or [])
                }
            }
        return response
