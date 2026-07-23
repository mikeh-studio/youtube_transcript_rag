"""Deterministic multi-vector video routing for large transcript libraries."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import sys
from collections import defaultdict
from typing import List, Set

import numpy as np

# Match the library compatibility shim for faiss-cpu wheels expecting NumPy 2.
try:
    import numpy.core as _numpy_core
    import numpy.core._multiarray_umath as _numpy_multiarray_umath

    sys.modules.setdefault("numpy._core", _numpy_core)
    sys.modules.setdefault("numpy._core._multiarray_umath", _numpy_multiarray_umath)
except Exception:
    pass

import faiss

from multilingual.youtube_metadata import normalize_youtube_source


ROUTING_PROFILE_VERSION = 1
DEFAULT_MAX_SECTIONS = 5
RRF_K = 60


def _chmod_private(path: str) -> None:
    try:
        os.chmod(path, 0o600)
    except OSError:
        pass


def _stable_hash(value) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _compact_text(value, limit=None) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    return text[:limit].rstrip() if limit and len(text) > limit else text


def _timestamp(seconds) -> str:
    value = max(0, int(float(seconds or 0)))
    hours, remainder = divmod(value, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def _lexical_tokens(text: str) -> Set[str]:
    return set(re.findall(r"[\w\u3040-\u30ff\u3400-\u9fff]+", text.lower()))


def _chunk_text(chunk: dict) -> str:
    return _compact_text(chunk.get("raw_text") or chunk.get("text"))


def _partition_chunks(chunks: List[dict], max_sections: int) -> List[List[dict]]:
    if not chunks:
        return []
    section_count = min(max(1, int(max_sections)), len(chunks))
    return [
        chunks[start:end]
        for index in range(section_count)
        for start, end in [
            (
                math.floor(index * len(chunks) / section_count),
                math.floor((index + 1) * len(chunks) / section_count),
            )
        ]
        if end > start
    ]


def build_routing_profile(
    video_id: str,
    video: dict,
    *,
    max_sections: int = DEFAULT_MAX_SECTIONS,
    overview_chars: int = 2400,
    section_chars: int = 1800,
) -> dict:
    """Build identity, overview, and timestamped section texts for one video."""
    video_id = str(video_id or "").strip()
    title = _compact_text(video.get("title") or f"Video {video_id}")
    source = normalize_youtube_source(
        video_id,
        video.get("source"),
        legacy=not isinstance(video.get("source"), dict),
    )
    channel = source["channel"]

    identity_lines = [
        f"Video title: {title}",
        "Source: YouTube",
        f"URL: {source['url']}",
        f"Video ID: {video_id}",
    ]
    if channel.get("name"):
        # Channel is useful provenance, but appears in exactly one vector.
        identity_lines.insert(1, f"Channel: {channel['name']}")
    identity = "\n".join(identity_lines)

    chunks = [
        chunk
        for chunk in video.get("chunks", [])
        if isinstance(chunk, dict) and _chunk_text(chunk)
    ]
    overview_parts = []
    if chunks:
        sample_count = min(8, len(chunks))
        sample_indices = sorted(
            {
                min(
                    len(chunks) - 1,
                    math.floor(index * len(chunks) / sample_count),
                )
                for index in range(sample_count)
            }
        )
        overview_parts = [_chunk_text(chunks[index]) for index in sample_indices]
    elif isinstance(video.get("full_transcript"), dict):
        overview_parts = [
            _compact_text(video["full_transcript"].get("text"), overview_chars)
        ]
    overview = _compact_text(" ".join(overview_parts), overview_chars)
    if not overview:
        overview = title

    sections = []
    for section_index, group in enumerate(
        _partition_chunks(chunks, max_sections=max_sections)
    ):
        start = float(group[0].get("start", 0.0))
        end = float(group[-1].get("end", start))
        section_text = _compact_text(
            " ".join(_chunk_text(chunk) for chunk in group),
            section_chars,
        )
        sections.append(
            {
                "index": section_index,
                "start": start,
                "end": end,
                "text": (
                    f"Transcript section [{_timestamp(start)}-{_timestamp(end)}]: "
                    f"{section_text}"
                ),
            }
        )

    vector_records = [
        {
            "video_id": video_id,
            "vector_type": "identity",
            "section_index": None,
            "start": None,
            "end": None,
            "text": identity,
            "channel_id": channel.get("id"),
        },
        {
            "video_id": video_id,
            "vector_type": "overview",
            "section_index": None,
            "start": None,
            "end": None,
            "text": f"Video overview: {overview}",
            "channel_id": channel.get("id"),
        },
    ]
    vector_records.extend(
        {
            "video_id": video_id,
            "vector_type": "section",
            "section_index": section["index"],
            "start": section["start"],
            "end": section["end"],
            "text": section["text"],
            "channel_id": channel.get("id"),
        }
        for section in sections
    )

    lexical_parts = [title]
    if channel.get("name"):
        lexical_parts.append(channel["name"])
    lexical_parts.append(overview)
    lexical_parts.extend(section["text"] for section in sections)
    chunking_version = str(
        (video.get("chunking") or {}).get("version") or "unknown"
    )
    fingerprint_payload = {
        "profile_version": ROUTING_PROFILE_VERSION,
        "video_id": video_id,
        "title": title,
        "source": {
            "platform": source["platform"],
            "video_id": source["video_id"],
            "url": source["url"],
            "channel": source["channel"],
        },
        "chunking_version": chunking_version,
        "chunks": [
            {
                "start": chunk.get("start"),
                "end": chunk.get("end"),
                "text": _chunk_text(chunk),
            }
            for chunk in chunks
        ],
    }
    return {
        "profile_version": ROUTING_PROFILE_VERSION,
        "video_id": video_id,
        "title": title,
        "source": source,
        "chunking_version": chunking_version,
        "source_fingerprint": _stable_hash(fingerprint_payload),
        "identity_text": identity,
        "overview_text": overview,
        "sections": sections,
        "lexical_text": "\n".join(part for part in lexical_parts if part),
        "vectors": vector_records,
    }


class MultiVectorVideoRouter:
    """Persistent, backend-injectable index for routing queries to videos."""

    def __init__(self, processor, *, data_dir="data", artifact_dir=None):
        self.processor = processor
        self.artifact_dir = artifact_dir or os.path.join(data_dir, "index")
        self.index = None
        self.vector_map = []
        self.profiles = {}
        self.manifest = None

    @property
    def index_path(self):
        return os.path.join(self.artifact_dir, "video_router.faiss")

    @property
    def map_path(self):
        return os.path.join(self.artifact_dir, "video_router_map.json")

    @property
    def profiles_path(self):
        return os.path.join(self.artifact_dir, "video_router_profiles.json")

    @property
    def manifest_path(self):
        return os.path.join(self.artifact_dir, "video_router_manifest.json")

    def _expected_manifest(self, profiles: dict, vector_count: int) -> dict:
        embedding = dict(self.processor.embedding_metadata())
        chunking_versions = {
            video_id: profile["chunking_version"]
            for video_id, profile in sorted(profiles.items())
        }
        profile_fingerprints = {
            video_id: profile["source_fingerprint"]
            for video_id, profile in sorted(profiles.items())
        }
        distinct_chunking = sorted(set(chunking_versions.values()))
        return {
            "profile_version": ROUTING_PROFILE_VERSION,
            "embedding_backend": embedding.get("backend"),
            "embedding_model": embedding.get("model"),
            "embedding_dim": int(embedding.get("dim") or 0),
            "source_fingerprint": _stable_hash(profile_fingerprints),
            "profile_fingerprints": profile_fingerprints,
            "chunking_version": (
                distinct_chunking[0] if len(distinct_chunking) == 1 else "mixed"
            ),
            "chunking_versions": chunking_versions,
            "vector_count": int(vector_count),
            "video_count": len(profiles),
        }

    def build(self, videos: dict, *, persist=True) -> dict:
        """Build the router from already-stored transcript records."""
        self.profiles = {
            video_id: build_routing_profile(video_id, video)
            for video_id, video in sorted(videos.items())
        }
        self.vector_map = [
            dict(record)
            for profile in self.profiles.values()
            for record in profile["vectors"]
        ]
        if self.vector_map:
            payloads = [
                {"raw_text": record["text"], "embed_text": record["text"]}
                for record in self.vector_map
            ]
            embeddings = np.asarray(
                self.processor.generate_embeddings(payloads), dtype="float32"
            )
            if embeddings.ndim != 2 or embeddings.shape[0] != len(self.vector_map):
                raise ValueError("Embedding backend returned an invalid router matrix.")
            expected_dim = int(self.processor.embedding_metadata().get("dim") or 0)
            if expected_dim and embeddings.shape[1] != expected_dim:
                raise ValueError(
                    "Embedding backend dimension does not match its metadata."
                )
            faiss.normalize_L2(embeddings)
            self.index = faiss.IndexFlatIP(int(embeddings.shape[1]))
            self.index.add(embeddings)
        else:
            self.index = None

        self.manifest = self._expected_manifest(
            self.profiles,
            len(self.vector_map),
        )
        if persist:
            self.save()
        return dict(self.manifest)

    def save(self) -> None:
        """Persist router-only artifacts separately from the chunk index."""
        os.makedirs(self.artifact_dir, exist_ok=True)
        if self.index is not None:
            faiss.write_index(self.index, self.index_path)
            _chmod_private(self.index_path)
        elif os.path.exists(self.index_path):
            os.remove(self.index_path)
        for path, payload in (
            (self.map_path, self.vector_map),
            (self.profiles_path, self.profiles),
            (self.manifest_path, self.manifest or {}),
        ):
            with open(path, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, ensure_ascii=False, indent=2)
            _chmod_private(path)

    def load(self) -> bool:
        """Load compatible artifacts; return False when files are incomplete."""
        required = (self.map_path, self.profiles_path, self.manifest_path)
        if not all(os.path.exists(path) for path in required):
            return False
        with open(self.map_path, "r", encoding="utf-8") as handle:
            self.vector_map = json.load(handle)
        with open(self.profiles_path, "r", encoding="utf-8") as handle:
            self.profiles = json.load(handle)
        with open(self.manifest_path, "r", encoding="utf-8") as handle:
            self.manifest = json.load(handle)
        self.index = (
            faiss.read_index(self.index_path)
            if os.path.exists(self.index_path)
            else None
        )
        return True

    def stale_reasons(self, videos: dict) -> List[str]:
        """Return deterministic reasons persisted artifacts need rebuilding."""
        if not isinstance(self.manifest, dict):
            return ["missing_manifest"]
        profiles = {
            video_id: build_routing_profile(video_id, video)
            for video_id, video in sorted(videos.items())
        }
        vector_count = sum(len(profile["vectors"]) for profile in profiles.values())
        expected = self._expected_manifest(profiles, vector_count)
        keys = (
            "profile_version",
            "embedding_backend",
            "embedding_model",
            "embedding_dim",
            "source_fingerprint",
            "chunking_version",
            "vector_count",
            "video_count",
        )
        return [
            f"{key}_changed"
            for key in keys
            if self.manifest.get(key) != expected.get(key)
        ] + self._artifact_stale_reasons()

    def _artifact_stale_reasons(self) -> List[str]:
        reasons = []
        expected_count = int((self.manifest or {}).get("vector_count") or 0)
        expected_dim = int((self.manifest or {}).get("embedding_dim") or 0)
        if len(self.vector_map) != expected_count:
            reasons.append("vector_map_count_changed")
        if expected_count and self.index is None:
            reasons.append("missing_index")
        if self.index is not None:
            if int(self.index.ntotal) != expected_count:
                reasons.append("index_count_changed")
            if expected_dim and int(self.index.d) != expected_dim:
                reasons.append("index_dim_changed")
        return reasons

    def is_stale(self, videos: dict) -> bool:
        """Return whether stored artifacts differ from current video inputs."""
        return bool(self.stale_reasons(videos))

    @staticmethod
    def _aggregate_dense_hits(vector_hits: dict) -> dict:
        scores = {}
        for video_id, hits in vector_hits.items():
            ordered = sorted(hits, key=lambda item: item["score"], reverse=True)
            leading = ordered[:2]
            if len(leading) == 1:
                multi_vector_score = leading[0]["score"]
            else:
                multi_vector_score = (
                    0.7 * leading[0]["score"] + 0.3 * leading[1]["score"]
                )
            identity = next(
                (
                    item["score"]
                    for item in ordered
                    if item["vector_type"] == "identity"
                ),
                multi_vector_score,
            )
            scores[video_id] = 0.85 * multi_vector_score + 0.15 * identity
        return scores

    def search(self, query: str, *, top_k=3, language=None) -> dict:
        """Rank videos with dense multi-vector aggregation plus lexical RRF."""
        query = _compact_text(query)
        top_k = max(0, int(top_k))
        if not query:
            return {
                "video_ids": [],
                "results": [],
                "used_fallback": True,
                "fallback_reason": "empty_query",
                "dense_available": False,
                "lexical_available": False,
            }
        if not self.profiles:
            return {
                "video_ids": [],
                "results": [],
                "used_fallback": True,
                "fallback_reason": "no_video_profiles",
                "dense_available": False,
                "lexical_available": False,
            }

        vector_hits = defaultdict(list)
        dense_available = self.index is not None and self.index.ntotal > 0
        if dense_available:
            query_vector = np.asarray(
                self.processor.encode_query(query, language=language),
                dtype="float32",
            )
            faiss.normalize_L2(query_vector)
            scores, indices = self.index.search(query_vector, int(self.index.ntotal))
            for score, vector_index in zip(scores[0], indices[0]):
                if vector_index < 0:
                    continue
                record = self.vector_map[int(vector_index)]
                vector_hits[record["video_id"]].append(
                    {
                        "score": float(score),
                        "vector_type": record["vector_type"],
                        "section_index": record.get("section_index"),
                    }
                )
        dense_scores = self._aggregate_dense_hits(vector_hits)

        query_tokens = _lexical_tokens(query)
        lexical_scores = {}
        if query_tokens:
            for video_id, profile in self.profiles.items():
                profile_tokens = _lexical_tokens(profile["lexical_text"])
                overlap = len(query_tokens & profile_tokens)
                if overlap:
                    lexical_scores[video_id] = overlap / math.sqrt(
                        max(1, len(query_tokens) * len(profile_tokens))
                    )
        lexical_available = bool(lexical_scores)

        dense_order = sorted(
            dense_scores,
            key=lambda video_id: (-dense_scores[video_id], video_id),
        )
        lexical_order = sorted(
            lexical_scores,
            key=lambda video_id: (-lexical_scores[video_id], video_id),
        )
        fused = defaultdict(float)
        for rank, video_id in enumerate(dense_order, start=1):
            fused[video_id] += 1.0 / (RRF_K + rank)
        for rank, video_id in enumerate(lexical_order, start=1):
            fused[video_id] += 1.0 / (RRF_K + rank)

        used_fallback = False
        fallback_reason = None
        if not fused:
            used_fallback = True
            fallback_reason = "no_dense_or_lexical_signal"
            ordered_ids = sorted(self.profiles)
        else:
            ordered_ids = sorted(
                fused,
                key=lambda video_id: (-fused[video_id], video_id),
            )

        results = []
        for rank, video_id in enumerate(ordered_ids[:top_k], start=1):
            profile = self.profiles[video_id]
            results.append(
                {
                    "rank": rank,
                    "video_id": video_id,
                    "title": profile["title"],
                    "source": profile["source"],
                    "score": float(fused.get(video_id, 0.0)),
                    "dense_score": (
                        float(dense_scores[video_id])
                        if video_id in dense_scores
                        else None
                    ),
                    "lexical_score": (
                        float(lexical_scores[video_id])
                        if video_id in lexical_scores
                        else None
                    ),
                    "vector_hits": sorted(
                        vector_hits.get(video_id, []),
                        key=lambda item: item["score"],
                        reverse=True,
                    )[:3],
                }
            )
        return {
            "video_ids": [result["video_id"] for result in results],
            "results": results,
            "used_fallback": used_fallback,
            "fallback_reason": fallback_reason,
            "dense_available": dense_available,
            "lexical_available": lexical_available,
            "fusion": "rrf",
        }
