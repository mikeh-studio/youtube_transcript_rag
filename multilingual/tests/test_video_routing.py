"""Tests for deterministic multi-vector video routing."""

import hashlib
import re

import numpy as np

from multilingual.video_routing import (
    MultiVectorVideoRouter,
    build_routing_profile,
)


class KeywordProcessor:
    def __init__(self, model="keyword-v1"):
        self.model = model
        self.vocabulary = [
            "sourdough",
            "fermentation",
            "starter",
            "basketball",
            "defense",
            "physics",
            "quantum",
            "history",
        ]

    def embedding_metadata(self):
        return {
            "backend": "test",
            "model": self.model,
            "dim": len(self.vocabulary) + 1,
        }

    def _embed(self, text):
        lowered = text.lower()
        values = [float(lowered.count(word)) for word in self.vocabulary]
        digest = hashlib.sha256(lowered.encode("utf-8")).digest()
        values.append(float(digest[0]) / 2550.0)
        vector = np.asarray(values, dtype="float32")
        norm = np.linalg.norm(vector)
        return vector / norm if norm else vector

    def generate_embeddings(self, chunks):
        return np.vstack([self._embed(chunk["embed_text"]) for chunk in chunks])

    def encode_query(self, query, language=None):
        del language
        return self._embed(query).reshape(1, -1)


def make_video(
    video_id,
    title,
    channel_name,
    channel_id,
    topic_words,
    *,
    language="en",
):
    chunks = []
    for index, words in enumerate(topic_words):
        chunks.append(
            {
                "start": index * 60.0,
                "end": (index + 1) * 60.0,
                "raw_text": words,
                "embed_text": words,
            }
        )
    return {
        "url": f"https://www.youtube.com/watch?v={video_id}",
        "title": title,
        "language": language,
        "chunks": chunks,
        "chunking": {"version": "time_v2_60s_15s"},
        "source": {
            "platform": "youtube",
            "video_id": video_id,
            "url": f"https://www.youtube.com/watch?v={video_id}",
            "channel": {
                "id": channel_id,
                "name": channel_name,
                "url": (
                    f"https://www.youtube.com/channel/{channel_id}"
                    if channel_id
                    else None
                ),
            },
            "metadata_provider": "youtube_data_api",
            "fetched_at": "2026-07-23T00:00:00+00:00",
        },
    }


def test_profile_has_bounded_sections_and_channel_provenance_once():
    video = make_video(
        "bread123456",
        "Bread at home",
        "Kitchen Physics",
        "UC-kitchen",
        [f"sourdough fermentation step {index}" for index in range(12)],
    )
    profile = build_routing_profile("bread123456", video)

    assert len(profile["sections"]) == 5
    assert len(profile["vectors"]) == 7
    assert profile["identity_text"].count("Kitchen Physics") == 1
    assert profile["lexical_text"].count("Kitchen Physics") == 1
    assert sum(
        "Kitchen Physics" in record["text"] for record in profile["vectors"]
    ) == 1
    assert all(
        record["channel_id"] == "UC-kitchen" for record in profile["vectors"]
    )
    assert not any(record["vector_type"] == "channel" for record in profile["vectors"])
    assert all(
        re.match(
            r"Transcript section \[\d\d:\d\d-\d\d:\d\d\]:",
            section["text"],
        )
        for section in profile["sections"]
    )


def test_missing_channel_keeps_valid_profile():
    video = make_video(
        "missing12345",
        "Unknown origin",
        None,
        None,
        ["history of early computing"],
    )
    profile = build_routing_profile("missing12345", video)

    assert "Channel:" not in profile["identity_text"]
    assert profile["source"]["channel"] == {"id": None, "name": None, "url": None}
    assert all(record["channel_id"] is None for record in profile["vectors"])


def test_multi_vector_hybrid_ranking_prefers_relevant_video(tmp_path):
    videos = {
        "bread123456": make_video(
            "bread123456",
            "Bread fundamentals",
            "Kitchen School",
            "UC-kitchen",
            [
                "mixing flour and water",
                "sourdough starter fermentation timing",
                "shape and bake the loaf",
            ],
        ),
        "sport123456": make_video(
            "sport123456",
            "Basketball fundamentals",
            "Court School",
            "UC-court",
            [
                "basketball defense footwork",
                "zone defense rotations",
                "rebounding drills",
            ],
        ),
    }
    router = MultiVectorVideoRouter(
        KeywordProcessor(),
        artifact_dir=str(tmp_path),
    )
    manifest = router.build(videos)
    result = router.search("sourdough starter fermentation", top_k=2)

    assert result["video_ids"] == ["bread123456", "sport123456"]
    assert result["results"][0]["vector_hits"]
    assert result["fusion"] == "rrf"
    assert result["used_fallback"] is False
    assert manifest["vector_count"] == sum(
        len(profile["vectors"]) for profile in router.profiles.values()
    )
    assert manifest["embedding_backend"] == "test"
    assert manifest["embedding_dim"] == 9
    assert manifest["chunking_version"] == "time_v2_60s_15s"


def test_channel_name_is_low_weight_identity_and_lexical_signal(tmp_path):
    videos = {
        "physics1234": make_video(
            "physics1234",
            "Weekly lesson",
            "Physics Lab",
            "UC-physics",
            ["quantum experiments and wave functions"],
        ),
        "history1234": make_video(
            "history1234",
            "Weekly lesson",
            "History Lab",
            "UC-history",
            ["history archives and primary sources"],
        ),
    }
    router = MultiVectorVideoRouter(
        KeywordProcessor(),
        artifact_dir=str(tmp_path),
    )
    router.build(videos, persist=False)
    result = router.search("Physics Lab", top_k=1)

    assert result["video_ids"] == ["physics1234"]
    assert result["results"][0]["lexical_score"] is not None


def test_manifest_detects_source_and_embedding_invalidation(tmp_path):
    videos = {
        "bread123456": make_video(
            "bread123456",
            "Bread fundamentals",
            "Original Channel",
            "UC-stable",
            ["sourdough fermentation"],
        )
    }
    processor = KeywordProcessor()
    router = MultiVectorVideoRouter(processor, artifact_dir=str(tmp_path))
    router.build(videos)
    loaded = MultiVectorVideoRouter(processor, artifact_dir=str(tmp_path))

    assert loaded.load() is True
    assert loaded.is_stale(videos) is False

    refreshed = {"bread123456": dict(videos["bread123456"])}
    refreshed["bread123456"]["source"] = dict(videos["bread123456"]["source"])
    refreshed["bread123456"]["source"]["fetched_at"] = "2026-07-24T00:00:00+00:00"
    assert loaded.is_stale(refreshed) is False

    renamed = {"bread123456": dict(videos["bread123456"])}
    renamed["bread123456"]["source"] = dict(videos["bread123456"]["source"])
    renamed["bread123456"]["source"]["channel"] = {
        "id": "UC-stable",
        "name": "Renamed Channel",
        "url": "https://www.youtube.com/channel/UC-stable",
    }
    assert "source_fingerprint_changed" in loaded.stale_reasons(renamed)

    changed_backend = MultiVectorVideoRouter(
        KeywordProcessor(model="keyword-v2"),
        artifact_dir=str(tmp_path),
    )
    changed_backend.load()
    assert "embedding_model_changed" in changed_backend.stale_reasons(videos)


def test_router_reports_graceful_fallback_without_profiles(tmp_path):
    router = MultiVectorVideoRouter(
        KeywordProcessor(),
        artifact_dir=str(tmp_path),
    )
    result = router.search("anything")

    assert result["used_fallback"] is True
    assert result["fallback_reason"] == "no_video_profiles"


def test_japanese_bigrams_and_language_scope_route_to_frieren_video(tmp_path):
    videos = {
        "frieren1234": make_video(
            "frieren1234",
            "葬送のフリーレン トークの魔法",
            "TOHO animation",
            "UC-frieren",
            ["フリーレンの収録とキャラクターについて話しました"],
            language="ja",
        ),
        "baseball123": make_video(
            "baseball123",
            "プロ野球ニュース",
            "Sports Japan",
            "UC-sports",
            ["投手と打者が今季の試合について話しました"],
            language="ja",
        ),
    }
    router = MultiVectorVideoRouter(
        KeywordProcessor(),
        artifact_dir=str(tmp_path),
    )
    router.build(videos, persist=False)

    result = router.search(
        "フリーレンのポッドキャストでは何が話されましたか",
        top_k=2,
        language="ja",
        video_ids=["frieren1234", "baseball123"],
    )

    assert result["video_ids"][0] == "frieren1234"
    assert result["results"][0]["language"] == "ja"
    assert result["results"][0]["lexical_score"] is not None

    scoped = router.search(
        "フリーレン",
        top_k=2,
        language="ja",
        video_ids=["baseball123"],
    )
    assert scoped["video_ids"] == ["baseball123"]
