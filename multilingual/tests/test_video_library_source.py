"""Focused VideoLibrary provenance and routed-scope compatibility tests."""

import json

import numpy as np
import pytest

# Import the library first so its NumPy compatibility shim is installed before
# importing faiss on environments with the older NumPy package layout.
from multilingual.video_library import VideoLibrary
import faiss


class TinyProcessor:
    def embedding_metadata(self):
        return {"backend": "hashing", "model": "local_hash", "dim": 2}

    def encode_query(self, query, language=None):
        del query, language
        return np.asarray([[1.0, 0.0]], dtype="float32")

    def generate_embeddings(self, chunks):
        values = []
        for chunk in chunks:
            values.append(
                [1.0, 0.0] if "alpha" in chunk["embed_text"] else [0.0, 1.0]
            )
        return np.asarray(values, dtype="float32")


class FakeMetadataClient:
    def __init__(self):
        self.calls = []

    def fetch_many(self, video_ids):
        self.calls.append(list(video_ids))
        return {
            video_id: {
                "title": "Fetched title",
                "source": {
                    "platform": "youtube",
                    "video_id": video_id,
                    "url": f"https://www.youtube.com/watch?v={video_id}",
                    "channel": {
                        "id": "UC-stable",
                        "name": "Fetched Channel",
                        "url": "https://www.youtube.com/channel/UC-stable",
                    },
                    "metadata_provider": "youtube_data_api",
                    "fetched_at": "2026-07-23T00:00:00+00:00",
                },
            }
            for video_id in video_ids
        }


def make_record(video_id, text):
    return {
        "url": f"https://www.youtube.com/watch?v={video_id}",
        "title": f"Video {video_id}",
        "language": "en",
        "chunks": [
            {
                "start": 0.0,
                "end": 10.0,
                "raw_text": text,
                "embed_text": text,
            }
        ],
        "summary_cache": {},
        "chunking": VideoLibrary.current_chunking_metadata(),
    }


def test_legacy_load_synthesizes_source_and_round_trip_retains_it(tmp_path):
    video_id = "legacy12345"
    library_dir = tmp_path / "library"
    record_dir = library_dir / "videos" / video_id
    record_dir.mkdir(parents=True)
    (library_dir / "library.json").write_text(
        json.dumps(
            {
                "format_version": 2,
                "library_metadata": {
                    "chunking": VideoLibrary.current_chunking_metadata(),
                    "embedding": {
                        "backend": "hashing",
                        "model": "local_hash",
                        "dim": 2,
                    },
                },
                "videos": [video_id],
            }
        ),
        encoding="utf-8",
    )
    (record_dir / "record.json").write_text(
        json.dumps(make_record(video_id, "alpha topic")),
        encoding="utf-8",
    )

    library = VideoLibrary(data_dir=str(tmp_path), processor=TinyProcessor())
    source = library.videos[video_id]["source"]
    assert source["platform"] == "youtube"
    assert source["metadata_provider"] == "legacy"
    assert source["channel"] == {"id": None, "name": None, "url": None}

    library.save()
    reloaded = VideoLibrary(data_dir=str(tmp_path), processor=TinyProcessor())
    assert reloaded.videos[video_id]["source"] == source


def test_metadata_backfill_is_idempotent_and_does_not_reingest(tmp_path):
    client = FakeMetadataClient()
    library = VideoLibrary(
        data_dir=str(tmp_path),
        processor=TinyProcessor(),
        metadata_client=client,
    )
    video_id = "source12345"
    library.videos[video_id] = make_record(video_id, "alpha transcript unchanged")
    original_chunks = list(library.videos[video_id]["chunks"])

    first = library.backfill_source_metadata(persist=False)
    second = library.backfill_source_metadata(persist=False)

    assert first["updated"] == [video_id]
    assert second["updated"] == []
    assert client.calls == [[video_id]]
    assert library.videos[video_id]["chunks"] == original_chunks
    assert library.videos[video_id]["source"]["channel"]["id"] == "UC-stable"


def test_search_accepts_routed_video_ids_and_returns_normalized_source(tmp_path):
    library = VideoLibrary(data_dir=str(tmp_path), processor=TinyProcessor())
    library.videos = {
        "alpha123456": make_record("alpha123456", "alpha relevant"),
        "beta1234567": make_record("beta1234567", "beta unrelated"),
    }
    library._rebuild_chunk_map()
    vectors = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype="float32")
    library.index = faiss.IndexFlatIP(2)
    library.index.add(vectors)

    results = library.search(
        "alpha",
        k=5,
        video_ids=["alpha123456"],
    )

    assert [result["video_id"] for result in results] == ["alpha123456"]
    assert results[0]["source"]["platform"] == "youtube"
    assert results[0]["source"]["channel"]["id"] is None
    with pytest.raises(ValueError):
        library.search(
            "alpha",
            video_id="alpha123456",
            video_ids=["alpha123456"],
        )
    with pytest.raises(KeyError):
        library.search("alpha", video_ids=["missing-video"])
    with pytest.raises(TypeError):
        library.search("alpha", video_ids="alpha123456")
