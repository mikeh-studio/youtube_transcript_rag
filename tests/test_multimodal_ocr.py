"""Tests for local-video OCR pipeline and retrieval schemas."""

import importlib
import os
import sys
import threading
import time
from pathlib import Path

import faiss
import numpy as np
import pytest

from pipelines.embed_ocr import build_ocr_embedding_records, embed_ocr
from pipelines.extract_frames import build_frame_record, planned_timestamps
from pipelines.run_ocr import _coerce_ocr_output, build_ocr_record
from pipelines.video_ocr_common import (
    format_timestamp,
    ocr_index_metadata_path,
    ocr_index_path,
    ocr_output_path,
    read_jsonl,
    validate_video_id,
    write_jsonl,
)
from retrieval.ocr_retriever import OCREvidenceRetriever
from retrieval.search_multimodal import merge_evidence


ROOT_DIR = Path(__file__).resolve().parents[1]
LOCAL_PREVIEW_DIR = ROOT_DIR / "local_preview"


class FakeProcessor:
    def embedding_metadata(self):
        return {"backend": "hashing", "model": "local_hash", "dim": 2}

    def clean_text(self, text):
        return " ".join(str(text or "").split())

    def make_embed_text(self, text, language):
        return self.clean_text(text)

    def generate_embeddings(self, chunks):
        return np.asarray(
            [self._vector(chunk.get("embed_text", "")) for chunk in chunks],
            dtype="float32",
        )

    def encode_query(self, query, language=None):
        return np.asarray([self._vector(query)], dtype="float32")

    @staticmethod
    def _vector(text):
        scoped = str(text or "").lower()
        if "inflation" in scoped:
            return [1.0, 0.0]
        if "revenue" in scoped:
            return [0.0, 1.0]
        return [0.707, 0.707]


class FakeEngine:
    def __init__(self):
        self.library = type("Library", (), {"processor": FakeProcessor()})()


def _load_local_api():
    os.environ["YT_RAG_SKIP_GLOBAL_SERVICE"] = "1"
    if str(LOCAL_PREVIEW_DIR) not in sys.path:
        sys.path.insert(0, str(LOCAL_PREVIEW_DIR))
    return importlib.import_module("local_api")


def _make_local_api_service(local_api, tmp_path):
    service = local_api.LocalRAGService.__new__(local_api.LocalRAGService)
    service.engine = FakeEngine()
    service.ocr_jobs = {}
    service.ocr_lock = threading.Lock()
    service.log_lock = threading.Lock()
    service.runtime_data_dir = tmp_path / "runtime"
    service.legacy_data_dir = tmp_path / "legacy"
    service.ingest_log_path = service.runtime_data_dir / "ingest_jobs.log"
    service.legacy_ingest_log_path = service.legacy_data_dir / "ingest_jobs.log"
    return service


def test_format_timestamp_and_frame_metadata_creation(tmp_path):
    assert format_timestamp(65) == "00:01:05"
    assert format_timestamp(3661.9) == "01:01:01"
    assert planned_timestamps(21, 10) == [0, 10, 20]

    frame_path = tmp_path / "frame_000010.jpg"
    row = build_frame_record(
        video_id="demo_001",
        frame_path=frame_path,
        timestamp_sec=10,
        created_at="2026-01-01T00:00:00+00:00",
    )

    assert row == {
        "video_id": "demo_001",
        "frame_id": "frame_000010",
        "timestamp_sec": 10,
        "timestamp_hhmmss": "00:00:10",
        "frame_path": str(frame_path),
        "extraction_method": "ffmpeg",
        "created_at": "2026-01-01T00:00:00+00:00",
    }


def test_local_video_id_rejects_path_like_dot_components():
    assert validate_video_id("demo_001") == "demo_001"
    with pytest.raises(ValueError, match="video_id"):
        validate_video_id("..")
    with pytest.raises(ValueError, match="video_id"):
        validate_video_id("demo.001")


def test_ocr_empty_text_handling():
    text, confidence = _coerce_ocr_output(
        [
            ([[0, 0]], "  ", 0.9),
            ([[1, 1]], "Revenue up", 0.8),
        ]
    )
    assert text == "Revenue up"
    assert confidence == 0.8

    empty = build_ocr_record(
        frame_row={
            "video_id": "demo_001",
            "frame_id": "frame_000010",
            "timestamp_sec": 10,
            "timestamp_hhmmss": "00:00:10",
            "frame_path": "frame.jpg",
        },
        ocr_text=" ",
        ocr_confidence=None,
    )
    assert empty is None


def test_build_ocr_embedding_records_schema():
    records = build_ocr_embedding_records(
        [
            {
                "video_id": "demo_001",
                "frame_id": "frame_000010",
                "timestamp_sec": 10,
                "timestamp_hhmmss": "00:00:10",
                "frame_path": "data/frames/demo_001/frame_000010.jpg",
                "ocr_text": "Inflation expectations",
                "ocr_confidence": 0.91,
                "ocr_engine": "easyocr",
            }
        ],
        processor=FakeProcessor(),
    )

    assert records[0]["id"] == "demo_001:frame_000010:ocr"
    assert records[0]["source_type"] == "ocr"
    assert records[0]["text"] == "Inflation expectations"
    assert records[0]["frame_path"].endswith("frame_000010.jpg")


def test_embed_ocr_persists_vector_metadata(tmp_path):
    data_dir = tmp_path / "data"
    video_id = "demo_001"
    ocr_path = ocr_output_path(data_dir, video_id)
    write_jsonl(
        ocr_path,
        [
            {
                "video_id": video_id,
                "frame_id": "frame_000010",
                "timestamp_sec": 10,
                "timestamp_hhmmss": "00:00:10",
                "frame_path": "data/frames/demo_001/frame_000010.jpg",
                "ocr_text": "Inflation expectations",
                "ocr_confidence": 0.91,
                "ocr_engine": "easyocr",
            }
        ],
    )

    records = embed_ocr(
        video_id=video_id,
        ocr_path=ocr_path,
        index_path=ocr_index_path(data_dir, video_id),
        metadata_path=ocr_index_metadata_path(data_dir, video_id),
        processor=FakeProcessor(),
    )
    persisted = list(read_jsonl(ocr_index_metadata_path(data_dir, video_id)))

    assert records[0]["embedding"] == [1.0, 0.0]
    assert persisted[0]["embedding"] == [1.0, 0.0]


def test_ocr_retriever_result_schema(tmp_path):
    data_dir = tmp_path / "data"
    video_id = "demo_001"
    metadata_path = ocr_index_metadata_path(data_dir, video_id)
    index_path = ocr_index_path(data_dir, video_id)
    records = [
        {
            "id": "demo_001:frame_000010:ocr",
            "video_id": video_id,
            "frame_id": "frame_000010",
            "timestamp_sec": 10.0,
            "timestamp_hhmmss": "00:00:10",
            "text": "Inflation expectations",
            "source_type": "ocr",
            "frame_path": "data/frames/demo_001/frame_000010.jpg",
            "ocr_confidence": 0.91,
            "ocr_engine": "easyocr",
        }
    ]
    write_jsonl(metadata_path, records)

    index_path.parent.mkdir(parents=True, exist_ok=True)
    index = faiss.IndexFlatIP(2)
    index.add(np.asarray([[1.0, 0.0]], dtype="float32"))
    faiss.write_index(index, str(index_path))

    retriever = OCREvidenceRetriever(data_dir=data_dir, processor=FakeProcessor())
    results = retriever.search("inflation slide", video_id=video_id, top_k=1)

    assert results[0]["source_type"] == "ocr"
    assert results[0]["video_id"] == video_id
    assert results[0]["timestamp_sec"] == 10.0
    assert results[0]["timestamp_hhmmss"] == "00:00:10"
    assert results[0]["ocr_text"] == "Inflation expectations"
    assert results[0]["frame_path"].endswith("frame_000010.jpg")
    assert isinstance(results[0]["score"], float)


def test_merged_evidence_schema():
    evidence = merge_evidence(
        transcript_results=[
            {
                "video_id": "abc12345678",
                "start": 65,
                "end": 88,
                "text": "Transcript answer",
                "score": 0.4,
                "video_title": "Demo",
                "url": "https://www.youtube.com/watch?v=abc12345678&t=65s",
            }
        ],
        ocr_results=[
            {
                "video_id": "demo_001",
                "timestamp_sec": 10,
                "timestamp_hhmmss": "00:00:10",
                "ocr_text": "Slide answer",
                "frame_path": "frame.jpg",
                "score": 0.9,
                "source_type": "ocr",
            }
        ],
        top_k=2,
    )

    assert evidence[0]["source_type"] == "ocr"
    assert evidence[0]["timestamp_sec"] == 10
    assert evidence[0]["text"] == "Slide answer"
    assert evidence[1]["source_type"] == "transcript"
    assert evidence[1]["start_sec"] == 65
    assert evidence[1]["timestamp_hhmmss"] == "00:01:05"
    assert "metadata" in evidence[1]


def test_local_api_local_video_path_validation(tmp_path):
    local_api = _load_local_api()
    service = _make_local_api_service(local_api, tmp_path)
    video_path = tmp_path / "demo.mp4"
    text_path = tmp_path / "notes.txt"
    video_path.write_bytes(b"video placeholder")
    text_path.write_text("not a video", encoding="utf-8")

    assert service._normalize_local_video_path(str(video_path)) == video_path.resolve()

    with pytest.raises(ValueError, match="local files only"):
        service._normalize_local_video_path("https://www.youtube.com/watch?v=abc12345678")

    with pytest.raises(ValueError, match="supported video file"):
        service._normalize_local_video_path(str(text_path))


def test_local_api_starts_local_video_ocr_job(monkeypatch, tmp_path):
    local_api = _load_local_api()
    service = _make_local_api_service(local_api, tmp_path)
    video_path = tmp_path / "demo.mp4"
    video_path.write_bytes(b"not a real video because pipeline calls are mocked")

    monkeypatch.setattr(
        local_api,
        "extract_frames",
        lambda **kwargs: [{"frame_id": "frame_000000"}, {"frame_id": "frame_000010"}],
    )
    monkeypatch.setattr(
        local_api,
        "run_ocr",
        lambda **kwargs: [{"ocr_text": "Inflation expectations"}],
    )
    monkeypatch.setattr(
        local_api,
        "embed_ocr",
        lambda **kwargs: [{"id": "demo_001:frame_000000:ocr"}],
    )

    response = service.start_local_video_ocr_job(
        video_id="demo_001",
        video_path=str(video_path),
        interval_sec=10,
    )

    job_id = response["job"]["job_id"]
    for _ in range(50):
        job = service.get_ocr_job(job_id)
        if job.status in {"completed", "failed"}:
            break
        time.sleep(0.02)

    job = service.get_ocr_job(job_id)
    assert job.status == "completed"
    assert job.step == "completed"
    assert job.frame_count == 2
    assert job.ocr_count == 1
    assert job.vector_count == 1


def test_local_api_multimodal_search_merges_transcript_and_ocr(monkeypatch, tmp_path):
    local_api = _load_local_api()
    service = _make_local_api_service(local_api, tmp_path)
    retrieval_kwargs = {}

    def fake_retrieve(*args, **kwargs):
        retrieval_kwargs.update(kwargs)
        return {
            "results": [
                {
                    "video_id": "abc12345678",
                    "video_title": "Transcript Video",
                    "language": "en",
                    "start": 12,
                    "end": 22,
                    "text": "Transcript evidence",
                    "score": 0.3,
                }
            ],
            "details": {"fusion": "weighted_normalized"},
        }

    service.retrieve = fake_retrieve

    class FakeRetriever:
        def __init__(self, *args, **kwargs):
            pass

        def search(self, *args, **kwargs):
            return [
                {
                    "video_id": "demo_001",
                    "timestamp_sec": 10,
                    "timestamp_hhmmss": "00:00:10",
                    "ocr_text": "Slide evidence",
                    "frame_path": "data/frames/demo_001/frame_000010.jpg",
                    "score": 0.9,
                    "source_type": "ocr",
                }
            ]

    monkeypatch.setattr(local_api, "OCREvidenceRetriever", FakeRetriever)

    result = service.search_multimodal(
        query="inflation",
        k=2,
        retrieval_mode="hybrid",
        retrieval_profile="optimized_v1",
        source_mode="both",
    )

    assert retrieval_kwargs["retrieval_profile"] == "optimized_v1"
    assert result["result_count"] == 2
    assert result["results"][0]["source_type"] == "ocr"
    assert result["results"][0]["text"] == "Slide evidence"
    assert result["results"][1]["source_type"] == "transcript"
    assert result["details"]["ocr_candidates"] == 1
    assert result["details"]["transcript_candidates"] == 1


def test_grounded_answer_citation_payload_preserves_ocr_metadata():
    _load_local_api()
    from grounded_answer import build_citation_catalog, build_grounded_answer_messages

    citations = build_citation_catalog(
        [
            {
                "source_type": "ocr",
                "video_id": "demo_001",
                "frame_id": "frame_000010",
                "frame_path": "data/frames/demo_001/frame_000010.jpg",
                "timestamp_hhmmss": "00:00:10",
                "start": 10,
                "text": "Inflation expectations",
                "score": 0.9,
            }
        ]
    )

    assert citations[0]["source_type"] == "ocr"
    assert citations[0]["frame_id"] == "frame_000010"
    assert citations[0]["frame_path"].endswith("frame_000010.jpg")
    assert citations[0]["timestamp_range_label"] == "00:00:10"
    assert citations[0]["url"] == ""

    system_prompt, user_message = build_grounded_answer_messages(
        question="What does the slide say?",
        citations=citations,
        answer_language="en",
    )
    assert "transcript and OCR evidence" in system_prompt
    assert "Frame: data/frames/demo_001/frame_000010.jpg" in user_message
    assert "OCR: Inflation expectations" in user_message


def test_embed_ocr_writes_embedding_sidecar(tmp_path):
    import json

    from pipelines.video_ocr_common import ocr_index_embed_meta_path

    data_dir = tmp_path / "data"
    video_id = "demo_001"
    ocr_path = ocr_output_path(data_dir, video_id)
    write_jsonl(
        ocr_path,
        [
            {
                "video_id": video_id,
                "frame_id": "frame_000010",
                "timestamp_sec": 10,
                "timestamp_hhmmss": "00:00:10",
                "frame_path": "data/frames/demo_001/frame_000010.jpg",
                "ocr_text": "Inflation expectations",
                "ocr_confidence": 0.91,
                "ocr_engine": "easyocr",
            }
        ],
    )

    index_path = ocr_index_path(data_dir, video_id)
    embed_ocr(
        video_id=video_id,
        ocr_path=ocr_path,
        index_path=index_path,
        metadata_path=ocr_index_metadata_path(data_dir, video_id),
        processor=FakeProcessor(),
    )

    sidecar = ocr_index_embed_meta_path(index_path)
    assert sidecar.exists()
    assert json.loads(sidecar.read_text()) == {
        "backend": "hashing",
        "model": "local_hash",
        "dim": 2,
    }


def _write_ocr_index_fixture(data_dir, video_id):
    metadata_path = ocr_index_metadata_path(data_dir, video_id)
    index_path = ocr_index_path(data_dir, video_id)
    write_jsonl(
        metadata_path,
        [
            {
                "id": f"{video_id}:frame_000010:ocr",
                "video_id": video_id,
                "frame_id": "frame_000010",
                "timestamp_sec": 10.0,
                "timestamp_hhmmss": "00:00:10",
                "text": "Inflation expectations",
                "source_type": "ocr",
                "frame_path": f"data/frames/{video_id}/frame_000010.jpg",
                "ocr_confidence": 0.91,
                "ocr_engine": "easyocr",
            }
        ],
    )
    index_path.parent.mkdir(parents=True, exist_ok=True)
    index = faiss.IndexFlatIP(2)
    index.add(np.asarray([[1.0, 0.0]], dtype="float32"))
    faiss.write_index(index, str(index_path))
    return index_path


def test_ocr_retriever_skips_sidecar_model_mismatch(tmp_path, capsys):
    import json

    from pipelines.video_ocr_common import ocr_index_embed_meta_path

    data_dir = tmp_path / "data"
    video_id = "demo_001"
    index_path = _write_ocr_index_fixture(data_dir, video_id)
    ocr_index_embed_meta_path(index_path).write_text(
        json.dumps(
            {
                "backend": "sentence_transformers",
                "model": "intfloat/multilingual-e5-base",
                "dim": 2,
            }
        )
    )

    retriever = OCREvidenceRetriever(data_dir=data_dir, processor=FakeProcessor())
    results = retriever.search("inflation slide", video_id=video_id, top_k=1)

    assert results == []
    assert "Skipping" in capsys.readouterr().out


def test_ocr_retriever_skips_legacy_dim_mismatch(tmp_path, capsys):
    data_dir = tmp_path / "data"
    video_id = "demo_001"
    # Legacy index (no sidecar) with dim 3 vs processor dim 2.
    metadata_path = ocr_index_metadata_path(data_dir, video_id)
    index_path = ocr_index_path(data_dir, video_id)
    write_jsonl(metadata_path, [{"id": "x", "video_id": video_id, "text": "t"}])
    index_path.parent.mkdir(parents=True, exist_ok=True)
    index = faiss.IndexFlatIP(3)
    index.add(np.asarray([[1.0, 0.0, 0.0]], dtype="float32"))
    faiss.write_index(index, str(index_path))

    retriever = OCREvidenceRetriever(data_dir=data_dir, processor=FakeProcessor())
    results = retriever.search("inflation slide", video_id=video_id, top_k=1)

    assert results == []
    assert "Skipping" in capsys.readouterr().out


def test_ocr_retriever_skips_sidecar_index_dim_mismatch(tmp_path, capsys):
    import json

    from pipelines.video_ocr_common import ocr_index_embed_meta_path

    data_dir = tmp_path / "data"
    video_id = "demo_001"
    index_path = _write_ocr_index_fixture(data_dir, video_id)

    # The sidecar matches the processor, but the FAISS file itself is stale.
    stale = faiss.IndexFlatIP(3)
    stale.add(np.asarray([[1.0, 0.0, 0.0]], dtype="float32"))
    faiss.write_index(stale, str(index_path))
    ocr_index_embed_meta_path(index_path).write_text(
        json.dumps({"backend": "hashing", "model": "local_hash", "dim": 2})
    )

    retriever = OCREvidenceRetriever(data_dir=data_dir, processor=FakeProcessor())
    results = retriever.search("inflation slide", video_id=video_id, top_k=1)

    assert results == []
    assert "FAISS index dimension is 3" in capsys.readouterr().out


def test_ocr_retriever_skips_vector_metadata_count_mismatch(tmp_path, capsys):
    import json

    from pipelines.video_ocr_common import ocr_index_embed_meta_path

    data_dir = tmp_path / "data"
    video_id = "demo_001"
    index_path = _write_ocr_index_fixture(data_dir, video_id)

    stale = faiss.IndexFlatIP(2)
    stale.add(np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype="float32"))
    faiss.write_index(stale, str(index_path))
    ocr_index_embed_meta_path(index_path).write_text(
        json.dumps({"backend": "hashing", "model": "local_hash", "dim": 2})
    )

    retriever = OCREvidenceRetriever(data_dir=data_dir, processor=FakeProcessor())
    results = retriever.search("inflation slide", video_id=video_id, top_k=1)

    assert results == []
    assert "2 vectors but 1 metadata rows" in capsys.readouterr().out


def test_ocr_retriever_accepts_matching_sidecar(tmp_path):
    import json

    from pipelines.video_ocr_common import ocr_index_embed_meta_path

    data_dir = tmp_path / "data"
    video_id = "demo_001"
    index_path = _write_ocr_index_fixture(data_dir, video_id)
    ocr_index_embed_meta_path(index_path).write_text(
        json.dumps({"backend": "hashing", "model": "local_hash", "dim": 2})
    )

    retriever = OCREvidenceRetriever(data_dir=data_dir, processor=FakeProcessor())
    results = retriever.search("inflation slide", video_id=video_id, top_k=1)

    assert len(results) == 1
    assert results[0]["ocr_text"] == "Inflation expectations"
