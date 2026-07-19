#!/usr/bin/env python3
"""Embed OCR rows into a separate local FAISS index."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional

import faiss
import numpy as np

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from multilingual.text_processing import TextProcessor
from pipelines.video_ocr_common import (
    DEFAULT_DATA_DIR,
    ocr_index_embed_meta_path,
    ocr_index_metadata_path,
    ocr_index_path,
    ocr_output_path,
    read_jsonl,
    validate_video_id,
    write_jsonl,
)


JP_CHAR_RE = __import__("re").compile(r"[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff]")


def infer_language(text: str) -> str:
    return "ja" if JP_CHAR_RE.search(text or "") else "en"


def make_embed_text(processor: TextProcessor, text: str, language: Optional[str]) -> str:
    """Build embedding text, falling back to cleaned raw OCR if tokenization fails."""
    scoped_text = processor.clean_text(text)
    if not scoped_text:
        return ""
    scoped_language = language or infer_language(scoped_text)
    try:
        return processor.make_embed_text(scoped_text, scoped_language)
    except Exception:
        return scoped_text


def build_ocr_embedding_records(
    ocr_rows: List[dict],
    *,
    processor: TextProcessor,
    language: Optional[str] = None,
) -> List[dict]:
    """Convert OCR metadata rows into index metadata records."""
    records: List[dict] = []
    for row in ocr_rows:
        text = processor.clean_text(str(row.get("ocr_text") or ""))
        if not text:
            continue
        video_id = validate_video_id(str(row.get("video_id") or ""))
        frame_id = str(row.get("frame_id") or "").strip()
        timestamp_sec = float(row.get("timestamp_sec", 0.0))
        record_id = f"{video_id}:{frame_id}:ocr"
        records.append(
            {
                "id": record_id,
                "video_id": video_id,
                "frame_id": frame_id,
                "timestamp_sec": timestamp_sec,
                "timestamp_hhmmss": str(row.get("timestamp_hhmmss") or ""),
                "text": text,
                "source_type": "ocr",
                "frame_path": str(row.get("frame_path") or ""),
                "ocr_confidence": row.get("ocr_confidence"),
                "ocr_engine": row.get("ocr_engine"),
                "embed_text": make_embed_text(processor, text, language),
            }
        )
    return records


def embed_ocr(
    *,
    video_id: str,
    ocr_path: Path | str,
    index_path: Path | str,
    metadata_path: Path | str,
    processor: Optional[TextProcessor] = None,
    language: Optional[str] = None,
) -> List[dict]:
    """Embed OCR rows and persist FAISS index plus metadata JSONL."""
    scoped_video_id = validate_video_id(video_id)
    scoped_ocr_path = Path(ocr_path)
    if not scoped_ocr_path.exists():
        raise FileNotFoundError(f"OCR metadata file not found: {scoped_ocr_path}")

    text_processor = processor or TextProcessor()
    ocr_rows = [
        row
        for row in read_jsonl(scoped_ocr_path)
        if str(row.get("video_id") or "") == scoped_video_id
    ]
    records = build_ocr_embedding_records(
        ocr_rows,
        processor=text_processor,
        language=language,
    )

    scoped_index_path = Path(index_path)
    scoped_index_path.parent.mkdir(parents=True, exist_ok=True)
    scoped_metadata_path = Path(metadata_path)
    embed_meta_path = ocr_index_embed_meta_path(scoped_index_path)
    if not records:
        write_jsonl(scoped_metadata_path, records)
        if scoped_index_path.exists():
            scoped_index_path.unlink()
        if embed_meta_path.exists():
            embed_meta_path.unlink()
        print(f"No OCR text rows found. Wrote empty metadata to {scoped_metadata_path}")
        return records

    chunks = [{"embed_text": row["embed_text"]} for row in records]
    embeddings = text_processor.generate_embeddings(chunks)
    embeddings = np.asarray(embeddings, dtype="float32")
    for idx, record in enumerate(records):
        record["embedding"] = embeddings[idx].tolist()

    write_jsonl(scoped_metadata_path, records)

    faiss.omp_set_num_threads(1)
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    faiss.write_index(index, str(scoped_index_path))
    embed_meta_path.write_text(
        json.dumps(text_processor.embedding_metadata(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Wrote {len(records)} OCR vectors to {scoped_index_path}")
    print(f"Wrote OCR index metadata to {scoped_metadata_path}")
    return records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Embed local-video OCR text.")
    parser.add_argument("--video-id", required=True, help="Local video identifier.")
    parser.add_argument(
        "--ocr-path",
        default=None,
        help="OCR JSONL path. Default: data/processed/{video_id}/frame_ocr.jsonl.",
    )
    parser.add_argument(
        "--index-path",
        default=None,
        help="FAISS output path. Default: data/index/ocr/{video_id}.faiss.",
    )
    parser.add_argument(
        "--metadata-path",
        default=None,
        help="Index metadata JSONL path. Default: data/index/ocr/{video_id}.jsonl.",
    )
    parser.add_argument(
        "--language",
        choices=["ja", "en"],
        default=None,
        help="Force a language for OCR embedding text. Default: infer per row.",
    )
    parser.add_argument(
        "--data-dir",
        default=str(DEFAULT_DATA_DIR),
        help="Data root. Default: data.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    video_id = validate_video_id(args.video_id)
    data_dir = Path(args.data_dir)
    embed_ocr(
        video_id=video_id,
        ocr_path=args.ocr_path or ocr_output_path(data_dir, video_id),
        index_path=args.index_path or ocr_index_path(data_dir, video_id),
        metadata_path=args.metadata_path or ocr_index_metadata_path(data_dir, video_id),
        language=args.language,
    )


if __name__ == "__main__":
    main()
