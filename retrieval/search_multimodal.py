#!/usr/bin/env python3
"""Search transcript and local-video OCR evidence with one schema."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from multilingual.rag_engine import RAGEngine
from pipelines.video_ocr_common import DEFAULT_DATA_DIR, format_timestamp
from retrieval.ocr_retriever import OCREvidenceRetriever


def transcript_row_to_evidence(row: dict) -> dict:
    """Normalize a transcript search row into merged evidence format."""
    start_sec = float(row.get("start", 0.0))
    end_sec = float(row.get("end", start_sec))
    return {
        "source_type": "transcript",
        "video_id": row.get("video_id"),
        "video_title": row.get("video_title"),
        "language": row.get("language"),
        "start_sec": start_sec,
        "end_sec": end_sec,
        "start": start_sec,
        "end": end_sec,
        "timestamp_hhmmss": format_timestamp(start_sec),
        "text": row.get("text", ""),
        "score": float(row.get("score", 0.0)),
        "url": row.get("url"),
        "metadata": {
            "video_title": row.get("video_title"),
            "video_url": row.get("video_url"),
            "url": row.get("url"),
            "language": row.get("language"),
            "chunk_index": row.get("chunk_index"),
        },
    }


def ocr_row_to_evidence(row: dict) -> dict:
    """Normalize an OCR search row into merged evidence format."""
    timestamp_sec = float(row.get("timestamp_sec", 0.0))
    return {
        "source_type": "ocr",
        "video_id": row.get("video_id"),
        "timestamp_sec": timestamp_sec,
        "start": timestamp_sec,
        "timestamp_hhmmss": row.get("timestamp_hhmmss") or format_timestamp(timestamp_sec),
        "text": row.get("ocr_text", ""),
        "score": float(row.get("score", 0.0)),
        "frame_path": row.get("frame_path"),
        "frame_id": row.get("frame_id"),
        "metadata": {
            "frame_path": row.get("frame_path"),
            "frame_id": row.get("frame_id"),
            **(row.get("metadata") or {}),
        },
    }


def merge_evidence(
    *,
    transcript_results: Optional[List[dict]] = None,
    ocr_results: Optional[List[dict]] = None,
    top_k: int = 5,
) -> List[dict]:
    """Merge transcript and OCR evidence rows into one ranked list."""
    evidence: List[dict] = []
    for row in transcript_results or []:
        evidence.append(transcript_row_to_evidence(row))
    for row in ocr_results or []:
        evidence.append(ocr_row_to_evidence(row))

    evidence.sort(key=lambda row: float(row.get("score", 0.0)), reverse=True)
    for rank, row in enumerate(evidence[: max(1, int(top_k))], start=1):
        row["rank"] = rank
    return evidence[: max(1, int(top_k))]


def search_multimodal(
    *,
    query: str,
    video_id: Optional[str] = None,
    top_k: int = 5,
    language: Optional[str] = None,
    data_dir: Path | str = DEFAULT_DATA_DIR,
    include_transcript: bool = True,
    include_ocr: bool = True,
) -> dict:
    """Search transcript and OCR indexes, returning merged evidence."""
    transcript_results: List[dict] = []
    ocr_results: List[dict] = []
    candidate_k = max(int(top_k), int(top_k) * 2)
    shared_processor = None

    if include_transcript:
        try:
            engine = RAGEngine()
            shared_processor = engine.library.processor
            transcript_results = engine.search(
                query,
                k=candidate_k,
                language=language,
                video_id=video_id,
            )
        except Exception as exc:
            print(f"Warning: transcript search unavailable: {type(exc).__name__}: {exc}")
            transcript_results = []

    if include_ocr:
        try:
            retriever = OCREvidenceRetriever(data_dir=data_dir, processor=shared_processor)
            ocr_results = retriever.search(
                query,
                video_id=video_id,
                top_k=candidate_k,
                language=language,
            )
        except Exception as exc:
            print(f"Warning: OCR search unavailable: {type(exc).__name__}: {exc}")
            ocr_results = []

    evidence = merge_evidence(
        transcript_results=transcript_results,
        ocr_results=ocr_results,
        top_k=top_k,
    )
    return {
        "query": query,
        "video_id_filter": video_id,
        "top_k": int(top_k),
        "result_count": len(evidence),
        "evidence": evidence,
        "details": {
            "transcript_candidates": len(transcript_results),
            "ocr_candidates": len(ocr_results),
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Search transcript and local-video OCR evidence."
    )
    parser.add_argument("--query", required=True, help="Natural-language query.")
    parser.add_argument("--video-id", default=None, help="Optional video ID filter.")
    parser.add_argument("--top-k", type=int, default=5, help="Number of results.")
    parser.add_argument("--language", choices=["ja", "en"], default=None)
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR), help="Data root.")
    parser.add_argument(
        "--ocr-only",
        action="store_true",
        help="Search OCR evidence only.",
    )
    parser.add_argument(
        "--transcript-only",
        action="store_true",
        help="Search transcript evidence only.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = search_multimodal(
        query=args.query,
        video_id=args.video_id,
        top_k=args.top_k,
        language=args.language,
        data_dir=args.data_dir,
        include_transcript=not args.ocr_only,
        include_ocr=not args.transcript_only,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
