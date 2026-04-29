#!/usr/bin/env python3
"""Run OCR over extracted local-video frames."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from pipelines.video_ocr_common import (
    DEFAULT_DATA_DIR,
    frames_metadata_path,
    now_iso,
    ocr_output_path,
    read_jsonl,
    validate_video_id,
    write_jsonl,
)


def _load_easyocr_reader(languages: List[str]):
    try:
        import easyocr
    except Exception as exc:  # pragma: no cover - exercised manually
        raise RuntimeError(
            "EasyOCR is required for OCR extraction. Install it with "
            "`pip install easyocr` or add it to your local environment."
        ) from exc
    return easyocr.Reader(languages)


def _coerce_ocr_output(raw_output) -> Tuple[str, Optional[float]]:
    """Convert EasyOCR-like output into text and average confidence."""
    text_parts: List[str] = []
    confidences: List[float] = []

    for item in raw_output or []:
        text = ""
        confidence = None
        if isinstance(item, (list, tuple)):
            if len(item) >= 2:
                text = str(item[1] or "").strip()
            if len(item) >= 3:
                try:
                    confidence = float(item[2])
                except (TypeError, ValueError):
                    confidence = None
        else:
            text = str(item or "").strip()

        if not text:
            continue
        text_parts.append(text)
        if confidence is not None:
            confidences.append(confidence)

    if not text_parts:
        return "", None
    average_confidence = (
        sum(confidences) / len(confidences) if confidences else None
    )
    return "\n".join(text_parts), average_confidence


def build_ocr_record(
    *,
    frame_row: dict,
    ocr_text: str,
    ocr_confidence: Optional[float],
    ocr_engine: str = "easyocr",
    created_at: str | None = None,
) -> Optional[dict]:
    """Build one OCR metadata row, returning None for empty OCR text."""
    text = str(ocr_text or "").strip()
    if not text:
        return None
    return {
        "video_id": str(frame_row.get("video_id") or ""),
        "frame_id": str(frame_row.get("frame_id") or ""),
        "timestamp_sec": float(frame_row.get("timestamp_sec", 0.0)),
        "timestamp_hhmmss": str(frame_row.get("timestamp_hhmmss") or ""),
        "frame_path": str(frame_row.get("frame_path") or ""),
        "ocr_text": text,
        "ocr_confidence": ocr_confidence,
        "ocr_engine": ocr_engine,
        "created_at": created_at or now_iso(),
    }


def run_ocr(
    *,
    frames_path: Path | str,
    output_path: Path | str,
    languages: Iterable[str] = ("ja", "en"),
    reader=None,
) -> List[dict]:
    """Run OCR over frame metadata and write non-empty OCR rows."""
    scoped_frames_path = Path(frames_path)
    if not scoped_frames_path.exists():
        raise FileNotFoundError(f"Frame metadata file not found: {scoped_frames_path}")

    scoped_languages = [str(lang).strip() for lang in languages if str(lang).strip()]
    if not scoped_languages:
        scoped_languages = ["ja", "en"]
    ocr_reader = reader or _load_easyocr_reader(scoped_languages)

    rows: List[dict] = []
    created_at = now_iso()
    for idx, frame_row in enumerate(read_jsonl(scoped_frames_path), start=1):
        frame_path = Path(str(frame_row.get("frame_path") or ""))
        if not frame_path.exists():
            print(f"Warning: missing frame file, skipping: {frame_path}")
            continue
        try:
            print(f"OCR {idx}: {frame_path}")
            raw_output = ocr_reader.readtext(str(frame_path), detail=1)
            text, confidence = _coerce_ocr_output(raw_output)
            record = build_ocr_record(
                frame_row=frame_row,
                ocr_text=text,
                ocr_confidence=confidence,
                created_at=created_at,
            )
            if record is None:
                continue
            rows.append(record)
        except Exception as exc:
            print(f"Warning: OCR failed for {frame_path}: {type(exc).__name__}: {exc}")
            continue

    write_jsonl(output_path, rows)
    print(f"Wrote {len(rows)} OCR metadata rows to {output_path}")
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Japanese/English OCR over extracted local-video frames."
    )
    parser.add_argument("--video-id", required=True, help="Local video identifier.")
    parser.add_argument(
        "--frames-path",
        default=None,
        help="Frame metadata JSONL path. Default: data/processed/{video_id}/frames.jsonl.",
    )
    parser.add_argument(
        "--output-path",
        default=None,
        help="OCR JSONL path. Default: data/processed/{video_id}/frame_ocr.jsonl.",
    )
    parser.add_argument(
        "--languages",
        default="ja,en",
        help="Comma-separated EasyOCR language list. Default: ja,en.",
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
    run_ocr(
        frames_path=args.frames_path or frames_metadata_path(data_dir, video_id),
        output_path=args.output_path or ocr_output_path(data_dir, video_id),
        languages=args.languages.split(","),
    )


if __name__ == "__main__":
    main()
