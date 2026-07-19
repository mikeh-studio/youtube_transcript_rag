"""Shared helpers for local, permissioned video OCR pipelines."""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Iterator


ROOT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_DATA_DIR = ROOT_DIR / "data"
VIDEO_ID_RE = re.compile(r"^[A-Za-z0-9_-]+$")


def now_iso() -> str:
    """Return the current UTC time in ISO-8601 format."""
    return datetime.now(timezone.utc).isoformat()


def validate_video_id(video_id: str) -> str:
    """Validate a local video identifier before using it in paths."""
    scoped = str(video_id or "").strip()
    if not scoped or not VIDEO_ID_RE.fullmatch(scoped):
        raise ValueError(
            "video_id must contain only letters, numbers, underscores, or hyphens."
        )
    return scoped


def format_timestamp(seconds: float) -> str:
    """Format seconds as HH:MM:SS for timestamp-grounded evidence."""
    total = max(0, int(float(seconds)))
    hours = total // 3600
    minutes = (total % 3600) // 60
    secs = total % 60
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def processed_dir(data_dir: Path | str, video_id: str) -> Path:
    return Path(data_dir) / "processed" / validate_video_id(video_id)


def frames_dir(data_dir: Path | str, video_id: str) -> Path:
    return Path(data_dir) / "frames" / validate_video_id(video_id)


def ocr_index_dir(data_dir: Path | str) -> Path:
    return Path(data_dir) / "index" / "ocr"


def frames_metadata_path(data_dir: Path | str, video_id: str) -> Path:
    return processed_dir(data_dir, video_id) / "frames.jsonl"


def ocr_output_path(data_dir: Path | str, video_id: str) -> Path:
    return processed_dir(data_dir, video_id) / "frame_ocr.jsonl"


def ocr_index_path(data_dir: Path | str, video_id: str) -> Path:
    return ocr_index_dir(data_dir) / f"{validate_video_id(video_id)}.faiss"


def ocr_index_metadata_path(data_dir: Path | str, video_id: str) -> Path:
    return ocr_index_dir(data_dir) / f"{validate_video_id(video_id)}.jsonl"


def ocr_index_embed_meta_path(index_path: Path | str) -> Path:
    """Sidecar JSON recording which embedding model built an OCR index."""
    return Path(index_path).with_suffix(".embed_meta.json")


def write_jsonl(path: Path | str, rows: Iterable[dict]) -> int:
    """Write dictionaries as UTF-8 JSONL and return the row count."""
    scoped_path = Path(path)
    scoped_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with scoped_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def read_jsonl(path: Path | str) -> Iterator[dict]:
    """Yield dictionaries from a UTF-8 JSONL file, skipping blank lines."""
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if not text:
                continue
            yield json.loads(text)
