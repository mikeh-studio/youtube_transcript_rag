#!/usr/bin/env python3
"""Extract timestamped frames from local, permissioned video files."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path
from typing import List

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from pipelines.video_ocr_common import (
    DEFAULT_DATA_DIR,
    format_timestamp,
    frames_dir,
    frames_metadata_path,
    now_iso,
    validate_video_id,
    write_jsonl,
)


def _require_tool(name: str) -> str:
    path = shutil.which(name)
    if not path:
        raise RuntimeError(
            f"{name} is required for frame extraction. Install ffmpeg and make sure "
            f"'{name}' is available on PATH."
        )
    return path


def _probe_duration(video_path: Path) -> float:
    ffprobe = _require_tool("ffprobe")
    result = subprocess.run(
        [
            ffprobe,
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(video_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    try:
        return max(0.0, float(result.stdout.strip()))
    except ValueError as exc:
        raise RuntimeError(f"Could not determine video duration for {video_path}") from exc


def planned_timestamps(duration_sec: float, interval_sec: int) -> List[int]:
    """Return extraction timestamps for a duration and interval."""
    if interval_sec <= 0:
        raise ValueError("interval_sec must be greater than zero.")
    if duration_sec <= 0:
        return [0]
    timestamps = list(range(0, int(duration_sec) + 1, int(interval_sec)))
    if not timestamps:
        return [0]
    return timestamps


def build_frame_record(
    *,
    video_id: str,
    frame_path: Path,
    timestamp_sec: int,
    extraction_method: str = "ffmpeg",
    created_at: str | None = None,
) -> dict:
    """Build one frame metadata row."""
    return {
        "video_id": validate_video_id(video_id),
        "frame_id": frame_path.stem,
        "timestamp_sec": int(timestamp_sec),
        "timestamp_hhmmss": format_timestamp(timestamp_sec),
        "frame_path": str(frame_path),
        "extraction_method": extraction_method,
        "created_at": created_at or now_iso(),
    }


def extract_frames(
    *,
    video_path: Path | str,
    video_id: str,
    output_dir: Path | str | None = None,
    metadata_path: Path | str | None = None,
    interval_sec: int = 10,
    data_dir: Path | str = DEFAULT_DATA_DIR,
) -> List[dict]:
    """Extract frames and write frame metadata JSONL."""
    scoped_video_id = validate_video_id(video_id)
    scoped_video_path = Path(video_path)
    if not scoped_video_path.exists():
        raise FileNotFoundError(f"Video file not found: {scoped_video_path}")
    if not scoped_video_path.is_file():
        raise ValueError(f"Video path is not a file: {scoped_video_path}")

    ffmpeg = _require_tool("ffmpeg")
    duration = _probe_duration(scoped_video_path)
    timestamps = planned_timestamps(duration, int(interval_sec))
    target_dir = Path(output_dir) if output_dir else frames_dir(data_dir, scoped_video_id)
    target_dir.mkdir(parents=True, exist_ok=True)

    records: List[dict] = []
    created_at = now_iso()
    for timestamp in timestamps:
        frame_path = target_dir / f"frame_{int(timestamp):06d}.jpg"
        command = [
            ffmpeg,
            "-y",
            "-ss",
            str(int(timestamp)),
            "-i",
            str(scoped_video_path),
            "-frames:v",
            "1",
            "-q:v",
            "2",
            str(frame_path),
        ]
        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode != 0:
            print(
                f"Warning: failed to extract frame at {timestamp}s: "
                f"{result.stderr.strip()}"
            )
            continue
        if not frame_path.exists():
            print(f"Warning: ffmpeg did not create expected frame: {frame_path}")
            continue
        records.append(
            build_frame_record(
                video_id=scoped_video_id,
                frame_path=frame_path,
                timestamp_sec=timestamp,
                created_at=created_at,
            )
        )

    target_metadata_path = (
        Path(metadata_path) if metadata_path else frames_metadata_path(data_dir, scoped_video_id)
    )
    write_jsonl(target_metadata_path, records)
    print(f"Wrote {len(records)} frame metadata rows to {target_metadata_path}")
    return records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract timestamped frames from a local, permissioned video file."
    )
    parser.add_argument("--video-id", required=True, help="Local video identifier.")
    parser.add_argument("--video-path", required=True, help="Path to a local video file.")
    parser.add_argument(
        "--interval-sec",
        type=int,
        default=10,
        help="Seconds between extracted frames. Default: 10.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Frame output directory. Default: data/frames/{video_id}.",
    )
    parser.add_argument(
        "--metadata-path",
        default=None,
        help="Frame metadata JSONL path. Default: data/processed/{video_id}/frames.jsonl.",
    )
    parser.add_argument(
        "--data-dir",
        default=str(DEFAULT_DATA_DIR),
        help="Data root. Default: data.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    extract_frames(
        video_path=args.video_path,
        video_id=args.video_id,
        output_dir=args.output_dir,
        metadata_path=args.metadata_path,
        interval_sec=args.interval_sec,
        data_dir=args.data_dir,
    )


if __name__ == "__main__":
    main()
