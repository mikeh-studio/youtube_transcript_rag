#!/usr/bin/env python3
"""Curate already-ingested transcript chunks into local evidence records.

Version 1A is intentionally deterministic and local-first: it reads transcript
chunks already stored by ``VideoLibrary``, derives quality/topic signals with
simple heuristics, and writes traceable JSON/JSONL artifacts under
``data/runtime``. It does not call LLM providers or fetch new source media.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
import sys
import time
import uuid
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Iterator, Optional


ROOT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_DATA_DIR = ROOT_DIR / "data"
PIPELINE_NAME = "transcript_evidence_curation"
PIPELINE_VERSION = "v1"
MODEL_NAME = "heuristic_quality_scorer"
MODEL_VERSION = "v1"
INFERENCE_TYPE = "heuristic_quality_scoring"

SOURCE_TYPE_TRANSCRIPT = "transcript"
QUALITY_LABEL_HIGH = "high_signal"
QUALITY_LABEL_MEDIUM = "medium_signal"
QUALITY_LABEL_LOW = "low_signal"
QUALITY_LABEL_INVALID = "invalid"
ELIGIBLE_LABELS = {QUALITY_LABEL_HIGH, QUALITY_LABEL_MEDIUM}

MEANINGFUL_CHAR_RE = re.compile(
    r"[A-Za-z0-9\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff]"
)
JAPANESE_CHAR_RE = re.compile(r"[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff]")
LATIN_CHAR_RE = re.compile(r"[A-Za-z]")
WORD_RE = re.compile(r"[A-Za-z0-9_]+|[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff]+")

BLOCKING_REASONS = {
    "empty_text",
    "too_short",
    "too_long",
    "missing_timestamp",
    "mostly_punctuation",
    "low_meaningful_text",
    "too_repetitive",
    "wrong_language",
}

TOPIC_KEYWORDS = {
    "anime": (
        "anime",
        "animation",
        "manga",
        "character",
        "episode",
        "voice actor",
        "アニメ",
        "漫画",
        "マンガ",
        "キャラ",
        "声優",
        "作品",
        "魔法",
        "フリーレン",
        "放送",
        "話",
    ),
    "business": (
        "business",
        "startup",
        "company",
        "customer",
        "revenue",
        "sales",
        "market",
        "企業",
        "会社",
        "ビジネス",
        "顧客",
        "売上",
        "経営",
        "営業",
    ),
    "finance": (
        "finance",
        "stock",
        "stocks",
        "inflation",
        "interest",
        "investor",
        "bank",
        "crypto",
        "金融",
        "株",
        "投資",
        "銀行",
        "金利",
        "インフレ",
        "為替",
        "資産",
    ),
    "technology": (
        "technology",
        "software",
        "data",
        "model",
        "api",
        "cloud",
        "computer",
        "ai",
        "技術",
        "データ",
        "ソフトウェア",
        "モデル",
        "API",
        "クラウド",
        "開発",
        "人工知能",
    ),
    "history": (
        "history",
        "war",
        "ancient",
        "empire",
        "century",
        "歴史",
        "戦争",
        "時代",
        "江戸",
        "明治",
        "古代",
        "世界史",
    ),
    "philosophy": (
        "philosophy",
        "ethics",
        "meaning",
        "mind",
        "thought",
        "哲学",
        "倫理",
        "意味",
        "思想",
        "心",
        "人生",
    ),
    "sports": (
        "sports",
        "baseball",
        "soccer",
        "basketball",
        "game",
        "team",
        "スポーツ",
        "野球",
        "サッカー",
        "バスケ",
        "試合",
        "選手",
        "チーム",
    ),
}


class MetadataOnlyProcessor:
    """Placeholder used to load VideoLibrary metadata without model startup."""


def now_iso() -> str:
    """Return a UTC ISO-8601 timestamp."""
    return datetime.now(timezone.utc).isoformat()


def make_pipeline_run_id(started_at: Optional[datetime] = None) -> str:
    """Create a unique run id that is sortable by start time."""
    scoped_started_at = started_at or datetime.now(timezone.utc)
    stamp = scoped_started_at.strftime("%Y%m%dT%H%M%SZ")
    return f"{PIPELINE_NAME}_{PIPELINE_VERSION}_{stamp}_{uuid.uuid4().hex[:8]}"


def read_jsonl(path: Path | str) -> Iterator[dict]:
    """Yield JSON objects from a JSONL file, skipping blank lines."""
    scoped_path = Path(path)
    with scoped_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped:
                yield json.loads(stripped)


def write_jsonl(path: Path | str, rows: Iterable[dict]) -> int:
    """Atomically write JSONL rows and return the row count."""
    scoped_path = Path(path)
    scoped_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = scoped_path.with_name(f"{scoped_path.name}.tmp")
    count = 0
    with tmp_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            count += 1
    tmp_path.replace(scoped_path)
    return count


def append_jsonl(path: Path | str, rows: Iterable[dict]) -> int:
    """Append JSONL rows and return the row count."""
    scoped_path = Path(path)
    scoped_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with scoped_path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            count += 1
    return count


def write_json(path: Path | str, payload: dict) -> None:
    """Atomically write one JSON object."""
    scoped_path = Path(path)
    scoped_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = scoped_path.with_name(f"{scoped_path.name}.tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    tmp_path.replace(scoped_path)


def runtime_paths(data_dir: Path | str) -> dict:
    """Return output artifact paths for a data directory."""
    runtime_dir = Path(data_dir) / "runtime"
    return {
        "pipeline_runs": runtime_dir / "pipeline_runs.jsonl",
        "model_inference_results": runtime_dir / "model_inference_results.jsonl",
        "manifest": runtime_dir / "curated_evidence_manifest.jsonl",
        "quality_report": runtime_dir / "evidence_quality_report.json",
    }


def load_video_library(data_dir: Path | str):
    """Load the existing VideoLibrary without initializing embedding models."""
    if str(ROOT_DIR) not in sys.path:
        sys.path.insert(0, str(ROOT_DIR))

    from multilingual.video_library import VideoLibrary

    return VideoLibrary(data_dir=str(Path(data_dir)), processor=MetadataOnlyProcessor())


def clean_text(text: object) -> str:
    """Normalize whitespace for curation without changing content semantics."""
    return re.sub(r"\s+", " ", str(text or "").replace("\u3000", " ")).strip()


def safe_float(value: object) -> Optional[float]:
    """Return a finite float or None."""
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(result):
        return None
    return result


def estimate_tokens(text: str, language: Optional[str] = None) -> int:
    """Estimate token count with simple local heuristics."""
    scoped_text = clean_text(text)
    if not scoped_text:
        return 0
    if language == "ja" or JAPANESE_CHAR_RE.search(scoped_text):
        non_space = re.sub(r"\s+", "", scoped_text)
        return max(1, math.ceil(len(non_space) / 2))
    words = WORD_RE.findall(scoped_text)
    if words:
        return len(words)
    return max(1, math.ceil(len(scoped_text) / 4))


def meaningful_char_count(text: str) -> int:
    """Count letters, digits, and CJK/kana characters."""
    return len(MEANINGFUL_CHAR_RE.findall(text or ""))


def language_match_score(text: str, language: Optional[str]) -> tuple[bool, dict]:
    """Return whether text roughly matches the requested language."""
    scoped_language = str(language or "").strip().lower()
    if not scoped_language:
        return True, {"language_check": "not_requested"}

    meaningful = max(1, meaningful_char_count(text))
    japanese_chars = len(JAPANESE_CHAR_RE.findall(text or ""))
    latin_chars = len(LATIN_CHAR_RE.findall(text or ""))
    japanese_ratio = japanese_chars / meaningful
    latin_ratio = latin_chars / meaningful

    if scoped_language.startswith("ja"):
        matches = japanese_ratio >= 0.20
    elif scoped_language.startswith("en"):
        matches = latin_ratio >= 0.45 and japanese_ratio < 0.20
    else:
        matches = True

    return matches, {
        "language_check": scoped_language,
        "japanese_char_ratio": round(japanese_ratio, 4),
        "latin_char_ratio": round(latin_ratio, 4),
    }


def repetition_score(text: str) -> tuple[bool, dict]:
    """Detect chunks dominated by repeated characters or repeated tokens."""
    scoped_text = clean_text(text).lower()
    meaningful_chars = MEANINGFUL_CHAR_RE.findall(scoped_text)
    if not meaningful_chars:
        return False, {
            "max_char_frequency_ratio": 1.0,
            "unique_token_ratio": 0.0,
        }

    max_char_frequency_ratio = Counter(meaningful_chars).most_common(1)[0][1] / len(
        meaningful_chars
    )
    tokens = WORD_RE.findall(scoped_text)
    unique_token_ratio = len(set(tokens)) / len(tokens) if tokens else 1.0
    repetitive = max_char_frequency_ratio >= 0.55 or (
        len(tokens) >= 8 and unique_token_ratio <= 0.25
    )
    return not repetitive, {
        "max_char_frequency_ratio": round(max_char_frequency_ratio, 4),
        "unique_token_ratio": round(unique_token_ratio, 4),
    }


def score_evidence_quality(
    *,
    text: str,
    start_sec: Optional[float],
    end_sec: Optional[float],
    language: Optional[str] = None,
    target_language: Optional[str] = None,
) -> dict:
    """Score one transcript chunk with deterministic local heuristics."""
    scoped_text = clean_text(text)
    text_length = len(scoped_text)
    meaningful = meaningful_char_count(scoped_text)
    has_timestamp = (
        start_sec is not None
        and end_sec is not None
        and start_sec >= 0
        and end_sec > start_sec
    )

    if not scoped_text:
        return {
            "score": 0.0,
            "label": QUALITY_LABEL_INVALID,
            "reasons": ["empty_text"],
            "signals": {
                "text_length": 0,
                "meaningful_char_count": 0,
                "has_timestamp": has_timestamp,
            },
        }

    meaningful_ratio = meaningful / max(1, text_length)
    mostly_punctuation = meaningful_ratio < 0.35
    too_short = meaningful < 12 or text_length < 20
    too_long = text_length > 4000
    useful_length = 40 <= text_length <= 1800
    pass_repetition, repetition_signals = repetition_score(scoped_text)

    scoped_target_language = target_language or language
    language_matches, language_signals = language_match_score(
        scoped_text, scoped_target_language
    )

    reasons = []
    if not has_timestamp:
        reasons.append("missing_timestamp")
    if too_short:
        reasons.append("too_short")
    if too_long:
        reasons.append("too_long")
    if mostly_punctuation:
        reasons.append("mostly_punctuation")
    if meaningful < 12:
        reasons.append("low_meaningful_text")
    if not pass_repetition:
        reasons.append("too_repetitive")
    if not language_matches:
        reasons.append("wrong_language")

    score = 0.0
    score += 0.15  # non-empty text
    score += 0.15 if useful_length else 0.07
    score += 0.15 if has_timestamp else 0.0
    score += 0.15 if meaningful_ratio >= 0.55 else (0.07 if meaningful_ratio >= 0.35 else 0.0)
    score += 0.15 if meaningful >= 30 else (0.05 if meaningful >= 12 else 0.0)
    score += 0.15 if pass_repetition else 0.0
    score += 0.10 if language_matches else 0.0
    if too_long:
        score -= 0.15

    score = max(0.0, min(1.0, round(score, 4)))
    if score >= 0.80:
        label = QUALITY_LABEL_HIGH
    elif score >= 0.60:
        label = QUALITY_LABEL_MEDIUM
    elif score >= 0.30:
        label = QUALITY_LABEL_LOW
    else:
        label = QUALITY_LABEL_INVALID

    signals = {
        "text_length": text_length,
        "meaningful_char_count": meaningful,
        "meaningful_char_ratio": round(meaningful_ratio, 4),
        "has_timestamp": has_timestamp,
        "useful_length": useful_length,
        "too_short": too_short,
        "too_long": too_long,
        "mostly_punctuation": mostly_punctuation,
        "language": language,
        "target_language": target_language,
        **repetition_signals,
        **language_signals,
    }
    return {
        "score": score,
        "label": label,
        "reasons": reasons,
        "signals": signals,
    }


def determine_eligibility(
    quality_result: dict,
    *,
    min_quality_score: float,
) -> tuple[bool, str, str]:
    """Translate quality signals into retrieval eligibility."""
    score = float(quality_result.get("score") or 0.0)
    label = str(quality_result.get("label") or QUALITY_LABEL_INVALID)
    reasons = list(quality_result.get("reasons") or [])
    blocking = [reason for reason in reasons if reason in BLOCKING_REASONS]

    if label in ELIGIBLE_LABELS and score >= min_quality_score and not blocking:
        return True, "quality_threshold_met", ""
    if blocking:
        return False, "", ";".join(blocking)
    if score < min_quality_score:
        return False, "", "below_quality_threshold"
    return False, "", "low_quality"


def tag_topics(text: str, language: Optional[str] = None) -> list[str]:
    """Assign simple deterministic topic tags by keyword matching."""
    scoped_text = clean_text(text)
    if not scoped_text:
        return ["unknown"]

    lower_text = scoped_text.lower()
    matched = []
    for tag, keywords in TOPIC_KEYWORDS.items():
        if any(str(keyword).lower() in lower_text for keyword in keywords):
            matched.append(tag)

    if matched:
        return matched
    if meaningful_char_count(scoped_text) > 0:
        return ["general"]
    return ["unknown"]


def make_evidence_id(
    dataset_id: str,
    dataset_version: str,
    video_id: str,
    chunk_index: int,
) -> str:
    """Create a stable evidence id for a dataset/video/chunk tuple."""
    return f"{dataset_id}:{dataset_version}:{video_id}:{SOURCE_TYPE_TRANSCRIPT}:{chunk_index:06d}"


def iter_transcript_chunks(
    library,
    *,
    video_id: Optional[str] = None,
    language: Optional[str] = None,
    limit: Optional[int] = None,
) -> Iterator[dict]:
    """Yield normalized transcript chunk records from a loaded library."""
    scoped_video_id = str(video_id or "").strip()
    scoped_language = str(language or "").strip()
    count = 0
    if limit is not None and limit <= 0:
        return

    videos = getattr(library, "videos", {}) or {}
    if scoped_video_id and scoped_video_id not in videos:
        raise KeyError(f"Video {scoped_video_id} not found in library.")

    for current_video_id, video in videos.items():
        if scoped_video_id and current_video_id != scoped_video_id:
            continue
        video_language = str(video.get("language") or "").strip()
        if scoped_language and video_language and video_language != scoped_language:
            continue

        for chunk_index, chunk in enumerate(video.get("chunks") or []):
            start_sec = safe_float(chunk.get("start"))
            end_sec = safe_float(chunk.get("end"))
            yield {
                "video_id": current_video_id,
                "video_title": str(video.get("title") or f"Video {current_video_id}"),
                "video_url": str(
                    video.get("url")
                    or f"https://www.youtube.com/watch?v={current_video_id}"
                ),
                "language": video_language or scoped_language or "",
                "chunk_index": int(chunk_index),
                "start_sec": start_sec,
                "end_sec": end_sec,
                "text": clean_text(chunk.get("raw_text") or chunk.get("text") or ""),
            }
            count += 1
            if limit is not None and count >= limit:
                return


def build_evidence_record(
    chunk: dict,
    *,
    dataset_id: str,
    dataset_version: str,
    pipeline_run_id: str,
    target_language: Optional[str],
    min_quality_score: float,
    created_at: str,
) -> dict:
    """Create one curated evidence record from a transcript chunk."""
    text = clean_text(chunk.get("text"))
    language = str(chunk.get("language") or "").strip()
    start_sec = chunk.get("start_sec")
    end_sec = chunk.get("end_sec")
    chunk_index = int(chunk.get("chunk_index") or 0)
    video_id = str(chunk.get("video_id") or "").strip()

    quality_result = score_evidence_quality(
        text=text,
        start_sec=start_sec,
        end_sec=end_sec,
        language=language,
        target_language=target_language,
    )
    retrieval_eligible, inclusion_reason, exclusion_reason = determine_eligibility(
        quality_result,
        min_quality_score=min_quality_score,
    )
    topics = tag_topics(text, language=language)

    record = {
        "evidence_id": make_evidence_id(
            dataset_id, dataset_version, video_id, chunk_index
        ),
        "dataset_id": dataset_id,
        "dataset_version": dataset_version,
        "pipeline_run_id": pipeline_run_id,
        "video_id": video_id,
        "video_title": str(chunk.get("video_title") or f"Video {video_id}"),
        "source_type": SOURCE_TYPE_TRANSCRIPT,
        "chunk_index": chunk_index,
        "start_sec": start_sec,
        "end_sec": end_sec,
        "text": text,
        "language": language,
        "text_length": len(text),
        "token_estimate": estimate_tokens(text, language=language),
        "has_timestamp": bool(
            start_sec is not None
            and end_sec is not None
            and start_sec >= 0
            and end_sec > start_sec
        ),
        "quality_score": quality_result["score"],
        "quality_label": quality_result["label"],
        "topic_tags": topics,
        "retrieval_eligible": retrieval_eligible,
        "inclusion_reason": inclusion_reason,
        "exclusion_reason": exclusion_reason,
        "created_at": created_at,
        "_quality_result": quality_result,
    }
    return record


def build_manifest_row(evidence_record: dict) -> dict:
    """Build the JSONL manifest row for one evidence record."""
    return {
        "dataset_id": evidence_record["dataset_id"],
        "dataset_version": evidence_record["dataset_version"],
        "pipeline_run_id": evidence_record["pipeline_run_id"],
        "evidence_id": evidence_record["evidence_id"],
        "video_id": evidence_record["video_id"],
        "video_title": evidence_record["video_title"],
        "source_type": evidence_record["source_type"],
        "chunk_index": evidence_record["chunk_index"],
        "start_sec": evidence_record["start_sec"],
        "end_sec": evidence_record["end_sec"],
        "text": evidence_record["text"],
        "language": evidence_record["language"],
        "text_length": evidence_record["text_length"],
        "token_estimate": evidence_record["token_estimate"],
        "has_timestamp": evidence_record["has_timestamp"],
        "quality_score": evidence_record["quality_score"],
        "quality_label": evidence_record["quality_label"],
        "topic_tags": evidence_record["topic_tags"],
        "included": evidence_record["retrieval_eligible"],
        "inclusion_reason": evidence_record["inclusion_reason"],
        "exclusion_reason": evidence_record["exclusion_reason"],
        "created_at": evidence_record["created_at"],
    }


def build_model_inference_row(evidence_record: dict, *, created_at: str) -> dict:
    """Represent the heuristic scorer as traceable model-inference metadata."""
    quality_result = evidence_record.get("_quality_result") or {}
    output_json = {
        "quality_score": evidence_record["quality_score"],
        "quality_label": evidence_record["quality_label"],
        "retrieval_eligible": evidence_record["retrieval_eligible"],
        "exclusion_reason": evidence_record["exclusion_reason"],
        "topic_tags": evidence_record["topic_tags"],
        "signals": quality_result.get("signals", {}),
        "reasons": quality_result.get("reasons", []),
    }
    return {
        "inference_run_id": f"{evidence_record['pipeline_run_id']}:{evidence_record['evidence_id']}",
        "pipeline_run_id": evidence_record["pipeline_run_id"],
        "evidence_id": evidence_record["evidence_id"],
        "source_type": evidence_record["source_type"],
        "model_name": MODEL_NAME,
        "model_version": MODEL_VERSION,
        "inference_type": INFERENCE_TYPE,
        "status": "completed",
        "score": evidence_record["quality_score"],
        "label": evidence_record["quality_label"],
        "output_json": output_json,
        "fallback_used": False,
        "error_message": "",
        "created_at": created_at,
    }


def summarize_quality(
    evidence_records: list[dict],
    *,
    pipeline_run_id: str,
    dataset_id: str,
    dataset_version: str,
    generated_at: str,
) -> dict:
    """Summarize quality and eligibility for a curation run."""
    total_records = len(evidence_records)
    eligible_records = sum(1 for row in evidence_records if row["retrieval_eligible"])
    excluded_records = total_records - eligible_records
    scores = [float(row["quality_score"]) for row in evidence_records]
    topic_counter = Counter(
        tag for row in evidence_records for tag in row.get("topic_tags", [])
    )
    label_counter = Counter(row["quality_label"] for row in evidence_records)

    return {
        "pipeline_run_id": pipeline_run_id,
        "dataset_id": dataset_id,
        "dataset_version": dataset_version,
        "total_records": total_records,
        "eligible_records": eligible_records,
        "excluded_records": excluded_records,
        "eligibility_rate": round(eligible_records / total_records, 4)
        if total_records
        else 0.0,
        "quality_label_counts": dict(sorted(label_counter.items())),
        "topic_tag_counts": dict(sorted(topic_counter.items())),
        "avg_quality_score": round(statistics.fmean(scores), 4) if scores else 0.0,
        "min_quality_score": round(min(scores), 4) if scores else 0.0,
        "max_quality_score": round(max(scores), 4) if scores else 0.0,
        "records_missing_timestamp": sum(
            1 for row in evidence_records if not row["has_timestamp"]
        ),
        "records_empty_text": sum(1 for row in evidence_records if not row["text"]),
        "generated_at": generated_at,
    }


def build_pipeline_run_row(
    *,
    pipeline_run_id: str,
    dataset_id: str,
    dataset_version: str,
    video_id_filter: Optional[str],
    language_filter: Optional[str],
    input_record_count: int,
    output_record_count: int,
    eligible_record_count: int,
    excluded_record_count: int,
    failed_record_count: int,
    status: str,
    started_at: str,
    finished_at: str,
    duration_ms: int,
    config_json: dict,
) -> dict:
    """Build the append-only pipeline run metadata row."""
    return {
        "pipeline_run_id": pipeline_run_id,
        "pipeline_name": PIPELINE_NAME,
        "pipeline_version": PIPELINE_VERSION,
        "dataset_id": dataset_id,
        "dataset_version": dataset_version,
        "video_id_filter": video_id_filter or "",
        "language_filter": language_filter or "",
        "input_record_count": input_record_count,
        "output_record_count": output_record_count,
        "eligible_record_count": eligible_record_count,
        "excluded_record_count": excluded_record_count,
        "failed_record_count": failed_record_count,
        "status": status,
        "started_at": started_at,
        "finished_at": finished_at,
        "duration_ms": duration_ms,
        "config_json": config_json,
    }


def run_curation_pipeline(
    *,
    library,
    dataset_id: str,
    dataset_version: str,
    data_dir: Path | str = DEFAULT_DATA_DIR,
    language: Optional[str] = None,
    video_id: Optional[str] = None,
    limit: Optional[int] = None,
    min_quality_score: float = 0.6,
    dry_run: bool = False,
) -> dict:
    """Curate transcript chunks and optionally persist local runtime artifacts."""
    if not dataset_id:
        raise ValueError("dataset_id is required.")
    if not dataset_version:
        raise ValueError("dataset_version is required.")
    if limit is not None and limit < 0:
        raise ValueError("limit must be non-negative.")
    if min_quality_score < 0 or min_quality_score > 1:
        raise ValueError("min_quality_score must be between 0 and 1.")

    started_dt = datetime.now(timezone.utc)
    started_at = started_dt.isoformat()
    monotonic_started = time.perf_counter()
    pipeline_run_id = make_pipeline_run_id(started_dt)
    created_at = started_at
    target_language = str(language or "").strip() or None

    chunks = list(
        iter_transcript_chunks(
            library,
            video_id=video_id,
            language=language,
            limit=limit,
        )
    )
    evidence_records: list[dict] = []
    manifest_rows: list[dict] = []
    inference_rows: list[dict] = []
    failed_record_count = 0

    for chunk in chunks:
        try:
            evidence_record = build_evidence_record(
                chunk,
                dataset_id=dataset_id,
                dataset_version=dataset_version,
                pipeline_run_id=pipeline_run_id,
                target_language=target_language,
                min_quality_score=min_quality_score,
                created_at=created_at,
            )
            evidence_records.append(evidence_record)
            manifest_rows.append(build_manifest_row(evidence_record))
            inference_rows.append(
                build_model_inference_row(evidence_record, created_at=created_at)
            )
        except Exception as exc:
            failed_record_count += 1
            chunk_video_id = str(chunk.get("video_id") or "unknown")
            chunk_index = int(chunk.get("chunk_index") or 0)
            evidence_id = make_evidence_id(
                dataset_id, dataset_version, chunk_video_id, chunk_index
            )
            inference_rows.append(
                {
                    "inference_run_id": f"{pipeline_run_id}:{evidence_id}",
                    "pipeline_run_id": pipeline_run_id,
                    "evidence_id": evidence_id,
                    "source_type": SOURCE_TYPE_TRANSCRIPT,
                    "model_name": MODEL_NAME,
                    "model_version": MODEL_VERSION,
                    "inference_type": INFERENCE_TYPE,
                    "status": "failed",
                    "score": 0.0,
                    "label": QUALITY_LABEL_INVALID,
                    "output_json": {},
                    "fallback_used": False,
                    "error_message": f"{type(exc).__name__}: {exc}",
                    "created_at": created_at,
                }
            )

    finished_at = now_iso()
    duration_ms = int((time.perf_counter() - monotonic_started) * 1000)
    eligible_record_count = sum(1 for row in evidence_records if row["retrieval_eligible"])
    excluded_record_count = len(evidence_records) - eligible_record_count
    status = "completed" if failed_record_count == 0 else "completed_with_errors"

    config_json = {
        "data_dir": str(Path(data_dir)),
        "dataset_id": dataset_id,
        "dataset_version": dataset_version,
        "video_id": video_id or "",
        "language": language or "",
        "limit": limit,
        "min_quality_score": min_quality_score,
        "dry_run": dry_run,
        "manifest_write_mode": "overwrite_current_run",
        "llm_provider_calls": False,
    }
    report = summarize_quality(
        evidence_records,
        pipeline_run_id=pipeline_run_id,
        dataset_id=dataset_id,
        dataset_version=dataset_version,
        generated_at=finished_at,
    )
    pipeline_run_row = build_pipeline_run_row(
        pipeline_run_id=pipeline_run_id,
        dataset_id=dataset_id,
        dataset_version=dataset_version,
        video_id_filter=video_id,
        language_filter=language,
        input_record_count=len(chunks),
        output_record_count=len(evidence_records),
        eligible_record_count=eligible_record_count,
        excluded_record_count=excluded_record_count,
        failed_record_count=failed_record_count,
        status=status,
        started_at=started_at,
        finished_at=finished_at,
        duration_ms=duration_ms,
        config_json=config_json,
    )

    paths = runtime_paths(data_dir)
    if not dry_run:
        write_jsonl(paths["manifest"], manifest_rows)
        append_jsonl(paths["model_inference_results"], inference_rows)
        write_json(paths["quality_report"], report)
        append_jsonl(paths["pipeline_runs"], [pipeline_run_row])

    return {
        "pipeline_run": pipeline_run_row,
        "report": report,
        "manifest_rows": manifest_rows,
        "model_inference_rows": inference_rows,
        "artifact_paths": {name: str(path) for name, path in paths.items()},
        "dry_run": dry_run,
    }


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Curate already-ingested transcript chunks into evidence JSONL."
    )
    parser.add_argument("--dataset-id", required=True)
    parser.add_argument("--dataset-version", required=True)
    parser.add_argument("--language", default=None)
    parser.add_argument("--video-id", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--min-quality-score", type=float, default=0.6)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help=argparse.SUPPRESS,
    )
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    """CLI entry point."""
    args = parse_args(argv)
    library = load_video_library(args.data_dir)
    result = run_curation_pipeline(
        library=library,
        dataset_id=args.dataset_id,
        dataset_version=args.dataset_version,
        data_dir=args.data_dir,
        language=args.language,
        video_id=args.video_id,
        limit=args.limit,
        min_quality_score=args.min_quality_score,
        dry_run=args.dry_run,
    )

    summary = {
        "pipeline_run_id": result["pipeline_run"]["pipeline_run_id"],
        "status": result["pipeline_run"]["status"],
        "input_record_count": result["pipeline_run"]["input_record_count"],
        "output_record_count": result["pipeline_run"]["output_record_count"],
        "eligible_record_count": result["pipeline_run"]["eligible_record_count"],
        "excluded_record_count": result["pipeline_run"]["excluded_record_count"],
        "failed_record_count": result["pipeline_run"]["failed_record_count"],
        "dry_run": result["dry_run"],
        "artifact_paths": result["artifact_paths"],
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
