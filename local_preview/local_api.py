#!/usr/bin/env python3
"""Local-first web preview API for Japanese YouTube RAG v2.

Runs a local HTTP server that serves both:
- Static web UI (`/`)
- JSON API routes (`/v1/*`)

This is intentionally local-only and does not require Cloudflare bindings.
"""

from __future__ import annotations

import json
import math
import os
import re
import sys
import threading
import time
import traceback
import uuid
from collections import Counter
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from html import unescape
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import parse_qs, quote, urlparse, unquote
from urllib.request import Request, urlopen

try:
    from openai import OpenAI
except Exception:  # pragma: no cover
    OpenAI = None


ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from multilingual.rag_engine import (  # noqa: E402
    CONTEXT_LABELS,
    SYSTEM_PROMPTS,
    RAGEngine,
    _detect_dominant_language,
    _format_context,
)
from multilingual.text_processing import LANGUAGE_CONFIG  # noqa: E402


VIDEO_ID_RE = re.compile(r"[a-zA-Z0-9_-]{11}")
PLAYLIST_VIDEO_RE = re.compile(r'"videoId":"([a-zA-Z0-9_-]{11})"')
TOKEN_RE = re.compile(r"[a-zA-Z0-9_]+")
JP_CHAR_RE = re.compile(r"[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff]")
RETRIEVAL_MODES = {"hybrid", "dense", "lexical"}
REVIEW_LABELS = {"relevant", "not_relevant"}
TITLE_PLACEHOLDER_RE = re.compile(r"^Video [a-zA-Z0-9_-]{11}$")
ASK_PROVIDERS = {"chatgpt", "claude"}
DEFAULT_ASK_PROVIDER = "chatgpt"
ASK_MAX_TOKENS = 2000
ASK_TEMPERATURE = 0.5
ASK_TONE_INSTRUCTION = "Use a light, engaging tone without inventing facts."
FEEDBACK_ALPHA_QUERY = 0.30
FEEDBACK_BETA_GLOBAL = 0.10
FEEDBACK_MAX_ADJUST = 0.35
FEEDBACK_MIN_SIMILARITY = 0.20
FEEDBACK_RECENCY_HALFLIFE_DAYS = 30.0
SUMMARY_LANGUAGES = {"en", "ja"}
SUMMARY_MAX_POINTS = 5
SUMMARY_MAP_POINTS = 3
SUMMARY_TEMPERATURE = 0.2
SUMMARY_MAX_TOKENS = 1800
SUMMARY_SINGLE_PASS_MAX_CHARS = 16000
SUMMARY_WINDOW_MAX_CHARS = 7000
SUMMARY_COMPACT_TARGET_CHARS = 18000
SUMMARY_RETRY_ATTEMPTS = 3
SUMMARY_MIN_SENTENCES = 4
SUMMARY_MAX_SENTENCES = 5
SUMMARY_ANCHOR_MIN_CHARS = 8
SUMMARY_ANCHOR_TOKEN_MATCH_THRESHOLD = 0.55

ASK_STYLE_INSTRUCTION = (
    "Answer in the same language as the user question. "
    "Use only provided excerpts. "
    "For every factual claim, include an inline citation like [Video Title mm:ss-mm:ss]. "
    "If evidence is missing, explicitly say so."
)


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class SummaryGenerationError(RuntimeError):
    """Raised when transcript summarization repeatedly fails to produce valid output."""


def _extract_json_payload(raw_text: str) -> Any:
    text = str(raw_text or "").strip()
    if not text:
        raise ValueError("LLM returned empty content.")

    fenced = re.findall(r"```(?:json)?\s*([\s\S]*?)```", text, flags=re.IGNORECASE)
    candidates = [entry.strip() for entry in fenced if entry.strip()]
    if text not in candidates:
        candidates.append(text)

    def _parse_candidate(candidate: str) -> Any:
        candidate = candidate.strip()
        if not candidate:
            raise ValueError("Empty JSON candidate.")
        return json.loads(candidate)

    for candidate in candidates:
        try:
            return _parse_candidate(candidate)
        except Exception:
            pass

        start_positions = []
        brace_idx = candidate.find("{")
        bracket_idx = candidate.find("[")
        if brace_idx >= 0:
            start_positions.append(brace_idx)
        if bracket_idx >= 0:
            start_positions.append(bracket_idx)
        if not start_positions:
            continue

        start = min(start_positions)
        for end in range(len(candidate), start + 1, -1):
            chunk = candidate[start:end].strip()
            if not chunk:
                continue
            if chunk[0] not in "{[":
                continue
            if (chunk[0] == "{" and chunk[-1] != "}") or (
                chunk[0] == "[" and chunk[-1] != "]"
            ):
                continue
            try:
                return _parse_candidate(chunk)
            except Exception:
                continue

    raise ValueError("Could not parse JSON payload from LLM output.")


def normalize_language(value: Optional[str], fallback: str = "ja") -> str:
    language = (value or fallback).strip().lower()
    return language if language in LANGUAGE_CONFIG else fallback


def extract_video_id(value: str) -> Optional[str]:
    raw = (value or "").strip()

    patterns = [
        r"(?:youtube\.com/watch\?[^\s]*v=)([a-zA-Z0-9_-]{11})",
        r"(?:youtu\.be/)([a-zA-Z0-9_-]{11})",
        r"(?:youtube\.com/embed/)([a-zA-Z0-9_-]{11})",
    ]

    for pattern in patterns:
        m = re.search(pattern, raw)
        if m:
            return m.group(1)

    if VIDEO_ID_RE.fullmatch(raw):
        return raw
    return None


def expand_playlist_ids(url: str) -> List[str]:
    parsed = urlparse(url)
    params = parse_qs(parsed.query)
    playlist_id = (params.get("list") or [None])[0]
    if not playlist_id:
        raise ValueError("Playlist URL must include list=<id>")

    req = Request(
        f"https://www.youtube.com/playlist?list={playlist_id}",
        headers={"User-Agent": "Mozilla/5.0"},
    )

    with urlopen(req, timeout=25) as response:
        html = response.read().decode("utf-8", errors="ignore")

    ids = list(dict.fromkeys(PLAYLIST_VIDEO_RE.findall(html)))
    if not ids:
        raise ValueError("Could not parse video IDs from playlist page")
    return ids


def fetch_video_title(video_id: str) -> Optional[str]:
    watch_url = f"https://www.youtube.com/watch?v={video_id}"
    user_agent = {"User-Agent": "Mozilla/5.0"}

    # First try oEmbed (lightweight and usually reliable without API keys).
    oembed_url = (
        "https://www.youtube.com/oembed"
        f"?url={quote(watch_url, safe='')}"
        "&format=json"
    )
    try:
        with urlopen(Request(oembed_url, headers=user_agent), timeout=10) as response:
            payload = json.loads(response.read().decode("utf-8", errors="ignore"))
            title = str(payload.get("title", "")).strip()
            if title:
                return title
    except Exception:
        pass

    # Fallback to watch page OpenGraph title.
    try:
        with urlopen(Request(watch_url, headers=user_agent), timeout=12) as response:
            html = response.read().decode("utf-8", errors="ignore")
        m = re.search(r'<meta property="og:title" content="([^"]+)"', html)
        if m:
            title = unescape(m.group(1)).strip()
            if title:
                return title
    except Exception:
        pass

    return None


@dataclass
class IngestJob:
    job_id: str
    video_id: str
    url: str
    language: str
    mode: str
    status: str
    attempts: int
    error_code: Optional[str]
    error_message: Optional[str]
    created_at: str
    updated_at: str


class LocalRAGService:
    def __init__(self):
        self.lock = threading.Lock()
        self.engine = RAGEngine()
        self.openai_model = (
            str(os.environ.get("OPENAI_MODEL") or "gpt-4o-mini").strip()
            or "gpt-4o-mini"
        )
        self._openai_client = None
        self.jobs: Dict[str, IngestJob] = {}
        self.title_cache: Dict[str, str] = {}
        self.feedback_lock = threading.Lock()
        self.feedback_path = (
            Path(__file__).resolve().parent / "data" / "search_feedback.json"
        )
        self.feedback: Dict[str, dict] = {}
        self.feedback_index: Dict[str, dict] = {}
        self.feedback_tuning_enabled = str(
            os.environ.get("YT_RAG_FEEDBACK_TUNING", "1")
        ).strip().lower() not in {"0", "false", "off", "no"}
        self.log_lock = threading.Lock()
        self.ingest_log_path = (
            Path(__file__).resolve().parent / "data" / "ingest_jobs.log"
        )
        self._load_feedback()

    @property
    def openai_client(self):
        if OpenAI is None:
            raise ValueError(
                "openai package is not installed. Install dependencies to use provider='chatgpt'."
            )

        if self._openai_client is None:
            api_key = str(os.environ.get("OPENAI_API_KEY") or "").strip()
            if not api_key:
                raise ValueError(
                    "OPENAI_API_KEY environment variable is not set. "
                    "Set it to use provider='chatgpt'."
                )
            self._openai_client = OpenAI(api_key=api_key)
        return self._openai_client

    def list_jobs(self) -> List[IngestJob]:
        with self.lock:
            return sorted(self.jobs.values(), key=lambda j: j.created_at, reverse=True)

    def get_job(self, job_id: str) -> Optional[IngestJob]:
        with self.lock:
            return self.jobs.get(job_id)

    def _store_job(self, job: IngestJob) -> None:
        with self.lock:
            self.jobs[job.job_id] = job

    def _update_job(self, job_id: str, **updates) -> None:
        with self.lock:
            job = self.jobs[job_id]
            for key, value in updates.items():
                setattr(job, key, value)
            job.updated_at = now_iso()

    @staticmethod
    def _is_placeholder_title(video_id: str, title: Optional[str]) -> bool:
        value = str(title or "").strip()
        return (
            (not value)
            or (value == f"Video {video_id}")
            or bool(TITLE_PLACEHOLDER_RE.fullmatch(value))
        )

    def _resolve_video_title(self, video_id: str, fallback: str) -> str:
        if video_id in self.title_cache:
            return self.title_cache[video_id]
        resolved = fetch_video_title(video_id) or fallback
        self.title_cache[video_id] = resolved
        return resolved

    def _hydrate_video_title(self, video_id: str) -> None:
        video = self.engine.library.videos.get(video_id)
        if not video:
            return
        current = str(video.get("title") or "").strip() or f"Video {video_id}"
        if not self._is_placeholder_title(video_id, current):
            return
        resolved = self._resolve_video_title(video_id, current)
        video["title"] = resolved

    def list_videos(self):
        results = []
        for video_id, data in self.engine.library.videos.items():
            self._hydrate_video_title(video_id)
            refreshed = self.engine.library.videos.get(video_id, data)
            results.append(
                {
                    "video_id": video_id,
                    "title": refreshed.get("title", f"Video {video_id}"),
                    "url": refreshed.get(
                        "url", f"https://www.youtube.com/watch?v={video_id}"
                    ),
                    "language": refreshed.get("language", "ja"),
                    "num_chunks": len(refreshed.get("chunks", [])),
                }
            )
        return results

    def delete_video(self, video_id: str):
        self.engine.library.remove_video(video_id)

    def _load_feedback(self) -> None:
        self.feedback_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.feedback_path.exists():
            return
        try:
            raw = json.loads(self.feedback_path.read_text(encoding="utf-8"))
            normalized: Dict[str, dict] = {}

            if isinstance(raw, dict):
                iterable = raw.values()
            elif isinstance(raw, list):
                iterable = raw
            else:
                iterable = []

            for row in iterable:
                record = self._normalize_feedback_record(row)
                if not record:
                    continue
                normalized[record["key"]] = record

            self.feedback = normalized
        except Exception:
            self.feedback = {}
        self._rebuild_feedback_index()

    def _persist_feedback(self) -> None:
        self.feedback_path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = self.feedback_path.with_suffix(".tmp")
        temp_path.write_text(
            json.dumps(self.feedback, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        temp_path.replace(self.feedback_path)

    @staticmethod
    def _coerce_optional_float(value):
        if value is None or value == "":
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _coerce_optional_int(value):
        if value is None or value == "":
            return None
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _feedback_key(
        video_id: str, chunk_index: Optional[int], start: float, end: float
    ) -> str:
        if chunk_index is not None:
            return f"{video_id}:{chunk_index}"
        start_ms = int(max(0.0, float(start)) * 1000)
        end_ms = int(max(0.0, float(end)) * 1000)
        return f"{video_id}:{start_ms}:{end_ms}"

    def _normalize_feedback_record(self, row: dict) -> Optional[dict]:
        if not isinstance(row, dict):
            return None

        video_id = str(row.get("video_id") or "").strip()
        if not video_id:
            return None

        label = str(row.get("label") or "").strip().lower()
        if label not in REVIEW_LABELS:
            return None

        try:
            chunk_index = self._coerce_optional_int(row.get("chunk_index"))
            start = float(row.get("start", 0.0))
            end = float(row.get("end", start))
        except Exception:
            return None

        key = str(row.get("key") or "").strip() or self._feedback_key(
            video_id, chunk_index, start, end
        )
        created_at = str(row.get("created_at") or now_iso())
        updated_at = str(row.get("updated_at") or created_at)
        query = str(row.get("query") or "")
        query_language = normalize_language(
            row.get("query_language"),
            fallback=self._infer_query_language(query),
        )
        query_tokens = row.get("query_tokens")
        if isinstance(query_tokens, list):
            normalized_tokens = [
                str(tok).strip().lower() for tok in query_tokens if str(tok).strip()
            ]
        else:
            normalized_tokens = self._tokenize_for_lexical(
                query, language=query_language
            )

        return {
            "id": str(row.get("id") or f"fb_{uuid.uuid4().hex[:12]}"),
            "key": key,
            "query": query,
            "query_language": query_language,
            "query_tokens": normalized_tokens,
            "retrieval_mode": str(row.get("retrieval_mode") or "hybrid"),
            "model": str(row.get("model") or row.get("retrieval_mode") or "hybrid"),
            "label": label,
            "video_id": video_id,
            "chunk_index": chunk_index,
            "start": start,
            "end": end,
            "url": str(
                row.get("url")
                or f"https://www.youtube.com/watch?v={video_id}&t={int(start)}s"
            ),
            "video_title": str(row.get("video_title") or f"Video {video_id}"),
            "language": normalize_language(row.get("language"), fallback="ja"),
            "score": self._coerce_optional_float(row.get("score")),
            "dense_score": self._coerce_optional_float(row.get("dense_score")),
            "lexical_score": self._coerce_optional_float(row.get("lexical_score")),
            "hybrid_score": self._coerce_optional_float(row.get("hybrid_score")),
            "rank": self._coerce_optional_int(row.get("rank")),
            "created_at": created_at,
            "updated_at": updated_at,
        }

    @staticmethod
    def _parse_iso_datetime(value: str) -> Optional[datetime]:
        text = str(value or "").strip()
        if not text:
            return None
        try:
            parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
        except Exception:
            return None
        if parsed.tzinfo is None:
            return parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)

    def _rebuild_feedback_index(self) -> None:
        index: Dict[str, dict] = {}
        for record in self.feedback.values():
            key = str(record.get("key") or "").strip()
            label = str(record.get("label") or "").strip().lower()
            if not key or label not in REVIEW_LABELS:
                continue
            entry = index.setdefault(
                key,
                {
                    "relevant_count": 0,
                    "not_relevant_count": 0,
                    "entries": [],
                },
            )
            if label == "relevant":
                entry["relevant_count"] += 1
            else:
                entry["not_relevant_count"] += 1
            entry["entries"].append(record)
        self.feedback_index = index

    @staticmethod
    def _jaccard_similarity(left: List[str], right: List[str]) -> float:
        left_set = {tok for tok in left if tok}
        right_set = {tok for tok in right if tok}
        if not left_set or not right_set:
            return 0.0
        union = left_set | right_set
        if not union:
            return 0.0
        return len(left_set & right_set) / len(union)

    def _compute_feedback_adjustment(
        self, *, query_tokens: List[str], chunk_key: str
    ) -> dict:
        feedback = self.feedback_index.get(chunk_key)
        if not feedback:
            return {
                "adjustment": 0.0,
                "query_matches": 0,
                "relevant_count": 0,
                "not_relevant_count": 0,
                "applied": False,
            }

        relevant_count = int(feedback.get("relevant_count", 0))
        not_relevant_count = int(feedback.get("not_relevant_count", 0))
        query_weight_sum = 0.0
        query_signal_sum = 0.0
        query_matches = 0
        now_dt = datetime.now(timezone.utc)

        for record in feedback.get("entries", []):
            label = str(record.get("label") or "").strip().lower()
            if label not in REVIEW_LABELS:
                continue
            past_tokens = record.get("query_tokens")
            if not isinstance(past_tokens, list):
                continue
            similarity = self._jaccard_similarity(
                query_tokens, [str(tok).strip().lower() for tok in past_tokens]
            )
            if similarity < FEEDBACK_MIN_SIMILARITY:
                continue
            updated_dt = self._parse_iso_datetime(
                record.get("updated_at")
            ) or self._parse_iso_datetime(record.get("created_at"))
            if updated_dt is None:
                age_days = 0.0
            else:
                age_days = max(0.0, (now_dt - updated_dt).total_seconds() / 86400.0)
            recency_weight = math.exp(
                -math.log(2.0) * age_days / FEEDBACK_RECENCY_HALFLIFE_DAYS
            )
            weight = similarity * recency_weight
            if weight <= 0.0:
                continue
            label_value = 1.0 if label == "relevant" else -1.0
            query_signal_sum += label_value * weight
            query_weight_sum += weight
            query_matches += 1

        query_component = (
            (query_signal_sum / query_weight_sum) if query_weight_sum > 0.0 else 0.0
        )
        global_component = (relevant_count - not_relevant_count) / float(
            relevant_count + not_relevant_count + 2
        )
        adjustment = (FEEDBACK_ALPHA_QUERY * query_component) + (
            FEEDBACK_BETA_GLOBAL * global_component
        )
        adjustment = max(-FEEDBACK_MAX_ADJUST, min(FEEDBACK_MAX_ADJUST, adjustment))
        return {
            "adjustment": float(adjustment),
            "query_matches": query_matches,
            "relevant_count": relevant_count,
            "not_relevant_count": not_relevant_count,
            "applied": (query_matches > 0) or (relevant_count + not_relevant_count > 0),
        }

    def _apply_feedback_rerank(self, *, query: str, rows: List[dict]) -> dict:
        if not rows:
            return {"results": [], "adjusted_count": 0}

        query_language = self._infer_query_language(query)
        query_tokens = self._tokenize_for_lexical(query, language=query_language)
        annotated: List[dict] = []
        adjusted_count = 0
        for row in rows:
            item = dict(row)
            base_score = float(item.get("score", 0.0))
            item["base_score"] = base_score
            if self.feedback_tuning_enabled:
                feedback = self._compute_feedback_adjustment(
                    query_tokens=query_tokens, chunk_key=self._chunk_identity(item)
                )
                adjustment = float(feedback["adjustment"])
            else:
                feedback = {
                    "query_matches": 0,
                    "relevant_count": 0,
                    "not_relevant_count": 0,
                    "applied": False,
                }
                adjustment = 0.0
            item["feedback_adjustment"] = adjustment
            item["feedback_signal"] = {
                "applied": bool(feedback["applied"] and self.feedback_tuning_enabled),
                "query_matches": int(feedback["query_matches"]),
                "relevant_count": int(feedback["relevant_count"]),
                "not_relevant_count": int(feedback["not_relevant_count"]),
            }
            item["score"] = base_score + adjustment
            if abs(adjustment) > 1e-12:
                adjusted_count += 1
            annotated.append(item)

        annotated.sort(
            key=lambda row: (
                float(row.get("score", 0.0)),
                -int(row.get("rank", 10**9)),
            ),
            reverse=True,
        )
        for idx, row in enumerate(annotated, start=1):
            row["rank"] = idx
        return {
            "results": annotated,
            "adjusted_count": adjusted_count,
        }

    def _append_ingest_log(
        self, *, level: str, event: str, message: str, **context
    ) -> None:
        record = {
            "ts": now_iso(),
            "level": str(level or "info").lower(),
            "event": event,
            "message": message,
            **context,
        }
        self.ingest_log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.log_lock:
            with self.ingest_log_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(record, ensure_ascii=False))
                fh.write("\n")

    def list_ingest_logs(
        self,
        *,
        limit: int = 200,
        level: Optional[str] = None,
        job_id: Optional[str] = None,
        video_id: Optional[str] = None,
        since: Optional[str] = None,
    ) -> List[dict]:
        safe_limit = max(1, min(int(limit), 5000))
        scoped_level = str(level or "").strip().lower()
        scoped_job = str(job_id or "").strip()
        scoped_video = str(video_id or "").strip()
        scoped_since = str(since or "").strip()

        if not self.ingest_log_path.exists():
            return []

        rows: List[dict] = []
        with self.log_lock:
            for line in self.ingest_log_path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue

                if scoped_level and str(row.get("level") or "").lower() != scoped_level:
                    continue
                if scoped_job and str(row.get("job_id") or "") != scoped_job:
                    continue
                if scoped_video and str(row.get("video_id") or "") != scoped_video:
                    continue
                if scoped_since and str(row.get("ts") or "") < scoped_since:
                    continue

                rows.append(row)

        rows.sort(key=lambda row: str(row.get("ts") or ""), reverse=True)
        return rows[:safe_limit]

    def save_search_feedback(self, payload: dict) -> dict:
        query = str(payload.get("query") or "").strip()
        retrieval_mode = str(payload.get("retrieval_mode") or "").strip().lower()
        model = str(payload.get("model") or "").strip().lower()
        label = str(payload.get("label") or "").strip().lower()
        video_id = str(payload.get("video_id") or "").strip()

        if not query:
            raise ValueError("query is required")
        if retrieval_mode not in RETRIEVAL_MODES:
            raise ValueError("retrieval_mode must be one of: dense, hybrid, lexical")
        if label not in REVIEW_LABELS:
            raise ValueError("label must be one of: relevant, not_relevant")
        if not video_id:
            raise ValueError("video_id is required")

        chunk_index = self._coerce_optional_int(payload.get("chunk_index"))
        start = float(payload.get("start", 0.0))
        end = float(payload.get("end", start))
        rank = self._coerce_optional_int(payload.get("rank"))
        identity = self._feedback_key(video_id, chunk_index, start, end)
        now = now_iso()
        query_language = self._infer_query_language(query)
        query_tokens = self._tokenize_for_lexical(query, language=query_language)

        with self.feedback_lock:
            existing = self.feedback.get(identity)
            resolved_model = (
                model
                or (
                    str(existing.get("model") or "").strip().lower()
                    if isinstance(existing, dict)
                    else ""
                )
                or retrieval_mode
            )
            record = {
                "id": existing.get("id")
                if isinstance(existing, dict)
                else f"fb_{uuid.uuid4().hex[:12]}",
                "key": identity,
                "query": query,
                "query_language": query_language,
                "query_tokens": query_tokens,
                "retrieval_mode": retrieval_mode,
                "model": resolved_model,
                "label": label,
                "video_id": video_id,
                "chunk_index": chunk_index,
                "start": start,
                "end": end,
                "url": str(
                    payload.get("url")
                    or f"https://www.youtube.com/watch?v={video_id}&t={int(start)}s"
                ),
                "video_title": str(payload.get("video_title") or f"Video {video_id}"),
                "language": normalize_language(payload.get("language"), fallback="ja"),
                "score": self._coerce_optional_float(payload.get("score")),
                "dense_score": self._coerce_optional_float(payload.get("dense_score")),
                "lexical_score": self._coerce_optional_float(
                    payload.get("lexical_score")
                ),
                "hybrid_score": self._coerce_optional_float(
                    payload.get("hybrid_score")
                ),
                "rank": rank,
                "created_at": existing.get("created_at")
                if isinstance(existing, dict)
                else now,
                "updated_at": now,
            }
            self.feedback[identity] = record
            self._persist_feedback()
            self._rebuild_feedback_index()
            return record

    def list_search_feedback(
        self,
        *,
        video_id: Optional[str] = None,
        label: Optional[str] = None,
        limit: int = 500,
    ) -> List[dict]:
        scoped_video = str(video_id or "").strip()
        scoped_label = str(label or "").strip().lower()
        safe_limit = max(1, min(int(limit), 5000))

        with self.feedback_lock:
            rows = list(self.feedback.values())

        if scoped_video:
            rows = [
                row for row in rows if str(row.get("video_id") or "") == scoped_video
            ]
        if scoped_label in REVIEW_LABELS:
            rows = [
                row
                for row in rows
                if str(row.get("label") or "").lower() == scoped_label
            ]

        rows.sort(key=lambda row: str(row.get("updated_at") or ""), reverse=True)
        return rows[:safe_limit]

    def delete_search_feedback(
        self,
        *,
        video_id: Optional[str] = None,
        label: Optional[str] = None,
    ) -> int:
        scoped_video = str(video_id or "").strip()
        scoped_label = str(label or "").strip().lower()

        if scoped_label and scoped_label not in REVIEW_LABELS:
            raise ValueError("label must be one of: relevant, not_relevant")
        if not scoped_video and not scoped_label:
            raise ValueError("at least one filter is required: video_id or label")

        with self.feedback_lock:
            keys = [
                key
                for key, row in self.feedback.items()
                if (not scoped_video or str(row.get("video_id") or "") == scoped_video)
                and (
                    not scoped_label
                    or str(row.get("label") or "").lower() == scoped_label
                )
            ]
            if not keys:
                return 0
            for key in keys:
                self.feedback.pop(key, None)
            self._persist_feedback()
            self._rebuild_feedback_index()
            return len(keys)

    def list_feedback_videos(self) -> List[dict]:
        with self.feedback_lock:
            rows = list(self.feedback.values())

        by_video: Dict[str, dict] = {}
        for row in rows:
            video_id = str(row.get("video_id") or "").strip()
            if not video_id:
                continue

            entry = by_video.setdefault(
                video_id,
                {
                    "video_id": video_id,
                    "title": str(row.get("video_title") or f"Video {video_id}"),
                    "video_title": str(row.get("video_title") or f"Video {video_id}"),
                    "review_count": 0,
                    "relevant_count": 0,
                    "not_relevant_count": 0,
                },
            )
            entry["review_count"] += 1
            if row.get("label") == "relevant":
                entry["relevant_count"] += 1
            elif row.get("label") == "not_relevant":
                entry["not_relevant_count"] += 1

        for video_id, video_data in self.engine.library.videos.items():
            if video_id not in by_video:
                continue
            title = str(video_data.get("title") or "").strip()
            if title:
                by_video[video_id]["title"] = title
                by_video[video_id]["video_title"] = title

        results = list(by_video.values())
        results.sort(key=lambda row: (-row["review_count"], row["video_id"]))
        return results

    @staticmethod
    def _chunk_identity(row: dict) -> str:
        video_id = row.get("video_id", "")
        chunk_index = row.get("chunk_index")
        if chunk_index is not None:
            return f"{video_id}:{chunk_index}"
        start_ms = int(float(row.get("start", 0.0)) * 1000)
        end_ms = int(float(row.get("end", 0.0)) * 1000)
        return f"{video_id}:{start_ms}:{end_ms}"

    @staticmethod
    def _infer_query_language(query: str) -> str:
        return "ja" if JP_CHAR_RE.search(query or "") else "en"

    @staticmethod
    def _tokenize_for_lexical(text: str, language: Optional[str] = None) -> List[str]:
        cleaned = re.sub(r"\s+", " ", str(text or "")).strip().lower()
        if not cleaned:
            return []

        inferred = language or ("ja" if JP_CHAR_RE.search(cleaned) else "en")
        lang = normalize_language(inferred, fallback="ja" if inferred == "ja" else "en")

        tokens: List[str] = []
        if lang == "ja" or JP_CHAR_RE.search(cleaned):
            compact = cleaned.replace(" ", "")
            if len(compact) == 1:
                tokens.append(compact)
            else:
                tokens.extend(compact[i : i + 2] for i in range(len(compact) - 1))
            tokens.extend([part for part in cleaned.split(" ") if part])
        else:
            tokens.extend(TOKEN_RE.findall(cleaned))

        return [tok for tok in tokens if tok]

    def _all_chunks(self) -> List[dict]:
        rows: List[dict] = []
        for video_id, video_data in self.engine.library.videos.items():
            title = video_data.get("title", f"Video {video_id}")
            video_url = video_data.get(
                "url", f"https://www.youtube.com/watch?v={video_id}"
            )
            language = video_data.get("language", "ja")
            chunks = video_data.get("chunks", [])
            for chunk_idx, chunk in enumerate(chunks):
                start = float(chunk.get("start", 0.0))
                rows.append(
                    {
                        "video_id": video_id,
                        "video_title": title,
                        "video_url": video_url,
                        "language": language,
                        "chunk_index": chunk_idx,
                        "text": chunk.get("raw_text", ""),
                        "start": start,
                        "end": float(chunk.get("end", start)),
                        "url": f"https://www.youtube.com/watch?v={video_id}&t={int(start)}s",
                    }
                )
        return rows

    def _dense_search(self, query: str, k: int, language: Optional[str]) -> List[dict]:
        try:
            dense_rows = self.engine.search(query, k=k, language=language)
        except Exception:
            return []
        results: List[dict] = []
        for rank, row in enumerate(dense_rows, start=1):
            item = dict(row)
            item["rank"] = rank
            item["dense_score"] = float(row.get("score", 0.0))
            item["score"] = float(row.get("score", 0.0))
            results.append(item)
        return results

    def _lexical_bm25_search(
        self, query: str, k: int, language: Optional[str]
    ) -> List[dict]:
        candidates = self._all_chunks()
        if not candidates:
            return []

        query_language = normalize_language(
            language, fallback=self._infer_query_language(query)
        )
        query_tokens = self._tokenize_for_lexical(query, language=query_language)
        if not query_tokens:
            return []

        doc_tokens: List[List[str]] = []
        doc_lens: List[int] = []
        doc_freq: Counter = Counter()
        for candidate in candidates:
            tokens = self._tokenize_for_lexical(
                candidate["text"], language=candidate.get("language")
            )
            doc_tokens.append(tokens)
            doc_len = len(tokens)
            doc_lens.append(doc_len)
            for token in set(tokens):
                doc_freq[token] += 1

        total_docs = len(candidates)
        avg_doc_len = (sum(doc_lens) / total_docs) if total_docs else 1.0
        query_tf = Counter(query_tokens)

        k1 = 1.5
        b = 0.75
        scored = []

        for idx, candidate in enumerate(candidates):
            tokens = doc_tokens[idx]
            if not tokens:
                continue
            tf = Counter(tokens)
            score = 0.0
            doc_len = max(1, doc_lens[idx])

            for term, q_weight in query_tf.items():
                if term not in tf:
                    continue
                n_qi = doc_freq.get(term, 0)
                idf = math.log(1.0 + (total_docs - n_qi + 0.5) / (n_qi + 0.5))
                freq = tf[term]
                denom = freq + k1 * (1.0 - b + b * (doc_len / max(avg_doc_len, 1e-9)))
                score += (
                    idf * ((freq * (k1 + 1.0)) / denom) * (1.0 + 0.2 * (q_weight - 1))
                )

            if score <= 0.0:
                continue

            item = dict(candidate)
            item["lexical_score"] = float(score)
            item["score"] = float(score)
            scored.append(item)

        scored.sort(key=lambda row: row["lexical_score"], reverse=True)
        for rank, row in enumerate(scored, start=1):
            row["rank"] = rank
        return scored[:k]

    def _rrf_fuse(
        self,
        dense_results: List[dict],
        lexical_results: List[dict],
        limit: Optional[int] = None,
        rrf_k: int = 60,
    ) -> List[dict]:
        merged: Dict[str, dict] = {}

        for dense_rank, row in enumerate(dense_results, start=1):
            key = self._chunk_identity(row)
            entry = merged.setdefault(key, dict(row))
            entry["dense_rank"] = min(dense_rank, entry.get("dense_rank", dense_rank))
            entry["dense_score"] = float(row.get("dense_score", row.get("score", 0.0)))
            entry["hybrid_score"] = float(entry.get("hybrid_score", 0.0)) + (
                1.0 / (rrf_k + dense_rank)
            )

        for lexical_rank, row in enumerate(lexical_results, start=1):
            key = self._chunk_identity(row)
            entry = merged.setdefault(key, dict(row))
            entry["lexical_rank"] = min(
                lexical_rank, entry.get("lexical_rank", lexical_rank)
            )
            entry["lexical_score"] = float(
                row.get("lexical_score", row.get("score", 0.0))
            )
            entry["hybrid_score"] = float(entry.get("hybrid_score", 0.0)) + (
                1.0 / (rrf_k + lexical_rank)
            )

        fused = list(merged.values())
        fused.sort(
            key=lambda row: (
                float(row.get("hybrid_score", 0.0)),
                float(row.get("dense_score", 0.0)),
                float(row.get("lexical_score", 0.0)),
            ),
            reverse=True,
        )

        for rank, row in enumerate(fused, start=1):
            row["rank"] = rank
            row["score"] = float(row.get("hybrid_score", 0.0))

        if limit is None:
            return fused
        return fused[: max(1, int(limit))]

    def _rows_near_duplicate(self, left: dict, right: dict) -> bool:
        if str(left.get("video_id") or "") != str(right.get("video_id") or ""):
            return False

        left_start = float(left.get("start", 0.0))
        right_start = float(right.get("start", 0.0))
        if abs(left_start - right_start) <= 8.0:
            return True

        left_tokens = self._tokenize_for_lexical(
            str(left.get("text") or ""), language=left.get("language")
        )
        right_tokens = self._tokenize_for_lexical(
            str(right.get("text") or ""), language=right.get("language")
        )
        return self._jaccard_similarity(left_tokens, right_tokens) >= 0.85

    def _apply_diversity_selection(self, rows: List[dict], top_k: int) -> dict:
        if not rows:
            return {
                "results": [],
                "diversity_applied": False,
                "max_per_video": max(2, math.ceil(max(1, int(top_k)) / 2)),
            }

        limit = max(1, int(top_k))
        max_per_video = max(2, math.ceil(limit / 2))

        selected: List[dict] = []
        fallback_rows: List[dict] = []
        seen_keys = set()
        per_video_counts: Dict[str, int] = {}
        diversity_applied = False

        for row in rows:
            item = dict(row)
            key = self._chunk_identity(item)
            if key in seen_keys:
                continue

            video_id = str(item.get("video_id") or "")
            if per_video_counts.get(video_id, 0) >= max_per_video:
                fallback_rows.append(item)
                diversity_applied = True
                continue

            if any(self._rows_near_duplicate(item, existing) for existing in selected):
                fallback_rows.append(item)
                diversity_applied = True
                continue

            selected.append(item)
            seen_keys.add(key)
            per_video_counts[video_id] = per_video_counts.get(video_id, 0) + 1
            if len(selected) >= limit:
                break

        if len(selected) < limit:
            for item in fallback_rows:
                key = self._chunk_identity(item)
                if key in seen_keys:
                    continue
                selected.append(item)
                seen_keys.add(key)
                if len(selected) >= limit:
                    break

        for idx, row in enumerate(selected, start=1):
            row["rank"] = idx

        return {
            "results": selected[:limit],
            "diversity_applied": diversity_applied,
            "max_per_video": max_per_video,
        }

    def retrieve(
        self,
        query: str,
        k: int = 5,
        language: Optional[str] = None,
        retrieval_mode: str = "hybrid",
    ) -> dict:
        mode = (retrieval_mode or "hybrid").strip().lower()
        if mode not in RETRIEVAL_MODES:
            raise ValueError(
                f"retrieval_mode must be one of: {', '.join(sorted(RETRIEVAL_MODES))}"
            )

        top_k = max(1, min(int(k), 12))
        candidate_k = max(30, top_k * 8)

        dense_results: List[dict] = []
        lexical_results: List[dict] = []
        candidate_rows: List[dict] = []

        if mode in {"dense", "hybrid"}:
            dense_results = self._dense_search(query, k=candidate_k, language=language)
        if mode in {"lexical", "hybrid"}:
            lexical_results = self._lexical_bm25_search(
                query, k=candidate_k, language=language
            )

        if mode == "dense":
            candidate_rows = dense_results
        elif mode == "lexical":
            candidate_rows = lexical_results
        else:
            if dense_results and lexical_results:
                candidate_rows = self._rrf_fuse(
                    dense_results, lexical_results, limit=None
                )
            elif dense_results:
                candidate_rows = dense_results
            else:
                candidate_rows = lexical_results

        pre_rerank_count = len(candidate_rows)
        reranked = self._apply_feedback_rerank(query=query, rows=candidate_rows)
        reranked_rows = reranked["results"]
        post_feedback_count = len(reranked_rows)

        diverse = self._apply_diversity_selection(reranked_rows, top_k=top_k)
        results = diverse["results"]

        details = {
            "fusion": "rrf" if mode == "hybrid" else "none",
            "dense_candidates": len(dense_results),
            "lexical_candidates": len(lexical_results),
            "candidate_k": candidate_k,
            "pre_rerank_candidate_count": pre_rerank_count,
            "post_feedback_candidate_count": post_feedback_count,
            "diversity_applied": bool(diverse["diversity_applied"]),
            "selected_per_video_cap": int(diverse["max_per_video"]),
            "feedback_tuning": {
                "enabled": bool(self.feedback_tuning_enabled),
                "adjusted_results": int(reranked["adjusted_count"]),
                "alpha_query": FEEDBACK_ALPHA_QUERY,
                "beta_global": FEEDBACK_BETA_GLOBAL,
                "max_adjust": FEEDBACK_MAX_ADJUST,
                "min_similarity": FEEDBACK_MIN_SIMILARITY,
            },
            "fallback": (
                "dense_only"
                if mode == "hybrid" and dense_results and not lexical_results
                else "lexical_only"
                if mode == "hybrid" and lexical_results and not dense_results
                else None
            ),
        }

        return {
            "retrieval_mode": mode,
            "details": details,
            "results": results,
        }

    def _llm_text_response(
        self,
        *,
        provider: str,
        system_prompt: str,
        user_message: str,
        max_tokens: int,
        temperature: float,
    ) -> dict:
        scoped_provider = str(provider or DEFAULT_ASK_PROVIDER).strip().lower()
        if scoped_provider not in ASK_PROVIDERS:
            raise ValueError("provider must be one of: chatgpt, claude")

        if scoped_provider == "claude":
            response = self.engine.client.messages.create(
                model=self.engine.model,
                max_tokens=max_tokens,
                system=system_prompt,
                messages=[{"role": "user", "content": user_message}],
                temperature=temperature,
            )
            text = response.content[0].text if response.content else ""
            model = self.engine.model
        else:
            response = self.openai_client.chat.completions.create(
                model=self.openai_model,
                max_tokens=max_tokens,
                temperature=temperature,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message},
                ],
            )
            text = ""
            if response.choices:
                text = str(response.choices[0].message.content or "")
            if not text.strip():
                raise ValueError("OpenAI response missing content.")
            model = self.openai_model

        if not str(text or "").strip():
            raise ValueError("LLM response missing content.")

        return {
            "provider": scoped_provider,
            "model": model,
            "text": text,
        }

    @staticmethod
    def _format_timestamp_for_prompt(seconds: float) -> str:
        seconds_int = max(0, int(float(seconds or 0.0)))
        minutes = seconds_int // 60
        remaining = seconds_int % 60
        return f"{minutes}:{remaining:02d}"

    @staticmethod
    def _summary_segment_text(segment: dict) -> str:
        return re.sub(
            r"\s+", " ", str(segment.get("text") or segment.get("raw_text") or "")
        ).strip()

    @classmethod
    def _coerce_summary_segments(cls, rows: List[dict]) -> List[dict]:
        segments: List[dict] = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            text = cls._summary_segment_text(row)
            if not text:
                continue
            try:
                start = float(row.get("start", 0.0))
            except Exception:
                start = 0.0
            try:
                end = float(row.get("end", row.get("timestamp_end", start)))
            except Exception:
                end = start
            end = max(start, end)
            segments.append(
                {
                    "start": start,
                    "end": end,
                    "text": text,
                }
            )
        segments.sort(
            key=lambda row: (float(row.get("start", 0.0)), float(row.get("end", 0.0)))
        )
        return segments

    def _summary_transcript_lines(self, segments: List[dict]) -> List[str]:
        lines: List[str] = []
        normalized_segments = self._coerce_summary_segments(segments)
        for idx, segment in enumerate(normalized_segments, start=1):
            start = float(segment.get("start", 0.0))
            end = float(segment.get("end", start))
            lines.append(
                f"[{idx:04d}] {self._format_timestamp_for_prompt(start)}-{self._format_timestamp_for_prompt(end)} {segment['text']}"
            )
        return lines

    def _summary_compact_transcript_lines(
        self,
        segments: List[dict],
        target_chars: int = SUMMARY_COMPACT_TARGET_CHARS,
    ) -> List[str]:
        normalized_segments = self._coerce_summary_segments(segments)
        if not normalized_segments:
            return []

        line_budget = max(
            42, min(180, int(target_chars / max(1, len(normalized_segments))) - 20)
        )
        lines: List[str] = []
        for idx, segment in enumerate(normalized_segments, start=1):
            text = segment["text"]
            if len(text) > line_budget:
                text = f"{text[:max(1, line_budget - 1)].rstrip()}…"
            start = float(segment.get("start", 0.0))
            lines.append(
                f"[{idx:04d}] {self._format_timestamp_for_prompt(start)} {text}"
            )
        return lines

    def _window_chunks_for_summary(
        self, segments: List[dict], max_chars: int
    ) -> List[List[dict]]:
        windows: List[List[dict]] = []
        current: List[dict] = []
        current_chars = 0

        for segment in self._coerce_summary_segments(segments):
            text = segment["text"]
            size = len(text) + 48
            if current and current_chars + size > max_chars:
                windows.append(current)
                current = []
                current_chars = 0
            current.append(segment)
            current_chars += size

        if current:
            windows.append(current)
        return windows

    @staticmethod
    def _summary_sentence_count(text: str, language: str) -> int:
        candidate = re.sub(r"\s+", " ", str(text or "")).strip()
        if not candidate:
            return 0
        if language == "ja":
            parts = re.split(r"[。！？]+", candidate)
        else:
            parts = re.split(r"(?<=[.!?])\s+", candidate)
        return len([part for part in parts if str(part).strip()])

    def _summary_items_have_required_sentence_count(
        self, items: List[dict], language: str
    ) -> bool:
        if not items:
            return False
        for row in items:
            sentence_count = self._summary_sentence_count(
                str(row.get("tldr") or ""), language
            )
            if (
                sentence_count < SUMMARY_MIN_SENTENCES
                or sentence_count > SUMMARY_MAX_SENTENCES
            ):
                return False
        return True

    def _normalize_summary_items(
        self,
        *,
        items: Any,
        segments: List[dict],
        language: str,
        max_points: int = SUMMARY_MAX_POINTS,
    ) -> List[dict]:
        max_points = max(1, int(max_points))
        prepared: List[tuple[int, Optional[int], dict]] = []

        if isinstance(items, dict):
            raw_items = items.get("items")
        else:
            raw_items = items

        if not isinstance(raw_items, list):
            raw_items = []

        normalized_segments = self._coerce_summary_segments(segments)
        min_start = (
            float(normalized_segments[0].get("start", 0.0))
            if normalized_segments
            else 0.0
        )
        max_end = (
            float(normalized_segments[-1].get("end", min_start))
            if normalized_segments
            else min_start
        )
        if max_end < min_start:
            max_end = min_start

        for input_idx, raw in enumerate(raw_items):
            if not isinstance(raw, dict):
                continue
            title = re.sub(r"\s+", " ", str(raw.get("title") or "")).strip()
            tldr = re.sub(
                r"\s+", " ", str(raw.get("tldr") or raw.get("summary") or "")
            ).strip()
            anchor_text = re.sub(
                r"\s+",
                " ",
                str(
                    raw.get("anchor_text")
                    or raw.get("anchor")
                    or raw.get("evidence")
                    or ""
                ),
            ).strip()
            if not tldr:
                continue
            try:
                start = float(raw.get("start", raw.get("timestamp_start", min_start)))
            except Exception:
                start = min_start
            try:
                end = float(raw.get("end", raw.get("timestamp_end", start)))
            except Exception:
                end = start
            start = max(min_start, min(start, max_end))
            end = max(start, min(end, max_end))

            rank = None
            try:
                parsed_rank = int(raw.get("rank"))
                if parsed_rank > 0:
                    rank = parsed_rank
            except Exception:
                rank = None

            prepared.append(
                (
                    input_idx,
                    rank,
                    {
                        "title": title,
                        "tldr": tldr,
                        "anchor_text": anchor_text,
                        "start": start,
                        "end": end,
                    },
                )
            )

        has_rank = any(row_rank is not None for _, row_rank, _ in prepared)
        if has_rank:
            prepared.sort(
                key=lambda row: (
                    row[1] is None,
                    row[1] if row[1] is not None else 10**6,
                    row[0],
                )
            )

        deduped: List[dict] = []
        seen: set[str] = set()
        for _, _, row in prepared:
            dedupe_key = (
                f"{str(row.get('title') or '').strip().lower()}|"
                f"{str(row.get('tldr') or '').strip().lower()}"
            )
            if dedupe_key in seen:
                continue
            seen.add(dedupe_key)
            deduped.append(row)

        deduped = deduped[:max_points]
        for idx, row in enumerate(deduped, start=1):
            if not row.get("title"):
                if language == "en":
                    row["title"] = "Intro" if idx == 1 else f"Topic {idx - 1}"
                else:
                    row["title"] = "導入" if idx == 1 else f"トピック{idx - 1}"
        return deduped

    @staticmethod
    def _summary_normalize_for_match(text: str) -> str:
        lowered = re.sub(r"\s+", " ", str(text or "")).strip().lower()
        return re.sub(r"[^\w\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff ]+", "", lowered)

    def _summary_anchor_tokens(self, text: str) -> set[str]:
        normalized = self._summary_normalize_for_match(text)
        if not normalized:
            return set()
        return {tok for tok in TOKEN_RE.findall(normalized) if tok}

    def _resolve_theme_anchor_timestamp(
        self,
        *,
        item: dict,
        segments: List[dict],
        used_segment_counts: Optional[Dict[int, int]] = None,
    ) -> dict:
        normalized_segments = self._coerce_summary_segments(segments)
        if not normalized_segments:
            return {"start": 0.0, "end": 0.0, "source": "model_fallback"}

        min_start = float(normalized_segments[0].get("start", 0.0))
        max_end = float(normalized_segments[-1].get("end", min_start))
        used_counts = used_segment_counts or {}

        try:
            target_start = float(item.get("start"))
        except Exception:
            target_start = None
        if target_start is not None:
            target_start = max(min_start, min(target_start, max_end))

        def _candidate_score(idx: int, match_strength: float) -> float:
            segment = normalized_segments[idx]
            segment_start = float(segment.get("start", min_start))
            reuse_penalty = 2.0 * float(used_counts.get(idx, 0))
            if target_start is None:
                proximity = 0.0
            else:
                # 20s distance ~= -1 score. Closer to model-provided timestamp wins.
                proximity = -abs(segment_start - target_start) / 20.0
            # Tiny tie-break toward earlier timeline when all else equal.
            tie_break = -0.0001 * idx
            return (2.5 * match_strength) + proximity - reuse_penalty + tie_break

        anchor_text = re.sub(r"\s+", " ", str(item.get("anchor_text") or "")).strip()
        if len(anchor_text) >= SUMMARY_ANCHOR_MIN_CHARS:
            normalized_anchor = self._summary_normalize_for_match(anchor_text)
            if normalized_anchor:
                exact_candidates: List[int] = []
                for idx, segment in enumerate(normalized_segments):
                    segment_text = self._summary_normalize_for_match(
                        segment.get("text", "")
                    )
                    if normalized_anchor and normalized_anchor in segment_text:
                        exact_candidates.append(idx)
                if exact_candidates:
                    best_idx = max(
                        exact_candidates, key=lambda idx: _candidate_score(idx, 1.0)
                    )
                    segment = normalized_segments[best_idx]
                    return {
                        "start": float(segment.get("start", min_start)),
                        "end": float(segment.get("end", min_start)),
                        "source": "anchor_match",
                        "segment_index": best_idx,
                    }

                anchor_tokens = self._summary_anchor_tokens(anchor_text)
                if anchor_tokens:
                    best_idx = -1
                    best_ranked_score = float("-inf")
                    best_match_strength = 0.0
                    for idx, segment in enumerate(normalized_segments):
                        segment_tokens = self._summary_anchor_tokens(
                            segment.get("text", "")
                        )
                        if not segment_tokens:
                            continue
                        overlap = len(anchor_tokens & segment_tokens)
                        if overlap <= 0:
                            continue
                        match_strength = overlap / float(len(anchor_tokens))
                        ranked_score = _candidate_score(idx, match_strength)
                        if ranked_score > best_ranked_score:
                            best_ranked_score = ranked_score
                            best_match_strength = match_strength
                            best_idx = idx
                    if (
                        best_idx >= 0
                        and best_match_strength >= SUMMARY_ANCHOR_TOKEN_MATCH_THRESHOLD
                    ):
                        segment = normalized_segments[best_idx]
                        return {
                            "start": float(segment.get("start", min_start)),
                            "end": float(segment.get("end", min_start)),
                            "source": "anchor_match",
                            "segment_index": best_idx,
                        }

        try:
            start = float(item.get("start", min_start))
        except Exception:
            start = min_start
        try:
            end = float(item.get("end", start))
        except Exception:
            end = start
        start = max(min_start, min(start, max_end))
        end = max(start, min(end, max_end))
        return {
            "start": start,
            "end": end,
            "source": "model_fallback",
        }

    def _summary_system_prompt(self, language: str, max_points: int) -> str:
        if language == "ja":
            language_rule = "出力は必ず日本語にしてください。文字起こし言語に関係なく日本語で要約してください。"
        else:
            language_rule = "Output must be in English. Even if the transcript is Japanese, translate and summarize into English."
        return (
            "You analyze YouTube transcripts and extract top themes across the entire video.\n"
            f"{language_rule}\n"
            "Review the entire transcript before deciding rankings.\n"
            "Rank themes by importance to the whole video, not by timestamp order.\n"
            "Do not copy transcript lines verbatim; paraphrase and summarize.\n"
            "Each tldr must be one paragraph of 4-5 sentences.\n"
            "Each item must include an anchor_text phrase from where the theme first appears in the transcript.\n"
            "Use distinct anchor_text values across items whenever possible.\n"
            "The start timestamp should match the anchor_text location.\n"
            "Include a start timestamp only if you are confident; otherwise you may omit it.\n"
            "Return strict JSON only, with no markdown.\n"
            "Use this schema:\n"
            "{\n"
            '  "items": [\n'
            '    {"title": "string", "tldr": "string", "anchor_text": "string", "start": 0.0}\n'
            "  ]\n"
            "}\n"
            f"Return exactly {max_points} ranked items in descending importance.\n"
            "Use the array order as the final rank order.\n"
        )

    @staticmethod
    def _summary_looks_like_transcript_copy(text: str, segments: List[dict]) -> bool:
        candidate = re.sub(r"\s+", " ", str(text or "")).strip()
        if len(candidate) < 24:
            return False
        lowered = candidate.lower()
        for segment in segments:
            segment_text = re.sub(
                r"\s+",
                " ",
                str(segment.get("text") or segment.get("raw_text") or ""),
            ).strip()
            if not segment_text:
                continue
            segment_lowered = segment_text.lower()
            if lowered in segment_lowered and len(lowered) >= 24:
                return True
        return False

    @staticmethod
    def _summary_items_match_language(items: List[dict], language: str) -> bool:
        if not items:
            return False
        text_rows = [
            f"{str(item.get('title') or '')} {str(item.get('tldr') or '')}".strip()
            for item in items
        ]
        if language == "en":
            has_japanese = any(JP_CHAR_RE.search(row) for row in text_rows if row)
            if has_japanese:
                return False
            return all(re.search(r"[A-Za-z]", row or "") for row in text_rows)
        if language == "ja":
            ja_count = sum(1 for row in text_rows if JP_CHAR_RE.search(row))
            return ja_count >= max(1, int(len(text_rows) * 0.7))
        return True

    def _rewrite_summary_items_for_quality(
        self,
        *,
        provider: str,
        language: str,
        max_points: int,
        segments: List[dict],
        items: List[dict],
    ) -> List[dict]:
        language_rule = (
            "Rewrite in Japanese only."
            if language == "ja"
            else "Rewrite in English only."
        )
        style_rule = (
            "Use titles like 導入, トピック1, トピック2, ... when appropriate."
            if language == "ja"
            else "Use titles like Intro, Topic 1, Topic 2, ... when appropriate."
        )
        payload = json.dumps({"items": items}, ensure_ascii=False, indent=2)
        system_prompt = (
            "You rewrite candidate transcript sections into clean final section summaries.\n"
            f"{language_rule}\n"
            f"{style_rule}\n"
            f"Each tldr must be exactly {SUMMARY_MIN_SENTENCES}-{SUMMARY_MAX_SENTENCES} sentences.\n"
            "Preserve or improve anchor_text so each item can be mapped to a real video timestamp.\n"
            "Avoid reusing the same anchor_text for multiple themes unless absolutely necessary.\n"
            "Do not copy transcript lines verbatim.\n"
            "Return strict JSON only.\n"
        )
        user_message = (
            "Rewrite and clean these section candidates.\n"
            "Keep exactly the requested number of items.\n"
            "Candidates:\n"
            f"{payload}\n"
        )
        llm = self._llm_text_response(
            provider=provider,
            system_prompt=system_prompt,
            user_message=user_message,
            max_tokens=900,
            temperature=SUMMARY_TEMPERATURE,
        )
        parsed = _extract_json_payload(llm["text"])
        rewritten = self._normalize_summary_items(
            items=parsed,
            segments=segments,
            language=language,
            max_points=max_points,
        )
        return rewritten

    def _summary_llm_with_retries(
        self,
        *,
        provider: str,
        system_prompt: str,
        user_message: str,
        segments: List[dict],
        language: str,
        max_points: int,
        stage: str,
        max_tokens: int,
        require_exact_count: bool = True,
    ) -> dict:
        last_error: Optional[Exception] = None
        attempts = max(1, int(SUMMARY_RETRY_ATTEMPTS))

        for attempt in range(1, attempts + 1):
            try:
                llm = self._llm_text_response(
                    provider=provider,
                    system_prompt=system_prompt,
                    user_message=user_message,
                    max_tokens=max_tokens,
                    temperature=SUMMARY_TEMPERATURE,
                )
                parsed = _extract_json_payload(llm["text"])
                items = self._normalize_summary_items(
                    items=parsed,
                    segments=segments,
                    language=language,
                    max_points=max_points,
                )
                if (
                    not self._summary_items_match_language(items, language)
                    or any(
                        self._summary_looks_like_transcript_copy(
                            row.get("tldr", ""), segments
                        )
                        for row in items
                    )
                    or not self._summary_items_have_required_sentence_count(
                        items, language
                    )
                ):
                    items = self._rewrite_summary_items_for_quality(
                        provider=provider,
                        language=language,
                        max_points=max_points,
                        segments=segments,
                        items=items,
                    )
                    if not self._summary_items_match_language(items, language):
                        raise ValueError(
                            "Summary output language mismatch after rewrite."
                        )
                    if any(
                        self._summary_looks_like_transcript_copy(
                            row.get("tldr", ""), segments
                        )
                        for row in items
                    ):
                        raise ValueError(
                            "Summary output still contains transcript-copy text after rewrite."
                        )
                    if not self._summary_items_have_required_sentence_count(
                        items, language
                    ):
                        raise ValueError(
                            f"Summary output must be {SUMMARY_MIN_SENTENCES}-{SUMMARY_MAX_SENTENCES} sentences per item after rewrite."
                        )
                if require_exact_count:
                    if len(items) != max_points:
                        raise ValueError(
                            f"Expected exactly {max_points} summary items, got {len(items)}."
                        )
                elif not items:
                    raise ValueError("Expected at least one summary item.")

                return {
                    "provider": llm["provider"],
                    "model": llm["model"],
                    "items": items,
                    "attempt_count": attempt,
                }
            except Exception as exc:
                last_error = exc

        raise SummaryGenerationError(
            f"{stage} failed after {attempts} attempts: {last_error}"
        )

    def _summarize_transcript_single_pass(
        self,
        *,
        transcript_lines: List[str],
        segments: List[dict],
        language: str,
        provider: str,
        max_points: int,
    ) -> dict:
        system_prompt = self._summary_system_prompt(
            language=language, max_points=max_points
        )
        transcript_blob = "\n".join(transcript_lines)
        user_message = (
            "Transcript with timestamps:\n\n"
            f"{transcript_blob}\n\n"
            "Summarize the full transcript into importance-ranked themes in the requested JSON format."
        )
        result = self._summary_llm_with_retries(
            provider=provider,
            system_prompt=system_prompt,
            user_message=user_message,
            segments=segments,
            language=language,
            max_points=max_points,
            stage="single-pass summary generation",
            max_tokens=SUMMARY_MAX_TOKENS,
            require_exact_count=True,
        )
        return {
            "provider": result["provider"],
            "model": result["model"],
            "items": result["items"],
            "strategy": "single_pass",
            "total_windows": 1,
            "processed_windows": 1,
            "retry_count": max(0, int(result.get("attempt_count", 1)) - 1),
        }

    def _summarize_transcript_compact_single_pass(
        self,
        *,
        segments: List[dict],
        language: str,
        provider: str,
        max_points: int,
    ) -> dict:
        compact_lines = self._summary_compact_transcript_lines(segments)
        if not compact_lines:
            raise ValueError("video has no transcript segments")

        system_prompt = self._summary_system_prompt(
            language=language, max_points=max_points
        )
        user_message = (
            "Compressed transcript outline covering the full video timeline:\n\n"
            f"{chr(10).join(compact_lines)}\n\n"
            "Review the entire outline and return importance-ranked themes (Intro, Topic 1, Topic 2...) in the required JSON format."
        )
        result = self._summary_llm_with_retries(
            provider=provider,
            system_prompt=system_prompt,
            user_message=user_message,
            segments=segments,
            language=language,
            max_points=max_points,
            stage="compact single-pass theme generation",
            max_tokens=SUMMARY_MAX_TOKENS,
            require_exact_count=True,
        )
        return {
            "provider": result["provider"],
            "model": result["model"],
            "items": result["items"],
            "strategy": "compact_single_pass",
            "total_windows": 1,
            "processed_windows": 1,
            "retry_count": max(0, int(result.get("attempt_count", 1)) - 1),
        }

    def _summarize_transcript_map_reduce(
        self,
        *,
        segments: List[dict],
        language: str,
        provider: str,
        max_points: int,
    ) -> dict:
        windows = self._window_chunks_for_summary(
            segments, max_chars=SUMMARY_WINDOW_MAX_CHARS
        )
        if len(windows) <= 1:
            lines = self._summary_transcript_lines(segments)
            return self._summarize_transcript_single_pass(
                transcript_lines=lines,
                segments=segments,
                language=language,
                provider=provider,
                max_points=max_points,
            )

        map_items: List[dict] = []
        resolved_provider = str(provider or DEFAULT_ASK_PROVIDER).strip().lower()
        resolved_model = (
            self.openai_model if resolved_provider == "chatgpt" else self.engine.model
        )
        retries = 0
        processed_windows = 0

        for window_idx, window_segments in enumerate(windows, start=1):
            lines = self._summary_transcript_lines(window_segments)
            if not lines:
                continue
            processed_windows += 1
            system_prompt = self._summary_system_prompt(
                language=language, max_points=SUMMARY_MAP_POINTS
            )
            user_message = (
                f"Window {window_idx}/{len(windows)} transcript:\n\n"
                f"{chr(10).join(lines)}\n\n"
                "Return local highlights for this window in strict JSON."
            )
            map_result = self._summary_llm_with_retries(
                provider=provider,
                system_prompt=system_prompt,
                user_message=user_message,
                segments=window_segments,
                language=language,
                max_points=SUMMARY_MAP_POINTS,
                stage=f"map stage window {window_idx}/{len(windows)}",
                max_tokens=900,
                require_exact_count=False,
            )
            resolved_provider = map_result["provider"]
            resolved_model = map_result["model"]
            retries += max(0, int(map_result.get("attempt_count", 1)) - 1)
            map_items.extend(map_result["items"])

        reduce_prompt = self._summary_system_prompt(
            language=language, max_points=max_points
        )
        reduce_payload = json.dumps({"items": map_items}, ensure_ascii=False, indent=2)
        reduce_message = (
            "Merge the candidate highlights below into a final, non-redundant summary.\n"
            "Make sure the final highlights represent the full transcript timeline.\n"
            "Candidates:\n"
            f"{reduce_payload}\n\n"
            "Return strict JSON using the required schema."
        )
        final_result = self._summary_llm_with_retries(
            provider=provider,
            system_prompt=reduce_prompt,
            user_message=reduce_message,
            segments=segments,
            language=language,
            max_points=max_points,
            stage="reduce stage summary generation",
            max_tokens=SUMMARY_MAX_TOKENS,
            require_exact_count=True,
        )
        resolved_provider = final_result["provider"]
        resolved_model = final_result["model"]
        retries += max(0, int(final_result.get("attempt_count", 1)) - 1)
        return {
            "provider": resolved_provider,
            "model": resolved_model,
            "items": final_result["items"],
            "strategy": "map_reduce",
            "total_windows": len(windows),
            "processed_windows": processed_windows,
            "retry_count": retries,
        }

    def _build_full_transcript_payload(self, rows: List[dict]) -> dict:
        segments = self._coerce_summary_segments(rows)
        full_text = "\n".join(segment["text"] for segment in segments)
        return {
            "version": 1,
            "text": full_text,
            "segments": segments,
            "segment_count": len(segments),
            "char_count": len(full_text),
        }

    def _resolve_summary_full_transcript(self, *, video: dict) -> tuple[dict, bool]:
        full_transcript = video.get("full_transcript")
        if isinstance(full_transcript, dict):
            existing_segments = self._coerce_summary_segments(
                full_transcript.get("segments") or []
            )
            if not existing_segments:
                existing_text = re.sub(
                    r"\s+", " ", str(full_transcript.get("text") or "")
                ).strip()
                if existing_text:
                    existing_segments = [
                        {"start": 0.0, "end": 0.0, "text": existing_text}
                    ]
            if existing_segments:
                payload = self._build_full_transcript_payload(existing_segments)
                video["full_transcript"] = payload
                return payload, False

        raw_chunks = video.get("chunks", [])
        payload = self._build_full_transcript_payload(raw_chunks)
        if not payload["segments"]:
            raise ValueError("video has no transcript segments")

        video["full_transcript"] = payload
        if hasattr(self.engine.library, "save"):
            try:
                self.engine.library.save()
            except Exception:
                pass
        return payload, True

    def summarize_video_transcript(
        self,
        *,
        video_id: str,
        language: str,
        provider: str = DEFAULT_ASK_PROVIDER,
        max_points: int = SUMMARY_MAX_POINTS,
    ) -> dict:
        scoped_video_id = str(video_id or "").strip()
        if not scoped_video_id:
            raise ValueError("video_id is required")
        if scoped_video_id not in self.engine.library.videos:
            raise KeyError(f"video_id not found: {scoped_video_id}")

        scoped_provider = str(provider or DEFAULT_ASK_PROVIDER).strip().lower()
        if scoped_provider not in ASK_PROVIDERS:
            raise ValueError("provider must be one of: chatgpt, claude")

        scoped_language = str(language or "").strip().lower() or "en"
        if scoped_language not in SUMMARY_LANGUAGES:
            raise ValueError("language must be one of: en, ja")

        safe_max_points = SUMMARY_MAX_POINTS

        video = self.engine.library.videos.get(scoped_video_id, {})
        raw_chunks = video.get("chunks", [])
        chunk_count = len(
            [row for row in raw_chunks if str(row.get("raw_text") or "").strip()]
        )
        (
            full_transcript,
            full_transcript_backfilled,
        ) = self._resolve_summary_full_transcript(
            video=video,
        )
        segments = self._coerce_summary_segments(full_transcript.get("segments") or [])
        if not segments:
            raise ValueError("video has no transcript segments")

        transcript_lines = self._summary_transcript_lines(segments)
        transcript_size = len("\n".join(transcript_lines))

        if transcript_size <= SUMMARY_SINGLE_PASS_MAX_CHARS:
            summary_result = self._summarize_transcript_single_pass(
                transcript_lines=transcript_lines,
                segments=segments,
                language=scoped_language,
                provider=scoped_provider,
                max_points=safe_max_points,
            )
        else:
            summary_result = self._summarize_transcript_compact_single_pass(
                segments=segments,
                language=scoped_language,
                provider=scoped_provider,
                max_points=safe_max_points,
            )

        summary_items: List[dict] = []
        timestamp_success_count = 0
        timestamp_failure_count = 0
        used_segment_counts: Dict[int, int] = {}
        for idx, item in enumerate(summary_result["items"], start=1):
            resolved_timestamp = self._resolve_theme_anchor_timestamp(
                item=item,
                segments=segments,
                used_segment_counts=used_segment_counts,
            )
            resolved_segment_index = resolved_timestamp.get("segment_index")
            if isinstance(resolved_segment_index, int):
                used_segment_counts[resolved_segment_index] = (
                    used_segment_counts.get(resolved_segment_index, 0) + 1
                )
            if resolved_timestamp["source"] == "anchor_match":
                timestamp_success_count += 1
            else:
                timestamp_failure_count += 1

            start = float(resolved_timestamp.get("start", 0.0))
            end = float(resolved_timestamp.get("end", start))
            title_default = (
                ("Intro" if idx == 1 else f"Topic {idx - 1}")
                if scoped_language == "en"
                else ("導入" if idx == 1 else f"トピック{idx - 1}")
            )
            summary_items.append(
                {
                    "rank": idx,
                    "title": str(item.get("title") or title_default),
                    "tldr": str(item.get("tldr") or "").strip(),
                    "anchor_text": str(item.get("anchor_text") or "").strip(),
                    "start": start,
                    "end": end,
                    "url": f"https://www.youtube.com/watch?v={scoped_video_id}&t={int(start)}s",
                }
            )

        if timestamp_failure_count == 0:
            timestamp_source = "anchor_match"
        elif timestamp_success_count == 0:
            timestamp_source = "model_fallback"
        else:
            timestamp_source = "mixed"

        return {
            "video_id": scoped_video_id,
            "language": scoped_language,
            "provider": summary_result["provider"],
            "model": summary_result["model"],
            "summary": summary_items,
            "generation_details": {
                "source_basis": "full_transcript",
                "source_chunk_count": chunk_count,
                "source_segment_count": len(segments),
                "full_transcript_backfilled": bool(full_transcript_backfilled),
                "timestamp_source": timestamp_source,
                "timestamp_resolution_success_count": timestamp_success_count,
                "timestamp_resolution_failure_count": timestamp_failure_count,
                "strategy": summary_result["strategy"],
                "total_windows": int(summary_result.get("total_windows", 1)),
                "processed_windows": int(summary_result.get("processed_windows", 1)),
                "retry_count": int(summary_result.get("retry_count", 0)),
            },
        }

    def ask_with_sources(
        self, question: str, sources: List[dict], provider: str = DEFAULT_ASK_PROVIDER
    ) -> dict:
        scoped_provider = str(provider or DEFAULT_ASK_PROVIDER).strip().lower()
        if scoped_provider not in ASK_PROVIDERS:
            raise ValueError("provider must be one of: chatgpt, claude")

        selected_model = (
            self.openai_model if scoped_provider == "chatgpt" else self.engine.model
        )
        if not sources:
            return {
                "answer": "No videos in the library. Please add videos first.",
                "sources": [],
                "provider": scoped_provider,
                "model": selected_model,
            }

        dominant_language = _detect_dominant_language(sources)
        system_prompt = SYSTEM_PROMPTS.get(dominant_language, SYSTEM_PROMPTS["en"])
        system_prompt = (
            f"{system_prompt}\n"
            f"- {ASK_TONE_INSTRUCTION}\n"
            f"- {ASK_STYLE_INSTRUCTION}"
        )
        labels = CONTEXT_LABELS.get(dominant_language, CONTEXT_LABELS["en"])
        context = _format_context(sources)
        user_message = (
            f"{labels['context']}:\n\n{context}\n\n{labels['question']}: {question}"
        )
        llm = self._llm_text_response(
            provider=scoped_provider,
            system_prompt=system_prompt,
            user_message=user_message,
            max_tokens=ASK_MAX_TOKENS,
            temperature=ASK_TEMPERATURE,
        )
        answer = str(llm["text"]).strip()

        return {
            "answer": answer,
            "sources": sources,
            "provider": scoped_provider,
            "model": llm["model"],
        }

    def ingest(self, *, url: str, mode: str, language: str, force: bool):
        started_at = time.time()
        mode = (mode or "single").strip().lower()
        language = normalize_language(language)

        if mode not in {"single", "playlist"}:
            raise ValueError("mode must be single or playlist")

        targets: List[str]
        if mode == "playlist":
            targets = expand_playlist_ids(url)
        else:
            video_id = extract_video_id(url)
            if not video_id:
                raise ValueError("Invalid YouTube URL or video ID")
            targets = [video_id]

        self._append_ingest_log(
            level="info",
            event="ingest.request",
            message="ingest request accepted",
            mode=mode,
            language=language,
            target_count=len(targets),
            force=bool(force),
        )

        created = []
        skipped = []

        for idx, video_id in enumerate(targets):
            if (not force) and (video_id in self.engine.library.videos):
                skipped.append(
                    {
                        "video_id": video_id,
                        "reason": "already_indexed",
                    }
                )
                self._append_ingest_log(
                    level="info",
                    event="job.skipped",
                    message="video already indexed; skipped",
                    video_id=video_id,
                    mode=mode,
                    language=language,
                )
                continue

            job_id = f"job_{uuid.uuid4()}"
            job_started = time.time()
            job = IngestJob(
                job_id=job_id,
                video_id=video_id,
                url=f"https://www.youtube.com/watch?v={video_id}",
                language=language,
                mode=mode,
                status="running",
                attempts=1,
                error_code=None,
                error_message=None,
                created_at=now_iso(),
                updated_at=now_iso(),
            )
            self._store_job(job)
            self._append_ingest_log(
                level="info",
                event="job.created",
                message="ingestion job created",
                job_id=job_id,
                video_id=video_id,
                mode=mode,
                language=language,
                attempts=1,
                status="running",
            )

            try:
                self.engine.library.add_video(video_id, language=language)
                self._hydrate_video_title(video_id)
                self._update_job(job_id, status="completed")
                self._append_ingest_log(
                    level="info",
                    event="job.completed",
                    message="ingestion job completed",
                    job_id=job_id,
                    video_id=video_id,
                    mode=mode,
                    language=language,
                    attempts=1,
                    status="completed",
                    duration_ms=int((time.time() - job_started) * 1000),
                )
                created.append(
                    {
                        "job_id": job_id,
                        "video_id": video_id,
                        "language": language,
                        "status": "completed",
                        "title": self.engine.library.videos.get(video_id, {}).get(
                            "title", f"Video {video_id}"
                        ),
                    }
                )
            except Exception as exc:
                self._update_job(
                    job_id,
                    status="failed",
                    error_code="INGESTION_FAILED",
                    error_message=str(exc)[:500],
                )
                self._append_ingest_log(
                    level="error",
                    event="job.failed",
                    message="ingestion job failed",
                    job_id=job_id,
                    video_id=video_id,
                    mode=mode,
                    language=language,
                    attempts=1,
                    status="failed",
                    error_code="INGESTION_FAILED",
                    error_message=str(exc)[:500],
                    duration_ms=int((time.time() - job_started) * 1000),
                )
                created.append(
                    {
                        "job_id": job_id,
                        "video_id": video_id,
                        "language": language,
                        "status": "failed",
                        "error": str(exc)[:200],
                    }
                )

            # Gentle pacing helps avoid YouTube 429 throttling on playlist ingestion.
            if mode == "playlist" and idx < len(targets) - 1:
                self._append_ingest_log(
                    level="debug",
                    event="playlist.pacing",
                    message="playlist pacing delay before next video",
                    video_id=video_id,
                    delay_ms=1200,
                )
                time.sleep(1.2)

        response = {
            "mode": mode,
            "queued_count": len(created),
            "skipped_count": len(skipped),
            "jobs": created,
            "skipped": skipped,
        }
        self._append_ingest_log(
            level="info",
            event="ingest.complete",
            message="ingest request finished",
            mode=mode,
            language=language,
            queued_count=len(created),
            skipped_count=len(skipped),
            duration_ms=int((time.time() - started_at) * 1000),
        )
        return response


SERVICE = (
    None
    if str(os.environ.get("YT_RAG_SKIP_GLOBAL_SERVICE", "0")).strip().lower()
    in {"1", "true", "yes", "on"}
    else LocalRAGService()
)
WEB_DIR = Path(__file__).resolve().parent / "web"


class Handler(BaseHTTPRequestHandler):
    server_version = "YouTubeRAGLocal/0.1"

    def _set_headers(self, status=200, content_type="application/json; charset=utf-8"):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, DELETE, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header(
            "Cache-Control", "no-store, no-cache, must-revalidate, max-age=0"
        )
        self.send_header("Pragma", "no-cache")
        self.send_header("Expires", "0")
        self.end_headers()

    def _json(self, payload, status=200):
        self._set_headers(status)
        self.wfile.write(
            json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
        )

    def _read_json_body(self):
        length = int(self.headers.get("Content-Length", "0") or "0")
        raw = self.rfile.read(length) if length > 0 else b"{}"
        return json.loads(raw.decode("utf-8"))

    def _serve_static(self, path: str):
        rel = "index.html" if path in {"/", ""} else path.lstrip("/")
        file_path = (WEB_DIR / rel).resolve()

        if WEB_DIR not in file_path.parents and file_path != WEB_DIR:
            self._json(
                {"ok": False, "error": {"code": "NOT_FOUND", "message": "Not found"}},
                404,
            )
            return

        if not file_path.exists() or not file_path.is_file():
            self._json(
                {"ok": False, "error": {"code": "NOT_FOUND", "message": "Not found"}},
                404,
            )
            return

        suffix = file_path.suffix.lower()
        content_type = {
            ".html": "text/html; charset=utf-8",
            ".css": "text/css; charset=utf-8",
            ".js": "application/javascript; charset=utf-8",
            ".json": "application/json; charset=utf-8",
            ".svg": "image/svg+xml",
            ".png": "image/png",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".webp": "image/webp",
            ".ico": "image/x-icon",
        }.get(suffix, "application/octet-stream")

        self._set_headers(200, content_type)
        self.wfile.write(file_path.read_bytes())

    def do_OPTIONS(self):
        self._set_headers(204)

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path

        try:
            if path == "/v1/health":
                self._json(
                    {
                        "ok": True,
                        "mode": "local_preview",
                        "services": {
                            "rag_engine": "ok",
                            "storage": "local_filesystem",
                        },
                    }
                )
                return

            if path == "/v1/ingest/jobs":
                jobs = [asdict(j) for j in SERVICE.list_jobs()[:100]]
                SERVICE._append_ingest_log(
                    level="info",
                    event="jobs.list",
                    message="jobs list requested",
                    count=len(jobs),
                )
                self._json({"ok": True, "jobs": jobs})
                return

            if path.startswith("/v1/ingest/jobs/"):
                job_id = unquote(path.rsplit("/", 1)[-1])
                job = SERVICE.get_job(job_id)
                if not job:
                    self._json(
                        {
                            "ok": False,
                            "error": {
                                "code": "JOB_NOT_FOUND",
                                "message": "Job not found",
                            },
                        },
                        404,
                    )
                    return
                self._json({"ok": True, "job": asdict(job)})
                return

            if path == "/v1/videos":
                videos = SERVICE.list_videos()
                SERVICE._append_ingest_log(
                    level="debug",
                    event="videos.list",
                    message="videos list requested",
                    count=len(videos),
                )
                self._json({"ok": True, "videos": videos})
                return

            if path == "/v1/logs/ingest-jobs":
                params = parse_qs(parsed.query or "")
                limit_raw = (params.get("limit") or [200])[0]
                level = (params.get("level") or [None])[0]
                job_id = (params.get("job_id") or [None])[0]
                video_id = (params.get("video_id") or [None])[0]
                since = (params.get("since") or [None])[0]
                try:
                    limit = int(limit_raw)
                except (TypeError, ValueError):
                    self._json(
                        {
                            "ok": False,
                            "error": {
                                "code": "INVALID_INPUT",
                                "message": "limit must be an integer",
                            },
                        },
                        400,
                    )
                    return

                rows = SERVICE.list_ingest_logs(
                    limit=limit,
                    level=level,
                    job_id=job_id,
                    video_id=video_id,
                    since=since,
                )
                self._json({"ok": True, "count": len(rows), "logs": rows})
                return

            if path == "/v1/feedback/search-review":
                params = parse_qs(parsed.query or "")
                video_id = (params.get("video_id") or [None])[0]
                label = (params.get("label") or [None])[0]
                limit_raw = (params.get("limit") or [500])[0]
                try:
                    limit = int(limit_raw)
                except (TypeError, ValueError):
                    self._json(
                        {
                            "ok": False,
                            "error": {
                                "code": "INVALID_INPUT",
                                "message": "limit must be an integer",
                            },
                        },
                        400,
                    )
                    return
                reviews = SERVICE.list_search_feedback(
                    video_id=video_id,
                    label=label,
                    limit=limit,
                )
                self._json({"ok": True, "count": len(reviews), "reviews": reviews})
                return

            if path == "/v1/feedback/videos":
                videos = SERVICE.list_feedback_videos()
                self._json({"ok": True, "count": len(videos), "videos": videos})
                return

            if path.startswith("/v1/"):
                self._json(
                    {
                        "ok": False,
                        "error": {"code": "NOT_FOUND", "message": "Route not found"},
                    },
                    404,
                )
                return

            self._serve_static(path)
        except Exception as exc:
            traceback.print_exc()
            self._json(
                {"ok": False, "error": {"code": "INTERNAL_ERROR", "message": str(exc)}},
                500,
            )

    def do_DELETE(self):
        parsed = urlparse(self.path)
        path = parsed.path

        try:
            if path == "/v1/feedback/search-review":
                params = parse_qs(parsed.query or "")
                video_id = (params.get("video_id") or [None])[0]
                label = (params.get("label") or [None])[0]
                deleted_count = SERVICE.delete_search_feedback(
                    video_id=video_id,
                    label=label,
                )
                self._json({"ok": True, "deleted_count": deleted_count})
                return

            if path.startswith("/v1/videos/"):
                video_id = unquote(path.rsplit("/", 1)[-1])
                SERVICE.delete_video(video_id)
                self._json({"ok": True, "deleted": {"video_id": video_id}})
                return

            self._json(
                {
                    "ok": False,
                    "error": {"code": "NOT_FOUND", "message": "Route not found"},
                },
                404,
            )
        except SummaryGenerationError as exc:
            self._json(
                {
                    "ok": False,
                    "error": {"code": "SUMMARY_GENERATION_FAILED", "message": str(exc)},
                },
                502,
            )
        except ValueError as exc:
            self._json(
                {"ok": False, "error": {"code": "INVALID_INPUT", "message": str(exc)}},
                400,
            )
        except KeyError as exc:
            self._json(
                {
                    "ok": False,
                    "error": {"code": "VIDEO_NOT_FOUND", "message": str(exc)},
                },
                404,
            )
        except Exception as exc:
            traceback.print_exc()
            self._json(
                {"ok": False, "error": {"code": "INTERNAL_ERROR", "message": str(exc)}},
                500,
            )

    def do_POST(self):
        parsed = urlparse(self.path)
        path = parsed.path

        try:
            body = self._read_json_body()

            if path == "/v1/ingest/videos":
                result = SERVICE.ingest(
                    url=(body.get("url") or "").strip(),
                    mode=(body.get("mode") or "single"),
                    language=(body.get("language") or "ja"),
                    force=bool(body.get("force", False)),
                )
                self._json({"ok": True, **result})
                return

            if path == "/v1/feedback/search-review":
                feedback = SERVICE.save_search_feedback(body)
                self._json({"ok": True, "feedback": feedback})
                return

            if path == "/v1/summaries/transcript":
                video_id = str(body.get("video_id") or "").strip()
                language = str(body.get("language") or "en").strip().lower()
                provider = (
                    str(body.get("provider") or DEFAULT_ASK_PROVIDER).strip().lower()
                )
                max_points = int(body.get("max_points", SUMMARY_MAX_POINTS))

                result = SERVICE.summarize_video_transcript(
                    video_id=video_id,
                    language=language,
                    provider=provider,
                    max_points=max_points,
                )
                self._json({"ok": True, **result})
                return

            if path == "/v1/search":
                query = (body.get("query") or "").strip()
                if not query:
                    self._json(
                        {
                            "ok": False,
                            "error": {
                                "code": "INVALID_QUERY",
                                "message": "query is required",
                            },
                        },
                        400,
                    )
                    return
                k = max(1, min(int(body.get("k", 5)), 12))
                language = body.get("language")
                retrieval_mode = (
                    str(body.get("retrieval_mode") or "hybrid").strip().lower()
                )
                if retrieval_mode not in RETRIEVAL_MODES:
                    self._json(
                        {
                            "ok": False,
                            "error": {
                                "code": "INVALID_INPUT",
                                "message": "retrieval_mode must be one of: dense, hybrid, lexical",
                            },
                        },
                        400,
                    )
                    return
                retrieval = SERVICE.retrieve(
                    query,
                    k=k,
                    language=normalize_language(language) if language else None,
                    retrieval_mode=retrieval_mode,
                )
                self._json(
                    {
                        "ok": True,
                        "query": query,
                        "k": k,
                        "retrieval_mode": retrieval["retrieval_mode"],
                        "retrieval_details": retrieval["details"],
                        "result_count": len(retrieval["results"]),
                        "results": retrieval["results"],
                    }
                )
                return

            if path == "/v1/ask":
                question = (body.get("question") or "").strip()
                if not question:
                    self._json(
                        {
                            "ok": False,
                            "error": {
                                "code": "INVALID_QUESTION",
                                "message": "question is required",
                            },
                        },
                        400,
                    )
                    return
                k = max(1, min(int(body.get("k", 5)), 12))
                language = body.get("language")
                retrieval_mode = (
                    str(body.get("retrieval_mode") or "hybrid").strip().lower()
                )
                provider = (
                    str(body.get("provider") or DEFAULT_ASK_PROVIDER).strip().lower()
                )
                if retrieval_mode not in RETRIEVAL_MODES:
                    self._json(
                        {
                            "ok": False,
                            "error": {
                                "code": "INVALID_INPUT",
                                "message": "retrieval_mode must be one of: dense, hybrid, lexical",
                            },
                        },
                        400,
                    )
                    return

                if provider not in ASK_PROVIDERS:
                    self._json(
                        {
                            "ok": False,
                            "error": {
                                "code": "INVALID_INPUT",
                                "message": "provider must be one of: chatgpt, claude",
                            },
                        },
                        400,
                    )
                    return

                retrieval = SERVICE.retrieve(
                    question,
                    k=k,
                    language=normalize_language(language) if language else None,
                    retrieval_mode=retrieval_mode,
                )
                result = SERVICE.ask_with_sources(
                    question, retrieval["results"], provider=provider
                )
                self._json(
                    {
                        "ok": True,
                        "retrieval_mode": retrieval["retrieval_mode"],
                        "retrieval_details": retrieval["details"],
                        **result,
                    }
                )
                return

            self._json(
                {
                    "ok": False,
                    "error": {"code": "NOT_FOUND", "message": "Route not found"},
                },
                404,
            )
        except ValueError as exc:
            self._json(
                {"ok": False, "error": {"code": "INVALID_INPUT", "message": str(exc)}},
                400,
            )
        except KeyError as exc:
            self._json(
                {
                    "ok": False,
                    "error": {"code": "VIDEO_NOT_FOUND", "message": str(exc)},
                },
                404,
            )
        except Exception as exc:
            traceback.print_exc()
            self._json(
                {"ok": False, "error": {"code": "INTERNAL_ERROR", "message": str(exc)}},
                500,
            )


def main(argv: List[str]) -> int:
    global SERVICE
    if SERVICE is None:
        SERVICE = LocalRAGService()

    host = "127.0.0.1"
    port = 8000
    if len(argv) >= 2:
        port = int(argv[1])

    server = ThreadingHTTPServer((host, port), Handler)
    print(f"Local preview server running at http://{host}:{port}")
    print("Press Ctrl+C to stop")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
