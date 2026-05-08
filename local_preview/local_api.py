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
from copy import deepcopy
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
import hashlib
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

try:
    from dotenv import load_dotenv
except Exception:  # pragma: no cover
    load_dotenv = None


ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

if load_dotenv is not None:
    load_dotenv(ROOT_DIR / ".env")
    load_dotenv(ROOT_DIR / ".env.local", override=True)

from multilingual.rag_engine import RAGEngine  # noqa: E402
from multilingual.text_processing import LANGUAGE_CONFIG  # noqa: E402
from pipelines.embed_ocr import embed_ocr  # noqa: E402
from pipelines.extract_frames import extract_frames  # noqa: E402
from pipelines.run_ocr import run_ocr  # noqa: E402
from pipelines.video_ocr_common import (  # noqa: E402
    ocr_index_metadata_path,
    ocr_index_path,
    ocr_output_path,
    frames_metadata_path,
    validate_video_id as validate_local_video_id,
)
from retrieval.ocr_retriever import OCREvidenceRetriever  # noqa: E402
from retrieval.search_multimodal import merge_evidence  # noqa: E402
from grounded_answer import (  # noqa: E402
    ANSWER_CONFIDENCE_LEVELS,
    ANSWER_STATUSES,
    assess_grounded_answer_evidence,
    build_citation_catalog,
    build_grounded_answer_messages,
    build_retrieved_chunks_payload,
    cap_confidence,
    default_error_answer,
    default_insufficient_answer,
    normalize_grounded_answer_payload,
)


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
FEEDBACK_ALPHA_QUERY = 0.30
FEEDBACK_BETA_GLOBAL = 0.10
FEEDBACK_MAX_ADJUST = 0.35
FEEDBACK_MIN_SIMILARITY = 0.20
FEEDBACK_RECENCY_HALFLIFE_DAYS = 30.0
SUMMARY_LANGUAGES = {"en", "ja"}
SUMMARY_MAX_POINTS = 5
SUMMARY_ALLOWED_POINTS = {5}
SUMMARY_MAP_POINTS = 3
SUMMARY_TEMPERATURE = 0.2
SUMMARY_MAX_TOKENS = 1800
SUMMARY_SINGLE_PASS_MAX_CHARS = 16000
SUMMARY_WINDOW_MAX_CHARS = 7000
SUMMARY_COMPACT_TARGET_CHARS = 18000
SUMMARY_RETRY_ATTEMPTS = 3
SUMMARY_MIN_SENTENCES = 4
SUMMARY_MAX_SENTENCES = 5
SUMMARY_RELAXED_MIN_SENTENCES = 3
SUMMARY_RELAXED_MAX_SENTENCES = 6
SUMMARY_ANCHOR_MIN_CHARS = 8
SUMMARY_ANCHOR_TOKEN_MATCH_THRESHOLD = 0.55
SUMMARY_CACHE_VERSION = 1
ASK_HISTORY_LIMIT_PER_VIDEO = 20
ASK_HISTORY_LIST_LIMIT_MAX = 100
LOCALHOST_CORS_HOSTS = {"127.0.0.1", "localhost", "::1"}
LOCAL_VIDEO_EXTENSIONS = {".mp4", ".m4v", ".mov", ".mkv", ".webm"}
EVIDENCE_INCLUDED_TRUE = {"1", "true", "yes", "included", "eligible"}
EVIDENCE_INCLUDED_FALSE = {"0", "false", "no", "excluded", "ineligible"}


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _chmod_private(path: Path) -> None:
    """Best-effort chmod for local files that may contain transcripts or user history."""
    try:
        os.chmod(path, 0o600)
    except OSError:
        pass


def _allowed_local_origin(origin: Optional[str]) -> Optional[str]:
    text = str(origin or "").strip()
    if not text:
        return None
    parsed = urlparse(text)
    if parsed.scheme not in {"http", "https"}:
        return None
    if parsed.hostname not in LOCALHOST_CORS_HOSTS:
        return None
    return text


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


@dataclass
class LocalVideoOCRJob:
    job_id: str
    video_id: str
    video_path: str
    interval_sec: int
    status: str
    step: str
    frame_count: int
    ocr_count: int
    vector_count: int
    error_code: Optional[str]
    error_message: Optional[str]
    created_at: str
    updated_at: str
    completed_at: Optional[str] = None


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
        self.ocr_jobs: Dict[str, LocalVideoOCRJob] = {}
        self.ocr_lock = threading.Lock()
        self.title_cache: Dict[str, str] = {}
        self.feedback_lock = threading.Lock()
        self.runtime_data_dir = ROOT_DIR / "data" / "runtime"
        self.cache_data_dir = ROOT_DIR / "data" / "cache"
        self.summary_cache_dir = self.cache_data_dir / "summaries"
        self.legacy_data_dir = Path(__file__).resolve().parent / "data"
        self.feedback_path = self.runtime_data_dir / "search_feedback.json"
        self.legacy_feedback_path = self.legacy_data_dir / "search_feedback.json"
        self.feedback: Dict[str, dict] = {}
        self.feedback_index: Dict[str, dict] = {}
        self.feedback_tuning_enabled = str(
            os.environ.get("YT_RAG_FEEDBACK_TUNING", "1")
        ).strip().lower() not in {"0", "false", "off", "no"}
        self.ask_history_lock = threading.Lock()
        self.ask_history_path = self.runtime_data_dir / "ask_history.json"
        self.legacy_ask_history_path = self.legacy_data_dir / "ask_history.json"
        self.ask_history: Dict[str, List[dict]] = {}
        self.log_lock = threading.Lock()
        self.ingest_log_path = self.runtime_data_dir / "ingest_jobs.log"
        self.legacy_ingest_log_path = self.legacy_data_dir / "ingest_jobs.log"
        self._load_feedback()
        self._load_ask_history()

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

    def _ensure_ocr_job_state(self) -> None:
        if not hasattr(self, "ocr_lock"):
            self.ocr_lock = threading.Lock()
        if not hasattr(self, "ocr_jobs") or not isinstance(self.ocr_jobs, dict):
            self.ocr_jobs = {}

    def list_ocr_jobs(self) -> List[LocalVideoOCRJob]:
        self._ensure_ocr_job_state()
        with self.ocr_lock:
            return sorted(
                self.ocr_jobs.values(),
                key=lambda job: job.created_at,
                reverse=True,
            )

    def get_ocr_job(self, job_id: str) -> Optional[LocalVideoOCRJob]:
        self._ensure_ocr_job_state()
        with self.ocr_lock:
            return self.ocr_jobs.get(job_id)

    def _store_ocr_job(self, job: LocalVideoOCRJob) -> None:
        self._ensure_ocr_job_state()
        with self.ocr_lock:
            self.ocr_jobs[job.job_id] = job

    def _update_ocr_job(self, job_id: str, **updates) -> None:
        self._ensure_ocr_job_state()
        with self.ocr_lock:
            job = self.ocr_jobs[job_id]
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
        self._ensure_ask_history_state()
        results = []
        for video_id, data in self.engine.library.videos.items():
            self._hydrate_video_title(video_id)
            refreshed = self.engine.library.videos.get(video_id, data)
            chunking = self.engine.library.get_video_chunking_metadata(video_id)
            history_rows = self.ask_history.get(video_id, [])
            results.append(
                {
                    "video_id": video_id,
                    "title": refreshed.get("title", f"Video {video_id}"),
                    "url": refreshed.get(
                        "url", f"https://www.youtube.com/watch?v={video_id}"
                    ),
                    "language": refreshed.get("language", "ja"),
                    "num_chunks": len(refreshed.get("chunks", [])),
                    "chunking_version": chunking["version"],
                    "chunking_stale": self.engine.library.video_chunking_is_stale(
                        video_id
                    ),
                    "ask_history_count": len(history_rows),
                    "last_ask_at": history_rows[0].get("created_at")
                    if history_rows
                    else None,
                }
            )
        return results

    def delete_video(self, video_id: str):
        self.engine.library.remove_video(video_id)
        self._delete_ask_history(video_id)

    @staticmethod
    def _normalize_local_video_path(video_path: str) -> Path:
        raw = str(video_path or "").strip()
        if not raw:
            raise ValueError("video_path is required")
        parsed = urlparse(raw)
        if parsed.scheme in {"http", "https"}:
            raise ValueError(
                "Local video OCR accepts local files only; do not provide public video URLs."
            )
        path = Path(raw).expanduser()
        if not path.is_absolute():
            path = ROOT_DIR / path
        path = path.resolve()
        if not path.exists():
            raise ValueError(f"video_path does not exist: {path}")
        if not path.is_file():
            raise ValueError(f"video_path must be a file: {path}")
        if path.suffix.lower() not in LOCAL_VIDEO_EXTENSIONS:
            extensions = ", ".join(sorted(LOCAL_VIDEO_EXTENSIONS))
            raise ValueError(f"video_path must be a supported video file: {extensions}")
        return path

    def _local_ocr_data_dir(self) -> Path:
        return ROOT_DIR / "data"

    def start_local_video_ocr_job(
        self,
        *,
        video_id: str,
        video_path: str,
        interval_sec: int = 10,
    ) -> dict:
        scoped_video_id = validate_local_video_id(video_id)
        scoped_path = self._normalize_local_video_path(video_path)
        interval = max(1, int(interval_sec or 10))
        job_id = f"ocr_{uuid.uuid4()}"
        job = LocalVideoOCRJob(
            job_id=job_id,
            video_id=scoped_video_id,
            video_path=str(scoped_path),
            interval_sec=interval,
            status="queued",
            step="queued",
            frame_count=0,
            ocr_count=0,
            vector_count=0,
            error_code=None,
            error_message=None,
            created_at=now_iso(),
            updated_at=now_iso(),
        )
        self._store_ocr_job(job)
        self._append_ingest_log(
            level="info",
            event="local_ocr.job.created",
            message="local video OCR job created",
            job_id=job_id,
            video_id=scoped_video_id,
            interval_sec=interval,
        )

        thread = threading.Thread(
            target=self._run_local_video_ocr_job,
            args=(job_id,),
            daemon=True,
        )
        thread.start()
        return {"job": asdict(job)}

    def _run_local_video_ocr_job(self, job_id: str) -> None:
        job = self.get_ocr_job(job_id)
        if not job:
            return
        data_dir = self._local_ocr_data_dir()
        started_at = time.time()
        try:
            self._update_ocr_job(job_id, status="running", step="extract_frames")
            frames = extract_frames(
                video_path=job.video_path,
                video_id=job.video_id,
                interval_sec=job.interval_sec,
                data_dir=data_dir,
            )
            self._update_ocr_job(
                job_id,
                frame_count=len(frames),
                step="run_ocr",
            )

            ocr_rows = run_ocr(
                frames_path=frames_metadata_path(data_dir, job.video_id),
                output_path=ocr_output_path(data_dir, job.video_id),
            )
            self._update_ocr_job(
                job_id,
                ocr_count=len(ocr_rows),
                step="embed_ocr",
            )

            records = embed_ocr(
                video_id=job.video_id,
                ocr_path=ocr_output_path(data_dir, job.video_id),
                index_path=ocr_index_path(data_dir, job.video_id),
                metadata_path=ocr_index_metadata_path(data_dir, job.video_id),
                processor=self.engine.library.processor,
            )
            completed_at = now_iso()
            self._update_ocr_job(
                job_id,
                status="completed",
                step="completed",
                vector_count=len(records),
                completed_at=completed_at,
            )
            self._append_ingest_log(
                level="info",
                event="local_ocr.job.completed",
                message="local video OCR job completed",
                job_id=job_id,
                video_id=job.video_id,
                frame_count=len(frames),
                ocr_count=len(ocr_rows),
                vector_count=len(records),
                duration_ms=int((time.time() - started_at) * 1000),
            )
        except Exception as exc:
            self._update_ocr_job(
                job_id,
                status="failed",
                step="failed",
                error_code="LOCAL_VIDEO_OCR_FAILED",
                error_message=str(exc)[:500],
                completed_at=now_iso(),
            )
            self._append_ingest_log(
                level="error",
                event="local_ocr.job.failed",
                message="local video OCR job failed",
                job_id=job_id,
                video_id=job.video_id,
                error_code="LOCAL_VIDEO_OCR_FAILED",
                error_message=str(exc)[:500],
                duration_ms=int((time.time() - started_at) * 1000),
            )

    def local_video_ocr_summary(self, video_id: str) -> dict:
        scoped_video_id = validate_local_video_id(video_id)
        data_dir = self._local_ocr_data_dir()

        def _count_jsonl(path: Path) -> int:
            if not path.exists():
                return 0
            return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())

        frame_meta = frames_metadata_path(data_dir, scoped_video_id)
        ocr_meta = ocr_output_path(data_dir, scoped_video_id)
        index_meta = ocr_index_metadata_path(data_dir, scoped_video_id)
        index_file = ocr_index_path(data_dir, scoped_video_id)
        return {
            "video_id": scoped_video_id,
            "frames_metadata_path": str(frame_meta),
            "ocr_metadata_path": str(ocr_meta),
            "ocr_index_path": str(index_file),
            "ocr_index_metadata_path": str(index_meta),
            "frame_count": _count_jsonl(frame_meta),
            "ocr_count": _count_jsonl(ocr_meta),
            "vector_count": _count_jsonl(index_meta),
            "index_exists": index_file.exists(),
        }

    def search_multimodal(
        self,
        *,
        query: str,
        k: int = 5,
        language: Optional[str] = None,
        retrieval_mode: str = "hybrid",
        video_id: Optional[str] = None,
        source_mode: str = "both",
    ) -> dict:
        scoped_source = str(source_mode or "both").strip().lower()
        if scoped_source not in {"both", "transcript", "ocr"}:
            raise ValueError("source_mode must be one of: both, transcript, ocr")

        top_k = max(1, min(int(k), 12))
        candidate_k = max(30, top_k * 8)
        transcript_results: List[dict] = []
        ocr_results: List[dict] = []
        retrieval_details = {}

        if scoped_source in {"both", "transcript"}:
            try:
                retrieval = self.retrieve(
                    query,
                    k=candidate_k,
                    language=language,
                    retrieval_mode=retrieval_mode,
                    video_id=video_id,
                )
                transcript_results = retrieval["results"]
                retrieval_details = retrieval["details"]
            except KeyError:
                if scoped_source == "transcript":
                    raise
                transcript_results = []

        if scoped_source in {"both", "ocr"}:
            retriever = OCREvidenceRetriever(
                data_dir=self._local_ocr_data_dir(),
                processor=self.engine.library.processor,
            )
            ocr_results = retriever.search(
                query,
                video_id=video_id,
                top_k=candidate_k,
                language=language,
            )

        evidence = merge_evidence(
            transcript_results=transcript_results,
            ocr_results=ocr_results,
            top_k=top_k,
        )
        return {
            "query": query,
            "k": top_k,
            "source_mode": scoped_source,
            "retrieval_mode": retrieval_mode,
            "video_id_filter": video_id,
            "result_count": len(evidence),
            "results": evidence,
            "evidence": evidence,
            "details": {
                "transcript_candidates": len(transcript_results),
                "ocr_candidates": len(ocr_results),
                "transcript_retrieval": retrieval_details,
            },
        }

    def _ensure_ask_history_state(self) -> None:
        if not hasattr(self, "ask_history_lock"):
            self.ask_history_lock = threading.Lock()
        if not hasattr(self, "ask_history_path"):
            self.ask_history_path = (
                Path(__file__).resolve().parent / "data" / "ask_history.json"
            )
        if not hasattr(self, "ask_history") or not isinstance(self.ask_history, dict):
            self.ask_history = {}

    @staticmethod
    def _canonical_video_url(video_id: str) -> str:
        return f"https://www.youtube.com/watch?v={video_id}"

    @staticmethod
    def _normalize_ask_history_sources(sources: List[dict]) -> List[dict]:
        normalized = []
        for row in sources or []:
            if not isinstance(row, dict):
                continue
            video_id = str(row.get("video_id") or "").strip()
            if not video_id:
                continue
            start = float(row.get("start", 0.0))
            end = float(row.get("end", start))
            chunk_index = row.get("chunk_index")
            normalized.append(
                {
                    "video_id": video_id,
                    "video_title": str(row.get("video_title") or video_id).strip(),
                    "video_url": str(
                        row.get("video_url")
                        or LocalRAGService._canonical_video_url(video_id)
                    ).strip(),
                    "language": normalize_language(row.get("language"), fallback="ja"),
                    "chunk_index": int(chunk_index)
                    if chunk_index is not None and str(chunk_index).strip() != ""
                    else None,
                    "text": str(row.get("text") or "").strip(),
                    "start": start,
                    "end": end,
                    "url": str(
                        row.get("url")
                        or f"{LocalRAGService._canonical_video_url(video_id)}&t={int(start)}s"
                    ).strip(),
                }
            )
        return normalized

    @staticmethod
    def _normalize_ask_history_citations(citations: List[dict]) -> List[dict]:
        normalized = []
        for row in citations or []:
            if not isinstance(row, dict):
                continue
            video_id = str(row.get("video_id") or "").strip()
            if not video_id:
                continue
            start_seconds = float(row.get("start_seconds", row.get("start", 0.0)))
            end_seconds = float(row.get("end_seconds", row.get("end", start_seconds)))
            citation_id = row.get("citation_id")
            try:
                normalized_citation_id = int(citation_id)
            except (TypeError, ValueError):
                normalized_citation_id = len(normalized) + 1
            chunk_index = row.get("chunk_index")
            try:
                normalized_chunk_index = (
                    int(chunk_index)
                    if chunk_index is not None and str(chunk_index).strip() != ""
                    else None
                )
            except (TypeError, ValueError):
                normalized_chunk_index = None
            normalized.append(
                {
                    "citation_id": normalized_citation_id,
                    "video_id": video_id,
                    "video_title": str(row.get("video_title") or video_id).strip(),
                    "chunk_id": str(
                        row.get("chunk_id")
                        or f"{video_id}:{int(start_seconds * 1000)}:{int(end_seconds * 1000)}"
                    ).strip(),
                    "chunk_index": normalized_chunk_index,
                    "start_seconds": start_seconds,
                    "end_seconds": end_seconds,
                    "timestamp_label": str(row.get("timestamp_label") or "").strip(),
                    "timestamp_range_label": str(
                        row.get("timestamp_range_label") or ""
                    ).strip(),
                    "snippet": str(row.get("snippet") or "").strip(),
                    "reason": str(row.get("reason") or "").strip(),
                    "url": str(row.get("url") or "").strip(),
                }
            )
        return normalized

    @classmethod
    def _ask_history_source_fingerprint(cls, sources: List[dict]) -> str:
        payload = [
            {
                "video_id": str(row.get("video_id") or "").strip(),
                "chunk_index": row.get("chunk_index"),
                "start": round(float(row.get("start", 0.0)), 3),
                "end": round(float(row.get("end", row.get("start", 0.0))), 3),
            }
            for row in cls._normalize_ask_history_sources(sources)
        ]
        serialized = json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()

    def _normalize_ask_history_record(self, video_id: str, row: dict) -> Optional[dict]:
        if not isinstance(row, dict):
            return None
        scoped_video_id = str(video_id or row.get("video_id") or "").strip()
        question = str(row.get("question") or "").strip()
        answer = str(row.get("answer") or "").strip()
        if not scoped_video_id or not question:
            return None

        video = self.engine.library.videos.get(scoped_video_id, {})
        title = str(
            row.get("video_title") or video.get("title") or f"Video {scoped_video_id}"
        ).strip()
        language = normalize_language(
            row.get("language"),
            fallback=normalize_language(video.get("language"), fallback="ja"),
        )
        sources = self._normalize_ask_history_sources(row.get("sources") or [])
        citations = self._normalize_ask_history_citations(row.get("citations") or [])
        chunking = (
            self.engine.library.get_video_chunking_metadata(scoped_video_id)
            if hasattr(self.engine.library, "get_video_chunking_metadata")
            else {"version": "unknown"}
        )
        return {
            "id": str(row.get("id") or f"ask_{uuid.uuid4()}").strip(),
            "video_id": scoped_video_id,
            "video_title": title,
            "url": str(
                row.get("url")
                or video.get("url")
                or self._canonical_video_url(scoped_video_id)
            ).strip(),
            "language": language,
            "question": question,
            "k": max(1, int(row.get("k", 5) or 5)),
            "retrieval_mode": str(row.get("retrieval_mode") or "hybrid")
            .strip()
            .lower(),
            "provider": str(row.get("provider") or DEFAULT_ASK_PROVIDER)
            .strip()
            .lower(),
            "model": str(row.get("model") or "").strip(),
            "status": str(row.get("status") or "answered").strip().lower()
            if str(row.get("status") or "answered").strip().lower() in ANSWER_STATUSES
            else "answered",
            "confidence": str(row.get("confidence") or "low").strip().lower()
            if str(row.get("confidence") or "low").strip().lower()
            in ANSWER_CONFIDENCE_LEVELS
            else "low",
            "answer": answer,
            "sources": sources,
            "citations": citations,
            "warnings": [
                str(item).strip()
                for item in (row.get("warnings") or [])
                if str(item).strip()
            ],
            "retrieved_chunks": deepcopy(row.get("retrieved_chunks") or []),
            "retrieval_details": deepcopy(row.get("retrieval_details") or {}),
            "created_at": str(row.get("created_at") or now_iso()).strip(),
            "chunking_version": str(
                row.get("chunking_version") or chunking.get("version") or "unknown"
            ).strip(),
            "source_fingerprint": str(
                row.get("source_fingerprint")
                or self._ask_history_source_fingerprint(sources)
            ).strip(),
        }

    @staticmethod
    def _resolve_existing_path(*paths: Optional[Path]) -> Optional[Path]:
        for path in paths:
            if path and path.exists():
                return path
        return None

    def _load_ask_history(self) -> None:
        self._ensure_ask_history_state()
        self.ask_history_path.parent.mkdir(parents=True, exist_ok=True)
        source_path = self._resolve_existing_path(
            self.ask_history_path, self.legacy_ask_history_path
        )
        if source_path is None:
            return
        try:
            raw = json.loads(source_path.read_text(encoding="utf-8"))
        except Exception:
            self.ask_history = {}
            return

        normalized: Dict[str, List[dict]] = {}
        if isinstance(raw, dict):
            iterable = raw.items()
        else:
            iterable = []

        for video_id, rows in iterable:
            if not isinstance(rows, list):
                continue
            items = []
            for row in rows:
                record = self._normalize_ask_history_record(video_id, row)
                if record:
                    items.append(record)
            items.sort(key=lambda row: row.get("created_at", ""), reverse=True)
            if items:
                normalized[str(video_id).strip()] = items[:ASK_HISTORY_LIMIT_PER_VIDEO]

        self.ask_history = normalized

    def _persist_ask_history(self) -> None:
        self._ensure_ask_history_state()
        self.ask_history_path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = self.ask_history_path.with_suffix(".tmp")
        temp_path.write_text(
            json.dumps(self.ask_history, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        _chmod_private(temp_path)
        temp_path.replace(self.ask_history_path)
        _chmod_private(self.ask_history_path)

    def save_ask_history(self, payload: dict) -> dict:
        self._ensure_ask_history_state()
        record = self._normalize_ask_history_record(payload.get("video_id"), payload)
        if not record:
            raise ValueError("video_id and question are required")
        video_id = record["video_id"]
        with self.ask_history_lock:
            rows = list(self.ask_history.get(video_id, []))
            rows.insert(0, record)
            rows.sort(key=lambda row: row.get("created_at", ""), reverse=True)
            self.ask_history[video_id] = rows[:ASK_HISTORY_LIMIT_PER_VIDEO]
            self._persist_ask_history()
        return deepcopy(record)

    def list_ask_history(
        self,
        *,
        video_id: Optional[str] = None,
        limit: int = ASK_HISTORY_LIMIT_PER_VIDEO,
    ) -> List[dict]:
        self._ensure_ask_history_state()
        safe_limit = max(
            1,
            min(int(limit or ASK_HISTORY_LIMIT_PER_VIDEO), ASK_HISTORY_LIST_LIMIT_MAX),
        )
        scoped_video_id = str(video_id or "").strip()
        if scoped_video_id:
            rows = list(self.ask_history.get(scoped_video_id, []))
        else:
            rows = [row for items in self.ask_history.values() for row in items]
        rows.sort(key=lambda row: row.get("created_at", ""), reverse=True)
        return deepcopy(rows[:safe_limit])

    def _delete_ask_history(self, video_id: str) -> None:
        self._ensure_ask_history_state()
        scoped_video_id = str(video_id or "").strip()
        if not scoped_video_id:
            return
        with self.ask_history_lock:
            if scoped_video_id not in self.ask_history:
                return
            del self.ask_history[scoped_video_id]
            self._persist_ask_history()

    def _load_feedback(self) -> None:
        self.feedback_path.parent.mkdir(parents=True, exist_ok=True)
        source_path = self._resolve_existing_path(
            self.feedback_path, self.legacy_feedback_path
        )
        if source_path is None:
            return
        try:
            raw = json.loads(source_path.read_text(encoding="utf-8"))
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
                existing = normalized.get(record["key"])
                if existing:
                    existing_dt = self._parse_iso_datetime(
                        existing.get("updated_at")
                    ) or self._parse_iso_datetime(existing.get("created_at"))
                    record_dt = self._parse_iso_datetime(
                        record.get("updated_at")
                    ) or self._parse_iso_datetime(record.get("created_at"))
                    if existing_dt and record_dt and existing_dt > record_dt:
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
        _chmod_private(temp_path)
        temp_path.replace(self.feedback_path)
        _chmod_private(self.feedback_path)

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

    @staticmethod
    def _normalize_feedback_query(query: str) -> str:
        return " ".join(str(query or "").strip().lower().split())

    @classmethod
    def _feedback_query_hash(cls, query: str, retrieval_mode: str) -> str:
        payload = json.dumps(
            {
                "query": cls._normalize_feedback_query(query),
                "retrieval_mode": str(retrieval_mode or "hybrid").strip().lower(),
            },
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]

    @staticmethod
    def _feedback_record_key(chunk_key: str, query_hash: str) -> str:
        return f"{chunk_key}:{query_hash}"

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

        chunk_key = str(
            row.get("chunk_key") or row.get("feedback_key") or ""
        ).strip() or self._feedback_key(video_id, chunk_index, start, end)
        created_at = str(row.get("created_at") or now_iso())
        updated_at = str(row.get("updated_at") or created_at)
        query = str(row.get("query") or "")
        retrieval_mode = str(row.get("retrieval_mode") or "hybrid").strip().lower()
        if retrieval_mode not in RETRIEVAL_MODES:
            retrieval_mode = "hybrid"
        query_hash = str(row.get("query_hash") or "").strip() or (
            self._feedback_query_hash(query, retrieval_mode)
        )
        key = self._feedback_record_key(chunk_key, query_hash)
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
            "chunk_key": chunk_key,
            "query_hash": query_hash,
            "query": query,
            "query_language": query_language,
            "query_tokens": normalized_tokens,
            "retrieval_mode": retrieval_mode,
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
            key = str(record.get("chunk_key") or record.get("key") or "").strip()
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

    def _existing_log_paths(self) -> List[Path]:
        paths: List[Path] = []
        seen = set()
        for candidate in (self.ingest_log_path, self.legacy_ingest_log_path):
            if candidate and candidate.exists():
                key = str(candidate.resolve())
                if key in seen:
                    continue
                seen.add(key)
                paths.append(candidate)
        return paths

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

        log_paths = self._existing_log_paths()
        if not log_paths:
            return []

        rows: List[dict] = []
        with self.log_lock:
            for log_path in log_paths:
                for line in log_path.read_text(encoding="utf-8").splitlines():
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    if (
                        scoped_level
                        and str(row.get("level") or "").lower() != scoped_level
                    ):
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

    # ------------------------------------------------------------------
    # Evidence curation artifacts (read-only)
    # ------------------------------------------------------------------

    def _evidence_artifact_paths(self) -> Dict[str, Path]:
        return {
            "pipeline_runs": self.runtime_data_dir / "pipeline_runs.jsonl",
            "model_inference_results": self.runtime_data_dir
            / "model_inference_results.jsonl",
            "manifest": self.runtime_data_dir / "curated_evidence_manifest.jsonl",
            "quality_report": self.runtime_data_dir / "evidence_quality_report.json",
        }

    @staticmethod
    def _read_jsonl_file(path: Path) -> List[dict]:
        if not path.exists():
            return []
        rows: List[dict] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            text = line.strip()
            if not text:
                continue
            try:
                row = json.loads(text)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict):
                rows.append(row)
        return rows

    @staticmethod
    def _count_jsonl_file(path: Path) -> int:
        if not path.exists():
            return 0
        return sum(
            1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
        )

    @staticmethod
    def _read_json_file(path: Path) -> Optional[dict]:
        if not path.exists():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return None
        return payload if isinstance(payload, dict) else None

    @staticmethod
    def _coerce_evidence_bool_filter(value: Optional[str]) -> Optional[bool]:
        scoped = str(value or "").strip().lower()
        if not scoped:
            return None
        if scoped in EVIDENCE_INCLUDED_TRUE:
            return True
        if scoped in EVIDENCE_INCLUDED_FALSE:
            return False
        raise ValueError("included must be one of: true, false, included, excluded")

    def evidence_curation_summary(self) -> dict:
        paths = self._evidence_artifact_paths()
        report = self._read_json_file(paths["quality_report"]) or {}
        runs = self._read_jsonl_file(paths["pipeline_runs"])
        latest_run = runs[-1] if runs else None
        artifacts = {
            name: {
                "path": str(path),
                "exists": path.exists(),
                "row_count": self._count_jsonl_file(path)
                if path.suffix == ".jsonl"
                else (1 if path.exists() else 0),
            }
            for name, path in paths.items()
        }
        return {
            "available": bool(report or artifacts["manifest"]["row_count"]),
            "report": report,
            "latest_run": latest_run,
            "artifacts": artifacts,
        }

    def list_evidence_curation_runs(self, *, limit: int = 25) -> List[dict]:
        safe_limit = max(1, min(int(limit), 500))
        rows = self._read_jsonl_file(self._evidence_artifact_paths()["pipeline_runs"])
        rows.sort(key=lambda row: str(row.get("started_at") or ""), reverse=True)
        return rows[:safe_limit]

    def list_evidence_manifest(
        self,
        *,
        video_id: Optional[str] = None,
        quality_label: Optional[str] = None,
        included: Optional[str] = None,
        topic: Optional[str] = None,
        q: Optional[str] = None,
        limit: int = 500,
        offset: int = 0,
    ) -> dict:
        safe_limit = max(1, min(int(limit), 5000))
        safe_offset = max(0, int(offset))
        scoped_video = str(video_id or "").strip()
        scoped_quality = str(quality_label or "").strip()
        scoped_topic = str(topic or "").strip().lower()
        scoped_query = str(q or "").strip().lower()
        scoped_included = self._coerce_evidence_bool_filter(included)

        rows = self._read_jsonl_file(self._evidence_artifact_paths()["manifest"])
        if scoped_video:
            rows = [
                row for row in rows if str(row.get("video_id") or "") == scoped_video
            ]
        if scoped_quality:
            rows = [
                row
                for row in rows
                if str(row.get("quality_label") or "") == scoped_quality
            ]
        if scoped_included is not None:
            rows = [
                row
                for row in rows
                if bool(
                    row.get("included")
                    if "included" in row
                    else row.get("retrieval_eligible")
                )
                is scoped_included
            ]
        if scoped_topic:
            rows = [
                row
                for row in rows
                if scoped_topic
                in {str(tag).lower() for tag in row.get("topic_tags") or []}
            ]
        if scoped_query:
            rows = [
                row
                for row in rows
                if scoped_query
                in " ".join(
                    [
                        str(row.get("evidence_id") or ""),
                        str(row.get("video_id") or ""),
                        str(row.get("video_title") or ""),
                        str(row.get("text") or ""),
                    ]
                ).lower()
            ]

        total = len(rows)
        paged_rows = rows[safe_offset : safe_offset + safe_limit]
        return {
            "count": len(paged_rows),
            "total": total,
            "limit": safe_limit,
            "offset": safe_offset,
            "rows": paged_rows,
        }

    def list_evidence_inferences(
        self,
        *,
        evidence_id: Optional[str] = None,
        pipeline_run_id: Optional[str] = None,
        limit: int = 100,
    ) -> List[dict]:
        safe_limit = max(1, min(int(limit), 1000))
        scoped_evidence_id = str(evidence_id or "").strip()
        scoped_pipeline_run_id = str(pipeline_run_id or "").strip()
        rows = self._read_jsonl_file(
            self._evidence_artifact_paths()["model_inference_results"]
        )
        if scoped_evidence_id:
            rows = [
                row
                for row in rows
                if str(row.get("evidence_id") or "") == scoped_evidence_id
            ]
        if scoped_pipeline_run_id:
            rows = [
                row
                for row in rows
                if str(row.get("pipeline_run_id") or "") == scoped_pipeline_run_id
            ]
        rows.sort(key=lambda row: str(row.get("created_at") or ""), reverse=True)
        return rows[:safe_limit]

    # ------------------------------------------------------------------
    # Chunking comparison (ephemeral re-chunk + search)
    # ------------------------------------------------------------------

    CHUNKING_STRATEGIES = {"time", "sentence", "token"}

    def _get_video_with_transcript(self, video_id: str):
        """Return video data dict, raising ValueError if missing transcript."""
        video_id = str(video_id or "").strip()
        if not video_id:
            raise ValueError("video_id is required")
        video = self.engine.library.videos.get(video_id)
        if not video:
            raise KeyError(f"Video {video_id} not found in library")
        ft = video.get("full_transcript")
        if not ft or not ft.get("segments"):
            raise ValueError(
                f"Video {video_id} has no full_transcript. "
                "Re-ingest this video to generate one."
            )
        return video, ft

    def preview_chunking(self, video_id, strategy, params):
        """Re-chunk a video in memory and return chunk preview (no embeddings)."""
        video, ft = self._get_video_with_transcript(video_id)
        language = video.get("language", "ja")
        processor = self.engine.library.processor

        lines = processor.reconstruct_lines_from_transcript(ft, language)
        chunks = self._apply_chunking_strategy(
            processor, lines, strategy, params, language
        )

        chunk_list = []
        for i, c in enumerate(chunks):
            chunk_list.append(
                {
                    "index": i,
                    "start": c["start"],
                    "end": c["end"],
                    "raw_text": c["raw_text"],
                    "char_count": len(c["raw_text"]),
                }
            )

        char_counts = [c["char_count"] for c in chunk_list] or [0]
        return {
            "chunk_count": len(chunk_list),
            "chunks": chunk_list,
            "stats": {
                "total_chars": sum(char_counts),
                "avg_chunk_chars": round(
                    sum(char_counts) / max(1, len(char_counts)), 1
                ),
                "min_chunk_chars": min(char_counts),
                "max_chunk_chars": max(char_counts),
            },
        }

    def search_with_chunking(
        self, video_id, strategy, params, query, k, language_override
    ):
        """Re-chunk, embed, build temp FAISS index, search, return ranked results."""
        import faiss as _faiss

        video, ft = self._get_video_with_transcript(video_id)
        language = language_override or video.get("language", "ja")
        processor = self.engine.library.processor

        lines = processor.reconstruct_lines_from_transcript(ft, language)
        chunks = self._apply_chunking_strategy(
            processor, lines, strategy, params, language
        )
        if not chunks:
            return {
                "chunk_count": 0,
                "results": [],
                "embedding_time_ms": 0,
                "search_time_ms": 0,
            }

        t0 = time.monotonic()
        embeddings = processor.generate_embeddings(chunks)
        embedding_time_ms = round((time.monotonic() - t0) * 1000, 1)

        _faiss.omp_set_num_threads(1)
        dim = embeddings.shape[1]
        index = _faiss.IndexFlatIP(dim)
        index.add(embeddings)

        t1 = time.monotonic()
        query_emb = processor.encode_query(query, language=language)
        safe_k = min(k, len(chunks))
        scores, indices = index.search(query_emb, safe_k)
        search_time_ms = round((time.monotonic() - t1) * 1000, 1)

        results = []
        for rank, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if idx < 0:
                continue
            c = chunks[idx]
            results.append(
                {
                    "rank": rank + 1,
                    "score": round(float(score), 4),
                    "chunk_index": int(idx),
                    "start": c["start"],
                    "end": c["end"],
                    "text": c["raw_text"],
                }
            )

        return {
            "chunk_count": len(chunks),
            "results": results,
            "embedding_time_ms": embedding_time_ms,
            "search_time_ms": search_time_ms,
        }

    @staticmethod
    def _apply_chunking_strategy(processor, lines, strategy, params, language):
        """Dispatch to the appropriate chunking method."""
        if strategy == "time":
            window = max(10, min(int(params.get("window", 60)), 300))
            overlap = max(0, min(int(params.get("overlap", 15)), window - 1))
            return processor.chunk_by_time_with_overlap(
                lines, window=window, overlap=overlap
            )
        elif strategy == "sentence":
            max_chars = max(100, min(int(params.get("max_chars", 1000)), 5000))
            return processor.chunk_by_sentence_boundary(lines, max_chars=max_chars)
        elif strategy == "token":
            token_count = max(32, min(int(params.get("token_count", 256)), 1024))
            overlap_fraction = max(
                0.0, min(float(params.get("overlap_fraction", 0.25)), 0.5)
            )
            return processor.chunk_by_token_count(
                lines,
                token_count=token_count,
                overlap_fraction=overlap_fraction,
                language=language,
            )
        else:
            raise ValueError(
                f"Unknown strategy: {strategy}. Must be one of: time, sentence, token"
            )

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
        chunk_key = self._feedback_key(video_id, chunk_index, start, end)
        query_hash = self._feedback_query_hash(query, retrieval_mode)
        identity = self._feedback_record_key(chunk_key, query_hash)
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
                "chunk_key": chunk_key,
                "query_hash": query_hash,
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

    def _all_chunks(self, video_id: Optional[str] = None) -> List[dict]:
        rows: List[dict] = []
        scoped_video_id = str(video_id or "").strip()
        for current_video_id, video_data in self.engine.library.videos.items():
            if scoped_video_id and current_video_id != scoped_video_id:
                continue
            title = video_data.get("title", f"Video {current_video_id}")
            video_url = video_data.get(
                "url", f"https://www.youtube.com/watch?v={current_video_id}"
            )
            language = video_data.get("language", "ja")
            chunks = video_data.get("chunks", [])
            for chunk_idx, chunk in enumerate(chunks):
                start = float(chunk.get("start", 0.0))
                rows.append(
                    {
                        "video_id": current_video_id,
                        "video_title": title,
                        "video_url": video_url,
                        "language": language,
                        "chunk_index": chunk_idx,
                        "text": chunk.get("raw_text", ""),
                        "start": start,
                        "end": float(chunk.get("end", start)),
                        "url": f"https://www.youtube.com/watch?v={current_video_id}&t={int(start)}s",
                    }
                )
        return rows

    def _dense_search(
        self,
        query: str,
        k: int,
        language: Optional[str],
        video_id: Optional[str] = None,
    ) -> List[dict]:
        try:
            if video_id is None:
                dense_rows = self.engine.search(query, k=k, language=language)
            else:
                dense_rows = self.engine.search(
                    query,
                    k=k,
                    language=language,
                    video_id=video_id,
                )
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
        self,
        query: str,
        k: int,
        language: Optional[str],
        video_id: Optional[str] = None,
    ) -> List[dict]:
        candidates = self._all_chunks(video_id=video_id)
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
        video_id: Optional[str] = None,
    ) -> dict:
        mode = (retrieval_mode or "hybrid").strip().lower()
        if mode not in RETRIEVAL_MODES:
            raise ValueError(
                f"retrieval_mode must be one of: {', '.join(sorted(RETRIEVAL_MODES))}"
            )
        scoped_video_id = str(video_id or "").strip()
        if scoped_video_id and scoped_video_id not in self.engine.library.videos:
            raise KeyError(f"video_id not found: {scoped_video_id}")

        top_k = max(1, min(int(k), 12))
        candidate_k = max(30, top_k * 8)

        dense_results: List[dict] = []
        lexical_results: List[dict] = []
        candidate_rows: List[dict] = []

        if mode in {"dense", "hybrid"}:
            if scoped_video_id:
                dense_results = self._dense_search(
                    query,
                    k=candidate_k,
                    language=language,
                    video_id=scoped_video_id,
                )
            else:
                dense_results = self._dense_search(
                    query,
                    k=candidate_k,
                    language=language,
                )
        if mode in {"lexical", "hybrid"}:
            if scoped_video_id:
                lexical_results = self._lexical_bm25_search(
                    query,
                    k=candidate_k,
                    language=language,
                    video_id=scoped_video_id,
                )
            else:
                lexical_results = self._lexical_bm25_search(
                    query,
                    k=candidate_k,
                    language=language,
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
            "video_id_filter": scoped_video_id or None,
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
        self,
        items: List[dict],
        language: str,
        *,
        min_sentences: int = SUMMARY_MIN_SENTENCES,
        max_sentences: int = SUMMARY_MAX_SENTENCES,
    ) -> bool:
        if not items:
            return False
        for row in items:
            sentence_count = self._summary_sentence_count(
                str(row.get("tldr") or ""), language
            )
            if sentence_count < min_sentences or sentence_count > max_sentences:
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
            "Make each theme title concrete and descriptive, not generic.\n"
            "When the transcript clearly identifies who is speaking, being discussed, or which character/persona they are playing, include that detail in the title or tldr.\n"
            "Only mention speaker names or character roles when the transcript supports them; never guess.\n"
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
        min_sentences: int = SUMMARY_MIN_SENTENCES,
        max_sentences: int = SUMMARY_MAX_SENTENCES,
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
            f"Each tldr must be exactly {min_sentences}-{max_sentences} sentences.\n"
            "Make titles more specific and informative when possible.\n"
            "If the transcript clearly names the speaker, participant, or character role, preserve that detail in the title or tldr.\n"
            "Do not invent names, roles, or casting information that is not explicit in the transcript.\n"
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
        min_sentences: int = SUMMARY_MIN_SENTENCES,
        max_sentences: int = SUMMARY_MAX_SENTENCES,
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
                        items,
                        language,
                        min_sentences=min_sentences,
                        max_sentences=max_sentences,
                    )
                ):
                    items = self._rewrite_summary_items_for_quality(
                        provider=provider,
                        language=language,
                        max_points=max_points,
                        segments=segments,
                        items=items,
                        min_sentences=min_sentences,
                        max_sentences=max_sentences,
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
                        items,
                        language,
                        min_sentences=min_sentences,
                        max_sentences=max_sentences,
                    ):
                        raise ValueError(
                            f"Summary output must be {min_sentences}-{max_sentences} sentences per item after rewrite."
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
        min_sentences: int = SUMMARY_MIN_SENTENCES,
        max_sentences: int = SUMMARY_MAX_SENTENCES,
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
            min_sentences=min_sentences,
            max_sentences=max_sentences,
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
        min_sentences: int = SUMMARY_MIN_SENTENCES,
        max_sentences: int = SUMMARY_MAX_SENTENCES,
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
            min_sentences=min_sentences,
            max_sentences=max_sentences,
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
        min_sentences: int = SUMMARY_MIN_SENTENCES,
        max_sentences: int = SUMMARY_MAX_SENTENCES,
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
                min_sentences=min_sentences,
                max_sentences=max_sentences,
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
                min_sentences=min_sentences,
                max_sentences=max_sentences,
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
            min_sentences=min_sentences,
            max_sentences=max_sentences,
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

    @staticmethod
    def _summary_cache_key(*, language: str, provider: str, max_points: int) -> str:
        return f"{str(language).strip().lower()}:{str(provider).strip().lower()}:{int(max_points)}"

    def _summary_cache_path(self, video_id: str) -> Path:
        safe_video_id = str(video_id or "").strip()
        return self.summary_cache_dir / f"{safe_video_id}.json"

    def _load_summary_cache_rows(self, *, video_id: str, video: dict) -> dict:
        cache_path = self._summary_cache_path(video_id)
        cache_rows: dict = {}
        if cache_path.exists():
            try:
                raw = json.loads(cache_path.read_text(encoding="utf-8"))
                if isinstance(raw, dict):
                    cache_rows = raw
            except Exception:
                cache_rows = {}
        elif isinstance(video.get("summary_cache"), dict):
            cache_rows = deepcopy(video.get("summary_cache") or {})

        video["summary_cache"] = deepcopy(cache_rows)
        return cache_rows

    def _persist_summary_cache_rows(self, *, video_id: str, cache_rows: dict) -> None:
        self.summary_cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path = self._summary_cache_path(video_id)
        temp_path = cache_path.with_suffix(".tmp")
        temp_path.write_text(
            json.dumps(cache_rows, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        _chmod_private(temp_path)
        temp_path.replace(cache_path)
        _chmod_private(cache_path)

    def _summary_source_fingerprint(self, *, full_transcript: dict) -> str:
        segments = self._coerce_summary_segments(full_transcript.get("segments") or [])
        normalized_segments = [
            {
                "start": round(float(seg.get("start", 0.0)), 3),
                "end": round(float(seg.get("end", seg.get("start", 0.0))), 3),
                "text": re.sub(r"\s+", " ", str(seg.get("text") or "")).strip(),
            }
            for seg in segments
        ]
        payload = {
            "segments": normalized_segments,
            "text": re.sub(r"\s+", " ", str(full_transcript.get("text") or "")).strip(),
            "segment_count": int(full_transcript.get("segment_count") or 0),
        }
        serialized = json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()

    def _get_summary_cache(
        self,
        *,
        video_id: str,
        video: dict,
        cache_key: str,
        source_fingerprint: str,
    ) -> Optional[dict]:
        cache_rows = self._load_summary_cache_rows(video_id=video_id, video=video)

        entry = cache_rows.get(cache_key)
        if not isinstance(entry, dict):
            return None

        if int(entry.get("version") or 0) != SUMMARY_CACHE_VERSION:
            return None
        if str(entry.get("source_fingerprint") or "") != source_fingerprint:
            return None

        result = entry.get("result")
        if not isinstance(result, dict):
            return None

        return {
            "result": deepcopy(result),
            "generated_at": str(entry.get("generated_at") or ""),
        }

    def _put_summary_cache(
        self,
        *,
        video_id: str,
        video: dict,
        cache_key: str,
        source_fingerprint: str,
        language: str,
        provider: str,
        max_points: int,
        model: str,
        response_payload: dict,
    ) -> None:
        cache_rows = self._load_summary_cache_rows(video_id=video_id, video=video)

        cache_rows[cache_key] = {
            "version": SUMMARY_CACHE_VERSION,
            "language": str(language).strip().lower(),
            "provider": str(provider).strip().lower(),
            "max_points": int(max_points),
            "model": str(model or "").strip(),
            "generated_at": now_iso(),
            "source_fingerprint": source_fingerprint,
            "result": deepcopy(response_payload),
        }
        video["summary_cache"] = deepcopy(cache_rows)
        self._persist_summary_cache_rows(video_id=video_id, cache_rows=cache_rows)

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

        safe_max_points = int(max_points or SUMMARY_MAX_POINTS)
        if safe_max_points not in SUMMARY_ALLOWED_POINTS:
            raise ValueError("max_points must be 5")

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

        cache_key = self._summary_cache_key(
            language=scoped_language,
            provider=scoped_provider,
            max_points=safe_max_points,
        )
        source_fingerprint = self._summary_source_fingerprint(
            full_transcript=full_transcript
        )
        cached_summary = self._get_summary_cache(
            video_id=scoped_video_id,
            video=video,
            cache_key=cache_key,
            source_fingerprint=source_fingerprint,
        )
        if cached_summary is not None:
            response = cached_summary["result"]
            response["video_id"] = scoped_video_id
            response["language"] = scoped_language
            response["provider"] = scoped_provider
            response["cached"] = True
            generation_details = dict(response.get("generation_details") or {})
            generation_details["source_basis"] = "full_transcript"
            generation_details["source_chunk_count"] = chunk_count
            generation_details["source_segment_count"] = len(segments)
            generation_details["full_transcript_backfilled"] = bool(
                full_transcript_backfilled
            )
            generation_details["cache_hit"] = True
            generation_details["cache_key"] = cache_key
            generation_details["cache_generated_at"] = (
                cached_summary.get("generated_at") or None
            )
            generation_details.setdefault(
                "primary_strategy", generation_details.get("strategy") or None
            )
            generation_details.setdefault("fallback_applied", False)
            generation_details.setdefault("fallback_reason", None)
            generation_details.setdefault("validation_relaxed", False)
            response["generation_details"] = generation_details
            return response

        transcript_lines = self._summary_transcript_lines(segments)
        transcript_size = len("\n".join(transcript_lines))
        fallback_applied = False
        fallback_reason: Optional[str] = None
        validation_relaxed = False

        if transcript_size <= SUMMARY_SINGLE_PASS_MAX_CHARS:
            primary_strategy = "single_pass"
            try:
                summary_result = self._summarize_transcript_single_pass(
                    transcript_lines=transcript_lines,
                    segments=segments,
                    language=scoped_language,
                    provider=scoped_provider,
                    max_points=safe_max_points,
                )
            except SummaryGenerationError as exc:
                fallback_applied = True
                fallback_reason = str(exc)
                validation_relaxed = True
                summary_result = self._summarize_transcript_single_pass(
                    transcript_lines=transcript_lines,
                    segments=segments,
                    language=scoped_language,
                    provider=scoped_provider,
                    max_points=safe_max_points,
                    min_sentences=SUMMARY_RELAXED_MIN_SENTENCES,
                    max_sentences=SUMMARY_RELAXED_MAX_SENTENCES,
                )
        else:
            primary_strategy = "compact_single_pass"
            try:
                summary_result = self._summarize_transcript_compact_single_pass(
                    segments=segments,
                    language=scoped_language,
                    provider=scoped_provider,
                    max_points=safe_max_points,
                )
            except SummaryGenerationError as exc:
                fallback_applied = True
                fallback_reason = str(exc)
                try:
                    summary_result = self._summarize_transcript_map_reduce(
                        segments=segments,
                        language=scoped_language,
                        provider=scoped_provider,
                        max_points=safe_max_points,
                    )
                except SummaryGenerationError as reduce_exc:
                    validation_relaxed = True
                    fallback_reason = (
                        f"{fallback_reason}; " f"reduce stage failed: {reduce_exc}"
                    )
                    summary_result = self._summarize_transcript_map_reduce(
                        segments=segments,
                        language=scoped_language,
                        provider=scoped_provider,
                        max_points=safe_max_points,
                        min_sentences=SUMMARY_RELAXED_MIN_SENTENCES,
                        max_sentences=SUMMARY_RELAXED_MAX_SENTENCES,
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

        response = {
            "video_id": scoped_video_id,
            "language": scoped_language,
            "provider": summary_result["provider"],
            "model": summary_result["model"],
            "cached": False,
            "summary": summary_items,
            "generation_details": {
                "source_basis": "full_transcript",
                "source_chunk_count": chunk_count,
                "source_segment_count": len(segments),
                "full_transcript_backfilled": bool(full_transcript_backfilled),
                "cache_hit": False,
                "cache_key": cache_key,
                "cache_generated_at": None,
                "timestamp_source": timestamp_source,
                "timestamp_resolution_success_count": timestamp_success_count,
                "timestamp_resolution_failure_count": timestamp_failure_count,
                "strategy": summary_result["strategy"],
                "primary_strategy": primary_strategy,
                "fallback_applied": bool(fallback_applied),
                "fallback_reason": fallback_reason,
                "validation_relaxed": bool(validation_relaxed),
                "total_windows": int(summary_result.get("total_windows", 1)),
                "processed_windows": int(summary_result.get("processed_windows", 1)),
                "retry_count": int(summary_result.get("retry_count", 0)),
            },
        }
        self._put_summary_cache(
            video_id=scoped_video_id,
            video=video,
            cache_key=cache_key,
            source_fingerprint=source_fingerprint,
            language=scoped_language,
            provider=scoped_provider,
            max_points=safe_max_points,
            model=str(summary_result.get("model") or ""),
            response_payload=response,
        )

        return response

    def ask_with_sources(
        self,
        question: str,
        sources: List[dict],
        provider: str = DEFAULT_ASK_PROVIDER,
        retrieval_mode: str = "hybrid",
    ) -> dict:
        scoped_provider = str(provider or DEFAULT_ASK_PROVIDER).strip().lower()
        if scoped_provider not in ASK_PROVIDERS:
            raise ValueError("provider must be one of: chatgpt, claude")

        answer_language = self._infer_query_language(question)
        selected_model = (
            self.openai_model if scoped_provider == "chatgpt" else self.engine.model
        )
        retrieved_chunks = build_retrieved_chunks_payload(sources)
        citations = build_citation_catalog(sources)

        if not sources:
            return {
                "status": "insufficient_evidence",
                "answer": default_insufficient_answer(answer_language),
                "confidence": "low",
                "citations": [],
                "retrieved_chunks": [],
                "warnings": [
                    "No transcript chunks matched the question."
                    if answer_language != "ja"
                    else "質問に合う書き起こしチャンクが見つかりませんでした。"
                ],
                "sources": [],
                "provider": scoped_provider,
                "model": selected_model,
            }

        evidence = assess_grounded_answer_evidence(
            question=question,
            rows=sources,
            retrieval_mode=retrieval_mode,
            tokenize_fn=self._tokenize_for_lexical,
            answer_language=answer_language,
        )
        if not evidence["sufficient"]:
            return {
                "status": "insufficient_evidence",
                "answer": default_insufficient_answer(answer_language),
                "confidence": "low",
                "citations": [],
                "retrieved_chunks": retrieved_chunks,
                "warnings": list(evidence["warnings"]),
                "sources": sources,
                "provider": scoped_provider,
                "model": selected_model,
            }

        system_prompt, user_message = build_grounded_answer_messages(
            question=question,
            citations=citations,
            answer_language=answer_language,
        )

        try:
            llm = self._llm_text_response(
                provider=scoped_provider,
                system_prompt=system_prompt,
                user_message=user_message,
                max_tokens=ASK_MAX_TOKENS,
                temperature=ASK_TEMPERATURE,
            )
            parsed = _extract_json_payload(llm["text"])
            if not isinstance(parsed, dict):
                raise ValueError("Grounded answer response must be a JSON object.")
            normalized = normalize_grounded_answer_payload(
                raw_payload=parsed,
                citations=citations,
                answer_language=answer_language,
            )
            model_name = llm["model"]
        except Exception as exc:
            return {
                "status": "error",
                "answer": default_error_answer(answer_language),
                "confidence": "low",
                "citations": [],
                "retrieved_chunks": retrieved_chunks,
                "warnings": [str(exc)],
                "sources": sources,
                "provider": scoped_provider,
                "model": selected_model,
            }

        warnings = list(evidence["warnings"])
        warnings.extend(
            warning for warning in normalized["warnings"] if warning not in warnings
        )
        confidence = cap_confidence(
            normalized["confidence"], evidence["confidence_cap"]
        )

        # Keep `sources` for compatibility with the existing local preview UI and history.
        return {
            "status": normalized["status"],
            "answer": normalized["answer"],
            "confidence": confidence
            if confidence in ANSWER_CONFIDENCE_LEVELS
            else "low",
            "citations": normalized["citations"],
            "retrieved_chunks": retrieved_chunks,
            "warnings": warnings,
            "sources": sources,
            "provider": scoped_provider,
            "model": model_name,
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
            video_exists = video_id in self.engine.library.videos
            video_is_stale = (
                video_exists and self.engine.library.video_chunking_is_stale(video_id)
            )
            if (not force) and video_exists and (not video_is_stale):
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
            if (not force) and video_is_stale:
                self._append_ingest_log(
                    level="info",
                    event="job.reingest.stale_chunking",
                    message="video uses stale chunking; re-ingesting",
                    video_id=video_id,
                    mode=mode,
                    language=language,
                )

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
                if hasattr(self.engine.library, "save"):
                    try:
                        self.engine.library.save()
                    except Exception as save_exc:
                        self._append_ingest_log(
                            level="warning",
                            event="library.save.failed",
                            message="library save failed after ingestion",
                            job_id=job_id,
                            video_id=video_id,
                            mode=mode,
                            language=language,
                            error_message=str(save_exc)[:300],
                        )
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
        origin = _allowed_local_origin(self.headers.get("Origin"))
        if origin:
            self.send_header("Access-Control-Allow-Origin", origin)
            self.send_header("Vary", "Origin")
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

            if path == "/v1/local-video-ocr/jobs":
                jobs = [asdict(j) for j in SERVICE.list_ocr_jobs()[:100]]
                self._json({"ok": True, "jobs": jobs})
                return

            if path.startswith("/v1/local-video-ocr/jobs/"):
                job_id = unquote(path.rsplit("/", 1)[-1])
                job = SERVICE.get_ocr_job(job_id)
                if not job:
                    self._json(
                        {
                            "ok": False,
                            "error": {
                                "code": "JOB_NOT_FOUND",
                                "message": "OCR job not found",
                            },
                        },
                        404,
                    )
                    return
                self._json({"ok": True, "job": asdict(job)})
                return

            if path.startswith("/v1/local-video-ocr/videos/"):
                video_id = unquote(path.rsplit("/", 1)[-1])
                summary = SERVICE.local_video_ocr_summary(video_id)
                self._json({"ok": True, "summary": summary})
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

            if path == "/v1/history/ask":
                params = parse_qs(parsed.query or "")
                video_id = (params.get("video_id") or [None])[0]
                limit_raw = (params.get("limit") or [ASK_HISTORY_LIMIT_PER_VIDEO])[0]
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
                items = SERVICE.list_ask_history(video_id=video_id, limit=limit)
                self._json({"ok": True, "count": len(items), "items": items})
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

            if path == "/v1/evidence-curation/summary":
                summary = SERVICE.evidence_curation_summary()
                self._json({"ok": True, **summary})
                return

            if path == "/v1/evidence-curation/runs":
                params = parse_qs(parsed.query or "")
                limit_raw = (params.get("limit") or [25])[0]
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
                runs = SERVICE.list_evidence_curation_runs(limit=limit)
                self._json({"ok": True, "count": len(runs), "runs": runs})
                return

            if path == "/v1/evidence-curation/manifest":
                params = parse_qs(parsed.query or "")
                try:
                    limit = int((params.get("limit") or [500])[0])
                    offset = int((params.get("offset") or [0])[0])
                    result = SERVICE.list_evidence_manifest(
                        video_id=(params.get("video_id") or [None])[0],
                        quality_label=(params.get("quality_label") or [None])[0],
                        included=(params.get("included") or [None])[0],
                        topic=(params.get("topic") or [None])[0],
                        q=(params.get("q") or [None])[0],
                        limit=limit,
                        offset=offset,
                    )
                except (TypeError, ValueError) as exc:
                    self._json(
                        {
                            "ok": False,
                            "error": {
                                "code": "INVALID_INPUT",
                                "message": str(exc),
                            },
                        },
                        400,
                    )
                    return
                self._json({"ok": True, **result})
                return

            if path == "/v1/evidence-curation/inferences":
                params = parse_qs(parsed.query or "")
                try:
                    limit = int((params.get("limit") or [100])[0])
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
                rows = SERVICE.list_evidence_inferences(
                    evidence_id=(params.get("evidence_id") or [None])[0],
                    pipeline_run_id=(params.get("pipeline_run_id") or [None])[0],
                    limit=limit,
                )
                self._json({"ok": True, "count": len(rows), "inferences": rows})
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

            if path == "/v1/local-video-ocr/jobs":
                result = SERVICE.start_local_video_ocr_job(
                    video_id=str(body.get("video_id") or "").strip(),
                    video_path=str(body.get("video_path") or "").strip(),
                    interval_sec=int(body.get("interval_sec") or 10),
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
                if max_points not in SUMMARY_ALLOWED_POINTS:
                    raise ValueError("max_points must be 5")

                result = SERVICE.summarize_video_transcript(
                    video_id=video_id,
                    language=language,
                    provider=provider,
                    max_points=max_points,
                )
                self._json({"ok": True, **result})
                return

            if path == "/v1/chunking/preview":
                video_id = str(body.get("video_id") or "").strip()
                strategy = str(body.get("strategy") or "").strip().lower()
                if strategy not in SERVICE.CHUNKING_STRATEGIES:
                    self._json(
                        {
                            "ok": False,
                            "error": {
                                "code": "INVALID_INPUT",
                                "message": f"strategy must be one of: {', '.join(sorted(SERVICE.CHUNKING_STRATEGIES))}",
                            },
                        },
                        400,
                    )
                    return
                params = body.get("params") or {}
                result = SERVICE.preview_chunking(video_id, strategy, params)
                self._json({"ok": True, **result})
                return

            if path == "/v1/chunking/search":
                video_id = str(body.get("video_id") or "").strip()
                strategy = str(body.get("strategy") or "").strip().lower()
                if strategy not in SERVICE.CHUNKING_STRATEGIES:
                    self._json(
                        {
                            "ok": False,
                            "error": {
                                "code": "INVALID_INPUT",
                                "message": f"strategy must be one of: {', '.join(sorted(SERVICE.CHUNKING_STRATEGIES))}",
                            },
                        },
                        400,
                    )
                    return
                query = str(body.get("query") or "").strip()
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
                params = body.get("params") or {}
                k = max(1, min(int(body.get("k", 5)), 12))
                language = body.get("language")
                result = SERVICE.search_with_chunking(
                    video_id,
                    strategy,
                    params,
                    query,
                    k,
                    normalize_language(language) if language else None,
                )
                self._json({"ok": True, **result})
                return

            if path == "/v1/search-multimodal":
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
                source_mode = str(body.get("source_mode") or "both").strip().lower()
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
                result = SERVICE.search_multimodal(
                    query=query,
                    k=k,
                    language=normalize_language(language) if language else None,
                    retrieval_mode=retrieval_mode,
                    video_id=str(body.get("video_id") or "").strip() or None,
                    source_mode=source_mode,
                )
                self._json({"ok": True, **result})
                return

            if path in {"/v1/ask-multimodal", "/v1/answer-multimodal"}:
                question = (body.get("question") or body.get("query") or "").strip()
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
                k = max(1, min(int(body.get("top_k", body.get("k", 5))), 12))
                language = body.get("language")
                retrieval_mode = (
                    str(body.get("retrieval_mode") or "hybrid").strip().lower()
                )
                source_mode = str(body.get("source_mode") or "both").strip().lower()
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

                retrieval = SERVICE.search_multimodal(
                    query=question,
                    k=k,
                    language=normalize_language(language) if language else None,
                    retrieval_mode=retrieval_mode,
                    video_id=str(body.get("video_id") or "").strip() or None,
                    source_mode=source_mode,
                )
                result = SERVICE.ask_with_sources(
                    question,
                    retrieval["results"],
                    provider=provider,
                    retrieval_mode=retrieval["retrieval_mode"],
                )
                if source_mode != "transcript" and not retrieval["results"]:
                    result["answer"] = (
                        "Insufficient evidence to answer confidently from the retrieved excerpts."
                    )
                    result["warnings"] = [
                        "No matching transcript or OCR evidence was found for the question."
                    ]
                self._json(
                    {
                        "ok": True,
                        "question": question,
                        "k": k,
                        "video_id": str(body.get("video_id") or "").strip() or None,
                        "source_mode": retrieval["source_mode"],
                        "retrieval_mode": retrieval["retrieval_mode"],
                        "retrieval_details": retrieval["details"],
                        "result_count": len(retrieval["results"]),
                        **result,
                    }
                )
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

            if path in {"/v1/ask", "/v1/answer"}:
                question = (body.get("question") or body.get("query") or "").strip()
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
                k = max(1, min(int(body.get("top_k", body.get("k", 5))), 12))
                language = body.get("language")
                video_id = str(body.get("video_id") or "").strip() or None
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
                    video_id=video_id,
                )
                result = SERVICE.ask_with_sources(
                    question,
                    retrieval["results"],
                    provider=provider,
                    retrieval_mode=retrieval["retrieval_mode"],
                )
                if video_id:
                    SERVICE.save_ask_history(
                        {
                            "video_id": video_id,
                            "question": question,
                            "k": k,
                            "language": language,
                            "retrieval_mode": retrieval["retrieval_mode"],
                            "provider": result["provider"],
                            "model": result["model"],
                            "status": result["status"],
                            "confidence": result["confidence"],
                            "answer": result["answer"],
                            "citations": result["citations"],
                            "warnings": result["warnings"],
                            "retrieved_chunks": result["retrieved_chunks"],
                            "sources": result["sources"],
                            "retrieval_details": retrieval["details"],
                        }
                    )
                self._json(
                    {
                        "ok": True,
                        "question": question,
                        "k": k,
                        "video_id": video_id,
                        "retrieval_mode": retrieval["retrieval_mode"],
                        "retrieval_details": retrieval["details"],
                        "result_count": len(retrieval["results"]),
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
