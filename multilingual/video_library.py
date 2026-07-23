"""
Multi-video library with persistent storage and multilingual support.

Manages a collection of YouTube video transcripts with a unified FAISS index
for cross-video semantic search. Each video has an associated language code
that determines transcript fetching and tokenization.
"""

import json
import os
import sys
import numpy as np

# Some faiss-cpu wheels import NumPy 2's private module path even when running
# against NumPy 1.x, where the same extension lives under numpy.core.
try:
    import numpy.core as _numpy_core
    import numpy.core._multiarray_umath as _numpy_multiarray_umath

    sys.modules.setdefault("numpy._core", _numpy_core)
    sys.modules.setdefault("numpy._core._multiarray_umath", _numpy_multiarray_umath)
except Exception:
    pass

import faiss
import re
import time
from html import unescape
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse
from urllib.request import Request, urlopen
from youtube_transcript_api import YouTubeTranscriptApi

from multilingual.text_processing import TextProcessor, LANGUAGE_CONFIG
from multilingual.youtube_metadata import (
    YouTubeMetadataClient,
    merge_youtube_sources,
    normalize_youtube_source,
)

CHUNK_WINDOW_SECONDS = 60
CHUNK_OVERLAP_SECONDS = 15
CHUNKING_VERSION = "time_v2_60s_15s"
LEGACY_CHUNKING_VERSION = "time_v1_45s_15s"
LIBRARY_FORMAT_VERSION = 2
# Manifests written before embedding metadata existed were always built with
# this hardcoded model.
LEGACY_EMBEDDING_METADATA = {
    "backend": "sentence_transformers",
    "model": "intfloat/multilingual-e5-base",
    "dim": 768,
}


def _chmod_private(path: str) -> None:
    """Best-effort chmod for locally persisted transcript metadata."""
    try:
        os.chmod(path, 0o600)
    except OSError:
        pass


class VideoLibrary:
    """Manages multiple video transcripts with a unified search index."""

    def __init__(self, data_dir="data", processor=None, metadata_client=None):
        """Initialize the video library.

        Args:
            data_dir: Directory for persistent storage.
            processor: TextProcessor instance (created if not provided).
            metadata_client: Optional YouTubeMetadataClient-compatible instance.
        """
        self.data_dir = data_dir
        self.processor = processor or TextProcessor()
        self.metadata_client = metadata_client or YouTubeMetadataClient()

        # Video storage: video_id -> video metadata and chunks
        self.videos = (
            {}
        )  # {video_id: {"url": str, "title": str, "language": str, "chunks": list, "summary_cache": dict}}

        # Unified index state
        self.index = None  # FAISS index
        self.chunk_map = []  # Maps global index position -> (video_id, chunk_index)
        self.library_metadata = {"chunking": self.current_chunking_metadata()}
        # Metadata describing the embedding space that built self.index. When
        # running on the hashing fallback with a good model-built index on
        # disk, _preserve_saved_index keeps save() from clobbering it.
        self.index_embedding_metadata = None
        self._preserve_saved_index = False

        # Try to auto-load saved library
        if self._save_exists():
            self.load()

    def _save_exists(self):
        """Check if a saved library exists on disk."""
        return os.path.exists(self._manifest_path()) or os.path.exists(
            self._legacy_meta_path()
        )

    def _library_dir(self):
        return os.path.join(self.data_dir, "library")

    def _videos_dir(self):
        return os.path.join(self._library_dir(), "videos")

    def _manifest_path(self):
        return os.path.join(self._library_dir(), "library.json")

    def _index_dir(self):
        return os.path.join(self.data_dir, "index")

    def _index_path(self):
        return os.path.join(self._index_dir(), "library.faiss")

    def _legacy_meta_path(self):
        return os.path.join(self.data_dir, "library_meta.json")

    def _legacy_index_path(self):
        return os.path.join(self.data_dir, "library.faiss")

    def _video_record_path(self, video_id):
        return os.path.join(self._videos_dir(), video_id, "record.json")

    def _rebuild_chunk_map(self):
        self.chunk_map = []
        for video_id, data in self.videos.items():
            for chunk_idx, _chunk in enumerate(data.get("chunks", [])):
                self.chunk_map.append((video_id, chunk_idx))

    def add_video(self, url, language="ja"):
        """Fetch, process, and add a video to the library.

        Args:
            url: YouTube video URL or video ID.
            language: Language code (e.g. "ja", "en"). Defaults to "ja".

        Returns:
            video_id: The YouTube video ID that was added.
        """
        if language not in LANGUAGE_CONFIG:
            raise ValueError(
                f"Unsupported language '{language}'. "
                f"Available: {list(LANGUAGE_CONFIG.keys())}"
            )

        video_id = self.processor.extract_video_id(url)
        if not video_id:
            raise ValueError(f"Invalid YouTube URL or video ID: {url}")

        existing_video = self.videos.get(video_id)
        if isinstance(existing_video, dict):
            existing_video.setdefault("summary_cache", {})
            if not self.video_chunking_is_stale(video_id):
                print(f"Video {video_id} is already in the library.")
                return video_id
            print(f"Video {video_id} uses stale chunking; re-ingesting.")

        print(f"\nFetching transcript for video: {video_id} (language: {language})")
        transcript_codes = LANGUAGE_CONFIG[language]["transcript_codes"]
        transcript = self._fetch_transcript(video_id, transcript_codes)

        print(f"Found {len(transcript)} transcript segments")

        # Process transcript into lines, then chunk
        print("Processing transcript...")
        lines = self.processor.process_transcript(transcript, language)
        print(f"Processed {len(lines)} lines")

        print("Creating time-based chunks...")
        chunks = self.processor.chunk_by_time_with_overlap(
            lines,
            window=CHUNK_WINDOW_SECONDS,
            overlap=CHUNK_OVERLAP_SECONDS,
        )
        print(f"Created {len(chunks)} chunks")
        full_transcript = self._build_full_transcript(lines)

        # Fetch video title
        title = self._fetch_title(video_id)
        source = normalize_youtube_source(video_id)
        if isinstance(existing_video, dict):
            source = normalize_youtube_source(
                video_id,
                existing_video.get("source"),
            )
            if (
                title == f"Video {video_id}"
                and str(existing_video.get("title") or "").strip()
            ):
                title = existing_video["title"]

        # Store video data with language
        self.videos[video_id] = {
            "url": f"https://www.youtube.com/watch?v={video_id}",
            "title": title,
            "source": source,
            "language": language,
            "chunks": chunks,
            "full_transcript": full_transcript,
            "summary_cache": {},
            "chunking": self.current_chunking_metadata(),
        }

        # Rebuild the unified index
        self._rebuild_index()

        print(f"Added video: {title} ({video_id}) [{language}] - {len(chunks)} chunks")
        return video_id

    @staticmethod
    def _derive_chunking_version(window_seconds, overlap_seconds):
        if int(window_seconds) == 45 and int(overlap_seconds) == 15:
            return LEGACY_CHUNKING_VERSION
        if (
            int(window_seconds) == CHUNK_WINDOW_SECONDS
            and int(overlap_seconds) == CHUNK_OVERLAP_SECONDS
        ):
            return CHUNKING_VERSION
        return f"time_vcustom_{int(window_seconds)}s_{int(overlap_seconds)}s"

    @classmethod
    def current_chunking_metadata(cls):
        return {
            "version": CHUNKING_VERSION,
            "strategy": "time_window_overlap",
            "window_seconds": CHUNK_WINDOW_SECONDS,
            "overlap_seconds": CHUNK_OVERLAP_SECONDS,
        }

    @classmethod
    def legacy_chunking_metadata(cls):
        return {
            "version": LEGACY_CHUNKING_VERSION,
            "strategy": "time_window_overlap",
            "window_seconds": 45,
            "overlap_seconds": 15,
        }

    @classmethod
    def _normalize_chunking_metadata(cls, chunking, fallback=None):
        if not isinstance(chunking, dict):
            chunking = fallback if isinstance(fallback, dict) else None
        if not isinstance(chunking, dict):
            return cls.legacy_chunking_metadata()

        try:
            window_seconds = int(chunking.get("window_seconds") or CHUNK_WINDOW_SECONDS)
        except (TypeError, ValueError):
            window_seconds = CHUNK_WINDOW_SECONDS
        try:
            overlap_seconds = int(
                chunking.get("overlap_seconds") or CHUNK_OVERLAP_SECONDS
            )
        except (TypeError, ValueError):
            overlap_seconds = CHUNK_OVERLAP_SECONDS

        version = str(chunking.get("version") or "").strip()
        if not version:
            version = cls._derive_chunking_version(window_seconds, overlap_seconds)
        strategy = str(chunking.get("strategy") or "time_window_overlap").strip()
        if not strategy:
            strategy = "time_window_overlap"
        return {
            "version": version,
            "strategy": strategy,
            "window_seconds": window_seconds,
            "overlap_seconds": overlap_seconds,
        }

    def get_video_chunking_metadata(self, video_id):
        video = self.videos.get(video_id, {})
        return self._normalize_chunking_metadata(
            video.get("chunking"),
            fallback=self.library_metadata.get("chunking"),
        )

    def video_chunking_is_stale(self, video_id):
        return (
            self.get_video_chunking_metadata(video_id).get("version")
            != CHUNKING_VERSION
        )

    @staticmethod
    def _build_full_transcript(lines):
        """Build a full-transcript payload from processed transcript lines."""
        segments = []
        for line in lines:
            raw_text = str(line.get("raw_text", "")).strip()
            if not raw_text:
                continue
            start = float(line.get("start", 0.0))
            duration = float(line.get("duration", 0.0))
            end = max(start, start + duration)
            segments.append(
                {
                    "start": start,
                    "end": end,
                    "text": raw_text,
                }
            )

        full_text = "\n".join(seg["text"] for seg in segments)
        return {
            "version": 1,
            "text": full_text,
            "segments": segments,
            "segment_count": len(segments),
            "char_count": len(full_text),
        }

    def _fetch_transcript(self, video_id, transcript_codes):
        """Fetch transcript using youtube-transcript-api, with watch-page fallback."""
        ytt_api = YouTubeTranscriptApi()
        primary_error = None
        try:
            return ytt_api.fetch(video_id, languages=transcript_codes)
        except Exception as err:
            primary_error = err

        # Fallback retries help with transient 429/5xx throttling.
        fallback_error = None
        for attempt in range(1, 4):
            try:
                fallback = self._fetch_transcript_from_watch_page(
                    video_id, transcript_codes
                )
                print(
                    "youtube-transcript-api failed; used watch-page caption fallback."
                )
                return fallback
            except Exception as err:
                fallback_error = err
                if attempt < 3 and self._is_retryable_network_error(err):
                    sleep_sec = 2 * attempt
                    print(
                        f"Fallback transcript fetch retry {attempt}/2 for {video_id} "
                        f"after {sleep_sec}s ({type(err).__name__}: {err})"
                    )
                    time.sleep(sleep_sec)
                    continue
                break

        raise Exception(
            f"Failed to fetch transcript for {video_id}: {primary_error}\n"
            f"Fallback also failed: {type(fallback_error).__name__}: {fallback_error}"
        )

    def _fetch_transcript_from_watch_page(self, video_id, transcript_codes):
        """Fallback transcript extraction using ytInitialPlayerResponse + json3 captions."""
        watch_url = f"https://www.youtube.com/watch?v={video_id}"
        req = Request(watch_url, headers={"User-Agent": "Mozilla/5.0"})
        with self._open_with_retry(req, timeout=20) as response:
            html = response.read().decode("utf-8", errors="ignore")

        player_response = self._extract_player_response(html)
        tracks = (
            player_response.get("captions", {})
            .get("playerCaptionsTracklistRenderer", {})
            .get("captionTracks", [])
        )
        if not tracks:
            raise ValueError("No caption tracks found in watch-page data.")

        track = self._pick_caption_track(tracks, transcript_codes)
        if not track or "baseUrl" not in track:
            raise ValueError("No compatible caption track available.")

        caption_url = self._build_json3_url(track["baseUrl"])
        caption_req = Request(caption_url, headers={"User-Agent": "Mozilla/5.0"})
        with self._open_with_retry(caption_req, timeout=20) as response:
            payload = json.loads(response.read().decode("utf-8", errors="ignore"))

        events = payload.get("events", [])
        transcript = []
        for event in events:
            segs = event.get("segs", []) or []
            text = "".join(seg.get("utf8", "") for seg in segs).strip()
            if not text:
                continue
            transcript.append(
                {
                    "text": unescape(text),
                    "start": float(event.get("tStartMs", 0.0)) / 1000.0,
                    "duration": float(event.get("dDurationMs", 0.0)) / 1000.0,
                }
            )

        if not transcript:
            raise ValueError("Caption track fetched but transcript was empty.")
        return transcript

    @staticmethod
    def _extract_player_response(html):
        patterns = [
            r"ytInitialPlayerResponse\s*=\s*(\{.*?\})\s*;\s*var\s+meta",
            r"ytInitialPlayerResponse\s*=\s*(\{.*?\})\s*;",
        ]
        for pattern in patterns:
            match = re.search(pattern, html, re.DOTALL)
            if not match:
                continue
            try:
                return json.loads(match.group(1))
            except Exception:
                continue
        raise ValueError("Could not parse ytInitialPlayerResponse from watch page.")

    @staticmethod
    def _pick_caption_track(tracks, transcript_codes):
        """Pick best caption track from preferred language codes, fallback to first."""
        for code in transcript_codes:
            for track in tracks:
                if track.get("languageCode") == code:
                    return track
        for code in transcript_codes:
            prefix = code.split("-")[0]
            for track in tracks:
                language_code = str(track.get("languageCode", ""))
                if language_code.startswith(prefix):
                    return track
        return tracks[0] if tracks else None

    @staticmethod
    def _build_json3_url(base_url):
        parsed = urlparse(base_url)
        query = dict(parse_qsl(parsed.query, keep_blank_values=True))
        query["fmt"] = "json3"
        new_query = urlencode(query, doseq=True)
        return urlunparse(
            (
                parsed.scheme,
                parsed.netloc,
                parsed.path,
                parsed.params,
                new_query,
                parsed.fragment,
            )
        )

    @staticmethod
    def _is_retryable_network_error(err):
        if isinstance(err, HTTPError):
            return err.code in {429, 500, 502, 503, 504}
        if isinstance(err, URLError):
            return True
        msg = str(err).lower()
        return any(
            token in msg
            for token in (
                "too many requests",
                "timed out",
                "timeout",
                "temporarily",
                "503",
                "429",
            )
        )

    def _open_with_retry(self, req, timeout=20, retries=3):
        """Open URL with small backoff on retryable network errors."""
        last_error = None
        for attempt in range(1, retries + 1):
            try:
                return urlopen(req, timeout=timeout)
            except Exception as err:
                last_error = err
                if attempt >= retries or not self._is_retryable_network_error(err):
                    raise
                time.sleep(attempt * 2)
        raise last_error

    def _fetch_title(self, video_id):
        """Attempt to get a video title. Falls back to video ID."""
        return f"Video {video_id}"

    def remove_video(self, video_id):
        """Remove a video from the library.

        Args:
            video_id: The YouTube video ID to remove.
        """
        if video_id not in self.videos:
            raise KeyError(f"Video {video_id} not found in library.")

        title = self.videos[video_id]["title"]
        del self.videos[video_id]
        self._rebuild_index()
        print(f"Removed video: {title} ({video_id})")

    def list_videos(self):
        """List all videos in the library.

        Returns:
            List of dicts with video_id, title, url, language, num_chunks.
        """
        result = []
        for vid, data in self.videos.items():
            result.append(
                {
                    "video_id": vid,
                    "title": data["title"],
                    "url": data["url"],
                    "source": normalize_youtube_source(
                        vid,
                        data.get("source"),
                    ),
                    "language": data.get("language", "ja"),
                    "num_chunks": len(data["chunks"]),
                    "chunking_version": self.get_video_chunking_metadata(vid)[
                        "version"
                    ],
                    "chunking_stale": self.video_chunking_is_stale(vid),
                }
            )
        return result

    def _rebuild_index(self):
        """Rebuild the unified FAISS index from all video chunks."""
        self.chunk_map = []
        all_chunks = []

        for video_id, data in self.videos.items():
            for chunk_idx, chunk in enumerate(data["chunks"]):
                all_chunks.append(chunk)
                self.chunk_map.append((video_id, chunk_idx))

        if not all_chunks:
            self.index = None
            self.index_embedding_metadata = self.processor.embedding_metadata()
            self._preserve_saved_index = False
            return

        print(f"Generating embeddings for {len(all_chunks)} total chunks...")
        embeddings = self.processor.generate_embeddings(all_chunks)

        faiss.omp_set_num_threads(1)
        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(embeddings)
        self.index_embedding_metadata = self.processor.embedding_metadata()
        self._preserve_saved_index = False
        print(f"FAISS index rebuilt with {self.index.ntotal} vectors")

    def _dominant_language(self):
        """Return the most common language across stored videos, or None if empty."""
        if not self.videos:
            return None
        langs = [data.get("language", "ja") for data in self.videos.values()]
        return max(set(langs), key=langs.count)

    def search(
        self,
        query,
        k=5,
        language=None,
        video_id=None,
        video_ids=None,
    ):
        """Semantic search across all videos.

        Args:
            query: Search query string.
            k: Number of results to return.
            language: Language code for query tokenization. If None, uses
                the dominant language of stored videos.
            video_id: Optional video ID to restrict retrieval to one video.
            video_ids: Optional iterable of video IDs to restrict retrieval to
                a routed subset. Cannot be combined with ``video_id``.

        Returns:
            List of result dicts with score, text, video info, timestamps, and language.
        """
        if self.index is None or self.index.ntotal == 0:
            return []

        k = min(int(k), int(self.index.ntotal))
        if k <= 0:
            return []

        scoped_video_id = str(video_id or "").strip()
        if scoped_video_id and video_ids is not None:
            raise ValueError("Use either video_id or video_ids, not both.")
        scoped_video_ids = None
        if video_ids is not None:
            if isinstance(video_ids, (str, bytes)):
                raise TypeError("video_ids must be an iterable of video IDs.")
            scoped_video_ids = set()
            for value in video_ids:
                normalized_id = str(value or "").strip()
                if normalized_id:
                    scoped_video_ids.add(normalized_id)
            if not scoped_video_ids:
                return []
        elif scoped_video_id:
            scoped_video_ids = {scoped_video_id}

        if scoped_video_ids:
            missing = sorted(scoped_video_ids - set(self.videos))
            if missing:
                raise KeyError(f"Video(s) not found in library: {', '.join(missing)}")
            if not any(self.videos[value].get("chunks") for value in scoped_video_ids):
                return []

        if language is None:
            if scoped_video_ids:
                scoped_languages = [
                    self.videos[value].get("language", "ja")
                    for value in scoped_video_ids
                ]
                language = max(set(scoped_languages), key=scoped_languages.count)
            else:
                language = self._dominant_language()
        query_emb = self.processor.encode_query(query, language=language)
        search_k = int(self.index.ntotal) if scoped_video_ids else k
        scores, indices = self.index.search(query_emb, search_k)

        results = []
        for rank, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if idx < 0:
                continue
            video_id, chunk_idx = self.chunk_map[idx]
            if scoped_video_ids and video_id not in scoped_video_ids:
                continue
            video_data = self.videos[video_id]
            chunk = video_data["chunks"][chunk_idx]

            results.append(
                {
                    "rank": len(results) + 1,
                    "score": float(score),
                    "video_id": video_id,
                    "video_title": video_data["title"],
                    "video_url": video_data["url"],
                    "source": normalize_youtube_source(
                        video_id,
                        video_data.get("source"),
                    ),
                    "language": video_data.get("language", "ja"),
                    "chunk_index": chunk_idx,
                    "text": chunk["raw_text"],
                    "start": chunk["start"],
                    "end": chunk["end"],
                    "url": f"https://www.youtube.com/watch?v={video_id}&t={int(chunk['start'])}s",
                }
            )
            if len(results) >= k:
                break

        return results

    def save(self):
        """Save library to disk."""
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self._library_dir(), exist_ok=True)
        os.makedirs(self._videos_dir(), exist_ok=True)
        os.makedirs(self._index_dir(), exist_ok=True)

        # Save FAISS index
        index_path = self._index_path()
        if self.index is not None:
            faiss.write_index(self.index, index_path)
        elif os.path.exists(index_path) and not self._preserve_saved_index:
            os.remove(index_path)

        # Save library manifest
        manifest = {
            "format_version": LIBRARY_FORMAT_VERSION,
            "library_metadata": {
                "chunking": self.current_chunking_metadata(),
                "embedding": self.index_embedding_metadata
                or self.processor.embedding_metadata(),
            },
            "videos": list(self.videos.keys()),
        }

        manifest_path = self._manifest_path()
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2)
        _chmod_private(manifest_path)

        existing_video_dirs = set()
        if os.path.isdir(self._videos_dir()):
            existing_video_dirs = {
                name
                for name in os.listdir(self._videos_dir())
                if os.path.isdir(os.path.join(self._videos_dir(), name))
            }

        for vid, data in self.videos.items():
            video_dir = os.path.join(self._videos_dir(), vid)
            os.makedirs(video_dir, exist_ok=True)
            record = {
                "url": data["url"],
                "title": data["title"],
                "source": normalize_youtube_source(
                    vid,
                    data.get("source"),
                ),
                "language": data.get("language", "ja"),
                "chunks": data["chunks"],
                "full_transcript": data.get("full_transcript"),
                "chunking": self._normalize_chunking_metadata(data.get("chunking")),
            }
            record_path = self._video_record_path(vid)
            with open(record_path, "w", encoding="utf-8") as f:
                json.dump(record, f, ensure_ascii=False, indent=2)
            _chmod_private(record_path)
            existing_video_dirs.discard(vid)

        for stale_video_id in existing_video_dirs:
            record_path = self._video_record_path(stale_video_id)
            stale_video_dir = os.path.dirname(record_path)
            if os.path.exists(record_path):
                os.remove(record_path)
            if os.path.isdir(stale_video_dir):
                try:
                    os.rmdir(stale_video_dir)
                except OSError:
                    pass

        print(f"Library saved to {self.data_dir}/")

    def load(self):
        """Load library from disk."""
        manifest_path = self._manifest_path()
        legacy_meta_path = self._legacy_meta_path()

        if os.path.exists(manifest_path):
            self._load_split_layout(manifest_path)
            index_path = self._index_path()
        elif os.path.exists(legacy_meta_path):
            self._load_legacy_layout(legacy_meta_path)
            index_path = self._legacy_index_path()
        else:
            raise FileNotFoundError(
                f"No saved library found at {manifest_path} or {legacy_meta_path}"
            )

        self._rebuild_chunk_map()

        if os.path.exists(index_path):
            self.index = faiss.read_index(index_path)
            print(
                f"Loaded library: {len(self.videos)} videos, {self.index.ntotal} vectors"
            )
        else:
            self.index = None
            print(f"Loaded library: {len(self.videos)} videos (no index)")

        self._reconcile_index_embedding(
            getattr(self, "_loaded_embedding_metadata", None)
        )

    def _reconcile_index_embedding(self, saved_metadata):
        """Ensure the loaded index matches the active embedding space.

        Rebuilds the index from persisted chunks when the embedding model
        changed, and preserves a model-built on-disk index (dense search
        disabled) when running degraded on the hashing fallback.
        """
        current = self.processor.embedding_metadata()

        if self.index is None:
            self.index_embedding_metadata = None
            if self.chunk_map and current.get("backend") == "sentence_transformers":
                print("No saved index but chunks exist; rebuilding index.")
                self._rebuild_index()
                self.save()
            return

        saved = saved_metadata
        if not isinstance(saved, dict) or not saved:
            saved = dict(LEGACY_EMBEDDING_METADATA)

        matches = all(
            saved.get(key) == current.get(key) for key in ("backend", "model", "dim")
        )
        count_ok = int(self.index.ntotal) == len(self.chunk_map)
        index_dim_ok = int(self.index.d) == int(current.get("dim") or 0)

        if matches and count_ok and index_dim_ok:
            self.index_embedding_metadata = saved
            return

        if current.get("backend") == "sentence_transformers":
            if not matches:
                reason = (
                    f"embedding model changed from {saved.get('model')} "
                    f"(dim {saved.get('dim')}) to {current.get('model')} "
                    f"(dim {current.get('dim')})"
                )
            elif not index_dim_ok:
                reason = (
                    f"index dimension is {int(self.index.d)} but the active "
                    f"embedding space expects {current.get('dim')}"
                )
            else:
                reason = (
                    f"index has {int(self.index.ntotal)} vectors "
                    f"but library has {len(self.chunk_map)} chunks"
                )
            print(f"Index is stale ({reason}); rebuilding from persisted chunks.")
            self._rebuild_index()
            self.save()
            return

        if saved.get("backend") == "sentence_transformers":
            # Degraded to hashing while a model-built index exists on disk:
            # never mix embedding spaces, never clobber the good index.
            print(
                "Warning: embedding model unavailable; dense search is disabled. "
                f"The saved index (built with {saved.get('model')}) is preserved."
            )
            self.index = None
            self.index_embedding_metadata = saved
            self._preserve_saved_index = True
            return

        # Both hashing (e.g. RAG_LOCAL_EMBED_DIM changed): rebuild cheaply.
        print("Hash-embedding parameters changed; rebuilding index.")
        self._rebuild_index()
        self.save()

    def _load_split_layout(self, manifest_path):
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)

        library_metadata = manifest.get("library_metadata")
        library_chunking = None
        self._loaded_embedding_metadata = None
        if isinstance(library_metadata, dict):
            library_chunking = self._normalize_chunking_metadata(
                library_metadata.get("chunking")
            )
            self.library_metadata = {"chunking": library_chunking}
            embedding = library_metadata.get("embedding")
            if isinstance(embedding, dict) and embedding:
                self._loaded_embedding_metadata = embedding
        else:
            self.library_metadata = {"chunking": self.current_chunking_metadata()}

        raw_video_ids = manifest.get("videos", [])
        if isinstance(raw_video_ids, dict):
            raw_video_ids = list(raw_video_ids.keys())

        self.videos = {}
        for raw_video_id in raw_video_ids:
            video_id = str(raw_video_id or "").strip()
            if not video_id:
                continue
            record_path = self._video_record_path(video_id)
            if not os.path.exists(record_path):
                continue
            with open(record_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.videos[video_id] = self._normalize_video_record(
                video_id,
                data,
                fallback_chunking=library_chunking,
                include_summary_cache=False,
            )

    def _load_legacy_layout(self, meta_path):
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

        self._loaded_embedding_metadata = None
        library_metadata = meta.get("library_metadata")
        library_chunking = None
        if isinstance(library_metadata, dict):
            library_chunking = self._normalize_chunking_metadata(
                library_metadata.get("chunking")
            )
            self.library_metadata = {"chunking": library_chunking}
        else:
            self.library_metadata = {"chunking": self.current_chunking_metadata()}

        raw_videos = meta.get("videos", {})
        self.videos = {}
        for video_id, data in raw_videos.items():
            self.videos[video_id] = self._normalize_video_record(
                video_id,
                data,
                fallback_chunking=library_chunking,
                include_summary_cache=True,
            )

    def _normalize_video_record(
        self, video_id, data, *, fallback_chunking=None, include_summary_cache=False
    ):
        normalized = {
            "url": data.get("url", f"https://www.youtube.com/watch?v={video_id}"),
            "title": data.get("title", f"Video {video_id}"),
            "source": normalize_youtube_source(
                video_id,
                data.get("source"),
                legacy=not isinstance(data.get("source"), dict),
            ),
            "language": data.get("language", "ja"),
            "chunks": data.get("chunks", []),
            "summary_cache": {},
            "chunking": self._normalize_chunking_metadata(
                data.get("chunking"),
                fallback=fallback_chunking,
            ),
        }
        if include_summary_cache and isinstance(data.get("summary_cache"), dict):
            normalized["summary_cache"] = data.get("summary_cache")
        if isinstance(data.get("full_transcript"), dict):
            normalized["full_transcript"] = data.get("full_transcript")
        return normalized

    def backfill_source_metadata(self, video_ids=None, *, force=False, persist=True):
        """Enrich stored video provenance without refetching transcripts.

        Complete Data API or oEmbed records are skipped unless ``force`` is
        true, making routine backfills idempotent. Failed lookups retain all
        previously stored metadata.
        """
        if video_ids is None:
            requested = list(self.videos)
        else:
            if isinstance(video_ids, (str, bytes)):
                raise TypeError("video_ids must be an iterable of video IDs.")
            requested = list(
                dict.fromkeys(
                    normalized
                    for value in video_ids
                    for normalized in [str(value or "").strip()]
                    if normalized
                )
            )
        missing = [video_id for video_id in requested if video_id not in self.videos]
        if missing:
            raise KeyError(f"Video(s) not found in library: {', '.join(missing)}")

        candidates = []
        for video_id in requested:
            source = normalize_youtube_source(
                video_id,
                self.videos[video_id].get("source"),
            )
            provider = source.get("metadata_provider")
            if force or provider in {"fallback", "legacy"}:
                candidates.append(video_id)

        if not candidates:
            return {"requested": len(requested), "updated": [], "skipped": requested}

        try:
            fetched = self.metadata_client.fetch_many(candidates)
        except Exception:
            fetched = {}
        updated = []
        for video_id in candidates:
            data = self.videos[video_id]
            result = fetched.get(video_id)
            if not isinstance(result, dict):
                continue
            incoming_source = result.get("source")
            existing_source = normalize_youtube_source(
                video_id,
                data.get("source"),
            )
            merged_source = merge_youtube_sources(existing_source, incoming_source)
            title = str(result.get("title") or "").strip()
            changed = merged_source != existing_source
            if title and title != data.get("title"):
                data["title"] = title
                changed = True
            if changed:
                data["source"] = merged_source
                # Keep the compatibility URL aligned with canonical provenance.
                data["url"] = merged_source["url"]
                updated.append(video_id)

        if updated and persist:
            self.save()
        skipped = [video_id for video_id in requested if video_id not in updated]
        return {"requested": len(requested), "updated": updated, "skipped": skipped}

    @property
    def total_chunks(self):
        """Total number of chunks across all videos."""
        return sum(len(d["chunks"]) for d in self.videos.values())
