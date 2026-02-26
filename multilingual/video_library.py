"""
Multi-video library with persistent storage and multilingual support.

Manages a collection of YouTube video transcripts with a unified FAISS index
for cross-video semantic search. Each video has an associated language code
that determines transcript fetching and tokenization.
"""

import os
import json
import numpy as np
import faiss
import re
import time
from html import unescape
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse
from urllib.request import Request, urlopen
from youtube_transcript_api import YouTubeTranscriptApi

from multilingual.text_processing import TextProcessor, LANGUAGE_CONFIG


class VideoLibrary:
    """Manages multiple video transcripts with a unified search index."""

    def __init__(self, data_dir="data", processor=None):
        """Initialize the video library.

        Args:
            data_dir: Directory for persistent storage.
            processor: TextProcessor instance (created if not provided).
        """
        self.data_dir = data_dir
        self.processor = processor or TextProcessor()

        # Video storage: video_id -> video metadata and chunks
        self.videos = {}  # {video_id: {"url": str, "title": str, "language": str, "chunks": list}}

        # Unified index state
        self.index = None  # FAISS index
        self.chunk_map = []  # Maps global index position -> (video_id, chunk_index)

        # Try to auto-load saved library
        if self._save_exists():
            self.load()

    def _save_exists(self):
        """Check if a saved library exists on disk."""
        meta_path = os.path.join(self.data_dir, "library_meta.json")
        index_path = os.path.join(self.data_dir, "library.faiss")
        return os.path.exists(meta_path) and os.path.exists(index_path)

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

        if video_id in self.videos:
            print(f"Video {video_id} is already in the library.")
            return video_id

        print(f"\nFetching transcript for video: {video_id} (language: {language})")
        transcript_codes = LANGUAGE_CONFIG[language]["transcript_codes"]
        transcript = self._fetch_transcript(video_id, transcript_codes)

        print(f"Found {len(transcript)} transcript segments")

        # Process transcript into lines, then chunk
        print("Processing transcript...")
        lines = self.processor.process_transcript(transcript, language)
        print(f"Processed {len(lines)} lines")

        print("Creating time-based chunks...")
        chunks = self.processor.chunk_by_time_with_overlap(lines, window=45, overlap=15)
        print(f"Created {len(chunks)} chunks")

        # Fetch video title
        title = self._fetch_title(video_id)

        # Store video data with language
        self.videos[video_id] = {
            "url": f"https://www.youtube.com/watch?v={video_id}",
            "title": title,
            "language": language,
            "chunks": chunks,
        }

        # Rebuild the unified index
        self._rebuild_index()

        print(f"Added video: {title} ({video_id}) [{language}] - {len(chunks)} chunks")
        return video_id

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
                fallback = self._fetch_transcript_from_watch_page(video_id, transcript_codes)
                print("youtube-transcript-api failed; used watch-page caption fallback.")
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
            transcript.append({
                "text": unescape(text),
                "start": float(event.get("tStartMs", 0.0)) / 1000.0,
                "duration": float(event.get("dDurationMs", 0.0)) / 1000.0,
            })

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
        return urlunparse((parsed.scheme, parsed.netloc, parsed.path, parsed.params, new_query, parsed.fragment))

    @staticmethod
    def _is_retryable_network_error(err):
        if isinstance(err, HTTPError):
            return err.code in {429, 500, 502, 503, 504}
        if isinstance(err, URLError):
            return True
        msg = str(err).lower()
        return any(token in msg for token in ("too many requests", "timed out", "timeout", "temporarily", "503", "429"))

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
            result.append({
                "video_id": vid,
                "title": data["title"],
                "url": data["url"],
                "language": data.get("language", "ja"),
                "num_chunks": len(data["chunks"]),
            })
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
            return

        print(f"Generating embeddings for {len(all_chunks)} total chunks...")
        embeddings = self.processor.generate_embeddings(all_chunks)

        faiss.omp_set_num_threads(1)
        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(embeddings)
        print(f"FAISS index rebuilt with {self.index.ntotal} vectors")

    def _dominant_language(self):
        """Return the most common language across stored videos, or None if empty."""
        if not self.videos:
            return None
        langs = [data.get("language", "ja") for data in self.videos.values()]
        return max(set(langs), key=langs.count)

    def search(self, query, k=5, language=None):
        """Semantic search across all videos.

        Args:
            query: Search query string.
            k: Number of results to return.
            language: Language code for query tokenization. If None, uses
                the dominant language of stored videos.

        Returns:
            List of result dicts with score, text, video info, timestamps, and language.
        """
        if self.index is None or self.index.ntotal == 0:
            return []

        k = min(int(k), int(self.index.ntotal))
        if k <= 0:
            return []

        if language is None:
            language = self._dominant_language()
        query_emb = self.processor.encode_query(query, language=language)
        scores, indices = self.index.search(query_emb, k)

        results = []
        for rank, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if idx < 0:
                continue
            video_id, chunk_idx = self.chunk_map[idx]
            video_data = self.videos[video_id]
            chunk = video_data["chunks"][chunk_idx]

            results.append({
                "rank": rank + 1,
                "score": float(score),
                "video_id": video_id,
                "video_title": video_data["title"],
                "video_url": video_data["url"],
                "language": video_data.get("language", "ja"),
                "chunk_index": chunk_idx,
                "text": chunk["raw_text"],
                "start": chunk["start"],
                "end": chunk["end"],
                "url": f"https://www.youtube.com/watch?v={video_id}&t={int(chunk['start'])}s",
            })

        return results

    def save(self):
        """Save library to disk."""
        os.makedirs(self.data_dir, exist_ok=True)

        # Save FAISS index
        index_path = os.path.join(self.data_dir, "library.faiss")
        if self.index is not None:
            faiss.write_index(self.index, index_path)
        elif os.path.exists(index_path):
            os.remove(index_path)

        # Save metadata (videos + chunk_map) as JSON
        meta = {
            "videos": {},
            "chunk_map": self.chunk_map,
        }
        for vid, data in self.videos.items():
            meta["videos"][vid] = {
                "url": data["url"],
                "title": data["title"],
                "language": data.get("language", "ja"),
                "chunks": data["chunks"],
            }

        meta_path = os.path.join(self.data_dir, "library_meta.json")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

        print(f"Library saved to {self.data_dir}/")

    def load(self):
        """Load library from disk."""
        meta_path = os.path.join(self.data_dir, "library_meta.json")
        index_path = os.path.join(self.data_dir, "library.faiss")

        if not os.path.exists(meta_path):
            raise FileNotFoundError(f"No saved library found at {meta_path}")

        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

        self.videos = meta["videos"]
        self.chunk_map = [tuple(x) for x in meta["chunk_map"]]

        if os.path.exists(index_path):
            self.index = faiss.read_index(index_path)
            print(f"Loaded library: {len(self.videos)} videos, {self.index.ntotal} vectors")
        else:
            self.index = None
            print(f"Loaded library: {len(self.videos)} videos (no index)")

    @property
    def total_chunks(self):
        """Total number of chunks across all videos."""
        return sum(len(d["chunks"]) for d in self.videos.values())
