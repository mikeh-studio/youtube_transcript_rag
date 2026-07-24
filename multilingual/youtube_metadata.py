"""Best-effort YouTube source metadata acquisition and normalization."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Callable, Dict, Iterable, List, Optional
from urllib.parse import urlencode
from urllib.request import Request, urlopen


YOUTUBE_DATA_API_URL = "https://www.googleapis.com/youtube/v3/videos"
YOUTUBE_OEMBED_URL = "https://www.youtube.com/oembed"
VALID_METADATA_PROVIDERS = {
    "youtube_data_api",
    "youtube_oembed",
    "fallback",
    "legacy",
}
METADATA_PROVIDER_PRIORITY = {
    "legacy": 0,
    "fallback": 0,
    "youtube_oembed": 1,
    "youtube_data_api": 2,
}


def canonical_youtube_url(video_id: str) -> str:
    """Return the stable watch URL for a YouTube video ID."""
    return f"https://www.youtube.com/watch?v={str(video_id).strip()}"


def canonical_channel_url(channel_id: Optional[str]) -> Optional[str]:
    """Return a stable channel URL when a channel ID is known."""
    value = str(channel_id or "").strip()
    return f"https://www.youtube.com/channel/{value}" if value else None


def utc_now_iso() -> str:
    """Return an ISO-8601 UTC timestamp."""
    return datetime.now(timezone.utc).isoformat()


def normalize_youtube_source(
    video_id: str,
    source: Optional[dict] = None,
    *,
    legacy: bool = False,
) -> dict:
    """Normalize a source payload without dropping known channel fields."""
    video_id = str(video_id or "").strip()
    raw = source if isinstance(source, dict) else {}
    raw_channel = raw.get("channel")
    channel = raw_channel if isinstance(raw_channel, dict) else {}
    channel_id = str(channel.get("id") or "").strip() or None
    channel_name = str(channel.get("name") or "").strip() or None
    channel_url = str(channel.get("url") or "").strip() or None
    if channel_id and not channel_url:
        channel_url = canonical_channel_url(channel_id)

    provider = str(raw.get("metadata_provider") or "").strip()
    if provider not in VALID_METADATA_PROVIDERS:
        provider = "legacy" if legacy else "fallback"

    fetched_at = str(raw.get("fetched_at") or "").strip() or None
    # A source record always uses the stable watch URL even if a legacy
    # compatibility field contains a shortened or timestamped URL.
    source_url = canonical_youtube_url(video_id)

    return {
        "platform": "youtube",
        "video_id": video_id,
        "url": source_url,
        "channel": {
            "id": channel_id,
            "name": channel_name,
            "url": channel_url,
        },
        "metadata_provider": provider,
        "fetched_at": fetched_at,
    }


def merge_youtube_sources(
    existing: Optional[dict], incoming: Optional[dict]
) -> dict:
    """Merge metadata while ensuring a failed refresh cannot erase known values."""
    current_video_id = ""
    if isinstance(incoming, dict):
        current_video_id = str(incoming.get("video_id") or "").strip()
    if not current_video_id and isinstance(existing, dict):
        current_video_id = str(existing.get("video_id") or "").strip()

    current = normalize_youtube_source(current_video_id, existing)
    update = normalize_youtube_source(current_video_id, incoming)
    current_channel = current["channel"]
    update_channel = update["channel"]

    update_is_useful = (
        update["metadata_provider"] not in {"fallback", "legacy"}
        or bool(update_channel["id"] or update_channel["name"])
    )
    update_is_preferred = (
        update_is_useful
        and METADATA_PROVIDER_PRIORITY[update["metadata_provider"]]
        >= METADATA_PROVIDER_PRIORITY[current["metadata_provider"]]
    )
    provider = (
        update["metadata_provider"]
        if update_is_preferred
        else current["metadata_provider"]
    )
    fetched_at = update["fetched_at"] if update_is_preferred else current["fetched_at"]
    return {
        "platform": "youtube",
        "video_id": current_video_id,
        "url": update.get("url") or current["url"],
        "channel": {
            "id": update_channel["id"] or current_channel["id"],
            "name": update_channel["name"] or current_channel["name"],
            "url": update_channel["url"] or current_channel["url"],
        },
        "metadata_provider": provider,
        "fetched_at": fetched_at,
    }


class YouTubeMetadataClient:
    """Fetch YouTube metadata through the Data API or the no-key oEmbed API."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        *,
        opener: Optional[Callable] = None,
        timeout: float = 10.0,
        clock: Callable[[], str] = utc_now_iso,
    ):
        self.api_key = str(
            api_key if api_key is not None else os.environ.get("YOUTUBE_API_KEY", "")
        ).strip()
        self.opener = opener or urlopen
        self.timeout = float(timeout)
        self.clock = clock

    def fetch_many(self, video_ids: Iterable[str]) -> Dict[str, dict]:
        """Fetch normalized metadata for unique IDs, falling back per video."""
        unique_ids = list(
            dict.fromkeys(
                str(video_id or "").strip()
                for video_id in video_ids
                if str(video_id or "").strip()
            )
        )
        results: Dict[str, dict] = {}
        if self.api_key:
            try:
                results.update(self._fetch_data_api(unique_ids))
            except Exception:
                # Metadata is optional; fall through to oEmbed for every miss.
                pass

        for video_id in unique_ids:
            if video_id in results:
                continue
            try:
                results[video_id] = self._fetch_oembed(video_id)
            except Exception:
                results[video_id] = {
                    "title": None,
                    "source": normalize_youtube_source(video_id),
                }
        return results

    def _read_json(self, url: str) -> dict:
        request = Request(url, headers={"User-Agent": "youtube-rag/1.0"})
        with self.opener(request, timeout=self.timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
        return payload if isinstance(payload, dict) else {}

    def _fetch_data_api(self, video_ids: List[str]) -> Dict[str, dict]:
        results: Dict[str, dict] = {}
        for offset in range(0, len(video_ids), 50):
            batch = video_ids[offset : offset + 50]
            query = urlencode(
                {
                    "part": "snippet",
                    "id": ",".join(batch),
                    "key": self.api_key,
                }
            )
            payload = self._read_json(f"{YOUTUBE_DATA_API_URL}?{query}")
            for item in payload.get("items", []):
                if not isinstance(item, dict):
                    continue
                video_id = str(item.get("id") or "").strip()
                snippet = item.get("snippet")
                if not video_id or not isinstance(snippet, dict):
                    continue
                channel_id = str(snippet.get("channelId") or "").strip() or None
                results[video_id] = {
                    "title": str(snippet.get("title") or "").strip() or None,
                    "source": normalize_youtube_source(
                        video_id,
                        {
                            "channel": {
                                "id": channel_id,
                                "name": str(
                                    snippet.get("channelTitle") or ""
                                ).strip()
                                or None,
                                "url": canonical_channel_url(channel_id),
                            },
                            "metadata_provider": "youtube_data_api",
                            "fetched_at": self.clock(),
                        },
                    ),
                }
        return results

    def _fetch_oembed(self, video_id: str) -> dict:
        query = urlencode(
            {
                "url": canonical_youtube_url(video_id),
                "format": "json",
            }
        )
        payload = self._read_json(f"{YOUTUBE_OEMBED_URL}?{query}")
        return {
            "title": str(payload.get("title") or "").strip() or None,
            "source": normalize_youtube_source(
                video_id,
                {
                    "channel": {
                        "id": None,
                        "name": str(payload.get("author_name") or "").strip() or None,
                        "url": str(payload.get("author_url") or "").strip() or None,
                    },
                    "metadata_provider": "youtube_oembed",
                    "fetched_at": self.clock(),
                },
            ),
        }
