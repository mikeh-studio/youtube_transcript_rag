"""Deterministic tests for YouTube source metadata acquisition."""

import json
from urllib.error import URLError
from urllib.parse import parse_qs, urlparse

from multilingual.youtube_metadata import (
    YouTubeMetadataClient,
    merge_youtube_sources,
    normalize_youtube_source,
)


class FakeResponse:
    def __init__(self, payload):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def read(self):
        return json.dumps(self.payload).encode("utf-8")


def test_data_api_batches_ids_and_keeps_stable_channel_id():
    calls = []

    def opener(request, timeout):
        del timeout
        parsed = urlparse(request.full_url)
        query = parse_qs(parsed.query)
        calls.append(query)
        items = [
            {
                "id": video_id,
                "snippet": {
                    "title": f"Title {video_id}",
                    "channelId": "stable-channel-id",
                    "channelTitle": "Mutable Channel Name",
                },
            }
            for video_id in query["id"][0].split(",")
        ]
        return FakeResponse({"items": items})

    client = YouTubeMetadataClient(
        api_key="not-persisted",
        opener=opener,
        clock=lambda: "2026-07-23T00:00:00+00:00",
    )
    results = client.fetch_many(f"video-{index}" for index in range(51))

    assert len(calls) == 2
    assert all(call["part"] == ["snippet"] for call in calls)
    source = results["video-0"]["source"]
    assert source["metadata_provider"] == "youtube_data_api"
    assert source["channel"] == {
        "id": "stable-channel-id",
        "name": "Mutable Channel Name",
        "url": "https://www.youtube.com/channel/stable-channel-id",
    }
    assert "not-persisted" not in json.dumps(results)


def test_no_key_uses_oembed_and_network_failure_returns_fallback():
    def opener(request, timeout):
        del timeout
        if "good-video" not in request.full_url:
            raise URLError("offline")
        return FakeResponse(
            {
                "title": "A useful title",
                "author_name": "Channel From oEmbed",
                "author_url": "https://www.youtube.com/@channel",
            }
        )

    client = YouTubeMetadataClient(
        api_key="",
        opener=opener,
        clock=lambda: "2026-07-23T00:00:00+00:00",
    )
    results = client.fetch_many(["good-video", "offline-video"])

    assert results["good-video"]["source"]["metadata_provider"] == "youtube_oembed"
    assert results["good-video"]["source"]["channel"]["name"] == "Channel From oEmbed"
    assert results["offline-video"]["title"] is None
    assert results["offline-video"]["source"]["metadata_provider"] == "fallback"


def test_failed_refresh_does_not_erase_known_channel_metadata():
    existing = normalize_youtube_source(
        "abcdefghijk",
        {
            "channel": {
                "id": "UC-stable",
                "name": "Old Display Name",
                "url": "https://www.youtube.com/channel/UC-stable",
            },
            "metadata_provider": "youtube_data_api",
            "fetched_at": "2026-07-01T00:00:00+00:00",
        },
    )
    merged = merge_youtube_sources(
        existing,
        normalize_youtube_source("abcdefghijk"),
    )

    assert merged == existing


def test_oembed_refresh_does_not_downgrade_data_api_provenance():
    existing = normalize_youtube_source(
        "abcdefghijk",
        {
            "channel": {
                "id": "UC-stable",
                "name": "Old Display Name",
            },
            "metadata_provider": "youtube_data_api",
            "fetched_at": "2026-07-01T00:00:00+00:00",
        },
    )
    incoming = normalize_youtube_source(
        "abcdefghijk",
        {
            "channel": {
                "id": None,
                "name": "Current Display Name",
                "url": "https://www.youtube.com/@current",
            },
            "metadata_provider": "youtube_oembed",
            "fetched_at": "2026-07-23T00:00:00+00:00",
        },
    )

    merged = merge_youtube_sources(existing, incoming)

    assert merged["metadata_provider"] == "youtube_data_api"
    assert merged["channel"]["id"] == "UC-stable"
    assert merged["channel"]["name"] == "Current Display Name"
    assert merged["fetched_at"] == "2026-07-01T00:00:00+00:00"
