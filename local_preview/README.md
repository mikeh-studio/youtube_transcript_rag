# Local Preview (No Cloudflare)

Run and evaluate YouTube Transcript RAG locally.

## Start

From `youtube_rag_v2_portfolio`:

```bash
python local_preview/local_api.py
```

Open:

- `http://127.0.0.1:8000/`
- `http://127.0.0.1:8000/reviews.html`
- `http://127.0.0.1:8000/evaluation.html`
- `http://127.0.0.1:8000/chunking.html`

## What This Includes

- Local UI + API in one process
- Ingest videos/playlists and index transcript chunks
- Retrieval modes: `hybrid`, `dense`, `lexical`
- Search result review (`relevant` / `not_relevant`)
- Reviewed chunks analytics page
- Chunking Lab for side-by-side strategy preview/search comparison
- Dedicated Evaluation workspace:
  - query sets
  - run snapshots
  - local labeling + reason codes
  - metrics (`P@K`, `Recall@K`, `MRR`, `nDCG@K`)
  - run comparison
- UI language switch: `English (US)` / `日本語`

## Update (March 7, 2026)

- TLDR summary generation now has a relaxed fallback path for long transcripts when strict compact/map-reduce validation fails.
- TLDR API responses now include strategy/fallback metadata to make summary generation diagnostics clearer.
- Header menu rendering is now aligned across main shell, Reviews, and Evaluation pages (icon + label parity and locale-safe labels).
- Ingest hero styling was simplified by removing the decorative boxed background.

## Update (March 13, 2026)

- Added `chunking.html`, a local Chunking Lab for comparing `time`, `sentence`, and `token` strategies against the same video.
- Added `POST /v1/chunking/preview` for chunk previews and `POST /v1/chunking/search` for search comparison over ephemeral re-chunked content.
- Chunking Lab can export the latest comparison into the browser-local Evaluation workspace as run snapshots.
- Chunk comparison requires ingested videos with persisted `full_transcript` data; re-ingest older videos if the lab reports that transcript data is missing.

## Safety Mode (Evaluation)

- Evaluation data is browser-local (`localStorage`) by design.
- No shared reviewer state is required for local evaluation runs.

## Notes

- Ingestion runs synchronously in local mode.
- If embedding model download is unavailable, local preview can fall back to local hashing embeddings.
- Local preview writes runtime state to `data/runtime/` and TLDR cache files to `data/cache/summaries/`.
- Library persistence is split across `data/library/` (manifest + per-video records) and `data/index/` (FAISS index).
- Chunking Lab uses the stored `full_transcript` artifact from ingest and does not mutate the saved library/index during preview/search comparison.
- Cloudflare production stack remains in `production_cloudflare/` for later.
