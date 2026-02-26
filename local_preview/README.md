# Local Preview (No Cloudflare)

Run and evaluate YouTube Transcript RAG locally.

## Start

From `japanese_youtube_rag_v2`:

```bash
python local_preview/local_api.py
```

Open:

- `http://127.0.0.1:8000/`
- `http://127.0.0.1:8000/reviews.html`
- `http://127.0.0.1:8000/evaluation.html`

## What This Includes

- Local UI + API in one process
- Ingest videos/playlists and index transcript chunks
- Retrieval modes: `hybrid`, `dense`, `lexical`
- Search result review (`relevant` / `not_relevant`)
- Reviewed chunks analytics page
- Dedicated Evaluation workspace:
  - query sets
  - run snapshots
  - local labeling + reason codes
  - metrics (`P@K`, `Recall@K`, `MRR`, `nDCG@K`)
  - run comparison
- UI language switch: `English (US)` / `日本語`

## Safety Mode (Evaluation)

- Evaluation data is browser-local (`localStorage`) by design.
- No shared reviewer state is required for local evaluation runs.

## Notes

- Ingestion runs synchronously in local mode.
- If embedding model download is unavailable, local preview can fall back to local hashing embeddings.
- Cloudflare production stack remains in `production_cloudflare/` for later.
