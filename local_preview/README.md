# Local Preview (No Cloudflare)

Run the local retrieval evaluation workbench for YouTube transcripts.

## Start

From the repository root:

```bash
python local_preview/local_api.py
```

For fully offline startup without loading the embedding model:

```bash
YT_RAG_FORCE_HASH_EMBEDDINGS=1 python local_preview/local_api.py
```

Open:

- `http://127.0.0.1:8000/`
- `http://127.0.0.1:8000/reviews.html`
- `http://127.0.0.1:8000/evidence.html`
- `http://127.0.0.1:8000/evaluation.html`
- `http://127.0.0.1:8000/chunking.html`

## What This Includes

- Local UI + API in one process
- Ingest videos/playlists and index transcript chunks
- Local/permissioned video OCR jobs for timestamped frame evidence
- Retrieval modes: `hybrid`, `dense`, `lexical`
- Citation-backed Q&A with grounded evidence and fallback warnings
- Search result review (`relevant` / `not_relevant`)
- Assisted labeling workflow helpers for batching search results and applying approved labels
- Reviewed chunks analytics page
- Read-only Evidence Curation workspace for pipeline quality signals and curated transcript evidence
- Chunking Lab for chunking strategy preview/search comparison
- Dedicated Evaluation workspace:
  - query sets
  - run snapshots
  - local labeling + reason codes
  - metrics (`P@K`, `Recall@K`, `MRR`, `nDCG@K`)
  - run comparison
- UI language switch: `English (US)` / `日本語`

## Update (May 7, 2026)

- Primary navigation now keeps the core flow visible as Ingest, TLDR Studio, and Q&A Studio, with Reviews, Evidence, Evaluation, and Chunking grouped under Tools.
- Ingest Gateway now centers the YouTube URL workflow, keeps Local Video OCR collapsed under **Local Video (Advanced)**, and shows thumbnails in the ingested-video carousel.
- Backend connection and non-JSON response failures now show friendly retryable messages with collapsible debug details.
- Q&A Studio now defaults to a tabbed **Ask | Search** workbench, with Ask presented as the primary grounded-answer flow.
- Empty YouTube player panels are hidden until a timestamp is selected.

## Update (March 7, 2026)

- TLDR summary generation now has a relaxed fallback path for long transcripts when strict compact/map-reduce validation fails.
- TLDR API responses now include strategy/fallback metadata to make summary generation diagnostics clearer.
- Header menu rendering is now aligned across main shell, Reviews, and Evaluation pages (icon + label parity and locale-safe labels).
- Ingest hero styling was simplified by removing the decorative boxed background.

## Update (March 13, 2026)

- Added `chunking.html`, a local Chunking Lab for chunking strategy comparison across `time`, `sentence`, and `token` strategies against the same video.
- Added `POST /v1/chunking/preview` for chunk previews and `POST /v1/chunking/search` for search comparison over ephemeral re-chunked content.
- Chunking Lab can export the latest comparison into the browser-local Evaluation workspace as run snapshots.
- Chunk comparison requires ingested videos with persisted `full_transcript` data; re-ingest older videos if the lab reports that transcript data is missing.

## Update (April 1, 2026)

- Citation-backed Q&A now shows grounded answer status, trust copy, and supporting evidence with EN/JP-safe answer-panel strings.
- Fallback evidence cards keep source links visible in insufficient-evidence and provider-error states.
- Local preview includes `review_agent_workflow.py` for building assisted labeling batches from live search results and posting approved labels back into `/v1/feedback/search-review`.

## Update (April 28, 2026)

- Ingest Gateway includes a Local Video OCR panel for local or permissioned `.mp4`, `.m4v`, `.mov`, `.mkv`, and `.webm` files.
- Q&A Studio can search transcript evidence, OCR evidence, or a merged transcript + OCR evidence list.
- New local endpoints: `POST /v1/local-video-ocr/jobs`, `GET /v1/local-video-ocr/jobs`, `GET /v1/local-video-ocr/videos/{video_id}`, `POST /v1/search-multimodal`, and `POST /v1/ask-multimodal`.

## Update (April 30, 2026)

- Added a read-only Evidence Curation workspace for transcript quality signals, eligibility decisions, topic tags, and pipeline run metadata generated under `data/runtime/`.

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

## Assisted Labeling Workflow

Use `local_preview/review_agent_workflow.py` to batch live `/v1/search` results for assisted review and to apply approved labels back through `/v1/feedback/search-review`.

Example:

```bash
python local_preview/review_agent_workflow.py build-batch \
  --query-file local_preview/examples/review_queries.example.json \
  --include-existing-feedback
```

The checked-in example query file is tuned for the currently ingested `葬送のフリーレン` radio-style transcript library and uses Japanese queries. If your local library is different, copy the file and swap in queries that match your own transcripts.

Render a reviewer prompt for one shard:

```bash
python local_preview/review_agent_workflow.py render-prompt \
  --kind reviewer \
  --input data/runtime/review_batches/<batch>.json \
  --shard-id shard-001
```

Group reviewer outputs into consensus and disagreement buckets:

```bash
python local_preview/review_agent_workflow.py build-adjudication \
  --input data/runtime/review_recommendations/<recommendations>.json
```

Apply only approved recommendations:

```bash
python local_preview/review_agent_workflow.py apply \
  --input data/runtime/review_recommendations/<approved>.json
```
