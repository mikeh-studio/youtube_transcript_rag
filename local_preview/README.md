# Local Preview

Run the local UI and API for transcript ingestion, retrieval testing, evidence review, and evaluation.

## Start

From the repository root:

```bash
python local_preview/local_api.py
```

Open:

- `http://127.0.0.1:8000/index.html` - Main console
- `http://127.0.0.1:8000/reviews.html` - Reviewed chunks
- `http://127.0.0.1:8000/evidence.html` - Evidence curation
- `http://127.0.0.1:8000/evaluation.html` - Evaluation workspace
- `http://127.0.0.1:8000/chunking.html` - Chunking Lab

Use this command for fully offline startup without loading the embedding model:

```bash
YT_RAG_FORCE_HASH_EMBEDDINGS=1 python local_preview/local_api.py
```

## What This Includes

- One local process for the HTTP API and static UI.
- YouTube video and playlist transcript ingestion.
- Transcript search with `hybrid`, `dense`, and `lexical` modes.
- Citation-backed Q&A with grounded evidence and fallback states.
- Search-result review with `relevant` and `not_relevant` labels.
- Reviewed-chunk analytics.
- Evidence curation reports generated from local pipeline artifacts.
- Evaluation query sets, labels, run snapshots, metrics, and run comparison.
- Chunking Lab for comparing chunking strategies on stored transcripts.
- Optional local-video OCR for files you own or have permission to process.
- UI language switch for English and Japanese.

## Runtime Data

Local preview writes generated state under `data/`:

- `data/library/` - library manifest and per-video records.
- `data/index/` - transcript FAISS indexes.
- `data/index/ocr/` - OCR FAISS indexes.
- `data/runtime/` - feedback, ask history, ingest logs, review batches, and curation artifacts.
- `data/cache/summaries/` - TLDR cache files.

Evaluation query sets, labels, and run snapshots are stored in browser `localStorage`.

## Ingestion Verification

Use the real backend for ingestion checks:

```bash
python local_preview/local_api.py
```

Then test through `http://127.0.0.1:8000/index.html` or the JSON API.

For an end-to-end ingestion check, capture:

- `POST /v1/ingest/videos` response
- `GET /v1/ingest/jobs` status, which should reach `completed`
- `GET /v1/videos` result with the ingested video and chunk count

Static hosting or mocked `/v1/*` responses are not valid ingestion verification.

## Local Video OCR

Local Video OCR is intentionally separate from the public YouTube transcript flow.

- It accepts local `.mp4`, `.m4v`, `.mov`, `.mkv`, and `.webm` files.
- It creates timestamped frame, OCR, and OCR-index artifacts under `data/`.
- Q&A Studio can search transcript evidence, OCR evidence, or both.
- It requires `ffmpeg` and `ffprobe` on `PATH`.

The main endpoints are:

- `POST /v1/local-video-ocr/jobs`
- `GET /v1/local-video-ocr/jobs`
- `GET /v1/local-video-ocr/jobs/{job_id}`
- `GET /v1/local-video-ocr/videos/{video_id}`
- `POST /v1/search-multimodal`
- `POST /v1/ask-multimodal`

## Assisted Labeling Workflow

Use `local_preview/review_agent_workflow.py` to batch live search results, render reviewer prompts, build adjudication inputs, and apply approved labels back through `/v1/feedback/search-review`.

Build a batch:

```bash
python local_preview/review_agent_workflow.py build-batch \
  --query-file local_preview/examples/review_queries.example.json \
  --include-existing-feedback
```

Render a reviewer prompt:

```bash
python local_preview/review_agent_workflow.py render-prompt \
  --kind reviewer \
  --input data/runtime/review_batches/<batch>.json \
  --shard-id shard-001
```

Build adjudication input:

```bash
python local_preview/review_agent_workflow.py build-adjudication \
  --input data/runtime/review_recommendations/<recommendations>.json
```

Apply approved recommendations:

```bash
python local_preview/review_agent_workflow.py apply \
  --input data/runtime/review_recommendations/<approved>.json
```

The checked-in example query file uses Japanese prompts aligned to the current local sample library. For other libraries, copy it and replace the queries with prompts that match your transcripts.

## Notes

- Ingestion runs synchronously in local mode.
- If embedding model download is unavailable, local preview can use hashing embeddings.
- Chunking Lab reads stored `full_transcript` data and does not mutate the saved library or index during preview/search comparison.
- The Cloudflare production stack remains separate in `production_cloudflare/`.
