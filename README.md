# YouTube Transcript RAG

Local-first YouTube transcript retrieval and evaluation platform.

This project lets you ingest YouTube transcripts, run semantic retrieval (`hybrid` / `dense` / `lexical`), and evaluate ranking quality with a dedicated local Evaluation workspace. It currently supports English (EN-US) and Japanese (JP) only, with a built-in UI Language switch for EN/JP.

## Language Support

- Current support is **English (EN-US)** and **Japanese (JP)** only.
- The UI includes a **Language** selector to switch between **EN** and **JP**.

## What's New (v2 Local Eval Release)

- Added **Evaluation page** for reproducible retrieval runs
- Added query set management (create/save/import/export)
- Added run snapshots with local labels and reason codes
- Added metrics: **P@K**, **Recall@K**, **MRR**, **nDCG@K**
- Added run-to-run comparison with aggregate + per-query deltas
- Added local-only evaluation storage (no shared label writes required)

## What's New (Theme TLDR + Timestamp Quality Update)

- TLDR generation now reads from a persisted **full transcript** artifact created at ingest time (with lazy backfill for older indexed videos).
- Theme output is ranked by **importance to the full video**, not auto-sorted by timeline order.
- Each TLDR point now uses a **paragraph-style summary** (4-5 sentences) for stronger contextual detail.
- Theme timestamp mapping now resolves against transcript anchors with better disambiguation and reuse penalties to reduce incorrect duplicate timestamps.
- Summary diagnostics now include source and timestamp-resolution details in `generation_details`.

## Quick Start

From `youtube_rag_v2_portfolio`:

```bash
pip install -r requirements.txt
python local_preview/local_api.py
```

Open:

- `http://127.0.0.1:8000/` (Main Console)
- `http://127.0.0.1:8000/reviews.html` (Reviewed Chunks)
- `http://127.0.0.1:8000/evaluation.html` (Evaluation Workspace)

Note:
- If model download/network is unavailable, the app falls back to local hashing embeddings in local preview mode.

## Run In 60 Seconds

```bash
cd youtube_rag_v2_portfolio
pip install -r requirements.txt
python local_preview/local_api.py
```

Open:

- `http://127.0.0.1:8000/`
- `http://127.0.0.1:8000/reviews.html`
- `http://127.0.0.1:8000/evaluation.html`

## Demo Flow (Portfolio-Friendly)

1. Ingest one or more videos/playlists.
2. Run Search in `hybrid` mode and inspect ranked chunks.
3. Open Evaluation page and create/import a query set.
4. Execute a full run, label results, and inspect quality metrics.
5. Compare two runs to show regression/lift.
6. Export run bundle JSON for reproducibility.

## Screenshots

### Main Console

![Main Console](docs/media/01-main-console.png)

### Evaluation Metrics

![Evaluation Metrics](docs/media/02-evaluation-metrics.png)

### Run Comparison

![Run Comparison](docs/media/03-run-comparison.png)

### UI Language Toggle (EN/JP)

![Main Console (JP)](docs/media/01-main-console-jp.png)

## Architecture

```text
Browser UI (index / reviews / evaluation)
          |
          v
local_preview/local_api.py  (local HTTP API + static files)
          |
          v
multilingual/*  (retrieval pipeline: chunking, embeddings, hybrid search)
          |
          v
local files:
  - data/ (indexed library)
  - browser localStorage (evaluation query sets/runs/labels)
```

## Project Structure

- `local_preview/` - local web UI + API for development and demos
- `multilingual/` - retrieval engine, multilingual tokenization, transcript processing
- `production_cloudflare/` - Cloudflare deployment stack (separate from local preview)
- `tests/` - unit/integration tests for core behavior

## Known Limitations

- Evaluation mode is currently **single-reviewer**.
- Evaluation labels are **browser-local** (not shared across devices by default).
- Inter-rater agreement/adjudication workflow is not implemented yet.
- Retrieval quality depends on transcript availability from YouTube.

## Tests

```bash
pytest tests/
```
