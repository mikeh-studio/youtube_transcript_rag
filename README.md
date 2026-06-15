# YouTube Transcript Retrieval Evaluation Workbench

Local-first RAG workbench for YouTube transcripts that evaluates retrieval quality before the answer layer. It ingests English and Japanese transcripts, chunks them with timestamps, builds local FAISS indexes, and supports dense, lexical, and hybrid retrieval for citation-backed Q&A.

![Evaluation Metrics](docs/media/02-evaluation-metrics.png)

Evaluation rigor: retrieval runs are scored with P@K, Recall@K, MRR, and nDCG@K, with browser-local labels and run comparison snapshots.

## Quick Start

From the repository root:

```bash
pip install -r requirements.txt
cp .env.example .env.local
python local_preview/local_api.py
```

Add provider keys to `.env.local` if you want citation-backed Q&A:

```env
OPENAI_API_KEY=your_openai_api_key_here
ANTHROPIC_API_KEY=your_anthropic_api_key_here
```

Open the real local app:

- `http://127.0.0.1:8000/index.html` - Main console
- `http://127.0.0.1:8000/reviews.html` - Reviewed chunks
- `http://127.0.0.1:8000/evidence.html` - Evidence curation
- `http://127.0.0.1:8000/evaluation.html` - Evaluation workspace
- `http://127.0.0.1:8000/chunking.html` - Chunking Lab

Notes:

- `.env` and `.env.local` are gitignored. `.env.example` is the checked-in template.
- `local_preview/local_api.py` loads `.env`, then `.env.local`; `.env.local` wins for machine-specific overrides.
- If model download is unavailable, local preview can fall back to local hashing embeddings.
- For fully offline startup, run `YT_RAG_FORCE_HASH_EMBEDDINGS=1 python local_preview/local_api.py`.

## What It Solves

Most RAG demos stop after retrieving chunks. This project focuses on whether retrieval is actually good.

The local workflow lets you:

- ingest transcript evidence
- test dense, lexical, and hybrid retrieval
- label retrieved results
- compute ranking metrics
- compare runs before and after changes
- use retrieved evidence in citation-backed answers
- add OCR evidence from local files you own or have permission to process

## Core Workflow

1. Ingest a YouTube transcript.
2. Normalize and chunk the transcript with timestamp metadata.
3. Build a local FAISS index.
4. Retrieve chunks with dense, lexical, or hybrid search.
5. Label results in the Evaluation workspace.
6. Compare retrieval runs with ranking metrics.
7. Ask questions backed by retrieved evidence.
8. Optionally merge transcript evidence with OCR evidence from local video frames.

## Current Capabilities

- English and Japanese transcript workflows.
- YouTube video and playlist transcript ingestion.
- Timestamped chunking with strategy comparison in Chunking Lab.
- Local FAISS indexes for transcript and OCR evidence.
- Retrieval modes: `dense`, `lexical`, and `hybrid`.
- Citation-backed Q&A with fallback states.
- Search-result review and assisted labeling helpers.
- Browser-local evaluation query sets, labels, run snapshots, and metrics.
- Read-only evidence curation reports from local pipeline artifacts.
- Local-video OCR for `.mp4`, `.m4v`, `.mov`, `.mkv`, and `.webm` files.

Historical release notes live in [`CHANGELOG.md`](CHANGELOG.md).

## Architecture

```text
Browser UI
  index.html / reviews.html / evidence.html / evaluation.html / chunking.html
      |
      v
local_preview/local_api.py
  local HTTP API + static files
      |
      v
multilingual/
  transcript processing, chunking, embeddings, retrieval
      |
      v
local files
  data/library/          library manifest and per-video records
  data/index/            transcript FAISS indexes
  data/index/ocr/        OCR FAISS indexes
  data/frames/           extracted local-video frames
  data/processed/        frame and OCR metadata
  data/runtime/          feedback, ask history, ingest logs, curation artifacts
  data/cache/summaries/  per-video TLDR cache files
  browser localStorage   evaluation query sets, runs, and labels
```

## Project Structure

- `local_preview/` - local web UI, API, and review workflow helpers.
- `multilingual/` - transcript processing, chunking, embeddings, and retrieval.
- `evals/` - offline retrieval benchmark datasets, configs, scoring, and reports.
- `pipelines/` - local OCR, embedding, and evidence curation scripts.
- `retrieval/` - multimodal search helpers.
- `production_cloudflare/` - separate Cloudflare deployment stack.
- `tests/` - repository-level regression tests.
- `multilingual/tests/` - multilingual module tests.

## Local Video OCR Boundary

Public YouTube videos remain transcript-first. The YouTube flow uses transcripts, metadata, and timestamp links.

Full frame extraction and OCR are only for local video files that you own or have permission to process. This repo does not add public YouTube video downloading, page scraping, or blocking-bypass logic.

For implementation details, see [`docs/multimodal_ocr_design.md`](docs/multimodal_ocr_design.md).

## Evidence Curation

`pipelines/curate_evidence.py` turns already-ingested transcript chunks into local evidence artifacts under `data/runtime/`. It adds heuristic quality signals, topic tags, retrieval eligibility, run metadata, and a quality report without calling external services.

Example:

```bash
python pipelines/curate_evidence.py \
  --dataset-id demo_transcript_evidence \
  --dataset-version v1 \
  --language ja \
  --limit 200
```

Open `http://127.0.0.1:8000/evidence.html` to inspect the generated artifacts.

## Assisted Labeling

`local_preview/review_agent_workflow.py` batches live `/v1/search` results, renders reviewer prompts, builds adjudication inputs, and applies approved labels through `/v1/feedback/search-review`.

See [`local_preview/README.md`](local_preview/README.md) for the full local operator flow.

## Offline Retrieval Benchmark

Run the checked-in benchmark without network access or provider keys.

```bash
python -m evals.runner \
  --dataset evals/datasets/jp_core_v1.example.jsonl \
  --config evals/configs/baseline.yaml \
  --out evals/reports/latest
```

The runner compares dense, lexical, baseline hybrid, and optimized hybrid
retrieval, then writes a compact leaderboard plus machine-readable metrics.
The local app keeps baseline RRF as the default hybrid profile unless
`retrieval_profile` or `YT_RAG_HYBRID_PROFILE` explicitly opts into another
profile.
See [`evals/README.md`](evals/README.md) for metrics, latest fixture results,
and sample-set limitations.

## Known Limitations

- Evaluation is single-reviewer.
- Evaluation labels are browser-local by default.
- Inter-rater agreement and adjudication UI are not implemented.
- Retrieval quality depends on transcript availability from YouTube.
- Chunking Lab requires videos with stored `full_transcript`; re-ingest older videos if preview/search comparison reports missing transcript data.
- The checked-in retrieval benchmark is a small deterministic fixture, not a statistically stable corpus.

## Tests

```bash
pytest tests/
pytest multilingual/tests/
pytest tests/ multilingual/tests/ -q
```

For chunking-specific checks:

```bash
HF_HUB_OFFLINE=1 pytest multilingual/tests/test_chunking_strategies.py -q
pytest tests/test_chunking_api.py -q
```
