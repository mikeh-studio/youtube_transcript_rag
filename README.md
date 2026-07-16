# YouTube Transcript Retrieval Evaluation Workbench

[![CI](https://github.com/mikeh-studio/youtube_transcript_rag/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/mikeh-studio/youtube_transcript_rag/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Local-first RAG workbench for YouTube transcripts that evaluates retrieval quality before the answer layer. It ingests English and Japanese transcripts, chunks them with timestamps, builds local FAISS indexes, and supports dense, lexical, and hybrid retrieval for citation-backed Q&A.

**Benchmark snapshot:** optimized hybrid improved nDCG@10 from 0.845 to 1.000
(+18.3%) versus baseline RRF on the checked-in eight-query Japanese regression
fixture; baseline RRF remains the application default. See the
[benchmark details](evals/README.md#fixture-result).

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
SAKANA_API_KEY=your_sakana_api_key_here
```

Optional Sakana overrides:

```env
SAKANA_MODEL=fugu
SAKANA_BASE_URL=https://api.sakana.ai/v1
```

If `OPENAI_MODEL` is not set, ChatGPT calls default to `gpt-5.4-mini`.

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
- Opt-in cross-encoder reranking stage for retrieval candidates.
- Opt-in agentic retrieval loop that retries weak-evidence questions with query rewrites, retrieval-mode switches, and broader top-k.
- Citation-backed Q&A with fallback states and selectable OpenAI, Claude, or Sakana AI providers.
- Study Studio for transcript-grounded flashcards, topic maps, per-topic explanations, run history, and study-quality checks.
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
  browser sessionStorage Study Studio run history
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

## Retrieval Add-ons

Both add-ons are opt-in; the default retrieval pipeline is unchanged.

Cross-encoder reranking rescores fused candidates with a multilingual
cross-encoder (default `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1`) before
feedback tuning and diversity selection. Enable it per request with
`"reranker": "cross_encoder"` on `/v1/search` or `/v1/ask`, or globally with
`YT_RAG_RERANKER=1`. Override the model with `YT_RAG_RERANKER_MODEL`. If the
model cannot be downloaded, reranking is skipped and the load error is
reported in `retrieval_details.reranker`.

The agentic retrieval loop retries `/v1/ask` retrieval when the grounded
evidence check finds the results too weak: it rewrites the query (with the
selected LLM provider, falling back to a deterministic heuristic), switches
retrieval mode, or broadens top-k, for up to three attempts. Enable it per
request with `"agentic": true` or globally with `YT_RAG_AGENTIC_RETRIEVAL=1`.
The attempt trace is returned in `retrieval_details.agentic_retrieval`. If no
attempt reaches sufficient evidence, the attempt with the most retrieved
evidence (the original query on ties) is returned and the normal
insufficient-evidence answer applies.

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
