# Changelog

Historical release notes moved out of the README so the project landing page can stay focused on the portfolio narrative.

## [Unreleased] - 2026-07-26

### Fixed
- Grounded Q&A (`/v1/ask`) no longer vetoes strong cross-lingual matches. The evidence-sufficiency gate now classifies each chunk against the language the query was tokenized in (the translated/evidence language when available) instead of the answer language, and for genuinely cross-lingual chunks it uses the normalized dense score as the overlap proxy rather than failing on zero lexical token overlap.
- Imported the missing `Optional` name in `local_preview/grounded_answer.py` (previously masked only by `from __future__ import annotations`; surfaces as `F821` and would break runtime annotation evaluation).

### Changed
- Lowered the grounded-answer sampling temperature from `0.5` to `0.2` so strict-JSON answers stay parseable and faithful to the retrieved evidence.

## Previous README What's New

### Retrieval Add-ons

- Added an opt-in **cross-encoder reranking stage** that rescores fused retrieval candidates before feedback tuning and diversity selection (`reranker: "cross_encoder"` per request or `YT_RAG_RERANKER=1` globally), with graceful pass-through when the model cannot be loaded offline.
- Added an opt-in **agentic retrieval loop** for `/v1/ask` that retries weak-evidence questions with LLM query rewrites (deterministic heuristic fallback), retrieval-mode switches, and broader top-k, returning the full attempt trace in `retrieval_details.agentic_retrieval`.
- Eval run configs now accept a `reranker` key so the reranking stage can be benchmarked against existing retrieval profiles.
- Reranker model loading is guarded against concurrent first requests, and both add-ons ship with unit and handler-level test coverage.

### Evaluation Workspace

- Added an **Evaluation page** for reproducible retrieval runs.
- Added query set management (create, save, import, export).
- Added run snapshots with local labels and reason codes.
- Added metrics: **P@K**, **Recall@K**, **MRR**, **nDCG@K**.
- Added run-to-run comparison with aggregate and per-query deltas.
- Added local-only evaluation storage with no shared label writes required.

### TLDR / Theme Quality

- TLDR generation now reads from a persisted **full transcript** artifact created at ingest time, with lazy backfill for older indexed videos.
- TLDR responses are now persisted in `data/cache/summaries/<video_id>.json` as a per-video summary cache keyed by `language + provider + model + max_points`.
- Cache entries automatically refresh when transcript content changes, preventing stale summaries from being reused.
- TLDR, Q&A, and Study generation can use selectable OpenAI, Claude, or Sakana AI providers.
- Study Studio now turns transcript sections into flashcards, topic maps, and study-quality checks.
- Study Studio topic maps now support an **Explain Topic** action for one speaker-aware, source-grounded topic expansion.
- Study Studio keeps recent generated results in session history so users can switch tabs and revisit prior runs.
- TLDR Studio now generates a fixed **5-theme** TLDR for more reliable output quality.
- Theme output is ranked by **importance to the full video**, not auto-sorted by timeline order.
- Each TLDR point now uses a paragraph-style summary for stronger contextual detail.
- TLDR prompts now try to include speaker or character-role context when that information is explicitly present in the transcript.
- Theme timestamp mapping now resolves against transcript anchors with better disambiguation and reuse penalties to reduce incorrect duplicate timestamps.
- Summary diagnostics now include source, cache, and timestamp-resolution details in `generation_details`.
- As of **March 7, 2026**, TLDR fallback handling retries with relaxed sentence validation when strict compact/map-reduce paths fail, and responses expose generation path metadata for debugging.

### Ingest & Video Management

- Ingest now includes an **Ingested Videos** management carousel with per-video **Review** and **Delete** actions.
- Ingest no longer auto-redirects to TLDR after success; it stays on Ingest and shows a success status so list management can continue.

### Chunking Lab (March 13, 2026)

- Added a dedicated **Chunking Lab** at `http://127.0.0.1:8000/chunking.html` for side-by-side chunk strategy comparison.
- Added in-memory chunk preview and search comparison routes: `POST /v1/chunking/preview` and `POST /v1/chunking/search`.
- Added support for comparing **time window**, **sentence boundary**, and **token count** chunking against the same ingested video.
- Chunking Lab can export the latest comparison into the local Evaluation workspace as ephemeral run snapshots.
- Chunk previews and search comparisons require videos with persisted `full_transcript` data; older videos may need re-ingest.

### UI / Reliability

- Header navigation is now consistent across `index`, `reviews`, and `evaluation`, including icon and label rendering under locale updates.
- Ingest hero visual was simplified by removing the boxed decorative background layer.
- Added regression coverage for long-transcript fallback behavior, cross-page header consistency, sentence-split timestamp handling, and chunking strategy parameter clamping.
