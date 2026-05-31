# Retrieval Benchmarks

This package provides a reproducible offline benchmark for transcript retrieval.
It runs without YouTube access, provider keys, Hugging Face downloads, or paid API
calls by using a checked-in fixture corpus.

## Run

From the repository root:

```bash
python -m evals.runner \
  --dataset evals/datasets/jp_core_v1.example.jsonl \
  --config evals/configs/baseline.yaml \
  --out evals/reports/latest
```

Outputs:

- `evals/reports/latest/results.json` - run metrics and threshold checks.
- `evals/reports/latest/leaderboard.md` - dense, lexical, baseline hybrid, and optimized hybrid comparison.
- `evals/reports/latest/failures.md` - query execution failures and threshold failures.
- `evals/reports/latest/per_query_results.jsonl` - one row per query/run.
- `evals/reports/latest/per_query/*.jsonl` - per-run query details.

The default config runs each query 5 times per retrieval configuration and
uses the median latency to reduce timer noise on the small fixture corpus.
Generated reports are ignored by git; rerun the command to refresh them locally.

## Dataset Schema

Each JSONL row contains:

- `id`
- `language`
- `query`
- `query_type`
- `difficulty`
- `gold_evidence`
- `required_facts`
- `notes`

Gold evidence can match either:

- exact chunk identity: `{"video_id": "...", "chunk_index": 3}`
- timestamp overlap: `{"video_id": "...", "start": 120, "end": 180}`

## Metrics

- `gold_recall@K`: share of gold evidence found in the top K results.
- `MRR@10`: reciprocal rank of the first gold hit in the top 10.
- `nDCG@10`: rank-sensitive score for gold hits in the top 10.
- `mean_latency_ms`: average retrieval latency per query.
- `p95_latency_ms`: 95th percentile retrieval latency.
- `failed_query_count`: query executions that raised errors.

## Retrieval Change

The app still exposes the same retrieval modes: `dense`, `lexical`, and
`hybrid`. The default hybrid path now uses `optimized_v1`, a weighted fusion
profile that combines normalized dense and lexical scores with a small dual-
signal bonus. The old RRF behavior is still available as `baseline_rrf` for
benchmark comparisons.

The implementation is intentionally narrow:

- no ingestion changes
- no UI route changes
- no paid API calls
- no network dependency in the benchmark
- no replacement of the existing dense, lexical, feedback, or diversity stages

## Fixture Result

The latest local fixture run selected `optimized_hybrid`.

| Run | Recall@5 | MRR@10 | nDCG@10 | p95 ms |
| --- | ---: | ---: | ---: | ---: |
| `hybrid_baseline` | 1.0000 | 0.7917 | 0.8452 | 0.83 |
| `optimized_hybrid` | 1.0000 | 1.0000 | 1.0000 | 0.84 |

The fixture has only 8 queries, so the runner reports small-sample mode and
uses absolute pass/fail checks instead of claiming stable relative percentages.
Use a larger project-specific dataset before making production-quality
percentage claims.
