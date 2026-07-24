# Video-routing evaluation

This package evaluates the video-selection stage that runs before transcript
chunk retrieval. It is deterministic, offline, and independent of any router
implementation.

The checked-in dataset contains synthetic YouTube records and independently
authored relevance labels. The runner sends adapters only:

```python
{
    "query_id": "weak_title_video_routing",
    "query": "How should RAG search a huge video library ...?",
    "language": "en",
    "top_k": 5,
}
```

Gold video IDs, gold chunks, category tags, and distractor annotations remain
inside the evaluator. This prevents an adapter from reading its own expected
answer.

## Adapter contract

Implement `route(request)` or provide a callable with the same shape:

```python
{
    "ranked_video_ids": ["ml-office-hours-42", "ml-vector-search-intro"],
    "ranked_chunks": [
        {"video_id": "ml-office-hours-42", "chunk_index": 0}
    ],
    "fallback_used": false,
    "latency_ms": 4.2,
    "ablation": {
        "without_channel": {
            "ranked_video_ids": ["infra-rag-at-scale", "ml-office-hours-42"]
        }
    },
}
```

Only `ranked_video_ids` is required. `ranked_chunks`, `fallback_used`,
`latency_ms`, and the channel ablation are optional. `run_adapter` supplies a
wall-clock latency when an adapter omits it. Precomputed partial predictions do
not invent missing optional observations; their report includes explicit
coverage counts and `null` metrics when no observation exists.

```python
from evals.video_routing import load_dataset
from evals.video_routing.runner import run_adapter

dataset = load_dataset("evals/datasets/video_routing_v1.json")
report = run_adapter(dataset, my_router_adapter)
```

To exercise the production router without exposing query labels:

```python
from evals.video_routing import MultiVectorRouterAdapter, load_dataset
from evals.video_routing.runner import run_adapter
from multilingual.text_processing import TextProcessor

dataset = load_dataset("evals/datasets/video_routing_v1.json")
adapter = MultiVectorRouterAdapter(dataset["videos"], TextProcessor())
report = run_adapter(dataset, adapter)
```

Only the label-free `videos` collection is passed to the production adapter.
The adapter also builds a channel-name-ablated router for the comparison
metrics.

For already captured results:

```bash
python -m evals.video_routing.runner \
  --dataset evals/datasets/video_routing_v1.json \
  --predictions /path/to/predictions.jsonl \
  --out /tmp/video-routing-report.json
```

Each JSONL prediction must include `query_id` and `ranked_video_ids`.

## Metrics

- video Recall@1, Recall@3, Recall@5, and video MRR
- final chunk Recall@5 when both gold chunks and ranked chunks exist
- fallback rate and missing-metadata fallback rate
- mean and p95 routing latency
- channel-targeted and ordinary-query video Recall@3
- same-channel and cross-channel distractor error rates
- with-channel versus without-channel ablation metrics and deltas

A distractor error means a labeled distractor ranks ahead of the first relevant
video. Missing or invalid predictions score as empty rankings for video recall,
while `status`, count fields, and errors expose incomplete coverage.
