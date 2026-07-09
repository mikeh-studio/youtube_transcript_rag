# TODOS

## Retrieval & Answering

### Answer-layer evals

**What:** Extend the offline benchmark to score grounded answers, not just retrieval ranking: citation precision (does the cited chunk support the claim?), faithfulness, and abstention correctness (does `insufficient_evidence` fire when it should?).

**Why:** The evidence-sufficiency thresholds in `local_preview/grounded_answer.py` (0.55/0.72/0.78) are hand-tuned with no measurement behind them, and the agentic retrieval loop's value can't be quantified without answer-level metrics.

**Context:** Retrieval evals live in `evals/` (P@K, Recall, MRR, nDCG over the fixture corpus). Study-quality eval runs under `data/runtime/study_eval_runs/` show the report pattern to follow. Start by adding an answer-eval dataset schema (question, gold citations, expect-abstain flag) and an LLM-judge or rule-based scorer in `evals/scoring.py`.

**Effort:** M
**Priority:** P1
**Depends on:** None

### Close the assisted-labeling loop with an LLM judge

**What:** Add an LLM judge to `local_preview/review_agent_workflow.py` that produces label + confidence + reason recommendations automatically, keeping the human as approver via the existing `/v1/feedback/search-review` apply step.

**Why:** Today a human does all the judging; auto-drafted labels turn the feedback-tuning system into a flywheel — more labels, better reranking, measurable eval deltas.

**Context:** The workflow already batches live `/v1/search` results and renders reviewer prompts; the missing piece is calling a provider (reuse `LocalRAGService._llm_text_response`) instead of rendering prompts for manual use, then writing recommendation files in the existing format under `data/runtime/review_recommendations/`.

**Effort:** M
**Priority:** P2
**Depends on:** None

### Contextual chunk enrichment

**What:** Prepend a short LLM-generated context line ("This section of [video] discusses X") to each chunk before embedding, as an A/B chunking strategy in Chunking Lab.

**Why:** Contextual retrieval measurably improves both dense and BM25 recall on conversational transcripts, where 60-second windows often lack the topic subject.

**Context:** Chunking strategies live in `multilingual/text_processing.py` and are compared in the Chunking Lab (`/v1/chunking/preview`, `/v1/chunking/search`). The offline benchmark in `evals/` can prove the gain before any default changes, same as the `optimized_v1` hybrid profile.

**Effort:** M
**Priority:** P2
**Depends on:** None

### Benchmark the cross-encoder reranker on a real corpus

**What:** Run `evals.runner` with `reranker: "cross_encoder"` run configs against a real ingested library (not the deterministic fixture) and record whether it should become a default.

**Why:** The reranking stage shipped opt-in with fixture-level tests only; the checked-in fixture uses hand-authored dense scores, so it can't prove the reranker generalizes.

**Context:** Eval configs accept a `reranker` key per run (see `evals/README.md`). Needs the model download (`cross-encoder/mmarco-mMiniLMv2-L12-H384-v1`, ~470MB) and a labeled query set over real videos.

**Effort:** S
**Priority:** P2
**Depends on:** None

## Study Studio UI

### Fix stale study run display and silent history write failures

**What:** In `local_preview/react_shell/src/App.jsx`, `activeStudyRun` falls back to the newest run matching only `studyMode` and ignores `selectedVideoId`, so switching videos can show the previous video's flashcards/topics under the new selection; `writeStudyHistory` also swallows sessionStorage quota errors while React state updates, so history silently diverges and vanishes on refresh.

**Why:** Users can act on study content attributed to the wrong video, and lost history is invisible until a refresh.

**Context:** Found by adversarial review during the retrieval add-ons ship (2026-07-08). Fix needs the React shell rebuilt (`local_preview/react_shell` → `local_preview/web/assets/index.js`), which is why it was deferred rather than patched in that PR. Include `videoId` in the fallback match or render the run's video title in the panel, and surface storage failures.

**Effort:** S
**Priority:** P2
**Depends on:** None

## Infrastructure

### Split local_api.py into service modules

**What:** Extract `local_preview/local_api.py` (~7,800 lines) into modules: retrieval, answering, study, summaries, feedback, OCR jobs, HTTP routing.

**Why:** Every feature lands in one file; test surface and merge conflicts grow with each PR (this branch resolved 4 conflicts in it). Future agent-loop work gets much cheaper to build and test after the split.

**Context:** `LocalRAGService` mixes all domains; handlers live in one `Handler` class. Existing tests import `local_api` directly, so keep a compatibility import surface during the split. `RAGEngine.ask()` in `multilingual/rag_engine.py` also duplicates a simpler `ask_with_sources()` and can be retired or delegated.

**Effort:** L
**Priority:** P2
**Depends on:** None

## Completed
