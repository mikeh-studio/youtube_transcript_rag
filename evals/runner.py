"""Offline benchmark runner for transcript retrieval configurations."""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from statistics import median
from pathlib import Path
from typing import Any, Dict, List, Optional


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
LOCAL_PREVIEW_DIR = ROOT_DIR / "local_preview"
if str(LOCAL_PREVIEW_DIR) not in sys.path:
    sys.path.insert(0, str(LOCAL_PREVIEW_DIR))

os.environ.setdefault("YT_RAG_SKIP_GLOBAL_SERVICE", "1")

from evals.scoring import (  # noqa: E402
    METRIC_MRR_10,
    METRIC_NDCG_10,
    METRIC_RECALL_1,
    METRIC_RECALL_5,
    METRIC_RECALL_10,
    aggregate_query_scores,
    evaluate_thresholds,
    score_query_results,
    select_optimized_run,
)
from local_preview.local_api import LocalRAGService, normalize_language  # noqa: E402


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)
        fh.write("\n")


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _resolve_path(value: str, *, config_path: Optional[Path] = None) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    cwd_candidate = (Path.cwd() / path).resolve()
    if cwd_candidate.exists():
        return cwd_candidate
    if config_path is not None:
        config_candidate = (config_path.parent / path).resolve()
        if config_candidate.exists():
            return config_candidate
    return cwd_candidate


def _display_path(path: Path) -> str:
    resolved = Path(path).resolve()
    try:
        return str(resolved.relative_to(Path.cwd().resolve()))
    except ValueError:
        return str(resolved)


def load_config(path: Path) -> dict:
    """Load JSON-compatible YAML without requiring PyYAML."""
    text = path.read_text(encoding="utf-8")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    try:
        import yaml  # type: ignore
    except Exception as exc:  # pragma: no cover - depends on optional dependency
        raise ValueError(
            f"{path} is not JSON-compatible YAML and PyYAML is not installed"
        ) from exc
    payload = yaml.safe_load(text)
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a mapping")
    return payload


def load_dataset(path: Path) -> List[dict]:
    rows: List[dict] = []
    with path.open("r", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            row = json.loads(stripped)
            _validate_dataset_row(row, line_no=line_no)
            rows.append(row)
    if not rows:
        raise ValueError(f"{path} did not contain any query rows")
    return rows


def _validate_dataset_row(row: dict, *, line_no: int) -> None:
    required_fields = (
        "id",
        "language",
        "query",
        "query_type",
        "difficulty",
        "gold_evidence",
        "required_facts",
        "notes",
    )
    missing = [field for field in required_fields if field not in row]
    if missing:
        raise ValueError(f"dataset line {line_no} missing fields: {', '.join(missing)}")
    if not isinstance(row["gold_evidence"], list) or not row["gold_evidence"]:
        raise ValueError(
            f"dataset line {line_no} gold_evidence must be a non-empty list"
        )


class FixtureLibrary:
    """In-memory library shaped like VideoLibrary for LocalRAGService."""

    def __init__(self, payload: dict):
        self.videos: Dict[str, dict] = {}
        for video in payload.get("videos", []):
            video_id = str(video["video_id"])
            chunks = []
            for chunk_index, chunk in enumerate(video.get("chunks", [])):
                start = float(chunk.get("start", 0.0))
                end = float(chunk.get("end", start))
                chunks.append(
                    {
                        "start": start,
                        "end": end,
                        "raw_text": str(chunk.get("text", "")),
                        "embed_text": str(
                            chunk.get("embed_text") or chunk.get("text", "")
                        ),
                        "dense_scores": dict(chunk.get("dense_scores") or {}),
                        "semantic_terms": list(chunk.get("semantic_terms") or []),
                    }
                )
            self.videos[video_id] = {
                "title": str(video.get("title") or f"Video {video_id}"),
                "url": str(
                    video.get("url") or f"https://www.youtube.com/watch?v={video_id}"
                ),
                "language": str(video.get("language") or "ja"),
                "chunks": chunks,
            }


class FixtureEngine:
    """Deterministic dense-search adapter for the checked-in fixture corpus."""

    def __init__(self, payload: dict):
        self.library = FixtureLibrary(payload)

    def search(
        self,
        query: str,
        k: int = 5,
        language: Optional[str] = None,
        video_id: Optional[str] = None,
    ) -> List[dict]:
        scoped_video_id = str(video_id or "").strip()
        query_language = normalize_language(
            language, fallback=LocalRAGService._infer_query_language(query)
        )
        query_tokens = LocalRAGService._tokenize_for_lexical(
            query, language=query_language
        )
        rows: List[dict] = []
        for current_video_id, video_data in self.library.videos.items():
            if scoped_video_id and current_video_id != scoped_video_id:
                continue
            for chunk_index, chunk in enumerate(video_data.get("chunks", [])):
                score = self._dense_score(query, query_tokens, chunk, query_language)
                start = float(chunk.get("start", 0.0))
                rows.append(
                    {
                        "video_id": current_video_id,
                        "video_title": video_data["title"],
                        "video_url": video_data["url"],
                        "language": video_data.get("language", "ja"),
                        "chunk_index": chunk_index,
                        "text": chunk.get("raw_text", ""),
                        "start": start,
                        "end": float(chunk.get("end", start)),
                        "url": (
                            f"https://www.youtube.com/watch?v={current_video_id}"
                            f"&t={int(start)}s"
                        ),
                        "score": score,
                    }
                )

        rows.sort(
            key=lambda row: (
                float(row.get("score", 0.0)),
                -int(row.get("chunk_index", 10**9)),
                str(row.get("video_id") or ""),
            ),
            reverse=True,
        )
        limit = max(1, int(k))
        for rank, row in enumerate(rows[:limit], start=1):
            row["rank"] = rank
        return rows[:limit]

    @staticmethod
    def _dense_score(
        query: str, query_tokens: List[str], chunk: dict, language: str
    ) -> float:
        dense_scores = chunk.get("dense_scores") or {}
        if query in dense_scores:
            return float(dense_scores[query])

        semantic_text = " ".join(
            [
                str(chunk.get("embed_text") or chunk.get("raw_text") or ""),
                " ".join(str(term) for term in chunk.get("semantic_terms") or []),
            ]
        )
        doc_tokens = LocalRAGService._tokenize_for_lexical(
            semantic_text, language=language
        )
        overlap = LocalRAGService._jaccard_similarity(query_tokens, doc_tokens)
        return float(0.05 + (0.75 * overlap))


def build_fixture_service(corpus_path: Path, *, feedback_tuning: bool = False):
    service = LocalRAGService.__new__(LocalRAGService)
    service.engine = FixtureEngine(_read_json(corpus_path))
    service.feedback = {}
    service.feedback_index = {}
    service.feedback_lock = threading.Lock()
    service.feedback_tuning_enabled = bool(feedback_tuning)
    service._persist_feedback = lambda: None
    return service


def build_service(config: dict, config_path: Path):
    corpus_fixture = str(config.get("corpus_fixture") or "").strip()
    if corpus_fixture:
        corpus_path = _resolve_path(corpus_fixture, config_path=config_path)
        return build_fixture_service(
            corpus_path,
            feedback_tuning=bool(config.get("feedback_tuning", False)),
        )
    return LocalRAGService()


def _slim_result(row: dict) -> dict:
    return {
        "rank": row.get("rank"),
        "video_id": row.get("video_id"),
        "chunk_index": row.get("chunk_index"),
        "start": row.get("start"),
        "end": row.get("end"),
        "score": row.get("score"),
        "dense_score": row.get("dense_score"),
        "lexical_score": row.get("lexical_score"),
        "hybrid_score": row.get("hybrid_score"),
        "text": str(row.get("text") or "")[:240],
    }


def run_one_config(
    service,
    dataset: List[dict],
    run_config: dict,
    default_top_k: int,
    latency_repetitions: int = 1,
) -> dict:
    name = str(run_config.get("name") or run_config.get("retrieval_mode") or "run")
    mode = str(run_config.get("retrieval_mode") or "hybrid")
    profile = run_config.get("retrieval_profile")
    top_k = int(run_config.get("top_k") or default_top_k)

    query_scores: List[dict] = []
    latencies_ms: List[float] = []
    per_query: List[dict] = []
    failures: List[dict] = []

    for query_case in dataset:
        response = None
        iteration_latencies: List[float] = []
        try:
            for _idx in range(max(1, int(latency_repetitions))):
                started = time.perf_counter()
                response = service.retrieve(
                    query_case["query"],
                    k=top_k,
                    language=query_case.get("language"),
                    retrieval_mode=mode,
                    retrieval_profile=profile,
                )
                iteration_latencies.append((time.perf_counter() - started) * 1000.0)
            latency_ms = float(median(iteration_latencies))
            rows = response.get("results") or []
            scores = score_query_results(query_case, rows)
            query_scores.append(scores)
            latencies_ms.append(latency_ms)
            per_query.append(
                {
                    "run": name,
                    "query_id": query_case["id"],
                    "query": query_case["query"],
                    "latency_ms": latency_ms,
                    "scores": scores,
                    "result_count": len(rows),
                    "retrieval_details": response.get("details") or {},
                    "top_results": [_slim_result(row) for row in rows[:top_k]],
                }
            )
        except Exception as exc:  # pragma: no cover - exercised by CLI failures
            latency_ms = (
                float(median(iteration_latencies)) if iteration_latencies else 0.0
            )
            failures.append(
                {
                    "run": name,
                    "query_id": query_case.get("id"),
                    "query": query_case.get("query"),
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            latencies_ms.append(latency_ms)
            per_query.append(
                {
                    "run": name,
                    "query_id": query_case.get("id"),
                    "query": query_case.get("query"),
                    "latency_ms": latency_ms,
                    "scores": {},
                    "result_count": 0,
                    "retrieval_details": (
                        response.get("details") if isinstance(response, dict) else {}
                    ),
                    "top_results": [],
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )

    metrics = aggregate_query_scores(
        query_scores, latencies_ms, failed_query_count=len(failures)
    )
    return {
        "name": name,
        "retrieval_mode": mode,
        "retrieval_profile": profile,
        "baseline": bool(run_config.get("baseline", False)),
        "metrics": metrics,
        "per_query": per_query,
        "failures": failures,
    }


def _format_float(value: Any, digits: int = 4) -> str:
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return "-"


def render_leaderboard(result: dict) -> str:
    selected_name = ((result.get("selected_run") or {}).get("name") or "").strip()
    threshold = result.get("thresholds") or {}
    lines = [
        "# Retrieval Benchmark Leaderboard",
        "",
        f"Dataset: `{result.get('dataset')}`",
        f"Query count: {result.get('query_count')}",
        "",
    ]
    if threshold.get("small_sample"):
        lines.extend(
            [
                "Small-sample mode: relative improvement percentages are not stable "
                "for this fixture set, so pass/fail uses configured absolute checks.",
                "",
            ]
        )
    lines.extend(
        [
            "| Selected | Run | Mode | Profile | Recall@1 | Recall@5 | Recall@10 | "
            "MRR@10 | nDCG@10 | Mean ms | p95 ms | Failures |",
            "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for run in result.get("runs", []):
        metrics = run.get("metrics") or {}
        selected = "yes" if run.get("name") == selected_name else ""
        lines.append(
            "| {selected} | {name} | {mode} | {profile} | {r1} | {r5} | {r10} | {mrr} | {ndcg} | {mean_ms} | {p95_ms} | {failures} |".format(
                selected=selected,
                name=run.get("name"),
                mode=run.get("retrieval_mode"),
                profile=run.get("retrieval_profile") or "-",
                r1=_format_float(metrics.get(METRIC_RECALL_1)),
                r5=_format_float(metrics.get(METRIC_RECALL_5)),
                r10=_format_float(metrics.get(METRIC_RECALL_10)),
                mrr=_format_float(metrics.get(METRIC_MRR_10)),
                ndcg=_format_float(metrics.get(METRIC_NDCG_10)),
                mean_ms=_format_float(metrics.get("mean_latency_ms"), digits=2),
                p95_ms=_format_float(metrics.get("p95_latency_ms"), digits=2),
                failures=int(metrics.get("failed_query_count", 0)),
            )
        )

    lines.extend(["", "## Thresholds", ""])
    if threshold:
        lines.append(f"Overall pass: `{str(bool(threshold.get('passed'))).lower()}`")
        lines.append("")
        lines.append("| Metric | Mode | Candidate | Target | Passed |")
        lines.append("| --- | --- | ---: | ---: | --- |")
        for check in threshold.get("checks", []):
            target = check.get(
                "target", check.get("absolute_target", check.get("max_allowed"))
            )
            lines.append(
                "| {metric} | {mode} | {candidate} | {target} | {passed} |".format(
                    metric=check.get("metric"),
                    mode=check.get("mode"),
                    candidate=_format_float(check.get("candidate")),
                    target=_format_float(target),
                    passed="yes" if check.get("passed") else "no",
                )
            )
    else:
        lines.append("No threshold comparison was run.")
    lines.append("")
    return "\n".join(lines)


def render_failures(result: dict) -> str:
    lines = ["# Retrieval Benchmark Failures", ""]
    all_failures = []
    for run in result.get("runs", []):
        all_failures.extend(run.get("failures") or [])

    threshold_failures = [
        check
        for check in (result.get("thresholds") or {}).get("checks", [])
        if not check.get("passed")
    ]
    if not all_failures and not threshold_failures:
        lines.append("No query execution failures or threshold failures.")
        lines.append("")
        return "\n".join(lines)

    if all_failures:
        lines.append("## Query Failures")
        lines.append("")
        for failure in all_failures:
            lines.append(
                f"- `{failure.get('run')}` `{failure.get('query_id')}`: {failure.get('error')}"
            )
        lines.append("")

    if threshold_failures:
        lines.append("## Threshold Failures")
        lines.append("")
        for check in threshold_failures:
            lines.append(
                "- `{metric}` candidate={candidate} mode={mode}".format(
                    metric=check.get("metric"),
                    candidate=_format_float(check.get("candidate")),
                    mode=check.get("mode"),
                )
            )
        lines.append("")
    return "\n".join(lines)


def write_reports(result: dict, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    public_runs = []
    for run in result["runs"]:
        public_run = dict(run)
        public_run.pop("per_query", None)
        public_runs.append(public_run)
    public_result = dict(result)
    public_result["runs"] = public_runs
    _write_json(out_dir / "results.json", public_result)
    _write_text(out_dir / "leaderboard.md", render_leaderboard(public_result))
    _write_text(out_dir / "failures.md", render_failures(public_result))

    per_query_dir = out_dir / "per_query"
    per_query_dir.mkdir(parents=True, exist_ok=True)
    combined_path = out_dir / "per_query_results.jsonl"
    with combined_path.open("w", encoding="utf-8") as combined:
        for run in result["runs"]:
            run_path = per_query_dir / f"{run['name']}.jsonl"
            with run_path.open("w", encoding="utf-8") as fh:
                for row in run.get("per_query", []):
                    line = json.dumps(row, ensure_ascii=False)
                    fh.write(line + "\n")
                    combined.write(line + "\n")


def run_benchmark(dataset_path: Path, config_path: Path, out_dir: Path) -> dict:
    config = load_config(config_path)
    dataset = load_dataset(dataset_path)
    service = build_service(config, config_path)
    default_top_k = int(config.get("top_k", 10))
    run_configs = config.get("runs") or []
    if not run_configs:
        raise ValueError("config must include at least one run")
    latency_repetitions = int(config.get("latency_repetitions", 1))

    runs = [
        run_one_config(
            service,
            dataset,
            run_config,
            default_top_k,
            latency_repetitions=latency_repetitions,
        )
        for run_config in run_configs
    ]
    selected_run = select_optimized_run(runs)
    baseline_run = next((run for run in runs if run.get("baseline")), runs[0])
    thresholds = {}
    if selected_run is not None:
        thresholds = evaluate_thresholds(
            baseline_run["metrics"],
            selected_run["metrics"],
            config.get("success_criteria") or {},
            query_count=len(dataset),
        )

    result = {
        "dataset": _display_path(dataset_path),
        "config": _display_path(config_path),
        "query_count": len(dataset),
        "latency_repetitions": latency_repetitions,
        "baseline_run": {
            "name": baseline_run["name"],
            "metrics": baseline_run["metrics"],
        },
        "selected_run": (
            {
                "name": selected_run["name"],
                "retrieval_mode": selected_run["retrieval_mode"],
                "retrieval_profile": selected_run.get("retrieval_profile"),
                "metrics": selected_run["metrics"],
            }
            if selected_run is not None
            else None
        ),
        "thresholds": thresholds,
        "runs": runs,
    }
    write_reports(result, out_dir)
    return result


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, help="JSONL evaluation dataset")
    parser.add_argument("--config", required=True, help="Benchmark config file")
    parser.add_argument("--out", required=True, help="Output report directory")
    parser.add_argument(
        "--fail-on-threshold",
        action="store_true",
        help="Exit non-zero when selected run fails configured thresholds.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    result = run_benchmark(
        _resolve_path(args.dataset),
        _resolve_path(args.config),
        _resolve_path(args.out),
    )
    selected = result.get("selected_run") or {}
    thresholds = result.get("thresholds") or {}
    print(f"Wrote benchmark reports to {_resolve_path(args.out)}")
    if selected:
        print(
            "Selected optimized run: "
            f"{selected.get('name')} ({selected.get('retrieval_mode')}, "
            f"{selected.get('retrieval_profile') or '-'})"
        )
    print(f"Threshold pass: {bool(thresholds.get('passed'))}")
    if args.fail_on_threshold and thresholds and not thresholds.get("passed"):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
