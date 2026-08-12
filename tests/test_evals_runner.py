"""Tests for the offline benchmark runner."""

from pathlib import Path

from evals.runner import run_benchmark


ROOT_DIR = Path(__file__).resolve().parents[1]


def test_runner_writes_reports_and_selects_optimized_config(tmp_path):
    result = run_benchmark(
        ROOT_DIR / "evals" / "datasets" / "jp_core_v1.example.jsonl",
        ROOT_DIR / "evals" / "configs" / "baseline.yaml",
        tmp_path / "latest",
    )

    assert (tmp_path / "latest" / "results.json").exists()
    assert (tmp_path / "latest" / "leaderboard.md").exists()
    assert (tmp_path / "latest" / "failures.md").exists()
    assert (tmp_path / "latest" / "per_query_results.jsonl").exists()
    assert result["selected_run"]["name"] == "optimized_hybrid"
    assert result["thresholds"]["small_sample"] is True
    assert result["thresholds"]["passed"] is True
    agentic = next(run for run in result["runs"] if run["name"] == "agentic")
    assert agentic["strategy"] == "agentic"
    assert "Precision@5" in agentic["metrics"]
    assert "Precision@10" in agentic["metrics"]
    trace = agentic["per_query"][0]["retrieval_details"]["agentic_retrieval"]
    assert trace["policy"] == "deterministic_tool_policy_v1"
    assert trace["attempts"][-1]["tool"] == "read_context"
