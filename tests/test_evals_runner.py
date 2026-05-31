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
