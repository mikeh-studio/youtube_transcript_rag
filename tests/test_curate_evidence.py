"""Tests for deterministic transcript evidence curation."""

from pathlib import Path

from pipelines.curate_evidence import (
    build_evidence_record,
    build_manifest_row,
    run_curation_pipeline,
    score_evidence_quality,
    tag_topics,
)


class FakeLibrary:
    """Minimal in-memory library matching VideoLibrary.videos shape."""

    def __init__(self):
        self.videos = {
            "vid123": {
                "title": "テスト動画",
                "url": "https://www.youtube.com/watch?v=vid123",
                "language": "ja",
                "chunks": [
                    {
                        "start": 0.0,
                        "end": 62.0,
                        "raw_text": (
                            "このアニメ作品では主人公が魔法について話しながら、"
                            "仲間との関係や物語の大事な場面を振り返っています。"
                        ),
                    },
                    {"start": 62.0, "end": 80.0, "raw_text": ""},
                    {
                        "start": None,
                        "end": None,
                        "raw_text": "これは十分な長さの日本語テキストですが時刻情報がありません。",
                    },
                ],
            }
        }


def test_heuristic_scoring_handles_empty_text():
    result = score_evidence_quality(
        text="",
        start_sec=0.0,
        end_sec=10.0,
        language="ja",
        target_language="ja",
    )

    assert result["score"] == 0.0
    assert result["label"] == "invalid"
    assert "empty_text" in result["reasons"]


def test_heuristic_scoring_ranks_useful_chunks_above_weak_chunks():
    useful = score_evidence_quality(
        text=(
            "このアニメ作品では主人公が魔法について話しながら、"
            "仲間との関係や物語の大事な場面を振り返っています。"
        ),
        start_sec=0.0,
        end_sec=62.0,
        language="ja",
        target_language="ja",
    )
    weak = score_evidence_quality(
        text="。。。。。。。。。。",
        start_sec=0.0,
        end_sec=10.0,
        language="ja",
        target_language="ja",
    )

    assert useful["score"] > weak["score"]
    assert useful["label"] in {"high_signal", "medium_signal"}
    assert weak["label"] in {"invalid", "low_signal"}


def test_topic_tagging_returns_expected_simple_tags():
    assert "anime" in tag_topics("アニメの声優と漫画作品について話す", "ja")
    assert "finance" in tag_topics("投資と金利、銀行の話題です", "ja")
    assert tag_topics("", "ja") == ["unknown"]


def test_manifest_rows_contain_required_fields():
    chunk = {
        "video_id": "vid123",
        "video_title": "テスト動画",
        "language": "ja",
        "chunk_index": 0,
        "start_sec": 0.0,
        "end_sec": 62.0,
        "text": "このアニメ作品では魔法と仲間について話しています。",
    }
    evidence = build_evidence_record(
        chunk,
        dataset_id="demo",
        dataset_version="v1",
        pipeline_run_id="run123",
        target_language="ja",
        min_quality_score=0.6,
        created_at="2026-01-01T00:00:00+00:00",
    )
    row = build_manifest_row(evidence)

    required_fields = {
        "dataset_id",
        "dataset_version",
        "pipeline_run_id",
        "evidence_id",
        "video_id",
        "source_type",
        "chunk_index",
        "start_sec",
        "end_sec",
        "text",
        "language",
        "quality_score",
        "quality_label",
        "topic_tags",
        "included",
        "inclusion_reason",
        "exclusion_reason",
        "created_at",
    }
    assert required_fields.issubset(row)
    assert row["source_type"] == "transcript"
    assert row["pipeline_run_id"] == "run123"


def test_pipeline_run_summary_counts_are_correct(tmp_path: Path):
    result = run_curation_pipeline(
        library=FakeLibrary(),
        dataset_id="demo",
        dataset_version="v1",
        data_dir=tmp_path,
        language="ja",
        min_quality_score=0.6,
        dry_run=True,
    )
    run = result["pipeline_run"]
    report = result["report"]

    assert run["input_record_count"] == 3
    assert run["output_record_count"] == 3
    assert run["eligible_record_count"] == 1
    assert run["excluded_record_count"] == 2
    assert run["failed_record_count"] == 0
    assert report["total_records"] == 3
    assert report["eligible_records"] == 1
    assert report["excluded_records"] == 2
    assert report["records_empty_text"] == 1
    assert report["records_missing_timestamp"] == 1


def test_dry_run_does_not_write_files(tmp_path: Path):
    result = run_curation_pipeline(
        library=FakeLibrary(),
        dataset_id="demo",
        dataset_version="v1",
        data_dir=tmp_path,
        language="ja",
        limit=2,
        dry_run=True,
    )

    assert result["dry_run"] is True
    assert result["manifest_rows"]
    assert not (tmp_path / "runtime").exists()
