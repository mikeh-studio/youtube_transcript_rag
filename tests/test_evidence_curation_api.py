"""Tests for local evidence curation artifact APIs."""

import importlib
import json
import os
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
LOCAL_PREVIEW_DIR = ROOT_DIR / "local_preview"
if str(LOCAL_PREVIEW_DIR) not in sys.path:
    sys.path.insert(0, str(LOCAL_PREVIEW_DIR))

os.environ["YT_RAG_SKIP_GLOBAL_SERVICE"] = "1"
local_api = importlib.import_module("local_api")
LocalRAGService = local_api.LocalRAGService


def _make_service(tmp_path: Path):
    service = LocalRAGService.__new__(LocalRAGService)
    service.runtime_data_dir = tmp_path / "runtime"
    service.runtime_data_dir.mkdir(parents=True, exist_ok=True)
    return service


def _write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )


def _seed_artifacts(service):
    paths = service._evidence_artifact_paths()
    report = {
        "pipeline_run_id": "run-2",
        "dataset_id": "demo",
        "dataset_version": "v1",
        "total_records": 3,
        "eligible_records": 2,
        "excluded_records": 1,
        "eligibility_rate": 0.6667,
        "avg_quality_score": 0.72,
        "generated_at": "2026-01-02T00:00:00+00:00",
    }
    paths["quality_report"].write_text(
        json.dumps(report, ensure_ascii=False), encoding="utf-8"
    )
    _write_jsonl(
        paths["pipeline_runs"],
        [
            {
                "pipeline_run_id": "run-1",
                "dataset_id": "demo",
                "dataset_version": "v1",
                "started_at": "2026-01-01T00:00:00+00:00",
                "status": "completed",
                "input_record_count": 1,
                "eligible_record_count": 1,
                "duration_ms": 10,
            },
            {
                "pipeline_run_id": "run-2",
                "dataset_id": "demo",
                "dataset_version": "v1",
                "started_at": "2026-01-02T00:00:00+00:00",
                "status": "completed",
                "input_record_count": 3,
                "eligible_record_count": 2,
                "duration_ms": 20,
            },
        ],
    )
    _write_jsonl(
        paths["manifest"],
        [
            {
                "evidence_id": "ev-1",
                "pipeline_run_id": "run-2",
                "video_id": "vid-a",
                "video_title": "Anime Demo",
                "quality_label": "high_signal",
                "quality_score": 0.9,
                "topic_tags": ["anime"],
                "included": True,
                "text": "アニメと魔法の話",
            },
            {
                "evidence_id": "ev-2",
                "pipeline_run_id": "run-2",
                "video_id": "vid-b",
                "video_title": "Finance Demo",
                "quality_label": "medium_signal",
                "quality_score": 0.68,
                "topic_tags": ["finance"],
                "included": True,
                "text": "投資と金利の話",
            },
            {
                "evidence_id": "ev-3",
                "pipeline_run_id": "run-2",
                "video_id": "vid-b",
                "video_title": "Finance Demo",
                "quality_label": "invalid",
                "quality_score": 0.0,
                "topic_tags": ["unknown"],
                "included": False,
                "exclusion_reason": "empty_text",
                "text": "",
            },
        ],
    )
    _write_jsonl(
        paths["model_inference_results"],
        [
            {
                "evidence_id": "ev-1",
                "pipeline_run_id": "run-2",
                "model_name": "heuristic_quality_scorer",
                "score": 0.9,
                "label": "high_signal",
                "created_at": "2026-01-02T00:00:00+00:00",
            },
            {
                "evidence_id": "ev-2",
                "pipeline_run_id": "run-2",
                "model_name": "heuristic_quality_scorer",
                "score": 0.68,
                "label": "medium_signal",
                "created_at": "2026-01-02T00:00:01+00:00",
            },
        ],
    )


def test_evidence_summary_handles_missing_artifacts(tmp_path):
    service = _make_service(tmp_path)

    summary = service.evidence_curation_summary()

    assert summary["available"] is False
    assert summary["report"] == {}
    assert summary["latest_run"] is None
    assert summary["artifacts"]["manifest"]["row_count"] == 0


def test_evidence_summary_returns_report_latest_run_and_counts(tmp_path):
    service = _make_service(tmp_path)
    _seed_artifacts(service)

    summary = service.evidence_curation_summary()

    assert summary["available"] is True
    assert summary["report"]["pipeline_run_id"] == "run-2"
    assert summary["latest_run"]["pipeline_run_id"] == "run-2"
    assert summary["artifacts"]["manifest"]["row_count"] == 3
    assert summary["artifacts"]["model_inference_results"]["row_count"] == 2


def test_evidence_manifest_filters_and_paginates(tmp_path):
    service = _make_service(tmp_path)
    _seed_artifacts(service)

    assert service.list_evidence_manifest(video_id="vid-a")["total"] == 1
    assert service.list_evidence_manifest(quality_label="invalid")["rows"][0][
        "evidence_id"
    ] == "ev-3"
    assert service.list_evidence_manifest(included="excluded")["rows"][0][
        "exclusion_reason"
    ] == "empty_text"
    assert service.list_evidence_manifest(topic="finance")["total"] == 1
    assert service.list_evidence_manifest(q="魔法")["rows"][0]["evidence_id"] == "ev-1"

    page = service.list_evidence_manifest(limit=1, offset=1)
    assert page["count"] == 1
    assert page["total"] == 3
    assert page["rows"][0]["evidence_id"] == "ev-2"


def test_evidence_runs_and_inferences_filter(tmp_path):
    service = _make_service(tmp_path)
    _seed_artifacts(service)

    runs = service.list_evidence_curation_runs(limit=1)
    inferences = service.list_evidence_inferences(evidence_id="ev-2")

    assert [row["pipeline_run_id"] for row in runs] == ["run-2"]
    assert len(inferences) == 1
    assert inferences[0]["evidence_id"] == "ev-2"


def test_evidence_manifest_route_parses_filters(tmp_path):
    service = _make_service(tmp_path)
    _seed_artifacts(service)
    original_service = local_api.SERVICE
    local_api.SERVICE = service

    class FakeHandler:
        path = "/v1/evidence-curation/manifest?included=false&limit=10"
        response_status = None
        response_payload = None

        def _json(self, payload, status=200):
            self.response_status = status
            self.response_payload = payload

    try:
        handler = FakeHandler()
        local_api.Handler.do_GET(handler)
    finally:
        local_api.SERVICE = original_service

    assert handler.response_status == 200
    assert handler.response_payload["ok"] is True
    assert handler.response_payload["total"] == 1
    assert handler.response_payload["rows"][0]["evidence_id"] == "ev-3"
