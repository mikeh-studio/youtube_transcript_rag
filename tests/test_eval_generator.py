"""Tests for the local Codex evaluation dataset generator."""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

import pytest


ROOT_DIR = Path(__file__).resolve().parents[1]
LOCAL_PREVIEW_DIR = ROOT_DIR / "local_preview"
if str(LOCAL_PREVIEW_DIR) not in sys.path:
    sys.path.insert(0, str(LOCAL_PREVIEW_DIR))

from eval_generator import EvalGeneratorError, EvalGeneratorService  # noqa: E402


def _video(video_id="video000001", language="en"):
    return {
        "title": "Demo Interview",
        "url": f"https://www.youtube.com/watch?v={video_id}",
        "language": language,
        "chunking": {"version": "time_v2"},
        "chunks": [
            {
                "start": index * 60.0,
                "end": (index + 1) * 60.0,
                "raw_text": f"Transcript chunk {index} explains supported fact {index}.",
            }
            for index in range(12)
        ],
    }


def _generated_payload(video_id="video000001"):
    case_types = [
        "direct_fact_1",
        "direct_fact_2",
        "semantic_paraphrase",
        "cross_lingual",
        "distractor_resistant",
        "multi_evidence",
    ]
    return {
        "cases": [
            {
                "case_type": case_type,
                "query": f"What supported fact is discussed in case {index}?",
                "language": "ja" if case_type == "cross_lingual" else "en",
                "query_type": "thematic" if index >= 2 else "factual",
                "difficulty": "medium",
                "gold_evidence": [
                    {"video_id": video_id, "chunk_index": index},
                    *(
                        [{"video_id": video_id, "chunk_index": index + 1}]
                        if case_type == "multi_evidence"
                        else []
                    ),
                ],
                "required_facts": [f"supported fact {index}"],
                "notes": "Agent proposal",
                "confidence": "high",
                "risk_flags": [],
            }
            for index, case_type in enumerate(case_types)
        ]
    }


class FakeCodexRunner:
    def __init__(self, payload=None):
        self.payload = payload or _generated_payload()
        self.calls = []

    def __call__(self, command, **kwargs):
        self.calls.append((list(command), kwargs))
        if command[-1] == "--version":
            return subprocess.CompletedProcess(command, 0, "codex-cli 0.test\n", "")
        if command[-2:] == ["login", "status"]:
            return subprocess.CompletedProcess(command, 0, "Logged in using ChatGPT\n", "")
        return subprocess.CompletedProcess(
            command, 0, json.dumps(self.payload, ensure_ascii=False), "progress"
        )


def _service(tmp_path, runner=None, videos=None):
    videos = videos or {"video000001": _video()}
    return EvalGeneratorService(
        root_dir=ROOT_DIR,
        runtime_dir=tmp_path,
        video_getter=lambda video_id: videos.get(video_id),
        schema_path=LOCAL_PREVIEW_DIR
        / "schemas"
        / "eval_generator_output.schema.json",
        command_runner=runner or FakeCodexRunner(),
    )


def _wait_for_job(service, job_id, timeout=3):
    deadline = time.time() + timeout
    while time.time() < deadline:
        job = service.get_job(job_id)
        if job and job["status"] in {"completed", "failed"}:
            return job
        time.sleep(0.01)
    raise AssertionError("generator job did not finish")


def test_codex_job_uses_read_only_structured_ephemeral_command(monkeypatch, tmp_path):
    runner = FakeCodexRunner()
    service = _service(tmp_path, runner=runner)
    monkeypatch.setattr(service, "_codex_path", lambda: "/usr/local/bin/codex")
    monkeypatch.setenv("OPENAI_API_KEY", "must-not-leak")

    job = service.start_job(["video000001"])
    completed = _wait_for_job(service, job["job_id"])

    assert completed["status"] == "completed"
    command, kwargs = runner.calls[-1]
    assert command[:2] == ["/usr/local/bin/codex", "exec"]
    assert "--ephemeral" in command
    assert command[command.index("--sandbox") + 1] == "read-only"
    assert command[command.index("--config") + 1] == 'approval_policy="never"'
    assert "--output-schema" in command
    assert "--strict-config" in command
    assert command[-1] == "-"
    assert "OPENAI_API_KEY" not in kwargs["env"]
    assert "Transcript context:" in kwargs["input"]

    draft = service.get_draft(completed["draft_id"])
    assert len(draft["cases"]) == 6
    assert draft["cases"][0]["gold_evidence"][0]["text"].startswith(
        "Transcript chunk 0"
    )
    assert draft["source_snapshot"]["videos"][0]["chunking_version"] == "time_v2"


def test_review_finalize_and_export_preserve_human_edits(tmp_path):
    service = _service(tmp_path)
    context = service._build_context(["video000001"])
    draft = service._canonicalize_draft(
        generated=_generated_payload(),
        context=context,
        capability={"version": "codex-cli test", "model_override": None},
    )
    service._write_json(service._draft_path(draft["draft_id"]), draft)

    decisions = []
    for index, case in enumerate(draft["cases"]):
        if index == 0:
            decisions.append(
                {
                    "id": case["id"],
                    "decision": "edited",
                    "query": "What fact was explicitly supported?",
                    "required_facts": ["A corrected atomic fact"],
                    "difficulty": "hard",
                    "notes": "Human corrected",
                    "kept_evidence_ids": [case["gold_evidence"][0]["evidence_id"]],
                }
            )
        elif index == 5:
            decisions.append({"id": case["id"], "decision": "rejected"})
        else:
            decisions.append({"id": case["id"], "decision": "approved"})

    reviewed = service.save_review(draft["draft_id"], decisions)
    assert reviewed["cases"][0]["review"]["decision"] == "edited"

    result = service.finalize(draft["draft_id"])
    assert result["dataset"]["row_count"] == 5
    assert len(result["query_set"]["queries"]) == 5
    assert result["query_set"]["queries"][0]["text"] == (
        "What fact was explicitly supported?"
    )
    filename, payload = service.export_dataset(result["dataset"]["dataset_id"])
    assert filename.endswith(".jsonl")
    rows = [json.loads(line) for line in payload.decode("utf-8").splitlines()]
    assert len(rows) == 5
    assert rows[0]["required_facts"] == ["A corrected atomic fact"]
    saved = service.list_datasets()
    assert len(saved) == 1
    assert saved[0]["dataset"]["dataset_id"] == result["dataset"]["dataset_id"]
    assert saved[0]["query_set"] == result["query_set"]


def test_finalize_requires_every_case_to_be_reviewed(tmp_path):
    service = _service(tmp_path)
    context = service._build_context(["video000001"])
    draft = service._canonicalize_draft(
        generated=_generated_payload(),
        context=context,
        capability={},
    )
    service._write_json(service._draft_path(draft["draft_id"]), draft)

    with pytest.raises(EvalGeneratorError, match="Review every") as exc_info:
        service.finalize(draft["draft_id"])

    assert exc_info.value.code == "REVIEW_INCOMPLETE"


def test_review_cannot_invent_a_new_evidence_reference(tmp_path):
    service = _service(tmp_path)
    context = service._build_context(["video000001"])
    draft = service._canonicalize_draft(
        generated=_generated_payload(),
        context=context,
        capability={},
    )
    service._write_json(service._draft_path(draft["draft_id"]), draft)

    with pytest.raises(EvalGeneratorError) as exc_info:
        service.save_review(
            draft["draft_id"],
            [
                {
                    "id": draft["cases"][0]["id"],
                    "decision": "edited",
                    "kept_evidence_ids": ["invented:999"],
                }
            ],
        )

    assert exc_info.value.code == "INVALID_EVIDENCE_EDIT"


def test_context_sampling_stays_within_total_character_budget(tmp_path):
    video = _video()
    video["chunks"] = [
        {"start": i, "end": i + 1, "raw_text": "x" * 10_000}
        for i in range(40)
    ]
    service = _service(tmp_path, videos={"video000001": video})

    context = service._build_context(["video000001"])

    assert context["context_char_count"] <= 150_000
    indices = context["videos"][0]["included_chunk_indices"]
    assert indices == sorted(indices)
    assert len(indices) < len(video["chunks"])
