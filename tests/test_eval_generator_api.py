"""HTTP route coverage for the local evaluation dataset generator."""

from __future__ import annotations

import importlib
import json
import os
import socket
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
LOCAL_PREVIEW_DIR = ROOT_DIR / "local_preview"
if str(LOCAL_PREVIEW_DIR) not in sys.path:
    sys.path.insert(0, str(LOCAL_PREVIEW_DIR))

os.environ["YT_RAG_SKIP_GLOBAL_SERVICE"] = "1"
local_api = importlib.import_module("local_api")


class FakeGenerator:
    def capabilities(self):
        return {"available": True, "authenticated": True, "version": "test"}

    def list_jobs(self):
        return [{"job_id": "job_test", "status": "completed"}]

    def get_job(self, job_id):
        return {"job_id": job_id, "status": "completed", "draft_id": "draft_test"}

    def start_job(self, video_ids):
        return {"job_id": "job_test", "status": "queued", "video_ids": video_ids}

    def list_drafts(self):
        return [{"draft_id": "draft_test", "case_count": 6, "decided_count": 0}]

    def get_draft(self, draft_id):
        return {"draft_id": draft_id, "status": "pending_review", "cases": []}

    def list_datasets(self):
        return [self.finalize("draft_test")]

    def get_dataset(self, dataset_id):
        return {
            "dataset_id": dataset_id,
            "status": "development",
            "created_at": "2026-08-08T12:00:00Z",
            "rows": [{"id": "case_1"}],
            "query_set": {"id": "qs_test", "queries": []},
        }

    @staticmethod
    def _finalize_response(dataset):
        return {
            "dataset": {
                "dataset_id": dataset["dataset_id"],
                "status": dataset["status"],
                "created_at": dataset["created_at"],
                "row_count": len(dataset["rows"]),
            },
            "query_set": dataset["query_set"],
            "export_url": (
                f"/v1/eval-generator/datasets/{dataset['dataset_id']}/export"
            ),
        }

    def save_review(self, draft_id, decisions):
        return {"draft_id": draft_id, "status": "pending_review", "decisions": decisions}

    def finalize(self, draft_id):
        return {
            "dataset": {"dataset_id": "dataset_test", "row_count": 5},
            "query_set": {"id": "qs_test", "queries": []},
            "export_url": "/v1/eval-generator/datasets/dataset_test/export",
        }

    def export_dataset(self, dataset_id):
        return f"{dataset_id}.jsonl", b'{"id":"case_1"}\n'


class FakeService:
    def __init__(self):
        self.generator = FakeGenerator()

    def _eval_generator_service(self):
        return self.generator


def _request(method, path, body=None):
    encoded = json.dumps(body).encode("utf-8") if body is not None else None
    request_lines = [f"{method} {path} HTTP/1.1", "Host: 127.0.0.1"]
    if encoded is not None:
        request_lines.extend(
            ["Content-Type: application/json", f"Content-Length: {len(encoded)}"]
        )
    raw_request = ("\r\n".join(request_lines) + "\r\n\r\n").encode("ascii")
    raw_request += encoded or b""

    client, handler_socket = socket.socketpair()
    try:
        client.sendall(raw_request)
        client.shutdown(socket.SHUT_WR)
        local_api.Handler(handler_socket, ("127.0.0.1", 50000), object())
        handler_socket.close()
        chunks = []
        while True:
            chunk = client.recv(65_536)
            if not chunk:
                break
            chunks.append(chunk)
    finally:
        handler_socket.close()
        client.close()

    header_blob, payload = b"".join(chunks).split(b"\r\n\r\n", 1)
    header_lines = header_blob.decode("iso-8859-1").split("\r\n")
    status = int(header_lines[0].split()[1])
    headers = {
        key: value.strip()
        for key, value in (line.split(":", 1) for line in header_lines[1:])
    }
    return status, headers, payload


def test_generator_http_lifecycle_routes(monkeypatch):
    monkeypatch.setattr(local_api, "SERVICE", FakeService())
    status, _, payload = _request("GET", "/v1/eval-generator/capabilities")
    assert status == 200
    assert json.loads(payload)["capabilities"]["authenticated"] is True

    status, _, payload = _request(
        "POST",
        "/v1/eval-generator/jobs",
        {"video_ids": ["demo-video"]},
    )
    assert status == 202
    assert json.loads(payload)["job"]["video_ids"] == ["demo-video"]

    status, _, payload = _request(
        "POST",
        "/v1/eval-generator/drafts/draft_test/review",
        {"decisions": [{"id": "case_1", "decision": "approved"}]},
    )
    assert status == 200
    assert json.loads(payload)["draft"]["decisions"][0]["decision"] == "approved"

    status, _, payload = _request(
        "POST",
        "/v1/eval-generator/drafts/draft_test/finalize",
        {},
    )
    assert status == 200
    assert json.loads(payload)["dataset"]["row_count"] == 5

    status, _, payload = _request("GET", "/v1/eval-generator/datasets")
    assert status == 200
    assert json.loads(payload)["datasets"][0]["query_set"]["id"] == "qs_test"

    status, _, payload = _request(
        "GET", "/v1/eval-generator/datasets/dataset_test"
    )
    assert status == 200
    assert json.loads(payload)["dataset"]["row_count"] == 1

    status, headers, payload = _request(
        "GET",
        "/v1/eval-generator/datasets/dataset_test/export",
    )
    assert status == 200
    assert headers["Content-Type"].startswith("application/x-ndjson")
    assert headers["Content-Disposition"] == (
        'attachment; filename="dataset_test.jsonl"'
    )
    assert payload == b'{"id":"case_1"}\n'
