# Multimodal OCR Design Note

## Why OCR First

OCR is the first multimodal milestone because it turns visible on-screen text into the same kind of searchable text evidence the project already understands. Slides, captions, charts, code snippets, and Japanese/English labels can be timestamped, embedded, retrieved, and cited without introducing visual-captioning models or image-embedding infrastructure.

## Local/Permissioned Mode Boundary

Public YouTube videos remain transcript-first. The existing YouTube flow uses transcripts, metadata, and timestamp links. Full frame extraction and OCR are intentionally separate and only operate on local video files that the operator owns or has permission to process. This milestone does not add YouTube page scraping, blocking bypasses, or public-video download logic.

## Storage Shape

Frame extraction writes:

- `data/frames/{video_id}/frame_000010.jpg`
- `data/processed/{video_id}/frames.jsonl`

OCR writes:

- `data/processed/{video_id}/frame_ocr.jsonl`

OCR embedding writes a separate index:

- `data/index/ocr/{video_id}.faiss`
- `data/index/ocr/{video_id}.jsonl`

The OCR index stays separate from the transcript FAISS index so the existing transcript library format and retrieval behavior remain unchanged.

## Local Preview Integration

The local preview keeps OCR behind an explicit local-file workflow:

- `POST /v1/local-video-ocr/jobs` starts extraction, OCR, and embedding for a local path.
- `GET /v1/local-video-ocr/jobs` and `GET /v1/local-video-ocr/jobs/{job_id}` expose progress.
- `GET /v1/local-video-ocr/videos/{video_id}` summarizes generated frame/OCR/index artifacts.
- `POST /v1/search-multimodal` returns transcript, OCR, or merged evidence.
- `POST /v1/ask-multimodal` builds grounded answers from the selected evidence mode.

The UI exposes this in Ingest Gateway under **Local Video (Advanced)** and in Q&A Studio as an Evidence selector inside the Ask/Search workbench. OCR-only cards show frame paths and timestamp labels instead of YouTube playback/review controls.

## Evidence Merge

Transcript rows are normalized into:

- `source_type = "transcript"`
- `video_id`
- `start_sec`
- `end_sec`
- `timestamp_hhmmss`
- `text`
- `score`
- `metadata`

OCR rows are normalized into:

- `source_type = "ocr"`
- `video_id`
- `timestamp_sec`
- `timestamp_hhmmss`
- `text`
- `score`
- `metadata.frame_path`

`retrieval/search_multimodal.py` and the local API search transcript evidence and OCR evidence, then sort the normalized rows by score into one evidence list. Grounded answer citations preserve `source_type` so transcript spans and OCR frames can be rendered differently while sharing one answer/evidence interface.

## Future Extensions

- Image embeddings for visually similar frames and non-text diagrams.
- Visual captioning for objects, scenes, and charts where OCR is insufficient.
- Scene-change frame extraction instead of fixed-interval sampling.
- Timestamp-based answer citations that can cite transcript spans and OCR frames together.
- Evaluation sets for `timestamp_hit@k` and `modality_hit`.
