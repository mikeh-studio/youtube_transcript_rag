# Multimodal OCR Design Note

OCR adds searchable visible text from local or permissioned video files. It is the first multimodal step because OCR output becomes timestamped text evidence, which fits the existing transcript retrieval and citation model.

## Scope Boundary

Public YouTube videos stay transcript-first. The YouTube flow uses transcripts, metadata, and timestamp links.

Frame extraction and OCR run only on local video files that the operator owns or has permission to process. This project does not add public YouTube video downloading, page scraping, or blocking-bypass logic.

## Storage Layout

Frame extraction writes:

- `data/frames/{video_id}/frame_000010.jpg`
- `data/processed/{video_id}/frames.jsonl`

OCR writes:

- `data/processed/{video_id}/frame_ocr.jsonl`

OCR embedding writes a separate index:

- `data/index/ocr/{video_id}.faiss`
- `data/index/ocr/{video_id}.jsonl`

The OCR index stays separate from the transcript index so the existing transcript library and retrieval behavior remain unchanged.

## Local Preview Integration

The local API exposes OCR through explicit local-file routes:

- `POST /v1/local-video-ocr/jobs` starts frame extraction, OCR, and embedding.
- `GET /v1/local-video-ocr/jobs` lists jobs.
- `GET /v1/local-video-ocr/jobs/{job_id}` returns job progress.
- `GET /v1/local-video-ocr/videos/{video_id}` summarizes generated artifacts.
- `POST /v1/search-multimodal` searches transcript, OCR, or merged evidence.
- `POST /v1/ask-multimodal` answers from the selected evidence mode.

The UI keeps OCR under **Local Video (Advanced)** in Ingest Gateway. Q&A Studio exposes the evidence mode in the Ask/Search workbench.

## Evidence Shape

Transcript evidence is normalized with:

- `source_type = "transcript"`
- `video_id`
- `start_sec`
- `end_sec`
- `timestamp_hhmmss`
- `text`
- `score`
- `metadata`

OCR evidence is normalized with:

- `source_type = "ocr"`
- `video_id`
- `timestamp_sec`
- `timestamp_hhmmss`
- `text`
- `score`
- `metadata.frame_path`

`retrieval/search_multimodal.py` and the local API merge normalized transcript and OCR rows, sort them by score, and preserve `source_type` so the UI can render transcript spans and OCR frames differently.

## Future Work

- Image embeddings for visually similar frames and non-text diagrams.
- Visual captioning for charts, objects, and scenes where OCR is insufficient.
- Scene-change frame extraction instead of fixed-interval sampling.
- Evaluation metrics such as `timestamp_hit@k` and `modality_hit`.
