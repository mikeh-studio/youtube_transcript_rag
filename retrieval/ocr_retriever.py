"""Retriever for local-video OCR FAISS indexes."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

import faiss

from multilingual.text_processing import TextProcessor
from pipelines.embed_ocr import infer_language
from pipelines.video_ocr_common import (
    DEFAULT_DATA_DIR,
    format_timestamp,
    ocr_index_dir,
    ocr_index_embed_meta_path,
    ocr_index_metadata_path,
    ocr_index_path,
    read_jsonl,
    validate_video_id,
)


class OCREvidenceRetriever:
    """Search OCR text extracted from local, permissioned video frames."""

    def __init__(self, data_dir: Path | str = DEFAULT_DATA_DIR, processor=None):
        self.data_dir = Path(data_dir)
        self.processor = processor or TextProcessor()
        self._cache: Dict[str, tuple] = {}

    def _available_video_ids(self) -> List[str]:
        root = ocr_index_dir(self.data_dir)
        if not root.exists():
            return []
        return sorted(path.stem for path in root.glob("*.jsonl"))

    def _load_video_index(self, video_id: str):
        scoped_video_id = validate_video_id(video_id)
        if scoped_video_id in self._cache:
            return self._cache[scoped_video_id]

        index_path = ocr_index_path(self.data_dir, scoped_video_id)
        metadata_path = ocr_index_metadata_path(self.data_dir, scoped_video_id)
        if not index_path.exists() or not metadata_path.exists():
            self._cache[scoped_video_id] = (None, [])
            return self._cache[scoped_video_id]

        records = list(read_jsonl(metadata_path))
        index = faiss.read_index(str(index_path))

        # Never search across mismatched embedding spaces: skip indexes built
        # with a different model than the active processor.
        current = self.processor.embedding_metadata()
        embed_meta_path = ocr_index_embed_meta_path(index_path)
        if embed_meta_path.exists():
            try:
                saved = json.loads(embed_meta_path.read_text(encoding="utf-8"))
            except Exception:
                saved = {}
            if any(
                saved.get(key) != current.get(key)
                for key in ("backend", "model", "dim")
            ):
                print(
                    f"Warning: OCR index for {scoped_video_id} was built with "
                    f"{saved.get('model')} (dim {saved.get('dim')}) but the active "
                    f"model is {current.get('model')} (dim {current.get('dim')}). "
                    f"Skipping; rebuild with: python pipelines/embed_ocr.py "
                    f"--video-id {scoped_video_id}"
                )
                self._cache[scoped_video_id] = (None, [])
                return self._cache[scoped_video_id]
        elif int(index.d) != int(current.get("dim") or 0):
            # Legacy index without a sidecar: dimension mismatch is the only
            # detectable signal (an equal-dim model change cannot be detected).
            print(
                f"Warning: OCR index for {scoped_video_id} has dim {index.d} but "
                f"the active model expects {current.get('dim')}. Skipping; rebuild "
                f"with: python pipelines/embed_ocr.py --video-id {scoped_video_id}"
            )
            self._cache[scoped_video_id] = (None, [])
            return self._cache[scoped_video_id]

        self._cache[scoped_video_id] = (index, records)
        return self._cache[scoped_video_id]

    def search(
        self,
        query: str,
        *,
        video_id: Optional[str] = None,
        top_k: int = 5,
        language: Optional[str] = None,
    ) -> List[dict]:
        """Search OCR text by natural-language query."""
        scoped_query = str(query or "").strip()
        if not scoped_query:
            return []
        limit = max(1, int(top_k))
        video_ids = [validate_video_id(video_id)] if video_id else self._available_video_ids()
        if not video_ids:
            return []

        query_language = language or infer_language(scoped_query)
        query_emb = self.processor.encode_query(scoped_query, language=query_language)
        rows: List[dict] = []

        for current_video_id in video_ids:
            index, records = self._load_video_index(current_video_id)
            if index is None or not records or index.ntotal == 0:
                continue
            search_k = min(limit, int(index.ntotal))
            scores, indices = index.search(query_emb, search_k)
            for score, idx in zip(scores[0], indices[0]):
                if idx < 0 or idx >= len(records):
                    continue
                record = records[idx]
                timestamp_sec = float(record.get("timestamp_sec", 0.0))
                rows.append(
                    {
                        "video_id": record.get("video_id"),
                        "timestamp_sec": timestamp_sec,
                        "timestamp_hhmmss": record.get("timestamp_hhmmss")
                        or format_timestamp(timestamp_sec),
                        "ocr_text": record.get("text", ""),
                        "frame_path": record.get("frame_path", ""),
                        "score": float(score),
                        "source_type": "ocr",
                        "frame_id": record.get("frame_id"),
                        "id": record.get("id"),
                        "metadata": {
                            "ocr_confidence": record.get("ocr_confidence"),
                            "ocr_engine": record.get("ocr_engine"),
                        },
                    }
                )

        rows.sort(key=lambda row: float(row.get("score", 0.0)), reverse=True)
        for rank, row in enumerate(rows[:limit], start=1):
            row["rank"] = rank
        return rows[:limit]
