"""Helpers for citation-backed grounded answers in the local preview API."""

from __future__ import annotations

import copy
import re
from typing import Callable, Dict, List, Tuple


ANSWER_STATUSES = {"answered", "insufficient_evidence", "error"}
ANSWER_CONFIDENCE_LEVELS = {"high", "medium", "low"}
INLINE_CITATION_RE = re.compile(r"\[(\d+)\]")
SNIPPET_MAX_CHARS = 240
CONFIDENCE_ORDER = {"low": 0, "medium": 1, "high": 2}


def format_timestamp_label(seconds: float) -> str:
    """Return a compact timestamp label for a YouTube offset."""
    total_seconds = max(0, int(float(seconds or 0.0)))
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    remaining = total_seconds % 60
    if hours > 0:
        return f"{hours}:{minutes:02d}:{remaining:02d}"
    return f"{minutes}:{remaining:02d}"


def format_timestamp_range_label(start_seconds: float, end_seconds: float) -> str:
    """Return a compact range label for transcript evidence."""
    return (
        f"{format_timestamp_label(start_seconds)}"
        f"-{format_timestamp_label(max(float(end_seconds or 0.0), float(start_seconds or 0.0)))}"
    )


def _canonical_video_url(video_id: str) -> str:
    return f"https://www.youtube.com/watch?v={video_id}"


def _truncate_snippet(text: str, max_chars: int = SNIPPET_MAX_CHARS) -> str:
    clean = re.sub(r"\s+", " ", str(text or "")).strip()
    if len(clean) <= max_chars:
        return clean
    return clean[: max_chars - 1].rstrip() + "…"


def _source_payload(row: dict, video_id: str, video_url: str) -> dict | None:
    """Return stable provenance without conflating it with evidence source_type."""
    raw = row.get("source")
    source_type = str(row.get("source_type") or "transcript").strip().lower()
    if not isinstance(raw, dict) and source_type == "ocr":
        return None
    source = copy.deepcopy(raw) if isinstance(raw, dict) else {}
    channel = source.get("channel")
    if not isinstance(channel, dict):
        channel = {}
    return {
        "platform": str(source.get("platform") or "youtube").strip().lower(),
        "video_id": str(source.get("video_id") or video_id).strip(),
        "url": str(
            source.get("url") or video_url or _canonical_video_url(video_id)
        ).strip(),
        "channel": {
            "id": str(channel.get("id") or "").strip() or None,
            "name": str(channel.get("name") or "").strip() or None,
            "url": str(channel.get("url") or "").strip() or None,
        },
        "metadata_provider": str(
            source.get("metadata_provider") or "legacy"
        ).strip(),
        "fetched_at": str(source.get("fetched_at") or "").strip() or None,
    }


def default_insufficient_answer(language: str) -> str:
    if language == "ja":
        return "取得された書き起こし断片だけでは、確かな根拠を持って回答できません。"
    return "Insufficient transcript evidence to answer confidently from the retrieved excerpts."


def default_error_answer(language: str) -> str:
    if language == "ja":
        return "回答生成は現在利用できません。取得された根拠チャンクを確認してください。"
    return "Answer generation is currently unavailable. You can still inspect the retrieved evidence."


def _warning_text(language: str, english: str, japanese: str) -> str:
    return japanese if language == "ja" else english


def _clamp(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    return max(lower, min(upper, float(value)))


def cap_confidence(confidence: str, confidence_cap: str) -> str:
    """Cap model confidence using a deterministic upper bound."""
    scoped_confidence = (
        str(confidence or "low").strip().lower()
        if str(confidence or "low").strip().lower() in ANSWER_CONFIDENCE_LEVELS
        else "low"
    )
    scoped_cap = (
        str(confidence_cap or "low").strip().lower()
        if str(confidence_cap or "low").strip().lower() in ANSWER_CONFIDENCE_LEVELS
        else "low"
    )
    return (
        scoped_confidence
        if CONFIDENCE_ORDER[scoped_confidence] <= CONFIDENCE_ORDER[scoped_cap]
        else scoped_cap
    )


def build_citation_catalog(rows: List[dict]) -> List[dict]:
    """Map retrieved evidence rows into stable citation payloads."""
    citations: List[dict] = []
    for citation_id, row in enumerate(rows or [], start=1):
        video_id = str(row.get("video_id") or "").strip()
        if not video_id:
            continue
        source_type = str(row.get("source_type") or "transcript").strip().lower()
        start_seconds = float(row.get("start", 0.0))
        end_seconds = float(row.get("end", start_seconds))
        if source_type == "ocr":
            end_seconds = start_seconds
            video_url = str(row.get("video_url") or "").strip()
            url = str(row.get("url") or "").strip()
            timestamp_range_label = str(
                row.get("timestamp_hhmmss") or format_timestamp_label(start_seconds)
            )
            reason = f"Retrieved OCR frame #{int(row.get('rank') or citation_id)} for the question."
        else:
            video_url = str(
                row.get("video_url") or _canonical_video_url(video_id)
            ).strip()
            url = str(
                row.get("url")
                or f"{_canonical_video_url(video_id)}&t={int(max(0.0, start_seconds))}s"
            ).strip()
            timestamp_range_label = format_timestamp_range_label(
                start_seconds, end_seconds
            )
            reason = f"Retrieved chunk #{int(row.get('rank') or citation_id)} for the question."
        citations.append(
            {
                "citation_id": citation_id,
                "source_type": source_type,
                "video_id": video_id,
                "video_title": str(row.get("video_title") or video_id).strip(),
                "video_url": video_url,
                "source": _source_payload(row, video_id, video_url),
                "chunk_id": (
                    f"{video_id}:{int(row['chunk_index'])}"
                    if row.get("chunk_index") is not None
                    else f"{video_id}:{source_type}:{int(start_seconds * 1000)}:{int(end_seconds * 1000)}"
                ),
                "chunk_index": row.get("chunk_index"),
                "frame_id": row.get("frame_id"),
                "frame_path": row.get("frame_path") or (row.get("metadata") or {}).get("frame_path"),
                "language": str(row.get("language") or "").strip(),
                "start_seconds": start_seconds,
                "end_seconds": end_seconds,
                "timestamp_label": format_timestamp_label(start_seconds),
                "timestamp_range_label": timestamp_range_label,
                "snippet": _truncate_snippet(row.get("text") or ""),
                "reason": reason,
                "url": url,
                "rank": int(row.get("rank") or citation_id),
                "score": row.get("score"),
                "dense_score": row.get("dense_score"),
                "lexical_score": row.get("lexical_score"),
                "hybrid_score": row.get("hybrid_score"),
                "pre_rerank_rank": row.get("pre_rerank_rank"),
                "pre_rerank_score": row.get("pre_rerank_score"),
                "rerank_score": row.get("rerank_score"),
                "text": str(row.get("text") or "").strip(),
            }
        )
    return citations


def build_retrieved_chunks_payload(rows: List[dict]) -> List[dict]:
    """Return a lightweight debug payload derived from retrieved rows."""
    return [
        {
            "source_type": str(row.get("source_type") or "transcript").strip().lower(),
            "video_id": video_id,
            "video_title": str(row.get("video_title") or "").strip(),
            "video_url": str(
                row.get("video_url")
                or ("" if str(row.get("source_type") or "").lower() == "ocr" else _canonical_video_url(video_id))
            ).strip(),
            "source": _source_payload(
                row,
                video_id,
                str(
                    row.get("video_url")
                    or (
                        ""
                        if str(row.get("source_type") or "").lower() == "ocr"
                        else _canonical_video_url(video_id)
                    )
                ).strip(),
            ),
            "language": str(row.get("language") or "").strip(),
            "chunk_index": row.get("chunk_index"),
            "frame_id": row.get("frame_id"),
            "frame_path": row.get("frame_path") or (row.get("metadata") or {}).get("frame_path"),
            "start_seconds": float(row.get("start", 0.0)),
            "end_seconds": float(row.get("end", row.get("start", 0.0))),
            "timestamp_label": format_timestamp_label(row.get("start", 0.0)),
            "timestamp_range_label": (
                str(row.get("timestamp_hhmmss") or format_timestamp_label(row.get("start", 0.0)))
                if str(row.get("source_type") or "").lower() == "ocr"
                else format_timestamp_range_label(
                    row.get("start", 0.0), row.get("end", row.get("start", 0.0))
                )
            ),
            "url": str(
                row.get("url")
                or (
                    ""
                    if str(row.get("source_type") or "").lower() == "ocr"
                    else f"{_canonical_video_url(video_id)}&t={int(max(0.0, float(row.get('start', 0.0))))}s"
                )
            ).strip(),
            "rank": int(row.get("rank") or 0) or None,
            "score": row.get("score"),
            "pre_rerank_rank": row.get("pre_rerank_rank"),
            "pre_rerank_score": row.get("pre_rerank_score"),
            "rerank_score": row.get("rerank_score"),
            "snippet": _truncate_snippet(row.get("text") or "", max_chars=180),
        }
        for row in rows or []
        for video_id in [str(row.get("video_id") or "").strip()]
        if video_id
    ]


def build_grounded_answer_messages(
    *, question: str, citations: List[dict], answer_language: str
) -> Tuple[str, str]:
    """Return the system and user prompt for grounded answer generation."""
    language_instruction = (
        "Answer in Japanese." if answer_language == "ja" else "Answer in English."
    )
    evidence_lines = []
    for citation in citations:
        source_label = "OCR" if citation.get("source_type") == "ocr" else "Transcript"
        locator_label = "Frame" if citation.get("source_type") == "ocr" else "URL"
        locator_value = citation.get("frame_path") or citation.get("url") or "-"
        source = (
            citation.get("source")
            if isinstance(citation.get("source"), dict)
            else {}
        )
        channel = (
            source.get("channel")
            if isinstance(source.get("channel"), dict)
            else {}
        )
        platform = str(source.get("platform") or "").strip()
        provenance = "YouTube" if platform.lower() == "youtube" else platform.title()
        if channel.get("name"):
            provenance = f"{provenance or 'YouTube'} channel: {channel['name']}"
        evidence_lines.append(
            "\n".join(
                [
                    f"[{citation['citation_id']}] {citation['video_title']} | {citation['timestamp_range_label']}",
                    f"Provenance: {provenance}" if provenance else "Provenance: unavailable",
                    f"{locator_label}: {locator_value}",
                    f"{source_label}: {citation['text']}",
                ]
            )
        )

    system_prompt = (
        "You answer questions only from the provided transcript and OCR evidence.\n"
        f"{language_instruction}\n"
        "Use only the provided evidence excerpts.\n"
        "Do not invent facts, names, dates, or conclusions.\n"
        "Cite every key claim inline with square-bracket markers like [1] or [2].\n"
        "If the evidence is weak, missing, or only partially relevant, return status='insufficient_evidence'.\n"
        "If the evidence conflicts, say so in the answer, keep confidence low, and add a warning.\n"
        "Return strict JSON only with this schema:\n"
        "{\n"
        '  "status": "answered" | "insufficient_evidence",\n'
        '  "confidence": "high" | "medium" | "low",\n'
        '  "answer": "concise answer text with inline citations like [1]",\n'
        '  "citations": [{"citation_id": 1, "reason": "why this excerpt supports the answer"}],\n'
        '  "warnings": ["optional warning"],\n'
        '  "has_conflict": false\n'
        "}\n"
        "Only cite citation_ids that exist in the evidence list."
    )
    user_message = (
        f"Question:\n{question.strip()}\n\n"
        "Evidence excerpts:\n"
        f"{chr(10).join(evidence_lines)}\n\n"
        "Return strict JSON only."
    )
    return system_prompt, user_message


def _normalize_dense_signal(row: dict) -> float:
    dense_score = row.get("dense_score", row.get("score"))
    if dense_score is None:
        return 0.0
    return _clamp((float(dense_score) - 0.55) / 0.30)


def _normalize_lexical_signal(row: dict, top_lexical_score: float) -> float:
    lexical_score = row.get("lexical_score", row.get("score"))
    if lexical_score is None or top_lexical_score <= 0.0:
        return 0.0
    return _clamp(float(lexical_score) / top_lexical_score)


def assess_grounded_answer_evidence(
    *,
    question: str,
    rows: List[dict],
    retrieval_mode: str,
    tokenize_fn: Callable[[str, str], List[str]],
    answer_language: str,
    query_language: Optional[str] = None,
) -> dict:
    """Decide whether retrieval evidence is strong enough to justify generation."""
    if not rows:
        return {
            "sufficient": False,
            "confidence_cap": "low",
            "warnings": [
                _warning_text(
                    answer_language,
                    "No transcript chunks matched the question.",
                    "質問に合う書き起こしチャンクが見つかりませんでした。",
                )
            ],
            "reason_code": "no_results",
        }

    query_tokens = tokenize_fn(question, query_language or answer_language)
    query_token_set = {token for token in query_tokens if token}
    top_lexical_score = max(
        float(row.get("lexical_score", 0.0) or 0.0) for row in rows or [{}]
    )
    scoped_mode = str(retrieval_mode or "hybrid").strip().lower()
    assessed_rows = []

    for row in rows:
        text = str(row.get("text") or "").strip()
        row_tokens = tokenize_fn(text, str(row.get("language") or answer_language))
        row_token_set = {token for token in row_tokens if token}
        overlap_ratio = (
            len(query_token_set & row_token_set) / max(len(query_token_set), 1)
            if query_token_set
            else 0.0
        )
        rank = int(row.get("rank") or (len(assessed_rows) + 1))
        rank_signal = _clamp(1.0 - ((rank - 1) / 5.0))
        dense_signal = _normalize_dense_signal(row)
        lexical_signal = _normalize_lexical_signal(row, top_lexical_score)
        overlap_signal = _clamp(overlap_ratio / 0.6)

        weighted_components = [
            (0.45, overlap_signal),
            (0.15, rank_signal),
        ]
        if scoped_mode in {"dense", "hybrid"}:
            weighted_components.append((0.25, dense_signal))
        if scoped_mode in {"lexical", "hybrid"}:
            weighted_components.append((0.15, lexical_signal))

        weight_sum = sum(weight for weight, _ in weighted_components)
        evidence_strength = (
            sum(weight * signal for weight, signal in weighted_components) / weight_sum
            if weight_sum > 0.0
            else 0.0
        )
        assessed_rows.append(
            {
                "row": row,
                "rank": rank,
                "overlap_ratio": overlap_ratio,
                "dense_signal": dense_signal,
                "lexical_signal": lexical_signal,
                "evidence_strength": evidence_strength,
            }
        )

    assessed_rows.sort(
        key=lambda item: (item["evidence_strength"], -item["rank"]),
        reverse=True,
    )
    top_row = assessed_rows[0]
    runner_up = assessed_rows[1] if len(assessed_rows) > 1 else None
    supporting_rows = [
        item
        for item in assessed_rows
        if item["evidence_strength"] >= 0.55 and item["overlap_ratio"] >= 0.18
    ]

    if len(supporting_rows) >= 2:
        confidence_cap = (
            "high"
            if top_row["evidence_strength"] >= 0.78
            and supporting_rows[1]["evidence_strength"] >= 0.60
            else "medium"
        )
        return {
            "sufficient": True,
            "confidence_cap": confidence_cap,
            "warnings": [],
            "reason_code": "multi_chunk_support",
        }

    strong_single = (
        top_row["evidence_strength"] >= 0.72 and top_row["overlap_ratio"] >= 0.28
    )
    weak_runner_up = runner_up is None or runner_up["evidence_strength"] < 0.48
    if strong_single and weak_runner_up:
        return {
            "sufficient": True,
            "confidence_cap": "medium",
            "warnings": [
                _warning_text(
                    answer_language,
                    "The answer relies mainly on a single excerpt.",
                    "根拠は主に1つの抜粋に依存しています。",
                )
            ],
            "reason_code": "strong_single_chunk",
        }

    if len(rows) == 1:
        warning = _warning_text(
            answer_language,
            "The best retrieved chunk is too weak to support a grounded answer.",
            "最上位の取得チャンクだけでは、根拠として弱すぎます。",
        )
        return {
            "sufficient": False,
            "confidence_cap": "low",
            "warnings": [warning],
            "reason_code": "single_weak_chunk",
        }

    warning = _warning_text(
        answer_language,
        "Retrieved evidence is too thin or weakly matched to answer confidently.",
        "取得された根拠は薄く、質問との一致も弱いため、確信を持って回答できません。",
    )
    if runner_up and runner_up["evidence_strength"] >= 0.45:
        warning = _warning_text(
            answer_language,
            "Retrieved evidence is split across weakly supported chunks.",
            "取得された根拠が複数の弱いチャンクに分散しています。",
        )
        reason_code = "mixed_signals"
    else:
        reason_code = "thin_support"
    return {
        "sufficient": False,
        "confidence_cap": "low",
        "warnings": [warning],
        "reason_code": reason_code,
    }


def normalize_grounded_answer_payload(
    *,
    raw_payload: dict,
    citations: List[dict],
    answer_language: str,
) -> dict:
    """Normalize model output into a stable grounded-answer response payload."""
    citation_index: Dict[int, dict] = {
        int(citation["citation_id"]): citation for citation in citations
    }
    status = str(raw_payload.get("status") or "insufficient_evidence").strip().lower()
    if status not in ANSWER_STATUSES - {"error"}:
        status = "insufficient_evidence"

    confidence = str(raw_payload.get("confidence") or "low").strip().lower()
    if confidence not in ANSWER_CONFIDENCE_LEVELS:
        confidence = "low"

    answer = str(raw_payload.get("answer") or "").strip()
    warnings = [
        str(item).strip()
        for item in (raw_payload.get("warnings") or [])
        if str(item).strip()
    ]
    has_conflict = bool(raw_payload.get("has_conflict"))

    normalized_citations: List[dict] = []
    seen_ids = set()
    for item in raw_payload.get("citations") or []:
        if not isinstance(item, dict):
            continue
        try:
            citation_id = int(item.get("citation_id"))
        except (TypeError, ValueError):
            continue
        if citation_id in seen_ids or citation_id not in citation_index:
            continue
        merged = copy.deepcopy(citation_index[citation_id])
        reason = str(item.get("reason") or "").strip()
        if reason:
            merged["reason"] = reason
        normalized_citations.append(merged)
        seen_ids.add(citation_id)

    if has_conflict:
        conflict_warning = (
            "Retrieved evidence contains conflicting statements."
            if answer_language != "ja"
            else "取得された根拠には食い違う記述があります。"
        )
        if conflict_warning not in warnings:
            warnings.append(conflict_warning)
        confidence = "low"

    if status == "answered" and not normalized_citations:
        status = "insufficient_evidence"
        warnings.append(
            "Model answer did not include valid supporting citations."
            if answer_language != "ja"
            else "モデルの回答に有効な根拠引用が含まれていませんでした。"
        )

    if status == "answered" and not answer:
        status = "insufficient_evidence"

    if status == "answered":
        inline_ids = {
            int(match.group(1)) for match in INLINE_CITATION_RE.finditer(answer)
        }
        missing_inline = [
            citation["citation_id"]
            for citation in normalized_citations
            if citation["citation_id"] not in inline_ids
        ]
        if missing_inline:
            suffix = " ".join(f"[{citation_id}]" for citation_id in missing_inline)
            answer = f"{answer} {suffix}".strip()
    else:
        if not answer:
            answer = default_insufficient_answer(answer_language)
        confidence = "low"

    return {
        "status": status,
        "confidence": confidence,
        "answer": answer,
        "citations": normalized_citations,
        "warnings": warnings,
    }
