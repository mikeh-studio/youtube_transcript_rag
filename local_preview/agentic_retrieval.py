"""Agentic retrieval loop that retries retrieval when evidence is insufficient.

Reuses the grounded-answer evidence assessment as the control signal.
Each insufficient attempt picks a follow-up strategy from the assessment
reason code (rewrite the query, switch retrieval mode, or broaden top-k),
retries, and stops at the first sufficient evidence set. If no attempt is
sufficient, the attempt with the most retrieved evidence (earliest on ties,
so the original query wins in the common case) is returned and the standard
insufficient-evidence handling still applies.
"""

from __future__ import annotations

import math
import re
from typing import Callable, List, Optional

AGENTIC_MAX_ATTEMPTS = 3
AGENTIC_MAX_TOP_K = 12

STRATEGY_INITIAL = "initial"
STRATEGY_REWRITE_QUERY = "rewrite_query"
STRATEGY_SWITCH_MODE = "switch_mode"
STRATEGY_BROADEN_TOP_K = "broaden_top_k"
STRATEGY_READ_CONTEXT = "read_context"

TOOL_SEMANTIC_SEARCH = "semantic_search"
TOOL_KEYWORD_SEARCH = "keyword_search"
TOOL_READ_CONTEXT = "read_context"
AGENTIC_CONTEXT_ANCHOR_LIMIT = 3

STOPPED_SUFFICIENT = "sufficient_evidence"
STOPPED_MAX_ATTEMPTS = "max_attempts"
STOPPED_NO_NEW_STRATEGY = "no_new_strategy"

_MODE_SWITCH_ORDER = {
    "hybrid": ("lexical", "dense"),
    "dense": ("hybrid", "lexical"),
    "lexical": ("hybrid", "dense"),
}

_JP_CHAR_RE = re.compile(r"[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff]")
_TOKEN_RE = re.compile(r"[a-zA-Z0-9_]+")

_EN_QUESTION_STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "am", "be", "been", "being",
    "do", "does", "did", "can", "could", "should", "would", "will", "shall",
    "may", "might", "what", "which", "who", "whom", "whose", "when", "where",
    "why", "how", "please", "tell", "explain", "describe", "about", "of",
    "in", "on", "at", "to", "for", "and", "or", "this", "that", "these",
    "those", "it", "its", "they", "their", "there", "me", "my", "i", "you",
    "your", "we", "us", "our",
}

# Sorted longest-first so e.g. "何ですか" is removed before "ですか" and no
# filler fragment is left behind in the rewritten query.
_JA_QUESTION_PHRASES = tuple(
    sorted(
        (
            "について教えてください",
            "について教えて",
            "を教えてください",
            "を教えて",
            "とは何ですか",
            "とはなんですか",
            "とは何か",
            "でしょうか",
            "ですか",
            "ますか",
            "ください",
            "何ですか",
            "どのように",
            "どうやって",
            "どうして",
            "なぜ",
        ),
        key=len,
        reverse=True,
    )
)


def normalize_query_for_compare(value: str) -> str:
    """Collapse whitespace and lowercase for attempted-query dedupe."""
    return re.sub(r"\s+", " ", str(value or "")).strip().lower()


def heuristic_query_rewrite(question: str, language: Optional[str] = None) -> Optional[str]:
    """Return a deterministic keyword-style retrieval rewrite.

    Strips question phrasing and filler words so retrieval focuses on
    distinctive content terms. Returns None when the rewrite would be
    empty or identical to the input.
    """
    original = normalize_query_for_compare(question)
    if not original:
        return None

    is_ja = language == "ja" or bool(_JP_CHAR_RE.search(str(question)))
    if is_ja:
        rewritten = str(question)
        for phrase in _JA_QUESTION_PHRASES:
            rewritten = rewritten.replace(phrase, " ")
        rewritten = re.sub(r"[?？!！。、，,.]", " ", rewritten)
        rewritten = re.sub(r"\s+", " ", rewritten).strip()
    else:
        tokens = _TOKEN_RE.findall(str(question).lower())
        kept = [token for token in tokens if token not in _EN_QUESTION_STOPWORDS]
        rewritten = " ".join(kept)

    if not rewritten:
        return None
    if normalize_query_for_compare(rewritten) == original:
        return None
    return rewritten


def choose_initial_search_tool(question: str, language: Optional[str] = None) -> str:
    """Choose a deterministic first retrieval tool for an agentic search."""
    text = str(question or "")
    is_japanese = language == "ja" or bool(_JP_CHAR_RE.search(text))
    has_quote = bool(re.search(r'["“”「」『』]', text))
    has_number = bool(re.search(r"\d", text))
    has_acronym = bool(re.search(r"\b[A-Z]{2,}\b", text))
    if is_japanese or has_quote or has_number or has_acronym:
        return TOOL_KEYWORD_SEARCH
    return TOOL_SEMANTIC_SEARCH


def _agentic_tool_attempt(
    *,
    attempt_no: int,
    strategy: str,
    tool: str,
    query: str,
    response: dict,
    assessment: dict,
    k: int,
) -> dict:
    rows = list(response.get("results") or [])
    return {
        "attempt": attempt_no,
        "strategy": strategy,
        "tool": tool,
        "query": query,
        "retrieval_mode": response.get("retrieval_mode"),
        "k": k,
        "result_count": len(rows),
        "sufficient": bool(assessment.get("sufficient")),
        "reason_code": str(assessment.get("reason_code") or ""),
        "confidence_cap": str(assessment.get("confidence_cap") or "low"),
    }


def run_agentic_tool_search(
    *,
    question: str,
    semantic_search_fn: Callable[..., dict],
    keyword_search_fn: Callable[..., dict],
    read_context_fn: Callable[..., dict],
    assess_fn: Callable[..., dict],
    rewrite_fn: Optional[Callable[..., Optional[str]]] = None,
    k: int = 5,
    language: Optional[str] = None,
    context_window: float = 30.0,
) -> dict:
    """Run a bounded deterministic semantic/keyword/context tool policy."""
    original_query = str(question or "").strip()
    initial_tool = choose_initial_search_tool(original_query, language=language)
    alternate_tool = (
        TOOL_SEMANTIC_SEARCH
        if initial_tool == TOOL_KEYWORD_SEARCH
        else TOOL_KEYWORD_SEARCH
    )
    search_fns = {
        TOOL_SEMANTIC_SEARCH: semantic_search_fn,
        TOOL_KEYWORD_SEARCH: keyword_search_fn,
    }
    attempts = []
    outcomes = []

    first_response = search_fns[initial_tool](query=original_query, k=k)
    first_rows = list(first_response.get("results") or [])
    first_assessment = assess_fn(
        rows=first_rows,
        retrieval_mode=str(first_response.get("retrieval_mode") or "dense"),
    )
    attempts.append(
        _agentic_tool_attempt(
            attempt_no=1,
            strategy=STRATEGY_INITIAL,
            tool=initial_tool,
            query=original_query,
            response=first_response,
            assessment=first_assessment,
            k=k,
        )
    )
    outcomes.append(
        {
            "retrieval": first_response,
            "rows": first_rows,
            "assessment": first_assessment,
            "query": original_query,
            "tool": initial_tool,
        }
    )

    if not first_assessment.get("sufficient"):
        rewritten = None
        if rewrite_fn is not None:
            rewritten = rewrite_fn(
                query=original_query,
                attempted_queries={normalize_query_for_compare(original_query)},
            )
        alternate_query = str(rewritten or original_query).strip()
        second_response = search_fns[alternate_tool](query=alternate_query, k=k)
        second_rows = list(second_response.get("results") or [])
        second_assessment = assess_fn(
            rows=second_rows,
            retrieval_mode=str(second_response.get("retrieval_mode") or "lexical"),
        )
        attempts.append(
            _agentic_tool_attempt(
                attempt_no=2,
                strategy=(
                    STRATEGY_REWRITE_QUERY if rewritten else STRATEGY_SWITCH_MODE
                ),
                tool=alternate_tool,
                query=alternate_query,
                response=second_response,
                assessment=second_assessment,
                k=k,
            )
        )
        outcomes.append(
            {
                "retrieval": second_response,
                "rows": second_rows,
                "assessment": second_assessment,
                "query": alternate_query,
                "tool": alternate_tool,
            }
        )

    best = outcomes[0]
    for candidate in outcomes[1:]:
        candidate_sufficient = bool(candidate["assessment"].get("sufficient"))
        best_sufficient = bool(best["assessment"].get("sufficient"))
        if candidate_sufficient and not best_sufficient:
            best = candidate
        elif candidate_sufficient == best_sufficient and len(candidate["rows"]) > len(
            best["rows"]
        ):
            best = candidate

    context_rows = []
    context_calls = []
    for row_index, row in enumerate(best["rows"]):
        if row_index >= AGENTIC_CONTEXT_ANCHOR_LIMIT:
            context_rows.append(row)
            continue
        video_id = str(row.get("video_id") or "").strip()
        if not video_id:
            context_rows.append(row)
            continue
        try:
            row_start = float(row.get("start", 0.0))
            row_end = float(row.get("end", row_start))
        except (TypeError, ValueError):
            context_rows.append(row)
            continue
        if not math.isfinite(row_start) or not math.isfinite(row_end):
            context_rows.append(row)
            continue
        if row_end < row_start:
            row_start, row_end = row_end, row_start
        timestamp = row_start + ((row_end - row_start) / 2.0)
        scoped_window = max(
            float(context_window),
            (row_end - row_start) / 2.0,
        )
        try:
            context = read_context_fn(
                video_id=video_id,
                timestamp=timestamp,
                window=scoped_window,
            )
        except (KeyError, TypeError, ValueError):
            context_rows.append(row)
            continue
        if not str(context.get("text") or "").strip():
            context_rows.append(row)
            continue
        expanded = dict(row)
        expanded.update(
            {
                "text": context["text"],
                "retrieval_start": row_start,
                "retrieval_end": row_end,
                "retrieval_url": row.get("url"),
                "start": context["start"],
                "end": context["end"],
                "url": context.get("url") or row.get("url"),
                "context_start": context["start"],
                "context_end": context["end"],
                "context_segments": context["segments"],
                "context_segment_count": context["segment_count"],
                "source_basis": context["source_basis"],
            }
        )
        context_rows.append(expanded)
        context_calls.append(
            {
                "video_id": video_id,
                "timestamp": timestamp,
                "window": scoped_window,
                "requested_start": context.get("requested_start"),
                "requested_end": context.get("requested_end"),
                "start": context["start"],
                "end": context["end"],
                "segment_count": context["segment_count"],
            }
        )

    final_assessment = best["assessment"]
    final_rows = best["rows"]
    context_accepted = False
    if context_calls:
        context_assessment = assess_fn(
            rows=context_rows,
            retrieval_mode=str(best["retrieval"].get("retrieval_mode") or "dense"),
        )
        confidence_order = {"low": 0, "medium": 1, "high": 2}
        best_sufficient = bool(best["assessment"].get("sufficient"))
        context_sufficient = bool(context_assessment.get("sufficient"))
        context_is_weaker = best_sufficient and (
            not context_sufficient
            or confidence_order.get(
                str(context_assessment.get("confidence_cap") or "low"), 0
            )
            < confidence_order.get(
                str(best["assessment"].get("confidence_cap") or "low"), 0
            )
        )
        context_accepted = not context_is_weaker
        if context_accepted:
            final_rows = context_rows
            final_assessment = context_assessment
        attempts.append(
            {
                "attempt": len(attempts) + 1,
                "strategy": STRATEGY_READ_CONTEXT,
                "tool": TOOL_READ_CONTEXT,
                "query": best["query"],
                "retrieval_mode": best["retrieval"].get("retrieval_mode"),
                "k": k,
                "result_count": len(final_rows),
                "sufficient": context_sufficient,
                "reason_code": str(context_assessment.get("reason_code") or ""),
                "confidence_cap": str(
                    context_assessment.get("confidence_cap") or "low"
                ),
                "accepted": context_accepted,
                "calls": context_calls,
            }
        )

    retrieval = dict(best["retrieval"])
    retrieval["results"] = final_rows
    retrieval["details"] = dict(retrieval.get("details") or {})
    stopped_reason = (
        STOPPED_SUFFICIENT
        if final_assessment.get("sufficient")
        else STOPPED_MAX_ATTEMPTS
    )
    return {
        "question": original_query,
        "retrieval": retrieval,
        "rows": final_rows,
        "assessment": final_assessment,
        "final_query": best["query"],
        "final_mode": retrieval.get("retrieval_mode"),
        "final_k": k,
        "final_tool": TOOL_READ_CONTEXT if context_accepted else best["tool"],
        "sufficient": bool(final_assessment.get("sufficient")),
        "agentic_applied": len(attempts) > 1,
        "stopped_reason": stopped_reason,
        "attempts": attempts,
    }


def _next_step(
    *,
    reason_code: str,
    query: str,
    mode: str,
    k: int,
    attempted_queries: set,
    attempted_modes: set,
    rewrite_fn: Optional[Callable],
    max_k: int,
) -> Optional[dict]:
    """Pick the next retrieval attempt from the assessment reason code."""

    def _try_rewrite() -> Optional[dict]:
        if rewrite_fn is None:
            return None
        candidate = rewrite_fn(query=query, attempted_queries=set(attempted_queries))
        candidate = str(candidate or "").strip()
        if not candidate:
            return None
        if normalize_query_for_compare(candidate) in attempted_queries:
            return None
        return {
            "strategy": STRATEGY_REWRITE_QUERY,
            "query": candidate,
            "mode": mode,
            "k": k,
        }

    def _try_switch_mode() -> Optional[dict]:
        for candidate_mode in _MODE_SWITCH_ORDER.get(mode, ()):
            if candidate_mode not in attempted_modes:
                return {
                    "strategy": STRATEGY_SWITCH_MODE,
                    "query": query,
                    "mode": candidate_mode,
                    "k": k,
                }
        return None

    def _try_broaden_top_k() -> Optional[dict]:
        if int(k) >= int(max_k):
            return None
        return {
            "strategy": STRATEGY_BROADEN_TOP_K,
            "query": query,
            "mode": mode,
            "k": min(int(max_k), max(int(k) + 1, int(k) * 2)),
        }

    scoped_reason = str(reason_code or "").strip().lower()
    if scoped_reason == "mixed_signals":
        order = (_try_switch_mode, _try_rewrite, _try_broaden_top_k)
    elif scoped_reason == "single_weak_chunk":
        order = (_try_broaden_top_k, _try_rewrite, _try_switch_mode)
    else:  # no_results, thin_support, and unknown codes
        order = (_try_rewrite, _try_switch_mode, _try_broaden_top_k)

    for factory in order:
        step = factory()
        if step is not None:
            return step
    return None


def _final_payload(
    outcome: dict, attempts: List[dict], question: str, *, stopped_reason: str
) -> dict:
    return {
        "question": question,
        "retrieval": outcome["retrieval"],
        "rows": outcome["rows"],
        "assessment": outcome["assessment"],
        "final_query": outcome["query"],
        "final_mode": outcome["mode"],
        "final_k": outcome["k"],
        "sufficient": bool(outcome["assessment"].get("sufficient")),
        "agentic_applied": len(attempts) > 1,
        "stopped_reason": stopped_reason,
        "attempts": attempts,
    }


def run_agentic_retrieval(
    *,
    question: str,
    retrieve_fn: Callable[..., dict],
    assess_fn: Callable[..., dict],
    rewrite_fn: Optional[Callable[..., Optional[str]]] = None,
    k: int = 5,
    retrieval_mode: str = "hybrid",
    max_attempts: int = AGENTIC_MAX_ATTEMPTS,
    max_k: int = AGENTIC_MAX_TOP_K,
) -> dict:
    """Run retrieval with evidence-driven retries.

    Args:
        question: The original user question. All attempts are assessed
            against this question, even when retrieval used a rewritten query.
        retrieve_fn: Callable(query=, retrieval_mode=, k=) -> retrieval dict
            with a 'results' list (LocalRAGService.retrieve shape).
        assess_fn: Callable(rows=, retrieval_mode=) -> evidence assessment
            dict with 'sufficient', 'reason_code', 'confidence_cap'.
        rewrite_fn: Optional Callable(query=, attempted_queries=) -> new
            query string or None when no useful rewrite exists.
        k: Initial top-k for retrieval.
        retrieval_mode: Initial retrieval mode.
        max_attempts: Total attempt budget including the initial attempt.
        max_k: Upper bound for the broaden_top_k strategy.

    Returns:
        Dict with the winning attempt's 'retrieval', 'rows', 'assessment',
        'final_query'/'final_mode'/'final_k', plus an 'attempts' trace,
        'agentic_applied', 'sufficient', and 'stopped_reason'.
    """
    scoped_mode = str(retrieval_mode or "hybrid").strip().lower()
    current = {
        "strategy": STRATEGY_INITIAL,
        "query": str(question or "").strip(),
        "mode": scoped_mode,
        "k": max(1, int(k)),
    }
    attempted_queries = {normalize_query_for_compare(current["query"])}
    attempted_modes = {scoped_mode}
    attempts: List[dict] = []
    best_outcome: Optional[dict] = None
    attempt_budget = max(1, int(max_attempts))

    for attempt_no in range(1, attempt_budget + 1):
        retrieval = retrieve_fn(
            query=current["query"],
            retrieval_mode=current["mode"],
            k=current["k"],
        )
        rows = list(retrieval.get("results") or [])
        assessment = assess_fn(rows=rows, retrieval_mode=current["mode"])
        attempts.append(
            {
                "attempt": attempt_no,
                "strategy": current["strategy"],
                "query": current["query"],
                "retrieval_mode": current["mode"],
                "k": current["k"],
                "result_count": len(rows),
                "sufficient": bool(assessment.get("sufficient")),
                "reason_code": str(assessment.get("reason_code") or ""),
                "confidence_cap": str(assessment.get("confidence_cap") or "low"),
            }
        )
        outcome = {
            "retrieval": retrieval,
            "rows": rows,
            "assessment": assessment,
            "query": current["query"],
            "mode": current["mode"],
            "k": current["k"],
        }
        # Fallback candidate when no attempt is sufficient: the attempt with
        # the most retrieved evidence, earliest on ties (so the original
        # query wins unless a retry found strictly more rows).
        if best_outcome is None or len(rows) > len(best_outcome["rows"]):
            best_outcome = outcome
        if assessment.get("sufficient"):
            return _final_payload(
                outcome, attempts, question, stopped_reason=STOPPED_SUFFICIENT
            )
        if attempt_no >= attempt_budget:
            break

        step = _next_step(
            reason_code=str(assessment.get("reason_code") or ""),
            query=current["query"],
            mode=current["mode"],
            k=current["k"],
            attempted_queries=attempted_queries,
            attempted_modes=attempted_modes,
            rewrite_fn=rewrite_fn,
            max_k=max_k,
        )
        if step is None:
            return _final_payload(
                best_outcome,
                attempts,
                question,
                stopped_reason=STOPPED_NO_NEW_STRATEGY,
            )
        current = {
            "strategy": step["strategy"],
            "query": step["query"],
            "mode": step["mode"],
            "k": step["k"],
        }
        attempted_queries.add(normalize_query_for_compare(step["query"]))
        attempted_modes.add(step["mode"])

    return _final_payload(
        best_outcome, attempts, question, stopped_reason=STOPPED_MAX_ATTEMPTS
    )
