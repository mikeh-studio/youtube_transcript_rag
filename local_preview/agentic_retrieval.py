"""Agentic retrieval loop that retries retrieval when evidence is insufficient.

Reuses the grounded-answer evidence assessment as the control signal.
Each insufficient attempt picks a follow-up strategy from the assessment
reason code (rewrite the query, switch retrieval mode, or broaden top-k),
retries, and stops at the first sufficient evidence set. If no attempt is
sufficient, the initial attempt's rows are returned so the standard
insufficient-evidence handling still applies to the original query.
"""

from __future__ import annotations

import re
from typing import Callable, List, Optional

AGENTIC_MAX_ATTEMPTS = 3
AGENTIC_MAX_TOP_K = 12

STRATEGY_INITIAL = "initial"
STRATEGY_REWRITE_QUERY = "rewrite_query"
STRATEGY_SWITCH_MODE = "switch_mode"
STRATEGY_BROADEN_TOP_K = "broaden_top_k"

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

# Longer phrases first so e.g. "について教えてください" is removed before "ください".
_JA_QUESTION_PHRASES = (
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
)


def normalize_query_for_compare(value: str) -> str:
    """Collapse whitespace and lowercase for attempted-query dedupe."""
    return re.sub(r"\s+", " ", str(value or "")).strip().lower()


def heuristic_query_rewrite(question: str, language: Optional[str] = None) -> Optional[str]:
    """Deterministic keyword-style rewrite used when no LLM is available.

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
    first_outcome: Optional[dict] = None
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
        if first_outcome is None:
            first_outcome = outcome
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
                first_outcome,
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
        first_outcome, attempts, question, stopped_reason=STOPPED_MAX_ATTEMPTS
    )
