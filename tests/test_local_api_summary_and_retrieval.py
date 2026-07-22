"""Tests for local preview summary generation and retrieval upgrades."""

import importlib
import json
import os
import re
import sys
import tempfile
import threading
from pathlib import Path

import pytest


ROOT_DIR = Path(__file__).resolve().parents[1]
LOCAL_PREVIEW_DIR = ROOT_DIR / "local_preview"
if str(LOCAL_PREVIEW_DIR) not in sys.path:
    sys.path.insert(0, str(LOCAL_PREVIEW_DIR))

os.environ["YT_RAG_SKIP_GLOBAL_SERVICE"] = "1"
local_api = importlib.import_module("local_api")
LocalRAGService = local_api.LocalRAGService


def _make_service(enabled=True):
    service = LocalRAGService.__new__(LocalRAGService)
    service.feedback = {}
    service.feedback_index = {}
    service.feedback_lock = threading.Lock()
    service.feedback_tuning_enabled = bool(enabled)
    service._persist_feedback = lambda: None
    runtime_dir = Path(tempfile.mkdtemp())
    cache_dir = Path(tempfile.mkdtemp())
    legacy_dir = Path(tempfile.mkdtemp())
    service.runtime_data_dir = runtime_dir
    service.cache_data_dir = cache_dir
    service.summary_cache_dir = cache_dir / "summaries"
    service.legacy_data_dir = legacy_dir
    service.feedback_path = runtime_dir / "search_feedback.json"
    service.legacy_feedback_path = legacy_dir / "search_feedback.json"
    service.openai_model = "gpt-5.4-mini"
    service.sakana_model = "fugu"
    service.sakana_base_url = local_api.SAKANA_DEFAULT_BASE_URL
    service._sakana_client = None
    service.ask_history_lock = threading.Lock()
    service.ask_history = {}
    service.ask_history_path = runtime_dir / "ask_history.json"
    service.legacy_ask_history_path = legacy_dir / "ask_history.json"
    service.ingest_log_path = runtime_dir / "ingest_jobs.log"
    service.legacy_ingest_log_path = legacy_dir / "ingest_jobs.log"
    service.log_lock = threading.Lock()
    return service


def _make_study_service():
    class DummyLibrary:
        def __init__(self):
            chunk_texts = [
                "The host opens the episode and explains why the video will become a study deck.",
                "The intro describes transcript evidence, timestamps, and source-grounded review.",
                "The first section frames the main idea as learning from processed video evidence.",
                "Serie is introduced as a character whose overwhelming power shapes the discussion.",
                "The speakers compare Serie with other characters and describe her loneliness.",
                "A second Serie moment focuses on playful behavior and how learners should interpret it.",
                "The acting section explains performance direction and how emotion should sound.",
                "The cast discusses voice choices, ordinary delivery, and character nuance.",
                "The review section names concrete details that are easy to miss while listening.",
                "The vocabulary section explains terms, phrasing, and translation nuance.",
                "The closing section returns to favorite magic and episode announcements.",
                "The final reminder asks learners to replay source links when checking answers.",
            ]
            chunks = [
                {
                    "raw_text": text,
                    "start": float(idx * 30),
                    "end": float(idx * 30 + 24),
                }
                for idx, text in enumerate(chunk_texts)
            ]
            segments = [
                {
                    "text": row["raw_text"],
                    "start": row["start"],
                    "end": row["end"],
                }
                for row in chunks
            ]
            self.videos = {
                "vidStudy001": {
                    "title": "Study Mode Demo",
                    "url": "https://www.youtube.com/watch?v=vidStudy001",
                    "language": "en",
                    "chunks": chunks,
                    "full_transcript": {
                        "version": 1,
                        "text": "\n".join(row["text"] for row in segments),
                        "segments": segments,
                        "segment_count": len(segments),
                        "char_count": sum(len(row["text"]) for row in segments),
                    },
                }
            }

    class DummyEngine:
        def __init__(self):
            self.library = DummyLibrary()
            self.model = "claude-sonnet-4-5-20250929"

    service = _make_service(enabled=False)
    service.engine = DummyEngine()
    service.title_cache = {}
    return service


def test_allowed_local_origin_accepts_only_loopback_hosts():
    assert (
        local_api._allowed_local_origin("http://127.0.0.1:8000")
        == "http://127.0.0.1:8000"
    )
    assert (
        local_api._allowed_local_origin("https://localhost:3000")
        == "https://localhost:3000"
    )
    assert local_api._allowed_local_origin("https://example.com") is None
    assert local_api._allowed_local_origin("file:///tmp/index.html") is None


def test_persist_ask_history_writes_private_file_mode():
    service = _make_service(enabled=False)
    service.ask_history = {"vid1": [{"question": "q1"}]}

    LocalRAGService._persist_ask_history(service)

    assert service.ask_history_path.exists()
    assert (service.ask_history_path.stat().st_mode & 0o777) == 0o600


def test_persist_feedback_writes_private_file_mode():
    service = _make_service(enabled=False)
    service.feedback = {"vid1:0": {"video_id": "vid1", "label": "relevant"}}

    LocalRAGService._persist_feedback(service)

    assert service.feedback_path.exists()
    assert (service.feedback_path.stat().st_mode & 0o777) == 0o600


def test_list_videos_includes_chunking_metadata():
    class DummyLibrary:
        def __init__(self):
            self.videos = {
                "vid1": {
                    "title": "Video vid1",
                    "url": "https://www.youtube.com/watch?v=vid1",
                    "language": "en",
                    "chunks": [{"raw_text": "hello", "start": 0.0, "end": 8.0}],
                }
            }

        def get_video_chunking_metadata(self, video_id):
            assert video_id == "vid1"
            return {"version": "time_v1_45s_15s"}

        def video_chunking_is_stale(self, video_id):
            assert video_id == "vid1"
            return True

    class DummyEngine:
        def __init__(self):
            self.library = DummyLibrary()

    service = _make_service(enabled=False)
    service.engine = DummyEngine()
    service.title_cache = {}
    service._hydrate_video_title = lambda video_id: None

    videos = service.list_videos()

    assert videos == [
        {
            "video_id": "vid1",
            "title": "Video vid1",
            "url": "https://www.youtube.com/watch?v=vid1",
            "language": "en",
            "num_chunks": 1,
            "chunking_version": "time_v1_45s_15s",
            "chunking_stale": True,
            "ask_history_count": 0,
            "last_ask_at": None,
        }
    ]


def test_generate_study_artifact_completes_all_three_modes_without_provider_keys(
    monkeypatch,
):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    service = _make_study_service()

    flashcards = service.generate_study_artifact(
        mode="flashcards",
        video_id="vidStudy001",
        language="en",
        provider="chatgpt",
        model=None,
        card_count=6,
        difficulty="balanced",
    )
    topics = service.generate_study_artifact(
        mode="topics",
        video_id="vidStudy001",
        language="en",
        provider="chatgpt",
        model=None,
        card_count=6,
        difficulty="balanced",
    )
    quality = service.generate_study_artifact(
        mode="quality",
        video_id="vidStudy001",
        language="en",
        provider="chatgpt",
        model=None,
        card_count=6,
        difficulty="balanced",
        cards=flashcards["deck"]["cards"],
    )

    assert flashcards["mode"] == "flashcards"
    assert flashcards["provider"] == "local"
    assert flashcards["generation_mode"] == "local_fallback"
    assert flashcards["focus"]["scope"] == "whole_video"
    assert flashcards["evidence_pack"]["section_count"] == 5
    assert flashcards["evidence_pack"]["selected_section_count"] == 5
    assert len(flashcards["deck"]["cards"]) == 6
    assert all(card["evidence"]["url"] for card in flashcards["deck"]["cards"])
    assert all(
        card["evidence"]["source_type"] == "study_section"
        for card in flashcards["deck"]["cards"]
    )
    assert all(card["card_type"] for card in flashcards["deck"]["cards"])
    assert all(card["learning_objective"] for card in flashcards["deck"]["cards"])
    assert all(card["why_it_matters"] for card in flashcards["deck"]["cards"])
    assert all(len(card["answer"]) <= 360 for card in flashcards["deck"]["cards"])
    assert not any(
        "What key point should you remember" in card["question"]
        for card in flashcards["deck"]["cards"]
    )
    assert not any(
        re.search(r"\b\d{1,2}:\d{2}(?::\d{2})?\b", card["question"])
        for card in flashcards["deck"]["cards"]
    )

    assert topics["mode"] == "topics"
    assert topics["provider"] == "local"
    assert topics["generation_mode"] == "section_cache"
    assert topics["evidence_pack"]["section_count"] == 5
    assert len(topics["topics"]) == 5
    assert all(topic["url"] for topic in topics["topics"])
    assert all(topic["key_points"] for topic in topics["topics"])
    assert not any(topic["title"].startswith("Topic ") for topic in topics["topics"])
    assert not any(
        "Evidence-backed topic from around" in topic["tldr"]
        for topic in topics["topics"]
    )

    assert quality["mode"] == "quality"
    assert quality["quality"]["metrics"]["cards_evaluated"] == 6
    assert quality["quality"]["metrics"]["citation_coverage"] == 1
    assert quality["quality"]["metrics"]["timestamp_coverage"] == 1
    assert quality["quality"]["metrics"]["specific_question_rate"] == 1
    assert quality["quality"]["metrics"]["timestamp_free_question_rate"] == 1
    assert quality["quality"]["metrics"]["concise_answer_rate"] == 1
    assert quality["quality"]["metrics"]["card_type_coverage"] == 1
    assert quality["quality"]["metrics"]["learning_objective_coverage"] == 1
    assert quality["quality"]["metrics"]["learner_value_coverage"] == 1
    assert quality["quality"]["verdict"] == "pass"


def test_generate_study_flashcards_use_requested_focus_without_timestamps(
    monkeypatch,
):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    service = _make_study_service()

    flashcards = service.generate_study_artifact(
        mode="flashcards",
        video_id="vidStudy001",
        language="en",
        provider="chatgpt",
        model=None,
        card_count=6,
        difficulty="balanced",
        focus="Serie character interpretation",
        focus_preset="characters",
        scope="focused_sections",
        model_profile="economy",
    )
    quality = service.generate_study_artifact(
        mode="quality",
        video_id="vidStudy001",
        language="en",
        provider="chatgpt",
        model=None,
        card_count=6,
        difficulty="balanced",
        cards=flashcards["deck"]["cards"],
        focus="Serie character interpretation",
        focus_preset="characters",
        scope="focused_sections",
        model_profile="economy",
    )

    assert flashcards["focus"]["query"] == "Serie character interpretation"
    assert flashcards["focus"]["preset"] == "characters"
    assert flashcards["focus"]["scope"] == "focused_sections"
    assert flashcards["focus"]["model_profile"] == "economy"
    assert flashcards["evidence_pack"]["selected_section_count"] >= 1
    assert "Serie" in flashcards["evidence_pack"]["selected_sections"][0]["title"]
    assert any(
        "Serie" in card["question"] or "Serie" in card["source_cue"]
        for card in flashcards["deck"]["cards"]
    )
    assert not any(
        re.search(r"\b\d{1,2}:\d{2}(?::\d{2})?\b", card["question"])
        for card in flashcards["deck"]["cards"]
    )
    assert all(
        card["retrieval_mode"] == "study_section_focus"
        for card in flashcards["deck"]["cards"]
    )
    assert quality["quality"]["metrics"]["focus_alignment_rate"] >= 0.6
    assert any(
        check["name"] == "Cards match requested focus" and check["status"] == "pass"
        for check in quality["quality"]["checks"]
    )


def test_study_flashcards_reuse_summary_sections_and_show_supported_speaker(
    monkeypatch,
):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    service = _make_study_service()
    video = service.engine.library.videos["vidStudy001"]
    video["title"] = (
        "The Stephen Curry Interview (Part 1) | LeBron James and Steve Nash"
    )

    before = service.generate_study_artifact(
        mode="flashcards",
        video_id="vidStudy001",
        language="en",
        provider="chatgpt",
        model=None,
        card_count=5,
        difficulty="balanced",
    )
    assert before["evidence_pack"]["basis"] == "deterministic"

    context = service._resolve_study_context("vidStudy001")
    summary_items = [
        {
            "rank": rank,
            "title": title,
            "tldr": tldr,
            "anchor_text": anchor,
            "start": float((rank - 1) * 60),
            "end": float((rank - 1) * 60 + 24),
            "url": (
                "https://www.youtube.com/watch?v=vidStudy001"
                f"&t={(rank - 1) * 60}s"
            ),
        }
        for rank, title, tldr, anchor in [
            (
                1,
                "Stephen Curry explains how early setbacks shaped his confidence",
                "Stephen Curry describes how setbacks became part of his preparation.",
                "The host opens the episode",
            ),
            (
                2,
                "Stephen Curry breaks down the discipline behind his shooting",
                "Stephen Curry connects repetition with trust in his mechanics.",
                "The first section frames the main idea",
            ),
            (
                3,
                "Stephen Curry reflects on leadership and team standards",
                "Stephen Curry explains how leaders reinforce habits for teammates.",
                "Serie is introduced as a character",
            ),
            (
                4,
                "Stephen Curry discusses adapting when defenses change",
                "Stephen Curry describes reading pressure and changing decisions.",
                "The acting section explains performance direction",
            ),
            (
                5,
                "Stephen Curry defines what keeps improvement sustainable",
                "Stephen Curry ties long-term growth to curiosity and consistent work.",
                "The closing section returns to favorite magic",
            ),
        ]
    ]
    service._put_summary_cache(
        video_id="vidStudy001",
        video=video,
        cache_key=service._summary_cache_key(
            language="en",
            provider="chatgpt",
            max_points=5,
            model="gpt-5.4-mini",
        ),
        source_fingerprint=context["source_fingerprint"],
        language="en",
        provider="chatgpt",
        max_points=5,
        model="gpt-5.4-mini",
        response_payload={"summary": summary_items},
    )

    after = service.generate_study_artifact(
        mode="flashcards",
        video_id="vidStudy001",
        language="en",
        provider="chatgpt",
        model=None,
        card_count=5,
        difficulty="balanced",
    )

    assert after["evidence_pack"]["basis"] == "summary_cache"
    assert after["evidence_pack"]["cache_status"] == "miss"
    assert all(
        not section["title"].startswith("Section ")
        for section in after["evidence_pack"]["selected_sections"]
    )
    assert all(
        card["speaker"] == "Stephen Curry"
        for card in after["deck"]["cards"]
    )
    assert all(
        card["speaker_role"] == "interview subject"
        for card in after["deck"]["cards"]
    )
    assert all(
        card["speaker_confidence"] == "named_in_section"
        for card in after["deck"]["cards"]
    )


def test_study_flashcards_keep_unknown_speaker_when_section_is_unattributed(
    monkeypatch,
):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    service = _make_study_service()

    flashcards = service.generate_study_artifact(
        mode="flashcards",
        video_id="vidStudy001",
        language="en",
        provider="chatgpt",
        model=None,
        card_count=4,
        difficulty="balanced",
    )

    assert all(
        card["speaker"] == "Unknown speaker"
        for card in flashcards["deck"]["cards"]
    )
    assert all(
        card["speaker_confidence"] == "unattributed"
        for card in flashcards["deck"]["cards"]
    )


def test_study_speaker_attribution_preserves_multi_person_interview_context():
    service = _make_study_service()

    attribution = service._study_speaker_attribution(
        value=(
            "Curry's gravity changed defenses and boosted teammates. "
            "The speakers explain how his movement creates open shots."
        ),
        episode_context={
            "primary_subject": "Stephen Curry",
            "participants_from_title": [
                "Stephen Curry",
                "LeBron James",
                "Steve Nash",
            ],
            "role_claims_from_intro": [],
        },
    )

    assert attribution == {
        "speaker": "Multiple speakers",
        "speaker_role": "Stephen Curry, LeBron James, Steve Nash",
        "speaker_confidence": "episode_context",
    }


def test_study_episode_context_extracts_people_but_not_show_brand():
    service = _make_study_service()
    context = {
        "video": {
            "title": (
                "The Kevin Durant Interview | LeBron James & Steve Nash | "
                "MIND THE GAME"
            ),
            "url": "https://www.youtube.com/watch?v=durant-demo",
        },
        "segments": [],
    }

    episode_context = service._study_episode_context_pack(context=context)

    assert episode_context["primary_subject"] == "Kevin Durant"
    assert episode_context["participants_from_title"] == [
        "Kevin Durant",
        "LeBron James",
        "Steve Nash",
    ]


def test_study_episode_context_excludes_title_case_show_brand():
    service = _make_study_service()
    context = {
        "video": {
            "title": "The Kevin Durant Interview | LeBron James | Mind The Game",
            "url": "https://www.youtube.com/watch?v=durant-demo",
        },
        "segments": [],
    }

    episode_context = service._study_episode_context_pack(context=context)

    assert episode_context["participants_from_title"] == [
        "Kevin Durant",
        "LeBron James",
    ]


def test_study_speaker_attribution_ignores_substring_name_collisions():
    service = _make_study_service()

    attribution = service._study_speaker_attribution(
        value=(
            "Steve Nash explains the pick and roll while breaking down "
            "Jamestown history."
        ),
        episode_context={
            "primary_subject": "Steve Nash",
            "participants_from_title": ["Steve Nash", "LeBron James"],
            "role_claims_from_intro": [],
        },
    )

    # "James" is only a substring of "Jamestown"; without word-boundary
    # matching it would falsely count as a second participant and blank out
    # the real speaker.
    assert attribution["speaker"] == "Steve Nash"
    assert attribution["speaker_confidence"] == "named_in_section"


def test_study_summary_sections_reject_transcriptless_video():
    service = _make_study_service()
    context = {
        "video_id": "empty-demo",
        "video": {"title": "Empty Demo", "chunks": []},
        "segments": [],
    }
    summary_bundle = {
        "summary": [
            {
                "rank": 1,
                "title": "A descriptive theme",
                "tldr": "Some summary text.",
                "start": 0.0,
                "end": 10.0,
            },
        ],
    }

    # No transcript rows survive text cleaning. The summary path must fail the
    # same explicit way as the deterministic path (via _study_evidence_rows)
    # rather than crashing on min() over an empty sequence.
    with pytest.raises(ValueError):
        service._study_summary_sections(
            context=context,
            language="en",
            summary_bundle=summary_bundle,
        )


def test_generate_study_flashcards_uses_chatgpt_when_key_is_configured(monkeypatch):
    monkeypatch.setattr(local_api, "OpenAI", object())
    monkeypatch.setenv("OPENAI_API_KEY", "openai-test-key")
    service = _make_study_service()
    captured = {}

    def fake_llm_flashcards(**kwargs):
        captured.update(kwargs)
        return {
            "provider": kwargs["provider"],
            "model": kwargs["model"],
            "cards": service._local_flashcards_from_evidence(
                evidence_rows=kwargs["evidence_rows"],
                card_count=kwargs["card_count"],
                language=kwargs["language"],
                difficulty=kwargs["difficulty"],
            ),
        }

    service._llm_flashcards_from_evidence = fake_llm_flashcards

    flashcards = service.generate_study_artifact(
        mode="flashcards",
        video_id="vidStudy001",
        language="en",
        provider="chatgpt",
        model="gpt-5.4-nano",
        card_count=4,
        difficulty="balanced",
        model_profile="economy",
    )

    assert captured["provider"] == "chatgpt"
    assert captured["model"] == "gpt-5.4-nano"
    assert flashcards["provider"] == "chatgpt"
    assert flashcards["model"] == "gpt-5.4-nano"
    assert flashcards["generation_mode"] == "llm"


def test_generate_study_flashcards_uses_sakana_when_key_is_configured(monkeypatch):
    monkeypatch.setenv("SAKANA_API_KEY", "sakana-test-key")
    service = _make_study_service()
    captured = {}

    def fake_llm_flashcards(**kwargs):
        captured.update(kwargs)
        return {
            "provider": kwargs["provider"],
            "model": kwargs["model"],
            "cards": service._local_flashcards_from_evidence(
                evidence_rows=kwargs["evidence_rows"],
                card_count=kwargs["card_count"],
                language=kwargs["language"],
                difficulty=kwargs["difficulty"],
            ),
        }

    service._llm_flashcards_from_evidence = fake_llm_flashcards

    flashcards = service.generate_study_artifact(
        mode="flashcards",
        video_id="vidStudy001",
        language="en",
        provider="sakana",
        model="fugu-ultra",
        card_count=4,
        difficulty="balanced",
        model_profile="economy",
    )

    assert captured["provider"] == "sakana"
    assert captured["model"] == "fugu-ultra"
    assert flashcards["provider"] == "sakana"
    assert flashcards["model"] == "fugu-ultra"
    assert flashcards["generation_mode"] == "llm"


def test_study_focus_tokenizer_supports_japanese_focus_terms():
    service = _make_study_service()
    focus_meta = service._study_focus_metadata(
        focus="",
        focus_preset="timeline",
        scope="focused_sections",
        model_profile="balanced",
    )

    assert "流れ" in service._study_focus_terms(focus_meta)
    assert "次" in service._study_focus_terms(focus_meta)
    assert any(
        "流れ" in token
        for token in local_api.LocalRAGService._study_search_tokens("話の流れ")
    )
    assert any(
        "ひらがな" in token
        for token in local_api.LocalRAGService._study_search_tokens("ひらがなの説明")
    )


def test_study_focused_sections_prefer_partial_query_matches_over_preset_only():
    service = _make_study_service()
    focus_meta = service._study_focus_metadata(
        focus="alpha beta",
        focus_preset="characters",
        scope="focused_sections",
        model_profile="balanced",
    )
    sections = [
        {
            "section_id": "section_1",
            "rank": 1,
            "title": "Alpha topic",
            "tldr": "This section explains alpha.",
            "key_points": [],
            "anchor_text": "",
            "keywords": ["alpha"],
            "section_type": "estimated",
        },
        {
            "section_id": "section_2",
            "rank": 2,
            "title": "Guest character setup",
            "tldr": "This section mentions guests and characters but not the query.",
            "key_points": [],
            "anchor_text": "",
            "keywords": ["guest", "character"],
            "section_type": "character",
        },
        {
            "section_id": "section_3",
            "rank": 3,
            "title": "Beta topic",
            "tldr": "This section explains beta.",
            "key_points": [],
            "anchor_text": "",
            "keywords": ["beta"],
            "section_type": "estimated",
        },
    ]

    selected = service._study_select_focus_sections(
        sections=sections,
        focus_meta=focus_meta,
        limit=2,
    )

    assert [section["title"] for section in selected] == [
        "Alpha topic",
        "Beta topic",
    ]


def test_study_evidence_rows_from_sections_pad_with_unique_review_ids():
    service = _make_study_service()
    rows = service._study_evidence_rows_from_sections(
        sections=[
            {
                "section_id": "section_1",
                "rank": 1,
                "title": "Serie focus",
                "tldr": "Serie is interpreted through power and loneliness.",
                "key_points": [],
                "anchor_text": "",
                "start": 120.0,
                "end": 150.0,
                "url": "https://www.youtube.com/watch?v=vidStudy001&t=120s",
                "evidence": {
                    "evidence_id": "base_1",
                    "start": 120.0,
                    "end": 150.0,
                },
            }
        ],
        max_rows=4,
        language="en",
        video_id="vidStudy001",
        video_title="Study Mode Demo",
    )

    assert len(rows) == 4
    assert len({row["evidence_id"] for row in rows}) == 4
    assert all(row["start"] == 120.0 for row in rows)


def test_study_quality_ignores_malformed_non_dict_cards(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    service = _make_study_service()
    flashcards = service.generate_study_artifact(
        mode="flashcards",
        video_id="vidStudy001",
        language="en",
        provider="chatgpt",
        model=None,
        card_count=4,
        difficulty="balanced",
    )

    quality = service.generate_study_artifact(
        mode="quality",
        video_id="vidStudy001",
        language="en",
        provider="chatgpt",
        model=None,
        card_count=4,
        difficulty="balanced",
        cards=[flashcards["deck"]["cards"][0], "malformed-card"],
    )

    assert quality["quality"]["metrics"]["cards_evaluated"] == 1
    assert quality["quality"]["metrics"]["citation_coverage"] == 1
    assert quality["quality"]["metrics"]["specific_question_rate"] == 1


def test_generate_study_topics_accepts_local_provider_without_llm_validation():
    service = _make_study_service()

    topics = service.generate_study_artifact(
        mode="topics",
        video_id="vidStudy001",
        language="en",
        provider="local",
        model=None,
        card_count=4,
        difficulty="balanced",
    )

    assert topics["provider"] == "local"
    assert topics["generation_mode"] == "section_cache"
    assert topics["topic_detail_level"] == "brief"
    assert topics["topics"]


def test_generate_study_topics_explain_uses_llm_when_key_is_configured(monkeypatch):
    monkeypatch.setattr(local_api, "OpenAI", object())
    monkeypatch.setenv("OPENAI_API_KEY", "openai-test-key")
    service = _make_study_service()
    captured = {}

    def fake_llm_explain_topic(**kwargs):
        captured.update(kwargs)
        topic = dict(kwargs["topics"][kwargs["topic_rank"] - 1])
        topic.update(
            {
                "topic_detail_level": "explain",
                "who_is_speaking": [
                    {
                        "name": "Guest",
                        "role": "Serie voice actor",
                        "confidence": "intro_text",
                        "evidence": "role claim from intro",
                    }
                ],
                "what_they_talked_about": (
                    "The speaker explains what makes the selected topic useful "
                    "for understanding the episode."
                ),
                "source_moments": [
                    {
                        "timestamp": "10:35",
                        "quote": "source quote",
                        "translation": "source translation",
                        "explanation": "why the source moment matters",
                    }
                ],
                "key_takeaways": ["A concrete takeaway from the evidence."],
                "learning_context": "It gives the learner a stronger review target.",
                "people_or_terms": ["Serie"],
                "review_questions": ["What should the learner remember here?"],
            }
        )
        return {
            "provider": kwargs["provider"],
            "model": kwargs["model"],
            "topics": [topic],
        }

    service._llm_explain_topic_from_section = fake_llm_explain_topic

    topics = service.generate_study_artifact(
        mode="topics",
        video_id="vidStudy001",
        language="en",
        provider="chatgpt",
        model="gpt-5.4-mini",
        card_count=4,
        difficulty="balanced",
        topic_detail_level="explain",
        topic_rank=3,
        model_profile="balanced",
    )

    assert captured["provider"] == "chatgpt"
    assert captured["model"] == "gpt-5.4-mini"
    assert captured["topic_rank"] == 3
    assert len(captured["selected_sections"]) == 5
    assert topics["provider"] == "chatgpt"
    assert topics["model"] == "gpt-5.4-mini"
    assert topics["generation_mode"] == "llm_topic_explain"
    assert topics["topic_detail_level"] == "explain"
    assert topics["topic_rank"] == 3
    assert len(topics["topics"]) == 1
    assert topics["topics"][0]["who_is_speaking"]
    assert topics["topics"][0]["source_moments"]


def test_study_topics_preserve_explicit_zero_end_timestamp():
    service = _make_study_service()

    topics = service._study_topics_from_sections(
        sections=[
            {
                "section_id": "section_zero",
                "rank": 1,
                "title": "Zero end section",
                "tldr": "Explicit zero end should not be replaced by start.",
                "key_points": [],
                "anchor_text": "zero",
                "start": 5.0,
                "end": 0.0,
                "url": "https://www.youtube.com/watch?v=vidStudy001&t=5s",
                "evidence": {},
            }
        ]
    )

    assert topics[0]["start"] == 5.0
    assert topics[0]["end"] == 0.0


def test_retrieve_rerank_happens_before_topk_cut():
    service = _make_service(enabled=True)

    service._dense_search = lambda query, k, language, video_id=None: [
        {
            "video_id": "vid1",
            "video_title": "Video 1",
            "start": float(i * 10),
            "end": float(i * 10 + 9),
            "chunk_index": i,
            "text": f"chunk {i} python list sort",
            "score": 0.70 - (0.01 * i),
            "rank": i + 1,
        }
        for i in range(10)
    ]
    service._lexical_bm25_search = lambda query, k, language, video_id=None: []

    service.save_search_feedback(
        {
            "query": "python list sort",
            "retrieval_mode": "dense",
            "label": "relevant",
            "video_id": "vid1",
            "chunk_index": 9,
            "start": 90.0,
            "end": 99.0,
        }
    )

    result = service.retrieve("python list sort", k=3, retrieval_mode="dense")
    chunk_ids = [row["chunk_index"] for row in result["results"]]

    assert 9 in chunk_ids
    assert result["details"]["pre_rerank_candidate_count"] == 10
    assert result["details"]["post_feedback_candidate_count"] == 10


def test_retrieve_applies_diversity_to_mix_videos():
    service = _make_service(enabled=False)

    dense_rows = [
        {
            "video_id": "vidA",
            "video_title": "Video A",
            "start": 0.0,
            "end": 10.0,
            "chunk_index": 0,
            "text": "python lists",
            "score": 0.95,
            "rank": 1,
        },
        {
            "video_id": "vidA",
            "video_title": "Video A",
            "start": 50.0,
            "end": 60.0,
            "chunk_index": 1,
            "text": "python tuples",
            "score": 0.94,
            "rank": 2,
        },
        {
            "video_id": "vidA",
            "video_title": "Video A",
            "start": 120.0,
            "end": 130.0,
            "chunk_index": 2,
            "text": "python dicts",
            "score": 0.93,
            "rank": 3,
        },
        {
            "video_id": "vidB",
            "video_title": "Video B",
            "start": 25.0,
            "end": 35.0,
            "chunk_index": 0,
            "text": "python generators",
            "score": 0.70,
            "rank": 4,
        },
    ]
    service._dense_search = lambda query, k, language, video_id=None: dense_rows
    service._lexical_bm25_search = lambda query, k, language, video_id=None: []

    result = service.retrieve("python", k=3, retrieval_mode="dense")
    video_ids = [row["video_id"] for row in result["results"]]

    assert "vidB" in video_ids
    assert result["details"]["diversity_applied"] is True
    assert result["details"]["selected_per_video_cap"] == 2


def test_retrieve_passes_video_id_filter_to_search_backends():
    service = _make_service(enabled=False)
    calls = {"dense": [], "lexical": []}

    def fake_dense(query, k, language, video_id=None):
        calls["dense"].append(
            {
                "query": query,
                "k": k,
                "language": language,
                "video_id": video_id,
            }
        )
        return [
            {
                "video_id": "vid1",
                "video_title": "Video 1",
                "start": 0.0,
                "end": 10.0,
                "chunk_index": 0,
                "text": "python lists",
                "score": 0.8,
                "rank": 1,
            }
        ]

    def fake_lexical(query, k, language, video_id=None):
        calls["lexical"].append(
            {
                "query": query,
                "k": k,
                "language": language,
                "video_id": video_id,
            }
        )
        return []

    service._dense_search = fake_dense
    service._lexical_bm25_search = fake_lexical
    service.engine = type(
        "DummyEngine",
        (),
        {"library": type("DummyLibrary", (), {"videos": {"vid1": {}}})()},
    )()

    result = service.retrieve(
        "python",
        k=2,
        retrieval_mode="dense",
        video_id="vid1",
    )

    assert result["details"]["video_id_filter"] == "vid1"
    assert calls["dense"][0]["video_id"] == "vid1"
    assert calls["lexical"] == []


def test_save_ask_history_persists_and_lists_records():
    class DummyLibrary:
        def __init__(self):
            self.videos = {
                "vid1": {
                    "title": "Video vid1",
                    "url": "https://www.youtube.com/watch?v=vid1",
                    "language": "en",
                }
            }

        def get_video_chunking_metadata(self, video_id):
            assert video_id == "vid1"
            return {"version": "time_v2_60s_15s"}

    service = _make_service(enabled=False)
    service.engine = type("DummyEngine", (), {"library": DummyLibrary()})()

    record = service.save_ask_history(
        {
            "video_id": "vid1",
            "question": "What is this video about?",
            "k": 5,
            "language": "en",
            "retrieval_mode": "hybrid",
            "provider": "chatgpt",
            "model": "gpt-4o-mini",
            "answer": "It explains Python basics.",
            "sources": [
                {
                    "video_id": "vid1",
                    "video_title": "Video vid1",
                    "video_url": "https://www.youtube.com/watch?v=vid1",
                    "language": "en",
                    "chunk_index": 0,
                    "text": "Python basics",
                    "start": 0.0,
                    "end": 30.0,
                    "url": "https://www.youtube.com/watch?v=vid1&t=0s",
                }
            ],
            "retrieval_details": {"fusion": "rrf"},
        }
    )

    listed = service.list_ask_history(video_id="vid1", limit=5)

    assert record["video_id"] == "vid1"
    assert record["chunking_version"] == "time_v2_60s_15s"
    assert record["source_fingerprint"]
    assert listed[0]["question"] == "What is this video about?"
    assert listed[0]["answer"] == "It explains Python basics."
    assert listed[0]["retrieval_details"]["fusion"] == "rrf"


def test_save_ask_history_trims_to_recent_limit():
    class DummyLibrary:
        def __init__(self):
            self.videos = {
                "vid1": {
                    "title": "Video vid1",
                    "url": "https://www.youtube.com/watch?v=vid1",
                    "language": "en",
                }
            }

        def get_video_chunking_metadata(self, video_id):
            assert video_id == "vid1"
            return {"version": "time_v2_60s_15s"}

    service = _make_service(enabled=False)
    service.engine = type("DummyEngine", (), {"library": DummyLibrary()})()

    for idx in range(25):
        service.save_ask_history(
            {
                "video_id": "vid1",
                "question": f"Question {idx}",
                "k": 5,
                "language": "en",
                "retrieval_mode": "hybrid",
                "provider": "chatgpt",
                "model": "gpt-4o-mini",
                "answer": f"Answer {idx}",
                "sources": [],
                "created_at": f"2026-01-{idx + 1:02d}T00:00:00+00:00",
            }
        )

    rows = service.list_ask_history(video_id="vid1", limit=50)

    assert len(rows) == local_api.ASK_HISTORY_LIMIT_PER_VIDEO
    assert rows[0]["question"] == "Question 24"
    assert rows[-1]["question"] == "Question 5"


def _make_answer_sources():
    return [
        {
            "video_id": "vid1",
            "video_title": "Demo Video",
            "video_url": "https://www.youtube.com/watch?v=vid1",
            "language": "en",
            "chunk_index": 2,
            "text": "The speaker says the product runs fully offline and stores runtime data locally.",
            "start": 65.0,
            "end": 88.0,
            "url": "https://www.youtube.com/watch?v=vid1&t=65s",
            "rank": 1,
            "score": 0.91,
            "dense_score": 0.87,
            "lexical_score": 6.3,
        },
        {
            "video_id": "vid2",
            "video_title": "Demo Video 2",
            "video_url": "https://www.youtube.com/watch?v=vid2",
            "language": "en",
            "chunk_index": 0,
            "text": "A later excerpt says the local preview is portfolio-friendly and optimized for explainability.",
            "start": 12.0,
            "end": 30.0,
            "url": "https://www.youtube.com/watch?v=vid2&t=12s",
            "rank": 2,
            "score": 0.84,
            "dense_score": 0.8,
            "lexical_score": 5.1,
        },
    ]


def test_ask_with_sources_returns_stable_grounded_answer_payload():
    service = _make_service(enabled=False)
    service._llm_text_response = lambda **kwargs: {
        "provider": "chatgpt",
        "model": "gpt-4o-mini",
        "text": json.dumps(
            {
                "status": "answered",
                "confidence": "high",
                "answer": "The app is designed to run locally and keep runtime data on the local filesystem [1]. It is also positioned as a portfolio-friendly, explainable demo experience [2].",
                "citations": [
                    {
                        "citation_id": 1,
                        "reason": "This excerpt states the app runs locally and stores local runtime data.",
                    },
                    {
                        "citation_id": 2,
                        "reason": "This excerpt frames the project as portfolio-friendly and explainable.",
                    },
                ],
                "warnings": [],
                "has_conflict": False,
            }
        ),
    }

    result = service.ask_with_sources(
        "Does the app run locally and present itself as a portfolio-friendly demo?",
        _make_answer_sources(),
        provider="chatgpt",
    )

    assert result["status"] == "answered"
    assert result["confidence"] == "medium"
    assert result["provider"] == "chatgpt"
    assert result["model"] == "gpt-4o-mini"
    assert result["sources"][0]["video_id"] == "vid1"
    assert result["retrieved_chunks"][0]["timestamp_range_label"] == "1:05-1:28"
    assert result["retrieved_chunks"][0]["url"].endswith("&t=65s")
    assert result["retrieved_chunks"][0]["video_url"].endswith("watch?v=vid1")
    assert result["citations"][0]["citation_id"] == 1
    assert result["citations"][0]["video_title"] == "Demo Video"
    assert result["citations"][0]["timestamp_label"] == "1:05"
    assert result["citations"][0]["timestamp_range_label"] == "1:05-1:28"
    assert result["citations"][0]["chunk_id"] == "vid1:2"
    assert result["citations"][0]["url"].endswith("&t=65s")
    assert "locally" in result["answer"]


def test_ask_with_sources_returns_insufficient_evidence_when_no_sources():
    service = _make_service(enabled=False)

    result = service.ask_with_sources(
        "What does the transcript say?",
        [],
        provider="chatgpt",
    )

    assert result["status"] == "insufficient_evidence"
    assert result["confidence"] == "low"
    assert result["citations"] == []
    assert result["retrieved_chunks"] == []
    assert result["sources"] == []
    assert "Insufficient transcript evidence" in result["answer"]


def test_ask_with_sources_refuses_weak_nonempty_evidence_without_llm_call():
    service = _make_service(enabled=False)
    llm_called = {"value": False}

    def _unexpected_llm_call(**kwargs):
        llm_called["value"] = True
        raise AssertionError("LLM should not be called for weak evidence.")

    service._llm_text_response = _unexpected_llm_call
    weak_sources = [
        {
            "video_id": "vid1",
            "video_title": "Demo Video",
            "video_url": "https://www.youtube.com/watch?v=vid1",
            "language": "en",
            "chunk_index": 9,
            "text": "This excerpt talks about local setup steps and startup logs.",
            "start": 140.0,
            "end": 168.0,
            "url": "https://www.youtube.com/watch?v=vid1&t=140s",
            "rank": 1,
            "score": 0.58,
            "dense_score": 0.58,
        }
    ]

    result = service.ask_with_sources(
        "What is the refund policy?",
        weak_sources,
        provider="chatgpt",
        retrieval_mode="dense",
    )

    assert llm_called["value"] is False
    assert result["status"] == "insufficient_evidence"
    assert result["confidence"] == "low"
    assert result["citations"] == []
    assert result["retrieved_chunks"][0]["url"].endswith("&t=140s")
    assert result["retrieved_chunks"][0]["video_url"].endswith("watch?v=vid1")
    assert any(
        "weak" in warning.lower() or "thin" in warning.lower()
        for warning in result["warnings"]
    )


def test_ask_with_sources_allows_strong_single_chunk_but_caps_confidence():
    service = _make_service(enabled=False)
    service._llm_text_response = lambda **kwargs: {
        "provider": "chatgpt",
        "model": "gpt-4o-mini",
        "text": json.dumps(
            {
                "status": "answered",
                "confidence": "high",
                "answer": "The product runs fully offline and stores runtime data locally [1].",
                "citations": [
                    {
                        "citation_id": 1,
                        "reason": "The excerpt directly states the offline/local runtime behavior.",
                    }
                ],
                "warnings": [],
                "has_conflict": False,
            }
        ),
    }
    single_source = [_make_answer_sources()[0]]

    result = service.ask_with_sources(
        "Does the product run fully offline and keep runtime data local?",
        single_source,
        provider="chatgpt",
        retrieval_mode="dense",
    )

    assert result["status"] == "answered"
    assert result["confidence"] == "medium"
    assert len(result["citations"]) == 1
    assert any("single excerpt" in warning.lower() for warning in result["warnings"])


def test_ask_with_sources_surfaces_conflicting_evidence_warning():
    service = _make_service(enabled=False)
    service._llm_text_response = lambda **kwargs: {
        "provider": "chatgpt",
        "model": "gpt-4o-mini",
        "text": json.dumps(
            {
                "status": "answered",
                "confidence": "medium",
                "answer": "The retrieved excerpts disagree about the rollout timing [1] [2].",
                "citations": [
                    {
                        "citation_id": 1,
                        "reason": "One excerpt says the rollout already happened.",
                    },
                    {
                        "citation_id": 2,
                        "reason": "Another excerpt says it is still upcoming.",
                    },
                ],
                "warnings": ["The timeline is inconsistent across retrieved chunks."],
                "has_conflict": True,
            }
        ),
    }

    result = service.ask_with_sources(
        "When did the rollout happen?",
        _make_answer_sources(),
        provider="chatgpt",
    )

    assert result["status"] == "answered"
    assert result["confidence"] == "low"
    assert any("conflicting" in warning.lower() for warning in result["warnings"])


def test_ask_with_sources_handles_provider_unavailable_gracefully():
    service = _make_service(enabled=False)

    def _raise_unconfigured_provider(**kwargs):
        raise ValueError("OPENAI_API_KEY environment variable is not set.")

    service._llm_text_response = _raise_unconfigured_provider

    result = service.ask_with_sources(
        "Does the product run fully offline and keep runtime data local?",
        _make_answer_sources(),
        provider="chatgpt",
    )

    assert result["status"] == "error"
    assert result["confidence"] == "low"
    assert result["citations"] == []
    assert result["retrieved_chunks"]
    assert "currently unavailable" in result["answer"]
    assert "OPENAI_API_KEY" in result["warnings"][0]


def test_answer_route_alias_returns_grounded_answer_payload(monkeypatch):
    saved_history = {}

    class StubService:
        def retrieve(
            self,
            query,
            k,
            language,
            retrieval_mode,
            retrieval_profile=None,
            video_id=None,
            reranker=None,
        ):
            assert reranker is None
            assert query == "What is the product intent?"
            assert k == 4
            assert retrieval_mode == "hybrid"
            assert retrieval_profile == "baseline_rrf"
            assert video_id == "vid1"
            return {
                "retrieval_mode": "hybrid",
                "details": {"fusion": "rrf"},
                "results": _make_answer_sources(),
            }

        def ask_with_sources(
            self, question, sources, provider, retrieval_mode="hybrid", model=None
        ):
            assert question == "What is the product intent?"
            assert provider == "chatgpt"
            assert len(sources) == 2
            assert retrieval_mode == "hybrid"
            assert model is None
            return {
                "status": "answered",
                "answer": "It is a local-first, explainable demo [1] [2].",
                "confidence": "high",
                "citations": [
                    {
                        "citation_id": 1,
                        "video_id": "vid1",
                        "video_title": "Demo Video",
                        "chunk_id": "vid1:2",
                        "start_seconds": 65.0,
                        "end_seconds": 88.0,
                        "timestamp_label": "1:05",
                        "timestamp_range_label": "1:05-1:28",
                        "snippet": "The speaker says the product runs fully offline...",
                        "reason": "States the local-first behavior.",
                        "url": "https://www.youtube.com/watch?v=vid1&t=65s",
                    }
                ],
                "retrieved_chunks": [
                    {
                        "video_id": "vid1",
                        "video_title": "Demo Video",
                        "chunk_index": 2,
                        "start_seconds": 65.0,
                        "end_seconds": 88.0,
                        "timestamp_label": "1:05",
                        "timestamp_range_label": "1:05-1:28",
                        "video_url": "https://www.youtube.com/watch?v=vid1",
                        "url": "https://www.youtube.com/watch?v=vid1&t=65s",
                        "score": 0.91,
                        "snippet": "The speaker says the product runs fully offline...",
                    }
                ],
                "warnings": [],
                "sources": sources,
                "provider": "chatgpt",
                "model": "gpt-4o-mini",
            }

        def save_ask_history(self, payload):
            saved_history.update(payload)
            return payload

    original_service = local_api.SERVICE
    try:
        local_api.SERVICE = StubService()

        class FakeHandler:
            def __init__(self, path, body):
                self.path = path
                self._body = body
                self.response_status = None
                self.response_payload = None

            def _read_json_body(self):
                return self._body

            def _json(self, payload, status=200):
                self.response_status = status
                self.response_payload = payload

        handler = FakeHandler(
            "/v1/answer",
            {
                "query": "What is the product intent?",
                "top_k": 4,
                "retrieval_mode": "hybrid",
                "retrieval_profile": "baseline_rrf",
                "provider": "chatgpt",
                "video_id": "vid1",
            },
        )
        local_api.Handler.do_POST(handler)
        payload = handler.response_payload
    finally:
        local_api.SERVICE = original_service

    assert handler.response_status == 200
    assert payload["ok"] is True
    assert payload["question"] == "What is the product intent?"
    assert payload["k"] == 4
    assert payload["status"] == "answered"
    assert payload["result_count"] == 2
    assert payload["citations"][0]["citation_id"] == 1
    assert payload["retrieval_details"]["fusion"] == "rrf"
    assert saved_history["status"] == "answered"
    assert saved_history["confidence"] == "high"
    assert saved_history["video_id"] == "vid1"


def test_search_route_plumbs_retrieval_profile():
    class StubService:
        def retrieve(
            self,
            query,
            k,
            language,
            retrieval_mode,
            retrieval_profile=None,
            video_id=None,
            reranker=None,
        ):
            assert reranker is None
            assert query == "semantic query"
            assert k == 3
            assert retrieval_mode == "hybrid"
            assert retrieval_profile == "optimized_v1"
            assert video_id is None
            return {
                "retrieval_mode": "hybrid",
                "details": {
                    "fusion": "weighted_normalized",
                    "fusion_profile": "optimized_v1",
                },
                "results": [],
            }

    original_service = local_api.SERVICE
    try:
        local_api.SERVICE = StubService()

        class FakeHandler:
            def __init__(self, path, body):
                self.path = path
                self._body = body
                self.response_status = None
                self.response_payload = None

            def _read_json_body(self):
                return self._body

            def _json(self, payload, status=200):
                self.response_status = status
                self.response_payload = payload

        handler = FakeHandler(
            "/v1/search",
            {
                "query": "semantic query",
                "k": 3,
                "retrieval_mode": "hybrid",
                "retrieval_profile": "optimized_v1",
            },
        )
        local_api.Handler.do_POST(handler)
        payload = handler.response_payload
    finally:
        local_api.SERVICE = original_service

    assert handler.response_status == 200
    assert payload["ok"] is True
    assert payload["retrieval_details"]["fusion_profile"] == "optimized_v1"


def test_summarize_video_transcript_returns_five_ranked_items():
    class DummyLibrary:
        def __init__(self):
            self.videos = {
                "vid1": {
                    "title": "Demo Video",
                    "full_transcript": {
                        "segments": [
                            {"text": "Intro", "start": 0.0, "end": 20.0},
                            {"text": "Main topic", "start": 20.0, "end": 60.0},
                            {"text": "Details", "start": 60.0, "end": 120.0},
                        ],
                        "text": "Intro\nMain topic\nDetails",
                    },
                    "chunks": [
                        {"raw_text": "Intro", "start": 0.0, "end": 20.0},
                        {"raw_text": "Main topic", "start": 20.0, "end": 60.0},
                        {"raw_text": "Details", "start": 60.0, "end": 120.0},
                    ],
                }
            }

    class DummyEngine:
        def __init__(self):
            self.library = DummyLibrary()
            self.model = "claude-sonnet-4-5-20250929"

    service = _make_service(enabled=False)
    service.engine = DummyEngine()
    service._summarize_transcript_single_pass = lambda **kwargs: {
        "provider": "chatgpt",
        "model": "gpt-4o-mini",
        "items": [
            {
                "title": "Point 1",
                "tldr": "A. B. C. D.",
                "anchor_text": "Intro",
                "start": 0.0,
                "end": 10.0,
            },
            {
                "title": "Point 2",
                "tldr": "A. B. C. D.",
                "anchor_text": "Main topic",
                "start": 15.0,
                "end": 25.0,
            },
            {
                "title": "Point 3",
                "tldr": "A. B. C. D.",
                "anchor_text": "Details",
                "start": 30.0,
                "end": 40.0,
            },
            {
                "title": "Point 4",
                "tldr": "A. B. C. D.",
                "anchor_text": "Intro",
                "start": 45.0,
                "end": 55.0,
            },
            {
                "title": "Point 5",
                "tldr": "A. B. C. D.",
                "anchor_text": "Main topic",
                "start": 60.0,
                "end": 70.0,
            },
        ],
        "strategy": "single_pass",
    }

    response = service.summarize_video_transcript(
        video_id="vid1",
        language="en",
        provider="chatgpt",
        max_points=5,
    )

    assert response["video_id"] == "vid1"
    assert response["language"] == "en"
    assert response["provider"] == "chatgpt"
    assert response["model"] == "gpt-4o-mini"
    assert len(response["summary"]) == 5
    assert response["summary"][0]["rank"] == 1
    assert response["summary"][4]["rank"] == 5
    assert response["generation_details"]["strategy"] == "single_pass"
    assert "youtube.com/watch?v=vid1&t=" in response["summary"][0]["url"]


def test_summarize_video_transcript_rejects_ten_points():
    class DummyLibrary:
        def __init__(self):
            self.videos = {
                "vid1": {
                    "title": "Demo Video",
                    "full_transcript": {
                        "segments": [
                            {"text": "Intro", "start": 0.0, "end": 20.0},
                            {"text": "Main topic", "start": 20.0, "end": 60.0},
                        ],
                        "text": "Intro\nMain topic",
                    },
                    "chunks": [
                        {"raw_text": "Intro", "start": 0.0, "end": 20.0},
                        {"raw_text": "Main topic", "start": 20.0, "end": 60.0},
                    ],
                }
            }

    class DummyEngine:
        def __init__(self):
            self.library = DummyLibrary()
            self.model = "claude-sonnet-4-5-20250929"

    service = _make_service(enabled=False)
    service.engine = DummyEngine()
    with pytest.raises(ValueError, match="max_points must be 5"):
        service.summarize_video_transcript(
            video_id="vid1",
            language="en",
            provider="chatgpt",
            max_points=10,
        )


def test_summarize_video_transcript_rejects_invalid_max_points():
    class DummyLibrary:
        def __init__(self):
            self.videos = {
                "vid1": {
                    "title": "Demo Video",
                    "full_transcript": {
                        "segments": [
                            {"text": "Intro", "start": 0.0, "end": 10.0},
                        ],
                        "text": "Intro",
                    },
                    "chunks": [
                        {"raw_text": "Intro", "start": 0.0, "end": 10.0},
                    ],
                }
            }

    class DummyEngine:
        def __init__(self):
            self.library = DummyLibrary()
            self.model = "claude-sonnet-4-5-20250929"

    service = _make_service(enabled=False)
    service.engine = DummyEngine()

    with pytest.raises(ValueError, match="max_points must be 5"):
        service.summarize_video_transcript(
            video_id="vid1",
            language="en",
            provider="chatgpt",
            max_points=7,
        )


def test_summarize_video_transcript_rejects_invalid_language():
    class DummyLibrary:
        def __init__(self):
            self.videos = {
                "vid1": {
                    "title": "Demo Video",
                    "full_transcript": {
                        "segments": [
                            {"text": "Hello", "start": 0.0, "end": 10.0},
                        ],
                        "text": "Hello",
                    },
                    "chunks": [
                        {"raw_text": "Hello", "start": 0.0, "end": 10.0},
                    ],
                }
            }

    class DummyEngine:
        def __init__(self):
            self.library = DummyLibrary()
            self.model = "claude-sonnet-4-5-20250929"

    service = _make_service(enabled=False)
    service.engine = DummyEngine()

    with pytest.raises(ValueError, match="language must be one of: en, ja"):
        service.summarize_video_transcript(
            video_id="vid1",
            language="fr",
            provider="chatgpt",
            max_points=5,
        )


def test_summarize_long_transcript_uses_compact_single_pass():
    class DummyLibrary:
        def __init__(self):
            self.videos = {
                "vid1": {
                    "title": "Long Video",
                    "full_transcript": {
                        "segments": [
                            {
                                "text": ("long transcript block " * 40).strip(),
                                "start": float(idx * 10),
                                "end": float(idx * 10 + 9),
                            }
                            for idx in range(120)
                        ],
                        "text": "\n".join(
                            ("long transcript block " * 40).strip() for _ in range(120)
                        ),
                    },
                    "chunks": [
                        {
                            "raw_text": ("long transcript block " * 40).strip(),
                            "start": float(idx * 10),
                            "end": float(idx * 10 + 9),
                        }
                        for idx in range(120)
                    ],
                }
            }

    class DummyEngine:
        def __init__(self):
            self.library = DummyLibrary()
            self.model = "claude-sonnet-4-5-20250929"

    service = _make_service(enabled=False)
    service.engine = DummyEngine()

    called = {"compact": False}

    def fake_compact_single_pass(**kwargs):
        called["compact"] = True
        return {
            "provider": "chatgpt",
            "model": "gpt-4o-mini",
            "items": [
                {
                    "title": "Theme 1",
                    "tldr": "A. B. C. D.",
                    "anchor_text": "long transcript block",
                    "start": 0.0,
                    "end": 0.0,
                },
                {
                    "title": "Theme 2",
                    "tldr": "A. B. C. D.",
                    "anchor_text": "long transcript block",
                    "start": 20.0,
                    "end": 20.0,
                },
                {
                    "title": "Theme 3",
                    "tldr": "A. B. C. D.",
                    "anchor_text": "long transcript block",
                    "start": 40.0,
                    "end": 40.0,
                },
                {
                    "title": "Theme 4",
                    "tldr": "A. B. C. D.",
                    "anchor_text": "long transcript block",
                    "start": 60.0,
                    "end": 60.0,
                },
                {
                    "title": "Theme 5",
                    "tldr": "A. B. C. D.",
                    "anchor_text": "long transcript block",
                    "start": 80.0,
                    "end": 80.0,
                },
            ],
            "strategy": "compact_single_pass",
            "total_windows": 1,
            "processed_windows": 1,
            "retry_count": 0,
            "attempt_count": 1,
        }

    service._summarize_transcript_compact_single_pass = fake_compact_single_pass
    response = service.summarize_video_transcript(
        video_id="vid1",
        language="en",
        provider="chatgpt",
        max_points=5,
    )

    assert called["compact"] is True
    assert response["generation_details"]["strategy"] == "compact_single_pass"
    assert len(response["summary"]) == 5


def test_summarize_long_transcript_relaxes_after_compact_and_reduce_failures():
    class DummyLibrary:
        def __init__(self):
            self.videos = {
                "vid1": {
                    "title": "Long Video",
                    "full_transcript": {
                        "segments": [
                            {
                                "text": ("long transcript block " * 40).strip(),
                                "start": float(idx * 10),
                                "end": float(idx * 10 + 9),
                            }
                            for idx in range(120)
                        ],
                        "text": "\n".join(
                            ("long transcript block " * 40).strip() for _ in range(120)
                        ),
                    },
                    "chunks": [
                        {
                            "raw_text": ("long transcript block " * 40).strip(),
                            "start": float(idx * 10),
                            "end": float(idx * 10 + 9),
                        }
                        for idx in range(120)
                    ],
                }
            }

    class DummyEngine:
        def __init__(self):
            self.library = DummyLibrary()
            self.model = "claude-sonnet-4-5-20250929"

    service = _make_service(enabled=False)
    service.engine = DummyEngine()
    call_counts = {"compact": 0, "reduce": 0}

    def fake_compact_single_pass(**kwargs):
        del kwargs
        call_counts["compact"] += 1
        raise local_api.SummaryGenerationError(
            "compact single-pass theme generation failed after retries"
        )

    def fake_map_reduce(**kwargs):
        call_counts["reduce"] += 1
        min_sentences = kwargs.get("min_sentences", local_api.SUMMARY_MIN_SENTENCES)
        max_sentences = kwargs.get("max_sentences", local_api.SUMMARY_MAX_SENTENCES)

        if (
            min_sentences == local_api.SUMMARY_MIN_SENTENCES
            and max_sentences == local_api.SUMMARY_MAX_SENTENCES
        ):
            raise local_api.SummaryGenerationError(
                "reduce stage summary generation failed after retries"
            )

        return {
            "provider": "chatgpt",
            "model": "gpt-4o-mini",
            "items": [
                {
                    "title": f"Theme {idx}",
                    "tldr": "One. Two. Three.",
                    "anchor_text": "long transcript block",
                    "start": float((idx - 1) * 20),
                    "end": float((idx - 1) * 20 + 10),
                }
                for idx in range(1, 6)
            ],
            "strategy": "map_reduce",
            "total_windows": 3,
            "processed_windows": 3,
            "retry_count": 2,
        }

    service._summarize_transcript_compact_single_pass = fake_compact_single_pass
    service._summarize_transcript_map_reduce = fake_map_reduce

    response = service.summarize_video_transcript(
        video_id="vid1",
        language="en",
        provider="chatgpt",
        max_points=5,
    )

    assert call_counts["compact"] == 1
    assert call_counts["reduce"] == 2
    assert len(response["summary"]) == 5
    assert response["generation_details"]["primary_strategy"] == "compact_single_pass"
    assert response["generation_details"]["strategy"] == "map_reduce"
    assert response["generation_details"]["fallback_applied"] is True
    assert response["generation_details"]["validation_relaxed"] is True
    assert "reduce stage failed" in str(
        response["generation_details"]["fallback_reason"]
    )


def test_summarize_single_pass_retries_then_fails():
    class DummyLibrary:
        def __init__(self):
            self.videos = {}

    class DummyEngine:
        def __init__(self):
            self.library = DummyLibrary()
            self.model = "claude-sonnet-4-5-20250929"

    service = _make_service(enabled=False)
    service.engine = DummyEngine()

    attempt_counter = {"count": 0}

    def failing_llm(**kwargs):
        attempt_counter["count"] += 1
        return {
            "provider": "chatgpt",
            "model": "gpt-4o-mini",
            "text": "not-json-output",
        }

    service._llm_text_response = failing_llm
    chunks = [{"raw_text": "hello", "start": 0.0, "end": 8.0}]
    lines = service._summary_transcript_lines(chunks)

    with pytest.raises(
        local_api.SummaryGenerationError, match="single-pass summary generation failed"
    ):
        service._summarize_transcript_single_pass(
            transcript_lines=lines,
            segments=chunks,
            language="en",
            provider="chatgpt",
            max_points=5,
        )

    assert attempt_counter["count"] == local_api.SUMMARY_RETRY_ATTEMPTS


def test_normalize_summary_items_does_not_fallback_to_chunks():
    service = _make_service(enabled=False)

    items = service._normalize_summary_items(
        items={"items": []},
        segments=[
            {"raw_text": "chunk A", "start": 0.0, "end": 10.0},
            {"raw_text": "chunk B", "start": 12.0, "end": 22.0},
        ],
        language="en",
        max_points=5,
    )

    assert items == []


def test_normalize_summary_items_preserves_importance_order():
    service = _make_service(enabled=False)
    segments = [
        {"raw_text": "first", "start": 0.0, "end": 10.0},
        {"raw_text": "second", "start": 10.0, "end": 300.0},
    ]

    items = service._normalize_summary_items(
        items={
            "items": [
                {"title": "Most Important", "tldr": "theme A", "start": 120.0},
                {"title": "Second", "tldr": "theme B", "start": 10.0},
                {"title": "Third", "tldr": "theme C", "start": 30.0},
                {"title": "Fourth", "tldr": "theme D", "start": 40.0},
                {"title": "Fifth", "tldr": "theme E", "start": 50.0},
            ]
        },
        segments=segments,
        language="en",
        max_points=5,
    )

    assert [row["title"] for row in items] == [
        "Most Important",
        "Second",
        "Third",
        "Fourth",
        "Fifth",
    ]


def test_summary_items_match_language_detects_non_english_output():
    service = _make_service(enabled=False)
    assert (
        service._summary_items_match_language(
            [
                {"title": "Intro", "tldr": "Overview of the video"},
                {"title": "Topic 1", "tldr": "Key argument and examples"},
            ],
            "en",
        )
        is True
    )
    assert (
        service._summary_items_match_language(
            [
                {"title": "イントロ", "tldr": "これは日本語です"},
                {"title": "Topic 1", "tldr": "Still mixed"},
            ],
            "en",
        )
        is False
    )


def test_summary_copy_detector_flags_verbatim_chunk_copy():
    service = _make_service(enabled=False)
    chunks = [
        {
            "raw_text": "This is a long transcript line with very specific wording that should not be copied directly."
        },
    ]
    assert (
        service._summary_looks_like_transcript_copy(
            "This is a long transcript line with very specific wording that should not be copied directly.",
            chunks,
        )
        is True
    )


def test_summary_sentence_count_validator_requires_four_to_five_sentences():
    service = _make_service(enabled=False)
    assert (
        service._summary_items_have_required_sentence_count(
            [{"tldr": "One. Two. Three. Four."}],
            "en",
        )
        is True
    )
    assert (
        service._summary_items_have_required_sentence_count(
            [{"tldr": "One. Two. Three."}],
            "en",
        )
        is False
    )


def test_resolve_theme_anchor_timestamp_prefers_anchor_match():
    service = _make_service(enabled=False)
    segments = [
        {"text": "intro setup", "start": 0.0, "end": 8.0},
        {"text": "important anchor appears here", "start": 90.0, "end": 108.0},
    ]
    resolved = service._resolve_theme_anchor_timestamp(
        item={"anchor_text": "anchor appears here", "start": 2.0},
        segments=segments,
    )
    assert resolved["source"] == "anchor_match"
    assert resolved["start"] == 90.0


def test_resolve_theme_anchor_timestamp_uses_model_start_for_ambiguous_anchor():
    service = _make_service(enabled=False)
    segments = [
        {"text": "repeated anchor text", "start": 10.0, "end": 20.0},
        {"text": "some middle section", "start": 40.0, "end": 50.0},
        {"text": "repeated anchor text again", "start": 90.0, "end": 100.0},
    ]

    resolved = service._resolve_theme_anchor_timestamp(
        item={"anchor_text": "repeated anchor text", "start": 95.0},
        segments=segments,
    )

    assert resolved["source"] == "anchor_match"
    assert resolved["start"] == 90.0


def test_resolve_theme_anchor_timestamp_penalizes_reused_segment():
    service = _make_service(enabled=False)
    segments = [
        {"text": "anchor phrase", "start": 10.0, "end": 20.0},
        {"text": "anchor phrase", "start": 80.0, "end": 90.0},
    ]

    resolved = service._resolve_theme_anchor_timestamp(
        item={"anchor_text": "anchor phrase", "start": 0.0},
        segments=segments,
        used_segment_counts={0: 2},
    )

    assert resolved["source"] == "anchor_match"
    assert resolved["start"] == 80.0


def test_summarize_video_transcript_backfills_full_transcript_once():
    class DummyLibrary:
        def __init__(self):
            self.videos = {
                "vid1": {
                    "title": "Legacy Video",
                    "chunks": [
                        {"raw_text": "Intro section", "start": 0.0, "end": 12.0},
                        {"raw_text": "Core explanation", "start": 30.0, "end": 42.0},
                    ],
                }
            }
            self.save_calls = 0

        def save(self):
            self.save_calls += 1

    class DummyEngine:
        def __init__(self):
            self.library = DummyLibrary()
            self.model = "claude-sonnet-4-5-20250929"

    service = _make_service(enabled=False)
    service.engine = DummyEngine()
    service._summarize_transcript_single_pass = lambda **kwargs: {
        "provider": "chatgpt",
        "model": "gpt-4o-mini",
        "items": [
            {
                "title": "Theme 1",
                "tldr": "One. Two. Three. Four.",
                "anchor_text": "Intro section",
            },
            {
                "title": "Theme 2",
                "tldr": "One. Two. Three. Four.",
                "anchor_text": "Core explanation",
            },
            {
                "title": "Theme 3",
                "tldr": "One. Two. Three. Four.",
                "anchor_text": "Intro section",
            },
            {
                "title": "Theme 4",
                "tldr": "One. Two. Three. Four.",
                "anchor_text": "Core explanation",
            },
            {
                "title": "Theme 5",
                "tldr": "One. Two. Three. Four.",
                "anchor_text": "Intro section",
            },
        ],
        "strategy": "single_pass",
    }

    first = service.summarize_video_transcript(
        video_id="vid1", language="en", provider="chatgpt"
    )
    second = service.summarize_video_transcript(
        video_id="vid1", language="en", provider="chatgpt"
    )

    assert first["generation_details"]["full_transcript_backfilled"] is True
    assert second["generation_details"]["full_transcript_backfilled"] is False
    assert service.engine.library.save_calls == 1
    assert "full_transcript" in service.engine.library.videos["vid1"]
    assert service._summary_cache_path("vid1").exists()


def test_summarize_video_transcript_uses_summary_cache_for_repeat_request():
    class DummyLibrary:
        def __init__(self):
            self.videos = {
                "vid1": {
                    "title": "Cached Video",
                    "full_transcript": {
                        "segments": [
                            {"text": "Intro section", "start": 0.0, "end": 12.0},
                            {"text": "Core explanation", "start": 30.0, "end": 42.0},
                        ],
                        "text": "Intro section\nCore explanation",
                        "segment_count": 2,
                    },
                    "chunks": [
                        {"raw_text": "Intro section", "start": 0.0, "end": 12.0},
                        {"raw_text": "Core explanation", "start": 30.0, "end": 42.0},
                    ],
                }
            }
            self.save_calls = 0

        def save(self):
            self.save_calls += 1

    class DummyEngine:
        def __init__(self):
            self.library = DummyLibrary()
            self.model = "claude-sonnet-4-5-20250929"

    service = _make_service(enabled=False)
    service.engine = DummyEngine()
    counter = {"count": 0}

    def fake_single_pass(**kwargs):
        counter["count"] += 1
        return {
            "provider": "chatgpt",
            "model": "gpt-4o-mini",
            "items": [
                {
                    "title": "Theme 1",
                    "tldr": "One. Two. Three. Four.",
                    "anchor_text": "Intro section",
                },
                {
                    "title": "Theme 2",
                    "tldr": "One. Two. Three. Four.",
                    "anchor_text": "Core explanation",
                },
                {
                    "title": "Theme 3",
                    "tldr": "One. Two. Three. Four.",
                    "anchor_text": "Intro section",
                },
                {
                    "title": "Theme 4",
                    "tldr": "One. Two. Three. Four.",
                    "anchor_text": "Core explanation",
                },
                {
                    "title": "Theme 5",
                    "tldr": "One. Two. Three. Four.",
                    "anchor_text": "Intro section",
                },
            ],
            "strategy": "single_pass",
        }

    service._summarize_transcript_single_pass = fake_single_pass

    first = service.summarize_video_transcript(
        video_id="vid1", language="en", provider="chatgpt"
    )
    second = service.summarize_video_transcript(
        video_id="vid1", language="en", provider="chatgpt"
    )

    assert first["cached"] is False
    assert second["cached"] is True
    assert first["generation_details"]["cache_hit"] is False
    assert second["generation_details"]["cache_hit"] is True
    assert second["generation_details"]["cache_generated_at"] is not None
    assert counter["count"] == 1
    assert service.engine.library.save_calls == 0
    assert service._summary_cache_path("vid1").exists()


def test_summarize_video_transcript_invalidates_cache_when_transcript_changes():
    class DummyLibrary:
        def __init__(self):
            self.videos = {
                "vid1": {
                    "title": "Cache Invalidation Video",
                    "full_transcript": {
                        "segments": [
                            {"text": "Intro section", "start": 0.0, "end": 12.0},
                            {"text": "Core explanation", "start": 30.0, "end": 42.0},
                        ],
                        "text": "Intro section\nCore explanation",
                        "segment_count": 2,
                    },
                    "chunks": [
                        {"raw_text": "Intro section", "start": 0.0, "end": 12.0},
                        {"raw_text": "Core explanation", "start": 30.0, "end": 42.0},
                    ],
                }
            }
            self.save_calls = 0

        def save(self):
            self.save_calls += 1

    class DummyEngine:
        def __init__(self):
            self.library = DummyLibrary()
            self.model = "claude-sonnet-4-5-20250929"

    service = _make_service(enabled=False)
    service.engine = DummyEngine()
    counter = {"count": 0}

    def fake_single_pass(**kwargs):
        counter["count"] += 1
        return {
            "provider": "chatgpt",
            "model": "gpt-4o-mini",
            "items": [
                {
                    "title": "Theme 1",
                    "tldr": "One. Two. Three. Four.",
                    "anchor_text": "Intro section",
                },
                {
                    "title": "Theme 2",
                    "tldr": "One. Two. Three. Four.",
                    "anchor_text": "Core explanation",
                },
                {
                    "title": "Theme 3",
                    "tldr": "One. Two. Three. Four.",
                    "anchor_text": "Intro section",
                },
                {
                    "title": "Theme 4",
                    "tldr": "One. Two. Three. Four.",
                    "anchor_text": "Core explanation",
                },
                {
                    "title": "Theme 5",
                    "tldr": "One. Two. Three. Four.",
                    "anchor_text": "Intro section",
                },
            ],
            "strategy": "single_pass",
        }

    service._summarize_transcript_single_pass = fake_single_pass

    first = service.summarize_video_transcript(
        video_id="vid1", language="en", provider="chatgpt"
    )
    service.engine.library.videos["vid1"]["full_transcript"]["segments"][0][
        "text"
    ] = "Updated intro section"
    second = service.summarize_video_transcript(
        video_id="vid1", language="en", provider="chatgpt"
    )

    assert first["cached"] is False
    assert second["cached"] is False
    assert second["generation_details"]["cache_hit"] is False
    assert counter["count"] == 2
    assert service.engine.library.save_calls == 0


def test_summarize_video_transcript_cache_key_partitions_requests():
    class DummyLibrary:
        def __init__(self):
            self.videos = {
                "vid1": {
                    "title": "Cache Key Video",
                    "full_transcript": {
                        "segments": [
                            {"text": "Intro section", "start": 0.0, "end": 12.0},
                            {"text": "Core explanation", "start": 30.0, "end": 42.0},
                        ],
                        "text": "Intro section\nCore explanation",
                        "segment_count": 2,
                    },
                    "chunks": [
                        {"raw_text": "Intro section", "start": 0.0, "end": 12.0},
                        {"raw_text": "Core explanation", "start": 30.0, "end": 42.0},
                    ],
                }
            }
            self.save_calls = 0

        def save(self):
            self.save_calls += 1

    class DummyEngine:
        def __init__(self):
            self.library = DummyLibrary()
            self.model = "claude-sonnet-4-5-20250929"

    service = _make_service(enabled=False)
    service.engine = DummyEngine()
    counter = {"count": 0}

    def fake_single_pass(**kwargs):
        counter["count"] += 1
        provider = kwargs.get("provider", "chatgpt")
        return {
            "provider": provider,
            "model": "gpt-4o-mini",
            "items": [
                {
                    "title": "Theme 1",
                    "tldr": "One. Two. Three. Four.",
                    "anchor_text": "Intro section",
                },
                {
                    "title": "Theme 2",
                    "tldr": "One. Two. Three. Four.",
                    "anchor_text": "Core explanation",
                },
                {
                    "title": "Theme 3",
                    "tldr": "One. Two. Three. Four.",
                    "anchor_text": "Intro section",
                },
                {
                    "title": "Theme 4",
                    "tldr": "One. Two. Three. Four.",
                    "anchor_text": "Core explanation",
                },
                {
                    "title": "Theme 5",
                    "tldr": "One. Two. Three. Four.",
                    "anchor_text": "Intro section",
                },
            ],
            "strategy": "single_pass",
        }

    service._summarize_transcript_single_pass = fake_single_pass

    first = service.summarize_video_transcript(
        video_id="vid1", language="en", provider="chatgpt"
    )
    second = service.summarize_video_transcript(
        video_id="vid1", language="en", provider="chatgpt"
    )
    third = service.summarize_video_transcript(
        video_id="vid1", language="ja", provider="chatgpt"
    )
    fourth = service.summarize_video_transcript(
        video_id="vid1", language="en", provider="claude"
    )
    assert first["cached"] is False
    assert second["cached"] is True
    assert third["cached"] is False
    assert fourth["cached"] is False
    assert counter["count"] == 3


def test_summary_prompt_requests_descriptive_speaker_aware_themes():
    service = _make_service(enabled=False)

    prompt = service._summary_system_prompt(language="en", max_points=5)

    assert "Make each theme title concrete and descriptive" in prompt
    assert "who is speaking" in prompt
    assert "never guess" in prompt


def test_ingest_persists_library_and_initializes_summary_cache():
    class DummyLibrary:
        def __init__(self):
            self.videos = {}
            self.save_calls = 0

        def add_video(self, video_id, language="ja"):
            self.videos[video_id] = {
                "title": f"Video {video_id}",
                "language": language,
                "chunks": [{"raw_text": "hello", "start": 0.0, "end": 8.0}],
                "full_transcript": {
                    "segments": [{"text": "hello", "start": 0.0, "end": 8.0}],
                    "text": "hello",
                    "segment_count": 1,
                },
            }
            return video_id

        def get_video_chunking_metadata(self, video_id):
            del video_id
            return {"version": "time_v2_60s_15s"}

        def video_chunking_is_stale(self, video_id):
            del video_id
            return False

        def save(self):
            self.save_calls += 1

    class DummyEngine:
        def __init__(self):
            self.library = DummyLibrary()
            self.model = "claude-sonnet-4-5-20250929"

    service = _make_service(enabled=False)
    service.engine = DummyEngine()
    service.lock = threading.Lock()
    service.jobs = {}
    service.title_cache = {}
    service._append_ingest_log = lambda **kwargs: None
    service._hydrate_video_title = lambda video_id: None

    result = service.ingest(
        url="https://www.youtube.com/watch?v=abc12345678",
        mode="single",
        language="en",
        force=False,
    )

    assert result["queued_count"] == 1
    assert service.engine.library.save_calls == 1
    assert service.engine.library.videos["abc12345678"].get("summary_cache") is None


def test_ingest_reingests_stale_video_without_force():
    class DummyLibrary:
        def __init__(self):
            self.videos = {
                "abc12345678": {
                    "title": "Old Video",
                    "language": "en",
                    "chunks": [{"raw_text": "old", "start": 0.0, "end": 45.0}],
                    "chunking": {"version": "time_v1_45s_15s"},
                }
            }
            self.save_calls = 0
            self.add_calls = []

        def add_video(self, video_id, language="ja"):
            self.add_calls.append((video_id, language))
            self.videos[video_id] = {
                "title": f"Video {video_id}",
                "language": language,
                "chunks": [{"raw_text": "new", "start": 0.0, "end": 60.0}],
                "full_transcript": {
                    "segments": [{"text": "new", "start": 0.0, "end": 60.0}],
                    "text": "new",
                    "segment_count": 1,
                },
                "chunking": {"version": "time_v2_60s_15s"},
            }
            return video_id

        def get_video_chunking_metadata(self, video_id):
            return self.videos[video_id].get("chunking", {"version": "unknown"})

        def video_chunking_is_stale(self, video_id):
            return (
                self.get_video_chunking_metadata(video_id).get("version")
                != "time_v2_60s_15s"
            )

        def save(self):
            self.save_calls += 1

    class DummyEngine:
        def __init__(self):
            self.library = DummyLibrary()
            self.model = "claude-sonnet-4-5-20250929"

    service = _make_service(enabled=False)
    service.engine = DummyEngine()
    service.lock = threading.Lock()
    service.jobs = {}
    service.title_cache = {}
    service._append_ingest_log = lambda **kwargs: None
    service._hydrate_video_title = lambda video_id: None

    result = service.ingest(
        url="https://www.youtube.com/watch?v=abc12345678",
        mode="single",
        language="en",
        force=False,
    )

    assert result["queued_count"] == 1
    assert result["skipped_count"] == 0
    assert service.engine.library.add_calls == [("abc12345678", "en")]
    assert service.engine.library.save_calls == 1
    assert (
        service.engine.library.videos["abc12345678"]["chunking"]["version"]
        == "time_v2_60s_15s"
    )


class _ModelSelectionEngine:
    def __init__(self, model="claude-sonnet-4-5-20250929"):
        self.model = model


def test_resolve_llm_model_defaults_to_provider_default():
    service = _make_service(enabled=False)
    service.engine = _ModelSelectionEngine()

    assert service._resolve_llm_model("chatgpt", None) == "gpt-5.4-mini"
    assert service._resolve_llm_model("chatgpt", "") == "gpt-5.4-mini"
    assert service._resolve_llm_model("claude", None) == "claude-sonnet-4-5-20250929"
    assert service._resolve_llm_model("sakana", None) == "fugu"


def test_resolve_llm_model_accepts_listed_and_default_models():
    service = _make_service(enabled=False)
    service.engine = _ModelSelectionEngine()

    assert service._resolve_llm_model("chatgpt", "gpt-5.4-mini") == "gpt-5.4-mini"
    assert service._resolve_llm_model("claude", "claude-opus-4-8") == "claude-opus-4-8"
    assert service._resolve_llm_model("sakana", "fugu-ultra") == "fugu-ultra"
    assert (
        service._resolve_llm_model("sakana", "fugu-ultra-20260615")
        == "fugu-ultra-20260615"
    )
    assert (
        service._resolve_llm_model("claude", "claude-sonnet-4-5-20250929")
        == "claude-sonnet-4-5-20250929"
    )
    # The env-configured default stays valid even when it is not in the list.
    service.openai_model = "gpt-4o-mini"
    assert service._resolve_llm_model("chatgpt", "gpt-4o-mini") == "gpt-4o-mini"


def test_resolve_llm_model_rejects_unknown_model():
    service = _make_service(enabled=False)
    service.engine = _ModelSelectionEngine()

    with pytest.raises(ValueError, match="model for provider 'chatgpt'"):
        service._resolve_llm_model("chatgpt", "gpt-3.5-turbo")
    with pytest.raises(ValueError, match="model for provider 'claude'"):
        service._resolve_llm_model("claude", "claude-2.1")
    with pytest.raises(ValueError, match="model for provider 'sakana'"):
        service._resolve_llm_model("sakana", "marlin")


def test_model_supports_temperature_gating():
    assert local_api._model_supports_temperature("claude", "claude-sonnet-4-6")
    assert local_api._model_supports_temperature("claude", "claude-haiku-4-5")
    assert not local_api._model_supports_temperature("claude", "claude-opus-4-8")
    assert local_api._model_supports_temperature("chatgpt", "gpt-4o-mini")
    assert not local_api._model_supports_temperature("chatgpt", "gpt-5.4-mini")
    assert not local_api._model_supports_temperature("chatgpt", "gpt-5.5")
    assert local_api._model_supports_temperature("sakana", "fugu")


def test_normalize_openai_compatible_base_url_appends_v1():
    assert (
        local_api._normalize_openai_compatible_base_url("https://api.sakana.ai")
        == "https://api.sakana.ai/v1"
    )
    assert (
        local_api._normalize_openai_compatible_base_url("https://api.sakana.ai/v1")
        == "https://api.sakana.ai/v1"
    )


def test_sakana_client_uses_sakana_key_and_base_url(monkeypatch):
    service = _make_service(enabled=False)
    service.sakana_base_url = "https://api.sakana.ai/v1"
    captured = {}

    class FakeOpenAI:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(local_api, "OpenAI", FakeOpenAI)
    monkeypatch.setenv("SAKANA_API_KEY", "sakana-test-key")
    monkeypatch.delenv("FUGU_API_KEY", raising=False)

    assert isinstance(service.sakana_client, FakeOpenAI)
    assert captured == {
        "api_key": "sakana-test-key",
        "base_url": "https://api.sakana.ai/v1",
    }


def test_llm_text_response_uses_sakana_chat_completions():
    service = _make_service(enabled=False)
    service.engine = _ModelSelectionEngine()
    captured = {}

    class FakeMessage:
        content = "{\"ok\": true}"

    class FakeChoice:
        message = FakeMessage()
        finish_reason = "stop"

    class FakeResponse:
        choices = [FakeChoice()]

    class FakeCompletions:
        def create(self, **kwargs):
            captured.update(kwargs)
            return FakeResponse()

    class FakeChat:
        completions = FakeCompletions()

    class FakeSakanaClient:
        chat = FakeChat()

    service._sakana_client = FakeSakanaClient()

    result = service._llm_text_response(
        provider="sakana",
        model="fugu-ultra",
        system_prompt="Answer as JSON.",
        user_message="Return ok.",
        max_tokens=123,
        temperature=0.2,
    )

    assert result == {
        "provider": "sakana",
        "model": "fugu-ultra",
        "text": "{\"ok\": true}",
    }
    assert captured["model"] == "fugu-ultra"
    assert captured["max_completion_tokens"] == 123
    assert captured["temperature"] == 0.2
    assert captured["response_format"] == {"type": "json_object"}
    assert captured["messages"] == [
        {"role": "system", "content": "Answer as JSON."},
        {"role": "user", "content": "Return ok."},
    ]


def test_ask_with_sources_passes_selected_model_to_llm():
    service = _make_service(enabled=False)
    service.engine = _ModelSelectionEngine()
    captured = {}

    def fake_llm(**kwargs):
        captured.update(kwargs)
        return {
            "provider": kwargs["provider"],
            "model": kwargs["model"],
            "text": json.dumps(
                {
                    "status": "answered",
                    "confidence": "high",
                    "answer": "The app runs locally [1] and is a portfolio demo [2].",
                    "citations": [
                        {"citation_id": 1, "reason": "Local-first behavior."},
                        {"citation_id": 2, "reason": "Portfolio framing."},
                    ],
                    "warnings": [],
                    "has_conflict": False,
                }
            ),
        }

    service._llm_text_response = fake_llm

    result = service.ask_with_sources(
        "Does the app run locally and present itself as a portfolio-friendly demo?",
        _make_answer_sources(),
        provider="claude",
        model="claude-opus-4-8",
    )

    assert captured["model"] == "claude-opus-4-8"
    assert result["model"] == "claude-opus-4-8"
    assert result["provider"] == "claude"
    assert result["status"] == "answered"


def test_ask_with_sources_rejects_unknown_model():
    service = _make_service(enabled=False)
    service.engine = _ModelSelectionEngine()

    with pytest.raises(ValueError, match="model for provider"):
        service.ask_with_sources(
            "Does the app run locally?",
            _make_answer_sources(),
            provider="chatgpt",
            model="gpt-2",
        )


def test_summary_cache_key_includes_model():
    key_a = LocalRAGService._summary_cache_key(
        language="en", provider="claude", max_points=5, model="claude-sonnet-4-6"
    )
    key_b = LocalRAGService._summary_cache_key(
        language="en", provider="claude", max_points=5, model="claude-opus-4-8"
    )
    assert key_a != key_b
    assert key_a.endswith(":claude-sonnet-4-6")


def test_llm_options_route_returns_models_and_defaults():
    class StubService:
        openai_model = "gpt-5.4-mini"
        sakana_model = "fugu"
        engine = _ModelSelectionEngine()

        def _resolve_llm_model(self, provider, model):
            return LocalRAGService._resolve_llm_model(self, provider, model)

    class FakeHandler:
        def __init__(self, path):
            self.path = path
            self.response_status = None
            self.response_payload = None

        def _json(self, payload, status=200):
            self.response_status = status
            self.response_payload = payload

    original_service = local_api.SERVICE
    try:
        local_api.SERVICE = StubService()
        handler = FakeHandler("/v1/llm-options")
        local_api.Handler.do_GET(handler)
    finally:
        local_api.SERVICE = original_service

    assert handler.response_status == 200
    payload = handler.response_payload
    assert payload["ok"] is True
    providers = payload["providers"]
    assert set(providers) == {"chatgpt", "claude", "sakana"}
    assert providers["chatgpt"]["default"] == "gpt-5.4-mini"
    assert providers["claude"]["default"] == "claude-sonnet-4-5-20250929"
    assert providers["sakana"]["default"] == "fugu"
    chatgpt_ids = [row["id"] for row in providers["chatgpt"]["models"]]
    claude_ids = [row["id"] for row in providers["claude"]["models"]]
    sakana_ids = [row["id"] for row in providers["sakana"]["models"]]
    assert chatgpt_ids[1] == "gpt-5.4-mini"
    assert claude_ids[0] == "claude-sonnet-4-5-20250929"
    assert {"gpt-5.4-nano", "gpt-5.4-mini", "gpt-5.5"} <= set(chatgpt_ids)
    assert {"claude-haiku-4-5", "claude-sonnet-4-6", "claude-opus-4-8"} <= set(
        claude_ids
    )
    assert {"fugu", "fugu-ultra", "fugu-ultra-20260615"} <= set(sakana_ids)
