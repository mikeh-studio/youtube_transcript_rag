import React, { useEffect, useMemo, useRef, useState } from "react";

const LOCK_KEY = "yt_rag_ingest_unlocked";
const LAST_VIDEO_KEY = "yt_rag_last_video_id";
const FEEDBACK_REVISION_STORAGE_KEY = "youtube-rag-feedback-revision";
const LOCALE_STORAGE_KEY = "youtube-rag-ui-locale";
const INTRO_SEEN_SESSION_KEY = "yt_rag_intro_seen";
const STUDY_HISTORY_SESSION_KEY = "yt_rag_study_history_v1";
const STUDY_HISTORY_LIMIT = 12;
const ROUTES = {
  INGEST: "/ingest",
  LOCAL_VIDEO: "/local-video",
  TLDR: "/tldr",
  STUDY: "/study",
  QA: "/qa",
};
const CORE_NAV_ITEMS = [
  { key: "ingest", label: "Ingest", icon: "/icons/icon-upload.svg", route: ROUTES.INGEST },
  {
    key: "studio",
    label: "Studio",
    icon: "/icons/icon-chat.svg",
    route: ROUTES.QA,
    activeRoutes: [ROUTES.QA, ROUTES.STUDY, ROUTES.TLDR],
    requiresUnlock: true,
  },
];
const LIBRARY_NAV_ITEMS = [
  { key: "reviews", label: "Reviews", icon: "/icons/icon-library.svg", href: "/reviews.html", requiresUnlock: true },
  { key: "evidence", label: "Evidence", icon: "/icons/icon-library.svg", href: "/evidence.html", requiresUnlock: true },
];
const MORE_NAV_ITEMS = [
  { key: "local-video", label: "Local Video", icon: "/icons/icon-search.svg", href: "/index.html#/local-video" },
  { key: "evaluation", label: "Evaluation", icon: "/icons/icon-jobs.svg", href: "/evaluation.html", requiresUnlock: true },
  { key: "chunking", label: "Chunking", icon: "/icons/icon-jobs.svg", href: "/chunking.html", requiresUnlock: true },
];
const YOUTUBE_ID_PATTERN = /^[a-zA-Z0-9_-]{11}$/;
const FALLBACK_LLM_PROVIDER_OPTIONS = {
  chatgpt: {
    default: "gpt-5.4-mini",
    models: [
      { id: "gpt-5.4-nano", label: "GPT-5.4 nano" },
      { id: "gpt-5.4-mini", label: "GPT-5.4 mini" },
      { id: "gpt-5.5", label: "GPT-5.5" },
    ],
  },
  claude: {
    default: "",
    models: [
      { id: "claude-haiku-4-5", label: "Claude Haiku 4.5" },
      { id: "claude-sonnet-4-6", label: "Claude Sonnet 4.6" },
      { id: "claude-opus-4-8", label: "Claude Opus 4.8" },
    ],
  },
  sakana: {
    default: "fugu",
    models: [
      { id: "fugu", label: "Sakana Fugu" },
      { id: "fugu-ultra", label: "Sakana Fugu Ultra" },
      { id: "fugu-ultra-20260615", label: "Sakana Fugu Ultra 20260615" },
    ],
  },
};
const STUDY_FOCUS_PRESETS = [
  { id: "main_ideas", label: "Main ideas" },
  { id: "characters", label: "People / characters" },
  { id: "vocabulary", label: "Vocabulary" },
  { id: "timeline", label: "Timeline / sequence" },
  { id: "quotes", label: "Quotes / context" },
  { id: "exam", label: "Exam prep" },
  { id: "language", label: "Language learning" },
  { id: "discussion", label: "Discussion questions" },
];
const STUDY_SCOPES = [
  { id: "whole_video", label: "Whole video" },
  { id: "focused_sections", label: "Strongest sections" },
];
const STUDY_MODEL_PROFILES = [
  { id: "economy", label: "Economy" },
  { id: "balanced", label: "Balanced" },
  { id: "quality", label: "Quality" },
];
const STUDY_TOPIC_DETAIL_LEVELS = [
  { id: "brief", label: "Brief Map" },
  { id: "explain", label: "Explain Topic" },
];
let llmProviderOptionsPromise = null;

function loadLlmProviderOptions() {
  if (!llmProviderOptionsPromise) {
    llmProviderOptionsPromise = apiRequest("/v1/llm-options")
      .then((payload) => payload?.providers || FALLBACK_LLM_PROVIDER_OPTIONS)
      .catch(() => FALLBACK_LLM_PROVIDER_OPTIONS);
  }
  return llmProviderOptionsPromise;
}

function useLlmProviderOptions() {
  const [options, setOptions] = useState(FALLBACK_LLM_PROVIDER_OPTIONS);
  useEffect(() => {
    let cancelled = false;
    loadLlmProviderOptions().then((value) => {
      if (!cancelled) {
        setOptions(value);
      }
    });
    return () => {
      cancelled = true;
    };
  }, []);
  return options;
}

function ModelSelectLabel({ providerOptions, provider, model, onChange, disabled = false }) {
  const providerEntry = providerOptions?.[provider] || {};
  const models = Array.isArray(providerEntry.models) ? providerEntry.models : [];
  const defaultLabel = providerEntry.default
    ? `Default (${providerEntry.default})`
    : "Default";
  return (
    <label>
      <span>Model</span>
      <select
        value={model}
        onChange={(event) => onChange(event.target.value)}
        disabled={disabled}
      >
        <option value="">{defaultLabel}</option>
        {models
          .map((row) => (
            <option key={row.id} value={row.id}>
              {row.id === providerEntry.default ? `${row.label} (default)` : row.label}
            </option>
          ))}
      </select>
    </label>
  );
}
const QA_ANSWER_I18N = {
  "en-US": {
    answerStatusAnswered: "Answered from evidence",
    answerStatusInsufficient: "Insufficient evidence",
    answerStatusError: "Answer unavailable",
    answerStatusDefault: "Answer",
    answerCitationCount: "{count} citation(s)",
    answerConfidence: "confidence: {value}",
    answerConfidenceHigh: "high",
    answerConfidenceMedium: "medium",
    answerConfidenceLow: "low",
    answerProvider: "provider: {value}",
    answerModel: "model: {value}",
    answerTrustNote: "Answer generated from retrieved evidence. May be incomplete if source coverage is limited.",
    answerSupportingEvidence: "Supporting evidence",
    answerHideEvidence: "Hide supporting evidence",
    answerShowEvidence: "Show supporting evidence",
    answerOpenSource: "Open source",
    answerPlayAtTimestamp: "Play at timestamp",
    answerRelevant: "Relevant",
    answerNotRelevant: "Not Relevant",
    answerNoEvidence: "No supporting evidence is available for this answer.",
    reviewSaving: "Saving...",
    reviewSaved: "Saved",
    reviewSaveFailed: "Save failed: {message}",
    askFailed: "Ask failed: {message}",
  },
  "ja-JP": {
    answerStatusAnswered: "根拠付きで回答",
    answerStatusInsufficient: "根拠が不十分",
    answerStatusError: "回答を生成できません",
    answerStatusDefault: "回答",
    answerCitationCount: "引用 {count} 件",
    answerConfidence: "信頼度: {value}",
    answerConfidenceHigh: "高",
    answerConfidenceMedium: "中",
    answerConfidenceLow: "低",
    answerProvider: "プロバイダー: {value}",
    answerModel: "モデル: {value}",
    answerTrustNote: "取得した根拠から生成した回答です。ソースの範囲次第で不完全な場合があります。",
    answerSupportingEvidence: "根拠チャンク",
    answerHideEvidence: "根拠を隠す",
    answerShowEvidence: "根拠を表示",
    answerOpenSource: "ソースを開く",
    answerPlayAtTimestamp: "この位置から再生",
    answerRelevant: "関連あり",
    answerNotRelevant: "関連なし",
    answerNoEvidence: "この回答に表示できる根拠チャンクはありません。",
    reviewSaving: "保存中...",
    reviewSaved: "保存しました",
    reviewSaveFailed: "保存に失敗しました: {message}",
    askFailed: "回答生成に失敗しました: {message}",
  },
};

function readHashRoute() {
  const raw = String(window.location.hash || "").replace(/^#/, "").trim();
  if (!raw) {
    return ROUTES.INGEST;
  }
  if (
    raw === ROUTES.INGEST
    || raw === ROUTES.LOCAL_VIDEO
    || raw === ROUTES.TLDR
    || raw === ROUTES.STUDY
    || raw === ROUTES.QA
  ) {
    return raw;
  }
  return ROUTES.INGEST;
}

function navigate(route) {
  window.location.hash = route;
}

function markFeedbackRevision() {
  try {
    localStorage.setItem(FEEDBACK_REVISION_STORAGE_KEY, String(Date.now()));
  } catch (_) {
    // best effort only for cross-tab review refreshes
  }
}

function readLocalStorage(key, fallback = "") {
  try {
    return localStorage.getItem(key) ?? fallback;
  } catch (_) {
    return fallback;
  }
}

function writeLocalStorage(key, value) {
  try {
    localStorage.setItem(key, value);
  } catch (_) {
    // best effort only; storage failures should not break generated output
  }
}

function readStudyHistory() {
  try {
    const parsed = JSON.parse(sessionStorage.getItem(STUDY_HISTORY_SESSION_KEY) || "[]");
    if (!Array.isArray(parsed)) {
      return [];
    }
    return parsed
      .filter((run) => run && typeof run === "object" && run.id && run.mode && run.payload)
      .slice(0, STUDY_HISTORY_LIMIT);
  } catch (_) {
    return [];
  }
}

function writeStudyHistory(runs) {
  try {
    sessionStorage.setItem(
      STUDY_HISTORY_SESSION_KEY,
      JSON.stringify(runs.slice(0, STUDY_HISTORY_LIMIT)),
    );
  } catch (_) {
    // best effort only; history should not block fresh generations
  }
}

function makeStudyRunId() {
  const randomPart = typeof crypto !== "undefined" && crypto.randomUUID
    ? crypto.randomUUID()
    : Math.random().toString(36).slice(2);
  return `study-${Date.now()}-${randomPart}`;
}

function optionalNumber(value) {
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : null;
}

async function apiRequest(path, { method = "GET", body } = {}) {
  let response;
  try {
    response = await fetch(path, {
      method,
      cache: "no-store",
      headers: {
        "content-type": "application/json",
      },
      body: body ? JSON.stringify(body) : undefined,
    });
  } catch (error) {
    throw makeFriendlyError(
      "Could not connect to the backend server.",
      String(error?.message || error),
    );
  }

  const rawText = await response.text();
  let payload = null;
  try {
    payload = rawText ? JSON.parse(rawText) : {};
  } catch (error) {
    throw makeFriendlyError(
      "Could not connect to the backend server.",
      [
        `Expected JSON from ${path}, but received a non-JSON response.`,
        `Parse error: ${String(error?.message || error)}`,
        rawText.slice(0, 1200),
      ].filter(Boolean).join("\n\n"),
    );
  }
  if (!response.ok) {
    const message = payload?.error?.message || `Request failed (${response.status})`;
    throw makeFriendlyError(message, JSON.stringify(payload, null, 2) || rawText || message);
  }
  return payload;
}

function makeFriendlyError(userMessage, debugMessage = "") {
  const error = new Error(userMessage);
  error.userMessage = userMessage;
  error.debugMessage = debugMessage || userMessage;
  return error;
}

function errorInfo(error, fallbackMessage = "Request failed.") {
  if (!error) {
    return null;
  }
  return {
    userMessage: error.userMessage || String(error.message || error || fallbackMessage),
    debugMessage: error.debugMessage || String(error.message || error || fallbackMessage),
  };
}

function FriendlyError({ error, title = "Something went wrong.", actionLabel = "Retry", onRetry }) {
  const info = errorInfo(error, title);
  if (!info) {
    return null;
  }
  return (
    <div className="friendly-error" role="alert">
      <div className="friendly-error-main">
        <strong>{title}</strong>
        <span>{info.userMessage}</span>
      </div>
      {onRetry ? (
        <button className="btn secondary" type="button" onClick={onRetry}>
          {actionLabel}
        </button>
      ) : null}
      <details className="friendly-error-debug">
        <summary>Debug details</summary>
        <pre>{info.debugMessage}</pre>
      </details>
    </div>
  );
}

function formatSeconds(value) {
  const seconds = Math.max(0, Math.floor(Number(value || 0)));
  const minutes = Math.floor(seconds / 60);
  const remaining = seconds % 60;
  return `${minutes}:${String(remaining).padStart(2, "0")}`;
}

function formatRange(startValue, endValue) {
  const start = Number(startValue || 0);
  const end = Number(endValue ?? startValue ?? 0);
  return `${formatSeconds(start)} - ${formatSeconds(end)}`;
}

function evidenceTimestamp(row) {
  if (row?.timestamp_hhmmss) {
    return row.timestamp_hhmmss;
  }
  if (row?.source_type === "ocr") {
    return formatSeconds(row.timestamp_sec ?? row.start);
  }
  return formatRange(row?.start_sec ?? row?.start_seconds ?? row?.start, row?.end_sec ?? row?.end_seconds ?? row?.end);
}

function evidenceTitle(row) {
  return row?.video_title || row?.metadata?.video_title || row?.video_id || "-";
}

function evidenceLanguage(row) {
  return row?.source_type === "ocr" ? "OCR" : row?.language || row?.metadata?.language || "-";
}

function evidenceFramePath(row) {
  return row?.frame_path || row?.metadata?.frame_path || "";
}

function extractVideoId(value) {
  const raw = String(value || "").trim();
  if (YOUTUBE_ID_PATTERN.test(raw)) {
    return raw;
  }
  const patterns = [
    /[?&]v=([a-zA-Z0-9_-]{11})/,
    /youtu\.be\/([a-zA-Z0-9_-]{11})/,
    /embed\/([a-zA-Z0-9_-]{11})/,
  ];
  for (const pattern of patterns) {
    const match = raw.match(pattern);
    if (match) {
      return match[1];
    }
  }
  return null;
}

function thumbnailUrlForVideo(videoId) {
  const scopedVideoId = String(videoId || "").trim();
  if (!YOUTUBE_ID_PATTERN.test(scopedVideoId)) {
    return "";
  }
  return `https://i.ytimg.com/vi/${encodeURIComponent(scopedVideoId)}/hqdefault.jpg`;
}

function safeExternalUrl(value) {
  const raw = String(value || "").trim();
  if (!raw) {
    return "";
  }
  try {
    const parsed = new URL(raw, window.location.origin);
    return parsed.protocol === "http:" || parsed.protocol === "https:" ? parsed.href : "";
  } catch (_) {
    return "";
  }
}

function qaAnswerText(locale, key, vars = {}) {
  const dictionary = QA_ANSWER_I18N[locale] || QA_ANSWER_I18N["en-US"];
  const template = dictionary[key] || QA_ANSWER_I18N["en-US"][key] || key;
  return template.replace(/\{(\w+)\}/g, (_, token) => String(vars[token] ?? ""));
}

function answerStatusLabel(status, locale) {
  if (status === "answered") {
    return qaAnswerText(locale, "answerStatusAnswered");
  }
  if (status === "insufficient_evidence") {
    return qaAnswerText(locale, "answerStatusInsufficient");
  }
  if (status === "error") {
    return qaAnswerText(locale, "answerStatusError");
  }
  return qaAnswerText(locale, "answerStatusDefault");
}

function answerConfidenceLabel(confidence, locale) {
  const scopedConfidence = String(confidence || "low").trim().toLowerCase();
  if (scopedConfidence === "high") {
    return qaAnswerText(locale, "answerConfidenceHigh");
  }
  if (scopedConfidence === "medium") {
    return qaAnswerText(locale, "answerConfidenceMedium");
  }
  return qaAnswerText(locale, "answerConfidenceLow");
}

function answerStatusTone(status) {
  if (status === "answered") {
    return "ok";
  }
  if (status === "error") {
    return "error";
  }
  return "pending";
}

function prefersReducedMotion() {
  if (typeof window === "undefined" || typeof window.matchMedia !== "function") {
    return false;
  }
  return window.matchMedia("(prefers-reduced-motion: reduce)").matches;
}

function IngestPage({ onSuccess }) {
  const [url, setUrl] = useState("");
  const [mode, setMode] = useState("single");
  const [language, setLanguage] = useState("ja");
  const [force, setForce] = useState(false);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [status, setStatus] = useState("Enter a YouTube URL to unlock the workspace.");
  const [rawResponse, setRawResponse] = useState("");
  const [ingestedVideos, setIngestedVideos] = useState([]);
  const [isLoadingVideos, setIsLoadingVideos] = useState(true);
  const [videosError, setVideosError] = useState("");
  const [deletingVideoId, setDeletingVideoId] = useState("");
  const carouselRef = useRef(null);

  async function loadIngestedVideos({ silent = false } = {}) {
    if (!silent) {
      setIsLoadingVideos(true);
      setVideosError(null);
    }
    try {
      const payload = await apiRequest("/v1/videos");
      const rows = Array.isArray(payload?.videos) ? payload.videos : [];
      const orderedRows = rows.slice().reverse();
      setIngestedVideos(orderedRows);
      return orderedRows;
    } catch (error) {
      if (!silent) {
        setVideosError(error);
      }
      return [];
    } finally {
      if (!silent) {
        setIsLoadingVideos(false);
      }
    }
  }

  async function refreshIngestedVideosUntilVisible(expectedVideoIds, timeoutMs = 8000, intervalMs = 300) {
    const required = Array.from(
      new Set((expectedVideoIds || []).map((row) => String(row || "").trim()).filter(Boolean)),
    );
    if (!required.length) {
      await loadIngestedVideos();
      return true;
    }

    const startedAt = Date.now();
    while (Date.now() - startedAt <= timeoutMs) {
      const rows = await loadIngestedVideos({ silent: true });
      const ids = new Set(rows.map((row) => String(row.video_id || "").trim()).filter(Boolean));
      const allFound = required.every((id) => ids.has(id));
      if (allFound) {
        return true;
      }
      await new Promise((resolve) => window.setTimeout(resolve, intervalMs));
    }
    await loadIngestedVideos();
    return false;
  }

  useEffect(() => {
    loadIngestedVideos().catch(() => {});
  }, []);

  async function handleSubmit(event) {
    event.preventDefault();
    if (!url.trim()) {
      return;
    }
    setIsSubmitting(true);
    setStatus("Running ingestion...");
    setRawResponse("");
    try {
      const payload = await apiRequest("/v1/ingest/videos", {
        method: "POST",
        body: {
          url: url.trim(),
          mode,
          language,
          force,
        },
      });
      const acceptedCount = Number(payload?.queued_count || 0) + Number(payload?.skipped_count || 0);
      if (!payload?.ok || acceptedCount <= 0) {
        throw new Error("Ingest request was accepted but no jobs were created.");
      }
      const completedJobs = (payload?.jobs || []).filter((job) => job?.status === "completed");
      const completedVideoIds = completedJobs
        .map((job) => String(job?.video_id || "").trim())
        .filter(Boolean);
      const firstVideo =
        payload?.jobs?.[0]?.video_id
        || payload?.skipped?.[0]?.video_id
        || "";
      setRawResponse(JSON.stringify(payload, null, 2));

      if (completedJobs.length) {
        setIngestedVideos((prev) => {
          const byId = new Map();
          prev.forEach((row) => {
            const id = String(row?.video_id || "").trim();
            if (id) {
              byId.set(id, row);
            }
          });
          completedJobs.forEach((job) => {
            const id = String(job?.video_id || "").trim();
            if (!id) {
              return;
            }
            const existing = byId.get(id) || {};
            byId.set(id, {
              video_id: id,
              title: String(job?.title || existing.title || `Video ${id}`),
              language: String(job?.language || existing.language || language),
              num_chunks: Number(existing.num_chunks || 0),
            });
          });
          return Array.from(byId.values());
        });
      }

      const refreshed = await refreshIngestedVideosUntilVisible(completedVideoIds);
      if (refreshed) {
        const label = completedVideoIds.length === 1 ? "video" : "videos";
        setStatus(`Ingest successful. ${completedVideoIds.length} ${label} added and list refreshed.`);
      } else {
        setStatus("Ingest completed, but list refresh is still catching up. Use Reload list.");
      }
      onSuccess(firstVideo);
    } catch (error) {
      const info = errorInfo(error, "Ingest failed.");
      setStatus(`Ingest failed. ${info.userMessage}`);
      setRawResponse(info.debugMessage);
    } finally {
      setIsSubmitting(false);
    }
  }

  async function deleteVideo(videoId) {
    const scopedVideoId = String(videoId || "").trim();
    if (!scopedVideoId) {
      return;
    }
    const confirmed = window.confirm(`Delete ${scopedVideoId} from the library?`);
    if (!confirmed) {
      return;
    }
    setDeletingVideoId(scopedVideoId);
    try {
      await apiRequest(`/v1/videos/${encodeURIComponent(scopedVideoId)}`, {
        method: "DELETE",
      });
      if (localStorage.getItem(LAST_VIDEO_KEY) === scopedVideoId) {
        localStorage.removeItem(LAST_VIDEO_KEY);
      }
      setStatus(`Deleted ${scopedVideoId} from the library.`);
      await loadIngestedVideos();
    } catch (error) {
      const info = errorInfo(error, "Delete failed.");
      setStatus(`Delete failed. ${info.userMessage}`);
    } finally {
      setDeletingVideoId("");
    }
  }

  function scrollCarousel(direction) {
    if (!carouselRef.current) {
      return;
    }
    const step = Math.max(carouselRef.current.clientWidth * 0.8, 280);
    carouselRef.current.scrollBy({
      left: direction * step,
      behavior: "smooth",
    });
  }

  return (
    <section className="ingest-stage">
      <header className="hero ingest-hero">
        <h1>Ingest Gateway</h1>
        <p className="subtitle">
          Start here. Once ingest succeeds, TLDR Studio, Study Studio, Q&amp;A Studio, Evaluation, and Reviews unlock.
        </p>
      </header>

      <section className="panel ingest-panel primary-ingest-panel">
        <h2 className="section-title">
          <img src="/icons/icon-upload.svg" alt="" aria-hidden="true" />
          <span>Paste a YouTube URL</span>
        </h2>
        <p className="section-desc">Add a video or playlist and unlock the analysis workspace.</p>
        <form className="grid ingest-compose-form" onSubmit={handleSubmit}>
          <div className="ingest-primary-row">
            <label className="ingest-url-field">
              <span>YouTube URL</span>
              <input
                required
                value={url}
                onChange={(event) => setUrl(event.target.value)}
                placeholder="Paste a video or playlist URL"
              />
            </label>
            <button className="btn ingest-submit-btn" type="submit" disabled={isSubmitting}>
              {isSubmitting ? "Submitting..." : "Run Ingestion"}
            </button>
          </div>

          <div className="ingest-advanced-grid">
            <label>
              <span>Mode</span>
              <select value={mode} onChange={(event) => setMode(event.target.value)}>
                <option value="single">single</option>
                <option value="playlist">playlist</option>
              </select>
            </label>
            <label>
              <span>Language</span>
              <select value={language} onChange={(event) => setLanguage(event.target.value)}>
                <option value="ja">ja</option>
                <option value="en">en</option>
              </select>
            </label>
            <label className="checkbox ingest-force-check">
              <input type="checkbox" checked={force} onChange={(event) => setForce(event.target.checked)} />
              <span>Force reingest</span>
            </label>
          </div>
        </form>
        <div className="search-summary">{status}</div>
        <details className="raw-json">
          <summary>Raw JSON</summary>
          <pre className="output">{rawResponse}</pre>
        </details>
      </section>

      <section className="panel ingest-library-panel">
        <div className="section-head">
          <h2 className="section-title">
            <img src="/icons/icon-library.svg" alt="" aria-hidden="true" />
            <span>Ingested Videos</span>
          </h2>
          <div className="ingest-carousel-controls">
            <button
              className="btn secondary ingest-reload-btn"
              type="button"
              onClick={() => loadIngestedVideos().catch(() => {})}
            >
              Reload list
            </button>
            <button
              className="btn secondary ingest-scroll-btn"
              type="button"
              onClick={() => scrollCarousel(-1)}
              disabled={!ingestedVideos.length}
              aria-label="Scroll videos left"
            >
              ◀
            </button>
            <button
              className="btn secondary ingest-scroll-btn"
              type="button"
              onClick={() => scrollCarousel(1)}
              disabled={!ingestedVideos.length}
              aria-label="Scroll videos right"
            >
              ▶
            </button>
          </div>
        </div>

        {videosError ? (
          <FriendlyError
            error={videosError}
            title="Could not load ingested videos."
            onRetry={() => loadIngestedVideos().catch(() => {})}
          />
        ) : null}

        {isLoadingVideos ? (
          <div className="search-empty">Loading ingested videos...</div>
        ) : null}

        {!isLoadingVideos && !ingestedVideos.length ? (
          <div className="search-empty">No ingested videos yet.</div>
        ) : null}

        {!isLoadingVideos && ingestedVideos.length ? (
          <div className="ingest-carousel" ref={carouselRef}>
            {ingestedVideos.map((video) => {
              const videoId = String(video.video_id || "").trim();
              const reviewHref = `/reviews.html?video_id=${encodeURIComponent(videoId)}`;
              const isDeleting = deletingVideoId === videoId;
              const thumbnailUrl = thumbnailUrlForVideo(videoId);
              return (
                <article className="ingest-video-card" key={videoId}>
                  <div className={`ingest-video-thumb ${thumbnailUrl ? "" : "fallback-only"}`}>
                    <span className="ingest-thumb-fallback">
                      <img src="/icons/icon-play.svg" alt="" aria-hidden="true" />
                    </span>
                    {thumbnailUrl ? (
                      <img
                        src={thumbnailUrl}
                        alt=""
                        loading="lazy"
                        referrerPolicy="no-referrer"
                        onError={(event) => {
                          event.currentTarget.style.display = "none";
                        }}
                      />
                    ) : null}
                  </div>
                  <h3 className="ingest-video-title">{video.title || videoId}</h3>
                  <p className="ingest-video-meta">ID: {videoId}</p>
                  <p className="ingest-video-meta">Language: {video.language || "-"}</p>
                  <p className="ingest-video-meta">Chunks: {Number(video.num_chunks || 0)}</p>
                  <div className="ingest-video-actions">
                    <a className="btn secondary" href={reviewHref}>Review</a>
                    <button
                      className="btn secondary danger"
                      type="button"
                      disabled={isDeleting}
                      onClick={() => deleteVideo(videoId).catch(() => {})}
                    >
                      {isDeleting ? "Deleting..." : "Delete"}
                    </button>
                  </div>
                </article>
              );
            })}
          </div>
        ) : null}
      </section>
    </section>
  );
}

function LocalVideoAnalysisPage({ locale }) {
  const [ocrVideoId, setOcrVideoId] = useState("");
  const [ocrVideoPath, setOcrVideoPath] = useState("");
  const [ocrIntervalSec, setOcrIntervalSec] = useState(10);
  const [ocrSubmitting, setOcrSubmitting] = useState(false);
  const [ocrStatus, setOcrStatus] = useState(
    "Process a local .mp4 you own or have permission to analyze.",
  );
  const [ocrRawResponse, setOcrRawResponse] = useState("");
  const [ocrJobs, setOcrJobs] = useState([]);
  const [ocrJobsError, setOcrJobsError] = useState("");

  const [analysisVideoId, setAnalysisVideoId] = useState("");
  const [question, setQuestion] = useState("");
  const [sourceMode, setSourceMode] = useState("ocr");
  const [topK, setTopK] = useState(5);
  const [provider, setProvider] = useState("chatgpt");
  const [model, setModel] = useState("");
  const [askLoading, setAskLoading] = useState(false);
  const [askError, setAskError] = useState("");
  const [askResponse, setAskResponse] = useState(null);
  const llmProviderOptions = useLlmProviderOptions();

  async function loadOcrJobs({ silent = false } = {}) {
    if (!silent) {
      setOcrJobsError("");
    }
    try {
      const payload = await apiRequest("/v1/local-video-ocr/jobs");
      const rows = Array.isArray(payload?.jobs) ? payload.jobs : [];
      setOcrJobs(rows);
      const readyJob = rows.find((job) => Number(job.vector_count || 0) > 0) || rows[0];
      if (readyJob?.video_id) {
        setAnalysisVideoId((current) => current || String(readyJob.video_id));
      }
    } catch (error) {
      if (!silent) {
        setOcrJobsError(error);
      }
    }
  }

  useEffect(() => {
    loadOcrJobs({ silent: true }).catch(() => {});
    const intervalId = window.setInterval(() => {
      loadOcrJobs({ silent: true }).catch(() => {});
    }, 3000);
    return () => {
      window.clearInterval(intervalId);
    };
  }, []);

  async function handleOcrSubmit(event) {
    event.preventDefault();
    const scopedVideoId = ocrVideoId.trim();
    const scopedPath = ocrVideoPath.trim();
    if (!scopedVideoId || !scopedPath) {
      return;
    }
    setOcrSubmitting(true);
    setOcrStatus("Starting local video OCR job...");
    setOcrRawResponse("");
    try {
      const payload = await apiRequest("/v1/local-video-ocr/jobs", {
        method: "POST",
        body: {
          video_id: scopedVideoId,
          video_path: scopedPath,
          interval_sec: Number(ocrIntervalSec || 10),
        },
      });
      setOcrRawResponse(JSON.stringify(payload, null, 2));
      setOcrStatus(`OCR job ${payload?.job?.job_id || ""} queued. Status updates below.`);
      setAnalysisVideoId(scopedVideoId);
      await loadOcrJobs();
    } catch (error) {
      const info = errorInfo(error, "OCR job failed to start.");
      setOcrStatus(`OCR job failed to start. ${info.userMessage}`);
      setOcrRawResponse(info.debugMessage);
    } finally {
      setOcrSubmitting(false);
    }
  }

  async function runLocalAsk(event) {
    event.preventDefault();
    const scopedVideoId = analysisVideoId.trim();
    if (!scopedVideoId || !question.trim()) {
      return;
    }
    setAskLoading(true);
    setAskError("");
    setAskResponse(null);
    try {
      const payload = await apiRequest("/v1/ask-multimodal", {
        method: "POST",
        body: {
          question: question.trim(),
          video_id: scopedVideoId,
          source_mode: sourceMode,
          retrieval_mode: "hybrid",
          k: Number(topK || 5),
          provider,
          model: model || undefined,
        },
      });
      setAskResponse(payload);
    } catch (error) {
      setAskError(String(error?.message || error));
    } finally {
      setAskLoading(false);
    }
  }

  const answerStatus = String(askResponse?.status || "").trim();
  const answerEvidence = Array.isArray(askResponse?.citations) && askResponse.citations.length
    ? askResponse.citations
    : Array.isArray(askResponse?.retrieved_chunks)
      ? askResponse.retrieved_chunks
      : [];

  return (
    <section className="ingest-stage local-video-stage">
      <header className="hero">
        <h1>Local Video Analysis</h1>
        <p className="subtitle">
          Extract visual evidence from a local video, monitor OCR indexing, and ask grounded questions.
        </p>
      </header>

      <section className="panel local-ocr-panel">
        <h2 className="section-title">
          <img src="/icons/icon-upload.svg" alt="" aria-hidden="true" />
          <span>Process Local Video</span>
        </h2>
        <p className="section-desc">
          Use only local video files you own or have permission to process. Public YouTube downloading is not supported.
        </p>
        <form className="grid local-ocr-grid" onSubmit={handleOcrSubmit}>
          <label>
            <span>Video ID</span>
            <input
              required
              value={ocrVideoId}
              onChange={(event) => setOcrVideoId(event.target.value)}
              placeholder="demo_001"
            />
          </label>
          <label>
            <span>Local video path</span>
            <input
              required
              value={ocrVideoPath}
              onChange={(event) => setOcrVideoPath(event.target.value)}
              placeholder="data/raw/demo_001.mp4"
            />
          </label>
          <label>
            <span>Frame interval</span>
            <input
              type="number"
              min="1"
              max="120"
              value={ocrIntervalSec}
              onChange={(event) => setOcrIntervalSec(event.target.value)}
            />
          </label>
          <button className="btn" type="submit" disabled={ocrSubmitting}>
            {ocrSubmitting ? "Queueing..." : "Run OCR"}
          </button>
        </form>
        <div className="search-summary">{ocrStatus}</div>
        <div className="ocr-job-grid">
          {ocrJobs.map((job) => (
            <article className="ocr-job-card" key={job.job_id}>
              <div className="search-card-head">
                <div className="search-rank">{job.status}</div>
                <div className="search-title">{job.video_id}</div>
                <div className="search-lang">{job.step}</div>
              </div>
              <div className="search-meta">
                <span>frames {Number(job.frame_count || 0)}</span>
                <span>ocr {Number(job.ocr_count || 0)}</span>
                <span>vectors {Number(job.vector_count || 0)}</span>
                <span>interval {Number(job.interval_sec || 0)}s</span>
              </div>
              {job.error_message ? <p className="search-snippet error-text">{job.error_message}</p> : null}
              <p className="frame-path">{job.video_path}</p>
              <button
                className="btn secondary"
                type="button"
                onClick={() => setAnalysisVideoId(String(job.video_id || ""))}
                disabled={!job.video_id || Number(job.vector_count || 0) <= 0}
              >
                Use for Ask
              </button>
            </article>
          ))}
          {!ocrJobs.length ? <div className="search-empty">No local OCR jobs yet.</div> : null}
        </div>
        {ocrJobsError ? (
          <FriendlyError
            error={ocrJobsError}
            title="Could not load local OCR jobs."
            onRetry={() => loadOcrJobs().catch(() => {})}
          />
        ) : null}
        <details className="raw-json">
          <summary>OCR Raw JSON</summary>
          <pre className="output">{ocrRawResponse}</pre>
        </details>
      </section>

      <section className="panel local-video-ask-panel">
        <h2 className="section-title">
          <img src="/icons/icon-chat.svg" alt="" aria-hidden="true" />
          <span>Ask About Local Video</span>
        </h2>
        <p className="section-desc">
          OCR only uses indexed frames. Transcript + OCR also uses transcript chunks when the same video ID exists in the transcript library.
        </p>
        <form className="qa-ask-form" onSubmit={runLocalAsk}>
          <label className="ask-question-field">
            <span>Question</span>
            <textarea
              required
              rows="3"
              value={question}
              onChange={(event) => setQuestion(event.target.value)}
              placeholder="What text or visual information appears in this video?"
            />
          </label>
          <div className="qa-controls-row ask-controls-row">
            <label>
              <span>Video ID</span>
              <input
                required
                list="local-video-job-ids"
                value={analysisVideoId}
                onChange={(event) => setAnalysisVideoId(event.target.value)}
                placeholder="demo_001"
              />
              <datalist id="local-video-job-ids">
                {ocrJobs.map((job) => (
                  <option key={job.job_id} value={job.video_id} />
                ))}
              </datalist>
            </label>
            <label>
              <span>Evidence</span>
              <select value={sourceMode} onChange={(event) => setSourceMode(event.target.value)}>
                <option value="ocr">OCR only</option>
                <option value="both">Transcript + OCR</option>
              </select>
            </label>
            <label>
              <span>Top K</span>
              <input
                type="number"
                min="1"
                max="12"
                value={topK}
                onChange={(event) => setTopK(event.target.value)}
              />
            </label>
            <label>
              <span>Provider</span>
              <select
                value={provider}
                onChange={(event) => {
                  setProvider(event.target.value);
                  setModel("");
                }}
              >
                <option value="chatgpt">ChatGPT</option>
                <option value="claude">Claude</option>
                <option value="sakana">Sakana AI</option>
              </select>
            </label>
            <ModelSelectLabel
              providerOptions={llmProviderOptions}
              provider={provider}
              model={model}
              onChange={setModel}
            />
            <button className="btn qa-primary-action" type="submit" disabled={askLoading}>
              {askLoading ? "Generating..." : "Ask Local Video"}
            </button>
          </div>
        </form>

        {askError ? <div className="search-summary">Ask failed: {askError}</div> : null}
        {askResponse ? (
          <>
            <div className="search-summary ask-summary-row">
              <span className={`ask-status-pill ${answerStatusTone(answerStatus)}`}>
                {answerStatusLabel(answerStatus, locale)}
              </span>
              <span>{answerEvidence.length} evidence item(s)</span>
              <span>confidence: {answerConfidenceLabel(askResponse.confidence, locale)}</span>
              <span>source: {askResponse.source_mode || sourceMode}</span>
            </div>
            <article
              className={`ask-answer ask-answer-${answerStatus || "default"}`}
              data-testid="local-video-answer-panel"
            >
              <p>{askResponse.answer}</p>
              {(askResponse.warnings || []).map((warning) => (
                <p className="error-text" key={warning}>{warning}</p>
              ))}
            </article>
            <div className="search-cards">
              {answerEvidence.map((row, index) => (
                <article
                  className="search-card"
                  data-testid="local-video-evidence-card"
                  key={`${row.video_id || analysisVideoId}-${row.frame_id || row.chunk_id || index}`}
                >
                  <div className="search-card-head">
                    <div className="search-rank">#{index + 1}</div>
                    <div className="search-title">{row.video_title || row.video_id || analysisVideoId}</div>
                    <div className={`search-lang source-badge ${row.source_type === "ocr" ? "ocr" : "transcript"}`}>
                      {row.source_type || "evidence"}
                    </div>
                  </div>
                  <div className="search-meta">
                    <span>{row.timestamp_range_label || row.timestamp_label || formatSeconds(row.start_seconds)}</span>
                    {row.frame_id ? <span>{row.frame_id}</span> : null}
                  </div>
                  <p className="search-snippet">{row.snippet || row.text || row.reason}</p>
                  {evidenceFramePath(row) ? <p className="frame-path">{evidenceFramePath(row)}</p> : null}
                </article>
              ))}
              {!answerEvidence.length ? (
                <div className="search-empty">No local video evidence was retrieved.</div>
              ) : null}
            </div>
            <details className="raw-json">
              <summary>Ask Raw JSON</summary>
              <pre className="output">{JSON.stringify(askResponse, null, 2)}</pre>
            </details>
          </>
        ) : null}
      </section>
    </section>
  );
}

function PlayerPanel({ videoId, startSeconds, title }) {
  if (!videoId) {
    return null;
  }

  const safeVideoId = encodeURIComponent(videoId);
  const start = Math.max(0, Math.floor(Number(startSeconds || 0)));
  const src = `https://www.youtube.com/embed/${safeVideoId}?start=${start}&autoplay=1&rel=0&modestbranding=1&playsinline=1`;

  return (
    <section className="panel">
      <h2 className="section-title">
        <img src="/icons/icon-play.svg" alt="" aria-hidden="true" />
        <span>YouTube Player</span>
      </h2>
      <p className="section-desc">{title ? `Now playing: ${title} (${formatSeconds(start)})` : "Now playing"}</p>
      <div className="player-shell player-active">
        <iframe
          key={`${videoId}-${start}`}
          className="player-frame"
          title="YouTube player"
          src={src}
          allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
          allowFullScreen
        />
      </div>
    </section>
  );
}

const RETRIEVAL_STRATEGY_LABELS = {
  initial: "Initial retrieval",
  rewrite_query: "Rewritten query",
  switch_mode: "Mode switch",
  broaden_top_k: "Broader search",
};

const RETRIEVAL_REASON_LABELS = {
  multi_chunk_support: "Multiple supporting chunks",
  strong_single_chunk: "One strong supporting chunk",
  no_results: "No matching evidence",
  thin_support: "Evidence too thin",
  single_weak_chunk: "One weak chunk",
  mixed_signals: "Retrieval signals disagree",
};

function finiteNumber(value) {
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : null;
}

function RetrievalAttemptTimeline({ details }) {
  const trace = details?.agentic_retrieval;
  const attempts = Array.isArray(trace?.attempts) ? trace.attempts : [];
  if (!attempts.length) {
    return null;
  }

  const selectedAttempt = attempts.findIndex((attempt) => (
    String(attempt.query || "") === String(trace.final_query || "")
    && String(attempt.retrieval_mode || "") === String(trace.final_retrieval_mode || "")
    && Number(attempt.k) === Number(trace.final_k)
  ));

  return (
    <section className="retrieval-visual-block agentic-trace" data-testid="agentic-attempt-timeline">
      <div className="retrieval-visual-heading">
        <div>
          <h3>Agentic attempt timeline</h3>
          <p>See why retrieval retried and which attempt supplied the final evidence.</p>
        </div>
        <span className={`retrieval-outcome ${trace.sufficient ? "sufficient" : "limited"}`}>
          {trace.sufficient ? "Evidence ready" : "Best available evidence"}
        </span>
      </div>
      <ol
        className="agentic-attempt-list"
        aria-label={`${attempts.length} retrieval attempt${attempts.length === 1 ? "" : "s"}`}
      >
        {attempts.map((attempt, index) => {
          const isSelected = index === selectedAttempt;
          const reason = RETRIEVAL_REASON_LABELS[attempt.reason_code]
            || String(attempt.reason_code || "Evidence assessed").replaceAll("_", " ");
          return (
            <li
              className={`agentic-attempt ${attempt.sufficient ? "sufficient" : ""} ${isSelected ? "selected" : ""}`}
              key={`${attempt.attempt}-${attempt.strategy}-${attempt.query}`}
            >
              <div className="agentic-attempt-marker" aria-hidden="true">
                {attempt.sufficient ? "✓" : attempt.attempt}
              </div>
              <div className="agentic-attempt-copy">
                <div className="agentic-attempt-head">
                  <strong>{RETRIEVAL_STRATEGY_LABELS[attempt.strategy] || attempt.strategy}</strong>
                  {isSelected ? <span>Selected</span> : null}
                </div>
                <p className="agentic-attempt-query">{attempt.query || "-"}</p>
                <div className="agentic-attempt-meta">
                  <span>{attempt.retrieval_mode}</span>
                  <span>top {attempt.k}</span>
                  <span>{attempt.result_count} result{Number(attempt.result_count) === 1 ? "" : "s"}</span>
                </div>
                <p className="agentic-attempt-reason">{reason}</p>
              </div>
            </li>
          );
        })}
      </ol>
      <p className="retrieval-visual-foot">
        Stopped: {String(trace.stopped_reason || "completed").replaceAll("_", " ")}
      </p>
    </section>
  );
}

function ResultRankingProfile({ rows }) {
  const rankedRows = (Array.isArray(rows) ? rows : [])
    .map((row, index) => {
      const score = finiteNumber(row?.rerank_score ?? row?.score);
      if (score === null) {
        return null;
      }
      return {
        row,
        score,
        finalRank: finiteNumber(row?.rank ?? row?.citation_id) ?? index + 1,
        preRank: finiteNumber(row?.pre_rerank_rank),
      };
    })
    .filter(Boolean)
    .slice(0, 8);
  if (!rankedRows.length) {
    return null;
  }

  const maxScore = Math.max(0.000001, ...rankedRows.map((item) => Math.max(0, item.score)));

  return (
    <section className="retrieval-visual-block" data-testid="result-ranking-profile">
      <div className="retrieval-visual-heading">
        <div>
          <h3>Result ranking profile</h3>
          <p>Compare final relevance and see which results moved after reranking.</p>
        </div>
        <span className="retrieval-scale-note">relative to top result</span>
      </div>
      <div className="ranking-profile-list">
        {rankedRows.map(({ row, score, finalRank, preRank }, index) => {
          const movement = preRank === null ? 0 : preRank - finalRank;
          const label = row.video_title || row.video_id || `Result ${index + 1}`;
          return (
            <div className="ranking-profile-row" key={`${row.video_id || "result"}-${row.chunk_index ?? row.frame_id ?? index}`}>
              <span className="ranking-profile-rank">#{finalRank}</span>
              <span className="ranking-profile-title">{label}</span>
              <span className="ranking-profile-track" aria-hidden="true">
                <span
                  className="ranking-profile-fill"
                  style={{ width: `${Math.max(2, (Math.max(0, score) / maxScore) * 100)}%` }}
                />
              </span>
              <strong>{score.toFixed(3)}</strong>
              {preRank !== null ? (
                <span className={`ranking-movement ${movement > 0 ? "up" : movement < 0 ? "down" : "same"}`}>
                  {movement > 0 ? `↑${movement}` : movement < 0 ? `↓${Math.abs(movement)}` : "—"}
                  <span className="sr-only">
                    {movement > 0
                      ? `moved up ${movement} places`
                      : movement < 0
                        ? `moved down ${Math.abs(movement)} places`
                        : "rank unchanged"}
                  </span>
                </span>
              ) : null}
            </div>
          );
        })}
      </div>
    </section>
  );
}

function RetrievalVisuals({ response, rows }) {
  if (!response) {
    return null;
  }
  return (
    <div className="retrieval-visuals" aria-label="Retrieval diagnostics">
      <RetrievalAttemptTimeline details={response.retrieval_details} />
      <ResultRankingProfile rows={rows} />
    </div>
  );
}

function QAStudioPage({ locale }) {
  const [query, setQuery] = useState("");
  const [kSearch, setKSearch] = useState(5);
  const [searchMode, setSearchMode] = useState("hybrid");
  const [searchReranker, setSearchReranker] = useState("none");
  const [searchLoading, setSearchLoading] = useState(false);
  const [searchResponse, setSearchResponse] = useState(null);
  const [searchError, setSearchError] = useState("");

  const [question, setQuestion] = useState("");
  const [kAsk, setKAsk] = useState(5);
  const [askMode, setAskMode] = useState("hybrid");
  const [askAgentic, setAskAgentic] = useState(false);
  const [askReranker, setAskReranker] = useState("none");
  const [provider, setProvider] = useState("chatgpt");
  const [model, setModel] = useState("");
  const llmProviderOptions = useLlmProviderOptions();
  const [askLoading, setAskLoading] = useState(false);
  const [askResponse, setAskResponse] = useState(null);
  const [askError, setAskError] = useState("");
  const [showEvidence, setShowEvidence] = useState(true);
  const [reviewStateByKey, setReviewStateByKey] = useState({});
  const [reviewPendingByKey, setReviewPendingByKey] = useState({});

  const [playerState, setPlayerState] = useState({
    videoId: "",
    start: 0,
    title: "",
  });
  const [activeQaTool, setActiveQaTool] = useState("ask");

  function resultIdentity(row) {
    const videoId = String(row?.video_id || "").trim() || extractVideoId(row?.url) || extractVideoId(row?.video_url);
    if (!videoId) {
      return "";
    }
    if (row?.source_type === "ocr") {
      return `${videoId}:ocr:${row?.frame_id || row?.timestamp_sec || row?.start || 0}`;
    }
    const chunkIndex = row?.chunk_index;
    if (chunkIndex !== undefined && chunkIndex !== null && chunkIndex !== "") {
      const parsed = Number(chunkIndex);
      if (Number.isFinite(parsed)) {
        return `${videoId}:${Math.trunc(parsed)}`;
      }
    }
    const start = Number((row?.start_seconds ?? row?.start) || 0);
    const end = Number((row?.end_seconds ?? row?.end ?? row?.start_seconds ?? row?.start) || 0);
    return `${videoId}:${start.toFixed(3)}-${end.toFixed(3)}`;
  }

  function videoIdFromRow(row) {
    return String(row?.video_id || "").trim() || extractVideoId(row?.url) || extractVideoId(row?.video_url) || "";
  }

  function playFromResult(row) {
    if (row?.source_type === "ocr") {
      return;
    }
    const videoId = videoIdFromRow(row);
    if (!videoId) {
      return;
    }
    setPlayerState({
      videoId,
      start: Number((row?.start_seconds ?? row?.start) || 0),
      title: row?.video_title || videoId,
    });
  }

  async function saveReviewForRow(row, label, context) {
    const key = resultIdentity(row);
    const videoId = videoIdFromRow(row);
    if (!key || !videoId || reviewPendingByKey[key]) {
      return;
    }

    setReviewPendingByKey((prev) => ({ ...prev, [key]: true }));
    setReviewStateByKey((prev) => ({
      ...prev,
      [key]: {
        label: prev[key]?.label || null,
        message: qaAnswerText(locale, "reviewSaving"),
        tone: "pending",
      },
    }));

    try {
      await apiRequest("/v1/feedback/search-review", {
        method: "POST",
        body: {
          query: context.query || "(unspecified)",
          retrieval_mode: context.retrievalMode || "hybrid",
          label,
          video_id: videoId,
          chunk_index: row?.chunk_index ?? null,
          start: Number((row?.start_seconds ?? row?.start) || 0),
          end: Number((row?.end_seconds ?? row?.end ?? row?.start_seconds ?? row?.start) || 0),
          url: row?.url || "",
          video_title: row?.video_title || "",
          language: row?.language || "",
          score: optionalNumber(row?.score),
          dense_score: optionalNumber(row?.dense_score),
          lexical_score: optionalNumber(row?.lexical_score),
          hybrid_score: optionalNumber(row?.hybrid_score),
          rank: row?.rank ?? null,
          model: context.retrievalMode || "hybrid",
        },
      });
      setReviewStateByKey((prev) => ({
        ...prev,
        [key]: {
          label,
          message: qaAnswerText(locale, "reviewSaved"),
          tone: "ok",
        },
      }));
      markFeedbackRevision();
    } catch (error) {
      setReviewStateByKey((prev) => ({
        ...prev,
        [key]: {
          label: prev[key]?.label || null,
          message: qaAnswerText(locale, "reviewSaveFailed", {
            message: String(error?.message || error),
          }),
          tone: "error",
        },
      }));
    } finally {
      setReviewPendingByKey((prev) => {
        const next = { ...prev };
        delete next[key];
        return next;
      });
    }
  }

  async function runSearch(event) {
    event.preventDefault();
    if (!query.trim()) {
      return;
    }
    setSearchLoading(true);
    setSearchError("");
    setSearchResponse(null);
    try {
      const payload = await apiRequest("/v1/search", {
        method: "POST",
        body: {
          query: query.trim(),
          k: Number(kSearch || 5),
          retrieval_mode: searchMode,
          reranker: searchReranker,
        },
      });
      setSearchResponse(payload);
    } catch (error) {
      setSearchError(String(error?.message || error));
    } finally {
      setSearchLoading(false);
    }
  }

  async function runAsk(event) {
    event.preventDefault();
    if (!question.trim()) {
      return;
    }
    setAskLoading(true);
    setAskError("");
    setAskResponse(null);
    try {
      const payload = await apiRequest("/v1/ask", {
        method: "POST",
        body: {
          question: question.trim(),
          k: Number(kAsk || 5),
          retrieval_mode: askMode,
          provider,
          model: model || undefined,
          agentic: askAgentic,
          reranker: askReranker,
        },
      });
      setAskResponse(payload);
      setShowEvidence(true);
    } catch (error) {
      setAskError(String(error?.message || error));
    } finally {
      setAskLoading(false);
    }
  }

  const answerStatus = String(askResponse?.status || "").trim();
  const answerTone = answerStatusTone(answerStatus);
  const answerCitationCount = Array.isArray(askResponse?.citations)
    ? askResponse.citations.length
    : 0;
  const answerEvidence = Array.isArray(askResponse?.citations) && askResponse.citations.length
    ? askResponse.citations
    : Array.isArray(askResponse?.retrieved_chunks)
      ? askResponse.retrieved_chunks
      : [];
  const answerWarnings = Array.isArray(askResponse?.warnings) ? askResponse.warnings : [];
  const answerSummaryFields = [
    qaAnswerText(locale, "answerCitationCount", { count: answerCitationCount }),
    qaAnswerText(locale, "answerConfidence", {
      value: answerConfidenceLabel(askResponse?.confidence, locale),
    }),
    qaAnswerText(locale, "answerProvider", {
      value: askResponse?.provider || "-",
    }),
    qaAnswerText(locale, "answerModel", {
      value: askResponse?.model || "-",
    }),
  ];

  return (
    <>
      <header className="hero">
        <h1>Q&amp;A Studio</h1>
        <p className="subtitle">Run retrieval and grounded answer generation in one page.</p>
      </header>

      <section className="panel qa-workbench-panel">
        <div className="qa-workbench-head">
          <h2 className="section-title">
            <img src={activeQaTool === "ask" ? "/icons/icon-chat.svg" : "/icons/icon-search.svg"} alt="" aria-hidden="true" />
            <span>Q&amp;A Workbench</span>
          </h2>
          <div className="qa-tabs" role="tablist" aria-label="Q&A mode">
            <button
              className={`qa-tab ${activeQaTool === "ask" ? "active" : ""}`}
              type="button"
              role="tab"
              aria-selected={activeQaTool === "ask"}
              onClick={() => setActiveQaTool("ask")}
            >
              Ask
            </button>
            <button
              className={`qa-tab ${activeQaTool === "search" ? "active" : ""}`}
              type="button"
              role="tab"
              aria-selected={activeQaTool === "search"}
              onClick={() => setActiveQaTool("search")}
            >
              Search
            </button>
          </div>
        </div>

        {activeQaTool === "ask" ? (
          <div className="qa-tab-panel" role="tabpanel">
            <form className="qa-ask-form" onSubmit={runAsk}>
              <label className="ask-question-field">
                <span>Question</span>
                <textarea
                  rows="3"
                  value={question}
                  onChange={(event) => setQuestion(event.target.value)}
                  placeholder="Ask a grounded question and get an answer with citations"
                />
              </label>
              <div className="qa-controls-row ask-controls-row">
                <label>
                  <span>Top K</span>
                  <input
                    type="number"
                    min="1"
                    max="12"
                    value={kAsk}
                    onChange={(event) => setKAsk(event.target.value)}
                  />
                </label>
                <label>
                  <span>Retrieval Mode</span>
                  <select value={askMode} onChange={(event) => setAskMode(event.target.value)}>
                    <option value="hybrid">hybrid</option>
                    <option value="dense">dense</option>
                    <option value="lexical">lexical</option>
                  </select>
                </label>
                <label>
                  <span>Provider</span>
                  <select
                    value={provider}
                    onChange={(event) => {
                      setProvider(event.target.value);
                      setModel("");
                    }}
                  >
                    <option value="chatgpt">ChatGPT</option>
                    <option value="claude">Claude</option>
                    <option value="sakana">Sakana AI</option>
                  </select>
                </label>
                <ModelSelectLabel
                  providerOptions={llmProviderOptions}
                  provider={provider}
                  model={model}
                  onChange={setModel}
                />
                <button className="btn qa-primary-action" type="submit" disabled={askLoading}>
                  {askLoading ? "Generating..." : "Generate Answer"}
                </button>
              </div>
              <fieldset className="qa-addon-controls">
                <legend>Retrieval add-ons</legend>
                <label className="qa-addon-switch">
                  <input
                    type="checkbox"
                    checked={askAgentic}
                    onChange={(event) => setAskAgentic(event.target.checked)}
                  />
                  <span>
                    <strong>Agentic retry</strong>
                    <small>Retry weak evidence with a rewrite, mode switch, or broader top K.</small>
                  </span>
                </label>
                <label className="qa-addon-select">
                  <span>
                    <strong>Reranker</strong>
                    <small>Rescore candidates with a multilingual cross-encoder.</small>
                  </span>
                  <select
                    value={askReranker}
                    onChange={(event) => setAskReranker(event.target.value)}
                  >
                    <option value="none">Off</option>
                    <option value="cross_encoder">Cross-encoder</option>
                  </select>
                </label>
              </fieldset>
            </form>

            {askError ? (
              <div className="search-summary">
                {qaAnswerText(locale, "askFailed", { message: askError })}
              </div>
            ) : null}
            {askResponse ? (
              <>
                <div className="search-summary ask-summary-row">
                  <span className={`ask-status-pill ${answerTone}`} data-testid="answer-status">
                    {answerStatusLabel(answerStatus, locale)}
                  </span>
                  {answerSummaryFields.map((value) => (
                    <span key={value}>{value}</span>
                  ))}
                </div>
                <div className="ask-trust-note">{qaAnswerText(locale, "answerTrustNote")}</div>
                <article
                  className={`ask-answer ask-answer-${answerStatus || "default"}`}
                  data-testid="answer-panel"
                >
                  {askResponse.answer || "-"}
                </article>
                {answerWarnings.length ? (
                  <div className="ask-warning-list">
                    {answerWarnings.map((warning, index) => (
                      <div className="ask-warning" key={`warning-${index}`}>{warning}</div>
                    ))}
                  </div>
                ) : null}
                <RetrievalVisuals response={askResponse} rows={answerEvidence} />
                <div className="ask-evidence-header">
                  <h3 className="ask-evidence-title">{qaAnswerText(locale, "answerSupportingEvidence")}</h3>
                  <button
                    className="btn secondary"
                    type="button"
                    onClick={() => setShowEvidence((current) => !current)}
                    data-testid="answer-evidence-toggle"
                  >
                    {showEvidence
                      ? qaAnswerText(locale, "answerHideEvidence")
                      : qaAnswerText(locale, "answerShowEvidence")}
                  </button>
                </div>
                {showEvidence ? (
                  <div className="search-cards">
                    {answerEvidence.map((row, index) => {
                      const key = resultIdentity(row);
                      const reviewState = key ? reviewStateByKey[key] : null;
                      const pending = !!(key && reviewPendingByKey[key]);
                      const active = reviewState?.label || null;
                      const sourceHref = safeExternalUrl(row.url);
                      const statusClass = reviewState?.tone
                        ? `review-status ${reviewState.tone}`
                        : "review-status";

                      return (
                        <article
                          className="search-card answer-citation-card"
                          key={`${row.video_id}-${row.chunk_index}-${row.citation_id || index}`}
                          data-testid="answer-citation-card"
                        >
                          <div className="search-card-head">
                            <div className="search-rank">[{row.citation_id ?? row.rank ?? index + 1}]</div>
                            <div className="search-title">{evidenceTitle(row)}</div>
                            <div className={`search-lang source-badge ${row.source_type === "ocr" ? "ocr" : "transcript"}`}>
                              {row.source_type || evidenceLanguage(row)}
                            </div>
                          </div>
                          <div className="search-meta">
                            <span>
                              {row.timestamp_range_label || evidenceTimestamp(row)}
                            </span>
                            {row.score !== undefined && row.score !== null ? (
                              <span>score {Number(row.score || 0).toFixed(4)}</span>
                            ) : null}
                          </div>
                          <p className="search-snippet">{row.snippet || row.text}</p>
                          {evidenceFramePath(row) ? <p className="frame-path">{evidenceFramePath(row)}</p> : null}
                          {row.reason ? <p className="citation-reason">{row.reason}</p> : null}
                          <div className="search-actions">
                            {row.source_type === "ocr" ? null : (
                              <button className="btn search-link-btn" type="button" onClick={() => playFromResult(row)}>
                                {qaAnswerText(locale, "answerPlayAtTimestamp")}
                              </button>
                            )}
                            {sourceHref ? (
                              <a className="citation-link" href={sourceHref} target="_blank" rel="noreferrer">
                                {qaAnswerText(locale, "answerOpenSource")}
                              </a>
                            ) : null}
                            {row.source_type === "ocr" ? null : (
                              <>
                                <div className="review-group">
                                  <button
                                    className={`btn secondary review-btn ${active === "relevant" ? "active relevant" : ""}`}
                                    type="button"
                                    disabled={pending}
                                    onClick={() => saveReviewForRow(row, "relevant", {
                                      query: question,
                                      retrievalMode: askMode,
                                    })}
                                  >
                                    {qaAnswerText(locale, "answerRelevant")}
                                  </button>
                                  <button
                                    className={`btn secondary review-btn ${active === "not_relevant" ? "active not-relevant" : ""}`}
                                    type="button"
                                    disabled={pending}
                                    onClick={() => saveReviewForRow(row, "not_relevant", {
                                      query: question,
                                      retrievalMode: askMode,
                                    })}
                                  >
                                    {qaAnswerText(locale, "answerNotRelevant")}
                                  </button>
                                </div>
                                {reviewState?.message ? <span className={statusClass}>{reviewState.message}</span> : null}
                              </>
                            )}
                          </div>
                        </article>
                      );
                    })}
                    {!answerEvidence.length ? (
                      <div className="search-empty">{qaAnswerText(locale, "answerNoEvidence")}</div>
                    ) : null}
                  </div>
                ) : null}
              </>
            ) : null}
          </div>
        ) : null}

        {activeQaTool === "search" ? (
          <div className="qa-tab-panel" role="tabpanel">
            <form className="qa-search-form" onSubmit={runSearch}>
              <label className="search-query-field">
                <span>Query</span>
                <input
                  value={query}
                  onChange={(event) => setQuery(event.target.value)}
                  placeholder="Search transcript evidence"
                />
              </label>
              <div className="qa-controls-row search-controls-row">
                <label>
                  <span>Top K</span>
                  <input
                    type="number"
                    min="1"
                    max="12"
                    value={kSearch}
                    onChange={(event) => setKSearch(event.target.value)}
                  />
                </label>
                <label>
                  <span>Retrieval Mode</span>
                  <select value={searchMode} onChange={(event) => setSearchMode(event.target.value)}>
                    <option value="hybrid">hybrid</option>
                    <option value="dense">dense</option>
                    <option value="lexical">lexical</option>
                  </select>
                </label>
                <button className="btn" type="submit" disabled={searchLoading}>
                  {searchLoading ? "Searching..." : "Run Search"}
                </button>
              </div>
              <fieldset className="qa-addon-controls qa-addon-controls-search">
                <legend>Retrieval add-ons</legend>
                <label className="qa-addon-select">
                  <span>
                    <strong>Reranker</strong>
                    <small>Rescore candidates with a multilingual cross-encoder.</small>
                  </span>
                  <select
                    value={searchReranker}
                    onChange={(event) => setSearchReranker(event.target.value)}
                  >
                    <option value="none">Off</option>
                    <option value="cross_encoder">Cross-encoder</option>
                  </select>
                </label>
              </fieldset>
            </form>

            {searchError ? <div className="search-summary">Search failed: {searchError}</div> : null}
            {!searchError && searchResponse ? (
              <div className="search-summary">
                {searchResponse.result_count} result(s) for "{searchResponse.query}"
              </div>
            ) : null}
            <RetrievalVisuals response={searchResponse} rows={searchResponse?.results || []} />
            <div className="search-cards">
              {(searchResponse?.results || []).map((row, index) => {
                const key = resultIdentity(row);
                const reviewState = key ? reviewStateByKey[key] : null;
                const pending = !!(key && reviewPendingByKey[key]);
                const active = reviewState?.label || null;
                const statusClass = reviewState?.tone ? `review-status ${reviewState.tone}` : "review-status";

                return (
                  <article className="search-card" key={`${row.video_id}-${row.chunk_index || row.frame_id || row.timestamp_sec}-${index}`}>
                    <div className="search-card-head">
                      <div className="search-rank">#{row.rank ?? index + 1}</div>
                      <div className="search-title">{evidenceTitle(row)}</div>
                      <div className={`search-lang source-badge ${row.source_type === "ocr" ? "ocr" : "transcript"}`}>
                        {row.source_type || evidenceLanguage(row)}
                      </div>
                    </div>
                    <div className="search-meta">
                      <span>{evidenceTimestamp(row)}</span>
                      <span>score {Number(row.score || 0).toFixed(4)}</span>
                    </div>
                    <p className="search-snippet">{row.text}</p>
                    {evidenceFramePath(row) ? <p className="frame-path">{evidenceFramePath(row)}</p> : null}
                    <div className="search-actions">
                      {row.source_type === "ocr" ? null : (
                        <>
                          <button className="btn search-link-btn" type="button" onClick={() => playFromResult(row)}>
                            Play at timestamp
                          </button>
                          <div className="review-group">
                            <button
                              className={`btn secondary review-btn ${active === "relevant" ? "active relevant" : ""}`}
                              type="button"
                              disabled={pending}
                              onClick={() => saveReviewForRow(row, "relevant", {
                                query,
                                retrievalMode: searchMode,
                              })}
                            >
                              Relevant
                            </button>
                            <button
                              className={`btn secondary review-btn ${active === "not_relevant" ? "active not-relevant" : ""}`}
                              type="button"
                              disabled={pending}
                              onClick={() => saveReviewForRow(row, "not_relevant", {
                                query,
                                retrievalMode: searchMode,
                              })}
                            >
                              Not Relevant
                            </button>
                          </div>
                          {reviewState?.message ? <span className={statusClass}>{reviewState.message}</span> : null}
                        </>
                      )}
                    </div>
                  </article>
                );
              })}
              {searchResponse && !searchResponse.results?.length ? (
                <div className="search-empty">No matching chunks found.</div>
              ) : null}
            </div>
          </div>
        ) : null}
      </section>

      <PlayerPanel videoId={playerState.videoId} startSeconds={playerState.start} title={playerState.title} />
    </>
  );
}

function TLDRStudioPage() {
  const [videos, setVideos] = useState([]);
  const [videoError, setVideoError] = useState("");
  const [isLoadingVideos, setIsLoadingVideos] = useState(true);
  const [selectedVideoId, setSelectedVideoId] = useState("");
  const [language, setLanguage] = useState("en");
  const [provider, setProvider] = useState("chatgpt");
  const [model, setModel] = useState("");
  const llmProviderOptions = useLlmProviderOptions();
  const [summaryResponse, setSummaryResponse] = useState(null);
  const [summaryError, setSummaryError] = useState("");
  const [isSummarizing, setIsSummarizing] = useState(false);

  const [playerState, setPlayerState] = useState({
    videoId: "",
    start: 0,
    title: "",
  });

  useEffect(() => {
    let cancelled = false;
    async function loadVideos() {
      setIsLoadingVideos(true);
      setVideoError("");
      try {
        const payload = await apiRequest("/v1/videos");
        if (cancelled) {
          return;
        }
        const rows = Array.isArray(payload?.videos) ? payload.videos : [];
        setVideos(rows);

        if (!rows.length) {
          setSelectedVideoId("");
          return;
        }

        const saved = String(readLocalStorage(LAST_VIDEO_KEY) || "").trim();
        const matched = rows.find((row) => String(row.video_id) === saved);
        setSelectedVideoId(matched ? matched.video_id : rows[0].video_id);
      } catch (error) {
        if (cancelled) {
          return;
        }
        setVideoError(String(error?.message || error));
      } finally {
        if (!cancelled) {
          setIsLoadingVideos(false);
        }
      }
    }

    loadVideos();
    return () => {
      cancelled = true;
    };
  }, []);

  const selectedVideo = useMemo(
    () => videos.find((row) => String(row.video_id) === String(selectedVideoId)) || null,
    [videos, selectedVideoId],
  );

  async function generateSummary(targetVideoId = selectedVideoId) {
    const scopedVideoId = String(targetVideoId || "").trim();
    if (!scopedVideoId) {
      return;
    }
    setSummaryError("");
    setIsSummarizing(true);
    try {
      const payload = await apiRequest("/v1/summaries/transcript", {
        method: "POST",
        body: {
          video_id: scopedVideoId,
          language,
          provider,
          model: model || undefined,
          max_points: 5,
        },
      });
      setSummaryResponse(payload);
      const first = payload?.summary?.[0];
      if (first) {
        setPlayerState({
          videoId: scopedVideoId,
          start: Number(first.start || 0),
          title: selectedVideo?.title || scopedVideoId,
        });
      }
      localStorage.setItem(LAST_VIDEO_KEY, scopedVideoId);
    } catch (error) {
      setSummaryError(String(error?.message || error));
    } finally {
      setIsSummarizing(false);
    }
  }

  const summaryItems = summaryResponse?.summary || [];

  function playSummaryItem(item) {
    const videoId = String(summaryResponse?.video_id || selectedVideoId || "").trim();
    if (!videoId) {
      return;
    }
    setPlayerState({
      videoId,
      start: Number(item.start || 0),
      title: selectedVideo?.title || videoId,
    });
  }

  return (
    <>
      <header className="hero">
        <h1>TLDR Studio</h1>
        <p className="subtitle">
          Generate the top 5 themes from the full transcript with richer, speaker-aware summaries when the transcript supports them.
        </p>
      </header>

      <section className="panel">
        <h2 className="section-title">
          <img src="/icons/icon-chat.svg" alt="" aria-hidden="true" />
          <span>Transcript TLDR</span>
        </h2>
        <form className="grid tldr-grid" onSubmit={(event) => { event.preventDefault(); generateSummary().catch(() => {}); }}>
          <label>
            <span>Video</span>
            <select
              value={selectedVideoId}
              onChange={(event) => {
                setSelectedVideoId(event.target.value);
                setSummaryResponse(null);
                setSummaryError("");
              }}
              disabled={isLoadingVideos || !videos.length}
            >
              {!videos.length ? <option value="">No videos</option> : null}
              {videos.map((row) => (
                <option key={row.video_id} value={row.video_id}>
                  {row.title} ({row.video_id})
                </option>
              ))}
            </select>
          </label>
          <label>
            <span>Output Language</span>
            <select value={language} onChange={(event) => setLanguage(event.target.value)}>
              <option value="en">English</option>
              <option value="ja">Japanese</option>
            </select>
          </label>
          <label>
            <span>Provider</span>
            <select
              value={provider}
              onChange={(event) => {
                setProvider(event.target.value);
                setModel("");
              }}
            >
              <option value="chatgpt">ChatGPT</option>
              <option value="claude">Claude</option>
              <option value="sakana">Sakana AI</option>
            </select>
          </label>
          <ModelSelectLabel
            providerOptions={llmProviderOptions}
            provider={provider}
            model={model}
            onChange={setModel}
          />
          <label>
            <span>Themes</span>
            <input value="5" disabled readOnly />
          </label>
          <button className="btn" type="submit" disabled={isSummarizing || !selectedVideoId}>
            {isSummarizing ? "Generating..." : "Generate TLDR"}
          </button>
        </form>
        {videoError ? <div className="search-summary">Video load failed: {videoError}</div> : null}
        {summaryError ? <div className="search-summary">TLDR failed: {summaryError}</div> : null}
      </section>

      <section className="panel">
        <h3 className="chart-heading">Top 5 Themes</h3>
        {!summaryItems.length ? (
          <div className="search-empty">Click Generate TLDR to create the top 5 themes.</div>
        ) : (
          <div className="search-cards">
            {summaryItems.map((item, index) => (
              <article
                key={`summary-${index}`}
                className="search-card summary-card"
              >
                <div className="search-card-head">
                  <div className="search-rank">#{item.rank ?? index + 1}</div>
                  <div className="search-title">{item.title}</div>
                  <div className="search-lang">{summaryResponse?.language || "-"}</div>
                </div>
                <div className="search-meta">
                  <span>Starts at {formatSeconds(item.start)}</span>
                </div>
                <p className="search-snippet">{item.tldr}</p>
                <div className="search-actions">
                  <button className="btn search-link-btn" type="button" onClick={() => playSummaryItem(item)}>
                    Play at timestamp
                  </button>
                </div>
              </article>
            ))}
          </div>
        )}
      </section>

      <PlayerPanel videoId={playerState.videoId} startSeconds={playerState.start} title={playerState.title} />
    </>
  );
}

function StudyStudioPage() {
  const [videos, setVideos] = useState([]);
  const [videoError, setVideoError] = useState("");
  const [isLoadingVideos, setIsLoadingVideos] = useState(true);
  const [selectedVideoId, setSelectedVideoId] = useState("");
  const [studyMode, setStudyMode] = useState("flashcards");
  const [language, setLanguage] = useState("en");
  const [provider, setProvider] = useState("chatgpt");
  const [model, setModel] = useState("");
  const [difficulty, setDifficulty] = useState("balanced");
  const [cardCount, setCardCount] = useState(8);
  const [focus, setFocus] = useState("");
  const [focusPreset, setFocusPreset] = useState("main_ideas");
  const [scope, setScope] = useState("whole_video");
  const [modelProfile, setModelProfile] = useState("balanced");
  const [topicDetailLevel, setTopicDetailLevel] = useState("brief");
  const [topicRank, setTopicRank] = useState(1);
  const [studyRuns, setStudyRuns] = useState(readStudyHistory);
  const [activeStudyRunId, setActiveStudyRunId] = useState("");
  const [studyError, setStudyError] = useState("");
  const [isGenerating, setIsGenerating] = useState(false);
  const llmProviderOptions = useLlmProviderOptions();
  const [playerState, setPlayerState] = useState({
    videoId: "",
    start: 0,
    title: "",
  });

  useEffect(() => {
    let cancelled = false;
    async function loadVideos() {
      setIsLoadingVideos(true);
      setVideoError("");
      try {
        const payload = await apiRequest("/v1/videos");
        if (cancelled) {
          return;
        }
        const rows = Array.isArray(payload?.videos) ? payload.videos : [];
        setVideos(rows);
        if (!rows.length) {
          setSelectedVideoId("");
          return;
        }
        const saved = String(localStorage.getItem(LAST_VIDEO_KEY) || "").trim();
        const matched = rows.find((row) => String(row.video_id) === saved);
        setSelectedVideoId(matched ? matched.video_id : rows[0].video_id);
      } catch (error) {
        if (!cancelled) {
          setVideoError(String(error?.message || error));
        }
      } finally {
        if (!cancelled) {
          setIsLoadingVideos(false);
        }
      }
    }
    loadVideos();
    return () => {
      cancelled = true;
    };
  }, []);

  const selectedVideo = useMemo(
    () => videos.find((row) => String(row.video_id) === String(selectedVideoId)) || null,
    [videos, selectedVideoId],
  );

  const modeMeta = {
    flashcards: {
      label: "Flashcards",
      action: "Generate Flashcards",
      loading: "Generating flashcards...",
    },
    topics: {
      label: "Topic Map",
      action: "Generate Topic Map",
      loading: "Generating topic map...",
    },
    quality: {
      label: "Quality",
      action: "Run Quality Evaluation",
      loading: "Evaluating deck quality...",
    },
  };
  const studyModes = Object.keys(modeMeta);
  const topicUsesLlm = studyMode === "topics" && topicDetailLevel === "explain";
  const llmControlsDisabled = studyMode !== "flashcards" && !topicUsesLlm;
  const activeStudyRun = useMemo(() => {
    const explicitRun = studyRuns.find((run) => run.id === activeStudyRunId && run.mode === studyMode);
    if (explicitRun) {
      return explicitRun;
    }
    return studyRuns.find((run) => run.mode === studyMode) || null;
  }, [activeStudyRunId, studyMode, studyRuns]);
  const studyResponse = activeStudyRun?.payload || null;
  const latestFlashcardsForVideo = useMemo(() => {
    const matchingRun = studyRuns.find((run) => {
      const cards = run?.payload?.deck?.cards;
      return (
        run.mode === "flashcards"
        && String(run.videoId || run.payload?.video_id || "") === String(selectedVideoId || "")
        && Array.isArray(cards)
        && cards.length
      );
    });
    return matchingRun?.payload?.deck?.cards || [];
  }, [selectedVideoId, studyRuns]);

  function selectStudyMode(mode) {
    setStudyMode(mode);
    setStudyError("");
  }

  function handleStudyTabKey(event, currentMode) {
    if (!["ArrowLeft", "ArrowRight", "Home", "End"].includes(event.key)) {
      return;
    }
    event.preventDefault();
    const currentIndex = studyModes.indexOf(currentMode);
    let nextIndex = currentIndex;
    if (event.key === "ArrowRight") {
      nextIndex = (currentIndex + 1) % studyModes.length;
    } else if (event.key === "ArrowLeft") {
      nextIndex = (currentIndex - 1 + studyModes.length) % studyModes.length;
    } else if (event.key === "Home") {
      nextIndex = 0;
    } else if (event.key === "End") {
      nextIndex = studyModes.length - 1;
    }
    selectStudyMode(studyModes[nextIndex]);
  }

  async function generateStudy(event) {
    event.preventDefault();
    const scopedVideoId = String(selectedVideoId || "").trim();
    if (!scopedVideoId) {
      return;
    }
    setStudyError("");
    setIsGenerating(true);
    const safeCardCount = Math.min(20, Math.max(4, Number(cardCount) || 8));
    try {
      const payload = await apiRequest("/v1/study/generate", {
        method: "POST",
        body: {
          mode: studyMode,
          video_id: scopedVideoId,
          language,
          provider: studyMode === "flashcards" || topicUsesLlm ? provider : "local",
          model: studyMode === "flashcards" || topicUsesLlm ? model || undefined : undefined,
          difficulty,
          card_count: safeCardCount,
          focus,
          focus_preset: focusPreset,
          scope,
          model_profile: modelProfile,
          topic_detail_level: studyMode === "topics" ? topicDetailLevel : undefined,
          topic_rank: studyMode === "topics" && topicDetailLevel === "explain" ? Number(topicRank) : undefined,
          cards: studyMode === "quality" && latestFlashcardsForVideo.length ? latestFlashcardsForVideo : undefined,
        },
      });
      const nextRun = {
        id: makeStudyRunId(),
        createdAt: new Date().toISOString(),
        mode: payload?.mode || studyMode,
        videoId: payload?.video_id || scopedVideoId,
        videoTitle: payload?.video_title || selectedVideo?.title || scopedVideoId,
        payload,
      };
      setStudyRuns((previousRuns) => {
        const nextRuns = [
          nextRun,
          ...previousRuns.filter((run) => run.id !== nextRun.id),
        ].slice(0, STUDY_HISTORY_LIMIT);
        writeStudyHistory(nextRuns);
        return nextRuns;
      });
      setActiveStudyRunId(nextRun.id);
      const firstEvidence =
        payload?.deck?.cards?.[0]?.evidence
        || payload?.topics?.[0]?.evidence
        || null;
      if (firstEvidence) {
        setPlayerState({
          videoId: scopedVideoId,
          start: Number(firstEvidence.start || 0),
          title: selectedVideo?.title || scopedVideoId,
        });
      }
      writeLocalStorage(LAST_VIDEO_KEY, scopedVideoId);
    } catch (error) {
      setStudyError(String(error?.message || error));
    } finally {
      setIsGenerating(false);
    }
  }

  function playEvidence(evidence) {
    const scopedVideoId = String(evidence?.video_id || selectedVideoId || "").trim();
    if (!scopedVideoId) {
      return;
    }
    setPlayerState({
      videoId: scopedVideoId,
      start: Number(evidence?.start || 0),
      title: evidence?.video_title || selectedVideo?.title || scopedVideoId,
    });
  }

  function restoreStudyRun(run) {
    if (!run?.id) {
      return;
    }
    const nextMode = run.mode || "flashcards";
    const nextVideoId = String(run.videoId || run.payload?.video_id || "").trim();
    setStudyMode(nextMode);
    setActiveStudyRunId(run.id);
    setStudyError("");
    if (nextVideoId && videos.some((row) => String(row.video_id) === nextVideoId)) {
      setSelectedVideoId(nextVideoId);
      writeLocalStorage(LAST_VIDEO_KEY, nextVideoId);
    }
    const firstEvidence =
      run.payload?.deck?.cards?.[0]?.evidence
      || run.payload?.topics?.[0]?.evidence
      || null;
    if (firstEvidence) {
      setPlayerState({
        videoId: nextVideoId || String(firstEvidence.video_id || selectedVideoId || ""),
        start: Number(firstEvidence.start || 0),
        title: firstEvidence.video_title || run.videoTitle || nextVideoId,
      });
    }
  }

  function clearStudyHistory() {
    setStudyRuns([]);
    setActiveStudyRunId("");
    writeStudyHistory([]);
    setStudyError("");
  }

  function formatStudyRunTime(value) {
    const parsed = new Date(value);
    if (Number.isNaN(parsed.getTime())) {
      return "";
    }
    return parsed.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
  }

  function studyRunMeta(run) {
    const payload = run?.payload || {};
    if (payload.mode === "flashcards") {
      const count = Array.isArray(payload.deck?.cards) ? payload.deck.cards.length : 0;
      return `${count} cards - ${payload.provider || "local"} / ${payload.model || "-"}`;
    }
    if (payload.mode === "topics") {
      const count = Array.isArray(payload.topics) ? payload.topics.length : 0;
      const detail = payload.topic_detail_level === "explain"
        ? `Explain #${payload.topic_rank || 1}`
        : "Brief map";
      return `${detail} - ${count} topic${count === 1 ? "" : "s"}`;
    }
    if (payload.mode === "quality") {
      const score = Number(payload.quality?.score || 0);
      return `${Math.round(score * 100)}% score`;
    }
    return payload.mode || "Study run";
  }

  const cards = Array.isArray(studyResponse?.deck?.cards) ? studyResponse.deck.cards : [];
  const topics = Array.isArray(studyResponse?.topics) ? studyResponse.topics : [];
  const quality = studyResponse?.quality || null;
  const selectedSections = Array.isArray(studyResponse?.evidence_pack?.selected_sections)
    ? studyResponse.evidence_pack.selected_sections
    : [];
  const focusLabel = studyResponse?.focus?.query
    || studyResponse?.focus?.preset_label
    || "Whole video";
  const activeMeta = modeMeta[studyMode] || modeMeta.flashcards;

  return (
    <>
      <header className="hero">
        <h1>Study Studio</h1>
        <p className="subtitle">
          Turn timestamped transcript evidence into flashcards, topic maps, and quality checks for a processed video.
        </p>
      </header>

      <section className="panel study-workbench-panel">
        <div className="qa-workbench-head">
          <h2 className="section-title">
            <img src="/icons/icon-library.svg" alt="" aria-hidden="true" />
            <span>Study Generator</span>
          </h2>
          <div className="qa-tabs study-tabs" role="tablist" aria-label="Study mode">
            {Object.entries(modeMeta).map(([mode, meta]) => (
              <button
                key={mode}
                id={`study-tab-${mode}`}
                className={`qa-tab ${studyMode === mode ? "active" : ""}`}
                type="button"
                role="tab"
                aria-selected={studyMode === mode}
                aria-controls={`study-panel-${mode}`}
                tabIndex={studyMode === mode ? 0 : -1}
                onClick={() => selectStudyMode(mode)}
                onKeyDown={(event) => handleStudyTabKey(event, mode)}
              >
                {meta.label}
              </button>
            ))}
          </div>
        </div>

        <form className="study-form" onSubmit={generateStudy}>
          <label className="study-video-field">
            <span>Video</span>
            <select
              value={selectedVideoId}
              onChange={(event) => {
                setSelectedVideoId(event.target.value);
                setStudyError("");
              }}
              disabled={isLoadingVideos || !videos.length}
              aria-busy={isLoadingVideos}
              aria-describedby="study-video-status"
            >
              {!videos.length ? <option value="">No videos</option> : null}
              {videos.map((row) => (
                <option key={row.video_id} value={row.video_id}>
                  {row.title} ({row.video_id})
                </option>
              ))}
            </select>
            <span className="sr-only" id="study-video-status" aria-live="polite">
              {isLoadingVideos ? "Loading videos." : `${videos.length} videos available.`}
            </span>
          </label>

          <div className="study-focus-row">
            <label className="study-focus-field">
              <span>Focus</span>
              <input
                type="text"
                value={focus}
                onChange={(event) => {
                  setFocus(event.target.value);
                }}
                placeholder="e.g. Serie character interpretation, quiz terms, discussion prompts"
              />
            </label>
            <label>
              <span>Preset</span>
              <select
                value={focusPreset}
                onChange={(event) => {
                  setFocusPreset(event.target.value);
                }}
              >
                {STUDY_FOCUS_PRESETS.map((row) => (
                  <option key={row.id} value={row.id}>{row.label}</option>
                ))}
              </select>
            </label>
            <label>
              <span>Scope</span>
              <select
                value={scope}
                onChange={(event) => {
                  setScope(event.target.value);
                }}
              >
                {STUDY_SCOPES.map((row) => (
                  <option key={row.id} value={row.id}>{row.label}</option>
                ))}
              </select>
            </label>
            <label>
              <span>Model Profile</span>
              <select
                value={modelProfile}
                onChange={(event) => {
                  setModelProfile(event.target.value);
                  setModel("");
                }}
              >
                {STUDY_MODEL_PROFILES.map((row) => (
                  <option key={row.id} value={row.id}>{row.label}</option>
                ))}
              </select>
            </label>
          </div>

          <div className="study-controls-row">
            <label>
              <span>Output Language</span>
              <select value={language} onChange={(event) => setLanguage(event.target.value)}>
                <option value="en">English</option>
                <option value="ja">Japanese</option>
              </select>
            </label>
            <label>
              <span>Topic Action</span>
              <select
                value={topicDetailLevel}
                onChange={(event) => {
                  setTopicDetailLevel(event.target.value);
                }}
                disabled={studyMode !== "topics"}
              >
                {STUDY_TOPIC_DETAIL_LEVELS.map((row) => (
                  <option key={row.id} value={row.id}>{row.label}</option>
                ))}
              </select>
            </label>
            <label>
              <span>Topic</span>
              <select
                value={topicRank}
                onChange={(event) => {
                  setTopicRank(event.target.value);
                }}
                disabled={studyMode !== "topics" || topicDetailLevel !== "explain"}
              >
                {[1, 2, 3, 4, 5].map((rank) => (
                  <option key={rank} value={rank}>#{rank}</option>
                ))}
              </select>
            </label>
            <label>
              <span>Provider</span>
              <select
                value={provider}
                onChange={(event) => {
                  setProvider(event.target.value);
                  setModel("");
                }}
                disabled={llmControlsDisabled}
              >
                <option value="chatgpt">ChatGPT</option>
                <option value="claude">Claude</option>
                <option value="sakana">Sakana AI</option>
              </select>
            </label>
            <ModelSelectLabel
              providerOptions={llmProviderOptions}
              provider={provider}
              model={model}
              onChange={setModel}
              disabled={llmControlsDisabled}
            />
            <label>
              <span>Difficulty</span>
              <select value={difficulty} onChange={(event) => setDifficulty(event.target.value)}>
                <option value="introductory">introductory</option>
                <option value="balanced">balanced</option>
                <option value="exam prep">exam prep</option>
                <option value="advanced review">advanced review</option>
              </select>
            </label>
            <label>
              <span>Cards</span>
              <input
                type="number"
                min="4"
                max="20"
                value={cardCount}
                onChange={(event) => setCardCount(event.target.value)}
                disabled={studyMode !== "flashcards"}
              />
            </label>
            <button className="btn study-primary-action" type="submit" disabled={isGenerating || !selectedVideoId}>
              {isGenerating ? activeMeta.loading : activeMeta.action}
            </button>
          </div>
        </form>
        {videoError ? <div className="search-summary">Video load failed: {videoError}</div> : null}
        {studyError ? <div className="search-summary">Study generation failed: {studyError}</div> : null}
        {studyRuns.length ? (
          <div className="study-history-wrap" aria-label="Study run history">
            <div className="study-history-head">
              <strong>Run History</strong>
              <button className="study-history-clear" type="button" onClick={clearStudyHistory}>
                Clear
              </button>
            </div>
            <div className="study-history-list">
              {studyRuns.map((run) => {
                const runModeLabel = modeMeta[run.mode]?.label || run.mode || "Study";
                const isActiveRun = activeStudyRun?.id === run.id;
                return (
                  <button
                    className={`study-history-item ${isActiveRun ? "active" : ""}`}
                    key={run.id}
                    type="button"
                    onClick={() => restoreStudyRun(run)}
                    aria-pressed={isActiveRun}
                  >
                    <strong>{runModeLabel}</strong>
                    <span>{formatStudyRunTime(run.createdAt)} - {run.videoTitle || run.videoId || "Video"}</span>
                    <em>{studyRunMeta(run)}</em>
                  </button>
                );
              })}
            </div>
          </div>
        ) : null}
        {studyResponse ? (
          <div className="search-summary ask-summary-row">
            <span>mode: {studyResponse.mode}</span>
            <span>provider: {studyResponse.provider}</span>
            <span>model: {studyResponse.model}</span>
            <span>profile: {studyResponse.focus?.model_profile_label || "-"}</span>
            {studyResponse.mode === "topics" ? (
              <span>topic: {studyResponse.topic_detail_level || "brief"}</span>
            ) : null}
            <span>focus: {focusLabel}</span>
            <span>segments: {studyResponse.source?.segment_count ?? "-"}</span>
            <span>chunks: {studyResponse.source?.chunk_count ?? "-"}</span>
            <span>sections: {studyResponse.evidence_pack?.selected_section_count ?? "-"}/{studyResponse.evidence_pack?.section_count ?? "-"}</span>
          </div>
        ) : null}
        {selectedSections.length ? (
          <div className="study-evidence-strip" aria-label="Selected study sections">
            {selectedSections.map((section) => (
              <button
                className="study-section-pill"
                key={section.section_id || `${section.rank}-${section.title}`}
                type="button"
                onClick={() => playEvidence(section)}
              >
                <strong>{section.title}</strong>
                <span>{section.timestamp || formatSeconds(section.start)}</span>
              </button>
            ))}
          </div>
        ) : null}
      </section>

      <section
        className="panel"
        role="tabpanel"
        id={`study-panel-${studyMode}`}
        aria-labelledby={`study-tab-${studyMode}`}
      >
        <h3 className="chart-heading">{activeMeta.label} Output</h3>
        {!studyResponse ? (
          <div className="search-empty">Choose a mode and generate study material from an ingested video.</div>
        ) : null}

        {studyResponse?.mode === "flashcards" ? (
          <div className="search-cards study-card-grid">
            {cards.map((card, index) => (
              <article className="search-card study-flashcard" key={`study-card-${index}`}>
                <div className="search-card-head">
                  <div className="search-rank">#{index + 1}</div>
                  <div className="search-title">{card.question}</div>
                  <div className="search-lang">{card.card_type || "card"}</div>
                </div>
                <div className="study-card-meta">
                  {card.learning_objective ? <span>{card.learning_objective}</span> : null}
                </div>
                <p className="study-answer">{card.answer}</p>
                <p className="search-snippet">{card.explanation}</p>
                {card.why_it_matters ? (
                  <p className="study-why">Why it matters: {card.why_it_matters}</p>
                ) : null}
                {card.language_note ? (
                  <p className="study-language-note">{card.language_note}</p>
                ) : null}
                <div className="chip-row">
                  {(card.tags || []).map((tag) => (
                    <span className="chip" key={`${index}-${tag}`}>{tag}</span>
                  ))}
                </div>
                <p className="citation-reason">Source cue: {card.source_cue}</p>
                <div className="search-actions">
                  <button className="btn search-link-btn" type="button" onClick={() => playEvidence(card.evidence)}>
                    Play source {card.evidence?.timestamp || ""}
                  </button>
                  {safeExternalUrl(card.evidence?.url) ? (
                    <a className="citation-link" href={card.evidence.url} target="_blank" rel="noreferrer">
                      Open source
                    </a>
                  ) : null}
                </div>
              </article>
            ))}
          </div>
        ) : null}

        {studyResponse?.mode === "topics" ? (
          <div className="search-cards">
            {topics.map((topic, index) => (
              <article className="search-card summary-card" key={`study-topic-${index}`}>
                <div className="search-card-head">
                  <div className="search-rank">#{topic.rank ?? index + 1}</div>
                  <div className="search-title">{topic.title}</div>
                  <div className="search-lang">{formatSeconds(topic.start)}</div>
                </div>
                <p className="search-snippet">{topic.tldr}</p>
                {Array.isArray(topic.who_is_speaking) && topic.who_is_speaking.length ? (
                  <div className="study-topic-speakers">
                    {topic.who_is_speaking.map((speaker) => (
                      <div className="study-topic-speaker" key={`${topic.rank}-${speaker.name}-${speaker.role}`}>
                        <strong>{speaker.name}</strong>
                        {speaker.role ? <span>{speaker.role}</span> : null}
                        {speaker.confidence ? <em>{speaker.confidence}</em> : null}
                      </div>
                    ))}
                  </div>
                ) : null}
                {topic.what_they_talked_about ? (
                  <p className="study-topic-detail">{topic.what_they_talked_about}</p>
                ) : null}
                {Array.isArray(topic.source_moments) && topic.source_moments.length ? (
                  <div className="study-source-moments">
                    {topic.source_moments.map((moment, momentIndex) => (
                      <div className="study-source-moment" key={`${topic.rank}-source-${momentIndex}`}>
                        <span>{moment.timestamp || formatSeconds(topic.start)}</span>
                        {moment.speaker ? <strong>{moment.speaker}</strong> : null}
                        <p>{moment.translation || moment.quote}</p>
                        {moment.quote && moment.translation ? <em>{moment.quote}</em> : null}
                        {moment.explanation ? <small>{moment.explanation}</small> : null}
                      </div>
                    ))}
                  </div>
                ) : null}
                {Array.isArray(topic.key_takeaways) && topic.key_takeaways.length ? (
                  <ul className="study-keypoints study-keymoments">
                    {topic.key_takeaways.map((point) => (
                      <li key={`${topic.rank}-takeaway-${point}`}>{point}</li>
                    ))}
                  </ul>
                ) : null}
                {Array.isArray(topic.key_points) && topic.key_points.length ? (
                  <ul className="study-keypoints">
                    {topic.key_points.map((point) => (
                      <li key={`${topic.rank}-${point}`}>{point}</li>
                    ))}
                  </ul>
                ) : null}
                {Array.isArray(topic.people_or_terms) && topic.people_or_terms.length ? (
                  <div className="chip-row">
                    {topic.people_or_terms.map((term) => (
                      <span className="chip" key={`${topic.rank}-${term}`}>{term}</span>
                    ))}
                  </div>
                ) : null}
                {Array.isArray(topic.review_questions) && topic.review_questions.length ? (
                  <ul className="study-keypoints study-review-questions">
                    {topic.review_questions.map((question) => (
                      <li key={`${topic.rank}-question-${question}`}>{question}</li>
                    ))}
                  </ul>
                ) : null}
                {topic.learning_context ? (
                  <p className="study-why">Learning context: {topic.learning_context}</p>
                ) : null}
                <p className="citation-reason">Source cue: {topic.anchor_text}</p>
                <div className="search-actions">
                  <button className="btn search-link-btn" type="button" onClick={() => playEvidence(topic.evidence)}>
                    Play source {formatSeconds(topic.start)}
                  </button>
                  {safeExternalUrl(topic.url) ? (
                    <a className="citation-link" href={topic.url} target="_blank" rel="noreferrer">
                      Open source
                    </a>
                  ) : null}
                </div>
              </article>
            ))}
          </div>
        ) : null}

        {studyResponse?.mode === "quality" && quality ? (
          <div className="study-quality-wrap">
            <div className="study-quality-score">
              <span>Quality score</span>
              <strong>{Math.round(Number(quality.score || 0) * 100)}%</strong>
              <em>{quality.verdict}</em>
            </div>
            <div className="study-metric-grid">
              {Object.entries(quality.metrics || {}).map(([key, value]) => (
                <div className="study-metric-card" key={key}>
                  <span>{key.replaceAll("_", " ")}</span>
                  <strong>{String(value)}</strong>
                </div>
              ))}
            </div>
            <div className="search-cards">
              {(quality.checks || []).map((check) => (
                <article className="search-card study-check-card" key={check.name}>
                  <div className="search-card-head">
                    <div className={`study-check-status ${check.status}`}>{check.status}</div>
                    <div className="search-title">{check.name}</div>
                    <div className="search-lang">{String(check.value)}</div>
                  </div>
                </article>
              ))}
            </div>
            <div className="chip-row">
              {(quality.recommendations || []).map((item) => (
                <span className="chip" key={item}>{item}</span>
              ))}
            </div>
          </div>
        ) : null}
      </section>

      <PlayerPanel videoId={playerState.videoId} startSeconds={playerState.start} title={playerState.title} />
    </>
  );
}

const STUDIO_MODES = [
  { route: ROUTES.QA, label: "Ask", icon: "/icons/icon-search.svg" },
  { route: ROUTES.STUDY, label: "Study", icon: "/icons/icon-library.svg" },
  { route: ROUTES.TLDR, label: "Summarize", icon: "/icons/icon-chat.svg" },
];

function StudioWorkspace({ route, locale }) {
  const activeMode = STUDIO_MODES.find((mode) => mode.route === route) || STUDIO_MODES[0];
  const panelId = `studio-panel-${activeMode.label.toLowerCase()}`;

  return (
    <section className="studio-workspace">
      <header className="hero studio-hero">
        <h1>Studio</h1>
        <p className="subtitle">Ask questions, build study material, or summarize an ingested video.</p>
      </header>
      <div className="studio-mode-tabs" role="tablist" aria-label="Studio mode">
        {STUDIO_MODES.map((mode) => {
          const active = mode.route === activeMode.route;
          return (
            <button
              className={`studio-mode-tab ${active ? "active" : ""}`}
              type="button"
              role="tab"
              aria-selected={active}
              aria-controls={active ? panelId : undefined}
              onClick={() => navigate(mode.route)}
              key={mode.route}
            >
              <img src={mode.icon} alt="" aria-hidden="true" />
              <span>{mode.label}</span>
            </button>
          );
        })}
      </div>
      <div className="studio-mode-content" id={panelId} role="tabpanel">
        {route === ROUTES.QA ? <QAStudioPage locale={locale} /> : null}
        {route === ROUTES.STUDY ? <StudyStudioPage /> : null}
        {route === ROUTES.TLDR ? <TLDRStudioPage /> : null}
      </div>
    </section>
  );
}

function App() {
  const [route, setRoute] = useState(readHashRoute());
  const [unlocked, setUnlocked] = useState(() => localStorage.getItem(LOCK_KEY) === "1");
  const [unlockChecked, setUnlockChecked] = useState(() => localStorage.getItem(LOCK_KEY) === "1");
  const [locale, setLocale] = useState(() => {
    const stored = String(localStorage.getItem(LOCALE_STORAGE_KEY) || "").trim();
    return stored === "ja-JP" ? "ja-JP" : "en-US";
  });
  const [routeAnimationKey, setRouteAnimationKey] = useState(0);
  const [showIntro, setShowIntro] = useState(() => {
    try {
      return sessionStorage.getItem(INTRO_SEEN_SESSION_KEY) !== "1";
    } catch (_) {
      return true;
    }
  });

  useEffect(() => {
    function onHashChange() {
      const next = readHashRoute();
      setRoute(next);
    }
    window.addEventListener("hashchange", onHashChange);
    if (!window.location.hash) {
      navigate(ROUTES.INGEST);
    }
    return () => {
      window.removeEventListener("hashchange", onHashChange);
    };
  }, []);

  useEffect(() => {
    if (unlocked) {
      setUnlockChecked(true);
      return undefined;
    }

    let cancelled = false;
    apiRequest("/v1/videos")
      .then((payload) => {
        if (cancelled) {
          return;
        }
        const rows = Array.isArray(payload?.videos) ? payload.videos : [];
        if (rows.length) {
          localStorage.setItem(LOCK_KEY, "1");
          setUnlocked(true);
        }
      })
      .catch(() => {
        // Keep the locked state when the library cannot be read.
      })
      .finally(() => {
        if (!cancelled) {
          setUnlockChecked(true);
        }
      });
    return () => {
      cancelled = true;
    };
  }, [unlocked]);

  useEffect(() => {
    if (
      unlockChecked
      && !unlocked
      && route !== ROUTES.INGEST
      && route !== ROUTES.LOCAL_VIDEO
    ) {
      navigate(ROUTES.INGEST);
    }
  }, [route, unlocked, unlockChecked]);

  useEffect(() => {
    setRouteAnimationKey((prev) => prev + 1);
  }, [route]);

  useEffect(() => {
    document.documentElement.lang = locale === "ja-JP" ? "ja" : "en";
    localStorage.setItem(LOCALE_STORAGE_KEY, locale);
  }, [locale]);

  useEffect(() => {
    if (!showIntro) {
      return undefined;
    }
    const timeoutMs = prefersReducedMotion() ? 70 : 1250;
    const timeoutId = window.setTimeout(() => {
      setShowIntro(false);
      try {
        sessionStorage.setItem(INTRO_SEEN_SESSION_KEY, "1");
      } catch (_) {
        // best effort only
      }
    }, timeoutMs);
    return () => {
      window.clearTimeout(timeoutId);
    };
  }, [showIntro]);

  function handleIngestSuccess(videoId) {
    localStorage.setItem(LOCK_KEY, "1");
    if (videoId) {
      localStorage.setItem(LAST_VIDEO_KEY, videoId);
    }
    setUnlocked(true);
    setUnlockChecked(true);
  }

  function isNavItemVisible(item) {
    return !item.requiresUnlock || unlocked;
  }

  function isNavItemActive(item) {
    if (!item.route) {
      return false;
    }
    const activeRoutes = Array.isArray(item.activeRoutes) ? item.activeRoutes : [item.route];
    return activeRoutes.includes(route);
  }

  function NavItem({ item, mobile = false }) {
    const active = isNavItemActive(item);
    const className = mobile ? `bottom-tab-link ${active ? "active" : ""}` : `app-nav-link nav-btn ${active ? "active" : ""}`;
    const content = (
      <>
        <img className={mobile ? "bottom-tab-icon" : "nav-icon"} src={item.icon} alt="" aria-hidden="true" />
        <span className={mobile ? "bottom-tab-label" : "nav-label"}>{item.label}</span>
      </>
    );

    if (item.route) {
      return (
        <button
          type="button"
          className={className}
          onClick={() => navigate(item.route)}
          aria-current={active ? "page" : undefined}
        >
          {content}
        </button>
      );
    }
    return (
      <a className={className} href={item.href}>
        {content}
      </a>
    );
  }

  function NavMenu({ label, icon, items, mobile = false }) {
    const visibleItems = items.filter(isNavItemVisible);
    if (!visibleItems.length) {
      return null;
    }

    if (mobile) {
      return (
        <details className="bottom-tools-menu" data-nav-group={label.toLowerCase()}>
          <summary className="bottom-tab-link">
            <img className="bottom-tab-icon" src={icon} alt="" aria-hidden="true" />
            <span className="bottom-tab-label">{label}</span>
          </summary>
          <div className="bottom-tools-panel">
            {visibleItems.map((item) => (
              <a className="bottom-tools-link" href={item.href} key={`bottom-tool-${item.key}`}>
                <img className="bottom-tab-icon" src={item.icon} alt="" aria-hidden="true" />
                <span>{item.label}</span>
              </a>
            ))}
          </div>
        </details>
      );
    }

    return (
      <details className="nav-tools-menu" data-nav-group={label.toLowerCase()}>
        <summary className="app-nav-link nav-btn nav-tools-summary">
          <img className="nav-icon" src={icon} alt="" aria-hidden="true" />
          <span className="nav-label">{label}</span>
        </summary>
        <div className="nav-tools-panel">
          {visibleItems.map((item) => (
            <a className="nav-tools-link" href={item.href} key={`tool-${item.key}`}>
              <img className="nav-icon" src={item.icon} alt="" aria-hidden="true" />
              <span>{item.label}</span>
            </a>
          ))}
        </div>
      </details>
    );
  }

  return (
    <div className={`react-shell-root ${showIntro ? "intro-playing" : ""}`}>
      {showIntro ? (
        <div className="site-intro" data-testid="intro-overlay" aria-hidden="true">
          <div className="site-intro-glow" />
          <div className="site-intro-content">
            <p className="site-intro-eyebrow">YouTube Transcript RAG</p>
            <p className="site-intro-title">Local Studio</p>
          </div>
        </div>
      ) : null}
      <header className="appbar">
        <a className="brand brand-link" href="#/ingest" onClick={() => navigate(ROUTES.INGEST)}>
          <img src="/icons/icon-play.svg" alt="" aria-hidden="true" />
          <div>
            <p className="eyebrow">YouTube Transcript RAG</p>
            <p className="brand-sub">Local Studio</p>
          </div>
        </a>
        <div className="appbar-actions">
          <nav className="appbar-nav" aria-label="Primary">
            {CORE_NAV_ITEMS.filter(isNavItemVisible).map((item) => (
              <NavItem key={item.key} item={item} />
            ))}
            <NavMenu label="Library" icon="/icons/icon-library.svg" items={LIBRARY_NAV_ITEMS} />
            <NavMenu label="More" icon="/icons/icon-jobs.svg" items={MORE_NAV_ITEMS} />
          </nav>
          <label className="locale-switch">
            <span>Language</span>
            <select value={locale} onChange={(event) => setLocale(event.target.value)}>
              <option value="en-US">English (US)</option>
              <option value="ja-JP">日本語</option>
            </select>
          </label>
        </div>
      </header>

      <main className={`shell react-shell-main ${route === ROUTES.INGEST ? "ingest-route" : ""}`}>
        <div className="route-scene" key={`scene-${route}-${routeAnimationKey}`}>
          {route === ROUTES.INGEST ? <IngestPage onSuccess={handleIngestSuccess} /> : null}
          {route === ROUTES.LOCAL_VIDEO ? <LocalVideoAnalysisPage locale={locale} /> : null}
          {route !== ROUTES.INGEST && route !== ROUTES.LOCAL_VIDEO && unlocked ? (
            <StudioWorkspace route={route} locale={locale} />
          ) : null}
        </div>
      </main>

      <nav className="bottom-tabbar" aria-label="Primary mobile navigation">
        {CORE_NAV_ITEMS.filter(isNavItemVisible).map((item) => (
          <NavItem key={`mobile-${item.key}`} item={item} mobile />
        ))}
        <NavMenu label="Library" icon="/icons/icon-library.svg" items={LIBRARY_NAV_ITEMS} mobile />
        <NavMenu label="More" icon="/icons/icon-jobs.svg" items={MORE_NAV_ITEMS} mobile />
      </nav>
    </div>
  );
}

export default App;
