import React, { useEffect, useMemo, useState } from "react";

const LOCK_KEY = "yt_rag_ingest_unlocked";
const LAST_VIDEO_KEY = "yt_rag_last_video_id";
const FEEDBACK_REVISION_STORAGE_KEY = "youtube-rag-feedback-revision";
const ROUTES = {
  INGEST: "/ingest",
  TLDR: "/tldr",
  QA: "/qa",
};

function readHashRoute() {
  const raw = String(window.location.hash || "").replace(/^#/, "").trim();
  if (!raw) {
    return ROUTES.INGEST;
  }
  if (raw === ROUTES.INGEST || raw === ROUTES.TLDR || raw === ROUTES.QA) {
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

function optionalNumber(value) {
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : null;
}

async function apiRequest(path, { method = "GET", body } = {}) {
  const response = await fetch(path, {
    method,
    cache: "no-store",
    headers: {
      "content-type": "application/json",
    },
    body: body ? JSON.stringify(body) : undefined,
  });
  const payload = await response.json();
  if (!response.ok) {
    throw new Error(payload?.error?.message || `Request failed (${response.status})`);
  }
  return payload;
}

function formatSeconds(value) {
  const seconds = Math.max(0, Math.floor(Number(value || 0)));
  const minutes = Math.floor(seconds / 60);
  const remaining = seconds % 60;
  return `${minutes}:${String(remaining).padStart(2, "0")}`;
}

function extractVideoId(value) {
  const raw = String(value || "").trim();
  if (/^[a-zA-Z0-9_-]{11}$/.test(raw)) {
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

function IngestPage({ onSuccess }) {
  const [url, setUrl] = useState("");
  const [mode, setMode] = useState("single");
  const [language, setLanguage] = useState("ja");
  const [force, setForce] = useState(false);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [status, setStatus] = useState("Enter a YouTube URL to unlock the workspace.");
  const [rawResponse, setRawResponse] = useState("");

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
      const firstVideo =
        payload?.jobs?.[0]?.video_id
        || payload?.skipped?.[0]?.video_id
        || "";
      setStatus("Ingest accepted. Preparing TLDR Studio...");
      setRawResponse(JSON.stringify(payload, null, 2));
      onSuccess(firstVideo);
    } catch (error) {
      setStatus(`Ingest failed: ${String(error?.message || error)}`);
      setRawResponse(String(error?.message || error));
    } finally {
      setIsSubmitting(false);
    }
  }

  return (
    <section className="ingest-stage">
      <header className="hero ingest-hero">
        <h1>Ingest Gateway</h1>
        <p className="subtitle">
          Start here. Once ingest succeeds, TLDR Studio, Q&amp;A Studio, Evaluation, and Reviews unlock.
        </p>
      </header>

      <section className="panel ingest-panel">
        <h2 className="section-title">
          <img src="/icons/icon-upload.svg" alt="" aria-hidden="true" />
          <span>Ingest</span>
        </h2>
        <p className="section-desc">Add a single video or playlist and unlock the rest of the workspace.</p>
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
    </section>
  );
}

function PlayerPanel({ videoId, startSeconds, title }) {
  if (!videoId) {
    return (
      <section className="panel">
        <h2 className="section-title">
          <img src="/icons/icon-play.svg" alt="" aria-hidden="true" />
          <span>YouTube Player</span>
        </h2>
        <div className="player-shell">
          <div className="player-placeholder">Select a timestamp to load the video.</div>
        </div>
      </section>
    );
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

function QAStudioPage() {
  const [query, setQuery] = useState("");
  const [kSearch, setKSearch] = useState(5);
  const [searchMode, setSearchMode] = useState("hybrid");
  const [searchLoading, setSearchLoading] = useState(false);
  const [searchResponse, setSearchResponse] = useState(null);
  const [searchError, setSearchError] = useState("");

  const [question, setQuestion] = useState("");
  const [kAsk, setKAsk] = useState(5);
  const [askMode, setAskMode] = useState("hybrid");
  const [provider, setProvider] = useState("chatgpt");
  const [askLoading, setAskLoading] = useState(false);
  const [askResponse, setAskResponse] = useState(null);
  const [askError, setAskError] = useState("");
  const [reviewStateByKey, setReviewStateByKey] = useState({});
  const [reviewPendingByKey, setReviewPendingByKey] = useState({});

  const [playerState, setPlayerState] = useState({
    videoId: "",
    start: 0,
    title: "",
  });

  function resultIdentity(row) {
    const videoId = extractVideoId(row?.video_id) || extractVideoId(row?.url) || extractVideoId(row?.video_url);
    if (!videoId) {
      return "";
    }
    const chunkIndex = row?.chunk_index;
    if (chunkIndex !== undefined && chunkIndex !== null && chunkIndex !== "") {
      const parsed = Number(chunkIndex);
      if (Number.isFinite(parsed)) {
        return `${videoId}:${Math.trunc(parsed)}`;
      }
    }
    const start = Number(row?.start || 0);
    const end = Number(row?.end || 0);
    return `${videoId}:${start.toFixed(3)}-${end.toFixed(3)}`;
  }

  function videoIdFromRow(row) {
    return extractVideoId(row?.video_id) || extractVideoId(row?.url) || extractVideoId(row?.video_url) || "";
  }

  function playFromResult(row) {
    const videoId = videoIdFromRow(row);
    if (!videoId) {
      return;
    }
    setPlayerState({
      videoId,
      start: Number(row?.start || 0),
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
        message: "Saving...",
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
          start: Number(row?.start || 0),
          end: Number(row?.end || 0),
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
          message: "Saved",
          tone: "ok",
        },
      }));
      markFeedbackRevision();
    } catch (error) {
      setReviewStateByKey((prev) => ({
        ...prev,
        [key]: {
          label: prev[key]?.label || null,
          message: `Save failed: ${String(error?.message || error)}`,
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
        },
      });
      setAskResponse(payload);
    } catch (error) {
      setAskError(String(error?.message || error));
    } finally {
      setAskLoading(false);
    }
  }

  return (
    <>
      <header className="hero">
        <h1>Q&amp;A Studio</h1>
        <p className="subtitle">Run retrieval and grounded answer generation in one page.</p>
      </header>

      <PlayerPanel videoId={playerState.videoId} startSeconds={playerState.start} title={playerState.title} />

      <section className="panel">
        <h2 className="section-title">
          <img src="/icons/icon-search.svg" alt="" aria-hidden="true" />
          <span>Search</span>
        </h2>
        <form className="grid qa-grid" onSubmit={runSearch}>
          <label>
            <span>Query</span>
            <input value={query} onChange={(event) => setQuery(event.target.value)} placeholder="Search transcript chunks" />
          </label>
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
        </form>

        {searchError ? <div className="search-summary">Search failed: {searchError}</div> : null}
        {!searchError && searchResponse ? (
          <div className="search-summary">
            {searchResponse.result_count} result(s) for "{searchResponse.query}"
          </div>
        ) : null}
        <div className="search-cards">
          {(searchResponse?.results || []).map((row, index) => {
            const key = resultIdentity(row);
            const reviewState = key ? reviewStateByKey[key] : null;
            const pending = !!(key && reviewPendingByKey[key]);
            const active = reviewState?.label || null;
            const statusClass = reviewState?.tone ? `review-status ${reviewState.tone}` : "review-status";

            return (
              <article className="search-card" key={`${row.video_id}-${row.chunk_index}-${index}`}>
                <div className="search-card-head">
                  <div className="search-rank">#{row.rank ?? index + 1}</div>
                  <div className="search-title">{row.video_title || row.video_id}</div>
                  <div className="search-lang">{row.language || "-"}</div>
                </div>
                <div className="search-meta">
                  <span>{formatSeconds(row.start)} - {formatSeconds(row.end)}</span>
                  <span>score {Number(row.score || 0).toFixed(4)}</span>
                </div>
                <p className="search-snippet">{row.text}</p>
                <div className="search-actions">
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
                </div>
              </article>
            );
          })}
          {searchResponse && !searchResponse.results?.length ? (
            <div className="search-empty">No matching chunks found.</div>
          ) : null}
        </div>
      </section>

      <section className="panel">
        <h2 className="section-title">
          <img src="/icons/icon-chat.svg" alt="" aria-hidden="true" />
          <span>Ask</span>
        </h2>
        <form className="grid qa-grid ask-grid" onSubmit={runAsk}>
          <label>
            <span>Question</span>
            <input
              value={question}
              onChange={(event) => setQuestion(event.target.value)}
              placeholder="Ask with citations"
            />
          </label>
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
            <select value={provider} onChange={(event) => setProvider(event.target.value)}>
              <option value="chatgpt">ChatGPT</option>
              <option value="claude">Claude</option>
            </select>
          </label>
          <button className="btn" type="submit" disabled={askLoading}>
            {askLoading ? "Generating..." : "Generate Answer"}
          </button>
        </form>

        {askError ? <div className="search-summary">Ask failed: {askError}</div> : null}
        {askResponse ? (
          <>
            <div className="search-summary">
              {(askResponse.sources || []).length} source chunk(s), provider: {askResponse.provider}, model: {askResponse.model}
            </div>
            <article className="ask-answer">{askResponse.answer || "-"}</article>
            <div className="search-cards">
              {(askResponse.sources || []).map((row, index) => {
                const key = resultIdentity(row);
                const reviewState = key ? reviewStateByKey[key] : null;
                const pending = !!(key && reviewPendingByKey[key]);
                const active = reviewState?.label || null;
                const statusClass = reviewState?.tone ? `review-status ${reviewState.tone}` : "review-status";

                return (
                  <article className="search-card" key={`${row.video_id}-${row.chunk_index}-${index}`}>
                    <div className="search-card-head">
                      <div className="search-rank">#{row.rank ?? index + 1}</div>
                      <div className="search-title">{row.video_title || row.video_id}</div>
                      <div className="search-lang">{row.language || "-"}</div>
                    </div>
                    <div className="search-meta">
                      <span>{formatSeconds(row.start)} - {formatSeconds(row.end)}</span>
                      <span>score {Number(row.score || 0).toFixed(4)}</span>
                    </div>
                    <p className="search-snippet">{row.text}</p>
                    <div className="search-actions">
                      <button className="btn search-link-btn" type="button" onClick={() => playFromResult(row)}>
                        Play at timestamp
                      </button>
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
                          Relevant
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
                          Not Relevant
                        </button>
                      </div>
                      {reviewState?.message ? <span className={statusClass}>{reviewState.message}</span> : null}
                    </div>
                  </article>
                );
              })}
            </div>
          </>
        ) : null}
      </section>
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

        const saved = String(localStorage.getItem(LAST_VIDEO_KEY) || "").trim();
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
        <p className="subtitle">Generate top 5 themes from the full transcript and jump to where each theme starts.</p>
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
            <select value={provider} onChange={(event) => setProvider(event.target.value)}>
              <option value="chatgpt">ChatGPT</option>
              <option value="claude">Claude</option>
            </select>
          </label>
          <button className="btn" type="submit" disabled={isSummarizing || !selectedVideoId}>
            {isSummarizing ? "Generating..." : "Generate TLDR"}
          </button>
        </form>
        {videoError ? <div className="search-summary">Video load failed: {videoError}</div> : null}
        {summaryError ? <div className="search-summary">TLDR failed: {summaryError}</div> : null}
      </section>

      <PlayerPanel videoId={playerState.videoId} startSeconds={playerState.start} title={playerState.title} />

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
    </>
  );
}

function App() {
  const [route, setRoute] = useState(readHashRoute());
  const [unlocked, setUnlocked] = useState(() => localStorage.getItem(LOCK_KEY) === "1");
  const [routeAnimationKey, setRouteAnimationKey] = useState(0);

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
    if (!unlocked && route !== ROUTES.INGEST) {
      navigate(ROUTES.INGEST);
    }
  }, [route, unlocked]);

  useEffect(() => {
    setRouteAnimationKey((prev) => prev + 1);
  }, [route]);

  function handleIngestSuccess(videoId) {
    localStorage.setItem(LOCK_KEY, "1");
    if (videoId) {
      localStorage.setItem(LAST_VIDEO_KEY, videoId);
    }
    setUnlocked(true);
    navigate(ROUTES.TLDR);
  }

  function NavButton({ target, label }) {
    const active = route === target;
    return (
      <button
        type="button"
        className={`app-nav-link nav-btn ${active ? "active" : ""}`}
        onClick={() => navigate(target)}
      >
        {label}
      </button>
    );
  }

  return (
    <div className="react-shell-root">
      <header className="appbar">
        <a className="brand brand-link" href="#/ingest" onClick={() => navigate(ROUTES.INGEST)}>
          <img src="/icons/icon-play.svg" alt="" aria-hidden="true" />
          <div>
            <p className="eyebrow">YouTube Transcript RAG</p>
            <p className="brand-sub">Local Studio</p>
          </div>
        </a>
        <div className="appbar-actions">
          <NavButton target={ROUTES.INGEST} label="Ingest" />
          {unlocked ? <NavButton target={ROUTES.TLDR} label="TLDR Studio" /> : null}
          {unlocked ? <NavButton target={ROUTES.QA} label="Q&A Studio" /> : null}
          {unlocked ? <a className="app-nav-link" href="/evaluation.html">Evaluation</a> : null}
          {unlocked ? <a className="app-nav-link" href="/reviews.html">Reviews</a> : null}
        </div>
      </header>

      <main className={`shell react-shell-main ${route === ROUTES.INGEST ? "ingest-route" : ""}`}>
        <div className="route-scene" key={`scene-${route}-${routeAnimationKey}`}>
          {route === ROUTES.INGEST ? <IngestPage onSuccess={handleIngestSuccess} /> : null}
          {route === ROUTES.TLDR && unlocked ? <TLDRStudioPage /> : null}
          {route === ROUTES.QA && unlocked ? <QAStudioPage /> : null}
        </div>
      </main>
    </div>
  );
}

export default App;
