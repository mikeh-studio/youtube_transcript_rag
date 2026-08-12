const els = {
  localeSelect: document.getElementById("localeSelect"),
  localeLabel: document.getElementById("localeLabel"),
  homeBrandLink: document.getElementById("homeBrandLink"),
  eyebrowText: document.getElementById("eyebrowText"),
  homeNavLink: document.getElementById("homeNavLink"),
  evaluationNavLink: document.getElementById("evaluationNavLink"),
  reviewsNavLink: document.getElementById("reviewsNavLink"),
  heroTitle: document.getElementById("heroTitle"),
  heroSubtitle: document.getElementById("heroSubtitle"),
  ingestHeading: document.getElementById("ingestHeading"),
  ingestDesc: document.getElementById("ingestDesc"),
  ingestUrlLabel: document.getElementById("ingestUrlLabel"),
  ingestModeLabel: document.getElementById("ingestModeLabel"),
  ingestLanguageLabel: document.getElementById("ingestLanguageLabel"),
  ingestForceLabel: document.getElementById("ingestForceLabel"),
  ingestForceText: document.getElementById("ingestForceText"),
  ingestSubmitBtn: document.getElementById("ingestSubmitBtn"),
  jobsHeading: document.getElementById("jobsHeading"),
  jobsDesc: document.getElementById("jobsDesc"),
  ingestForm: document.getElementById("ingestForm"),
  ingestUrl: document.getElementById("ingestUrl"),
  ingestMode: document.getElementById("ingestMode"),
  ingestLanguage: document.getElementById("ingestLanguage"),
  ingestForce: document.getElementById("ingestForce"),
  ingestResult: document.getElementById("ingestResult"),
  videosHeading: document.getElementById("videosHeading"),
  jobsTable: document.getElementById("jobsTable"),
  videosTable: document.getElementById("videosTable"),
  refreshJobs: document.getElementById("refreshJobs"),
  refreshVideos: document.getElementById("refreshVideos"),
  posterHeading: document.getElementById("posterHeading"),
  posterDesc: document.getElementById("posterDesc"),
  refreshPosters: document.getElementById("refreshPosters"),
  posterGrid: document.getElementById("posterGrid"),
  logsHeading: document.getElementById("logsHeading"),
  logsToggleBtn: document.getElementById("logsToggleBtn"),
  logsPanel: document.getElementById("logsPanel"),
  logsDesc: document.getElementById("logsDesc"),
  logsLevelLabel: document.getElementById("logsLevelLabel"),
  logsJobIdLabel: document.getElementById("logsJobIdLabel"),
  logsVideoIdLabel: document.getElementById("logsVideoIdLabel"),
  logsRefreshBtn: document.getElementById("logsRefreshBtn"),
  logsLevel: document.getElementById("logsLevel"),
  logsJobId: document.getElementById("logsJobId"),
  logsVideoId: document.getElementById("logsVideoId"),
  logsResult: document.getElementById("logsResult"),
  playerShell: document.getElementById("playerShell"),
  playerHeading: document.getElementById("playerHeading"),
  playerDesc: document.getElementById("playerDesc"),
  playerPlaceholder: document.getElementById("playerPlaceholder"),
  youtubePlayerFrame: document.getElementById("youtubePlayerFrame"),
  playerNowPlaying: document.getElementById("playerNowPlaying"),
  playerMuteToggle: document.getElementById("playerMuteToggle"),
  searchHeading: document.getElementById("searchHeading"),
  searchDesc: document.getElementById("searchDesc"),
  searchQueryLabel: document.getElementById("searchQueryLabel"),
  searchKLabel: document.getElementById("searchKLabel"),
  searchModeLabel: document.getElementById("searchModeLabel"),
  searchSubmitBtn: document.getElementById("searchSubmitBtn"),
  searchRawJsonSummary: document.getElementById("searchRawJsonSummary"),
  searchForm: document.getElementById("searchForm"),
  searchQuery: document.getElementById("searchQuery"),
  searchK: document.getElementById("searchK"),
  searchMode: document.getElementById("searchMode"),
  searchSummary: document.getElementById("searchSummary"),
  searchCards: document.getElementById("searchCards"),
  searchResult: document.getElementById("searchResult"),
  askHeading: document.getElementById("askHeading"),
  askDesc: document.getElementById("askDesc"),
  askQuestionLabel: document.getElementById("askQuestionLabel"),
  askKLabel: document.getElementById("askKLabel"),
  askModeLabel: document.getElementById("askModeLabel"),
  askProviderLabel: document.getElementById("askProviderLabel"),
  askSubmitBtn: document.getElementById("askSubmitBtn"),
  askForm: document.getElementById("askForm"),
  askQuestion: document.getElementById("askQuestion"),
  askK: document.getElementById("askK"),
  askMode: document.getElementById("askMode"),
  askProvider: document.getElementById("askProvider"),
  askSummary: document.getElementById("askSummary"),
  askAnswer: document.getElementById("askAnswer"),
  askSources: document.getElementById("askSources"),
  askRawJsonSummary: document.getElementById("askRawJsonSummary"),
  askResult: document.getElementById("askResult"),
  summaryHeading: document.getElementById("summaryHeading"),
  summaryDesc: document.getElementById("summaryDesc"),
  summaryForm: document.getElementById("summaryForm"),
  summaryVideoLabel: document.getElementById("summaryVideoLabel"),
  summaryVideoId: document.getElementById("summaryVideoId"),
  summaryLanguageLabel: document.getElementById("summaryLanguageLabel"),
  summaryLanguage: document.getElementById("summaryLanguage"),
  summarySubmitBtn: document.getElementById("summarySubmitBtn"),
  summaryStatus: document.getElementById("summaryStatus"),
  summaryCards: document.getElementById("summaryCards"),
  summaryRawJsonSummary: document.getElementById("summaryRawJsonSummary"),
  summaryResult: document.getElementById("summaryResult"),
};

const LOCALE_STORAGE_KEY = "youtube-rag-ui-locale";
const SEARCH_STATE_STORAGE_KEY = "youtube-rag-search-state-v1";
const FEEDBACK_REVISION_STORAGE_KEY = "youtube-rag-feedback-revision";
const DEFAULT_LOCALE = "en-US";
const VIDEO_ID_RE = /^[a-zA-Z0-9_-]{11}$/;

const I18N = {
  "en-US": {
    pageTitle: "YouTube Transcript Retrieval Lab | Local Preview",
    localeLabel: "Language",
    navHome: "Home",
    eyebrowText: "YouTube Transcript Retrieval Lab",
    navEvaluation: "Evaluation",
    navReviews: "Reviews",
    heroTitle: "Local Preview Console",
    heroSubtitle: "No Cloudflare setup needed. This runs on localhost only.",
    ingestHeading: "Ingest",
    ingestDesc: "Add single videos or playlists to build your searchable transcript library.",
    ingestUrlLabel: "YouTube URL",
    ingestUrlPlaceholder: "Video or playlist URL",
    ingestModeLabel: "Mode",
    ingestLanguageLabel: "Language",
    ingestForceLabel: "Force reingest",
    ingestSubmitBtn: "Run Ingestion",
    jobsHeading: "Jobs",
    jobsDesc: "Track ingestion status, failures, and retry outcomes for each queued video.",
    videosHeading: "Videos",
    posterHeading: "Poster Gallery",
    posterDesc: "Open video detail pages by clicking a poster card.",
    posterOpen: "Open details",
    postersEmpty: "No videos available yet.",
    refresh: "Refresh",
    logsHeading: "Troubleshooting Logs",
    logsShowBtn: "Show Troubleshooting Logs",
    logsHideBtn: "Hide Troubleshooting Logs",
    logsDesc: "Inspect ingest and jobs logs with quick filters for failures and retries.",
    logsLevelLabel: "Level",
    logsJobIdLabel: "Job ID",
    logsVideoIdLabel: "Video ID",
    logsLevelAll: "all",
    logsLevelError: "error",
    logsLevelInfo: "info",
    logsLevelDebug: "debug",
    logsJobIdPlaceholder: "job_xxx",
    logsVideoIdPlaceholder: "YouTube video id",
    logsRefreshBtn: "Refresh Logs",
    logsLoading: "Loading logs...",
    logsError: "Log load failed: {message}",
    playerHeading: "Player",
    playerDesc: "Play a result directly in the page. Timestamp actions update this player.",
    playerEmpty: "Run search and open a timestamp to load video.",
    playerNowPlaying: "Now playing: {title} ({time})",
    playerMute: "Mute",
    playerUnmute: "Unmute",
    playerError: "Player error: {message}",
    searchHeading: "Search",
    searchDesc: "Retrieve ranked transcript chunks using hybrid, dense, or lexical retrieval.",
    searchQueryLabel: "Query",
    searchQueryPlaceholder: "Search indexed chunks",
    searchKLabel: "Top K",
    searchModeLabel: "Retrieval Mode",
    searchSubmitBtn: "Run Search",
    searchSummaryDefault: "Run a query to see ranked chunks.",
    searchSummaryCount: "{count} result(s) for {query}",
    searchModeChip: "mode: {value}",
    searchDenseChip: "dense: {value}",
    searchLexicalChip: "lexical: {value}",
    searchFallbackChip: "fallback: {value}",
    searchNoMatch: "No matching chunks found.",
    searchLoadFailed: "Could not load results.",
    searchSearching: "Searching...",
    searchErrorPrefix: "Search failed: {message}",
    searchOpenTimestamp: "Play at timestamp",
    searchRawJsonSummary: "Raw JSON",
    reviewRelevant: "Relevant",
    reviewNotRelevant: "Not relevant",
    reviewSaving: "Saving...",
    reviewSaved: "Saved",
    reviewSaveFailed: "Save failed: {message}",
    askHeading: "Ask",
    askDesc: "Generate a grounded answer from retrieved chunks with source citations.",
    askQuestionLabel: "Question",
    askQuestionPlaceholder: "Ask with citations",
    askKLabel: "Top K",
    askModeLabel: "Retrieval Mode",
    askProviderLabel: "Provider",
    askProviderChatgpt: "ChatGPT",
    askProviderClaude: "Claude",
    askSubmitBtn: "Generate Answer",
    askGenerating: "Generating answer...",
    askSummaryDefault: "Run a question to generate an answer with sources.",
    askSummaryCount: "{count} source chunk(s) for {question}",
    askModeChip: "mode: {value}",
    askProviderChip: "provider: {value}",
    askModelChip: "model: {value}",
    askNoSources: "No sources found for this answer.",
    askRawJsonSummary: "Raw JSON",
    summaryHeading: "Transcript TLDR",
    summaryDesc: "Generate five timestamped highlights from a full transcript.",
    summaryVideoLabel: "Video",
    summaryLanguageLabel: "Output Language",
    summaryLangEnglish: "English",
    summaryLangJapanese: "Japanese",
    summarySelectVideo: "Select a video",
    summarySubmitBtn: "Generate TLDR",
    summaryGenerating: "Generating TLDR...",
    summaryStatusDefault: "Choose a video and generate TLDR highlights.",
    summaryStatusCount: "{count} TLDR point(s) for {video}",
    summaryProviderChip: "provider: {value}",
    summaryModelChip: "model: {value}",
    summaryLanguageChip: "language: {value}",
    summaryNoVideo: "No videos available. Ingest a video first.",
    summaryNoData: "No TLDR generated yet.",
    summaryPlayAtTimestamp: "Play at timestamp",
    summaryRawJsonSummary: "Raw JSON",
    summaryError: "Summary failed: {message}",
    jobsEmpty: "No data.",
    videosEmpty: "No data.",
    apiConnectionError: "API connection error: {message}",
    ingestRunning: "Running ingestion...",
    delete: "Delete",
    deleteError: "Delete failed: {message}",
    tableJobId: "job_id",
    tableVideoId: "video_id",
    tableLang: "lang",
    tableStatus: "status",
    tableAttempts: "attempts",
    tableError: "error",
    actions: "Actions",
    tableTitle: "title",
    tableChunks: "chunks",
    score: "score",
    dense: "dense",
    lexical: "lexical",
    rrf: "rrf",
    untitledVideo: "Untitled Video",
    askError: "Ask failed: {message}",
  },
  "ja-JP": {
    pageTitle: "YouTube Transcript Retrieval Lab | ローカルプレビュー",
    localeLabel: "言語",
    navHome: "ホーム",
    eyebrowText: "YouTube Transcript Retrieval Lab",
    navEvaluation: "評価",
    navReviews: "レビュー",
    heroTitle: "ローカルプレビューコンソール",
    heroSubtitle: "Cloudflare 設定なしで localhost で動作します。",
    ingestHeading: "取り込み",
    ingestDesc: "単体動画またはプレイリストを追加して、検索可能な文字起こしライブラリを作成します。",
    ingestUrlLabel: "YouTube URL",
    ingestUrlPlaceholder: "動画URLまたはプレイリストURL",
    ingestModeLabel: "モード",
    ingestLanguageLabel: "言語",
    ingestForceLabel: "再取り込みを強制",
    ingestSubmitBtn: "取り込み実行",
    jobsHeading: "ジョブ",
    jobsDesc: "動画ごとの取り込み状態、失敗、再試行結果を確認できます。",
    videosHeading: "動画",
    posterHeading: "ポスターギャラリー",
    posterDesc: "ポスターカードをクリックして詳細ページを開きます。",
    posterOpen: "詳細を見る",
    postersEmpty: "まだ動画がありません。",
    refresh: "更新",
    logsHeading: "トラブルシュートログ",
    logsShowBtn: "トラブルシュートログを表示",
    logsHideBtn: "トラブルシュートログを非表示",
    logsDesc: "取り込み・ジョブのログを確認し、失敗や再試行を絞り込めます。",
    logsLevelLabel: "レベル",
    logsJobIdLabel: "ジョブID",
    logsVideoIdLabel: "動画ID",
    logsLevelAll: "すべて",
    logsLevelError: "error",
    logsLevelInfo: "info",
    logsLevelDebug: "debug",
    logsJobIdPlaceholder: "job_xxx",
    logsVideoIdPlaceholder: "YouTube 動画ID",
    logsRefreshBtn: "ログ更新",
    logsLoading: "ログ読み込み中...",
    logsError: "ログ取得エラー: {message}",
    playerHeading: "プレイヤー",
    playerDesc: "検索結果をページ内で再生します。タイムスタンプ操作はこのプレイヤーを更新します。",
    playerEmpty: "検索を実行し、タイムスタンプを開くと動画を読み込みます。",
    playerNowPlaying: "再生中: {title} ({time})",
    playerMute: "ミュート",
    playerUnmute: "ミュート解除",
    playerError: "プレイヤーエラー: {message}",
    searchHeading: "検索",
    searchDesc: "ハイブリッド / dense / lexical の検索モードでチャンクをランキング表示します。",
    searchQueryLabel: "クエリ",
    searchQueryPlaceholder: "インデックス済みチャンクを検索",
    searchKLabel: "Top K",
    searchModeLabel: "検索モード",
    searchSubmitBtn: "検索実行",
    searchSummaryDefault: "クエリを実行するとランキング結果を表示します。",
    searchSummaryCount: "{query} の結果: {count} 件",
    searchModeChip: "モード: {value}",
    searchDenseChip: "dense: {value}",
    searchLexicalChip: "lexical: {value}",
    searchFallbackChip: "フォールバック: {value}",
    searchNoMatch: "一致するチャンクが見つかりませんでした。",
    searchLoadFailed: "結果を読み込めませんでした。",
    searchSearching: "検索中...",
    searchErrorPrefix: "検索エラー: {message}",
    searchOpenTimestamp: "タイムスタンプ再生",
    searchRawJsonSummary: "Raw JSON",
    reviewRelevant: "関連あり",
    reviewNotRelevant: "関連なし",
    reviewSaving: "保存中...",
    reviewSaved: "保存済み",
    reviewSaveFailed: "保存エラー: {message}",
    askHeading: "質問",
    askDesc: "取得したチャンクを根拠に、出典付きで回答を生成します。",
    askQuestionLabel: "質問",
    askQuestionPlaceholder: "出典付きで質問",
    askKLabel: "Top K",
    askModeLabel: "検索モード",
    askProviderLabel: "プロバイダー",
    askProviderChatgpt: "ChatGPT",
    askProviderClaude: "Claude",
    askSubmitBtn: "回答生成",
    askGenerating: "回答を生成中...",
    askSummaryDefault: "質問を実行すると、出典付き回答を表示します。",
    askSummaryCount: "{question} の出典チャンク: {count} 件",
    askModeChip: "モード: {value}",
    askProviderChip: "プロバイダー: {value}",
    askModelChip: "モデル: {value}",
    askNoSources: "この回答に対応する出典が見つかりませんでした。",
    askRawJsonSummary: "Raw JSON",
    summaryHeading: "Transcript TLDR",
    summaryDesc: "文字起こし全体からタイムスタンプ付きの要点5件を生成します。",
    summaryVideoLabel: "動画",
    summaryLanguageLabel: "出力言語",
    summaryLangEnglish: "英語",
    summaryLangJapanese: "日本語",
    summarySelectVideo: "動画を選択",
    summarySubmitBtn: "TLDR生成",
    summaryGenerating: "TLDRを生成中...",
    summaryStatusDefault: "動画を選択してTLDRを生成してください。",
    summaryStatusCount: "{video} のTLDR: {count} 件",
    summaryProviderChip: "プロバイダー: {value}",
    summaryModelChip: "モデル: {value}",
    summaryLanguageChip: "言語: {value}",
    summaryNoVideo: "動画がありません。先に取り込みを実行してください。",
    summaryNoData: "まだTLDRが生成されていません。",
    summaryPlayAtTimestamp: "タイムスタンプ再生",
    summaryRawJsonSummary: "Raw JSON",
    summaryError: "サマリー生成エラー: {message}",
    jobsEmpty: "データがありません。",
    videosEmpty: "データがありません。",
    apiConnectionError: "API接続エラー: {message}",
    ingestRunning: "取り込み中...",
    delete: "削除",
    deleteError: "削除エラー: {message}",
    tableJobId: "job_id",
    tableVideoId: "video_id",
    tableLang: "lang",
    tableStatus: "status",
    tableAttempts: "attempts",
    tableError: "error",
    actions: "操作",
    tableTitle: "title",
    tableChunks: "chunks",
    score: "score",
    dense: "dense",
    lexical: "lexical",
    rrf: "rrf",
    untitledVideo: "タイトルなし動画",
    askError: "質問エラー: {message}",
  },
};

let currentLocale = localStorage.getItem(LOCALE_STORAGE_KEY) || DEFAULT_LOCALE;
if (!I18N[currentLocale]) {
  currentLocale = DEFAULT_LOCALE;
}

let latestSearchResponse = null;
let latestSearchContext = {
  query: "",
  retrieval_mode: "hybrid",
};
let latestAskResponse = null;
let latestAskContext = {
  query: "",
  retrieval_mode: "hybrid",
};
let latestSummaryResponse = null;
let activePlayerState = null;
const reviewStateByKey = new Map();
const reviewPendingKeys = new Set();

let ytPlayer = null;
let ytPlayerReady = false;
let ytApiReadyPromise = null;
let pendingPlayback = null;
let playerMuted = false;
let logsPanelExpanded = false;

function t(key, vars = {}) {
  const base = I18N[currentLocale] || I18N[DEFAULT_LOCALE];
  const fallback = I18N[DEFAULT_LOCALE];
  let template = base[key] || fallback[key] || key;
  Object.entries(vars).forEach(([name, value]) => {
    template = template.replaceAll(`{${name}}`, String(value));
  });
  return template;
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

  const json = await response.json();
  if (!response.ok) {
    throw new Error(json?.error?.message || `Request failed (${response.status})`);
  }
  return json;
}

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll("\"", "&quot;")
    .replaceAll("'", "&#39;");
}

function formatSeconds(sec) {
  const seconds = Math.max(0, Math.floor(Number(sec || 0)));
  const mm = Math.floor(seconds / 60);
  const ss = seconds % 60;
  return `${mm}:${String(ss).padStart(2, "0")}`;
}

function truncate(value, max = 260) {
  const text = String(value || "").trim();
  if (text.length <= max) {
    return text;
  }
  return `${text.slice(0, max)}...`;
}

function optionalNumber(value) {
  if (value == null || value === "") {
    return null;
  }
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : null;
}

function extractVideoId(value) {
  const raw = String(value || "").trim();
  if (VIDEO_ID_RE.test(raw)) {
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

function resultIdentity(row) {
  const videoId = (
    extractVideoId(row?.video_id)
    || extractVideoId(row?.url)
    || extractVideoId(row?.video_url)
    || ""
  ).trim();
  if (!videoId) {
    return "";
  }

  if (row?.chunk_index != null && row?.chunk_index !== "") {
    const chunkIndex = Number(row.chunk_index);
    if (Number.isFinite(chunkIndex)) {
      return `${videoId}:${Math.trunc(chunkIndex)}`;
    }
  }

  const startMs = Math.max(0, Math.floor(Number(row?.start || 0) * 1000));
  const endMs = Math.max(0, Math.floor(Number(row?.end || 0) * 1000));
  return `${videoId}:${startMs}:${endMs}`;
}

function updateNowPlayingText() {
  if (!activePlayerState) {
    els.playerNowPlaying.textContent = "";
    return;
  }

  els.playerNowPlaying.textContent = t("playerNowPlaying", {
    title: activePlayerState.title,
    time: formatSeconds(activePlayerState.start),
  });
}

function updatePlayerMuteButton() {
  els.playerMuteToggle.textContent = playerMuted ? t("playerUnmute") : t("playerMute");
  els.playerMuteToggle.setAttribute("aria-pressed", playerMuted ? "true" : "false");
}

function applyPlayerMuteState() {
  if (!ytPlayer || !ytPlayerReady) {
    return;
  }
  if (playerMuted) {
    ytPlayer.mute();
  } else {
    ytPlayer.unMute();
  }
}

function playWithPlayer(videoId, start, autoplay) {
  if (!ytPlayer || !ytPlayerReady) {
    pendingPlayback = { videoId, start, autoplay };
    return;
  }

  ytPlayer.loadVideoById({
    videoId,
    startSeconds: start,
  });

  if (!autoplay) {
    ytPlayer.pauseVideo();
  }

  applyPlayerMuteState();
  els.playerShell.classList.add("player-active");
}

function loadYouTubeApi() {
  if (window.YT && window.YT.Player) {
    return Promise.resolve();
  }

  if (ytApiReadyPromise) {
    return ytApiReadyPromise;
  }

  ytApiReadyPromise = new Promise((resolve) => {
    const previousCallback = window.onYouTubeIframeAPIReady;
    window.onYouTubeIframeAPIReady = () => {
      if (typeof previousCallback === "function") {
        previousCallback();
      }
      resolve();
    };

    if (document.querySelector("script[data-youtube-iframe-api='1']")) {
      return;
    }

    const script = document.createElement("script");
    script.src = "https://www.youtube.com/iframe_api";
    script.async = true;
    script.dataset.youtubeIframeApi = "1";
    document.head.appendChild(script);
  });

  return ytApiReadyPromise;
}

async function ensurePlayer() {
  if (ytPlayer) {
    return ytPlayer;
  }

  await loadYouTubeApi();
  if (!(window.YT && window.YT.Player)) {
    throw new Error("YouTube IFrame API unavailable");
  }

  ytPlayer = new window.YT.Player("youtubePlayerFrame", {
    width: "100%",
    height: "100%",
    playerVars: {
      autoplay: 1,
      controls: 1,
      rel: 0,
      modestbranding: 1,
      playsinline: 1,
    },
    events: {
      onReady: () => {
        ytPlayerReady = true;
        els.playerMuteToggle.disabled = false;
        updatePlayerMuteButton();
        applyPlayerMuteState();

        if (pendingPlayback) {
          const queued = pendingPlayback;
          pendingPlayback = null;
          playWithPlayer(queued.videoId, queued.start, queued.autoplay);
        }
      },
      onError: (event) => {
        const code = event?.data ?? "unknown";
        els.playerNowPlaying.textContent = t("playerError", { message: String(code) });
      },
    },
  });

  return ytPlayer;
}

async function loadPlayerForResult(row, autoplay = true) {
  const videoId = extractVideoId(row?.video_id) || extractVideoId(row?.url) || extractVideoId(row?.video_url);
  if (!videoId) {
    return;
  }

  const start = Math.max(0, Math.floor(Number(row?.start || 0)));
  activePlayerState = {
    title: String(row?.video_title || t("untitledVideo")),
    start,
  };
  updateNowPlayingText();

  try {
    await ensurePlayer();
    playWithPlayer(videoId, start, autoplay);
  } catch (err) {
    els.playerNowPlaying.textContent = t("playerError", {
      message: String(err?.message || err),
    });
  }
}

function renderSearchResults(response) {
  const details = response?.retrieval_details || {};
  const chips = [
    t("searchModeChip", { value: response?.retrieval_mode || "-" }),
    t("searchDenseChip", { value: details.dense_candidates ?? 0 }),
    t("searchLexicalChip", { value: details.lexical_candidates ?? 0 }),
    details.fallback ? t("searchFallbackChip", { value: details.fallback }) : null,
  ].filter(Boolean);

  els.searchSummary.innerHTML = `
    <div>${escapeHtml(t("searchSummaryCount", { count: response?.result_count ?? 0, query: response?.query || "" }))}</div>
    <div class="chip-row">${chips.map((chip) => `<span class="chip">${escapeHtml(chip)}</span>`).join("")}</div>
  `;

  const rows = response?.results || [];
  if (!rows.length) {
    els.searchCards.innerHTML = `<div class="search-empty">${escapeHtml(t("searchNoMatch"))}</div>`;
    return;
  }

  els.searchCards.innerHTML = rows.map((row, index) => {
    const score = Number(row?.score ?? 0).toFixed(4);
    const dense = row?.dense_score != null ? Number(row.dense_score).toFixed(4) : null;
    const lexical = row?.lexical_score != null ? Number(row.lexical_score).toFixed(4) : null;
    const hybrid = row?.hybrid_score != null ? Number(row.hybrid_score).toFixed(5) : null;

    const scoreBits = [
      `${t("score")} ${score}`,
      dense ? `${t("dense")} ${dense}` : null,
      lexical ? `${t("lexical")} ${lexical}` : null,
      hybrid ? `${t("rrf")} ${hybrid}` : null,
    ].filter(Boolean);

    const key = resultIdentity(row);
    const reviewState = key ? reviewStateByKey.get(key) : null;
    const selectedLabel = reviewState?.label || null;
    const pending = key ? reviewPendingKeys.has(key) : false;

    const statusText = pending
      ? t("reviewSaving")
      : (reviewState?.message || "");
    const statusClass = pending
      ? "pending"
      : reviewState?.tone === "error"
      ? "error"
      : reviewState?.tone === "ok"
      ? "ok"
      : "";

    return `
      <article class="search-card">
        <div class="search-card-head">
          <div class="search-rank">#${escapeHtml(row?.rank ?? "-")}</div>
          <div class="search-title">${escapeHtml(row?.video_title || t("untitledVideo"))}</div>
          <div class="search-lang">${escapeHtml(row?.language || "-")}</div>
        </div>
        <div class="search-meta">
          <span>${escapeHtml(formatSeconds(row?.start))} - ${escapeHtml(formatSeconds(row?.end))}</span>
          <span>${escapeHtml(scoreBits.join(" · "))}</span>
        </div>
        <p class="search-snippet">${escapeHtml(truncate(row?.text, 340))}</p>
        <div class="search-actions">
          <button class="btn search-link-btn" type="button" data-action="play" data-result-index="${index}">${escapeHtml(t("searchOpenTimestamp"))}</button>
          <div class="review-group">
            <button class="btn secondary review-btn ${selectedLabel === "relevant" ? "active relevant" : ""}" type="button" data-action="review" data-review-label="relevant" data-result-index="${index}" ${pending ? "disabled" : ""}>${escapeHtml(t("reviewRelevant"))}</button>
            <button class="btn secondary review-btn ${selectedLabel === "not_relevant" ? "active not-relevant" : ""}" type="button" data-action="review" data-review-label="not_relevant" data-result-index="${index}" ${pending ? "disabled" : ""}>${escapeHtml(t("reviewNotRelevant"))}</button>
          </div>
          ${statusText ? `<span class="review-status ${statusClass}">${escapeHtml(statusText)}</span>` : ""}
        </div>
      </article>
    `;
  }).join("");
}

function renderAskResponse(response) {
  const details = response?.retrieval_details || {};
  const chips = [
    t("askModeChip", { value: response?.retrieval_mode || "-" }),
    t("askProviderChip", { value: response?.provider || "-" }),
    t("askModelChip", { value: response?.model || "-" }),
    t("searchDenseChip", { value: details.dense_candidates ?? 0 }),
    t("searchLexicalChip", { value: details.lexical_candidates ?? 0 }),
    details.fallback ? t("searchFallbackChip", { value: details.fallback }) : null,
  ].filter(Boolean);

  const question = latestAskContext.query || els.askQuestion.value.trim();
  const sources = Array.isArray(response?.sources) ? response.sources : [];
  els.askSummary.innerHTML = `
    <div>${escapeHtml(t("askSummaryCount", { count: sources.length, question }))}</div>
    <div class="chip-row">${chips.map((chip) => `<span class="chip">${escapeHtml(chip)}</span>`).join("")}</div>
  `;

  els.askAnswer.textContent = String(response?.answer || "").trim() || "-";

  if (!sources.length) {
    els.askSources.innerHTML = `<div class="search-empty">${escapeHtml(t("askNoSources"))}</div>`;
    return;
  }

  els.askSources.innerHTML = sources.map((row, index) => {
    const score = Number(row?.score ?? 0).toFixed(4);
    const dense = row?.dense_score != null ? Number(row.dense_score).toFixed(4) : null;
    const lexical = row?.lexical_score != null ? Number(row.lexical_score).toFixed(4) : null;
    const hybrid = row?.hybrid_score != null ? Number(row.hybrid_score).toFixed(5) : null;
    const scoreBits = [
      `${t("score")} ${score}`,
      dense ? `${t("dense")} ${dense}` : null,
      lexical ? `${t("lexical")} ${lexical}` : null,
      hybrid ? `${t("rrf")} ${hybrid}` : null,
    ].filter(Boolean);

    const key = resultIdentity(row);
    const reviewState = key ? reviewStateByKey.get(key) : null;
    const selectedLabel = reviewState?.label || null;
    const pending = key ? reviewPendingKeys.has(key) : false;
    const statusText = pending ? t("reviewSaving") : (reviewState?.message || "");
    const statusClass = pending
      ? "pending"
      : reviewState?.tone === "error"
      ? "error"
      : reviewState?.tone === "ok"
      ? "ok"
      : "";

    return `
      <article class="search-card">
        <div class="search-card-head">
          <div class="search-rank">#${escapeHtml(row?.rank ?? (index + 1))}</div>
          <div class="search-title">${escapeHtml(row?.video_title || t("untitledVideo"))}</div>
          <div class="search-lang">${escapeHtml(row?.language || "-")}</div>
        </div>
        <div class="search-meta">
          <span>${escapeHtml(formatSeconds(row?.start))} - ${escapeHtml(formatSeconds(row?.end))}</span>
          <span>${escapeHtml(scoreBits.join(" · "))}</span>
        </div>
        <p class="search-snippet">${escapeHtml(truncate(row?.text, 340))}</p>
        <div class="search-actions">
          <button class="btn search-link-btn" type="button" data-action="ask-play" data-result-index="${index}">${escapeHtml(t("searchOpenTimestamp"))}</button>
          <div class="review-group">
            <button class="btn secondary review-btn ${selectedLabel === "relevant" ? "active relevant" : ""}" type="button" data-action="ask-review" data-review-label="relevant" data-result-index="${index}" ${pending ? "disabled" : ""}>${escapeHtml(t("reviewRelevant"))}</button>
            <button class="btn secondary review-btn ${selectedLabel === "not_relevant" ? "active not-relevant" : ""}" type="button" data-action="ask-review" data-review-label="not_relevant" data-result-index="${index}" ${pending ? "disabled" : ""}>${escapeHtml(t("reviewNotRelevant"))}</button>
          </div>
          ${statusText ? `<span class="review-status ${statusClass}">${escapeHtml(statusText)}</span>` : ""}
        </div>
      </article>
    `;
  }).join("");
}

function renderSummaryVideoOptions(videos) {
  const rows = Array.isArray(videos) ? videos : [];
  const selected = els.summaryVideoId.value;

  const options = [
    `<option value="">${escapeHtml(t("summarySelectVideo"))}</option>`,
    ...rows.map((row) => {
      const videoId = String(row?.video_id || "").trim();
      const title = String(row?.title || t("untitledVideo")).trim();
      if (!videoId) {
        return "";
      }
      return `<option value="${escapeHtml(videoId)}">${escapeHtml(`${title} (${videoId})`)}</option>`;
    }),
  ].join("");

  els.summaryVideoId.innerHTML = options;
  if (selected && rows.some((row) => String(row?.video_id || "").trim() === selected)) {
    els.summaryVideoId.value = selected;
  } else if (rows.length) {
    els.summaryVideoId.value = String(rows[0]?.video_id || "");
  } else {
    els.summaryVideoId.value = "";
    if (!latestSummaryResponse) {
      els.summaryStatus.textContent = t("summaryNoVideo");
      els.summaryCards.innerHTML = `<div class="search-empty">${escapeHtml(t("summaryNoVideo"))}</div>`;
    }
  }
}

function renderSummaryResponse(response) {
  const rows = Array.isArray(response?.summary) ? response.summary : [];
  const selectedOption = els.summaryVideoId.options[els.summaryVideoId.selectedIndex];
  const selectedText = selectedOption?.textContent || response?.video_id || "-";

  const chips = [
    t("summaryLanguageChip", { value: response?.language || "-" }),
    t("summaryProviderChip", { value: response?.provider || "-" }),
    t("summaryModelChip", { value: response?.model || "-" }),
  ];

  els.summaryStatus.innerHTML = `
    <div>${escapeHtml(t("summaryStatusCount", { count: rows.length, video: selectedText }))}</div>
    <div class="chip-row">${chips.map((chip) => `<span class="chip">${escapeHtml(chip)}</span>`).join("")}</div>
  `;

  if (!rows.length) {
    els.summaryCards.innerHTML = `<div class="search-empty">${escapeHtml(t("summaryNoData"))}</div>`;
    return;
  }

  els.summaryCards.innerHTML = rows.map((row, index) => {
    const title = String(row?.title || `${index + 1}`);
    const tldrText = String(row?.tldr || "").trim();
    return `
      <article class="search-card summary-card">
        <div class="search-card-head">
          <div class="search-rank">#${escapeHtml(row?.rank ?? index + 1)}</div>
          <div class="search-title">${escapeHtml(title)}</div>
          <div class="search-lang">${escapeHtml(response?.language || "-")}</div>
        </div>
        <div class="search-meta">
          <span>${escapeHtml(formatSeconds(row?.start))} - ${escapeHtml(formatSeconds(row?.end))}</span>
        </div>
        <p class="search-snippet">${escapeHtml(tldrText)}</p>
        <div class="search-actions">
          <button class="btn search-link-btn" type="button" data-action="summary-play" data-summary-index="${index}">
            ${escapeHtml(t("summaryPlayAtTimestamp"))}
          </button>
        </div>
      </article>
    `;
  }).join("");
}

function renderTable(container, columns, rows, actions = null, emptyMessageKey = "jobsEmpty") {
  if (!rows.length) {
    container.innerHTML = `<p style="padding:0.7rem">${escapeHtml(t(emptyMessageKey))}</p>`;
    return;
  }

  const head = columns.map((col) => `<th>${escapeHtml(col.label)}</th>`).join("");
  const body = rows
    .map((row) => {
      const tds = columns.map((col) => `<td>${escapeHtml(row[col.key] ?? "")}</td>`).join("");
      const actionTd = actions ? `<td>${actions(row)}</td>` : "";
      return `<tr>${tds}${actionTd}</tr>`;
    })
    .join("");

  const actionHead = actions ? `<th>${escapeHtml(t("actions"))}</th>` : "";
  container.innerHTML = `
    <table>
      <thead><tr>${head}${actionHead}</tr></thead>
      <tbody>${body}</tbody>
    </table>
  `;
}

function posterThumbnailUrl(videoId) {
  const safeVideoId = encodeURIComponent(String(videoId || "").trim());
  return `https://i.ytimg.com/vi/${safeVideoId}/hqdefault.jpg`;
}

function renderPosterGallery(rows) {
  if (!Array.isArray(rows) || !rows.length) {
    els.posterGrid.innerHTML = `<div class="search-empty">${escapeHtml(t("postersEmpty"))}</div>`;
    return;
  }

  els.posterGrid.innerHTML = rows.map((row) => {
    const videoId = String(row?.video_id || "").trim();
    if (!videoId) {
      return "";
    }
    const title = String(row?.title || t("untitledVideo"));
    const href = `./video_detail.html?video_id=${encodeURIComponent(videoId)}`;
    return `
      <a class="poster-card" href="${href}" data-testid="poster-card" data-video-id="${escapeHtml(videoId)}">
        <img class="poster-image" src="${posterThumbnailUrl(videoId)}" alt="${escapeHtml(title)} poster" loading="lazy" />
        <div class="poster-content">
          <h3 class="poster-title" data-testid="poster-title">${escapeHtml(title)}</h3>
          <p class="poster-meta">${escapeHtml(videoId)}</p>
          <span class="poster-link">${escapeHtml(t("posterOpen"))}</span>
        </div>
      </a>
    `;
  }).join("");
}

function saveSearchState(response = latestSearchResponse) {
  const payload = {
    query: els.searchQuery.value,
    k: Number(els.searchK.value || 5),
    retrieval_mode: els.searchMode.value,
    response: response || null,
    saved_at: Date.now(),
  };
  try {
    localStorage.setItem(SEARCH_STATE_STORAGE_KEY, JSON.stringify(payload));
  } catch (_) {
    // Ignore storage quota and serialization issues in local preview.
  }
}

function restoreSearchState() {
  try {
    const raw = localStorage.getItem(SEARCH_STATE_STORAGE_KEY);
    if (!raw) {
      return false;
    }
    const saved = JSON.parse(raw);
    if (!saved || typeof saved !== "object") {
      return false;
    }

    if (typeof saved.query === "string") {
      els.searchQuery.value = saved.query;
    }
    if (saved.k != null) {
      const parsedK = Number(saved.k);
      if (Number.isFinite(parsedK)) {
        els.searchK.value = String(Math.max(1, Math.min(12, Math.floor(parsedK))));
      }
    }
    if (typeof saved.retrieval_mode === "string" && saved.retrieval_mode) {
      els.searchMode.value = saved.retrieval_mode;
    }

    const response = saved.response;
    if (response && typeof response === "object" && Array.isArray(response.results)) {
      latestSearchResponse = response;
      latestSearchContext = {
        query: response?.query || els.searchQuery.value.trim(),
        retrieval_mode: response?.retrieval_mode || els.searchMode.value,
      };
      renderSearchResults(response);
      els.searchResult.textContent = JSON.stringify(response, null, 2);
      return true;
    }
    return false;
  } catch (_) {
    // Ignore malformed local storage snapshots.
    return false;
  }
}

function markFeedbackRevision() {
  try {
    localStorage.setItem(FEEDBACK_REVISION_STORAGE_KEY, String(Date.now()));
  } catch (_) {
    // Ignore storage write errors.
  }
}

async function hydrateReviewStateFromServer(rows) {
  if (!Array.isArray(rows) || !rows.length) {
    return;
  }

  try {
    const payload = await apiRequest("/v1/feedback/search-review?limit=5000");
    const payloadRows = Array.isArray(payload?.reviews) ? payload.reviews : [];

    const allowedKeys = new Set();
    rows.forEach((row) => {
      const key = resultIdentity(row);
      if (key) {
        allowedKeys.add(key);
      }
    });

    rows.forEach((row) => {
      const key = String(row?.key || "").trim() || resultIdentity(row);
      if (!key || !allowedKeys.has(key)) {
        return;
      }
      if (reviewPendingKeys.has(key)) {
        return;
      }
      reviewStateByKey.set(key, {
        label: row.label,
        message: "",
        tone: "",
        updatedAt: row.updated_at || null,
      });
    });
  } catch (_) {
    // Ignore hydration failure and allow manual review save flow.
  }
}

function updateLogsToggleButton() {
  els.logsToggleBtn.textContent = logsPanelExpanded ? t("logsHideBtn") : t("logsShowBtn");
  els.logsToggleBtn.setAttribute("aria-expanded", logsPanelExpanded ? "true" : "false");
}

function setLogsPanelExpanded(expanded) {
  logsPanelExpanded = Boolean(expanded);
  els.logsPanel.hidden = !logsPanelExpanded;
  updateLogsToggleButton();
}

async function refreshLogs() {
  els.logsResult.textContent = t("logsLoading");
  try {
    const params = new URLSearchParams();
    params.set("limit", "200");
    if (els.logsLevel.value) {
      params.set("level", els.logsLevel.value);
    }
    if (els.logsJobId.value.trim()) {
      params.set("job_id", els.logsJobId.value.trim());
    }
    if (els.logsVideoId.value.trim()) {
      params.set("video_id", els.logsVideoId.value.trim());
    }

    const data = await apiRequest(`/v1/logs/ingest-jobs?${params.toString()}`);
    els.logsResult.textContent = JSON.stringify(data, null, 2);
  } catch (err) {
    els.logsResult.textContent = t("logsError", { message: String(err.message || err) });
  }
}

async function refreshJobs() {
  const data = await apiRequest("/v1/ingest/jobs");
  renderTable(
    els.jobsTable,
    [
      { key: "job_id", label: t("tableJobId") },
      { key: "video_id", label: t("tableVideoId") },
      { key: "language", label: t("tableLang") },
      { key: "status", label: t("tableStatus") },
      { key: "attempts", label: t("tableAttempts") },
      { key: "error_code", label: t("tableError") },
    ],
    data.jobs || [],
    null,
    "jobsEmpty",
  );
}

async function refreshVideos() {
  const data = await apiRequest("/v1/videos");
  const videos = data.videos || [];
  renderTable(
    els.videosTable,
    [
      { key: "video_id", label: t("tableVideoId") },
      { key: "title", label: t("tableTitle") },
      { key: "language", label: t("tableLang") },
      { key: "num_chunks", label: t("tableChunks") },
    ],
    videos,
    (row) => `<button class="btn secondary delete-video" data-video-id="${escapeHtml(row.video_id)}">${escapeHtml(t("delete"))}</button>`,
    "videosEmpty",
  );
  renderPosterGallery(videos);
  renderSummaryVideoOptions(videos);
}

async function runIngest(event) {
  event.preventDefault();
  els.ingestResult.textContent = t("ingestRunning");

  try {
    const response = await apiRequest("/v1/ingest/videos", {
      method: "POST",
      body: {
        url: els.ingestUrl.value.trim(),
        mode: els.ingestMode.value,
        language: els.ingestLanguage.value,
        force: els.ingestForce.checked,
      },
    });
    els.ingestResult.textContent = JSON.stringify(response, null, 2);
    await Promise.all([refreshJobs(), refreshVideos()]);
  } catch (err) {
    els.ingestResult.textContent = String(err.message || err);
  }
}

async function runSearch(event) {
  event.preventDefault();
  els.searchSummary.textContent = t("searchSearching");
  els.searchCards.innerHTML = "";
  els.searchResult.textContent = "";

  try {
    const response = await apiRequest("/v1/search", {
      method: "POST",
      body: {
        query: els.searchQuery.value.trim(),
        k: Number(els.searchK.value || 5),
        retrieval_mode: els.searchMode.value,
      },
    });
    latestSearchResponse = response;
    latestSearchContext = {
      query: response?.query || els.searchQuery.value.trim(),
      retrieval_mode: response?.retrieval_mode || els.searchMode.value,
    };
    await hydrateReviewStateFromServer(response?.results || []);
    renderSearchResults(response);
    els.searchResult.textContent = JSON.stringify(response, null, 2);
    saveSearchState(response);
  } catch (err) {
    latestSearchResponse = null;
    els.searchSummary.textContent = t("searchErrorPrefix", { message: String(err.message || err) });
    els.searchCards.innerHTML = `<div class="search-empty">${escapeHtml(t("searchLoadFailed"))}</div>`;
    els.searchResult.textContent = String(err.message || err);
  }
}

async function runAsk(event) {
  event.preventDefault();
  els.askSummary.textContent = t("askGenerating");
  els.askAnswer.textContent = t("askGenerating");
  els.askSources.innerHTML = "";
  els.askResult.textContent = t("askGenerating");

  try {
    const question = els.askQuestion.value.trim();
    const retrievalMode = els.askMode.value;
    const response = await apiRequest("/v1/ask", {
      method: "POST",
      body: {
        question,
        k: Number(els.askK.value || 5),
        retrieval_mode: retrievalMode,
        provider: els.askProvider.value,
      },
    });
    latestAskResponse = response;
    latestAskContext = {
      query: question,
      retrieval_mode: response?.retrieval_mode || retrievalMode,
    };
    await hydrateReviewStateFromServer(response?.sources || []);
    renderAskResponse(response);
    els.askResult.textContent = JSON.stringify(response, null, 2);
  } catch (err) {
    latestAskResponse = null;
    els.askSummary.textContent = t("askSummaryDefault");
    els.askAnswer.textContent = t("askError", { message: String(err.message || err) });
    els.askSources.innerHTML = `<div class="search-empty">${escapeHtml(t("askNoSources"))}</div>`;
    els.askResult.textContent = t("askError", { message: String(err.message || err) });
  }
}

async function runSummary(event) {
  event.preventDefault();
  els.summaryStatus.textContent = t("summaryGenerating");
  els.summaryCards.innerHTML = "";
  els.summaryResult.textContent = t("summaryGenerating");

  const videoId = String(els.summaryVideoId.value || "").trim();
  if (!videoId) {
    latestSummaryResponse = null;
    els.summaryStatus.textContent = t("summaryNoVideo");
    els.summaryCards.innerHTML = `<div class="search-empty">${escapeHtml(t("summaryNoVideo"))}</div>`;
    els.summaryResult.textContent = t("summaryNoVideo");
    return;
  }

  try {
    const response = await apiRequest("/v1/summaries/transcript", {
      method: "POST",
      body: {
        video_id: videoId,
        language: els.summaryLanguage.value,
        provider: els.askProvider.value,
        max_points: 5,
      },
    });
    latestSummaryResponse = response;
    renderSummaryResponse(response);
    els.summaryResult.textContent = JSON.stringify(response, null, 2);
  } catch (err) {
    latestSummaryResponse = null;
    const message = t("summaryError", { message: String(err.message || err) });
    els.summaryStatus.textContent = message;
    els.summaryCards.innerHTML = `<div class="search-empty">${escapeHtml(message)}</div>`;
    els.summaryResult.textContent = message;
  }
}

async function onVideosClick(event) {
  const button = event.target.closest("button.delete-video");
  if (!button) {
    return;
  }

  const videoId = button.getAttribute("data-video-id");
  if (!videoId) {
    return;
  }

  button.disabled = true;
  try {
    await apiRequest(`/v1/videos/${encodeURIComponent(videoId)}`, {
      method: "DELETE",
    });
    await refreshVideos();
  } catch (err) {
    alert(t("deleteError", { message: String(err.message || err) }));
  } finally {
    button.disabled = false;
  }
}

async function saveReviewForRow(row, label, context, rerender) {
  const videoId = extractVideoId(row?.video_id) || extractVideoId(row?.url) || extractVideoId(row?.video_url);
  if (!videoId) {
    return;
  }

  const key = resultIdentity(row);
  if (!key || reviewPendingKeys.has(key)) {
    return;
  }

  const previous = reviewStateByKey.get(key) || null;
  reviewPendingKeys.add(key);
  reviewStateByKey.set(key, {
    label: previous?.label || null,
    message: t("reviewSaving"),
    tone: "pending",
  });
  rerender();

  try {
    const response = await apiRequest("/v1/feedback/search-review", {
      method: "POST",
      body: {
        query: context.query || "(unspecified)",
        retrieval_mode: context.retrieval_mode || "hybrid",
        label,
        video_id: videoId,
        chunk_index: row.chunk_index ?? null,
        start: Number(row.start || 0),
        end: Number(row.end || 0),
        url: row.url || "",
        video_title: row.video_title || "",
        language: row.language || "",
        score: optionalNumber(row.score),
        dense_score: optionalNumber(row.dense_score),
        lexical_score: optionalNumber(row.lexical_score),
        hybrid_score: optionalNumber(row.hybrid_score),
        rank: row.rank ?? null,
        model: context.retrieval_mode || "hybrid",
      },
    });

    reviewStateByKey.set(key, {
      label,
      message: t("reviewSaved"),
      tone: "ok",
      updatedAt: response?.feedback?.updated_at || null,
    });
    markFeedbackRevision();
  } catch (err) {
    reviewStateByKey.set(key, {
      label: previous?.label || null,
      message: t("reviewSaveFailed", { message: String(err.message || err) }),
      tone: "error",
    });
  } finally {
    reviewPendingKeys.delete(key);
    rerender();
  }
}

function onSearchCardsClick(event) {
  const button = event.target.closest("button[data-action]");
  if (!button) {
    return;
  }

  const idx = Number(button.getAttribute("data-result-index"));
  if (!Number.isInteger(idx) || idx < 0) {
    return;
  }

  const row = latestSearchResponse?.results?.[idx];
  if (!row) {
    return;
  }

  const action = button.getAttribute("data-action");
  if (action === "play") {
    loadPlayerForResult(row, true).catch(() => {});
    return;
  }

  if (action === "review") {
    const label = button.getAttribute("data-review-label");
    if (label === "relevant" || label === "not_relevant") {
      saveReviewForRow(
        row,
        label,
        latestSearchContext,
        () => {
          if (latestSearchResponse) {
            renderSearchResults(latestSearchResponse);
          }
        },
      ).catch(() => {});
    }
  }
}

function onAskSourcesClick(event) {
  const button = event.target.closest("button[data-action]");
  if (!button) {
    return;
  }

  const idx = Number(button.getAttribute("data-result-index"));
  if (!Number.isInteger(idx) || idx < 0) {
    return;
  }

  const row = latestAskResponse?.sources?.[idx];
  if (!row) {
    return;
  }

  const action = button.getAttribute("data-action");
  if (action === "ask-play") {
    loadPlayerForResult(row, true).catch(() => {});
    return;
  }

  if (action === "ask-review") {
    const label = button.getAttribute("data-review-label");
    if (label === "relevant" || label === "not_relevant") {
      saveReviewForRow(
        row,
        label,
        latestAskContext,
        () => {
          if (latestAskResponse) {
            renderAskResponse(latestAskResponse);
          }
        },
      ).catch(() => {});
    }
  }
}

function onSummaryCardsClick(event) {
  const button = event.target.closest("button[data-action='summary-play']");
  if (!button) {
    return;
  }

  const idx = Number(button.getAttribute("data-summary-index"));
  if (!Number.isInteger(idx) || idx < 0) {
    return;
  }

  const row = latestSummaryResponse?.summary?.[idx];
  if (!row) {
    return;
  }

  const videoId = String(latestSummaryResponse?.video_id || els.summaryVideoId.value || "").trim();
  if (!videoId) {
    return;
  }

  const playbackRow = {
    video_id: videoId,
    video_title: els.summaryVideoId.options[els.summaryVideoId.selectedIndex]?.textContent || videoId,
    start: Number(row?.start || 0),
    end: Number(row?.end || row?.start || 0),
    url: row?.url || "",
  };
  loadPlayerForResult(playbackRow, true).catch(() => {});
}

function onPlayerMuteToggle() {
  if (!ytPlayer || !ytPlayerReady) {
    return;
  }

  if (playerMuted) {
    ytPlayer.unMute();
    playerMuted = false;
  } else {
    ytPlayer.mute();
    playerMuted = true;
  }
  updatePlayerMuteButton();
}

function applyLocale(locale, options = {}) {
  if (!I18N[locale]) {
    return;
  }
  currentLocale = locale;
  localStorage.setItem(LOCALE_STORAGE_KEY, locale);
  document.documentElement.lang = locale === "ja-JP" ? "ja" : "en";
  document.title = t("pageTitle");

  els.localeSelect.value = locale;
  els.localeLabel.textContent = t("localeLabel");
  els.homeBrandLink?.setAttribute("aria-label", t("navHome"));
  els.eyebrowText.textContent = t("eyebrowText");
  els.homeNavLink.textContent = t("navHome");
  els.evaluationNavLink.textContent = t("navEvaluation");
  els.reviewsNavLink.textContent = t("navReviews");
  els.heroTitle.textContent = t("heroTitle");
  els.heroSubtitle.textContent = t("heroSubtitle");

  els.ingestHeading.textContent = t("ingestHeading");
  els.ingestDesc.textContent = t("ingestDesc");
  els.ingestUrlLabel.textContent = t("ingestUrlLabel");
  els.ingestUrl.placeholder = t("ingestUrlPlaceholder");
  els.ingestModeLabel.textContent = t("ingestModeLabel");
  els.ingestLanguageLabel.textContent = t("ingestLanguageLabel");
  els.ingestForceText.textContent = t("ingestForceLabel");
  els.ingestSubmitBtn.textContent = t("ingestSubmitBtn");

  els.jobsHeading.textContent = t("jobsHeading");
  els.jobsDesc.textContent = t("jobsDesc");
  els.videosHeading.textContent = t("videosHeading");
  els.posterHeading.textContent = t("posterHeading");
  els.posterDesc.textContent = t("posterDesc");
  els.refreshJobs.textContent = t("refresh");
  els.refreshVideos.textContent = t("refresh");
  els.refreshPosters.textContent = t("refresh");
  if (!els.posterGrid.children.length) {
    els.posterGrid.innerHTML = `<div class="search-empty">${escapeHtml(t("postersEmpty"))}</div>`;
  }
  els.logsHeading.textContent = t("logsHeading");
  updateLogsToggleButton();
  els.logsDesc.textContent = t("logsDesc");
  els.logsLevelLabel.textContent = t("logsLevelLabel");
  els.logsJobIdLabel.textContent = t("logsJobIdLabel");
  els.logsVideoIdLabel.textContent = t("logsVideoIdLabel");
  els.logsRefreshBtn.textContent = t("logsRefreshBtn");
  els.logsLevel.innerHTML = `
    <option value="">${escapeHtml(t("logsLevelAll"))}</option>
    <option value="error">${escapeHtml(t("logsLevelError"))}</option>
    <option value="info">${escapeHtml(t("logsLevelInfo"))}</option>
    <option value="debug">${escapeHtml(t("logsLevelDebug"))}</option>
  `;
  els.logsJobId.placeholder = t("logsJobIdPlaceholder");
  els.logsVideoId.placeholder = t("logsVideoIdPlaceholder");

  els.playerHeading.textContent = t("playerHeading");
  els.playerDesc.textContent = t("playerDesc");
  els.playerPlaceholder.textContent = t("playerEmpty");
  updateNowPlayingText();
  updatePlayerMuteButton();

  els.searchHeading.textContent = t("searchHeading");
  els.searchDesc.textContent = t("searchDesc");
  els.searchQueryLabel.textContent = t("searchQueryLabel");
  els.searchQuery.placeholder = t("searchQueryPlaceholder");
  els.searchKLabel.textContent = t("searchKLabel");
  els.searchModeLabel.textContent = t("searchModeLabel");
  els.searchSubmitBtn.textContent = t("searchSubmitBtn");
  els.searchRawJsonSummary.textContent = t("searchRawJsonSummary");
  if (latestSearchResponse) {
    renderSearchResults(latestSearchResponse);
  } else {
    els.searchSummary.textContent = t("searchSummaryDefault");
  }

  els.askHeading.textContent = t("askHeading");
  els.askDesc.textContent = t("askDesc");
  els.askQuestionLabel.textContent = t("askQuestionLabel");
  els.askQuestion.placeholder = t("askQuestionPlaceholder");
  els.askKLabel.textContent = t("askKLabel");
  els.askModeLabel.textContent = t("askModeLabel");
  els.askProviderLabel.textContent = t("askProviderLabel");
  const selectedAskProvider = els.askProvider.value;
  els.askProvider.innerHTML = `
    <option value="chatgpt">${escapeHtml(t("askProviderChatgpt"))}</option>
    <option value="claude">${escapeHtml(t("askProviderClaude"))}</option>
  `;
  if (selectedAskProvider === "chatgpt" || selectedAskProvider === "claude") {
    els.askProvider.value = selectedAskProvider;
  }
  els.askSubmitBtn.textContent = t("askSubmitBtn");
  els.askRawJsonSummary.textContent = t("askRawJsonSummary");
  if (latestAskResponse) {
    renderAskResponse(latestAskResponse);
    els.askResult.textContent = JSON.stringify(latestAskResponse, null, 2);
  } else {
    els.askSummary.textContent = t("askSummaryDefault");
    els.askAnswer.textContent = "";
    els.askSources.innerHTML = "";
  }

  els.summaryHeading.textContent = t("summaryHeading");
  els.summaryDesc.textContent = t("summaryDesc");
  els.summaryVideoLabel.textContent = t("summaryVideoLabel");
  els.summaryLanguageLabel.textContent = t("summaryLanguageLabel");
  const selectedSummaryLanguage = els.summaryLanguage.value;
  els.summaryLanguage.innerHTML = `
    <option value="en">${escapeHtml(t("summaryLangEnglish"))}</option>
    <option value="ja">${escapeHtml(t("summaryLangJapanese"))}</option>
  `;
  if (selectedSummaryLanguage === "en" || selectedSummaryLanguage === "ja") {
    els.summaryLanguage.value = selectedSummaryLanguage;
  }
  els.summarySubmitBtn.textContent = t("summarySubmitBtn");
  els.summaryRawJsonSummary.textContent = t("summaryRawJsonSummary");
  if (latestSummaryResponse) {
    renderSummaryResponse(latestSummaryResponse);
    els.summaryResult.textContent = JSON.stringify(latestSummaryResponse, null, 2);
  } else {
    els.summaryStatus.textContent = t("summaryStatusDefault");
    if (!els.summaryCards.innerHTML.trim()) {
      els.summaryCards.innerHTML = `<div class="search-empty">${escapeHtml(t("summaryNoData"))}</div>`;
    }
  }

  if (!options.skipRefresh) {
    Promise.all([refreshJobs(), refreshVideos()]).catch((err) => {
      const msg = t("apiConnectionError", { message: String(err.message || err) });
      els.jobsTable.innerHTML = `<p style="padding:0.7rem">${escapeHtml(msg)}</p>`;
      els.videosTable.innerHTML = `<p style="padding:0.7rem">${escapeHtml(msg)}</p>`;
      els.posterGrid.innerHTML = `<div class="search-empty">${escapeHtml(msg)}</div>`;
      els.summaryCards.innerHTML = `<div class="search-empty">${escapeHtml(msg)}</div>`;
    });
  }
}

function wireEvents() {
  els.ingestForm.addEventListener("submit", runIngest);
  els.searchForm.addEventListener("submit", runSearch);
  els.askForm.addEventListener("submit", runAsk);
  els.summaryForm.addEventListener("submit", runSummary);
  els.localeSelect.addEventListener("change", () => {
    applyLocale(els.localeSelect.value);
  });

  els.refreshJobs.addEventListener("click", async () => {
    els.refreshJobs.disabled = true;
    try {
      await refreshJobs();
    } catch (err) {
      els.jobsTable.innerHTML = `<p style="padding:0.7rem">${escapeHtml(String(err.message || err))}</p>`;
    } finally {
      els.refreshJobs.disabled = false;
    }
  });

  els.refreshVideos.addEventListener("click", async () => {
    els.refreshVideos.disabled = true;
    try {
      await refreshVideos();
    } catch (err) {
      els.videosTable.innerHTML = `<p style="padding:0.7rem">${escapeHtml(String(err.message || err))}</p>`;
    } finally {
      els.refreshVideos.disabled = false;
    }
  });

  els.refreshPosters.addEventListener("click", async () => {
    els.refreshPosters.disabled = true;
    try {
      await refreshVideos();
    } catch (err) {
      els.posterGrid.innerHTML = `<div class="search-empty">${escapeHtml(String(err.message || err))}</div>`;
    } finally {
      els.refreshPosters.disabled = false;
    }
  });

  els.logsToggleBtn.addEventListener("click", async () => {
    const shouldOpen = !logsPanelExpanded;
    setLogsPanelExpanded(shouldOpen);
    if (shouldOpen) {
      try {
        await refreshLogs();
      } catch (_) {
        // refreshLogs handles user-visible errors in logsResult.
      }
    }
  });

  els.logsRefreshBtn.addEventListener("click", async () => {
    els.logsRefreshBtn.disabled = true;
    try {
      await refreshLogs();
    } finally {
      els.logsRefreshBtn.disabled = false;
    }
  });
  els.searchQuery.addEventListener("input", () => saveSearchState());
  els.searchK.addEventListener("change", () => saveSearchState());
  els.searchMode.addEventListener("change", () => saveSearchState());

  els.playerMuteToggle.addEventListener("click", onPlayerMuteToggle);
  els.videosTable.addEventListener("click", onVideosClick);
  els.searchCards.addEventListener("click", onSearchCardsClick);
  els.askSources.addEventListener("click", onAskSourcesClick);
  els.summaryCards.addEventListener("click", onSummaryCardsClick);
}

async function boot() {
  const restoredSearch = restoreSearchState();
  applyLocale(currentLocale, { skipRefresh: true });
  setLogsPanelExpanded(false);
  wireEvents();
  loadYouTubeApi().catch(() => {});

  if (restoredSearch) {
    await hydrateReviewStateFromServer(latestSearchResponse?.results || []);
    if (latestSearchResponse) {
      renderSearchResults(latestSearchResponse);
    }
  }

  try {
    await Promise.all([refreshJobs(), refreshVideos()]);
  } catch (err) {
    const msg = t("apiConnectionError", { message: String(err.message || err) });
    els.jobsTable.innerHTML = `<p style="padding:0.7rem">${escapeHtml(msg)}</p>`;
    els.videosTable.innerHTML = `<p style="padding:0.7rem">${escapeHtml(msg)}</p>`;
    els.posterGrid.innerHTML = `<div class="search-empty">${escapeHtml(msg)}</div>`;
    if (logsPanelExpanded) {
      els.logsResult.textContent = msg;
    }
  }
}

boot();
