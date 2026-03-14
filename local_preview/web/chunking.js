const INGEST_UNLOCK_KEY = "yt_rag_ingest_unlocked";
if (localStorage.getItem(INGEST_UNLOCK_KEY) !== "1") {
  window.location.replace("./index.html#/ingest");
}

const LOCALE_STORAGE_KEY = "youtube-rag-ui-locale";
const EVAL_STORAGE_KEY = "youtube-rag-eval-v1";
const CHUNKING_PREFS_KEY = "youtube-rag-chunking-prefs";
const DEFAULT_LOCALE = "en-US";

// ------------------------------------------------------------------
// i18n
// ------------------------------------------------------------------
const I18N = {
  "en-US": {
    heroTitle: "Chunking Strategy Comparison",
    heroSubtitle: "Compare chunking strategies side-by-side and measure their effect on search results.",
    videoHeading: "Video & Strategy Selection",
    videoSelectLabel: "Video",
    strategyALegend: "Strategy A",
    strategyBLegend: "Strategy B",
    strategyALabel: "Strategy",
    strategyBLabel: "Strategy",
    previewBtn: "Preview Chunks",
    previewHeading: "Side-by-Side Chunk Preview",
    searchHeading: "Search Comparison",
    searchQueryLabel: "Query",
    searchKLabel: "Top K",
    searchBtn: "Search Both",
    exportHeading: "Export to Evaluation",
    exportDesc: "Export the most recent search comparison as an evaluation run snapshot.",
    exportBtn: "Export to Evaluation",
    colAHeading: "Strategy A",
    colBHeading: "Strategy B",
    searchColAHeading: "Strategy A Results",
    searchColBHeading: "Strategy B Results",
    noVideo: "No videos available",
    loading: "Loading...",
    previewSuccess: "Preview loaded.",
    searchSuccess: "Search completed.",
    exportSuccess: "Exported to evaluation workspace.",
    exportNoData: "Run a search comparison first.",
    strategyTime: "Time Window",
    strategySentence: "Sentence Boundary",
    strategyToken: "Token Count",
    paramWindow: "Window (s)",
    paramOverlap: "Overlap (s)",
    paramMaxChars: "Max Chars",
    paramTokenCount: "Token Count",
    paramOverlapFraction: "Overlap Fraction",
    navIngest: "Ingest",
    navTldr: "TLDR Studio",
    navQa: "Q&A Studio",
    navReviews: "Reviews",
    navEval: "Evaluation",
    navChunking: "Chunking",
  },
  "ja-JP": {
    heroTitle: "チャンキング戦略比較",
    heroSubtitle: "チャンキング戦略を並べて比較し、検索結果への影響を測定します。",
    videoHeading: "動画と戦略の選択",
    videoSelectLabel: "動画",
    strategyALegend: "戦略 A",
    strategyBLegend: "戦略 B",
    strategyALabel: "戦略",
    strategyBLabel: "戦略",
    previewBtn: "チャンクをプレビュー",
    previewHeading: "チャンクの並列プレビュー",
    searchHeading: "検索比較",
    searchQueryLabel: "クエリ",
    searchKLabel: "Top K",
    searchBtn: "両方を検索",
    exportHeading: "評価へエクスポート",
    exportDesc: "最新の検索比較を評価ランスナップショットとしてエクスポートします。",
    exportBtn: "評価へエクスポート",
    colAHeading: "戦略 A",
    colBHeading: "戦略 B",
    searchColAHeading: "戦略 A の結果",
    searchColBHeading: "戦略 B の結果",
    noVideo: "動画がありません",
    loading: "読み込み中...",
    previewSuccess: "プレビューを読み込みました。",
    searchSuccess: "検索が完了しました。",
    exportSuccess: "評価ワークスペースにエクスポートしました。",
    exportNoData: "先に検索比較を実行してください。",
    strategyTime: "時間ウィンドウ",
    strategySentence: "文境界",
    strategyToken: "トークン数",
    paramWindow: "ウィンドウ (秒)",
    paramOverlap: "オーバーラップ (秒)",
    paramMaxChars: "最大文字数",
    paramTokenCount: "トークン数",
    paramOverlapFraction: "オーバーラップ率",
    navIngest: "取り込み",
    navTldr: "TLDR スタジオ",
    navQa: "Q&A スタジオ",
    navReviews: "レビュー",
    navEval: "評価",
    navChunking: "チャンキング",
  },
};

// ------------------------------------------------------------------
// DOM refs
// ------------------------------------------------------------------
const els = {
  localeSelect: document.getElementById("localeSelect"),
  heroTitle: document.getElementById("heroTitle"),
  heroSubtitle: document.getElementById("heroSubtitle"),
  videoHeading: document.getElementById("videoHeading"),
  videoSelectLabel: document.getElementById("videoSelectLabel"),
  videoSelect: document.getElementById("videoSelect"),
  strategyALegend: document.getElementById("strategyALegend"),
  strategyBLegend: document.getElementById("strategyBLegend"),
  strategyALabel: document.getElementById("strategyALabel"),
  strategyBLabel: document.getElementById("strategyBLabel"),
  strategyASelect: document.getElementById("strategyASelect"),
  strategyBSelect: document.getElementById("strategyBSelect"),
  strategyAParams: document.getElementById("strategyAParams"),
  strategyBParams: document.getElementById("strategyBParams"),
  previewBtn: document.getElementById("previewBtn"),
  previewStatus: document.getElementById("previewStatus"),
  previewHeading: document.getElementById("previewHeading"),
  colAHeading: document.getElementById("colAHeading"),
  colBHeading: document.getElementById("colBHeading"),
  colAStats: document.getElementById("colAStats"),
  colBStats: document.getElementById("colBStats"),
  colAChunks: document.getElementById("colAChunks"),
  colBChunks: document.getElementById("colBChunks"),
  searchHeading: document.getElementById("searchHeading"),
  searchQueryLabel: document.getElementById("searchQueryLabel"),
  searchKLabel: document.getElementById("searchKLabel"),
  searchQuery: document.getElementById("searchQuery"),
  searchK: document.getElementById("searchK"),
  searchBtn: document.getElementById("searchBtn"),
  searchStatus: document.getElementById("searchStatus"),
  searchColAHeading: document.getElementById("searchColAHeading"),
  searchColBHeading: document.getElementById("searchColBHeading"),
  searchColAMeta: document.getElementById("searchColAMeta"),
  searchColBMeta: document.getElementById("searchColBMeta"),
  searchColAResults: document.getElementById("searchColAResults"),
  searchColBResults: document.getElementById("searchColBResults"),
  exportHeading: document.getElementById("exportHeading"),
  exportDesc: document.getElementById("exportDesc"),
  exportBtn: document.getElementById("exportBtn"),
  exportStatus: document.getElementById("exportStatus"),
  // nav
  homeNavLink: document.getElementById("homeNavLink"),
  tldrNavLink: document.getElementById("tldrNavLink"),
  qaNavLink: document.getElementById("qaNavLink"),
  reviewsNavLink: document.getElementById("reviewsNavLink"),
  evaluationNavLink: document.getElementById("evaluationNavLink"),
  chunkingNavLink: document.getElementById("chunkingNavLink"),
  mobileHomeNavText: document.getElementById("mobileHomeNavText"),
  mobileTldrNavText: document.getElementById("mobileTldrNavText"),
  mobileQaNavText: document.getElementById("mobileQaNavText"),
  mobileReviewsNavText: document.getElementById("mobileReviewsNavText"),
  mobileEvalNavText: document.getElementById("mobileEvalNavText"),
  mobileChunkingNavText: document.getElementById("mobileChunkingNavText"),
};

// ------------------------------------------------------------------
// Helpers
// ------------------------------------------------------------------
let currentLocale = DEFAULT_LOCALE;
let lastSearchA = null;
let lastSearchB = null;
let lastSearchQuery = "";
let lastSearchK = 5;

function t(key) {
  return (I18N[currentLocale] || I18N[DEFAULT_LOCALE])[key] || key;
}

function setNavLabel(el, text) {
  if (!el) return;
  const label = el.querySelector(".nav-label-text");
  if (label) { label.textContent = text; return; }
  el.textContent = text;
}

async function apiRequest(path, { method = "GET", body } = {}) {
  const response = await fetch(path, {
    method,
    cache: "no-store",
    headers: body ? { "content-type": "application/json" } : undefined,
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
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}

function truncate(value, max = 200) {
  const text = String(value || "");
  return text.length > max ? `${text.slice(0, max)}...` : text;
}

function formatSeconds(value) {
  const seconds = Math.max(0, Math.floor(Number(value || 0)));
  const minutes = Math.floor(seconds / 60);
  const remaining = seconds % 60;
  return `${minutes}:${String(remaining).padStart(2, "0")}`;
}

function savePrefs() {
  try {
    localStorage.setItem(CHUNKING_PREFS_KEY, JSON.stringify({
      videoId: els.videoSelect.value,
      strategyA: els.strategyASelect.value,
      strategyB: els.strategyBSelect.value,
    }));
  } catch (_) { /* best effort */ }
}

function loadPrefs() {
  try {
    return JSON.parse(localStorage.getItem(CHUNKING_PREFS_KEY) || "{}");
  } catch (_) { return {}; }
}

// ------------------------------------------------------------------
// Strategy param rendering
// ------------------------------------------------------------------
const STRATEGY_PARAMS = {
  time: [
    { key: "window", type: "select", options: [30, 45, 60, 90, 120], default: 60 },
    { key: "overlap", type: "select", options: [0, 10, 15, 30], default: 15 },
  ],
  sentence: [
    { key: "max_chars", type: "select", options: [500, 1000, 1500, 2000], default: 1000 },
  ],
  token: [
    { key: "token_count", type: "select", options: [128, 256, 512], default: 256 },
    { key: "overlap_fraction", type: "select", options: [0, 0.1, 0.25, 0.5], default: 0.25 },
  ],
};

const PARAM_LABELS = {
  window: "paramWindow",
  overlap: "paramOverlap",
  max_chars: "paramMaxChars",
  token_count: "paramTokenCount",
  overlap_fraction: "paramOverlapFraction",
};

function renderParams(container, strategy) {
  container.innerHTML = "";
  const defs = STRATEGY_PARAMS[strategy] || [];
  for (const def of defs) {
    const label = document.createElement("label");
    label.style.marginTop = "0.5rem";
    const span = document.createElement("span");
    span.textContent = t(PARAM_LABELS[def.key] || def.key);
    const select = document.createElement("select");
    select.dataset.paramKey = def.key;
    for (const opt of def.options) {
      const option = document.createElement("option");
      option.value = String(opt);
      option.textContent = String(opt);
      if (opt === def.default) option.selected = true;
      select.appendChild(option);
    }
    label.appendChild(span);
    label.appendChild(select);
    container.appendChild(label);
  }
}

function getParams(container) {
  const result = {};
  for (const select of container.querySelectorAll("select[data-param-key]")) {
    const key = select.dataset.paramKey;
    const val = select.value;
    result[key] = val.includes(".") ? parseFloat(val) : parseInt(val, 10);
  }
  return result;
}

// ------------------------------------------------------------------
// Chunk rendering
// ------------------------------------------------------------------
function renderChunks(container, chunks) {
  container.innerHTML = "";
  if (!chunks || !chunks.length) {
    container.innerHTML = '<div class="search-empty">No chunks.</div>';
    return;
  }
  for (const chunk of chunks) {
    const card = document.createElement("article");
    card.className = "search-card";
    card.innerHTML = `
      <div class="search-card-head">
        <div class="search-rank">#${chunk.index + 1}</div>
        <div class="search-meta">${formatSeconds(chunk.start)} - ${formatSeconds(chunk.end)}</div>
        <div class="search-lang">${chunk.char_count} chars</div>
      </div>
      <p class="search-snippet">${escapeHtml(truncate(chunk.raw_text))}</p>
    `;
    container.appendChild(card);
  }
}

function renderStats(el, stats, chunkCount) {
  if (!stats) {
    el.textContent = "";
    return;
  }
  el.textContent = `${chunkCount} chunks | avg ${Math.round(stats.avg_chunk_chars)} chars | min ${stats.min_chunk_chars} | max ${stats.max_chunk_chars}`;
}

function renderSearchResults(container, results) {
  container.innerHTML = "";
  if (!results || !results.length) {
    container.innerHTML = '<div class="search-empty">No results.</div>';
    return;
  }
  for (const r of results) {
    const card = document.createElement("article");
    card.className = "search-card";
    card.innerHTML = `
      <div class="search-card-head">
        <div class="search-rank">#${r.rank}</div>
        <div class="search-meta">${formatSeconds(r.start)} - ${formatSeconds(r.end)}</div>
        <div class="search-lang">score ${Number(r.score).toFixed(4)}</div>
      </div>
      <p class="search-snippet">${escapeHtml(truncate(r.text, 280))}</p>
    `;
    container.appendChild(card);
  }
}

// ------------------------------------------------------------------
// Actions
// ------------------------------------------------------------------
async function loadVideos() {
  try {
    const payload = await apiRequest("/v1/videos");
    const videos = Array.isArray(payload?.videos) ? payload.videos : [];
    els.videoSelect.innerHTML = "";
    if (!videos.length) {
      const opt = document.createElement("option");
      opt.value = "";
      opt.textContent = t("noVideo");
      els.videoSelect.appendChild(opt);
      els.videoSelect.disabled = true;
      return;
    }
    const prefs = loadPrefs();
    for (const v of videos) {
      const opt = document.createElement("option");
      opt.value = v.video_id;
      opt.textContent = `${v.title} (${v.video_id})`;
      if (v.video_id === prefs.videoId) opt.selected = true;
      els.videoSelect.appendChild(opt);
    }
    els.videoSelect.disabled = false;
    els.previewBtn.disabled = false;
    els.searchBtn.disabled = false;
  } catch (err) {
    els.previewStatus.textContent = `Failed to load videos: ${err.message}`;
  }
}

async function previewChunks() {
  const videoId = els.videoSelect.value;
  if (!videoId) return;
  els.previewBtn.disabled = true;
  els.previewStatus.textContent = t("loading");
  els.colAChunks.innerHTML = "";
  els.colBChunks.innerHTML = "";
  els.colAStats.textContent = "";
  els.colBStats.textContent = "";

  try {
    const [resA, resB] = await Promise.all([
      apiRequest("/v1/chunking/preview", {
        method: "POST",
        body: {
          video_id: videoId,
          strategy: els.strategyASelect.value,
          params: getParams(els.strategyAParams),
        },
      }),
      apiRequest("/v1/chunking/preview", {
        method: "POST",
        body: {
          video_id: videoId,
          strategy: els.strategyBSelect.value,
          params: getParams(els.strategyBParams),
        },
      }),
    ]);
    renderChunks(els.colAChunks, resA.chunks);
    renderChunks(els.colBChunks, resB.chunks);
    renderStats(els.colAStats, resA.stats, resA.chunk_count);
    renderStats(els.colBStats, resB.stats, resB.chunk_count);
    els.previewStatus.textContent = t("previewSuccess");
    savePrefs();
  } catch (err) {
    els.previewStatus.textContent = `Error: ${err.message}`;
  } finally {
    els.previewBtn.disabled = false;
  }
}

async function searchBoth() {
  const videoId = els.videoSelect.value;
  const query = els.searchQuery.value.trim();
  if (!videoId || !query) return;
  const k = Math.max(1, Math.min(parseInt(els.searchK.value, 10) || 5, 12));
  els.searchBtn.disabled = true;
  els.searchStatus.textContent = t("loading");
  els.searchColAResults.innerHTML = "";
  els.searchColBResults.innerHTML = "";
  els.searchColAMeta.textContent = "";
  els.searchColBMeta.textContent = "";

  try {
    const [resA, resB] = await Promise.all([
      apiRequest("/v1/chunking/search", {
        method: "POST",
        body: {
          video_id: videoId,
          strategy: els.strategyASelect.value,
          params: getParams(els.strategyAParams),
          query,
          k,
        },
      }),
      apiRequest("/v1/chunking/search", {
        method: "POST",
        body: {
          video_id: videoId,
          strategy: els.strategyBSelect.value,
          params: getParams(els.strategyBParams),
          query,
          k,
        },
      }),
    ]);
    lastSearchA = resA;
    lastSearchB = resB;
    lastSearchQuery = query;
    lastSearchK = k;
    renderSearchResults(els.searchColAResults, resA.results);
    renderSearchResults(els.searchColBResults, resB.results);
    els.searchColAMeta.textContent = `${resA.chunk_count} chunks | embed ${resA.embedding_time_ms}ms | search ${resA.search_time_ms}ms`;
    els.searchColBMeta.textContent = `${resB.chunk_count} chunks | embed ${resB.embedding_time_ms}ms | search ${resB.search_time_ms}ms`;
    els.searchStatus.textContent = t("searchSuccess");
    els.exportBtn.disabled = false;
    savePrefs();
  } catch (err) {
    els.searchStatus.textContent = `Error: ${err.message}`;
  } finally {
    els.searchBtn.disabled = false;
  }
}

function exportToEvaluation() {
  if (!lastSearchA || !lastSearchB) {
    els.exportStatus.textContent = t("exportNoData");
    return;
  }

  const now = new Date().toISOString();
  const runId = `chunking_${Date.now()}`;

  function makeRunResult(query, results) {
    return {
      query_id: `q_${Date.now()}_${Math.random().toString(36).slice(2, 8)}`,
      query_text: query,
      query_type: "factual",
      expected_relevant_min: null,
      result_count: results.length,
      retrieval_details: {},
      items: results.map((r) => ({
        video_id: els.videoSelect.value,
        chunk_index: r.chunk_index,
        start: r.start,
        end: r.end,
        text: r.text,
        score: r.score,
        rank: r.rank,
      })),
      labels: {},
      error: null,
    };
  }

  const strategyALabel = `${els.strategyASelect.value}(${JSON.stringify(getParams(els.strategyAParams))})`;
  const strategyBLabel = `${els.strategyBSelect.value}(${JSON.stringify(getParams(els.strategyBParams))})`;

  function buildRun(label, searchRes) {
    return {
      id: `${runId}_${label.replace(/[^a-zA-Z0-9]/g, "_")}`,
      query_set_id: "",
      started_at: now,
      completed_at: now,
      retrieval_mode: "dense",
      k: lastSearchK,
      candidate_k: 0,
      index_snapshot_id: "chunking_ephemeral",
      system_version: {
        app_version: "chunking_lab_v1",
        retriever_version: "local",
        chunking_version: label,
      },
      results: [makeRunResult(lastSearchQuery, searchRes.results)],
      metrics: null,
    };
  }

  try {
    const raw = localStorage.getItem(EVAL_STORAGE_KEY);
    const state = raw ? JSON.parse(raw) : { version: 1, query_sets: [], runs: [] };
    if (!Array.isArray(state.runs)) state.runs = [];
    state.runs.push(buildRun(strategyALabel, lastSearchA));
    state.runs.push(buildRun(strategyBLabel, lastSearchB));
    localStorage.setItem(EVAL_STORAGE_KEY, JSON.stringify(state));
    els.exportStatus.textContent = t("exportSuccess");
  } catch (err) {
    els.exportStatus.textContent = `Export failed: ${err.message}`;
  }
}

// ------------------------------------------------------------------
// i18n apply
// ------------------------------------------------------------------
function applyLocale() {
  const textKeys = [
    "heroTitle", "heroSubtitle", "videoHeading", "videoSelectLabel",
    "strategyALegend", "strategyBLegend", "strategyALabel", "strategyBLabel",
    "previewHeading", "searchHeading", "searchQueryLabel", "searchKLabel",
    "exportHeading", "exportDesc", "colAHeading", "colBHeading",
    "searchColAHeading", "searchColBHeading",
  ];
  for (const key of textKeys) {
    if (els[key]) els[key].textContent = t(key);
  }
  if (els.previewBtn) els.previewBtn.textContent = t("previewBtn");
  if (els.searchBtn) els.searchBtn.textContent = t("searchBtn");
  if (els.exportBtn) els.exportBtn.textContent = t("exportBtn");

  // nav labels
  setNavLabel(els.homeNavLink, t("navIngest"));
  setNavLabel(els.tldrNavLink, t("navTldr"));
  setNavLabel(els.qaNavLink, t("navQa"));
  setNavLabel(els.reviewsNavLink, t("navReviews"));
  setNavLabel(els.evaluationNavLink, t("navEval"));
  setNavLabel(els.chunkingNavLink, t("navChunking"));
  if (els.mobileHomeNavText) els.mobileHomeNavText.textContent = t("navIngest");
  if (els.mobileTldrNavText) els.mobileTldrNavText.textContent = t("navTldr");
  if (els.mobileQaNavText) els.mobileQaNavText.textContent = t("navQa");
  if (els.mobileReviewsNavText) els.mobileReviewsNavText.textContent = t("navReviews");
  if (els.mobileEvalNavText) els.mobileEvalNavText.textContent = t("navEval");
  if (els.mobileChunkingNavText) els.mobileChunkingNavText.textContent = t("navChunking");

  // re-render strategy param labels
  renderParams(els.strategyAParams, els.strategyASelect.value);
  renderParams(els.strategyBParams, els.strategyBSelect.value);
}

// ------------------------------------------------------------------
// Init
// ------------------------------------------------------------------
function init() {
  const storedLocale = localStorage.getItem(LOCALE_STORAGE_KEY) || DEFAULT_LOCALE;
  currentLocale = storedLocale === "ja-JP" ? "ja-JP" : "en-US";
  if (els.localeSelect) els.localeSelect.value = currentLocale;

  renderParams(els.strategyAParams, els.strategyASelect.value);
  renderParams(els.strategyBParams, els.strategyBSelect.value);

  const prefs = loadPrefs();
  if (prefs.strategyA && els.strategyASelect) {
    els.strategyASelect.value = prefs.strategyA;
    renderParams(els.strategyAParams, prefs.strategyA);
  }
  if (prefs.strategyB && els.strategyBSelect) {
    els.strategyBSelect.value = prefs.strategyB;
    renderParams(els.strategyBParams, prefs.strategyB);
  }

  applyLocale();
  loadVideos();

  // event listeners
  els.localeSelect.addEventListener("change", () => {
    currentLocale = els.localeSelect.value;
    localStorage.setItem(LOCALE_STORAGE_KEY, currentLocale);
    document.documentElement.lang = currentLocale === "ja-JP" ? "ja" : "en";
    applyLocale();
  });

  els.strategyASelect.addEventListener("change", () => {
    renderParams(els.strategyAParams, els.strategyASelect.value);
  });
  els.strategyBSelect.addEventListener("change", () => {
    renderParams(els.strategyBParams, els.strategyBSelect.value);
  });

  els.previewBtn.addEventListener("click", previewChunks);
  els.searchBtn.addEventListener("click", searchBoth);
  els.exportBtn.addEventListener("click", exportToEvaluation);
}

init();
