const INGEST_UNLOCK_KEY = "yt_rag_ingest_unlocked";
if (localStorage.getItem(INGEST_UNLOCK_KEY) !== "1") {
  window.location.replace("./index.html#/ingest");
}

const LOCALE_STORAGE_KEY = "youtube-rag-ui-locale";
const EVAL_STORAGE_KEY = "youtube-rag-eval-v1";
const EVAL_SCHEMA_VERSION = 1;
const DEFAULT_LOCALE = "en-US";
const APP_VERSION = "local_eval_v1";

const QUERY_TYPES = ["navigational", "factual", "thematic"];
const REASON_CODES = [
  "wrong_entity",
  "too_broad",
  "wrong_language",
  "timestamp_mismatch",
  "low_specificity",
  "other",
];

const els = {
  localeSelect: document.getElementById("localeSelect"),
  localeLabel: document.getElementById("localeLabel"),
  homeBrandLink: document.getElementById("homeBrandLink"),
  eyebrowText: document.getElementById("eyebrowText"),
  homeNavLink: document.getElementById("homeNavLink"),
  studioNavLink: document.getElementById("studioNavLink"),
  reviewsNavLink: document.getElementById("reviewsNavLink"),
  evidenceNavLink: document.getElementById("evidenceNavLink"),
  evaluationNavLink: document.getElementById("evaluationNavLink"),
  chunkingNavLink: document.getElementById("chunkingNavLink"),
  mobileHomeNavText: document.getElementById("mobileHomeNavText"),
  mobileStudioNavText: document.getElementById("mobileStudioNavText"),
  mobileReviewsNavText: document.getElementById("mobileReviewsNavText"),
  mobileEvidenceNavText: document.getElementById("mobileEvidenceNavText"),
  mobileEvaluationNavText: document.getElementById("mobileEvaluationNavText"),
  mobileChunkingNavText: document.getElementById("mobileChunkingNavText"),
  heroTitle: document.getElementById("heroTitle"),
  heroSubtitle: document.getElementById("heroSubtitle"),
  guidelinesHeading: document.getElementById("guidelinesHeading"),
  guidelinesList: document.getElementById("guidelinesList"),
  generatorHeading: document.getElementById("generatorHeading"),
  generatorIntro: document.getElementById("generatorIntro"),
  generatorCapability: document.getElementById("generatorCapability"),
  generatorVideosHeading: document.getElementById("generatorVideosHeading"),
  generatorSelectionCount: document.getElementById("generatorSelectionCount"),
  generatorVideoList: document.getElementById("generatorVideoList"),
  generatorDraftLabel: document.getElementById("generatorDraftLabel"),
  generatorDraftSelect: document.getElementById("generatorDraftSelect"),
  loadGeneratorDraftBtn: document.getElementById("loadGeneratorDraftBtn"),
  generateEvalDatasetBtn: document.getElementById("generateEvalDatasetBtn"),
  generatorStatus: document.getElementById("generatorStatus"),
  generatorWarnings: document.getElementById("generatorWarnings"),
  generatorReviewCards: document.getElementById("generatorReviewCards"),
  generatorFinalizeActions: document.getElementById("generatorFinalizeActions"),
  generatorReviewSummary: document.getElementById("generatorReviewSummary"),
  finalizeEvalDatasetBtn: document.getElementById("finalizeEvalDatasetBtn"),
  downloadEvalDatasetLink: document.getElementById("downloadEvalDatasetLink"),
  generatorStepSelect: document.getElementById("generatorStepSelect"),
  generatorStepGenerate: document.getElementById("generatorStepGenerate"),
  generatorStepReview: document.getElementById("generatorStepReview"),
  generatorStepFinalize: document.getElementById("generatorStepFinalize"),
  querySetsHeading: document.getElementById("querySetsHeading"),
  newQuerySetBtn: document.getElementById("newQuerySetBtn"),
  saveQuerySetBtn: document.getElementById("saveQuerySetBtn"),
  deleteQuerySetBtn: document.getElementById("deleteQuerySetBtn"),
  exportQuerySetBtn: document.getElementById("exportQuerySetBtn"),
  importQuerySetBtn: document.getElementById("importQuerySetBtn"),
  importQuerySetFile: document.getElementById("importQuerySetFile"),
  querySetSelectLabel: document.getElementById("querySetSelectLabel"),
  querySetSelect: document.getElementById("querySetSelect"),
  querySetNameLabel: document.getElementById("querySetNameLabel"),
  querySetName: document.getElementById("querySetName"),
  querySetLanguageLabel: document.getElementById("querySetLanguageLabel"),
  querySetLanguage: document.getElementById("querySetLanguage"),
  thQueryText: document.getElementById("thQueryText"),
  thQueryType: document.getElementById("thQueryType"),
  thExpectedMin: document.getElementById("thExpectedMin"),
  thQueryNotes: document.getElementById("thQueryNotes"),
  thQueryActions: document.getElementById("thQueryActions"),
  queryRowsBody: document.getElementById("queryRowsBody"),
  addQueryRowBtn: document.getElementById("addQueryRowBtn"),
  runsHeading: document.getElementById("runsHeading"),
  runQuerySetBtn: document.getElementById("runQuerySetBtn"),
  exportRunBtn: document.getElementById("exportRunBtn"),
  resetEvalDataBtn: document.getElementById("resetEvalDataBtn"),
  runModeLabel: document.getElementById("runModeLabel"),
  runMode: document.getElementById("runMode"),
  runKLabel: document.getElementById("runKLabel"),
  runK: document.getElementById("runK"),
  runSelectLabel: document.getElementById("runSelectLabel"),
  runSelect: document.getElementById("runSelect"),
  runStatus: document.getElementById("runStatus"),
  metricsHeading: document.getElementById("metricsHeading"),
  metricPrecision: document.getElementById("metricPrecision"),
  metricRecall: document.getElementById("metricRecall"),
  metricMRR: document.getElementById("metricMRR"),
  metricNDCG: document.getElementById("metricNDCG"),
  thMetricQuery: document.getElementById("thMetricQuery"),
  thMetricPAtK: document.getElementById("thMetricPAtK"),
  thMetricRecall: document.getElementById("thMetricRecall"),
  thMetricMRR: document.getElementById("thMetricMRR"),
  thMetricNDCG: document.getElementById("thMetricNDCG"),
  thMetricLabeled: document.getElementById("thMetricLabeled"),
  metricsTableBody: document.getElementById("metricsTableBody"),
  reviewHeading: document.getElementById("reviewHeading"),
  runQuerySelectLabel: document.getElementById("runQuerySelectLabel"),
  runQuerySelect: document.getElementById("runQuerySelect"),
  resultReasonLabel: document.getElementById("resultReasonLabel"),
  reasonLegend: document.getElementById("reasonLegend"),
  runResultCards: document.getElementById("runResultCards"),
  compareHeading: document.getElementById("compareHeading"),
  compareRunALabel: document.getElementById("compareRunALabel"),
  compareRunA: document.getElementById("compareRunA"),
  compareRunBLabel: document.getElementById("compareRunBLabel"),
  compareRunB: document.getElementById("compareRunB"),
  compareRunsBtn: document.getElementById("compareRunsBtn"),
  compareResult: document.getElementById("compareResult"),
};

const I18N = {
  "en-US": {
    pageTitle: "YouTube Transcript RAG | Evaluation",
    localeLabel: "Language",
    navIngest: "Ingest",
    eyebrowText: "YouTube Transcript RAG",
    navStudio: "Studio",
    navReviews: "Reviews",
    navEvidence: "Evidence",
    navEvaluation: "Evaluation",
    navChunking: "Chunking",
    heroTitle: "Semantic Retrieval Evaluation",
    heroSubtitle: "Create query sets, run ranked retrieval snapshots, label results locally, and compare metrics across runs.",
    guidelinesHeading: "Relevance Guidelines",
    guidelines: [
      "Relevant: chunk directly helps answer the query without extra assumptions.",
      "Not relevant: chunk is off-topic, wrong entity, wrong language, or too broad.",
      "Unsure: partial overlap, unclear intent match, or insufficient context.",
      "Use reason codes for not relevant/unsure labels to speed error triage.",
    ],
    generatorHeading: "Generate a Dataset",
    generatorIntro: "Select up to three videos. Codex drafts six retrieval cases; you approve the evidence before they become evaluation data.",
    generatorVideosHeading: "Choose 1–3 videos",
    generatorDraftLabel: "Saved draft",
    loadGeneratorDraftBtn: "Load Draft",
    generateEvalDatasetBtn: "Generate 6 Cases",
    generatorChecking: "Checking Codex CLI…",
    generatorReady: "Codex CLI ready",
    generatorUnavailable: "Codex CLI unavailable",
    generatorLoading: "Loading videos and saved drafts…",
    generatorSelectPrompt: "Select at least one video to start.",
    generatorNoVideos: "No ingested videos with transcript chunks are available.",
    generatorNoDrafts: "No saved drafts",
    generatorQueued: "Generation queued…",
    generatorStepStatus: "Generator: {step}",
    generatorFailed: "Generation failed: {message}",
    generatorDraftReady: "Draft ready. Review every case before finalizing.",
    generatorWarningsHeading: "Generator warnings",
    generatorFactsHeading: "Required facts",
    generatorEvidenceHeading: "Gold evidence",
    generatorApprove: "Approve",
    generatorEdit: "Edit",
    generatorReject: "Reject",
    generatorSaveEdit: "Save & Approve",
    generatorCancelEdit: "Cancel",
    generatorQuestionLabel: "Question",
    generatorFactsLabel: "Required facts (one per line)",
    generatorNotesLabel: "Notes",
    generatorDifficultyLabel: "Difficulty",
    generatorPending: "Pending",
    generatorApproved: "Approved",
    generatorEdited: "Edited",
    generatorRejected: "Rejected",
    generatorReviewSummary: "{decided}/{total} reviewed · {accepted} accepted",
    generatorCreateQuerySet: "Create Query Set",
    generatorDownload: "Download JSONL",
    generatorFinalized: "Dataset finalized with {count} cases. The query set is ready to run.",
    generatorStepSelectLabel: "Select",
    generatorStepGenerateLabel: "Generate",
    generatorStepReviewLabel: "Review",
    generatorStepFinalizeLabel: "Finalize",
    querySetsHeading: "Query Sets",
    newQuerySetBtn: "New Set",
    saveQuerySetBtn: "Save Set",
    deleteQuerySetBtn: "Delete Set",
    exportQuerySetBtn: "Export Set",
    importQuerySetBtn: "Import Set",
    querySetSelectLabel: "Query Set",
    querySetNameLabel: "Set Name",
    querySetNamePlaceholder: "Core retrieval checks",
    querySetLanguageLabel: "Language",
    thQueryText: "query",
    thQueryType: "type",
    thExpectedMin: "expected_min_relevant",
    thQueryNotes: "notes",
    thQueryActions: "actions",
    addQueryRowBtn: "Add Query",
    removeQuery: "Remove",
    runsHeading: "Runs",
    runQuerySetBtn: "Run Query Set",
    exportRunBtn: "Export Run",
    resetEvalDataBtn: "Reset Local Eval Data",
    runModeLabel: "Retrieval Mode",
    runKLabel: "Top K",
    runSelectLabel: "Run Snapshot",
    runIdle: "Ready.",
    runMissingSet: "No query set selected.",
    runEmptySet: "Add at least one query before running.",
    runProgress: "Running {index}/{total}: {query}",
    runDone: "Run complete. {total} queries executed.",
    runFailedQuery: "Query failed: {query} -> {message}",
    metricsHeading: "Metrics",
    thMetricQuery: "query",
    thMetricPAtK: "P@K",
    thMetricRecall: "Recall@K",
    thMetricMRR: "MRR",
    thMetricNDCG: "nDCG@K",
    thMetricLabeled: "labeled",
    reviewHeading: "Review Run Results",
    runQuerySelectLabel: "Query",
    resultReasonLabel: "Reason Codes",
    labelRelevant: "Relevant",
    labelNotRelevant: "Not relevant",
    labelUnsure: "Unsure",
    reasonPlaceholder: "reason",
    reviewEmpty: "No run results available yet.",
    compareHeading: "Run Comparison",
    compareRunALabel: "Run A",
    compareRunBLabel: "Run B",
    compareRunsBtn: "Compare",
    compareNeedTwo: "Select two runs to compare.",
    compareNoMetrics: "Run metrics unavailable for one or both snapshots.",
    noData: "No data.",
    importFailed: "Import failed: {message}",
    exportFailed: "Export failed: {message}",
    deleteSetConfirm: "Delete this query set? Existing runs are kept.",
    resetEvalConfirm: "Reset all local evaluation data on this browser?",
    mode: "mode",
    model: "model",
    score: "score",
    rank: "rank",
    time: "time",
    reason: "reason",
    unlabeled: "unlabeled",
  },
  "ja-JP": {
    pageTitle: "YouTube Transcript RAG | 評価",
    localeLabel: "言語",
    navIngest: "取り込み",
    eyebrowText: "YouTube Transcript RAG",
    navStudio: "Studio",
    navReviews: "レビュー",
    navEvidence: "エビデンス",
    navEvaluation: "評価",
    navChunking: "チャンキング",
    heroTitle: "セマンティック検索評価",
    heroSubtitle: "クエリセット作成、検索スナップショット実行、ローカルラベリング、ラン比較を行います。",
    guidelinesHeading: "関連性ラベルガイド",
    guidelines: [
      "関連あり: 追加の推測なしでクエリ回答に直接役立つチャンク。",
      "関連なし: オフトピック、別エンティティ、言語不一致、または広すぎる内容。",
      "不明: 一部一致するが意図一致が曖昧、または文脈不足。",
      "関連なし/不明には理由コードを付与して原因分析を高速化します。",
    ],
    generatorHeading: "評価データセットを生成",
    generatorIntro: "最大3本の動画を選択します。Codexが6件の検索ケースを提案し、承認後に評価データになります。",
    generatorVideosHeading: "動画を1〜3本選択",
    generatorDraftLabel: "保存済みドラフト",
    loadGeneratorDraftBtn: "ドラフトを開く",
    generateEvalDatasetBtn: "6ケースを生成",
    generatorChecking: "Codex CLIを確認中…",
    generatorReady: "Codex CLI 準備完了",
    generatorUnavailable: "Codex CLIを利用できません",
    generatorLoading: "動画とドラフトを読み込み中…",
    generatorSelectPrompt: "開始するには動画を1本以上選択してください。",
    generatorNoVideos: "文字起こしチャンク付きの動画がありません。",
    generatorNoDrafts: "保存済みドラフトなし",
    generatorQueued: "生成を待機中…",
    generatorStepStatus: "生成状況: {step}",
    generatorFailed: "生成失敗: {message}",
    generatorDraftReady: "ドラフト準備完了。確定前に全ケースをレビューしてください。",
    generatorWarningsHeading: "生成時の警告",
    generatorFactsHeading: "必須の事実",
    generatorEvidenceHeading: "正解エビデンス",
    generatorApprove: "承認",
    generatorEdit: "編集",
    generatorReject: "却下",
    generatorSaveEdit: "編集して承認",
    generatorCancelEdit: "キャンセル",
    generatorQuestionLabel: "質問",
    generatorFactsLabel: "必須の事実（1行に1件）",
    generatorNotesLabel: "メモ",
    generatorDifficultyLabel: "難易度",
    generatorPending: "未確認",
    generatorApproved: "承認済み",
    generatorEdited: "編集済み",
    generatorRejected: "却下済み",
    generatorReviewSummary: "{decided}/{total} 件確認 · {accepted} 件採用",
    generatorCreateQuerySet: "クエリセットを作成",
    generatorDownload: "JSONLをダウンロード",
    generatorFinalized: "{count}件でデータセットを確定しました。クエリセットを実行できます。",
    generatorStepSelectLabel: "選択",
    generatorStepGenerateLabel: "生成",
    generatorStepReviewLabel: "レビュー",
    generatorStepFinalizeLabel: "確定",
    querySetsHeading: "クエリセット",
    newQuerySetBtn: "新規セット",
    saveQuerySetBtn: "保存",
    deleteQuerySetBtn: "削除",
    exportQuerySetBtn: "エクスポート",
    importQuerySetBtn: "インポート",
    querySetSelectLabel: "クエリセット",
    querySetNameLabel: "セット名",
    querySetNamePlaceholder: "基本検索チェック",
    querySetLanguageLabel: "言語",
    thQueryText: "query",
    thQueryType: "type",
    thExpectedMin: "expected_min_relevant",
    thQueryNotes: "notes",
    thQueryActions: "actions",
    addQueryRowBtn: "クエリ追加",
    removeQuery: "削除",
    runsHeading: "ラン",
    runQuerySetBtn: "クエリセット実行",
    exportRunBtn: "ランをエクスポート",
    resetEvalDataBtn: "ローカル評価データ初期化",
    runModeLabel: "検索モード",
    runKLabel: "Top K",
    runSelectLabel: "ランスナップショット",
    runIdle: "準備完了。",
    runMissingSet: "クエリセットが選択されていません。",
    runEmptySet: "実行前に1件以上のクエリを追加してください。",
    runProgress: "実行中 {index}/{total}: {query}",
    runDone: "実行完了。{total} 件のクエリを処理しました。",
    runFailedQuery: "クエリ失敗: {query} -> {message}",
    metricsHeading: "指標",
    thMetricQuery: "query",
    thMetricPAtK: "P@K",
    thMetricRecall: "Recall@K",
    thMetricMRR: "MRR",
    thMetricNDCG: "nDCG@K",
    thMetricLabeled: "labeled",
    reviewHeading: "ラン結果レビュー",
    runQuerySelectLabel: "クエリ",
    resultReasonLabel: "理由コード",
    labelRelevant: "関連あり",
    labelNotRelevant: "関連なし",
    labelUnsure: "不明",
    reasonPlaceholder: "理由",
    reviewEmpty: "レビュー可能なラン結果がありません。",
    compareHeading: "ラン比較",
    compareRunALabel: "Run A",
    compareRunBLabel: "Run B",
    compareRunsBtn: "比較",
    compareNeedTwo: "比較する2つのランを選択してください。",
    compareNoMetrics: "どちらかのランで指標がありません。",
    noData: "データなし。",
    importFailed: "インポート失敗: {message}",
    exportFailed: "エクスポート失敗: {message}",
    deleteSetConfirm: "このクエリセットを削除しますか？既存ランは保持されます。",
    resetEvalConfirm: "このブラウザの評価データをすべて初期化しますか？",
    mode: "mode",
    model: "model",
    score: "score",
    rank: "rank",
    time: "time",
    reason: "reason",
    unlabeled: "未ラベル",
  },
};

let currentLocale = localStorage.getItem(LOCALE_STORAGE_KEY) || DEFAULT_LOCALE;
if (!I18N[currentLocale]) {
  currentLocale = DEFAULT_LOCALE;
}

let state = loadEvalState();
let currentQuerySetId = null;
let currentRunId = null;
let currentRunQueryId = null;
let generatorCapability = null;
let generatorVideos = [];
let generatorDrafts = [];
let currentGeneratorDraft = null;
let activeGeneratorJobId = null;
let generatorPollTimer = null;
let editingGeneratorCaseId = null;

function t(key, vars = {}) {
  const base = I18N[currentLocale] || I18N[DEFAULT_LOCALE];
  const fallback = I18N[DEFAULT_LOCALE];
  let template = base[key] || fallback[key] || key;
  Object.entries(vars).forEach(([name, value]) => {
    template = template.replaceAll(`{${name}}`, String(value));
  });
  return template;
}

function nowIso() {
  return new Date().toISOString();
}

function makeId(prefix) {
  return `${prefix}_${Math.random().toString(36).slice(2, 10)}`;
}

function formatSeconds(sec) {
  const seconds = Math.max(0, Math.floor(Number(sec || 0)));
  const mm = Math.floor(seconds / 60);
  const ss = seconds % 60;
  return `${mm}:${String(ss).padStart(2, "0")}`;
}

function truncate(value, max = 280) {
  const text = String(value || "");
  return text.length > max ? `${text.slice(0, max)}...` : text;
}

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll("\"", "&quot;")
    .replaceAll("'", "&#39;");
}

function setNavLabel(element, text) {
  if (!element) {
    return;
  }
  const label = element.querySelector(".nav-label-text");
  if (label) {
    label.textContent = text;
    return;
  }
  element.textContent = text;
}

function formatMetric(value) {
  if (value == null || Number.isNaN(value)) {
    return "-";
  }
  return Number(value).toFixed(3);
}

function parseOptionalInt(value) {
  if (value == null || value === "") {
    return null;
  }
  const parsed = Number(value);
  if (!Number.isFinite(parsed)) {
    return null;
  }
  return Math.max(0, Math.trunc(parsed));
}

function getChunkKey(item) {
  const videoId = String(item.video_id || "").trim();
  if (!videoId) {
    return "";
  }
  if (item.chunk_index != null && item.chunk_index !== "") {
    return `${videoId}:${Math.trunc(Number(item.chunk_index))}`;
  }
  const startMs = Math.floor(Number(item.start || 0) * 1000);
  const endMs = Math.floor(Number(item.end || 0) * 1000);
  return `${videoId}:${startMs}:${endMs}`;
}

function defaultQuerySet() {
  const id = makeId("qs");
  return {
    id,
    name: "Starter Eval Set",
    language: "mixed",
    created_at: nowIso(),
    queries: [
      { id: makeId("q"), text: "イケメンすぎる", type: "thematic", expected_relevant_min: 1, notes: "" },
      { id: makeId("q"), text: "favorite hot sauce", type: "factual", expected_relevant_min: 1, notes: "" },
      { id: makeId("q"), text: "self introduction", type: "navigational", expected_relevant_min: 1, notes: "" },
    ],
  };
}

function createEmptyState() {
  return {
    version: EVAL_SCHEMA_VERSION,
    query_sets: [defaultQuerySet()],
    runs: [],
  };
}

function normalizeQuery(row) {
  return {
    id: String(row?.id || makeId("q")),
    text: String(row?.text || "").trim(),
    type: QUERY_TYPES.includes(row?.type) ? row.type : "factual",
    expected_relevant_min: parseOptionalInt(row?.expected_relevant_min),
    notes: String(row?.notes || ""),
  };
}

function normalizeQuerySet(row) {
  const queries = Array.isArray(row?.queries) ? row.queries.map(normalizeQuery) : [];
  return {
    id: String(row?.id || makeId("qs")),
    name: String(row?.name || "Untitled Query Set"),
    language: ["ja", "en", "mixed"].includes(row?.language) ? row.language : "mixed",
    created_at: String(row?.created_at || nowIso()),
    queries,
  };
}

function normalizeRunResult(row) {
  const items = Array.isArray(row?.items) ? row.items : [];
  const labels = (row?.labels && typeof row.labels === "object") ? row.labels : {};
  return {
    query_id: String(row?.query_id || makeId("q")),
    query_text: String(row?.query_text || ""),
    query_type: QUERY_TYPES.includes(row?.query_type) ? row.query_type : "factual",
    expected_relevant_min: parseOptionalInt(row?.expected_relevant_min),
    result_count: Number(row?.result_count || items.length || 0),
    retrieval_details: row?.retrieval_details || {},
    items,
    labels,
    error: row?.error ? String(row.error) : null,
  };
}

function normalizeRun(row) {
  return {
    id: String(row?.id || makeId("run")),
    query_set_id: String(row?.query_set_id || ""),
    started_at: String(row?.started_at || nowIso()),
    completed_at: String(row?.completed_at || row?.started_at || nowIso()),
    retrieval_mode: ["hybrid", "dense", "lexical"].includes(row?.retrieval_mode) ? row.retrieval_mode : "hybrid",
    k: Number(row?.k || 5),
    candidate_k: Number(row?.candidate_k || 0),
    index_snapshot_id: String(row?.index_snapshot_id || "local"),
    system_version: {
      app_version: String(row?.system_version?.app_version || APP_VERSION),
      retriever_version: String(row?.system_version?.retriever_version || "local"),
      chunking_version: String(row?.system_version?.chunking_version || "unknown"),
    },
    results: Array.isArray(row?.results) ? row.results.map(normalizeRunResult) : [],
    metrics: row?.metrics || null,
  };
}

function loadEvalState() {
  try {
    const raw = localStorage.getItem(EVAL_STORAGE_KEY);
    if (!raw) {
      const created = createEmptyState();
      localStorage.setItem(EVAL_STORAGE_KEY, JSON.stringify(created));
      return created;
    }
    const parsed = JSON.parse(raw);
    if (!parsed || typeof parsed !== "object") {
      return createEmptyState();
    }

    const normalized = {
      version: EVAL_SCHEMA_VERSION,
      query_sets: Array.isArray(parsed.query_sets)
        ? parsed.query_sets.map(normalizeQuerySet)
        : [defaultQuerySet()],
      runs: Array.isArray(parsed.runs) ? parsed.runs.map(normalizeRun) : [],
    };
    if (!normalized.query_sets.length) {
      normalized.query_sets.push(defaultQuerySet());
    }
    localStorage.setItem(EVAL_STORAGE_KEY, JSON.stringify(normalized));
    return normalized;
  } catch (_) {
    return createEmptyState();
  }
}

function saveEvalState() {
  localStorage.setItem(EVAL_STORAGE_KEY, JSON.stringify(state));
}

function restoreGeneratedQuerySet(row, { select = false } = {}) {
  const normalized = normalizeQuerySet(row);
  const exists = state.query_sets.some((querySet) => querySet.id === normalized.id);
  if (!exists) {
    state.query_sets.push(normalized);
  }
  if (select) {
    currentQuerySetId = normalized.id;
  }
  if (!exists || select) {
    saveEvalState();
  }
  return { normalized, restored: !exists };
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

function setGeneratorStatus(message, { error = false } = {}) {
  els.generatorStatus.textContent = message;
  els.generatorStatus.classList.toggle("error", error);
}

function selectedGeneratorVideoIds() {
  return Array.from(
    els.generatorVideoList.querySelectorAll("input[type='checkbox']:checked"),
  ).map((input) => input.value);
}

function renderGeneratorCapability() {
  if (!generatorCapability) {
    els.generatorCapability.textContent = t("generatorChecking");
    els.generatorCapability.className = "eval-generator-capability pending";
    return;
  }
  const ready = Boolean(
    generatorCapability.available && generatorCapability.authenticated,
  );
  els.generatorCapability.textContent = ready
    ? `${t("generatorReady")} · ${generatorCapability.version || "codex"}`
    : `${t("generatorUnavailable")} · ${generatorCapability.message || ""}`;
  els.generatorCapability.className = `eval-generator-capability ${ready ? "ok" : "error"}`;
}

function renderGeneratorVideos() {
  const selected = new Set(selectedGeneratorVideoIds());
  if (!generatorVideos.length) {
    els.generatorVideoList.innerHTML = `<p class="section-desc">${escapeHtml(t("generatorNoVideos"))}</p>`;
    els.generatorSelectionCount.textContent = "0 / 3";
    updateGeneratorControls();
    return;
  }
  els.generatorVideoList.innerHTML = generatorVideos.map((video) => {
    const checked = selected.has(video.video_id) ? "checked" : "";
    return `
      <label class="eval-generator-video-option">
        <input type="checkbox" value="${escapeHtml(video.video_id)}" ${checked} />
        <span>
          <strong class="eval-generator-video-title">${escapeHtml(video.title)}</strong>
          <span class="eval-generator-video-meta">${escapeHtml(video.language || "-")} · ${Number(video.num_chunks || 0)} chunks</span>
        </span>
      </label>
    `;
  }).join("");
  updateGeneratorControls();
}

function renderGeneratorDraftSelector() {
  if (!generatorDrafts.length) {
    els.generatorDraftSelect.innerHTML = `<option value="">${escapeHtml(t("generatorNoDrafts"))}</option>`;
    els.loadGeneratorDraftBtn.disabled = true;
    return;
  }
  els.generatorDraftSelect.innerHTML = generatorDrafts.map((draft) => {
    const label = `${String(draft.created_at || "").slice(0, 19).replace("T", " ")} · ${draft.decided_count}/${draft.case_count}`;
    return `<option value="${escapeHtml(draft.draft_id)}">${escapeHtml(label)}</option>`;
  }).join("");
  if (currentGeneratorDraft?.draft_id) {
    els.generatorDraftSelect.value = currentGeneratorDraft.draft_id;
  }
  els.loadGeneratorDraftBtn.disabled = false;
}

function updateGeneratorControls() {
  const selectionCount = selectedGeneratorVideoIds().length;
  const ready = Boolean(
    generatorCapability?.available && generatorCapability?.authenticated,
  );
  els.generatorSelectionCount.textContent = `${selectionCount} / 3`;
  els.generateEvalDatasetBtn.disabled = !ready
    || selectionCount < 1
    || selectionCount > 3
    || Boolean(activeGeneratorJobId);
  els.generatorVideoList.querySelectorAll("input[type='checkbox']").forEach((input) => {
    input.disabled = Boolean(activeGeneratorJobId)
      || (!input.checked && selectionCount >= 3);
  });
}

function setGeneratorStep(step) {
  const steps = [
    ["select", els.generatorStepSelect],
    ["generate", els.generatorStepGenerate],
    ["review", els.generatorStepReview],
    ["finalize", els.generatorStepFinalize],
  ];
  const activeIndex = Math.max(0, steps.findIndex(([name]) => name === step));
  steps.forEach(([, element], index) => {
    element.classList.toggle("active", index === activeIndex);
    element.classList.toggle("complete", index < activeIndex);
  });
}

function generatorCaseValues(caseRow) {
  const finalValues = caseRow.review?.final_values || {};
  return {
    query: finalValues.query || caseRow.query,
    required_facts: finalValues.required_facts || caseRow.required_facts || [],
    gold_evidence: finalValues.gold_evidence || caseRow.gold_evidence || [],
    difficulty: finalValues.difficulty || caseRow.difficulty,
    notes: finalValues.notes ?? caseRow.notes ?? "",
  };
}

function generatorDecisionLabel(decision) {
  return {
    approved: t("generatorApproved"),
    edited: t("generatorEdited"),
    rejected: t("generatorRejected"),
    pending: t("generatorPending"),
  }[decision] || t("generatorPending");
}

function renderGeneratorReview() {
  const draft = currentGeneratorDraft;
  if (!draft) {
    els.generatorWarnings.hidden = true;
    els.generatorReviewCards.innerHTML = "";
    els.generatorFinalizeActions.hidden = true;
    setGeneratorStep(activeGeneratorJobId ? "generate" : "select");
    return;
  }
  setGeneratorStep(draft.status === "finalized" ? "finalize" : "review");
  const warnings = Array.isArray(draft.warnings) ? draft.warnings : [];
  els.generatorWarnings.hidden = warnings.length === 0;
  els.generatorWarnings.innerHTML = warnings.length
    ? `<strong>${escapeHtml(t("generatorWarningsHeading"))}</strong><ul>${warnings.map((warning) => `<li>${escapeHtml(warning)}</li>`).join("")}</ul>`
    : "";

  const cases = Array.isArray(draft.cases) ? draft.cases : [];
  els.generatorReviewCards.innerHTML = cases.map((caseRow, index) => {
    const decision = caseRow.review?.decision || "pending";
    const values = generatorCaseValues(caseRow);
    const isEditing = editingGeneratorCaseId === caseRow.id;
    const keptEvidenceIds = new Set(
      values.gold_evidence.map((row) => row.evidence_id),
    );
    const badges = [caseRow.case_type, caseRow.language, values.difficulty, decision]
      .map((value) => `<span class="eval-generator-case-badge">${escapeHtml(value)}</span>`)
      .join("");
    const warningsHtml = [...(caseRow.warnings || []), ...(caseRow.risk_flags || [])]
      .map((warning) => `<span class="eval-generator-case-badge">⚠ ${escapeHtml(warning)}</span>`)
      .join("");
    const evidenceHtml = (caseRow.gold_evidence || []).map((evidence) => `
      <label class="eval-generator-evidence">
        <input type="checkbox" data-evidence-id="${escapeHtml(evidence.evidence_id)}" ${keptEvidenceIds.has(evidence.evidence_id) ? "checked" : ""} ${isEditing ? "" : "disabled"} />
        <span>
          <a href="${escapeHtml(evidence.url)}" target="_blank" rel="noreferrer">${escapeHtml(evidence.video_title)} · ${formatSeconds(evidence.start)} · chunk ${Number(evidence.chunk_index)}</a>
          <p>${escapeHtml(truncate(evidence.text, 360))}</p>
        </span>
      </label>
    `).join("");
    const editHtml = isEditing ? `
      <div class="eval-generator-edit-grid">
        <label class="full">
          <span>${escapeHtml(t("generatorQuestionLabel"))}</span>
          <input data-edit-field="query" value="${escapeHtml(values.query)}" />
        </label>
        <label>
          <span>${escapeHtml(t("generatorDifficultyLabel"))}</span>
          <select data-edit-field="difficulty">
            ${["easy", "medium", "hard"].map((value) => `<option value="${value}" ${values.difficulty === value ? "selected" : ""}>${value}</option>`).join("")}
          </select>
        </label>
        <label>
          <span>${escapeHtml(t("generatorNotesLabel"))}</span>
          <input data-edit-field="notes" value="${escapeHtml(values.notes)}" />
        </label>
        <label class="full">
          <span>${escapeHtml(t("generatorFactsLabel"))}</span>
          <textarea data-edit-field="required_facts">${escapeHtml(values.required_facts.join("\n"))}</textarea>
        </label>
      </div>
    ` : `
      <div>
        <strong class="eval-generator-video-meta">${escapeHtml(t("generatorFactsHeading"))}</strong>
        <ul class="eval-guidelines">${values.required_facts.map((fact) => `<li>${escapeHtml(fact)}</li>`).join("")}</ul>
      </div>
    `;
    let actions = "";
    if (isEditing) {
      actions = `
        <button class="btn" type="button" data-generator-action="save-edit">${escapeHtml(t("generatorSaveEdit"))}</button>
        <button class="btn secondary" type="button" data-generator-action="cancel-edit">${escapeHtml(t("generatorCancelEdit"))}</button>
      `;
    } else {
      actions = `
        ${decision !== "edited" ? `<button class="btn secondary ${decision === "approved" ? "active" : ""}" type="button" data-generator-action="approve">${escapeHtml(t("generatorApprove"))}</button>` : ""}
        <button class="btn secondary" type="button" data-generator-action="edit">${escapeHtml(t("generatorEdit"))}</button>
        <button class="btn secondary danger" type="button" data-generator-action="reject">${escapeHtml(t("generatorReject"))}</button>
      `;
    }
    return `
      <article class="eval-generator-case ${escapeHtml(decision)}" data-case-id="${escapeHtml(caseRow.id)}">
        <div class="eval-generator-case-head">
          <h3 class="eval-generator-case-title">${index + 1}. ${escapeHtml(values.query)}</h3>
          <span class="review-status ${decision === "rejected" ? "error" : decision === "pending" ? "pending" : "ok"}">${escapeHtml(generatorDecisionLabel(decision))}</span>
        </div>
        <div class="eval-generator-case-badges">${badges}${warningsHtml}</div>
        ${editHtml}
        <div class="eval-generator-evidence-list">
          <strong class="eval-generator-video-meta">${escapeHtml(t("generatorEvidenceHeading"))}</strong>
          ${evidenceHtml}
        </div>
        <div class="eval-generator-case-actions">${actions}</div>
      </article>
    `;
  }).join("");

  const counts = cases.reduce((acc, caseRow) => {
    const decision = caseRow.review?.decision || "pending";
    if (decision !== "pending") acc.decided += 1;
    if (decision === "approved" || decision === "edited") acc.accepted += 1;
    return acc;
  }, { decided: 0, accepted: 0 });
  els.generatorFinalizeActions.hidden = false;
  els.generatorReviewSummary.textContent = t("generatorReviewSummary", {
    decided: counts.decided,
    total: cases.length,
    accepted: counts.accepted,
  });
  els.finalizeEvalDatasetBtn.disabled = draft.status === "finalized"
    || counts.decided !== cases.length
    || counts.accepted < 1;
}

async function refreshGeneratorDrafts() {
  const response = await apiRequest("/v1/eval-generator/drafts");
  generatorDrafts = Array.isArray(response.drafts) ? response.drafts : [];
  renderGeneratorDraftSelector();
}

async function loadGeneratorDraft(draftId) {
  if (!draftId) return;
  const response = await apiRequest(`/v1/eval-generator/drafts/${encodeURIComponent(draftId)}`);
  currentGeneratorDraft = response.draft;
  editingGeneratorCaseId = null;
  renderGeneratorDraftSelector();
  renderGeneratorReview();
  if (currentGeneratorDraft.status === "finalized" && currentGeneratorDraft.dataset_id) {
    const datasetResponse = await apiRequest(
      `/v1/eval-generator/datasets/${encodeURIComponent(currentGeneratorDraft.dataset_id)}`,
    );
    restoreGeneratedQuerySet(datasetResponse.query_set, { select: true });
    els.downloadEvalDatasetLink.href = `/v1/eval-generator/datasets/${encodeURIComponent(currentGeneratorDraft.dataset_id)}/export`;
    els.downloadEvalDatasetLink.download = `${currentGeneratorDraft.dataset_id}.jsonl`;
    els.downloadEvalDatasetLink.hidden = false;
    rerender();
  } else {
    els.downloadEvalDatasetLink.hidden = true;
  }
  setGeneratorStatus(t("generatorDraftReady"));
}

async function pollGeneratorJob(jobId) {
  if (generatorPollTimer) {
    clearTimeout(generatorPollTimer);
    generatorPollTimer = null;
  }
  try {
    const response = await apiRequest(`/v1/eval-generator/jobs/${encodeURIComponent(jobId)}`);
    const job = response.job;
    if (job.status === "completed") {
      activeGeneratorJobId = null;
      await refreshGeneratorDrafts();
      await loadGeneratorDraft(job.draft_id);
      updateGeneratorControls();
      return;
    }
    if (job.status === "failed") {
      activeGeneratorJobId = null;
      setGeneratorStatus(t("generatorFailed", { message: job.error_message || job.error_code }), { error: true });
      setGeneratorStep("select");
      updateGeneratorControls();
      return;
    }
    setGeneratorStatus(t("generatorStepStatus", { step: job.step || job.status }));
    setGeneratorStep("generate");
    generatorPollTimer = setTimeout(() => {
      pollGeneratorJob(jobId);
    }, 1200);
  } catch (err) {
    activeGeneratorJobId = null;
    setGeneratorStatus(t("generatorFailed", { message: String(err.message || err) }), { error: true });
    updateGeneratorControls();
  }
}

async function startGeneratorJob() {
  const videoIds = selectedGeneratorVideoIds();
  if (videoIds.length < 1 || videoIds.length > 3) return;
  els.downloadEvalDatasetLink.hidden = true;
  currentGeneratorDraft = null;
  editingGeneratorCaseId = null;
  renderGeneratorReview();
  setGeneratorStatus(t("generatorQueued"));
  setGeneratorStep("generate");
  const response = await apiRequest("/v1/eval-generator/jobs", {
    method: "POST",
    body: { video_ids: videoIds },
  });
  activeGeneratorJobId = response.job.job_id;
  updateGeneratorControls();
  await pollGeneratorJob(activeGeneratorJobId);
}

async function saveGeneratorDecision(caseRow, payload) {
  const response = await apiRequest(
    `/v1/eval-generator/drafts/${encodeURIComponent(currentGeneratorDraft.draft_id)}/review`,
    { method: "POST", body: { decisions: [{ id: caseRow.id, ...payload }] } },
  );
  currentGeneratorDraft = response.draft;
  editingGeneratorCaseId = null;
  await refreshGeneratorDrafts();
  renderGeneratorReview();
}

async function finalizeGeneratorDraft() {
  if (!currentGeneratorDraft) return;
  const response = await apiRequest(
    `/v1/eval-generator/drafts/${encodeURIComponent(currentGeneratorDraft.draft_id)}/finalize`,
    { method: "POST", body: {} },
  );
  restoreGeneratedQuerySet(response.query_set, { select: true });
  currentGeneratorDraft.status = "finalized";
  currentGeneratorDraft.dataset_id = response.dataset.dataset_id;
  els.downloadEvalDatasetLink.href = response.export_url;
  els.downloadEvalDatasetLink.download = `${response.dataset.dataset_id}.jsonl`;
  els.downloadEvalDatasetLink.hidden = false;
  setGeneratorStep("finalize");
  setGeneratorStatus(t("generatorFinalized", { count: response.dataset.row_count }));
  rerender();
  renderGeneratorReview();
}

async function initEvalGenerator() {
  setGeneratorStatus(t("generatorLoading"));
  try {
    const [capabilityResponse, videosResponse, draftsResponse, datasetsResponse] = await Promise.all([
      apiRequest("/v1/eval-generator/capabilities"),
      apiRequest("/v1/videos"),
      apiRequest("/v1/eval-generator/drafts"),
      apiRequest("/v1/eval-generator/datasets"),
    ]);
    generatorCapability = capabilityResponse.capabilities || null;
    generatorVideos = (videosResponse.videos || []).filter((video) => Number(video.num_chunks || 0) > 0);
    generatorDrafts = draftsResponse.drafts || [];
    let restoredCount = 0;
    let selectedRestoredSet = false;
    for (const dataset of datasetsResponse.datasets || []) {
      const result = restoreGeneratedQuerySet(dataset.query_set);
      if (result.restored) {
        restoredCount += 1;
        if (!selectedRestoredSet) {
          currentQuerySetId = result.normalized.id;
          selectedRestoredSet = true;
        }
      }
    }
    renderGeneratorCapability();
    renderGeneratorVideos();
    renderGeneratorDraftSelector();
    if (restoredCount > 0) rerender();
    setGeneratorStatus(t("generatorSelectPrompt"));
  } catch (err) {
    generatorCapability = {
      available: false,
      authenticated: false,
      message: String(err.message || err),
    };
    renderGeneratorCapability();
    setGeneratorStatus(t("generatorFailed", { message: String(err.message || err) }), { error: true });
    updateGeneratorControls();
  }
}

function selectedQuerySet() {
  return state.query_sets.find((row) => row.id === currentQuerySetId) || null;
}

function selectedRun() {
  return state.runs.find((row) => row.id === currentRunId) || null;
}

function selectedRunResult(run) {
  if (!run) {
    return null;
  }
  return run.results.find((row) => row.query_id === currentRunQueryId) || run.results[0] || null;
}

function renderQuerySetSelector() {
  if (!state.query_sets.length) {
    state.query_sets.push(defaultQuerySet());
  }
  if (!currentQuerySetId || !state.query_sets.some((row) => row.id === currentQuerySetId)) {
    currentQuerySetId = state.query_sets[0].id;
  }
  els.querySetSelect.innerHTML = state.query_sets
    .map((row) => `<option value="${escapeHtml(row.id)}">${escapeHtml(row.name)}</option>`)
    .join("");
  els.querySetSelect.value = currentQuerySetId;
}

function renderQuerySetEditor() {
  const set = selectedQuerySet();
  if (!set) {
    els.queryRowsBody.innerHTML = `<tr><td colspan="5">${escapeHtml(t("noData"))}</td></tr>`;
    return;
  }

  els.querySetName.value = set.name;
  els.querySetLanguage.value = set.language;

  if (!set.queries.length) {
    els.queryRowsBody.innerHTML = `<tr><td colspan="5">${escapeHtml(t("noData"))}</td></tr>`;
    return;
  }

  els.queryRowsBody.innerHTML = set.queries.map((query, index) => {
    const typeOptions = QUERY_TYPES
      .map((type) => `<option value="${type}" ${query.type === type ? "selected" : ""}>${type}</option>`)
      .join("");
    return `
      <tr data-query-index="${index}">
        <td><input data-field="text" value="${escapeHtml(query.text)}" /></td>
        <td><select data-field="type">${typeOptions}</select></td>
        <td><input data-field="expected_relevant_min" type="number" min="0" value="${query.expected_relevant_min ?? ""}" /></td>
        <td><input data-field="notes" value="${escapeHtml(query.notes)}" /></td>
        <td><button class="btn secondary" type="button" data-action="remove-query">${escapeHtml(t("removeQuery"))}</button></td>
      </tr>
    `;
  }).join("");
}

function runLabel(run) {
  const when = String(run.completed_at || run.started_at || "").replace("T", " ").slice(0, 19);
  return `${run.id} | ${run.retrieval_mode} | k=${run.k} | ${when}`;
}

function renderRunSelectors() {
  const orderedRuns = state.runs.slice().sort((a, b) => String(b.completed_at).localeCompare(String(a.completed_at)));
  if (!orderedRuns.length) {
    currentRunId = null;
    currentRunQueryId = null;
    els.runSelect.innerHTML = `<option value="">${escapeHtml(t("noData"))}</option>`;
    els.compareRunA.innerHTML = `<option value="">${escapeHtml(t("noData"))}</option>`;
    els.compareRunB.innerHTML = `<option value="">${escapeHtml(t("noData"))}</option>`;
    return;
  }

  if (!currentRunId || !orderedRuns.some((row) => row.id === currentRunId)) {
    currentRunId = orderedRuns[0].id;
  }

  const options = orderedRuns
    .map((run) => `<option value="${escapeHtml(run.id)}">${escapeHtml(runLabel(run))}</option>`)
    .join("");
  els.runSelect.innerHTML = options;
  els.runSelect.value = currentRunId;

  const compareOptions = `<option value="">-</option>${options}`;
  els.compareRunA.innerHTML = compareOptions;
  els.compareRunB.innerHTML = compareOptions;

  if (orderedRuns.length >= 2) {
    els.compareRunA.value = orderedRuns[0].id;
    els.compareRunB.value = orderedRuns[1].id;
  } else {
    els.compareRunA.value = orderedRuns[0].id;
    els.compareRunB.value = "";
  }
}

function renderRunStatus() {
  const run = selectedRun();
  if (!run) {
    els.runStatus.textContent = t("runIdle");
    return;
  }
  els.runStatus.textContent = JSON.stringify({
    run_id: run.id,
    query_set_id: run.query_set_id,
    retrieval_mode: run.retrieval_mode,
    k: run.k,
    completed_at: run.completed_at,
    total_queries: run.results.length,
    metrics: run.metrics || null,
  }, null, 2);
}

function computeNdcgAtK(binaryRelevances, k) {
  const top = binaryRelevances.slice(0, k);
  let dcg = 0;
  top.forEach((rel, idx) => {
    const gain = rel > 0 ? 1 : 0;
    dcg += gain / Math.log2(idx + 2);
  });
  const ideal = binaryRelevances.slice().sort((a, b) => b - a).slice(0, k);
  let idcg = 0;
  ideal.forEach((rel, idx) => {
    const gain = rel > 0 ? 1 : 0;
    idcg += gain / Math.log2(idx + 2);
  });
  if (idcg === 0) {
    return 0;
  }
  return dcg / idcg;
}

function recomputeRunMetrics(run) {
  const perQuery = run.results.map((queryRun) => {
    const topItems = queryRun.items.slice(0, run.k);
    const relevances = topItems.map((item) => {
      const key = getChunkKey(item);
      const entry = queryRun.labels?.[key];
      return entry?.label === "relevant" ? 1 : 0;
    });
    const relevantInTopK = relevances.reduce((acc, value) => acc + value, 0);
    const precisionAtK = run.k > 0 ? (relevantInTopK / run.k) : 0;

    const totalRelevantLabeled = Object.values(queryRun.labels || {})
      .filter((entry) => entry?.label === "relevant")
      .length;
    const recallAtK = totalRelevantLabeled > 0
      ? (relevantInTopK / totalRelevantLabeled)
      : null;

    let mrr = 0;
    for (let idx = 0; idx < relevances.length; idx += 1) {
      if (relevances[idx] > 0) {
        mrr = 1 / (idx + 1);
        break;
      }
    }

    const ndcgAtK = computeNdcgAtK(relevances, run.k);
    const labeledCount = topItems
      .map((item) => queryRun.labels?.[getChunkKey(item)]?.label || null)
      .filter((label) => Boolean(label))
      .length;

    return {
      query_id: queryRun.query_id,
      query_text: queryRun.query_text,
      precision_at_k: precisionAtK,
      recall_at_k: recallAtK,
      mrr,
      ndcg_at_k: ndcgAtK,
      labeled_count: labeledCount,
      result_count: queryRun.result_count,
    };
  });

  const avg = (rows, key, includeNull = false) => {
    const values = rows
      .map((row) => row[key])
      .filter((value) => (includeNull ? true : value != null));
    if (!values.length) {
      return null;
    }
    return values.reduce((acc, value) => acc + Number(value || 0), 0) / values.length;
  };

  run.metrics = {
    aggregate: {
      precision_at_k: avg(perQuery, "precision_at_k", true),
      recall_at_k: avg(perQuery, "recall_at_k", false),
      mrr: avg(perQuery, "mrr", true),
      ndcg_at_k: avg(perQuery, "ndcg_at_k", true),
      total_queries: perQuery.length,
    },
    per_query: perQuery,
  };
}

function renderMetrics() {
  const run = selectedRun();
  const metrics = run?.metrics?.aggregate;
  els.metricPrecision.textContent = metrics ? formatMetric(metrics.precision_at_k) : "-";
  els.metricRecall.textContent = metrics ? formatMetric(metrics.recall_at_k) : "-";
  els.metricMRR.textContent = metrics ? formatMetric(metrics.mrr) : "-";
  els.metricNDCG.textContent = metrics ? formatMetric(metrics.ndcg_at_k) : "-";

  const rows = run?.metrics?.per_query || [];
  if (!rows.length) {
    els.metricsTableBody.innerHTML = `<tr><td colspan="6">${escapeHtml(t("noData"))}</td></tr>`;
    return;
  }

  els.metricsTableBody.innerHTML = rows.map((row) => `
    <tr>
      <td>${escapeHtml(truncate(row.query_text, 120))}</td>
      <td>${escapeHtml(formatMetric(row.precision_at_k))}</td>
      <td>${escapeHtml(formatMetric(row.recall_at_k))}</td>
      <td>${escapeHtml(formatMetric(row.mrr))}</td>
      <td>${escapeHtml(formatMetric(row.ndcg_at_k))}</td>
      <td>${escapeHtml(`${row.labeled_count}/${Math.min(run.k, row.result_count || run.k)}`)}</td>
    </tr>
  `).join("");
}

function renderRunQuerySelector() {
  const run = selectedRun();
  if (!run || !run.results.length) {
    currentRunQueryId = null;
    els.runQuerySelect.innerHTML = `<option value="">${escapeHtml(t("noData"))}</option>`;
    return;
  }

  if (!currentRunQueryId || !run.results.some((row) => row.query_id === currentRunQueryId)) {
    currentRunQueryId = run.results[0].query_id;
  }

  els.runQuerySelect.innerHTML = run.results
    .map((row) => `<option value="${escapeHtml(row.query_id)}">${escapeHtml(row.query_text)}</option>`)
    .join("");
  els.runQuerySelect.value = currentRunQueryId;
}

function renderRunResultCards() {
  const run = selectedRun();
  const queryRun = selectedRunResult(run);
  if (!run || !queryRun) {
    els.runResultCards.innerHTML = `<div class="search-empty">${escapeHtml(t("reviewEmpty"))}</div>`;
    return;
  }
  if (queryRun.error) {
    els.runResultCards.innerHTML = `<div class="search-empty">${escapeHtml(queryRun.error)}</div>`;
    return;
  }
  if (!queryRun.items.length) {
    els.runResultCards.innerHTML = `<div class="search-empty">${escapeHtml(t("noData"))}</div>`;
    return;
  }

  els.runResultCards.innerHTML = queryRun.items.map((item, idx) => {
    const key = getChunkKey(item);
    const labelState = queryRun.labels?.[key] || {};
    const active = labelState.label || "";
    const reason = labelState.reason_code || "";
    const reasonOptions = [`<option value="">${escapeHtml(t("reasonPlaceholder"))}</option>`, ...REASON_CODES.map((code) => (
      `<option value="${code}" ${code === reason ? "selected" : ""}>${escapeHtml(code)}</option>`
    ))].join("");

    return `
      <article class="search-card">
        <div class="search-card-head">
          <div class="search-rank">#${escapeHtml(item.rank ?? idx + 1)}</div>
          <div class="search-title">${escapeHtml(item.video_title || item.video_id || "-")}</div>
          <div class="search-lang">${escapeHtml(item.language || "-")}</div>
        </div>
        <div class="search-meta">
          <span>${escapeHtml(t("time"))}: ${escapeHtml(formatSeconds(item.start))} - ${escapeHtml(formatSeconds(item.end))}</span>
          <span>${escapeHtml(t("score"))}: ${escapeHtml(Number(item.score ?? 0).toFixed(4))} · ${escapeHtml(t("rank"))}: ${escapeHtml(item.rank ?? idx + 1)}</span>
        </div>
        <p class="search-snippet">${escapeHtml(truncate(item.text, 360))}</p>
        <div class="eval-review-actions">
          <a class="btn secondary" href="${escapeHtml(item.url || "#")}" target="_blank" rel="noopener noreferrer">Open</a>
          <div class="review-group">
            <button class="btn secondary review-btn ${active === "relevant" ? "active relevant" : ""}" type="button" data-action="label" data-label="relevant" data-key="${escapeHtml(key)}">${escapeHtml(t("labelRelevant"))}</button>
            <button class="btn secondary review-btn ${active === "not_relevant" ? "active not-relevant" : ""}" type="button" data-action="label" data-label="not_relevant" data-key="${escapeHtml(key)}">${escapeHtml(t("labelNotRelevant"))}</button>
            <button class="btn secondary review-btn ${active === "unsure" ? "active" : ""}" type="button" data-action="label" data-label="unsure" data-key="${escapeHtml(key)}">${escapeHtml(t("labelUnsure"))}</button>
          </div>
          <label class="eval-reason-wrap">
            <span>${escapeHtml(t("reason"))}</span>
            <select data-action="reason" data-key="${escapeHtml(key)}" ${active === "relevant" ? "disabled" : ""}>${reasonOptions}</select>
          </label>
        </div>
      </article>
    `;
  }).join("");
}

function renderCompareOutput(text) {
  els.compareResult.textContent = text;
}

function rerender() {
  renderQuerySetSelector();
  renderQuerySetEditor();
  renderRunSelectors();
  renderRunStatus();
  renderRunQuerySelector();
  renderMetrics();
  renderRunResultCards();
}

function downloadJson(filename, payload) {
  const blob = new Blob([JSON.stringify(payload, null, 2)], { type: "application/json;charset=utf-8" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(url);
}

function updateQuerySetField(field, value) {
  const set = selectedQuerySet();
  if (!set) {
    return;
  }
  set[field] = value;
  saveEvalState();
}

function updateQueryField(index, field, value) {
  const set = selectedQuerySet();
  if (!set || !set.queries[index]) {
    return;
  }
  if (field === "expected_relevant_min") {
    set.queries[index][field] = parseOptionalInt(value);
  } else {
    set.queries[index][field] = value;
  }
  saveEvalState();
}

function addQueryRow() {
  const set = selectedQuerySet();
  if (!set) {
    return;
  }
  set.queries.push({
    id: makeId("q"),
    text: "",
    type: "factual",
    expected_relevant_min: null,
    notes: "",
  });
  saveEvalState();
  renderQuerySetEditor();
}

function removeQueryRow(index) {
  const set = selectedQuerySet();
  if (!set || !set.queries[index]) {
    return;
  }
  set.queries.splice(index, 1);
  saveEvalState();
  renderQuerySetEditor();
}

function createQuerySet() {
  const newSet = {
    id: makeId("qs"),
    name: `Query Set ${state.query_sets.length + 1}`,
    language: "mixed",
    created_at: nowIso(),
    queries: [],
  };
  state.query_sets.push(newSet);
  currentQuerySetId = newSet.id;
  saveEvalState();
  rerender();
}

function saveQuerySet() {
  const set = selectedQuerySet();
  if (!set) {
    return;
  }
  set.name = (els.querySetName.value || "").trim() || "Untitled Query Set";
  set.language = els.querySetLanguage.value;
  set.queries = set.queries
    .map(normalizeQuery)
    .filter((row) => row.text.trim().length > 0);
  saveEvalState();
  rerender();
}

function deleteQuerySet() {
  const set = selectedQuerySet();
  if (!set) {
    return;
  }
  if (!window.confirm(t("deleteSetConfirm"))) {
    return;
  }
  state.query_sets = state.query_sets.filter((row) => row.id !== set.id);
  if (!state.query_sets.length) {
    state.query_sets.push(defaultQuerySet());
  }
  currentQuerySetId = state.query_sets[0].id;
  saveEvalState();
  rerender();
}

function exportQuerySet() {
  const set = selectedQuerySet();
  if (!set) {
    return;
  }
  try {
    downloadJson(`query_set_${set.id}.json`, {
      type: "yt_rag_query_set_v1",
      exported_at: nowIso(),
      query_set: set,
    });
  } catch (err) {
    els.runStatus.textContent = t("exportFailed", { message: String(err.message || err) });
  }
}

function importQuerySetFromObject(payload) {
  const candidate = payload?.query_set || payload;
  if (!candidate || typeof candidate !== "object") {
    throw new Error("Invalid query set payload");
  }
  const normalized = normalizeQuerySet(candidate);
  const exists = state.query_sets.some((row) => row.id === normalized.id);
  if (exists) {
    normalized.id = makeId("qs");
    normalized.name = `${normalized.name} (imported)`;
  }
  state.query_sets.push(normalized);
  currentQuerySetId = normalized.id;
  saveEvalState();
  rerender();
}

async function importQuerySetFromFile(file) {
  const text = await file.text();
  const payload = JSON.parse(text);
  importQuerySetFromObject(payload);
}

function setRunStatus(message) {
  els.runStatus.textContent = message;
}

async function runQuerySet() {
  const set = selectedQuerySet();
  if (!set) {
    setRunStatus(t("runMissingSet"));
    return;
  }
  const queries = set.queries.filter((row) => row.text.trim().length > 0);
  if (!queries.length) {
    setRunStatus(t("runEmptySet"));
    return;
  }

  const mode = els.runMode.value;
  const k = Math.max(1, Math.min(12, Number(els.runK.value || 5)));
  els.runK.value = String(k);

  const run = {
    id: makeId("run"),
    query_set_id: set.id,
    started_at: nowIso(),
    completed_at: "",
    retrieval_mode: mode,
    k,
    candidate_k: Math.max(20, k * 4),
    index_snapshot_id: "local_library_latest",
    system_version: {
      app_version: APP_VERSION,
      retriever_version: mode,
      chunking_version: "unknown",
    },
    results: [],
    metrics: null,
  };

  els.runQuerySetBtn.disabled = true;
  try {
    for (let i = 0; i < queries.length; i += 1) {
      const query = queries[i];
      setRunStatus(t("runProgress", {
        index: i + 1,
        total: queries.length,
        query: query.text,
      }));

      try {
        const response = await apiRequest("/v1/search", {
          method: "POST",
          body: {
            query: query.text,
            k,
            retrieval_mode: mode,
            language: set.language === "mixed" ? undefined : set.language,
          },
        });
        run.results.push({
          query_id: query.id,
          query_text: query.text,
          query_type: query.type,
          expected_relevant_min: query.expected_relevant_min,
          result_count: response.result_count || 0,
          retrieval_details: response.retrieval_details || {},
          items: Array.isArray(response.results) ? response.results : [],
          labels: {},
          error: null,
        });
      } catch (err) {
        const message = String(err?.message || err);
        run.results.push({
          query_id: query.id,
          query_text: query.text,
          query_type: query.type,
          expected_relevant_min: query.expected_relevant_min,
          result_count: 0,
          retrieval_details: {},
          items: [],
          labels: {},
          error: t("runFailedQuery", { query: query.text, message }),
        });
      }
    }

    run.completed_at = nowIso();
    recomputeRunMetrics(run);
    state.runs.unshift(run);
    currentRunId = run.id;
    currentRunQueryId = run.results[0]?.query_id || null;
    saveEvalState();
    rerender();
    setRunStatus(t("runDone", { total: queries.length }));
  } finally {
    els.runQuerySetBtn.disabled = false;
  }
}

function setLabelForSelectedQuery(key, label) {
  const run = selectedRun();
  const queryRun = selectedRunResult(run);
  if (!run || !queryRun || !key) {
    return;
  }
  const existing = queryRun.labels[key] || {};
  queryRun.labels[key] = {
    label,
    reason_code: label === "relevant" ? null : (existing.reason_code || "other"),
    updated_at: nowIso(),
  };
  recomputeRunMetrics(run);
  saveEvalState();
  renderMetrics();
  renderRunResultCards();
  renderRunStatus();
}

function setReasonForSelectedQuery(key, reasonCode) {
  const run = selectedRun();
  const queryRun = selectedRunResult(run);
  if (!run || !queryRun || !key) {
    return;
  }
  const existing = queryRun.labels[key];
  if (!existing || existing.label === "relevant") {
    return;
  }
  queryRun.labels[key] = {
    ...existing,
    reason_code: reasonCode || null,
    updated_at: nowIso(),
  };
  saveEvalState();
  renderRunResultCards();
}

function exportRunBundle() {
  const run = selectedRun();
  if (!run) {
    return;
  }
  const set = state.query_sets.find((row) => row.id === run.query_set_id) || null;
  try {
    downloadJson(`eval_run_${run.id}.json`, {
      type: "yt_rag_run_bundle_v1",
      exported_at: nowIso(),
      query_set: set,
      run,
    });
  } catch (err) {
    setRunStatus(t("exportFailed", { message: String(err?.message || err) }));
  }
}

function resetEvalData() {
  if (!window.confirm(t("resetEvalConfirm"))) {
    return;
  }
  state = createEmptyState();
  currentQuerySetId = state.query_sets[0].id;
  currentRunId = null;
  currentRunQueryId = null;
  saveEvalState();
  rerender();
  renderCompareOutput("");
  setRunStatus(t("runIdle"));
}

function compareRuns() {
  const runAId = els.compareRunA.value;
  const runBId = els.compareRunB.value;
  if (!runAId || !runBId || runAId === runBId) {
    renderCompareOutput(t("compareNeedTwo"));
    return;
  }
  const runA = state.runs.find((row) => row.id === runAId);
  const runB = state.runs.find((row) => row.id === runBId);
  if (!runA?.metrics?.aggregate || !runB?.metrics?.aggregate) {
    renderCompareOutput(t("compareNoMetrics"));
    return;
  }

  const delta = (a, b) => (a == null || b == null ? null : Number(b) - Number(a));

  const perQueryA = Array.isArray(runA.metrics?.per_query) ? runA.metrics.per_query : [];
  const perQueryB = Array.isArray(runB.metrics?.per_query) ? runB.metrics.per_query : [];
  const byQueryA = new Map(perQueryA.map((row) => [String(row.query_text || "").trim(), row]));
  const byQueryB = new Map(perQueryB.map((row) => [String(row.query_text || "").trim(), row]));
  const sharedQueries = Array.from(byQueryA.keys()).filter((key) => byQueryB.has(key));
  const perQueryDelta = sharedQueries.map((queryText) => {
    const a = byQueryA.get(queryText);
    const b = byQueryB.get(queryText);
    return {
      query: queryText,
      precision_at_k: delta(a?.precision_at_k, b?.precision_at_k),
      recall_at_k: delta(a?.recall_at_k, b?.recall_at_k),
      mrr: delta(a?.mrr, b?.mrr),
      ndcg_at_k: delta(a?.ndcg_at_k, b?.ndcg_at_k),
    };
  });

  const output = {
    run_a: {
      id: runA.id,
      mode: runA.retrieval_mode,
      k: runA.k,
      completed_at: runA.completed_at,
      metrics: runA.metrics.aggregate,
    },
    run_b: {
      id: runB.id,
      mode: runB.retrieval_mode,
      k: runB.k,
      completed_at: runB.completed_at,
      metrics: runB.metrics.aggregate,
    },
    delta: {
      precision_at_k: delta(runA.metrics.aggregate.precision_at_k, runB.metrics.aggregate.precision_at_k),
      recall_at_k: delta(runA.metrics.aggregate.recall_at_k, runB.metrics.aggregate.recall_at_k),
      mrr: delta(runA.metrics.aggregate.mrr, runB.metrics.aggregate.mrr),
      ndcg_at_k: delta(runA.metrics.aggregate.ndcg_at_k, runB.metrics.aggregate.ndcg_at_k),
    },
    per_query_delta: perQueryDelta,
  };

  renderCompareOutput(JSON.stringify(output, null, 2));
}

function applyLocale(locale) {
  if (!I18N[locale]) {
    return;
  }
  currentLocale = locale;
  localStorage.setItem(LOCALE_STORAGE_KEY, locale);
  document.documentElement.lang = locale === "ja-JP" ? "ja" : "en";
  document.title = t("pageTitle");

  els.localeSelect.value = locale;
  els.localeLabel.textContent = t("localeLabel");
  els.homeBrandLink?.setAttribute("aria-label", t("navIngest"));
  els.eyebrowText.textContent = t("eyebrowText");
  setNavLabel(els.homeNavLink, t("navIngest"));
  setNavLabel(els.studioNavLink, t("navStudio"));
  setNavLabel(els.reviewsNavLink, t("navReviews"));
  setNavLabel(els.evidenceNavLink, t("navEvidence"));
  setNavLabel(els.evaluationNavLink, t("navEvaluation"));
  setNavLabel(els.chunkingNavLink, t("navChunking"));
  if (els.mobileHomeNavText) {
    els.mobileHomeNavText.textContent = t("navIngest");
  }
  if (els.mobileStudioNavText) {
    els.mobileStudioNavText.textContent = t("navStudio");
  }
  if (els.mobileReviewsNavText) {
    els.mobileReviewsNavText.textContent = t("navReviews");
  }
  if (els.mobileEvidenceNavText) {
    els.mobileEvidenceNavText.textContent = t("navEvidence");
  }
  if (els.mobileEvaluationNavText) {
    els.mobileEvaluationNavText.textContent = t("navEvaluation");
  }
  if (els.mobileChunkingNavText) {
    els.mobileChunkingNavText.textContent = t("navChunking");
  }
  els.heroTitle.textContent = t("heroTitle");
  els.heroSubtitle.textContent = t("heroSubtitle");
  els.guidelinesHeading.textContent = t("guidelinesHeading");
  els.guidelinesList.innerHTML = (I18N[currentLocale].guidelines || [])
    .map((line) => `<li>${escapeHtml(line)}</li>`)
    .join("");

  els.generatorHeading.textContent = t("generatorHeading");
  els.generatorIntro.textContent = t("generatorIntro");
  els.generatorVideosHeading.textContent = t("generatorVideosHeading");
  els.generatorDraftLabel.textContent = t("generatorDraftLabel");
  els.loadGeneratorDraftBtn.textContent = t("loadGeneratorDraftBtn");
  els.generateEvalDatasetBtn.textContent = t("generateEvalDatasetBtn");
  els.finalizeEvalDatasetBtn.textContent = t("generatorCreateQuerySet");
  els.downloadEvalDatasetLink.textContent = t("generatorDownload");
  els.generatorStepSelect.querySelector("strong").textContent = t("generatorStepSelectLabel");
  els.generatorStepGenerate.querySelector("strong").textContent = t("generatorStepGenerateLabel");
  els.generatorStepReview.querySelector("strong").textContent = t("generatorStepReviewLabel");
  els.generatorStepFinalize.querySelector("strong").textContent = t("generatorStepFinalizeLabel");
  renderGeneratorCapability();
  renderGeneratorVideos();
  renderGeneratorDraftSelector();
  renderGeneratorReview();

  els.querySetsHeading.textContent = t("querySetsHeading");
  els.newQuerySetBtn.textContent = t("newQuerySetBtn");
  els.saveQuerySetBtn.textContent = t("saveQuerySetBtn");
  els.deleteQuerySetBtn.textContent = t("deleteQuerySetBtn");
  els.exportQuerySetBtn.textContent = t("exportQuerySetBtn");
  els.importQuerySetBtn.textContent = t("importQuerySetBtn");
  els.querySetSelectLabel.textContent = t("querySetSelectLabel");
  els.querySetNameLabel.textContent = t("querySetNameLabel");
  els.querySetName.placeholder = t("querySetNamePlaceholder");
  els.querySetLanguageLabel.textContent = t("querySetLanguageLabel");
  els.thQueryText.textContent = t("thQueryText");
  els.thQueryType.textContent = t("thQueryType");
  els.thExpectedMin.textContent = t("thExpectedMin");
  els.thQueryNotes.textContent = t("thQueryNotes");
  els.thQueryActions.textContent = t("thQueryActions");
  els.addQueryRowBtn.textContent = t("addQueryRowBtn");

  els.runsHeading.textContent = t("runsHeading");
  els.runQuerySetBtn.textContent = t("runQuerySetBtn");
  els.exportRunBtn.textContent = t("exportRunBtn");
  els.resetEvalDataBtn.textContent = t("resetEvalDataBtn");
  els.runModeLabel.textContent = t("runModeLabel");
  els.runKLabel.textContent = t("runKLabel");
  els.runSelectLabel.textContent = t("runSelectLabel");

  els.metricsHeading.textContent = t("metricsHeading");
  els.thMetricQuery.textContent = t("thMetricQuery");
  els.thMetricPAtK.textContent = t("thMetricPAtK");
  els.thMetricRecall.textContent = t("thMetricRecall");
  els.thMetricMRR.textContent = t("thMetricMRR");
  els.thMetricNDCG.textContent = t("thMetricNDCG");
  els.thMetricLabeled.textContent = t("thMetricLabeled");

  els.reviewHeading.textContent = t("reviewHeading");
  els.runQuerySelectLabel.textContent = t("runQuerySelectLabel");
  els.resultReasonLabel.textContent = t("resultReasonLabel");

  els.compareHeading.textContent = t("compareHeading");
  els.compareRunALabel.textContent = t("compareRunALabel");
  els.compareRunBLabel.textContent = t("compareRunBLabel");
  els.compareRunsBtn.textContent = t("compareRunsBtn");

  rerender();
}

function wireEvents() {
  els.localeSelect.addEventListener("change", () => {
    applyLocale(els.localeSelect.value);
  });

  els.generatorVideoList.addEventListener("change", (event) => {
    if (event.target.matches("input[type='checkbox']")) {
      updateGeneratorControls();
    }
  });
  els.generateEvalDatasetBtn.addEventListener("click", () => {
    startGeneratorJob().catch((err) => {
      activeGeneratorJobId = null;
      setGeneratorStatus(
        t("generatorFailed", { message: String(err?.message || err) }),
        { error: true },
      );
      setGeneratorStep("select");
      updateGeneratorControls();
    });
  });
  els.loadGeneratorDraftBtn.addEventListener("click", () => {
    loadGeneratorDraft(els.generatorDraftSelect.value).catch((err) => {
      setGeneratorStatus(
        t("generatorFailed", { message: String(err?.message || err) }),
        { error: true },
      );
    });
  });
  els.generatorReviewCards.addEventListener("click", (event) => {
    const button = event.target.closest("button[data-generator-action]");
    const card = button?.closest("article[data-case-id]");
    if (!button || !card || !currentGeneratorDraft) return;
    const caseRow = currentGeneratorDraft.cases.find(
      (row) => row.id === card.dataset.caseId,
    );
    if (!caseRow) return;
    const action = button.dataset.generatorAction;
    if (action === "edit") {
      editingGeneratorCaseId = caseRow.id;
      renderGeneratorReview();
      return;
    }
    if (action === "cancel-edit") {
      editingGeneratorCaseId = null;
      renderGeneratorReview();
      return;
    }
    let payload;
    if (action === "approve" || action === "reject") {
      payload = { decision: action === "approve" ? "approved" : "rejected" };
    } else if (action === "save-edit") {
      const query = card.querySelector("[data-edit-field='query']").value;
      const facts = card.querySelector("[data-edit-field='required_facts']").value
        .split("\n")
        .map((value) => value.trim())
        .filter(Boolean);
      payload = {
        decision: "edited",
        query,
        required_facts: facts,
        difficulty: card.querySelector("[data-edit-field='difficulty']").value,
        notes: card.querySelector("[data-edit-field='notes']").value,
        kept_evidence_ids: Array.from(
          card.querySelectorAll("input[data-evidence-id]:checked"),
        ).map((input) => input.dataset.evidenceId),
      };
    } else {
      return;
    }
    button.disabled = true;
    saveGeneratorDecision(caseRow, payload).catch((err) => {
      button.disabled = false;
      setGeneratorStatus(
        t("generatorFailed", { message: String(err?.message || err) }),
        { error: true },
      );
    });
  });
  els.finalizeEvalDatasetBtn.addEventListener("click", () => {
    els.finalizeEvalDatasetBtn.disabled = true;
    finalizeGeneratorDraft().catch((err) => {
      setGeneratorStatus(
        t("generatorFailed", { message: String(err?.message || err) }),
        { error: true },
      );
      renderGeneratorReview();
    });
  });

  els.querySetSelect.addEventListener("change", () => {
    currentQuerySetId = els.querySetSelect.value;
    renderQuerySetEditor();
  });

  els.querySetName.addEventListener("input", () => {
    updateQuerySetField("name", els.querySetName.value);
    renderQuerySetSelector();
  });
  els.querySetLanguage.addEventListener("change", () => {
    updateQuerySetField("language", els.querySetLanguage.value);
  });

  els.queryRowsBody.addEventListener("input", (event) => {
    const target = event.target;
    if (!(target instanceof HTMLInputElement || target instanceof HTMLSelectElement)) {
      return;
    }
    const tr = target.closest("tr[data-query-index]");
    if (!tr) {
      return;
    }
    const index = Number(tr.getAttribute("data-query-index"));
    const field = target.getAttribute("data-field");
    if (!Number.isInteger(index) || !field) {
      return;
    }
    updateQueryField(index, field, target.value);
  });

  els.queryRowsBody.addEventListener("click", (event) => {
    const button = event.target.closest("button[data-action='remove-query']");
    if (!button) {
      return;
    }
    const tr = button.closest("tr[data-query-index]");
    if (!tr) {
      return;
    }
    const index = Number(tr.getAttribute("data-query-index"));
    removeQueryRow(index);
  });

  els.newQuerySetBtn.addEventListener("click", createQuerySet);
  els.saveQuerySetBtn.addEventListener("click", saveQuerySet);
  els.deleteQuerySetBtn.addEventListener("click", deleteQuerySet);
  els.exportQuerySetBtn.addEventListener("click", exportQuerySet);
  els.importQuerySetBtn.addEventListener("click", () => {
    els.importQuerySetFile.value = "";
    els.importQuerySetFile.click();
  });
  els.importQuerySetFile.addEventListener("change", async () => {
    const file = els.importQuerySetFile.files?.[0];
    if (!file) {
      return;
    }
    try {
      await importQuerySetFromFile(file);
    } catch (err) {
      setRunStatus(t("importFailed", { message: String(err?.message || err) }));
    }
  });
  els.addQueryRowBtn.addEventListener("click", addQueryRow);

  els.runSelect.addEventListener("change", () => {
    currentRunId = els.runSelect.value || null;
    currentRunQueryId = null;
    renderRunStatus();
    renderRunQuerySelector();
    renderMetrics();
    renderRunResultCards();
  });

  els.runQuerySetBtn.addEventListener("click", () => {
    runQuerySet().catch((err) => {
      setRunStatus(String(err?.message || err));
      els.runQuerySetBtn.disabled = false;
    });
  });
  els.exportRunBtn.addEventListener("click", exportRunBundle);
  els.resetEvalDataBtn.addEventListener("click", resetEvalData);

  els.runQuerySelect.addEventListener("change", () => {
    currentRunQueryId = els.runQuerySelect.value || null;
    renderRunResultCards();
  });

  els.runResultCards.addEventListener("click", (event) => {
    const button = event.target.closest("button[data-action='label']");
    if (!button) {
      return;
    }
    const key = button.getAttribute("data-key");
    const label = button.getAttribute("data-label");
    if (!key || !label) {
      return;
    }
    setLabelForSelectedQuery(key, label);
  });

  els.runResultCards.addEventListener("change", (event) => {
    const select = event.target.closest("select[data-action='reason']");
    if (!select) {
      return;
    }
    const key = select.getAttribute("data-key");
    if (!key) {
      return;
    }
    setReasonForSelectedQuery(key, select.value);
  });

  els.compareRunsBtn.addEventListener("click", compareRuns);
}

function init() {
  if (state.runs.length) {
    currentRunId = state.runs[0].id;
  }
  currentQuerySetId = state.query_sets[0]?.id || null;
  wireEvents();
  applyLocale(currentLocale);
  if (!els.runStatus.textContent.trim()) {
    setRunStatus(t("runIdle"));
  }
  initEvalGenerator().catch((err) => {
    setGeneratorStatus(
      t("generatorFailed", { message: String(err?.message || err) }),
      { error: true },
    );
  });
}

init();
