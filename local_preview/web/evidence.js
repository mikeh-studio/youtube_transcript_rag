const INGEST_UNLOCK_KEY = "yt_rag_ingest_unlocked";
if (localStorage.getItem(INGEST_UNLOCK_KEY) !== "1") {
  window.location.replace("./index.html#/ingest");
}

const LOCALE_STORAGE_KEY = "youtube-rag-ui-locale";
const DEFAULT_LOCALE = "en-US";
const MANIFEST_LIMIT = 5000;

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
  emptyStatePanel: document.getElementById("emptyStatePanel"),
  emptyStateHeading: document.getElementById("emptyStateHeading"),
  emptyStateText: document.getElementById("emptyStateText"),
  emptyStateCommand: document.getElementById("emptyStateCommand"),
  runHeading: document.getElementById("runHeading"),
  refreshBtn: document.getElementById("refreshBtn"),
  runMeta: document.getElementById("runMeta"),
  loadStatus: document.getElementById("loadStatus"),
  metricsHeading: document.getElementById("metricsHeading"),
  metricTotalLabel: document.getElementById("metricTotalLabel"),
  metricTotal: document.getElementById("metricTotal"),
  metricEligibleLabel: document.getElementById("metricEligibleLabel"),
  metricEligible: document.getElementById("metricEligible"),
  metricRateLabel: document.getElementById("metricRateLabel"),
  metricRate: document.getElementById("metricRate"),
  metricScoreLabel: document.getElementById("metricScoreLabel"),
  metricScore: document.getElementById("metricScore"),
  qualityDistHeading: document.getElementById("qualityDistHeading"),
  qualityBars: document.getElementById("qualityBars"),
  topicDistHeading: document.getElementById("topicDistHeading"),
  topicBars: document.getElementById("topicBars"),
  filtersHeading: document.getElementById("filtersHeading"),
  filterVideoLabel: document.getElementById("filterVideoLabel"),
  filterVideo: document.getElementById("filterVideo"),
  filterQualityLabel: document.getElementById("filterQualityLabel"),
  filterQuality: document.getElementById("filterQuality"),
  filterIncludedLabel: document.getElementById("filterIncludedLabel"),
  filterIncluded: document.getElementById("filterIncluded"),
  filterTopicLabel: document.getElementById("filterTopicLabel"),
  filterTopic: document.getElementById("filterTopic"),
  filterSearchLabel: document.getElementById("filterSearchLabel"),
  filterSearch: document.getElementById("filterSearch"),
  exportCsvBtn: document.getElementById("exportCsvBtn"),
  evidenceCountText: document.getElementById("evidenceCountText"),
  evidenceListHeading: document.getElementById("evidenceListHeading"),
  thVideo: document.getElementById("thVideo"),
  thTime: document.getElementById("thTime"),
  thQuality: document.getElementById("thQuality"),
  thScore: document.getElementById("thScore"),
  thTopics: document.getElementById("thTopics"),
  thIncluded: document.getElementById("thIncluded"),
  thReason: document.getElementById("thReason"),
  thText: document.getElementById("thText"),
  thAction: document.getElementById("thAction"),
  evidenceTableBody: document.getElementById("evidenceTableBody"),
  runsListHeading: document.getElementById("runsListHeading"),
  thRunStarted: document.getElementById("thRunStarted"),
  thRunDataset: document.getElementById("thRunDataset"),
  thRunStatus: document.getElementById("thRunStatus"),
  thRunInput: document.getElementById("thRunInput"),
  thRunEligible: document.getElementById("thRunEligible"),
  thRunDuration: document.getElementById("thRunDuration"),
  runsTableBody: document.getElementById("runsTableBody"),
};

const I18N = {
  "en-US": {
    pageTitle: "YouTube Transcript RAG | Evidence",
    localeLabel: "Language",
    eyebrowText: "YouTube Transcript RAG",
    navIngest: "Ingest",
    navStudio: "Studio",
    navReviews: "Reviews",
    navEvidence: "Evidence",
    navEvaluation: "Evaluation",
    navChunking: "Chunking",
    heroTitle: "Evidence Curation",
    heroSubtitle: "Review curated transcript evidence, eligibility decisions, and pipeline run metadata.",
    emptyStateHeading: "No curation artifacts found",
    emptyStateText: "Run the local curation pipeline, then refresh this workspace.",
    runHeading: "Latest Curation Run",
    refreshBtn: "Refresh",
    loading: "Loading evidence curation artifacts...",
    loadError: "Failed to load evidence curation data: {message}",
    availableRows: "{count} evidence record(s) shown",
    noRows: "No curated evidence matches these filters.",
    metricsHeading: "Evidence Quality",
    metricTotalLabel: "Total Records",
    metricEligibleLabel: "Eligible Records",
    metricRateLabel: "Eligibility Rate",
    metricScoreLabel: "Avg Quality Score",
    qualityDistHeading: "Quality Labels",
    topicDistHeading: "Topic Tags",
    filtersHeading: "Filters",
    filterVideoLabel: "Video",
    filterAllVideos: "All videos",
    filterQualityLabel: "Quality",
    filterAllQualities: "All quality labels",
    filterIncludedLabel: "Included",
    filterAllIncluded: "All",
    filterIncludedYes: "Included",
    filterIncludedNo: "Excluded",
    filterTopicLabel: "Topic",
    filterAllTopics: "All topics",
    filterSearchLabel: "Search",
    filterSearchPlaceholder: "Search text, title, or id",
    exportCsv: "Export CSV",
    evidenceListHeading: "Curated Evidence",
    thVideo: "video",
    thTime: "time",
    thQuality: "quality",
    thScore: "score",
    thTopics: "topics",
    thIncluded: "included",
    thReason: "reason",
    thText: "text",
    thAction: "action",
    details: "Details",
    hide: "Hide",
    fullText: "Full text",
    inference: "Heuristic inference",
    inferenceLoading: "Loading inference metadata...",
    noInference: "No inference rows found.",
    runsListHeading: "Recent Pipeline Runs",
    thRunStarted: "started_at",
    thRunDataset: "dataset",
    thRunStatus: "status",
    thRunInput: "input",
    thRunEligible: "eligible",
    thRunDuration: "duration_ms",
    yes: "yes",
    no: "no",
    noData: "No data.",
    generated: "generated",
    runId: "run",
    dataset: "dataset",
    status: "status",
    duration: "duration",
  },
  "ja-JP": {
    pageTitle: "YouTube Transcript RAG | エビデンス",
    localeLabel: "言語",
    eyebrowText: "YouTube Transcript RAG",
    navIngest: "取り込み",
    navStudio: "Studio",
    navReviews: "レビュー",
    navEvidence: "エビデンス",
    navEvaluation: "評価",
    navChunking: "チャンキング",
    heroTitle: "エビデンスキュレーション",
    heroSubtitle: "キュレーション済み transcript エビデンス、採用判定、パイプラインランを確認します。",
    emptyStateHeading: "キュレーション成果物がありません",
    emptyStateText: "ローカルのキュレーションパイプラインを実行してから、このワークスペースを更新してください。",
    runHeading: "最新キュレーションラン",
    refreshBtn: "更新",
    loading: "エビデンスキュレーション成果物を読み込み中...",
    loadError: "エビデンスキュレーションデータの読み込み失敗: {message}",
    availableRows: "{count} 件のエビデンスを表示",
    noRows: "この条件に一致するキュレーション済みエビデンスはありません。",
    metricsHeading: "エビデンス品質",
    metricTotalLabel: "総レコード数",
    metricEligibleLabel: "採用レコード",
    metricRateLabel: "採用率",
    metricScoreLabel: "平均品質スコア",
    qualityDistHeading: "品質ラベル",
    topicDistHeading: "トピックタグ",
    filtersHeading: "フィルター",
    filterVideoLabel: "動画",
    filterAllVideos: "すべての動画",
    filterQualityLabel: "品質",
    filterAllQualities: "すべての品質ラベル",
    filterIncludedLabel: "採用",
    filterAllIncluded: "すべて",
    filterIncludedYes: "採用",
    filterIncludedNo: "除外",
    filterTopicLabel: "トピック",
    filterAllTopics: "すべてのトピック",
    filterSearchLabel: "検索",
    filterSearchPlaceholder: "本文、タイトル、ID を検索",
    exportCsv: "CSV エクスポート",
    evidenceListHeading: "キュレーション済みエビデンス",
    thVideo: "video",
    thTime: "time",
    thQuality: "quality",
    thScore: "score",
    thTopics: "topics",
    thIncluded: "included",
    thReason: "reason",
    thText: "text",
    thAction: "action",
    details: "詳細",
    hide: "閉じる",
    fullText: "全文",
    inference: "ヒューリスティック推論",
    inferenceLoading: "推論メタデータを読み込み中...",
    noInference: "推論行がありません。",
    runsListHeading: "最近のパイプラインラン",
    thRunStarted: "started_at",
    thRunDataset: "dataset",
    thRunStatus: "status",
    thRunInput: "input",
    thRunEligible: "eligible",
    thRunDuration: "duration_ms",
    yes: "yes",
    no: "no",
    noData: "データなし。",
    generated: "generated",
    runId: "run",
    dataset: "dataset",
    status: "status",
    duration: "duration",
  },
};

let currentLocale = localStorage.getItem(LOCALE_STORAGE_KEY) || DEFAULT_LOCALE;
if (!I18N[currentLocale]) {
  currentLocale = DEFAULT_LOCALE;
}

let summaryPayload = null;
let allRows = [];
let runRows = [];
let expandedEvidenceId = "";
let inferenceCache = {};

function t(key, vars = {}) {
  const base = I18N[currentLocale] || I18N[DEFAULT_LOCALE];
  const fallback = I18N[DEFAULT_LOCALE];
  let template = base[key] || fallback[key] || key;
  Object.entries(vars).forEach(([name, value]) => {
    template = template.replaceAll(`{${name}}`, String(value));
  });
  return template;
}

async function apiRequest(path) {
  const response = await fetch(path, { method: "GET", cache: "no-store" });
  const raw = await response.text();
  let json = null;
  try {
    json = raw ? JSON.parse(raw) : {};
  } catch (_) {
    throw new Error(`Non-JSON response (${response.status}): ${raw.slice(0, 160)}`);
  }
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

function formatSeconds(value) {
  const total = Math.max(0, Math.floor(Number(value || 0)));
  const minutes = Math.floor(total / 60);
  const seconds = total % 60;
  return `${minutes}:${String(seconds).padStart(2, "0")}`;
}

function formatTimeRange(row) {
  const start = row?.start_sec ?? 0;
  const end = row?.end_sec ?? start;
  return `${formatSeconds(start)} - ${formatSeconds(end)}`;
}

function formatNumber(value, digits = 0) {
  const number = Number(value);
  if (!Number.isFinite(number)) {
    return "-";
  }
  return number.toLocaleString(undefined, {
    maximumFractionDigits: digits,
    minimumFractionDigits: digits,
  });
}

function formatPercent(value) {
  const number = Number(value);
  if (!Number.isFinite(number)) {
    return "-";
  }
  return `${(number * 100).toFixed(1)}%`;
}

function truncate(value, length = 180) {
  const text = String(value || "");
  if (text.length <= length) {
    return text;
  }
  return `${text.slice(0, length - 1)}...`;
}

function countBy(rows, keyFn) {
  const counts = new Map();
  rows.forEach((row) => {
    const keys = keyFn(row);
    (Array.isArray(keys) ? keys : [keys]).forEach((key) => {
      const scoped = String(key || "").trim();
      if (!scoped) {
        return;
      }
      counts.set(scoped, (counts.get(scoped) || 0) + 1);
    });
  });
  return Array.from(counts.entries()).sort((a, b) => b[1] - a[1] || a[0].localeCompare(b[0]));
}

function uniqueValues(rows, keyFn) {
  return Array.from(new Set(rows.map(keyFn).filter(Boolean))).sort((a, b) => String(a).localeCompare(String(b)));
}

function getFilteredRows() {
  const video = els.filterVideo.value;
  const quality = els.filterQuality.value;
  const included = els.filterIncluded.value;
  const topic = els.filterTopic.value;
  const query = els.filterSearch.value.trim().toLowerCase();
  return allRows
    .filter((row) => !video || row.video_id === video)
    .filter((row) => !quality || row.quality_label === quality)
    .filter((row) => {
      if (!included) {
        return true;
      }
      return Boolean(row.included) === (included === "true");
    })
    .filter((row) => {
      if (!topic) {
        return true;
      }
      return (row.topic_tags || []).includes(topic);
    })
    .filter((row) => {
      if (!query) {
        return true;
      }
      return [
        row.evidence_id,
        row.video_id,
        row.video_title,
        row.text,
        row.quality_label,
        (row.topic_tags || []).join(" "),
      ]
        .join(" ")
        .toLowerCase()
        .includes(query);
    });
}

function renderOptions(select, options, selectedValue) {
  select.innerHTML = options
    .map((option) => `<option value="${escapeHtml(option.value)}">${escapeHtml(option.label)}</option>`)
    .join("");
  if (options.some((option) => option.value === selectedValue)) {
    select.value = selectedValue;
  }
}

function populateFilters() {
  const selected = {
    video: els.filterVideo.value,
    quality: els.filterQuality.value,
    included: els.filterIncluded.value,
    topic: els.filterTopic.value,
  };
  const videoOptions = [
    { value: "", label: t("filterAllVideos") },
    ...uniqueValues(allRows, (row) => row.video_id).map((videoId) => {
      const row = allRows.find((entry) => entry.video_id === videoId) || {};
      const label = row.video_title ? `${row.video_title} (${videoId})` : videoId;
      return { value: videoId, label };
    }),
  ];
  const qualityOptions = [
    { value: "", label: t("filterAllQualities") },
    ...uniqueValues(allRows, (row) => row.quality_label).map((label) => ({
      value: label,
      label,
    })),
  ];
  const topicOptions = [
    { value: "", label: t("filterAllTopics") },
    ...uniqueValues(allRows.flatMap((row) => row.topic_tags || []), (tag) => tag).map((tag) => ({
      value: tag,
      label: tag,
    })),
  ];
  renderOptions(els.filterVideo, videoOptions, selected.video);
  renderOptions(els.filterQuality, qualityOptions, selected.quality);
  renderOptions(
    els.filterIncluded,
    [
      { value: "", label: t("filterAllIncluded") },
      { value: "true", label: t("filterIncludedYes") },
      { value: "false", label: t("filterIncludedNo") },
    ],
    selected.included,
  );
  renderOptions(els.filterTopic, topicOptions, selected.topic);
}

function renderRunMeta() {
  const latest = summaryPayload?.latest_run || null;
  const report = summaryPayload?.report || {};
  if (!latest && !Object.keys(report).length) {
    els.runMeta.innerHTML = `<div class="search-empty">${escapeHtml(t("noData"))}</div>`;
    return;
  }
  const dataset = latest
    ? `${latest.dataset_id || "-"} / ${latest.dataset_version || "-"}`
    : `${report.dataset_id || "-"} / ${report.dataset_version || "-"}`;
  const runId = latest?.pipeline_run_id || report.pipeline_run_id || "-";
  const generated = report.generated_at || latest?.finished_at || "-";
  const status = latest?.status || "-";
  const duration = latest?.duration_ms != null ? `${latest.duration_ms} ms` : "-";
  els.runMeta.innerHTML = `
    <span class="evidence-meta-pill">${escapeHtml(t("dataset"))}: ${escapeHtml(dataset)}</span>
    <span class="evidence-meta-pill">${escapeHtml(t("runId"))}: ${escapeHtml(runId)}</span>
    <span class="evidence-meta-pill">${escapeHtml(t("status"))}: ${escapeHtml(status)}</span>
    <span class="evidence-meta-pill">${escapeHtml(t("generated"))}: ${escapeHtml(generated)}</span>
    <span class="evidence-meta-pill">${escapeHtml(t("duration"))}: ${escapeHtml(duration)}</span>
  `;
}

function renderMetrics(rows) {
  const total = rows.length;
  const eligible = rows.filter((row) => row.included).length;
  const scores = rows
    .map((row) => Number(row.quality_score))
    .filter((score) => Number.isFinite(score));
  const avgScore = scores.length
    ? scores.reduce((sum, score) => sum + score, 0) / scores.length
    : null;

  els.metricTotal.textContent = formatNumber(total);
  els.metricEligible.textContent = formatNumber(eligible);
  els.metricRate.textContent = total ? formatPercent(eligible / total) : "-";
  els.metricScore.textContent = avgScore == null ? "-" : avgScore.toFixed(3);
}

function renderBars(container, rows, countEntries) {
  if (!countEntries.length) {
    container.innerHTML = `<div class="search-empty">${escapeHtml(t("noData"))}</div>`;
    return;
  }
  const maxCount = Math.max(...countEntries.map(([, count]) => count), 1);
  container.innerHTML = countEntries
    .map(([label, count]) => {
      const width = Math.max(4, (count / maxCount) * 100);
      return `
        <div class="bar-row evidence-bar-row">
          <div class="bar-label">${escapeHtml(label)}</div>
          <div class="bar-track"><div class="bar-fill evidence-fill" style="width:${width}%"></div></div>
          <div class="bar-value">${escapeHtml(count)}</div>
        </div>
      `;
    })
    .join("");
}

function renderDistributions(rows) {
  renderBars(
    els.qualityBars,
    rows,
    countBy(rows, (row) => row.quality_label || "unknown"),
  );
  renderBars(
    els.topicBars,
    rows,
    countBy(rows, (row) => row.topic_tags || ["unknown"]).slice(0, 10),
  );
}

function topicChips(row) {
  return (row.topic_tags || [])
    .map((tag) => `<span class="evidence-chip">${escapeHtml(tag)}</span>`)
    .join("");
}

function includedPill(row) {
  const included = Boolean(row.included);
  const text = included ? t("yes") : t("no");
  const tone = included ? "included" : "excluded";
  return `<span class="evidence-pill ${tone}">${escapeHtml(text)}</span>`;
}

function qualityPill(row) {
  return `<span class="evidence-pill quality">${escapeHtml(row.quality_label || "-")}</span>`;
}

function renderInferenceDetail(evidenceId) {
  const rows = inferenceCache[evidenceId];
  if (!rows) {
    return `<div class="search-empty">${escapeHtml(t("inferenceLoading"))}</div>`;
  }
  if (!rows.length) {
    return `<div class="search-empty">${escapeHtml(t("noInference"))}</div>`;
  }
  return rows
    .map((row) => {
      const output = JSON.stringify(row.output_json || {}, null, 2);
      return `
        <div class="evidence-inference-block">
          <p class="search-summary">${escapeHtml(row.model_name || "-")} ${escapeHtml(row.model_version || "")} / ${escapeHtml(row.status || "-")} / ${escapeHtml(row.label || "-")}</p>
          <pre class="output raw-json">${escapeHtml(output)}</pre>
        </div>
      `;
    })
    .join("");
}

function renderTable(rows) {
  if (!rows.length) {
    els.evidenceTableBody.innerHTML = `<tr><td colspan="9">${escapeHtml(t("noRows"))}</td></tr>`;
    return;
  }
  els.evidenceTableBody.innerHTML = rows
    .map((row) => {
      const evidenceId = row.evidence_id || "";
      const expanded = evidenceId && expandedEvidenceId === evidenceId;
      const detail = expanded
        ? `
          <tr class="evidence-detail-row">
            <td colspan="9">
              <div class="evidence-detail-grid">
                <div>
                  <h3 class="chart-heading">${escapeHtml(t("fullText"))}</h3>
                  <p class="evidence-full-text">${escapeHtml(row.text || "")}</p>
                </div>
                <div>
                  <h3 class="chart-heading">${escapeHtml(t("inference"))}</h3>
                  ${renderInferenceDetail(evidenceId)}
                </div>
              </div>
            </td>
          </tr>
        `
        : "";
      return `
        <tr>
          <td>${escapeHtml(row.video_title || row.video_id || "-")}</td>
          <td>${escapeHtml(formatTimeRange(row))}</td>
          <td>${qualityPill(row)}</td>
          <td>${escapeHtml(Number(row.quality_score || 0).toFixed(3))}</td>
          <td class="evidence-topic-cell">${topicChips(row) || "-"}</td>
          <td>${includedPill(row)}</td>
          <td>${escapeHtml(row.exclusion_reason || row.inclusion_reason || "-")}</td>
          <td class="evidence-text-cell">${escapeHtml(truncate(row.text, 220))}</td>
          <td><button class="btn secondary evidence-detail-btn" type="button" data-action="toggle-detail" data-evidence-id="${escapeHtml(evidenceId)}">${escapeHtml(expanded ? t("hide") : t("details"))}</button></td>
        </tr>
        ${detail}
      `;
    })
    .join("");
}

function renderRuns() {
  if (!runRows.length) {
    els.runsTableBody.innerHTML = `<tr><td colspan="6">${escapeHtml(t("noData"))}</td></tr>`;
    return;
  }
  els.runsTableBody.innerHTML = runRows
    .map((row) => {
      const dataset = `${row.dataset_id || "-"} / ${row.dataset_version || "-"}`;
      return `
        <tr>
          <td>${escapeHtml(row.started_at || "-")}</td>
          <td>${escapeHtml(dataset)}</td>
          <td>${escapeHtml(row.status || "-")}</td>
          <td>${escapeHtml(row.input_record_count ?? "-")}</td>
          <td>${escapeHtml(row.eligible_record_count ?? "-")}</td>
          <td>${escapeHtml(row.duration_ms ?? "-")}</td>
        </tr>
      `;
    })
    .join("");
}

function renderAll() {
  const rows = getFilteredRows();
  els.emptyStatePanel.hidden = Boolean(summaryPayload?.available || allRows.length);
  els.evidenceCountText.textContent = t("availableRows", { count: rows.length });
  renderRunMeta();
  renderMetrics(rows);
  renderDistributions(rows);
  renderTable(rows);
  renderRuns();
}

function toCsvValue(value) {
  const text = String(value ?? "");
  if (/[",\n\r]/.test(text)) {
    return `"${text.replaceAll("\"", "\"\"")}"`;
  }
  return text;
}

function exportCsv() {
  const rows = getFilteredRows();
  const headers = [
    "evidence_id",
    "pipeline_run_id",
    "video_id",
    "video_title",
    "chunk_index",
    "start_sec",
    "end_sec",
    "language",
    "quality_score",
    "quality_label",
    "topic_tags",
    "included",
    "inclusion_reason",
    "exclusion_reason",
    "text",
  ];
  const lines = [
    headers.join(","),
    ...rows.map((row) =>
      headers
        .map((header) => {
          const value = header === "topic_tags" ? (row.topic_tags || []).join("|") : row[header];
          return toCsvValue(value);
        })
        .join(","),
    ),
  ];
  const blob = new Blob([lines.join("\n")], { type: "text/csv;charset=utf-8" });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  const stamp = new Date().toISOString().replace(/[:.]/g, "-");
  link.href = url;
  link.download = `curated_evidence_${stamp}.csv`;
  document.body.appendChild(link);
  link.click();
  link.remove();
  URL.revokeObjectURL(url);
}

async function fetchInference(evidenceId) {
  if (!evidenceId || inferenceCache[evidenceId]) {
    return;
  }
  const payload = await apiRequest(`/v1/evidence-curation/inferences?evidence_id=${encodeURIComponent(evidenceId)}&limit=20`);
  inferenceCache = {
    ...inferenceCache,
    [evidenceId]: payload.inferences || [],
  };
}

async function toggleDetail(evidenceId) {
  if (!evidenceId) {
    return;
  }
  expandedEvidenceId = expandedEvidenceId === evidenceId ? "" : evidenceId;
  renderAll();
  if (expandedEvidenceId) {
    try {
      await fetchInference(expandedEvidenceId);
    } catch (err) {
      inferenceCache = { ...inferenceCache, [expandedEvidenceId]: [] };
    }
    renderAll();
  }
}

async function loadData() {
  els.loadStatus.textContent = t("loading");
  els.refreshBtn.disabled = true;
  try {
    const [summary, manifest, runs] = await Promise.all([
      apiRequest("/v1/evidence-curation/summary"),
      apiRequest(`/v1/evidence-curation/manifest?limit=${MANIFEST_LIMIT}`),
      apiRequest("/v1/evidence-curation/runs?limit=25"),
    ]);
    summaryPayload = summary;
    allRows = manifest.rows || [];
    runRows = runs.runs || [];
    expandedEvidenceId = "";
    inferenceCache = {};
    populateFilters();
    renderAll();
    els.loadStatus.textContent = summary.available || allRows.length ? "" : t("noData");
  } catch (err) {
    els.loadStatus.textContent = t("loadError", { message: err.message || err });
  } finally {
    els.refreshBtn.disabled = false;
  }
}

function applyI18n() {
  document.title = t("pageTitle");
  document.documentElement.lang = currentLocale === "ja-JP" ? "ja" : "en";
  els.localeLabel.textContent = t("localeLabel");
  els.eyebrowText.textContent = t("eyebrowText");
  setNavLabel(els.homeNavLink, t("navIngest"));
  setNavLabel(els.studioNavLink, t("navStudio"));
  setNavLabel(els.reviewsNavLink, t("navReviews"));
  setNavLabel(els.evidenceNavLink, t("navEvidence"));
  setNavLabel(els.evaluationNavLink, t("navEvaluation"));
  setNavLabel(els.chunkingNavLink, t("navChunking"));
  els.mobileHomeNavText.textContent = t("navIngest");
  els.mobileStudioNavText.textContent = t("navStudio");
  els.mobileReviewsNavText.textContent = t("navReviews");
  els.mobileEvidenceNavText.textContent = t("navEvidence");
  els.mobileEvaluationNavText.textContent = t("navEvaluation");
  els.mobileChunkingNavText.textContent = t("navChunking");
  els.heroTitle.textContent = t("heroTitle");
  els.heroSubtitle.textContent = t("heroSubtitle");
  els.emptyStateHeading.textContent = t("emptyStateHeading");
  els.emptyStateText.textContent = t("emptyStateText");
  els.emptyStateCommand.textContent = "python pipelines/curate_evidence.py --dataset-id demo_transcript_evidence --dataset-version v1 --limit 50";
  els.runHeading.textContent = t("runHeading");
  els.refreshBtn.textContent = t("refreshBtn");
  els.metricsHeading.textContent = t("metricsHeading");
  els.metricTotalLabel.textContent = t("metricTotalLabel");
  els.metricEligibleLabel.textContent = t("metricEligibleLabel");
  els.metricRateLabel.textContent = t("metricRateLabel");
  els.metricScoreLabel.textContent = t("metricScoreLabel");
  els.qualityDistHeading.textContent = t("qualityDistHeading");
  els.topicDistHeading.textContent = t("topicDistHeading");
  els.filtersHeading.textContent = t("filtersHeading");
  els.filterVideoLabel.textContent = t("filterVideoLabel");
  els.filterQualityLabel.textContent = t("filterQualityLabel");
  els.filterIncludedLabel.textContent = t("filterIncludedLabel");
  els.filterTopicLabel.textContent = t("filterTopicLabel");
  els.filterSearchLabel.textContent = t("filterSearchLabel");
  els.filterSearch.placeholder = t("filterSearchPlaceholder");
  els.exportCsvBtn.textContent = t("exportCsv");
  els.evidenceListHeading.textContent = t("evidenceListHeading");
  els.thVideo.textContent = t("thVideo");
  els.thTime.textContent = t("thTime");
  els.thQuality.textContent = t("thQuality");
  els.thScore.textContent = t("thScore");
  els.thTopics.textContent = t("thTopics");
  els.thIncluded.textContent = t("thIncluded");
  els.thReason.textContent = t("thReason");
  els.thText.textContent = t("thText");
  els.thAction.textContent = t("thAction");
  els.runsListHeading.textContent = t("runsListHeading");
  els.thRunStarted.textContent = t("thRunStarted");
  els.thRunDataset.textContent = t("thRunDataset");
  els.thRunStatus.textContent = t("thRunStatus");
  els.thRunInput.textContent = t("thRunInput");
  els.thRunEligible.textContent = t("thRunEligible");
  els.thRunDuration.textContent = t("thRunDuration");
  populateFilters();
  renderAll();
}

function init() {
  els.localeSelect.value = currentLocale;
  applyI18n();
  els.localeSelect.addEventListener("change", () => {
    currentLocale = els.localeSelect.value;
    if (!I18N[currentLocale]) {
      currentLocale = DEFAULT_LOCALE;
    }
    localStorage.setItem(LOCALE_STORAGE_KEY, currentLocale);
    applyI18n();
  });
  [
    els.filterVideo,
    els.filterQuality,
    els.filterIncluded,
    els.filterTopic,
  ].forEach((element) => element.addEventListener("change", renderAll));
  els.filterSearch.addEventListener("input", renderAll);
  els.refreshBtn.addEventListener("click", loadData);
  els.exportCsvBtn.addEventListener("click", exportCsv);
  els.evidenceTableBody.addEventListener("click", (event) => {
    const button = event.target.closest("[data-action='toggle-detail']");
    if (!button) {
      return;
    }
    toggleDetail(button.getAttribute("data-evidence-id") || "").catch(() => {});
  });
  loadData().catch(() => {});
}

init();
