const INGEST_UNLOCK_KEY = "yt_rag_ingest_unlocked";
if (localStorage.getItem(INGEST_UNLOCK_KEY) !== "1") {
  window.location.replace("./index.html#/ingest");
}

const LOCALE_STORAGE_KEY = "youtube-rag-ui-locale";
const FEEDBACK_REVISION_STORAGE_KEY = "youtube-rag-feedback-revision";
const DEFAULT_LOCALE = "en-US";

const els = {
  localeSelect: document.getElementById("localeSelect"),
  localeLabel: document.getElementById("localeLabel"),
  homeBrandLink: document.getElementById("homeBrandLink"),
  eyebrowText: document.getElementById("eyebrowText"),
  homeNavLink: document.getElementById("homeNavLink"),
  tldrNavLink: document.getElementById("tldrNavLink"),
  qaNavLink: document.getElementById("qaNavLink"),
  reviewsNavLink: document.getElementById("reviewsNavLink"),
  evidenceNavLink: document.getElementById("evidenceNavLink"),
  evaluationNavLink: document.getElementById("evaluationNavLink"),
  chunkingNavLink: document.getElementById("chunkingNavLink"),
  mobileHomeNavText: document.getElementById("mobileHomeNavText"),
  mobileTldrNavText: document.getElementById("mobileTldrNavText"),
  mobileQaNavText: document.getElementById("mobileQaNavText"),
  mobileReviewsNavText: document.getElementById("mobileReviewsNavText"),
  mobileEvidenceNavText: document.getElementById("mobileEvidenceNavText"),
  mobileEvaluationNavText: document.getElementById("mobileEvaluationNavText"),
  mobileChunkingNavText: document.getElementById("mobileChunkingNavText"),
  heroTitle: document.getElementById("heroTitle"),
  heroSubtitle: document.getElementById("heroSubtitle"),
  filtersHeading: document.getElementById("filtersHeading"),
  filterVideoLabel: document.getElementById("filterVideoLabel"),
  filterLabelLabel: document.getElementById("filterLabelLabel"),
  filterVideo: document.getElementById("filterVideo"),
  filterLabel: document.getElementById("filterLabel"),
  refreshReviewsBtn: document.getElementById("refreshReviewsBtn"),
  exportCsvBtn: document.getElementById("exportCsvBtn"),
  reviewCountText: document.getElementById("reviewCountText"),
  tpHeading: document.getElementById("tpHeading"),
  tpValue: document.getElementById("tpValue"),
  tpFill: document.getElementById("tpFill"),
  relevantBarHeading: document.getElementById("relevantBarHeading"),
  barRelevantLabel: document.getElementById("barRelevantLabel"),
  barNotRelevantLabel: document.getElementById("barNotRelevantLabel"),
  barRelevantFill: document.getElementById("barRelevantFill"),
  barNotRelevantFill: document.getElementById("barNotRelevantFill"),
  barRelevantValue: document.getElementById("barRelevantValue"),
  barNotRelevantValue: document.getElementById("barNotRelevantValue"),
  reviewListHeading: document.getElementById("reviewListHeading"),
  thUpdated: document.getElementById("thUpdated"),
  thVideo: document.getElementById("thVideo"),
  thTime: document.getElementById("thTime"),
  thLabel: document.getElementById("thLabel"),
  thMode: document.getElementById("thMode"),
  thQuery: document.getElementById("thQuery"),
  thScore: document.getElementById("thScore"),
  thRank: document.getElementById("thRank"),
  reviewsTableBody: document.getElementById("reviewsTableBody"),
  clearReviewsBtn: document.getElementById("clearReviewsBtn"),
};

const I18N = {
  "en-US": {
    pageTitle: "YouTube Transcript RAG | Reviews",
    localeLabel: "Language",
    navIngest: "Ingest",
    navTLDR: "TLDR Studio",
    navQA: "Q&A Studio",
    navReviews: "Reviews",
    navEvidence: "Evidence",
    eyebrowText: "YouTube Transcript RAG",
    navEvaluation: "Evaluation",
    navChunking: "Chunking",
    heroTitle: "Review Analytics",
    heroSubtitle: "Inspect reviewed chunks, filter by video, and export data.",
    filtersHeading: "Filters",
    filterVideoLabel: "Video",
    filterLabelLabel: "Review Label",
    filterAllVideos: "All videos",
    filterAllLabels: "All",
    labelRelevant: "Relevant",
    labelNotRelevant: "Not relevant",
    refreshReviewsBtn: "Refresh Reviews",
    exportCsv: "Export CSV",
    clearReviewsBtn: "Clear Reviewed Chunks",
    clearNeedFilter: "Select a video or label filter before clearing.",
    clearConfirm: "Clear reviewed chunks for the current filters?",
    clearSuccess: "Cleared {count} reviewed chunk(s).",
    loadingReviews: "Loading reviews...",
    reviewCountText: "{count} review(s) shown",
    noReviews: "No reviews found for this filter.",
    tpHeading: "True Positive (Precision Proxy)",
    tpNone: "No reviewed items",
    relevantBarHeading: "Relevant Distribution",
    reviewListHeading: "Reviewed Chunks",
    thUpdated: "update_date",
    thVideo: "video_id",
    thTime: "chunk time",
    thLabel: "label",
    thMode: "model",
    thQuery: "query",
    thScore: "score",
    thRank: "rank",
    loadError: "Failed to load reviews: {message}",
    partialVideoLoadError: "Video summary load failed. Showing reviews without video filter stats.",
  },
  "ja-JP": {
    pageTitle: "YouTube Transcript RAG | レビュー",
    localeLabel: "言語",
    navIngest: "取り込み",
    navTLDR: "TLDR Studio",
    navQA: "Q&A Studio",
    navReviews: "レビュー",
    navEvidence: "エビデンス",
    eyebrowText: "YouTube Transcript RAG",
    navEvaluation: "評価",
    navChunking: "チャンキング",
    heroTitle: "レビュー分析",
    heroSubtitle: "レビュー済みチャンクを確認し、動画で絞り込み、CSVをエクスポートします。",
    filtersHeading: "フィルター",
    filterVideoLabel: "動画",
    filterLabelLabel: "レビューラベル",
    filterAllVideos: "すべての動画",
    filterAllLabels: "すべて",
    labelRelevant: "関連あり",
    labelNotRelevant: "関連なし",
    refreshReviewsBtn: "レビュー更新",
    exportCsv: "CSV エクスポート",
    clearReviewsBtn: "レビュー済みチャンクをクリア",
    clearNeedFilter: "クリアする前に動画またはラベルを選択してください。",
    clearConfirm: "現在のフィルター対象のレビュー済みチャンクを削除しますか？",
    clearSuccess: "{count} 件のレビュー済みチャンクを削除しました。",
    loadingReviews: "レビュー読み込み中...",
    reviewCountText: "{count} 件のレビューを表示",
    noReviews: "この条件に一致するレビューはありません。",
    tpHeading: "True Positive（精度プロキシ）",
    tpNone: "レビュー項目がありません",
    relevantBarHeading: "関連ラベル分布",
    reviewListHeading: "レビュー済みチャンク",
    thUpdated: "update_date",
    thVideo: "video_id",
    thTime: "chunk time",
    thLabel: "label",
    thMode: "model",
    thQuery: "query",
    thScore: "score",
    thRank: "rank",
    loadError: "レビュー読み込み失敗: {message}",
    partialVideoLoadError: "動画サマリーの読み込みに失敗しました。動画フィルター統計なしで表示します。",
  },
};

let currentLocale = localStorage.getItem(LOCALE_STORAGE_KEY) || DEFAULT_LOCALE;
if (!I18N[currentLocale]) {
  currentLocale = DEFAULT_LOCALE;
}
let initialVideoFilter = String(
  new URLSearchParams(window.location.search || "").get("video_id") || "",
).trim();

let allReviews = [];
let reviewedVideos = [];
let reloadPromise = null;

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
    headers: body ? { "content-type": "application/json" } : undefined,
    body: body ? JSON.stringify(body) : undefined,
  });
  const raw = await response.text();
  let json = null;
  try {
    json = raw ? JSON.parse(raw) : {};
  } catch (_) {
    const snippet = raw ? raw.slice(0, 160) : "";
    throw new Error(`Non-JSON response (${response.status}): ${snippet}`);
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

function formatSeconds(sec) {
  const total = Math.max(0, Math.floor(Number(sec || 0)));
  const mm = Math.floor(total / 60);
  const ss = total % 60;
  return `${mm}:${String(ss).padStart(2, "0")}`;
}

function labelText(value) {
  if (value === "relevant") {
    return t("labelRelevant");
  }
  if (value === "not_relevant") {
    return t("labelNotRelevant");
  }
  return String(value || "-");
}

function populateFilters() {
  const selectedVideo = els.filterVideo.value;

  const options = [
    `<option value="">${escapeHtml(t("filterAllVideos"))}</option>`,
    ...reviewedVideos.map((row) => {
      const title = row.video_title || row.title || row.video_id;
      const text = `${title} (${row.review_count})`;
      return `<option value="${escapeHtml(row.video_id)}">${escapeHtml(text)}</option>`;
    }),
  ];

  els.filterVideo.innerHTML = options.join("");
  if (["", ...reviewedVideos.map((v) => v.video_id)].includes(selectedVideo)) {
    els.filterVideo.value = selectedVideo;
  }

  const selectedLabel = els.filterLabel.value;
  els.filterLabel.innerHTML = `
    <option value="">${escapeHtml(t("filterAllLabels"))}</option>
    <option value="relevant">${escapeHtml(t("labelRelevant"))}</option>
    <option value="not_relevant">${escapeHtml(t("labelNotRelevant"))}</option>
  `;
  if (["", "relevant", "not_relevant"].includes(selectedLabel)) {
    els.filterLabel.value = selectedLabel;
  }
}

function filteredReviews() {
  const videoId = els.filterVideo.value;
  const label = els.filterLabel.value;

  return allReviews
    .filter((row) => (!videoId || row.video_id === videoId))
    .filter((row) => (!label || row.label === label));
}

function hasActiveClearFilter() {
  return Boolean(els.filterVideo.value || els.filterLabel.value);
}

function updateClearButtonState() {
  els.clearReviewsBtn.disabled = !hasActiveClearFilter();
}

function renderCharts(rows) {
  const relevant = rows.filter((row) => row.label === "relevant").length;
  const notRelevant = rows.filter((row) => row.label === "not_relevant").length;
  const total = relevant + notRelevant;

  if (!total) {
    els.tpValue.textContent = t("tpNone");
    els.tpFill.style.width = "0%";
  } else {
    const precision = (relevant / total) * 100;
    els.tpValue.textContent = `${precision.toFixed(1)}%`;
    els.tpFill.style.width = `${precision.toFixed(1)}%`;
  }

  const maxCount = Math.max(relevant, notRelevant, 1);
  els.barRelevantValue.textContent = String(relevant);
  els.barNotRelevantValue.textContent = String(notRelevant);
  els.barRelevantFill.style.width = `${(relevant / maxCount) * 100}%`;
  els.barNotRelevantFill.style.width = `${(notRelevant / maxCount) * 100}%`;
}

function renderTable(rows) {
  if (!rows.length) {
    els.reviewsTableBody.innerHTML = `<tr><td colspan="8">${escapeHtml(t("noReviews"))}</td></tr>`;
    return;
  }

  const body = rows
    .map((row) => {
      const score = row.score != null ? Number(row.score).toFixed(4) : "-";
      const rank = row.rank != null ? String(row.rank) : "-";
      const model = row.model || row.retrieval_mode || "-";
      return `
        <tr>
          <td>${escapeHtml(row.updated_at || "-")}</td>
          <td>${escapeHtml(row.video_id || "-")}</td>
          <td>${escapeHtml(`${formatSeconds(row.start)} - ${formatSeconds(row.end)}`)}</td>
          <td>${escapeHtml(labelText(row.label))}</td>
          <td>${escapeHtml(model)}</td>
          <td>${escapeHtml(row.query || "-")}</td>
          <td>${escapeHtml(score)}</td>
          <td>${escapeHtml(rank)}</td>
        </tr>
      `;
    })
    .join("");

  els.reviewsTableBody.innerHTML = body;
}

function render() {
  const rows = filteredReviews();
  els.reviewCountText.textContent = t("reviewCountText", { count: rows.length });
  renderCharts(rows);
  renderTable(rows);
  updateClearButtonState();
}

function toCsv(rows) {
  const headers = [
    "update_date",
    "video_id",
    "chunk_time",
    "label",
    "model",
    "query",
    "retrieval_mode",
    "score",
    "rank",
  ];

  const escapeCsv = (value) => {
    const raw = String(value ?? "");
    if (raw.includes(",") || raw.includes("\n") || raw.includes("\"")) {
      return `"${raw.replaceAll("\"", "\"\"")}"`;
    }
    return raw;
  };

  const lines = [headers.join(",")];
  rows.forEach((row) => {
    const values = [
      row.updated_at || "",
      row.video_id || "",
      `${formatSeconds(row.start)} - ${formatSeconds(row.end)}`,
      row.label || "",
      row.model || row.retrieval_mode || "",
      row.query || "",
      row.retrieval_mode || "",
      row.score != null ? Number(row.score).toFixed(4) : "",
      row.rank != null ? String(row.rank) : "",
    ].map((value) => escapeCsv(value));
    lines.push(values.join(","));
  });

  return `${lines.join("\n")}\n`;
}

function exportCsv() {
  const rows = filteredReviews();
  const csv = toCsv(rows);
  const blob = new Blob([csv], { type: "text/csv;charset=utf-8" });
  const url = URL.createObjectURL(blob);
  const stamp = new Date().toISOString().slice(0, 19).replaceAll(":", "-");

  const link = document.createElement("a");
  link.href = url;
  link.download = `reviewed_chunks_${stamp}.csv`;
  document.body.appendChild(link);
  link.click();
  link.remove();
  URL.revokeObjectURL(url);
}

async function clearReviews() {
  if (!hasActiveClearFilter()) {
    els.reviewCountText.textContent = t("clearNeedFilter");
    return;
  }

  const confirmed = window.confirm(t("clearConfirm"));
  if (!confirmed) {
    return;
  }

  const params = new URLSearchParams();
  if (els.filterVideo.value) {
    params.set("video_id", els.filterVideo.value);
  }
  if (els.filterLabel.value) {
    params.set("label", els.filterLabel.value);
  }

  els.clearReviewsBtn.disabled = true;
  try {
    const response = await apiRequest(`/v1/feedback/search-review?${params.toString()}`, {
      method: "DELETE",
    });
    await reloadData({ showLoading: true });
    els.reviewCountText.textContent = t("clearSuccess", {
      count: Number(response?.deleted_count || 0),
    });
  } catch (err) {
    els.reviewCountText.textContent = t("loadError", {
      message: String(err?.message || err),
    });
  } finally {
    updateClearButtonState();
  }
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
  setNavLabel(els.tldrNavLink, t("navTLDR"));
  setNavLabel(els.qaNavLink, t("navQA"));
  setNavLabel(els.reviewsNavLink, t("navReviews"));
  setNavLabel(els.evidenceNavLink, t("navEvidence"));
  setNavLabel(els.evaluationNavLink, t("navEvaluation"));
  setNavLabel(els.chunkingNavLink, t("navChunking"));
  if (els.mobileHomeNavText) {
    els.mobileHomeNavText.textContent = t("navIngest");
  }
  if (els.mobileTldrNavText) {
    els.mobileTldrNavText.textContent = t("navTLDR");
  }
  if (els.mobileQaNavText) {
    els.mobileQaNavText.textContent = t("navQA");
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
  els.filtersHeading.querySelector("span").textContent = t("filtersHeading");
  els.filterVideoLabel.textContent = t("filterVideoLabel");
  els.filterLabelLabel.textContent = t("filterLabelLabel");
  els.refreshReviewsBtn.textContent = t("refreshReviewsBtn");
  els.exportCsvBtn.textContent = t("exportCsv");
  els.clearReviewsBtn.textContent = t("clearReviewsBtn");

  els.tpHeading.textContent = t("tpHeading");
  els.relevantBarHeading.textContent = t("relevantBarHeading");
  els.barRelevantLabel.textContent = t("labelRelevant");
  els.barNotRelevantLabel.textContent = t("labelNotRelevant");

  els.reviewListHeading.querySelector("span").textContent = t("reviewListHeading");
  els.thUpdated.textContent = t("thUpdated");
  els.thVideo.textContent = t("thVideo");
  els.thTime.textContent = t("thTime");
  els.thLabel.textContent = t("thLabel");
  els.thMode.textContent = t("thMode");
  els.thQuery.textContent = t("thQuery");
  els.thScore.textContent = t("thScore");
  els.thRank.textContent = t("thRank");

  populateFilters();
  render();
}

async function loadData() {
  const selectedVideo = els.filterVideo.value;
  const selectedLabel = els.filterLabel.value;

  const [videosResult, reviewsResult] = await Promise.allSettled([
    apiRequest("/v1/feedback/videos"),
    apiRequest("/v1/feedback/search-review?limit=5000"),
  ]);

  if (reviewsResult.status !== "fulfilled") {
    throw reviewsResult.reason;
  }

  const reviewsResponse = reviewsResult.value;
  const videosResponse = videosResult.status === "fulfilled" ? videosResult.value : null;

  reviewedVideos = videosResponse?.videos || [];
  allReviews = (reviewsResponse.reviews || []).slice().sort((a, b) => {
    const aTime = String(a.updated_at || "");
    const bTime = String(b.updated_at || "");
    return bTime.localeCompare(aTime);
  });

  // Fallback: build video list from reviews when video summary endpoint fails.
  if (!reviewedVideos.length && allReviews.length) {
    const byVideo = new Map();
    allReviews.forEach((row) => {
      const videoId = String(row.video_id || "").trim();
      if (!videoId) {
        return;
      }
      if (!byVideo.has(videoId)) {
        byVideo.set(videoId, {
          video_id: videoId,
          video_title: row.video_title || videoId,
          review_count: 0,
        });
      }
      byVideo.get(videoId).review_count += 1;
    });
    reviewedVideos = Array.from(byVideo.values()).sort((a, b) => {
      return Number(b.review_count || 0) - Number(a.review_count || 0);
    });
  }

  populateFilters();
  const preferredVideo = selectedVideo || initialVideoFilter;
  if (preferredVideo && [...els.filterVideo.options].some((opt) => opt.value === preferredVideo)) {
    els.filterVideo.value = preferredVideo;
    if (preferredVideo === initialVideoFilter) {
      initialVideoFilter = "";
    }
  }
  if ([...els.filterLabel.options].some((opt) => opt.value === selectedLabel)) {
    els.filterLabel.value = selectedLabel;
  }
  render();

  if (videosResult.status !== "fulfilled") {
    els.reviewCountText.textContent = `${t("reviewCountText", { count: filteredReviews().length })} ${t("partialVideoLoadError")}`;
  }
}

function reloadData({ showLoading = false } = {}) {
  if (reloadPromise) {
    return reloadPromise;
  }

  if (showLoading) {
    els.reviewCountText.textContent = t("loadingReviews");
  }

  reloadPromise = loadData().finally(() => {
    reloadPromise = null;
  });
  return reloadPromise;
}

function wireEvents() {
  els.localeSelect.addEventListener("change", () => {
    applyLocale(els.localeSelect.value);
  });
  els.filterVideo.addEventListener("change", () => {
    render();
    updateClearButtonState();
  });
  els.filterLabel.addEventListener("change", () => {
    render();
    updateClearButtonState();
  });
  els.refreshReviewsBtn.addEventListener("click", async () => {
    els.refreshReviewsBtn.disabled = true;
    try {
      await reloadData({ showLoading: true });
    } catch (err) {
      els.reviewCountText.textContent = t("loadError", {
        message: String(err?.message || err),
      });
    } finally {
      els.refreshReviewsBtn.disabled = false;
    }
  });
  els.exportCsvBtn.addEventListener("click", exportCsv);
  els.clearReviewsBtn.addEventListener("click", () => {
    clearReviews().catch((err) => {
      els.reviewCountText.textContent = t("loadError", {
        message: String(err?.message || err),
      });
      updateClearButtonState();
    });
  });
  window.addEventListener("focus", () => {
    reloadData().catch(() => {});
  });
  document.addEventListener("visibilitychange", () => {
    if (document.visibilityState === "visible") {
      reloadData().catch(() => {});
    }
  });
  window.addEventListener("storage", (event) => {
    if (event.key === FEEDBACK_REVISION_STORAGE_KEY) {
      reloadData().catch(() => {});
    }
  });
}

async function boot() {
  wireEvents();

  try {
    await reloadData();
    applyLocale(currentLocale);
  } catch (err) {
    applyLocale(currentLocale);
    els.reviewCountText.textContent = t("loadError", {
      message: String(err?.message || err),
    });
  }
}

boot();
