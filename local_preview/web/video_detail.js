function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}

function posterThumbnailUrl(videoId) {
  const safeVideoId = encodeURIComponent(String(videoId || "").trim());
  return `https://i.ytimg.com/vi/${safeVideoId}/hqdefault.jpg`;
}

async function apiRequest(path) {
  const response = await fetch(path, {
    method: "GET",
    cache: "no-store",
  });
  const raw = await response.text();
  let json = null;
  try {
    json = raw ? JSON.parse(raw) : {};
  } catch (_) {
    throw new Error(`Non-JSON response (${response.status})`);
  }
  if (!response.ok) {
    throw new Error(json?.error?.message || `Request failed (${response.status})`);
  }
  return json;
}

function currentVideoId() {
  const params = new URLSearchParams(window.location.search || "");
  return String(params.get("video_id") || "").trim();
}

function setDetailError(message) {
  const titleEl = document.getElementById("detailTitle");
  const metaEl = document.getElementById("detailMeta");
  titleEl.textContent = "Video not found";
  metaEl.textContent = String(message || "");
}

async function loadDetail() {
  const videoId = currentVideoId();
  const titleEl = document.getElementById("detailTitle");
  const videoIdEl = document.getElementById("detailVideoId");
  const metaEl = document.getElementById("detailMeta");
  const posterEl = document.getElementById("detailPoster");

  if (!videoId) {
    setDetailError("Missing video_id query parameter.");
    return;
  }

  videoIdEl.innerHTML = `video_id: ${escapeHtml(videoId)}`;
  posterEl.src = posterThumbnailUrl(videoId);

  try {
    const data = await apiRequest("/v1/videos");
    const videos = Array.isArray(data?.videos) ? data.videos : [];
    const match = videos.find((row) => String(row?.video_id || "").trim() === videoId);

    if (!match) {
      setDetailError(`No entry found for ${videoId}.`);
      return;
    }

    const title = String(match.title || "Untitled Video");
    titleEl.textContent = title;
    metaEl.textContent = `Language: ${match.language || "-"} | Chunks: ${match.num_chunks ?? "-"}`;
    document.title = `${title} | Video Detail`;
  } catch (err) {
    setDetailError(String(err?.message || err));
  }
}

loadDetail();
