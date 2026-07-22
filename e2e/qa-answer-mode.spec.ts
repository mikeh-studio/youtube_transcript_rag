import { expect, test } from "@playwright/test";

async function prepareQAStudio(page, options = {}) {
  const { locale } = options;
  await page.addInitScript((initLocale) => {
    localStorage.setItem("yt_rag_ingest_unlocked", "1");
    if (initLocale) {
      localStorage.setItem("youtube-rag-ui-locale", initLocale);
    }
  }, locale ?? null);
  await page.goto("/index.html#/qa");
}

async function mockAskResponse(page, payload) {
  await page.route("**/v1/ask", async (route) => {
    await route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify(payload),
    });
  });
}

test("renders citation-backed answer state and evidence toggle in Q&A studio", async ({ page }) => {
  await mockAskResponse(page, {
    ok: true,
    question: "What is the product intent?",
    k: 4,
    status: "answered",
    answer: "It is designed as a local-first, explainable demo [1] [2].",
    confidence: "high",
    citations: [
      {
        citation_id: 1,
        video_id: "vid1",
        video_title: "Demo Video",
        chunk_id: "vid1:2",
        start_seconds: 65,
        end_seconds: 88,
        timestamp_label: "1:05",
        timestamp_range_label: "1:05-1:28",
        snippet: "The speaker says the product runs fully offline and stores runtime data locally.",
        reason: "This excerpt supports the local-first claim.",
        url: "https://www.youtube.com/watch?v=vid1&t=65s",
        language: "en",
        score: 0.91,
      },
      {
        citation_id: 2,
        video_id: "vid2",
        video_title: "Demo Video 2",
        chunk_id: "vid2:0",
        start_seconds: 12,
        end_seconds: 30,
        timestamp_label: "0:12",
        timestamp_range_label: "0:12-0:30",
        snippet: "A later excerpt says the local preview is portfolio-friendly and optimized for explainability.",
        reason: "This excerpt supports the portfolio/demo framing.",
        url: "https://www.youtube.com/watch?v=vid2&t=12s",
        language: "en",
        score: 0.84,
      },
    ],
    retrieved_chunks: [],
    warnings: [],
    provider: "chatgpt",
    model: "gpt-4o-mini",
    sources: [],
    retrieval_mode: "hybrid",
    retrieval_details: { fusion: "rrf" },
    result_count: 2,
  });
  await prepareQAStudio(page);

  await page.getByLabel("Question").fill("What is the product intent?");
  await page.getByRole("button", { name: "Generate Answer" }).click();

  await expect(page.getByTestId("answer-status")).toHaveText("Answered from evidence");
  await expect(page.getByTestId("answer-panel")).toContainText("local-first, explainable demo [1] [2]");
  await expect(
    page.getByText(
      "Answer generated from retrieved evidence. May be incomplete if source coverage is limited.",
    ),
  ).toBeVisible();
  await expect(page.getByTestId("answer-citation-card")).toHaveCount(2);

  await page.getByTestId("answer-evidence-toggle").click();
  await expect(page.getByTestId("answer-citation-card")).toHaveCount(0);
});

test("keeps source links visible when answer mode falls back to insufficient evidence", async ({ page }) => {
  await mockAskResponse(page, {
    ok: true,
    question: "What is the refund policy?",
    k: 4,
    status: "insufficient_evidence",
    answer: "Insufficient transcript evidence to answer confidently from the retrieved excerpts.",
    confidence: "low",
    citations: [],
    retrieved_chunks: [
      {
        video_id: "vid1",
        video_title: "Demo Video",
        video_url: "https://www.youtube.com/watch?v=vid1",
        chunk_index: 9,
        start_seconds: 140,
        end_seconds: 168,
        timestamp_label: "2:20",
        timestamp_range_label: "2:20-2:48",
        url: "https://www.youtube.com/watch?v=vid1&t=140s",
        rank: 1,
        score: 0.58,
        snippet: "This excerpt talks about local setup steps and startup logs.",
        language: "en",
      },
    ],
    warnings: ["The best retrieved chunk is too weak to support a grounded answer."],
    provider: "chatgpt",
    model: "gpt-4o-mini",
    sources: [],
    retrieval_mode: "dense",
    retrieval_details: {},
    result_count: 1,
  });
  await prepareQAStudio(page);

  await page.getByLabel("Question").fill("What is the refund policy?");
  await page.getByRole("button", { name: "Generate Answer" }).click();

  await expect(page.getByTestId("answer-status")).toHaveText("Insufficient evidence");
  await expect(page.getByTestId("answer-citation-card")).toHaveCount(1);
  await expect(page.getByRole("link", { name: "Open source" })).toHaveAttribute(
    "href",
    "https://www.youtube.com/watch?v=vid1&t=140s",
  );
});

test("renders localized answer panel copy in Japanese locale", async ({ page }) => {
  await mockAskResponse(page, {
    ok: true,
    question: "指輪のシーンについて何と言っている？",
    k: 4,
    status: "answered",
    answer: "指輪のシーンは感情の余韻を大事にした演出として語られています [1]。",
    confidence: "medium",
    citations: [
      {
        citation_id: 1,
        video_id: "vid-ja-1",
        video_title: "葬送のフリーレン ラジオ",
        chunk_id: "vid-ja-1:3",
        start_seconds: 182,
        end_seconds: 215,
        timestamp_label: "3:02",
        timestamp_range_label: "3:02-3:35",
        snippet: "指輪のシーンでは余韻が残るように音楽の入り方を調整したと話している。",
        reason: "この引用が指輪のシーンに関する説明を支えている。",
        url: "https://www.youtube.com/watch?v=vid-ja-1&t=182s",
        language: "ja",
        score: 0.88,
      },
    ],
    retrieved_chunks: [],
    warnings: [],
    provider: "chatgpt",
    model: "gpt-4o-mini",
    sources: [],
    retrieval_mode: "hybrid",
    retrieval_details: { fusion: "rrf" },
    result_count: 1,
  });
  await prepareQAStudio(page, { locale: "ja-JP" });

  await page.getByLabel("Question").fill("指輪のシーンについて何と言っている？");
  await page.getByRole("button", { name: "Generate Answer" }).click();

  await expect(page.getByTestId("answer-status")).toHaveText("根拠付きで回答");
  await expect(
    page.getByText("取得した根拠から生成した回答です。ソースの範囲次第で不完全な場合があります。"),
  ).toBeVisible();
  await expect(page.getByRole("button", { name: "根拠を隠す" })).toBeVisible();
  await expect(page.getByRole("button", { name: "この位置から再生" })).toBeVisible();
  await expect(page.getByRole("link", { name: "ソースを開く" })).toHaveAttribute(
    "href",
    "https://www.youtube.com/watch?v=vid-ja-1&t=182s",
  );
});

test("visualizes agentic attempts and reranked results", async ({ page }) => {
  let requestBody = null;
  await page.route("**/v1/ask", async (route) => {
    requestBody = route.request().postDataJSON();
    await route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify({
        ok: true,
        question: "How is the retrieval update different?",
        k: 5,
        status: "answered",
        answer: "The system retries weak retrieval and reranks candidate evidence [1] [2].",
        confidence: "high",
        citations: [
          {
            citation_id: 1,
            video_id: "vid1",
            video_title: "Retrieval Architecture",
            chunk_index: 7,
            start_seconds: 65,
            end_seconds: 88,
            timestamp_range_label: "1:05-1:28",
            snippet: "Weak evidence triggers a rewritten query before answer generation.",
            reason: "Supports the retry loop.",
            url: "https://www.youtube.com/watch?v=vid1&t=65s",
            language: "en",
            rank: 1,
            pre_rerank_rank: 3,
            pre_rerank_score: 0.42,
            rerank_score: 0.94,
            score: 0.94,
          },
          {
            citation_id: 2,
            video_id: "vid2",
            video_title: "Ranking Pipeline",
            chunk_index: 2,
            start_seconds: 12,
            end_seconds: 30,
            timestamp_range_label: "0:12-0:30",
            snippet: "A cross-encoder rescored the fused candidate pool.",
            reason: "Supports the reranking stage.",
            url: "https://www.youtube.com/watch?v=vid2&t=12s",
            language: "en",
            rank: 2,
            pre_rerank_rank: 1,
            pre_rerank_score: 0.81,
            rerank_score: 0.78,
            score: 0.78,
          },
        ],
        retrieved_chunks: [],
        warnings: [],
        provider: "chatgpt",
        model: "gpt-5.4-mini",
        retrieval_mode: "hybrid",
        retrieval_details: {
          fusion: "rrf",
          dense_candidates: 30,
          lexical_candidates: 30,
          pre_rerank_candidate_count: 42,
          post_feedback_candidate_count: 42,
          feedback_tuning: { enabled: true, adjusted_results: 2 },
          reranker: {
            requested: "cross_encoder",
            applied: true,
            model: "cross-encoder/test-model",
            scored_count: 42,
            error: null,
          },
          agentic_retrieval: {
            enabled: true,
            applied: true,
            sufficient: true,
            stopped_reason: "sufficient_evidence",
            final_query: "retrieval retry reranking",
            final_retrieval_mode: "hybrid",
            final_k: 5,
            attempts: [
              {
                attempt: 1,
                strategy: "initial",
                query: "How is the retrieval update different?",
                retrieval_mode: "hybrid",
                k: 5,
                result_count: 1,
                sufficient: false,
                reason_code: "single_weak_chunk",
                confidence_cap: "low",
              },
              {
                attempt: 2,
                strategy: "rewrite_query",
                query: "retrieval retry reranking",
                retrieval_mode: "hybrid",
                k: 5,
                result_count: 5,
                sufficient: true,
                reason_code: "multi_chunk_support",
                confidence_cap: "high",
              },
            ],
          },
        },
        result_count: 2,
      }),
    });
  });

  await prepareQAStudio(page);
  await page.getByLabel("Question").fill("How is the retrieval update different?");
  await page.getByLabel("Agentic retry").check();
  await page.getByLabel("Reranker").selectOption("cross_encoder");
  await page.getByRole("button", { name: "Generate Answer" }).click();

  expect(requestBody).toMatchObject({
    agentic: true,
    reranker: "cross_encoder",
    retrieval_mode: "hybrid",
  });
  expect(requestBody).not.toHaveProperty("source_mode");
  await expect(page.locator(".qa-ask-form").getByLabel("Evidence", { exact: true })).toHaveCount(0);
  await expect(page.getByTestId("agentic-attempt-timeline")).toBeVisible();
  await expect(page.locator(".agentic-attempt")).toHaveCount(2);
  await expect(page.getByTestId("agentic-attempt-timeline").getByText("Rewritten query")).toBeVisible();
  await expect(page.getByTestId("retrieval-stage-funnel")).toHaveCount(0);
  await expect(page.getByTestId("result-ranking-profile")).toContainText("↑2");
  await expect(page.getByTestId("result-ranking-profile")).toContainText("↓1");
  await page.setViewportSize({ width: 390, height: 844 });
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth)).toBe(true);
});

test("keeps OCR ingestion and multimodal Ask in the Local Video tool", async ({ page }) => {
  let requestBody = null;
  await page.route("**/v1/local-video-ocr/jobs", async (route) => {
    await route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify({
        ok: true,
        jobs: [
          {
            job_id: "ocr-demo-1",
            video_id: "demo_001",
            video_path: "data/raw/demo_001.mp4",
            status: "completed",
            step: "indexed",
            frame_count: 12,
            ocr_count: 10,
            vector_count: 10,
            interval_sec: 10,
          },
        ],
      }),
    });
  });
  await page.route("**/v1/ask-multimodal", async (route) => {
    requestBody = route.request().postDataJSON();
    await route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify({
        ok: true,
        question: "What score is visible?",
        video_id: "demo_001",
        source_mode: "ocr",
        status: "answered",
        confidence: "high",
        answer: "The scoreboard shows 98-96 in the fourth quarter [1].",
        citations: [
          {
            citation_id: 1,
            source_type: "ocr",
            video_id: "demo_001",
            video_title: "demo_001",
            frame_id: "frame_000120",
            frame_path: "data/frames/demo_001/frame_000120.jpg",
            timestamp_label: "2:00",
            timestamp_range_label: "2:00",
            snippet: "Q4 98 96",
          },
        ],
        retrieved_chunks: [],
        warnings: [],
        provider: "chatgpt",
        model: "gpt-5.4-mini",
      }),
    });
  });

  await page.goto("/index.html#/local-video");
  await expect(page.getByRole("heading", { name: "Local Video Analysis" })).toBeVisible();

  const askPanel = page.locator(".local-video-ask-panel");
  await expect(askPanel.getByLabel("Video ID")).toHaveValue("demo_001");
  await askPanel.getByLabel("Question").fill("What score is visible?");
  await askPanel.getByRole("button", { name: "Ask Local Video" }).click();

  expect(requestBody).toMatchObject({
    question: "What score is visible?",
    video_id: "demo_001",
    source_mode: "ocr",
    retrieval_mode: "hybrid",
  });
  await expect(page.getByTestId("local-video-answer-panel")).toContainText("98-96");
  await expect(page.getByTestId("local-video-evidence-card")).toHaveCount(1);
  await expect(page.getByTestId("local-video-evidence-card")).toContainText("Q4 98 96");
});
