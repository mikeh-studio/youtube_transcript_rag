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
  await expect(page.getByText("Answer generated from retrieved transcript evidence.")).toBeVisible();
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

  await page.locator('input[placeholder="Ask with citations"]').fill("指輪のシーンについて何と言っている？");
  await page.getByRole("button", { name: "Generate Answer" }).click();

  await expect(page.getByTestId("answer-status")).toHaveText("根拠付きで回答");
  await expect(page.getByText("取得した書き起こしの根拠から生成した回答です。")).toBeVisible();
  await expect(page.getByRole("button", { name: "根拠を隠す" })).toBeVisible();
  await expect(page.getByRole("button", { name: "この位置から再生" })).toBeVisible();
  await expect(page.getByRole("link", { name: "ソースを開く" })).toHaveAttribute(
    "href",
    "https://www.youtube.com/watch?v=vid-ja-1&t=182s",
  );
});
