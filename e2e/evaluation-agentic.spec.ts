import { expect, test } from "@playwright/test";


test("runs an Evaluation query set with the agentic tool strategy", async ({ page }) => {
  const requestBodies = [];
  await page.addInitScript(() => {
    localStorage.setItem("yt_rag_ingest_unlocked", "1");
    localStorage.setItem("youtube-rag-eval-v1", JSON.stringify({
      version: 1,
      query_sets: [{
        id: "qs_agentic_ui",
        name: "Agentic UI Check",
        language: "ja",
        created_at: "2026-08-11T00:00:00Z",
        queries: [{
          id: "q_agentic_ui",
          text: "エージェント検索",
          type: "factual",
          expected_relevant_min: 1,
          notes: "",
        }],
      }],
      runs: [],
    }));
  });

  await page.route("**/v1/eval-generator/capabilities", async (route) => {
    await route.fulfill({
      contentType: "application/json",
      body: JSON.stringify({ capabilities: { available: false, authenticated: false } }),
    });
  });
  await page.route("**/v1/eval-generator/drafts", async (route) => {
    await route.fulfill({ contentType: "application/json", body: JSON.stringify({ drafts: [] }) });
  });
  await page.route(/\/v1\/eval-generator\/datasets$/, async (route) => {
    await route.fulfill({ contentType: "application/json", body: JSON.stringify({ datasets: [] }) });
  });
  await page.route("**/v1/videos", async (route) => {
    await route.fulfill({ contentType: "application/json", body: JSON.stringify({ videos: [] }) });
  });
  await page.route("**/v1/search", async (route) => {
    requestBodies.push(route.request().postDataJSON());
    await route.fulfill({
      contentType: "application/json",
      body: JSON.stringify({
        ok: true,
        query: "エージェント検索",
        k: 5,
        agentic: true,
        retrieval_mode: "lexical",
        result_count: 1,
        results: [{
          rank: 1,
          video_id: "vid1",
          video_title: "Demo Video",
          chunk_index: 1,
          start: 30,
          end: 60,
          score: 0.9,
          text: "生の文字起こしコンテキスト",
          language: "ja",
          url: "https://www.youtube.com/watch?v=vid1&t=30s",
        }],
        retrieval_details: {
          agentic_retrieval: {
            final_tool: "read_context",
            attempts: [
              {
                attempt: 1,
                strategy: "initial",
                tool: "keyword_search",
                retrieval_mode: "lexical",
              },
              {
                attempt: 2,
                strategy: "read_context",
                tool: "read_context",
                retrieval_mode: "lexical",
              },
            ],
          },
        },
      }),
    });
  });

  await page.goto("/evaluation.html");
  await page.getByLabel("Search Strategy").selectOption("agentic");
  await expect(page.getByLabel("Direct Retrieval Mode")).toBeDisabled();
  await page.getByRole("button", { name: "Run Query Set" }).click();

  await expect(page.locator("#runStatus")).toHaveText("Run complete. 1 queries executed.");
  expect(requestBodies).toEqual([expect.objectContaining({
    query: "エージェント検索",
    agentic: true,
    retrieval_mode: "hybrid",
  })]);
  await expect(page.locator("#runSelect option").first()).toContainText("agentic · hybrid");
  await expect(page.locator(".eval-agentic-trace")).toBeVisible();
  await expect(page.locator(".eval-agentic-tool")).toHaveText([
    "Keyword · Japanese BM25",
    "Raw transcript context",
  ]);
});
