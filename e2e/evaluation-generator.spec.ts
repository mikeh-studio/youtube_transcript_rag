import { expect, test } from "@playwright/test";

test("generates, reviews, and finalizes a Codex retrieval dataset", async ({ page }) => {
  await page.addInitScript(() => {
    localStorage.setItem("yt_rag_ingest_unlocked", "1");
  });

  const caseTypes = [
    "direct_fact_1",
    "direct_fact_2",
    "semantic_paraphrase",
    "cross_lingual",
    "distractor_resistant",
    "multi_evidence",
  ];
  const draft = {
    version: 1,
    draft_id: "draft_browser_test",
    status: "pending_review",
    created_at: "2026-08-06T12:00:00Z",
    updated_at: "2026-08-06T12:00:00Z",
    warnings: [],
    cases: caseTypes.map((caseType, index) => ({
      id: `case_${index + 1}`,
      case_type: caseType,
      query: `Generated retrieval question ${index + 1}?`,
      language: caseType === "cross_lingual" ? "ja" : "en",
      query_type: index < 2 ? "factual" : "thematic",
      difficulty: "medium",
      required_facts: [`Supported fact ${index + 1}`],
      notes: "Codex proposal",
      warnings: [],
      risk_flags: [],
      gold_evidence: [{
        evidence_id: `demo-video:${index}`,
        video_id: "demo-video",
        video_title: "Demo Interview",
        chunk_index: index,
        start: index * 60,
        end: (index + 1) * 60,
        url: `https://www.youtube.com/watch?v=demo-video&t=${index * 60}s`,
        text: `Transcript evidence for supported fact ${index + 1}.`,
      }],
      review: { decision: "pending", reviewed_at: null, final_values: {} },
    })),
  };

  await page.route("**/v1/eval-generator/capabilities", async (route) => {
    await route.fulfill({
      contentType: "application/json",
      body: JSON.stringify({
        capabilities: {
          available: true,
          authenticated: true,
          version: "codex-cli 0.test",
        },
      }),
    });
  });
  await page.route("**/v1/videos", async (route) => {
    await route.fulfill({
      contentType: "application/json",
      body: JSON.stringify({
        videos: [{
          video_id: "demo-video",
          title: "Demo Interview",
          language: "en",
          num_chunks: 12,
        }],
      }),
    });
  });
  await page.route("**/v1/eval-generator/drafts", async (route) => {
    await route.fulfill({
      contentType: "application/json",
      body: JSON.stringify({
        drafts: draft.cases.some((row) => row.review.decision !== "pending")
          ? [{
              draft_id: draft.draft_id,
              status: draft.status,
              created_at: draft.created_at,
              updated_at: draft.updated_at,
              case_count: 6,
              decided_count: draft.cases.filter(
                (row) => row.review.decision !== "pending",
              ).length,
            }]
          : [],
      }),
    });
  });
  await page.route(/\/v1\/eval-generator\/datasets$/, async (route) => {
    const accepted = draft.cases.filter((row) => row.review.decision !== "rejected");
    await route.fulfill({
      contentType: "application/json",
      body: JSON.stringify({
        datasets: draft.status === "finalized" ? [{
          dataset: { dataset_id: "dataset_browser_test", row_count: accepted.length },
          query_set: {
            id: "qs_dataset_browser_test",
            name: "Generated Eval Dataset",
            language: "mixed",
            created_at: "2026-08-06T12:06:00Z",
            queries: accepted.map((row) => ({
              id: `q_${row.id}`,
              text: row.review.final_values.query || row.query,
              type: row.query_type,
              expected_relevant_min: 1,
              notes: row.notes,
            })),
          },
          export_url: "/v1/eval-generator/datasets/dataset_browser_test/export",
        }] : [],
      }),
    });
  });
  await page.route("**/v1/eval-generator/jobs", async (route) => {
    expect(route.request().postDataJSON()).toEqual({ video_ids: ["demo-video"] });
    await route.fulfill({
      contentType: "application/json",
      body: JSON.stringify({ job: { job_id: "job_browser_test", status: "queued" } }),
    });
  });
  await page.route("**/v1/eval-generator/jobs/job_browser_test", async (route) => {
    await route.fulfill({
      contentType: "application/json",
      body: JSON.stringify({
        job: {
          job_id: "job_browser_test",
          status: "completed",
          step: "completed",
          draft_id: draft.draft_id,
        },
      }),
    });
  });
  await page.route("**/v1/eval-generator/drafts/draft_browser_test", async (route) => {
    await route.fulfill({ contentType: "application/json", body: JSON.stringify({ draft }) });
  });
  await page.route("**/v1/eval-generator/drafts/draft_browser_test/review", async (route) => {
    const { decisions } = route.request().postDataJSON();
    for (const decision of decisions) {
      const row = draft.cases.find((item) => item.id === decision.id);
      if (!row) continue;
      row.review = {
        decision: decision.decision,
        reviewed_at: "2026-08-06T12:05:00Z",
        final_values: decision.decision === "edited"
          ? {
              query: decision.query,
              required_facts: decision.required_facts,
              difficulty: decision.difficulty,
              notes: decision.notes,
              gold_evidence: row.gold_evidence.filter((evidence) => (
                decision.kept_evidence_ids.includes(evidence.evidence_id)
              )),
            }
          : {},
      };
    }
    await route.fulfill({ contentType: "application/json", body: JSON.stringify({ draft }) });
  });
  await page.route("**/v1/eval-generator/drafts/draft_browser_test/finalize", async (route) => {
    draft.status = "finalized";
    const accepted = draft.cases.filter((row) => row.review.decision !== "rejected");
    await route.fulfill({
      contentType: "application/json",
      body: JSON.stringify({
        dataset: { dataset_id: "dataset_browser_test", row_count: accepted.length },
        query_set: {
          id: "qs_dataset_browser_test",
          name: "Generated Eval Dataset",
          language: "mixed",
          created_at: "2026-08-06T12:06:00Z",
          queries: accepted.map((row) => ({
            id: `q_${row.id}`,
            text: row.review.final_values.query || row.query,
            type: row.query_type,
            expected_relevant_min: 1,
            notes: row.notes,
          })),
        },
        export_url: "/v1/eval-generator/datasets/dataset_browser_test/export",
      }),
    });
  });

  await page.goto("/evaluation.html");
  await expect(page.getByText("Codex CLI ready")).toBeVisible();
  await page.getByLabel("Demo Interview").check();
  await page.getByRole("button", { name: "Generate 6 Cases" }).click();
  await expect(page.locator(".eval-generator-case")).toHaveCount(6);

  const firstCard = page.locator(".eval-generator-case").nth(0);
  await firstCard.getByRole("button", { name: "Edit" }).click();
  await firstCard.locator("[data-edit-field='query']").fill("Human-edited retrieval question?");
  await firstCard.getByRole("button", { name: "Save & Approve" }).click();
  await expect(page.locator("#generatorReviewSummary")).toContainText("1/6 reviewed");

  await page.locator(".eval-generator-case").nth(1).getByRole("button", { name: "Reject" }).click();
  await expect(page.locator("#generatorReviewSummary")).toContainText("2/6 reviewed");
  for (let index = 2; index < 6; index += 1) {
    await page.locator(".eval-generator-case").nth(index).getByRole("button", { name: "Approve" }).click();
    await expect(page.locator("#generatorReviewSummary")).toContainText(`${index + 1}/6 reviewed`);
  }

  await page.getByRole("button", { name: "Create Query Set" }).click();
  await expect(page.locator("#generatorStatus")).toContainText("Dataset finalized with 5 cases");
  await expect(page.locator("#querySetSelect")).toHaveValue("qs_dataset_browser_test");
  await expect(page.getByRole("link", { name: "Download JSONL" })).toHaveAttribute(
    "href",
    "/v1/eval-generator/datasets/dataset_browser_test/export",
  );

  await page.evaluate(() => localStorage.removeItem("youtube-rag-eval-v1"));
  await page.reload();
  await expect(page.locator("#querySetSelect")).toHaveValue("qs_dataset_browser_test");
  await expect(
    page.locator("#queryRowsBody input[data-field='text']").first(),
  ).toHaveValue("Human-edited retrieval question?");

  const firstQuestion = page.locator("#queryRowsBody input[data-field='text']").first();
  await firstQuestion.fill("Browser-customized generated question?");
  await page.getByRole("button", { name: "Save Set" }).click();
  await page.reload();
  await page.locator("#querySetSelect").selectOption("qs_dataset_browser_test");
  await expect(
    page.locator("#queryRowsBody input[data-field='text']").first(),
  ).toHaveValue("Browser-customized generated question?");
});
