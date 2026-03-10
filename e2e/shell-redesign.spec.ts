import { expect, test } from "@playwright/test";

test("plays cinematic intro only once per browser session", async ({ page }) => {
  await page.goto("/index.html#/ingest");

  const introOverlay = page.locator("[data-testid='intro-overlay']");
  await expect(introOverlay).toHaveCount(1);
  await introOverlay.waitFor({ state: "detached", timeout: 5000 });

  await page.reload();
  await expect(page.locator("[data-testid='intro-overlay']")).toHaveCount(0);
});

test.describe("mobile bottom tabs", () => {
  test.use({ viewport: { width: 390, height: 844 } });

  test("shows tab bar and supports route tab switching when unlocked", async ({ page }) => {
    await page.addInitScript(() => {
      localStorage.setItem("yt_rag_ingest_unlocked", "1");
    });

    await page.goto("/index.html#/ingest");

    const tabBar = page.locator(".bottom-tabbar");
    await expect(tabBar).toBeVisible();
    await expect(tabBar.locator(".bottom-tab-link")).toHaveCount(5);

    await tabBar.getByRole("button", { name: "TLDR Studio" }).click();
    await expect(page).toHaveURL(/#\/tldr$/);
    await expect(tabBar.locator(".bottom-tab-link.active")).toContainText("TLDR Studio");
  });
});

test("header menu labels are consistent across pages", async ({ page }) => {
  const expectedLabels = ["Ingest", "TLDR Studio", "Q&A Studio", "Reviews", "Evaluation"];

  await page.addInitScript(() => {
    localStorage.setItem("yt_rag_ingest_unlocked", "1");
  });

  await page.goto("/index.html#/ingest");
  const indexNav = page.locator(".appbar-nav");
  await expect(indexNav.locator(".nav-icon")).toHaveCount(5);
  for (const label of expectedLabels) {
    await expect(indexNav.getByText(label, { exact: true })).toBeVisible();
  }

  await page.goto("/reviews.html");
  const reviewsNav = page.locator(".appbar-nav");
  await expect(reviewsNav.locator(".nav-icon")).toHaveCount(5);
  for (const label of expectedLabels) {
    await expect(reviewsNav.getByText(label, { exact: true })).toBeVisible();
  }

  await page.goto("/evaluation.html");
  const evalNav = page.locator(".appbar-nav");
  await expect(evalNav.locator(".nav-icon")).toHaveCount(5);
  for (const label of expectedLabels) {
    await expect(evalNav.getByText(label, { exact: true })).toBeVisible();
  }
});
