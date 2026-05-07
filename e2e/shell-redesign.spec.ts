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
    await expect(tabBar.locator(".bottom-tab-link")).toHaveCount(4);

    await tabBar.getByRole("button", { name: "TLDR Studio" }).click();
    await expect(page).toHaveURL(/#\/tldr$/);
    await expect(tabBar.locator(".bottom-tab-link.active")).toContainText("TLDR Studio");
  });
});

test("header menu labels are consistent across pages", async ({ page }) => {
  const expectedPrimaryLabels = ["Ingest", "TLDR Studio", "Q&A Studio", "Tools"];
  const expectedToolLabels = ["Reviews", "Evidence", "Evaluation", "Chunking"];

  async function expectGroupedNav(nav) {
    const directItems = nav.locator(":scope > .app-nav-link, :scope > .nav-tools-menu > summary");
    await expect(directItems).toHaveCount(4);
    for (const label of expectedPrimaryLabels) {
      await expect(directItems.getByText(label, { exact: true })).toBeVisible();
    }

    const toolsMenu = nav.locator(".nav-tools-menu");
    await toolsMenu.evaluate((element) => element.setAttribute("open", ""));
    for (const label of expectedToolLabels) {
      await expect(toolsMenu.getByText(label, { exact: true })).toBeVisible();
    }
  }

  await page.addInitScript(() => {
    localStorage.setItem("yt_rag_ingest_unlocked", "1");
  });

  await page.goto("/index.html#/ingest");
  await expectGroupedNav(page.locator(".appbar-nav"));

  await page.goto("/reviews.html");
  await expectGroupedNav(page.locator(".appbar-nav"));

  await page.goto("/evaluation.html");
  await expectGroupedNav(page.locator(".appbar-nav"));
});
