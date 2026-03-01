import { expect, test } from '@playwright/test';

test('navigates from home poster to detail page and preserves title', async ({ page }) => {
  const mockVideos = [
    {
      video_id: 'TWbQXYQMCxg',
      title: 'Frieren Radio Episode 3',
      language: 'ja',
      num_chunks: 123,
    },
  ];

  await page.route('**/v1/videos', async (route) => {
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({ ok: true, videos: mockVideos }),
    });
  });

  await page.goto('/index.html');

  const posterCard = page.locator('[data-testid="poster-card"]').first();
  await expect(posterCard).toBeVisible();

  const homeTitle = (await posterCard.locator('[data-testid="poster-title"]').innerText()).trim();
  await posterCard.click();

  await expect(page).toHaveURL(/video_detail\.html\?video_id=/);

  const detailTitleLocator = page.locator('[data-testid="detail-title"]');
  await expect(detailTitleLocator).toHaveText(homeTitle);
  const detailTitle = (await detailTitleLocator.innerText()).trim();
  await expect(detailTitle).toBe(homeTitle);
});
