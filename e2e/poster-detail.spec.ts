import { expect, test } from '@playwright/test';

test('renders ingested video thumbnails and review links', async ({ page }) => {
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

  await page.goto('/index.html#/ingest');

  const videoCard = page.locator('.ingest-video-card').first();
  await expect(videoCard).toBeVisible();

  await expect(videoCard.locator('.ingest-video-title')).toHaveText('Frieren Radio Episode 3');
  await expect(videoCard.locator('.ingest-video-thumb > img')).toHaveAttribute(
    'src',
    /https:\/\/i\.ytimg\.com\/vi\/TWbQXYQMCxg\/hqdefault\.jpg/,
  );

  await expect(videoCard.getByRole('link', { name: 'Review' })).toHaveAttribute(
    'href',
    /\/reviews\.html\?video_id=TWbQXYQMCxg$/,
  );
});
