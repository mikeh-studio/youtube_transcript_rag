import { existsSync } from "node:fs";
import { defineConfig } from "@playwright/test";

const macChromePath = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome";
const launchOptions =
  process.platform === "darwin" && !process.env.CI && existsSync(macChromePath)
    ? { executablePath: macChromePath }
    : undefined;

export default defineConfig({
  testDir: "./e2e",
  timeout: 30_000,
  expect: { timeout: 10_000 },
  fullyParallel: false,
  retries: 0,
  use: {
    baseURL: "http://127.0.0.1:4173",
    trace: "on-first-retry",
    screenshot: "only-on-failure",
    launchOptions,
  },
  webServer: {
    command: "python -m http.server 4173 --directory local_preview/web",
    url: "http://127.0.0.1:4173/index.html",
    reuseExistingServer: true,
    timeout: 30_000,
  },
});
