import path from "node:path";
import { fileURLToPath } from "node:url";

import { defineConfig } from "@playwright/test";

const repositoryRoot = fileURLToPath(new URL("../../../../", import.meta.url));

export default defineConfig({
  testDir: ".",
  testMatch: "*.spec.ts",
  fullyParallel: false,
  workers: 1,
  timeout: 90_000,
  expect: { timeout: 10_000 },
  outputDir: path.resolve(
    process.cwd(),
    "output/playwright/linear-frontend-architecture-spike/test-results",
  ),
  reporter: [["line"]],
  use: {
    channel: "chrome",
    baseURL: "http://127.0.0.1:4173",
    viewport: { width: 1440, height: 900 },
    trace: "on",
    screenshot: "only-on-failure",
    video: "off",
  },
  webServer: {
    command: "npm run prototype",
    cwd: repositoryRoot,
    url: "http://127.0.0.1:4173/__prototype/react/",
    reuseExistingServer: false,
    timeout: 120_000,
  },
});
