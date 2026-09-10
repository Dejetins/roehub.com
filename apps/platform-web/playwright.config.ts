import { defineConfig } from '@playwright/test';

export default defineConfig({
  testDir: './e2e',
  workers: 1,
  retries: 0,
  timeout: 45_000,
  outputDir: '../../.local_artifacts/platform-web-test-results',
  reporter: [['list']],
  use: { baseURL: 'http://localhost:18480', browserName: 'chromium',
    viewport: { width: 1440, height: 1000 }, trace: 'off', screenshot: 'off' },
  webServer: {
    command: '.venv/bin/python -m tools.qa.backtests_client_fixture',
    cwd: '../..',
    url: 'http://127.0.0.1:18480/health/ready',
    timeout: 180_000,
    gracefulShutdown: { signal: 'SIGTERM', timeout: 30_000 },
    reuseExistingServer: process.env.ROEHUB_PROOF_REUSE === 'true',
  },
});
