import { test, expect, chromium, type Page } from '@playwright/test';
import { jobSchema } from '../src/library-api';
import AxeBuilder from '@axe-core/playwright';
import { readFileSync, writeFileSync, mkdirSync, mkdtempSync, rmSync } from 'node:fs';
import { resolve } from 'node:path';
import { tmpdir } from 'node:os';
import { execFileSync } from 'node:child_process';
import { randomUUID } from 'node:crypto';
const root = resolve(import.meta.dirname, '../../..');
const evidence = resolve(root, `.codex/delivery/evidence/roehub-backtests-client-v1/browser/${process.env.ROEHUB_PROOF_STAGE === 'S6' ? 'S6-library-regression' : process.env.ROEHUB_PROOF_STAGE === 'S5' ? 'S5-library-regression' : process.env.ROEHUB_PROOF_STAGE === 'S4' ? 'S4-library-regression' : process.env.ROEHUB_PROOF_STAGE === 'S3' ? 'S3-library-regression' : 'S2'}`);
async function signIn(page: Page) {
  await page.goto('http://localhost:18480/backtests');
  const credentials = JSON.parse(readFileSync(resolve(root, '.local_artifacts/backtests-client/credentials.json'), 'utf8'));
  const form = page.locator('[data-password-login]');
  await form.locator('..').locator('summary').click();
  await form.locator('[name="username"]').fill(credentials.username);
  await form.locator('[name="password"]').fill(credentials.password);
  await form.locator('button[type="submit"]').click();
  await expect(page.locator('[data-platform-client]')).toBeVisible();
}
async function seedJob(page: Page, label: string) {
  // Disposable setup through the production endpoint, not a client submit/recovery acceptance claim.
  const payload = await page.evaluate(async ({ label, key }) => {
    const defaults = await (await fetch('/api/backtests/runtime-defaults')).json();
    const window = defaults.indicator_param_specs['ma.ema'].params.window.values[0];
    const body = { strategy_name: label, coordinates: { exchange: 'binance', market_type: 'spot', symbol: 'BTCUSDT' },
      timeframe: '15m', time_range: { start: '2026-03-26T00:00:00Z', end: '2026-03-29T00:00:00Z' },
      indicators: [{ indicator_id: 'ma.ema', sources: ['close'], window: { start: window, stop: window, step: 1 } }],
      risk: { mode: 'none' }, execution: { ...defaults.execution_defaults, direction_mode: 'long_only' },
      ranking: defaults.ranking_default, top_n: defaults.top_n_default };
    const preflight = await fetch('/api/backtests/preflight', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
    if (preflight.status !== 200 || (await preflight.json()).errors.length) throw new Error('Disposable job preflight failed');
    const reply = await fetch('/api/backtests/jobs', { method: 'POST', headers: { 'Content-Type': 'application/json', 'Idempotency-Key': key }, body: JSON.stringify(body) });
    if (reply.status !== 201) throw new Error(`Disposable create failed: ${reply.status}`);
    const job = await reply.json();
    return job;
  }, { label, key: randomUUID() });
  const parsed = jobSchema.safeParse(payload);
  if (!parsed.success) throw new Error(JSON.stringify(parsed.error.issues.map(issue => ({ path: issue.path, code: issue.code, message: issue.message }))));
  return { job_id: parsed.data.job_id, status: 201, state: parsed.data.state };
}

test('real library, empty/cursor filters, independent deep link, focus, locale and visual proof', async ({ page }) => {
  mkdirSync(evidence, { recursive: true });
  await signIn(page);
  const errors: string[] = []; const http: { path: string; status: number }[] = []; const failures: { path: string; reason?: string }[] = [];
  page.on('pageerror', error => errors.push(error.message));
  page.on('console', message => { if (message.type() === 'error') errors.push('console error'); });
  page.on('response', response => { if (response.status() >= 400) http.push({ path: new URL(response.url()).pathname, status: response.status() }); });
  page.on('requestfailed', request => failures.push({ path: new URL(request.url()).pathname, reason: request.failure()?.errorText }));
  await expect(page.getByRole('heading', { name: 'No backtests yet' })).toBeVisible();
  await page.screenshot({ path: resolve(evidence, 'empty.png'), fullPage: true });
  const first = await seedJob(page, 'EMA · BTC research A');
  const second = await seedJob(page, 'EMA · BTC research B');
  expect(first.job_id).not.toBe(second.job_id);
  await page.reload();
  await expect(page.getByRole('link', { name: 'EMA · BTC research A' })).toBeVisible();
  await expect(page.getByRole('link', { name: 'EMA · BTC research B' })).toBeVisible();
  await page.getByText('Search, instrument and date filters', { exact: true }).click();
  await expect(page.getByText(/server job projection is unavailable/)).toBeVisible();
  await expect(page.getByRole('searchbox')).toBeDisabled();
  await page.screenshot({ path: resolve(evidence, 'projection-unavailable.png'), fullPage: true });
  await page.getByText('Search, instrument and date filters', { exact: true }).click();
  const filters = page.getByRole('button', { name: 'Filters', exact: true });
  await filters.focus(); await page.keyboard.press('Enter');
  const dialog = page.getByRole('dialog');
  await expect(dialog).toBeVisible();
  await expect(dialog.getByRole('button', { name: 'Close filters' })).toBeFocused();
  await page.keyboard.press('Shift+Tab');
  await expect(dialog.getByRole('button', { name: 'Done' })).toBeFocused();
  await page.keyboard.press('Tab');
  await expect(dialog.getByRole('button', { name: 'Close filters' })).toBeFocused();
  expect((await new AxeBuilder({ page }).analyze()).violations).toEqual([]);
  await dialog.getByLabel('Risk mode').selectOption('tp_sl_grid');
  await dialog.getByLabel('Page size (1–250)').fill('1');
  await dialog.getByLabel('Page size (1–250)').press('Tab');
  await page.keyboard.press('Escape');
  await expect(dialog).not.toBeVisible(); await expect(filters).toBeFocused();
  await expect(page.getByRole('heading', { name: 'No matching jobs on this page' })).toBeVisible();
  await page.getByRole('button', { name: 'Next page' }).click();
  await expect(page).toHaveURL(/cursor=/);
  await page.goBack();
  expect(new URL(page.url()).searchParams.has('cursor')).toBe(false);
  await page.goForward(); expect(new URL(page.url()).searchParams.has('cursor')).toBe(true);
  await page.reload(); await expect(page.locator('[data-platform-client]')).toBeVisible();
  expect(new URL(page.url()).searchParams.get('risk_mode')).toBe('tp_sl_grid');
  // State filter remains real; detail reads work even when the list excludes the selected job.
  await page.goto(`/backtests/${first.job_id}?state=cancelled&variant=a%2Fb`);
  await expect(page.getByRole('heading', { name: 'EMA · BTC research A' })).toBeVisible();
  await expect(page.getByRole('heading', { name: 'No jobs on this page' })).toBeVisible();
  expect(new URL(page.url()).searchParams.get('variant')).toBe('a/b');
  await page.reload(); await expect(page.getByRole('heading', { name: 'EMA · BTC research A' })).toBeVisible();
  await page.getByRole('link', { name: 'Back to library' }).click();
  await expect(page).toHaveURL(/\/backtests\?state=cancelled$/);
  await page.goBack(); await expect(page.getByRole('heading', { name: 'EMA · BTC research A' })).toBeVisible();
  // Show a real selected row for the visual checkpoint.
  await page.goto(`/backtests/${first.job_id}`);
  await expect(page.getByRole('heading', { name: 'EMA · BTC research A' })).toBeVisible();
  for (const locale of ['en', 'ru']) {
    await page.getByRole('link', { name: locale === 'en' ? 'English' : 'Русский', exact: true }).click();
    await expect(page.locator('html')).toHaveAttribute('lang', locale);
    for (const width of [820, 1024, 1440]) {
      await page.setViewportSize({ width, height: 1000 });
      await expect(page.getByRole('heading', { name: 'EMA · BTC research A' })).toBeVisible();
      expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
      expect((await new AxeBuilder({ page }).analyze()).violations).toEqual([]);
      await page.screenshot({ path: resolve(evidence, `${locale}-${width}.png`), fullPage: true });
    }
  }
  expect(errors).toEqual([]); expect(http).toEqual([]);
  expect(failures.filter(item => item.reason !== 'net::ERR_ABORTED')).toEqual([]);
  writeFileSync(resolve(evidence, 'observations.json'), JSON.stringify({
    boundary: 'Production local Web/auth/API + disposable PostgreSQL/ClickHouse; two real API-created jobs from 4320 synthetic candles',
    jobs: [first, second], javascriptAndConsoleErrors: errors, httpErrors: http, transportFailures: failures,
    widths: [820, 1024, 1440], locales: ['en', 'ru'], axeViolations: 0,
    checks: ['real empty list', 'two real records', 'server risk empty-page cursor', 'state filter', 'deep-link refresh/back/forward', 'encoded variant', 'dialog keyboard containment/return', 'projection unavailable'],
  }, null, 2) + '\n');
});

test('native Chromium 200% zoom retains usable RU/EN shell and dialog', async () => {
  // An isolated test extension sets native tab zoom; this is not CSS zoom or pinch emulation.
  const directory = mkdtempSync(resolve(tmpdir(), 'roehub-s2-zoom-'));
  const extension = resolve(directory, 'extension'); mkdirSync(extension);
  writeFileSync(resolve(extension, 'manifest.json'), JSON.stringify({ manifest_version: 3, name: 'Local zoom proof', version: '1.0', permissions: ['tabs'], background: { service_worker: 'worker.js' } }));
  writeFileSync(resolve(extension, 'worker.js'), 'chrome.runtime.onInstalled.addListener(() => {});');
  const context = await chromium.launchPersistentContext(resolve(directory, 'profile'), {
    channel: 'chromium', headless: true, viewport: null,
    args: [`--disable-extensions-except=${extension}`, `--load-extension=${extension}`, '--window-size=1440,1100'],
  });
  try {
    const page = await context.newPage(); await signIn(page);
    const cdp = await context.newCDPSession(page);
    const nativeScreenshot = async (name: string) => {
      const shot = await cdp.send('Page.captureScreenshot', { format: 'png', fromSurface: true });
      writeFileSync(resolve(evidence, name), Buffer.from(shot.data, 'base64'));
    };
    const worker = context.serviceWorkers()[0] ?? await context.waitForEvent('serviceworker');
    await worker.evaluate("chrome.tabs.query({url:'http://localhost:18480/*'}).then(tabs=>Promise.all(tabs.map(tab=>chrome.tabs.setZoom(tab.id,2))))");
    const nativeZoom = await worker.evaluate("chrome.tabs.query({url:'http://localhost:18480/*'}).then(tabs=>chrome.tabs.getZoom(tabs[0].id))");
    expect(nativeZoom).toBe(2);
    const sizes = [];
    for (const locale of ['en', 'ru']) {
      await page.getByRole('link', { name: locale === 'en' ? 'English' : 'Русский', exact: true }).click();
      await expect(page.locator('html')).toHaveAttribute('lang', locale);
      await expect(page.getByRole('link', { name: 'EMA · BTC research A' })).toBeVisible();
      const metrics = await page.evaluate(() => ({ innerWidth, outerWidth, devicePixelRatio, scrollWidth: document.documentElement.scrollWidth }));
      expect(metrics.innerWidth).toBeLessThan(800); expect(metrics.scrollWidth).toBeLessThanOrEqual(metrics.innerWidth);
      sizes.push({ locale, ...metrics });
      expect((await new AxeBuilder({ page }).analyze()).violations).toEqual([]);
      await page.getByRole('button', { name: locale === 'en' ? 'Filters' : 'Фильтры', exact: true }).click();
      await expect(page.getByRole('dialog')).toBeVisible();
      expect((await new AxeBuilder({ page }).analyze()).violations).toEqual([]);
      const box = await page.getByRole('dialog').boundingBox();
      expect(box!.x).toBeGreaterThanOrEqual(0);
      expect(box!.x + box!.width).toBeLessThanOrEqual(metrics.innerWidth);
      await nativeScreenshot(`${locale}-native-zoom-200-dialog.png`);
      await page.keyboard.press('Escape');
      await nativeScreenshot(`${locale}-native-zoom-200.png`);
    }
    writeFileSync(resolve(evidence, 'zoom-observations.json'), JSON.stringify({ nativeZoom, sizes, axeViolations: 0 }, null, 2) + '\n');
  } finally { await context.close(); rmSync(directory, { recursive: true, force: true }); }
});


test('real API expiry removes private content and stops protected reads', async ({ page }) => {
  await signIn(page);
  await expect(page.getByRole('link', { name: 'EMA · BTC research A' })).toBeVisible();
  execFileSync(resolve(root, '.venv/bin/python'), ['-m', 'tools.qa.backtests_client_fixture', 'expire'], { cwd: root });
  await page.getByRole('button', { name: 'Filters', exact: true }).click();
  const expiredRead = page.waitForResponse(response => new URL(response.url()).pathname === '/api/backtests/jobs' && response.status() === 401);
  await page.getByRole('dialog').getByRole('combobox', { name: 'State', exact: true }).selectOption('cancelled');
  await expiredRead;
  await expect(page.getByRole('link', { name: 'Sign in', exact: true })).toBeVisible();
  await expect(page.locator('[data-platform-client]')).toHaveCount(0);
  const afterExpiry: string[] = [];
  page.on('request', request => { const path = new URL(request.url()).pathname; if (path.startsWith('/api/')) afterExpiry.push(path); });
  // Exceeds the identity refresh interval, proving polling stopped after the API 401.
  await page.waitForTimeout(31000);
  expect(afterExpiry).toEqual([]);
  await page.screenshot({ path: resolve(evidence, 'expired.png'), fullPage: true });
  writeFileSync(resolve(evidence, 'expiry-observations.json'), JSON.stringify({
    fault: 'Real persisted session expiration; direct jobs GET returned 401',
    privateContentRemoved: true, observationMs: 31000, protectedRequestsAfterExpiry: afterExpiry,
  }, null, 2) + '\n');
});
