import { test, expect } from '@playwright/test';
import AxeBuilder from '@axe-core/playwright';
import { readFileSync, writeFileSync, unlinkSync, mkdirSync } from 'node:fs';
import { execFileSync } from 'node:child_process';
import { resolve } from 'node:path';

const root = resolve(import.meta.dirname, '../../..');
const state = resolve(root, '.local_artifacts/backtests-client');
const evidence = resolve(root, `.codex/delivery/evidence/roehub-backtests-client-v1/browser/${process.env.ROEHUB_PROOF_STAGE === 'S6' ? 'S6-foundation-regression' : process.env.ROEHUB_PROOF_STAGE === 'S5' ? 'S5-foundation-regression' : process.env.ROEHUB_PROOF_STAGE === 'S4' ? 'S4-foundation-regression' : process.env.ROEHUB_PROOF_STAGE === 'S3' ? 'S3-foundation-regression' : 'S2-foundation-regression'}`);

// Real password -> production auth -> persisted cookie. Never emit credentials or storage.
async function signIn(page: import('@playwright/test').Page) {
  const credentials = JSON.parse(readFileSync(resolve(state, 'credentials.json'), 'utf8'));
  const form = page.locator('[data-password-login]');
  await form.locator('..').locator('summary').click();
  await form.locator('[name="username"]').fill(credentials.username);
  await form.locator('[name="password"]').fill(credentials.password);
  await form.locator('button[type="submit"]').click();
  await expect(page.locator('[data-platform-client]')).toBeVisible();
}

test('real session, selected routes, assets, locale, SSR navigation and rollback', async ({ page, request }) => {
  const errors: string[] = [];
  const failedRequests: { phase: string; path: string; reason: string | undefined }[] = [];
  const consoleErrors: { phase: string; page: string }[] = [];
  const httpErrors: { phase: string; path: string; status: number }[] = [];
  let phase = 'preserved-auth';
  page.on('console', message => {
    if (message.type() === 'error') consoleErrors.push({ phase, page: new URL(page.url()).pathname });
  });
  page.on('response', response => {
    if (response.status() >= 400) httpErrors.push({ phase, path: new URL(response.url()).pathname, status: response.status() });
  });
  page.on('pageerror', error => errors.push(`${new URL(page.url()).pathname}: ${error.message}`));
  page.on('requestfailed', request => failedRequests.push({ phase, path: new URL(request.url()).pathname, reason: request.failure()?.errorText }));
  mkdirSync(evidence, { recursive: true });
  const response = await page.goto('/backtests/local-job?variant=a%2Fb');
  expect(response?.status()).toBe(200); // completed anonymous redirect to public login
  expect(new URL(page.url()).searchParams.get('next')).toBe('/backtests/local-job?variant=a%2Fb');
  await expect(page.locator('#platform-root')).toHaveCount(0);
  await signIn(page);
  phase = 'foundation';
  expect(new URL(page.url()).pathname).toBe('/backtests/local-job');
  expect(new URL(page.url()).searchParams.get('variant')).toBe('a/b');
  const reload = await page.reload();
  expect(reload?.headers()['cache-control']).toBe('private, no-store');
  await expect(page.locator('[data-platform-client]')).toBeVisible();
  const assets = await page.locator('script[type="module"]').getAttribute('src');
  expect(assets).toMatch(/^\/platform-assets\/assets\/main-.*\.js$/);
  expect((await request.get(assets!)).status()).toBe(200);
  expect((await request.get('/platform-assets/index.html')).status()).toBe(404);

  // The browser goes through Web's unchanged same-origin proxy into the real API.
  const api = await page.evaluate(async () => {
    const paths = ['/api/auth/current-user', '/api/backtests/runtime-defaults', '/api/backtests/jobs',
      '/api/ui/backtests/workstation',
      '/api/ui/backtests/artifact-date-bounds?exchange=binance&market_type=spot&symbol=BTCUSDT'];
    const results = [];
    for (const path of paths) {
      const reply = await fetch(path);
      const data = await reply.json();
      results.push({ path, status: reply.status, keys: Object.keys(data),
        sources: path.includes('workstation') ? data.sources.map((source: { name: string; status: string }) => ({ name: source.name, status: source.status })) : undefined,
        state: path.includes('artifact-date-bounds') ? data.state : undefined });
    }
    return results;
  });
  for (const reply of api) expect(reply.status, reply.path).toBe(200);
  expect(api.find(reply => reply.path.includes('artifact-date-bounds'))?.state).toBe('ready');
  const sources = api.find(reply => reply.path.includes('workstation'))!.sources;
  for (const name of ['runtime_defaults', 'market_data_reference']) {
    expect(sources.find((source: { name: string }) => source.name === name)?.status).toBe('available');
  }
  const preflight = await page.evaluate(async () => {
    const defaults = await (await fetch('/api/backtests/runtime-defaults')).json();
    const windowValue = defaults.indicator_param_specs['ma.ema'].params.window.values[0];
    const response = await fetch('/api/backtests/preflight', { method: 'POST',
      headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({
        coordinates: { exchange: 'binance', market_type: 'spot', symbol: 'BTCUSDT' },
        timeframe: '15m', time_range: { start: '2026-03-26T00:00:00Z', end: '2026-03-29T00:00:00Z' },
        indicators: [{ indicator_id: 'ma.ema', sources: ['close'], window: { start: windowValue, stop: windowValue, step: 1 } }],
        risk: { mode: 'none' }, execution: { ...defaults.execution_defaults, direction_mode: 'long_only' },
        ranking: defaults.ranking_default, top_n: defaults.top_n_default,
      }),
    });
    const body = await response.json();
    return { status: response.status, errorCount: body.errors?.length, cost: body.cost_estimate };
  });
  expect(preflight.status).toBe(200);
  expect(preflight.errorCount).toBe(0);
  expect((await request.get('http://127.0.0.1:18483/metrics')).status()).toBe(200);
  await page.getByRole('link', { name: 'Русский', exact: true }).click();
  await expect(page.locator('html')).toHaveAttribute('lang', 'ru');
  await expect(page.getByRole('heading', { name: 'Результаты бэктеста' })).toBeVisible();
  await page.screenshot({ path: resolve(evidence, 'foundation-ru.png') });
  expect(new URL(page.url()).searchParams.get('variant')).toBe('a/b');
  await page.getByRole('link', { name: 'English', exact: true }).click();
  await expect(page.getByRole('heading', { name: 'Backtest detail' })).toBeVisible();

  mkdirSync(evidence, { recursive: true });
  for (const width of [820, 1024, 1440]) {
    await page.setViewportSize({ width, height: 1000 });
    expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
    await page.screenshot({ path: resolve(evidence, `foundation-${width}.png`) });
  }
  expect((await new AxeBuilder({ page }).analyze()).violations).toEqual([]);
  await page.keyboard.press('Tab');
  expect(await page.locator(':focus').count()).toBe(1);
  expect(consoleErrors.filter(error => error.phase === 'foundation')).toEqual([]);
  expect(httpErrors.filter(error => error.phase === 'foundation')).toEqual([]);
  await expect(page.getByRole('link', { name: 'Data', exact: true })).toHaveCount(0);
  await expect(page.locator('.nav-unavailable')).toHaveAttribute('aria-disabled', 'true');
  phase = 'preserved-ssr';
  await page.getByRole('link', { name: 'Strategies', exact: true }).click();
  await expect(page.locator('[data-page="strategies"]')).toBeVisible();
  await expect(page.locator('#platform-root')).toHaveCount(0);
  await page.goBack();
  await expect(page.locator('[data-platform-client]')).toBeVisible();
  for (const [name, path] of [['Overview', '/dashboard'], ['Settings', '/settings']]) {
    await page.getByRole('link', { name, exact: true }).click();
    expect(new URL(page.url()).pathname).toBe(path);
    await expect(page.locator('#platform-root')).toHaveCount(0);
    await page.waitForLoadState('load');
    await page.goBack();
    await expect(page.locator('[data-platform-client]')).toBeVisible();
  }
  await page.goto('/backtests/new');
  await expect(page.getByRole('heading', { name: 'New backtest', level: 1 })).toBeVisible();
  const ssr = await page.goto('http://localhost:18482/backtests/new');
  expect(ssr?.headers()['cache-control']).toBe('private, no-store');
  await expect(page.locator('[data-page="backtests"]')).toBeVisible();
  await expect(page.locator('#platform-root')).toHaveCount(0);
  await page.screenshot({ path: resolve(evidence, 'ssr-rollback.png'), animations: 'disabled' });
  await page.goto('http://localhost:18480/backtests');
  await expect(page.locator('[data-platform-client]')).toBeVisible();
  const logout = page.waitForResponse(response => new URL(response.url()).pathname === '/api/auth/local/logout');
  await page.getByRole('link', { name: 'Sign out', exact: true }).click();
  expect((await logout).status()).toBe(204);
  await expect(page).toHaveURL(/\/login/);
  await page.goto('http://localhost:18480/backtests');
  await expect(page).toHaveURL(/\/login\?next=/);
  await expect(page.locator('#platform-root')).toHaveCount(0);
  // SSR pages can call unrelated projections outside this focused fixture; inspect separately.
  expect(errors).toEqual([]);
  expect(failedRequests.filter(failure => failure.phase === 'foundation' || failure.reason !== 'net::ERR_ABORTED')).toEqual([]);
  writeFileSync(resolve(evidence, 'foundation-observations.json'), JSON.stringify({
    boundary: 'Real Web + production auth/API + disposable PostgreSQL/ClickHouse + synthetic artifact files',
    api, preflight, idleRunnerMetrics: 200,
    javascriptErrors: errors, transportFailures: failedRequests, consoleErrors, httpErrors,
    preservedSsrNote: 'Production strategy CRUD/dashboard are installed. Optional OIDC and unrelated overview/account projections are omitted; existing Settings market-data CSS/JS assets are absent. Navigation is proved, not complete Settings functionality.',
    viewportWidths: [820, 1024, 1440], axeViolations: 0,
  }, null, 2) + '\n');
});

test('identity outage stays unavailable; expired real session requires safe sign-in', async ({ page }) => {
  await page.goto('/backtests');
  await signIn(page);
  writeFileSync(resolve(state, 'identity-unavailable'), 'controlled fault');
  try {
    const response = await page.goto('/backtests/new');
    expect(response?.status()).toBe(502);
    expect(new URL(page.url()).pathname).toBe('/backtests/new');
    await expect(page.locator('#platform-root')).toHaveCount(0);
    expect(response?.headers()['cache-control']).toBe('private, no-store');
  } finally { unlinkSync(resolve(state, 'identity-unavailable')); }
  await page.goto('/backtests');
  await expect(page.locator('[data-platform-client]')).toBeVisible();
  execFileSync(resolve(root, '.venv/bin/python'), ['-m', 'tools.qa.backtests_client_fixture', 'expire'], { cwd: root });
  await page.goto('/backtests/local-job?variant=a%2Fb');
  expect(new URL(page.url()).pathname).toBe('/login');
  expect(new URL(page.url()).searchParams.get('next')).toBe('/backtests/local-job?variant=a%2Fb');
  await expect(page.locator('#platform-root')).toHaveCount(0);
});
