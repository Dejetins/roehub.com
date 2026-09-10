import { test, expect, type Page } from '@playwright/test';
import AxeBuilder from '@axe-core/playwright';
import { readFileSync, writeFileSync, mkdirSync } from 'node:fs';
import { resolve } from 'node:path';
import { spawn, execFileSync } from 'node:child_process';

const root = resolve(import.meta.dirname, '../../..');
const evidence = resolve(root, '.codex/delivery/evidence/roehub-backtests-client-v1/browser/S6');
const fixture = (action: string) => execFileSync(resolve(root, '.venv/bin/python'),
  ['-m', 'tools.qa.backtests_client_fixture', action], { cwd: root });

async function configure(page: Page) {
  await page.getByLabel('Job label (optional)').fill('S6 integrated journey');
  await page.getByLabel('Timeframe', { exact: true }).selectOption('15m');
  await page.getByLabel('Start date', { exact: true }).fill('2026-03-26');
  await page.getByLabel('End date', { exact: true }).fill('2026-03-29');
  await page.getByLabel('Indicator', { exact: true }).selectOption('ma.ema');
}

test('integrated real configure worker results save and flag-off domain preservation', async ({ page }) => {
  test.setTimeout(240_000);
  mkdirSync(evidence, { recursive: true });
  await page.goto('file://' + resolve(root, '.codex/delivery/evidence/roehub-ui-agent-governed-pilot/specimens/2026-08-03-linear-black-workbench-v23.html'));
  await page.screenshot({ path: resolve(evidence, 'v23-reference.png'), fullPage: true });
  const network: { phase: string; path: string; method: string; status: number }[] = [];
  const consoleErrors: { phase: string; message: string }[] = [];
  const failed: { phase: string; path: string; reason?: string }[] = [];
  const pageErrors: string[] = [];
  const jobReads: { state: string; processed: number; total: number }[] = [];
  let phase = 'login', createPosts = 0, savePosts = 0, starts = 0;
  page.on('pageerror', () => pageErrors.push(phase));
  page.on('console', m => { if (m.type() === 'error') consoleErrors.push({ phase,
    message: m.text().replace(/http[^ ]+/g, '[URL]').slice(0, 200) }); });
  page.on('requestfailed', r => failed.push({ phase, path: new URL(r.url()).pathname, reason: r.failure()?.errorText }));
  page.on('request', r => {
    const path = new URL(r.url()).pathname;
    if (r.method() !== 'POST') return;
    if (path === '/api/backtests/jobs') createPosts++;
    if (path.endsWith('/strategies')) savePosts++;
    if (/\/(run|start|runs)(\/|$)/.test(path)) starts++;
  });
  page.on('response', async r => {
    const path = new URL(r.url()).pathname;
    if (path.startsWith('/api/') || r.status() >= 400) network.push({ phase, path, method: r.request().method(), status: r.status() });
    if (phase === 'client' && /\/api\/backtests\/jobs\/[\da-f-]+$/.test(path) && r.ok()) {
      const body = await r.json().catch(() => null);
      if (body) jobReads.push({ state: body.state, processed: body.progress.processed_units, total: body.progress.total_units });
    }
  });
  await page.goto('/backtests/new');
  const credentials = JSON.parse(readFileSync(resolve(root, '.local_artifacts/backtests-client/credentials.json'), 'utf8'));
  const login = page.locator('[data-password-login]');
  await login.locator('..').locator('summary').click();
  await login.locator('[name=username]').fill(credentials.username);
  await login.locator('[name=password]').fill(credentials.password);
  await login.locator('button[type=submit]').click();
  await expect(page.locator('[data-platform-client]')).toBeVisible();
  await page.waitForLoadState('networkidle');
  phase = 'client';
  await configure(page);
  const preflightResponse = page.waitForResponse(r => r.url().endsWith('/preflight'));
  await page.getByRole('button', { name: 'Check configuration', exact: true }).click();
  const preflight = await preflightResponse;
  expect(preflight.status()).toBe(200);
  expect((await preflight.json()).errors).toHaveLength(0);
  await expect(page.getByRole('button', { name: 'Submit backtest', exact: true })).toBeEnabled();
  await page.screenshot({ path: resolve(evidence, 'integrated-preflight.png'), fullPage: true });

  // Pause only this fixture's scheduler so the browser observes an actual queued job.
  // The observer is read-only and never changes domain state or refresh scheduling.
  fixture('pause-runner');
  const workerStates: string[] = [];
  const observer = spawn(resolve(root, '.venv/bin/python'), ['-u', '-c', `
import json,time
from pathlib import Path
import psycopg
private=json.loads(Path('.local_artifacts/backtests-client/credentials.json').read_text())
with psycopg.connect(private['dsn'],autocommit=True) as db:
    last=None
    for _ in range(12000):
        rows=db.execute("SELECT state FROM backtest_jobs ORDER BY created_at DESC LIMIT 1").fetchall()
        if rows and rows[0][0]!=last:
            last=rows[0][0];print(last,flush=True)
            if last in ('succeeded','failed','cancelled'):break
        time.sleep(.01)
`], { cwd: root, stdio: ['ignore', 'pipe', 'ignore'] });
  observer.stdout.on('data', data => workerStates.push(...String(data).trim().split('\n')));
  try {
    const creationResponse = page.waitForResponse(r => new URL(r.url()).pathname === '/api/backtests/jobs' && r.request().method() === 'POST');
    await page.getByRole('button', { name: 'Submit backtest', exact: true }).dblclick();
    const creation = await creationResponse;
    expect(creation.status()).toBe(201);
    const job = await creation.json();
    expect(job.state).toBe('queued');
    await expect(page.locator('.job-detail').getByRole('heading', { name: 'S6 integrated journey' })).toBeVisible();
    await expect(page.getByRole('button', { name: 'Cancel backtest', exact: true })).toBeEnabled();
    await page.screenshot({ path: resolve(evidence, 'integrated-queued.png'), fullPage: true });
    fixture('resume-runner');
    await expect(page.getByText('The server reports completion.', { exact: true })).toBeVisible({ timeout: 180_000 });
    await expect(page.getByRole('img', { name: 'Equity', exact: true })).toBeVisible({ timeout: 60_000 });
    const variant = new URL(page.url()).searchParams.get('variant')!;
    expect(variant).toBeTruthy();
    expect(workerStates).toContain('running');
    expect(workerStates).toContain('succeeded');
    expect(jobReads.some(r => r.state === 'queued')).toBe(true);
    expect(jobReads.some(r => r.state === 'succeeded')).toBe(true);
    const base = `/api/backtests/jobs/${job.job_id}/variants/${encodeURIComponent(variant)}`;
    await expect(page.getByRole('img', { name: 'Equity', exact: true })).toHaveAccessibleDescription(/Equity in quote currency.*UTC/);
    await page.getByText('Show accessible data table', { exact: true }).click();
    await expect(page.getByRole('region', { name: 'Equity', exact: true })).toHaveAccessibleDescription(/Equity in quote currency.*UTC/);
    await expect(page.getByRole('region', { name: 'Equity', exact: true }).locator('tbody tr')).not.toHaveCount(0);
    for (const tab of ['Drawdown · %', 'Monthly statistics', 'Symbol statistics', 'Trades']) {
      await page.getByRole('tab', { name: tab, exact: true }).click();
      if (tab === 'Drawdown · %') await expect(page.getByRole('img', { name: tab, exact: true })).toBeVisible();
      else await expect(page.getByRole('tabpanel').locator('tbody tr')).not.toHaveCount(0);
    }
    const downloadEvent = page.waitForEvent('download');
    await page.getByRole('button', { name: 'Download CSV', exact: true }).click();
    const download = await downloadEvent;
    const csv = readFileSync((await download.path())!, 'utf8');
    expect(csv).toContain('trade_index,entry_timestamp');
    expect(csv.trim().split('\n')).toHaveLength(2);
    const readiness = await (await page.request.get(`${base}/compatibility-readiness`)).json();
    expect(readiness.compatibility_state).toBe('not_launchable');
    await page.getByRole('button', { name: 'Save strategy', exact: true }).click();
    const saveResponse = page.waitForResponse(r => r.url().endsWith('/strategies') && r.request().method() === 'POST');
    await page.getByRole('button', { name: 'Confirm save', exact: true }).click();
    const savedResponse = await saveResponse;
    expect(savedResponse.status()).toBe(201);
    expect(savedResponse.request().postData()).toBeNull();
    const saved = await savedResponse.json();
    const strategyId = saved.strategy.strategy_id;
    const preservedJob = await (await page.request.get(`/api/backtests/jobs/${job.job_id}`)).json();
    const preservedStrategy = await (await page.request.get(`/api/strategies/${strategyId}`)).json();
    const deepLink = `/backtests/${job.job_id}?variant=${encodeURIComponent(variant)}`;
    await page.reload();
    await expect(page.locator('[data-result-variant]')).toHaveAttribute('data-result-variant', variant);
    await page.getByRole('tab', { name: 'Equity', exact: true }).click();
    await expect(page.getByRole('img', { name: 'Equity', exact: true })).toBeVisible();
    expect((await new AxeBuilder({ page }).analyze()).violations).toEqual([]);
    await page.screenshot({ path: resolve(evidence, 'integrated-result.png'), fullPage: true });

    // Both Web processes use the exact same API/databases; only the feature flag differs.
    phase = 'flag-off';
    const off = await page.goto(`http://localhost:18482${deepLink}`);
    expect(off?.headers()['cache-control']).toBe('private, no-store');
    await expect(page.locator('#platform-root')).toHaveCount(0);
    await expect(page.locator('[data-backtests-root]')).toHaveAttribute('data-initial-job-id', job.job_id);
    expect(new URL(page.url()).searchParams.get('variant')).toBe(variant);
    const oldJob = await page.request.get(`http://localhost:18482/api/backtests/jobs/${job.job_id}`);
    expect(oldJob.status()).toBe(200);
    expect(await oldJob.json()).toMatchObject({ job_id: preservedJob.job_id, state: preservedJob.state,
      created_at: preservedJob.created_at, updated_at: preservedJob.updated_at,
      request: preservedJob.request, terminal_summary: preservedJob.terminal_summary });
    const oldStrategy = await page.request.get(`http://localhost:18482/api/strategies/${strategyId}`);
    expect(oldStrategy.status()).toBe(200);
    expect(await oldStrategy.json()).toEqual(preservedStrategy);
    const oldTrades = await page.request.get(`http://localhost:18482${base}/trades`);
    expect(oldTrades.status()).toBe(200);
    expect((await oldTrades.json()).items).toHaveLength(1);
    await page.screenshot({ path: resolve(evidence, 'flag-off-created-job.png'), fullPage: true });
    await page.goto(`http://localhost:18482/strategies/${strategyId}`);
    await expect(page.locator('[data-strategies-root]')).toHaveAttribute('data-initial-strategy-id', strategyId);
    await expect(page.locator(`[data-saved-strategy-id="${strategyId}"]`)).toBeVisible();
    await page.screenshot({ path: resolve(evidence, 'flag-off-saved-strategy.png'), fullPage: true });
    phase = 'restored-client';
    await page.goto(`http://localhost:18480${deepLink}`);
    await expect(page.locator('[data-result-variant]')).toHaveAttribute('data-result-variant', variant);
    await expect(page.getByRole('img', { name: 'Equity', exact: true })).toBeVisible();
    const restoredStrategy = await page.request.get(`/api/strategies/${strategyId}`);
    expect(await restoredStrategy.json()).toEqual(preservedStrategy);
    expect(createPosts).toBe(1); expect(savePosts).toBe(1); expect(starts).toBe(0);
    expect(pageErrors).toEqual([]);
    expect(network.filter(r => r.status >= 400).every(r => r.phase === 'login' && r.path === '/api/auth/oidc/status' && r.status === 404)).toBe(true);
    expect(consoleErrors.every(r => r.phase === 'login' && /status of 404/.test(r.message))).toBe(true);
    expect(failed.every(r => r.reason === 'net::ERR_ABORTED')).toBe(true);
    writeFileSync(resolve(evidence, 'integrated-journey.json'), JSON.stringify({
      boundary: 'Real production API/auth/worker, disposable PostgreSQL/ClickHouse and synthetic production artifacts; no mocked response',
      jobId: job.job_id, variant, strategyId, preflight: 200, create: 201, createPosts, savePosts,
      workerStates, jobReads, terminal: preservedJob.state, csvRows: 1, liveReadiness: readiness.compatibility_state,
      save: 201, bodyless: true, starts, axe: 0,
      rollback: { flagOff: true, ssrRoot: true, jobUnchanged: true, strategyUnchanged: true,
        existingTrades: 1, ssrStrategyRow: true, clientRestored: true, migration: false },
      network, consoleErrors, pageErrors, failed,
    }, null, 2));
  } finally { observer.kill(); fixture('resume-runner'); }
});
