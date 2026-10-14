import { test, expect } from '@playwright/test';
import { readFileSync, writeFileSync, mkdirSync, renameSync } from 'node:fs';
import { resolve } from 'node:path';

const root = resolve(import.meta.dirname, '../../..');
const evidence = resolve(root, '.codex/agents/generated/postgresql-only-storage-v1/reports/S08/browser');

test('S08 accepted platform and integrated operational route evidence', async ({ page }) => {
  mkdirSync(evidence, { recursive: true });
  await page.goto('/backtests');
  const credentials = JSON.parse(readFileSync(resolve(root, '.local_artifacts/s08-browser/credentials.json'), 'utf8'));
  const form = page.locator('[data-password-login]');
  await form.locator('..').locator('summary').click();
  await form.locator('[name=username]').fill(credentials.username);
  await form.locator('[name=password]').fill(credentials.password);
  await form.locator('button[type=submit]').click();
  await expect(page.locator('[data-platform-client]')).toBeVisible();
  // Capture only after login; never retain secrets, cookies or authentication traces.
  const responses: {path:string;status:number}[] = [];
  const errors: string[] = [];
  page.on('response', r => { if (r.status() >= 400) responses.push({path:new URL(r.url()).pathname,status:r.status()}); });
  page.on('pageerror', () => errors.push('pageerror'));
  await expect(page.locator('.nav-unavailable')).toHaveAttribute('aria-disabled', 'true');
  await page.screenshot({ path: resolve(evidence, 'accepted-platform.png'), fullPage:true });
  const routes = await page.locator('.sidebar a').evaluateAll(nodes => nodes.map(n => n.getAttribute('href')));
  await page.locator('.sidebar a[href="/settings"]').click();
  await page.locator('[data-settings-tab-button="market_data"]').click();
  await expect(page.locator('[data-market-data-settings]')).toBeVisible();
  const missingScript = await page.request.get('/assets/js/pages/market-data-settings.js');
  const markets = await page.locator('[data-market-data-market] option').count();
  await page.screenshot({path:resolve(evidence, 'settings-market-data.png'),fullPage:true});
  const settings = {url:new URL(page.url()).pathname, surface:'apps/web integration template', scriptStatus:missingScript.status(), marketOptions:markets};
  await page.goto('/monitoring');
  await expect(page.locator('[data-monitoring-root]')).toBeVisible();
  const summary = await page.request.get('/api/ui/dashboard/summary');
  await page.screenshot({path:resolve(evidence,'integrated-monitoring.png'),fullPage:true});
  await page.goto('/backtests/new');
  await expect(page.locator('[data-platform-client]')).toBeVisible();
  const noHistory = await page.evaluate(async () => {
    const r = await fetch('/api/ui/backtests/artifact-date-bounds?exchange=binance&market_type=spot&symbol=ETHUSDT');
    const data = await r.json(); return {status:r.status,state:data.state};
  });
  await expect(page.getByLabel('Job label (optional)')).toBeVisible();
  await page.screenshot({path:resolve(evidence,'backtest-builder.png'),fullPage:true});
  const pointer = resolve(root, '.local_artifacts/s08-browser/artifacts/backtest/v2/binance/spot/BTCUSDT/current.yaml');
  renameSync(pointer, pointer+'.s08-held');
  try {
    await page.reload();
    await expect(page.getByLabel('Job label (optional)')).toBeVisible();
    await page.getByLabel('Start date',{exact:true}).fill('2026-03-26');
    await page.getByLabel('End date',{exact:true}).fill('2026-03-29');
    await page.getByRole('button',{name:'Check configuration',exact:true}).click();
    await expect(page.getByRole('button',{name:'Submit backtest',exact:true})).toBeDisabled();
    await expect(page.locator('[id="error-time_range.start"]')).toBeVisible();
    await page.screenshot({path:resolve(evidence,'backtest-no-history.png'),fullPage:true});
  } finally { renameSync(pointer+'.s08-held',pointer); }
  writeFileSync(resolve(evidence,'routes.json'),JSON.stringify({
    acceptedSurface:'apps/platform-web', routes, dataNavigation:'disabled', settings,
    monitoring:{url:'/monitoring',surface:'apps/web integration template',summaryStatus:summary.status()},
    noHistoryApi:noHistory, noHistoryUI:{validationError:true,submitDisabled:true}, responses, errors,
    requiredHistoryProgressJourney:'separate existing Navigator consumer; see navigator.json',
    requiredEventJournalJourney:'separate existing Navigator consumer; see navigator.json',
    proofBoundary:'Accepted platform no-history and integration monitoring; legacy settings missing assets recorded separately; Navigator consumer proof separate',
  },null,2));
});

test('S08 Navigator existing history progress and stream journal on real PG', async ({ page }) => {
  const {execFileSync} = await import('node:child_process');
  await page.goto('/backtests');
  const credentials=JSON.parse(readFileSync(resolve(root,'.local_artifacts/s08-browser/credentials.json'),'utf8'));
  const form=page.locator('[data-password-login]');
  await form.locator('..').locator('summary').click();
  await form.locator('[name=username]').fill(credentials.username);
  await form.locator('[name=password]').fill(credentials.password);
  await form.locator('button[type=submit]').click();
  await expect(page.locator('[data-platform-client]')).toBeVisible();
  await page.goto('http://localhost:18686/data?market=1&symbol=BTCUSDT');
  await expect(page.getByRole('heading',{name:'BTCUSDT',exact:true})).toBeVisible();
  await page.getByRole('button',{name:'Show log',exact:true}).click();
  await page.getByRole('button',{name:'Stream events',exact:true}).click();
  const journal=page.getByRole('region',{name:'Stream event journal'});
  await expect(journal.getByText('Connection established',{exact:true})).toBeVisible();
  await page.screenshot({path:resolve(evidence,'navigator-stream-events.png'),fullPage:true});
  await page.goto('http://localhost:18686/data?market=1&symbol=BTCUSDT');
  await expect(page.getByRole('heading',{name:'BTCUSDT',exact:true})).toBeVisible();
  const manual=page.getByRole('button',{name:'Enter period manually'});
  if(await manual.isVisible()) await manual.click();
  await page.locator('.data-custom-period > summary').click();
  const range=JSON.parse(execFileSync(resolve(root,'.venv/bin/python'),[resolve(root,'.codex/agents/generated/postgresql-only-storage-v1/reports/S08/prove_ui_work.py'),'range'],{cwd:root,encoding:'utf8'}));
  await page.getByLabel('From · UTC',{exact:true}).fill(range.start);
  await page.getByLabel('To · UTC',{exact:true}).fill(range.end);
  await page.getByRole('button',{name:'Fill missing history',exact:true}).click();
  const request=page.getByRole('region',{name:'Download request'});
  await expect(request).toBeVisible();
  await expect(request.getByText('Queued',{exact:true})).toBeVisible();
  await page.screenshot({path:resolve(evidence,'navigator-queued.png'),fullPage:true});
  const tick=()=>JSON.parse(execFileSync(resolve(root,'.venv/bin/python'),[resolve(root,'.codex/agents/generated/postgresql-only-storage-v1/reports/S08/prove_ui_work.py'),'tick'],{cwd:root,encoding:'utf8'}).trim());
  const progress=tick();
  expect(progress.canonical_count).toBe(2);
  expect(progress.job.slice(1)).toEqual([2,6,2,2]);
  await expect(request.getByRole('progressbar',{name:'Job progress'})).toHaveAttribute('value',/^33/,{timeout:15000});
  await page.screenshot({path:resolve(evidence,'navigator-progress.png'),fullPage:true});
  tick();
  const complete=tick();
  expect(complete.canonical_count).toBe(6);
  expect(complete.job).toEqual(['succeeded',6,6,6,6]);
  await expect(request.getByText(/Download completed up to/)).toBeVisible({timeout:15000});
  await expect(request.getByText('Job progress: 6 / 6 minutes',{exact:true})).toBeVisible();
  await page.screenshot({path:resolve(evidence,'navigator-complete.png'),fullPage:true});
  await page.getByRole('button',{name:'Show log',exact:true}).click();
  await expect(page.getByRole('region',{name:'Download log'}).locator('.data-job-state.succeeded').first()).toBeVisible();
  await page.screenshot({path:resolve(evidence,'navigator-download-log.png'),fullPage:true});
  await page.goto('http://localhost:18686/monitoring/postgresql');
  await expect(page.getByText('Operational health',{exact:true})).toBeVisible();
  await page.screenshot({path:resolve(evidence,'navigator-monitoring.png'),fullPage:true});
  const apiPids=execFileSync('pgrep',['-f','^.*python3 -m uvicorn tools.qa.backtests_client_fixture:create_api --factory --host 127.0.0.1 --port 18681 --no-access-log$'],{encoding:'utf8'}).trim().split(/\s+/);
  expect(apiPids).toHaveLength(1);
  const apiPid=Number(apiPids[0]);
  process.kill(apiPid,'SIGSTOP');
  try {
    await page.getByRole('button',{name:'Refresh',exact:true}).first().click();
    await expect(page.getByRole('alert').first()).toBeVisible({timeout:20000});
    await expect(page.getByText('Operational health',{exact:true})).toBeVisible();
    await page.screenshot({path:resolve(evidence,'navigator-offline.png'),fullPage:true});
  } finally { process.kill(apiPid,'SIGCONT'); }
  await page.getByRole('button',{name:'Refresh',exact:true}).first().click();
  await expect(page.getByRole('alert')).toHaveCount(0);
  writeFileSync(resolve(evidence,'navigator.json'),JSON.stringify({surface:'apps/navigator-web existing candidate consumer',provider:'synthetic closed candles',storage:'production PostgreSQL adapters and jobs',range,progress,complete,streamEvent:'production connection transition trigger, real API and rendered journal',monitoring:'real observer TCP reachability only; Unknown / readiness not confirmed; no SQL health metric claim',offline:'owned fixture API SIGSTOP: HTTP timeout, explicit error, retained snapshot; SIGCONT recovery passed',promotedToAccepted:false},null,2));
});
