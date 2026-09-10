import { test, expect, chromium, type Page } from '@playwright/test';
import AxeBuilder from '@axe-core/playwright';
import { readFileSync, writeFileSync, mkdirSync, mkdtempSync, rmSync } from 'node:fs';
import { resolve } from 'node:path';
import { tmpdir } from 'node:os';
import { spawn, execFileSync } from 'node:child_process';
const root = resolve(import.meta.dirname, '../../..');
const evidence = resolve(root, `.codex/delivery/evidence/roehub-backtests-client-v1/browser/${process.env.ROEHUB_PROOF_STAGE === 'S6' ? 'S6-execution-regression' : process.env.ROEHUB_PROOF_STAGE==='S5'?'S5-execution-regression':'S4'}`);
const terminal = (state: string) => ['succeeded', 'failed', 'cancelled'].includes(state);
let completed: any;
async function signIn(page: Page, path = '/backtests/new') {
  await page.goto(path);
  const c = JSON.parse(readFileSync(resolve(root, '.local_artifacts/backtests-client/credentials.json'), 'utf8'));
  const form = page.locator('[data-password-login]'); await form.locator('..').locator('summary').click();
  await form.locator('[name=username]').fill(c.username); await form.locator('[name=password]').fill(c.password); await form.locator('button[type=submit]').click();
  await expect(page.locator('[data-platform-client]')).toBeVisible();
}
async function create(page: Page, label: string) {
  await page.getByLabel('Job label (optional)').fill(label);
  await page.getByLabel('Timeframe', { exact: true }).selectOption('15m');
  await page.getByLabel('Start date', { exact: true }).fill('2026-03-26');
  await page.getByLabel('End date', { exact: true }).fill('2026-03-29');
  await page.getByLabel('Indicator', { exact: true }).selectOption('ma.ema');
  await page.getByRole('button', { name: 'Check configuration', exact: true }).click();
  await expect(page.getByRole('button', { name: 'Submit backtest', exact: true })).toBeEnabled();
  const response = page.waitForResponse(r => new URL(r.url()).pathname === '/api/backtests/jobs' && r.request().method() === 'POST');
  await page.getByRole('button', { name: 'Submit backtest', exact: true }).click();
  const r = await response; expect(r.status()).toBe(201); return r.json();
}
const detail = (page: Page) => page.locator('.job-detail');

test('real worker lifecycle and real cancellation, reload and history', async ({ page }) => {
  test.setTimeout(240_000); mkdirSync(evidence, { recursive: true });
  const reads: { jobId: string; state: string; processed: number; total: number }[] = [];
  const errors: string[] = []; const http: { path: string; status: number }[] = [];
  page.on('pageerror', () => errors.push('pageerror'));
  page.on('console', m => { if (m.type() === 'error') errors.push(m.text().replace(/http[^ ]+/g, '[URL]').slice(0,200)); });
  page.on('response', async r => {
    if (r.status() >= 400) http.push({ path: new URL(r.url()).pathname, status: r.status() });
    if (/\/api\/backtests\/jobs\/[\da-f-]+$/.test(new URL(r.url()).pathname) && r.ok()) {
      const j = await r.json().catch(() => null); if (j) reads.push({ jobId: j.job_id, state: j.state, processed: j.progress.processed_units, total: j.progress.total_units });
    }
  });
  await signIn(page); await page.waitForLoadState('networkidle');
  const loginHttp = [...http], loginErrors = [...errors];
  expect(loginHttp.every(item => item.path === '/api/auth/oidc/status' && item.status === 404)).toBe(true);
  expect(loginErrors.every(item => item.includes('status of 404'))).toBe(true);
  http.length = 0; errors.length = 0;
  // Read-only high-frequency DB observation captures the short worker running
  // interval without bypassing the browser/API's 2s/5s refresh hints.
  const observedStates: string[] = [];
  const observer = spawn(resolve(root, '.venv/bin/python'), ['-u', '-c', `
import json,time
from pathlib import Path
import psycopg
private=json.loads(Path('.local_artifacts/backtests-client/credentials.json').read_text())
with psycopg.connect(private['dsn'],autocommit=True) as db:
    last=None
    for _ in range(1200):
        rows=db.execute("SELECT state FROM backtest_jobs ORDER BY created_at DESC LIMIT 1").fetchall()
        if rows and rows[0][0]!=last:
            last=rows[0][0];print(last,flush=True)
            if last in ('succeeded','failed','cancelled'):break
        time.sleep(.01)
`], {cwd:root,stdio:['ignore','pipe','ignore']});
  observer.stdout.on('data', data => observedStates.push(...String(data).trim().split('\n')));
  const first = await create(page, 'S4 real completion');
  await expect(detail(page).getByRole('heading', { name: 'S4 real completion' })).toBeVisible();
  await page.screenshot({ path: resolve(evidence, 'real-active.png'), fullPage: true });
  await expect(detail(page).locator('.job-status').getByText('Completed', { exact: true })).toBeVisible({ timeout: 180_000 });
  completed = await (await page.request.get(`/api/backtests/jobs/${first.job_id}`)).json();
  expect(completed.state).toBe('succeeded'); expect(completed.terminal_summary.top_variants_count).toBeGreaterThan(0);
  await page.screenshot({ path: resolve(evidence, 'real-completed.png'), fullPage: true });
  await page.reload(); await expect(detail(page).locator('.job-status').getByText('Completed', { exact: true })).toBeVisible();
  await page.getByRole('link', { name: 'Backtests', exact: true }).click(); await page.goBack();
  await expect(detail(page).getByRole('heading', { name: 'S4 real completion' })).toBeVisible(); await page.goForward();
  await expect(page).toHaveURL(/\/backtests$/);
  // The first job is actually terminal, releasing admission before the second create.
  await page.getByRole('link', { name: 'New backtest', exact: true }).click();
  execFileSync(resolve(root,'.venv/bin/python'),['-m','tools.qa.backtests_client_fixture','pause-runner'],{cwd:root});
  const second = await create(page, 'S4 real cancellation');
  const cancel = detail(page).getByRole('button', { name: 'Cancel backtest', exact: true }); await expect(cancel).toBeEnabled();
  let posts = 0; page.on('request', r => { if (r.url().endsWith('/cancel') && r.method() === 'POST') posts++; });
  await cancel.click(); await page.keyboard.press('Escape'); await expect(cancel).toBeFocused(); expect(posts).toBe(0);
  await cancel.click(); const response = page.waitForResponse(r => r.url().endsWith('/cancel'));
  await page.getByRole('button', { name: 'Confirm cancellation', exact: true }).click();
  const reply = await response; expect(reply.status()).toBe(200); const accepted = await reply.json();
  execFileSync(resolve(root,'.venv/bin/python'),['-m','tools.qa.backtests_client_fixture','resume-runner'],{cwd:root});
  await expect.poll(async () => (await (await page.request.get(`/api/backtests/jobs/${second.job_id}`)).json()).state, { timeout: 120_000, intervals: [2000] }).toBe('cancelled');
  await expect(detail(page).getByText('The server confirms cancellation.', { exact: true })).toBeVisible({ timeout: 20_000 });
  const cancelled = await (await page.request.get(`/api/backtests/jobs/${second.job_id}`)).json();
  await page.screenshot({ path: resolve(evidence, 'real-cancelled.png'), fullPage: true }); expect(posts).toBe(1);
  const again = await page.request.post(`/api/backtests/jobs/${second.job_id}/cancel`); expect(again.status()).toBe(200); expect((await again.json()).state).toBe('cancelled');
  observer.kill(); expect(observedStates).toContain('running'); expect(first.state).toBe('queued');
  writeFileSync(resolve(evidence, 'real-lifecycle.json'), JSON.stringify({ first: { jobId: first.job_id, initial: first.state, terminal: completed.state, variants: completed.terminal_summary.top_variants_count }, second: { jobId: second.job_id, initial: second.state, cancelResponse: accepted.state, cancelRequested: !!accepted.cancel_requested_at, terminal: cancelled.state }, reads, observedWorkerStates: observedStates, queuedSchedulerPaused:true, uiCancelPosts: posts, terminalCancelStatus: again.status(), reloadAndHistory: true, errors, http, loginHttp, loginErrors }, null, 2));
  expect(errors).toEqual([]); expect(http).toEqual([]);
});

const controlledId = '20000000-0000-4000-8000-000000000001';
function controlledJob(state = 'running', id = controlledId) {
  const date = new Date().toISOString();
  return { job_id:id,state,created_at:date,updated_at:date,generated_at:date,refresh_status:terminal(state)?'terminal':'poll',next_allowed_refresh_at:date,retry_after_seconds:0,cancel_requested_at:null,
    progress:{percent:50,processed_units:5,total_units:10,updated_at:date,pipeline_stage:'simulation'},terminal_summary: terminal(state)?{top_variants_count:3}:{},
    request:{coordinates:{exchange:'binance',market_type:'spot',symbol:'BTCUSDT'},timeframe:'15m',time_range:{start:date,end:date},risk_mode:'none',ui_metadata:{strategy_name:'Controlled S4 job'}} };
}
async function controlled(page: Page, get: ()=>any = ()=>controlledJob()) {
  await signIn(page, '/backtests');
  await page.route(`**/api/backtests/jobs/${controlledId}`, route=>route.fulfill({status:200,json:get()}));
  await page.goto(`/backtests/${controlledId}`);
  await expect(detail(page).getByRole('heading', {name:'Controlled S4 job'})).toBeVisible();
}
async function confirm(page:Page) {
  await detail(page).getByRole('button',{name:'Cancel backtest',exact:true}).click();
  await page.getByRole('button',{name:'Confirm cancellation',exact:true}).click();
}

test('controlled pending cancellation and terminal completion wins a late command response',async({page})=>{
  mkdirSync(evidence,{recursive:true});let current=controlledJob();await controlled(page,()=>current);
  let release!:()=>void;const gate=new Promise<void>(r=>{release=r;});let posts=0;
  await page.route(`**/api/backtests/jobs/${controlledId}/cancel`,async route=>{posts++;await gate;await route.fulfill({status:200,json:{...controlledJob(),cancel_requested_at:new Date().toISOString()}});});
  await confirm(page);await expect(detail(page).getByText('Sending cancellation request…')).toBeVisible();
  await expect(detail(page).getByRole('button',{name:'Cancel backtest',exact:true})).toBeDisabled();
  current=controlledJob('succeeded');
  await expect(detail(page).locator('.job-status').getByText('Completed',{exact:true})).toBeVisible({timeout:10000});
  release();await expect(detail(page).getByText('Result variants: 3',{exact:true})).toBeVisible();
  await page.waitForTimeout(500);expect(posts).toBe(1);await expect(detail(page).locator('.job-status').getByText('Completed',{exact:true})).toBeVisible();
  await page.screenshot({path:resolve(evidence,'controlled-terminal-race.png'),fullPage:true});
  writeFileSync(resolve(evidence,'controlled-race.json'),JSON.stringify({controlled:true,posts,terminal:'succeeded',variants:3,latePendingIgnored:true}));
});

test('controlled cancel409 then status429 obeys the GET cooldown and preserves eligibility',async({page})=>{
  test.setTimeout(90_000);let phase='initial';let reads=0;let posts=0;const times:number[]=[];
  await controlled(page);
  await page.unroute(`**/api/backtests/jobs/${controlledId}`);
  await page.route(`**/api/backtests/jobs/${controlledId}`,async route=>{
    reads++;times.push(Date.now());
    if(phase==='limited'){phase='ready';await route.fulfill({status:429,headers:{'Retry-After':'60'},json:{error:{code:'controlled_rate_limit'}}});}
    else await route.fulfill({status:200,json:controlledJob()});
  });
  await page.route(`**/api/backtests/jobs/${controlledId}/cancel`,route=>{posts++;phase='limited';return route.fulfill({status:409,json:{error:{code:'backtest.job_not_cancellable'}}});});
  await confirm(page);await expect.poll(()=>reads).toBe(1);await expect(detail(page).getByText(/Too many requests/)).toBeVisible();
  await page.waitForTimeout(3000);expect(reads).toBe(1);expect(posts).toBe(1);
  await page.waitForTimeout(55_000);expect(reads).toBe(1);
  await expect.poll(()=>reads,{timeout:5000}).toBe(2);
  await expect(detail(page).getByRole('button',{name:'Cancel backtest',exact:true})).toBeEnabled();
  writeFileSync(resolve(evidence,'controlled-cooldown.json'),JSON.stringify({controlled:true,cancelStatus:409,status429RetryAfter:60,reads,posts,noReadsBeforeDeadline:true},null,2));
});

test('controlled lost cancel response reconciles by GET and reload never repeats the command',async({page})=>{
  let current=controlledJob();await controlled(page,()=>current);let posts=0;
  await page.route(`**/api/backtests/jobs/${controlledId}/cancel`,async route=>{posts++;current={...controlledJob(),cancel_requested_at:new Date().toISOString()} as any;await route.abort('failed');});
  await confirm(page);await expect(detail(page).getByText(/Cancellation outcome is pending/)).toBeVisible();
  await expect(detail(page).getByText(/Cancellation requested/)).toBeVisible({timeout:10000});
  await page.reload();await expect(detail(page).getByText(/Cancellation requested/)).toBeVisible();
  await expect(detail(page).getByRole('button',{name:'Cancel backtest',exact:true})).toBeDisabled();expect(posts).toBe(1);
  current=controlledJob('cancelled');await expect(detail(page).getByText('The server confirms cancellation.',{exact:true})).toBeVisible({timeout:10000});
  await page.screenshot({path:resolve(evidence,'controlled-transport-reload.png'),fullPage:true});
});

test('controlled read403 and cancel403 stop protected job polling; cancel429 requires an explicit command',async({page})=>{
  await controlled(page);let reads=0;let posts=0;
  await page.route(`**/api/backtests/jobs/${controlledId}/cancel`,route=>{posts++;return route.fulfill({status:429,headers:{'Retry-After':'2'},json:{error:{code:'limited'}}});});
  await confirm(page);await expect(detail(page).getByText(/Too many requests/)).toBeVisible();
  await expect(detail(page).getByRole('button',{name:'Cancel backtest',exact:true})).toBeEnabled({timeout:10000});expect(posts).toBe(1);
  await page.unroute(`**/api/backtests/jobs/${controlledId}/cancel`);
  await page.route(`**/api/backtests/jobs/${controlledId}/cancel`,route=>{posts++;return route.fulfill({status:403,json:{error:{code:'forbidden'}}});});
  await confirm(page);await expect(detail(page).getByText(/You do not have access/)).toBeVisible();
  page.on('request',r=>{if(new URL(r.url()).pathname===`/api/backtests/jobs/${controlledId}`)reads++;});
  await page.waitForTimeout(5500);expect(reads).toBe(0);expect(posts).toBe(2);
  await page.reload();await expect(detail(page).getByRole('button',{name:'Cancel backtest',exact:true})).toBeEnabled();
  await page.unroute(`**/api/backtests/jobs/${controlledId}`);
  await page.route(`**/api/backtests/jobs/${controlledId}`,route=>route.fulfill({status:403,json:{error:{code:'forbidden'}}}));
  await expect(detail(page).getByText(/You do not have access/)).toBeVisible({timeout:10000});
  const stopped=reads;await page.waitForTimeout(5000);expect(reads).toBe(stopped);
});

test('controlled stale measurement, dialog keyboard, terminal during dialog, late job read and responsive RU EN',async({page})=>{
  let current=controlledJob();current.progress.updated_at='2020-01-01T00:00:00Z';await controlled(page,()=>current);
  await expect(detail(page).getByText('Progress measurement is stale. No progress is inferred.')).toBeVisible();
  await expect(detail(page).getByRole('progressbar')).toHaveAttribute('value','50');
  const cancel=detail(page).getByRole('button',{name:'Cancel backtest',exact:true});await cancel.click();
  await expect(page.getByRole('button',{name:'Keep running',exact:true})).toBeFocused();
  await page.keyboard.press('Shift+Tab');await expect(page.getByRole('button',{name:'Confirm cancellation',exact:true})).toBeFocused();
  await page.keyboard.press('Tab');await expect(page.getByRole('button',{name:'Keep running',exact:true})).toBeFocused();
  await page.keyboard.press('Escape');await expect(cancel).toBeFocused();
  await cancel.click();current=controlledJob('failed');await expect(page.getByRole('dialog')).not.toBeVisible({timeout:10000});
  await expect(detail(page).getByRole('status')).toBeFocused();await expect(detail(page).getByText('The server reports failure.',{exact:true})).toBeVisible();
  const checks=[];
  for(const locale of ['en','ru']){
    if(locale==='ru'){await page.getByRole('link',{name:'Русский',exact:true}).click();await expect(detail(page).getByText('Сервер сообщает об ошибке.',{exact:true})).toBeVisible();}
    for(const width of [820,1024,1440]){
      await page.setViewportSize({width,height:1000});await detail(page).scrollIntoViewIfNeeded();
      const axe=await new AxeBuilder({page}).analyze();expect(axe.violations).toEqual([]);
      const overflow=await page.evaluate(()=>document.documentElement.scrollWidth>innerWidth);expect(overflow).toBe(false);
      await page.screenshot({path:resolve(evidence,`${locale}-${width}.png`),fullPage:true});checks.push({locale,width,axe:0,overflow});
    }
  }
  writeFileSync(resolve(evidence,'controlled-visual.json'),JSON.stringify({controlled:true,checks,keyboardTrap:true,dismissRestored:true,terminalClosedDialog:true},null,2));
  // Return to a running job, delay its next read, navigate away and reject old content.
  current=controlledJob();await page.reload();await expect(detail(page).getByRole('button',{name:'Отменить бэктест',exact:true})).toBeVisible();
  let release!:()=>void;const gate=new Promise<void>(r=>{release=r;});let started=false;
  await page.unroute(`**/api/backtests/jobs/${controlledId}`);
  await page.route(`**/api/backtests/jobs/${controlledId}`,async route=>{started=true;await gate;await route.fulfill({status:200,json:{...controlledJob('succeeded'),request:{...current.request,ui_metadata:{strategy_name:'Late stale job'}}}}).catch(()=>{});});
  await expect.poll(()=>started).toBe(true);
  await detail(page).getByRole('link',{name:'К библиотеке',exact:true}).click();release();await page.waitForTimeout(500);
  expect(await page.getByText('Late stale job',{exact:true}).count()).toBe(0);
});

test('controlled network failure retains stale measurement and identity outage blocks the cancel gate without logout',async({page})=>{
  await controlled(page);let posts=0;page.on('request',r=>{if(r.method()==='POST')posts++;});
  await page.unroute(`**/api/backtests/jobs/${controlledId}`);
  await page.route(`**/api/backtests/jobs/${controlledId}`,route=>route.abort('failed'));
  await expect(detail(page).getByText(/Stale snapshot/)).toBeVisible({timeout:10000});
  await expect(detail(page).getByRole('progressbar')).toHaveAttribute('value','50');
  await expect(detail(page).getByRole('button',{name:'Cancel backtest',exact:true})).toBeDisabled();
  await page.unroute(`**/api/backtests/jobs/${controlledId}`);
  await page.route(`**/api/backtests/jobs/${controlledId}`,route=>route.fulfill({status:200,json:controlledJob()}));
  await expect(detail(page).getByRole('button',{name:'Cancel backtest',exact:true})).toBeEnabled({timeout:10000});
  await page.route('**/api/auth/current-user',route=>route.fulfill({status:503,json:{error:{code:'identity_unavailable'}}}));
  await confirm(page);await expect(page.getByText('Session service is unavailable. Reload this page to check again.',{exact:true})).toBeVisible();
  await expect(page.getByRole('link',{name:'Sign in',exact:true})).toHaveCount(0);expect(posts).toBe(0);
  let requests=0;page.on('request',r=>{if(new URL(r.url()).pathname.startsWith('/api/'))requests++;});
  await page.waitForTimeout(5500);expect(requests).toBe(0);
  await page.screenshot({path:resolve(evidence,'identity-outage.png'),fullPage:true});
});

test('real expired session at cancel gate removes private UI and stops all protected requests',async({page})=>{
  test.setTimeout(50000);await controlled(page);let posts=0;page.on('request',r=>{if(r.method()==='POST')posts++;});
  execFileSync(resolve(root,'.venv/bin/python'),['-m','tools.qa.backtests_client_fixture','expire'],{cwd:root});
  const response=page.waitForResponse(r=>new URL(r.url()).pathname==='/api/auth/current-user'&&r.status()===401);
  await confirm(page);await response;await expect(page.getByRole('link',{name:'Sign in',exact:true})).toBeVisible();
  await expect(page.locator('[data-platform-client]')).toHaveCount(0);expect(posts).toBe(0);
  let requests=0;page.on('request',r=>{if(new URL(r.url()).pathname.startsWith('/api/'))requests++;});
  await page.waitForTimeout(31000);expect(requests).toBe(0);
  await page.screenshot({path:resolve(evidence,'session-expired.png'),fullPage:true});
  writeFileSync(resolve(evidence,'session-expiry.json'),JSON.stringify({realSessionStatus:401,controlledJob:true,cancelPosts:0,privateUiRemoved:true,quietMs:31000,protectedRequests:requests},null,2));
});

test('native 200 percent zoom keeps RU EN execution and cancel dialog usable',async()=>{
  const directory=mkdtempSync(resolve(tmpdir(),'roehub-s4-zoom-'));const extension=resolve(directory,'extension');mkdirSync(extension);
  writeFileSync(resolve(extension,'manifest.json'),JSON.stringify({manifest_version:3,name:'Local zoom proof',version:'1.0',permissions:['tabs'],background:{service_worker:'worker.js'}}));
  writeFileSync(resolve(extension,'worker.js'),'chrome.runtime.onInstalled.addListener(() => {});');
  const context=await chromium.launchPersistentContext(resolve(directory,'profile'),{baseURL:'http://localhost:18480',channel:'chromium',headless:true,viewport:null,args:[`--disable-extensions-except=${extension}`,`--load-extension=${extension}`,'--window-size=1440,1100']});
  try{
    const page=await context.newPage();await controlled(page);const worker=context.serviceWorkers()[0]??await context.waitForEvent('serviceworker');
    await worker.evaluate("chrome.tabs.query({url:'http://localhost:18480/*'}).then(tabs=>Promise.all(tabs.map(tab=>chrome.tabs.setZoom(tab.id,2))))");
    const zoom=await worker.evaluate("chrome.tabs.query({url:'http://localhost:18480/*'}).then(tabs=>chrome.tabs.getZoom(tabs[0].id))");expect(zoom).toBe(2);
    const sizes=[];const cdp=await context.newCDPSession(page);
    for(const locale of ['en','ru']){
      if(locale==='ru')await page.getByRole('link',{name:'Русский',exact:true}).click();
      const cancel=detail(page).getByRole('button',{name:locale==='en'?'Cancel backtest':'Отменить бэктест',exact:true});await cancel.scrollIntoViewIfNeeded();await cancel.focus();await expect(cancel).toBeFocused();
      const dimensions=await page.evaluate(()=>({innerWidth,outerWidth,devicePixelRatio,scrollWidth:document.documentElement.scrollWidth}));expect(dimensions.scrollWidth).toBeLessThanOrEqual(dimensions.innerWidth);
      expect((await new AxeBuilder({page}).analyze()).violations).toEqual([]);
      let shot=await cdp.send('Page.captureScreenshot',{format:'png',fromSurface:true});writeFileSync(resolve(evidence,`${locale}-native-zoom-200.png`),Buffer.from(shot.data,'base64'));
      await cancel.click();await expect(page.getByRole('dialog')).toBeVisible();expect((await new AxeBuilder({page}).analyze()).violations).toEqual([]);
      const confirm=page.getByRole('button',{name:locale==='en'?'Confirm cancellation':'Подтвердить отмену',exact:true});await confirm.focus();await expect(confirm).toBeFocused();const box=await confirm.boundingBox();expect(box!.x).toBeGreaterThanOrEqual(0);expect(box!.x+box!.width).toBeLessThanOrEqual(dimensions.innerWidth);
      shot=await cdp.send('Page.captureScreenshot',{format:'png',fromSurface:true});writeFileSync(resolve(evidence,`${locale}-native-zoom-200-dialog.png`),Buffer.from(shot.data,'base64'));
      await page.keyboard.press('Escape');await expect(cancel).toBeFocused();sizes.push({locale,...dimensions});
    }
    writeFileSync(resolve(evidence,'zoom.json'),JSON.stringify({controlled:true,nativeZoom:zoom,sizes,axe:0,dialogAndCancelReachable:true},null,2));
  }finally{await context.close();rmSync(directory,{recursive:true,force:true});}
});
