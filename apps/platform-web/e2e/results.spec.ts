import { test, expect, chromium, type Page } from '@playwright/test';
import AxeBuilder from '@axe-core/playwright';
import { readFileSync, writeFileSync, mkdirSync, mkdtempSync, rmSync } from 'node:fs';
import { resolve } from 'node:path';
import { tmpdir } from 'node:os';
import { execFileSync } from 'node:child_process';
const root=resolve(import.meta.dirname,'../../..');
const evidence=resolve(root,`.codex/delivery/evidence/roehub-backtests-client-v1/browser/${process.env.ROEHUB_PROOF_STAGE==='S6'?'S6-results-regression':'S5'}`);
let realJob:string,realVariant:string;
async function closePanels(page:Page){
 for(const selector of ['details.report-actions[open]','details.job-information[open]']){
  const panel=page.locator(selector);if(await panel.count())await panel.locator('summary').click();
 }
}
async function openActions(page:Page){
 const info=page.locator('details.job-information[open]');if(await info.count())await info.locator('summary').click();
 const disclosure=page.locator('details.report-actions');
 await expect(disclosure).toBeAttached();
 if(await disclosure.getAttribute('open')===null)await disclosure.locator('summary').click();
}
async function openJobInfo(page:Page){
 const actions=page.locator('details.report-actions[open]');if(await actions.count())await actions.locator('summary').click();
 const disclosure=page.locator('details.job-information');
 await expect(disclosure).toBeAttached();
 if(await disclosure.getAttribute('open')===null)await disclosure.locator('summary').click();
}
async function chooseVariant(page:Page,rank:number){
 await closePanels(page);await page.getByRole('region',{name:'Ranked variants',exact:true}).getByRole('link',{name:`Variant ${rank}`,exact:true}).click();
 await openActions(page);
}

async function signIn(page:Page,path='/backtests/new'){
 await page.goto(path);const c=JSON.parse(readFileSync(resolve(root,'.local_artifacts/backtests-client/credentials.json'),'utf8'));
 const form=page.locator('[data-password-login]');await form.locator('..').locator('summary').click();await form.locator('[name=username]').fill(c.username);await form.locator('[name=password]').fill(c.password);await form.locator('button[type=submit]').click();await expect(page.locator('[data-platform-client]')).toBeVisible();
}
async function create(page:Page){
 await page.getByLabel('Job label (optional)').fill('S5 real results');await page.getByLabel('Timeframe',{exact:true}).selectOption('15m');await page.getByLabel('Start date',{exact:true}).fill('2026-03-26');await page.getByLabel('End date',{exact:true}).fill('2026-03-29');await page.getByLabel('Indicator',{exact:true}).selectOption('ma.ema');await page.getByRole('button',{name:'Check configuration',exact:true}).click();await expect(page.getByRole('button',{name:'Submit backtest',exact:true})).toBeEnabled();const response=page.waitForResponse(r=>new URL(r.url()).pathname==='/api/backtests/jobs'&&r.request().method()==='POST');await page.getByRole('button',{name:'Submit backtest',exact:true}).click();return (await response).json();
}
test('real results materialization CSV provenance dedupe strategy SSR and terminal deletion',async({page})=>{
 test.setTimeout(240000);mkdirSync(evidence,{recursive:true});const network:{path:string;method:string;status:number}[]=[];let starts=0;const errors:string[]=[];const consoleErrors:string[]=[];const httpErrors:{path:string;status:number}[]=[];
 page.on('pageerror',error=>errors.push(`${new URL(page.url()).pathname}: ${error.name}: ${error.message} ${error.stack || "[no stack]"}`.replace(/https?:\/\/[^/ ]+/g,'[origin]')));page.on('console',m=>{if(m.type()==='error')consoleErrors.push(m.text().replace(/http[^ ]+/g,'[URL]').slice(0,200));});page.on('response',r=>{if(r.status()>=400)httpErrors.push({path:new URL(r.url()).pathname,status:r.status()});});page.on('request',r=>{if(/\/(run|start|runs)(\/|$)/.test(new URL(r.url()).pathname)&&r.method()==='POST')starts++;});page.on('response',r=>{if(r.url().includes('/api/backtests/'))network.push({path:new URL(r.url()).pathname,method:r.request().method(),status:r.status()});});
 await signIn(page);const job=await create(page);realJob=job.job_id;await expect(page.locator('.job-status').getByText('Completed',{exact:true})).toBeVisible({timeout:180000});
 await expect(page.getByRole('img',{name:'Equity',exact:true})).toBeVisible({timeout:90000});realVariant=new URL(page.url()).searchParams.get('variant')!;expect(realVariant).toBeTruthy();
 const base=`/api/backtests/jobs/${realJob}/variants/${encodeURIComponent(realVariant)}`;
 await page.getByText('Show accessible data table',{exact:true}).click();await expect(page.getByRole('region',{name:'Equity',exact:true}).locator('tbody tr')).not.toHaveCount(0);
 const counts:Record<string,number>={};
 for(const tab of ['drawdown','monthly-stats','trades']){
  await closePanels(page);
  if(tab==='drawdown')await page.getByRole('button',{name:'Drawdown · %',exact:true}).click();
  else await page.getByRole('tab',{name:tab==='monthly-stats'?'Monthly statistics':'Trades',exact:true}).click();
  const r=await page.request.get(`${base}/${tab}`);expect(r.status()).toBe(200);
  const data=await r.json();counts[tab]=(data.items??data.points).length;expect(counts[tab]).toBeGreaterThan(0);
  if(tab==='drawdown')await expect(page.getByRole('img',{name:'Drawdown · %',exact:true})).toBeVisible();
  else await expect(page.getByRole('tabpanel').locator('tbody tr')).not.toHaveCount(0);
 }
 await expect(page.getByRole('tab',{name:'Symbol statistics',exact:true})).toHaveCount(0);
 await openActions(page);
 await page.getByLabel('CSV maximum rows',{exact:true}).fill('1');const download=page.waitForEvent('download');await page.getByRole('button',{name:'Download CSV',exact:true}).click();const file=await download;const csv=readFileSync((await file.path())!,'utf8');expect(csv).toContain('trade_index,entry_timestamp');expect(csv.trim().split('\n')).toHaveLength(2);await expect(page.getByText('CSV downloaded: 1 of 1 rows (limit 1).',{exact:false})).toBeVisible();
 const materialized=await page.request.post(`${base}/trades`);expect(materialized.status()).toBe(200);const postDetail=await materialized.json();expect(postDetail.job_id).toBe(realJob);expect(postDetail.variant_key).toBe(realVariant);expect(Array.isArray(postDetail.trades)).toBe(true);
 const ready=await (await page.request.get(`${base}/compatibility-readiness`)).json();expect(ready.compatibility_state).toBe('not_launchable');expect(ready.compatibility_reason_codes).toContain('unsupported_live_evaluator');
 const save=page.getByRole('button',{name:'Save strategy',exact:true});await expect(save).toBeEnabled();await save.click();await expect(page.getByRole('button',{name:'Go back',exact:true})).toBeFocused();await page.keyboard.press('Escape');await expect(save).toBeFocused();await save.click();const reply=page.waitForResponse(r=>r.url().endsWith('/strategies')&&r.request().method()==='POST');await page.getByRole('button',{name:'Confirm save',exact:true}).click();const response=await reply;expect(response.status()).toBe(201);expect(response.request().postData()).toBeNull();const saved=await response.json();const key=response.request().headers()['idempotency-key'];
 const replay=await page.request.post(`${base}/strategies`,{headers:{'Idempotency-Key':key!}});expect(replay.status()).toBe(200);const duplicate=await replay.json();expect(duplicate.strategy.strategy_id).toBe(saved.strategy.strategy_id);expect(duplicate.duplicate_reason).toBe('idempotent_replay');
 const provenance=await page.request.post(`${base}/strategies`,{headers:{'Idempotency-Key':crypto.randomUUID()}});expect(provenance.status()).toBe(200);const dedupe=await provenance.json();expect(dedupe.strategy.strategy_id).toBe(saved.strategy.strategy_id);expect(dedupe.duplicate_reason).toBe('source_variant_exists');
 const strategy=await page.request.get(`/api/strategies/${saved.strategy.strategy_id}`);expect(strategy.status()).toBe(200);await page.getByRole('link',{name:'Open strategy',exact:true}).click();await expect(page).toHaveURL(new RegExp(`/strategies/${saved.strategy.strategy_id}$`));await expect(page.locator('[data-strategies-root]')).toHaveAttribute('data-initial-strategy-id',saved.strategy.strategy_id);await expect(page.locator(`[data-saved-strategy-id="${saved.strategy.strategy_id}"]`)).toBeVisible();await page.goBack();await openActions(page);await openJobInfo(page);await expect(page.getByRole('button',{name:'Delete history',exact:true})).toBeEnabled();
 await closePanels(page);await page.getByRole('tab',{name:'Overview',exact:true}).click();await closePanels(page);await page.getByRole('button',{name:'Equity',exact:true}).click();await expect(page.getByRole('img',{name:'Equity',exact:true})).toBeVisible();await page.screenshot({path:resolve(evidence,'real-results.png'),fullPage:true});
 // The server actually dedupes the second UI save; only its response is dropped.
 let lostSavePosts=0,lostSaveAcceptedStatus=0;
 await page.route(`**${base}/strategies`,async route=>{lostSavePosts++;const accepted=await route.fetch();lostSaveAcceptedStatus=accepted.status();expect((await accepted.json()).strategy.strategy_id).toBe(saved.strategy.strategy_id);await route.abort('failed');});
 await confirmSave(page);await expect(page.getByText(/Save outcome is unknown/)).toBeVisible();expect(lostSaveAcceptedStatus).toBe(200);await page.reload();await openActions(page);await expect(page.getByText(/Save outcome is unknown/)).toBeVisible();await expect(page.getByRole('button',{name:'Save strategy',exact:true})).toBeDisabled();expect(lostSavePosts).toBe(1);await page.unroute(`**${base}/strategies`);
 // Keep source job for remaining cold-fixture tests; delete a separate genuinely cancelled history job.
 await page.goto('/backtests/new');execFileSync(resolve(root,'.venv/bin/python'),['-m','tools.qa.backtests_client_fixture','pause-runner'],{cwd:root});const second=await create(page);const conflict=await page.request.delete(`/api/backtests/jobs/${second.job_id}`);expect(conflict.status()).toBe(409);await page.request.post(`/api/backtests/jobs/${second.job_id}/cancel`);execFileSync(resolve(root,'.venv/bin/python'),['-m','tools.qa.backtests_client_fixture','resume-runner'],{cwd:root});await expect(page.getByText('The server confirms cancellation.',{exact:true})).toBeVisible({timeout:20000});
 await openJobInfo(page);await page.getByRole('button',{name:'Delete history',exact:true}).click();const deletion=page.waitForResponse(r=>r.request().method()==='DELETE');await page.getByRole('button',{name:'Confirm deletion',exact:true}).click();expect((await deletion).status()).toBe(204);await expect(page).toHaveURL(/\/backtests$/);await expect(page.getByText('Backtest history deleted.',{exact:true})).toBeVisible();expect((await page.request.get(`/api/backtests/jobs/${second.job_id}`)).status()).toBe(404);
 expect(starts).toBe(0);expect(errors).toEqual([]);expect(httpErrors.every(r=>r.path==='/api/auth/oidc/status'&&r.status===404||r.path===`/api/backtests/jobs/${second.job_id}`&&[409,404].includes(r.status))).toBe(true);expect(consoleErrors.every(e=>/status of (404|409)|net::ERR_FAILED/.test(e))).toBe(true);writeFileSync(resolve(evidence,'real-results.json'),JSON.stringify({jobId:realJob,variant:realVariant,counts,csvRows:1,csvTruncated:false,strategyId:saved.strategy.strategy_id,lostSavePosts,lostSaveAcceptedStatus,lostSaveReloadReplay:false,lazyTradesPostStatus:materialized.status(),lazyTradesPostInlineRows:postDetail.trades.length,liveCompatibility:ready.compatibility_state,saveStatus:response.status(),bodyless:response.request().postData()===null,replayStatus:replay.status(),duplicateReason:duplicate.duplicate_reason,provenanceReason:dedupe.duplicate_reason,strategyReadStatus:strategy.status(),ssrStrategy:true,deleteStatus:204,activeDeleteConflict:conflict.status(),starts,errors,consoleErrors,httpErrors,network},null,2));
});
const id='40000000-0000-4000-8000-000000000001';
const org='50000000-0000-4000-8000-000000000001';
function controlledJob(){const date=new Date().toISOString();return {job_id:id,state:'succeeded',created_at:date,updated_at:date,generated_at:date,refresh_status:'terminal',next_allowed_refresh_at:date,retry_after_seconds:0,cancel_requested_at:null,progress:{percent:100,processed_units:10,total_units:10,updated_at:date,pipeline_stage:'complete'},terminal_summary:{top_variants_count:2},request:{coordinates:{exchange:'binance',market_type:'spot',symbol:'BTCUSDT'},timeframe:'15m',time_range:{start:date,end:date},risk_mode:'none',ui_metadata:{strategy_name:'Controlled S5 results'}}};}
const variant=(v='v1')=>({rank:v==='v1'?1:2,variant_key:v,variant_hash:'hash',summary_metrics:{total_return_pct:v==='v1'?12.34:99.5,sharpe:1.2,max_drawdown_pct:3,profit_factor:1.5,win_rate_pct:60,trade_count:2},best_tp_pct:null,best_sl_pct:null});
const pending=(v='v1',status='queued')=>({job_id:id,variant_key:v,status,materialization:{status,retryable:false,retry_after_seconds:2},cache:{cache_path:'/private/materialization'},timing:{}});
const series=(v='v1',kind='equity')=>({job_id:id,variant_key:v,kind,points:[{x:'2026-03-26T00:00:00Z',value:100},{x:'2026-03-27T00:00:00Z',value:v==='v1'?120:200}],returned_points:2,source_points:2,downsampled:false,cache:{status:'hit'}});
async function controlled(page:Page){await signIn(page,'/backtests');await page.route(`**/api/backtests/jobs/${id}**`,async route=>{const path=new URL(route.request().url()).pathname;const v=path.includes('/v2')?'v2':'v1';
 if(path.endsWith('/summary'))return route.fulfill({json:{job:controlledJob(),top_variants:{items:[variant(),variant('v2')]},selected_variant_key:'v1',retry_after_seconds:0}});
 if(path.endsWith('/top'))return route.fulfill({json:{items:[variant(),variant('v2')]}});
 if(path.endsWith('/compatibility-readiness'))return route.fulfill({json:{source_job_id:id,source_variant_key:v,strategy_spec_hash:'hash',compatibility_state:'launchable',market_data_state:'unavailable',checked_at:new Date().toISOString()}});
 if(/\/(equity|drawdown)$/.test(path))return route.fulfill({json:series(v,path.endsWith('drawdown')?'drawdown':'equity')});
 if(path.endsWith('/trades'))return route.fulfill({json:{job_id:id,variant_key:v,items:[],pagination:{page:1,page_size:50,total:0,has_next:false,has_previous:false},cache:{status:'hit'}}});
 if(path.endsWith('-stats'))return route.fulfill({json:{job_id:id,variant_key:v,kind:path.endsWith('monthly-stats')?'monthly':'symbol',items:[],bounds:{truncated:false,returned_items:0,source_items:0},cache:{status:'hit'}}});
 if(path.includes('/variants/'))return route.fulfill({json:variant(v)});
 return route.fulfill({json:controlledJob()});});await page.goto(`/backtests/${id}?variant=v1`);await expect(page.getByRole('img',{name:'Equity',exact:true})).toBeVisible();await openActions(page);}
const selected=(page:Page)=>page.locator('[data-result-variant]');
async function confirmSave(page:Page){await openActions(page);await page.getByRole('button',{name:'Save strategy',exact:true}).click();await page.getByRole('button',{name:'Confirm save',exact:true}).click();}
test('controlled lazy202 GET429, degraded empty and failed detail; tabs keyboard and late variant reads',async({page})=>{
 await controlled(page);let reads=0;const times:number[]=[];
 await page.route(`**/api/backtests/jobs/${id}/variants/v1/drawdown*`,route=>{reads++;times.push(Date.now());return reads===1?route.fulfill({status:202,json:pending()}):reads===2?route.fulfill({status:429,headers:{'Retry-After':'3'},json:{error:{code:'limited'}}}):route.fulfill({json:{...series('v1','drawdown'),cache:{status:'degraded',warning:'private details'}}});});
 await closePanels(page);await page.getByRole('tab',{name:'Overview',exact:true}).focus();await page.keyboard.press('ArrowRight');await expect(page.getByRole('tab',{name:'Metrics',exact:true})).toBeFocused();await closePanels(page);await page.getByRole('tab',{name:'Overview',exact:true}).click();await closePanels(page);await page.getByRole('button',{name:'Drawdown · %',exact:true}).click();await expect(page.getByText('Result detail is materializing. Reads follow the server delay.')).toBeVisible();await expect(page.getByText(/Too many requests/)).toBeVisible({timeout:10000});await expect(page.getByRole('tabpanel').getByRole('button',{name:'Refresh results'})).toBeDisabled();await closePanels(page);await page.getByRole('button',{name:'Equity',exact:true}).click();await closePanels(page);await page.getByRole('button',{name:'Drawdown · %',exact:true}).click();await expect(page.getByRole('tabpanel').getByRole('button',{name:'Refresh results'})).toBeDisabled();await page.waitForTimeout(1000);expect(reads).toBe(2);await expect(page.getByText(/Some detail is degraded/)).toBeVisible({timeout:10000});expect(times[2]!-times[1]!).toBeGreaterThanOrEqual(2900);
 await closePanels(page);await page.getByRole('tab',{name:'Monthly statistics',exact:true}).click();await expect(page.getByRole('tabpanel').getByText('No result data available.')).toBeVisible();
 await page.route(`**/api/backtests/jobs/${id}/variants/v1/trades*`,route=>route.fulfill({status:202,json:pending('v1','failed')}));await closePanels(page);await page.getByRole('tab',{name:'Trades',exact:true}).click();await expect(page.getByText('Detail materialization failed. Job state is unchanged.')).toBeVisible();
 let release!:()=>void;const gate=new Promise<void>(r=>release=r);await page.route(`**/api/backtests/jobs/${id}/variants/v2`,async route=>{await gate;await route.fulfill({json:variant('v2')});});
 await chooseVariant(page,2);await chooseVariant(page,1);release();await expect(selected(page)).toHaveAttribute('data-result-variant','v1');await expect(selected(page)).not.toContainText('99.5');await page.goBack();await expect(selected(page)).toHaveAttribute('data-result-variant','v2');await page.goForward();await expect(selected(page)).toHaveAttribute('data-result-variant','v1');await page.reload();await openActions(page);await expect(selected(page)).toHaveAttribute('data-result-variant','v1');
 writeFileSync(resolve(evidence,'controlled-materialization.json'),JSON.stringify({controlled:true,reads,readIntervals:times.slice(1).map((v,i)=>v-times[i]!),empty:true,degraded:true,failed:true,lateVariantDiscarded:true,historyAndReload:true}));
});
test('controlled JSON export never downloads and truncation is visible',async({page})=>{
 await controlled(page);let calls=0,downloads=0;page.on('download',()=>downloads++);
 await page.route(`**/api/backtests/jobs/${id}/variants/v1/trades.csv*`,route=>{calls++;return calls===1?route.fulfill({status:202,json:pending()}):route.fulfill({contentType:'text/csv',headers:{'x-roehub-trades-row-count':'1','x-roehub-trades-total-rows':'5','x-roehub-trades-max-rows':'1','x-roehub-trades-truncated':'true'},body:'trade_index\n1\n'});});
 await page.getByLabel('CSV maximum rows').fill('1');await page.getByRole('button',{name:'Download CSV',exact:true}).click();await expect(page.getByText(/CSV is materializing/)).toBeVisible();expect(downloads).toBe(0);await expect(page.getByRole('button',{name:'Download CSV',exact:true})).toBeEnabled({timeout:5000});const download=page.waitForEvent('download');await page.getByRole('button',{name:'Download CSV',exact:true}).click();await download;await expect(page.getByText(/Export is truncated/)).toBeVisible();expect(calls).toBe(2);writeFileSync(resolve(evidence,'controlled-export.json'),JSON.stringify({controlled:true,pendingDownloads:0,downloads,calls,truncated:true}));
});
test('controlled readiness transport and failed save recover explicitly; lost save reload never replays',async({page})=>{
 await controlled(page);let reads=0,posts=0;
 await page.route(`**/api/backtests/jobs/${id}/variants/v1/compatibility-readiness`,route=>{reads++;return reads===1?route.abort('failed'):route.fulfill({json:{source_job_id:id,source_variant_key:'v1',strategy_spec_hash:'hash',compatibility_state:'launchable',market_data_state:'unavailable',checked_at:new Date().toISOString()}});});
 const section=page.getByRole('region',{name:'Save strategy',exact:true});await section.getByRole('button',{name:'Refresh results'}).click();await expect(section.getByText(/request|connection/i)).toBeVisible();await expect(section.getByRole('button',{name:'Save strategy',exact:true})).toBeDisabled();await section.getByRole('button',{name:'Refresh results'}).click();await expect(section.getByRole('button',{name:'Save strategy',exact:true})).toBeEnabled();
 await page.route(`**/api/backtests/jobs/${id}/variants/v1/strategies`,route=>{posts++;return posts===1?route.fulfill({status:429,headers:{'Retry-After':'3'},json:{error:{code:'limited'}}}):route.abort('failed');});
 await confirmSave(page);await expect(section.getByText(/Too many requests/)).toBeVisible();await expect(section.getByRole('button',{name:'Refresh results'})).toBeDisabled();await page.waitForTimeout(1000);expect(posts).toBe(1);await expect(section.getByRole('button',{name:'Refresh results'})).toBeEnabled({timeout:5000});await section.getByRole('button',{name:'Refresh results'}).click();await expect(section.getByRole('button',{name:'Save strategy',exact:true})).toBeEnabled();await confirmSave(page);await expect(section.getByText(/Save outcome is unknown/)).toBeVisible();expect(posts).toBe(2);
 const record=await page.evaluate(()=>JSON.parse(sessionStorage.getItem('roehub.backtests.strategy-recovery.v1')!));expect(Object.keys(record).sort()).toEqual(['createdAt','jobId','key','operation','organization','resultId','subject','variant'].sort());await page.reload();await openActions(page);await expect(section.getByText(/Save outcome is unknown/)).toBeVisible();await expect(section.getByRole('button',{name:'Save strategy',exact:true})).toBeDisabled();expect(posts).toBe(2);await chooseVariant(page,2);await expect(section.getByText(/unresolved strategy save exists/)).toBeVisible();
 await page.getByRole('link',{name:'Sign out',exact:true}).click();await expect.poll(()=>page.evaluate(()=>sessionStorage.getItem('roehub.backtests.strategy-recovery.v1'))).toBeNull();
 writeFileSync(resolve(evidence,'controlled-save.json'),JSON.stringify({controlled:true,readinessTransportRecovery:true,save429Delay:3,posts,lostResponse:true,reloadPosts:0,scopeUnboundReplayDisabled:true,recoveryAllowlist:true,logoutCleared:true}));
});
test('controlled delete conflict requires authoritative read; ambiguous delete is retained',async({page})=>{
 await controlled(page);let deletes=0,reads=0;await page.route(`**/api/backtests/jobs/${id}`,route=>{if(route.request().method()==='DELETE'){deletes++;return deletes===1?route.fulfill({status:409,json:{error:{code:'backtest.job_not_deletable'}}}):route.abort('failed');}reads++;return route.fulfill({json:controlledJob()});});
 await openJobInfo(page);const section=page.getByRole('region',{name:'Delete history',exact:true});const remove=section.getByRole('button',{name:'Delete history',exact:true});await remove.click();await page.keyboard.press('Escape');await expect(remove).toBeFocused();expect(deletes).toBe(0);await remove.click();await page.getByRole('button',{name:'Confirm deletion',exact:true}).click();await expect(remove).toBeDisabled();await section.getByRole('button',{name:'Check deletion outcome'}).click();await expect(remove).toBeEnabled();await remove.click();await page.getByRole('button',{name:'Confirm deletion',exact:true}).click();await expect(section.getByText(/Deletion outcome is unknown/)).toBeVisible();await section.getByRole('button',{name:'Check deletion outcome'}).click();await expect(remove).toBeDisabled();await expect(page).toHaveURL(new RegExp(`/backtests/${id}`));expect(deletes).toBe(2);expect(reads).toBe(2);writeFileSync(resolve(evidence,'controlled-delete.json'),JSON.stringify({controlled:true,deletes,reads,conflictFreshRead:true,ambiguousContextRetained:true,noRetry:true}));
});
test('controlled results RU EN widths focus and axe',async({page})=>{
 await controlled(page);const captures=[];for(const locale of ['ru','en']){await page.getByRole('link',{name:locale==='ru'?'Русский':'English',exact:true}).click();await expect(selected(page)).toBeVisible();await closePanels(page);for(const width of [820,1024,1440]){await page.setViewportSize({width,height:1100});await expect(page.getByRole('img')).toBeVisible();await expect(page.getByRole('img')).toHaveAccessibleDescription(locale==='ru'?/Наведите/:/Hover for values/);await expect.poll(()=>page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth)).toBe(true);const axe=await new AxeBuilder({page}).analyze();expect(axe.violations).toEqual([]);await page.screenshot({path:resolve(evidence,`${locale}-${width}.png`),fullPage:true});captures.push({locale,width,axe:0,overflow:false});}}
 writeFileSync(resolve(evidence,'visual.json'),JSON.stringify({controlled:true,captures}));
});
test('controlled 202 then forbidden or missing detail stops polling across tab remount',async({page})=>{
 await controlled(page);for(const status of [403,404]){let reads=0;await page.route(`**/api/backtests/jobs/${id}/variants/v1/drawdown*`,route=>{reads++;return reads===1?route.fulfill({status:202,json:pending()}):route.fulfill({status,json:{error:{code:'restricted'}}});});await closePanels(page);await page.getByRole('button',{name:'Drawdown · %',exact:true}).click();await expect(page.getByRole('tabpanel').getByRole('alert')).toBeVisible({timeout:10000});await page.waitForTimeout(2500);expect(reads).toBe(2);await closePanels(page);await page.getByRole('button',{name:'Equity',exact:true}).click();await closePanels(page);await page.getByRole('button',{name:'Drawdown · %',exact:true}).click();await page.waitForTimeout(2500);expect(reads).toBe(2);await page.unroute(`**/api/backtests/jobs/${id}/variants/v1/drawdown*`);if(status===403)await page.reload();await openActions(page);}
});
test('native 200 percent zoom results and save dialog RU EN',async()=>{
 const directory=mkdtempSync(resolve(tmpdir(),'roehub-s5-zoom-'));const extension=resolve(directory,'extension');mkdirSync(extension);writeFileSync(resolve(extension,'manifest.json'),JSON.stringify({manifest_version:3,name:'Local zoom proof',version:'1.0',permissions:['tabs'],background:{service_worker:'worker.js'}}));writeFileSync(resolve(extension,'worker.js'),'chrome.runtime.onInstalled.addListener(() => {});');
 const context=await chromium.launchPersistentContext(resolve(directory,'profile'),{baseURL:'http://localhost:18480',channel:'chromium',headless:true,viewport:null,args:[`--disable-extensions-except=${extension}`,`--load-extension=${extension}`,'--window-size=1440,1100']});
 try{const page=await context.newPage();await controlled(page);const worker=context.serviceWorkers()[0]??await context.waitForEvent('serviceworker');await worker.evaluate("chrome.tabs.query({url:'http://localhost:18480/*'}).then(tabs=>Promise.all(tabs.map(tab=>chrome.tabs.setZoom(tab.id,2))))");const zoom=await worker.evaluate("chrome.tabs.query({url:'http://localhost:18480/*'}).then(tabs=>chrome.tabs.getZoom(tabs[0].id))");expect(zoom).toBe(2);const sizes=[];const cdp=await context.newCDPSession(page);
 for(const locale of ['en','ru']){if(locale==='ru')await page.getByRole('link',{name:'Русский',exact:true}).click();await openActions(page);const save=page.getByRole('button',{name:locale==='en'?'Save strategy':'Сохранить стратегию',exact:true});await save.scrollIntoViewIfNeeded();await save.focus();await expect(save).toBeFocused();const dimensions=await page.evaluate(()=>({innerWidth,outerWidth,devicePixelRatio,scrollWidth:document.documentElement.scrollWidth}));expect(dimensions.scrollWidth).toBeLessThanOrEqual(dimensions.innerWidth);expect((await new AxeBuilder({page}).analyze()).violations).toEqual([]);let shot=await cdp.send('Page.captureScreenshot',{format:'png',fromSurface:true});writeFileSync(resolve(evidence,`${locale}-native-zoom-200.png`),Buffer.from(shot.data,'base64'));await save.click();await expect(page.getByRole('dialog')).toBeVisible();expect((await new AxeBuilder({page}).analyze()).violations).toEqual([]);const confirm=page.getByRole('button',{name:locale==='en'?'Confirm save':'Подтвердить сохранение',exact:true});await confirm.focus();await expect(confirm).toBeFocused();const box=await confirm.boundingBox();expect(box!.x).toBeGreaterThanOrEqual(0);expect(box!.x+box!.width).toBeLessThanOrEqual(dimensions.innerWidth);shot=await cdp.send('Page.captureScreenshot',{format:'png',fromSurface:true});writeFileSync(resolve(evidence,`${locale}-native-zoom-200-dialog.png`),Buffer.from(shot.data,'base64'));await page.keyboard.press('Escape');await expect(save).toBeFocused();sizes.push({locale,...dimensions});}
 writeFileSync(resolve(evidence,'zoom.json'),JSON.stringify({controlled:true,nativeZoom:zoom,sizes,axe:0},null,2));}finally{await context.close();rmSync(directory,{recursive:true,force:true});}
});
test('controlled save storage degradation, source mismatch and session outage never claim success',async({page})=>{
 await page.addInitScript(()=>{const original=Storage.prototype.setItem;Storage.prototype.setItem=function(key,value){if(key.startsWith('roehub.backtests.'))throw new Error('controlled storage unavailable');return original.call(this,key,value);};});await controlled(page);await expect(page.getByText(/Tab storage is unavailable/)).toBeVisible();let posts=0;
 await page.route(`**/api/backtests/jobs/${id}/variants/v1/strategies`,route=>{posts++;return route.abort('failed');});await confirmSave(page);await expect(page.getByText(/Save outcome is unknown/)).toBeVisible();expect(posts).toBe(1);await chooseVariant(page,2);await expect(page.getByText(/unresolved strategy save exists/)).toBeVisible();
 await page.reload();await openActions(page);await expect(page.getByRole('button',{name:'Save strategy',exact:true})).toBeEnabled();await page.route(`**/api/backtests/jobs/${id}/variants/v2/compatibility-readiness`,route=>route.fulfill({json:{source_job_id:id,source_variant_key:'other',strategy_spec_hash:'hash',compatibility_state:'launchable',market_data_state:'ready',checked_at:new Date().toISOString()}}));await page.getByRole('region',{name:'Save strategy',exact:true}).getByRole('button',{name:'Refresh results'}).click();await expect(page.getByRole('button',{name:'Save strategy',exact:true})).toBeDisabled();expect(posts).toBe(1);
 await chooseVariant(page,1);await page.route('**/api/auth/current-user',route=>route.fulfill({status:503,json:{error:{code:'identity_unavailable'}}}));await confirmSave(page);await expect(page.getByText('Session service is unavailable. Reload this page to check again.',{exact:true})).toBeVisible();expect(posts).toBe(1);
});
test('controlled pending result401 closes private surface before malformed response and stops reads',async({page})=>{
 await controlled(page);let reads=0;await page.route(`**/api/backtests/jobs/${id}/variants/v1/drawdown*`,route=>{reads++;return reads===1?route.fulfill({status:202,json:pending()}):route.fulfill({status:401,body:'not-json'});});await closePanels(page);await page.getByRole('button',{name:'Drawdown · %',exact:true}).click();await expect(page.getByRole('link',{name:'Sign in',exact:true})).toBeVisible({timeout:10000});let requests=0;page.on('request',r=>{if(new URL(r.url()).pathname.startsWith('/api/'))requests++;});await page.waitForTimeout(4000);expect(requests).toBe(0);expect(reads).toBe(2);await expect(page.locator('[data-platform-client]')).toHaveCount(0);
});
test('controlled pending then transport permits explicit read recovery without inheriting old202',async({page})=>{
 await controlled(page);let reads=0;await page.route(`**/api/backtests/jobs/${id}/variants/v1/drawdown*`,route=>{reads++;return reads===1?route.fulfill({status:202,json:pending()}):reads===2?route.abort('failed'):route.fulfill({json:series('v1','drawdown')});});await closePanels(page);await page.getByRole('button',{name:'Drawdown · %',exact:true}).click();await expect(page.getByRole('tabpanel').getByRole('alert')).toBeVisible({timeout:10000});const refresh=page.getByRole('tabpanel').getByRole('button',{name:'Refresh results'});await expect(refresh).toBeEnabled();await page.waitForTimeout(2200);expect(reads).toBe(2);await refresh.click();await expect(page.getByRole('img',{name:'Drawdown · %',exact:true})).toBeVisible();expect(reads).toBe(3);
});
test('controlled delete403 closes selected private results and stops their pending reads',async({page})=>{
 await controlled(page);let reads=0;await page.route(`**/api/backtests/jobs/${id}/variants/v1/drawdown*`,route=>{reads++;return route.fulfill({status:202,json:pending()});});await closePanels(page);await page.getByRole('button',{name:'Drawdown · %',exact:true}).click();await expect(page.getByText('Result detail is materializing. Reads follow the server delay.')).toBeVisible();await page.route(`**/api/backtests/jobs/${id}`,route=>route.request().method()==='DELETE'?route.fulfill({status:403,json:{error:{code:'forbidden'}}}):route.fulfill({json:controlledJob()}));await openJobInfo(page);await page.getByRole('button',{name:'Delete history',exact:true}).click();await page.getByRole('button',{name:'Confirm deletion',exact:true}).click();await expect(page.locator('.job-detail').getByText(/You do not have access/)).toBeVisible();await expect(selected(page)).toHaveCount(0);await page.waitForTimeout(3000);expect(reads).toBe(1);
});

test('controlled stale readiness and changed source at confirmation block save before POST',async({page})=>{
 await controlled(page);
 let posts=0,mode='stale';
 const path=`**/api/backtests/jobs/${id}/variants/v1/compatibility-readiness`;
 await page.route(`**/api/backtests/jobs/${id}/variants/v1/strategies`,route=>{posts++;return route.abort('failed');});
 await page.route(path,route=>route.fulfill({json:{source_job_id:id,source_variant_key:'v1',strategy_spec_hash:mode==='changed'?'different-hash':'hash',compatibility_state:'not_launchable',compatibility_reason_codes:['unsupported_live_evaluator'],market_data_state:'unavailable',checked_at:new Date(Date.now()+(mode==='stale'?-120000:mode==='future'?120000:0)).toISOString()}}));
 const section=page.getByRole('region',{name:'Save strategy',exact:true});
 const refresh=section.getByRole('button',{name:'Refresh results',exact:true});
 const save=section.getByRole('button',{name:'Save strategy',exact:true});
 await refresh.click();await expect(save).toBeDisabled();
 mode='future';await refresh.click();await expect(save).toBeDisabled();
 mode='fresh';await refresh.click();await expect(save).toBeEnabled();
 await save.click();mode='changed';await page.getByRole('button',{name:'Confirm save',exact:true}).click();
 await expect(section.getByRole('alert')).toBeVisible();await expect(save).toBeDisabled();
 expect(posts).toBe(0);
 expect(await page.evaluate(()=>sessionStorage.getItem('roehub.backtests.strategy-recovery.v1'))).toBeNull();
 await expect(refresh).toBeEnabled({timeout:5000});mode='fresh';await refresh.click();await expect(save).toBeEnabled();expect(posts).toBe(0);
 writeFileSync(resolve(evidence,'controlled-readiness-freshness.json'),JSON.stringify({controlled:true,staleBlocked:true,futureBlocked:true,sourceChangedAtConfirmationBlocked:true,posts,recoveryRecordAbsent:true,explicitFreshReadRestoresEligibility:true}));
});

test('controlled result429 without hints or with short hint shares manual and remount cooldown',async({page})=>{
 await controlled(page);const observations=[];
 for(const hint of [null,1]){
  if(hint!==null)await page.reload();await openActions(page);
  let reads=0;const times:number[]=[];
  const path=`**/api/backtests/jobs/${id}/variants/v1/drawdown*`;
  await page.route(path,route=>{reads++;times.push(Date.now());return reads===1?route.fulfill({status:429,headers:hint===null?{}:{'Retry-After':String(hint)},json:{error:{code:'limited'}}}):route.fulfill({json:series('v1','drawdown')});});
  await closePanels(page);await page.getByRole('button',{name:'Drawdown · %',exact:true}).click();
  const refresh=page.getByRole('tabpanel').getByRole('button',{name:'Refresh results',exact:true});
  await expect(page.getByRole('tabpanel').getByRole('alert')).toBeVisible();
  const initiallyEnabled=await refresh.isEnabled();
  observations.push({hint,initiallyEnabled});
  writeFileSync(resolve(evidence,'controlled-unhinted-cooldown.json'),JSON.stringify({controlled:true,observations}));
  await expect(refresh).toBeDisabled({timeout:500});
  await page.waitForTimeout(hint===null?1500:1200);
  Object.assign(observations.at(-1)!,{enabledDuringCooldown:await refresh.isEnabled()});
  writeFileSync(resolve(evidence,'controlled-unhinted-cooldown.json'),JSON.stringify({controlled:true,observations}));
  await expect(refresh).toBeDisabled({timeout:250});
  await closePanels(page);await page.getByRole('button',{name:'Equity',exact:true}).click();await closePanels(page);await page.getByRole('button',{name:'Drawdown · %',exact:true}).click();
  await expect(refresh).toBeDisabled();await page.waitForTimeout(hint===null?1500:100);expect(reads).toBe(1);
  await expect(page.getByRole('img',{name:'Drawdown · %',exact:true})).toBeVisible({timeout:6000});
  expect(reads).toBe(2);expect(times[1]!-times[0]!).toBeGreaterThanOrEqual(hint===null?4900:1900);
  Object.assign(observations.at(-1)!,{reads,intervalMs:times[1]!-times[0]!,remountCannotBypass:true});
  await page.unroute(path);
 }
 writeFileSync(resolve(evidence,'controlled-unhinted-cooldown.json'),JSON.stringify({controlled:true,observations}));
});
