import { test, expect, chromium, type Page } from '@playwright/test';
import AxeBuilder from '@axe-core/playwright';
import { readFileSync, writeFileSync, mkdirSync, mkdtempSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { execFileSync } from 'node:child_process';
import { resolve } from 'node:path';
const root=resolve(import.meta.dirname,'../../..');
const evidence=process.env.ROEHUB_PROOF_DIR ? resolve(root,process.env.ROEHUB_PROOF_DIR) : resolve(root,'.codex/delivery/evidence/roehub-backtests-client-v1/browser/'+(process.env.ROEHUB_PROOF_STAGE==='S6'?'S6-builder-regression':process.env.ROEHUB_PROOF_STAGE==='S5'?'S5-builder-regression':process.env.ROEHUB_PROOF_STAGE==='S4'?'S4-builder-regression':'S3'));
const recoveryKey='roehub.backtests.create-recovery.v1';
async function signIn(page:Page) {
  await page.goto('/backtests/new');
  const c=JSON.parse(readFileSync(resolve(root,'.local_artifacts/backtests-client/credentials.json'),'utf8'));
  const form=page.locator('[data-password-login]');await form.locator('..').locator('summary').click();
  await form.locator('[name=username]').fill(c.username);await form.locator('[name=password]').fill(c.password);await form.locator('button[type=submit]').click();
  await expect(page.getByLabel('Job label (optional)')).toBeVisible();
}
async function configure(page:Page,label:string) {
  await page.getByLabel('Job label (optional)').fill(label);
  await page.getByLabel('Timeframe',{exact:true}).selectOption('15m');
  await page.getByLabel('Start date',{exact:true}).fill('2026-03-26');
  await page.getByLabel('End date',{exact:true}).fill('2026-03-29');
  await page.getByLabel('Indicator',{exact:true}).selectOption('ma.ema');
}
async function check(page:Page) {
  await page.getByRole('button',{name:'Check configuration',exact:true}).click();
  await expect(page.getByRole('button',{name:'Submit backtest',exact:true})).toBeEnabled();
}

test('real configure/preflight/create, exact rates, double click, label metadata and normalized conflict',async({page})=>{
  mkdirSync(evidence,{recursive:true});await signIn(page);const errors:string[]=[];const http:{path:string;status:number}[]=[];
  page.on('pageerror',()=>errors.push('pageerror'));page.on('console',m=>{if(m.type()==='error')errors.push('console error');});
  page.on('response',r=>{if(r.status()>=400)http.push({path:new URL(r.url()).pathname,status:r.status()});});
  await configure(page,'S3 real create');await page.getByLabel('Fee (%)',{exact:true}).fill('0.15');await page.getByLabel('Slippage (%)',{exact:true}).fill('0.025');
  await check(page);
  await page.evaluate(()=>scrollTo(0,0));
  await page.screenshot({path:resolve(evidence,'real-preflight.png'),fullPage:true});
  await page.getByText('Minimum closed trades: 1',{exact:true}).scrollIntoViewIfNeeded();
  await expect(page.getByText('Minimum closed trades: 1',{exact:true})).toBeVisible();
  await page.screenshot({path:resolve(evidence,'real-preflight-effective.png')});
  const requests:{key:string|null;body:Record<string,unknown>}[]=[];
  page.on('request',r=>{if(new URL(r.url()).pathname==='/api/backtests/jobs' && r.method()==='POST')requests.push({key:r.headers()['idempotency-key'],body:r.postDataJSON()});});
  const response=page.waitForResponse(r=>new URL(r.url()).pathname==='/api/backtests/jobs' && r.request().method()==='POST');
  await page.getByRole('button',{name:'Submit backtest',exact:true}).evaluate(el=>{(el as HTMLButtonElement).click();(el as HTMLButtonElement).click();});
  const created=await response;expect(created.status()).toBe(201);const job=await created.json();
  await expect(page).toHaveURL(new RegExp(`/backtests/${job.job_id}$`));expect(requests).toHaveLength(1);
  expect(requests[0].body.execution).toMatchObject({fee_rate:0.0015,slippage_rate:0.00025});
  expect(await page.evaluate(k=>sessionStorage.getItem(k),recoveryKey)).toBeNull();
  // Explicit controlled same-key probes prove actual server identity semantics, not UI recovery/replay.
  const uiCreateRequests=requests.length;
  const probe=await page.evaluate(async({request})=>{
    const post=async(body:unknown)=>{const r=await fetch('/api/backtests/jobs',{method:'POST',headers:{'Content-Type':'application/json','Idempotency-Key':request.key!},body:JSON.stringify(body)});const data=await r.json();return{status:r.status,jobId:data.job_id};};
    const label=await post({...request.body,strategy_name:'label-only changed'});
    const conflict=await post({...request.body,top_n:2});return{label,conflict};
  },{request:requests[0]});
  expect(probe.label).toEqual({status:200,jobId:job.job_id});expect(probe.conflict.status).toBe(409);
  await page.screenshot({path:resolve(evidence,'created.png'),fullPage:true});
  // Two HTTP/console error observations belong to the deliberate 409 probe.
  expect(http).toEqual([{path:'/api/backtests/jobs',status:409}]);expect(errors.filter(x=>x==='pageerror')).toEqual([]);
  writeFileSync(resolve(evidence,'real-create.json'),JSON.stringify({jobId:job.job_id,status:201,createRequests:uiCreateRequests,feeRate:0.0015,slippageRate:0.00025,labelReplayStatus:probe.label.status,changedRequestStatus:probe.conflict.status,http,errors},null,2));
});

test('accepted create with lost response retains exact body/key across reload without replay, then logout clears',async({page})=>{
  await signIn(page);await configure(page,'S3 lost response');await check(page);
  let count=0;let acceptedId='';let submitted:unknown;let key='';
  await page.route('**/api/backtests/jobs',async route=>{if(route.request().method()!=='POST'){await route.continue();return;}
    count++;submitted=route.request().postDataJSON();key=route.request().headers()['idempotency-key'];const response=await route.fetch();expect(response.status()).toBe(201);acceptedId=(await response.json()).job_id;await route.abort('failed');});
  await page.getByRole('button',{name:'Submit backtest',exact:true}).click();
  await expect(page.getByRole('heading',{name:'Submission outcome is unresolved'})).toBeVisible();
  const record=await page.evaluate(k=>JSON.parse(sessionStorage.getItem(k)!),recoveryKey);expect(record.body).toEqual(submitted);expect(record.key).toBe(key);expect(record.organization).toBeNull();expect(Object.keys(record).sort()).toEqual(['body','createdAt','key','operation','organization','resultId','subject']);
  await page.reload();await expect(page.getByRole('heading',{name:'Submission outcome is unresolved'})).toBeVisible();
  expect(await page.getByRole('button',{name:'Submit backtest',exact:true}).count()).toBe(0);
  const history=await page.request.get('/api/backtests/jobs');expect((await history.json()).items.filter((j:{job_id:string})=>j.job_id===acceptedId)).toHaveLength(1);expect(count).toBe(1);
  await page.screenshot({path:resolve(evidence,'lost-response-reload.png'),fullPage:true});
  // Switch this disposable actor's sole membership after acceptance. No new request is sent.
  const scope=JSON.parse(execFileSync(resolve(root,'.venv/bin/python'),['-c',`
import json
from pathlib import Path
from uuid import uuid4
import psycopg
private=json.loads(Path('.local_artifacts/backtests-client/credentials.json').read_text())
with psycopg.connect(private['dsn']) as db:
    row=db.execute('SELECT organization_id FROM identity_memberships WHERE user_id=%s',(private['subject'],)).fetchall()
    assert len(row)==1
    old=row[0][0]; new=uuid4()
    db.execute("INSERT INTO identity_organizations (organization_id,installation_id,slug,display_name,created_at) SELECT %s,installation_id,%s,'Disposable changed scope',now() FROM identity_organizations WHERE organization_id=%s",(new,'proof-scope-'+new.hex,old))
    replacement_owner=uuid4()
    db.execute("INSERT INTO identity_users (user_id,paid_level,created_at) VALUES (%s,'free',now())",(replacement_owner,))
    db.execute("INSERT INTO identity_memberships (organization_id,user_id,role,created_at,updated_at) VALUES (%s,%s,'owner',now(),now())",(old,replacement_owner))
    db.execute("UPDATE identity_memberships SET status='suspended',updated_at=now() WHERE user_id=%s AND organization_id=%s",(private['subject'],old))
    db.execute("INSERT INTO identity_memberships (organization_id,user_id,role,created_at,updated_at) VALUES (%s,%s,'owner',now(),now())",(new,private['subject']))
    assert db.execute("SELECT count(*) FROM identity_memberships WHERE user_id=%s AND status='active'",(private['subject'],)).fetchone()[0]==1
print(json.dumps({'soleOrganizationChanged':str(old)!=str(new)}))
`],{cwd:root,encoding:'utf8'}));expect(scope.soleOrganizationChanged).toBe(true);
  await page.reload();await expect(page.getByRole('heading',{name:'Submission outcome is unresolved'})).toBeVisible();expect(count).toBe(1);
  const changedHistory=await page.request.get('/api/backtests/jobs');expect(changedHistory.status()).toBe(200);expect((await changedHistory.json()).items).toEqual([]);
  // Same-subject re-auth must keep unresolved record and must not treat the new empty scope as failed create.
  execFileSync(resolve(root,'.venv/bin/python'),['-m','tools.qa.backtests_client_fixture','expire'],{cwd:root});
  await page.reload();await expect(page).toHaveURL(/login/);
  const credentials=JSON.parse(readFileSync(resolve(root,'.local_artifacts/backtests-client/credentials.json'),'utf8'));const login=page.locator('[data-password-login]');await login.locator('..').locator('summary').click();await login.locator('[name=username]').fill(credentials.username);await login.locator('[name=password]').fill(credentials.password);await login.locator('button[type=submit]').click();
  await expect(page.getByRole('heading',{name:'Submission outcome is unresolved'})).toBeVisible();expect(await page.evaluate(k=>JSON.parse(sessionStorage.getItem(k)!).key,recoveryKey)).toBe(key);expect(count).toBe(1);

  await page.getByRole('link',{name:'Sign out',exact:true}).click();await expect(page).toHaveURL(/login/);
  expect(await page.evaluate(k=>sessionStorage.getItem(k),recoveryKey)).toBeNull();
  writeFileSync(resolve(evidence,'lost-response.json'),JSON.stringify({acceptedId,createRequests:count,exactBodyAndKeyRetained:true,organization:null,soleOrganizationChanged:true,newScopeHistoryEmpty:true,sameSubjectReauthRetained:true,automaticReplay:false,logoutCleared:true},null,2));
});

test('body errors, stale preflight, 422 field focus, renewed admission and memory-only recovery',async({page})=>{
  await signIn(page);await configure(page,'S3 faults');
  let mode='body';
  await page.route('**/api/backtests/preflight',async route=>{
    if(mode==='422'){await route.fulfill({status:422,json:{error:{code:'validation_error',details:{errors:[{path:'body.execution.fee_rate',code:'invalid_value',message:'Fee rejected by fixture fault'}]}}}});return;}
    // Controlled fault response; real preflight/create is proved separately above.
    const body={normalized_request:route.request().postDataJSON(),request_hash:'controlled',result_config_hash:'controlled',artifact_metadata:{},cost_estimate:{indicator_rows:1,candidate_combinations:1,tp_sl_cells:0,cost_class:'small'},warnings:[],errors:mode==='body'?[{path:'risk',code:'fixture_block',message:'Controlled body rejection'}]:[],funding_readiness:{status:'not_applicable',coverage_policy:'not_applicable',coverage_ratio:null,rows_count:0,expected_event_count:0,missing_event_count:0,warning_codes:[]},direction_market_compatibility:{compatible:true,market_type:'spot',direction_mode:'long_only'}};
    await route.fulfill({status:200,json:body});
  });
  await page.getByRole('button',{name:'Check configuration',exact:true}).click();await expect(page.getByRole('button',{name:'Submit backtest',exact:true})).toBeDisabled();await expect(page.getByRole('alert')).toContainText('Controlled body rejection');
  mode='422';await page.getByRole('button',{name:'Check configuration',exact:true}).click();await expect(page.getByRole('alert')).toBeFocused();await page.getByRole('link',{name:/execution.fee_rate/}).click();await expect(page.getByLabel('Fee (%)',{exact:true})).toBeFocused();
  mode='valid';await check(page);await page.getByLabel('Job label (optional)').fill('S3 metadata edit');await expect(page.getByRole('button',{name:'Submit backtest',exact:true})).toBeEnabled();
  await page.getByLabel('Top N',{exact:true}).fill('3');await expect(page.getByRole('button',{name:'Submit backtest',exact:true})).toBeDisabled();await expect(page.getByText('Configuration changed. Check it again before submitting.')).toBeVisible();await check(page);
  let posts=0;await page.route('**/api/backtests/jobs',async route=>{if(route.request().method()==='POST'){posts++;await route.fulfill({status:429,headers:{'Retry-After':'1'},json:{error:{code:'backtest.rate_limited'}}});}else await route.continue();});
  await page.getByRole('button',{name:'Submit backtest',exact:true}).click();await expect(page.getByRole('heading',{name:'The server rejected submission. Your inputs are retained.'})).toBeVisible();await expect(page.getByRole('button',{name:'Correct rejected configuration'})).toBeEnabled();expect(posts).toBe(1);
  await page.getByRole('button',{name:'Correct rejected configuration'}).click();await expect(page.getByLabel('Top N',{exact:true})).toHaveValue('3');
  await page.unroute('**/api/backtests/jobs');await check(page);
  await page.evaluate(()=>{Storage.prototype.setItem=()=>{throw new DOMException('unavailable','QuotaExceededError');};});
  await page.getByRole('button',{name:'Submit backtest',exact:true}).click();await expect(page.getByText(/The request has NOT been sent/)).toBeVisible();
  await page.route('**/api/backtests/jobs',async route=>{if(route.request().method()==='POST'){posts++;await route.fulfill({status:503,json:{error:{code:'backtest.artifacts_unavailable'}}});}else await route.continue();});
  await page.getByRole('button',{name:'Submit backtest',exact:true}).click();await expect(page.getByRole('heading',{name:'Submission outcome is unresolved'})).toBeVisible();await expect(page.getByText(/Recovery is kept in memory only/)).toBeVisible();expect(posts).toBe(2);
  await page.screenshot({path:resolve(evidence,'memory-only-unresolved.png'),fullPage:true});
  writeFileSync(resolve(evidence,'faults.json'),JSON.stringify({bodyErrorsBlocked:true,field422Focused:true,labelOnlyKeepsPreflight:true,resultEditInvalidates:true,admission429:1,storageFailureBeforePost:true,unknown503:1,automaticRetries:0},null,2));
});

test('dirty discard, catalog form in RU/EN at 820/1024/1440, axe and keyboard',async({page})=>{
  await signIn(page);await configure(page,'S3 unsent');let dialogs=0;page.on('dialog',async dialog=>{dialogs++;await dialog.dismiss();});
  await page.getByRole('link',{name:'Back to library'}).click();await expect(page).toHaveURL(/backtests\/new/);await expect.poll(()=>dialogs).toBe(1);await expect(page.getByLabel('Job label (optional)')).toHaveValue('S3 unsent');
  page.removeAllListeners('dialog');page.once('dialog',d=>d.accept());await page.getByRole('link',{name:'Back to library'}).click();await expect(page).toHaveURL(/backtests$/);
  await page.getByRole('link',{name:'New backtest',exact:true}).click();await expect(page.getByLabel('Job label (optional)')).toHaveValue('');
  await page.getByLabel('Job label (optional)').fill('Back guard');let backDialog=false;page.once('dialog',async d=>{backDialog=true;await d.dismiss();});await page.goBack();await expect.poll(()=>backDialog).toBe(true);await expect(page).toHaveURL(/backtests\/new/);await expect(page.getByLabel('Job label (optional)')).toHaveValue('Back guard');
  page.once('dialog',d=>d.accept());await page.goBack();await expect(page).toHaveURL(/backtests$/);await page.goForward();await expect(page.getByLabel('Job label (optional)')).toHaveValue('');
  const observations=[];
  for(const locale of ['en','ru']){
    await page.getByRole('link',{name:locale==='ru'?'Русский':'English',exact:true}).click();
    await expect(page.getByRole('heading',{name:locale==='ru'?'Новый бэктест':'New backtest',level:1})).toBeVisible();
    for(const width of [820,1024,1440]){await page.setViewportSize({width,height:1000});await expect(page.locator('.builder-form form')).toBeVisible();
      expect((await new AxeBuilder({page}).analyze()).violations).toEqual([]);
      const dims=await page.evaluate(()=>({inner:innerWidth,scroll:document.documentElement.scrollWidth}));expect(dims.scroll).toBeLessThanOrEqual(dims.inner);
      await page.screenshot({path:resolve(evidence,`${locale}-${width}.png`),fullPage:true});observations.push({locale,width,...dims,axe:0});
    }
  }
  writeFileSync(resolve(evidence,'visual.json'),JSON.stringify({discardCancelledAndConfirmed:true,backForwardDiscardVerified:true,observations},null,2));
});

test('native Chromium 200% zoom: RU/EN builder and primary actions remain reachable',async()=>{
 const directory=mkdtempSync(resolve(tmpdir(),'roehub-s3-zoom-'));const extension=resolve(directory,'extension');mkdirSync(extension);
 writeFileSync(resolve(extension,'manifest.json'),JSON.stringify({manifest_version:3,name:'Local zoom proof',version:'1.0',permissions:['tabs'],background:{service_worker:'worker.js'}}));writeFileSync(resolve(extension,'worker.js'),'chrome.runtime.onInstalled.addListener(() => {});');
 const context=await chromium.launchPersistentContext(resolve(directory,'profile'),{channel:'chromium',headless:true,viewport:null,args:[`--disable-extensions-except=${extension}`,`--load-extension=${extension}`,'--window-size=1440,1100']});
 try{const page=await context.newPage();await page.goto('http://localhost:18480');
 // Standalone persistent context has no config baseURL.
 const c=JSON.parse(readFileSync(resolve(root,'.local_artifacts/backtests-client/credentials.json'),'utf8'));await page.goto('http://localhost:18480/backtests/new');const f=page.locator('[data-password-login]');await f.locator('..').locator('summary').click();await f.locator('[name=username]').fill(c.username);await f.locator('[name=password]').fill(c.password);await f.locator('button[type=submit]').click();await expect(page.getByLabel('Job label (optional)')).toBeVisible();
 const worker=context.serviceWorkers()[0] ?? await context.waitForEvent('serviceworker');await worker.evaluate("chrome.tabs.query({url:'http://localhost:18480/*'}).then(tabs=>Promise.all(tabs.map(tab=>chrome.tabs.setZoom(tab.id,2))))");
 const zoom=await worker.evaluate("chrome.tabs.query({url:'http://localhost:18480/*'}).then(tabs=>chrome.tabs.getZoom(tabs[0].id))");expect(zoom).toBe(2);const sizes=[];const cdp=await context.newCDPSession(page);
 for(const locale of ['en','ru']){await page.getByRole('link',{name:locale==='en'?'English':'Русский',exact:true}).click();await expect(page.locator('html')).toHaveAttribute('lang',locale);await expect(page.locator('.builder-form form')).toBeVisible();
  const dimensions=await page.evaluate(()=>({innerWidth,outerWidth,devicePixelRatio,scrollWidth:document.documentElement.scrollWidth}));expect(dimensions.scrollWidth).toBeLessThanOrEqual(dimensions.innerWidth);expect(dimensions.innerWidth).toBeLessThan(800);expect((await new AxeBuilder({page}).analyze()).violations).toEqual([]);
  await page.evaluate(()=>scrollTo(0,0));let shot=await cdp.send('Page.captureScreenshot',{format:'png',fromSurface:true});writeFileSync(resolve(evidence,`${locale}-native-zoom-200.png`),Buffer.from(shot.data,'base64'));
  const action=page.getByRole('button',{name:locale==='en'?'Check configuration':'Проверить параметры',exact:true});await action.scrollIntoViewIfNeeded();await action.focus();await expect(action).toBeFocused();const box=await action.boundingBox();expect(box!.x).toBeGreaterThanOrEqual(0);expect(box!.x+box!.width).toBeLessThanOrEqual(dimensions.innerWidth);
  shot=await cdp.send('Page.captureScreenshot',{format:'png',fromSurface:true});writeFileSync(resolve(evidence,`${locale}-native-zoom-200-actions.png`),Buffer.from(shot.data,'base64'));sizes.push({locale,...dimensions});
 }
 writeFileSync(resolve(evidence,'zoom.json'),JSON.stringify({nativeZoom:zoom,sizes,axe:0,primaryActionsReachable:true},null,2));
 }finally{await context.close();rmSync(directory,{recursive:true,force:true});}
});

test('real preflight 401 immediately closes private UI and stays quiet, then same-subject login returns',async({page})=>{
 await signIn(page);await configure(page,'S3 expiry');execFileSync(resolve(root,'.venv/bin/python'),['-m','tools.qa.backtests_client_fixture','expire'],{cwd:root});
 const expired=page.waitForResponse(r=>new URL(r.url()).pathname==='/api/backtests/preflight' && r.status()===401);await page.getByRole('button',{name:'Check configuration',exact:true}).click();await expired;
 await expect(page.getByRole('link',{name:'Sign in',exact:true})).toBeVisible();await expect(page.locator('[data-platform-client]')).toHaveCount(0);const reads:string[]=[];page.on('request',r=>{if(new URL(r.url()).pathname.startsWith('/api/'))reads.push(new URL(r.url()).pathname);});
 await page.waitForTimeout(31000);expect(reads).toEqual([]);await page.screenshot({path:resolve(evidence,'preflight-expired.png'),fullPage:true});
 await page.getByRole('link',{name:'Sign in',exact:true}).click();const c=JSON.parse(readFileSync(resolve(root,'.local_artifacts/backtests-client/credentials.json'),'utf8'));const f=page.locator('[data-password-login]');await f.locator('..').locator('summary').click();await f.locator('[name=username]').fill(c.username);await f.locator('[name=password]').fill(c.password);await f.locator('button[type=submit]').click();await expect(page.getByLabel('Job label (optional)')).toHaveValue('');
 writeFileSync(resolve(evidence,'preflight-expiry.json'),JSON.stringify({realPreflightStatus:401,privateUiRemoved:true,quietMs:31000,protectedRequests:0,sameSubjectLoginReturned:true,unsentDraftNotRestored:true},null,2));
});
