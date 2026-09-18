import { test, expect, type Page } from '@playwright/test';
import AxeBuilder from '@axe-core/playwright';
import { readFileSync, mkdirSync, writeFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { randomUUID } from 'node:crypto';
test.use({actionTimeout:15000});
const root=resolve(import.meta.dirname,'../../..');
const port=Number(process.env.ROEHUB_PROOF_PORT ?? 18480);
const evidence=resolve(root,process.env.ROEHUB_PROOF_EVIDENCE ?? '.codex/delivery/evidence/roehub-strategies-client-2026-09-12','strategies');
async function login(page:Page,path='/strategies') {
  await page.goto(path);
  const credentials=JSON.parse(readFileSync(resolve(root,process.env.ROEHUB_PROOF_STATE ?? '.local_artifacts/backtests-client','credentials.json'),'utf8'));
  const form=page.locator('[data-password-login]');
  await form.locator('..').locator('summary').click();
  await form.locator('[name=username]').fill(credentials.username);await form.locator('[name=password]').fill(credentials.password);
  await form.locator('button[type=submit]').click(); await expect(page.locator('[data-platform-client]')).toBeVisible();
}
async function seed(page:Page,fast:number) {
  const response=await page.request.post('/api/strategies',{data:{instrument_id:{market_id:1,symbol:'BTCUSDT'},instrument_key:'binance:spot:BTCUSDT',market_type:'spot',timeframe:'1m',indicators:[{name:'MA',params:{fast,slow:50}}],signal_template:`MA(${fast},50)`}});
  expect(response.status()).toBe(201);return response.json();
}
const heading=(page:Page)=>page.locator('#selected-strategy-heading');

test('real library, independent detail, filters, keyboard, status, return context, layouts and motion',async({page})=>{
  test.setTimeout(120000);mkdirSync(evidence,{recursive:true});await login(page);
  await expect(page.getByText('No saved strategies yet',{exact:true})).toBeVisible();
  await page.screenshot({path:resolve(evidence,'empty.png')});
  const first=await seed(page,20),second=await seed(page,30);
  const commands:string[]=[];const errors:string[]=[];const httpErrors:{path:string;status:number}[]=[];const consoleErrors:string[]=[];
  page.on('response',response=>{if(response.status()>=400)httpErrors.push({path:new URL(response.url()).pathname,status:response.status()});});
  page.on('console',message=>{if(message.type()==='error')consoleErrors.push(message.text().replace(/https?:\/\/\S+/g,'[URL]').slice(0,180));});
  page.on('request',r=>{if(r.method()!=='GET' && /\/api\//.test(r.url()))commands.push(new URL(r.url()).pathname);});
  page.on('pageerror',error=>errors.push(`${new URL(page.url()).pathname}: ${error.message.replace(/https?:\/\/\S+/g,'[URL]').slice(0,300)}`));
  await page.reload();await expect(page.locator('.strategy-row')).toHaveCount(2);
  const firstRow=page.locator(`.strategy-row[href^="/strategies/${first.strategy_id}"]`);
  await firstRow.hover();expect(await firstRow.evaluate(element=>element.matches(':hover'))).toBe(true);
  await firstRow.focus();await page.keyboard.press('Enter');await expect(heading(page)).toHaveText(first.name);
  await expect(firstRow).toHaveAttribute('aria-current','true');await expect(heading(page)).toBeFocused();
  await page.getByRole('tab',{name:'Settings',exact:true}).click();
  await expect(page.getByText('MA(20,50)',{exact:true})).toBeVisible();await expect(page.getByRole('button',{name:'Refresh',exact:true})).toHaveCount(1);await expect(page.getByText('Fast period',{exact:true})).toBeVisible();
  await expect(page.getByRole('link',{name:'Return to report',exact:true})).toHaveCount(0);
  await page.getByLabel('Search strategies').fill('never-matches');
  await expect(page.getByText('No matching strategies',{exact:true})).toBeVisible();
  await expect(page.getByText('The selected strategy is outside the current filters.')).toBeVisible();
  await expect(heading(page)).toHaveText(first.name);await page.getByRole('button',{name:'Reset filters'}).click();
  await page.getByRole('combobox',{name:'Market type',exact:true}).selectOption('spot');
  await page.locator(`.strategy-row[href^="/strategies/${second.strategy_id}"]`).click();await expect(heading(page)).toHaveText(second.name);
  await page.goBack();await expect(heading(page)).toHaveText(first.name);expect(new URL(page.url()).searchParams.get('market')).toBe('spot');
  await page.goForward();await expect(heading(page)).toHaveText(second.name);
  await page.reload();await expect(heading(page)).toHaveText(second.name);
  await page.route('**/api/strategies',route=>route.fulfill({status:503,json:{}}));
  await page.goto(`/strategies/${first.strategy_id}`);await expect(heading(page)).toHaveText(first.name);
  await expect(page.locator('#strategies-library [role=alert]')).toBeVisible();await page.unroute('**/api/strategies');
  await page.goto(`/strategies/${first.strategy_id}?from_job=bad&from_variant=x&return=https://evil.example`);
  await expect(heading(page)).toHaveText(first.name);await expect(page.getByRole('link',{name:'Return to report',exact:true})).toHaveCount(0);
  const absent=randomUUID();
  await page.goto(`/strategies/${first.strategy_id}?from_job=${absent}&from_variant=original`);
  await expect(page.getByRole('link',{name:'Return to report',exact:true})).toHaveCount(0);
  await expect(heading(page)).toHaveText(first.name);
  await page.goto(`/strategies/${absent}`);await expect(page.getByText('This strategy was not found or is not visible to your account.',{exact:true})).toBeVisible();
  await expect(heading(page)).not.toHaveText(first.name);
  await page.goto(`/strategies/${first.strategy_id}`);await expect(heading(page)).toHaveText(first.name);
  // Current real Backtests reference at the same viewport; no historical specimen writes.
  for(const locale of ['en','ru']) {
    await page.getByRole('link',{name:locale==='ru'?'Русский':'English',exact:true}).click();
    for(const width of [820,1024,1440]) {
      await page.setViewportSize({width,height:1000});await expect(heading(page)).toHaveText(first.name);
      await expect(page.locator('.strategies-refresh')).not.toHaveText(/Refreshing|Обновляем/);
      expect(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth)).toBe(true);
      await page.screenshot({path:resolve(evidence,`strategies-${locale}-${width}.png`),fullPage:true});
      expect((await new AxeBuilder({page}).analyze()).violations).toEqual([]);
    }
    await page.evaluate(()=>{document.documentElement.style.zoom='2';});
    expect(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth)).toBe(true);
    await page.screenshot({path:resolve(evidence,`strategies-${locale}-zoom200.png`),fullPage:true});
    expect((await new AxeBuilder({page}).analyze()).violations).toEqual([]);
    await page.evaluate(()=>{document.documentElement.style.zoom='';});
  }
  await page.getByRole('link',{name:'English',exact:true}).click();
  await page.evaluate(()=>{document.documentElement.style.zoom='2';});
  expect(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth)).toBe(true);
  await page.screenshot({path:resolve(evidence,'strategies-en-zoom200-repeat.png'),fullPage:true});
  await page.evaluate(()=>{document.documentElement.style.zoom='';});
  await page.getByLabel('Animation',{exact:true}).selectOption('slow');
  for(let i=0;i<4;i++){await page.getByRole('button',{name:/Hide strategy list|Show strategy list/,exact:true}).click();}
  await firstRow.click();await page.locator(`.strategy-row[href^="/strategies/${second.strategy_id}"]`).click();await expect(heading(page)).toHaveText(second.name);
  await page.getByRole('tab',{name:'Settings',exact:true}).click();
  await page.getByText('Technical details',{exact:true}).click();await page.getByText('Technical details',{exact:true}).click();
  await page.getByLabel('Animation',{exact:true}).selectOption('off');
  await page.getByRole('link',{name:'Backtests',exact:true}).click();await expect(page.getByLabel('Animation',{exact:true})).toHaveValue('off');
  for(const locale of ['ru','en']) {
    await page.getByRole('link',{name:locale==='ru'?'Русский':'English',exact:true}).click();
    for(const width of [820,1024,1440]) {await page.setViewportSize({width,height:1000});await page.screenshot({path:resolve(evidence,`backtests-${locale}-${width}.png`)});}
    await page.evaluate(()=>{document.documentElement.style.zoom='2';});
    expect(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth)).toBe(true);
    await page.screenshot({path:resolve(evidence,`backtests-${locale}-zoom200.png`),fullPage:true});
    await page.evaluate(()=>{document.documentElement.style.zoom='';});
  }
  await page.getByRole('link',{name:'Strategies',exact:true}).click();await expect(page.getByLabel('Animation',{exact:true})).toHaveValue('off');
  await page.emulateMedia({reducedMotion:'reduce'});expect(await page.evaluate(()=>getComputedStyle(document.documentElement).getPropertyValue('--motion-duration').trim())).toMatch(/^0(?:ms|s)$/);await firstRow.click();await expect(heading(page)).toHaveText(first.name);
  expect(commands).toEqual([]);expect(errors).toEqual([]);
  writeFileSync(resolve(evidence,'real-observations.json'),JSON.stringify({first:first.strategy_id,second:second.strategy_id,realList:true,realDetail:true,standaloneProvenance:false,invalidReturnIgnored:true,absentSourceTruthful:true,keyboard:true,filterBackForward:true,viewportWidths:[820,1024,1440],locales:['ru','en'],zoom:200,axeViolations:0,commands,errors,httpErrors,consoleErrors},null,2));
});

test('all four presentation gates, classic/new/mode and locale reload',async({page})=>{
  await login(page);
  const strategy=await seed(page,40);
  for(const [offset,backtests,strategies] of [[0,true,true],[2,false,false],[4,true,false],[5,false,true]] as const){
    const base=`http://localhost:${port+offset}`;
    for(const [path,client] of [[`/strategies/${strategy.strategy_id}`,strategies],['/strategies',strategies],['/strategies?view=classic&view=client',strategies],['/strategies?mode=rl_ml&mode=dashboard',strategies],['/strategies?view=client&view=classic',false],['/backtests',backtests],['/strategies/new',false],['/strategies?mode=rl_ml',false],[`/strategies/${strategy.strategy_id}?view=classic`,false],['/strategies?view=classic',false]] as const){
      const response=await page.goto(base+path);expect(response?.headers()['cache-control']).toBe('private, no-store');
      await expect(page.locator('#platform-root')).toHaveCount(client?1:0);
      await page.reload();await expect(page.locator('#platform-root')).toHaveCount(client?1:0);
    }
    const from=strategies?'/strategies':'/backtests';
    if(backtests||strategies){await page.goto(base+from);await expect(page.locator('[data-platform-client]')).toBeVisible();await page.getByRole('link',{name:strategies?'Backtests':'Strategies',exact:true}).click();await expect(page.locator('#platform-root')).toHaveCount(backtests&&strategies?1:0);}
    await page.goto(`${base}/locale?${new URLSearchParams({locale:'ru',next:`/strategies/${strategy.strategy_id}?view=classic`})}`);
    expect(new URL(page.url()).searchParams.get('view')).toBe('classic');await expect(page.locator('#platform-root')).toHaveCount(0);
    await page.goto(`${base}/locale?${new URLSearchParams({locale:'en',next:'/strategies'})}`);
  }
});

test('controlled optional errors, stale spec, late identity, cooldown, logout and session boundary',async({page})=>{
  await login(page);const first=await seed(page,60),second=await seed(page,70);
  const dashboard='**/api/ui/strategies/dashboard?*';
  const path=`**/api/strategies/${first.strategy_id}`;
  await page.goto(`/strategies/${first.strategy_id}`);await expect(heading(page)).toHaveText(first.name);
  await page.route(path,route=>route.fulfill({status:503,json:{}}));
  await page.getByRole('button',{name:'Refresh',exact:true}).click();
  await expect(page.locator('.strategy-detail>[role=alert]')).toBeVisible();await expect(heading(page)).toHaveText(first.name);
  await page.unroute(path);
  for(const status of [403,404,200]){
    await page.route(path,route=>route.fulfill({status,json:status===200?{invalid:true}:{}}));await page.reload();
    await expect(page.locator('.strategy-detail>[role=alert]')).toBeVisible();await expect(heading(page)).not.toHaveText(first.name);await page.unroute(path);
  }
  await page.route(dashboard,route=>route.fulfill({status:429,headers:{'Retry-After':'60'},json:{}}));await page.reload();
  await expect(heading(page)).toHaveText(first.name);await expect(page.locator('.operations-workspace')).toBeVisible();
  await expect(page.locator('.strategies-refresh')).toBeEnabled();let cooldownReads=0;page.on('request',request=>{if(request.url().includes('/api/ui/strategies/dashboard'))cooldownReads++;});await page.locator('.strategies-refresh').click();await expect(page.locator('.strategies-refresh')).toHaveText('Refresh');expect(cooldownReads).toBe(0);await page.unrouteAll({behavior:'wait'});
  await page.route(dashboard,route=>route.fulfill({status:503,json:{}}));await page.reload();await expect(heading(page)).toHaveText(first.name);
  await expect(page.locator('.operations-workspace')).toBeVisible();await page.unrouteAll({behavior:'wait'});
  await page.route(dashboard,async route=>{const response=await route.fetch();const body=await response.json();body.selected_strategy.strategy_id=second.strategy_id;await route.fulfill({json:body});});
  await page.reload();await expect(heading(page)).toHaveText(first.name);await expect(page.locator('.operations-workspace')).toBeVisible();await page.unrouteAll({behavior:'wait'});
  await page.route(dashboard,async route=>{const response=await route.fetch();const body=await response.json();body.live_profile.state='unavailable';body.live_profile.readiness_status='ready';body.compatibility_readiness.state='empty';body.compatibility_readiness.compatibility_state='launchable';body.runtime_status.state='ready';body.runtime_status.producer_status='blocked';await route.fulfill({json:body});});
  await page.reload();await expect(heading(page)).toHaveText(first.name);
  await expect(page.locator('.operations-workspace')).toBeVisible();await page.unrouteAll({behavior:'wait'});
  await page.route(dashboard,async route=>{const response=await route.fetch();const body=await response.json();if(new URL(route.request().url()).searchParams.get('strategy_id')===first.strategy_id)await new Promise(resolve=>setTimeout(resolve,700));await route.fulfill({json:body});});
  await page.route(path,async route=>{await new Promise(resolve=>setTimeout(resolve,700));try{await route.continue();}catch(error){if(!page.url().includes(second.strategy_id))throw error;}});
  await page.goto(`/strategies/${first.strategy_id}`);await page.locator(`.strategy-row[href^="/strategies/${second.strategy_id}"]`).click();await expect(heading(page)).toHaveText(second.name);
  await page.waitForTimeout(800);await expect(heading(page)).toHaveText(second.name);await page.unroute(path);await page.unrouteAll({behavior:'wait'});
  await page.route('**/api/auth/current-user',route=>route.fulfill({json:{user_id:randomUUID(),paid_level:'free'}}));await page.reload();await expect(page.getByRole('alert')).toContainText('account has changed');await expect(page.locator('.strategy-row')).toHaveCount(0);await page.unroute('**/api/auth/current-user');
  await page.goto(`/strategies/${first.strategy_id}`);await expect(heading(page)).toHaveText(first.name);
  await page.route(path,route=>route.fulfill({status:401,json:{}}));await page.getByRole('button',{name:'Refresh',exact:true}).click();
  await expect(page.getByRole('link',{name:'Sign in',exact:true})).toBeVisible();await expect(page.locator('.strategy-row')).toHaveCount(0);await expect(page.locator('.strategy-detail')).toHaveCount(0);await page.unroute(path);
  await page.reload();await expect(heading(page)).toHaveText(first.name);await page.getByRole('link',{name:'Sign out',exact:true}).click();await expect(page).toHaveURL(/\/login/);await page.goBack();await expect(page.locator('.strategy-detail')).toHaveCount(0);
});


test('long library keeps its scroll, filters and selected identity across Back/Forward',async({page})=>{
  await login(page);
  const prior=await (await page.request.get('/api/strategies')).json();
  const strategies=[];
  for(let index=0;index<14;index++)strategies.push(await seed(page,100+index));
  await page.reload();
  const library=page.locator('#strategies-library');
  await expect(page.locator('.strategy-row')).toHaveCount(prior.length+14);
  const first=strategies[0],second=strategies[1];
  await page.locator(`.strategy-row[href^="/strategies/${first.strategy_id}"]`).click();
  await expect(heading(page)).toHaveText(first.name);
  await library.evaluate(element=>{element.scrollTop=element.scrollHeight;});
  const scroll=await library.evaluate(element=>element.scrollTop);
  expect(scroll).toBeGreaterThan(0);
  // Navigate with the actual keyboard link while keeping the list's viewport.
  await page.locator(`.strategy-row[href^="/strategies/${second.strategy_id}"]`).focus();
  await page.keyboard.press('Enter');await expect(heading(page)).toHaveText(second.name);
  const selectedScroll=await library.evaluate(element=>element.scrollTop);
  await page.goBack();await expect(heading(page)).toHaveText(first.name);
  expect(await library.evaluate(element=>element.scrollTop)).toBe(selectedScroll);
  await page.goForward();await expect(heading(page)).toHaveText(second.name);
  expect(await library.evaluate(element=>element.scrollTop)).toBe(selectedScroll);
});

test('strategy list animates intermediate geometry and reverses without scaling text',async({page})=>{
 await login(page);const strategy=await seed(page,20);await page.goto(`/strategies/${strategy.strategy_id}`);await expect(heading(page)).toHaveText(strategy.name);
 await page.getByLabel('Animation',{exact:true}).selectOption('slow');
 const observations=[];
 for(const width of [1440,820]){
  await page.setViewportSize({width,height:1000});
  const samples=await page.evaluate(async()=>{
   const button=document.querySelector<HTMLButtonElement>('.history-toggle')!;
   const slot=document.querySelector<HTMLElement>('.strategy-list-slot')!;
   const library=document.querySelector<HTMLElement>('#strategies-library')!;
   const frames:{time:number;size:number;textWidth:number;overflow:boolean}[]=[];
   const measure=(time:number)=>frames.push({time,size:innerWidth>928?slot.getBoundingClientRect().width:slot.getBoundingClientRect().height,textWidth:library.getBoundingClientRect().width,overflow:document.documentElement.scrollWidth>innerWidth});
   measure(0);button.focus();button.click();const start=performance.now();let reversed=false;
   await new Promise<void>(resolve=>{function tick(now:number){const elapsed=now-start;measure(elapsed);if(elapsed>=180&&!reversed){button.click();reversed=true;}if(elapsed<850)requestAnimationFrame(tick);else resolve();}requestAnimationFrame(tick);});
   return frames;
  });
  const initial=samples[0]!.size;
  expect(samples.some(frame=>frame.size<initial-5&&frame.size>5)).toBe(true);
  expect(Math.abs(samples.at(-1)!.size-initial)).toBeLessThan(1);
  expect(samples.every(frame=>!frame.overflow)).toBe(true);
  expect(Math.max(...samples.map(f=>f.textWidth))-Math.min(...samples.map(f=>f.textWidth))).toBeLessThan(1);
  await expect(page.getByRole('button',{name:'Hide strategy list',exact:true})).toBeFocused();
  observations.push({width,samples});
 }
 mkdirSync(evidence,{recursive:true});writeFileSync(resolve(evidence,'list-motion-frames.json'),JSON.stringify(observations,null,2));
 await page.getByLabel('Animation',{exact:true}).selectOption('off');await page.getByRole('button',{name:'Hide strategy list',exact:true}).click();await expect(page.locator('.strategy-list-slot')).toHaveAttribute('inert','');
});
