import {test,expect} from '@playwright/test';
import AxeBuilder from '@axe-core/playwright';
import {readFileSync,writeFileSync,mkdirSync,rmSync} from 'node:fs';
import {resolve} from 'node:path';
const root=resolve(import.meta.dirname,'../../..');
const state=resolve(root,process.env.ROEHUB_PROOF_STATE??'.local_artifacts/backtests-client');
const evidence=resolve(root,process.env.ROEHUB_PROOF_EVIDENCE??'.codex/delivery/evidence/roehub-strategies-client-2026-09-12/operations');
test('operational strategy: executed chart, position, commands, reasons, layouts and keyboard',async({page,request})=>{
 test.setTimeout(120000);mkdirSync(evidence,{recursive:true});
 await page.goto('/strategies');const credentials=JSON.parse(readFileSync(resolve(state,'credentials.json'),'utf8'));
 const form=page.locator('[data-password-login]');await form.locator('..').locator('summary').click();await form.locator('[name=username]').fill(credentials.username);await form.locator('[name=password]').fill(credentials.password);await form.locator('button[type=submit]').click();await expect(page.locator('[data-platform-client]')).toBeVisible();
 const response=await page.request.post('/api/strategies',{data:{instrument_id:{market_id:1,symbol:'BTCUSDT'},instrument_key:'binance:spot:BTCUSDT',market_type:'spot',timeframe:'15m',indicators:[{name:'MA',params:{fast:20,slow:50}}],signal_template:'MA(20,50)'}});expect(response.status()).toBe(201);const strategy=await response.json();
 writeFileSync(resolve(state,'execution-demo-strategy.txt'),strategy.strategy_id);rmSync(resolve(state,'execution-demo-state.json'),{force:true});
 const errors:string[]=[];page.on('pageerror',e=>errors.push(e.message));
 await page.goto(`/strategies/${strategy.strategy_id}`);await expect(page.getByRole('button',{name:'Stop',exact:true})).toBeEnabled();
 await expect(page.locator('.operations-chart canvas')).toBeVisible();await expect(page.getByRole('tab',{name:'Trades · 7'})).toBeVisible();await expect(page.getByRole('tab',{name:'Backtest',exact:true})).toHaveCount(0);
 const geometry=async()=>page.locator('.operations-technical').evaluate(el=>({top:el.getBoundingClientRect().top,scroll:window.scrollY,height:document.documentElement.scrollHeight}));
 const baseline=await geometry();
 const eventTabWidth=await page.locator('#operations-events').evaluate(el=>el.getBoundingClientRect().width);
 await page.getByRole('tab',{name:'Equity',exact:true}).click();
 const benchmark=page.getByRole('button',{name:'Buy & Hold',exact:true});
 await expect(benchmark).toHaveAttribute('aria-pressed','true');await benchmark.click();await expect(benchmark).toHaveAttribute('aria-pressed','false');await benchmark.click();
 await page.screenshot({path:resolve(evidence,'equity-controls-footer.png'),fullPage:true});
 await page.getByRole('tab',{name:'Price & executions',exact:true}).click();

 for(const name of ['Equity','Drawdown','Trades · 7','Events','Price & executions']){
  await page.getByRole('tab',{name:new RegExp('^'+name.replace(/[.*+?^${}()|[\]\\]/g,'\\$&'))}).click();
  await expect.poll(geometry).toEqual(baseline);
  expect(await page.locator('#operations-events').evaluate(el=>el.getBoundingClientRect().width)).toBe(eventTabWidth);
  await expect(page.locator('.operations-position')).toHaveCount(name==='Price & executions'?1:0);
  await expect(page.locator('.operations-lifecycle')).toHaveCount(name==='Price & executions'?1:0);
  await page.locator('.overview-expand').click();
  await expect(page.locator('.overview-expanded')).toBeVisible();
  if(name==='Events'||name==='Trades · 7'){
   expect(await page.locator('.operations-secondary').evaluate(el=>el.scrollHeight<=el.clientHeight+1)).toBe(true);
   for(const table of await page.locator('.operations-secondary .table-scroll').all()){
    expect(await table.evaluate(el=>el.scrollHeight<=el.clientHeight+1)).toBe(true);
   }
  }
  await page.screenshot({path:resolve(evidence,`expanded-${name.split(' ')[0]}.png`),fullPage:true});
  await page.keyboard.press('Escape');
  await expect(page.locator('.overview-expanded')).toHaveCount(0);
  await expect(page.locator('.overview-expand')).toBeFocused();
  await expect.poll(geometry).toEqual(baseline);
 }

 for(const expanded of [false,true]){
  if(expanded)await page.locator('.overview-expand').click();
  await page.getByRole('tab',{name:/Trades ·/}).click();
  const tableOrigin=await page.locator('.operations-trades').evaluate(el=>({x:el.getBoundingClientRect().x,y:el.getBoundingClientRect().y}));
  await page.getByRole('tab',{name:/Events/}).click();
  expect(await page.locator('.operations-events').evaluate(el=>({x:el.getBoundingClientRect().x,y:el.getBoundingClientRect().y}))).toEqual(tableOrigin);
  await expect(page.locator('#operations-detail table')).toHaveCount(1);
  await expect(page.locator('#operations-detail .freshness')).toHaveCount(0);
  await expect(page.locator('.operations-events tbody tr')).toHaveCount(5);
  await page.getByRole('button',{name:'Next page',exact:true}).click();
  await expect(page.locator('.operations-events tbody tr')).toHaveCount(5);
  await page.getByRole('button',{name:'Previous page',exact:true}).click();
  expect(await page.locator('.operations-secondary').evaluate(el=>el.scrollHeight<=el.clientHeight+1)).toBe(true);
  await page.getByLabel('Rows per page',{exact:true}).selectOption('10');
  expect(await page.locator('.operations-secondary .table-scroll').evaluate(el=>el.scrollHeight<=el.clientHeight+1)).toBe(true);
  const toolsTop=await page.locator('.operations-trade-tools').evaluate(el=>el.getBoundingClientRect().top);
  const footerTop=await page.locator('.operations-event-pagination').evaluate(el=>el.getBoundingClientRect().top);
  await page.getByLabel('Rows per page',{exact:true}).selectOption('50');
  if(!expanded){
   await page.locator('.operations-secondary .table-scroll').evaluate(el=>{el.scrollTop=100;});
   expect(await page.locator('.operations-trade-tools').evaluate(el=>el.getBoundingClientRect().top)).toBe(toolsTop);
   expect(await page.locator('.operations-event-pagination').evaluate(el=>el.getBoundingClientRect().top)).toBe(footerTop);
  }
  await page.getByRole('button',{name:'Event type',exact:true}).click();await page.getByRole('checkbox',{name:'All events',exact:true}).click();await page.getByRole('checkbox',{name:'Exit',exact:true}).click();await page.getByRole('checkbox',{name:'Exit',exact:true}).press('Escape');
  await expect(page.locator('.operations-events tbody tr')).toHaveCount(6);
  await page.getByRole('button',{name:'Reason',exact:true}).click();await page.getByRole('checkbox',{name:'All reasons',exact:true}).click();await page.getByRole('checkbox',{name:'Manual',exact:true}).click();await page.getByRole('checkbox',{name:'Manual',exact:true}).press('Escape');
  await expect(page.locator('.operations-events tbody tr')).toHaveCount(1);
  await page.getByRole('button',{name:'Event type',exact:true}).click();await page.getByRole('checkbox',{name:'All events',exact:true}).click();await page.getByRole('checkbox',{name:'All events',exact:true}).press('Escape');
  await page.getByRole('button',{name:'Reason',exact:true}).click();await page.getByRole('checkbox',{name:'All reasons',exact:true}).click();await page.getByRole('checkbox',{name:'All reasons',exact:true}).press('Escape');
  await expect(page.locator('.operations-events tbody tr')).toHaveCount(14);
  await page.screenshot({path:resolve(evidence,`events-aligned-${expanded}.png`),fullPage:true});
  await page.getByLabel('Rows per page',{exact:true}).selectOption('5');
  if(expanded)await page.locator('.overview-expand').click();
 }
 await expect(page.getByRole('button',{name:'Restart',exact:true})).toHaveCount(0);
 await expect(page.getByRole('tab',{name:'Settings',exact:true})).toHaveCount(0);

 await page.getByRole('tab',{name:/Events/}).click();await expect(page.locator('.events-unread')).toHaveCount(0);
 await page.getByRole('tab',{name:/Trades ·/}).click();
 await page.locator('.strategy-filter-panel summary').click();await page.getByRole('button',{name:'State',exact:true}).click();await page.getByRole('checkbox',{name:'All',exact:true}).click();await page.getByRole('checkbox',{name:'Stopped',exact:true}).click();await page.getByRole('checkbox',{name:'Stopped',exact:true}).press('Escape');await expect(page.locator('.strategy-row')).toHaveCount(0);
 await page.getByRole('button',{name:'State',exact:true}).click();await page.getByRole('checkbox',{name:'All',exact:true}).click();await page.getByRole('checkbox',{name:'All',exact:true}).click();await page.getByRole('checkbox',{name:'Running',exact:true}).click();await page.getByRole('checkbox',{name:'Running',exact:true}).press('Escape');await expect(page.locator('.strategy-row')).toHaveCount(1);await expect(page.locator('.strategy-row-status')).toHaveClass(/is-running/);
 await page.getByRole('button',{name:'State',exact:true}).click();await page.getByRole('checkbox',{name:'All',exact:true}).click();await page.getByRole('checkbox',{name:'All',exact:true}).press('Escape');await page.locator('.strategy-filter-panel summary').click();


 const firstCell=page.locator('.operations-trades tbody tr').first().locator('td').first();expect((await firstCell.boundingBox())!.width).toBeLessThan(130);
 await expect(page.locator('.operations-trades tbody').getByText(/\d{4}-\d{2}-\d{2} \d{2}:\d{2}/).first()).toBeVisible();
 await expect(page.locator('.operations-chart')).toBeHidden();
 await page.getByRole('tab',{name:'Price & executions',exact:true}).click();
 await expect(page.locator('.operations-trades')).toBeHidden();
 await page.getByRole('button',{name:'Chart display',exact:true}).click();
 for(const name of ['Entry price','Stop loss','Take profit']){const toggle=page.getByRole('checkbox',{name,exact:true});await toggle.uncheck();await expect(toggle).not.toBeChecked();await toggle.check();}
 await expect(page.getByRole('group',{name:'Chart display'})).toBeVisible();
 await page.keyboard.press('Escape');await expect(page.getByRole('button',{name:'Chart display',exact:true})).toBeFocused();
 await expect(page.getByRole('group',{name:'Chart display'})).toHaveCount(0);
 const timeframe=page.getByRole('button',{name:'Chart timeframe',exact:true});
 const chooseTimeframe=async(value:string)=>{await timeframe.click();await page.getByRole('radio',{name:value,exact:true}).click();};
 await chooseTimeframe('1h');await expect(timeframe).toHaveText('1h');
 await expect(page.locator('.operations-chart canvas')).toBeVisible();
 await page.locator('.overview-expand').click();await expect(timeframe).toHaveText('1h');await page.locator('.overview-expand').click();
 await chooseTimeframe('15m');
 await page.getByRole('button',{name:'Chart display',exact:true}).click();
 const markerToggle=page.getByRole('checkbox',{name:'Trades',exact:true});
 await expect(markerToggle).toBeChecked();await markerToggle.click({position:{x:(await markerToggle.boundingBox())!.width-5,y:16}});await expect(markerToggle).not.toBeChecked();await markerToggle.getByText('Trades',{exact:true}).click();await expect(markerToggle).toBeChecked();
 await page.screenshot({path:resolve(evidence,'chart-display-menu.png'),fullPage:true});
 await page.keyboard.press('Escape');
 const chart=page.locator('.operations-chart');await chart.scrollIntoViewIfNeeded();const box=(await chart.boundingBox())!;
 await page.mouse.move(box.x+box.width*.6,box.y+box.height*.4);await page.keyboard.down('Control');await page.mouse.wheel(0,-500);await page.keyboard.up('Control');
 await page.screenshot({path:resolve(evidence,'operations-chart-zoom.png'),fullPage:true});
 await page.getByRole('tab',{name:/Trades ·/}).click();await page.locator('.operations-trades button').last().click();await expect(page.locator('.operations-trade-detail tbody tr')).toHaveCount(3);await expect(page.getByRole('columnheader',{name:'Slippage · %',exact:true})).toBeVisible();
 await page.getByRole('tab',{name:'Price & executions',exact:true}).click();
 await page.route(`**/api/strategies/${strategy.strategy_id}/manual-exit`,async route=>{await route.fetch();await route.fulfill({status:503,json:{}});},{times:1});
 await page.getByRole('button',{name:'Close position',exact:true}).click();await page.getByRole('button',{name:'Confirm',exact:true}).click();await expect(page.getByRole('button',{name:'Check outcome',exact:true})).toBeVisible();await page.reload();await expect(page.getByRole('button',{name:'Open trade',exact:true})).toBeDisabled();await page.getByRole('button',{name:'Check outcome',exact:true}).click();await expect(page.getByText('No open position',{exact:true})).toBeVisible();
 await page.getByRole('button',{name:'Open trade',exact:true}).click();await page.getByLabel('Order amount · quote currency').fill('1000');await page.getByRole('button',{name:'Confirm',exact:true}).click();await expect(page.getByRole('button',{name:'Close position',exact:true})).toBeEnabled();
 await expect(page.getByRole('tab',{name:'Trades · 8'})).toBeVisible();
 await page.getByRole('button',{name:'Stop',exact:true}).click();await page.getByRole('button',{name:'Confirm',exact:true}).click();await expect(page.getByRole('button',{name:'Start',exact:true})).toBeEnabled();
 await expect(page.getByText('No open position',{exact:true})).toHaveCount(0);
 await page.getByRole('button',{name:'Start',exact:true}).click();await page.getByRole('button',{name:'Confirm',exact:true}).click();await expect(page.getByRole('button',{name:'Stop',exact:true})).toBeEnabled();
 for(const locale of ['en','ru']){await page.getByRole('link',{name:locale==='en'?'English':'Русский',exact:true}).click();for(const width of [820,1024,1440]){await page.setViewportSize({width,height:1000});await expect(page.locator('.operations-chart canvas')).toBeVisible();expect(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth)).toBe(true);await page.screenshot({path:resolve(evidence,`operations-${locale}-${width}.png`),fullPage:true});expect((await new AxeBuilder({page}).analyze()).violations).toEqual([]);}}
 expect((await request.post(`/api/strategies/${strategy.strategy_id}/run`)).status()).toBe(401);
 await page.getByRole('link',{name:'English',exact:true}).click();
 await page.getByRole('button',{name:'Close position',exact:true}).click();await page.getByRole('button',{name:'Confirm',exact:true}).click();await expect(page.getByText('No open position',{exact:true})).toBeVisible();
 await page.getByRole('button',{name:'Stop',exact:true}).click();await page.getByRole('button',{name:'Confirm',exact:true}).click();await expect(page.getByRole('button',{name:'Start',exact:true})).toBeEnabled();
 await page.getByRole('button',{name:'Delete',exact:true}).click();await page.getByRole('button',{name:'Confirm',exact:true}).click();await expect(page).toHaveURL(/\/strategies$/);await expect(page.locator('.strategy-row')).toHaveCount(0);
 expect(errors).toEqual([]);writeFileSync(resolve(evidence,'observations.json'),JSON.stringify({strategy_id:strategy.strategy_id,source:'synthetic_demo',closedAndReopened:true,stoppedWithoutClosing:true,resumed:true,deleted204:true,unauthenticatedCommand401:true,realProviderCommands:0,consoleErrors:errors}));
});
