import { test, expect, type Page } from '@playwright/test';
import AxeBuilder from '@axe-core/playwright';
import { readFileSync, writeFileSync, mkdirSync } from 'node:fs';
import { resolve } from 'node:path';

const root=resolve(import.meta.dirname,'../../..');
const evidence=resolve(root,'.codex/delivery/evidence/roehub-backtests-compact-2026-09-08');
test.use({timezoneId:'America/New_York'});
async function signIn(page:Page) {
  mkdirSync(evidence,{recursive:true});
  await page.goto('/backtests/new');
  const c=JSON.parse(readFileSync(resolve(root,'.local_artifacts/backtests-client/credentials.json'),'utf8'));
  const form=page.locator('[data-password-login]');
  await form.locator('..').locator('summary').click();
  await form.locator('[name=username]').fill(c.username);
  await form.locator('[name=password]').fill(c.password);
  await form.locator('button[type=submit]').click();
  await expect(page.getByLabel('Job label (optional)')).toBeVisible();
}

test('shared axes, compact short-value controls, date-only fields and accessible responsive layout',async({page})=>{
  await signIn(page);
  const errors:string[]=[];page.on('pageerror',()=>errors.push('pageerror'));
  page.on('console',m=>{if(m.type()==='error')errors.push('console error');});
  const observations=[];
  for(const locale of ['en','ru']) {
    await page.getByRole('link',{name:locale==='ru'?'Русский':'English',exact:true}).click();
    await expect(page.locator('.builder-fields')).toBeVisible();
    for(const width of [390,820,1024,1440,1672]) {
      await page.setViewportSize({width,height:1000});
      const dims=await page.locator('.builder-fields').evaluate(root=>{
        const fields=Array.from(root.querySelectorAll<HTMLElement>('input:not([type=checkbox]),select,.source-picker>summary'));
        const boxes=fields.map(e=>({id:e.id,x:e.getBoundingClientRect().x,y:e.getBoundingClientRect().y,width:e.getBoundingClientRect().width}));
        const axes=[...new Set(boxes.map(b=>Math.round(b.x)))];
        return {inner:innerWidth,scroll:document.documentElement.scrollWidth,boxes,axes,
          min:Math.min(...boxes.map(b=>b.width)),max:Math.max(...boxes.map(b=>b.width))};
      });
      expect(dims.scroll).toBeLessThanOrEqual(dims.inner);
      const compact=dims.boxes.filter(b=>b.id==='field-timeframe' || b.id==='field-top_n' || b.id==='field-execution.fee_rate' || b.id==='field-execution.slippage_rate' || b.id.includes('.window.'));
      expect(compact).toHaveLength(7);
      expect(Math.max(...compact.map(b=>b.width))).toBeLessThanOrEqual(100);
      expect(Math.max(...compact.map(b=>b.width))-Math.min(...compact.map(b=>b.width))).toBeLessThan(1);
      expect(dims.boxes.find(b=>b.id==='field-coordinates.symbol')!.width).toBeGreaterThan(compact[0].width);
      if(width>=1440)expect(dims.axes).toHaveLength(5);
      expect(await page.locator('input[type=date]').count()).toBe(2);
      expect(await page.locator('input[type=time],input[type=datetime-local]').count()).toBe(0);
      expect((await new AxeBuilder({page}).analyze()).violations).toEqual([]);
      const action=page.getByRole('button',{name:locale==='en'?'Check configuration':'Проверить параметры',exact:true});
      await action.focus();await expect(action).toBeFocused();
      await page.evaluate(()=>{(document.activeElement as HTMLElement)?.blur();scrollTo(0,0);});
      await page.screenshot({path:resolve(evidence,`${locale}-${width}.png`),fullPage:true});
      observations.push({locale,width,...dims,axe:0});
    }
  }
  expect(errors).toEqual([]);
  writeFileSync(resolve(evidence,'layout.json'),JSON.stringify({observations,errors},null,2));
});

test('sources, ordered indicators, risk levels and sizing remain editable in compact groups',async({page})=>{
  await signIn(page);
  await page.getByLabel('Indicator',{exact:true}).selectOption('ma.ema');
  await page.locator('.source-picker summary').click();
  const source=page.getByRole('checkbox',{name:'close',exact:true});
  await source.uncheck();await source.check();
  await page.locator('.source-picker summary').click();
  await page.getByRole('button',{name:'Add indicator',exact:true}).click();
  await expect(page.locator('.indicator-row')).toHaveCount(2);
  const rows=page.locator('.indicator-row');
  await rows.nth(1).getByLabel('Indicator',{exact:true}).selectOption('ma.vwma');
  await rows.nth(1).getByRole('button',{name:'Move up',exact:true}).click();
  await expect(rows.nth(0).getByLabel('Indicator',{exact:true})).toHaveValue('ma.vwma');
  await rows.nth(0).getByRole('button',{name:'Remove indicator',exact:true}).click();
  await expect(rows).toHaveCount(1);
  await expect(rows.getByLabel('Indicator',{exact:true})).toHaveValue('ma.ema');
  await page.getByLabel('Risk mode',{exact:true}).selectOption('tp_sl_grid');
  await page.getByRole('checkbox',{name:'Take profit',exact:true}).check();
  const risk=page.getByRole('region',{name:'Take profit',exact:true});
  await risk.getByLabel('Start (%)',{exact:true}).fill('1');
  await risk.getByLabel('Stop (%)',{exact:true}).fill('1');
  await risk.getByLabel('Step (%)',{exact:true}).fill('1');
  await page.getByLabel('Position sizing',{exact:true}).selectOption('fixed_quote');
  await page.getByLabel('Quote amount',{exact:true}).fill('1000');
  await page.locator('.builder-fields .policies>summary').click();
  await expect(page.locator('.builder-fields .policies dl')).toBeVisible();
  expect((await new AxeBuilder({page}).analyze()).violations).toEqual([]);
  await page.evaluate(()=>{(document.activeElement as HTMLElement)?.blur();scrollTo(0,0);});
  await page.screenshot({path:resolve(evidence,'expanded.png'),fullPage:true});
});

test('real date-only preflight and create preserve UTC boundaries and reach completed results',async({page})=>{
  test.setTimeout(180_000);
  await signIn(page);
  const failures:{path:string;status:number}[]=[];const errors:string[]=[];
  page.on('response',r=>{if(r.status()>=400)failures.push({path:new URL(r.url()).pathname,status:r.status()});});
  page.on('pageerror',()=>errors.push('pageerror'));
  page.on('console',m=>{if(m.type()==='error')errors.push('console error');});
  await page.getByLabel('Job label (optional)').fill('Compact console · date-only proof');
  await page.getByLabel('Timeframe',{exact:true}).selectOption('15m');
  await page.getByLabel('Indicator',{exact:true}).selectOption('ma.ema');
  await page.getByLabel('Start date',{exact:true}).fill('2026-03-26');
  await page.getByLabel('End date',{exact:true}).fill('2026-03-29');
  await page.getByLabel('Fee (%)',{exact:true}).fill('0.075');
  const checked=page.waitForResponse(r=>r.url().endsWith('/preflight'));
  await page.getByRole('button',{name:'Check configuration',exact:true}).click();
  const check=await checked;expect(check.status()).toBe(200);
  expect((await check.json()).errors).toHaveLength(0);
  const range={start:'2026-03-26T00:00:00Z',end:'2026-03-29T00:00:00Z'};
  expect(check.request().postDataJSON().time_range).toEqual(range);
  await expect(page.getByRole('button',{name:'Submit backtest',exact:true})).toBeEnabled();
  await expect(page.locator('.builder-review')).toHaveAttribute('open','');
  await page.screenshot({path:resolve(evidence,'preflight.png'),fullPage:true});
  const created=page.waitForResponse(r=>new URL(r.url()).pathname==='/api/backtests/jobs' && r.request().method()==='POST');
  await page.getByRole('button',{name:'Submit backtest',exact:true}).click();
  const response=await created;expect(response.status()).toBe(201);
  expect(response.request().postDataJSON().time_range).toEqual(range);
  expect(response.request().postDataJSON().execution.fee_rate).toBe(0.00075);
  const job=await response.json();
  await expect(page).toHaveURL(new RegExp(`/backtests/${job.job_id}`));
  await expect(page.locator('.job-status').getByText('Completed',{exact:true})).toBeVisible({timeout:120_000});
  await expect(page.getByRole('img',{name:'Equity',exact:true})).toBeVisible({timeout:30_000});
  await page.screenshot({path:resolve(evidence,'completed.png'),fullPage:true});
  expect(errors).toEqual([]);expect(failures).toEqual([]);
  writeFileSync(resolve(evidence,'journey.json'),JSON.stringify({jobId:job.job_id,browserTimezone:'America/New_York',range,preflight:200,create:201,completed:true,errors,failures},null,2));
});
