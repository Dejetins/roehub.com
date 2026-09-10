import '@testing-library/jest-dom/vitest';
import { afterEach, expect, it, vi } from 'vitest';
import { cleanup, fireEvent, render, screen, waitFor, within } from '@testing-library/react';
import { MemoryRouter, useLocation } from 'react-router';
import { QueryClientProvider } from '@tanstack/react-query';
import { I18nextProvider } from 'react-i18next';
import { createI18n } from './i18n';
import { createQueryClient } from './query-client';
import { Results, chartDate, monthlyReturns } from './results';
const charts=vi.hoisted(()=>({options:[] as any[]}));
vi.mock('echarts/core',()=>({use:vi.fn(),connect:vi.fn(),init:()=>({setOption:(o:unknown)=>charts.options.push(o),resize:vi.fn(),dispose:vi.fn()})}));
const job='10000000-0000-4000-8000-000000000001';
const variants=[{canonical_variant_params:{execution:{initial_cash_quote:1000}},rank:1,variant_key:'v1',variant_hash:'h1',summary_metrics:{total_return_pct:20,max_drawdown_pct:5,trade_count:40},readable_params:{indicators:[{indicator_id:'ma.ema',window:5}]}},{rank:2,variant_key:'v2',variant_hash:'h2',summary_metrics:{total_return_pct:10,max_drawdown_pct:2,trade_count:80},readable_params:{indicators:[{indicator_id:'ma.ema',window:10}]}}];
function Location(){return <output aria-label="location">{useLocation().search}</output>;}
function setup(items=variants){
 charts.options=[];
 vi.spyOn(HTMLElement.prototype,'clientWidth','get').mockReturnValue(900);
 vi.spyOn(HTMLElement.prototype,'clientHeight','get').mockReturnValue(320);
 vi.stubGlobal('ResizeObserver',class {observe(){} disconnect(){}});
 vi.stubGlobal('fetch',vi.fn(async(url:string|URL)=>{
 const path=new URL(url).pathname;const variant=path.includes('/v2')?'v2':'v1';
 const data=path.endsWith('/candles')?{job_id:job,variant_key:variant,timeframe:new URL(url).searchParams.get('timeframe')??'15m',source_bars:2,group_size:1,candles:[{time:'2026-02-27T00:00:00Z',open:100,high:120,low:90,close:110},{time:'2026-02-27T00:15:00Z',open:110,high:125,low:105,close:120}]}:path.endsWith('/trades')?{job_id:job,variant_key:variant,items:[{trade_index:0,entry_timestamp:'2026-02-27T00:00:00Z',exit_timestamp:'2026-02-27T00:15:00Z',side:'long',entry_price:100,exit_price:120}],pagination:{page:1,page_size:100,total:1,has_previous:false,has_next:false}}:path.endsWith('/monthly-stats')?{job_id:job,variant_key:variant,kind:'monthly',items:[{month:'2025-12',net_pnl_quote:100,trades_count:1,return_pct:999},{month:'2026-01',net_pnl_quote:-100,trades_count:2,return_pct:999}],bounds:{truncated:false,source_items:2,returned_items:2}}:path.endsWith('/top')?{items}:path.endsWith('/summary')?{}:path.endsWith('/equity')||path.endsWith('/drawdown')?{job_id:job,variant_key:variant,kind:path.endsWith('/equity')?'equity':'drawdown',points:[{x:'2026-02-27',value:100},{x:'2026-02-28',value:110}],returned_points:2,source_points:2,downsampled:false}:path.endsWith('/v1')?variants[0]:path.endsWith('/v2')?variants[1]:{};
 return new Response(JSON.stringify(data),{status:path.endsWith('/summary')?503:200});
 }));
 render(<I18nextProvider i18n={createI18n('en')}><QueryClientProvider client={createQueryClient()}><MemoryRouter initialEntries={['/?variant=v1']}><Results job={job} subject="actor" now={Date.now()}/><Location/></MemoryRouter></QueryClientProvider></I18nextProvider>);
}
afterEach(()=>{cleanup();vi.restoreAllMocks();vi.unstubAllGlobals();});
it('opens a readable variant link and sorts result rows without changing selection',async()=>{
 setup();await screen.findByRole('link',{name:'EMA 10'});
 fireEvent.click(screen.getByRole('button',{name:/Maximum drawdown/}));
 const table=screen.getByRole('region',{name:'Ranked variants'});
 expect(within(table).getAllByRole('link')[0]).toHaveTextContent('EMA 10');
 expect(screen.getByLabelText('location')).toHaveTextContent('variant=v1');
 fireEvent.click(screen.getByRole('link',{name:'EMA 10'}));
 expect(screen.getByLabelText('location')).toHaveTextContent('variant=v2');
 await screen.findByRole('tab',{name:'Metrics'});
 fireEvent.click(screen.getByRole('tab',{name:'Metrics'}));
 expect(screen.getByRole('columnheader',{name:'Metric'})).toBeVisible();
});
it('configures axis hover and zoom on both real report chart components',async()=>{
 setup();await waitFor(()=>expect(charts.options).toHaveLength(1));
 fireEvent.click(screen.getByRole('button',{name:'Drawdown · %'}));
 await waitFor(()=>expect(charts.options).toHaveLength(2));
 for(const option of charts.options){
 expect(option.tooltip).toMatchObject({trigger:'axis',renderMode:'richText',confine:true,axisPointer:{type:'cross'}});
 expect(option.dataZoom.map((v:any)=>v.type)).toEqual(['inside','slider']);
 expect(option.tooltip.valueFormatter(1234.567)).toBe((1234.567).toLocaleString(undefined,{maximumFractionDigits:2}));
 }
});

it('keeps distinct intraday points while formatting dates and toggles trade markers',async()=>{
 setup();await waitFor(()=>expect(charts.options).toHaveLength(1));
 const option=charts.options[0];
 const timestamp='2026-02-27T23:45:00Z';
 expect(option.xAxis.axisLabel.formatter(timestamp)).toBe(chartDate(timestamp,'en'));
 expect(option.xAxis.axisLabel.formatter(timestamp)).not.toContain('23:45');
 fireEvent.click(screen.getByRole('checkbox',{name:'Trade exits'}));
 await waitFor(()=>expect(charts.options).toHaveLength(2));
 expect(charts.options[1].series[1].type).toBe('scatter');
 expect(screen.queryByRole('tab',{name:'Symbol statistics'})).not.toBeInTheDocument();
});
it('uses opening monthly balance instead of adding individual trade returns, across years',()=>{
 const rows=monthlyReturns([{month:'2026-01',net_pnl_quote:-100,trades_count:2,return_pct:999},{month:'2025-12',net_pnl_quote:100,trades_count:1,return_pct:999}],1000);
 expect(rows.map(r=>r.month)).toEqual(['2025-12','2026-01']);
 expect(rows[0]?.percent).toBe(10);
 expect(rows[1]?.percent).toBeCloseTo(-100/1100*100);
 expect(monthlyReturns([{month:'2026-01',net_pnl_quote:100,trades_count:1,return_pct:999}])[0]?.percent).toBeNull();
});

it('renders years as rows and twelve month columns with one-decimal P&L and blanks',async()=>{
 setup();await screen.findByRole('tab',{name:'Monthly statistics'});
 fireEvent.click(screen.getByRole('tab',{name:'Monthly statistics'}));
 const matrix=await screen.findByRole('region',{name:'Monthly statistics'});
 expect(within(matrix).getAllByRole('columnheader')).toHaveLength(13);
 expect(within(matrix).getAllByRole('rowheader').map(v=>v.textContent)).toEqual(['2025','2026']);
 expect(within(matrix).getByText('10.0%')).toBeVisible();
 expect(within(matrix).getByText('-9.1%')).toBeVisible();
 expect(within(matrix).getByText('100.0')).toBeVisible();
 expect(within(matrix).getAllByText('—')).toHaveLength(22);
});

it('renders actual OHLC in ECharts candle order and entry/exit at execution prices',async()=>{
 setup();await screen.findByRole('button',{name:'Price & trades'});
 fireEvent.click(screen.getByRole('button',{name:'Price & trades'}));
 await waitFor(()=>expect(charts.options.some(o=>o.series[0].type==='candlestick'&&o.series[1]?.data.length===1)).toBe(true));
 const option=[...charts.options].reverse().find(o=>o.series[0].type==='candlestick');
 expect(option.dataZoom[0].filterMode).toBe('filter');
 expect(option.series[0].data).toEqual([[100,110,90,120],[110,120,105,125]]);
 expect(option.series[1].data[0].value).toEqual([0,100]);
 expect(option.series[2].data[0].value).toEqual([1,120]);
});

it('paginates ranked variants in groups of ten',async()=>{
 setup(Array.from({length:11},(_,i)=>({...variants[0]!,rank:i+1,variant_key:`v${i+1}`,readable_params:{indicators:[{indicator_id:'ma.ema',window:i+1}]}})));
 const ranking=await screen.findByRole('region',{name:'Ranked variants'});
 expect(within(ranking).getAllByRole('link')).toHaveLength(10);
 fireEvent.click(screen.getByRole('button',{name:'Next variants'}));
 expect(within(ranking).getAllByRole('link')).toHaveLength(1);
 expect(within(ranking).getByRole('link')).toHaveTextContent('EMA 11');
});

it('requests the selected candle timeframe independently of the backtest',async()=>{
 setup();await screen.findByRole('button',{name:'Price & trades'});
 fireEvent.click(screen.getByRole('button',{name:'Price & trades'}));
 fireEvent.change(screen.getByRole('combobox',{name:'Chart timeframe'}),{target:{value:'1h'}});
 await waitFor(()=>expect(vi.mocked(fetch).mock.calls.some(([url])=>String(url).includes('/candles?timeframe=1h&max_bars=60000'))).toBe(true));
 expect(screen.getByLabelText('location')).toHaveTextContent('variant=v1');
});
it('expands the mounted overview, switches all charts and restores focus on Escape',async()=>{
 setup();await waitFor(()=>expect(charts.options).toHaveLength(1));
 const button=screen.getByRole('button',{name:'Full screen'});
 expect(button.textContent).toBe('');
 expect(button).toHaveAttribute('title','Full screen');
 const chart=screen.getByRole('img',{name:'Equity'});
 fireEvent.click(button);
 expect(screen.getByRole('dialog',{name:'Overview'})).toContainElement(chart);
 expect(screen.getByRole('img',{name:'Equity'})).toBe(chart);
 expect(document.body.style.overflow).toBe('hidden');
 expect(document.querySelector('.result-tabs')).toHaveProperty('inert',true);
 fireEvent.click(screen.getByRole('button',{name:'Drawdown · %'}));
 await screen.findByRole('img',{name:'Drawdown · %'});
 fireEvent.click(screen.getByRole('button',{name:'Price & trades'}));
 const price=await screen.findByRole('img',{name:'Price & trades'});
 fireEvent.change(screen.getByLabelText('Chart timeframe'),{target:{value:'1h'}});
 await screen.findByText('1h · 2 candles');
 fireEvent.keyDown(screen.getByRole('dialog',{name:'Overview'}),{key:'Escape'});
 expect(screen.queryByRole('dialog',{name:'Overview'})).not.toBeInTheDocument();
 expect(screen.getByRole('img',{name:'Price & trades'})).toBe(price);
 expect(screen.getByLabelText('Chart timeframe')).toHaveValue('1h');
 expect(document.body.style.overflow).not.toBe('hidden');
 expect(screen.getByRole('button',{name:'Full screen'})).toHaveFocus();
 expect(document.querySelector('.result-tabs')).not.toHaveProperty('inert',true);
});
