import {TradesScroll} from './trades-scroll';
import {Download,Save} from 'lucide-react';
import {DateRangeControl} from './date-range-control';
import {NavigatorTable} from './navigator-table';
import {chartPeriods,chartTimes,periodStart,periodZoom,usePeriodZoom,type ChartPeriod,type PeriodView} from './chart-period';
import {useChartInstance} from './chart-instance';
import {motionDuration} from './motion';
import { LoadingData, ReadStatus } from './loading-data';
import { savedStrategyHref } from './strategies-api';
import { MotionLink as Link, transitionUI } from './motion';
import { ExpandableOverview } from './expandable-overview';
import { useEffect, useLayoutEffect, useMemo, useRef, useState, type ReactNode } from 'react';
import { useQueryClient } from '@tanstack/react-query';
import { useSearchParams } from 'react-router';
import { useTranslation } from 'react-i18next';
import { z } from 'zod';
import { use, connect } from 'echarts/core';
import { LineChart, ScatterChart } from 'echarts/charts';
import { GridComponent, TooltipComponent, DataZoomComponent } from 'echarts/components';
import { CanvasRenderer } from 'echarts/renderers';
import { ApiError, requestJson, readSession } from './api';
import { ReadError, isRestricted } from './library';
import { storageWorks } from './recovery';
import { clearStrategyRecovery, loadStrategyRecovery, storeStrategyRecovery, type StrategyRecovery } from './strategy-recovery';
import * as api from './results-api';
import {PriceChart, usePriceRead} from './price-chart';
import {useReadSnapshot} from './read-snapshot';
use([LineChart, ScatterChart, GridComponent, TooltipComponent, DataZoomComponent, CanvasRenderer]);
export {useResultRead} from './result-read';
import {useResultRead} from './result-read';
function ReadState({query,children}:{query:ReturnType<typeof useResultRead<any>>;children:ReactNode}){
  const {t}=useTranslation();const data=query.data?.data;
  return <><ReadError error={query.error}/>{query.data?.status===202?(['queued','running','pending'].includes(data.status)?(!query.isFetching&&<LoadingData />):<p role="status" className="notice">{t('results.failed')}</p>):!isRestricted(query.error)&&children}</>;
}
export function Confirm({id,title,help,source,onConfirm,disabled=false,trigger,icon,triggerTitle,confirmText}:{id:string;title:string;help:string;source:string;onConfirm:()=>void;disabled?:boolean;trigger:string;icon?:ReactNode;triggerTitle?:string;confirmText?:string}){
  const {t}=useTranslation();const dialog=useRef<HTMLDialogElement>(null),button=useRef<HTMLButtonElement>(null);
  return <><button ref={button} className={icon?"library-icon-action":undefined} aria-label={trigger} title={triggerTitle??trigger} disabled={disabled} onClick={()=>transitionUI(()=>dialog.current?.showModal())}>{icon??trigger}</button><dialog ref={dialog} aria-labelledby={`${id}-title`} aria-describedby={`${id}-help`} onClose={()=>button.current?.focus()} onKeyDown={event=>{if(event.key!=='Tab')return;const buttons=event.currentTarget.querySelectorAll<HTMLButtonElement>('button:not(:disabled)');const first=buttons[0],last=buttons[buttons.length-1];if(event.shiftKey&&document.activeElement===first){event.preventDefault();last?.focus();}else if(!event.shiftKey&&document.activeElement===last){event.preventDefault();first?.focus();}}}><div className="panel-head"><h2 id={`${id}-title`}>{title}</h2></div><div className="filter-fields"><p id={`${id}-help`}>{help}</p><p className="identity">{source}</p><button autoFocus onClick={()=>transitionUI(()=>dialog.current?.close())}>{t('results.keep')}</button><button disabled={disabled} onClick={()=>{dialog.current?.close();onConfirm();}}>{confirmText??t(`results.${id==='save'?'confirmSave':'confirmDelete'}`)}</button></div></dialog></>;
}
export function DataTable({items,columns,label,descriptionId}:{items:Record<string,unknown>[];columns:string[];label:string;descriptionId?:string}){
  const {t,i18n}=useTranslation();return items.length?<div className="table-scroll" tabIndex={0} role="region" aria-label={label} aria-describedby={descriptionId}><table><caption className="sr-only">{label}</caption><thead><tr>{columns.map(c=><th key={c} scope="col">{t(`results.columns.${c}`)}</th>)}</tr></thead><tbody>{items.map((row,i)=><tr key={i}>{columns.map(c=><td key={c}>{typeof row[c]==='number'?(row[c] as number).toLocaleString(i18n.language,{maximumFractionDigits:6}):typeof row[c]==='string'?String(row[c]):'—'}</td>)}</tr>)}</tbody></table></div>:<p>{t('results.empty')}</p>;
}
export function chartDate(value:string|number,locale:string) {
  if(typeof value==='number'||!/^\d{4}-\d{2}-\d{2}/.test(value))return String(value);
  const date=new Date(value);return Number.isNaN(date.getTime())?'—':date.toLocaleDateString(locale,{day:'2-digit',month:'short',year:'numeric',timeZone:'UTC'});
}
export function monthlyReturns(items:z.infer<typeof api.statsSchema>['items'],initialCash?:number) {
  let balance=initialCash;
  return [...items].sort((a,b)=>(a.month??'').localeCompare(b.month??'')).map(row=>{
    const percent=balance!=null&&balance>0?row.net_pnl_quote/balance*100:null;
    if(balance!=null)balance+=row.net_pnl_quote;
    return {...row,percent};
  });
}
function MonthlyMatrix({data,initialCash}:{data:z.infer<typeof api.statsSchema>;initialCash?:number}) {
  const {t,i18n}=useTranslation();const rows=monthlyReturns(data.items,initialCash);
  const years=[...new Set(rows.flatMap(r=>r.month?.match(/^\d{4}-\d{2}$/)?[r.month.slice(0,4)]:[]))];
  const number=(n:number)=>n.toLocaleString(i18n.language,{minimumFractionDigits:1,maximumFractionDigits:1});
  return <><p className="muted report-explanation">{t('results.monthlyHelp')}</p><div className="table-scroll monthly-matrix" role="region" tabIndex={0} aria-label={t('results.monthly-stats')}><table><thead><tr><th scope="col">{t('results.year')}</th>{Array.from({length:12},(_,m)=><th key={m} scope="col">{new Date(Date.UTC(2020,m,1)).toLocaleDateString(i18n.language,{month:'short',timeZone:'UTC'})}</th>)}</tr></thead><tbody>{years.map(year=><tr key={year}><th scope="row">{year}</th>{Array.from({length:12},(_,m)=>{const row=rows.find(r=>r.month===`${year}-${String(m+1).padStart(2,'0')}`);return <td key={m} className={row?row.net_pnl_quote>0?'pnl-positive':row.net_pnl_quote<0?'pnl-negative':'':''}>{row?<><strong>{row.percent==null?'—':`${number(row.percent)}%`}</strong><span>{number(row.net_pnl_quote)}</span></>:'—'}</td>})}</tr>)}</tbody></table></div>{!years.length&&<p>{t('results.empty')}</p>}</>;
}
export function Chart({data,label,group,periodView,showTrades=false}:{showTrades?:boolean;periodView?:PeriodView;data:z.infer<typeof api.seriesSchema>;label:string;group:string}){
  const ref=useRef<HTMLDivElement>(null);const {t,i18n}=useTranslation();const instance=useChartInstance(ref);
  const times=useMemo(()=>chartTimes(data.points.map(p=>p.x)),[data]);
  usePeriodZoom(instance,times,periodView);
  useLayoutEffect(()=>{
    if(!ref.current||!data.points.length)return;
    const element=ref.current,colors=getComputedStyle(element),muted=colors.getPropertyValue('--muted').trim();
    const drawdown=data.kind==='drawdown',accent=drawdown?'#ed958b':colors.getPropertyValue('--violet').trim();
    const date=(value:string|number)=>chartDate(value,i18n.language);
    const number=(value:unknown)=>typeof value==='number'?value.toLocaleString(i18n.language,{maximumFractionDigits:2}):String(value);
    const chart=instance.current;if(!chart)return;chart.group=group;connect(group);
      chart.setOption({animation:motionDuration()>0,animationDuration:0,animationDurationUpdate:Math.min(200,motionDuration()),tooltip:{trigger:'axis',renderMode:'richText',confine:true,backgroundColor:'#171d23',borderColor:'#39424b',textStyle:{color:'#e6e9ee'},axisPointer:{type:'cross',label:{backgroundColor:'#343d48'}},valueFormatter:number,
        formatter:(raw:any)=>{const values=Array.isArray(raw)?raw:[raw],point=data.points[values[0]?.dataIndex];if(!point)return '';return `${date(point.x)}\n${label}: ${number(point.value)}${drawdown?'%':''}${showTrades&&point.trade_index!=null?`\n${t('results.columns.trade_index')} #${point.trade_index} · P&L ${number(point.net_pnl_quote??'—')}`:''}`;}},
        dataZoom:[{type:'inside',filterMode:'none',zoomOnMouseWheel:'ctrl',...(periodView?periodZoom(times,periodView.range):{})},{type:'slider',bottom:5,height:18,labelFormatter:(_:number,value:string)=>date(value),borderColor:'#39424b',fillerColor:'rgba(131,89,235,.18)',textStyle:{color:muted}}],
        grid:{left:72,right:18,top:18,bottom:62},textStyle:{color:muted},
        xAxis:{type:'category',data:data.points.map(p=>String(p.x)),axisLabel:{formatter:date,hideOverlap:true,color:muted},axisPointer:{label:{formatter:(p:any)=>date(p.value)}},axisLine:{lineStyle:{color:muted}}},
        yAxis:{type:'value',scale:true,max:drawdown?0:null,axisLabel:{color:muted,formatter:(v:number)=>`${number(v)}${drawdown?'%':''}`},splitLine:{lineStyle:{color:'#293139'}}},
        series:[{name:label,type:'line',data:data.points.map(p=>p.value),showSymbol:data.points.length===1,symbolSize:6,itemStyle:{color:accent},lineStyle:{color:accent,width:2},areaStyle:{color:accent,opacity:drawdown?.22:.08}},
          ...(showTrades&&!drawdown?[{name:t('results.tradeClosures'),type:'scatter',symbolSize:7,data:data.points.map((p,index)=>({value:[index,p.value],itemStyle:{color:(p.net_pnl_quote??0)<0?'#ed958b':'#65d69b'}}))}]:[])]},{replaceMerge:['series']});
  },[data,label,group,showTrades,i18n.language,t,periodView]);
  return <div className="chart-view">{showTrades&&data.downsampled&&<p className="muted chart-help">{t('results.points',{returned:data.returned_points,source:data.source_points})}</p>}<div ref={ref} className="result-chart" role="img" aria-label={label}/><details className="chart-data"><summary>{t('results.alternative')}</summary><p className="muted">{t('results.points',{returned:data.returned_points,source:data.source_points})}</p><DataTable items={data.points} columns={['x','value']} label={label}/></details></div>;
}
export function Results({job,subject,now,active=true,information}:{information?:ReactNode;job:string;subject:string;now:number;active?:boolean}){
  const {t,i18n}=useTranslation();const [params,setParams]=useSearchParams();
  const summary=useResultRead(subject,[job,'summary'],async signal=>{const r=await requestJson(`${api.jobPath(job)}/summary`,api.summarySchema,{signal});if(r.data.job.job_id!==job)throw new ApiError('invalid-response',200,'failed');return r;},now);
  const top=useResultRead(subject,[job,'top'],signal=>requestJson(`${api.jobPath(job)}/top`,api.topSchema,{signal}),now);
  const committedSelection=useRef<string|null>(null);
  const selected=!active&&committedSelection.current?committedSelection.current:params.get('variant');const [variantPage,setVariantPage]=useState(0);
  useEffect(()=>{if(active&&!selected&&summary.data?.data.selected_variant_key){setParams(old=>{const next=new URLSearchParams(old);next.set('variant',summary.data!.data.selected_variant_key!);return next;},{replace:true});}},[active,selected,summary.data,setParams]);
  const [sort,setSort]=useState<{key:keyof z.infer<typeof api.metricsSchema>;ascending:boolean}>({key:'total_return_pct',ascending:false});
  const variants=top.data?.data.items.slice().sort((a,b)=>{const x=a.summary_metrics[sort.key],y=b.summary_metrics[sort.key];if(x==null)return y==null?0:1;if(y==null)return -1;return (x-y)*(sort.ascending?1:-1);});
  const metricColumns=['total_return_pct','max_drawdown_pct','profit_factor','win_rate_pct','trade_count'] as const;
  const number=(v:number|null|undefined)=>v==null?'—':v.toLocaleString(i18n.language,{maximumFractionDigits:2});
  if(summary.isPending||top.isPending)return <section id="results" className="results data-pending"><LoadingData /></section>;
  const ranking=<><div className="result-heading"><h3>{t('results.variants')}</h3><ReadStatus pending={summary.isFetching||top.isFetching}/></div><ReadState query={summary}>{null}</ReadState><ReadState query={top}>{variants?.length?<div className="table-scroll variant-ranking" role="region" aria-label={t('results.variants')} tabIndex={0}><table><thead><tr><th>{t('results.variant')}</th>{metricColumns.map(k=><th key={k} aria-sort={sort.key===k?(sort.ascending?'ascending':'descending'):'none'}><button onClick={()=>{setVariantPage(0);setSort(old=>({key:k,ascending:old.key===k?!old.ascending:k==='max_drawdown_pct'}));}}>{t(`results.metrics.${k}`)}{sort.key===k?(sort.ascending?' ↑':' ↓'):''}</button></th>)}</tr></thead><tbody>{variants.slice(variantPage*10,variantPage*10+10).map(v=>{const next=new URLSearchParams(params);next.set('variant',v.variant_key);return <tr key={v.variant_key} className={selected===v.variant_key?'selected':''} onClick={e=>{if(!(e.target as HTMLElement).closest('a'))transitionUI(()=>setParams(next));}}><td><Link aria-current={selected===v.variant_key?'true':undefined} to={`?${next}`} preventScrollReset>{v.readable_params?.indicators.map(i=>`${i.indicator_id.replace('ma.','').toUpperCase()} ${i.window??''}`).join(' + ')||t('results.selected',{rank:v.rank})}</Link><span className="job-meta">#{v.rank} · TP {number(v.best_tp_pct)}% · SL {number(v.best_sl_pct)}%</span></td>{metricColumns.map(k=><td key={k}>{number(v.summary_metrics[k]??(k==='trade_count'?v.summary_metrics.trades_count:undefined))}</td>)}</tr>})}</tbody></table></div>:<p>{t('results.empty')}</p>}</ReadState>
  {variants&&variants.length>10&&<nav className="variant-pages" aria-label={t('results.variants')}><button aria-label={t('results.previousVariants')} disabled={variantPage===0} onClick={()=>setVariantPage(p=>p-1)}>←</button><span>{variantPage*10+1}–{Math.min(variantPage*10+10,variants.length)} / {variants.length}</span><button aria-label={t('results.nextVariants')} disabled={(variantPage+1)*10>=variants.length} onClick={()=>setVariantPage(p=>p+1)}>→</button></nav>}
  {(summary.error||top.error)&&<button disabled={!summary.canRefresh||!top.canRefresh} onClick={()=>{void summary.refetch();void top.refetch();}}>{t('results.refresh')}</button>}
</>;
  return <section id="results" className="results" tabIndex={-1}>
  {selected&&api.variantKeySchema.safeParse(selected).success&&!isRestricted(summary.error)&&!isRestricted(top.error)&&<Variant information={information} ranking={ranking} timeRange={summary.data?.data.job.request.time_range} onCommit={value=>{committedSelection.current=value;}} key={job} job={job} variant={selected} subject={subject} now={now}/>}</section>;
}
function useDetailRead(job:string,variant:string,subject:string,now:number,tab:string,page:number,enabled:boolean) {
  const schema=tab==='trades'?api.tradesSchema:tab.endsWith('stats')?api.statsSchema:api.seriesSchema;
  const suffix=tab==='trades'?`trades?page=${page}&page_size=6`:tab.endsWith('stats')?tab:`${tab}?points=400`;
  return useResultRead(subject,[job,variant,suffix],async signal=>{
    const reply=await api.readResult(job,variant,suffix,schema as z.ZodType<any>,signal);
    if(reply.status===200 && ((reply.data.pagination && reply.data.pagination.page!==page) || (reply.data.kind && reply.data.kind!==(tab==='monthly-stats'?'monthly':tab))))throw new ApiError('invalid-response',200,'failed');
    return reply;
  },now,enabled);
}
function Variant({job,variant,subject,now,onCommit,timeRange,ranking,information}:{information?:ReactNode;ranking:ReactNode;timeRange?:{start:string;end:string};onCommit:(value:string)=>void;job:string;variant:string;subject:string;now:number}){
  const {t,i18n}=useTranslation();
  const [showTrades,setShowTrades]=useState(false),[priceMarkers,setPriceMarkers]=useState(true);
  const [tab,setTab]=useState('overview'),[chartKind,setChartKind]=useState('equity'),[timeframe,setTimeframe]=useState('15m');
  const [period,setPeriod]=useState<{preset:ChartPeriod|null;range:PeriodView['range']}>({preset:'all',range:null});
  const [pagination,setPagination]=useState({variant,page:1});
  const page=pagination.variant===variant?pagination.page:1;
  const setPage=(page:number)=>setPagination({variant,page});
  const kind=chartKind;
  const detail=useResultRead(subject,[job,variant,'variant'],async signal=>{const r=await requestJson(api.variantPath(job,variant),api.variantSchema,{signal});if(r.data.variant_key!==variant)throw new ApiError('invalid-response',200,'failed');return r;},now);
  // Independent reads start together; only a complete visible bundle is committed.
  const panel=useDetailRead(job,variant,subject,now,kind,page,!['metrics','price'].includes(kind));
  const price=usePriceRead(job,variant,subject,now,timeframe,kind==='price');
  const tableRead=useDetailRead(job,variant,subject,now,tab,page,tab==='monthly-stats');
  const reads=[detail,...(kind==='price'?[price.prices,price.trades]:[panel]),...(tab==='monthly-stats'?[tableRead]:[])];
  const blocked=reads.some(read=>isRestricted(read.error));
  const pending=reads.some(read=>read.isFetching || (read.data?.status===202&&['queued','running','pending'].includes(String((read.data.data as {status?:string}).status))));
  const complete=reads.every(read=>read.data?.status===200&&!read.isFetching&&!read.error);
  const selection=JSON.stringify([variant,tab,chartKind,page,timeframe]);
  const snapshot=useReadSnapshot(`${subject}:${job}`,selection,complete&&detail.data?{
    variant,tab,kind,page,timeframe,detail:detail.data.data, table:tableRead.data?.status===200?tableRead.data.data:undefined,
    panel:panel.data?.status===200?panel.data.data:undefined,
    prices:price.prices.data?.data,trades:price.trades.data?.data,
  }:undefined,blocked);
  const shown=snapshot.data;
  useEffect(()=>{if(shown)onCommit(shown.variant);},[shown?.variant,onCommit]);
  const tabs=['overview','metrics','trades','monthly-stats'];
  const observedTimes=chartTimes(shown?.kind==='price'?shown.prices?.candles.map(c=>c.time)??[]:shown?.panel?.points?.map((p:{x:string|number})=>p.x)??[]);
  const bounds=timeRange?{start:Date.parse(timeRange.start),end:Date.parse(timeRange.end)}:observedTimes.length?{start:observedTimes[0]!,end:observedTimes.at(-1)!}:null;
  const periodView=useMemo<PeriodView>(()=>({range:period.range,onZoom:()=>{}}),[period.range]);
  const periodControls=<div className="chart-timeframes chart-periods view-switch" role="group" aria-label={t('results.chartPeriod')}>{chartPeriods.map(value=><button key={value} type="button" aria-pressed={period.preset===value} disabled={value!=='all'&&(!bounds||!observedTimes.length||periodStart(bounds.end,value)<bounds.start)} onClick={()=>setPeriod({preset:value,range:value==='all'||!bounds?null:{start:periodStart(bounds.end,value),end:bounds.end}})}>{t(`results.periods.${value}`)}</button>)}<DateRangeControl bounds={bounds&&observedTimes.length?bounds:null} value={period.range} active={period.preset===null} onApply={range=>setPeriod({preset:null,range})}/></div>;
  const chartControls=<div className="report-chart-controls"><div className="chart-switch view-switch" role="group" aria-label={t('results.chartType')}>{['equity','price','drawdown'].map(kind=><button key={kind} aria-pressed={chartKind===kind} onClick={()=>setChartKind(kind)}>{t(`results.${kind}`)}</button>)}</div><div className="report-chart-ranges">{chartKind!=='price'&&periodControls}{chartKind==='price'&&<div className="chart-timeframes view-switch" role="group" aria-label={t('results.chartTimeframe')}>{['1m','5m','15m','1h','1d'].map(tf=><button key={tf} type="button" aria-pressed={timeframe===tf} onClick={()=>setTimeframe(tf)}>{tf}</button>)}<DateRangeControl bounds={bounds&&observedTimes.length?bounds:null} value={period.range} active={period.preset===null} onApply={range=>setPeriod({preset:null,range})}/></div>}{chartKind==='equity'&&<label className="trade-toggle"><input type="checkbox" checked={showTrades} onChange={e=>setShowTrades(e.target.checked)}/>{t('results.tradeClosures')}</label>}{chartKind==='price'&&<label className="trade-toggle"><input type="checkbox" checked={priceMarkers} onChange={e=>setPriceMarkers(e.target.checked)}/>{t('results.priceTrades')}</label>}<ReadStatus className="expanded-read-status" pending={pending} retained={snapshot.retained}/></div></div>;
  const content=shown&&(shown.kind==='price'&&shown.prices&&shown.trades?<PriceChart markers={priceMarkers} periodView={periodView} data={shown.prices} entries={shown.trades.items} timeframe={shown.timeframe}/>:shown.panel?<DetailContent showTrades={showTrades} periodView={periodView} data={shown.panel} tab={shown.kind} page={shown.page} setPage={setPage} pending={snapshot.retained} initialCash={shown.detail.canonical_variant_params?.execution?.initial_cash_quote} group={job}/>:null);
  return <section className="navigator-result" aria-label={t('results.variant')} data-result-variant={shown?.variant} data-requested-variant={variant} aria-busy={pending}>
    <div className="variant-heading"><h3 className="selected-variant-title">{t('results.variant')} · {shown?.detail.rank??'—'}</h3><ReadStatus pending={pending} retained={snapshot.retained}/></div>
    {reads.map((read,index)=><ReadError key={index} error={read.error}/>)}
    {reads.some(read=>read.data?.status===202&&!['queued','running','pending'].includes(String((read.data.data as {status?:string}).status)))&&<p role="status" className="notice">{t('results.failed')}</p>}
    {!shown&&pending&&<div className="data-pending"><LoadingData/></div>}
    {shown&&<>
      <dl className="result-metrics">{['total_return_pct','max_drawdown_pct','profit_factor','win_rate_pct','trade_count'].map(k=>{const metrics=shown.detail.summary_metrics;const v=metrics[k as keyof typeof metrics]??(k==='trade_count'?metrics.trades_count:undefined);return <div key={k}><dt>{t(`results.metrics.${k}`)}</dt><dd>{v==null?'—':v.toLocaleString(i18n.language,{maximumFractionDigits:1})}</dd></div>;})}</dl>
    </>}
    {!blocked&&<><ExpandableOverview className="navigator-chart" controls={chartControls}>{content}</ExpandableOverview>
    <NavigatorTable actions={shown&&<div inert={snapshot.retained}><Export key={`export:${shown.variant}`} job={job} variant={shown.variant} now={now}/></div>} value={tab} onChange={setTab} tabs={tabs.map(name=>({id:name,label:t(name==='overview'?'results.variants':name==='metrics'?'results.metricsTab':`results.${name}`),content:name==='overview'?ranking:name==='metrics'&&shown?<table className="full-metrics"><thead><tr><th>{t('results.metric')}</th><th>{t('results.value')}</th></tr></thead><tbody>{Object.entries(shown.detail.summary_metrics).map(([k,v])=><tr key={k}><th scope="row">{t(`results.metrics.${k}`)}</th><td>{v==null?'—':v.toLocaleString(i18n.language,{maximumFractionDigits:3})}</td></tr>)}</tbody></table>:name==='trades'?tab==='trades'&&shown?<TradesScroll key={`${job}:${shown.variant}`} subject={subject} job={job} variant={shown.variant} now={now}/>:null:shown?.tab===name&&shown.table?<DetailContent data={shown.table} tab={name} page={shown.page} setPage={setPage} pending={snapshot.retained} initialCash={shown.detail.canonical_variant_params?.execution?.initial_cash_quote} group={job}/>:null}))}/>
    <aside className="navigator-result-actions navigator-inspector panel">{information}
      {shown&&<div><div inert={snapshot.retained}><div className="report-actions"><SaveStrategy key={`save:${shown.variant}`} job={job} variant={shown.variant} subject={subject} now={now}/></div></div></div>}
    </aside></>}
  </section>;
}
function DetailContent({data,tab,page,setPage,initialCash,group,pending,periodView,showTrades}:{showTrades?:boolean;periodView?:PeriodView;data:any;tab:string;page:number;setPage:(n:number)=>void;initialCash?:number;group:string;pending:boolean}){
  const {t}=useTranslation();
  return <div className="result-detail-view" data-ready="true">
    {(data.cache?.degraded||data.bounds?.truncated)&&<p className="notice">{t('results.degraded')}</p>}
    {data.points?data.points.length?<Chart showTrades={showTrades} periodView={periodView} data={data} label={t(`results.${tab}`)} group={group}/>:<p>{t('results.empty')}</p>:tab==='monthly-stats'?<MonthlyMatrix data={data} initialCash={initialCash}/>:<DataTable items={data.items} columns={tab==='trades'?['trade_index','entry_timestamp','exit_timestamp','side','entry_price','exit_price','quantity','net_pnl_quote','return_pct','fee_quote','exit_reason','equity_after']:[tab==='monthly-stats'?'month':'symbol','trades_count','net_pnl_quote','return_pct','win_rate_pct']} label={t(`results.${tab}`)}/>}
    {data.pagination&&<nav className="pagination" aria-label={t('results.trades')}><button disabled={!data.pagination.has_previous||pending} onClick={()=>setPage(Math.max(1,page-1))}>{t('results.previous')}</button><span>{t('results.page',{page,total:data.pagination.total})}</span><button disabled={!data.pagination.has_next||page>=10000||pending} onClick={()=>setPage(page+1)}>{t('results.next')}</button></nav>}
  </div>;
}
function SaveStrategy({job,variant,subject,now}:{job:string;variant:string;subject:string;now:number}){
  const {t}=useTranslation(),client=useQueryClient();const [record,setRecord]=useState(()=>loadStrategyRecovery(subject));const [saved,setSaved]=useState<z.infer<typeof api.savedSchema>|null>(null);const [error,setError]=useState<Error|null>(null);const [failureDeadline,setFailureDeadline]=useState(0);const [sending,setSending]=useState(false);const [storage]=useState(storageWorks);const lock=useRef(false),controller=useRef<AbortController|null>(null);
  useEffect(()=>()=>controller.current?.abort(),[]);
  const readiness=useResultRead(subject,[job,variant,'compatibility'],async signal=>{const r=await requestJson(`${api.variantPath(job,variant)}/compatibility-readiness`,api.readinessSchema,{signal});api.assertSource(r.data,job,variant);return r;},now);
  const compatible=!!readiness.data&&!readiness.error&&!readiness.isFetching && now-Date.parse(readiness.data.data.checked_at)<60_000 && Date.parse(readiness.data.data.checked_at)<=now+5_000;
  const liveState=readiness.data?.data.compatibility_state;
  const reasons=readiness.data?.data.compatibility_reason_codes.map(reason=>t(`results.reasons.${reason}`)).join(' · ');
  async function save(){if(lock.current||record||!compatible)return;lock.current=true;setSending(true);setError(null);const abort=new AbortController();controller.current=abort;let attempt:StrategyRecovery|null=null;
    try{const identity=await client.fetchQuery({queryKey:['session',subject],queryFn:()=>readSession(abort.signal),staleTime:0,retry:false});if(abort.signal.aborted||identity.user_id!==subject)return;
      // Recheck source compatibility at the command gate; feed readiness is not a save requirement.
      const fresh=await requestJson(`${api.variantPath(job,variant)}/compatibility-readiness`,api.readinessSchema,{signal:abort.signal});api.assertSource(fresh.data,job,variant);if(Date.now()-Date.parse(fresh.data.checked_at)>=60_000 || Date.parse(fresh.data.checked_at)>Date.now()+5_000 || fresh.data.strategy_spec_hash!==readiness.data?.data.strategy_spec_hash)throw new ApiError('conflict',409,'failed');
      attempt={operation:'save-strategy',jobId:job,variant,key:crypto.randomUUID(),createdAt:Date.now(),subject,organization:null,resultId:null};storeStrategyRecovery(attempt);setRecord(attempt);
      const result=await api.saveStrategy(job,variant,attempt.key,subject,abort.signal);if(abort.signal.aborted)return;
      storeStrategyRecovery({...attempt,resultId:result.strategy.strategy_id});setSaved(result);clearStrategyRecovery();setRecord(null);
    }catch(e){if(abort.signal.aborted)return;const failure=e instanceof Error?e:new Error();setError(failure);setFailureDeadline(Date.now()+(failure instanceof ApiError?Math.max(2,failure.retryAfterSeconds??0):2)*1000);if(attempt&&failure instanceof ApiError&&failure.outcome==='failed'){clearStrategyRecovery();setRecord(null);}}
    finally{lock.current=false;setSending(false);}
  }
  return <section className="result-command" aria-label={t('results.save')}><ReadError error={readiness.error}/>{!storage&&<p className="notice">{t('results.storage')}</p>}
    <Confirm id="save" title={t('results.saveTitle')} help={`${t('results.saveHelp')} ${liveState!=='launchable'?t('results.liveWarning'):''} ${reasons??''}`} source={`${job} · ${variant}`} trigger={t('results.save')} icon={<Save size={16} aria-hidden="true"/>} triggerTitle={t('results.saveHelp')} disabled={!compatible||!!record||sending||!!saved||isRestricted(error)||!!error} onConfirm={()=>void save()}/>
    {sending&&<p role="status">{t('results.saving')}</p>}<ReadError error={error}/>{record&&!sending&&<p role="status" className="notice">{t(record.jobId===job&&record.variant===variant?'results.unresolved':'results.recoveryOther')} <a href="/strategies">{t('results.history')}</a>{record.resultId&&<a href={`/strategies/${record.resultId}`}>{t('results.open')}</a>}</p>}
    {saved&&<p role="status">{t('results.saved')}. {saved.duplicate&&t('results.duplicate')} <a className="button-link" href={savedStrategyHref(saved.strategy.strategy_id, job, variant)}>{t('results.open')}</a></p>}
  </section>;
}
function Export({job,variant,now}:{job:string;variant:string;now:number}){
  const {t,i18n}=useTranslation();const [open,setOpen]=useState(false),[sending,setSending]=useState(false),[deadline,setDeadline]=useState(0),[message,setMessage]=useState(''),[error,setError]=useState<Error|null>(null);const controller=useRef<AbortController|null>(null),lock=useRef(false);
  useEffect(()=>()=>controller.current?.abort(),[]);
  async function download(format:'csv'|'xlsx'){setOpen(false);if(lock.current||now<deadline)return;lock.current=true;setSending(true);setError(null);const abort=new AbortController();controller.current=abort;
    try{const r=await api.readCsv(job,variant,100000,abort.signal,{allRows:true,format});if(abort.signal.aborted)return;if(r.status===202){setDeadline(Date.now()+r.delay*1000);setMessage(t(['queued','running','pending'].includes(r.pending.status)?'results.csvPending':'results.failed'));return;}
      const url=URL.createObjectURL(r.blob),a=document.createElement('a');a.href=url;a.download=`backtest-${job}-trades.${format}`;document.body.append(a);a.click();a.remove();setTimeout(()=>URL.revokeObjectURL(url),1000);setMessage('');
    }catch(e){if(!abort.signal.aborted){setError(e as Error);if(e instanceof ApiError)setDeadline(Date.now()+Math.max(2,e.retryAfterSeconds??0)*1000);}}finally{lock.current=false;setSending(false);}}
  return <section className="result-export" aria-label={i18n.language.startsWith('ru')?'Экспорт сделок':'Export trades'}><div className="export-controls" onBlur={e=>{if(!e.currentTarget.contains(e.relatedTarget))setOpen(false);}} onKeyDown={e=>{if(e.key==='Escape'&&open){e.preventDefault();e.stopPropagation();setOpen(false);e.currentTarget.querySelector('button')?.focus();}}}><button className="library-icon-action" aria-label={i18n.language.startsWith('ru')?'Скачать':'Download'} title={i18n.language.startsWith('ru')?'Скачать':'Download'} aria-expanded={open} disabled={sending||now<deadline||isRestricted(error)} onClick={()=>setOpen(value=>!value)}><Download size={16} aria-hidden="true"/></button>{open&&<div className="export-format-popup" role="group" aria-label={i18n.language.startsWith('ru')?'Формат файла':'File format'}>{(['csv','xlsx'] as const).map(format=><button key={format} onClick={()=>void download(format)}>{format.toUpperCase()}</button>)}</div>}</div><p role="status">{sending?t('results.loading'):message}</p><ReadError error={error}/></section>;
}
