import { MotionLink as Link, useMotionState, transitionUI } from './motion';
import { ExpandableOverview } from './expandable-overview';
import { useEffect, useId, useRef, useState, type ReactNode } from 'react';
import { useQuery, useQueryClient } from '@tanstack/react-query';
import { useSearchParams } from 'react-router';
import { useTranslation } from 'react-i18next';
import { z } from 'zod';
import { init, use, connect } from 'echarts/core';
import { LineChart, ScatterChart } from 'echarts/charts';
import { GridComponent, TooltipComponent, DataZoomComponent } from 'echarts/components';
import { CanvasRenderer } from 'echarts/renderers';
import { ApiError, requestJson, readSession, type ApiReply } from './api';
import { ReadError, isRestricted } from './library';
import { storageWorks } from './recovery';
import { clearStrategyRecovery, loadStrategyRecovery, storeStrategyRecovery, type StrategyRecovery } from './strategy-recovery';
import * as api from './results-api';
import {PriceView} from './price-chart';
use([LineChart, ScatterChart, GridComponent, TooltipComponent, DataZoomComponent, CanvasRenderer]);
export function useResultRead<T>(subject:string, scope:string[], read:(signal:AbortSignal)=>Promise<ApiReply<T>>, now:number) {
  const query=useQuery({queryKey:['private',subject,'results',...scope],queryFn:({signal})=>read(signal),refetchOnMount:false,refetchOnReconnect:false,retryOnMount:false});
  const delay=api.resultDelay(query.data,query.error);
  const deadline=(query.error?query.errorUpdatedAt:query.dataUpdatedAt)+delay;
  useEffect(()=>{if(query.isFetching || !Number.isFinite(deadline))return;const timer=setTimeout(()=>void query.refetch(),Math.min(2147483647,Math.max(0,deadline-Date.now())));return()=>clearTimeout(timer);},[deadline,query.isFetching,query.refetch]);
  const manualDeadline=query.error ? isRestricted(query.error) ? Infinity : query.error instanceof ApiError && query.error.kind==='rate-limited' ? deadline : query.errorUpdatedAt + (query.error instanceof ApiError ? query.error.retryAfterSeconds??0 : 0)*1000 : query.data?.status===202?deadline:0;
  return {...query, canRefresh:!query.isFetching&&!isRestricted(query.error)&&now>=manualDeadline, deadline:manualDeadline};
}
function ReadState({query,children}:{query:ReturnType<typeof useResultRead<any>>;children:ReactNode}){
  const {t}=useTranslation();const data=query.data?.data;
  return <>{query.isPending&&<p role="status">{t('results.loading')}</p>}<ReadError error={query.error}/>{query.data?.status===202?<p role="status" className="notice">{t(`results.${['queued','running','pending'].includes(data.status)?'pending':'failed'}`)}</p>:!isRestricted(query.error)&&children}</>;
}
export function Confirm({id,title,help,source,onConfirm,disabled=false,trigger}:{id:string;title:string;help:string;source:string;onConfirm:()=>void;disabled?:boolean;trigger:string}){
  const {t}=useTranslation();const dialog=useRef<HTMLDialogElement>(null),button=useRef<HTMLButtonElement>(null);
  return <><button ref={button} disabled={disabled} onClick={()=>transitionUI(()=>dialog.current?.showModal())}>{trigger}</button><dialog ref={dialog} aria-labelledby={`${id}-title`} aria-describedby={`${id}-help`} onClose={()=>button.current?.focus()} onKeyDown={event=>{if(event.key!=='Tab')return;const buttons=event.currentTarget.querySelectorAll<HTMLButtonElement>('button:not(:disabled)');const first=buttons[0],last=buttons[buttons.length-1];if(event.shiftKey&&document.activeElement===first){event.preventDefault();last?.focus();}else if(!event.shiftKey&&document.activeElement===last){event.preventDefault();first?.focus();}}}><div className="panel-head"><h2 id={`${id}-title`}>{title}</h2></div><div className="filter-fields"><p id={`${id}-help`}>{help}</p><p className="identity">{source}</p><button autoFocus onClick={()=>transitionUI(()=>dialog.current?.close())}>{t('results.keep')}</button><button disabled={disabled} onClick={()=>{dialog.current?.close();onConfirm();}}>{t(`results.${id==='save'?'confirmSave':'confirmDelete'}`)}</button></div></dialog></>;
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
function Chart({data,label,group}:{data:z.infer<typeof api.seriesSchema>;label:string;group:string}){
  const ref=useRef<HTMLDivElement>(null);const descriptionId=useId();const {t,i18n}=useTranslation();const [showTrades,setShowTrades]=useState(false);
  useEffect(()=>{
    if(!ref.current||!data.points.length)return;
    const element=ref.current,colors=getComputedStyle(element),muted=colors.getPropertyValue('--muted').trim();
    const drawdown=data.kind==='drawdown',accent=drawdown?'#ed958b':colors.getPropertyValue('--violet').trim();
    const date=(value:string|number)=>chartDate(value,i18n.language);
    const number=(value:unknown)=>typeof value==='number'?value.toLocaleString(i18n.language,{maximumFractionDigits:2}):String(value);
    let chart:ReturnType<typeof init>|undefined;
    const draw=()=>{if(!element.clientWidth||!element.clientHeight)return;if(chart){chart.resize();return;}
      chart=init(element);chart.group=group;connect(group);
      chart.setOption({animation:false,tooltip:{trigger:'axis',renderMode:'richText',confine:true,backgroundColor:'#171d23',borderColor:'#39424b',textStyle:{color:'#e6e9ee'},axisPointer:{type:'cross',label:{backgroundColor:'#343d48'}},valueFormatter:number,
        formatter:(raw:any)=>{const values=Array.isArray(raw)?raw:[raw],point=data.points[values[0]?.dataIndex];if(!point)return '';return `${date(point.x)}\n${label}: ${number(point.value)}${drawdown?'%':''}${showTrades&&point.trade_index!=null?`\n${t('results.columns.trade_index')} #${point.trade_index} · P&L ${number(point.net_pnl_quote??'—')}`:''}`;}},
        dataZoom:[{type:'inside',filterMode:'none',zoomOnMouseWheel:'ctrl'},{type:'slider',bottom:5,height:18,labelFormatter:(_:number,value:string)=>date(value),borderColor:'#39424b',fillerColor:'rgba(131,89,235,.18)',textStyle:{color:muted}}],
        grid:{left:65,right:18,top:18,bottom:64},textStyle:{color:muted},
        xAxis:{type:'category',data:data.points.map(p=>String(p.x)),axisLabel:{formatter:date,hideOverlap:true,color:muted},axisPointer:{label:{formatter:(p:any)=>date(p.value)}},axisLine:{lineStyle:{color:muted}}},
        yAxis:{type:'value',scale:true,...(drawdown?{max:0}:{}),axisLabel:{color:muted,formatter:(v:number)=>`${number(v)}${drawdown?'%':''}`},splitLine:{lineStyle:{color:'#293139'}}},
        series:[{name:label,type:'line',data:data.points.map(p=>p.value),showSymbol:data.points.length===1,symbolSize:6,itemStyle:{color:accent},lineStyle:{color:accent,width:2},areaStyle:{color:accent,opacity:drawdown?.22:.08}},
          ...(showTrades&&!drawdown?[{name:t('results.tradeClosures'),type:'scatter',symbolSize:7,data:data.points.map((p,index)=>({value:[index,p.value],itemStyle:{color:(p.net_pnl_quote??0)<0?'#ed958b':'#65d69b'}}))}]:[])]});
    };draw();const resize=new ResizeObserver(draw);resize.observe(element);return()=>{resize.disconnect();chart?.dispose();};
  },[data,label,group,showTrades,i18n.language,t]);
  return <div className="chart-view"><div className="chart-toolbar"><p id={descriptionId} className="chart-help muted">{t(data.kind==='drawdown'?'results.drawdownHelp':'results.chartHint')}</p>{data.kind==='equity'&&<label className="trade-toggle"><input type="checkbox" checked={showTrades} onChange={e=>setShowTrades(e.target.checked)}/>{t('results.tradeClosures')}</label>}</div>{showTrades&&data.downsampled&&<p className="muted chart-help">{t('results.points',{returned:data.returned_points,source:data.source_points})}</p>}<div ref={ref} className="result-chart" role="img" aria-label={label} aria-describedby={descriptionId}/><details className="chart-data"><summary>{t('results.alternative')}</summary><p className="muted">{t('results.points',{returned:data.returned_points,source:data.source_points})}</p><DataTable items={data.points} columns={['x','value']} label={label}/></details></div>;
}
export function Results({job,subject,now,active=true}:{job:string;subject:string;now:number;active?:boolean}){
  const {t,i18n}=useTranslation();const [params,setParams]=useSearchParams();
  const summary=useResultRead(subject,[job,'summary'],async signal=>{const r=await requestJson(`${api.jobPath(job)}/summary`,api.summarySchema,{signal});if(r.data.job.job_id!==job)throw new ApiError('invalid-response',200,'failed');return r;},now);
  const top=useResultRead(subject,[job,'top'],signal=>requestJson(`${api.jobPath(job)}/top`,api.topSchema,{signal}),now);
  const selected=params.get('variant');const [variantPage,setVariantPage]=useState(0);
  useEffect(()=>{if(active&&!selected&&summary.data?.data.selected_variant_key){setParams(old=>{const next=new URLSearchParams(old);next.set('variant',summary.data!.data.selected_variant_key!);return next;},{replace:true});}},[active,selected,summary.data,setParams]);
  const [sort,setSort]=useState<{key:keyof z.infer<typeof api.metricsSchema>;ascending:boolean}>({key:'total_return_pct',ascending:false});
  const variants=top.data?.data.items.slice().sort((a,b)=>{const x=a.summary_metrics[sort.key],y=b.summary_metrics[sort.key];if(x==null)return y==null?0:1;if(y==null)return -1;return (x-y)*(sort.ascending?1:-1);});
  const metricColumns=['total_return_pct','max_drawdown_pct','profit_factor','win_rate_pct','trade_count'] as const;
  const number=(v:number|null|undefined)=>v==null?'—':v.toLocaleString(i18n.language,{maximumFractionDigits:2});
  return <section id="results" className="results" tabIndex={-1}><h3>{t('results.variants')}</h3><ReadState query={summary}>{null}</ReadState><ReadState query={top}>{variants?.length?<div className="table-scroll variant-ranking" role="region" aria-label={t('results.variants')} tabIndex={0}><table><thead><tr><th>{t('results.variant')}</th>{metricColumns.map(k=><th key={k} aria-sort={sort.key===k?(sort.ascending?'ascending':'descending'):'none'}><button onClick={()=>{setVariantPage(0);setSort(old=>({key:k,ascending:old.key===k?!old.ascending:k==='max_drawdown_pct'}));}}>{t(`results.metrics.${k}`)}{sort.key===k?(sort.ascending?' ↑':' ↓'):''}</button></th>)}</tr></thead><tbody>{variants.slice(variantPage*10,variantPage*10+10).map(v=>{const next=new URLSearchParams(params);next.set('variant',v.variant_key);return <tr key={v.variant_key} className={selected===v.variant_key?'selected':''} onClick={e=>{if(!(e.target as HTMLElement).closest('a'))transitionUI(()=>setParams(next));}}><td><Link aria-current={selected===v.variant_key?'true':undefined} to={`?${next}`} preventScrollReset>{v.readable_params?.indicators.map(i=>`${i.indicator_id.replace('ma.','').toUpperCase()} ${i.window??''}`).join(' + ')||t('results.selected',{rank:v.rank})}</Link><span className="job-meta">#{v.rank} · TP {number(v.best_tp_pct)}% · SL {number(v.best_sl_pct)}%</span></td>{metricColumns.map(k=><td key={k}>{number(v.summary_metrics[k]??(k==='trade_count'?v.summary_metrics.trades_count:undefined))}</td>)}</tr>})}</tbody></table></div>:<p>{t('results.empty')}</p>}</ReadState>
  {variants&&variants.length>10&&<nav className="variant-pages" aria-label={t('results.variants')}><button aria-label={t('results.previousVariants')} disabled={variantPage===0} onClick={()=>setVariantPage(p=>p-1)}>←</button><span>{variantPage*10+1}–{Math.min(variantPage*10+10,variants.length)} / {variants.length}</span><button aria-label={t('results.nextVariants')} disabled={(variantPage+1)*10>=variants.length} onClick={()=>setVariantPage(p=>p+1)}>→</button></nav>}
  {(summary.error||top.error)&&<button disabled={!summary.canRefresh||!top.canRefresh} onClick={()=>{void summary.refetch();void top.refetch();}}>{t('results.refresh')}</button>}
  {selected&&api.variantKeySchema.safeParse(selected).success&&!isRestricted(summary.error)&&!isRestricted(top.error)&&<Variant key={`${job}:${selected}`} job={job} variant={selected} subject={subject} now={now}/>}</section>;
}
function Variant({job,variant,subject,now}:{job:string;variant:string;subject:string;now:number}){
  const {t,i18n}=useTranslation();const [tab,setTab]=useMotionState('overview'),[page,setPage]=useMotionState(1),[chartKind,setChartKind]=useMotionState('equity');
  const detail=useResultRead(subject,[job,variant,'variant'],async signal=>{const r=await requestJson(api.variantPath(job,variant),api.variantSchema,{signal});if(r.data.variant_key!==variant)throw new ApiError('invalid-response',200,'failed');return r;},now);
  const tabs=['overview','metrics','trades','monthly-stats'];
  return <section aria-label={t('results.variant')} data-result-variant={variant}><h3 className="selected-variant-title">{t('results.variant')} · {detail.data?.data.rank??'—'}</h3><ReadState query={detail}>{detail.data&&<dl className="result-metrics">{Object.entries(detail.data.data.summary_metrics).filter(([k])=>['total_return_pct','max_drawdown_pct','profit_factor','win_rate_pct','trade_count','trades_count'].includes(k)).map(([k,v])=><div key={k}><dt>{t(`results.metrics.${k}`)}</dt><dd>{v==null?'—':v.toLocaleString(i18n.language,{maximumFractionDigits:1})}</dd></div>)}</dl>}</ReadState>
  {!isRestricted(detail.error)&&<><div className="result-tabs" role="tablist" aria-label={t('results.title')}>{tabs.map((name,index)=><button key={name} role="tab" id={`tab-${name}`} aria-controls="result-panel" aria-selected={tab===name} tabIndex={tab===name?0:-1} onClick={()=>setTab(name)} onKeyDown={e=>{const move=e.key==='ArrowRight'?1:e.key==='ArrowLeft'?-1:0;let next=move?(index+move+tabs.length)%tabs.length:e.key==='Home'?0:e.key==='End'?tabs.length-1:-1;if(next>=0){e.preventDefault();setTab(tabs[next]!);document.getElementById(`tab-${tabs[next]}`)?.focus();}}}>{t(name==='metrics'?'results.metricsTab':`results.${name}`)}</button>)}</div><div id="result-panel" data-motion-content role="tabpanel" aria-labelledby={`tab-${tab}`}>{tab==='overview'?<ExpandableOverview controls={<div className="chart-switch" role="group" aria-label={t('results.chartType')}>{['equity','drawdown','price'].map(kind=><button key={kind} aria-pressed={chartKind===kind} onClick={()=>setChartKind(kind)}>{t(`results.${kind}`)}</button>)}</div>}>{chartKind==='price'?<PriceView job={job} variant={variant} subject={subject} now={now}/>:<DetailView key={chartKind} job={job} variant={variant} subject={subject} now={now} tab={chartKind} page={1} setPage={setPage}/>}</ExpandableOverview>:tab==='metrics'?<table className="full-metrics"><thead><tr><th>{t('results.metric')}</th><th>{t('results.value')}</th></tr></thead><tbody>{Object.entries(detail.data?.data.summary_metrics??{}).map(([k,v])=><tr key={k}><th scope="row">{t(`results.metrics.${k}`)}</th><td>{v==null?'—':v.toLocaleString(i18n.language,{maximumFractionDigits:3})}</td></tr>)}</tbody></table>:<DetailView key={`${tab}:${page}`} job={job} variant={variant} subject={subject} now={now} tab={tab} page={page} setPage={setPage} initialCash={detail.data?.data.canonical_variant_params?.execution?.initial_cash_quote}/>}</div><details className="report-actions"><summary>{t('results.actions')}</summary><SaveStrategy job={job} variant={variant} subject={subject} now={now}/><Export job={job} variant={variant} now={now}/></details></>}
  </section>;
}
function DetailView({job,variant,subject,now,tab,page,setPage,initialCash}:{job:string;variant:string;subject:string;now:number;tab:string;page:number;setPage:(n:number)=>void;initialCash?:number}){
  const {t}=useTranslation();const schema=tab==='trades'?api.tradesSchema:tab.endsWith('stats')?api.statsSchema:api.seriesSchema;
  const suffix=tab==='trades'?`trades?page=${page}&page_size=6`:tab.endsWith('stats')?tab:`${tab}?points=400`;
  const query=useResultRead(subject,[job,variant,suffix],async signal=>{const reply=await api.readResult(job,variant,suffix,schema as z.ZodType<any>,signal);if(reply.status===200 && ((reply.data.pagination && reply.data.pagination.page!==page) || (reply.data.kind && reply.data.kind!==(tab==='monthly-stats'?'monthly':tab==='symbol-stats'?'symbol':tab))))throw new ApiError('invalid-response',200,'failed');return reply;},now);
  const data=query.data?.status===200?query.data.data:undefined;
  return <div className="result-detail-view" data-motion-content><ReadState query={query}>{data&&<>{(data.cache?.degraded||data.bounds?.truncated)&&<p className="notice">{t('results.degraded')}</p>}{data.points?data.points.length?<Chart data={data} label={t(`results.${tab}`)} group={`${job}:${variant}`}/>:<p>{t('results.empty')}</p>:tab==='monthly-stats'?<MonthlyMatrix data={data} initialCash={initialCash}/>:<DataTable items={data.items} columns={tab==='trades'?['trade_index','entry_timestamp','exit_timestamp','side','entry_price','exit_price','quantity','net_pnl_quote','return_pct','fee_quote','exit_reason','equity_after']:[tab==='monthly-stats'?'month':'symbol','trades_count','net_pnl_quote','return_pct','win_rate_pct']} label={t(`results.${tab}`)}/>} {data.pagination&&<nav className="pagination" aria-label={t('results.trades')}><button disabled={!data.pagination.has_previous||query.isFetching} onClick={()=>setPage(Math.max(1,page-1))}>{t('results.previous')}</button><span>{t('results.page',{page,total:data.pagination.total})}</span><button disabled={!data.pagination.has_next||page>=10000||query.isFetching} onClick={()=>setPage(page+1)}>{t('results.next')}</button></nav>}</>}</ReadState><button disabled={!query.canRefresh} onClick={()=>void query.refetch()}>{t('results.refresh')}</button></div>;
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
  return <section className="result-command" aria-label={t('results.save')}><p>{t('results.readiness')}: {t(`results.live.${liveState??'unavailable'}`)}{reasons&&` · ${reasons}`}</p><ReadError error={readiness.error}/><button disabled={!readiness.canRefresh||sending||now<failureDeadline||isRestricted(error)} onClick={async()=>{const checked=await readiness.refetch();if(!checked.error && !record && error instanceof ApiError && error.outcome==='failed')setError(null);}}>{t('results.refresh')}</button>{readiness.data&&<p className="muted">{t('results.feed',{state:t(`results.feedStates.${readiness.data.data.market_data_state}`,{defaultValue:t('results.unavailable')})})}</p>}{!storage&&<p className="notice">{t('results.storage')}</p>}
    <Confirm id="save" title={t('results.saveTitle')} help={`${t('results.saveHelp')} ${liveState!=='launchable'?t('results.liveWarning'):''} ${reasons??''}`} source={`${job} · ${variant}`} trigger={t('results.save')} disabled={!compatible||!!record||sending||!!saved||isRestricted(error)||!!error} onConfirm={()=>void save()}/>
    {sending&&<p role="status">{t('results.saving')}</p>}<ReadError error={error}/>{record&&!sending&&<p role="status" className="notice">{t(record.jobId===job&&record.variant===variant?'results.unresolved':'results.recoveryOther')} <a href="/strategies">{t('results.history')}</a>{record.resultId&&<a href={`/strategies/${record.resultId}`}>{t('results.open')}</a>}</p>}
    {saved&&<p role="status">{t('results.saved')}. {saved.duplicate&&t('results.duplicate')} <a className="button-link" href={`/strategies/${saved.strategy.strategy_id}`}>{t('results.open')}</a></p>}
  </section>;
}
function Export({job,variant,now}:{job:string;variant:string;now:number}){
  const {t}=useTranslation();const [rows,setRows]=useState(10000),[sending,setSending]=useState(false),[deadline,setDeadline]=useState(0),[message,setMessage]=useState(''),[error,setError]=useState<Error|null>(null);const controller=useRef<AbortController|null>(null),lock=useRef(false);
  useEffect(()=>()=>controller.current?.abort(),[]);
  async function download(){if(lock.current||now<deadline)return;lock.current=true;setSending(true);setError(null);const abort=new AbortController();controller.current=abort;
    try{const r=await api.readCsv(job,variant,rows,abort.signal);if(abort.signal.aborted)return;if(r.status===202){setDeadline(Date.now()+r.delay*1000);setMessage(t(['queued','running','pending'].includes(r.pending.status)?'results.csvPending':'results.failed'));return;}
      const url=URL.createObjectURL(r.blob),a=document.createElement('a');a.href=url;a.download=`backtest-${job}-trades.csv`;document.body.append(a);a.click();a.remove();setTimeout(()=>URL.revokeObjectURL(url),1000);setMessage(`${t('results.csvDone',r)} ${r.truncated?t('results.truncated'):''}`);
    }catch(e){if(!abort.signal.aborted){setError(e as Error);if(e instanceof ApiError)setDeadline(Date.now()+Math.max(2,e.retryAfterSeconds??0)*1000);}}finally{lock.current=false;setSending(false);}}
  return <section className="result-export" aria-label={t('results.csv')}><label>{t('results.maxRows')}<input type="number" min={1} max={100000} value={rows} onChange={e=>setRows(Number(e.target.value))}/></label><button disabled={sending||now<deadline||!Number.isInteger(rows)||rows<1||rows>100000||isRestricted(error)} onClick={()=>void download()}>{now<deadline?t('results.retry',{seconds:Math.ceil((deadline-now)/1000)}):t('results.csv')}</button><p role="status">{message}</p><ReadError error={error}/></section>;
}
