import {z} from 'zod';
import {type ReactNode} from 'react';
import {useQuery} from '@tanstack/react-query';
import {useTranslation} from 'react-i18next';
import {useMotionState} from './motion';
import {ApiError,requestJson} from './api';
import {isRestricted,formatDate,ReadError} from './library';
import {Chart,DataTable,useResultRead} from './results';
import {readResult,seriesSchema,tradesSchema,summarySchema,jobPath,pendingSchema} from './results-api';
import {readResearchSource,readResearchVariant} from './strategy-insights-api';
import type {Strategy,readStrategyStatus} from './strategies-api';

type Observation=Awaited<ReturnType<typeof readStrategyStatus>>|undefined;
const usable=(value:{state:string}|undefined)=>!!value && ['ready','degraded'].includes(value.state);
export function StrategyInsights({strategy,subject,observation,now,children}:{strategy:Strategy;subject:string;observation:Observation;now:number;children:ReactNode}) {
 const {t,i18n}=useTranslation();
 const [tab,setTab]=useMotionState('overview');
 const source=useQuery({queryKey:['private',subject,'strategy-insights',strategy.strategy_id,'source'],queryFn:({signal})=>readResearchSource(strategy.strategy_id,signal)});
 const origin=isRestricted(source.error)?undefined:source.data;
 const tabs=['overview','backtest','execution','specification'];
 return <div className="strategy-insights">
  <div className="result-tabs" role="tablist" aria-label={t('strategy.workspace')}>{tabs.map((name,index)=><button key={name} id={`strategy-tab-${name}`} role="tab" aria-selected={tab===name} aria-controls="strategy-tab-panel" tabIndex={tab===name?0:-1} onClick={()=>setTab(name)} onKeyDown={event=>{const next=event.key==='ArrowRight'?(index+1)%tabs.length:event.key==='ArrowLeft'?(index+tabs.length-1)%tabs.length:event.key==='Home'?0:event.key==='End'?tabs.length-1:-1;if(next>=0){event.preventDefault();setTab(tabs[next]!);document.getElementById(`strategy-tab-${tabs[next]}`)?.focus();}}}>{t(`strategy.tabs.${name}`)}</button>)}</div>
  <div id="strategy-tab-panel" role="tabpanel" aria-labelledby={`strategy-tab-${tab}`} data-motion-content data-motion-style="quiet">
   {tab==='specification'&&children}
   {tab==='execution'&&<Trading observation={observation}/>}
   {(tab==='overview'||tab==='backtest')&&<>
    <section className="strategy-research"><h3>{t('strategy.researchResult')}</h3>
     {source.isPending&&<p role="status">{t('strategy.sourceLoading')}</p>}
     <ReadError error={source.error}/>
     {source.error&&origin&&<p className="notice">{t('stale')}</p>}
     {origin?.source_job_id&&origin.source_variant_key?<Research key={`${origin.source_job_id}:${origin.source_variant_key}`} subject={subject} job={origin.source_job_id} variant={origin.source_variant_key} now={now} expanded={tab==='backtest'}/>:origin&&!source.error&&<div className="strategy-empty-research"><p>{t('strategy.noResearch')}</p><p className="muted">{t('strategy.noResearchHelp')}</p><a href="/backtests/new">{t('strategy.newResearch')}</a></div>}
    </section>
    {tab==='overview'&&<><p className="freshness">{t('createdUtc')} · {formatDate(strategy.created_at,i18n.language)}</p></>}
   </>}
  </div>
 </div>;
}
function Research({subject,job,variant,now,expanded}:{subject:string;job:string;variant:string;now:number;expanded:boolean}){
 const {t,i18n}=useTranslation();const [view,setView]=useMotionState('equity'),[page,setPage]=useMotionState(1);
 const query=useResultRead(subject,[job,variant,'strategy-research'],signal=>readResearchVariant(job,variant,signal),now);
 const summary=useResultRead(subject,[job,'strategy-research-summary'],async signal=>{const result=await requestJson(`${jobPath(job)}/summary`,summarySchema,{signal});if(result.data.job.job_id!==job)throw new ApiError('invalid-response',200,'failed');return result;},now);
 const data=isRestricted(query.error)||isRestricted(summary.error)?undefined:query.data?.data;
 const execution=data?.canonical_variant_params?.execution;
 const num=(value:number|null|undefined)=>value==null?'—':value.toLocaleString(i18n.language,{maximumFractionDigits:3});
 const metrics=data?.summary_metrics;
 return <>
  <ReadError error={query.error}/><ReadError error={summary.error}/>
  {query.isPending&&<p role="status">{t('refreshing')}</p>}
  {data&&<>
   {(query.error||summary.error)&&<p className="notice">{t('stale')}</p>}
   {summary.data&&!isRestricted(summary.error)&&<p className="freshness">{formatDate(summary.data.data.job.request.time_range.start,i18n.language)} — {formatDate(summary.data.data.job.request.time_range.end,i18n.language)} UTC</p>}
   <dl className="result-metrics strategy-research-metrics">{(['total_return_pct','max_drawdown_pct','profit_factor','win_rate_pct','trades_count','sharpe_trades'] as const).map(key=><div key={key}><dt>{t(`results.metrics.${key}`)}</dt><dd>{num(key==='trades_count'?metrics?.trades_count??metrics?.trade_count:metrics?.[key])}</dd></div>)}</dl>
   <p className="freshness">{t('strategy.researchOnly')}</p>
   {expanded&&<>
    <h3>{t('strategy.researchConditions')}</h3><dl className="strategy-observations">
     <Field label={t('strategy.direction')} value={execution?.direction_mode?t(`strategy.values.${execution.direction_mode}`,{defaultValue:execution.direction_mode}):'—'}/>
     <Field label={t('strategy.initialCash')} value={num(execution?.initial_cash_quote)}/>
     <Field label={t('strategy.fees')} value={execution?.fee_rate==null?'—':`${num(execution.fee_rate*100)}%`}/>
     <Field label={t('strategy.slippage')} value={execution?.slippage_rate==null?'—':`${num(execution.slippage_rate*100)}%`}/>
     <Field label={t('strategy.takeProfit')} value={data.best_tp_pct==null?t('strategy.notSet'):`${num(data.best_tp_pct)}%`}/>
     <Field label={t('strategy.stopLoss')} value={data.best_sl_pct==null?t('strategy.notSet'):`${num(data.best_sl_pct)}%`}/>
    </dl>
    <div className="chart-switch" role="group" aria-label={t('results.title')}>{['equity','drawdown','trades'].map(name=><button key={name} aria-pressed={view===name} onClick={()=>setView(name)}>{t(`results.${name}`)}</button>)}</div>
    <ResearchData key={`${view}:${page}`} job={job} variant={variant} subject={subject} now={now} view={view} page={page} setPage={setPage}/>
   </>}
  </>}
 </>;
}
function ResearchData({subject,job,variant,now,view,page,setPage}:{subject:string;job:string;variant:string;now:number;view:string;page:number;setPage:(p:number)=>void}){
 const {t}=useTranslation();
 const suffix=view==='trades'?`trades?page=${page}&page_size=10`:`${view}?points=400`;
 const series=useResultRead<z.infer<typeof tradesSchema>|z.infer<typeof seriesSchema>|z.infer<typeof pendingSchema>>(subject,[job,variant,'strategy-series',suffix],async signal=>{
  if(view==='trades'){const reply=await readResult(job,variant,suffix,tradesSchema,signal);if('pagination' in reply.data && reply.data.pagination.page!==page)throw new ApiError('invalid-response',200,'failed');return reply;}
  const reply=await readResult(job,variant,suffix,seriesSchema,signal);if('kind' in reply.data && reply.data.kind!==view)throw new ApiError('invalid-response',200,'failed');return reply;
 },now);
 const data=isRestricted(series.error)?undefined:series.data?.status===200?series.data.data:undefined;
 return <div className="strategy-research-data"><ReadError error={series.error}/>{(series.isPending||series.data?.status===202)&&<p role="status">{t('refreshing')}</p>}{data&&'points' in data?<Chart data={data} label={t(`results.${view}`)} group={`${job}:${variant}`}/>:data&&'items' in data?<><DataTable items={data.items} columns={['entry_timestamp','exit_timestamp','side','entry_price','exit_price','net_pnl_quote','return_pct','fee_quote']} label={t('results.trades')}/><nav className="pagination" aria-label={t('results.trades')}><button disabled={!data.pagination.has_previous} onClick={()=>setPage(page-1)}>{t('results.previous')}</button><span>{t('results.page',{page,total:data.pagination.total})}</span><button disabled={!data.pagination.has_next} onClick={()=>setPage(page+1)}>{t('results.next')}</button></nav></>:null}</div>;
}
function Field({label,value}:{label:string;value:ReactNode}){return <div><dt>{label}</dt><dd>{value}</dd></div>}
function Trading({observation}:{observation:Observation}){
 const {t,i18n}=useTranslation();const accounting=observation?.paper_accounting,signals=observation?.signal_journal;
 const num=(n:number|null|undefined)=>n==null?'—':n.toLocaleString(i18n.language,{maximumFractionDigits:6});
 return <>{accounting?.source==='synthetic_demo'&&<p className="notice">{t('strategy.executionDemo')}</p>}<section><h3>{t('strategy.paperAccount')}</h3><p className="freshness">{t('strategy.paperOnly')}</p>{usable(accounting)&&accounting?.updated_at?<><dl className="strategy-observations">{(['position_quantity','average_entry_price','equity','realized_pnl','unrealized_pnl','fee_total','funding_total'] as const).map(key=><Field key={key} label={t(`strategy.account.${key}`)} value={num(accounting[key])}/>)}</dl>{!accounting.pnl_complete&&<p className="notice">{t('strategy.incompletePnl')}</p>}<p className="freshness">{formatDate(accounting.updated_at,i18n.language)} UTC</p></>:<p>{t('strategy.noAccounting')}</p>}</section>
 <section><h3>{t('strategy.signals')}</h3>{usable(signals)&&signals?.items.length?<div className="table-scroll"><table><thead><tr>{['time','action','side','price','environment'].map(key=><th key={key}>{t(`strategy.${key}`)}</th>)}</tr></thead><tbody>{signals.items.map((signal,index)=><tr key={index}><td>{formatDate(signal.bar_ts_close,i18n.language)}</td><td>{t(`strategy.values.${signal.signal_action==='close'?'close_position':signal.signal_action}`,{defaultValue:signal.signal_action})}</td><td>{signal.side??'—'}</td><td>{num(signal.reference_price)}</td><td>{t(`strategy.values.${signal.mode}`,{defaultValue:signal.mode})}</td></tr>)}</tbody></table></div>:<p>{t('strategy.noSignals')}</p>}</section></>;
}
