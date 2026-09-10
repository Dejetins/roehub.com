import { useContext, useEffect, useRef, useState, type ReactNode } from 'react';
import { useQuery, useQueryClient, type UseQueryResult } from '@tanstack/react-query';
import { useForm, useWatch, type FieldPath } from 'react-hook-form';
import { Link, UNSAFE_DataRouterContext, useBlocker, useNavigate } from 'react-router';
import { useTranslation } from 'react-i18next';
import { ArrowUp, ArrowDown, X, Plus } from 'lucide-react';
import { readJob } from './library-api';
import { ApiError, readSession } from './api';
import { createJob, readBuilderCatalog, readDefaults, readBounds, preflightRequest, type Research, type Catalog, type Defaults, type Preflight } from './builder-api';
import { dateBoundary, fieldPath, firstWindow, initialResearch, syntheticResearch, normalizedLabel, prepareResearch, validateResearch, type Issue } from './builder-model';
import { clearRecovery, freezeAttempt, loadRecovery, saveRecovery, storageWorks, recoveryReason, type Recovery } from './recovery';

function RouterDiscard({ dirty, message, embedded }: { dirty: boolean; message: string; embedded: boolean }) {
  const blocker=useBlocker(({nextLocation})=>dirty && !(embedded && /^\/backtests(?:\/|$)/.test(nextLocation.pathname)));
  useEffect(()=>{ if(blocker.state==='blocked') { if(window.confirm(message)) blocker.proceed(); else blocker.reset(); } },[blocker,message]);
  return null;
}
function DiscardGuard({ dirty, embedded = false }: { dirty: boolean; embedded?: boolean }) {
  const {t}=useTranslation(); const dataRouter=useContext(UNSAFE_DataRouterContext);const approvedLeave=useRef(false);
  useEffect(()=>{
    const unload=(e:BeforeUnloadEvent)=>{ if(dirty && !approvedLeave.current) { e.preventDefault(); e.returnValue=''; } };
    const click=(e:MouseEvent)=>{ const a=(e.target as Element)?.closest('a'); if(!dirty || !a || e.metaKey || e.ctrlKey || e.shiftKey || e.button!==0 || a.target==='_blank' || a.getAttribute('href')?.startsWith('#')) return;
      if (embedded && a.origin===window.location.origin && /^\/backtests(?:\/|$)/.test(a.pathname)) return;
      // Router links are guarded by the blocker. SSR/locale/logout links leave the document.
      if(a.dataset.clientLink==='true' && dataRouter) return;
      if(!window.confirm(t('builder.discard'))) {e.preventDefault(); e.stopPropagation();} else approvedLeave.current=true;
    };
    window.addEventListener('beforeunload',unload); document.addEventListener('click',click,true);
    return ()=>{window.removeEventListener('beforeunload',unload);document.removeEventListener('click',click,true);};
  },[dirty,dataRouter,t,embedded]);
  return dataRouter ? <RouterDiscard dirty={dirty} embedded={embedded} message={t('builder.discard')} /> : null;
}
export function RecoveryNotice({ record, rejected=false }: { record: Recovery; rejected?:boolean }) {
  const {t}=useTranslation();
  return <section className="notice" aria-labelledby="recovery-heading"><h2 id="recovery-heading" tabIndex={-1}>{t(rejected?'builder.rejected':'builder.unresolved')}</h2>
    <p>{t(rejected?'builder.frozen':'builder.recoveryHelp')}</p>{!rejected && <p>{t(`builderRecovery.${recoveryReason(record,record.subject,null,null)}`)}</p>}<dl><dt>{t('builder.recoveryTime')}</dt><dd>{new Date(record.createdAt).toISOString()}</dd><dt>{t('builder.recoveryKey')}</dt><dd className="identity">{record.key}</dd></dl>
    <Link data-client-link="true" to="/backtests">{t('builder.history')}</Link>{record.resultId && <> · <Link data-client-link="true" to={`/backtests/${record.resultId}`}>{t('builder.recoveryJob')}</Link></>}
  </section>;
}
function ReadRetry({queries}:{queries:UseQueryResult<unknown,Error>[]}) {
  const {t}=useTranslation();const [now,setNow]=useState(Date.now);
  useEffect(()=>{const timer=setInterval(()=>setNow(Date.now()),1000);return()=>clearInterval(timer);},[]);
  const failed=queries.filter(q=>q.error);
  if(!failed.length)return null;
  const deadline=Math.max(...failed.map(q=>q.errorUpdatedAt+(q.error instanceof ApiError?q.error.retryAfterSeconds ?? 0:0)*1000));
  const seconds=Math.max(0,Math.ceil((deadline-now)/1000));
  const forbidden=failed.some(q=>q.error instanceof ApiError && ['forbidden','unauthenticated'].includes(q.error.kind));
  return <div className="notice" role="alert">{failed.map((q,i)=><p key={i}>{t(`builder.errors.${q.error instanceof ApiError?q.error.kind:'unavailable'}`)}</p>)}
    <button type="button" disabled={forbidden || seconds>0 || queries.some(q=>q.isFetching)} onClick={()=>failed.forEach(q=>void q.refetch())}>{seconds?t('wait',{seconds}):t('builder.reloadInputs')}</button></div>;
}
export type BuilderSummary = { symbol: string; timeframe: string; start: string; end: string; dirty: boolean; status: 'unchecked' | 'checked' | 'staleShort' | 'checking' | 'submitting' | 'unresolved' };
type BuilderPlacement = { demoPreset?: boolean; embedded?: boolean; onSummary?: (summary: BuilderSummary) => void; onClose?: () => void; onCreated?: (id: string) => void };
export function Builder({subject,embedded=false,onClose,onCreated,onSummary,demoPreset=false}:{subject:string} & BuilderPlacement) {
  const {t}=useTranslation();
  const defaults=useQuery({queryKey:['private',subject,'builder-defaults'],queryFn:({signal})=>readDefaults(signal),refetchOnMount:false});
  const catalog=useQuery({queryKey:['private',subject,'builder-catalog'],queryFn:({signal})=>readBuilderCatalog(signal),refetchOnMount:false});
  const [recovery]=useState(()=>loadRecovery(subject));
  return <>{!embedded && <div className="workspace-title"><h1 id="workspace-heading" tabIndex={-1}>{t('new')}</h1><span className="scope-label">{t('artifactOnly')}</span></div>}
    {recovery ? <KnownRecovery record={recovery} onCreated={onCreated}/> : defaults.isPending || catalog.isPending ? <p role="status">{t('builder.loading')}</p> :
      defaults.isError || catalog.isError || !defaults.data || !catalog.data ? <section className="notice" role="alert"><p>{t('builder.unavailable')}</p><ReadRetry queries={[defaults,catalog]}/></section> :
      <Configure demoPreset={demoPreset} subject={subject} onSummary={onSummary} embedded={embedded} onClose={onClose} onCreated={onCreated} defaults={defaults.data} catalog={catalog.data}/>}</>;
}
function KnownRecovery({record,onCreated}:{record:Recovery;onCreated?: (id:string)=>void}) {
  const navigate=useNavigate();
  const known=useQuery({queryKey:['private',record.subject,'recovered-job',record.resultId],enabled:!!record.resultId,queryFn:({signal})=>readJob(record.resultId!,signal),refetchOnMount:false});
  useEffect(()=>{if(known.data){clearRecovery();if(onCreated) onCreated(known.data.job_id);else navigate(`/backtests/${known.data.job_id}`,{replace:true});}},[known.data,navigate]);
  const {t}=useTranslation();
  return <><RecoveryNotice record={record}/>{known.error && <p role="alert" className="notice">{t(`builder.errors.${known.error instanceof ApiError?known.error.kind:'unavailable'}`)}</p>}</>;
}
function Configure({subject,defaults:d,catalog:initialCatalog,embedded=false,onClose,onCreated,onSummary,demoPreset=false}:{subject:string;defaults:Defaults;catalog:Catalog} & BuilderPlacement) {
  const {t}=useTranslation(); const navigate=useNavigate(); const client=useQueryClient();
  const initial=()=>demoPreset ? syntheticResearch(d,initialCatalog) : initialResearch(d,initialCatalog);
  const form=useForm<Research>({defaultValues:initial()});
  const autoChecked=useRef(false);
  const b=useWatch({control:form.control}) as Research;
  const [issues,setIssues]=useState<Issue[]>([]); const [failure,setFailure]=useState<ApiError|null>(null);
  const [review,setReview]=useState<{data:Preflight;fingerprint:string;revision:number}|null>(null);
  const [pending,setPending]=useState<'preflight'|'submit'|null>(null); const [record,setRecord]=useState<Recovery|null>(null);
  const [rejected,setRejected]=useState(false); const [memoryOnly,setMemoryOnly]=useState(()=>!storageWorks()); const [storageChanged,setStorageChanged]=useState(false);
  const [deadline,setDeadline]=useState(0); const [now,setNow]=useState(Date.now); const [tooLarge,setTooLarge]=useState(false);
  const revision=useRef(0);
  const lock=useRef(false); const controller=useRef<AbortController|null>(null); const alive=useRef(true); const summary=useRef<HTMLDivElement>(null);
  useEffect(()=>{alive.current=true;return ()=>{alive.current=false;controller.current?.abort();};},[]);
  useEffect(()=>{const timer=setInterval(()=>setNow(Date.now()),1000);return()=>clearInterval(timer);},[]);
  const {strategy_name:_name,...computation}=b; const fingerprint=JSON.stringify(computation); const fingerprintRef=useRef(fingerprint); fingerprintRef.current=fingerprint;
  const catalog=useQuery({queryKey:['private',subject,'builder-catalog',b.coordinates.exchange,b.coordinates.market_type],queryFn:({signal})=>readBuilderCatalog(signal,b.coordinates),refetchOnMount:false});
  const c=catalog.data ?? initialCatalog;
  const catalogCurrent=!!catalog.data && !catalog.isError && !catalog.isFetching;
  const bounds=useQuery({queryKey:['private',subject,'builder-bounds',b.coordinates],queryFn:({signal})=>readBounds(b.coordinates,signal),refetchOnMount:false});
  const validReview=!!review && review.fingerprint===fingerprint && review.revision===revision.current && !review.data.errors.length && review.data.direction_market_compatibility.compatible;
  const dirty=form.formState.isDirty && !record;
  const draftStatus: BuilderSummary['status'] = pending==='submit' ? 'submitting' : record ? 'unresolved' : pending==='preflight' ? 'checking' : validReview ? 'checked' : review ? 'staleShort' : 'unchecked';
  useEffect(()=>{onSummary?.({symbol:b.coordinates.symbol,timeframe:b.timeframe,start:b.time_range.start,end:b.time_range.end,dirty:!!dirty,status:draftStatus});},[onSummary,b.coordinates.symbol,b.timeframe,b.time_range.start,b.time_range.end,dirty,draftStatus]);
  const wait=Math.max(0,Math.ceil((deadline-now)/1000));
  useEffect(()=>{if(issues.length || failure) summary.current?.focus();},[issues,failure]);
  function set(path:FieldPath<Research>,value:unknown) {
    if(path!=='strategy_name') revision.current++;
    if(path==='execution.sizing.mode') form.setValue('execution.sizing',{mode:String(value)},{shouldDirty:true});
    else form.setValue(path,value as never,{shouldDirty:true});
  }
  function issueFor(path:string) { return issues.filter(i=>{const p=fieldPath(i.path);return p===path || path.startsWith(`${p}.`);}); }
  function field(path:FieldPath<Research>,label:string,type='text',options?:{value:string;label?:string;disabled?:boolean}[],factor=1):ReactNode {
    const value=path.split('.').reduce<unknown>((v,k)=>(v as Record<string,unknown>)?.[k],b); const errors=issueFor(path);
    const props={id:`field-${path}`,name:path,'aria-invalid':errors.length>0 as boolean,'aria-describedby':errors.length?`error-${path}`:undefined};
    return <div className="field" key={path}><label htmlFor={props.id}>{label}</label>{options ? <select {...props} value={String(value ?? '')} onChange={e=>set(path,e.target.value)}>
      {!options.some(o=>o.value===value) && <option value={String(value ?? '')}>{String(value ?? '')} — {t('soon')}</option>}
      {options.map(o=><option key={o.value} value={o.value} disabled={o.disabled}>{o.label ?? o.value}</option>)}</select> :
      <input {...props} type={type} step={type==='number'?'any':undefined} value={type==='date'?String(value ?? '').slice(0,10):typeof value==='number' ? (Number.isFinite(value)?Number((value*factor).toPrecision(14)):'') : String(value ?? '')}
        onChange={e=>set(path,type==='date'?dateBoundary(e.target.value):type==='number'?(e.target.value===''?NaN:Number(e.target.value)/factor):e.target.value)}/>}
      {errors.length>0 && <span className="field-error" id={`error-${path}`}>{errors.map(i=>i.message===i.code?t(`builder.errors.${i.code}`):i.message).join(' · ')}</span>}</div>;
  }
  const options=(values:string[])=>values.map(value=>({value,label:t(`builderTerms.${value}`,{defaultValue:value})}));
  function serverFailure(error:unknown) {
    const e=error instanceof ApiError?error:new ApiError('unavailable',null,'failed'); setFailure(e);setIssues([...e.issues]);
    setDeadline(Date.now()+(e.retryAfterSeconds ?? 0)*1000);
  }
  async function check() {
    if(lock.current || record || wait) return;
    setFailure(null);setIssues([]);setReview(null);
    const body=form.getValues(); const errors=validateResearch(body,d,c,bounds.data);
    if(!catalogCurrent) errors.push({path:'coordinates',code:'catalogUnavailable',message:'catalogUnavailable'});
    if(errors.length) {setIssues(errors);return;}
    const stamp=fingerprintRef.current; const generation=revision.current; lock.current=true; setPending('preflight');controller.current=new AbortController();
    try {
      const result=await preflightRequest(prepareResearch(body),controller.current.signal);
      if(!alive.current || fingerprintRef.current!==stamp || revision.current!==generation) return;
      setReview({data:result,fingerprint:stamp,revision:generation});setIssues(result.errors);
    } catch(e) {if(alive.current && fingerprintRef.current===stamp && revision.current===generation) serverFailure(e);}
    finally {lock.current=false;if(alive.current)setPending(null);}
  }
  useEffect(()=>{
    if(!demoPreset || autoChecked.current || !catalogCurrent || !bounds.isSuccess || bounds.isFetching || record) return;
    autoChecked.current=true;
    // Check once after the live catalog is ready; edits and failures require explicit recheck.
    if(!form.formState.isDirty && revision.current===0) void check();
  },[demoPreset,catalogCurrent,bounds.isSuccess,bounds.isFetching,record]);
  async function submit() {
    if(lock.current || record || !validReview || wait || !catalogCurrent || bounds.isFetching || bounds.isError) return;
    lock.current=true;setPending('submit');setFailure(null);setTooLarge(false);
    controller.current=new AbortController();
    try {
      // Fresh subject check is a command gate; it does not invent organization scope.
      const identity=await readSession(controller.current.signal);
      if(identity.user_id!==subject) {clearRecovery();client.setQueryData(['session',subject],identity);return;}
      if(!alive.current || fingerprintRef.current!==review!.fingerprint || revision.current!==review!.revision) return;
      const body=prepareResearch({...form.getValues(),strategy_name:normalizedLabel(form.getValues('strategy_name') ?? '')});
      const attempt=freezeAttempt(body,subject);
      let stored:boolean;
      try {stored=saveRecovery(attempt);} catch {setTooLarge(true);return;}
      if(!stored && !memoryOnly) {clearRecovery();setMemoryOnly(true);setStorageChanged(true);return;}
      setRecord(attempt);setStorageChanged(false);
      const result=await createJob(attempt.body,attempt.key,controller.current.signal);
      if(!alive.current) return;
      saveRecovery({...attempt,organization:result.data.organization_id,resultId:result.data.job_id});
      clearRecovery();form.reset(form.getValues());
      void client.invalidateQueries({queryKey:['private',subject,'jobs'],refetchType:'active'});
      if(onCreated) onCreated(result.data.job_id); else navigate(`/backtests/${result.data.job_id}`);
    } catch(e) {
      if(!alive.current)return;
      serverFailure(e);
      if(e instanceof ApiError && e.outcome==='failed' && [403,404,422,429].includes(e.status ?? 0)) setRejected(true);
    } finally {lock.current=false;if(alive.current)setPending(null);}
  }
  return <div className={embedded ? "builder settings-workspace" : "builder"}><DiscardGuard dirty={dirty} embedded={embedded}/>
    <section className={embedded ? "builder-form" : "panel builder-form"} aria-labelledby="configure-heading"><div className={embedded ? "sr-only" : "panel-head"}><h2 id="configure-heading">{t('builder.configure')}</h2>{!embedded && <button type="button" onClick={()=>onClose ? onClose() : navigate('/backtests')} aria-label={t('builder.closeSettings')}><X aria-hidden="true"/>{t('builder.closeSettings')}</button>}</div>
      <form noValidate onSubmit={e=>{e.preventDefault();void check();}}>
        {demoPreset && <p className="notice">{t('builder.syntheticDemo')}</p>}<ReadRetry queries={[catalog,bounds]}/>{(catalog.isFetching || bounds.isFetching) && <p role="status">{t('builder.loading')}</p>}
        {memoryOnly && <p className="notice">{t('builder.storageWarning')}</p>}{storageChanged && <p role="alert" className="notice">{t('builder.storageChanged')}</p>}
        <div ref={summary} tabIndex={-1} className={issues.length || failure?'notice error':''} role={issues.length || failure?'alert':undefined}>
          {failure && <p>{t(`builder.errors.${failure.kind}`)}{failure.code && <> · {failure.code}</>}{failure.kind==='conflict' && <> {t('builder.conflict')}</>}</p>}
          {failure?.admission && <dl>{Object.entries(failure.admission).map(([key,value])=><div key={key}><dt>{t(`builderTerms.${key}`,{defaultValue:key})}</dt><dd>{typeof value==='string'?t(`builderLimits.${value.replaceAll('.','_')}`,{defaultValue:value}):value}</dd></div>)}</dl>}
          {issues.length>0 && <><h3>{t('builder.invalid')}</h3><ul>{issues.map((i,index)=><li key={index}><a href={`#field-${fieldPath(i.path)}`} onClick={e=>{e.preventDefault();let path=fieldPath(i.path);let el=document.getElementById(`field-${path}`); if(!el) el=Array.from(document.querySelectorAll<HTMLElement>('[id^="field-"]')).find(x=>x.id.startsWith(`field-${path}.`)) ?? null;if(el){let parent=el.parentElement;while(parent){if(parent instanceof HTMLDetailsElement)parent.open=true;parent=parent.parentElement;}el.focus();}}}>{fieldPath(i.path)}: {i.message===i.code?t(`builder.errors.${i.code}`):i.message}</a></li>)}</ul></>}
        </div>
        {tooLarge && <p className="notice error">{t('builder.tooLarge')}</p>}
        {record && <><RecoveryNotice record={record} rejected={rejected}/>{rejected && <div className="notice"><button type="button" disabled={wait>0} onClick={()=>{clearRecovery();setRecord(null);setReview(null);setRejected(false);}}>{t('builder.revise')}</button></div>}</>}
        <fieldset className="builder-fields" disabled={!!record || pending==='submit'}><legend className="sr-only">{t('builder.configure')}</legend>
          <div className="builder-meta">{field('strategy_name',t('builder.label'))}<p className="muted" title={t('builder.draftHelp')}>{t('builder.draftShort')}</p></div>
          <section className="config-band" aria-labelledby="market-heading"><h3 id="market-heading">{t('builder.groupMarket')}</h3><div className="band-content">
            <div className="config-grid">{field('coordinates.exchange',t('builder.exchange'),'text',c.instrument_universe.markets.map(o=>({...o,disabled:o.status!=='available'})))}
              {field('coordinates.market_type',t('builder.market'),'text',c.instrument_universe.market_types.map(o=>({...o,disabled:o.status!=='available'})))}
              {field('coordinates.symbol',t('builder.symbol'),'text',c.instrument_universe.symbols.map(o=>({...o,disabled:o.status!=='available'})))}
              {field('timeframe',t('builder.timeframe'),'text',options(d.supported_timeframes))}</div>
            <div className="config-grid date-grid">{field('time_range.start',t('builder.start'),'date')}{field('time_range.end',t('builder.end'),'date')}<p className="muted date-help">{t('builder.utcHelp')}</p></div>
          </div></section>
          <section className="config-band" aria-labelledby="signal-heading"><h3 id="signal-heading">{t('builder.groupSignal')}</h3><div className="band-content">
            {b.indicators.map((indicator,index)=><section className="indicator-row" key={index} aria-label={`${t('builder.indicator')} ${index+1}`}>
              <div className="indicator-tools"><span className="indicator-number">{index+1}</span><button type="button" aria-label={t('builder.up')} title={t('builder.up')} disabled={index===0} onClick={()=>{const list=[...b.indicators];[list[index-1],list[index]]=[list[index],list[index-1]];set('indicators',list);}}><ArrowUp aria-hidden="true"/></button>
                <button type="button" aria-label={t('builder.down')} title={t('builder.down')} disabled={index===b.indicators.length-1} onClick={()=>{const list=[...b.indicators];[list[index+1],list[index]]=[list[index],list[index+1]];set('indicators',list);}}><ArrowDown aria-hidden="true"/></button>
                <button type="button" aria-label={t('builder.remove')} title={t('builder.remove')} onClick={()=>set('indicators',b.indicators.filter((_,i)=>i!==index))}><X aria-hidden="true"/></button></div>
              <div className="config-grid">
                {field(`indicators.${index}.indicator_id`,t('builder.indicator'),'text',d.supported_indicator_ids.map(id=>({value:id,label:`${id}${firstWindow(d,id)===undefined?` — ${t('builder.noWindow')}`:''}`,disabled:firstWindow(d,id)===undefined || !c.indicator_catalog.items.some(i=>i.indicator_id===id && i.status==='available')})))}
                <div className="field"><span id={`sources-label-${index}`}>{t('builder.sources')}</span><details className="source-picker">
                  <summary id={`field-indicators.${index}.sources`} aria-labelledby={`sources-label-${index} sources-value-${index}`} aria-invalid={issues.some(i=>fieldPath(i.path).startsWith(`indicators.${index}.sources`))}><span id={`sources-value-${index}`}>{indicator.sources.join(', ') || t('builder.chooseSources')}</span></summary>
                  <fieldset className="sources"><legend className="sr-only">{t('builder.sources')}</legend>{[...new Set([...(d.indicator_sources[indicator.indicator_id] ?? []),...indicator.sources])].map((source,sourceIndex)=><label key={source}><input id={`field-indicators.${index}.sources.${sourceIndex}`} type="checkbox" aria-invalid={issueFor(`indicators.${index}.sources.${sourceIndex}`).length>0} checked={indicator.sources.includes(source)} onChange={e=>set(`indicators.${index}.sources`,e.target.checked?[...indicator.sources,source]:indicator.sources.filter(s=>s!==source))}/>{source}{!d.indicator_sources[indicator.indicator_id]?.includes(source) && ` — ${t('soon')}`}</label>)}</fieldset>
                  {!(d.indicator_sources[indicator.indicator_id]?.length) && <p className="muted">{t('builder.sourceLess')}</p>}
                </details>{issues.filter(i=>fieldPath(i.path).startsWith(`indicators.${index}.sources`)).map((i,n)=><p key={n} className="field-error">{i.message===i.code?t(`builder.errors.${i.code}`):i.message}</p>)}</div>
                {field(`indicators.${index}.window.start`,t('builder.windowStart'),'number')}{field(`indicators.${index}.window.stop`,t('builder.windowStop'),'number')}{field(`indicators.${index}.window.step`,t('builder.windowStep'),'number')}
              </div>
              <details className="window-help"><summary>{t('builder.supportedWindow')}</summary><p className="muted">{(()=>{const spec=d.indicator_param_specs[indicator.indicator_id]?.params.window;return spec?.mode==='explicit'?spec.values.join(', '):spec?`${spec.start}…${spec.stop_incl} / ${spec.step}`:t('soon');})()}</p></details>
            </section>)}
            <button className="add-indicator" type="button" disabled={b.indicators.length>=d.guardrails.max_indicator_arity} onClick={()=>{const id=d.supported_indicator_ids.find(id=>firstWindow(d,id)!==undefined && c.indicator_catalog.items.some(i=>i.indicator_id===id && i.status==='available'));if(id)set('indicators',[...b.indicators,{indicator_id:id,sources:[...(d.indicator_sources[id] ?? [])],window:{start:firstWindow(d,id),stop:firstWindow(d,id),step:1}}]);}}><Plus aria-hidden="true"/>{t('builder.add')}</button>
          </div></section>
          <section className="config-band" aria-labelledby="trade-heading"><h3 id="trade-heading">{t('builder.groupTrade')}</h3><div className="band-content">
            <div className="config-grid">{field('execution.direction_mode',t('builder.direction'),'text',options(d.direction_modes))}{field('execution.initial_cash_quote',t('builder.cash'),'number')}
              {field('execution.fee_rate',t('builder.fee'),'number',undefined,100)}{field('execution.slippage_rate',t('builder.slippage'),'number',undefined,100)}{field('execution.sizing.mode',t('builder.sizing'),'text',options(d.sizing_modes))}</div>
            {b.execution.sizing.mode!=='all_in' && <div className="config-grid sizing-grid">
              {b.execution.sizing.mode==='fixed_quote' && field('execution.sizing.quote_amount',t('builder.quote'),'number')}
              {b.execution.sizing.mode.startsWith('fixed_equity_pct') && field('execution.sizing.equity_pct',t('builder.equity'),'number')}
              {b.execution.sizing.mode.endsWith('min_quote') && field('execution.sizing.min_quote',t('builder.minQuote'),'number')}
              {b.execution.sizing.mode.endsWith('max_quote') && field('execution.sizing.max_quote',t('builder.maxQuote'),'number')}</div>}
          </div></section>
          <section className="config-band" aria-labelledby="risk-heading"><h3 id="risk-heading">{t('builder.groupRisk')}</h3><div className="band-content">
            <div className="config-grid">{field('risk.mode',t('builder.risk'),'text',options(d.risk_modes))}</div>
            {b.risk.mode==='tp_sl_grid' && <><p className="muted">{d.hit_times_grid.timeframe} · {t('builder.covered')}</p>{(['tp','sl'] as const).map(side=><section key={side} className="risk-row" aria-label={t(`builder.${side}`)}>
              <div className="config-grid"><label className="check-label"><input id={`field-risk.${side}.enabled`} type="checkbox" checked={b.risk[side]?.enabled ?? false} onChange={e=>set(`risk.${side}`,{...b.risk[side],enabled:e.target.checked})}/>{t(`builder.${side}`)}</label>
                {b.risk[side]?.enabled && <>{field(`risk.${side}.start_pct`,t('builder.startPct'),'number')}{field(`risk.${side}.stop_pct`,t('builder.stopPct'),'number')}{field(`risk.${side}.step_pct`,t('builder.stepPct'),'number')}</>}</div>
              <p className="muted">{t('builder.covered')}: {d.hit_times_grid[`${side}_levels_pct`].join(', ')}</p>
            </section>)}</>}
          </div></section>
          <section className="config-band" aria-labelledby="ranking-heading"><h3 id="ranking-heading">{t('builder.groupRanking')}</h3><div className="band-content">
            <div className="config-grid">{field('ranking.primary_metric',t('builder.metric'),'text',options(d.ranking_metrics))}{field('ranking.direction',t('builder.rankingDirection'),'text',options(['asc','desc']))}{field('top_n',t('builder.topN'),'number')}</div>
            <Policies body={b}/>
          </div></section>
        </fieldset>
        <div className="builder-actions"><span className="check-state" role="status">{pending==='preflight'?t('builder.checking'):validReview?t('builder.checked'):review?t('builder.staleShort'):t('builder.unchecked')}</span><button type="button" disabled={!!record || !!pending} onClick={()=>{revision.current++;form.reset(initial());setIssues([]);setFailure(null);setReview(null);setTooLarge(false);}}>{t('builder.reset')}</button><button className={!validReview?'primary':undefined} type="submit" disabled={!!record || !!pending || wait>0}>{pending==='preflight'?t('builder.checking'):t('builder.preflight')}</button>
          <button className="primary" type="button" disabled={!!record || !!pending || !validReview || wait>0 || !catalogCurrent || bounds.isFetching || bounds.isError} onClick={()=>void submit()}>{pending==='submit'?t('builder.submitting'):t('builder.submit')}</button>
          {wait>0 && <span role="status">{t('wait',{seconds:wait})}</span>}
        </div>
      </form></section>
    <details className={embedded ? "builder-review" : "panel builder-review"} open={!!review}><summary id="review-heading">{t('builder.review')}<span className="muted">{t('builder.reviewHelp')}</span></summary><div className="review-body">
      <p>{t('builder.advisory')}</p><dl><dt>{t('builder.sourceFreshness')}</dt><dd>{c.generated_at}</dd><dt>{t('builder.bounds')}</dt><dd>{bounds.data?.max_end ?? t('soon')}</dd>
        <dt>{t('builder.asof')}</dt><dd>{bounds.data?.artifact_metadata?.artifact_asof_date ?? t('soon')}</dd><dt>{t('builder.published')}</dt><dd>{bounds.data?.artifact_metadata?.published_at_utc ?? t('soon')}</dd></dl>
      {review && (review.fingerprint!==fingerprint || review.revision!==revision.current) && <p className="notice" role="status">{t('builder.stale')}</p>}
      {review && review.fingerprint===fingerprint && review.revision===revision.current && <><h3>{t('builder.cost')}</h3><dl>{Object.entries(review.data.cost_estimate).map(([k,v])=><div key={k}><dt>{t(`builderTerms.${k}`,{defaultValue:k})}</dt><dd>{typeof v==='string'?t(`builderTerms.${v}`,{defaultValue:v}):v}</dd></div>)}</dl>
        <h3>{t('builder.fundingQuality')}</h3><dl>{Object.entries(review.data.funding_readiness).map(([k,v])=><div key={k}><dt>{t(`builderTerms.${k}`,{defaultValue:k})}</dt><dd>{Array.isArray(v)?v.join(', ') || '—':v===null?t('soon'):typeof v==='string'?t(`builderTerms.${v}`,{defaultValue:v}):k==='coverage_ratio'?v*100:v}</dd></div>)}</dl>
        {review.data.warnings.length>0 && <div className="notice"><h3>{t('builder.warnings')}</h3><ul>{review.data.warnings.map((w,i)=><li key={i}>{w.message} · {w.code}</li>)}</ul></div>}
        <h3>{t('builder.normalized')}</h3><Policies body={review.data.normalized_request} expanded/><dl><dt>{t('builder.direction')}</dt><dd>{t(`builderTerms.${review.data.normalized_request.execution.direction_mode}`)}</dd><dt>{t('builder.fee')}</dt><dd>{review.data.normalized_request.execution.fee_rate*100}</dd><dt>{t('builder.slippage')}</dt><dd>{review.data.normalized_request.execution.slippage_rate*100}</dd><dt>{t('builder.ranking')}</dt><dd>{t(`builderTerms.${review.data.normalized_request.ranking.effective_primary_metric ?? review.data.normalized_request.ranking.primary_metric}`)}</dd><dt>{t('builder.cash')}</dt><dd>{review.data.normalized_request.execution.initial_cash_quote}</dd><dt>{t('builder.sizing')}</dt><dd>{Object.entries(review.data.normalized_request.execution.sizing).map(([k,v])=>`${t(`builderTerms.${k}`,{defaultValue:k})}: ${typeof v==='string'?t(`builderTerms.${v}`,{defaultValue:v}):v}`).join(' · ')}</dd><dt>{t('builder.topN')}</dt><dd>{review.data.normalized_request.top_n}</dd><dt>{t('builder.start')}</dt><dd>{review.data.normalized_request.time_range.start.slice(0,10)}</dd><dt>{t('builder.end')}</dt><dd>{review.data.normalized_request.time_range.end.slice(0,10)}</dd><dt>{t('builder.asof')}</dt><dd>{review.data.artifact_metadata.artifact_asof_date ?? t('soon')}</dd><dt>{t('builder.published')}</dt><dd>{review.data.artifact_metadata.published_at_utc ?? t('soon')}</dd></dl>
      </>}
    </div></details></div>;
}
function Policies({body,expanded=false}:{body:Research;expanded?:boolean}) {
  const {t}=useTranslation();return <details className="policies" open={expanded}><summary>{t('builder.policies')}</summary><p className="muted">{t('builder.defaultsHelp')}</p><dl>{([
    ['funding',body.execution.funding],['profitLock',body.execution.profit_lock],['closeOnEnd',body.execution.close_on_end],['quality',body.quality_constraints],
  ] as const).map(([key,value])=><div key={key}><dt>{t(`builder.${key}`)}</dt><dd>{typeof value==='boolean'?t(`builderTerms.${value?'on':'off'}`):Object.entries(value).map(([k,v])=>`${t(`builderTerms.${k}`,{defaultValue:k})}: ${typeof v==='boolean'?t(`builderTerms.${v?'on':'off'}`):typeof v==='string'?t(`builderTerms.${v}`,{defaultValue:v}):v}`).join(' · ')}</dd></div>)}</dl></details>;
}
