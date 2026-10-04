import {useLibraryFilterPopup} from './library-filter-popup';
import {useReadSnapshot} from './read-snapshot';
import { LoadingData, ReadStatus } from './loading-data';
import { useEffect, useRef, useState, type ReactNode } from 'react';
import { useQuery, type UseQueryResult } from '@tanstack/react-query';
import { useLocation, useNavigate, useSearchParams } from 'react-router';
import { useTranslation } from 'react-i18next';
import { z } from 'zod';
import {ChartDisplayMenu} from './chart-display-menu';
import {StrategyOperations} from './strategy-operations';
import { ApiError } from './api';
import { MotionLink } from './motion';
import { RefreshCw, PanelRight, Filter } from 'lucide-react';
import { formatDate, isRestricted } from './library';
import { filterStrategies, newestStrategy, newestFirst, preferredStrategy, readStrategies, readStrategy, readStrategyStatus } from './strategies-api';

function StrategyError({ error }: { error: Error | null }) {
  const { t } = useTranslation();
  if (!error) return null;
  const kind = error instanceof ApiError ? error.kind : 'unavailable';
  return <p role="alert" className="notice error">{t(kind === 'not-found' ? 'strategy.missing' : `errors.${kind}`)}</p>;
}
export function StrategiesPage({ subject }: { subject: string }) {
  const { t, i18n } = useTranslation();
  const location = useLocation();
  const navigate = useNavigate();
  const [params, setParams] = useSearchParams();
  const id = location.pathname.split('/')[2] || params.getAll('strategy_id').at(-1) || '';
  const [collapsed, setCollapsed] = useState(false);
  const {open:filtersOpen,setOpen:setFiltersOpen,trigger:filterTrigger,popup:filterPopup,toggle:toggleFilters}=useLibraryFilterPopup();
  const listCollapsed = !!id && collapsed;
  const [now, setNow] = useState(Date.now);
  useEffect(() => { const timer = setInterval(() => setNow(Date.now()), 1000); return () => clearInterval(timer); }, []);
  const list = useQuery({ queryKey: ['private', subject, 'strategies'], queryFn: ({signal}) => readStrategies(subject, signal) });
  const validId = z.uuid().safeParse(id).success;
  // The existing dashboard response includes statuses for the authorized library.
  // Read it before redirecting a bare route, without temporarily selecting a row.
  const statusId=id||(!isRestricted(list.error)?newestStrategy(list.data??[])?.strategy_id:'')||'';
  const detail = useQuery({queryKey:['private',subject,'strategy',id], enabled:validId, queryFn:({signal})=>readStrategy(subject,id,signal)});
  const status = useQuery<Awaited<ReturnType<typeof readStrategyStatus>>>({queryKey:['private',subject,'strategy-status',statusId], enabled:z.uuid().safeParse(statusId).success, refetchOnMount:false, refetchInterval:query=>{if(isRestricted(query.state.error)||query.state.errorUpdateCount>=3)return false;return Math.max(15000,(query.state.data?.deadline??0)-Date.now(),query.state.error instanceof ApiError?(query.state.error.retryAfterSeconds??0)*1000:0);}, queryFn:({signal})=>readStrategyStatus(statusId,signal)});
  const snapshot=useReadSnapshot(subject,id,validId&&detail.data&&!detail.isFetching&&!status.isFetching&&(status.data||status.error)?{id,detail,status}:undefined,
    !validId||isRestricted(detail.error)||isRestricted(status.error));
  const displayed=snapshot.data;
  const reads = [list, ...(validId ? [detail, status] : [])];
  const eligible = reads.filter(read => !read.isFetching && !isRestricted(read.error) && now >= Math.max(read.errorUpdatedAt + (read.error instanceof ApiError ? read.error.retryAfterSeconds ?? 0 : 0)*1000, read === status ? status.data?.deadline ?? 0 : 0) && (read !== status || status.data?.refresh_control.manual_refresh_available !== false));
  const items = isRestricted(list.error) ? undefined : list.data;
  const query = params.get('q')?.slice(0, 200) ?? '', market = params.get('market') ?? '', timeframe = params.get('timeframe') ?? '';
  const stateFilter=params.get('state')??'';
  // Library status belongs to the whole list, not to the current route. Keep one
  // fresh observation while a newly selected dashboard is loading.
  const libraryStatus=useReadSnapshot(subject,'strategy-library-status',status.data,
    isRestricted(list.error)||isRestricted(status.error)).data;
  function observedState(strategyId:string){
    if(!libraryStatus||now-Date.parse(libraryStatus.generated_at)>=60000)return 'unknown';
    const value=strategyId===libraryStatus.selected_strategy.strategy_id?libraryStatus.runtime_status.producer_status:libraryStatus.strategy_selector?.items.find(row=>row.strategy_id===strategyId)?.status;
    return value==='live'?'running':value??'unknown';
  }
  const filtered = filterStrategies(newestFirst(items ?? []), query, '', '').filter(item=>(!market||market.split(',').includes(item.spec.market_type))&&(!timeframe||timeframe.split(',').includes(item.spec.timeframe))&&(!stateFilter||stateFilter.split(',').includes(observedState(item.strategy_id))));
  function filter(key: string, value: string) { setParams(old => { const next = new URLSearchParams(old); if (value) next.set(key, value); else next.delete(key); return next; }, { replace: true, flushSync: true }); }
  function selection(nextId: string) {
    const next = new URLSearchParams(params); next.delete('strategy_id'); next.delete('from_job'); next.delete('from_variant');
    return `/strategies/${nextId}${next.size ? `?${next}` : ''}`;
  }
  const defaultStrategy = !id && !list.isFetching && !list.error && !status.isPending && !status.isFetching && !isRestricted(status.error) ? preferredStrategy(filtered,observedState) : undefined;
  const defaultHref = defaultStrategy ? selection(defaultStrategy.strategy_id) : undefined;
  useEffect(() => {
    if (defaultHref) void navigate(defaultHref, {replace: true});
  }, [defaultHref, navigate]);
  const toolbar=<div className="strategy-panel-tools"><button className="strategy-list-toggle" aria-label={t(collapsed ? 'strategy.showList' : 'strategy.hideList')} title={t(collapsed ? 'strategy.showList' : 'strategy.hideList')} aria-expanded={!collapsed} aria-controls="strategies-library" onClick={() => setCollapsed(value => !value)}><PanelRight aria-hidden="true"/></button></div>;
  return <div className="strategies-page">
    <h1 className="sr-only" id="workspace-heading" tabIndex={-1}>{t('strategies')}</h1>
    <div className={`panes report-workspace strategies-panes ${listCollapsed ? 'list-collapsed' : ''}`}>

      <section className="panel context strategy-context" aria-label={t('strategy.workspace')}>
        {validId&&<ReadStatus pending={detail.isFetching||status.isFetching} retained={snapshot.retained}/>}
        {snapshot.retained&&<><StrategyError error={detail.error}/><StrategyError error={status.error}/></>}
        <div>{snapshot.retained&&(detail.error||status.error)&&<button disabled={!eligible.length} onClick={()=>eligible.forEach(read=>void read.refetch())}>{t('refresh')}</button>}</div><div inert={snapshot.retained&&displayed?.id!==id}>{id ? z.uuid().safeParse(id).success ? <StrategyDetail updating={status.isFetching} toolbar={toolbar} key={`${subject}:${displayed?.id??id}`} detail={displayed?.detail??detail} status={displayed?.status??status} subject={subject} id={displayed?.id??id} now={now} filteredOut={!!items?.some(item => item.strategy_id===id) && !filtered.some(item=>item.strategy_id===id)} /> : <>{toolbar}<p role="alert">{t('strategy.invalid')}</p></> : (list.isPending || (!!statusId&&status.isPending) || defaultHref) ? <LoadingData /> : <p className="empty">{t('strategy.inspect')}</p>}</div>

      </section>
      <div className="strategy-list-slot" inert={listCollapsed} aria-hidden={listCollapsed}><div className="strategy-list-clip"><section id="strategies-library" className="panel library navigator-library" aria-labelledby="strategies-library-heading">
        <div className="panel-head"><h2 id="strategies-library-heading">{t('strategy.library')}</h2><span className="count">{items?.length ?? '—'}</span><div className="actions"><button className="library-icon-action strategies-refresh" aria-label={t('refresh')} title={t('refresh')} disabled={!eligible.length} onClick={() => {eligible.forEach(read => void read.refetch());}}><RefreshCw aria-hidden="true" /></button><button ref={filterTrigger} type="button" className="library-icon-action" aria-label={t('filters')} title={t('filters')} aria-expanded={filtersOpen} aria-controls="strategy-library-filters" onClick={toggleFilters}><Filter aria-hidden="true"/></button></div></div>
        <div className="library-tabs view-switch" role="group" aria-label={t('strategy.statesLabel',{defaultValue:i18n.language.startsWith('ru')?'Состояние стратегии':'Strategy state'})}>{['','running','stopped'].map(value=><button type="button" key={value} aria-pressed={stateFilter===value} onClick={()=>filter('state',value)}>{value?t(`strategy.states.${value}`):t('strategy.all')}</button>)}</div>
        <div className="library-body">
          <section ref={filterPopup} popover="auto" role="dialog" onToggle={event=>setFiltersOpen(event.newState==='open')} id="strategy-library-filters" className="library-filter-popup strategy-filter-popover" aria-label={t('filters')} onKeyDown={event=>{if(event.key==='Escape'){event.preventDefault();event.stopPropagation();filterPopup.current?.hidePopover();filterTrigger.current?.focus();}}}><div className="strategy-filters"><label>{t('strategy.search')}<input maxLength={200} value={query} onChange={e => filter('q', e.target.value)} /></label>
            <LibraryFilter label={t('strategy.market')} all={t('strategy.all')} value={market} options={[...new Set(items?.map(item=>item.spec.market_type))].map(value=>({value,label:value}))} onChange={value=>filter('market',value)}/>
            <LibraryFilter label={t('timeframe')} all={t('strategy.all')} value={timeframe} options={[...new Set(items?.map(item=>item.spec.timeframe))].map(value=>({value,label:value}))} onChange={value=>filter('timeframe',value)}/>
            <LibraryFilter label={i18n.language.startsWith('ru')?'Состояние':'State'} all={t('strategy.all')} value={stateFilter} options={['running','stopped','starting','stopping','blocked','unknown'].map(value=>({value,label:t(`strategy.states.${value}`,{defaultValue:value})}))} onChange={value=>filter('state',value)}/>
            {<button className="library-filter-reset" disabled={!query&&!market&&!timeframe&&!stateFilter} onClick={() => setParams(old => { const next = new URLSearchParams(old); ['q','market','timeframe','state'].forEach(key => next.delete(key)); return next; }, {replace:true,flushSync:true})}>{t('resetFilters')}</button>}
          </div></section>
          <ReadStatus pending={list.isFetching}/>
          <StrategyError error={list.error} />
          {items && !filtered.length && <p className="empty">{t(items.length ? 'strategy.noMatches' : 'strategy.empty')}</p>}
          <div className="strategy-rows">{filtered.map(item => <MotionLink key={item.strategy_id} className="strategy-row" aria-current={id === item.strategy_id ? 'true' : undefined} to={selection(item.strategy_id)} preventScrollReset>
            <strong>{item.name}</strong><span>{item.spec.instrument_id.symbol} · {item.spec.market_type} · {item.spec.timeframe}</span><span className={`strategy-row-status ${observedState(item.strategy_id)==='running'?'is-running':''}`}><span aria-hidden="true" className="strategy-state-dot">●</span> {t(`strategy.states.${observedState(item.strategy_id)}`)}</span>
          </MotionLink>)}</div>
        </div>
      </section></div></div>

    </div>
  </div>;
}
type DetailQuery = UseQueryResult<Awaited<ReturnType<typeof readStrategy>>, Error>;
type StatusQuery = UseQueryResult<Awaited<ReturnType<typeof readStrategyStatus>>, Error>;
function StrategyDetail({toolbar,detail,status,subject,id,now,filteredOut,updating=false}:{updating?:boolean;toolbar:ReactNode;subject:string;detail:DetailQuery;status:StatusQuery;id:string;now:number;filteredOut:boolean}) {
  const {t,i18n} = useTranslation();
  const data = isRestricted(detail.error) ? undefined : detail.data;
  const observation = isRestricted(detail.error) || isRestricted(status.error) ? undefined : status.data;
  useEffect(() => { document.getElementById('selected-strategy-heading')?.focus({preventScroll:true}); }, [id, !!data]);
  return <>
    {!data&&!detail.isPending&&toolbar}{!data&&!detail.isPending&&<div className="panel-head"><h2 id="selected-strategy-heading" tabIndex={-1}>{t('strategy.saved')}</h2></div>}
    <div className="strategy-detail">
      {detail.isFetching&&!data && <LoadingData />}<StrategyError error={detail.error} />
      {data && <>
        {detail.isError && <p role="status" className="notice">{t('stale')}</p>}
        {filteredOut && <p role="status" className="notice">{t('strategy.filtered')}</p>}
        <StrategyOperations toolbar={toolbar} strategy={data} subject={subject} observation={observation} loading={updating||status.isFetching} now={now} error={status.error}><details><summary>{t('strategy.technical')}</summary><p className="freshness">{t('strategy.immutable')}</p><dl><dt>ID</dt><dd>{id}</dd><dt>{t('createdUtc')}</dt><dd>{formatDate(data.created_at,i18n.language)}</dd><dt>schema_version · spec_kind</dt><dd>{data.spec.schema_version} · {data.spec.spec_kind}</dd><dt>{t('instrument')}</dt><dd>{data.spec.instrument_key}</dd></dl><details><summary>{t('strategy.json')}</summary><pre>{JSON.stringify(data.spec,null,2)}</pre></details></details></StrategyOperations>
      </>}

    </div>
  </>;
}


function LibraryFilter({label,all,value,options,onChange}:{label:string;all:string;value:string;options:{value:string;label:string}[];onChange:(value:string)=>void}){
 const selected=value?value.split(',').filter(v=>v!=='none'):options.map(option=>option.value);
 return <div className="strategy-library-filter"><span>{label}</span><ChartDisplayMenu multiple label={label} value={!value?all:selected.length===1?options.find(option=>option.value===selected[0])?.label??selected[0]:`${label} · ${selected.length}`} options={[{label:all,checked:!value,onChange:checked=>onChange(checked?'':'none')},...options.map(option=>({label:option.label,checked:selected.includes(option.value),onChange:(checked:boolean)=>{const next=checked?[...selected,option.value]:selected.filter(v=>v!==option.value);onChange(next.length===options.length?'':next.join(',')||'none');}}))]}/></div>;
}
