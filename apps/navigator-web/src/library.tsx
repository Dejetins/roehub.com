import {useLibraryFilterPopup} from './library-filter-popup';
import {useReadSnapshot} from './read-snapshot';
import {JobSelection} from './job-selection';
import { LoadingData, ReadStatus } from './loading-data';
import {ChartDisplayMenu} from './chart-display-menu';
import { MotionLink as Link, useMotionState } from './motion';
import { useEffect, useRef, useState, type ReactNode } from 'react';
import { useQuery, type UseQueryResult } from '@tanstack/react-query';
import { useNavigate, useParams, useSearchParams } from 'react-router';
import { useTranslation } from 'react-i18next';
import { ChevronsLeft, ArrowRight, Filter, Plus, RefreshCw, Search, CircleCheck, CircleX, Clock, LoaderCircle, CircleSlash } from 'lucide-react';
import { useLocation } from 'react-router';
import { ApiError } from './api';
import { jobLabel, jobStates, libraryParams, readJobs, refreshDeadline, type Job } from './library-api';

function useNow() {
  const [now, setNow] = useState(Date.now);
  useEffect(() => { const timer = setInterval(() => setNow(Date.now()), 1000); return () => clearInterval(timer); }, []);
  return now;
}
export function formatDate(value: string, locale: string) {
  return new Intl.DateTimeFormat(locale, { dateStyle: 'medium', timeStyle: 'short', timeZone: 'UTC' }).format(new Date(value));
}
export function isRestricted(error: Error | null) {
  return error instanceof ApiError && ['unauthenticated', 'forbidden', 'not-found'].includes(error.kind);
}
export function ReadError({ error }: { error: Error | null }) {
  const { t } = useTranslation();
  if (!error) return null;
  const kind = error instanceof ApiError ? error.kind : 'unavailable';
  return <p className="notice error" role="alert">{t(`errors.${kind}`)}</p>;
}
export function Refresh({ query, deadline = 0, now, iconOnly = false }: { query: UseQueryResult<unknown, Error>; deadline?: number; now: number; iconOnly?: boolean }) {
  const { t } = useTranslation();
  const delay = query.error instanceof ApiError ? query.error.retryAfterSeconds ?? 0 : 0;
  const wait = Math.max(0, Math.ceil((Math.max(deadline, query.errorUpdatedAt + delay * 1000) - now) / 1000));
  return <button type="button" className={iconOnly?'library-icon-action':undefined} title={iconOnly?(query.isFetching?t('refreshing'):wait?t('wait',{seconds:wait}):t('refresh')):undefined} disabled={query.isFetching || wait > 0 || isRestricted(query.error)}
    onClick={() => void query.refetch()} aria-label={t('refresh')}>
    <RefreshCw aria-hidden="true" />{!iconOnly&&(query.isFetching ? t('refreshing') : wait ? t('wait', { seconds: wait }) : t('refresh'))}
  </button>;
}
export function JobState({ job }: { job: Job }) {
  const { t } = useTranslation();
  return <span className={`job-state state-${job.state}`}>{t(`states.${job.state}`)}{
    job.cancel_requested_at && ['queued', 'running'].includes(job.state) ? ` · ${t('cancelPending')}` : ''}</span>;
}

export function BacktestsWorkspace({ subject, mode = 'list', embedded = false, active = true, onNew, configuration, configurationActive=false, onHistory }: { subject: string; mode?: 'list' | 'new' | 'detail'; embedded?: boolean; active?: boolean; onNew?: () => void; configuration?: ReactNode; configurationActive?:boolean; onHistory?:()=>void }) {
  const { t, i18n } = useTranslation();
  const { jobId } = useParams();
  const navigate = useNavigate();
  const [historyCollapsed,setHistoryCollapsed]=useMotionState(false, 'layout');
  const [params, setParams] = useSearchParams();
  const normalized = libraryParams(params);
  if(normalized.get('limit')!=='10')normalized.delete('cursor');
  normalized.set('limit','10');
  const listQuery = normalized.toString();
  const variant = params.get('variant');
  if (mode === 'detail' && variant && variant.length <= 2048 && !/[\x00-\x1f\x7f]/.test(variant)) normalized.set('variant', variant);
  const routeQuery = normalized.toString();
  useEffect(() => { if (active && params.toString() !== routeQuery) setParams(routeQuery, { replace: true }); }, [active, params, routeQuery, setParams]);
  const now = useNow();
  const location = useLocation();
  const jobs = useQuery({ queryKey: ['private', subject, 'jobs', listQuery],
    refetchOnMount: false, queryFn: ({ signal }) => readJobs(new URLSearchParams(listQuery), signal) });
  const {open:filtersOpen,setOpen:setFiltersOpen,trigger:filterTrigger,popup:filterPopup,toggle:toggleFilters,close:closeFilters}=useLibraryFilterPopup();
  const [searchOpen,setSearchOpen]=useState(false);
  const searchTrigger=useRef<HTMLButtonElement>(null);
  useEffect(()=>{if(configurationActive)filterPopup.current?.hidePopover();},[configurationActive]);
  const hasFilters = normalized.has('state')||normalized.has('risk_mode');
  const listSnapshot=useReadSnapshot(subject,listQuery,jobs.data&&!jobs.isFetching&&!jobs.error?{data:jobs.data,received:jobs.dataUpdatedAt}:undefined,isRestricted(jobs.error));
  const data = listSnapshot.data?.data;
  // Server history is ordered newest first. Never redirect a deep link or builder,
  // or select from a retained list belonging to a previous filter request.
  const defaultJob = active && mode === 'list' && !configurationActive && !jobs.isFetching && !jobs.error
    ? jobs.data?.items[0]?.job_id : undefined;
  useEffect(() => {
    if (defaultJob) void navigate(`/backtests/${encodeURIComponent(defaultJob)}?${listQuery}`, {replace: true});
  }, [defaultJob, listQuery, navigate]);


  const deadline = Math.max(0, ...(data?.items.map(job => refreshDeadline(job, jobs.dataUpdatedAt)) ?? []));
  function change(key: string, value: string) {
    const next = new URLSearchParams(listQuery);
    next.delete('cursor');
    if (value) next.set(key, value); else next.delete(key);
    if (mode === 'detail' && normalized.has('variant')) next.set('variant', normalized.get('variant')!);
    setParams(next);
  }
  const backLink = `/backtests${listQuery ? `?${listQuery}` : ''}`;
  return <>
    {!embedded && <div className="workspace-title"><h1 id="workspace-heading" tabIndex={-1}>{t(mode === 'detail' ? 'detail' : mode === 'new' ? 'new' : 'title')}</h1>
      <span className="scope-label">{t('research')}</span></div>}
    <div className={`${configuration ? "panes with-configuration" : "panes"} ${mode==='detail'?'report-workspace backtest-detail-workspace':''} ${historyCollapsed?'history-collapsed':''}`}>
      {mode==='detail'&&<button className="history-toggle navigator-history-toggle" aria-expanded={!historyCollapsed} onClick={()=>setHistoryCollapsed(v=>!v)}>{t(historyCollapsed?'results.showHistory':'results.hideHistory')}</button>}
      <section className="panel library navigator-library" aria-labelledby="jobs-heading">
        <div className="panel-head"><h2 id="jobs-heading">{t('jobs')}</h2>
          {data && <span className="count" aria-label={t('pageCount')}>{data.items.length}</span>}
          <Link className="button-link library-icon-action" title={t('new')} data-client-link="true" to={`/backtests/new${listQuery ? `?${listQuery}` : ''}`} aria-label={t('new')} onClick={event => { if (onNew && !event.metaKey && !event.ctrlKey && !event.shiftKey && !event.altKey) { event.preventDefault(); onNew(); } }}><Plus aria-hidden="true" /></Link>
            {!configurationActive&&<button ref={searchTrigger} type="button" className="library-icon-action" aria-label={t('search')} title={t('search')} aria-expanded={searchOpen} aria-controls="navigator-job-search" onClick={()=>setSearchOpen(value=>!value)}><Search aria-hidden="true"/></button>}
            {!configurationActive&&<div className="actions"><Refresh iconOnly query={jobs} deadline={deadline} now={now} />
            <div className="library-filter-menu">
            <button className="library-icon-action" title={t('filters')} ref={filterTrigger} onClick={toggleFilters} aria-label={t('filters')} aria-expanded={filtersOpen} aria-controls="library-filter-popup"><Filter aria-hidden="true" /></button>
            <section ref={filterPopup} popover="auto" onToggle={event=>setFiltersOpen(event.newState==='open')} id="library-filter-popup" className="library-filter-popup strategy-filter-popover" role="dialog" aria-label={t('filters')} onKeyDown={event=>{if(event.key==='Escape'){event.preventDefault();closeFilters();}}}>
              <div className="library-filter-field"><span>{t('state')}</span><ChartDisplayMenu label={t('state')} value={params.get('state')?t(`states.${params.get('state')}`):t('allStates')} options={['',...jobStates].map(value=>({label:value?t(`states.${value}`):t('allStates'),checked:(params.get('state')??'')===value,onChange:()=>change('state',value)}))}/></div>
              <div className="library-filter-field"><span>{t('risk')}</span><ChartDisplayMenu label={t('risk')} value={params.get('risk_mode')?t(`risks.${params.get('risk_mode')}`):t('allRisk')} options={['','none','tp_sl_grid'].map(value=>({label:value?t(`risks.${value}`):t('allRisk'),checked:(params.get('risk_mode')??'')===value,onChange:()=>change('risk_mode',value)}))}/></div>

              <button className="library-filter-reset" disabled={!hasFilters} onClick={()=>setParams(mode==='detail'&&normalized.has('variant')?{variant:normalized.get('variant')!}:{})}>{t('resetFilters')}</button>
            </section>
            </div></div>}
        </div>
        {configuration&&<div className="result-tabs library-tabs" role="tablist" aria-label={t('jobs')}>
          {[{id:'new',label:t('new'),selected:configurationActive,action:onNew},{id:'history',label:i18n.language.startsWith('ru')?'История':'History',selected:!configurationActive,action:onHistory}].map((tab,index)=><button key={tab.id} id={`library-tab-${tab.id}`} role="tab" aria-selected={tab.selected} aria-controls={`library-panel-${tab.id}`} tabIndex={tab.selected?0:-1} onClick={tab.action} onKeyDown={event=>{if(['ArrowLeft','ArrowRight','Home','End'].includes(event.key)){event.preventDefault();const next=event.key==='Home'?'new':event.key==='End'?'history':index===0?'history':'new';(next==='new'?onNew:onHistory)?.();document.getElementById(`library-tab-${next}`)?.focus();}}}>{tab.label}</button>)}
        </div>}
        {configuration&&<div id="library-panel-new" role="tabpanel" aria-labelledby="library-tab-new" hidden={!configurationActive}>{configuration}</div>}
        <div className="library-body" id="library-panel-history" role={configuration?'tabpanel':undefined} aria-labelledby={configuration?'library-tab-history':undefined} hidden={!!configuration&&configurationActive}>{location.state?.historyDeleted && <p role="status">{t('results.removed')}</p>}

          {searchOpen&&<section id="navigator-job-search" className="navigator-job-search" aria-label={t('extendedFilters')} onKeyDown={event=>{if(event.key==='Escape'){setSearchOpen(false);searchTrigger.current?.focus();}}}><h3>{t('extendedFilters')}</h3>
            <fieldset disabled aria-label={t('extendedFilters')}><label>{t('search')}<input type="search" /></label>
              <label>{t('instrument')}<input /></label><label>{t('from')}<input type="date" /></label><label>{t('to')}<input type="date" /></label></fieldset>
          </section>}
          {(jobs.isFetching||jobs.error)&&<ReadStatus pending={jobs.isFetching} retained={listSnapshot.retained}/>}
          <ReadError error={jobs.error} />
          {data && data.items.length === 0 && <div className="empty"><h3>{t(data.next_cursor ? 'emptyPage' : hasFilters ? 'noMatches' : 'empty')}</h3>
            <p>{t(data.next_cursor ? 'emptyPageHelp' : hasFilters ? 'noMatchesHelp' : 'emptyHelp')}</p></div>}
          {data && data.items.length > 0 && <div className="table-scroll" role="region" aria-label={t('jobTable')} tabIndex={0}>
            <table><caption className="sr-only">{t('pageCount')}</caption><thead><tr><th scope="col">{t('job')}</th><th scope="col">{t('state')}</th><th scope="col">{t('createdUtc')}</th></tr></thead>
              <tbody>{data.items.map(job => <tr key={job.job_id} className={jobId === job.job_id ? 'selected' : ''}>
                <td><Link aria-current={jobId === job.job_id ? 'page' : undefined} to={`/backtests/${encodeURIComponent(job.job_id)}${listQuery ? `?${listQuery}` : ''}`}>
                  {jobLabel(job)}</Link><span className="job-meta">{job.request.coordinates.symbol} · {job.request.coordinates.market_type} · {job.request.timeframe}</span></td>
                <td className="navigator-job-state"><span role="img" aria-label={t(`states.${job.state}`)} title={t(`states.${job.state}`)} className={`navigator-status-icon state-${job.state}`}>{job.state==='succeeded'?<CircleCheck/>:job.state==='failed'?<CircleX/>:job.state==='running'?<LoaderCircle/>:job.state==='cancelled'?<CircleSlash/>:<Clock/>}</span></td>
                <td><time dateTime={job.created_at}>{formatDate(job.created_at, i18n.language)}</time></td>
              </tr>)}</tbody></table>
          </div>}
          <nav className="pagination library-pagination" aria-label={t('pagination')}>
            <span className="navigator-page-count">{data?.items.length??'—'} {i18n.language.startsWith('ru')?'записей':'jobs'}</span><div className="navigator-page-arrows">
            <button title={t('firstPage')} aria-label={t('firstPage')} disabled={!params.has('cursor')} onClick={() => change('cursor', '')}><ChevronsLeft aria-hidden="true" /></button>
            <button title={t('nextPage')} aria-label={t('nextPage')} disabled={!data?.next_cursor || jobs.isFetching || listSnapshot.retained || !!jobs.error} onClick={() => change('cursor', data!.next_cursor!)}><ArrowRight aria-hidden="true" /></button>
          </div></nav>

        </div>
      </section>
      <section className="panel context" aria-label={t(mode === 'detail' ? 'detail' : mode === 'new' ? 'new' : 'selection')}>
        {mode === 'detail' ? <JobSelection active={active} id={jobId ?? ''} subject={subject} now={now} backLink={backLink} /> : mode === 'list' && (jobs.isPending || defaultJob) ? <LoadingData /> :
          <div className="context-empty"><h2>{t(mode === 'new' ? 'new' : 'selection')}</h2>
            <p>{t(mode === 'new' ? 'futureBuilder' : 'selectHelp')}</p>
            {mode === 'new' && <Link to={backLink}>{t('backToList')}</Link>}
            <p className="muted">{t('futureResults')}</p></div>}
      </section>
    </div>

  </>;
}
