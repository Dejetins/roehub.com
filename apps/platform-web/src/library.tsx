import { MotionLink as Link, useMotionState, transitionUI } from './motion';
import { useEffect, useRef, useState, type ReactNode } from 'react';
import { useQuery, type UseQueryResult } from '@tanstack/react-query';
import { useParams, useSearchParams } from 'react-router';
import { useTranslation } from 'react-i18next';
import { ArrowLeft, ArrowRight, Filter, Plus, RefreshCw, X } from 'lucide-react';
import { useLocation } from 'react-router';
import { JobEntry } from './execution';
import { ApiError } from './api';
import { jobLabel, jobStates, libraryParams, readJobs, readWorkstation, refreshDeadline, type Job } from './library-api';

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
export function Refresh({ query, deadline = 0, now }: { query: UseQueryResult<unknown, Error>; deadline?: number; now: number }) {
  const { t } = useTranslation();
  const delay = query.error instanceof ApiError ? query.error.retryAfterSeconds ?? 0 : 0;
  const wait = Math.max(0, Math.ceil((Math.max(deadline, query.errorUpdatedAt + delay * 1000) - now) / 1000));
  return <button type="button" disabled={query.isFetching || wait > 0 || isRestricted(query.error)}
    onClick={() => void query.refetch()} aria-label={t('refresh')}>
    <RefreshCw aria-hidden="true" />{query.isFetching ? t('refreshing') : wait ? t('wait', { seconds: wait }) : t('refresh')}
  </button>;
}
export function JobState({ job }: { job: Job }) {
  const { t } = useTranslation();
  return <span className={`job-state state-${job.state}`}>{t(`states.${job.state}`)}{
    job.cancel_requested_at && ['queued', 'running'].includes(job.state) ? ` · ${t('cancelPending')}` : ''}</span>;
}

export function BacktestsWorkspace({ subject, mode = 'list', embedded = false, active = true, onNew, configuration }: { subject: string; mode?: 'list' | 'new' | 'detail'; embedded?: boolean; active?: boolean; onNew?: () => void; configuration?: ReactNode }) {
  const { t, i18n } = useTranslation();
  const { jobId } = useParams();
  const [historyCollapsed,setHistoryCollapsed]=useMotionState(false, 'layout');
  const [params, setParams] = useSearchParams();
  const normalized = libraryParams(params);
  const listQuery = normalized.toString();
  const variant = params.get('variant');
  if (mode === 'detail' && variant && variant.length <= 2048 && !/[\x00-\x1f\x7f]/.test(variant)) normalized.set('variant', variant);
  const routeQuery = normalized.toString();
  useEffect(() => { if (active && params.toString() !== routeQuery) setParams(routeQuery, { replace: true }); }, [active, params, routeQuery, setParams]);
  const now = useNow();
  const location = useLocation();
  const jobs = useQuery({ queryKey: ['private', subject, 'jobs', listQuery],
    refetchOnMount: false, queryFn: ({ signal }) => readJobs(new URLSearchParams(listQuery), signal) });
  const workstation = useQuery({ queryKey: ['private', subject, 'workstation'], refetchOnMount: false, queryFn: ({ signal }) => readWorkstation(signal) });
  const dialog = useRef<HTMLDialogElement>(null);
  const filterTrigger = useRef<HTMLButtonElement>(null);
  const hasFilters = !!listQuery;
  const data = isRestricted(jobs.error) ? undefined : jobs.data;
  const stale = !!data && (jobs.isError || now - jobs.dataUpdatedAt > 60_000 || data.items.some(job => now - Date.parse(job.generated_at) > 60_000));
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
    <div className={`${configuration ? "panes with-configuration" : "panes"} ${mode==='detail'?'report-workspace':''} ${historyCollapsed?'history-collapsed':''}`}>
      {mode==='detail'&&<button className="history-toggle" aria-expanded={!historyCollapsed} onClick={()=>setHistoryCollapsed(v=>!v)}>{t(historyCollapsed?'results.showHistory':'results.hideHistory')}</button>}
      <section className="panel library" aria-labelledby="jobs-heading">
        <div className="panel-head"><h2 id="jobs-heading">{t('jobs')}</h2>
          {data && <span className="count" aria-label={t('pageCount')}>{data.items.length}</span>}
          <div className="actions"><Link className="button-link" data-client-link="true" to={`/backtests/new${listQuery ? `?${listQuery}` : ''}`} aria-label={t('new')} onClick={event => { if (onNew && !event.metaKey && !event.ctrlKey && !event.shiftKey && !event.altKey) { event.preventDefault(); onNew(); } }}><Plus aria-hidden="true" />{t('new')}</Link>
            <Refresh query={jobs} deadline={deadline} now={now} />
            <button ref={filterTrigger} onClick={() => transitionUI(() => dialog.current?.showModal())} aria-label={t('filters')}><Filter aria-hidden="true" />{t('filters')}</button></div>
        </div>
        {configuration}
        <div className="library-body">{location.state?.historyDeleted && <p role="status">{t('results.removed')}</p>}

          <p className="filter-summary">{t('filterSummary', { state: params.get('state') ? t(`states.${params.get('state')}`) : t('allStates'),
            risk: params.get('risk_mode') ? t(`risks.${params.get('risk_mode')}`) : t('allRisk') })}</p>
          <div role="status" className={stale ? 'notice' : 'freshness'}>{jobs.isPending ? t('loadingJobs') : data ?
            `${t(stale ? 'stale' : 'snapshot')} · ${formatDate(new Date(jobs.dataUpdatedAt).toISOString(), i18n.language)} UTC` : ''}</div>
          <ReadError error={jobs.error} />
          {data && data.items.length === 0 && <div className="empty"><h3>{t(data.next_cursor ? 'emptyPage' : hasFilters ? 'noMatches' : 'empty')}</h3>
            <p>{t(data.next_cursor ? 'emptyPageHelp' : hasFilters ? 'noMatchesHelp' : 'emptyHelp')}</p></div>}
          {data && data.items.length > 0 && <div className="table-scroll" role="region" aria-label={t('jobTable')} tabIndex={0}>
            <table><caption className="sr-only">{t('pageCount')}</caption><thead><tr><th scope="col">{t('job')}</th><th scope="col">{t('state')}</th><th scope="col">{t('createdUtc')}</th></tr></thead>
              <tbody>{data.items.map(job => <tr key={job.job_id} className={jobId === job.job_id ? 'selected' : ''}>
                <td><Link aria-current={jobId === job.job_id ? 'page' : undefined} to={`/backtests/${encodeURIComponent(job.job_id)}${listQuery ? `?${listQuery}` : ''}`}>
                  {jobLabel(job)}</Link><span className="job-meta">{job.request.coordinates.symbol} · {job.request.coordinates.market_type} · {job.request.timeframe}</span></td>
                <td><JobState job={job} />{now - Date.parse(job.generated_at) > 60_000 && <span className="job-meta">{t('stale')}</span>}</td>
                <td><time dateTime={job.created_at}>{formatDate(job.created_at, i18n.language)}</time></td>
              </tr>)}</tbody></table>
          </div>}
          <nav className="pagination" aria-label={t('pagination')}>
            <button disabled={!params.has('cursor')} onClick={() => change('cursor', '')}><ArrowLeft aria-hidden="true" />{t('firstPage')}</button>
            <span>{t('pageCount')}</span>
            <button disabled={!data?.next_cursor || jobs.isFetching || !!jobs.error} onClick={() => change('cursor', data!.next_cursor!)}>{t('nextPage')}<ArrowRight aria-hidden="true" /></button>
          </nav>
          <p className="muted compact">{t('riskPagination')}</p>
          <details className="projection"><summary>{t('extendedFilters')}</summary>
            <p className="notice">{t(workstation.isPending ? 'loadingProjection' : workstation.error ? 'projectionError' :
              workstation.data?.job_table.state === 'unavailable' ? 'projectionUnavailable' : 'projectionNotBound')}</p>
            <ReadError error={workstation.error} />
            <Refresh query={workstation} deadline={workstation.data ? refreshDeadline(workstation.data, workstation.dataUpdatedAt) : 0} now={now} />
            <fieldset disabled aria-label={t('extendedFilters')}><label>{t('search')}<input type="search" /></label>
              <label>{t('instrument')}<input /></label><label>{t('from')}<input type="date" /></label><label>{t('to')}<input type="date" /></label></fieldset>
          </details>
        </div>
      </section>
      <section className="panel context" aria-label={t(mode === 'detail' ? 'detail' : mode === 'new' ? 'new' : 'selection')}>
        {mode === 'detail' ? <JobEntry active={active} key={jobId} id={jobId ?? ''} subject={subject} now={now} backLink={backLink} /> :
          <div className="context-empty"><h2>{t(mode === 'new' ? 'new' : 'selection')}</h2>
            <p>{t(mode === 'new' ? 'futureBuilder' : 'selectHelp')}</p>
            {mode === 'new' && <Link to={backLink}>{t('backToList')}</Link>}
            <p className="muted">{t('futureResults')}</p></div>}
      </section>
    </div>
    <dialog ref={dialog} aria-labelledby="filter-title" onKeyDown={event => {
      if (event.key !== 'Tab') return;
      const controls = Array.from(event.currentTarget.querySelectorAll<HTMLElement>('button:not([disabled]), input:not([disabled]), select:not([disabled]), a[href]'));
      const first = controls[0]; const last = controls[controls.length - 1];
      if (event.shiftKey && document.activeElement === first) { event.preventDefault(); last?.focus(); }
      else if (!event.shiftKey && document.activeElement === last) { event.preventDefault(); first?.focus(); }
    }} onClose={() => filterTrigger.current?.focus()}>
      <div className="panel-head"><h2 id="filter-title">{t('filters')}</h2><button onClick={() => transitionUI(() => dialog.current?.close())} aria-label={t('close')}><X aria-hidden="true" /></button></div>
      <div className="filter-fields"><label>{t('state')}<select value={params.get('state') ?? ''} onChange={event => change('state', event.target.value)}>
        <option value="">{t('allStates')}</option>{jobStates.map(state => <option key={state} value={state}>{t(`states.${state}`)}</option>)}</select></label>
        <label>{t('risk')}<select value={params.get('risk_mode') ?? ''} onChange={event => change('risk_mode', event.target.value)}>
          <option value="">{t('allRisk')}</option><option value="none">{t('risks.none')}</option><option value="tp_sl_grid">{t('risks.tp_sl_grid')}</option></select></label>
        <label>{t('pageSize')}<input type="number" min="1" max="250" defaultValue={params.get('limit') ?? '50'} key={params.get('limit')}
          onBlur={event => { if (event.target.validity.valid && event.target.value) change('limit', event.target.value); }} /></label>
        <p className="muted">{t('filterHelp')}</p><button onClick={() => { setParams(mode === 'detail' && normalized.has('variant') ? { variant: normalized.get('variant')! } : {}); }}>{t('resetFilters')}</button>
        <button className="primary" onClick={() => transitionUI(() => dialog.current?.close())}>{t('done')}</button></div>
    </dialog>
  </>;
}
