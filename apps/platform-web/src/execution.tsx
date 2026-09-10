import { transitionUI } from './motion';
import { Results } from './results';
import { DeleteHistory, deletionIdle, type DeleteState } from './delete-history';
import { useEffect, useRef, useState } from 'react';
import { useQuery, useQueryClient } from '@tanstack/react-query';
import { Link } from 'react-router';
import { useTranslation } from 'react-i18next';
import { ApiError, readSession } from './api';
import { cancelJob, jobIdSchema, jobLabel, readJob, refreshDeadline, type Job } from './library-api';
import { ReadError, Refresh, formatDate, isRestricted, JobState } from './library';

export const terminal = (job: Job) => ['succeeded', 'failed', 'cancelled'].includes(job.state);
/** A delayed read/command response cannot roll back an observed terminal or newer snapshot. */
export function reconcileJob(previous: Job | undefined, incoming: Job): Job {
  if (!previous || previous.job_id !== incoming.job_id) return incoming;
  if (terminal(previous) && !terminal(incoming)) return previous;
  if (Date.parse(incoming.updated_at) < Date.parse(previous.updated_at)) return previous;
  if (Date.parse(incoming.generated_at) < Date.parse(previous.generated_at) && terminal(previous) === terminal(incoming)) return previous;
  return incoming;
}
export function nextJobRead(job: Job | undefined, received: number, error: Error | null, errorAt: number) {
  if (isRestricted(error) || (job && terminal(job))) return Infinity;
  return Math.max(job ? refreshDeadline(job, received) : 0, received + 2_000,
    errorAt + (error instanceof ApiError ? Math.max(5, error.retryAfterSeconds ?? 0) : error ? 5 : 0) * 1000);
}
interface CommandState { phase: 'idle' | 'sending' | 'unresolved' | 'rejected'; error: Error | null; deadline: number }
const idle: CommandState = { phase: 'idle', error: null, deadline: 0 };

export function JobEntry({ id, subject, now, backLink, active = true }: { id: string; subject: string; now: number; backLink: string; active?: boolean }) {
  const { t, i18n } = useTranslation(); const client = useQueryClient();
  const key = ['private', subject, 'job', id];
  const commandKey = ['private', subject, 'cancel', id];
  // Query cache retains the pending marker across client-side navigation, and is
  // cleared by the existing subject/logout boundary. Reload always starts with GET.
  const command = useQuery<CommandState>({ queryKey: commandKey, initialData: idle, enabled: false, staleTime: Infinity });
  const state = command.data!;
  const deletion = useQuery<DeleteState>({ queryKey: ['private', subject, 'delete', id], initialData: deletionIdle, enabled: false }).data!;
  const valid = jobIdSchema.safeParse(id).success;
  const restricted = isRestricted(state.error) || isRestricted(deletion.error);
  const query = useQuery({ queryKey: key, enabled: valid && !restricted, refetchOnMount: false, refetchOnReconnect: false,
    queryFn: async ({ signal }) => {
      const commandAtStart = client.getQueryData<CommandState>(commandKey);
      const incoming = await readJob(id, signal);
      if (incoming.job_id !== id) throw new ApiError('invalid-response', 200, 'failed');
      if (commandAtStart?.phase === 'rejected' && client.getQueryData(commandKey) === commandAtStart) client.setQueryData(commandKey, idle);
      return reconcileJob(client.getQueryData<Job>(key), incoming);
    } });
  const job = isRestricted(query.error) || restricted ? undefined : query.data;
  const done = !!job && terminal(job);
  const readDeadline = Math.max(nextJobRead(job, query.dataUpdatedAt, query.error, query.errorUpdatedAt), state.deadline);
  const manualDeadline = Math.max(job ? refreshDeadline(job, query.dataUpdatedAt) : 0, state.deadline);
  useEffect(() => {
    if (!valid || restricted || query.isFetching || !Number.isFinite(readDeadline)) return;
    const timer = setTimeout(() => void query.refetch(), Math.min(2_147_483_647, Math.max(0, readDeadline - Date.now())));
    return () => clearTimeout(timer);
  }, [valid, restricted, query.isFetching, readDeadline, query.refetch]);
  const controller = useRef<AbortController | null>(null); const lock = useRef(false);
  useEffect(() => () => {
    controller.current?.abort();
    if (lock.current && client.getQueryData(commandKey)) client.setQueryData(commandKey, { ...idle, phase: 'unresolved' });
  }, []);
  const dialog = useRef<HTMLDialogElement>(null); const trigger = useRef<HTMLButtonElement>(null);
  const status = useRef<HTMLDivElement>(null);
  const [open, setOpen] = useState(false);
  useEffect(() => { if (done && open) dialog.current?.close(); }, [done, open]);
  const waiting = state.phase === 'sending' || state.phase === 'unresolved' || !!job?.cancel_requested_at;
  const canCancel = !!job && !done && !waiting && !query.error && !restricted && now >= state.deadline;
  const stale = !!job && (query.isError || now - Date.parse(job.generated_at) > 60_000 || now - query.dataUpdatedAt > 60_000);
  const measured = !!job?.progress.updated_at && job.progress.total_units > 0;
  const staleProgress = !!job && !done && (!job.progress.updated_at || now - Date.parse(job.progress.updated_at) > 60_000);
  async function confirmCancel() {
    if (lock.current || !canCancel) return;
    lock.current = true;
    const abort = new AbortController(); controller.current = abort;
    client.setQueryData(commandKey, { phase: 'sending', error: null, deadline: 0 });
    dialog.current?.close();
    try {
      const identity = await client.fetchQuery({ queryKey: ['session', subject], queryFn: () => readSession(abort.signal), staleTime: 0, retry: false });
      if (abort.signal.aborted) return;
      if (identity.user_id !== subject) { client.setQueryData(['session', subject], identity); return; }
      // Identity outages close the private surface just like the shell's read gate.
      const latest = client.getQueryData<Job>(key);
      if (latest && terminal(latest)) { client.setQueryData(commandKey, idle); return; }
      const current = await cancelJob(id, abort.signal);
      if (abort.signal.aborted) return;
      client.setQueryData<Job>(key, old => reconcileJob(old, current));
      client.setQueryData(commandKey, { phase: terminal(current) ? 'idle' : 'unresolved', error: null, deadline: refreshDeadline(current, Date.now()) });
      void client.invalidateQueries({ queryKey: ['private', subject, 'jobs'], refetchType: 'none' });
    } catch (error) {
      if (abort.signal.aborted) return;
      const failure = error instanceof Error ? error : new Error('unavailable');
      const definite = failure instanceof ApiError && ['forbidden', 'not-found', 'conflict', 'rate-limited', 'validation'].includes(failure.kind);
      client.setQueryData(commandKey, { phase: definite ? 'rejected' : 'unresolved', error: failure,
        deadline: Date.now() + (failure instanceof ApiError ? failure.retryAfterSeconds ?? 0 : 0) * 1000 });
    } finally { lock.current = false; }
  }
  // Rejected 409/429 requires a new authoritative read before another explicit command.
  const needsRead = state.phase === 'rejected' && !restricted;
  return <><div className="panel-head"><h2 id="selected-job-heading" tabIndex={-1}>{job?jobLabel(job):t('selectedJob')}</h2>{valid && !restricted && <Refresh query={query} deadline={manualDeadline} now={now} />}</div>
    <div className="job-detail">
      {!valid ? <p role="alert" className="notice error">{t('invalidJob')}</p> : query.isPending ? <p role="status">{t('loadingJob')}</p> : null}
      <ReadError error={query.error} />{restricted && <ReadError error={deletion.error} />}{query.error instanceof ApiError && query.error.status === 404 && deletion.phase === 'unknown' && <p role="status" className="notice">{t('results.absent')}</p>}{!done && <ReadError error={state.error} />}
      {job && <><div className="job-status" ref={status} tabIndex={-1} role="status"><JobState job={job} /></div>
        {job.state!=='succeeded'&&<section className="execution-progress" aria-label={t('execution.progress')}>
          {!done && <><p>{t(measured ? 'execution.units' : 'execution.unmeasured', { processed: job.progress.processed_units, total: job.progress.total_units })}</p>
            {measured && <progress aria-label={t('execution.progress')} value={job.progress.percent} max={100} />}
            {staleProgress && <p className="notice">{t('execution.staleProgress')}</p>}
            {job.progress.updated_at && <p className="freshness">{t('execution.measuredAt')} · {formatDate(job.progress.updated_at, i18n.language)} UTC</p>}
            <p className="muted">{t('execution.eta')}</p><p className="muted">{t('execution.polling')}</p></>}
          {!done && (state.phase === 'sending' || state.phase === 'unresolved') && <p role="status" className="notice">{t(state.phase === 'sending' ? 'execution.sending' : 'execution.unresolved')}</p>}
          {!done && <button ref={trigger} disabled={!canCancel || needsRead} onClick={() => { setOpen(true); transitionUI(() => dialog.current?.showModal()); }}>{t('execution.cancel')}</button>}
          {done && <p>{t(`execution.${job.state}`)}</p>}
          {done && job.terminal_summary?.top_variants_count != null && <p>{t('execution.variants', { count: job.terminal_summary.top_variants_count })}</p>}
        </section>}
        {job.state === 'succeeded' && <Results active={active} job={id} subject={subject} now={now} />}
        <details className="job-information"><summary>{t('results.jobInformation')}</summary>        <p className="freshness">{t(stale ? 'stale' : 'snapshot')} · {formatDate(job.generated_at, i18n.language)} UTC</p>
        <dl><dt>{t('jobId')}</dt><dd className="identity">{job.job_id}</dd><dt>{t('instrument')}</dt><dd>{job.request.coordinates.exchange} · {job.request.coordinates.market_type} · {job.request.coordinates.symbol}</dd>
          <dt>{t('timeframe')}</dt><dd>{job.request.timeframe}</dd><dt>{t('period')}</dt><dd>{formatDate(job.request.time_range.start, i18n.language)} → {formatDate(job.request.time_range.end, i18n.language)} UTC</dd>
          <dt>{t('risk')}</dt><dd>{t(`risks.${job.request.risk_mode}`, { defaultValue: job.request.risk_mode })}</dd><dt>{t('createdUtc')}</dt><dd>{formatDate(job.created_at, i18n.language)}</dd>
          {job.started_at && <><dt>{t('execution.started')}</dt><dd>{formatDate(job.started_at, i18n.language)}</dd></>}{job.finished_at && <><dt>{t('execution.finished')}</dt><dd>{formatDate(job.finished_at, i18n.language)}</dd></>}</dl>
        <DeleteHistory job={job} subject={subject} now={now} backLink={backLink} read={() => query.refetch()} readAt={query.dataUpdatedAt} readError={query.error} canRead={!query.isFetching && !isRestricted(query.error) && now >= Math.max(manualDeadline, query.error instanceof ApiError ? query.errorUpdatedAt + Math.max(5, query.error.retryAfterSeconds ?? 0) * 1000 : 0)} />
        </details>
      </>}
    </div>
    <dialog ref={dialog} aria-labelledby="cancel-title" aria-describedby="cancel-help" onClose={() => { setOpen(false); (trigger.current && !trigger.current.disabled ? trigger.current : status.current)?.focus(); }} onKeyDown={event => {
      if (event.key !== 'Tab') return;
      const controls = event.currentTarget.querySelectorAll<HTMLButtonElement>('button:not([disabled])');
      const first = controls[0], last = controls[controls.length - 1];
      if (event.shiftKey && document.activeElement === first) { event.preventDefault(); last?.focus(); }
      else if (!event.shiftKey && document.activeElement === last) { event.preventDefault(); first?.focus(); }
    }}><div className="panel-head"><h2 id="cancel-title">{t('execution.confirmTitle')}</h2></div><div className="filter-fields">
      <p id="cancel-help">{t('execution.confirmHelp')}</p><p className="identity">{id}</p>
      <button autoFocus onClick={() => transitionUI(() => dialog.current?.close())}>{t('execution.keep')}</button>
      <button disabled={!canCancel || needsRead} onClick={() => void confirmCancel()}>{t('execution.confirm')}</button>
    </div></dialog>
  </>;
}
