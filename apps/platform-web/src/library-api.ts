import { z } from 'zod';
import { requestJson } from './api';

export const jobStates = ['queued', 'running', 'succeeded', 'failed', 'cancelled'] as const;
export const jobIdSchema = z.uuid();
const date = z.string().refine(value => Number.isFinite(Date.parse(value)));
// Allowlist only the fields used by the library; private artifact metadata/paths never enter its cache.
export const jobSchema = z.object({
  job_id: jobIdSchema, state: z.enum(jobStates), created_at: date, updated_at: date,
  generated_at: date, refresh_status: z.enum(['poll', 'terminal']), next_allowed_refresh_at: date,
  retry_after_seconds: z.number().nonnegative(), cancel_requested_at: date.nullable(),
  started_at: date.nullable().optional(), finished_at: date.nullable().optional(),
  terminal_summary: z.object({ top_variants_count: z.number().nonnegative().nullable().optional() }).optional(),
  progress: z.object({ percent: z.number().min(0).max(100), processed_units: z.number().nonnegative(),
    total_units: z.number().nonnegative(), updated_at: date.nullable(), pipeline_stage: z.string() }),
  request: z.object({
    ui_metadata: z.object({ strategy_name: z.string().optional() }).optional(),
    coordinates: z.object({ exchange: z.string(), market_type: z.string(), symbol: z.string() }),
    timeframe: z.string(), time_range: z.object({ start: date, end: date }),
    risk_mode: z.string(),
  }),
});
export type Job = z.infer<typeof jobSchema>;
export const listSchema = z.object({ items: z.array(jobSchema), next_cursor: z.string().nullable() });
const workstationSchema = z.object({ generated_at: date, refresh_status: z.string(),
  retry_after_seconds: z.number().nullable(), next_allowed_refresh_at: date.nullable(),
  sources: z.array(z.object({ name: z.string(), status: z.enum(['available', 'degraded', 'unavailable']) })),
  job_table: z.object({ state: z.enum(['ready', 'empty', 'degraded', 'unavailable']) }),
});

export function libraryParams(input: URLSearchParams): URLSearchParams {
  const result = new URLSearchParams();
  const state = input.get('state');
  if (jobStates.includes(state as typeof jobStates[number])) result.set('state', state!);
  const risk = input.get('risk_mode');
  if (risk === 'none' || risk === 'tp_sl_grid') result.set('risk_mode', risk);
  const limit = input.get('limit');
  if (limit && /^\d+$/.test(limit) && Number(limit) >= 1 && Number(limit) <= 250) result.set('limit', String(Number(limit)));
  const cursor = input.get('cursor');
  if (cursor && cursor.length <= 4096 && /^[A-Za-z0-9_=-]+$/.test(cursor)) result.set('cursor', cursor);
  return result;
}
export function readJobs(params: URLSearchParams, signal: AbortSignal) {
  const query = libraryParams(params);
  if (!query.has('limit')) query.set('limit', '50');
  return requestJson(`/api/backtests/jobs?${query}`, listSchema, { signal }).then(reply => reply.data);
}
export function readJob(id: string, signal: AbortSignal) {
  jobIdSchema.parse(id);
  return requestJson(`/api/backtests/jobs/${encodeURIComponent(id)}`, jobSchema, { signal }).then(reply => reply.data);
}
export function cancelJob(id: string, signal: AbortSignal) {
  jobIdSchema.parse(id);
  return requestJson(`/api/backtests/jobs/${encodeURIComponent(id)}/cancel`, jobSchema, { method: 'POST', signal }).then(reply => {
    if (reply.data.job_id !== id) throw new Error('Mismatched job identity');
    return { ...reply.data, retry_after_seconds: Math.max(reply.data.retry_after_seconds, reply.retryAfterSeconds ?? 0) };
  });
}
export function readWorkstation(signal: AbortSignal) {
  return requestJson('/api/ui/backtests/workstation', workstationSchema, { signal }).then(reply => reply.data);
}
export function jobLabel(job: Job) { return job.request.ui_metadata?.strategy_name || job.job_id; }
export function refreshDeadline(data: { retry_after_seconds?: number | null; next_allowed_refresh_at?: string | null }, received: number) {
  return Math.max(received + (data.retry_after_seconds ?? 0) * 1000,
    data.next_allowed_refresh_at ? Date.parse(data.next_allowed_refresh_at) : 0);
}
