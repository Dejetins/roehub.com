import {operationalDashboard} from './strategy-operations-api';
import { z } from 'zod';
import { profileFields, tradingObservation } from './strategy-insights-api';
import { ApiError, requestJson } from './api';
import { variantKeySchema } from './results-api';
import { refreshDeadline } from './library-api';

const date = z.string().refine(value => Number.isFinite(Date.parse(value)));
export const strategySchema = z.object({
  strategy_id: z.uuid(), user_id: z.uuid(), name: z.string(), created_at: date, is_deleted: z.boolean(),
  spec: z.object({ instrument_id: z.object({ market_id: z.number().int(), symbol: z.string() }),
    instrument_key: z.string(), market_type: z.string(), timeframe: z.string(),
    indicators: z.array(z.record(z.string(), z.json())), signal_template: z.string(),
    schema_version: z.number().int(), spec_kind: z.string() }),
});
export type Strategy = z.infer<typeof strategySchema>;
export async function readStrategies(subject: string, signal: AbortSignal) {
  const { data } = await requestJson('/api/strategies', z.array(strategySchema), { signal });
  if (data.some(item => item.user_id !== subject || item.is_deleted)) throw new ApiError('invalid-response', 200, 'failed');
  return data;
}
export async function readStrategy(subject: string, id: string, signal: AbortSignal) {
  const { data } = await requestJson(`/api/strategies/${encodeURIComponent(id)}`, strategySchema, { signal });
  if (data.strategy_id !== id || data.user_id !== subject || data.is_deleted) throw new ApiError('invalid-response', 200, 'failed');
  return data;
}
const reason = z.string().transform(value => [
  'unsupported_strategy_schema_version','unsupported_strategy_spec_kind','unsupported_live_evaluator',
  'timeframe_rollup_required','large_warmup_live_start_slow','supported_live_evaluator',
].includes(value) ? value : 'status_unavailable');
const panel = z.enum(['ready', 'empty', 'degraded', 'unavailable']);
const refresh = { next_allowed_refresh_at: date.nullable(), retry_after_seconds: z.number().nonnegative().nullable() };
const dashboardSchema = z.object({ ...tradingObservation, generated_at: date, refresh_status: z.enum(['fresh', 'degraded', 'rate_limited']), ...refresh,
  ...operationalDashboard,
  sources: z.array(z.object({ name: z.string(), status: z.enum(['available','degraded','unavailable']),
    generated_at: date.nullable().optional(), age_seconds: z.number().nullable().optional(),
    next_allowed_refresh_at: date.nullable().optional(), retry_after_seconds: z.number().nullable().optional() })),
  runtime_status: z.object({ state: panel, producer_status: z.enum(['running','stopped','blocked','unknown']), environment: z.enum(['monitor_only','paper','testnet','mainnet_unavailable','unknown']), run_state: z.string().nullable(), run_id:z.string().nullable().optional(), run_updated_at:date.nullable().optional() }),
  live_profile: z.object({ ...profileFields, state: panel, readiness_status: z.enum(['ready','blocked']), readiness_reason: reason }),
  compatibility_readiness: z.object({ state: panel, compatibility_state: z.enum(['launchable','not_launchable','degraded']), compatibility_reason_codes: z.array(reason),
    market_data_state: z.enum(['ready','missing','stale','pending']), market_data_reason_codes: z.array(reason), checked_at: date.nullable() }),
  refresh_control: z.object({ ...refresh, manual_refresh_available: z.boolean(), interval_seconds: z.number().nonnegative() }),
});
export async function readStrategyStatus(id: string, signal: AbortSignal) {
  // A bounded initial/manual read; no polling and no direct compatibility command.
  const reply = await requestJson(`/api/ui/strategies/dashboard?${new URLSearchParams({strategy_id: id, refresh: 'manual'})}`, dashboardSchema, { signal });
  if (reply.data.selected_strategy.strategy_id !== id) throw new ApiError('invalid-response', 200, 'failed');
  const received = Date.now();
  return { ...reply.data, deadline: Math.max(refreshDeadline(reply.data, received),
    refreshDeadline(reply.data.refresh_control, received), received + (reply.retryAfterSeconds ?? 0) * 1000,
    received + reply.data.refresh_control.interval_seconds * 1000,
    ...reply.data.sources.map(source => refreshDeadline(source, received))) };
}
/** Navigation context is neither provenance nor an authorization grant. */
export function reportReturn(params: URLSearchParams): string | null {
  const job = params.get('from_job'), variant = params.get('from_variant');
  if (!z.uuid().safeParse(job).success || !variantKeySchema.safeParse(variant).success) return null;
  return `/backtests/${job}?${new URLSearchParams({variant: variant!})}`;
}
export function savedStrategyHref(id: string, job: string, variant: string) {
  const params = new URLSearchParams({ from_job: job, from_variant: variant });
  return `/strategies/${encodeURIComponent(id)}${reportReturn(params) ? `?${params}` : ''}`;
}
export function filterStrategies(items: Strategy[], query: string, market: string, timeframe: string) {
  const search = query.trim().toLocaleLowerCase();
  return items.filter(item => (!market || item.spec.market_type === market) && (!timeframe || item.spec.timeframe === timeframe) &&
    [item.name, item.spec.instrument_key, item.spec.instrument_id.symbol, item.spec.market_type, item.spec.timeframe].join(' ').toLocaleLowerCase().includes(search));
}
