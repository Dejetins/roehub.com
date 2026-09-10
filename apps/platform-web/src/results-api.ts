import { z } from 'zod';
import { ApiError, requestJson, type ApiReply } from './api';
import { jobSchema } from './library-api';
const n = z.number().finite();
const optionalNumber = n.nullable().optional();
export const variantKeySchema = z.string().min(1).max(2048).refine(v => !/[\x00-\x1f\x7f]/.test(v));
export const metricsSchema = z.object({ total_return_pct: optionalNumber, total_return_pct_net_of_funding: optionalNumber, sharpe: optionalNumber, sharpe_trades:optionalNumber, avg_trade_exec_bars:optionalNumber, avg_trade_ret_pct:optionalNumber, exposure_pct:optionalNumber, return_over_max_drawdown:optionalNumber, max_drawdown_pct: optionalNumber, profit_factor: optionalNumber, win_rate_pct: optionalNumber, trade_count: optionalNumber, trades_count: optionalNumber });
export const variantSchema = z.object({ canonical_variant_params: z.object({execution:z.object({initial_cash_quote:n.optional()}).optional()}).optional(), rank: n, variant_key: variantKeySchema, variant_hash: z.string(), summary_metrics: metricsSchema, best_tp_pct: optionalNumber, best_sl_pct: optionalNumber, readable_params:z.object({indicators:z.array(z.object({indicator_id:z.string(),source:z.string().optional(),window:n.optional()}))}).optional() });
export const topSchema = z.object({ items: z.array(variantSchema).max(1000) });
export const summarySchema = z.object({ job: jobSchema, top_variants: topSchema, selected_variant_key: variantKeySchema.nullable(), retry_after_seconds: n });
const identity = { job_id: z.uuid(), variant_key: variantKeySchema };
export const pendingSchema = z.object({ ...identity, status: z.string(), materialization: z.object({ status: z.string(), retry_after_seconds: n.nonnegative(), retryable: z.boolean() }) });
const cache = z.object({ status: z.string().optional(), warning: z.string().optional() }).optional().transform(v => ({ degraded: !!v?.warning || v?.status === 'degraded' }));
export const seriesSchema = z.object({ ...identity, kind: z.enum(['equity','drawdown']), points: z.array(z.object({ x: z.union([z.string(), n]), value: n, trade_index:n.optional(), net_pnl_quote:n.optional(), equity:n.optional() })).max(1500), returned_points: n, source_points: n, downsampled: z.boolean(), cache });
const statsItem = z.object({ month: z.string().optional(), symbol: z.string().optional(), trades_count: n, net_pnl_quote: n, return_pct: n, win_rate_pct: optionalNumber, wins: optionalNumber, losses: optionalNumber });
export const statsSchema = z.object({ ...identity, kind: z.enum(['monthly','symbol']), items: z.array(statsItem).max(600), bounds: z.object({ truncated: z.boolean(), source_items: n, returned_items: n }), cache });
export const tradeSchema = z.object({ trade_index: n, entry_timestamp: z.string(), exit_timestamp: z.string(), side: z.string(), entry_price: optionalNumber, exit_price: optionalNumber, quantity: optionalNumber, net_pnl_quote: optionalNumber, return_pct: optionalNumber, fee_quote: optionalNumber, exit_reason: z.string().optional(), equity_after: optionalNumber });
export const tradesSchema = z.object({ ...identity, items: z.array(tradeSchema).max(100), pagination: z.object({ page: n.int().min(1).max(10000), page_size: n.int().min(1).max(100), total: n, has_next: z.boolean(), has_previous: z.boolean() }), cache });
export const readinessSchema = z.object({ source_job_id: z.uuid(), source_variant_key: variantKeySchema, strategy_spec_hash: z.string(), compatibility_state: z.enum(['launchable','not_launchable','degraded']), compatibility_reason_codes: z.array(z.string()).optional().transform(v => (v ?? []).filter(code => ['unsupported_strategy_schema_version','unsupported_strategy_spec_kind','unsupported_live_evaluator','timeframe_rollup_required','large_warmup_live_start_slow','supported_live_evaluator'].includes(code))), market_data_state: z.string(), checked_at: z.string().refine(v => Number.isFinite(Date.parse(v))) });
export const savedSchema = z.object({ status: z.enum(['created','duplicate']), duplicate: z.boolean(), duplicate_reason: z.string().nullable(), strategy: z.object({ strategy_id: z.uuid(), user_id: z.string(), is_deleted: z.boolean() }), provenance: z.object({ source_job_id: z.uuid(), source_variant_key: variantKeySchema, strategy_spec_hash: z.string() }) });
export const jobPath = (job: string) => `/api/backtests/jobs/${z.uuid().parse(job)}`;
export const variantPath = (job: string, variant: string) => `${jobPath(job)}/variants/${encodeURIComponent(variantKeySchema.parse(variant))}`;
export function assertSource(data: { job_id?: string; variant_key?: string; source_job_id?: string; source_variant_key?: string }, job: string, variant: string) {
  if ((data.job_id ?? data.source_job_id) !== job || (data.variant_key ?? data.source_variant_key) !== variant) throw new ApiError('invalid-response', 200, 'failed');
}
export async function readResult<T extends {job_id: string; variant_key: string}>(job: string, variant: string, suffix: string, schema: z.ZodType<T>, signal: AbortSignal) {
  const reply = await requestJson(`${variantPath(job, variant)}/${suffix}`, z.union([pendingSchema, schema]), { signal });
  assertSource(reply.data, job, variant);
  if ((reply.status === 202) !== ('materialization' in reply.data)) throw new ApiError('invalid-response', reply.status, 'failed');
  return reply;
}
export async function saveStrategy(job: string, variant: string, key: string, subject: string, signal: AbortSignal) {
  const reply = await requestJson(`${variantPath(job, variant)}/strategies`, savedSchema, { method:'POST', idempotencyKey:key, signal });
  try { assertSource(reply.data.provenance, job, variant); if (reply.data.strategy.user_id !== subject || ![200,201].includes(reply.status)) throw new Error(); }
  catch { throw new ApiError('invalid-response', reply.status, 'unknown'); }
  return reply.data;
}
export async function deleteJob(job: string, signal: AbortSignal) {
  const reply = await requestJson(jobPath(job), z.undefined(), { method:'DELETE', signal });
  if (reply.status !== 204) throw new ApiError('invalid-response', reply.status, 'unknown');
}
export type CsvReply = { status:200; blob:Blob; rows:number; total:number; max:number; truncated:boolean } | {status:202; pending:z.infer<typeof pendingSchema>; delay:number};
/** CSV is not JSON. Never trust a content-disposition filename or download a pending body. */
export async function readCsv(job:string, variant:string, maxRows:number, signal:AbortSignal):Promise<CsvReply> {
  if (!Number.isInteger(maxRows) || maxRows<1 || maxRows>100000) throw new TypeError('CSV bound');
  const controller=new AbortController(); const abort=()=>controller.abort(); signal.addEventListener('abort',abort,{once:true}); if(signal.aborted) abort();
  const timer=setTimeout(abort,15000);
  try {
    const response=await fetch(`${variantPath(job,variant)}/trades.csv?max_rows=${maxRows}`,{credentials:'same-origin',cache:'no-store',redirect:'error',signal:controller.signal,headers:{Accept:'text/csv, application/json'}});
    if(response.status===401){window.dispatchEvent(new Event('roehub:unauthenticated'));controller.abort();throw new ApiError('unauthenticated',401,'failed');}
    const delay=Number(response.headers.get('Retry-After')) || 0;
    if(!response.ok) {
      const payload=await response.json().catch(()=>null);
      const hint=z.object({error:z.object({details:z.object({retry_after_seconds:z.number().finite().nonnegative().optional()}).optional()}).optional()}).safeParse(payload);
      const retry=Math.max(delay,hint.success?hint.data.error?.details?.retry_after_seconds??0:0);
      throw new ApiError(response.status===429?'rate-limited':response.status===403?'forbidden':response.status===404?'not-found':'unavailable',response.status,'failed',retry);
    }
    const type=response.headers.get('Content-Type')?.split(';')[0].trim();
    if(response.status===202 && type==='application/json') { const pending=pendingSchema.parse(await response.json());assertSource(pending,job,variant);return {status:202,pending,delay:Math.max(delay,pending.materialization.retry_after_seconds,2)}; }
    if(response.status!==200 || type!=='text/csv') throw new ApiError('invalid-response',response.status,'failed');
    const count=(name:string)=>{const raw=response.headers.get(`x-roehub-trades-${name}`);if(!raw || !/^\d+$/.test(raw))throw new ApiError('invalid-response',200,'failed');return Number(raw);};
    const rows=count('row-count'),total=count('total-rows'),max=count('max-rows'), truncated=response.headers.get('x-roehub-trades-truncated');
    if(rows>maxRows || rows>max || rows>total || !['true','false'].includes(truncated??''))throw new ApiError('invalid-response',200,'failed');
    const blob=await response.blob();if(controller.signal.aborted)throw new Error();
    return {status:200,blob,rows,total,max,truncated:truncated==='true'};
  }catch(error){if(signal.aborted)throw error;if(error instanceof ApiError)throw error;throw new ApiError('transport',null,'failed');}
  finally{clearTimeout(timer);signal.removeEventListener('abort',abort);}
}
export function resultDelay(reply: ApiReply<unknown>|undefined, error: Error|null) {
  if(error) return error instanceof ApiError && error.kind==='rate-limited' ? Math.max(2,error.retryAfterSeconds??5)*1000 : Infinity;
  if(reply?.status===202){const data=pendingSchema.parse(reply.data);if(['queued','running','pending'].includes(data.status))return Math.max(2,reply.retryAfterSeconds??0,data.materialization.retry_after_seconds)*1000;}
  return Infinity;
}
