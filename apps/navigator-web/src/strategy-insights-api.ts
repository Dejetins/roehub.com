import { z } from 'zod';
import { ApiError, requestJson } from './api';
import { variantKeySchema, variantSchema, variantPath } from './results-api';
const number = z.number().finite();
const decimal = z.union([number,z.string().regex(/^-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?$/).transform(Number)]).refine(Number.isFinite);
const panel = z.enum(['ready','empty','degraded','unavailable']);
const date = z.string().refine(v=>Number.isFinite(Date.parse(v)));
export const tradingObservation = {
  paper_accounting:z.object({source:z.string().optional(),state:panel,position_quantity:decimal.nullable(),average_entry_price:decimal.nullable(),equity:decimal.nullable(),realized_pnl:decimal.nullable(),unrealized_pnl:decimal.nullable(),fee_total:decimal.nullable(),funding_total:decimal.nullable(),pnl_complete:z.boolean(),updated_at:date.nullable()}).optional(),
  signal_journal:z.object({state:panel,items:z.array(z.object({outcome:z.enum(['warmup','no_signal','signal','blocked']),signal_action:z.enum(['none','open','close','reduce','reverse']),side:z.enum(['buy','sell']).nullable(),reference_price:decimal,bar_ts_close:date,mode:z.enum(['monitor_only','paper','live','testnet'])})).max(200)}).optional(),
};
export const profileFields = {
  mode:z.enum(['monitor_only','paper','live','testnet']).optional(),updated_at:date.nullable().optional(),
  exchange_connection_id:z.string().nullable().optional(),sizing_method:z.enum(['fixed_quote','fixed_equity_pct']).optional(),
  sizing_value:decimal.optional(),max_position_notional:decimal.nullable().optional(),max_orders_per_run:number.optional(),max_notional_per_run:decimal.optional(),
};
export const sourceSchema = z.object({strategy_id:z.uuid(),source_job_id:z.uuid().nullable(),source_variant_key:variantKeySchema.nullable()}).refine(v=>(v.source_job_id===null)===(v.source_variant_key===null));
export async function readResearchSource(id:string,signal:AbortSignal){
 const {data}=await requestJson(`/api/strategies/${id}/research-source`,sourceSchema,{signal});
 if(data.strategy_id!==id)throw new ApiError('invalid-response',200,'failed');
 return data;
}
export const researchVariantSchema = variantSchema.extend({canonical_variant_params:z.object({execution:z.object({direction_mode:z.string().optional(),initial_cash_quote:number.optional(),fee_rate:number.optional(),slippage_rate:number.optional(),close_on_end:z.boolean().optional(),sizing:z.object({mode:z.string(),quote_amount:number.optional(),equity_pct:number.optional()}).optional()}).optional(),risk:z.object({mode:z.string()}).optional()}).optional()});
export async function readResearchVariant(job:string,variant:string,signal:AbortSignal){
 const reply=await requestJson(variantPath(job,variant),researchVariantSchema,{signal});if(reply.data.variant_key!==variant)throw new ApiError('invalid-response',200,'failed');return reply;
}
