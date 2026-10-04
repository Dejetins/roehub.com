import {z} from 'zod';
import {requestJson,ApiError} from './api';
const n=z.union([z.number(),z.string().regex(/^-?\d+(?:\.\d+)?$/).transform(Number)]).refine(Number.isFinite);
const time=z.string().datetime({offset:true});
export const fillSchema=z.object({fill_id:z.string(),time,price:n,reference_price:n.nullable().optional(),quantity:n,fee:n.nullable(),action:z.enum(['entry','exit']),reason:z.string(),side:z.enum(['buy','sell'])});
export const tradeSchema=z.object({trade_id:z.string(),side:z.enum(['long','short']),entry_time:time,exit_time:time.nullable(),entry:n,exit:n.nullable(),quantity:n,remaining_quantity:n,entry_reason:z.string(),exit_reason:z.string().nullable(),gross_pnl:n.nullable(),fees:n.nullable(),net_pnl:n.nullable(),fills:z.array(fillSchema)});
export const operationsSchema=z.object({source:z.string(),state:z.enum(['ready','empty','degraded','unavailable']),scope:z.enum(['current_run','history']),partial:z.boolean(),reason:z.string().nullable(),initial_cash:n.nullable(),trades:z.array(tradeSchema),equity:z.array(z.object({timestamp:time,value:n})),drawdown:z.array(z.object({timestamp:time,value:n})),current_price:n.nullable(),price_at:time.nullable(),stop_loss:n.nullable(),take_profit:n.nullable()});
export type Operations=z.infer<typeof operationsSchema>;export type OperationTrade=z.infer<typeof tradeSchema>;
export const operationalDashboard={
 strategy_selector:z.object({state:z.string(),items:z.array(z.object({strategy_id:z.uuid(),status:z.string(),run_state:z.string().nullable()}))}).optional(),
 execution_outcomes:z.object({items:z.array(z.object({intent_id:z.uuid().nullable(),order_status:z.string().nullable(),intent_status:z.string().nullable(),reconciliation_status:z.string().nullable()}))}).optional(),
 operations:operationsSchema.nullable().optional(),
 chart:z.object({state:z.string(),candles:z.array(z.object({timestamp:time,open:n.nullable(),high:n.nullable(),low:n.nullable(),close:n.nullable()}))}).optional(),
 selected_strategy:z.object({strategy_id:z.uuid().nullable(),actions:z.object({can_run:z.boolean(),can_stop:z.boolean(),can_delete:z.boolean(),can_create:z.boolean(),can_clone:z.boolean()}).optional()}),
};
export type Command='run'|'stop'|'restart'|'delete'|'manual-entry'|'manual-exit';
const run=z.object({run_id:z.uuid(),state:z.string()});
const manual=z.object({status:z.enum(['pending','accepted','rejected','unknown']),intent_id:z.uuid(),outcome_reason:z.string(),duplicate:z.boolean(),paper_order_state:z.string().nullable().optional()});
export async function sendStrategyCommand(id:string,command:Command,key:string,notional?:number,runId?:string|null,price?:number){
 const path=`/api/strategies/${encodeURIComponent(id)}`;
 if(command==='delete'){await requestJson(path,z.undefined(),{method:'DELETE'});return {status:'accepted',reason:''};}
 if(command==='manual-entry'||command==='manual-exit'){
  const response=await requestJson(`${path}/${command}`,manual,{method:'POST',idempotencyKey:key,body:{client_request_id:key,...(runId?{expected_run_id:runId}:{}),...(price?{reference_price:price}:{}),...(command==='manual-entry'&&notional?{quote_notional:notional}:{})}});
  return {status:response.data.status,reason:response.data.outcome_reason,intent:response.data.intent_id,terminal:response.data.status==='rejected'||response.data.paper_order_state==='filled'};
 }
 const response=await requestJson(`${path}/${command}`,run,{method:'POST'});return {status:'accepted',reason:response.data.state};
}
export function commandError(error:unknown){return error instanceof ApiError&&error.outcome==='unknown'?'unknown':'failed';}
