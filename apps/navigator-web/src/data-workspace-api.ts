import {z} from 'zod';
import {ApiError} from './api';
import {instrumentSchema,marketRead,readInstrument,coverageSchema,rangeIso} from './market-data-api';
const streamStateSchema=z.enum(['disabled','unknown','offline','disconnected','blocked','retry_wait','recovering','running']);
const strategySchema=z.object({strategy_id:z.string(),name:z.string(),state:z.string()});
export const catalogRowSchema=instrumentSchema.extend({
 stream_state:streamStateSchema.optional(),exchange_name:z.string(),market_type:z.string(),refreshed_at:z.string(),strategies:z.array(strategySchema),
 coverage_state:z.enum(['unknown','empty','partial','complete']),coverage_percent:z.number().nullable(),actual_candles:z.number(),expected_candles:z.number(),
 history_start_at:z.string().nullable().optional(),history_end_at:z.string().optional(),
 job:z.object({job_id:z.string(),state:z.string(),progress_percent:z.number()}).nullable(),
});
export const exchangeCatalogSchema=z.object({exchange:z.string(),start_at:z.string(),end_at:z.string(),observed_at:z.string(),total:z.number(),offset:z.number(),limit:z.number(),snapshot:z.string(),missing_market_ids:z.array(z.number()),items:z.array(catalogRowSchema)});
export const collectionSchema=z.object({market_id:z.number(),symbol:z.string(),selected:z.boolean(),effective:z.boolean(),strategy_pinned:z.boolean(),strategies:z.array(strategySchema),last_candle_at:z.string().nullable(),history_repair_error:z.string().nullable().optional(),stream_state:streamStateSchema.optional(),next_retry_at:z.string().nullable().optional(),observed_at:z.string()});
export type CatalogRow=z.infer<typeof catalogRowSchema>;
export const readExchangeCatalog=(params:URLSearchParams,signal?:AbortSignal)=>marketRead(`workspace/instruments?${params}`,exchangeCatalogSchema,signal);
export const fullHistorySchema=z.object({market_id:z.number(),symbol:z.string(),timeframe:z.string(),start_at:z.string().nullable(),end_at:z.string(),state:z.enum(['unknown','empty','partial','complete']),expected_candles:z.number(),actual_candles:z.number(),missing_candles:z.number().nullable(),source_absent_minutes:z.number().default(0),coverage_percent:z.number().nullable(),observed_at:z.string(),origin_state:z.string()});
export async function readDataInspector(market:number,symbol:string,from:string,to:string,signal?:AbortSignal){
 const p=new URLSearchParams({market_id:String(market),symbol,start_at:rangeIso(from),end_at:rangeIso(to),timeframe:'1m'});
 const [instrument,collection,coverage]=await Promise.all([readInstrument(market,symbol,signal),marketRead(`workspace/collection?${p}`,collectionSchema,signal),marketRead(`workspace/history?${p}`,fullHistorySchema,signal)]);
 return {instrument:{...instrument,...collection},coverage};
}
export type DataInspector=Awaited<ReturnType<typeof readDataInspector>>;

const historyBoundsSchema=z.object({market_id:z.number(),symbol:z.string(),state:z.enum(['queued','running','ready','unavailable']),first_open_at:z.string().nullable(),end_at:z.string(),observed_at:z.string()});
export const readHistoryBounds=(market:number,symbol:string,signal?:AbortSignal)=>marketRead(`workspace/history-bounds?${new URLSearchParams({market_id:String(market),symbol})}`,historyBoundsSchema,signal);

/** Retry only safe transient reads. Keep failures visible while waiting, and
 * reset the bounded backoff after a successful response or query-key change. */
export const transientDataRead=(error:unknown)=>error instanceof ApiError&&
 (error.kind==='transport'||error.status===408||error.status===429||(error.status??0)>=500);
export function createDataReadRecovery(){
 let lastErrorAt=-1,failures=0;
 return (state:{error:unknown;errorUpdatedAt:number},normal:number|false):number|false=>{
  if(!state.error){lastErrorAt=-1;failures=0;return normal;}
  if(!transientDataRead(state.error))return false;
  if(state.errorUpdatedAt!==lastErrorAt){lastErrorAt=state.errorUpdatedAt;failures++;}
  const retryAfter=(state.error as ApiError).retryAfterSeconds??0;
  return Math.max(retryAfter*1000,Math.min(30000,2000*2**Math.min(failures-1,4)));
 };
}
