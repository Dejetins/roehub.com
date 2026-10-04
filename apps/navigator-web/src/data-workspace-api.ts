import {z} from 'zod';
import {instrumentSchema,marketRead,readInstrument,coverageSchema,rangeIso} from './market-data-api';
const strategySchema=z.object({strategy_id:z.string(),name:z.string(),state:z.string()});
export const catalogRowSchema=instrumentSchema.extend({
 exchange_name:z.string(),market_type:z.string(),refreshed_at:z.string(),strategies:z.array(strategySchema),
 coverage_state:z.enum(['empty','partial','complete']),coverage_percent:z.number(),actual_candles:z.number(),expected_candles:z.number(),
 job:z.object({job_id:z.string(),state:z.string(),progress_percent:z.number()}).nullable(),
});
export const exchangeCatalogSchema=z.object({exchange:z.string(),start_at:z.string(),end_at:z.string(),observed_at:z.string(),total:z.number(),offset:z.number(),limit:z.number(),snapshot:z.string(),missing_market_ids:z.array(z.number()),items:z.array(catalogRowSchema)});
export const collectionSchema=z.object({market_id:z.number(),symbol:z.string(),selected:z.boolean(),effective:z.boolean(),strategy_pinned:z.boolean(),strategies:z.array(strategySchema),last_candle_at:z.string().nullable(),observed_at:z.string()});
export type CatalogRow=z.infer<typeof catalogRowSchema>;
export const readExchangeCatalog=(params:URLSearchParams,signal?:AbortSignal)=>marketRead(`workspace/instruments?${params}`,exchangeCatalogSchema,signal);
export async function readDataInspector(market:number,symbol:string,from:string,to:string,signal?:AbortSignal){
 const p=new URLSearchParams({market_id:String(market),symbol,start_at:rangeIso(from),end_at:rangeIso(to),timeframe:'1m'});
 const [instrument,collection,coverage]=await Promise.all([readInstrument(market,symbol,signal),marketRead(`workspace/collection?${p}`,collectionSchema,signal),marketRead(`workspace/coverage?${p}`,coverageSchema,signal)]);
 return {instrument:{...instrument,...collection},coverage};
}
export type DataInspector=Awaited<ReturnType<typeof readDataInspector>>;

const historyBoundsSchema=z.object({market_id:z.number(),symbol:z.string(),state:z.enum(['queued','running','ready','unavailable']),first_open_at:z.string().nullable(),end_at:z.string(),observed_at:z.string()});
export const readHistoryBounds=(market:number,symbol:string,signal?:AbortSignal)=>marketRead(`workspace/history-bounds?${new URLSearchParams({market_id:String(market),symbol})}`,historyBoundsSchema,signal);
