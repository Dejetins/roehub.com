import { z } from 'zod';
import { variantKeySchema } from './results-api';
export const STRATEGY_RECOVERY_KEY='roehub.backtests.strategy-recovery.v1';
const schema=z.object({operation:z.literal('save-strategy'),jobId:z.uuid(),variant:variantKeySchema,key:z.uuid(),createdAt:z.number().positive(),subject:z.string(),organization:z.uuid().nullable(),resultId:z.uuid().nullable()});
export type StrategyRecovery=z.infer<typeof schema>;
let memory:StrategyRecovery|null=null;
export function clearStrategyRecovery(){memory=null;try{sessionStorage.removeItem(STRATEGY_RECOVERY_KEY);}catch{/* In-memory fallback. */}}
export function loadStrategyRecovery(subject:string){try{const raw=sessionStorage.getItem(STRATEGY_RECOVERY_KEY);if(raw){if(raw.length>10000)throw new Error();memory=schema.parse(JSON.parse(raw));}}catch{ /* Keep only an already validated in-memory attempt. */ }
  if(memory?.subject!==subject)clearStrategyRecovery();return memory;
}
export function storeStrategyRecovery(record:StrategyRecovery){memory=schema.parse(record);try{sessionStorage.setItem(STRATEGY_RECOVERY_KEY,JSON.stringify(memory));return true;}catch{return false;}}
/** Durable server provenance has no job-create TTL. Current API cannot bind a replay to original org. */
export function strategyReplayAllowed(){return false;}
