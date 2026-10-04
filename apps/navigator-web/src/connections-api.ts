import {z} from 'zod';
import {requestJson} from './api';
export const connectionSchema=z.object({connection_id:z.string(),exchange_name:z.string(),market_type:z.string(),environment:z.string(),label:z.string().nullable(),
  requested_permissions:z.string(),effective_permissions:z.string(),effective_capability:z.string(),connection_readiness:z.string(),connection_readiness_reason:z.string(),
  status:z.enum(['active','disabled','archived']),validation_status:z.string(),validation_reason:z.string().nullable(),last_validated_at:z.string().nullable(),
  created_at:z.string(),updated_at:z.string(),used_by_strategies_count:z.number(),active_strategy_bindings_count:z.number()});
// Secrets, including masked key fragments, never enter the client query cache.
export type Connection=z.infer<typeof connectionSchema>;
export const bindingsSchema=z.object({items:z.array(z.object({binding_id:z.string(),strategy_id:z.string(),binding_status:z.string(),usage_mode:z.string(),updated_at:z.string()})),next_cursor:z.string().nullable()});
export const readConnectionBindings=(id:string,cursor:string,signal?:AbortSignal)=>requestJson(`/api/ui/account/exchange-connections/${encodeURIComponent(id)}/bindings?${new URLSearchParams({limit:'30',...(cursor?{cursor}:{})})}`,bindingsSchema,{signal}).then(r=>r.data);
export const connectionPageSchema=z.object({items:z.array(connectionSchema),next_cursor:z.string().nullable()});
export const marketsSchema=z.object({items:z.array(z.object({market_id:z.number(),exchange_name:z.string(),market_type:z.string(),market_code:z.string()}))});
export type Market=z.infer<typeof marketsSchema>['items'][number];
export async function readMarkets(signal?:AbortSignal){return (await requestJson('/api/market-data/markets',marketsSchema,{signal})).data;}
export async function readConnections(params:URLSearchParams,signal?:AbortSignal){return (await requestJson(`/api/ui/account/exchange-connections?${params}`,connectionPageSchema,{signal})).data;}
export async function readConnection(id:string,signal?:AbortSignal){return (await requestJson(`/api/ui/account/exchange-connections/${encodeURIComponent(id)}`,connectionSchema,{signal})).data;}
export async function connectionCommand(path:string,body?:unknown){return (await requestJson(`/api/ui/account/exchange-connections${path}`,connectionSchema,{method:'POST',body,timeoutMs:60_000})).data;}
