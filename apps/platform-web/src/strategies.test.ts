import { afterEach, expect, it, vi } from 'vitest';
import { clientRoute } from './client-routes';
import { filterStrategies, readStrategies, readStrategy, reportReturn, savedStrategyHref, strategySchema, type Strategy } from './strategies-api';
const subject='00000000-0000-4000-8000-000000000001', id='00000000-0000-4000-8000-000000000002';
const strategy: Strategy={strategy_id:id,user_id:subject,name:'MA snapshot',created_at:'2026-09-12T00:00:00Z',is_deleted:false,
  spec:{instrument_id:{market_id:1,symbol:'BTCUSDT'},instrument_key:'binance:spot:BTCUSDT',market_type:'spot',timeframe:'1m',indicators:[{name:'MA',params:{fast:20,slow:50}},{name:'new-supported',input:'close'}],signal_template:'MA(20,50)',schema_version:1,spec_kind:'legacy'}};
afterEach(()=>vi.unstubAllGlobals());
it('defaults old bootstrap to Backtests only and never enables classic/new/mode',()=>{
  const old={locale:'en' as const,subject};
  expect(clientRoute('/backtests/new',old)).toBe(true); expect(clientRoute('/strategies',old)).toBe(false);
  for(const client_routes of [[],['/backtests'],['/strategies'],['/backtests','/strategies']] as const){
    const bootstrap={...old,client_routes:[...client_routes]};
    expect(clientRoute('/strategies/'+id,bootstrap)).toBe(client_routes.some(x=>x==='/strategies'));
    expect(clientRoute('/backtests/'+id,bootstrap)).toBe(client_routes.some(x=>x==='/backtests'));
    for(const path of ['/strategies/new','/strategies?view=classic','/strategies/'+id+'?mode=rl_ml','https://example.org/strategies']) expect(clientRoute(path,bootstrap)).toBe(false);
  }
});
it('return context roundtrips the full variant through an encoded local route',()=>{
  const variant='ema/close?window=20&direction=long';
  const href=savedStrategyHref(id,subject,variant);
  const target=reportReturn(new URL(href,location.origin).searchParams)!;
  expect(new URL(target,location.origin).pathname).toBe('/backtests/'+subject);
  expect(new URL(target,location.origin).searchParams.get('variant')).toBe(variant);
});
it.each(['','from_job=oops&from_variant=a',`from_job=${id}&from_variant=%00`,`from_job=${id}`,`return=https://evil.example`])('ignores malformed or absent return context %s',query=>expect(reportReturn(new URLSearchParams(query))).toBeNull());
it('keeps ordered immutable indicator values and strips unrelated response payload',()=>{
  const parsed=strategySchema.parse({...strategy,provider_payload:'private'});
  expect(parsed.spec.indicators).toEqual(strategy.spec.indicators); expect(parsed).not.toHaveProperty('provider_payload');
});
it('filters the complete direct list on supported real fields',()=>{
  for(const query of ['btc','MA snapshot','spot','1m']) expect(filterStrategies([strategy],query,'','')).toHaveLength(1);
  expect(filterStrategies([strategy],'','futures','')).toEqual([]); expect(filterStrategies([strategy],'','spot','15m')).toEqual([]);
});
it('rejects another subject in owned list and another identity in detail',async()=>{
  vi.stubGlobal('fetch',vi.fn().mockResolvedValueOnce(new Response(JSON.stringify([{...strategy,user_id:id}]))).mockResolvedValueOnce(new Response(JSON.stringify({...strategy,strategy_id:subject}))));
  await expect(readStrategies(subject,new AbortController().signal)).rejects.toMatchObject({kind:'invalid-response'});
  await expect(readStrategy(subject,id,new AbortController().signal)).rejects.toMatchObject({kind:'invalid-response'});
});

it('matches server last-value semantics for repeated presentation queries',()=>{
  const bootstrap={locale:'en' as const,subject,client_routes:['/strategies' as const]};
  expect(clientRoute('/strategies?view=classic&view=client',bootstrap)).toBe(true);
  expect(clientRoute('/strategies?view=client&view=classic',bootstrap)).toBe(false);
  expect(clientRoute('/strategies?mode=rl_ml&mode=dashboard',bootstrap)).toBe(true);
  expect(clientRoute('/strategies?mode=dashboard&mode=rl_ml',bootstrap)).toBe(false);
});
