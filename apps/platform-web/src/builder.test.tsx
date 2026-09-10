import '@testing-library/jest-dom/vitest';
import { afterEach, expect, it, vi } from 'vitest';
import { act, cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react';
import { QueryClientProvider } from '@tanstack/react-query';
import { createMemoryRouter, RouterProvider } from 'react-router';
import { I18nextProvider } from 'react-i18next';
import fixture from './test-data/builder-catalog.json';
import { defaultsSchema,builderCatalogSchema,preflightSchema,researchSchema, type Research } from './builder-api';
import { computationIdentity,dateBoundary,fieldPath,normalizedLabel,initialResearch,prepareResearch,validateResearch } from './builder-model';
import { clearRecovery,freezeAttempt,loadRecovery,saveRecovery,storageWorks,recoveryReason,RECOVERY_KEY } from './recovery';
import { App } from './app';
import { createQueryClient } from './query-client';
import { createI18n } from './i18n';
const defaults=defaultsSchema.parse(fixture.defaults),catalog=builderCatalogSchema.parse(fixture.catalog);
const bounds={state:'ready' as const,coordinates:catalog.config_draft.coordinates,default_end:'2026-03-29',max_end:'2026-03-29'};
function body():Research {const b=initialResearch(defaults,catalog);b.indicators[0].indicator_id='ma.ema';b.timeframe='15m';b.time_range={start:'2026-03-26T00:00:00Z',end:'2026-03-29T00:00:00Z'};return b;}
function result(b=body()) {return {normalized_request:{...b,quality_constraints:{min_closed_trades:1}},request_hash:'effective-hash',result_config_hash:'config-hash',artifact_metadata:{},cost_estimate:{indicator_rows:1,candidate_combinations:1,tp_sl_cells:0,cost_class:'small'},warnings:[],errors:[],funding_readiness:{status:'not_applicable',coverage_policy:'not_applicable',coverage_ratio:null,rows_count:0,expected_event_count:0,missing_event_count:0,warning_codes:[]},direction_market_compatibility:{compatible:true,market_type:'spot',direction_mode:'long_only'}};}
afterEach(()=>{cleanup();vi.restoreAllMocks();vi.unstubAllGlobals();clearRecovery();sessionStorage.clear();});
it('uses authoritative execution/quality defaults and preserves workstation selection without specimen substitution',()=>{
 const b=initialResearch(defaults,catalog);expect(b.timeframe).toBe('1h');expect(b.time_range.start).toBe('2023-01-01T00:00:00Z');expect(b.execution).toEqual({...defaults.execution_defaults,direction_mode:'long_only'});expect(b.quality_constraints).toEqual(defaults.quality_constraints_default);expect(validateResearch(body(),defaults,catalog,bounds)).toEqual([]);
});
it.each([
 ['empty interval',(b:Research):unknown=>b.time_range.end=b.time_range.start,'time_range.end'],
 ['invalid calendar date',(b:Research):unknown=>b.time_range.start='2026-02-30T00:00:00Z','time_range.start'],
 ['offset timestamp',(b:Research):unknown=>b.time_range.start='2026-03-26T00:00:00+00:00','time_range.start'],
 ['artifact end',(b:Research):unknown=>b.time_range.end='2026-03-30T00:00:00Z','time_range.end'],
 ['timeframe',(b:Research):unknown=>b.timeframe='5m','timeframe'],
 ['market',(b:Research):unknown=>b.coordinates.market_type='futures','coordinates.market_type'],
 ['symbol',(b:Research):unknown=>b.coordinates.symbol='UNKNOWN','coordinates.symbol'],
 ['empty indicators',(b:Research):unknown=>b.indicators=[],'indicators'],
 ['unknown ID',(b:Research):unknown=>b.indicators[0].indicator_id='injected','indicators.0.indicator_id'],
 ['source',(b:Research):unknown=>b.indicators[0].sources=['open'],'indicators.0.sources'],
 ['uncovered window',(b:Research):unknown=>b.indicators[0].window.stop=11,'indicators.0.window'],
 ['noninteger window',(b:Research):unknown=>b.indicators[0].window.start=1.5,'indicators.0.window.start'],
 ['zero grid step',(b:Research):unknown=>b.indicators[0].window.step=0,'indicators.0.window.step'],
 ['infinite grid',(b:Research):unknown=>b.indicators[0].window.stop=1e99,'indicators.0.window.stop'],
 ['short spot',(b:Research):unknown=>b.execution.direction_mode='short','execution.direction_mode'],
 ['cash',(b:Research):unknown=>b.execution.initial_cash_quote=0,'execution.initial_cash_quote'],
 ['NaN rate',(b:Research):unknown=>b.execution.fee_rate=NaN,'execution.fee_rate'],
 ['negative rate',(b:Research):unknown=>b.execution.slippage_rate=-1,'execution.slippage_rate'],
 ['sizing missing quote',(b:Research):unknown=>b.execution.sizing={mode:'fixed_quote'},'execution.sizing.quote_amount'],
 ['sizing pct',(b:Research):unknown=>b.execution.sizing={mode:'fixed_equity_pct',equity_pct:101},'execution.sizing.equity_pct'],
 ['sizing min',(b:Research):unknown=>b.execution.sizing={mode:'fixed_equity_pct_min_quote',equity_pct:20},'execution.sizing.min_quote'],
 ['sizing max',(b:Research):unknown=>b.execution.sizing={mode:'fixed_equity_pct_max_quote',equity_pct:20,max_quote:0},'execution.sizing.max_quote'],
 ['top limit',(b:Research):unknown=>b.top_n=51,'top_n'],
 ['top integral',(b:Research):unknown=>b.top_n=1.5,'top_n'],
 ['funding rank',(b:Research):unknown=>b.ranking.primary_metric='total_return_pct_net_of_funding','ranking.primary_metric'],
] as const)('blocks %s without silently changing values',(_name,edit,path)=>{const b=body();edit(b);const snapshot=JSON.stringify(b);expect(validateResearch(b,defaults,catalog,bounds).map(i=>i.path)).toContain(path);expect(JSON.stringify(b)).toBe(snapshot);});
it('validates source-less and materialized arithmetic grids without allocating arbitrary input ranges',()=>{
 const d=structuredClone(defaults);d.indicator_param_specs['ma.ema'].params.window={mode:'range',start:2,stop_incl:20,step:2};const b=body();b.indicators[0].window={start:2,stop:20,step:2};expect(validateResearch(b,d,catalog,bounds)).toEqual([]);b.indicators[0].window.step=3;expect(validateResearch(b,d,catalog,bounds).map(i=>i.code)).toContain('grid');
 b.indicators=[{indicator_id:'ma.vwma',sources:[],window:{start:10,stop:10,step:1}}];expect(validateResearch(b,defaults,catalog,bounds)).toEqual([]);
});
it('validates both explicit risk toggles, covered levels, percent units and resource guardrails',()=>{
 const b=body();b.risk={mode:'tp_sl_grid',tp:{enabled:true,start_pct:1,stop_pct:2,step_pct:1},sl:{enabled:false}};expect(validateResearch(b,defaults,catalog,bounds)).toEqual([]);
 b.risk.tp!.step_pct=0.3;expect(validateResearch(b,defaults,catalog,bounds).map(i=>i.code)).toContain('riskGrid');b.risk.tp!.enabled=false;expect(validateResearch(b,defaults,catalog,bounds).map(i=>i.code)).toContain('riskEmpty');
 const d=structuredClone(defaults);d.guardrails.max_indicator_arity=1;b.indicators.push(b.indicators[0]);expect(validateResearch(b,d,catalog,bounds).map(i=>i.code)).toContain('arity');
});
it('strips unknown nested data from the submitted/recovery allowlist; label alone is not computation identity',()=>{
 const b=body();expect(computationIdentity(b)).toBe(computationIdentity({...b,strategy_name:'other label'}));expect(computationIdentity(b)).not.toBe(computationIdentity({...b,top_n:2}));
 const clean=researchSchema.parse({...b,token:'secret',execution:{...b.execution,provider_payload:'secret'}});expect(JSON.stringify(clean)).not.toContain('secret');expect(fieldPath('body.indicators[0].window.start')).toBe('indicators.0.window.start');
 expect(preflightSchema.parse({...result(),artifact_metadata:{storage_path:'secret'}}).artifact_metadata).toEqual({});
});
it('retains exact allowed request/key on reload and clears on actor change',()=>{const a=freezeAttempt(body(),'actor');expect(saveRecovery(a)).toBe(true);expect(loadRecovery('actor')).toEqual(a);expect(loadRecovery('other')).toBeNull();expect(sessionStorage.getItem(RECOVERY_KEY)).toBeNull();});
it.each([
 [null,null,null,'identityUnknown'],['actor',null,86400,'scopeUnknown'],['actor','20000000-0000-4000-8000-000000000002',86400,'scopeChanged'],['actor','10000000-0000-4000-8000-000000000001',null,'retentionUnknown'],['actor','10000000-0000-4000-8000-000000000001',1,'retentionExpired'],['actor','10000000-0000-4000-8000-000000000001',86400,'serverScopeUnbound']
] as const)('never allows replay for actor=%s org=%s retention=%s',(subject,org,ttl,reason)=>{const r={...freezeAttempt(body(),'actor'),organization:'10000000-0000-4000-8000-000000000001',createdAt:1000};expect(recoveryReason(r,subject,org,ttl,3000)).toBe(reason);});
it('storage denial keeps bounded in-memory recovery and reports degradation',()=>{vi.spyOn(Storage.prototype,'setItem').mockImplementation(()=>{throw new Error('denied');});expect(storageWorks()).toBe(false);const a=freezeAttempt(body(),'actor');expect(saveRecovery(a)).toBe(false);expect(loadRecovery('actor')).toEqual(a);clearRecovery();expect(loadRecovery('actor')).toBeNull();});
it('rejects oversized recovery before writing and strips extra stored fields',()=>{const a=freezeAttempt(body(),'actor');expect(()=>saveRecovery({...a,body:{...a.body,strategy_name:'x'.repeat(256001)}})).toThrow('recoveryTooLarge');sessionStorage.setItem(RECOVERY_KEY,JSON.stringify({...a,token:'private'}));expect(loadRecovery('actor')).toEqual(a);});
function response(data:unknown,status=200){
 // The same workstation endpoint now supplies the simultaneously visible history and editor.
 const value=data && typeof data==='object' && 'config_draft' in data ? {refresh_status:'fresh',retry_after_seconds:0,next_allowed_refresh_at:null,job_table:{state:'unavailable'},...data} : data;
 return new Response(JSON.stringify(value),{status});
}
function renderBuilder(override?:(url:URL,init:RequestInit)=>Response|Promise<Response>|undefined,path='/backtests/new'){
 const fetch=vi.fn((url:string|URL,init:RequestInit={})=>{
  const u=new URL(url);const custom=override?.(u,init);if(custom)return Promise.resolve(custom);
  if(u.pathname.endsWith('current-user'))return Promise.resolve(response({user_id:'actor',paid_level:'free'}));
  if(u.pathname.endsWith('runtime-defaults'))return Promise.resolve(response(defaults));
  if(u.pathname.endsWith('workstation'))return Promise.resolve(response(catalog));
  if(u.pathname.endsWith('artifact-date-bounds'))return Promise.resolve(response({...bounds,coordinates:Object.fromEntries(u.searchParams)}));
  if(u.pathname.endsWith('preflight'))return Promise.resolve(response(result(JSON.parse(String(init.body)))));
  if(u.pathname.endsWith('jobs'))return Promise.resolve(response({items:[],next_cursor:null}));
  return Promise.resolve(response({},404));
 });vi.stubGlobal('fetch',fetch);const client=createQueryClient();const router=createMemoryRouter([{path:'*',element:<App bootstrap={{subject:'actor',locale:'en'}}/>}],{initialEntries:[path]});
 render(<I18nextProvider i18n={createI18n('en')}><QueryClientProvider client={client}><RouterProvider router={router}/></QueryClientProvider></I18nextProvider>);return{fetch,client,router};
}
async function configure(){await screen.findByLabelText('Job label (optional)');fireEvent.change(screen.getByLabelText('Timeframe',{exact:true}),{target:{value:'15m'}});fireEvent.change(screen.getByLabelText('Indicator',{exact:true}),{target:{value:'ma.ema'}});fireEvent.change(screen.getByLabelText('Start date',{exact:true}),{target:{value:'2026-03-26'}});}
async function check(){fireEvent.click(screen.getByRole('button',{name:'Check configuration'}));await waitFor(()=>expect(screen.getByRole('button',{name:'Submit backtest'})).toBeEnabled());}
it('date-only controls submit UTC day boundaries, invalidate preflight and reject a missing date',async()=>{
 const {fetch}=renderBuilder();await configure();
 expect(screen.getByLabelText('Start date')).toHaveAttribute('type','date');
 expect(screen.getByLabelText('End date')).toHaveAttribute('type','date');
 expect(document.querySelector('input[type=time],input[type=datetime-local]')).toBeNull();
 await check();
 const posted=JSON.parse(String(fetch.mock.calls.find(([u])=>String(u).endsWith('/preflight'))![1]?.body));
 expect(posted.time_range).toEqual({start:'2026-03-26T00:00:00Z',end:'2026-03-29T00:00:00Z'});
 fireEvent.change(screen.getByLabelText('End date'),{target:{value:'2026-03-28'}});
 expect(screen.getByRole('button',{name:'Submit backtest'})).toBeDisabled();
 fireEvent.change(screen.getByLabelText('End date'),{target:{value:''}});
 fireEvent.click(screen.getByRole('button',{name:'Check configuration'}));
 expect(screen.getByRole('alert')).toHaveTextContent('Choose a valid calendar date.');
 expect(fetch.mock.calls.filter(([u])=>String(u).endsWith('/preflight'))).toHaveLength(1);
});
it('converts calendar dates without local-time parsing, and never rolls invalid dates into another month',()=>{
 expect(dateBoundary('2026-03-08')).toBe('2026-03-08T00:00:00Z');
 expect(dateBoundary('')).toBe('');
 const b=body();b.time_range.start=dateBoundary('2026-02-30');
 expect(validateResearch(b,defaults,catalog,bounds)).toContainEqual({path:'time_range.start',code:'utc',message:'utc'});
});
it('invalidates an in-flight preflight after edits and ignores its late success',async()=>{
 let resolveOld!:(r:Response)=>void;let posted:Research|undefined;renderBuilder((u,i)=>u.pathname.endsWith('preflight')?new Promise(r=>{resolveOld=r;posted=JSON.parse(String(i.body));}):undefined);await configure();
 fireEvent.click(screen.getByRole('button',{name:'Check configuration'}));await waitFor(()=>expect(posted).toBeDefined());fireEvent.change(screen.getByLabelText('Top N',{exact:true}),{target:{value:'2'}});
 await act(async()=>resolveOld(response(result(posted))));expect(screen.getByRole('button',{name:'Submit backtest'})).toBeDisabled();
});
it('coordinates requery catalog, preserve selection, and cannot validate a new market using a late old symbol list',async()=>{
 let oldResolve!:(r:Response)=>void;const requests:string[]=[];
 const expanded={...catalog,instrument_universe:{...catalog.instrument_universe,market_types:[{value:'spot',label:'Spot',status:'available'},{value:'futures',label:'Futures',status:'available'}]}};
 renderBuilder((u)=>{if(!u.pathname.endsWith('workstation'))return;requests.push(u.search);if(u.searchParams.get('instrument_market_type')==='spot')return new Promise(r=>{oldResolve=r;});if(u.searchParams.get('instrument_market_type')==='futures')return response({...expanded,instrument_universe:{...expanded.instrument_universe,symbols:[{value:'ETHUSDT',label:'ETHUSDT',status:'available'}]}});return response(expanded);});
 await configure();fireEvent.change(screen.getByLabelText('Market type'),{target:{value:'futures'}});await waitFor(()=>expect(requests.some(s=>s.includes('instrument_market_type=futures'))).toBe(true));
 await waitFor(()=>expect(screen.getByRole('option',{name:'ETHUSDT'})).toBeInTheDocument());await act(async()=>oldResolve(response(expanded)));
 expect(screen.getByLabelText('Symbol')).toHaveValue('BTCUSDT');expect(screen.getByRole('option',{name:'ETHUSDT'})).toBeInTheDocument();fireEvent.click(screen.getByRole('button',{name:'Check configuration'}));await waitFor(()=>expect(screen.getByRole('alert')).toHaveTextContent('coordinates.symbol'));expect(screen.getByRole('button',{name:'Submit backtest'})).toBeDisabled();
});
it('body errors on HTTP 200 block submit and map parent errors to the actual child fields',async()=>{
 renderBuilder((u)=>u.pathname.endsWith('preflight')?response({...result(),errors:[{path:'indicators.0.window',code:'invalid_window',message:'Not materialized'}]}):undefined);await configure();fireEvent.click(screen.getByRole('button',{name:'Check configuration'}));await waitFor(()=>expect(screen.getByRole('alert')).toHaveTextContent('Not materialized'));expect(screen.getByLabelText('Window start',{exact:false})).toHaveAttribute('aria-invalid','true');expect(screen.getByRole('button',{name:'Submit backtest'})).toBeDisabled();
});
it('double submit uses a single exact frozen attempt; 503 retains its identity and stops edits',async()=>{
 let count=0;let accept!:(r:Response)=>void;const{fetch}=renderBuilder((u,i)=>u.pathname.endsWith('/jobs') && i?.method==='POST'?new Promise(r=>{count++;accept=r;}):undefined);await configure();fireEvent.change(screen.getByLabelText('Fee (%)',{exact:true}),{target:{value:'0.15'}});await check();const button=screen.getByRole('button',{name:'Submit backtest'});fireEvent.click(button);fireEvent.click(button);await waitFor(()=>expect(count).toBe(1));
 const record=loadRecovery('actor')!;expect(record.body.execution.fee_rate).toBe(0.0015);expect(record.organization).toBeNull();await act(async()=>accept(response({},503)));expect(loadRecovery('actor')?.key).toBe(record.key);expect(screen.getByLabelText('Job label (optional)')).toBeDisabled();expect(fetch.mock.calls.filter(([u,i])=>String(u).endsWith('/jobs') && i?.method==='POST')).toHaveLength(1);
});
it.each([403,404,409,422,429])('preserves identity and input on submit HTTP %s',async(status)=>{
 renderBuilder((u,i)=>u.pathname.endsWith('/jobs') && i?.method==='POST'?response({error:{code:'controlled_rejection',details:{errors:status===422?[{path:'body.top_n',code:'limit',message:'Admission changed'}]:[]}}},status):undefined);await configure();await check();fireEvent.click(screen.getByRole('button',{name:'Submit backtest'}));await waitFor(()=>expect(screen.getByRole('alert')).toHaveTextContent('controlled_rejection'));expect(loadRecovery('actor')).not.toBeNull();if(status===409)expect(screen.queryByRole('button',{name:'Correct rejected configuration'})).toBeNull();
});
it('401 from a command closes the private UI; retained recovery survives same-subject reentry and subject change clears it',async()=>{
 renderBuilder((u,i)=>u.pathname.endsWith('/jobs') && i?.method==='POST'?response({},401):undefined);await configure();await check();fireEvent.click(screen.getByRole('button',{name:'Submit backtest'}));await screen.findByRole('link',{name:'Sign in'});expect(screen.queryByLabelText('Job label (optional)')).toBeNull();expect(loadRecovery('actor')).not.toBeNull();cleanup();renderBuilder();await screen.findByRole('heading',{name:'Submission outcome is unresolved'});expect(screen.queryByRole('button',{name:'Submit backtest'})).toBeNull();
 cleanup();renderBuilder(u=>u.pathname.endsWith('current-user')?response({user_id:'other',paid_level:'free'}):undefined);await screen.findByText('Your account has changed. Reload this page to continue.');await waitFor(()=>expect(sessionStorage.getItem(RECOVERY_KEY)).toBeNull());
});
it('unavailable defaults prevents configuration and submit',async()=>{renderBuilder(u=>u.pathname.endsWith('runtime-defaults')?response({},503):undefined);await screen.findByText('Defaults or catalog are unavailable. Submission is blocked.');expect(screen.queryByRole('button',{name:'Submit backtest'})).toBeNull();});
it('edit then revert cannot revive a successful preflight; a label edit preserves it',async()=>{
 renderBuilder();await configure();await check();fireEvent.change(screen.getByLabelText('Job label (optional)'),{target:{value:'metadata'}});expect(screen.getByRole('button',{name:'Submit backtest'})).toBeEnabled();
 fireEvent.change(screen.getByLabelText('Top N'),{target:{value:'2'}});fireEvent.change(screen.getByLabelText('Top N'),{target:{value:'10'}});expect(screen.getByRole('button',{name:'Submit backtest'})).toBeDisabled();await check();expect(screen.getByRole('button',{name:'Submit backtest'})).toBeEnabled();
});
it('edit then revert while preflight is pending discards its late success',async()=>{
 let resolveOld!:(r:Response)=>void;let posted:Research|undefined;renderBuilder((u,i)=>u.pathname.endsWith('preflight')?new Promise(r=>{resolveOld=r;posted=JSON.parse(String(i.body));}):undefined);await configure();fireEvent.click(screen.getByRole('button',{name:'Check configuration'}));await waitFor(()=>expect(posted).toBeDefined());
 fireEvent.change(screen.getByLabelText('Top N'),{target:{value:'2'}});fireEvent.change(screen.getByLabelText('Top N'),{target:{value:'10'}});await act(async()=>resolveOld(response(result(posted))));expect(screen.getByRole('button',{name:'Submit backtest'})).toBeDisabled();
});
it('401 during preflight immediately stops private commands and protected reads',async()=>{
 const {fetch}=renderBuilder(u=>u.pathname.endsWith('preflight')?response({},401):undefined);await configure();fireEvent.click(screen.getByRole('button',{name:'Check configuration'}));await screen.findByRole('link',{name:'Sign in'});expect(screen.queryByLabelText('Job label (optional)')).toBeNull();expect(loadRecovery('actor')).toBeNull();const count=fetch.mock.calls.length;await act(async()=>{window.dispatchEvent(new Event('focus'));});expect(fetch.mock.calls.length).toBe(count);
});
it('funding warnings are retained and effective policies reflect normalized server values',async()=>{
 const r=result();r.normalized_request.execution={...r.normalized_request.execution,profit_lock:{enabled:true,safe_profit_percent:2},funding:{mode:'off',coverage_policy:'degraded_with_warning'}};
 renderBuilder(u=>u.pathname.endsWith('preflight')?response({...r,warnings:[{path:'funding_readiness',code:'funding_readiness_degraded',message:'Funding has missing events'}],funding_readiness:{...r.funding_readiness,status:'degraded',coverage_ratio:0.5,expected_event_count:2,missing_event_count:1}}):undefined);await configure();await check();expect(screen.getByText(/Funding has missing events/)).toBeInTheDocument();expect(screen.getByText(/Safe profit \(%\): 2/)).toBeInTheDocument();expect(screen.getByText('Minimum closed trades: 1')).toBeInTheDocument();
});
it('same subject with changed organization claims cannot enable replay or clear the unresolved record',async()=>{
 const record={...freezeAttempt(body(),'actor'),organization:'10000000-0000-4000-8000-000000000001'};saveRecovery(record);const {fetch}=renderBuilder(u=>u.pathname.endsWith('current-user')?response({user_id:'actor',paid_level:'free',organization_id:'20000000-0000-4000-8000-000000000002'}):undefined);
 await screen.findByRole('heading',{name:'Submission outcome is unresolved'});expect(loadRecovery('actor')).toEqual(record);expect(fetch.mock.calls.some(([,i])=>i?.method==='POST')).toBe(false);expect(screen.queryByRole('button',{name:'Submit backtest'})).toBeNull();
});
it('a known result identity is read authoritatively after reload and its resolved payload is removed',async()=>{
 const id='10000000-0000-4000-8000-000000000001', now=new Date().toISOString();saveRecovery({...freezeAttempt(body(),'actor'),organization:id,resultId:id});
 const {router,fetch}=renderBuilder(u=>u.pathname.endsWith(`/jobs/${id}`)?response({job_id:id,state:'queued',created_at:now,updated_at:now,generated_at:now,refresh_status:'poll',next_allowed_refresh_at:now,retry_after_seconds:0,cancel_requested_at:null,progress:{percent:0,processed_units:0,total_units:1,updated_at:now,pipeline_stage:'queued'},request:{coordinates:body().coordinates,timeframe:'15m',time_range:body().time_range,risk_mode:'none'}}):undefined);
 await waitFor(()=>expect(router.state.location.pathname).toBe(`/backtests/${id}`));expect(loadRecovery('actor')).toBeNull();expect(fetch.mock.calls.some(([,i])=>i?.method==='POST')).toBe(false);
});
it('an absent known result remains unresolved; 404 is not proof that create failed',async()=>{
 const id='10000000-0000-4000-8000-000000000001';saveRecovery({...freezeAttempt(body(),'actor'),resultId:id});renderBuilder(u=>u.pathname.endsWith(`/jobs/${id}`)?response({},404):undefined);
 await screen.findByRole('alert');expect(loadRecovery('actor')?.resultId).toBe(id);expect(screen.queryByRole('button',{name:'Submit backtest'})).toBeNull();
});
it('read 429 body delay blocks catalog retry without discarding the draft',async()=>{
 renderBuilder(u=>u.pathname.endsWith('artifact-date-bounds')?response({error:{code:'backtest.rate_limited',details:{retry_after_seconds:60}}},429):undefined);await configure();expect(screen.getByRole('button',{name:/Wait \d+s/})).toBeDisabled();expect(screen.getByLabelText('Start date')).toHaveValue('2026-03-26');expect(screen.getByRole('button',{name:'Submit backtest'})).toBeDisabled();
});
it('renewed admission shows the server resource limit and usage, retaining the submitted identity',async()=>{
 renderBuilder((u,i)=>u.pathname.endsWith('/jobs') && i.method==='POST'?response({error:{code:'backtest.rate_limited',details:{limit_scope:'full_jobs.active',limit:2,used:2,retry_after_seconds:60,provider_payload:'secret'}}},429):undefined);await configure();await check();fireEvent.click(screen.getByRole('button',{name:'Submit backtest'}));await screen.findByText('Server limit');expect(screen.getByText('Active jobs')).toBeInTheDocument();expect(screen.getByText('In use')).toBeInTheDocument();expect(screen.queryByText('secret')).toBeNull();expect(loadRecovery('actor')).not.toBeNull();expect(screen.getByRole('button',{name:'Correct rejected configuration'})).toBeDisabled();
});
it('inactive TP/SL fields and sizing fields are omitted from the exact submission without erasing draft inputs',()=>{
 const b=body();b.risk={mode:'none',tp:{enabled:true,start_pct:NaN,stop_pct:2,step_pct:1}};b.execution.sizing={mode:'all_in',quote_amount:NaN};expect(prepareResearch(b).risk).toEqual({mode:'none'});expect(prepareResearch(b).execution.sizing).toEqual({mode:'all_in'});expect(b.risk.tp?.start_pct).toBeNaN();
 b.risk={mode:'tp_sl_grid',tp:{enabled:true,start_pct:1,stop_pct:1,step_pct:1},sl:{enabled:false,start_pct:NaN}};expect(prepareResearch(b).risk.sl).toEqual({enabled:false});
});
it('validates actual materialized values rather than inventing a timeframe or upper-stop restriction',()=>{
 const b=body();b.timeframe='30m';b.risk={mode:'tp_sl_grid',tp:{enabled:true,start_pct:1,stop_pct:1,step_pct:1},sl:{enabled:false}};expect(validateResearch(b,defaults,catalog,bounds)).toEqual([]);
 const d=structuredClone(defaults);d.indicator_param_specs['ma.ema'].params.window={mode:'range',start:2,stop_incl:20,step:2};b.indicators[0].window={start:2,stop:21,step:20};expect(validateResearch(b,d,catalog,bounds)).toEqual([]);
});

it('job metadata normalization matches the server Unicode character limit',()=>{expect(normalizedLabel('  hello \n world  ')).toBe('hello world');expect(normalizedLabel('🚀'.repeat(97))).toBe('🚀'.repeat(96));});

it('keeps Jobs visible, retains draft and preflight across collapse, and resets separately',async()=>{
 const confirm=vi.spyOn(window,'confirm');vi.spyOn(window,'scrollTo').mockImplementation(()=>{});
 const {router}=renderBuilder();await configure();
 fireEvent.change(screen.getByLabelText('Job label (optional)'),{target:{value:'Retained draft'}});
 await check();const cash=screen.getByLabelText('Initial cash (quote)');
 const library=document.querySelector('.library')!;
 expect(library).toContainElement(document.querySelector('.config-disclosure') as HTMLElement);
 expect(library.querySelector('.panel-head')).toContainElement(screen.getByRole('heading',{name:'Jobs'}));
 expect(document.querySelector('.context')?.parentElement).toBe(library.parentElement);
 expect(document.querySelectorAll('.config-disclosure')).toHaveLength(1);
 fireEvent.click(screen.getByRole('button',{name:'Collapse settings'}));
 expect(router.state.location.pathname).toBe('/backtests');expect(document.getElementById('configuration-body')).toHaveAttribute('inert');expect(screen.getByRole('heading',{name:'Jobs'})).toBeVisible();
 fireEvent.click(screen.getByRole('link',{name:'New backtest'}));
 expect(screen.getByLabelText('Initial cash (quote)')).toBe(cash);
 expect(screen.getByLabelText('Job label (optional)')).toHaveValue('Retained draft');
 expect(screen.getByRole('button',{name:'Submit backtest'})).toBeEnabled();
 fireEvent.keyDown(screen.getByLabelText('Job label (optional)'),{key:'Escape'});
 expect(screen.getByRole('button',{name:'Continue setup'})).toHaveAttribute('aria-expanded','false');
 expect(screen.getByRole('button',{name:'Continue setup'})).toHaveFocus();
 fireEvent.click(screen.getByRole('button',{name:'Continue setup'}));
 expect(screen.getByLabelText('Job label (optional)')).toHaveValue('Retained draft');
 fireEvent.click(screen.getByRole('button',{name:'Reset form'}));
 expect(screen.getByLabelText('Job label (optional)')).toHaveValue('');
 expect(screen.getByLabelText('Start date')).toHaveValue('2023-01-01');
 expect(screen.getByRole('button',{name:'Submit backtest'})).toBeDisabled();
 expect(confirm).not.toHaveBeenCalled();
});
it('retains history filters, selected job and draft without changing route on expansion',async()=>{
 vi.spyOn(window,'scrollTo').mockImplementation(()=>{});
 const {router}=renderBuilder();await configure();
 fireEvent.change(screen.getByLabelText('Top N',{exact:true}),{target:{value:'3'}});
 await act(async()=>{await router.navigate('/backtests/not-a-uuid?state=running&risk_mode=none&limit=10');});
 const history=document.getElementById('panel-history')!;history.scrollTop=150;
 fireEvent.click(screen.getByRole('button',{name:'Continue setup'}));
 expect(router.state.location.search).toBe('?state=running&risk_mode=none&limit=10');
 expect(screen.getByLabelText('Top N',{exact:true})).toHaveValue(3);
 fireEvent.click(screen.getByRole('button',{name:'Collapse settings'}));
 expect(router.state.location.pathname).toBe('/backtests/not-a-uuid');
 expect(router.state.location.search).toBe('?state=running&risk_mode=none&limit=10');
 expect(document.getElementById('panel-history')).toBe(history);expect(history.scrollTop).toBe(150);
 await act(async()=>{await router.navigate(-1);});
 expect(router.state.location.pathname).toBe('/backtests/new');
 expect(screen.getByRole('button',{name:'Collapse settings'})).toHaveAttribute('aria-expanded','true');
 expect(screen.getByLabelText('Top N',{exact:true})).toHaveValue(3);
 await act(async()=>{await router.navigate(1);});
 expect(router.state.location.pathname).toBe('/backtests/not-a-uuid');
 expect(screen.getByRole('button',{name:'Continue setup'})).toHaveAttribute('aria-expanded','false');
});
it('continues guarding document departure while the dirty draft is hidden',async()=>{
 vi.spyOn(window,'scrollTo').mockImplementation(()=>{});const confirm=vi.spyOn(window,'confirm').mockReturnValue(false);
 renderBuilder();await configure();fireEvent.click(screen.getByRole('button',{name:'Collapse settings'}));
 const unload=new Event('beforeunload',{cancelable:true});window.dispatchEvent(unload);expect(unload.defaultPrevented).toBe(true);
 fireEvent.click(screen.getByRole('link',{name:'Overview'}));expect(confirm).toHaveBeenCalledTimes(1);
});

it('finishes one submission while collapsed, opens its job, and starts a fresh draft',async()=>{
 vi.spyOn(window,'scrollTo').mockImplementation(()=>{});
 const id='30000000-0000-4000-8000-000000000003',now=new Date().toISOString();let accept!:(r:Response)=>void;
 const {router,fetch}=renderBuilder((u,i)=>{
  if(u.pathname.endsWith('/jobs') && i.method==='POST')return new Promise(r=>{accept=r;});
  if(u.pathname.endsWith(`/jobs/${id}`))return response({job_id:id,state:'queued',created_at:now,updated_at:now,generated_at:now,refresh_status:'poll',next_allowed_refresh_at:now,retry_after_seconds:0,cancel_requested_at:null,progress:{percent:0,processed_units:0,total_units:1,updated_at:now,pipeline_stage:'queued'},request:{coordinates:body().coordinates,timeframe:'15m',time_range:body().time_range,risk_mode:'none'}});
 });
 await configure();await check();fireEvent.click(screen.getByRole('button',{name:'Submit backtest'}));
 await waitFor(()=>expect(accept).toBeDefined());fireEvent.click(screen.getByRole('button',{name:'Collapse settings'}));
 await act(async()=>accept(response({job_id:id,organization_id:id,request_hash:'hash',result_config_hash:'config',state:'queued'},201)));
 await waitFor(()=>expect(router.state.location.pathname).toBe(`/backtests/${id}`));
 expect(screen.getByRole('button',{name:'New backtest'})).toHaveAttribute('aria-expanded','false');
 await waitFor(()=>expect(document.getElementById('selected-job-heading')).toBeVisible());
 expect(loadRecovery('actor')).toBeNull();
 expect(fetch.mock.calls.filter(([u,i])=>new URL(u).pathname.endsWith('/jobs') && i?.method==='POST')).toHaveLength(1);
 await act(async()=>{await router.navigate(-2);});
 expect(router.state.location.pathname).toBe('/backtests/new');
 expect(await screen.findByLabelText('Start date')).toHaveValue('2023-01-01');
 expect(screen.getByRole('button',{name:'Check configuration'})).toBeEnabled();
 expect(screen.getByRole('button',{name:'Submit backtest'})).toBeDisabled();
});
it('keeps the editor expanded when the visible job receives its result selection',async()=>{
 vi.spyOn(window,'scrollTo').mockImplementation(()=>{});
 const id='40000000-0000-4000-8000-000000000004',now=new Date().toISOString();let finish!:(r:Response)=>void;
 const job={job_id:id,state:'succeeded',created_at:now,updated_at:now,generated_at:now,refresh_status:'terminal',next_allowed_refresh_at:now,retry_after_seconds:0,cancel_requested_at:null,progress:{percent:100,processed_units:1,total_units:1,updated_at:now,pipeline_stage:'done'},request:{coordinates:body().coordinates,timeframe:'15m',time_range:body().time_range,risk_mode:'none'}};
 const {router}=renderBuilder(u=>{
  if(u.pathname.endsWith(`/jobs/${id}`))return response(job);
  if(u.pathname.endsWith('/summary'))return new Promise(r=>{finish=r;});
  if(u.pathname.endsWith('/top'))return response({items:[]});
 });
 await configure();await act(async()=>{await router.navigate(`/backtests/${id}`);});
 await waitFor(()=>expect(finish).toBeDefined());
 fireEvent.click(screen.getByRole('button',{name:'Continue setup'}));
 await act(async()=>finish(response({job,top_variants:{items:[]},selected_variant_key:'v1',retry_after_seconds:0})));
 expect(router.state.location.pathname).toBe(`/backtests/${id}`);
 expect(screen.getByRole('button',{name:'Collapse settings'})).toHaveAttribute('aria-expanded','true');
 expect(screen.getByLabelText('Start date')).toHaveValue('2026-03-26');
 fireEvent.click(screen.getByRole('button',{name:'Collapse settings'}));
 await waitFor(()=>expect(router.state.location.search).toBe('?variant=v1'));
 expect(router.state.location.pathname).toBe(`/backtests/${id}`);
});

it('persists only an allowlisted animation preference and supports immediate reversals',async()=>{
 localStorage.setItem('roehub.backtests.motion','invalid');renderBuilder();await configure();
 expect(screen.getByLabelText('Animation')).toHaveValue('normal');
 fireEvent.change(screen.getByLabelText('Animation'),{target:{value:'slow'}});
 expect(localStorage.getItem('roehub.backtests.motion')).toBe('slow');
 expect(document.documentElement).toHaveStyle('--motion-duration: 520ms');
 fireEvent.click(screen.getByRole('button',{name:'Collapse settings'}));
 expect(document.getElementById('configuration-body')).toHaveAttribute('aria-hidden','true');
 fireEvent.click(screen.getByRole('button',{name:'Continue setup'}));
 expect(document.getElementById('configuration-body')).not.toHaveAttribute('inert');
 fireEvent.change(screen.getByLabelText('Animation'),{target:{value:'off'}});
 expect(document.documentElement).toHaveStyle('--motion-duration: 0ms');
 expect(screen.getByLabelText('Start date')).toHaveValue('2026-03-26');
 localStorage.removeItem('roehub.backtests.motion');
});

it('prefills the explicit synthetic preset and checks once without creating a job',async()=>{
 const expanded=structuredClone(defaults);expanded.indicator_param_specs['ma.ema'].params.window={mode:'explicit',values:[5,10,15,20,25,30,35,40,45,50]};
 const {fetch}=renderBuilder(u=>u.pathname.endsWith('runtime-defaults')?response(expanded):undefined,'/backtests/new?preset=synthetic');
 await waitFor(()=>expect(screen.getByRole('button',{name:'Submit backtest'})).toBeEnabled());
 expect(screen.getByLabelText('Start date')).toHaveValue('2026-02-27');
 expect(screen.getByLabelText('End date')).toHaveValue('2026-03-29');
 expect(screen.getByLabelText('Indicator',{exact:true})).toHaveValue('ma.ema');
 expect(screen.getByLabelText('Timeframe',{exact:true})).toHaveValue('15m');
 const posted=JSON.parse(String(fetch.mock.calls.find(([u])=>new URL(u).pathname.endsWith('/preflight'))![1]?.body));
 expect(posted.indicators[0].window).toEqual({start:5,stop:50,step:5});
 expect(posted.risk).toEqual({mode:'tp_sl_grid',tp:{enabled:true,start_pct:1,stop_pct:2,step_pct:1},sl:{enabled:true,start_pct:1,stop_pct:2,step_pct:1}});
 expect(fetch.mock.calls.filter(([u])=>new URL(u).pathname.endsWith('/preflight'))).toHaveLength(1);
 expect(fetch.mock.calls.filter(([u,i])=>new URL(u).pathname.endsWith('/jobs')&&i?.method==='POST')).toHaveLength(0);
 fireEvent.change(screen.getByLabelText('Top N',{exact:true}),{target:{value:'2'}});
 expect(screen.getByRole('button',{name:'Submit backtest'})).toBeDisabled();
 expect(fetch.mock.calls.filter(([u])=>new URL(u).pathname.endsWith('/preflight'))).toHaveLength(1);
});
