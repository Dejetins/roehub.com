import '@testing-library/jest-dom/vitest';
import {afterEach,beforeEach,expect,it,vi} from 'vitest';
import {act,cleanup,fireEvent,render,screen,waitFor,within} from '@testing-library/react';
import {createMemoryRouter,RouterProvider} from 'react-router';
import {QueryClientProvider} from '@tanstack/react-query';
import {I18nextProvider} from 'react-i18next';
import {DataPage} from './data-page';
import {ArtifactProgress,TimeframeCoverage,ArtifactSummary} from './data-artifacts';
import {profileDraft} from './artifact-profile';
import {createQueryClient} from './query-client';
import {createI18n} from './i18n';
import type {ArtifactDetail,ArtifactDefaults,ArtifactRequest} from './artifacts-api';
const indicator=(id:string)=>({indicator_id:id,sources:['close'],params:{window:{mode:'explicit' as const,values:[5,10],start:null,stop_incl:null,step:null}},signal_params:{}});
const profile={profile_id:'profile',revision:1,catalog_revision:'a'.repeat(64),mode:'on_demand' as const,history:{kind:'all_loaded' as const,start_utc:null,end_utc:null},derivative_timeframes:['1h'],indicators:[indicator('ema'),indicator('sma')],catalog_indicators:[indicator('ema'),indicator('sma')],tp_levels_pct:[1],sl_levels_pct:[1]};
const schedule={interval_days:1 as const,timezone:'Europe/Moscow',local_time:'02:00:00',anchor:'2026-10-08'};
const defaults:ArtifactDefaults={profile,schedule,base_timeframes:['1m','1h'],unavailable_indicator_ids:[],can_mutate:true};
const row=(symbol='BTCUSDT'):ArtifactDetail=>({coordinate:{exchange:'binance',market_type:'spot',symbol},control_version:1,profile,schedule,inherits_schedule:true,next_run:'2026-10-09T00:00:00Z',last_success:null,source_through:null,covered_through:null,base_state:'waiting_for_source',timeframes:[],base_request:null,derivative_request:null,size_bytes:null,location_label:'Backtest artifacts'});
function mount(handler?:(u:URL,o:RequestInit)=>Response|Promise<Response>|undefined,initial='/data?view=artifacts&market=1&symbol=BTCUSDT'){
 vi.stubGlobal('fetch',vi.fn((u:URL,o:RequestInit)=>handler?.(u,o)??Promise.resolve(Response.json(u.pathname.endsWith('/markets')?{items:[{market_id:1,exchange_name:'binance',market_type:'spot',market_code:'binance:spot'}]}:u.pathname.endsWith('/defaults')?defaults:u.pathname.endsWith('/inventory')?{snapshot_id:'s',observed_at:'2026-10-08',items:[row(),row('ETHUSDT')],next_cursor:null}:u.pathname.endsWith('/requests')?{items:[],next_cursor:null}:row(u.pathname.split('/').pop())))));
 const client=createQueryClient(),router=createMemoryRouter([{path:'*',element:<DataPage subject="owner"/>}],{initialEntries:[initial]});
 render(<I18nextProvider i18n={createI18n('en')}><QueryClientProvider client={client}><RouterProvider router={router}/></QueryClientProvider></I18nextProvider>);return {client,router};
}
beforeEach(()=>{HTMLDialogElement.prototype.showModal=function(){this.setAttribute('open','');};HTMLDialogElement.prototype.close=function(){this.removeAttribute('open');};});
afterEach(()=>{cleanup();vi.restoreAllMocks();vi.unstubAllGlobals();});
const result=()=>Response.json({command_id:'cmd',items:[{coordinate:row().coordinate,request_id:null,state:null,control_version:2,outcome:'policy_updated'}]});
it('persists runtime catalog exclusions without unrelated fields or null job polling',async()=>{let body:any;mount((u,o)=>o.method==='POST'?(body=JSON.parse(String(o.body)),result()):undefined);await screen.findByRole('heading',{name:'BTCUSDT'});fireEvent.click(screen.getByRole('button',{name:'Edit profile and schedule'}));fireEvent.click(screen.getByRole('button',{name:'Configure indicators'}));expect(screen.getByRole('checkbox',{name:'ema'})).toBeChecked();fireEvent.click(screen.getByRole('checkbox',{name:'sma'}));fireEvent.click(screen.getByRole('button',{name:'Done'}));fireEvent.click(screen.getByRole('button',{name:'Save changes'}));await waitFor(()=>expect(body).toBeDefined());expect(body.items[0].patch.profile.indicators.map((i:any)=>i.indicator_id)).toEqual(['ema']);expect(Object.keys(body.items[0].patch)).toEqual(['profile']);await screen.findByText('Changes saved. Scheduled anchor is unchanged.');expect(vi.mocked(fetch).mock.calls.some(([u])=>String(u).includes('/requests/null'))).toBe(false);});
it.each(['1','3','7','inherit'])('bulk schedule %s preserves profile and anchor',async(value)=>{let body:any;mount((u,o)=>o.method==='POST'?(body=JSON.parse(String(o.body)),result()):undefined);await screen.findByRole('heading',{name:'BTCUSDT'});fireEvent.click(screen.getByRole('checkbox',{name:'Select page'}));fireEvent.click(screen.getByRole('button',{name:'Edit profile and schedule'}));fireEvent.change(screen.getByLabelText('Refresh interval'),{target:{value}});fireEvent.click(screen.getByRole('button',{name:'Save changes'}));await waitFor(()=>expect(body).toBeDefined());expect(body.items).toHaveLength(2);for(const i of body.items){expect(Object.keys(i.patch)).toEqual(['schedule']);expect(i.patch.schedule).toEqual(value==='inherit'?null:{...schedule,interval_days:Number(value)});}});
it('rejects malformed TP instead of silently saving previous values',async()=>{mount();await screen.findByRole('heading',{name:'BTCUSDT'});fireEvent.click(screen.getByRole('button',{name:'Edit profile and schedule'}));fireEvent.change(screen.getByLabelText('TP · % · comma separated'),{target:{value:'1, nope'}});expect(screen.getByRole('button',{name:'Save changes'})).toBeDisabled();expect(screen.getByRole('alert')).toHaveTextContent('positive numeric');});
it('locks retained identity, rejects late selection and clears denied content',async()=>{let finish!:(r:Response)=>void;let denied=false;const {router,client}=mount(u=>u.pathname.endsWith('/ETHUSDT')?new Promise(r=>{finish=r;}):denied&&u.pathname.endsWith('/BTCUSDT')?Response.json({}, {status:403}):undefined);await screen.findByRole('heading',{name:'BTCUSDT'});await act(()=>router.navigate('/data?view=artifacts&market=1&symbol=ETHUSDT'));expect(screen.getByRole('heading',{name:'BTCUSDT'})).toBeVisible();expect(screen.getByRole('button',{name:'Update now'})).toBeDisabled();await act(()=>router.navigate('/data?view=artifacts&market=1&symbol=BTCUSDT'));await act(async()=>finish(Response.json(row('ETHUSDT'))));expect(screen.getByRole('heading',{name:'BTCUSDT'})).toBeVisible();denied=true;await act(()=>client.refetchQueries({queryKey:['artifacts','owner','detail']}));await waitFor(()=>expect(screen.queryByRole('button',{name:'Update now'})).not.toBeInTheDocument());expect(screen.queryByRole('button',{name:'BTCUSDT'})).not.toBeInTheDocument();});
it('recovers unknown outcome via same-key GET and never resends command',async()=>{let posts=0,recovery='';const {router}=mount((u,o)=>o.method==='POST'?(posts++,Promise.reject(new TypeError('lost'))):u.pathname.includes('/commands/')?(recovery=u.pathname,result()):undefined);await screen.findByRole('heading',{name:'BTCUSDT'});fireEvent.click(screen.getByRole('button',{name:'Update now'}));await waitFor(()=>expect(screen.getByRole('button',{name:'Check command result'})).toBeEnabled());const key=new URLSearchParams(router.state.location.search).get('artifactkey')!;expect(key).toMatch(/^art-/);expect(screen.getByRole('button',{name:'Update now'})).toBeDisabled();fireEvent.click(screen.getByRole('button',{name:'Check command result'}));await waitFor(()=>expect(recovery).toContain(key));expect(posts).toBe(1);});
it('uses no fake percent for unknown shared progress',()=>{const r:ArtifactRequest={request_id:'r',coordinate:row().coordinate,kind:'derivatives',state:'running',reason_code:null,created_at:'2026-10-08',progress:[{phase:'derivatives',timeframe:null,completed_units:null,total_units:null,unit:'bars',progress_epoch:1,observed_at:null}]};render(<I18nextProvider i18n={createI18n('en')}><ArtifactProgress request={r} label="Derivatives"/></I18nextProvider>);expect(screen.getByText('Calculating indicators')).toBeVisible();expect(screen.queryByRole('progressbar')).toBeNull();});
it('adds new catalog entries to explicit draft, retaining saved exclusions',()=>{const saved={...profile,indicators:[indicator('ema')]};const expanded={...defaults,profile:{...profile,catalog_indicators:[...profile.catalog_indicators,indicator('new')]}};expect(profileDraft(saved,expanded).indicators.map(i=>i.indicator_id)).toEqual(['ema','new']);expect(saved.indicators).toHaveLength(1);});
it('honors read Retry-After and retains continuation through empty page',async()=>{let failed=false;const {client}=mount(u=>u.pathname.endsWith('/inventory')?(failed?Response.json({}, {status:503,headers:{'Retry-After':'60'}}):Response.json({snapshot_id:'s',observed_at:'2026-10-08',items:[],next_cursor:'next'})):undefined,'/data?view=artifacts&market=1&type=1');await screen.findByRole('button',{name:'Next page'});failed=true;await act(()=>client.refetchQueries({queryKey:['artifacts','owner','inventory']}));await waitFor(()=>expect(screen.getByRole('button',{name:'Refresh'})).toBeDisabled());});
it('does not lose a valid SL edit while TP is invalid',async()=>{let body:any;mount((u,o)=>o.method==='POST'?(body=JSON.parse(String(o.body)),result()):undefined);await screen.findByRole('heading',{name:'BTCUSDT'});fireEvent.click(screen.getByRole('button',{name:'Edit profile and schedule'}));fireEvent.change(screen.getByLabelText('TP · % · comma separated'),{target:{value:'bad'}});fireEvent.change(screen.getByLabelText('SL · % · comma separated'),{target:{value:'2'}});fireEvent.change(screen.getByLabelText('TP · % · comma separated'),{target:{value:'3'}});fireEvent.click(screen.getByRole('button',{name:'Save changes'}));await waitFor(()=>expect(body).toBeDefined());expect(body.items[0].patch.profile.sl_levels_pct).toEqual([2]);expect(body.items[0].patch.profile.tp_levels_pct).toEqual([3]);});
it('keeps independent grid errors until their own field is repaired',async()=>{mount();await screen.findByRole('heading',{name:'BTCUSDT'});fireEvent.click(screen.getByRole('button',{name:'Edit profile and schedule'}));fireEvent.click(screen.getByRole('button',{name:'Configure indicators'}));fireEvent.change(screen.getByLabelText('ema · Compute · window · values'),{target:{value:'bad'}});fireEvent.change(screen.getByLabelText('sma · Compute · window · values'),{target:{value:'8'}});expect(screen.getByRole('button',{name:'Done'})).toBeDisabled();fireEvent.change(screen.getByLabelText('ema · Compute · window · values'),{target:{value:'6'}});expect(screen.getByRole('button',{name:'Done'})).toBeEnabled();});
it('guards draft departure and retains the draft on cancellation',async()=>{const confirm=vi.spyOn(window,'confirm').mockReturnValue(false);const {router}=mount();await screen.findByRole('heading',{name:'BTCUSDT'});fireEvent.click(screen.getByRole('button',{name:'Edit profile and schedule'}));fireEvent.click(screen.getByRole('button',{name:'Configure indicators'}));fireEvent.click(screen.getByRole('checkbox',{name:'sma'}));fireEvent.click(screen.getByRole('button',{name:'Done'}));fireEvent.click(screen.getByRole('link',{name:'Candles'}));await waitFor(()=>expect(confirm).toHaveBeenCalled());expect(router.state.location.search).toContain('view=artifacts');fireEvent.click(screen.getByRole('button',{name:'Configure indicators'}));expect(screen.getByRole('checkbox',{name:'sma'})).not.toBeChecked();fireEvent.click(screen.getByRole('button',{name:'Done'}));fireEvent.click(screen.getByRole('button',{name:'Cancel'}));expect(screen.getByRole('button',{name:'Edit profile and schedule'})).toHaveFocus();});
it('shows active shared base work instead of a stale completed organization request',()=>{
 const request:ArtifactRequest={request_id:'old',coordinate:row().coordinate,kind:'base',state:'error',reason_code:'old_failure',created_at:'2026-10-08',progress:[]};
 render(<I18nextProvider i18n={createI18n('en')}><ArtifactProgress request={request} state="queued" label="Base NPY"/></I18nextProvider>);
 expect(screen.getByText('Queued')).toBeVisible();expect(screen.queryByText('old_failure')).toBeNull();
});
it('polls shared base work without an organization request ID',async()=>{
 let reads=0;
 mount(u=>u.pathname.endsWith('/BTCUSDT')?Response.json({...row(),base_state:++reads>1?'ready':'queued'}):u.pathname.endsWith('/inventory')?Response.json({items:[{...row(),base_state:'queued'}],next_cursor:null}):undefined);
 await screen.findByRole('heading',{name:'BTCUSDT'});
 await waitFor(()=>expect(screen.getByText('Ready')).toBeVisible(),{timeout:4000});
 expect(reads).toBeGreaterThan(1);
 expect(vi.mocked(fetch).mock.calls.some(([u])=>String(u).includes('/requests/null'))).toBe(false);
});

it('keeps errors compact in the catalog and opens row details with the keyboard',async()=>{
 const failed={...row('ETHUSDT'),base_state:'error',base_request:{request_id:'failed',coordinate:row('ETHUSDT').coordinate,kind:'base',state:'error',reason_code:'source_or_storage_unavailable',created_at:'2026-10-08',progress:[]}};
 mount(u=>u.pathname.endsWith('/inventory')?Response.json({snapshot_id:'s',observed_at:'2026-10-08',items:[row(),failed],next_cursor:null}):u.pathname.endsWith('/ETHUSDT')?Response.json(failed):undefined);
 const button=await screen.findByRole('button',{name:'ETHUSDT'}),tr=button.closest('tr')!;
 expect(within(tr).queryByText('source_or_storage_unavailable')).toBeNull();
 expect(within(tr).getByText('Error · select instrument for details')).toHaveClass('sr-only');
 expect(within(tr).getByText('Off')).toBeInTheDocument();
 expect(within(tr).queryByText('Not requested')).toBeNull();
 fireEvent.keyDown(tr,{key:'Enter'});
 await screen.findByRole('heading',{name:'ETHUSDT'});
 expect(await screen.findByText('source_or_storage_unavailable')).toBeVisible();
});

it('sorts compact coverage and discloses only relevant details',()=>{
 const interval=(timeframe:string,state:string,closed_bars:number,missing_bars:number|null)=>({timeframe,state,closed_bars,missing_bars,start_utc:null,end_utc:null,retry_after_source_end_ms:null});
 render(<I18nextProvider i18n={createI18n('en')}><TimeframeCoverage timeframes={[interval('1d','insufficient_history',0,null),interval('2h','ready',1,0),interval('1m','ready',120,3)]}/></I18nextProvider>);
 fireEvent.click(screen.getByText('Timeframe coverage'));
 const buttons=screen.getAllByRole('button');expect(buttons.map(b=>b.textContent)).toEqual(['1m','2h','1d']);
 expect(screen.queryByText(/Missing candles:/)).toBeNull();
 fireEvent.click(buttons[0]);expect(screen.getByText('Missing candles: 3')).toBeVisible();expect(buttons[0]).toHaveAttribute('aria-expanded','true');
 fireEvent.click(buttons[2]);expect(screen.queryByText('Missing candles: 3')).toBeNull();expect(screen.getByText('Not enough candles to form this interval.')).toBeVisible();
 expect(screen.queryByText('insufficient_history')).toBeNull();
});

it('deduplicates artifact dates while retaining source lag and failed attempts',()=>{
 const timestamp='2026-10-09T12:00:00Z';
 const data={...row(),source_through:timestamp,covered_through:timestamp,last_success:timestamp};
 const wrap=(value:ArtifactDetail)=><I18nextProvider i18n={createI18n('en')}><ArtifactSummary row={value}/></I18nextProvider>;
 const {rerender}=render(wrap(data));
 expect(screen.queryByText('Candles through')).toBeNull();expect(screen.queryByText('Last attempt')).toBeNull();
 expect(screen.getByText('Updated')).toBeVisible();expect(screen.getByText('Daily · 02:00 Europe/Moscow')).toBeVisible();
 rerender(wrap({...data,source_through:'2026-10-09T13:00:00Z',base_request:{request_id:'failed',coordinate:data.coordinate,kind:'base',state:'error',reason_code:'failed',created_at:'2026-10-09T14:00:00Z',progress:[]}}));
 expect(screen.getByText('Candles through')).toBeVisible();expect(screen.getByText('Last attempt')).toBeVisible();
});

it('opens the global journal, filters on the server and separates job details from instrument controls',async()=>{
 const job:ArtifactRequest={request_id:'job-1',coordinate:{exchange:'bybit',market_type:'futures',symbol:'SOLUSDT'},kind:'base',state:'error',reason_code:'source_or_storage_unavailable',created_at:'2026-10-09T12:00:00Z',progress:[]};
 const queries:string[]=[];
 mount(u=>u.pathname.endsWith('/requests')?(queries.push(u.search),Response.json({items:[job],next_cursor:null})):u.pathname.endsWith('/requests/job-1')?Response.json(job):u.pathname.endsWith('/events')?Response.json({items:[{state:'error',reason_code:job.reason_code,observed_at:job.created_at,progress:[]}]}):undefined);
 await screen.findByRole('heading',{name:'BTCUSDT'});
 fireEvent.click(screen.getByRole('button',{name:'Show journal'}));
 const journal=await screen.findByRole('region',{name:'Job journal'});
 expect(within(journal).getByText('All exchanges')).toBeVisible();
 fireEvent.change(within(journal).getByLabelText('Job state'),{target:{value:'error'}});
 await waitFor(()=>expect(queries.at(-1)).toContain('state=error'));
 fireEvent.change(within(journal).getByLabelText('Job type'),{target:{value:'base'}});
 await waitFor(()=>expect(queries.at(-1)).toContain('kind=base'));
 await waitFor(()=>expect(within(journal).getByRole('link',{name:'SOLUSDT · Base files'})).toBeVisible());
 fireEvent.click(within(journal).getByRole('link',{name:'SOLUSDT · Base files'}));
 await screen.findByRole('heading',{name:'SOLUSDT'});
 expect(screen.queryByRole('button',{name:'Update now'})).toBeNull();
 expect(screen.queryByText('Timeframe coverage')).toBeNull();
 fireEvent.click(screen.getByRole('button',{name:'Back to instrument'}));
 await screen.findByRole('heading',{name:'BTCUSDT'});
 expect(screen.getByRole('button',{name:'Update now'})).toBeVisible();
 fireEvent.keyDown(journal,{key:'Escape'});
 expect(screen.queryByRole('region',{name:'Job journal'})).toBeNull();
 expect(screen.getByRole('button',{name:'Show journal'})).toHaveFocus();
});

it('removes journal and inspector data on denied job reads',async()=>{
 const job:ArtifactRequest={request_id:'denied',coordinate:row().coordinate,kind:'base',state:'ready',reason_code:null,created_at:'2026-10-09T12:00:00Z',progress:[]};
 mount(u=>u.pathname.endsWith('/requests')?Response.json({items:[job],next_cursor:null}):u.pathname.endsWith('/requests/denied')?Response.json({}, {status:403}):undefined);
 await screen.findByRole('heading',{name:'BTCUSDT'});
 fireEvent.click(screen.getByRole('button',{name:'Show journal'}));
 fireEvent.click(await screen.findByRole('link',{name:'BTCUSDT · Base files'}));
 await waitFor(()=>expect(screen.queryByRole('link',{name:'BTCUSDT · Base files'})).toBeNull());
 expect(screen.queryByRole('heading',{name:'BTCUSDT'})).toBeNull();
 expect(screen.queryByRole('button',{name:'Update now'})).toBeNull();
});

it('continues journal pagination through an empty page and keeps filters on subsequent requests',async()=>{
 const cursors:string[]=[];
 mount(u=>u.pathname.endsWith('/requests')?(cursors.push(u.search),Response.json({items:[],next_cursor:u.searchParams.has('cursor')?null:'older'})):undefined,'/data?view=artifacts&artifactjournal=1&artifactstate=error&artifactkind=base');
 fireEvent.click(await screen.findByRole('button',{name:'Older jobs'}));
 await waitFor(()=>expect(cursors.at(-1)).toContain('cursor=older'));
 expect(cursors.at(-1)).toContain('state=error');expect(cursors.at(-1)).toContain('kind=base');
 await waitFor(()=>expect(screen.queryByRole('button',{name:'Older jobs'})).toBeNull());
});

it.each(['queued','running','ready'])('keeps %s artifact progress compact regardless of timeframe count',state=>{
 const request:ArtifactRequest={request_id:'compact',coordinate:row().coordinate,kind:'base',state:state as ArtifactRequest['state'],reason_code:null,created_at:'2026-10-09',progress:['1m','1h','1d'].map(timeframe=>({phase:'base',timeframe,completed_units:120,total_units:120,unit:'bars',progress_epoch:1,observed_at:null}))};
 render(<I18nextProvider i18n={createI18n('en')}><ArtifactProgress request={request} label="Base NPY"/></I18nextProvider>);
 expect(screen.queryAllByRole('progressbar')).toHaveLength(0);
 expect(screen.queryByText(/120 \/ 120/)).toBeNull();
 expect(screen.queryByText('Preparing files')).toBe(state==='running'?screen.getByText('Preparing files'):null);
});

it('keeps the inspector mounted and visually steady while refresh locks commands',async()=>{
 let finish!:(r:Response)=>void,hold=false;
 const {client}=mount(u=>u.pathname.endsWith('/BTCUSDT')&&hold?new Promise(r=>{finish=r;}):undefined);
 const heading=await screen.findByRole('heading',{name:'BTCUSDT'});
 const update=screen.getByRole('button',{name:'Update now'});
 await waitFor(()=>expect(update).toBeEnabled());
 hold=true;let refresh!:Promise<void>;
 act(()=>{refresh=client.refetchQueries({queryKey:['artifacts','owner','detail']});});
 await waitFor(()=>expect(update).toBeDisabled());
 expect(update.parentElement).toHaveClass('artifact-quiet-read');
 expect(screen.getByRole('heading',{name:'BTCUSDT'})).toBe(heading);
 expect(screen.queryByText('Loading data…')).toBeNull();
 await act(async()=>{finish(Response.json(row()));await refresh;});
 await waitFor(()=>expect(update).toBeEnabled());expect(screen.getByRole('heading',{name:'BTCUSDT'})).toBe(heading);
});

it('cancels modal edits, restores focus, and applies only to the profile draft',async()=>{
 let writes=0;mount((u,o)=>o.method==='POST'?(writes++,result()):undefined);
 await screen.findByRole('heading',{name:'BTCUSDT'});
 fireEvent.click(screen.getByRole('button',{name:'Edit profile and schedule'}));
 const trigger=screen.getByRole('button',{name:'Configure indicators'});
 trigger.focus();fireEvent.click(trigger);
 let dialog=screen.getByRole('dialog',{name:'Indicators'});
 fireEvent.click(within(dialog).getByRole('checkbox',{name:'sma'}));
 fireEvent.click(within(dialog).getByRole('button',{name:'Cancel'}));
 expect(trigger).toHaveFocus();expect(writes).toBe(0);
 fireEvent.click(trigger);dialog=screen.getByRole('dialog',{name:'Indicators'});
 expect(within(dialog).getByRole('checkbox',{name:'sma'})).toBeChecked();
 fireEvent.change(within(dialog).getByLabelText('ema · Compute · window · values'),{target:{value:'20'}});
 fireEvent.click(within(dialog).getByRole('button',{name:'Parameters SMA'}));
 fireEvent.change(within(dialog).getByRole('searchbox',{name:'Search indicators'}),{target:{value:'ema'}});
 expect(within(dialog).queryByRole('button',{name:'Parameters SMA'})).toBeNull();
 fireEvent.click(within(dialog).getByRole('button',{name:'Parameters EMA'}));
 expect(within(dialog).getByLabelText('ema · Compute · window · values')).toHaveValue('20');
 fireEvent.click(within(dialog).getByRole('button',{name:'Done'}));
 expect(writes).toBe(0);
 fireEvent.click(trigger);dialog=screen.getByRole('dialog',{name:'Indicators'});
 expect(within(dialog).getByLabelText('ema · Compute · window · values')).toHaveValue('20');
 fireEvent(dialog,new Event('cancel',{bubbles:false,cancelable:true}));expect(screen.queryByRole('dialog')).toBeNull();expect(trigger).toHaveFocus();
});
