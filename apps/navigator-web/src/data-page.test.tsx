import '@testing-library/jest-dom/vitest';
import {afterEach,expect,it,vi} from 'vitest';
import {act,cleanup,fireEvent,render,screen,waitFor,within} from '@testing-library/react';
import {createMemoryRouter,RouterProvider} from 'react-router';
import {QueryClientProvider} from '@tanstack/react-query';
import {I18nextProvider} from 'react-i18next';
import {DataPage} from './data-page';
import {LegacyDownloadsRedirect} from './data-downloads';
import {createQueryClient} from './query-client';
import {createI18n} from './i18n';
const suffix='&from=2024-01-01T00:00&to=2024-01-01T00:05';
const item=(symbol:string,pinned=false)=>({market_id:1,symbol,base_asset:symbol.slice(0,3),quote_asset:'USDT',price_step:.01,qty_step:.001,min_notional:5,selected:false,strategy_pinned:pinned,effective:pinned});
const pins=[{strategy_id:'strategy-a',name:'Running EMA',state:'running'}];
const catalog=(symbol:string)=>({snapshot_id:'snapshot',refreshed_at:'2026-10-02T00:00Z',catalog_state:'fresh',total:2,items:[item(symbol)],next_cursor:null});
const rows=['BTCUSDT','ETHUSDT'].map(symbol=>({...item(symbol),exchange_name:'binance',market_type:'spot',refreshed_at:'2026-10-02T00:00Z',strategies:[],coverage_state:'complete',coverage_percent:100,actual_candles:5,expected_candles:5,job:null}));
const coverage=(symbol:string)=>({market_id:1,symbol,timeframe:'1m',start_at:'2024-01-01T00:00Z',end_at:'2024-01-01T00:05Z',state:'complete',expected_candles:5,actual_candles:5,coverage_percent:100,observed_at:'2026-10-02T00:00Z',gap_count:0,gaps_truncated:false,gaps:[]});
const collection=(symbol:string,pinned=false)=>({...item(symbol,pinned),strategies:pinned?pins:[],last_candle_at:null,observed_at:'2026-10-02T00:00Z'});
const exchangeCatalog=(q='',exchange='binance')=>({exchange,start_at:'2024-01-01T00:00Z',end_at:'2024-01-01T00:05Z',observed_at:'2026-10-02T00:00Z',total:q==='ZZ'?0:2,offset:0,limit:50,snapshot:'snapshot',missing_market_ids:[],items:q==='ZZ'?[]:rows});
function mount(handler?: (url:URL,options:RequestInit)=>Response|Promise<Response>|undefined,initial='/data?market=1&symbol=BTCUSDT'+suffix){
 vi.stubGlobal('fetch',vi.fn((url:URL,options:RequestInit)=>handler?.(url,options)??Promise.resolve(Response.json(url.pathname.endsWith('/markets')?{items:[{market_id:1,exchange_name:'binance',market_type:'spot',market_code:'binance:spot'},{market_id:3,exchange_name:'bybit',market_type:'spot',market_code:'bybit:spot'}]}:url.pathname.endsWith('/instruments')?exchangeCatalog(url.searchParams.get('q')??'',url.searchParams.get('exchange')??'binance'):url.pathname.endsWith('/catalog')?catalog(url.searchParams.get('q')??''):url.pathname.endsWith('/collection')?collection(url.searchParams.get('symbol')??''):url.pathname.endsWith('/history-bounds')?{market_id:Number(url.searchParams.get('market_id')),symbol:url.searchParams.get('symbol'),state:'ready',first_open_at:'2024-01-01T00:00Z',end_at:'2024-01-01T00:05Z',observed_at:'2026-10-03T00:00Z'}:url.pathname.endsWith('/coverage')?coverage(url.searchParams.get('symbol')??''):{}))));
 const client=createQueryClient(),router=createMemoryRouter([{path:'/data/ingestion',element:<LegacyDownloadsRedirect/>},{path:'*',element:<DataPage subject="owner"/>}],{initialEntries:[initial]});
 render(<I18nextProvider i18n={createI18n('en')}><QueryClientProvider client={client}><RouterProvider router={router}/></QueryClientProvider></I18nextProvider>);return {client,router};
}
afterEach(()=>{cleanup();vi.useRealTimers();vi.restoreAllMocks();vi.unstubAllGlobals();});
it('retains the previous identity with commands locked and rejects a late selection response',async()=>{
 let finish!:(response:Response)=>void;
 const {router}=mount(url=>url.pathname.endsWith('/collection')&&url.searchParams.get('symbol')==='ETHUSDT'?new Promise<Response>(r=>{finish=r;}):undefined);
 await screen.findByRole('heading',{name:'BTCUSDT'});
 await act(()=>router.navigate('/data?market=1&symbol=ETHUSDT'+suffix));await waitFor(()=>expect(finish).toBeDefined());
 expect(screen.getByRole('heading',{name:'BTCUSDT'})).toBeVisible();expect(screen.getByRole('switch',{name:'Streaming'})).toBeDisabled();
 expect(screen.getByRole('button',{name:'Download period'})).toBeDisabled();
 await act(()=>router.navigate('/data?market=1&symbol=BTCUSDT'+suffix));await act(async()=>finish(Response.json(collection('ETHUSDT'))));
 expect(screen.getByRole('heading',{name:'BTCUSDT'})).toBeVisible();expect(screen.queryByRole('heading',{name:'ETHUSDT'})).not.toBeInTheDocument();
});
it('commits an empty filtered catalog and hides protected state when inspector access is denied',async()=>{
 let denied=false;const {client,router}=mount(url=>denied&&url.pathname.endsWith('/collection')?Response.json({error:{code:'denied'}},{status:403}):undefined);
 await screen.findByRole('heading',{name:'BTCUSDT'});
 await act(()=>router.navigate('/data?market=1&symbol=BTCUSDT&q=ZZ'+suffix));await screen.findByText('No matching instruments');expect(document.querySelectorAll('.data-catalog-table tbody tr')).toHaveLength(0);
 denied=true;await act(()=>client.refetchQueries({queryKey:['data-inspector','owner']}));
 await waitFor(()=>expect(screen.queryByRole('switch')).not.toBeInTheDocument());expect(screen.queryByText('100%')).not.toBeInTheDocument();
});
it('has one stream control and locks strategy-required collection with its reason',async()=>{
 mount(url=>url.pathname.endsWith('/collection')?Response.json(collection('BTCUSDT',true)):undefined);
 await screen.findByRole('heading',{name:'BTCUSDT'});
 const switches=screen.getAllByRole('switch');expect(switches).toHaveLength(1);expect(switches[0]).toBeDisabled();expect(switches[0]).toHaveAttribute('aria-checked','true');
 expect(screen.getByRole('link',{name:'Running EMA'})).toHaveAttribute('href','/strategies/strategy-a');
 expect(screen.getByText(/Cannot disable/)).toBeVisible();
});
it('keeps an unknown stream command disabled until explicit read reconciliation',async()=>{
 let writes=0;
 mount((url,options)=>url.pathname.includes('/selections/')&&options.method==='PUT'?(writes++,Response.json({}, {status:503})):undefined);
 await screen.findByRole('heading',{name:'BTCUSDT'});fireEvent.click(screen.getByRole('switch'));
 await screen.findByRole('button',{name:'Check saved state'});expect(screen.getByRole('switch')).toBeDisabled();expect(writes).toBe(1);
 fireEvent.click(screen.getByRole('switch'));expect(writes).toBe(1);
 fireEvent.click(screen.getByRole('button',{name:'Check saved state'}));await waitFor(()=>expect(screen.getByRole('switch')).toBeEnabled());
});
it('keeps server filters/sort in the URL and chooses exchanges rather than instruments in the library',async()=>{
 const {router}=mount();await screen.findByRole('heading',{name:'BTCUSDT'});
 expect(within(screen.getByRole('navigation',{name:'Exchanges'})).getAllByRole('link')).toHaveLength(2);
 fireEvent.change(screen.getByRole('combobox',{name:'History'}),{target:{value:'empty'}});
 await waitFor(()=>expect(router.state.location.search).toContain('history=empty'));
 fireEvent.click(screen.getByRole('button',{name:'History / download'}));await waitFor(()=>expect(router.state.location.search).toContain('sort=coverage'));
 fireEvent.click(within(screen.getByRole('navigation',{name:'Exchanges'})).getByRole('link',{name:'Bybit'}));
 await waitFor(()=>expect(router.state.location.search).toContain('exchange=bybit'));expect(router.state.location.search).not.toContain('symbol=');
});
it('validates the total download range before sending a command',async()=>{
 let writes=0;mount((url,options)=>{if(options.method==='POST')writes++;return undefined;});
 await screen.findByRole('heading',{name:'BTCUSDT'});
 fireEvent.change(within(screen.getByRole('complementary',{name:'Instrument settings'})).getByLabelText('From · UTC'),{target:{value:'2016-01-01T00:00'}});
 fireEvent.click(screen.getByRole('button',{name:'Download period'}));expect(screen.getByRole('button',{name:'Download period'})).toBeDisabled();expect(writes).toBe(0);
});
const jobResult=(state='succeeded')=>({job_id:'bounded-job',kind:'candle_ingestion',market_id:1,symbols:['BTCUSDT'],timeframe:'1m',start_at:'2024-01-01T00:00Z',end_at:'2024-01-01T00:05Z',state,attempt:1,completed_units:state==='queued'?0:5,total_units:5,rows_read:5,rows_written:5,error_code:null,created_at:'2026-10-02T00:00Z',updated_at:'2026-10-02T00:00Z',started_at:state==='queued'?null:'2026-10-02T00:00Z',finished_at:['succeeded','failed','cancelled'].includes(state)?'2026-10-02T00:00Z':null,can_cancel:false,can_retry:false,attempts:[]});
it('submits exactly one bounded historical download and exposes completion',async()=>{
 const writes:unknown[]=[];mount((url,options)=>{if(options.method==='POST'){writes.push(JSON.parse(String(options.body)));return Response.json(jobResult('queued'));}if(url.pathname.endsWith('/bounded-job'))return Response.json(jobResult());});
 await screen.findByRole('heading',{name:'BTCUSDT'});await waitFor(()=>expect(screen.getByRole('button',{name:'Download period'})).toBeEnabled());fireEvent.click(screen.getByRole('button',{name:'Download period'}));
 await screen.findByRole('region',{name:'Download request'},{timeout:3500});
 expect(writes).toHaveLength(1);expect(writes[0]).toMatchObject({kind:'candle_ingestion',market_id:1,symbols:['BTCUSDT'],start_at:'2024-01-01T00:00:00.000Z',end_at:'2024-01-01T00:05:00.000Z'});
});
it.each(['command','progress'])('hides protected instruments after a denied %s',async(kind)=>{
 mount((url,options)=>{if(options.method==='POST')return kind==='command'?Response.json({}, {status:403}):Response.json(jobResult('queued'));if(url.pathname.endsWith('/bounded-job'))return Response.json({}, {status:403});});
 await screen.findByRole('heading',{name:'BTCUSDT'});await waitFor(()=>expect(screen.getByRole('button',{name:'Download period'})).toBeEnabled());fireEvent.click(screen.getByRole('button',{name:'Download period'}));
 await waitFor(()=>expect(screen.queryByRole('switch')).not.toBeInTheDocument(),{timeout:3500});expect(document.querySelectorAll('.data-catalog-table tbody tr')).toHaveLength(0);
});
it('keeps the download default independent of the catalog coverage period',async()=>{
 const {router}=mount(url=>url.pathname.endsWith('/coverage')?Response.json({...coverage('BTCUSDT'),start_at:url.searchParams.get('start_at'),end_at:url.searchParams.get('end_at')}):undefined);
 await screen.findByRole('heading',{name:'BTCUSDT'});
 await act(()=>router.navigate('/data?market=1&symbol=BTCUSDT&from=2024-02-01T00:00&to=2024-02-01T00:05'));
 await waitFor(()=>expect(within(screen.getByRole('complementary',{name:'Instrument settings'})).getByLabelText('From · UTC')).toHaveValue('2024-01-01T00:00'));
 expect(document.querySelector<HTMLFormElement>('.data-range-picker input')?.value).toBe('2024-02-01T00:00');
 await act(()=>router.navigate(-1));await waitFor(()=>expect(within(screen.getByRole('complementary',{name:'Instrument settings'})).getByLabelText('From · UTC')).toHaveValue('2024-01-01T00:00'));
});
it('clears a batch when Back moves to an instrument in another market',async()=>{
 const writes:unknown[]=[];
 const {router}=mount((url,options)=>{const market=Number(url.searchParams.get('market_id')??1),symbol=url.searchParams.get('symbol')??url.searchParams.get('q')??'BTCUSDT';if(options.method==='POST'){writes.push(JSON.parse(String(options.body)));return Response.json(jobResult());}if(url.pathname.endsWith('/bounded-job'))return Response.json(jobResult());if(url.pathname.endsWith('/collection'))return Response.json({...collection(symbol),market_id:market});if(url.pathname.endsWith('/catalog'))return Response.json({...catalog(symbol),items:[{...item(symbol),market_id:market}]});if(url.pathname.endsWith('/coverage'))return Response.json({...coverage(symbol),market_id:market});if(url.pathname.endsWith('/instruments')&&url.searchParams.get('exchange')==='bybit')return Response.json({...exchangeCatalog('','bybit'),items:[{...rows[1],market_id:3,exchange_name:'bybit'}]});});
 await screen.findByRole('heading',{name:'BTCUSDT'});await act(()=>router.navigate('/data?exchange=bybit&market=3&symbol=ETHUSDT'+suffix));await screen.findByRole('heading',{name:'ETHUSDT'});
 fireEvent.click(screen.getByRole('checkbox',{name:'Download ETHUSDT · Spot'}));expect(screen.getByText(/Selected: 1/)).toBeVisible();
 await act(()=>router.navigate('/data?market=1&symbol=BTCUSDT'+suffix));await screen.findByRole('heading',{name:'BTCUSDT'});expect(screen.queryByText(/Selected: 1/)).not.toBeInTheDocument();
 await waitFor(()=>expect(screen.getByRole('button',{name:'Download period'})).toBeEnabled());fireEvent.click(screen.getByRole('button',{name:'Download period'}));await screen.findByRole('region',{name:'Download request'},{timeout:3500});expect(writes[0]).toMatchObject({market_id:1,symbols:['BTCUSDT']});
});
it('shows a failed download even when no candles have been stored',async()=>{
 mount(url=>url.pathname.endsWith('/instruments')?Response.json({...exchangeCatalog(),items:[{...rows[0],coverage_state:'empty',coverage_percent:0,actual_candles:0,job:{job_id:'failed-job',state:'failed',progress_percent:0}}]}):undefined);
 await screen.findByRole('link',{name:'Failed'});expect(within(screen.getByRole('table')).queryByText('No data')).not.toBeInTheDocument();
});

it('submits native datetime input events and commits the accepted range without restoring a recovery key',async()=>{
 const writes:Record<string,unknown>[]=[];const {router}=mount((url,options)=>{if(options.method==='POST'){const body=JSON.parse(String(options.body));writes.push(body);return Response.json({...jobResult(),start_at:body.start_at,end_at:body.end_at});}if(url.pathname.endsWith('/bounded-job'))return Response.json(jobResult());if(url.pathname.endsWith('/coverage'))return Response.json({...coverage('BTCUSDT'),start_at:url.searchParams.get('start_at'),end_at:url.searchParams.get('end_at')});});
 await screen.findByRole('heading',{name:'BTCUSDT'});const inspector=within(screen.getByRole('complementary',{name:'Instrument settings'}));
 fireEvent.input(inspector.getByLabelText('From · UTC'),{target:{value:'2024-02-01T00:00'}});fireEvent.input(inspector.getByLabelText('To · UTC'),{target:{value:'2024-02-01T00:03'}});
 await waitFor(()=>expect(screen.getByRole('button',{name:'Download period'})).toBeEnabled());fireEvent.click(screen.getByRole('button',{name:'Download period'}));await screen.findByRole('region',{name:'Download request'});
 expect(writes[0]).toMatchObject({start_at:'2024-02-01T00:00:00.000Z',end_at:'2024-02-01T00:03:00.000Z'});await waitFor(()=>expect(new URLSearchParams(router.state.location.search).get('from')).toBe('2024-02-01T00:00'));expect(router.state.location.search).not.toContain('requestkey');
});

it('uses one recovery action after reload and never resubmits a recovered request',async()=>{
 let posts=0;const {router}=mount((url,options)=>{if(options.method==='POST')posts++;if(url.pathname.endsWith('/lookup')||url.pathname.endsWith('/bounded-job'))return Response.json(jobResult());},'/data?market=1&symbol=BTCUSDT&requestkey=lost-response'+suffix);
 await screen.findByRole('heading',{name:'BTCUSDT'});expect(screen.getAllByRole('button',{name:'Check request'})).toHaveLength(1);expect(screen.getByRole('button',{name:'Download period'})).toBeDisabled();fireEvent.click(screen.getByRole('button',{name:'Check request'}));await screen.findByRole('region',{name:'Download request'});expect(router.state.location.search).not.toContain('requestkey');expect(posts).toBe(0);expect(screen.getByRole('button',{name:'Download period'})).toBeEnabled();
});


it('defaults to the entire confirmed history, preserves a manual draft, and keeps coverage reads bounded',async()=>{
 const writes:Record<string,unknown>[]=[];
 const {router,client}=mount((url,options)=>{
  if(url.pathname.endsWith('/history-bounds'))return Response.json({market_id:1,symbol:'BTCUSDT',state:'ready',first_open_at:'2017-08-17T04:00Z',end_at:'2026-10-02T12:00Z',observed_at:'2026-10-02T12:00Z'});
  if(options.method==='POST'){const body=JSON.parse(String(options.body));writes.push(body);return Response.json({...jobResult('queued'),start_at:body.start_at,end_at:body.end_at});}
  if(url.pathname.endsWith('/bounded-job'))return Response.json(jobResult('running'));
 });
 const inspector=within(await screen.findByRole('complementary',{name:'Instrument settings'}));
 await waitFor(()=>expect(inspector.getByLabelText('From · UTC')).toHaveValue('2017-08-17T04:00'));
 expect(inspector.getByLabelText('To · UTC')).toHaveValue('2026-10-02T12:00');
 fireEvent.input(inspector.getByLabelText('From · UTC'),{target:{value:'2018-01-01T00:00'}});
 await act(()=>client.refetchQueries({queryKey:['history-bounds','owner']}));
 expect(inspector.getByLabelText('From · UTC')).toHaveValue('2018-01-01T00:00');
 fireEvent.click(screen.getByRole('button',{name:'Download period'}));
 await waitFor(()=>expect(writes).toHaveLength(1));
 expect(writes[0]).toMatchObject({start_at:'2018-01-01T00:00:00.000Z',end_at:'2026-10-02T12:00:00.000Z'});
 expect(new URLSearchParams(router.state.location.search).get('from')).toBe('2024-01-01T00:00');
});

it('uses a named icon button and waits for the selected instrument history bounds',async()=>{
 let finish!:(r:Response)=>void;
 mount(url=>url.pathname.endsWith('/instruments')?Response.json({...exchangeCatalog(),items:rows.map(row=>({...row,coverage_state:'empty',coverage_percent:0}))}):url.pathname.endsWith('/history-bounds')&&url.searchParams.get('symbol')==='ETHUSDT'?new Promise<Response>(r=>finish=r):undefined);
 const button=await screen.findByRole('button',{name:'Download ETHUSDT · Spot'});
 expect(button.querySelector('svg')).not.toBeNull();expect(button.textContent).toBe('');
 fireEvent.click(button);await screen.findByRole('heading',{name:'ETHUSDT'});
 expect(screen.getByRole('button',{name:'Download period'})).toBeDisabled();
 await act(async()=>finish(Response.json({market_id:1,symbol:'ETHUSDT',state:'ready',first_open_at:'2019-03-01T00:00Z',end_at:'2026-10-02T12:00Z',observed_at:'2026-10-02T12:00Z'})));
 await waitFor(()=>expect(document.getElementById('data-download-from')).toHaveFocus());
 expect(document.getElementById('data-download-from')).toHaveValue('2019-03-01T00:00');
});


it('allows manual dates while history discovery is queued and never overwrites that draft',async()=>{
 let ready=false;
 const {client}=mount(url=>url.pathname.endsWith('/history-bounds')?Response.json({market_id:1,symbol:'BTCUSDT',state:ready?'ready':'queued',first_open_at:ready?'2017-08-17T04:00Z':null,end_at:'2026-10-02T12:00Z',observed_at:'2026-10-02T12:00Z'}):undefined);
 fireEvent.click(await screen.findByRole('button',{name:'Enter period manually'}));
 const inspector=within(screen.getByRole('complementary',{name:'Instrument settings'}));
 fireEvent.input(inspector.getByLabelText('From · UTC'),{target:{value:'2020-01-01T00:00'}});
 ready=true;await act(()=>client.refetchQueries({queryKey:['history-bounds','owner']}));
 expect(inspector.getByLabelText('From · UTC')).toHaveValue('2020-01-01T00:00');
 expect(screen.getByRole('button',{name:'Download period'})).toBeEnabled();
});

it('uses explicit monthly dates for a batch without starting history discovery for every instrument',async()=>{
 let batch=false,reads=0;
 mount(url=>{if(batch&&url.pathname.endsWith('/history-bounds'))reads++;return undefined;});
 await screen.findByRole('heading',{name:'BTCUSDT'});
 fireEvent.click(await screen.findByRole('checkbox',{name:'Download BTCUSDT · Spot'}));
 batch=true;fireEvent.click(screen.getByRole('checkbox',{name:'Download ETHUSDT · Spot'}));
 const inspector=within(screen.getByRole('complementary',{name:'Instrument settings'}));
 expect(inspector.getByLabelText('From · UTC')).not.toHaveValue('');
 expect(inspector.getByLabelText('To · UTC')).not.toHaveValue('');
 expect(reads).toBe(0);
});

it('loads the journal only on demand, keeps the catalog mounted, and restores trigger focus on Escape',async()=>{
 let historyReads=0;
 mount(url=>url.pathname.endsWith('/work-requests')?(historyReads++,Response.json({items:[jobResult()],next_cursor:null})):undefined,'/data?market=1'+suffix);
 await screen.findByRole('link',{name:'BTCUSDT'});expect(historyReads).toBe(0);
 const table=document.querySelector('.data-catalog-table');
 const trigger=screen.getByRole('button',{name:'Download log'});fireEvent.click(trigger);
 const journal=await screen.findByRole('region',{name:'Download log'});await within(journal).findByText('Completed');
 expect(document.querySelector('.data-catalog-table')).toBe(table);expect(historyReads).toBe(1);
 fireEvent.keyDown(journal,{key:'Escape'});await waitFor(()=>expect(trigger).toHaveFocus());expect(screen.queryByRole('region',{name:'Download log'})).not.toBeInTheDocument();
});

it('opens a failed catalog-row job inside its instrument and retries once with the correct attempt',async()=>{
 let finish!:(r:Response)=>void;const posts:RequestInit[]=[];let state='failed';
 const {router}=mount((url,options)=>{
  if(url.pathname.endsWith('/instruments'))return Response.json({...exchangeCatalog(),items:[{...rows[0],job:{job_id:'bounded-job',state,progress_percent:20}}]});
  if(url.pathname.endsWith('/retry')){posts.push(options);return new Promise<Response>(r=>finish=r);}
  if(url.pathname.endsWith('/bounded-job'))return Response.json({...jobResult(state),attempt:2,can_retry:state==='failed',error_code:state==='failed'?'source_or_storage_unavailable':null});
 },'/data?market=1'+suffix);
 fireEvent.click(await screen.findByRole('link',{name:'Failed'}));
 const retry=await screen.findByRole('button',{name:'Continue download'});await waitFor(()=>expect(retry).toBeEnabled());fireEvent.click(retry);fireEvent.click(retry);
 expect(posts).toHaveLength(1);expect(JSON.parse(String(posts[0].body))).toEqual({attempt:2});await waitFor(()=>expect(retry).toBeDisabled());
 state='queued';await act(async()=>finish(Response.json({...jobResult('queued'),attempt:3,can_cancel:true})));
 await within(screen.getByRole('region',{name:'Download request'})).findByText('Queued');expect(screen.queryByRole('button',{name:'Continue download'})).not.toBeInTheDocument();expect(router.state.location.pathname).toBe('/data');
});

it('cancels an active download through confirmation without navigating away',async()=>{
 HTMLDialogElement.prototype.showModal=function(){this.open=true;};
 HTMLDialogElement.prototype.close=function(){this.open=false;this.dispatchEvent(new Event('close'));};
 let state='running',posts=0;
 mount((url,options)=>{
  if(url.pathname.endsWith('/cancel')&&options.method==='POST'){posts++;state='cancel_requested';return Response.json({...jobResult(state),can_cancel:false});}
  if(url.pathname.endsWith('/bounded-job'))return Response.json({...jobResult(state),can_cancel:state==='running'});
 },'/data?market=1&symbol=BTCUSDT&job=bounded-job'+suffix);
 const cancel=await screen.findByRole('button',{name:'Cancel download'});await waitFor(()=>expect(cancel).toBeEnabled());fireEvent.click(cancel);expect(posts).toBe(0);
 fireEvent.click(within(screen.getByRole('dialog',{name:'Cancel this download?'})).getByRole('button',{name:'Cancel download'}));
 await screen.findByText('Cancellation requested');expect(posts).toBe(1);expect(screen.getByText(/Stopping after the current write/)).toBeVisible();
 expect(screen.getByRole('progressbar',{name:'Job progress'})).toBeVisible();expect(screen.getByRole('progressbar',{name:'Selected period coverage'})).toBeVisible();
});

it('reconciles an unknown retry without replay and retains recovery when reopening the same job',async()=>{
 let posts=0;
 const {router}=mount((url,options)=>{
  if(url.pathname.endsWith('/retry')&&options.method==='POST'){posts++;return Response.json({}, {status:503});}
  if(url.pathname.endsWith('/bounded-job'))return Response.json({...jobResult('failed'),can_retry:true});
 },'/data?market=1&symbol=BTCUSDT&job=bounded-job'+suffix);
 const retry=await screen.findByRole('button',{name:'Continue download'});await waitFor(()=>expect(retry).toBeEnabled());fireEvent.click(retry);
 await screen.findByRole('button',{name:'Check saved state'});await waitFor(()=>expect(retry).toBeDisabled());
 await act(()=>router.navigate('/data?market=1'+suffix));await act(()=>router.navigate('/data?market=1&symbol=BTCUSDT&job=bounded-job'+suffix));
 const check=await screen.findByRole('button',{name:'Check saved state'});await waitFor(()=>expect(check).toBeEnabled());expect(screen.getByRole('button',{name:'Continue download'})).toBeDisabled();
 fireEvent.click(check);await waitFor(()=>expect(screen.queryByRole('button',{name:'Check saved state'})).not.toBeInTheDocument());expect(posts).toBe(1);
});

it('keeps failed job reads local, and hides protected data when a job action is denied',async()=>{
 let denied=false,missing=true;
 const {client}=mount((url,options)=>{
  if(url.pathname.endsWith('/retry')&&options.method==='POST'){denied=true;return Response.json({}, {status:403});}
  if(url.pathname.endsWith('/bounded-job'))return missing?Response.json({}, {status:404}):Response.json({...jobResult('failed'),can_retry:true});
 },'/data?market=1&symbol=BTCUSDT&job=bounded-job'+suffix);
 await screen.findByText('The resource is unavailable or no longer exists.');expect(screen.getByRole('switch')).toBeInTheDocument();
 missing=false;await act(()=>client.refetchQueries({queryKey:['data-job','owner']}));const retry=await screen.findByRole('button',{name:'Continue download'});await waitFor(()=>expect(retry).toBeEnabled());fireEvent.click(retry);
 await waitFor(()=>expect(screen.queryByRole('switch')).not.toBeInTheDocument());expect(denied).toBe(true);expect(document.querySelectorAll('.data-catalog-table tbody tr')).toHaveLength(0);
});

it('redirects old Downloads links to the journal, preserving selected task and filters',async()=>{
 mount(url=>url.pathname.endsWith('/work-requests')?Response.json({items:[],next_cursor:null}):url.pathname.endsWith('/catalog-job')?Response.json({...jobResult(),job_id:'catalog-job',kind:'catalog_refresh',symbols:[],start_at:null,end_at:null}):undefined,'/data/ingestion?job=catalog-job&state=queued&kind=catalog_refresh');
 await screen.findByRole('region',{name:'Download log'});expect(screen.getByRole('combobox',{name:'Job state'})).toHaveValue('queued');expect(screen.getByRole('combobox',{name:'Job type'})).toHaveValue('catalog_refresh');
 await screen.findByRole('heading',{name:'Catalog refresh'});expect(screen.queryByRole('heading',{name:'New candle download'})).not.toBeInTheDocument();
});

it('loads journal cursor pages, marks retained rows noninteractive, and hides inventory on journal denial',async()=>{
 let status=200,finish!:(r:Response)=>void;
 const {client}=mount(url=>{
  if(url.pathname.endsWith('/work-requests')){
   if(status===403)return Response.json({}, {status});
   if(url.searchParams.get('state'))return new Promise<Response>(r=>finish=r);
   return Response.json({items:[{...jobResult(),job_id:url.searchParams.has('before')?'older':'newer',symbols:[url.searchParams.has('before')?'OLDUSDT':'NEWUSDT']}],next_cursor:url.searchParams.has('before')?null:{before:'2024-01-01',before_id:'newer'}});
  }
 },'/data?market=1&journal=1'+suffix);
 await screen.findByRole('link',{name:'NEWUSDT'});fireEvent.click(screen.getByRole('button',{name:'Earlier requests'}));await screen.findByRole('link',{name:'OLDUSDT'});
 fireEvent.change(screen.getByRole('combobox',{name:'Job state'}),{target:{value:'failed'}});await waitFor(()=>expect(screen.getByRole('link',{name:'OLDUSDT'})).toHaveAttribute('aria-disabled','true'));
 await act(async()=>finish(Response.json({items:[],next_cursor:null})));await screen.findByText('No requests');
 status=403;await act(()=>client.refetchQueries({queryKey:['data-jobs','owner']}));await waitFor(()=>expect(document.querySelectorAll('.data-catalog-table tbody tr')).toHaveLength(0));
});

it('does not allow a second download while inspecting an older request for an actively loading instrument',async()=>{
 mount(url=>url.pathname.endsWith('/instruments')?Response.json({...exchangeCatalog(),items:[{...rows[0],job:{job_id:'current-job',state:'running',progress_percent:10}}]}):url.pathname.endsWith('/bounded-job')?Response.json(jobResult()):undefined,'/data?market=1&symbol=BTCUSDT&job=bounded-job'+suffix);
 await screen.findByRole('region',{name:'Download request'});expect(screen.getByRole('button',{name:'Download period'})).toBeDisabled();expect(screen.getByRole('link',{name:'Open current download'})).toHaveAttribute('href','/data?market=1&symbol=BTCUSDT&job=current-job');
});

it('opens a newly submitted catalog refresh in the inspector even if an instrument was selected',async()=>{
 const catalogJob={...jobResult('queued'),kind:'catalog_refresh',symbols:[],start_at:null,end_at:null,can_cancel:true};
 const {router}=mount((url,options)=>{
  if(options.method==='POST'||url.pathname.endsWith('/bounded-job'))return Response.json(catalogJob);
  if(url.pathname.endsWith('/instruments'))return Response.json({...exchangeCatalog(),missing_market_ids:[1]});
 });
 await screen.findByRole('heading',{name:'BTCUSDT'});fireEvent.click(await screen.findByRole('button',{name:'Load catalog · Spot'}));
 await screen.findByRole('heading',{name:'Request details'});await screen.findByRole('heading',{name:'Catalog refresh'});expect(router.state.location.search).not.toContain('symbol=');expect(screen.queryByRole('switch')).not.toBeInTheDocument();
});

it('shares one progress poll across the row, journal and inspector without reloading other data or disabling cancellation',async()=>{
 vi.useFakeTimers();
 let state='running',units=600,reads=0,delay=false,finish!:(r:Response)=>void;
 const counts=new Map<string,number>();
 const current=()=>({...jobResult(state),completed_units:units,total_units:12000,can_cancel:true,started_at:'2026-10-02T00:00Z'});
 const {client}=mount(url=>{
  counts.set(url.pathname,(counts.get(url.pathname)??0)+1);
  if(url.pathname.endsWith('/preferences'))return Response.json({theme:'graphite',locale:'en',density:'compact',updated_at:'2026-10-02Z',autorefresh:{preset_key:'15s',refresh_interval_seconds:15,allowed_presets:['15s'],min_custom_interval_seconds:10,max_custom_interval_seconds:300}});
  if(url.pathname.endsWith('/instruments'))return Response.json({...exchangeCatalog(),items:[{...rows[0],job:{job_id:'bounded-job',state,progress_percent:5}}]});
  if(url.pathname.endsWith('/work-requests'))return Response.json({items:[current()],next_cursor:null});
  if(url.pathname.endsWith('/bounded-job')){reads++;return delay?new Promise<Response>(r=>finish=r):Response.json(current());}
 },'/data?market=1&symbol=BTCUSDT&job=bounded-job&journal=1'+suffix);
 await act(async()=>{await vi.advanceTimersByTimeAsync(100);});
 const cancel=screen.getByRole('button',{name:'Cancel download'}),table=document.querySelector('.data-catalog-table'),inspector=screen.getByRole('complementary',{name:'Instrument settings'});
 expect(cancel).toBeEnabled();cancel.focus();inspector.scrollTop=120;
 const initialReads=reads,initialCounts=new Map(counts);
 for(let i=0;i<12;i++){
  state=i%2?'running':'queued';units+=60;
  await act(async()=>{await vi.advanceTimersByTimeAsync(2000);});
  expect(reads).toBe(initialReads+i+1);
  expect(document.querySelector('.data-job-state.queued')).not.toBeInTheDocument();
  expect(screen.queryByText('Updating…')).not.toBeInTheDocument();
  expect(screen.getAllByText('Downloading')).toHaveLength(3);
  expect(cancel).toBeEnabled();expect(cancel).toHaveFocus();
 }
 for(const [path,count] of initialCounts)if(!path.endsWith('/bounded-job'))expect(counts.get(path),path).toBe(count);
 expect(document.querySelector('.data-catalog-table')).toBe(table);expect(inspector.scrollTop).toBe(120);
 expect(document.querySelectorAll('.data-download-eta')).toHaveLength(3);
 const labels=[...document.querySelectorAll('.data-download-eta')].map(el=>el.textContent);
 expect(new Set(labels).size).toBe(1);expect(labels[0]).toMatch(/^≈ /);
 delay=true;await act(async()=>{await vi.advanceTimersByTimeAsync(2000);});expect(finish).toBeDefined();
 expect(cancel).toBeEnabled();expect(cancel).toHaveFocus();expect(screen.queryByText('Updating…')).not.toBeInTheDocument();
 delay=false;await act(async()=>finish(Response.json(current())));
 // Completion refreshes the dependent snapshots once, not on each checkpoint.
 const before=new Map(counts);state='succeeded';units=12000;
 await act(async()=>{await vi.advanceTimersByTimeAsync(2000);});
 expect(counts.get('/api/market-data/workspace/instruments')).toBe((before.get('/api/market-data/workspace/instruments')??0)+1);
 expect(counts.get('/api/market-data/work-requests')).toBe((before.get('/api/market-data/work-requests')??0)+1);
 client.clear();
});

it('keeps read failures visible, stops the ETA, and locks commands until a successful explicit refresh',async()=>{
 let fail=false;
 const {client}=mount(url=>url.pathname.endsWith('/bounded-job')?fail?Response.json({}, {status:503}):Response.json({...jobResult('running'),total_units:12000,can_cancel:true}):undefined,'/data?market=1&symbol=BTCUSDT&job=bounded-job'+suffix);
 const cancel=await screen.findByRole('button',{name:'Cancel download'});expect(cancel).toBeEnabled();
 fail=true;await act(()=>client.refetchQueries({queryKey:['data-job','owner','bounded-job']}));
 expect(cancel).toBeDisabled();expect(screen.getByText('No connection')).toBeVisible();
 expect(screen.getByText('The service could not be read. Try refreshing.')).toBeVisible();
 fail=false;fireEvent.click(screen.getByRole('button',{name:'Refresh request'}));await waitFor(()=>expect(cancel).toBeEnabled());
});

it('pauses and resumes the same job with a version fence and preserves the manual range',async()=>{
 let saved={...jobResult('running'),completed_units:2,total_units:100,can_pause:true,can_resume:false,can_cancel:true,control_version:0};
 const writes:{path:string;body:unknown}[]=[];
 mount((url,options)=>{
  if(options.method==='POST'&&/\/(pause|resume)$/.test(url.pathname)){
   writes.push({path:url.pathname,body:JSON.parse(String(options.body))});
   saved={...saved,state:url.pathname.endsWith('/pause')?'paused':'queued',can_pause:url.pathname.endsWith('/resume'),can_resume:url.pathname.endsWith('/pause'),control_version:saved.control_version+1};
   return Response.json(saved);
  }
  if(url.pathname.endsWith('/bounded-job'))return Response.json(saved);
 },'/data?market=1&symbol=BTCUSDT&job=bounded-job'+suffix);
 const pause=await screen.findByRole('button',{name:'Pause'});
 fireEvent.click(pause);
 await screen.findByRole('button',{name:'Resume'});
 expect(screen.getByText('Progress is saved. Resume when ready.')).toBeVisible();
 expect(screen.getByRole('button',{name:'Download period'})).toBeDisabled();
 expect(screen.getByText('2%')).toBeVisible();
 fireEvent.click(screen.getByRole('button',{name:'Resume'}));
 await screen.findByRole('button',{name:'Pause'});
 expect(writes).toEqual([{path:'/api/market-data/work-requests/bounded-job/pause',body:{control_version:0}},{path:'/api/market-data/work-requests/bounded-job/resume',body:{control_version:1}}]);
 expect(screen.getByText('2%')).toBeVisible();
});

it('automatically reconnects only the job read after a temporary browser connection failure',async()=>{
 let fail=false,jobReads=0,catalogReads=0,posts=0;
 const {client}=mount((url,options)=>{
  if(options.method==='POST')posts++;
  if(url.pathname.endsWith('/instruments'))catalogReads++;
  if(url.pathname.endsWith('/bounded-job')){jobReads++;return fail?Response.json({}, {status:503}):Response.json({...jobResult('running'),can_pause:true,can_cancel:true});}
 },'/data?market=1&symbol=BTCUSDT&job=bounded-job'+suffix);
 await screen.findByRole('button',{name:'Pause'});
 fail=true;await act(()=>client.refetchQueries({queryKey:['data-job','owner','bounded-job']}));
 expect(screen.getByRole('button',{name:'Pause'})).toBeDisabled();
 expect(screen.getByText('No connection')).toBeVisible();
 const before=jobReads,oldCatalog=catalogReads;fail=false;
 await waitFor(()=>expect(screen.getByRole('button',{name:'Pause'})).toBeEnabled(),{timeout:4000});
 expect(jobReads).toBeGreaterThan(before);expect(catalogReads).toBe(oldCatalog);expect(posts).toBe(0);
});

it('does not let a delayed pre-pause GET overwrite the acknowledged command',async()=>{
 let delay=false,finish!:(r:Response)=>void;
 let saved={...jobResult('running'),completed_units:2,total_units:100,can_pause:true,can_resume:false,can_cancel:true,control_version:0};
 const {client}=mount((url,options)=>{
  if(url.pathname.endsWith('/pause')&&options.method==='POST'){
   saved={...saved,state:'paused',can_pause:false,can_resume:true,control_version:1};
   return Response.json(saved);
  }
  if(url.pathname.endsWith('/bounded-job'))return delay?new Promise<Response>(r=>finish=r):Response.json(saved);
 },'/data?market=1&symbol=BTCUSDT&job=bounded-job'+suffix);
 await screen.findByRole('button',{name:'Pause'});
 delay=true;void client.refetchQueries({queryKey:['data-job','owner','bounded-job']});
 await waitFor(()=>expect(finish).toBeDefined());
 fireEvent.click(screen.getByRole('button',{name:'Pause'}));
 await screen.findByRole('button',{name:'Resume'});
 delay=false;await act(async()=>finish(Response.json({...saved,state:'running',control_version:0,can_pause:true,can_resume:false})));
 expect(screen.getByRole('button',{name:'Resume'})).toBeVisible();
 expect(client.getQueryData(['data-job','owner','bounded-job'])).toMatchObject({state:'paused',control_version:1});
});

it('shares a command fence and unknown outcome between row and inspector controls',async()=>{
 let posts=0,finish!:(r:Response)=>void;
 const saved={...jobResult('running'),can_pause:true,can_cancel:true,completed_units:2,total_units:100};
 mount((url,options)=>{
  if(url.pathname.endsWith('/instruments'))return Response.json({...exchangeCatalog(),items:[{...rows[0],job:{job_id:'bounded-job',state:'running',progress_percent:2}}]});
  if(url.pathname.endsWith('/bounded-job'))return Response.json(saved);
  if(url.pathname.endsWith('/pause')&&options.method==='POST'){posts++;return new Promise<Response>(r=>finish=r);}
 },'/data?market=1&symbol=BTCUSDT&job=bounded-job'+suffix);
 const row=await screen.findByRole('button',{name:'Pause · BTCUSDT'});
 const inspector=await screen.findByRole('button',{name:'Pause'});
 fireEvent.click(row);fireEvent.click(inspector);expect(posts).toBe(1);
 await act(async()=>finish(Response.json({}, {status:503})));
 await screen.findByRole('button',{name:'Check saved state · BTCUSDT'});
 expect(row).toBeDisabled();expect(inspector).toBeDisabled();
 fireEvent.click(screen.getByRole('button',{name:'Check saved state'}));
 await waitFor(()=>expect(row).toBeEnabled());expect(posts).toBe(1);
});

it('renders temporary memory failure, saved progress and paginated events without exposing technical details by default',async()=>{
 const reads:string[]=[];
 const saved={...jobResult('retry_wait'),completed_units:60,total_units:180,rows_written:60,error_code:'storage_memory_pressure',error_phase:'fill_window',next_retry_at:'2026-10-04T12:00:00Z',can_pause:true,can_cancel:true};
 mount(url=>{
  if(url.pathname.endsWith('/bounded-job'))return Response.json(saved);
  if(url.pathname.endsWith('/events')){reads.push(url.search);return Response.json({items:[{event_id:url.searchParams.has('before')?1:2,event_type:url.searchParams.has('before')?'started':'retry_scheduled',occurred_at:'2026-10-04T11:59:45Z',state:'retry_wait',attempt:1,completed_units:60,total_units:180,rows_written:60,error_code:'storage_memory_pressure',error_phase:'fill_window',next_retry_at:'2026-10-04T12:00:00Z'}],next_cursor:url.searchParams.has('before')?null:2,retention_days:30,max_events:512});}
 },'/data?market=1&symbol=BTCUSDT&job=bounded-job'+suffix);
 await screen.findByText(/Storage is waiting for memory/);expect(reads).toHaveLength(0);
 const events=screen.getByText('Download events').closest('details')!;events.open=true;fireEvent(events,new Event('toggle'));
 await screen.findByText('Temporary failure; retry scheduled');
 expect(screen.getByText('storage_memory_pressure · fill_window',{selector:'code'})).not.toBeVisible();
 fireEvent.click(screen.getByRole('button',{name:'Earlier events'}));
 await screen.findByText('Download started');expect(reads.some(s=>s.includes('before=2'))).toBe(true);
});
