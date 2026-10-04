import '@testing-library/jest-dom/vitest';
import {afterEach,expect,it,vi} from 'vitest';
import {act,cleanup,fireEvent,render,screen,waitFor} from '@testing-library/react';
import {createMemoryRouter,RouterProvider} from 'react-router';
import {QueryClientProvider} from '@tanstack/react-query';
import {I18nextProvider} from 'react-i18next';
import {ConnectionsPage} from './connections-page';
import {createQueryClient} from './query-client';
import {createI18n} from './i18n';
import {connectionSchema} from './connections-api';
const row={connection_id:'00000000-0000-4000-8000-000000000001',exchange_name:'binance',market_type:'spot',environment:'testnet',label:'UI test account',requested_permissions:'trade',effective_permissions:'trade',effective_capability:'trading',connection_readiness:'ready_for_trading',connection_readiness_reason:'trading_policy_ok',status:'active',validation_status:'valid_trade_enabled',validation_reason:null,last_validated_at:'2026-10-02T00:00Z',created_at:'2026-10-02T00:00Z',updated_at:'2026-10-02T00:00Z',used_by_strategies_count:0,active_strategy_bindings_count:0};
function mount(handler:(url:URL,init:RequestInit)=>Response|Promise<Response>|undefined){
 vi.stubGlobal('IntersectionObserver',class{observe(){}disconnect(){}});
 vi.stubGlobal('fetch',vi.fn((url:URL,init:RequestInit)=>handler(url,init)??Promise.resolve(Response.json(url.pathname.endsWith('/markets')?{items:[]}:url.pathname.endsWith('/bindings')?{items:[],next_cursor:null}:url.pathname.endsWith('/exchange-connections')?{items:[],next_cursor:null}:row))));
 const client=createQueryClient(),router=createMemoryRouter([{path:'*',element:<ConnectionsPage subject="owner"/>}],{initialEntries:['/connections?new=1']});
 render(<I18nextProvider i18n={createI18n('en')}><QueryClientProvider client={client}><RouterProvider router={router}/></QueryClientProvider></I18nextProvider>);return {client,router};
}
afterEach(()=>{cleanup();vi.restoreAllMocks();vi.unstubAllGlobals();document.cookie='roehub_csrf=; max-age=0';});
function fill(){fireEvent.change(screen.getByLabelText('API key'),{target:{value:'UI_TEST_NOT_A_REAL_KEY'}});fireEvent.change(screen.getByLabelText('API secret'),{target:{value:'UI_TEST_NOT_A_REAL_SECRET'}});return screen.getByLabelText('API key').closest('form')!;}
it('submits once, strips response key fragments and clears secret inputs',async()=>{
 let writes=0,finish!:(r:Response)=>void;
 const {client,router}=mount((_url,init)=>init.method==='POST'?(writes++,new Promise<Response>(r=>{finish=r;})):undefined);
 const form=fill();fireEvent.submit(form);fireEvent.submit(form);expect(writes).toBe(1);expect(screen.getByLabelText('API key')).toBeDisabled();
 await act(async()=>finish(Response.json({...row,api_key_last4:'NEVER-CACHE',api_secret:'NEVER-CACHE'})));
 await screen.findByRole('heading',{name:'UI test account'});expect(router.state.location.search).not.toContain('UI_TEST');expect(JSON.stringify(client.getQueryCache().getAll().map(q=>q.state.data))).not.toContain('NEVER-CACHE');
 expect(connectionSchema.parse({...row,api_key:'NEVER-CACHE'})).not.toHaveProperty('api_key');
});
it('requires explicit step-up, clears credentials on rejection and never replays the connection command',async()=>{
 let writes=0;const authCalls:string[]=[];
 Object.defineProperty(navigator,'credentials',{configurable:true,value:{get:vi.fn(async()=>({id:'test-passkey',rawId:new Uint8Array([1]).buffer,type:'public-key',authenticatorAttachment:'platform',getClientExtensionResults:()=>({}),response:{clientDataJSON:new Uint8Array([2]).buffer,authenticatorData:new Uint8Array([3]).buffer,signature:new Uint8Array([4]).buffer,userHandle:null}}))}});
 document.cookie='roehub_csrf=UI_TEST_CSRF';
 mount((url,init)=>{
  if(url.pathname.includes('/recent-auth/')){authCalls.push(url.pathname);expect((init.headers as Record<string,string>)['X-CSRF-Token']).toBe('UI_TEST_CSRF');return Response.json(url.pathname.endsWith('/options')?{challenge_id:'test-challenge',publicKey:{challenge:'AQ',allowCredentials:[{id:'Ag',type:'public-key'}]}}:{authenticated:true});}
  if(init.method==='POST'){writes++;return Response.json({error:{code:'recent_auth_required'}},{status:403});}
 });
 fireEvent.submit(fill());await screen.findByRole('button',{name:'Confirm with passkey'});expect(screen.getByLabelText('API key')).toHaveValue('');expect(screen.getByLabelText('API secret')).toHaveValue('');
 const login=new URL(screen.getByRole('link',{name:'Sign in again'}).getAttribute('href')!,'http://localhost');expect(login.pathname).toBe('/logout');expect(login.searchParams.get('next')).toBe('/login?next=%2Fconnections%3Fnew%3D1');
 fireEvent.click(screen.getByRole('button',{name:'Confirm with passkey'}));await waitFor(()=>expect(screen.queryByRole('button',{name:'Confirm with passkey'})).not.toBeInTheDocument());expect(authCalls).toHaveLength(2);expect(writes).toBe(1);
});
