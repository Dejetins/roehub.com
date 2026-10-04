import '@testing-library/jest-dom/vitest';
import {afterEach,expect,it,vi} from 'vitest';
import {act,cleanup,fireEvent,render,screen,waitFor} from '@testing-library/react';
import {createMemoryRouter,RouterProvider} from 'react-router';
import {I18nextProvider} from 'react-i18next';
import {createI18n} from './i18n';
import {DirtyGuard} from './workspace-ui';
import {useWorkSubmission} from './market-data-ui';
const job={job_id:'test-request',kind:'catalog_refresh',market_id:1,symbols:[],timeframe:'1m',start_at:null,end_at:null,state:'queued',attempt:1,completed_units:0,total_units:1,rows_read:0,rows_written:0,error_code:null,created_at:'2026-10-03T00:00Z',updated_at:'2026-10-03T00:00Z',started_at:null,finished_at:null,can_cancel:true,can_retry:false,attempts:[]};
afterEach(()=>{cleanup();vi.restoreAllMocks();vi.unstubAllGlobals();});
it('persists recovery key without a dirty prompt, blocks duplicates and recovers after reload',async()=>{
 let reject!:(e:Error)=>void;const saved=vi.fn();let writes=0;
 vi.stubGlobal('fetch',vi.fn((_url,options)=>{if(options.method==='POST'){writes++;return new Promise<Response>((_resolve,r)=>{reject=r;});}return Promise.resolve(Response.json(job));}));
 const confirm=vi.spyOn(window,'confirm').mockReturnValue(false);
 function Form(){const c=useWorkSubmission(saved);return <><DirtyGuard dirty/><button onClick={()=>void c.submit({kind:'catalog_refresh',market_id:1})}>Submit</button><button disabled={!c.unresolved} onClick={()=>void c.reconcile()}>Recover</button></>;}
 function mount(url:string){const router=createMemoryRouter([{path:'*',element:<Form/>}],{initialEntries:[url]});render(<I18nextProvider i18n={createI18n('en')}><RouterProvider router={router}/></I18nextProvider>);return router;}
 const router=mount('/data?market=1');fireEvent.click(screen.getByText('Submit'));fireEvent.click(screen.getByText('Submit'));
 expect(writes).toBe(1);expect(confirm).not.toHaveBeenCalled();expect(new URLSearchParams(router.state.location.search).get('requestkey')).toMatch(/^[0-9a-f-]{36}$/);
 await act(async()=>reject(new TypeError('offline')));await waitFor(()=>expect(screen.getByText('Recover')).not.toBeDisabled());
 const url=router.state.location.pathname+router.state.location.search;cleanup();mount(url);
 fireEvent.click(screen.getByText('Submit'));expect(writes).toBe(1);fireEvent.click(screen.getByText('Recover'));await waitFor(()=>expect(saved).toHaveBeenCalledWith(job));expect(writes).toBe(1);
});
