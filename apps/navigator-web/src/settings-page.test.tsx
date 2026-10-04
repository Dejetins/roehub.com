import '@testing-library/jest-dom/vitest';
import {afterEach,expect,it,vi} from 'vitest';
import {act,cleanup,fireEvent,render,screen,waitFor} from '@testing-library/react';
import {createMemoryRouter,RouterProvider} from 'react-router';
import {QueryClientProvider} from '@tanstack/react-query';
import {I18nextProvider} from 'react-i18next';
import {SettingsPage} from './settings-page';
import {createQueryClient} from './query-client';
import {createI18n} from './i18n';
import {clientRoute} from './client-routes';
const original={user_id:'owner',username:'QA owner',email:null,timezone:'UTC',locale:'en',telegram_discord:null,subscription_status:'free',updated_at:'2026-10-03T00:00:00Z'};
function mount(path='/settings/profile'){const client=createQueryClient();const router=createMemoryRouter([{path:'*',element:<SettingsPage subject="owner"/>}],{initialEntries:[path]});render(<I18nextProvider i18n={createI18n('en')}><QueryClientProvider client={client}><RouterProvider router={router}/></QueryClientProvider></I18nextProvider>);return {client,router};}
afterEach(()=>{cleanup();vi.restoreAllMocks();vi.unstubAllGlobals();});
it('saves once while pending and commits server-normalized values',async()=>{
 let finish!:(r:Response)=>void;let writes=0;
 vi.stubGlobal('fetch',vi.fn((_url,options)=>{if(options.method==='PUT'){writes++;return new Promise<Response>(r=>{finish=r;});}return Promise.resolve(Response.json(original));}));
 mount();const name=await screen.findByLabelText('Display name');fireEvent.change(name,{target:{value:'New name'}});
 const form=name.closest('form')!;fireEvent.submit(form);fireEvent.submit(form);
 expect(writes).toBe(1);expect(name).toBeDisabled();
 await act(async()=>finish(Response.json({...original,username:'New name'})));
 await waitFor(()=>expect(screen.getByRole('button',{name:'Save changes'})).toBeDisabled());
 expect(name).toHaveValue('New name');expect(screen.getByText('Saved')).toBeVisible();
});
it('validates by field without sending a command, and protects dirty navigation',async()=>{
 const fetch=vi.fn(()=>Promise.resolve(Response.json(original)));vi.stubGlobal('fetch',fetch);
 const {router}=mount();const email=await screen.findByLabelText('Email');fireEvent.change(email,{target:{value:'bad'}});fireEvent.submit(email.closest('form')!);
 await waitFor(()=>expect(email).toHaveFocus());expect(email).toHaveAttribute('aria-invalid','true');expect(fetch).toHaveBeenCalledTimes(1);
 vi.spyOn(window,'confirm').mockReturnValue(false);await act(()=>router.navigate('/settings/preferences'));expect(router.state.location.pathname).toBe('/settings/profile');expect(email).toHaveValue('bad');
 const unload=new Event('beforeunload',{cancelable:true});window.dispatchEvent(unload);expect(unload.defaultPrevented).toBe(true);
 fireEvent.click(screen.getByRole('button',{name:'Cancel'}));expect(email).toHaveValue('');
});
it('retains coherent data on a failed read, then hides it immediately on denial',async()=>{
 let status=200;vi.stubGlobal('fetch',vi.fn(()=>Promise.resolve(status===200?Response.json(original):Response.json({error:{code:'read_failed'}},{status}))));
 const {client}=mount();const name=await screen.findByLabelText('Display name');expect(name).toHaveValue('QA owner');status=503;
 await act(()=>client.invalidateQueries({queryKey:['account','owner','profile']}));await screen.findByText('Showing previous data');expect(name).toHaveValue('QA owner');expect(name).toBeDisabled();
 status=403;fireEvent.click(screen.getByRole('button',{name:'Try again'}));await screen.findByText('Access denied.');expect(screen.queryByLabelText('Display name')).not.toBeInTheDocument();
});
it('freezes an unknown save outcome until an explicit read reconciles it',async()=>{
 let writes=0;vi.stubGlobal('fetch',vi.fn((_url,options)=>{if(options.method==='PUT'){writes++;return Promise.reject(new TypeError('offline'));}return Promise.resolve(Response.json(original));}));
 mount();const name=await screen.findByLabelText('Display name');fireEvent.change(name,{target:{value:'Draft'}});fireEvent.submit(name.closest('form')!);
 await screen.findByText('Result unknown');expect(screen.getByRole('button',{name:'Save changes'})).toBeDisabled();expect(writes).toBe(1);expect(name).toHaveValue('Draft');
 fireEvent.click(screen.getByRole('button',{name:'Check saved state'}));await waitFor(()=>expect(name).not.toBeDisabled());expect(writes).toBe(1);expect(name).toHaveValue('Draft');
});
it('hides form data after a mutation revokes access',async()=>{
 vi.stubGlobal('fetch',vi.fn((_url,options)=>Promise.resolve(options.method==='PUT'?Response.json({error:{code:'denied'}},{status:403}):Response.json(original))));
 mount();const name=await screen.findByLabelText('Display name');fireEvent.change(name,{target:{value:'Draft'}});fireEvent.submit(name.closest('form')!);
 await waitFor(()=>expect(screen.queryByLabelText('Display name')).not.toBeInTheDocument());
});
it('keeps security tabs keyboard-operable without expansion or stale rows from the other tab',async()=>{
 vi.stubGlobal('IntersectionObserver',class {observe(){} disconnect(){}});
 let finish!:(r:Response)=>void;
 vi.stubGlobal('fetch',vi.fn(url=>String(url).includes('audit-events')?new Promise<Response>(resolve=>{finish=resolve;}):Promise.resolve(Response.json({items:[{session_id:'view-only-id',created_at:'2026-10-03T00:00:00Z',last_seen_at:'2026-10-03T00:00:00Z',idle_expires_at:'2050-01-01T00:00:00Z',absolute_expires_at:'2050-01-01T00:00:00Z',revoked_at:null,is_current:true}],next_cursor:null}))));
 mount('/settings/security');await screen.findByText('Active');
 expect(screen.queryByRole('button',{name:'Expand tables'})).not.toBeInTheDocument();expect(screen.queryByRole('button',{name:'Refresh'})).not.toBeInTheDocument();
 fireEvent.keyDown(screen.getByRole('tab',{name:'Sessions'}),{key:'ArrowRight'});
 expect(screen.getByRole('tab',{name:'Activity'})).toHaveFocus();expect(screen.queryByText('Active')).not.toBeInTheDocument();
 await act(async()=>finish(Response.json({items:[{event_id:'event',created_at:'2026-10-03T00:00:00Z',event_type:'profile_updated',summary:'Profile updated'}],next_cursor:null})));
 await screen.findByText('Profile updated');
 fireEvent.keyDown(screen.getByRole('tab',{name:'Activity'}),{key:'Home'});await screen.findByText('Active');expect(screen.queryByText('Profile updated')).not.toBeInTheDocument();
});
it('accepts only opted-in canonical work page routes',()=>{
 const bootstrap={subject:'owner',locale:'en' as const,client_routes:['/settings','/connections','/data','/monitoring']};
 for(const path of ['/settings/profile','/settings/preferences','/settings/notifications','/settings/security','/connections','/data','/data/ingestion','/monitoring','/monitoring/api'])expect(clientRoute(path,bootstrap)).toBe(true);
 for(const path of ['/settings/arbitrary','/data/unknown','https://evil.invalid/data','/admin'])expect(clientRoute(path,bootstrap)).toBe(false);
 expect(clientRoute('/settings/profile',{...bootstrap,client_routes:[]})).toBe(false);
});
