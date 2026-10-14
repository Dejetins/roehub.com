import '@testing-library/jest-dom/vitest';
import {afterEach,expect,it,vi} from 'vitest';
import {act,cleanup,fireEvent,render,screen,waitFor} from '@testing-library/react';
import {QueryClientProvider} from '@tanstack/react-query';
import {I18nextProvider} from 'react-i18next';
import {SessionDevices} from './session-devices';
import {createQueryClient} from './query-client';
import {createI18n} from './i18n';
const devices={items:[{device_id:'current',label:'Chrome · macOS',last_seen_at:'2026-10-10T10:00:00Z',expires_at:'2027-10-10T10:00:00Z',active:true,is_current:true},{device_id:'other',label:'Firefox · Windows',last_seen_at:'2026-10-09T10:00:00Z',expires_at:'2027-10-10T10:00:00Z',active:true,is_current:false}],next_cursor:null,legacy_active_count:12};
function mount(){const client=createQueryClient();render(<I18nextProvider i18n={createI18n('en')}><QueryClientProvider client={client}><SessionDevices subject="owner"/></QueryClientProvider></I18nextProvider>);return client;}
afterEach(()=>{cleanup();vi.restoreAllMocks();vi.unstubAllGlobals();});
it('shows devices and one legacy group, revokes once and reconciles without hiding rows',async()=>{
 let finish!:(r:Response)=>void,writes=0,ended=false;
 vi.stubGlobal('fetch',vi.fn((_url,options)=>options.method==='POST'?(writes++,new Promise<Response>(r=>{finish=r;})):Promise.resolve(Response.json({...devices,items:devices.items.map(d=>d.device_id==='other'?{...d,active:!ended}:d)}))));
 mount();await screen.findByText('This device');expect(screen.getByText('Earlier sign-ins · 12')).toBeVisible();
 const button=screen.getByRole('button',{name:'End session: Firefox · Windows'});fireEvent.click(button);fireEvent.click(button);expect(writes).toBe(1);expect(button).toBeDisabled();expect(screen.getByText('Chrome · macOS')).toBeVisible();
 ended=true;await act(async()=>finish(Response.json({revoked:true,current:false})));await screen.findByText('Ended');expect(screen.queryByRole('button',{name:'End session: Firefox · Windows'})).not.toBeInTheDocument();
});
it('hides private rows immediately on a denied mutation',async()=>{
 vi.stubGlobal('fetch',vi.fn((_url,options)=>Promise.resolve(options.method==='POST'?Response.json({error:{code:'denied'}},{status:403}):Response.json(devices))));
 mount();fireEvent.click(await screen.findByRole('button',{name:'End session: Firefox · Windows'}));
 await screen.findByText('Access denied.');expect(screen.queryByText('Chrome · macOS')).not.toBeInTheDocument();expect(screen.queryByText('Earlier sign-ins · 12')).not.toBeInTheDocument();
});
it('keeps an unknown command outcome locked until an explicit successful read',async()=>{
 let writes=0;
 vi.stubGlobal('fetch',vi.fn((_url,options)=>options.method==='POST'?(writes++,Promise.reject(new TypeError('offline'))):Promise.resolve(Response.json(devices))));
 mount();fireEvent.click(await screen.findByRole('button',{name:'End earlier sign-ins'}));
 await screen.findByText('Refresh the list to check the result before trying again.');expect(screen.getByRole('button',{name:'End earlier sign-ins'})).toBeDisabled();expect(writes).toBe(1);
 fireEvent.click(screen.getByRole('button',{name:'Refresh devices'}));await waitFor(()=>expect(screen.getByRole('button',{name:'End earlier sign-ins'})).not.toBeDisabled());expect(writes).toBe(1);
});
it('retains read failures without enabling commands and removes rows on read denial',async()=>{
 let status=200;
 vi.stubGlobal('fetch',vi.fn(()=>Promise.resolve(status===200?Response.json(devices):Response.json({error:{code:'failed'}},{status}))));
 const client=mount();await screen.findByText('Chrome · macOS');status=503;await act(()=>client.invalidateQueries({queryKey:['account','owner','devices']}));
 await screen.findByText('Showing previous data');expect(screen.getByRole('button',{name:'End earlier sign-ins'})).toBeDisabled();status=401;
 fireEvent.click(screen.getByRole('button',{name:'Refresh devices'}));await waitFor(()=>expect(screen.queryByText('Chrome · macOS')).not.toBeInTheDocument());
});
