import { afterEach, expect, it, vi } from 'vitest';
import { act, cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react';
import { onlineManager, QueryClientProvider, QueryObserver } from '@tanstack/react-query';
import { MemoryRouter } from 'react-router';
import { I18nextProvider } from 'react-i18next';
import { ApiError } from './api';
import { App } from './app';
import { createQueryClient } from './query-client';
import { createI18n } from './i18n';

vi.mock('./backtests-page',()=>({BacktestsPage:()=> <input aria-label="Private draft"/>}));

afterEach(() => { cleanup(); vi.unstubAllGlobals(); });
function renderApp() {
  const client=createQueryClient();
  render(<I18nextProvider i18n={createI18n('en')}><QueryClientProvider client={client}>
    <MemoryRouter initialEntries={['/backtests/job?variant=a%2Fb']}><App bootstrap={{ locale: 'en', subject: 'actor' }} /></MemoryRouter>
  </QueryClientProvider></I18nextProvider>);
  return client;
}
it('shows re-authentication only for 401, preserving the deep link', async () => {
  vi.stubGlobal('fetch', vi.fn().mockResolvedValue(new Response('{}', { status: 401 })));
  renderApp();
  const link = await screen.findByRole('link', { name: 'Sign in' });
  expect(link.getAttribute('href')).toBe('/login?next=%2Fbacktests%2Fjob%3Fvariant%3Da%252Fb');
});
it('identity outage does not offer logout or expose protected content', async () => {
  vi.stubGlobal('fetch', vi.fn().mockResolvedValue(new Response('{}', { status: 503 })));
  renderApp();
  expect((await screen.findByRole('alert')).textContent).toContain('unavailable');
  expect(screen.queryByRole('link', { name: 'Sign in' })).toBeNull();
  expect(screen.queryByRole('navigation')).toBeNull();
});
it('a changed subject prevents reuse of the original account view', async () => {
  vi.stubGlobal('fetch', vi.fn().mockResolvedValue(new Response('{"user_id":"other","paid_level":"free"}')));
  renderApp();
  expect((await screen.findByRole('alert')).textContent).toContain('account has changed');
  expect(screen.queryByRole('navigation')).toBeNull();
});

it.each([401,403])('never retries session denial %s',async(status)=>{
 const {sessionReadInterval}=await import('./session-recovery');
 const {ApiError}=await import('./api');
 expect(sessionReadInterval(new ApiError(status===401?'unauthenticated':'forbidden',status,'failed'))).toBe(false);
});
it('bounds session read recovery and honors a longer Retry-After',async()=>{
 const {sessionReadInterval}=await import('./session-recovery');
 const {ApiError}=await import('./api');
 expect(sessionReadInterval(new ApiError('unavailable',503,'failed'),1)).toBe(2000);
 expect(sessionReadInterval(new ApiError('unavailable',503,'failed'),20)).toBe(30000);
 expect(sessionReadInterval(new ApiError('rate-limited',429,'failed',90),2)).toBe(90000);
 expect(sessionReadInterval(new ApiError('unavailable',200,'failed'))).toBe(false);
});


it('quarantines the mounted workspace and command fences during a temporary session read, then restores the same draft',async()=>{
 let status=200;
 const fetcher=vi.fn().mockImplementation(()=>Promise.resolve(status===200?Response.json({user_id:'actor',paid_level:'free'}):Response.json({}, {status})));
 vi.stubGlobal('fetch',fetcher);
 const client=renderApp();
 const read=new QueryObserver(client,{queryKey:['proof-read'],retry:false,queryFn:async()=>{if(status!==200)throw new ApiError('unavailable',503,'failed');return 'fresh';}});
 const stopRead=read.subscribe(()=>{});
 const draft=await screen.findByRole('textbox',{name:'Private draft'});
 fireEvent.input(draft,{target:{value:'keep this draft'}});
 client.setQueryData(['data-job-action','actor','job'],{pending:false,unknown:true});
 status=503;await act(async()=>{await read.refetch();await client.refetchQueries({queryKey:['session']});});
 expect((await screen.findByRole('alert')).textContent).toContain('automatically');
 expect(screen.queryByRole('textbox',{name:'Private draft'})).toBeNull();
 expect(client.getQueryData(['data-job-action','actor','job'])).toEqual({pending:false,unknown:true});
 expect(draft.isConnected).toBe(true);
 status=200;
 // Allow the real configured read timer to recover; no write is sent.
 await waitFor(()=>expect(screen.queryByRole('textbox',{name:'Private draft'})).toBe(draft),{timeout:4000});
 expect((draft as HTMLInputElement).value).toBe('keep this draft');
 await waitFor(()=>expect(client.getQueryState(['proof-read'])?.error).toBeNull());
 stopRead();
 status=403;await act(()=>client.refetchQueries({queryKey:['session']}));
 await waitFor(()=>expect(client.getQueryData(['data-job-action','actor','job'])).toBeUndefined());
 expect(draft.isConnected).toBe(false);
 const reads=fetcher.mock.calls.length;
 await act(async()=>{onlineManager.setOnline(false);onlineManager.setOnline(true);await new Promise(resolve=>setTimeout(resolve,20));});
 expect(fetcher.mock.calls.length).toBe(reads);
});
