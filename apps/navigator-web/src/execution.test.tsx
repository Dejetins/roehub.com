import '@testing-library/jest-dom/vitest';
import { afterEach, expect, it, vi } from 'vitest';
import { cleanup, render, screen, waitFor, act } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { QueryClientProvider, onlineManager } from '@tanstack/react-query';
import { I18nextProvider } from 'react-i18next';
import { MemoryRouter } from 'react-router';
import { createQueryClient } from './query-client';
import { createI18n } from './i18n';
import { JobEntry, reconcileJob, nextJobRead } from './execution';
import { ApiError } from './api';
import { jobSchema, type Job } from './library-api';
const id = '10000000-0000-4000-8000-000000000001';
const date = new Date().toISOString();
const job: Job = { job_id: id, state: 'running', created_at: date, updated_at: date, generated_at: date,
 refresh_status: 'poll', next_allowed_refresh_at: date, retry_after_seconds: 0, cancel_requested_at: null,
 progress: { percent: 50, processed_units: 5, total_units: 10, updated_at: date, pipeline_stage: 'simulation' },
 request: { coordinates: { exchange:'binance', market_type:'spot', symbol:'BTCUSDT' }, timeframe:'15m', time_range:{start:date,end:date},risk_mode:'none' } };
const reply = (body: unknown, status = 200) => new Response(JSON.stringify(body), {status});
function mount(handler?: (url:URL, init:RequestInit)=>Response|Promise<Response>|undefined) {
 const fetch=vi.fn((url:URL,init:RequestInit)=>Promise.resolve(handler?.(url,init)??reply(url.pathname.includes('current-user')?{user_id:'actor',paid_level:'free'}:job)));
 vi.stubGlobal('fetch',fetch);
 HTMLDialogElement.prototype.showModal=function(){this.open=true;};HTMLDialogElement.prototype.close=function(){this.open=false;this.dispatchEvent(new Event('close'));};
 const client=createQueryClient();
 render(<I18nextProvider i18n={createI18n('en')}><QueryClientProvider client={client}><MemoryRouter><JobEntry id={id} subject="actor" now={Date.now()+100} backLink="/backtests" /></MemoryRouter></QueryClientProvider></I18nextProvider>);
 return {client,fetch};
}
afterEach(()=>{cleanup();vi.unstubAllGlobals();});
it('terminal beats late running, and older responses cannot discard result metadata',()=>{
 const result={...job,state:'succeeded' as const,terminal_summary:{top_variants_count:4}};
 expect(reconcileJob(result,job)).toBe(result);
 expect(reconcileJob(result,{...result,updated_at:'2020-01-01T00:00:00Z',terminal_summary:{top_variants_count:0}})).toBe(result);
 expect(reconcileJob(job,result)).toEqual(result);
});
it('uses server read deadlines, error delay, and stops terminal or restricted polling',()=>{
 expect(nextJobRead({...job,retry_after_seconds:12},1000,null,0)).toBeGreaterThanOrEqual(13000);
 expect(nextJobRead(undefined,0,new ApiError('rate-limited',429,'failed',90),1000)).toBe(91000);
 expect(nextJobRead({...job,state:'failed'},0,null,0)).toBe(Infinity);
 expect(nextJobRead(job,0,new ApiError('forbidden',403,'failed'),0)).toBe(Infinity);
});
it('strips private terminal failure/provider payloads while preserving public count',()=>{
 expect(jobSchema.parse({...job,terminal_summary:{top_variants_count:2,last_error:{private:'secret'}}}).terminal_summary).toEqual({top_variants_count:2});
});
it('renders only measured progress and explicitly unavailable ETA',async()=>{
 mount();expect(await screen.findByRole('progressbar')).toHaveAttribute('value','50');expect(screen.getByText('5 / 10 units processed')).toBeVisible();expect(screen.getByText('Completion estimate unavailable')).toBeVisible();
});
it('100 percent running remains running; terminal hides stale progress',async()=>{
 const {client}=mount(()=>reply({...job,progress:{...job.progress,percent:100}}));
 expect(await screen.findByText('Running')).toBeVisible();
 act(()=>client.setQueryData(['private','actor','job',id],{...job,state:'failed'}));
 expect(await screen.findByText('The server reports failure.')).toBeVisible();expect(screen.queryByRole('progressbar')).toBeNull();
});
it('dismiss sends no command, double confirmation sends one and pending is never cancelled',async()=>{
 let finish!:(r:Response)=>void;const pending=new Promise<Response>(r=>{finish=r;});
 const {fetch}=mount((url)=>url.pathname.endsWith('/cancel')?pending:undefined);
 await userEvent.click(await screen.findByRole('button',{name:'Cancel backtest'}));
 await userEvent.click(screen.getByRole('button',{name:'Keep running'}));
 expect(fetch.mock.calls.filter(([url])=>url.pathname.endsWith('/cancel'))).toHaveLength(0);
 await userEvent.click(screen.getByRole('button',{name:'Cancel backtest'}));
 const confirm=screen.getByRole('button',{name:'Confirm cancellation'});
 act(()=>{confirm.click();confirm.click();});
 await waitFor(()=>expect(fetch.mock.calls.filter(([url])=>url.pathname.endsWith('/cancel'))).toHaveLength(1));
 await act(async()=>finish(reply({...job,cancel_requested_at:date})));
 expect(await screen.findByText(/Cancellation outcome is pending/)).toBeVisible();expect(screen.queryByText('The server confirms cancellation.')).toBeNull();
 expect(screen.getByRole('button',{name:'Cancel backtest'})).toBeDisabled();
});
it('a terminal read during cancel wins the late cancel response',async()=>{
 let finish!:(r:Response)=>void;const pending=new Promise<Response>(r=>{finish=r;});
 const {client}=mount(url=>url.pathname.endsWith('/cancel')?pending:undefined);
 await userEvent.click(await screen.findByRole('button',{name:'Cancel backtest'}));await userEvent.click(screen.getByRole('button',{name:'Confirm cancellation'}));
 await waitFor(()=>expect(screen.getByText('Sending cancellation request…')).toBeVisible());
 act(()=>client.setQueryData(['private','actor','job',id],{...job,state:'succeeded',terminal_summary:{top_variants_count:2}}));
 await act(async()=>finish(reply({...job,cancel_requested_at:date})));
 expect(await screen.findByText('Completed')).toBeVisible();expect(client.getQueryData(['private','actor','job',id])).toMatchObject({state:'succeeded',terminal_summary:{top_variants_count:2}});expect(screen.queryByText(/Cancellation outcome is pending/)).toBeNull();
});
it.each([403,404,409,429,503])('cancel %i preserves identity and never claims cancellation',async status=>{
 const {client}=mount(url=>url.pathname.endsWith('/cancel')?reply({error:{code:'fault',details:{retry_after_seconds:30}}},status):undefined);
 await userEvent.click(await screen.findByRole('button',{name:'Cancel backtest'}));await userEvent.click(screen.getByRole('button',{name:'Confirm cancellation'}));
 await waitFor(()=>expect(client.getQueryData<{error:ApiError}>(['private','actor','cancel',id])?.error?.status).toBe(status));
 expect(client.getQueryData<Job>(['private','actor','job',id])?.job_id).toBe(id);
 expect(screen.queryByText('The server confirms cancellation.')).toBeNull();
 if(status!==403&&status!==404)expect(screen.getByRole('button',{name:'Cancel backtest'})).toBeDisabled();
});
it('subject change during cancellation gate sends no POST',async()=>{
 const {fetch,client}=mount(url=>url.pathname.includes('current-user')?reply({user_id:'other',paid_level:'free'}):undefined);
 await userEvent.click(await screen.findByRole('button',{name:'Cancel backtest'}));await userEvent.click(screen.getByRole('button',{name:'Confirm cancellation'}));
 await waitFor(()=>expect(client.getQueryData(['session','actor'])).toMatchObject({user_id:'other'}));
 expect(fetch.mock.calls.filter(([,init])=>init.method==='POST')).toHaveLength(0);
});

it.each([403,429])('reconnect cannot bypass a job read %i stop/cooldown',async status=>{
 const {fetch}=mount(url=>url.pathname.endsWith('/'+id)?reply({error:{details:{retry_after_seconds:60}}},status):undefined);
 await screen.findByRole('alert');
 await act(async()=>{onlineManager.setOnline(false);onlineManager.setOnline(true);await new Promise(r=>setTimeout(r,20));});
 expect(fetch.mock.calls.filter(([url])=>url.pathname.endsWith('/'+id))).toHaveLength(1);
});
