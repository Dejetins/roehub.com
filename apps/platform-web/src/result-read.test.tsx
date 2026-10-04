import {afterEach,expect,it,vi} from 'vitest';
import {act,cleanup,renderHook} from '@testing-library/react';
import {QueryClientProvider} from '@tanstack/react-query';
import type {ReactNode} from 'react';
import {ApiError} from './api';
import {createQueryClient} from './query-client';
import {useResultRead} from './result-read';

afterEach(()=>{cleanup();vi.useRealTimers();});

it.each([
 ['rate-limited',429,3],
 ['forbidden',403,null],
 ['not-found',404,null],
] as const)('preserves cached %s scheduling when returning to a result key',async(kind,status,delay)=>{
 vi.useFakeTimers();
 const client=createQueryClient();
 const read=vi.fn(async()=>{throw new ApiError(kind,status,'failed',delay);});
 const wrapper=({children}:{children:ReactNode})=><QueryClientProvider client={client}>{children}</QueryClientProvider>;
 const view=renderHook(({key})=>useResultRead('actor',['job','variant',key],read,Date.now(),key==='drawdown'),{wrapper,initialProps:{key:'drawdown'}});
 await act(async()=>{await vi.advanceTimersByTimeAsync(10);});
 expect(read).toHaveBeenCalledTimes(1);
 expect(view.result.current.error).toBeInstanceOf(ApiError);
 view.rerender({key:'equity'});
 view.rerender({key:'drawdown'});
 await act(async()=>{await vi.advanceTimersByTimeAsync(1000);});
 expect(read).toHaveBeenCalledTimes(1);
 if(status===429){
  await act(async()=>{await vi.advanceTimersByTimeAsync(2000);});
  expect(read).toHaveBeenCalledTimes(2);
 }else{
  await act(async()=>{await vi.advanceTimersByTimeAsync(10_000);});
  expect(read).toHaveBeenCalledTimes(1);
 }
 view.unmount();client.clear();
});
