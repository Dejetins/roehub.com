import {afterEach,expect,it,vi} from 'vitest';
import {QueryObserver} from '@tanstack/react-query';
import {createQueryClient,READ_CACHE} from './query-client';
afterEach(()=>vi.useRealTimers());
it('bounds inactive entries while retaining observed data and command recovery',async()=>{
 const client=createQueryClient();const observed=['private','actor','results','visible'];
 client.setQueryData(observed,{points:[1,2]});
 const observer=new QueryObserver(client,{queryKey:observed,enabled:false});const stop=observer.subscribe(()=>{});
 client.setQueryData(['private','actor','cancel','pending'],{phase:'unresolved'});
 for(let i=0;i<100;i++)client.setQueryData(['private','actor','results',i],{points:[i]});
 await Promise.resolve();
 expect(client.getQueryData(observed)).toEqual({points:[1,2]});
 expect(client.getQueryCache().findAll({queryKey:['private','actor','results']}).length).toBe(READ_CACHE.maxInactiveEntries+1);
 expect(client.getQueryData(['private','actor','results',99])).toEqual({points:[99]});
 expect(client.getQueryData(['private','actor','cancel','pending'])).toEqual({phase:'unresolved'});
 stop();client.clear();
});
it('evicts oversized inactive data and expires unused reads after two minutes',async()=>{
 vi.useFakeTimers();const client=createQueryClient();
 client.setQueryData(['private','actor','results','large'],'x'.repeat(READ_CACHE.maxInactiveBytes));await Promise.resolve();
 expect(client.getQueryData(['private','actor','results','large'])).toBeUndefined();
 client.setQueryData(['private','actor','results','small'],[1]);await Promise.resolve();
 await vi.advanceTimersByTimeAsync(READ_CACHE.gcTime+1);
 expect(client.getQueryData(['private','actor','results','small'])).toBeUndefined();client.clear();
});
it('retains unresolved download actions outside the read budget for five minutes',async()=>{
 vi.useFakeTimers();const client=createQueryClient(),key=['data-job-action','owner','job'];client.setQueryData(key,true);
 for(let i=0;i<100;i++)client.setQueryData(['read',i],i);await Promise.resolve();
 await vi.advanceTimersByTimeAsync(READ_CACHE.gcTime+1);expect(client.getQueryData(key)).toBe(true);
 await vi.advanceTimersByTimeAsync(300000);expect(client.getQueryData(key)).toBeUndefined();client.clear();
});
