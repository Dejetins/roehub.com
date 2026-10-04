import {afterEach,expect,it,vi} from 'vitest';
import {QueryClient,QueryObserver} from '@tanstack/react-query';
import {ApiError} from './api';
import {restoreTemporaryReads} from './session-recovery';

afterEach(()=>vi.useRealTimers());
it.each(['denied','resolved','extended'])('rechecks delayed reads after their state becomes %s',async(kind)=>{
 vi.useFakeTimers();
 const client=new QueryClient();let error:ApiError|null=new ApiError('rate-limited',429,'failed',90);
 const fn=vi.fn(async()=>{if(error)throw error;return 'ok';});
 const observer=new QueryObserver(client,{queryKey:['proof'],queryFn:fn,retry:false});
 const stop=observer.subscribe(()=>{});await observer.refetch();
 const cleanup=restoreTemporaryReads(client);
 await vi.advanceTimersByTimeAsync(30000);
 error=kind==='denied'?new ApiError('forbidden',403,'failed'):kind==='resolved'?null:new ApiError('rate-limited',429,'failed',180);
 await observer.refetch();const reads=fn.mock.calls.length;
 await vi.advanceTimersByTimeAsync(60000);
 expect(fn.mock.calls.length).toBe(reads);
 if(kind==='extended'){
  error=null;await vi.advanceTimersByTimeAsync(120000);
  expect(fn.mock.calls.length).toBe(reads+1);expect(observer.getCurrentResult().data).toBe('ok');
 }
 cleanup();stop();client.clear();
});
