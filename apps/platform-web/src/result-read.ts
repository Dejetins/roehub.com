import {useEffect} from 'react';
import {useQuery} from '@tanstack/react-query';
import {ApiError,type ApiReply} from './api';
import {isRestricted} from './library';
import * as api from './results-api';
export function useResultRead<T>(subject:string, scope:string[], read:(signal:AbortSignal)=>Promise<ApiReply<T>>, now:number, enabled=true) {
  const query=useQuery({enabled,queryKey:['private',subject,'results',...scope],queryFn:({signal})=>read(signal),refetchOnMount:false,refetchOnReconnect:false,retryOnMount:false});
  const delay=api.resultDelay(query.data,query.error);
  const deadline=(query.error?query.errorUpdatedAt:query.dataUpdatedAt)+delay;
  useEffect(()=>{if(!enabled || query.isFetching || !Number.isFinite(deadline))return;const timer=setTimeout(()=>void query.refetch(),Math.min(2147483647,Math.max(0,deadline-Date.now())));return()=>clearTimeout(timer);},[enabled,deadline,query.isFetching,query.refetch]);
  const manualDeadline=query.error ? isRestricted(query.error) ? Infinity : query.error instanceof ApiError && query.error.kind==='rate-limited' ? deadline : query.errorUpdatedAt + (query.error instanceof ApiError ? query.error.retryAfterSeconds??0 : 0)*1000 : query.data?.status===202?deadline:0;
  return {...query, canRefresh:!query.isFetching&&!isRestricted(query.error)&&now>=manualDeadline, deadline:manualDeadline};
}
