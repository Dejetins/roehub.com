import {useEffect,useRef,useState} from 'react';
import {useQuery} from '@tanstack/react-query';
import {useSearchParams} from 'react-router';
import {accountRead,preferencesSchema} from './account-api';
import {ApiError} from './api';
import {lookupWork,submitWork,type WorkCommand,type WorkJob} from './market-data-api';
import {useWords} from './workspace-ui';

export function useWorkInterval(subject:string){const p=useQuery({queryKey:['account',subject,'preferences'],queryFn:({signal})=>accountRead('preferences',preferencesSchema,signal),retry:false});const interval=p.data?.autorefresh.refresh_interval_seconds??0;return interval>0?Math.max(10000,interval*1000):false as const;}
export function useAutoLoad(hasMore:boolean|undefined,pending:boolean,error:unknown,load:()=>unknown){const ref=useRef<HTMLDivElement>(null);useEffect(()=>{const el=ref.current;if(!el)return;const observer=new IntersectionObserver(entries=>{if(entries.some(e=>e.isIntersecting)&&hasMore&&!pending&&!error)void load();});observer.observe(el);return()=>observer.disconnect();},[hasMore,pending,error,load]);return ref;}
export function workState(state:string,w:ReturnType<typeof useWords>){return ({queued:w('Queued','В очереди'),running:w('Running','Выполняется'),pause_requested:w('Pausing','Приостановка'),paused:w('Paused','На паузе'),retry_wait:w('Waiting to retry','Ожидание повтора'),cancel_requested:w('Cancellation requested','Запрошена отмена'),cancelled:w('Cancelled','Отменено'),succeeded:w('Completed','Завершено'),failed:w('Failed','Ошибка'),fresh:w('Fresh','Актуален'),stale:w('Stale','Устарел'),complete:w('Complete','Полное'),partial:w('Partial','Частичное'),empty:w('No candles','Нет свечей')})[state]??state;}

/** One command, with the recovery key in the URL before dispatch. No automatic retry. */
export function useWorkSubmission(onSaved:(job:WorkJob)=>void){
 const [params,setParams]=useSearchParams(),[pending,setPending]=useState(false),[error,setError]=useState<unknown>(),[unresolved,setUnresolved]=useState(!!params.get('requestkey')),busy=useRef(false);
 async function submit(body:WorkCommand){if(busy.current||unresolved)return;busy.current=true;setPending(true);setError(undefined);const key=crypto.randomUUID();const next=new URLSearchParams(params);next.set('requestkey',key);setParams(next,{replace:true});try{const job=await submitWork(body,key);next.delete('requestkey');setParams(next,{replace:true});onSaved(job);}catch(e){setError(e);if(e instanceof ApiError&&e.outcome==='unknown'){setUnresolved(true);}else{next.delete('requestkey');setParams(next,{replace:true});}}finally{busy.current=false;setPending(false);}}
 async function reconcile(){const key=params.get('requestkey');if(!key||busy.current)return;busy.current=true;setPending(true);try{const job=await lookupWork(key);setError(undefined);setUnresolved(false);const next=new URLSearchParams(params);next.delete('requestkey');setParams(next,{replace:true});onSaved(job);}catch(e){setError(e);}finally{busy.current=false;setPending(false);}}
 return {submit,reconcile,pending,error,unresolved};
}
