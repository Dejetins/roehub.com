import { useEffect, useRef } from 'react';
import { useQuery, useQueryClient } from '@tanstack/react-query';
import { useNavigate } from 'react-router';
import { useTranslation } from 'react-i18next';
import { ApiError, readSession } from './api';
import { deleteJob } from './results-api';
import { Confirm } from './results';
import { ReadError, isRestricted } from './library';
import type { Job } from './library-api';
export type DeleteState={phase:'idle'|'sending'|'unknown'|'rejected';error:Error|null;deadline:number;readAt:number};
export const deletionIdle:DeleteState={phase:'idle',error:null,deadline:0,readAt:0};
export function DeleteHistory({job,subject,now,backLink,read,canRead,readAt,readError}:{job:Job;subject:string;now:number;backLink:string;read:()=>Promise<unknown>;canRead:boolean;readAt:number;readError:Error|null}){
  const {t}=useTranslation(),client=useQueryClient(),navigate=useNavigate();const key=['private',subject,'delete',job.job_id];
  const state=useQuery<DeleteState>({queryKey:key,enabled:false,initialData:deletionIdle}).data!;
  const controller=useRef<AbortController|null>(null),lock=useRef(false);
  useEffect(()=>()=>{controller.current?.abort();if(lock.current&&client.getQueryData(key))client.setQueryData(key,{...deletionIdle,phase:'unknown'});},[]);
  const terminal=['succeeded','failed','cancelled'].includes(job.state);
  // A successful fresh read after rejection re-establishes eligibility. An unknown
  // outcome never repeats DELETE automatically, even when the job remains visible.
  const rejected=state.phase==='rejected'&&readAt<=state.readAt;
  const allowed=terminal&&!readError&&!isRestricted(state.error)&&!['sending','unknown'].includes(state.phase)&&!rejected&&now>=state.deadline;
  async function remove(){if(lock.current||!allowed)return;lock.current=true;const abort=new AbortController();controller.current=abort;client.setQueryData(key,{...deletionIdle,phase:'sending'});
    try{const identity=await client.fetchQuery({queryKey:['session',subject],queryFn:()=>readSession(abort.signal),staleTime:0,retry:false});if(abort.signal.aborted||identity.user_id!==subject)return;
      await deleteJob(job.job_id,abort.signal);if(abort.signal.aborted)return;
      client.removeQueries({queryKey:['private',subject,'job',job.job_id]});client.removeQueries({queryKey:['private',subject,'results',job.job_id]});client.removeQueries({queryKey:key});
      await client.invalidateQueries({queryKey:['private',subject,'jobs']});navigate(backLink,{replace:true,state:{historyDeleted:true}});
    }catch(e){if(abort.signal.aborted)return;const error=e instanceof Error?e:new Error();client.setQueryData(key,{phase:error instanceof ApiError&&error.outcome==='failed'?'rejected':'unknown',error,deadline:Date.now()+(error instanceof ApiError?error.retryAfterSeconds??0:0)*1000,readAt});}
    finally{lock.current=false;}}
  return <section className="result-command" aria-label={t('results.delete')}><p className="muted">{t('results.deleteEligibility')}</p><Confirm id="delete" trigger={t('results.delete')} title={t('results.deleteTitle')} help={t('results.deleteHelp')} source={job.job_id} onConfirm={()=>void remove()} disabled={!allowed}/><ReadError error={state.error}/>{state.phase==='sending'&&<p role="status">{t('results.deleting')}</p>}{state.phase==='unknown'&&<p role="status" className="notice">{t('results.deleteUnknown')}</p>}{['unknown','rejected'].includes(state.phase)&&<button disabled={!canRead||now<state.deadline||isRestricted(state.error)} onClick={()=>void read()}>{t('results.reconcile')}</button>}</section>;
}
