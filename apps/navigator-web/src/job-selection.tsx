import {useRef,useLayoutEffect} from 'react';
import {useQuery,useQueryClient} from '@tanstack/react-query';
import {useSearchParams} from 'react-router';
import {useTranslation} from 'react-i18next';
import {JobEntry,reconcileJob} from './execution';
import {readJob,jobIdSchema,type Job} from './library-api';
import {ReadError,isRestricted} from './library';
import {ReadStatus} from './loading-data';
import {useReadSnapshot} from './read-snapshot';
import {useResultRead} from './result-read';
import {requestJson,ApiError} from './api';
import * as api from './results-api';

/** Prepare a new entity before replacing the command-owning panel. Identity
 * remounts isolate commands; refreshing the displayed entity never remounts it. */
export function JobSelection(props:Parameters<typeof JobEntry>[0]) {
  const {id,subject,now}=props;const {t}=useTranslation();const [params]=useSearchParams();const client=useQueryClient();
  const valid=jobIdSchema.safeParse(id).success;
  const query=useQuery({queryKey:['private',subject,'job',id],enabled:valid,refetchOnMount:false,queryFn:async({signal})=>{
    const incoming=await readJob(id,signal);if(incoming.job_id!==id)throw new ApiError('invalid-response',200,'failed');
    return reconcileJob(client.getQueryData<Job>(['private',subject,'job',id]),incoming);
  }});
  const succeeded=query.data?.state==='succeeded';
  const summary=useResultRead(subject,[id,'summary'],signal=>requestJson(`${api.jobPath(id)}/summary`,api.summarySchema,{signal}),now,valid&&succeeded);
  const top=useResultRead(subject,[id,'top'],signal=>requestJson(`${api.jobPath(id)}/top`,api.topSchema,{signal}),now,valid&&succeeded);
  const choice=useRef({id,variant:params.get('variant')});
  if(choice.current.id!==id)choice.current={id,variant:params.get('variant')};
  const variant=choice.current.variant??summary.data?.data.selected_variant_key??'';
  const prepare=valid&&succeeded&&api.variantKeySchema.safeParse(variant).success;
  const detail=useResultRead(subject,[id,variant,'variant'],async signal=>{const r=await requestJson(api.variantPath(id,variant),api.variantSchema,{signal});if(r.data.variant_key!==variant)throw new ApiError('invalid-response',200,'failed');return r;},now,prepare);
  const equity=useResultRead(subject,[id,variant,'equity?points=400'],signal=>api.readResult(id,variant,'equity?points=400',api.seriesSchema,signal),now,prepare);
  const reads=succeeded?[summary,top,...(prepare?[detail,equity]:[])]:[];
  const existing=useRef<string|undefined>(undefined);
  const ready=!!query.data&&!query.isFetching&&(!existing.current||existing.current===id||!succeeded||reads.every(read=>read.data?.status===200&&!read.isFetching&&!read.error));
  const snapshot=useReadSnapshot(subject,id,ready?{id,variant}:undefined,!valid||isRestricted(query.error)||reads.some(read=>isRestricted(read.error)));
  const displayed=snapshot.data?.id;
  useLayoutEffect(()=>{existing.current=displayed;},[displayed]);
  const retaining=snapshot.retained&&displayed!==id;
  // Once an entity is displayed its own observers own polling, refresh and errors.
  const materializationFailed=reads.some(read=>read.data?.status===202&&!['queued','running','pending'].includes(String((read.data.data as {status?:string}).status)));
  const pending=query.isFetching||reads.some(read=>read.isFetching||(read.data?.status===202&&['queued','running','pending'].includes(String((read.data.data as {status?:string}).status))));
  return <>
    <ReadStatus className="read-status-overlay" pending={pending&&displayed!==id} retained={retaining}/><ReadError error={query.error}/>{reads.map((read,index)=><ReadError key={index} error={read.error}/>)}
    {materializationFailed&&<p role="status" className="notice">{t('results.failed')}</p>}
    {(query.error||reads.some(read=>read.error)||materializationFailed)&&<button disabled={pending} onClick={()=>{void query.refetch();reads.forEach(read=>{if(read.canRefresh)void read.refetch();});}}>{t('refresh')}</button>}
    {displayed?<div className="retained-job" inert={retaining}><JobEntry {...props} id={displayed} active={props.active&&!retaining} key={displayed}/></div>:!valid?<JobEntry {...props}/>:null}
  </>;
}
