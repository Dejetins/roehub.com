import {useMemo,useSyncExternalStore} from 'react';
import {QueryClient,useQuery,useQueryClient} from '@tanstack/react-query';
import {ApiError} from './api';
import {readJob,type WorkJob} from './market-data-api';

export const activeDownload=(state:string|undefined)=>!!state&&['queued','running','pause_requested','paused','retry_wait','cancel_requested'].includes(state);
export function downloadState(job:Pick<WorkJob,'state'|'started_at'|'completed_units'|'queue_waiting'|'storage_retry_at'>){
 return job.state==='running'||(job.state==='queued'&&!job.queue_waiting&&!job.storage_retry_at&&(job.started_at!==null||job.completed_units>0))?'downloading':job.state;
}
export function downloadPercent(job:Pick<WorkJob,'state'|'completed_units'|'total_units'>){
 const value=job.total_units?job.completed_units/job.total_units*100:0;
 return Math.max(0,Math.min(job.state==='succeeded'?100:99.99,Math.floor(value*100)/100));
}

export type DownloadEta={kind:'estimating'|'ready'|'stalled'|'unavailable'|'paused'|'waiting'|'hidden';seconds?:number};
const hidden:DownloadEta={kind:'hidden'};

/** Only observed work counts. The rolling wall-clock window includes worker pauses. */
export class DownloadRate {
 private attempt:number|undefined;
 private epoch:number|undefined;
 private interrupted=false;
 private samples:{at:number;units:number}[]=[];
 private lastAdvance=0;
 private calculated=0;
 private rate=0;
 private eta:DownloadEta=hidden;
 observe(job:WorkJob,now:number,unavailable=false):DownloadEta{
  if(this.interrupted||this.epoch!==(job.progress_epoch??0)||this.attempt!==job.attempt||job.completed_units<(this.samples.at(-1)?.units??0)){
   this.attempt=job.attempt;this.epoch=job.progress_epoch??0;this.interrupted=false;this.samples=[];this.lastAdvance=now;this.calculated=0;this.rate=0;this.eta=hidden;
  }
  if(!activeDownload(job.state)||job.state==='cancel_requested')return this.eta=hidden;
  if(unavailable){this.interrupted=true;return this.eta={kind:'unavailable'};}
  if(['paused','pause_requested','retry_wait'].includes(job.state)){this.interrupted=true;return this.eta={kind:job.state==='retry_wait'?'waiting':'paused'};}
  if(job.state==='queued'&&job.storage_retry_at){this.interrupted=true;return this.eta={kind:'waiting'};}
  const queued=job.state==='queued'&&job.queue_waiting;
  const previous=this.samples.at(-1);
  if(previous&&job.completed_units>previous.units)this.lastAdvance=now;
  if(!previous||(now>previous.at&&(!queued||job.completed_units>previous.units)))this.samples.push({at:now,units:job.completed_units});
  while(this.samples.length>1&&this.samples[1].at<=now-120_000)this.samples.shift();
  if(queued)return this.eta={kind:'waiting'};
  if(downloadState(job)!=='downloading')return this.eta=hidden;
  if(now-this.lastAdvance>=60_000){this.rate=0;return this.eta={kind:'stalled'};}
  const first=this.samples[0],elapsed=(now-first.at)/1000,units=job.completed_units-first.units;
  if(elapsed<20||units<=0)return this.eta={kind:'estimating'};
  if(now-this.calculated<10_000&&this.eta.kind==='ready')return this.eta;
  const rate=units/elapsed;
  this.rate=this.rate?this.rate*.6+rate*.4:rate;this.calculated=now;
  return this.eta={kind:'ready',seconds:Math.max(1,(job.total_units-job.completed_units)/this.rate)};
 }
}

/** A single lightweight timer per visible job, shared by row, journal and inspector.
 * Payloads remain in QueryClient under its normal bounds; this registry only owns
 * live subscriptions and a two-minute rate window, discarded on the last unmount. */
const registries=new WeakMap<QueryClient,Map<string,ProgressSubscription>>();
class ProgressSubscription {
 private listeners=new Set<()=>void>();
 private timer:ReturnType<typeof setInterval>|undefined;
 private stop:(()=>void)|undefined;
 private rate=new DownloadRate();
 private value:DownloadEta=hidden;
 private previous:WorkJob|undefined;
 private readFailures=0;
 private nextReadAt=0;
 readonly key:readonly string[];
 constructor(private client:QueryClient,private subject:string,private id:string,private entries:Map<string,ProgressSubscription>,private identity:string,private expectsActive:boolean){this.key=['data-job',subject,id];}
 snapshot=()=>this.value;
 private update=()=>{
  const state=this.client.getQueryState<WorkJob>(this.key),job=state?.data;
  if(!job)return;
  const next=this.rate.observe(job,Date.now(),!!state.error);
  if(next.kind!==this.value.kind||next.seconds!==this.value.seconds){this.value=next;this.listeners.forEach(listener=>listener());}
  const previous=this.previous,wasActive=this.expectsActive;this.previous=job;this.expectsActive=false;
  // Refresh derived coverage and history once at a terminal transition, never per tick.
  if(!activeDownload(job.state)&&(wasActive||(previous&&previous.attempt===job.attempt&&activeDownload(previous.state)))){
   for(const name of ['exchange-catalog','data-inspector','instrument-catalog','data-jobs'])void this.client.invalidateQueries({queryKey:[name,this.subject]});
  }
 };
 subscribe=(listener:()=>void)=>{
  this.listeners.add(listener);
  if(this.listeners.size===1){
   this.entries.set(this.identity,this);
   this.stop=this.client.getQueryCache().subscribe(event=>{if(event.type==='updated'&&event.query.queryKey.length===3&&event.query.queryKey.every((part:unknown,i:number)=>part===this.key[i]))this.update();});
   this.update();
   this.timer=setInterval(()=>{
    this.update();
    const state=this.client.getQueryState<WorkJob>(this.key);
    if(!state||state.fetchStatus==='fetching'||Date.now()<this.nextReadAt)return;
    const error=state.error;
    const transient=error instanceof ApiError&&(error.status===null||error.status===408||error.status===429||(error.status??0)>=500);
    if(error&&!transient)return;
    if(activeDownload(state.data?.state)||transient){
     this.nextReadAt=Date.now()+2000;
     void this.client.fetchQuery({queryKey:this.key,queryFn:({signal})=>readJob(this.id,signal),staleTime:0,retry:false}).then(()=>{this.readFailures=0;this.nextReadAt=0;}).catch((error:unknown)=>{
      this.readFailures++;const retryAfter=error instanceof ApiError?(error.retryAfterSeconds??0)*1000:0;
      this.nextReadAt=Date.now()+Math.max(retryAfter,Math.min(30000,2000*2**Math.min(this.readFailures,4)));
     });
    }
   },2000);
  }
  return()=>{this.listeners.delete(listener);if(!this.listeners.size){clearInterval(this.timer);this.stop?.();this.entries.delete(this.identity);}};
 };
}

export function useSharedDownload(subject:string,id:string|undefined,seed?:WorkJob,expectActive=false){
 const client=useQueryClient();
 const query=useQuery({queryKey:['data-job',subject,id],enabled:!!id,retry:false,queryFn:({signal})=>readJob(id!,signal),initialData:seed});
 const subscription=useMemo(()=>{
  if(!id)return undefined;
  let entries=registries.get(client);if(!entries){entries=new Map();registries.set(client,entries);}
  const key=JSON.stringify([subject,id]);let entry=entries.get(key);
  if(!entry){entry=new ProgressSubscription(client,subject,id,entries,key,expectActive);entries.set(key,entry);}
  return entry;
 },[client,subject,id]);
 const eta=useSyncExternalStore(subscription?.subscribe??noSubscription,subscription?.snapshot??noEta);
 return {query,eta};
}
const noSubscription=()=>()=>{};
const noEta=()=>hidden;
