import {useEffect,useRef,useState} from 'react';
import {Link,Navigate,useSearchParams} from 'react-router';
import {skipToken,useInfiniteQuery,useQuery,useQueryClient} from '@tanstack/react-query';
import {useTranslation} from 'react-i18next';
import {CheckCircle2,ChevronDown,Clock3,Pause,Play,RefreshCw,TriangleAlert,XCircle,X} from 'lucide-react';
import {ApiError} from './api';
import type {Market} from './connections-api';
import {marketRead,readJob,validateCoverageRange,workAction,workHistorySchema,workEventsSchema,type WorkJob} from './market-data-api';
import {workState} from './market-data-ui';
import {useReadSnapshot} from './read-snapshot';
import {deniesRead,formatTime,ReadFeedback,useWords,WorkError} from './workspace-ui';
import {Confirm} from './results';

import {activeDownload,downloadState,downloadPercent,useSharedDownload,type DownloadEta} from './download-progress';
export {activeDownload} from './download-progress';
const accessDenied=(error:unknown)=>error instanceof ApiError&&[401,403].includes(error.status??0);

/** Old bookmarks keep their selected job, but no separate Downloads page remains. */
export function legacyDownloadsTarget(search:string){
 const params=new URLSearchParams(search);params.set('journal','1');
 if((params.has('from')||params.has('to'))&&!validateCoverageRange(params.get('from')??'',params.get('to')??'')){params.delete('from');params.delete('to');}
 for(const [oldKey,key] of [['state','jobstate'],['kind','jobkind']]){if(params.has(oldKey))params.set(key,params.get(oldKey)!);params.delete(oldKey);}
 return `/data?${params}`;
}
export function LegacyDownloadsRedirect(){const [params]=useSearchParams();return <Navigate replace to={legacyDownloadsTarget(params.toString())}/>;}

export function useDownloadJob(subject:string,id:string|undefined,onDenied:()=>void,seed?:WorkJob,expectActive=false){
 const {query,eta}=useSharedDownload(subject,id,seed,expectActive);
 const view=useReadSnapshot(subject,id??'',query.isSuccess?query.data:undefined,!id||deniesRead(query.error));
 useEffect(()=>{if(accessDenied(query.error))onDenied();},[query.error]);
 return {query,view,eta};
}

export function JobState({state,started_at=null,completed_units=0,queue_waiting=false,storage_retry_at=null}:{state:string;started_at?:string|null;completed_units?:number;queue_waiting?:boolean;storage_retry_at?:string|null}){
 const w=useWords(),display=downloadState({state,started_at,completed_units,queue_waiting,storage_retry_at}),Icon=display==='succeeded'?CheckCircle2:display==='failed'?TriangleAlert:display==='cancelled'?XCircle:display==='downloading'?RefreshCw:display==='paused'||display==='pause_requested'?Pause:Clock3;
 return <span className={`data-job-state ${display}`}><Icon aria-hidden="true"/>{display==='downloading'?w('Downloading','Загрузка'):workState(display,w)}</span>;
}
export function DownloadEtaLabel({eta}:{eta:DownloadEta}){
 const w=useWords();let text='';
 if(eta.kind==='ready'){
  const minutes=Math.max(1,Math.ceil((eta.seconds??0)/60/5)*5);
  text=minutes>=1440?`≈ ${Math.floor(minutes/1440)} ${w('d','д')} ${Math.ceil(minutes%1440/60)} ${w('h','ч')}`:minutes>=60?`≈ ${Math.floor(minutes/60)} ${w('h','ч')} ${minutes%60} ${w('min','мин')}`:`≈ ${minutes} ${w('min','мин')}`;
  if((eta.seconds??0)<300)text=`≈ ${Math.max(1,Math.ceil((eta.seconds??0)/60))} ${w('min','мин')}`;
 }else if(eta.kind==='estimating')text=w('Estimating…','Расчёт…');
 else if(eta.kind==='stalled')text=w('No progress','Нет прогресса');
 else if(eta.kind==='unavailable')text=w('No connection','Нет связи');
 else if(eta.kind==='paused')text=w('Paused','На паузе');
 else if(eta.kind==='waiting')text=w('Waiting','Ожидание');
 return <span className="data-download-eta" title={w('Estimated time remaining at the current speed','Осталось при текущей скорости')} aria-label={text?`${w('Time remaining','Осталось')}: ${text}`:undefined}>{text}</span>;
}
export function JobProgressBar({job,eta}:{job:WorkJob;eta:DownloadEta}){
 const w=useWords();return <div className={`data-job-progress-bar ${job.state==='succeeded'?'complete':''}`}><progress max={100} value={downloadPercent(job)} aria-label={w('Job progress','Ход задания')}/><DownloadEtaLabel eta={eta}/></div>;
}
export function jobSource(job:WorkJob,markets:Market[],w:ReturnType<typeof useWords>){
 const market=markets.find(m=>m.market_id===job.market_id);
 return market?`${market.exchange_name==='bybit'?'Bybit':market.exchange_name==='binance'?'Binance':market.exchange_name} · ${market.market_type==='spot'?w('Spot','Спот'):w('Futures','Фьючерсы')}`:`${w('Market','Рынок')} ${job.market_id}`;
}

/** Loaded only while expanded; the catalogue remains mounted above this journal. */
export function DownloadJournal({subject,markets,blocked,onDenied,onClose,onSelect,jobHref}:{subject:string;markets:Market[];blocked:boolean;onDenied:()=>void;onClose:()=>void;onSelect:()=>void;jobHref:(job:WorkJob)=>string}){
 const w=useWords(),{i18n}=useTranslation(),[params,setParams]=useSearchParams();
 const state=params.get('jobstate')??'',kind=params.get('jobkind')??'';
 const history=useInfiniteQuery({queryKey:['data-jobs',subject,state,kind],enabled:!blocked,initialPageParam:{},maxPages:5,retry:false,
  queryFn:({signal,pageParam})=>marketRead(`work-requests?${new URLSearchParams({limit:'30',...(state?{state}:{}),...(kind?{kind}:{}),...pageParam})}`,workHistorySchema,signal),getNextPageParam:p=>p.next_cursor??undefined});
 const view=useReadSnapshot(subject,`${state}:${kind}`,history.isSuccess?history.data:undefined,blocked||deniesRead(history.error));
 useEffect(()=>{if(accessDenied(history.error))onDenied();},[history.error]);
 function filter(name:string,value:string){const next=new URLSearchParams(params);value?next.set(name,value):next.delete(name);setParams(next,{replace:true});}
 return <section id="data-download-journal" className="data-download-journal" aria-label={w('Download log','Журнал загрузок')} onKeyDown={e=>{if(e.key==='Escape'){e.stopPropagation();onClose();}}}>
  <header><h2>{w('Download log','Журнал загрузок')}</h2><span>{w('All exchanges','Все биржи')}</span><button className="library-icon-action" aria-label={w('Refresh download log','Обновить журнал загрузок')} disabled={history.isFetching||blocked} onClick={()=>void history.refetch()}><RefreshCw aria-hidden="true"/></button><button className="library-icon-action" aria-label={w('Close download log','Закрыть журнал загрузок')} onClick={onClose}><X aria-hidden="true"/></button></header>
  <div className="data-journal-filters">
   <label className="data-select"><span className="sr-only">{w('Job state','Состояние задания')}</span><select value={state} onChange={e=>filter('jobstate',e.target.value)}><option value="">{w('All states','Все состояния')}</option>{['queued','running','pause_requested','paused','retry_wait','cancel_requested','cancelled','succeeded','failed'].map(s=><option key={s} value={s}>{workState(s,w)}</option>)}</select><ChevronDown aria-hidden="true"/></label>
   <label className="data-select"><span className="sr-only">{w('Job type','Тип задания')}</span><select value={kind} onChange={e=>filter('jobkind',e.target.value)}><option value="">{w('All tasks','Все задачи')}</option><option value="candle_ingestion">{w('Candles','Свечи')}</option><option value="catalog_refresh">{w('Catalog refresh','Обновление каталога')}</option></select><ChevronDown aria-hidden="true"/></label>
  </div>
  <div className="data-read-status"><ReadFeedback pending={false} retained={view.retained} initial={!view.data&&!blocked} error={history.error}/></div>
  <div className="data-journal-scroll" tabIndex={0} aria-label={w('Download log entries','Записи журнала загрузок')}>
   <table><thead><tr><th>{w('Task / source','Задача / источник')}</th><th>{w('State','Состояние')}</th><th>{w('Job progress','Ход задания')}</th><th>{w('Created · UTC','Создано · UTC')}</th></tr></thead><tbody>{view.data?.pages.flatMap(p=>p.items).map(job=><JournalRow key={job.job_id} subject={subject} seed={job} selected={params.get('job')===job.job_id} retained={view.retained} markets={markets} jobHref={jobHref} onSelect={onSelect} onDenied={onDenied}/>) }</tbody></table>
   {view.data&&!view.data.pages[0]?.items.length&&<p className="work-empty">{w('No requests','Нет заданий')}</p>}
   {history.hasNextPage&&<button className="data-journal-more" disabled={history.isFetching||view.retained||blocked} onClick={()=>void history.fetchNextPage()}>{w('Earlier requests','Более ранние задания')}</button>}
  </div>
 </section>;
}

function JournalRow({subject,seed,selected,retained,markets,jobHref,onSelect,onDenied}:{subject:string;seed:WorkJob;selected:boolean;retained:boolean;markets:Market[];jobHref:(job:WorkJob)=>string;onSelect:()=>void;onDenied:()=>void}){
 const w=useWords(),{i18n}=useTranslation(),{query,view,eta}=useDownloadJob(subject,seed.job_id,onDenied,seed),job=view.data;
 if(!job)return null;
 return <tr className={selected?'selected':''}><td><Link to={jobHref(job)} aria-disabled={retained} onClick={e=>{if(retained)e.preventDefault();else onSelect();}}>{job.kind==='catalog_refresh'?w('Catalog refresh','Обновление каталога'):job.symbols.join(', ')}</Link><small>{jobSource(job,markets,w)}</small></td><td><JobState {...job}/>{query.error&&<span className="data-job-read-error" title={w('Progress update unavailable. Refresh the request.','Прогресс недоступен. Обновите задание.')}><TriangleAlert aria-label={w('Update unavailable','Нет обновления')}/></span>}</td><td><div className="data-journal-progress"><span>{downloadPercent(job)}%</span><JobProgressBar job={job} eta={eta}/></div></td><td>{formatTime(job.created_at,i18n.language)}</td></tr>;
}

export function DownloadJobDetails({subject,job,eta,locked,markets,onDenied,onChanged}:{subject:string;job:WorkJob;eta:DownloadEta;locked:boolean;markets:Market[];onDenied:()=>void;onChanged:()=>Promise<void>}){
 const w=useWords(),{i18n}=useTranslation(),percent=downloadPercent(job);
 return <section className="data-inspector-section data-job-detail" aria-label={w('Download request','Задание загрузки')}>
  <h3>{job.kind==='catalog_refresh'?w('Catalog refresh','Обновление каталога'):w('Download request','Задание загрузки')}</h3><p>{jobSource(job,markets,w)}{job.symbols.length>1?` · ${job.symbols.join(', ')}`:''}</p>
  <div className="data-job-summary"><JobState {...job}/><strong>{percent}%</strong></div><JobProgressBar job={job} eta={eta}/>
  <p>{w('Job progress','Ход задания')}: {job.completed_units.toLocaleString(i18n.language)} / {job.total_units.toLocaleString(i18n.language)} {job.kind==='candle_ingestion'?w('minutes','минут'):w('catalog','каталог')}</p>
  <p>{w('Attempt','Попытка')}: {job.attempt} / 5</p>
  {job.start_at&&<p>{formatTime(job.start_at,i18n.language)} – {formatTime(job.end_at,i18n.language)} · UTC</p>}
  {job.state==='failed'&&job.error_code&&<p className="notice error" role="alert">{job.error_code==='worker_lost'?w('The worker stopped responding.','Обработчик перестал отвечать.'):w('The source or storage could not complete this request.','Источник или хранилище не смогли завершить задание.')} <span>{w('Progress is saved. Continue from the last confirmed position.','Прогресс сохранён. Продолжение — с последней подтверждённой позиции.')}</span></p>}
  {job.state==='failed'&&!job.can_retry&&<p>{w('The continuation limit has been reached. Resolve the cause, then create a new request.','Лимит продолжений исчерпан. Устраните причину, затем создайте новое задание.')}</p>}
  {job.state==='cancel_requested'&&<p role="status">{w('Stopping after the current write. Stored candles are preserved.','Загрузка остановится после текущей записи. Сохранённые свечи останутся.')}</p>}
  {job.state==='pause_requested'&&<p role="status">{w('Finishing the current write before pausing.','Завершаем текущую запись перед паузой.')}</p>}
  {job.state==='paused'&&<p>{w('Progress is saved. Resume when ready.','Прогресс сохранён. Можно продолжить в любое время.')}</p>}
  {job.state==='retry_wait'&&<p role="status">{job.error_code==='source_rate_limited'?w('Exchange rate limit.','Ограничение частоты запросов биржи.'):job.error_code==='storage_memory_pressure'?w('Storage is waiting for memory to become available.','Хранилище ожидает освобождения памяти.'):job.error_code==='storage_unavailable'?w('Storage temporarily unavailable.','Хранилище временно недоступно.'):job.error_code==='worker_lost'?w('Recovering the download worker.','Восстановление обработчика загрузки.'):w('Source temporarily unavailable.','Источник временно недоступен.')} {w('Will retry automatically','Автоповтор')}{job.next_retry_at?` · ${formatTime(job.next_retry_at,i18n.language)} · UTC`:''}. {w('Progress is saved.','Прогресс сохранён.')}</p>}
  {job.state==='queued'&&job.storage_retry_at&&<p role="status">{w('Storage recovery. Queue will continue automatically after','Восстановление хранилища. Очередь автоматически продолжится после')} {formatTime(job.storage_retry_at,i18n.language)} · UTC.</p>}
  {job.state==='queued'&&job.queue_waiting&&!job.storage_retry_at&&<p>{w('Waiting for an available download slot.','Ожидание свободного слота загрузки.')}</p>}
  <DownloadJobActions subject={subject} job={job} locked={locked} onDenied={onDenied} onChanged={onChanged}/>
  <JobEvents subject={subject} job={job} onDenied={onDenied}/>
 </section>;
}

export function StandaloneDownloadJob({subject,id,markets,onDenied,onChanged,onClose}:{subject:string;id:string;markets:Market[];onDenied:()=>void;onChanged:()=>Promise<void>;onClose:()=>void}){
 const w=useWords(),{query,view,eta}=useDownloadJob(subject,id,onDenied);
 return <><header className="data-inspector-heading"><h2>{w('Request details','Подробности задания')}</h2><button className="library-icon-action" aria-label={w('Close request','Закрыть задание')} onClick={onClose}><X aria-hidden="true"/></button></header><ReadFeedback pending={false} retained={view.retained} initial={!view.data} error={query.error}/>
  {query.error&&<button disabled={query.isFetching} onClick={()=>void query.refetch()}>{w('Refresh request','Обновить задание')}</button>}
  {view.data&&<><p className="data-job-symbols">{view.data.symbols.map(symbol=><Link key={symbol} to={`/data?${new URLSearchParams({market:String(view.data!.market_id),symbol,job:view.data!.job_id,journal:'1'})}`}>{symbol}</Link>)}</p><DownloadJobDetails key={view.data.job_id} subject={subject} job={view.data} eta={eta} markets={markets} locked={view.retained||!!query.error} onDenied={onDenied} onChanged={onChanged}/></>}
 </>;
}

type ActionState={pending:boolean;unknown:boolean;error?:unknown};
/** All visible controls share the same command fence and unknown-outcome state. */
export function DownloadJobActions({subject,job,locked,onDenied,onChanged,compact=false}:{subject:string;job:WorkJob;locked:boolean;onDenied:()=>void;onChanged?:()=>Promise<void>;compact?:boolean}){
 const w=useWords(),client=useQueryClient(),key=['data-job-action',subject,job.job_id];
 const state=useQuery<ActionState>({queryKey:key,queryFn:skipToken,enabled:false,initialData:{pending:false,unknown:false}}).data!;
 const write=(value:ActionState)=>client.setQueryData(key,value);
 const disabled=locked||state.pending||state.unknown;
 async function save(value:WorkJob){
  const progressKey=['data-job',subject,value.job_id];
  await client.cancelQueries({queryKey:progressKey,exact:true});client.setQueryData(progressKey,value);
  await Promise.all([onChanged?.(),...['exchange-catalog','data-inspector','instrument-catalog','data-jobs','data-job-events'].map(name=>client.invalidateQueries({queryKey:[name,subject]}))]);
 }
 async function action(name:'cancel'|'retry'|'pause'|'resume'){
  const live=client.getQueryData<ActionState>(key);
  if(locked||live?.pending||live?.unknown||!job[`can_${name}`])return;
  write({pending:true,unknown:false});
  try{await save(await workAction(job.job_id,name,job.attempt,job.control_version??0));write({pending:false,unknown:false});}
  catch(error){write({pending:false,unknown:error instanceof ApiError&&error.outcome==='unknown',error});if(accessDenied(error))onDenied();}
 }
 async function reconcile(){
  if(locked||client.getQueryData<ActionState>(key)?.pending)return;
  write({...state,pending:true});
  try{await save(await readJob(job.job_id));write({pending:false,unknown:false});}
  catch(error){write({...state,pending:false,error});if(accessDenied(error))onDenied();}
 }
 const suffix=compact?` · ${job.symbols.join(', ')}`:'';
 const label=(name:string)=>name+suffix;
 const button=(name:'pause'|'resume'|'retry',text:string,icon:React.ReactNode)=>job[`can_${name}`]&&<button type="button" className={compact?'library-icon-action':undefined} aria-label={label(text)} title={locked?w('Refresh the saved state to enable commands.','Обновите сохранённое состояние для управления.'):label(text)} disabled={disabled} onClick={()=>void action(name)}>{icon}{!compact&&text}</button>;
 return <div className={compact?'data-row-actions':'work-actions data-job-actions'} aria-label={w('Download controls','Управление загрузкой')}>
  {button('pause',w('Pause','Пауза'),<Pause aria-hidden="true"/>)}
  {button('resume',w('Resume','Продолжить'),<Play aria-hidden="true"/>)}
  {button('retry',w('Continue download','Продолжить загрузку'),<Play aria-hidden="true"/>)}
  {job.can_cancel&&<Confirm id={`cancel-data-${compact?'row':'detail'}-${job.job_id}`} title={w('Cancel this download?','Отменить загрузку?')} help={w('Stored candles are preserved. The current write may finish before cancellation.','Сохранённые свечи останутся. Текущая запись может завершиться до отмены.')} source={job.symbols.join(', ')||w('Catalog refresh','Обновление каталога')} trigger={label(w('Cancel download','Отменить загрузку'))} icon={<X aria-hidden="true"/>} confirmText={w('Cancel download','Отменить загрузку')} disabled={disabled} onConfirm={()=>void action('cancel')}/>}
  {state.unknown&&<button type="button" disabled={locked||state.pending} aria-label={label(w('Check saved state','Проверить сохранённое состояние'))} title={w('The command result is unknown. Read its saved state; the command will not be resent.','Результат команды неизвестен. Проверить сохранённое состояние без повторной отправки.')} onClick={()=>void reconcile()}>{compact?<RefreshCw aria-hidden="true"/>:w('Check saved state','Проверить сохранённое состояние')}</button>}
  <WorkError error={state.error}/>
 </div>;
}

function eventReason(code:string|null,w:ReturnType<typeof useWords>){
 return code==='storage_memory_pressure'?w('ClickHouse needs more free memory.','ClickHouse ожидает освобождения памяти.'):
 code==='storage_unavailable'?w('Storage is temporarily unavailable.','Хранилище временно недоступно.'):
 code==='source_rate_limited'?w('The exchange requested a cooldown.','Биржа запросила перерыв между запросами.'):
 code==='source_unavailable'?w('The exchange is temporarily unavailable.','Биржа временно недоступна.'):
 code==='worker_lost'?w('The worker stopped responding.','Обработчик перестал отвечать.'):
 code?w('The source or storage could not complete this step.','Источник или хранилище не смогли выполнить этот шаг.'):'';
}
function JobEvents({subject,job,onDenied}:{subject:string;job:WorkJob;onDenied:()=>void}){
 const w=useWords(),[open,setOpen]=useState(false);
 return <details className="data-job-events" onToggle={e=>setOpen(e.currentTarget.open)}><summary>{w('Download events','События загрузки')}</summary>{open&&<JobEventList subject={subject} job={job} onDenied={onDenied}/>}</details>;
}
function JobEventList({subject,job,onDenied}:{subject:string;job:WorkJob;onDenied:()=>void}){
 const w=useWords(),{i18n}=useTranslation();
 const query=useInfiniteQuery({queryKey:['data-job-events',subject,job.job_id],initialPageParam:null as number|null,maxPages:5,retry:false,
  queryFn:({signal,pageParam})=>marketRead(`work-requests/${encodeURIComponent(job.job_id)}/events?limit=30${pageParam?`&before=${pageParam}`:''}`,workEventsSchema,signal),getNextPageParam:page=>page.next_cursor??undefined,
  refetchInterval:read=>{const e=read.state.error;if(e instanceof ApiError){if(e.status!==null&&e.status!==429&&(e.status??0)<500)return false;return Math.max(10000,(e.retryAfterSeconds??0)*1000);}return activeDownload(job.state)?5000:false;}});
 useEffect(()=>{if(accessDenied(query.error))onDenied();},[query.error]);
 const labels:Record<string,string>={queued:w('Added to queue','Добавлено в очередь'),started:w('Download started','Загрузка началась'),pause_requested:w('Pause requested','Запрошена пауза'),paused:w('Paused','Приостановлено'),resumed:w('Resumed from saved position','Продолжено с сохранённой позиции'),continued:w('Continued after failure','Продолжено после ошибки'),retry_scheduled:w('Temporary failure; retry scheduled','Временный отказ; назначен повтор'),recovered:w('Automatic recovery started','Началось автоматическое восстановление'),cancel_requested:w('Cancellation requested','Запрошена отмена'),cancelled:w('Cancelled; stored candles preserved','Отменено; свечи сохранены'),succeeded:w('Download completed','Загрузка завершена'),failed:w('Download stopped','Загрузка остановлена'),history_started:w('Event recording started; earlier history unavailable','Начало записи событий; предыдущая история недоступна')};
 const items=query.data?.pages.flatMap(p=>p.items)??[];
 function download(){const blob=new Blob([JSON.stringify({job_id:job.job_id,items,scope:'currently loaded events',retention_days:30,max_events:512},null,2)],{type:'application/json'});const url=URL.createObjectURL(blob),a=document.createElement('a');a.href=url;a.download=`download-${job.job_id}.json`;a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);}
 return <div><ReadFeedback pending={false} initial={!query.data} error={query.error}/><div className="work-actions"><button type="button" disabled={query.isFetching} onClick={()=>void query.refetch()}>{w('Refresh events','Обновить события')}</button><button type="button" disabled={!items.length} onClick={download}>{w('Download shown events','Скачать показанные события')}</button></div>
 <p>{w('Latest 512 events, retained for 30 days.','Последние 512 событий, срок хранения — 30 дней.')}</p>
 <ol className="data-event-list">{items.map(event=><li key={event.event_id}><time dateTime={event.occurred_at}>{formatTime(event.occurred_at,i18n.language)} · UTC</time><strong>{labels[event.event_type]??event.event_type}</strong>{event.error_code&&<p>{eventReason(event.error_code,w)}</p>}<p>{w('Saved progress','Сохранённый прогресс')}: {event.completed_units.toLocaleString(i18n.language)} / {event.total_units.toLocaleString(i18n.language)} · {w('Written','Записано')}: {event.rows_written.toLocaleString(i18n.language)}</p>{event.next_retry_at&&<p>{w('Automatic retry','Автоповтор')}: {formatTime(event.next_retry_at,i18n.language)} · UTC{event.state==='paused'?` · ${w('after Resume','после продолжения пользователем')}`:''}</p>}{event.state==='failed'&&<p>{w('Continue the request after resolving the cause.','После устранения причины продолжите задание.')}</p>}{(event.error_code||event.error_phase)&&<details><summary>{w('Technical details','Технические сведения')}</summary><code>{event.error_code??'—'} · {event.error_phase??'—'}</code><p>{w('Attempt','Попытка')}: {event.attempt}</p></details>}</li>)}</ol>
 {query.hasNextPage&&<button type="button" disabled={query.isFetching} onClick={()=>void query.fetchNextPage()}>{w('Earlier events','Более ранние события')}</button>}</div>;
}
