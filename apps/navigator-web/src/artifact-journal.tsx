import {useEffect,useState} from 'react';
import {useInfiniteQuery,useQuery} from '@tanstack/react-query';
import {Link,useSearchParams} from 'react-router';
import {useTranslation} from 'react-i18next';
import {ChevronDown,X,Clock,CheckCircle2,AlertCircle} from 'lucide-react';
import {ApiError} from './api';
import {useWords,ReadFeedback,RefreshButton,deniesRead,formatTime} from './workspace-ui';
import {useReadSnapshot} from './read-snapshot';
import {readArtifactRequests,readArtifactRequest,readArtifactEvents,type ArtifactRequest} from './artifacts-api';

const active=(state:string)=>!['ready','error'].includes(state);
function useLabels(){const w=useWords();return {
 states:{waiting_for_source:w('Waiting for candles','Ожидание свечей'),queued:w('Queued','В очереди'),running:w('Running','Выполняется'),retry_wait:w('Retry pending','Ожидание повтора'),ready:w('Ready','Готово'),error:w('Error','Ошибка')},
 kinds:{base:w('Base files','Базовые файлы'),derivatives:w('Precompute','Предрасчёт')},
 phases:{source_snapshot:w('Reading candles','Чтение свечей'),base:w('Preparing files','Подготовка файлов'),funding:w('Funding rates','Ставки финансирования'),derivatives:w('Calculating indicators','Расчёт индикаторов')}
};}
function phase(request:ArtifactRequest,labels:ReturnType<typeof useLabels>){
 const latest=[...request.progress].sort((a,b)=>(Date.parse(b.observed_at??'')||0)-(Date.parse(a.observed_at??'')||0))[0];
 return latest?labels.phases[latest.phase as keyof typeof labels.phases]??labels.kinds[request.kind]:'—';
}
function Status({state}:{state:ArtifactRequest['state']}){const labels=useLabels(),Icon=state==='ready'?CheckCircle2:state==='error'?AlertCircle:Clock;return <span className={`artifact-cell-status${state==='ready'?' ready':state==='error'?' failed':''}`}><Icon aria-hidden="true"/>{labels.states[state]}</span>;}
function useCooldown(errors:unknown[]){const [until,setUntil]=useState(0);useEffect(()=>{const seconds=Math.max(0,...errors.map(e=>e instanceof ApiError?e.retryAfterSeconds??0:0));if(seconds)setUntil(Date.now()+seconds*1000);},errors);useEffect(()=>{if(!until)return;const timer=setTimeout(()=>setUntil(0),Math.max(1,until-Date.now()));return()=>clearTimeout(timer);},[until]);return until>Date.now();}

export function ArtifactJournal({subject,blocked,onDenied,onClose,onSelect}:{subject:string;blocked:boolean;onDenied:()=>void;onClose:()=>void;onSelect:()=>void}){
 const w=useWords(),labels=useLabels(),{i18n}=useTranslation(),[params,setParams]=useSearchParams();
 const state=params.get('artifactstate')??'',kind=params.get('artifactkind')??'';
 const history=useInfiniteQuery({queryKey:['artifacts',subject,'history',state,kind],enabled:!blocked,initialPageParam:'',maxPages:5,retry:false,refetchOnWindowFocus:false,
 queryFn:({signal,pageParam})=>readArtifactRequests(pageParam,signal,{state,kind}),getNextPageParam:p=>p.next_cursor??undefined,
 refetchInterval:q=>q.state.error?false:q.state.data?.pages.some(p=>p.items.some(r=>active(r.state)))?5000:false});
 const cooling=useCooldown([history.error]),denied=blocked||deniesRead(history.error);
 const view=useReadSnapshot(subject,`${state}:${kind}`,history.isSuccess?history.data:undefined,denied);
 useEffect(()=>{if(deniesRead(history.error))onDenied();},[history.error]);
 function filter(key:string,value:string){setParams(current=>{const next=new URLSearchParams(current);value?next.set(key,value):next.delete(key);return next;},{replace:true});}
 const rows=[...new Map(view.data?.pages.flatMap(p=>p.items).map(r=>[r.request_id,r])??[]).values()];
 return <section id="artifact-job-journal" className="data-download-journal" aria-label={w('Job journal','Журнал заданий')} onKeyDown={e=>{if(e.key==='Escape'){e.stopPropagation();onClose();}}}>
 <header className={history.isFetching&&history.isSuccess&&!denied&&!cooling?'artifact-quiet-read':undefined}><h2>{w('Job journal','Журнал заданий')}</h2><span>{w('All exchanges','Все биржи')}</span><RefreshButton pending={history.isFetching} disabled={denied||cooling} onClick={()=>void history.refetch()}/><button className="library-icon-action work-icon" aria-label={w('Close journal','Закрыть журнал')} onClick={onClose}><X/></button></header>
 <div className="data-journal-filters"><label><span className="sr-only">{w('Job state','Состояние задания')}</span><select className="data-select" value={state} disabled={denied||cooling} onChange={e=>filter('artifactstate',e.target.value)}><option value="">{w('All states','Все состояния')}</option>{Object.entries(labels.states).map(([key,label])=><option key={key} value={key}>{label}</option>)}</select><ChevronDown aria-hidden="true"/></label><label><span className="sr-only">{w('Job type','Тип задания')}</span><select className="data-select" value={kind} disabled={denied||cooling} onChange={e=>filter('artifactkind',e.target.value)}><option value="">{w('All types','Все типы')}</option>{Object.entries(labels.kinds).map(([key,label])=><option key={key} value={key}>{label}</option>)}</select><ChevronDown aria-hidden="true"/></label></div>
 <div className="data-read-status"><ReadFeedback pending={false} retained={view.retained} initial={!view.data&&!denied} error={history.error}/></div>
 <div className="data-journal-scroll" tabIndex={0} aria-label={w('Job history rows','Строки журнала заданий')}><table><thead><tr>{[w('Task / source','Задача / источник'),w('State','Состояние'),w('Stage','Этап'),w('Created · UTC','Создано · UTC')].map(t=><th key={t} scope="col">{t}</th>)}</tr></thead><tbody>{rows.map(r=>{const next=new URLSearchParams(params);next.set('artifactjob',r.request_id);return <tr key={r.request_id} className={params.get('artifactjob')===r.request_id?'selected':''}><td>{view.retained?<span>{r.coordinate.symbol} · {labels.kinds[r.kind]}</span>:<Link to={`/data?${next}`} onClick={onSelect}>{r.coordinate.symbol} · {labels.kinds[r.kind]}</Link>}<small>{r.coordinate.exchange} · {r.coordinate.market_type}</small></td><td><Status state={r.state}/></td><td>{phase(r,labels)}</td><td><time dateTime={r.created_at}>{formatTime(r.created_at,i18n.language)}</time></td></tr>;})}</tbody></table>
 {!rows.length&&history.isSuccess&&!denied&&<p className="artifact-journal-empty">{w('No jobs match these filters.','Заданий с такими фильтрами нет.')}</p>}
 {history.hasNextPage&&<button className="data-journal-more" disabled={history.isFetching||view.retained||denied||cooling} onClick={()=>void history.fetchNextPage()}>{w('Older jobs','Предыдущие задания')}</button>}</div></section>;
}

export function ArtifactJobInspector({subject,id,blocked,onDenied,onBack,hasInstrument}:{subject:string;id:string;blocked:boolean;onDenied:()=>void;onBack:()=>void;hasInstrument:boolean}){
 const w=useWords(),labels=useLabels(),{i18n}=useTranslation(),[showEvents,setShowEvents]=useState(false);
 const job=useQuery({queryKey:['artifacts',subject,'request',id],enabled:!!id&&!blocked,retry:false,refetchOnWindowFocus:false,queryFn:({signal})=>readArtifactRequest(id,signal),refetchInterval:q=>q.state.error?false:q.state.data&&active(q.state.data.state)?2000:false});
 const events=useQuery({queryKey:['artifacts',subject,'events',id],enabled:showEvents&&!blocked&&job.isSuccess,retry:false,refetchOnWindowFocus:false,queryFn:({signal})=>readArtifactEvents(id,signal),refetchInterval:q=>q.state.error?false:job.data&&active(job.data.state)?5000:false});
 const cooling=useCooldown([job.error,events.error]),denied=blocked||deniesRead(job.error)||deniesRead(events.error);
 const view=useReadSnapshot(subject,id,job.isSuccess?job.data:undefined,denied),request=view.data;
 useEffect(()=>{if(deniesRead(job.error)||deniesRead(events.error))onDenied();},[job.error,events.error]);
 return <><div className={`work-actions${job.isSuccess&&!denied&&!cooling?' artifact-quiet-read':''}`}><button className="work-link-button" onClick={onBack}>{hasInstrument?w('Back to instrument','К инструменту'):w('Back to catalog','К каталогу')}</button><RefreshButton pending={job.isFetching||events.isFetching} disabled={denied||cooling} onClick={()=>{void job.refetch();if(showEvents)void events.refetch();}}/></div><ReadFeedback pending={false} retained={view.retained} initial={!request&&!denied} error={job.error}/>
 {request&&<><header className="data-inspector-heading"><div><h2>{request.coordinate.symbol}</h2><p>{request.coordinate.exchange} · {request.coordinate.market_type}</p></div></header><section className="data-inspector-section"><h3>{labels.kinds[request.kind]}</h3><Status state={request.state}/><dl className="artifact-facts"><div><dt>{w('Created · UTC','Создано · UTC')}</dt><dd>{formatTime(request.created_at,i18n.language)}</dd></div><div><dt>{w('Last stage','Последний этап')}</dt><dd>{phase(request,labels)}</dd></div></dl>
 {request.reason_code&&<div className="notice error"><p>{request.reason_code==='source_unavailable'||request.reason_code==='source_or_storage_unavailable'?w('Candles or file storage are unavailable.','Свечи или хранилище файлов недоступны.'):w('The job could not finish.','Задание не удалось завершить.')}</p><details><summary>{w('Error code','Код ошибки')}</summary><code>{request.reason_code}</code></details></div>}
 </section><section className="data-inspector-section"><details open={showEvents} onToggle={e=>setShowEvents(e.currentTarget.open)}><summary>{w('Event history','История событий')}</summary><ReadFeedback pending={false} initial={showEvents&&!events.data} error={events.error}/>{!view.retained&&events.data&&!denied&&<ol className="artifact-job-events">{events.data.items.filter((event,index,items)=>index===0||event.state!==items[index-1].state||event.reason_code!==items[index-1].reason_code).map((event,index)=><li key={index}><Status state={event.state}/><time dateTime={event.observed_at}>{formatTime(event.observed_at,i18n.language)} · UTC</time>{event.reason_code&&<details><summary>{w('Error code','Код ошибки')}</summary><code>{event.reason_code}</code></details>}</li>)}</ol>}</details></section></>}
 </>;
}
