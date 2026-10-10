import {useState} from 'react';
import {Link} from 'react-router';
import {useQuery} from '@tanstack/react-query';
import {ChevronDown,Database} from 'lucide-react';
import {readArtifactRequests} from './artifacts-api';
import {useReadSnapshot} from './read-snapshot';
import {deniesRead,ReadFeedback,useWords} from './workspace-ui';

/** A bounded view of recent requests, never presented as the global queue count. */
export function ArtifactActivity({subject}:{subject:string}) {
 const w=useWords(),[open,setOpen]=useState(false);
 const query=useQuery({queryKey:['private',subject,'storage-activity'],queryFn:({signal})=>readArtifactRequests('',signal),retry:false,refetchInterval:q=>q.state.error?false:15000});
 const snapshot=useReadSnapshot(subject,'storage-activity',query.isSuccess?query.data:undefined,deniesRead(query.error));
 const items=snapshot.data?.items??[],running=items.filter(r=>r.state==='running'),waiting=items.filter(r=>['queued','retry_wait','waiting_for_source'].includes(r.state));
 const state=query.error?w('Status unavailable','Состояние недоступно'):!snapshot.data?w('Checking…','Проверяем…'):running.length?w('Calculations in progress','Выполняются расчёты'):w('No running tasks in recent requests','Среди последних запросов нет выполняемых');
 const stateName=(state:string)=>({running:w('Running','Выполняется'),queued:w('Queued','В очереди'),retry_wait:w('Waiting to retry','Ожидает повтора'),waiting_for_source:w('Waiting for candles','Ожидает свечей')})[state]??state;
 return <section className="storage-activity"><button type="button" className="storage-activity-toggle" aria-expanded={open} aria-controls="storage-activity-details" onClick={()=>setOpen(!open)}><ChevronDown aria-hidden="true"/><strong>{w('Background calculations','Фоновые расчёты')}</strong><span>{state}</span></button>
 {open&&<div id="storage-activity-details" className="storage-activity-details"><ReadFeedback pending={query.isFetching} initial={!snapshot.data} retained={snapshot.retained} error={query.error}/>
 {snapshot.data&&<><p className="storage-hint">{w('Latest 30 requests','Последние 30 запросов')} · {w('Running','Выполняются')}: {running.length} · {w('Waiting','Ожидают')}: {waiting.length}</p>
 {[...running,...waiting].slice(0,3).map(item=><div className="storage-task" key={item.request_id}><Database aria-hidden="true"/><div><strong>{item.coordinate.symbol}</strong><p>{item.coordinate.exchange} · {item.coordinate.market_type} · {item.kind==='base'?w('Base data','Базовые данные'):w('Indicators','Индикаторы')}</p></div><span>{stateName(item.state)}</span></div>)}</>}
 <div className="storage-activity-links"><Link className="work-link-button" to="/monitoring/backtest-artifact-publisher">{w('Manage calculation service','Управление сервисом расчётов')}</Link><Link to="/data?view=artifacts&journal=1">{w('Request history','История запросов')}</Link>{query.error&&<button type="button" disabled={query.isFetching} onClick={()=>void query.refetch()}>{w('Try again','Повторить')}</button>}</div>
 <p className="storage-hint">{w('Start and stop the service in Monitoring. Pausing an individual calculation is not yet supported.','Запуск и остановка сервиса доступны в мониторинге. Пауза отдельного расчёта пока не поддерживается.')}</p>
 </div>}
 </section>;
}
