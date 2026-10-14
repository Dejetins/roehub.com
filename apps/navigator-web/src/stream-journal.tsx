import {useEffect} from 'react';
import {useSearchParams} from 'react-router';
import {useInfiniteQuery} from '@tanstack/react-query';
import {useTranslation} from 'react-i18next';
import {z} from 'zod';
import {RefreshCw,X} from 'lucide-react';
import type {Market} from './connections-api';
import {marketRead} from './market-data-api';
import {useReadSnapshot} from './read-snapshot';
import {deniesRead,formatTime,ReadFeedback,useWords} from './workspace-ui';

const schema=z.object({items:z.array(z.object({event_id:z.number(),market_id:z.number(),symbol:z.string(),event_type:z.string(),occurred_at:z.string(),reason:z.string().nullable(),error_code:z.string().nullable(),attempt:z.number(),start_at:z.string().nullable(),end_at:z.string().nullable()})),next_cursor:z.number().nullable(),retention_days:z.number()});
const labels:Record<string,[string,string]>={connection_opened:['Connection established','Соединение установлено'],connection_lost:['Connection lost; reconnecting','Соединение потеряно; переподключаемся'],recovery_queued:['Gap recovery queued','Дозагрузка пропуска запланирована'],recovery_progress:['History interval checked','Участок истории проверен'],recovery_waiting:['Recovery waiting to retry','Дозагрузка ожидает повторной попытки'],recovery_failed:['Recovery needs attention','Дозагрузка требует внимания'],recovery_closed:['Recovery task closed','Задание восстановления закрыто']};
const errors:Record<string,[string,string]>={storage_busy:['Storage query slots are busy','Все слоты запросов хранилища заняты'],source_unavailable:['Exchange unavailable','Биржа недоступна'],storage_unavailable:['Storage temporarily unavailable','Хранилище временно недоступно'],storage_memory_pressure:['Storage memory limit reached','Достигнут лимит памяти хранилища'],source_rate_limited:['Exchange request limit reached','Достигнут лимит запросов биржи'],source_history_incomplete:['Exchange history is incomplete','История биржи неполна']};

export function StreamJournal({subject,markets,blocked,onDenied,onClose,onJobs}:{subject:string;markets:Market[];blocked:boolean;onDenied:()=>void;onClose:()=>void;onJobs:()=>void}){
 const w=useWords(),{i18n}=useTranslation(),[params,setParams]=useSearchParams();
 const market=params.get('eventmarket')??params.get('market')??String(markets[0]?.market_id??1);
 const query=useInfiniteQuery({queryKey:['stream-events',subject,market],enabled:!blocked,initialPageParam:null as number|null,maxPages:5,retry:false,
  queryFn:({signal,pageParam})=>marketRead(`workspace/stream-events?market_id=${encodeURIComponent(market)}&limit=30${pageParam?`&before=${pageParam}`:''}`,schema,signal),getNextPageParam:p=>p.next_cursor??undefined});
 const denied=deniesRead(query.error),view=useReadSnapshot(subject,market,query.isSuccess?query.data:undefined,blocked||denied);
 useEffect(()=>{if(denied)onDenied();},[denied,onDenied]);
 return <section id="data-download-journal" className="data-download-journal" aria-label={w('Stream event journal','Журнал событий потока')} onKeyDown={e=>{if(e.key==='Escape'){e.stopPropagation();onClose();}}}>
  <header><h2>{w('Stream events','События потока')}</h2><button className="work-link-button" onClick={onJobs}>{w('Downloads','Загрузки')}</button><span>{w('30 days','30 дней')}</span><button className="library-icon-action work-icon" aria-label={w('Refresh events','Обновить события')} disabled={query.isFetching||blocked} onClick={()=>void query.refetch()}><RefreshCw aria-hidden="true"/></button><button className="library-icon-action work-icon" aria-label={w('Close journal','Закрыть журнал')} onClick={onClose}><X aria-hidden="true"/></button></header>
  <div className="data-journal-filters"><label>{w('Source','Источник')}<select value={market} onChange={e=>{const next=new URLSearchParams(params);next.set('eventmarket',e.target.value);setParams(next,{replace:true});}}>{markets.map(m=><option key={m.market_id} value={m.market_id}>{m.exchange_name} · {m.market_type}</option>)}</select></label></div>
  <ReadFeedback pending={query.isFetching} initial={!view.data&&!blocked} retained={view.retained} error={query.error}/>
  <div className="data-journal-scroll" tabIndex={0} aria-label={w('Stream events','События потока')}><table><thead><tr><th>{w('Time · UTC','Время · UTC')}</th><th>{w('Instrument','Инструмент')}</th><th>{w('Event','Событие')}</th><th>{w('Details','Подробности')}</th></tr></thead><tbody>{view.data?.pages.flatMap(p=>p.items).map(e=><tr key={e.event_id}><td>{formatTime(e.occurred_at,i18n.language)}</td><td>{e.symbol}</td><td>{w(...(labels[e.event_type]??['Stream state changed','Состояние потока изменилось']))}</td><td>{e.error_code&&<span>{w(...(errors[e.error_code]??['Temporary processing failure','Временная ошибка обработки']))}</span>}{e.attempt>0&&<small>{w('Attempt','Попытка')}: {e.attempt}</small>}{e.start_at&&e.end_at&&<small>{formatTime(e.start_at,i18n.language)} — {formatTime(e.end_at,i18n.language)}</small>}</td></tr>)}</tbody></table>
   {view.data&&!view.data.pages.some(p=>p.items.length)&&<p className="work-empty">{w('No recorded events for this source. Events are collected from journal activation.','Для этого источника событий пока нет. Запись ведётся с момента включения журнала.')}</p>}
   {query.hasNextPage&&<button disabled={query.isFetching||blocked||view.retained} onClick={()=>void query.fetchNextPage()}>{w('Earlier events','Более ранние события')}</button>}
  </div>
 </section>;
}
