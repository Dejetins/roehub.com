import {useEffect,useMemo,useRef,useState,type FormEvent} from 'react';
import {Link,useSearchParams} from 'react-router';
import {useQueries,useQuery,useQueryClient} from '@tanstack/react-query';
import {useTranslation} from 'react-i18next';
import {ArrowDown,ArrowUp,ArrowUpDown,CheckCircle2,ChevronLeft,ChevronRight,Circle,Clock3,Download,LockKeyhole,Search,TriangleAlert,X} from 'lucide-react';
import {readMarkets,type Market} from './connections-api';
import {ApiError} from './api';
import {closedDayRange,rangeIso,selectInstrument,validateCoverageRange,validateWorkRange,type WorkJob} from './market-data-api';
import {createDataReadRecovery,transientDataRead,readDataInspector,readExchangeCatalog,readHistoryBounds,type CatalogRow,type DataInspector} from './data-workspace-api';
import {useReadSnapshot} from './read-snapshot';
import {CollectionLibrary,deniesRead,DirtyGuard,Field,formatTime,ReadFeedback,RefreshButton,useWords,WorkError} from './workspace-ui';
import {useWorkInterval,useWorkSubmission,workState} from './market-data-ui';
import {DownloadJournal,DownloadJobDetails,DownloadJobActions,StandaloneDownloadJob,useDownloadJob,JobState,DownloadEtaLabel} from './data-downloads';
import {ChartDisplayMenu} from './chart-display-menu';
import './data-workspace.css';
import {DataMode} from './data-mode';
import {ArtifactsPage} from './data-artifacts';

import {ExchangeLinkContent,exchangeName} from './data-exchange-link';
const itemKey=(market:number,symbol:string)=>`${market}:${symbol}`;
const activeJob=(state:string|undefined)=>!!state&&['queued','running','pause_requested','paused','retry_wait','cancel_requested'].includes(state);

export function DataPage({subject}:{subject:string}) {
 const [params]=useSearchParams();
 return params.get('view')==='artifacts'?<ArtifactsPage key={subject} subject={subject}/>:<CandlesPage key={subject} subject={subject}/>;
}

function CandlesPage({subject}:{subject:string}) {
 const w=useWords(),{i18n}=useTranslation(),[params,setParams]=useSearchParams(),client=useQueryClient(),interval=useWorkInterval(subject);
 const defaults=useRef(closedDayRange()).current;
 const market=Number(params.get('market')??'1'),symbol=params.get('symbol')??'';
 const sources=useQuery({queryKey:['market-sources',subject],queryFn:({signal})=>readMarkets(signal),retry:false});
 const exchange=params.get('exchange')??sources.data?.items.find(m=>m.market_id===market)?.exchange_name??'binance';
 const q=params.get('q')??'',marketFilter=params.get('type')??'',history=params.get('history')??'all',stream=params.get('stream')??(params.get('selected')==='1'?'on':'all');
 const sort=params.get('sort')??'symbol',descending=params.get('direction')==='desc',offset=Math.max(0,Number(params.get('offset'))||0);
 const from=defaults.from,to=defaults.to,valid=true;
 const requestParams=new URLSearchParams({history_scope:'full',exchange,q,history,stream,sort,descending:String(descending),offset:String(offset),limit:'50',...(marketFilter?{market_id:marketFilter}:{}),...(params.get('catalog')?{snapshot:params.get('catalog')!}:{}),...(valid?{start_at:rangeIso(from),end_at:rangeIso(to)}:{})});
 const collectionKey=requestParams.toString();
 const catalogRecovery=useMemo(createDataReadRecovery,[subject,collectionKey]);
 const query=useQuery({queryKey:['exchange-catalog',subject,collectionKey],enabled:!!sources.data&&valid,retry:false,
  queryFn:({signal})=>readExchangeCatalog(requestParams,signal),refetchInterval:read=>catalogRecovery(read.state,read.state.data?.items.some(row=>row.effective)?2000:read.state.data?.items.some(row=>activeJob(row.job?.state))?5000:interval)});
 const detailKey=`${market}:${symbol}:${from}:${to}`;
 const inspectorRecovery=useMemo(createDataReadRecovery,[subject,detailKey]);
 const detail=useQuery({queryKey:['data-inspector',subject,detailKey],enabled:!!symbol&&valid,retry:false,
  queryFn:({signal})=>readDataInspector(market,symbol,from,to,signal),refetchInterval:read=>inspectorRecovery(read.state,(read.state.data?.instrument.effective||read.state.data?.coverage.state==='unknown')?2000:query.data?.items.some(row=>row.market_id===market&&row.symbol===symbol&&activeJob(row.job?.state))?5000:interval)});
 const [commandDenied,setCommandDenied]=useState(false);
 const accessLost=commandDenied||[sources.error,query.error,detail.error].some(e=>e instanceof ApiError&&[401,403].includes(e.status??0));
 const collection=useReadSnapshot(subject,collectionKey,query.isSuccess?query.data:undefined,accessLost||deniesRead(query.error));
 const inspector=useReadSnapshot(subject,detailKey,detail.isSuccess?detail.data:undefined,!symbol||accessLost||deniesRead(detail.error));
 const visible=collection.data,view=inspector.data;
 // Share the inspector's complete facts only with its matching current row.
 // Membership/order and command eligibility still belong to the catalog snapshot.
 const selectedFacts=view&&!inspector.retained&&collection.displayedKey===collectionKey?view:undefined;
 const displayRow=(row:CatalogRow):CatalogRow=>selectedFacts&&row.market_id===selectedFacts.instrument.market_id&&row.symbol===selectedFacts.instrument.symbol?{
  ...row,effective:selectedFacts.instrument.effective,selected:selectedFacts.instrument.selected,
  strategy_pinned:selectedFacts.instrument.strategy_pinned,strategies:selectedFacts.instrument.strategies,
  stream_state:selectedFacts.instrument.stream_state,coverage_state:selectedFacts.coverage.state,
  coverage_percent:selectedFacts.coverage.coverage_percent,actual_candles:selectedFacts.coverage.actual_candles,
  expected_candles:selectedFacts.coverage.expected_candles,history_start_at:selectedFacts.coverage.start_at,
  history_end_at:selectedFacts.coverage.end_at,
 }:row;
 const currentRows=visible?.items.map(displayRow)??[];
 const locked=inspector.retained||!view||!!detail.error||!valid||accessLost;
 const exchanges=[...new Set(sources.data?.items.map(m=>m.exchange_name)??[])];
 const selectedSource=sources.data?.items.find(m=>m.market_id===view?.instrument.market_id);
 const [selected,setSelected]=useState<CatalogRow[]>([]);
 const [manualRefreshing,setManualRefreshing]=useState(false);
 useEffect(()=>setSelected([]),[subject,exchange]);
 useEffect(()=>setSelected(current=>current.filter(row=>row.market_id===market)),[market]);
 const [focusDownload,setFocusDownload]=useState<string>();
 const journalOpen=params.get('journal')==='1',journalTrigger=useRef<HTMLButtonElement>(null);
 function closeJournal(){change({journal:''});journalTrigger.current?.focus();}
 function jobHref(job:WorkJob){const next=new URLSearchParams(params);if(job.market_id!==market){for(const k of ['exchange','type','offset','catalog','q','history','stream'])next.delete(k);}next.set('market',String(job.market_id));next.set('job',job.job_id);job.symbols[0]?next.set('symbol',job.symbols[0]):next.delete('symbol');return `/data?${next}`;}
 const scroll=useRef<HTMLDivElement>(null),lastCollection=useRef(collectionKey);
 useEffect(()=>{if(query.isSuccess&&lastCollection.current!==collectionKey){scroll.current?.scrollTo?.({top:0});lastCollection.current=collectionKey;}},[query.isSuccess,collectionKey]);
 function change(values:Record<string,string>,replace=true){const next=new URLSearchParams(params);for(const [k,v] of Object.entries(values)){v?next.set(k,v):next.delete(k);}setParams(next,{replace});}
 function filter(values:Record<string,string>){setSelected([]);change({...values,offset:'',catalog:''});}
 function select(row:CatalogRow){setSelected([]);change({market:String(row.market_id),symbol:row.symbol,job:row.job?.job_id??''},false);}
 function href(row:CatalogRow){const next=new URLSearchParams(params);next.set('market',String(row.market_id));next.set('symbol',row.symbol);row.job?next.set('job',row.job.job_id):next.delete('job');return `/data?${next}`;}
 function chooseExchange(name:string){setSelected([]);change({exchange:name,market:String(sources.data?.items.find(m=>m.exchange_name===name)?.market_id??1),symbol:'',job:'',type:'',offset:'',catalog:'',q:'',history:'',stream:'',selected:''},false);}
 function check(row:CatalogRow,on:boolean){if(on){if(selected.length>=50||selected.some(s=>s.market_id!==row.market_id))return;setSelected([...selected,row]);change({market:String(row.market_id),symbol:row.symbol,job:row.job?.job_id??''},false);}else setSelected(selected.filter(s=>itemKey(s.market_id,s.symbol)!==itemKey(row.market_id,row.symbol)));}
 async function refresh(){
  if(manualRefreshing)return;
  setManualRefreshing(true);
  try{
   setSelected([]);
   if(params.get('catalog')){
    const latest=new URLSearchParams(requestParams);latest.delete('snapshot');latest.set('offset','0');
    change({catalog:'',offset:''});
    // Await the new page's read too; its errors remain visible through useQuery.
    await client.fetchQuery({queryKey:['exchange-catalog',subject,latest.toString()],queryFn:({signal})=>readExchangeCatalog(latest,signal),retry:false}).catch(()=>undefined);
   }else await query.refetch();
   if(symbol)await detail.refetch();
  }finally{setManualRefreshing(false);}
 }
 async function changed(){await Promise.all([client.invalidateQueries({queryKey:['exchange-catalog',subject]}),client.invalidateQueries({queryKey:['data-inspector',subject]}),client.invalidateQueries({queryKey:['instrument-catalog',subject]}),client.invalidateQueries({queryKey:['data-jobs',subject]})]);}
 const [savedWork,setSavedWork]=useState<WorkJob>();
 const catalogJobId=savedWork?.kind==='catalog_refresh'?savedWork.job_id:undefined;
 const catalogCommand=useWorkSubmission(job=>{setSavedWork(job);client.setQueryData(['data-job',subject,job.job_id],job);void changed();});
 const {query:catalogJob}=useDownloadJob(subject,catalogJobId,()=>setCommandDenied(true));
 useEffect(()=>{if(deniesRead(catalogCommand.error))setCommandDenied(true);},[catalogCommand.error]);
 useEffect(()=>{if(savedWork&&!catalogCommand.pending){const range:Record<string,string>=savedWork.start_at&&savedWork.end_at&&validateCoverageRange(savedWork.start_at.slice(0,16),savedWork.end_at.slice(0,16))?{from:savedWork.start_at.slice(0,16),to:savedWork.end_at.slice(0,16)}:{};change({job:savedWork.job_id,requestkey:'',symbol:savedWork.symbols[0]??'',market:String(savedWork.market_id),...range});setSelected([]);}},[savedWork?.job_id,catalogCommand.pending]);
 const changeSort=(field:string)=>filter({sort:field,direction:sort===field&&!descending?'desc':''});
 const sortLabel=(field:string,label:string)=><button type="button" onClick={()=>changeSort(field)}>{label}{sort===field?(descending?<ArrowDown/>:<ArrowUp/>):<ArrowUpDown/>}</button>;
 return <>
  <CollectionLibrary title={w('Exchanges','Биржи')}><nav className="work-categories data-exchanges" aria-label={w('Exchanges','Биржи')}>
   {exchanges.map(name=><Link key={name} aria-label={exchangeName(name)} to={`/data?exchange=${name}&market=${sources.data?.items.find(m=>m.exchange_name===name)?.market_id}`} aria-current={exchange===name?'page':undefined} onClick={event=>{if(!event.ctrlKey&&!event.metaKey&&!event.shiftKey&&!event.altKey){event.preventDefault();chooseExchange(name);}}}><ExchangeLinkContent name={name}/></Link>)}
  </nav><ReadFeedback pending={sources.isFetching} initial={!sources.data} error={sources.error}/></CollectionLibrary>
  <h1 id="workspace-heading" className="sr-only" tabIndex={-1}>{w('Data','Данные')}</h1>
  <section className="data-catalog panel" aria-label={w('Instrument catalog','Каталог инструментов')}>
   <DataMode/>
   <div className="data-catalog-tools">
    <label className="data-search"><Search aria-hidden="true"/><span className="sr-only">{w('Search instruments','Поиск инструментов')}</span><input type="search" maxLength={64} value={q} placeholder={w('Search instruments','Поиск инструментов')} onChange={e=>filter({q:e.target.value.toUpperCase()})}/></label>
    <DataFilter label={w('Market','Рынок')} value={marketFilter} options={[['',w('All markets','Все рынки')],...(sources.data?.items.filter(m=>m.exchange_name===exchange).map(m=>[String(m.market_id),marketName(m.market_type,w)] as [string,string])??[])]} onChange={value=>filter({type:value})}/>
    <DataFilter label={w('History','История')} value={history} options={[["all",w('History: all','История: все')],['complete',w('Downloaded','Загружено')],['partial',w('Partial','Частично')],['empty',w('No data','Нет данных')],['active',w('Downloading / queued','Загрузка / очередь')],['failed',w('Failed','Ошибка')]]} onChange={value=>filter({history:value})}/>
    <DataFilter label={w('Stream','Поток')} value={stream} options={[["all",w('Stream: all','Поток: все')],['on',w('Stream: enabled','Поток: включён')],['off',w('Stream: disabled','Поток: выключен')],['pinned',w('Required by strategy','Требуется стратегии')]]} onChange={value=>filter({stream:value})}/>
   </div>
   <div className="data-catalog-context"><span>{w('All available history · 1m · through the latest closed minute','Вся доступная история · 1m · до последней закрытой минуты')}</span>{(q||marketFilter||history!=='all'||stream!=='all')&&<button className="data-text-action" onClick={()=>filter({q:'',type:'',history:'',stream:'',selected:''})}>{w('Reset','Сбросить')}</button>}<div className="work-actions"><button ref={journalTrigger} className="work-link-button" aria-expanded={journalOpen} aria-controls="data-download-journal" onClick={()=>journalOpen?closeJournal():change({journal:'1'})}>{journalOpen?w('Hide log','Скрыть журнал'):w('Show log','Показать журнал')}</button><RefreshButton pending={manualRefreshing} onClick={()=>void refresh()}/></div></div>
   <div className="data-read-status"><DataReadFeedback retained={collection.retained} initial={!visible&&valid} error={query.error} catalog/>{!valid&&<p role="alert">{w('Choose a closed period of up to 7 days.','Выберите завершённый период не более 7 дней.')}</p>}</div>
   {!!visible?.missing_market_ids.length&&<div className="data-catalog-missing"><span>{w('Some markets have no catalog snapshot.','Для части рынков каталог ещё не загружен.')}</span>{visible.missing_market_ids.map(id=><button key={id} disabled={catalogCommand.pending||catalogCommand.unresolved||activeJob(catalogJob.data?.state)||collection.retained||accessLost} onClick={()=>void catalogCommand.submit({kind:'catalog_refresh',market_id:id})}>{w('Load catalog','Загрузить каталог')} · {marketName(sources.data?.items.find(m=>m.market_id===id)?.market_type??'',w)}</button>)}</div>}
   <WorkError error={!view?catalogCommand.error:undefined}/><WorkError error={catalogJob.error}/>{catalogJob.data&&!accessLost&&sources.data?.items.find(m=>m.market_id===catalogJob.data?.market_id)?.exchange_name===exchange&&<Link className="data-job-result" to={jobHref(catalogJob.data)}>{workState(catalogJob.data.state,w)} · {marketName(sources.data?.items.find(m=>m.market_id===catalogJob.data?.market_id)?.market_type??'',w)}<ChevronRight/></Link>}{catalogCommand.unresolved&&!view&&<button disabled={catalogCommand.pending} onClick={()=>void catalogCommand.reconcile()}>{w('Check request','Проверить задание')}</button>}
   <div className="data-catalog-scroll" ref={scroll} tabIndex={0} aria-label={w('Catalog rows','Строки каталога')}><table className="data-catalog-table"><thead><tr><th className="data-check-column"><input type="checkbox" aria-label={w('Select this page for download','Выбрать страницу для загрузки')} title={w('Select up to 50 available instruments from one market','Выбрать до 50 доступных инструментов одного рынка')} disabled={collection.retained||accessLost||!currentRows.some(row=>row.market_id===Number(marketFilter||market)&&!activeJob(row.job?.state)&&!row.effective)} checked={!!selected.length&&!!currentRows.filter(row=>row.market_id===Number(marketFilter||market)&&!activeJob(row.job?.state)&&!row.effective).every(row=>selected.some(s=>itemKey(s.market_id,s.symbol)===itemKey(row.market_id,row.symbol)))} onChange={e=>{const batch=e.target.checked?currentRows.filter(row=>row.market_id===Number(marketFilter||market)&&!activeJob(row.job?.state)&&!row.effective).slice(0,50):[];setSelected(batch);if(batch.length)change({market:String(batch[0].market_id),symbol:batch[0].symbol,job:batch[0].job?.job_id??''},false);}}/></th>{[['symbol',w('Instrument','Инструмент')],['market',w('Market','Рынок')],['coverage',w('History / download','История / загрузка')],['stream',w('Stream','Поток')]].map(([field,label])=><th key={field} scope="col" aria-sort={sort===field?(descending?'descending':'ascending'):'none'}>{sortLabel(field,label)}</th>)}</tr></thead><tbody>
    {visible?.items.map(storedRow=>{const row=displayRow(storedRow),stale=collection.retained&&row===storedRow;const key=itemKey(row.market_id,row.symbol);return <tr key={key} className={market===row.market_id&&symbol===row.symbol?'selected':''} data-selectable={!collection.retained&&!accessLost} aria-current={market===row.market_id&&symbol===row.symbol?'true':undefined} aria-disabled={collection.retained||accessLost} tabIndex={collection.retained||accessLost?-1:0} onKeyDown={e=>{
     if(e.target!==e.currentTarget||collection.retained||accessLost||e.altKey||e.ctrlKey||e.metaKey||e.shiftKey)return;
     if(e.key==='Enter'||e.key===' '){e.preventDefault();select(row);}
    }} onClick={e=>{
     if(collection.retained||accessLost||e.defaultPrevented||e.button!==0||e.metaKey||e.ctrlKey||e.shiftKey||e.altKey)return;
     // Keep nested controls and native links responsible for their own actions.
     if(e.target instanceof Element&&e.target.closest('a,button,input,select,textarea,label,summary,[role="button"],[role="checkbox"],[role="switch"],[contenteditable="true"]'))return;
     select(row);
    }}><td><input type="checkbox" aria-label={`${w('Download','Загрузить')} ${row.symbol} · ${marketName(row.market_type,w)}`} checked={selected.some(s=>itemKey(s.market_id,s.symbol)===key)} title={row.effective?w('Streaming manages this history automatically.','Стриминг поддерживает историю автоматически.'):undefined} disabled={collection.retained||accessLost||(activeJob(row.job?.state)||row.effective)||(!selected.some(s=>itemKey(s.market_id,s.symbol)===key)&&(selected.length>=50||selected.some(s=>s.market_id!==row.market_id)))} onChange={e=>check(row,e.target.checked)}/></td><td><span className="data-instrument-name">{row.symbol}</span></td><td>{marketName(row.market_type,w)}</td><td><HistoryCell subject={subject} onDenied={()=>setCommandDenied(true)} row={row} jobHref={href(row)} onDownload={()=>{setFocusDownload(itemKey(row.market_id,row.symbol));select(row);}} disabled={collection.retained||accessLost}/></td><td><StreamStatus effective={row.effective} pinned={row.strategy_pinned} state={row.stream_state} complete={row.coverage_state==='complete'} stale={stale}/></td></tr>;})}
   </tbody></table>{visible&&!visible.items.length&&<p className="work-empty">{w('No matching instruments','Нет подходящих инструментов')}</p>}</div>
   <footer className="data-catalog-footer pagination library-pagination"><span className="navigator-page-count">{selected.length?`${w('Selected','Выбрано')}: ${selected.length} · ${w('one market, up to 50','один рынок, до 50')}`:visible?`${visible.total?visible.offset+1:0}–${Math.min(visible.offset+visible.limit,visible.total)} ${w('of','из')} ${visible.total.toLocaleString(i18n.language)}`:'—'}</span><div className="navigator-page-arrows"><button className="library-icon-action work-icon" title={w('Previous page','Предыдущая страница')} aria-label={w('Previous page','Предыдущая страница')} disabled={!visible||visible.offset===0||collection.retained||query.isFetching} onClick={()=>change({offset:String(Math.max(0,visible!.offset-50)),catalog:visible!.snapshot})}><ChevronLeft aria-hidden="true"/></button><button className="library-icon-action work-icon" title={w('Next page','Следующая страница')} aria-label={w('Next page','Следующая страница')} disabled={!visible||visible.offset+visible.limit>=visible.total||collection.retained||query.isFetching} onClick={()=>change({offset:String(visible!.offset+50),catalog:visible!.snapshot})}><ChevronRight aria-hidden="true"/></button></div></footer>
   {journalOpen&&<DownloadJournal subject={subject} markets={sources.data?.items??[]} blocked={accessLost} onDenied={()=>setCommandDenied(true)} onClose={closeJournal} onSelect={()=>setSelected([])} jobHref={jobHref}/>}
  </section>
  <aside className="navigator-inspector data-inspector" aria-label={w('Instrument settings','Настройки инструмента')}>
   <DataReadFeedback retained={inspector.retained} initial={!!symbol&&!view&&valid} error={detail.error}/>
   {view?<InstrumentInspector key={itemKey(view.instrument.market_id,view.instrument.symbol)} subject={subject} latestJob={visible?.items.find(row=>row.market_id===view.instrument.market_id&&row.symbol===view.instrument.symbol)?.job} markets={sources.data?.items??[]} requestJobId={params.get('job')??(savedWork?.market_id===view.instrument.market_id&&savedWork.symbols.includes(view.instrument.symbol)?savedWork.job_id:visible?.items.find(row=>row.market_id===view.instrument.market_id&&row.symbol===view.instrument.symbol)?.job?.job_id)} focusDownload={focusDownload===itemKey(view.instrument.market_id,view.instrument.symbol)} onFocused={()=>setFocusDownload(undefined)} value={view} source={selectedSource} command={catalogCommand} saved={savedWork?.kind==='candle_ingestion'&&savedWork.market_id===view.instrument.market_id?savedWork:undefined} locked={locked} from={from} to={to} selected={collection.retained?[]:selected} onClose={()=>{setSelected([]);change({symbol:'',job:''},false);}} onChanged={changed} onDenied={()=>setCommandDenied(true)}/>:!symbol&&params.get('job')&&!accessLost?<StandaloneDownloadJob subject={subject} id={params.get('job')!} markets={sources.data?.items??[]} onDenied={()=>setCommandDenied(true)} onChanged={changed} onClose={()=>change({job:''})}/>:!symbol?<p className="work-empty">{w('Select an instrument to manage its data.','Выберите инструмент для управления данными.')}</p>:null}
  </aside>
 </>;
}

function DataReadFeedback({retained,initial,error,catalog=false}:{retained:boolean;initial:boolean;error:unknown;catalog?:boolean}){
 const w=useWords(),recovering=transientDataRead(error);
 const local=!!error&&!deniesRead(error);
 return <><ReadFeedback pending={false} retained={retained} initial={initial&&!error} error={local?undefined:error}/>{local&&<p className="notice error" role="alert">{
  catalog?w('Could not update the instrument list: history coverage and stream states may be out of date.','Не удалось обновить список инструментов: заполненность истории и состояния потоков могут быть устаревшими.'):
   w('Could not update this instrument’s history and stream state.','Не удалось обновить историю и состояние потока этого инструмента.')
 }{' '}{recovering?w('Retrying automatically.','Повторим автоматически.'):w('Refresh to try again.','Повторите обновление.')}</p>}</>;
}

function marketName(type:string,w:ReturnType<typeof useWords>){return type==='spot'?w('Spot','Спот'):type==='futures'?w('Futures','Фьючерсы'):type;}
function rangeLabel(a:string,b:string,locale:string){const start=Date.parse(a.endsWith('Z')||/[+]\d\d:\d\d$/.test(a)?a:`${a}Z`),end=Date.parse(b.endsWith('Z')||/[+]\d\d:\d\d$/.test(b)?b:`${b}Z`);return Number.isFinite(start)&&Number.isFinite(end)?new Intl.DateTimeFormat(locale,{day:'numeric',month:'short',year:'numeric',hour:'2-digit',minute:'2-digit',timeZone:'UTC'}).formatRange(start,end):'—';}
function DataFilter({label,value,options,onChange}:{label:string;value:string;options:[string,string][];onChange:(value:string)=>void}){
 return <ChartDisplayMenu label={label} value={options.find(([key])=>key===value)?.[1]??options[0][1]} options={options.map(([key,text])=>({label:text,checked:key===value,onChange:()=>onChange(key)}))}/>;
}

function StreamStatus({effective,pinned,state,complete,stale=false}:{effective:boolean;pinned:boolean;state?:CatalogRow['stream_state'];complete:boolean;stale?:boolean}){
 const w=useWords(),active=!stale&&effective&&state==='running',failed=!stale&&effective&&['disconnected','blocked'].includes(state??'');
 const label=stale?w('Out of date','Нет обновления'):!effective?w('Disabled','Выключен'):state==='disconnected'?w('No connection','Нет связи'):state==='offline'?w('Stopped','Остановлен'):state==='blocked'?w('Error','Ошибка'):state==='retry_wait'?w('Retry pending','Ждём повтора'):state==='recovering'?w('Recovering','Дозагрузка'):active?(complete?w('Running','Работает'):w('Catching up','Дозагрузка')):w('Checking','Проверяем');
 return <span className={`data-stream-state ${failed?'failed':active&&complete?'enabled':''}`} title={label} aria-label={label}>{failed?<TriangleAlert aria-hidden="true"/>:pinned?<LockKeyhole aria-hidden="true"/>:<Circle aria-hidden="true"/>}{label}</span>;
}

function StreamMessage({item,complete,stale}:{item:DataInspector['instrument'];complete:boolean;stale:boolean}){const w=useWords();return <p id="data-stream-help" role={item.stream_state==='disconnected'?'alert':undefined}>{stale?w('The current stream state could not be verified.','Текущее состояние потока не удалось проверить.'):!item.effective?w('New candles are not added automatically. Stored history is preserved.','Новые свечи автоматически не добавляются. Сохранённая история остаётся.'):item.stream_state==='disconnected'?w('No connection to the exchange. New candles are not being received. Reconnecting automatically.','Нет соединения с биржей. Новые свечи не поступают. Пробуем подключиться повторно.'):item.stream_state==='offline'?w('Collector is offline. Streaming is requested, but new candles are not being collected.','Сборщик остановлен. Стриминг запрошен, но новые свечи не собираются.'):item.stream_state==='blocked'?(item.history_repair_error?w('Automatic recovery is blocked. Repair the missing history below.','Автовосстановление заблокировано. Дозагрузите недостающую историю ниже.'):w('Collection recovery failed. Check the collector service.','Восстановление сбора завершилось ошибкой. Проверьте сервис сборщика.')):item.stream_state==='retry_wait'?w('Temporary interruption. Recovery will retry automatically.','Временный сбой. Восстановление повторится автоматически.'):item.stream_state==='recovering'?w('Restoring missing candles automatically before continuing the stream.','Автоматически восстанавливаем пропуски перед продолжением потока.'):item.stream_state==='running'?(complete?w('New closed candles are saved automatically. History is up to date.','Новые закрытые свечи сохраняются автоматически. История актуальна.'):w('Collector is running. Waiting for missing candles; history updates automatically.','Сборщик работает. Ожидаем недостающие свечи; история обновляется автоматически.')):w('Streaming is requested. Collector status is unavailable.','Поток включён. Проверяем его состояние.')}</p>;}

function HistoryCell({subject,onDenied,row,jobHref,onDownload,disabled}:{subject:string;onDenied:()=>void;row:CatalogRow;jobHref:string;onDownload:()=>void;disabled:boolean}){
 const w=useWords(),summary=row.job,{query,view,eta}=useDownloadJob(subject,summary?.job_id,onDenied,undefined,true),job=view.data;
 const state=job?.state??summary?.state,active=activeJob(state),failed=state==='failed',controlled=active||failed;
 return <div className="data-history-cell"><div className="data-row-coverage">
  <span className={`data-progress ${row.coverage_state==='complete'?'complete':''}`} title={`${row.actual_candles} / ${row.expected_candles} · ${w('closed candles across the full available history','завершённых свечей за всю доступную историю')}`}>
   {active?<JobState compact {...(job??{state:state!})}/>:failed?<Link className="data-job-state failed data-job-failed" to={jobHref} aria-disabled={disabled} onClick={e=>{if(disabled)e.preventDefault();}}><TriangleAlert aria-hidden="true"/>{w('Failed','Ошибка')}</Link>:row.coverage_state==='complete'?<CheckCircle2 aria-hidden="true"/>:<Circle aria-hidden="true"/>}
   <span className="data-history-meter"><span>{row.effective&&['running','recovering','retry_wait'].includes(row.stream_state??'')&&row.coverage_state!=='complete'?w('Updating history','Обновление истории'):row.coverage_state==='unknown'?w('History start unknown','Начало истории неизвестно'):`${w('Full history','Вся история')}: ${row.coverage_percent}%`}</span>
   {row.coverage_percent!==null&&<progress max={100} value={row.coverage_percent} aria-label={w('Candle coverage','Покрытие свечей')}/>}
   {active&&<span className="data-history-eta" title={w('Estimated time until this download task completes','Оценка времени до завершения задания')}><DownloadEtaLabel eta={eta}/></span>}</span>
   {controlled&&query.error&&<TriangleAlert aria-label={w('Update unavailable','Нет обновления')}/>}
  </span>
  {controlled&&job?<DownloadJobActions compact subject={subject} job={job} locked={disabled||view.retained||!!query.error} onDenied={onDenied}/>:!controlled&&['unknown','empty','partial'].includes(row.coverage_state)&&!row.effective&&<button type="button" className="library-icon-action data-download-icon" aria-label={`${w('Download','Загрузить')} ${row.symbol} · ${marketName(row.market_type,w)}`} title={w('Download','Загрузить')} disabled={disabled} onClick={onDownload}><Download aria-hidden="true"/></button>}
 </div></div>;
}

function InstrumentInspector({subject,latestJob,markets,requestJobId,focusDownload,onFocused,value,source,command,saved,locked,from,to,selected,onClose,onChanged,onDenied}:{subject:string;latestJob:CatalogRow['job']|undefined;markets:Market[];requestJobId:string|undefined;focusDownload:boolean;onFocused:()=>void;value:DataInspector;source:Market|undefined;command:ReturnType<typeof useWorkSubmission>;saved:WorkJob|undefined;locked:boolean;from:string;to:string;selected:CatalogRow[];onClose:()=>void;onChanged:()=>Promise<void>;onDenied:()=>void}){
 const w=useWords(),{i18n}=useTranslation(),item=value.instrument,coverage=value.coverage,client=useQueryClient();
 const [pending,setPending]=useState(false),[error,setError]=useState<unknown>(),[unknown,setUnknown]=useState(false),busy=useRef(false);
 const [draft,setDraft]=useState<{key:string;from:string;to:string}>(),[validation,setValidation]=useState(false);
 const batchMatches=selected.every(row=>row.market_id===item.market_id);
 const names=selected.length?selected.map(r=>r.symbol):[item.symbol];
 const [manual,setManual]=useState(false),batch=names.length>1;
 const rangeKey=`${item.market_id}:${[...names].sort().join(',')}`;
 const bounds=useQueries({queries:names.map(name=>({queryKey:['history-bounds',subject,item.market_id,name],queryFn:({signal}:{signal:AbortSignal})=>readHistoryBounds(item.market_id,name,signal),enabled:!locked&&!manual,retry:false,staleTime:60_000,refetchInterval:(read:{state:{error:unknown;data?:{state:string}}})=>read.state.error?false:read.state.data&&['queued','running'].includes(read.state.data.state)?2000:false}))});
 const allReady=bounds.every(read=>read.data?.state==='ready'&&read.data.first_open_at);
 useEffect(()=>{if(allReady){void client.invalidateQueries({queryKey:['exchange-catalog',subject]});void client.invalidateQueries({queryKey:['data-inspector',subject]});}},[allReady]);
 const resolving=!manual&&bounds.some(read=>!read.error&&(!read.data||['queued','running'].includes(read.data.state)));
 const boundsError=bounds.find(read=>read.error)?.error,boundsDenied=bounds.some(read=>deniesRead(read.error));
 const defaultFrom=allReady?bounds.map(read=>read.data!.first_open_at!).sort()[0].slice(0,16):'';
 const defaultTo=bounds.map(read=>read.data?.end_at??'').filter(Boolean).sort()[0]?.slice(0,16)??'';
 const a=draft?.key===rangeKey?draft.from:defaultFrom,b=draft?.key===rangeKey?draft.to:defaultTo;
 const setA=(from:string)=>setDraft({key:rangeKey,from,to:b}),setB=(to:string)=>setDraft({key:rangeKey,from:a,to});
 const manualRepairAllowed=!item.effective||(item.stream_state==='blocked'&&!!item.history_repair_error);
 const syncing=item.effective&&['running','recovering','retry_wait'].includes(item.stream_state??'')&&coverage.state!=='complete';
 const downloadLocked=!manualRepairAllowed||locked||activeJob(latestJob?.state)||(resolving&&!manual)||command.pending||command.unresolved;
 useEffect(()=>{if(focusDownload&&!downloadLocked){const input=document.getElementById('data-download-from');input?.closest('details')?.setAttribute('open','');input?.focus();onFocused();}},[focusDownload,downloadLocked]);
 useEffect(()=>{if(boundsDenied)onDenied();},[boundsDenied]);
 const {query:job,view:jobView,eta}=useDownloadJob(subject,requestJobId,onDenied);
 const jobMatches=jobView.data?.market_id===item.market_id&&jobView.data.symbols.includes(item.symbol);
 useEffect(()=>{if(deniesRead(command.error))onDenied();},[command.error,job.error]);
 const dirty=!!draft&&draft.key===rangeKey&&(a!==defaultFrom||b!==defaultTo);
 async function toggle(){if(locked||item.strategy_pinned||busy.current||unknown)return;busy.current=true;setPending(true);setError(undefined);try{await selectInstrument(item,!item.effective);await onChanged();}catch(e){setError(e);setUnknown(e instanceof ApiError&&e.outcome==='unknown');if(deniesRead(e))onDenied();}finally{busy.current=false;setPending(false);}}
 async function reconcile(){setPending(true);try{await onChanged();const result=await readDataInspector(item.market_id,item.symbol,from,to);client.setQueryData(['data-inspector',subject,`${item.market_id}:${item.symbol}:${from}:${to}`],result);setUnknown(false);setError(undefined);}catch(e){setError(e);if(deniesRead(e))onDenied();}finally{setPending(false);}}
 function submit(e:FormEvent<HTMLFormElement>){e.preventDefault();setValidation(true);if(!validateWorkRange(a,b,names)){e.currentTarget.querySelector('input')?.focus();return;}if(downloadLocked||!batchMatches||(jobMatches&&activeJob(jobView.data?.state))||!!(requestJobId&&(job.isPending||job.error||jobView.retained)))return;void command.submit({kind:'candle_ingestion',market_id:item.market_id,symbols:names,timeframe:'1m',start_at:rangeIso(a),end_at:!dirty&&!manual?new Date(Math.floor(Date.now()/60000)*60000).toISOString():rangeIso(b),full_history:!dirty&&!manual});}
 return <><DirtyGuard dirty={(dirty&&!(saved?.start_at?.slice(0,16)===a&&saved?.end_at?.slice(0,16)===b))||command.pending}/><header className="data-inspector-heading"><div><h2>{item.symbol}</h2><p>{exchangeName(source?.exchange_name??'')} · {marketName(source?.market_type??'',w)} · 1m</p></div><button className="library-icon-action" aria-label={w('Close instrument','Закрыть инструмент')} onClick={onClose}><X/></button></header>
  {latestJob&&activeJob(latestJob.state)&&latestJob.job_id!==requestJobId&&<Link className="data-job-result" to={`/data?${new URLSearchParams({market:String(item.market_id),symbol:item.symbol,job:latestJob.job_id})}`}>{w('Open current download','Открыть текущую загрузку')}<ChevronRight aria-hidden="true"/></Link>}
  {requestJobId&&<><ReadFeedback pending={false} retained={jobView.retained} initial={!jobView.data} error={job.error}/>{job.error&&<button disabled={job.isFetching} onClick={()=>void job.refetch()}>{w('Refresh request','Обновить задание')}</button>}{jobMatches&&jobView.data&&<DownloadJobDetails key={jobView.data.job_id} subject={subject} job={jobView.data} eta={eta} markets={markets} locked={locked||jobView.retained||!!job.error} onDenied={onDenied} onChanged={onChanged}/>}</>}
  <section className="data-inspector-section"><div className="data-stream-control"><h3>{w('Candle stream','Поток свечей')}</h3><button type="button" role="switch" aria-label={w('Candle stream','Поток свечей')} aria-checked={item.effective} aria-describedby="data-stream-help" className="data-stream-switch" disabled={locked||pending||unknown||item.strategy_pinned} onClick={()=>void toggle()}><span className="data-stream-track" aria-hidden="true"/></button>{item.strategy_pinned&&<LockKeyhole className="data-lock" aria-hidden="true"/>}</div>
  {item.effective&&!['disconnected','offline','unknown'].includes(item.stream_state??'unknown')&&coverage.state!=='complete'&&<p role="status">{coverage.missing_candles===null?w('Finding history start. Streaming will start automatically after preparation.','Определяем начало истории. После подготовки стриминг запустится автоматически.'):`${w('Preparing history · missing candles','Подготовка истории · осталось свечей')}: ${coverage.missing_candles.toLocaleString(i18n.language)}`}</p>}
  <StreamStatus effective={item.effective} pinned={item.strategy_pinned} state={item.stream_state} complete={coverage.state==='complete'} stale={locked}/><StreamMessage item={item} complete={coverage.state==='complete'} stale={locked}/>{item.strategy_pinned&&<p>{w('Cannot disable: this instrument is required by active strategies.','Нельзя отключить: инструмент нужен активным стратегиям.')}</p>}{item.next_retry_at&&<p>{w('Next retry','Следующая попытка')}: {formatTime(item.next_retry_at,i18n.language)}</p>}
  {!item.effective&&coverage.state!=='complete'&&<p className="data-stream-observation">{w('Enable streaming to fill missing history automatically, then collect new candles.','Включите стриминг: история дозагрузится автоматически, затем начнётся сбор новых свечей.')}</p>}
  {item.history_repair_error&&coverage.state!=='complete'&&<p role="alert">{w('Streaming is blocked: the exchange did not provide all required candles. Use Fill missing history to retry; streaming starts only after the gaps are filled.','Стриминг заблокирован: биржа не вернула все необходимые свечи. Повторите дозагрузку истории; поток включится только после устранения пропусков.')}</p>}
  {!!item.strategies.length&&<ul className="data-strategy-pins">{item.strategies.map(s=><li key={s.strategy_id}><Link to={`/strategies/${s.strategy_id}`}>{s.name}</Link><span>{workState(s.state,w)}</span></li>)}</ul>}
  <dl className="data-last-candle"><dt>{w('Last candle · UTC','Последняя свеча · UTC')}</dt><dd>{formatTime(item.last_candle_at,i18n.language)}</dd></dl>

  <WorkError error={error}/>{unknown&&<button disabled={pending} onClick={()=>void reconcile()}>{w('Check saved state','Проверить сохранённое состояние')}</button>}</section>
  <section className="data-inspector-section"><h3>{w('Full history','Вся история')}</h3><div className={`data-inspector-coverage ${coverage.state==='complete'?'complete':''}`}>{coverage.state==='complete'?<CheckCircle2 aria-hidden="true"/>:<Circle aria-hidden="true"/>}<strong>{syncing?w('Updating…','Обновляется…'):coverage.coverage_percent===null?'—':`${coverage.coverage_percent}%`}</strong><span>{coverage.state==='unknown'?w('Checking history start…','Определяем начало истории…'):`${coverage.actual_candles.toLocaleString(i18n.language)} / ${coverage.expected_candles.toLocaleString(i18n.language)}`}</span></div>{coverage.coverage_percent!==null&&<progress max={100} value={coverage.coverage_percent} aria-label={w('Full history coverage','Заполненность всей истории')}/>}<p>{coverage.start_at?`${rangeLabel(coverage.start_at,coverage.end_at,i18n.language)} · UTC`:w('The exchange history start is not confirmed yet.','Начало истории на бирже ещё не подтверждено.')}</p>{coverage.missing_candles!==null&&<p>{w('Missing candles','Не хватает свечей')}: {coverage.missing_candles.toLocaleString(i18n.language)}</p>}{coverage.source_absent_minutes>0&&<p>{w('No candles on the exchange','Нет свечей на бирже')}: {coverage.source_absent_minutes.toLocaleString(i18n.language)} {w('min. Excluded from expected history.','мин. Исключены из ожидаемой истории.')}</p>}
   {manualRepairAllowed&&<form onSubmit={submit}><h3>{selected.length>1?`${w('Download','Загрузить')}: ${selected.length} ${w('instruments','инструмента')}`:w('Fill missing history','Дозагрузить историю')}</h3>{selected.length>1&&<p>{names.join(', ')}</p>}{resolving&&<p role="status">{w('Finding the first available candle…','Определяем первую доступную свечу…')}</p>}{resolving&&!manual&&<button type="button" disabled={locked} onClick={()=>setManual(true)}>{w('Enter period manually','Указать период вручную')}</button>}{!batch&&!resolving&&!allReady&&<p>{w('History start is unavailable. Retry or enter a period manually.','Не удалось определить начало истории. Повторите проверку или укажите период вручную.')}</p>}<WorkError error={boundsError}/>{!batch&&!resolving&&!allReady&&<button type="button" disabled={locked} onClick={()=>void Promise.all(bounds.map(read=>read.refetch()))}>{w('Find full history','Определить всю историю')}</button>}<details className="data-custom-period"><summary>{w('Custom period','Указать период вручную')}</summary><fieldset className="data-range-fields" disabled={downloadLocked}><Field name="data-download-from" label={w('From · UTC','С · UTC')}><input id="data-download-from" type="datetime-local" required value={a} aria-invalid={validation&&!validateWorkRange(a,b,names)} onInput={e=>setA(e.currentTarget.value)} onChange={e=>setA(e.target.value)}/></Field><Field name="data-download-to" label={w('To · UTC','По · UTC')}><input id="data-download-to" type="datetime-local" required value={b} onInput={e=>setB(e.currentTarget.value)} onChange={e=>setB(e.target.value)}/></Field></fieldset></details>
   {validation&&!validateWorkRange(a,b,names)&&<p className="notice error" role="alert">{w('Choose up to 50 instruments and a valid period of closed candles.','Выберите до 50 инструментов и корректный период завершённых свечей.')}</p>}
   <button className="primary data-download-button" type="submit" disabled={downloadLocked||!batchMatches||!validateWorkRange(a,b,names)||(jobMatches&&activeJob(jobView.data?.state))||!!(requestJobId&&(job.isPending||job.error||jobView.retained))}><Download aria-hidden="true"/>{command.pending?w('Requesting…','Отправка…'):w('Fill missing history','Дозагрузить историю')}</button><p>{batch?w('Each instrument gets a separate queued request. Only missing candles across its full history will be fetched.','Для каждого инструмента создаётся задание. Загружаются только пропуски во всей его истории.'):w('Checks the full history and downloads only gaps and the missing tail. Existing candles are skipped.','Проверяем всю историю и загружаем только пропуски и недостающий конец. Сохранённые свечи пропускаются.')}</p>
  </form>}<WorkError error={command.error}/>{command.unresolved&&<button disabled={command.pending} onClick={()=>void command.reconcile()}>{w('Check request','Проверить задание')}</button>}
  </section>
  <details className="data-instrument-details"><summary>{w('Instrument details','Параметры инструмента')}</summary><dl><dt>{w('Assets','Активы')}</dt><dd>{item.base_asset} / {item.quote_asset}</dd><dt>{w('Price step','Шаг цены')}</dt><dd>{item.price_step??'—'}</dd><dt>{w('Quantity step','Шаг количества')}</dt><dd>{item.qty_step??'—'}</dd><dt>{w('Minimum order','Минимальный ордер')}</dt><dd>{item.min_notional??'—'} {item.quote_asset}</dd><dt>{w('Catalog updated · UTC','Каталог обновлён · UTC')}</dt><dd>{formatTime(item.refreshed_at,i18n.language)}</dd></dl></details>
 </>;
}
