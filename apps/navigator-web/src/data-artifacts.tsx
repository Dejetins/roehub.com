import {Fragment,useEffect,useRef,useState} from 'react';
import {Link,useSearchParams,useBlocker} from 'react-router';
import {useQuery,useQueryClient} from '@tanstack/react-query';
import {useTranslation} from 'react-i18next';
import {Search,X,CheckCircle2,Circle,Clock,AlertCircle,ChevronRight} from 'lucide-react';
import {ChartDisplayMenu} from './chart-display-menu';
import {ApiError} from './api';
import {readMarkets} from './connections-api';
import {CollectionLibrary,formatTime,ReadFeedback,RefreshButton,useWords,WorkError,deniesRead} from './workspace-ui';
import {useReadSnapshot} from './read-snapshot';
import {DataMode} from './data-mode';
import {ExchangeLinkContent,exchangeName} from './data-exchange-link';
import {ArtifactProfileEditor} from './artifact-profile';
import {coordinateKey,readArtifactDefaults,readArtifactDetail,readArtifactInventory,recoverArtifactCommand,submitArtifactCommand,type ArtifactDetail,type ArtifactRequest,type ArtifactPatch} from './artifacts-api';
import {ArtifactJournal,ArtifactJobInspector} from './artifact-journal';
import './data-artifacts.css';
const activeState=(state:string|undefined)=>!!state&&['waiting_for_source','queued','running','retry_wait'].includes(state);
const active=(r:ArtifactRequest|null)=>!!r&&activeState(r.state);
const updating=(r:ArtifactDetail)=>activeState(r.base_state)||active(r.base_request)||active(r.derivative_request);
const denied=(e:unknown)=>deniesRead(e);
export function ArtifactProgress({request,label,state}:{request:ArtifactRequest|null;label:string;state?:string}){
 if(activeState(state)&&!active(request))request=null;
 const w=useWords();const names:Record<string,string>={waiting_for_source:w('Waiting for source','Ожидание источника'),queued:w('Queued','В очереди'),running:w('Running','Выполняется'),retry_wait:w('Waiting to retry','Ожидание повтора'),ready:w('Ready','Готово'),error:w('Failed','Ошибка'),unknown:w('Unknown','Неизвестно'),insufficient_history:w('Insufficient history','Недостаточно истории')};
 const current=request?.state??state??'unknown';
 const latest=[...(request?.progress??[])].sort((a,b)=>(Date.parse(b.observed_at??'')||0)-(Date.parse(a.observed_at??'')||0))[0];
 const phases:Record<string,string>={base:w('Preparing files','Подготовка файлов'),source_snapshot:w('Reading candles','Чтение свечей'),funding:w('Funding rates','Ставки финансирования'),derivatives:w('Calculating indicators','Расчёт индикаторов')};
 const Icon=current==='ready'?CheckCircle2:current==='error'?AlertCircle:activeState(current)?Clock:Circle;
 return <div className="artifact-progress" aria-label={label}><span className={`artifact-cell-status${current==='ready'?' ready':current==='error'?' failed':''}`}><Icon aria-hidden="true"/>{!request&&!state?w('Not requested','Не запрошено'):names[current]??w('Unknown','Неизвестно')}</span>
 {current==='running'&&latest&&phases[latest.phase]&&<span className="artifact-current-phase">{phases[latest.phase]}</span>}
 {request?.reason_code&&<span className="notice error">{request.reason_code==='source_unavailable'?w('Source unavailable','Источник недоступен'):request.reason_code}</span>}</div>;
}
export function ArtifactCellStatus({request,state,enabled}:{request:ArtifactRequest|null;state?:string;enabled?:boolean}) {
 const w=useWords();
 const observed=activeState(state)&&!active(request)?state:request?.state??state;
 const current=enabled===false&&!activeState(observed)&&observed!=='error'?undefined:observed;
 const failed=current==='error';
 const label=failed?w('Error · select instrument for details','Ошибка · выберите инструмент для подробностей'):
  current==='ready'?w('Ready','Готово'):current==='running'?w('Calculating','Расчёт'):
  current==='queued'?w('Queued','В очереди'):current==='retry_wait'?w('Retry pending','Ожидание повтора'):
  current==='waiting_for_source'?w('Waiting for candles','Ожидание свечей'):current==='insufficient_history'?w('Short history','Мало истории'):
  enabled===false?w('Off','Выкл'):enabled===true?w('On','Вкл'):w('Unknown','Неизвестно');
 const Icon=failed?AlertCircle:current==='ready'?CheckCircle2:activeState(current)?Clock:Circle;
 return <span className={`artifact-cell-status${failed?' failed':current==='ready'?' ready':''}`} title={label}>
  <Icon aria-hidden="true"/>{failed?<span className="sr-only">{label}</span>:<span>{label}</span>}
 </span>;
}
function artifactDate(value:string|null,locale:string){
 if(!value)return '—';
 const date=new Date(value);
 return Number.isNaN(date.getTime())?'—':new Intl.DateTimeFormat(locale,{day:'2-digit',month:'2-digit',year:'numeric',hour:'2-digit',minute:'2-digit',timeZone:'UTC'}).format(date);
}
export function ArtifactsPage({subject}:{subject:string}){
 const w=useWords(),{i18n}=useTranslation(),[params,setParams]=useSearchParams(),client=useQueryClient();
 const markets=useQuery({queryKey:['market-sources',subject],queryFn:({signal})=>readMarkets(signal),retry:false,refetchOnWindowFocus:false});
 const journalOpen=params.get('artifactjournal')==='1',jobId=params.get('artifactjob')??'',journalTrigger=useRef<HTMLButtonElement>(null);
 const market=Number(params.get('market')??1),exchange=params.get('exchange')??markets.data?.items.find(m=>m.market_id===market)?.exchange_name??'binance',symbol=params.get('symbol')??'',search=params.get('q')??'',filter=params.get('type')??'';
 const sources=markets.data?.items.filter(m=>m.exchange_name===exchange&&(filter?m.market_id===Number(filter):true))??[];
 const marketOptions=[{value:'',label:w('All markets','Все рынки')},...(markets.data?.items.filter(m=>m.exchange_name===exchange).map(m=>({value:String(m.market_id),label:m.market_type==='spot'?w('Spot','Спот'):m.market_type==='futures'?w('Futures','Фьючерсы'):m.market_type}))??[])];
 const scopeKey=`${exchange}:${filter}:${search}:${params.get('artifactcursor')??''}`;
 const [accessLost,setAccessLost]=useState(false),[selected,setSelected]=useState<string[]>([]),[editing,setEditing]=useState(false),[commandError,setCommandError]=useState<unknown>(),[pending,setPending]=useState(false),[cooldown,setCooldown]=useState(0),[conflict,setConflict]=useState(false),[message,setMessage]=useState('');
 const busy=useRef(false),mounted=useRef(true),restoreFocus=useRef(false),editTrigger=useRef<HTMLButtonElement>(null),controller=useRef<AbortController|undefined>(undefined);
 useEffect(()=>{mounted.current=true;return()=>{mounted.current=false;controller.current?.abort();void client.cancelQueries({queryKey:['artifacts',subject]});client.removeQueries({queryKey:['artifacts',subject]});};},[]);
 const defaults=useQuery({queryKey:['artifacts',subject,'defaults'],queryFn:({signal})=>readArtifactDefaults(signal),retry:false,refetchOnWindowFocus:false,enabled:!accessLost&&cooldown<=Date.now()});
 const cursor=params.get('artifactcursor')??'';
 const inventory=useQuery({queryKey:['artifacts',subject,'inventory',scopeKey],enabled:!!markets.data&&!accessLost&&cooldown<=Date.now(),retry:false,refetchOnWindowFocus:false,queryFn:async({signal})=>{
  // Each market page is independently bounded. A cursor applies to the selected market only.
  const pages=await Promise.all(sources.map(s=>readArtifactInventory(new URLSearchParams({exchange,market_type:s.market_type,search,limit:'50',...(cursor&&sources.length===1?{cursor}:{})}),signal)));
  return {items:pages.flatMap(p=>p.items),next:sources.length===1?pages[0]?.next_cursor:null};
 },refetchInterval:q=>q.state.error?false:q.state.data?.items.some(updating)?2000:false});
 const source=markets.data?.items.find(m=>m.market_id===market),coordinate={exchange:source?.exchange_name??exchange,market_type:source?.market_type??'spot',symbol};
 const detailKey=coordinateKey(coordinate);
 const detail=useQuery({queryKey:['artifacts',subject,'detail',detailKey],enabled:!!symbol&&!!source&&!accessLost&&cooldown<=Date.now(),retry:false,refetchOnWindowFocus:false,queryFn:({signal})=>readArtifactDetail(coordinate,signal),refetchInterval:q=>q.state.error?false:q.state.data&&updating(q.state.data)?2000:false});
 const readCooldown=useArtifactCooldown([markets.error,defaults.error,inventory.error,detail.error]);
 useEffect(()=>{if(readCooldown)setCooldown(current=>Math.max(current,readCooldown));},[readCooldown]);
 const lost=accessLost||[markets.error,defaults.error,inventory.error,detail.error].some(denied);
 const view=useReadSnapshot(subject,scopeKey,inventory.isSuccess?inventory.data:undefined,lost),inspector=useReadSnapshot(subject,detailKey,detail.isSuccess?detail.data:undefined,lost||!symbol);
 function clearAccess(){setAccessLost(true);setSelected([]);setEditing(false);controller.current?.abort();void client.cancelQueries({queryKey:['artifacts',subject]});client.removeQueries({queryKey:['artifacts',subject]});}
 useEffect(()=>{if(lost&&!accessLost)clearAccess();},[lost,accessLost]);
 useEffect(()=>{setSelected([]);setEditing(false);},[scopeKey,detailKey]);
 useEffect(()=>{if(!cooldown)return;const timer=setTimeout(()=>setCooldown(0),Math.max(1,cooldown-Date.now()));return()=>clearTimeout(timer);},[cooldown]);
 const key=params.get('artifactkey')??'';
 const rows=selected.length?(view.data?.items.filter(r=>selected.includes(coordinateKey(r.coordinate)))??[]):inspector.data?[inspector.data]:[];
 const unavailable=lost||!defaults.isSuccess||!!defaults.error||!defaults.data?.can_mutate||!inventory.isSuccess||!!inventory.error||view.retained||inspector.retained||!!detail.error||pending||!!key||conflict||(cooldown>Date.now()||readCooldown>Date.now())||!rows.length||rows.length>50||selected.length>0&&rows.length!==selected.length;
 const checking=defaults.isFetching||inventory.isFetching||detail.isFetching;
 const locked=unavailable||checking;
 const quietRead=checking&&!unavailable&&!editing;
 function change(values:Record<string,string>,replace=false){setParams(current=>{const next=new URLSearchParams(current);for(const [k,v] of Object.entries(values))v?next.set(k,v):next.delete(k);return next;},{replace});}
 function closeEdit(){restoreFocus.current=true;setEditing(false);}
 useEffect(()=>{if(restoreFocus.current&&!pending&&!editing){editTrigger.current?.focus();restoreFocus.current=false;}},[pending,editing]);
 async function refresh(){if((cooldown>Date.now()||readCooldown>Date.now())||pending||lost)return;setConflict(true);await client.cancelQueries({queryKey:['artifacts',subject]});const results=await Promise.all([defaults.refetch(),inventory.refetch(),...(symbol?[detail.refetch()]:[])]);if(results.every(r=>r.isSuccess)){setConflict(false);setCommandError(undefined);}}
 function failure(error:unknown){setCommandError(error);if(denied(error))clearAccess();if(error instanceof ApiError){if(error.status===409)setConflict(true);if(error.retryAfterSeconds)setCooldown(Date.now()+error.retryAfterSeconds*1000);}}
 async function finish(){await client.cancelQueries({queryKey:['artifacts',subject]});await client.invalidateQueries({queryKey:['artifacts',subject]});if(mounted.current){change({artifactkey:''},true);setConflict(false);setCommandError(undefined);setMessage(w('Changes saved. Scheduled anchor is unchanged.','Изменения сохранены. Начало отсчёта расписания не изменено.'));closeEdit();}}
 async function submit(patches?:ArtifactPatch[]){if(locked||busy.current)return;busy.current=true;setPending(true);setCommandError(undefined);setMessage('');const commandKey=`art-${crypto.randomUUID()}`;change({artifactkey:commandKey},true);controller.current=new AbortController();
  try{await submitArtifactCommand(rows.map((r,i)=>({coordinate:r.coordinate,expected_version:r.control_version,...(patches?{patch:patches[i]}:{})})),commandKey,!patches,controller.current.signal);if(mounted.current)await finish();}
  catch(error){if(mounted.current){failure(error);if(!(error instanceof ApiError&&error.outcome==='unknown'))change({artifactkey:''},true);}}
  finally{busy.current=false;if(mounted.current)setPending(false);}
 }
 async function recover(){if(!key||busy.current||lost||(cooldown>Date.now()||readCooldown>Date.now()))return;busy.current=true;setPending(true);controller.current=new AbortController();try{await recoverArtifactCommand(key,controller.current.signal);if(mounted.current)await finish();}catch(error){if(mounted.current)failure(error);}finally{busy.current=false;if(mounted.current)setPending(false);}}
 const selectRow=(r:ArtifactDetail)=>{setSelected([]);const marketId=markets.data?.items.find(m=>m.exchange_name===r.coordinate.exchange&&m.market_type===r.coordinate.market_type)?.market_id;change({market:String(marketId??market),symbol:r.coordinate.symbol,artifactjob:''});};
 function closeJournal(){change({artifactjournal:'',artifactjob:''});journalTrigger.current?.focus();}
 const allRows=view.data?.items??[],exchanges=[...new Set(markets.data?.items.map(m=>m.exchange_name)??[])];
 return <><ArtifactDraftGuard dirty={editing||pending||!!key}/><CollectionLibrary title={w('Exchanges','Биржи')}><nav className="work-categories data-exchanges" aria-label={w('Exchanges','Биржи')}>{exchanges.map(name=>{const next=new URLSearchParams(params);next.set('exchange',name);next.set('market',String(markets.data?.items.find(m=>m.exchange_name===name)?.market_id??1));for(const k of ['symbol','type','artifactcursor'])next.delete(k);return <Link key={name} aria-label={exchangeName(name)} to={`/data?${next}`} aria-current={name===exchange?'page':undefined}><ExchangeLinkContent name={name}/></Link>;})}</nav></CollectionLibrary>
  <h1 id="workspace-heading" className="sr-only" tabIndex={-1}>{w('Data','Данные')}</h1>
  <section className="data-catalog panel" aria-label={w('Artifact catalog','Каталог артефактов')}><DataMode/><div className="data-catalog-tools">
   <label className="data-search"><Search aria-hidden="true"/><span className="sr-only">{w('Search instruments','Поиск инструментов')}</span><input type="search" maxLength={64} value={search} placeholder={w('Search instruments','Поиск инструментов')} onChange={e=>change({q:e.target.value.toUpperCase(),artifactcursor:''},true)}/></label>
   <ChartDisplayMenu label={w('Market','Рынок')} value={marketOptions.find(option=>option.value===filter)?.label??marketOptions[0].label} options={marketOptions.map(option=>({label:option.label,checked:option.value===filter,onChange:()=>change({type:option.value,artifactcursor:''})}))}/>
  </div>
  <div className="data-catalog-context"><span>{w('Backtest files · updated from candles','Файлы для бэктестов · обновляются из свечей')}</span><div className="work-actions"><button ref={journalTrigger} className="work-link-button" aria-expanded={journalOpen} aria-controls="artifact-job-journal" onClick={()=>journalOpen?closeJournal():change({artifactjournal:'1'})}>{journalOpen?w('Hide journal','Скрыть журнал'):w('Show journal','Показать журнал')}</button><span className={inventory.isFetching&&inventory.isSuccess&&!conflict&&!lost&&!cooldown&&!readCooldown?'artifact-quiet-read':undefined}><RefreshButton pending={inventory.isFetching} disabled={lost||(cooldown>Date.now()||readCooldown>Date.now())} onClick={()=>void refresh()}/></span></div></div>
  <div className="data-read-status"><ReadFeedback pending={false} retained={view.retained} initial={!view.data} error={markets.error??defaults.error??inventory.error??(lost?new ApiError('forbidden',403,'failed'):undefined)}/></div>
  <div className="data-catalog-scroll" tabIndex={0} aria-label={w('Artifact rows','Строки артефактов')}><table className="data-catalog-table artifact-table"><thead><tr><th><input type="checkbox" aria-label={w('Select page','Выбрать страницу')} disabled={editing||pending||!!key||lost||view.retained||!inventory.isSuccess||!!inventory.error} checked={!!allRows.length&&allRows.slice(0,50).every(r=>selected.includes(coordinateKey(r.coordinate)))} onChange={e=>{setEditing(false);if(jobId)change({artifactjob:''});setSelected(e.target.checked?allRows.slice(0,50).map(r=>coordinateKey(r.coordinate)):[]);}}/></th>{[w('Instrument','Инструмент'),w('Market','Рынок'),w('Base files','Базовые файлы'),w('Precompute','Предрасчёт'),w('Ready through · UTC','Готово по · UTC')].map(t=><th key={t} scope="col">{t}</th>)}</tr></thead><tbody>{allRows.map(r=>{const id=coordinateKey(r.coordinate);return <tr key={id} className={id===detailKey?'selected':''} data-selectable="true" tabIndex={view.retained||lost?-1:0} onKeyDown={e=>{
 if(e.target!==e.currentTarget||view.retained||lost||e.altKey||e.ctrlKey||e.metaKey||e.shiftKey)return;
 if(e.key==='Enter'||e.key===' '){e.preventDefault();selectRow(r);}
 }} onClick={e=>{
 if(view.retained||lost||e.defaultPrevented||e.button!==0||e.metaKey||e.ctrlKey||e.shiftKey||e.altKey)return;
 if(e.target instanceof Element&&e.target.closest('a,button,input,select,textarea,label'))return;
 selectRow(r);
 }}><td><input type="checkbox" aria-label={`${w('Select','Выбрать')} ${r.coordinate.symbol} ${r.coordinate.market_type}`} checked={selected.includes(id)} disabled={editing||pending||!!key||lost||view.retained||!inventory.isSuccess||!!inventory.error||!selected.includes(id)&&selected.length>=50} onChange={e=>{setEditing(false);if(jobId)change({artifactjob:''});setSelected(s=>e.target.checked?[...s,id]:s.filter(x=>x!==id));}}/></td><td><button className="data-instrument-name artifact-instrument" disabled={view.retained||lost} onClick={()=>selectRow(r)}>{r.coordinate.symbol}</button></td><td>{r.coordinate.market_type==='spot'?w('Spot','Спот'):r.coordinate.market_type==='futures'?w('Futures','Фьючерсы'):r.coordinate.market_type}</td><td><ArtifactCellStatus request={r.base_request} state={r.base_state}/></td><td><ArtifactCellStatus request={r.derivative_request} enabled={r.profile.mode==='precompute'}/></td><td><time dateTime={r.covered_through??undefined}>{artifactDate(r.covered_through,i18n.language)}</time></td></tr>;})}</tbody></table>{!allRows.length&&inventory.isSuccess&&!lost&&<p>{w('No artifacts on this page. Candles prepare base files automatically.','На странице нет артефактов. Базовые файлы готовятся автоматически из свечей.')}</p>}</div>
  <footer className="data-catalog-footer library-pagination"><span>{selected.length?`${selected.length} ${w('selected','выбрано')}`:w('Up to 50 rows per market','До 50 строк на рынок')}</span>{cursor&&<button onClick={()=>change({artifactcursor:''})}>{w('First page','Первая страница')}</button>}{view.data?.next&&<button disabled={view.retained||!!inventory.error} onClick={()=>change({artifactcursor:view.data!.next!})}>{w('Next page','Следующая страница')}</button>}{!filter&&sources.length>1&&<span>{w('Choose a market for more pages','Выберите рынок для следующих страниц')}</span>}</footer>
  {journalOpen&&<ArtifactJournal subject={subject} blocked={lost} onDenied={clearAccess} onClose={closeJournal} onSelect={()=>setSelected([])}/>}
  </section><aside className="data-inspector panel" aria-label={jobId?w('Job details','Подробности задания'):w('Artifact settings','Настройки артефактов')}>{jobId?<ArtifactJobInspector subject={subject} id={jobId} hasInstrument={!!symbol} blocked={lost} onDenied={clearAccess} onBack={()=>change({artifactjob:''})}/>:<><ReadFeedback pending={false} retained={inspector.retained} initial={!!symbol&&!inspector.data} error={detail.error}/>
  {!lost&&!!rows.length&&<><header className="data-inspector-heading"><div><h2>{selected.length>1?`${selected.length} ${w('instruments','инструмента')}`:rows[0].coordinate.symbol}</h2><p>{rows[0].coordinate.exchange} · {rows[0].coordinate.market_type}</p></div><button className="library-icon-action" aria-label={w('Close instrument','Закрыть инструмент')} onClick={()=>change({symbol:''})}><X/></button></header>
  {!editing&&<div className={`work-actions artifact-actions${quietRead?' artifact-quiet-read':''}`}><button disabled={locked||editing} onClick={()=>void submit()}>{w('Update now','Обновить сейчас')}</button><button ref={editTrigger} disabled={locked||editing} onClick={()=>setEditing(true)}>{w('Edit profile and schedule','Изменить профиль и расписание')}</button></div>}
  {editing&&defaults.data&&<ArtifactProfileEditor key={rows.map(r=>coordinateKey(r.coordinate)).join('|')} rows={rows} defaults={defaults.data} locked={locked} onCancel={closeEdit} onSave={patches=>void submit(patches)}/>}
  {!editing&&rows.length===1&&<ArtifactInspector row={rows[0]}/>}</>}
  <WorkError error={commandError}/>{(cooldown>Date.now()||readCooldown>Date.now())&&<p role="status">{w('Retry allowed after','Повтор доступен после')} {formatTime(new Date(Math.max(cooldown,readCooldown)).toISOString(),i18n.language)}</p>}{key&&!lost&&<button disabled={pending||(cooldown>Date.now()||readCooldown>Date.now())} onClick={()=>void recover()}>{w('Check command result','Проверить результат команды')}</button>}{conflict&&!lost&&<button disabled={pending||(cooldown>Date.now()||readCooldown>Date.now())} onClick={()=>void refresh()}>{w('Refresh changed state','Обновить изменённое состояние')}</button>}<p role="status">{message}</p>
  {!rows.length&&!lost&&<p>{w('Select an instrument to inspect artifacts.','Выберите инструмент для просмотра артефактов.')}</p>}
  </>}
  </aside></>;
}
function timeframeDuration(value:string){
 const match=/^(\d+)([mhdw])$/.exec(value);
 return match?Number(match[1])*({m:1,h:60,d:1440,w:10080}[match[2]]??1):Infinity;
}
export function TimeframeCoverage({timeframes}:{timeframes:ArtifactDetail['timeframes']}){
 const w=useWords(),{i18n}=useTranslation(),[expanded,setExpanded]=useState<string|null>(null);
 return <details className="artifact-coverage"><summary>{w('Timeframe coverage','Покрытие таймфреймов')}</summary>
 {timeframes.length?<table className="artifact-coverage-table"><thead><tr><th scope="col">{w('Interval','Интервал')}</th><th scope="col">{w('State','Состояние')}</th><th scope="col">{w('Candles','Свечей')}</th></tr></thead><tbody>
 {[...timeframes].sort((a,b)=>timeframeDuration(a.timeframe)-timeframeDuration(b.timeframe)||a.timeframe.localeCompare(b.timeframe)).map(t=>{
 const open=expanded===t.timeframe,missing=(t.missing_bars??0)>0;
 const label=missing?w('Gaps','Пропуски'):({ready:w('Ready','Готово'),insufficient_history:w('Short history','Мало истории'),error:w('Error','Ошибка'),waiting_for_source:w('Waiting','Ожидание'),queued:w('Queued','В очереди'),running:w('Calculating','Расчёт'),retry_wait:w('Retry pending','Ожидание повтора')}[t.state]??w('Unknown','Неизвестно'));
 const Icon=missing||t.state==='error'?AlertCircle:t.state==='ready'?CheckCircle2:t.state==='insufficient_history'||activeState(t.state)?Clock:Circle;
 const toggle=()=>setExpanded(open?null:t.timeframe);
 return <Fragment key={t.timeframe}><tr className="artifact-coverage-row" onClick={toggle}><th scope="row"><button type="button" aria-expanded={open} aria-label={`${t.timeframe} · ${label}`} onClick={e=>{e.stopPropagation();toggle();}}><ChevronRight aria-hidden="true"/>{t.timeframe}</button></th><td><span className="artifact-coverage-state"><Icon aria-hidden="true"/>{label}</span></td><td>{t.closed_bars===null||t.closed_bars===0&&t.state==='insufficient_history'?'—':t.closed_bars.toLocaleString(i18n.language)}</td></tr>
 {open&&<tr className="artifact-coverage-detail"><td colSpan={3}>
 {(t.start_utc||t.end_utc)&&<p>{w('Period','Период')} · UTC<br/>{t.start_utc?artifactDate(t.start_utc,i18n.language):'—'} — {t.end_utc?artifactDate(t.end_utc,i18n.language):'—'}</p>}
 {missing&&<p>{w('Missing candles','Пропущено свечей')}: {t.missing_bars!.toLocaleString(i18n.language)}</p>}
 {t.state==='insufficient_history'&&<p>{w('Not enough candles to form this interval.','Недостаточно свечей для этого интервала.')}</p>}
 {t.state==='error'&&<p>{w('Could not prepare this interval. See the base file error above.','Не удалось подготовить интервал. Подробности ошибки — выше, в базовых файлах.')}</p>}
 {!t.start_utc&&!t.end_utc&&t.state!=='insufficient_history'&&t.state!=='error'&&<p>{w('Coverage period is not available yet.','Период покрытия пока неизвестен.')}</p>}
 </td></tr>}
 </Fragment>;
 })}</tbody></table>:<p>{w('No interval data yet.','Данных по интервалам пока нет.')}</p>}</details>;
}
export function ArtifactSummary({row}:{row:ArtifactDetail}){
 const w=useWords(),{i18n}=useTranslation();
 const latest=[row.base_request,row.derivative_request].filter((r):r is ArtifactRequest=>!!r).sort((a,b)=>Date.parse(b.created_at)-Date.parse(a.created_at))[0];
 const sameTime=(a:string|null,b:string|null)=>a===b||!!a&&!!b&&Date.parse(a)===Date.parse(b);
 const facts:[string,string|null][]=[
  [w('Data through','Данные по'),row.covered_through],
  ...(!sameTime(row.source_through,row.covered_through)&&row.source_through?[[w('Candles through','Свечи доступны по'),row.source_through] as [string,string]]:[]),
  [w('Updated','Обновлено'),row.last_success],
  ...(latest&&latest.state!=='ready'&&(!row.last_success||Date.parse(latest.created_at)>Date.parse(row.last_success))?[[w('Last attempt','Последняя попытка'),latest.created_at] as [string,string]]:[]),
  [w('Next update','Следующее'),row.next_run],
 ];
 const cadence=row.schedule.interval_days===1?w('Daily','Ежедневно'):row.schedule.interval_days===7?w('Weekly','Раз в неделю'):w(`Every ${row.schedule.interval_days} days`,`Раз в ${row.schedule.interval_days} дня`);
 return <div className="artifact-summary"><div className="artifact-summary-title">{w('Updates','Обновление')}<span>UTC</span></div>
 <dl className="artifact-summary-facts">{facts.map(([label,value])=><div key={label}><dt>{label}</dt><dd>{value?<time dateTime={value}>{artifactDate(value,i18n.language)}</time>:'—'}</dd></div>)}</dl>
 <p className="artifact-schedule">{cadence} · {row.schedule.local_time.slice(0,5)} {row.schedule.timezone}</p>
 {row.size_bytes!==null&&<p>{w('Size','Размер')}: {row.size_bytes.toLocaleString(i18n.language)} B</p>}
 </div>;
}
function ArtifactInspector({row}:{row:ArtifactDetail}){const w=useWords();return <><section className="data-inspector-section"><h3>{w('Base NPY','База NPY')}</h3><ArtifactProgress label="Base NPY" request={row.base_request} state={row.base_state}/><ArtifactSummary row={row}/><TimeframeCoverage key={coordinateKey(row.coordinate)} timeframes={row.timeframes}/></section><section className="data-inspector-section"><h3>{w('Precompute','Предрасчёт')}</h3><p>{w('Calculates indicators and risk levels ahead of time for backtests. Optional; base files update automatically.','Заранее рассчитывает индикаторы и уровни риска для бэктестов. Необязателен: базовые файлы обновляются автоматически.')}</p><p>{row.profile.mode==='precompute'?w('Enabled','Включён'):w('Off · calculated when needed','Выключен · расчёт при необходимости')} · {row.profile.indicators.length} / {row.profile.catalog_indicators.length} {w('indicators','индикаторов')}</p>{row.derivative_request&&<ArtifactProgress label="Derivatives" request={row.derivative_request}/>}<p>{row.profile.derivative_timeframes.join(', ')}</p><p>{row.profile.history.kind==='all_loaded'?w('All loaded history','Вся загруженная история'):`${row.profile.history.start_utc} — ${row.profile.history.end_utc}`}</p><details><summary>{w('Risk levels','Уровни риска')} · TP {row.profile.tp_levels_pct.length} / SL {row.profile.sl_levels_pct.length}</summary><p>TP: {row.profile.tp_levels_pct.join(', ')}% · SL: {row.profile.sl_levels_pct.join(', ')}%</p></details><p>{w('Turning precompute off retains existing files.','Отключение предрасчёта сохраняет существующие файлы.')}</p></section></>;}

function ArtifactDraftGuard({dirty}:{dirty:boolean}){const w=useWords(),blocker=useBlocker(({currentLocation,nextLocation})=>{const a=new URLSearchParams(currentLocation.search),b=new URLSearchParams(nextLocation.search);a.delete('artifactkey');b.delete('artifactkey');a.sort();b.sort();return dirty&&(currentLocation.pathname!==nextLocation.pathname||a.toString()!==b.toString());});const asked=useRef(false);useEffect(()=>{if(blocker.state!=='blocked'){asked.current=false;return;}if(asked.current)return;asked.current=true;if(window.confirm(w('Discard unsaved changes?','Отменить несохранённые изменения?')))blocker.proceed();else blocker.reset();},[blocker]);useEffect(()=>{const before=(e:BeforeUnloadEvent)=>{if(dirty){e.preventDefault();e.returnValue='';}};window.addEventListener('beforeunload',before);return()=>window.removeEventListener('beforeunload',before);},[dirty]);return null;}

function useArtifactCooldown(errors:unknown[]){const [until,setUntil]=useState(0);useEffect(()=>{const seconds=Math.max(0,...errors.map(e=>e instanceof ApiError?e.retryAfterSeconds??0:0));if(seconds)setUntil(Date.now()+seconds*1000);},errors);useEffect(()=>{if(!until)return;const t=setTimeout(()=>setUntil(0),Math.max(1,until-Date.now()));return()=>clearTimeout(t);},[until]);return until;}
