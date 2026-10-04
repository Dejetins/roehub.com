import {useEffect, useRef, useState, type FormEvent} from 'react';
import {Link, useNavigate, useSearchParams} from 'react-router';
import {useInfiniteQuery, useQuery, useQueryClient} from '@tanstack/react-query';
import {Plus} from 'lucide-react';
import {useTranslation} from 'react-i18next';
import {ApiError} from './api';
import {connectionCommand,readConnection,readConnections,readConnectionBindings,readMarkets,type Connection,type Market} from './connections-api';
import {useReadSnapshot} from './read-snapshot';
import {CollectionLibrary,deniesRead,DirtyGuard,Field,formatTime,ReadFeedback,RefreshButton,useWords,WorkError,WorkHeading} from './workspace-ui';
import {Confirm} from './results';
import {NavigatorTable} from './navigator-table';
import {readCatalog} from './market-data-api';
import {useAutoLoad,useWorkSubmission,workState} from './market-data-ui';
import {RecentAuth} from './recent-auth';

type Detail={kind:'private';connection:Connection}|{kind:'public';market:Market};
export function ConnectionsPage({subject}:{subject:string}) {
  const w=useWords(),[params,setParams]=useSearchParams(),client=useQueryClient(),sentinel=useRef<HTMLDivElement>(null);
  const q=params.get('q')??'',status=params.get('status')??'active',environment=params.get('environment')??'',kind=params.get('kind')??'all';
  const id=params.get('connection'),source=params.get('source'),creating=params.get('new')==='1',selection=id?`private:${id}`:source?`public:${source}`:'';
  const query=useInfiniteQuery({queryKey:['connections',subject,q,status,environment],initialPageParam:'',maxPages:5,retry:false,
    queryFn:({signal,pageParam})=>readConnections(new URLSearchParams({status,q,limit:'50',...(environment?{environment}:{}),...(pageParam?{cursor:pageParam}:{})}),signal),getNextPageParam:p=>p.next_cursor??undefined});
  const markets=useQuery({queryKey:['market-sources',subject],queryFn:({signal})=>readMarkets(signal),retry:false});
  const selected=useQuery({queryKey:['connection-detail',subject,selection],enabled:!!selection&&!creating,retry:false,
    queryFn:async({signal}):Promise<Detail>=>{if(id)return {kind:'private',connection:await readConnection(id,signal)};
      const result=await readMarkets(signal),market=result.items.find(m=>String(m.market_id)===source);
      if(!market)throw new ApiError('not-found',404,'failed');return {kind:'public',market};}});
  const view=useReadSnapshot(subject,selection,selected.isSuccess?selected.data:undefined,deniesRead(selected.error)||deniesRead(id?query.error:markets.error)||!selection||creating);
  const collection=useReadSnapshot(subject,`${q}:${status}:${environment}`,query.isSuccess?query.data:undefined,deniesRead(query.error)||deniesRead(selected.error));
  const list=collection.data?.pages.flatMap(p=>p.items)??[];
  function filter(name:string,value:string){const next=new URLSearchParams(params);value?next.set(name,value):next.delete(name);setParams(next,{replace:true});}
  function href(name:string,value:string){const next=new URLSearchParams(params);['connection','source','new'].forEach(k=>next.delete(k));next.set(name,value);return `/connections?${next}`;}
  function refresh(){void query.refetch();void markets.refetch();if(selection)void selected.refetch();}
  function changed(row:Connection){client.setQueryData(['connection-detail',subject,`private:${row.connection_id}`],{kind:'private',connection:row});void client.invalidateQueries({queryKey:['connections',subject]});setParams(new URLSearchParams({connection:row.connection_id,status:'all'}));}
  useEffect(()=>{const el=sentinel.current;if(!el)return;const observer=new IntersectionObserver(entries=>{if(entries.some(e=>e.isIntersecting)&&query.hasNextPage&&!query.isFetching&&!query.error)void query.fetchNextPage();});observer.observe(el);return()=>observer.disconnect();},[query.hasNextPage,query.isFetching,query.error,query.fetchNextPage]);
  const detail=view.data;
  const heading=creating?w('New connection','Новое подключение'):detail?.kind==='private'?(detail.connection.label||`${detail.connection.exchange_name} · ${detail.connection.market_type}`):detail?.kind==='public'?`${detail.market.exchange_name} · ${detail.market.market_type}`:w('Connections','Подключения');
  return <><CollectionLibrary title={w('Connections','Подключения')} actions={<><button type="button" className="library-icon-action" aria-label={w('New connection','Новое подключение')} title={w('New connection','Новое подключение')} disabled={!query.isSuccess} onClick={()=>setParams(new URLSearchParams({new:'1'}))}><Plus aria-hidden="true"/></button><RefreshButton pending={query.isFetching||markets.isFetching} onClick={refresh}/></>}
    filters={<><label>{w('Search','Поиск')}<input value={q} maxLength={80} onChange={e=>filter('q',e.target.value)}/></label><label>{w('Access','Доступ')}<select value={kind} onChange={e=>filter('kind',e.target.value)}><option value="all">{w('All','Все')}</option><option value="public">{w('Public data','Публичные данные')}</option><option value="private">{w('Private account','Приватный аккаунт')}</option></select></label>
      <label>{w('State','Состояние')}<select value={status} onChange={e=>filter('status',e.target.value)}>{['all','active','disabled','archived'].map(s=><option key={s} value={s}>{stateLabel(s,w)}</option>)}</select></label>
      <label>{w('Environment','Среда')}<select value={environment} onChange={e=>filter('environment',e.target.value)}><option value="">{w('All','Все')}</option><option value="testnet">Testnet</option><option value="mainnet">Mainnet</option></select></label></>}
    onReset={()=>{const next=new URLSearchParams(params);['q','kind','status','environment'].forEach(k=>next.delete(k));setParams(next,{replace:true});}}>
      {kind!=='private'&&<><h3 className="work-list-group">{w('Public data','Публичные данные')}</h3><ReadFeedback pending={markets.isFetching} initial={!markets.data} error={markets.error}/>
        {!deniesRead(markets.error)&&markets.data?.items.filter(m=>`${m.exchange_name} ${m.market_type}`.toLowerCase().includes(q.toLowerCase())&&(!environment||environment==='mainnet')).map(m=><Link className="work-object-row" key={m.market_id} to={href('source',String(m.market_id))} aria-current={source===String(m.market_id)?'true':undefined}><strong>{m.exchange_name} · {m.market_type}</strong><span>{w('Public market data','Публичные рыночные данные')}</span></Link>)}</>}
      {kind!=='public'&&<><h3 className="work-list-group">{w('Private accounts','Приватные аккаунты')}</h3><ReadFeedback pending={query.isFetching} retained={collection.retained} initial={!collection.data} error={query.error}/>
        {list.map(c=><Link key={c.connection_id} className="work-object-row" to={href('connection',c.connection_id)} aria-current={id===c.connection_id?'true':undefined}><strong>{c.label||`${c.exchange_name} · ${c.market_type}`}</strong><span>{c.environment} · {stateLabel(c.status,w)}</span></Link>)}
        {!query.isPending&&!query.error&&!list.length&&<p className="work-empty">{w('No connections','Нет подключений')}</p>}<div ref={sentinel}/></>}
    </CollectionLibrary><WorkHeading title={heading}/><div className="work-content">
    {creating?<ConnectionForm subject={subject} onSaved={changed} onCancel={()=>setParams(new URLSearchParams())}/>:<>
      <ReadFeedback pending={selected.isFetching} initial={!!selection&&!detail} retained={view.retained} error={selected.error}/>
      {detail?.kind==='private'?<ConnectionDetail key={detail.connection.connection_id} subject={subject} row={detail.connection} locked={view.retained||selected.isFetching} onChanged={changed}/>:detail?.kind==='public'?<PublicSource key={detail.market.market_id} subject={subject} market={detail.market} locked={view.retained||selected.isFetching}/>:!selection?<section className="panel work-empty">{w('Select a connection or data source.','Выберите подключение или источник данных.')}</section>:null}
    </>}</div></>;
}

function stateLabel(state:string,w:ReturnType<typeof useWords>){return ({all:w('All','Все'),active:w('Active','Активно'),disabled:w('Disconnected','Отключено'),archived:w('Archived','В архиве')})[state]??state;}
const needsRecentAuth=(error:unknown)=>error instanceof ApiError&&error.code==='recent_auth_required';
function connectionLabel(state:string,w:ReturnType<typeof useWords>){return ({
 read:w('Read only','Только чтение'),trade:w('Trading','Торговля'),none:w('None','Нет'),trading:w('Trading','Торговля'),
 valid_trade_enabled:w('Trading access verified','Торговый доступ проверен'),valid_readonly:w('Read access verified','Доступ на чтение проверен'),permission_mismatch:w('Permissions do not match','Права не соответствуют запросу'),
 skipped_external_validation:w('Not checked','Не проверено'),invalid_credentials:w('Invalid credentials','Неверные ключи'),invalid_permissions:w('Invalid permissions','Недопустимые права'),invalid_ip_restriction:w('IP restriction required','Нужно ограничение по IP'),unsupported_account_mode:w('Unsupported account mode','Режим аккаунта не поддерживается'),
 ready_for_trading:w('Ready for trading','Готово к торговле'),rejected:w('Rejected','Отклонено'),needs_action:w('Action required','Требуется действие'),disconnected:w('Disconnected','Отключено'),archived:w('Archived','В архиве'),
 active:w('Active','Активно'),paused:w('Paused','Приостановлено'),disabled:w('Disabled','Отключено')
 })[state]??w('Unknown','Неизвестно');}

function ConnectionBindings({subject,id}:{subject:string;id:string}){
 const w=useWords(),{i18n}=useTranslation();
 const query=useInfiniteQuery({queryKey:['connection-bindings',subject,id],initialPageParam:'',maxPages:5,retry:false,queryFn:({signal,pageParam})=>readConnectionBindings(id,pageParam,signal),getNextPageParam:p=>p.next_cursor??undefined});
 const view=useReadSnapshot(subject,id,query.isSuccess?query.data:undefined,deniesRead(query.error));
 const sentinel=useAutoLoad(query.hasNextPage,query.isFetching,query.error,query.fetchNextPage),rows=view.data?.pages.flatMap(p=>p.items)??[];
 return <NavigatorTable tabs={[{id:'bindings',label:w('Used by strategies','Используют стратегии'),content:<><ReadFeedback pending={query.isFetching} retained={view.retained} initial={!view.data} error={query.error}/><table><thead><tr><th>{w('Strategy','Стратегия')}</th><th>{w('State','Состояние')}</th><th>{w('Updated · UTC','Обновлено · UTC')}</th></tr></thead><tbody>{rows.map(b=><tr key={b.binding_id}><td><Link to={`/strategies/${encodeURIComponent(b.strategy_id)}`}>{b.strategy_id}</Link></td><td>{connectionLabel(b.binding_status,w)}</td><td>{formatTime(b.updated_at,i18n.language)}</td></tr>)}</tbody></table>{view.data&&!rows.length&&<p className="work-empty">{w('No bindings in this organization','В этой организации нет привязок')}</p>}<div ref={sentinel}/></>}]}/>;
}

function PublicSource({subject,market,locked}:{subject:string;market:Market;locked:boolean}) {
 const w=useWords(),{i18n}=useTranslation(),navigate=useNavigate();
 const query=useQuery({queryKey:['source-catalog',subject,market.market_id],queryFn:({signal})=>readCatalog(new URLSearchParams({market_id:String(market.market_id),limit:'1'}),signal),retry:false});
 const view=useReadSnapshot(subject,String(market.market_id),query.isSuccess?query.data:undefined,deniesRead(query.error));
 const command=useWorkSubmission(job=>navigate(`/data?market=${job.market_id}&journal=1&job=${encodeURIComponent(job.job_id)}`));
 const denied=deniesRead(query.error)||deniesRead(command.error);
 return <section className="panel"><ReadFeedback pending={query.isFetching} retained={view.retained} initial={!view.data} error={query.error}/>{!denied&&<><dl className="work-kv"><dt>{w('Provider','Провайдер')}</dt><dd>{market.exchange_name}</dd><dt>{w('Market','Рынок')}</dt><dd>{market.market_type}</dd><dt>{w('Access','Доступ')}</dt><dd>{w('Public · no private API key','Публичный · без приватного API-ключа')}</dd><dt>{w('Capabilities','Возможности')}</dt><dd>{w('Instrument catalog · historical 1m candles','Каталог инструментов · исторические свечи 1m')}</dd><dt>{w('Catalog','Каталог')}</dt><dd>{view.data?.snapshot_id?`${workState(view.data.catalog_state,w)} · ${view.data.total.toLocaleString(i18n.language)}`:w('Not loaded','Не загружен')}</dd><dt>{w('Last successful read · UTC','Успешное чтение · UTC')}</dt><dd>{formatTime(view.data?.refreshed_at,i18n.language)}</dd></dl>
 <div className="work-actions work-section-actions"><button disabled={locked||query.isFetching||view.retained||command.pending||command.unresolved} onClick={()=>void command.submit({kind:'catalog_refresh',market_id:market.market_id})}>{command.pending?w('Requesting…','Отправка…'):w('Check source and refresh catalog','Проверить источник и обновить каталог')}</button><Link className="work-link-button" to={`/data?market=${market.market_id}`}>{w('Instruments','Инструменты')}</Link></div></>}<WorkError error={command.error}/>{command.unresolved&&<button disabled={command.pending} onClick={()=>void command.reconcile()}>{w('Check saved request','Проверить сохранённое задание')}</button>}</section>;
}

function ConnectionDetail({subject,row,locked,onChanged}:{subject:string;row:Connection;locked:boolean;onChanged:(row:Connection)=>void}) {
 const w=useWords(),{i18n}=useTranslation(),[editing,setEditing]=useState(false),[pending,setPending]=useState(false),[error,setError]=useState<unknown>(),[unresolved,setUnresolved]=useState(false),[denied,setDenied]=useState(false),busy=useRef(false);
 async function command(action:string){if(busy.current||locked||unresolved)return;busy.current=true;setPending(true);setError(undefined);try{onChanged(await connectionCommand(`/${encodeURIComponent(row.connection_id)}/${action}`));}catch(e){setError(e);setUnresolved(e instanceof ApiError&&e.outcome==='unknown');setDenied(deniesRead(e)&&!needsRecentAuth(e));}finally{busy.current=false;setPending(false);}}
 if(denied)return <WorkError error={error}/>;
 if(editing)return <ConnectionForm subject="" existing={row} locked={locked} onSaved={value=>{setEditing(false);onChanged(value);}} onCancel={()=>setEditing(false)}/>;
 const disabled=locked||pending||unresolved||needsRecentAuth(error);
 return <><section className="panel"><dl className="work-kv">
  <dt>{w('Provider / market','Провайдер / рынок')}</dt><dd>{row.exchange_name} · {row.market_type}</dd><dt>{w('Environment','Среда')}</dt><dd>{row.environment}</dd>
  <dt>{w('Connection','Подключение')}</dt><dd>{stateLabel(row.status,w)}</dd><dt>{w('Validation','Проверка')}</dt><dd>{connectionLabel(row.validation_status,w)}</dd><dt>{w('Readiness','Готовность')}</dt><dd>{connectionLabel(row.connection_readiness,w)}</dd>
  <dt>{w('Requested access','Запрошенный доступ')}</dt><dd>{connectionLabel(row.requested_permissions,w)}</dd><dt>{w('Effective access','Фактический доступ')}</dt><dd>{connectionLabel(row.effective_permissions,w)}</dd>
  <dt>{w('Last checked · UTC','Проверено · UTC')}</dt><dd>{formatTime(row.last_validated_at,i18n.language)}</dd><dt>{w('Used by strategies','Используют стратегии')}</dt><dd>{row.used_by_strategies_count} · {w('active bindings','активных привязок')}: {row.active_strategy_bindings_count}</dd>
  </dl><WorkError error={error}/>{needsRecentAuth(error)&&<RecentAuth onVerified={()=>setError(undefined)}/>}<div className="work-actions work-section-actions"><button disabled={disabled||row.status==='archived'} onClick={()=>void command('validate')}>{pending?w('Checking…','Проверка…'):w('Check connection','Проверить подключение')}</button>
  <button disabled={disabled||row.status!=='active'} onClick={()=>setEditing(true)}>{w('Replace credentials','Заменить ключи')}</button>
  <Confirm id="disconnect" title={w('Disconnect this account?','Отключить этот аккаунт?')} help={w('Active strategy dependencies are checked by the server.','Сервер проверит зависимости активных стратегий.')} source={row.label||row.exchange_name} trigger={w('Disconnect','Отключить')} confirmText={w('Disconnect','Отключить')} disabled={disabled||row.status!=='active'||row.active_strategy_bindings_count>0} onConfirm={()=>void command('disable')}/>
  <Confirm id="archive-connection" title={w('Archive this connection?','Архивировать подключение?')} help={w('The disconnected connection will leave the active list. Its history remains available.','Отключённое подключение исчезнет из активного списка. История сохранится.')} source={row.label||row.exchange_name} trigger={w('Archive','Архивировать')} confirmText={w('Archive','Архивировать')} disabled={disabled||row.status!=='disabled'||row.active_strategy_bindings_count>0} onConfirm={()=>void command('archive')}/>
  </div>{unresolved&&<button onClick={async()=>{try{onChanged(await readConnection(row.connection_id));setUnresolved(false);setError(undefined);}catch(e){setError(e);setDenied(deniesRead(e)&&!needsRecentAuth(e));}}}>{w('Check saved state','Проверить сохранённое состояние')}</button>}</section><ConnectionBindings subject={subject} id={row.connection_id}/></>;
}

function ConnectionForm({existing,locked=false,onSaved,onCancel}:{subject:string;existing?:Connection;locked?:boolean;onSaved:(row:Connection)=>void;onCancel:()=>void}) {
 const w=useWords(),form=useRef<HTMLFormElement>(null),busy=useRef(false),[dirty,setDirty]=useState(false),[pending,setPending]=useState(false),[error,setError]=useState<unknown>(),[unresolved,setUnresolved]=useState(false),[saved,setSaved]=useState<Connection>(),[cancelled,setCancelled]=useState(false);
 useEffect(()=>{if(!pending&&!dirty){if(saved)onSaved(saved);else if(cancelled)onCancel();}},[pending,dirty,saved,cancelled,onSaved,onCancel]);
 async function submit(event:FormEvent){event.preventDefault();if(locked||busy.current||unresolved||needsRecentAuth(error)||!form.current)return;
   const data=new FormData(form.current);if(!form.current.reportValidity())return;
   busy.current=true;setPending(true);setError(undefined);
   const secrets={api_key:String(data.get('api_key')??''),api_secret:String(data.get('api_secret')??'')};
   const body=existing?secrets:{...secrets,exchange_name:data.get('exchange_name'),market_type:data.get('market_type'),environment:data.get('environment'),label:data.get('label')||null,permissions:data.get('permissions')};
   try{const row=await connectionCommand(existing?`/${encodeURIComponent(existing.connection_id)}/rotate`:'',body);setDirty(false);setSaved(row);}
   catch(e){setError(e);setUnresolved(e instanceof ApiError&&e.outcome==='unknown');}
   finally{form.current?.querySelectorAll<HTMLInputElement>('input[type=password]').forEach(input=>{input.value='';});secrets.api_key='';secrets.api_secret='';body.api_key='';body.api_secret='';busy.current=false;setPending(false);}
 }
 if(deniesRead(error)&&!needsRecentAuth(error))return <WorkError error={error}/>;
 return <section className="panel"><form ref={form} className="work-form" onSubmit={submit} onChange={()=>setDirty(true)} autoComplete="off"><DirtyGuard dirty={dirty||pending}/><h2>{existing?w('Replace credentials','Заменить ключи'):w('Private exchange account','Приватный биржевой аккаунт')}</h2>
 <fieldset className="work-fields" disabled={locked||pending||unresolved||needsRecentAuth(error)}>{!existing&&<>
  <Field name="label" label={w('Name','Название')}><input id="label" name="label" maxLength={80}/></Field>
  <Field name="exchange_name" label={w('Exchange','Биржа')}><select id="exchange_name" name="exchange_name"><option value="binance">Binance</option><option value="bybit">Bybit</option></select></Field>
  <Field name="market_type" label={w('Market','Рынок')}><select id="market_type" name="market_type"><option value="spot">Spot</option><option value="futures">Futures</option></select></Field>
  <Field name="environment" label={w('Environment','Среда')}><select id="environment" name="environment" defaultValue="testnet"><option value="testnet">Testnet</option><option value="mainnet">Mainnet</option></select></Field>
  <Field name="permissions" label={w('Requested access','Запрошенный доступ')}><select id="permissions" name="permissions"><option value="read">{w('Read only','Только чтение')}</option><option value="trade">{w('Trading','Торговля')}</option></select></Field>
 </>}<Field name="api_key" label="API key"><input id="api_key" name="api_key" type="password" autoComplete="new-password" required maxLength={4096}/></Field><Field name="api_secret" label="API secret"><input id="api_secret" name="api_secret" type="password" autoComplete="new-password" required maxLength={4096}/></Field></fieldset>
 <WorkError error={error}/>{needsRecentAuth(error)&&<RecentAuth onVerified={()=>setError(undefined)}/>}<div className="work-actions"><button type="submit" className="primary" disabled={locked||pending||unresolved||needsRecentAuth(error)}>{pending?w('Saving…','Сохранение…'):w('Save connection','Сохранить подключение')}</button><button type="button" disabled={pending} onClick={()=>{if(!dirty||window.confirm(w('Discard unsaved changes?','Отменить несохранённые изменения?'))){setDirty(false);form.current?.reset();setCancelled(true);}}}>{w('Cancel','Отмена')}</button></div>{unresolved&&<p role="status">{w('Return to the list and check whether the connection was saved before starting again.','Вернитесь к списку и проверьте, сохранилось ли подключение, прежде чем создавать его снова.')}</p>}</form></section>;
}
