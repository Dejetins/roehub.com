import {useEffect,useRef,useState} from 'react';
import {useBlocker,useSearchParams} from 'react-router';
import {Cpu,Database,HardDrive,MemoryStick,RefreshCw,Sparkles} from 'lucide-react';
import {ArtifactActivity} from './storage-activity';
import {useQuery,useQueryClient} from '@tanstack/react-query';
import {ApiError} from './api';
import {useReadSnapshot} from './read-snapshot';
import {deniesRead,Field,ReadFeedback,useWords,WorkError} from './workspace-ui';
import {readStorageOrganizations,readStorageAccess,readPublicStorage,readStorage,storageCommand,type Storage,type StorageCommand} from './storage-api';
import './storage-settings.css';

export function StorageSettings({subject}:{subject:string}) {
 const w=useWords(),client=useQueryClient();
 const organizations=useQuery({queryKey:['private',subject,'storage-organizations'],queryFn:({signal})=>readStorageOrganizations(signal)});
 const [selected,setSelected]=useState('');
 const org=selected||organizations.data?.[0]?.organization.organization_id||'';
 const access=useQuery({queryKey:['private',subject,'storage-access',org],enabled:!!org,queryFn:({signal})=>readStorageAccess(org,signal),refetchInterval:15_000});
 const safe=useQuery({queryKey:['private',subject,'storage-public'],queryFn:({signal})=>readPublicStorage(signal)});
 const allowed=organizations.isSuccess&&access.isSuccess&&access.data.organization_id===org&&access.data.installation_owner&&access.data.permissions.includes('operations.execute');
 useEffect(()=>{if(!allowed){void client.cancelQueries({queryKey:['storage-owner',subject]});client.removeQueries({queryKey:['storage-owner',subject]});}},[allowed,client,subject]);
 useEffect(()=>()=>{void client.cancelQueries({queryKey:['storage-owner',subject]});client.removeQueries({queryKey:['storage-owner',subject]});},[client,subject]);
 return <section className="panel storage-panel"><header className="storage-heading"><h2>{w('Data and storage','Данные и хранение')}</h2><p>{w('Resources for calculations and space for prepared market data.','Ресурсы для расчётов и место для подготовленных рыночных данных.')}</p></header>
 <ReadFeedback pending={(safe.isFetching&&!safe.data)||(organizations.isFetching&&!organizations.data)||(access.isFetching&&!access.data)} initial={!safe.data} error={safe.error||organizations.error||access.error}/>
 
 {organizations.data&&organizations.data.length>1&&<Field name="storage-org" label={w('Organization','Организация')}><select id="storage-org" value={org} onChange={e=>setSelected(e.target.value)}>{organizations.data.map(o=><option key={o.organization.organization_id} value={o.organization.organization_id}>{o.organization.display_name}</option>)}</select></Field>}

 {allowed?<OwnerStorage key={`${subject}:${org}`} subject={subject} org={org} accessRefreshing={access.isFetching}/>:organizations.isSuccess&&access.isSuccess?<p>{w('Installation-owner access is required to inspect the server path and manage storage. Organization ownership alone does not grant this access.','Для просмотра пути и управления хранилищем нужны права владельца установки. Владение организацией не даёт этих прав.')}</p>:null}
 </section>;
}
function OwnerStorage({subject,org,accessRefreshing}:{subject:string;org:string;accessRefreshing:boolean}) {
 const w=useWords(),client=useQueryClient(),[params,setParams]=useSearchParams();
 const key=['storage-owner',subject,org];
 const query=useQuery({queryKey:key,queryFn:({signal})=>readStorage(org,signal),gcTime:0});
 const [failure,setFailure]=useState<unknown>(null),[pending,setPending]=useState(false),[denied,setDenied]=useState(false);
 const [outcome,setOutcome]=useState('');
 const [result,setResult]=useState<Storage|null>(null),[now,setNow]=useState(Date.now),[deadline,setDeadline]=useState(0);
 const [budget,setBudget]=useState(''),[schedule,setSchedule]=useState<Storage['schedule']>(null);
 const [target,setTarget]=useState({exchange:'binance',market_type:'spot',symbol:''});
 const [cleanupOpen,setCleanupOpen]=useState(false);
 const edited=useRef(false);
 const lock=useRef(false),alive=useRef(true),controller=useRef<AbortController|null>(null);
 const recoveryKey=`roehub:storage-unresolved:${subject}:${org}`;
 const [recovery,setRecovery]=useState(()=>{try{return sessionStorage.getItem(recoveryKey)||params.get('storage-unresolved');}catch{return 'recovery-storage-unavailable';}});
 const unknown=!!recovery;
 function remember(value:string|null){try{if(value)sessionStorage.setItem(recoveryKey,value);else sessionStorage.removeItem(recoveryKey);}catch{throw new Error('recovery-storage-unavailable');}setRecovery(value);setParams(p=>{if(value)p.set('storage-unresolved',value);else p.delete('storage-unresolved');return p;},{replace:true});}
 const blocked=denied||deniesRead(query.error);
 const snapshot=useReadSnapshot(`${subject}:${org}:owner`,'root',query.isSuccess?query.data:undefined,blocked);
 const data=snapshot.data;
 const blockedRef=useRef(blocked);blockedRef.current=blocked;
 useEffect(()=>{alive.current=true;return()=>{alive.current=false;controller.current?.abort();};},[]);
 useEffect(()=>{const timer=setInterval(()=>setNow(Date.now()),1000);return()=>clearInterval(timer);},[]);
 useEffect(()=>{if(query.data){setSchedule(query.data.schedule);if(!edited.current)setBudget(String(query.data.budget_bytes??''));setResult(null);}},[query.data]);
 useEffect(()=>{if(blocked){setDenied(true);controller.current?.abort();setResult(null);setBudget('');setSchedule(null);void client.cancelQueries({queryKey:key});client.removeQueries({queryKey:key});}},[blocked,client,subject,org]);
 const until=Math.max(deadline,query.errorUpdatedAt+(query.error instanceof ApiError?(query.error.retryAfterSeconds??0)*1000:0));
 const disabled=!data||blocked||snapshot.retained||query.isFetching||accessRefreshing||pending||unknown||now<until;
 const bytes=(value:number|null|undefined)=>formatStorageSize(value,w('en','ru'));
 const dirty=!!data&&budget!==String(data.budget_bytes??'');
 const validBudget=Number.isSafeInteger(Number(budget))&&Number(budget)>=1;
 const maxBudget=Math.max(100*GIB,Number(data?.budget_bytes??0));
 const budgetOptions=Array.from(new Set([GIB,2*GIB,5*GIB,10*GIB,20*GIB,50*GIB,100*GIB,maxBudget,...(data?.budget_bytes?[data.budget_bytes]:[])])).sort((a,b)=>a-b);
 const budgetIndex=Math.max(0,budgetOptions.indexOf(Number(budget)));
 const usage=data?.used_bytes!=null&&data.budget_bytes?Math.min(100,100*data.used_bytes/data.budget_bytes):null;
 const validToken=!!result?.expires_at&&Date.parse(result.expires_at)>now&&result.control_version===data?.control_version;
 async function run(command:StorageCommand,prepare=false) {
  if(disabled||lock.current)return;lock.current=true;setPending(true);setFailure(null);setOutcome('');controller.current=new AbortController();
  const id=crypto.randomUUID(),durable=command.action==='configure'||command.action==='cleanup_confirm';
  let dispatched=false;
  try {
   if(prepare){
    const probe=await storageCommand(org,{action:'validate',root_id:command.root_id,expected_version:command.expected_version},crypto.randomUUID(),controller.current.signal);
    if(!alive.current||blockedRef.current||controller.current.signal.aborted)return;
    if(probe.state!=='validated'||!probe.validation_id||!probe.expires_at||Date.parse(probe.expires_at)<=Date.now()||probe.control_version!==command.expected_version)throw new Error('Storage validation failed');
    command={...command,validation_id:probe.validation_id};
   }
   if(durable)remember(id);
   dispatched=true;
   let reply:Storage;
   try{reply=await storageCommand(org,command,id,controller.current.signal);}
   catch(error){
    if(command.action!=='cleanup_preview'||command.slot!=='slot_a'||!(error instanceof ApiError)||error.code!=='cleanup_current')throw error;
    if(!alive.current||blockedRef.current||controller.current.signal.aborted)return;
    reply=await storageCommand(org,{...command,slot:'slot_b'},crypto.randomUUID(),controller.current.signal);
   }if(!alive.current||blockedRef.current||controller.current?.signal.aborted)return;
   if(durable&&reply.state!=='unknown')remember(null);
   await client.cancelQueries({queryKey:key});if(!alive.current||blockedRef.current||controller.current?.signal.aborted)return;setResult(reply);setOutcome(reply.state);
   if(reply.state==='configured'||reply.state==='cleaned'){if(reply.state==='configured')edited.current=false;client.setQueryData(key,{...reply,state:'ready'});void client.invalidateQueries({queryKey:['private',subject,'storage-public']});void query.refetch();}
  } catch(error) {if(!alive.current||blockedRef.current||controller.current?.signal.aborted)return;setFailure(error);setResult(null);
   if(error instanceof ApiError){setDeadline(Date.now()+(error.retryAfterSeconds??0)*1000);if(deniesRead(error))setDenied(true);
    if(durable&&dispatched&&error.outcome!=='unknown')remember(null);}
  } finally {lock.current=false;if(alive.current)setPending(false);}
 }
 const command=(action:StorageCommand['action']):StorageCommand=>({action,root_id:'backtest-primary',expected_version:data!.control_version});
 return <div className="storage-owner"><StorageDraftGuard dirty={dirty||pending}/>
 <div className="storage-read-row"><ReadFeedback pending={query.isFetching} initial={!data&&!blocked} retained={snapshot.retained} error={query.error??(denied?new ApiError('forbidden',403,'failed'):null)}/>
 <button type="button" className="library-icon-action work-icon" aria-label={w('Refresh storage','Обновить состояние')} title={w('Refresh storage','Обновить состояние')} disabled={pending||query.isFetching||now<until||blocked} onClick={()=>void query.refetch()}><RefreshCw aria-hidden="true"/></button></div>
 {unknown&&!pending&&<p role="alert">{w('Operation outcome unresolved. Check the existing operation before saving or deleting again.','Исход операции не подтверждён. Проверьте результат текущей операции перед повторным сохранением или удалением.')} <code>{recovery}</code></p>}
 {data&&<>
 <ArtifactActivity subject={subject}/>
 <section className="storage-section" aria-labelledby="storage-resource-title"><h3 id="storage-resource-title">{w('Artifact calculation limits','Лимиты для расчётов артефактов')}</h3>
 <p>{w('Maximum resources available to calculations.','Максимум ресурсов, доступных для расчётов.')}</p>
 <div className="storage-resource-grid">{[{id:'cpu',title:w('Processor','Процессор'),icon:Cpu},{id:'memory',title:w('Memory','Оперативная память'),icon:MemoryStick}].map(({id,title,icon:Icon})=><div className="storage-resource" key={id}>
 <label htmlFor={`storage-${id}`}><span><Icon aria-hidden="true"/>{title}</span><span className="storage-muted">{w('Managed by installation','Задано в установке')}</span></label>
 <input id={`storage-${id}`} type="range" min="0" max="1" value="0" disabled aria-valuetext={w('Limit unavailable','Лимит недоступен')} aria-describedby="storage-resource-help"/>
 </div>)}</div>
 <p id="storage-resource-help" className="storage-hint">{w('This installation does not yet support changing CPU and memory limits here.','Эта установка пока не поддерживает изменение лимитов CPU и памяти на этой странице.')}</p>
 <div className="storage-ml"><Sparkles aria-hidden="true"/><div><strong>{w('ML models','ML-модели')}</strong><p>{w('Separate resource limits will appear when the module is connected.','Отдельные лимиты ресурсов появятся после подключения модуля.')}</p></div><span className="storage-badge">{w('Later','Позже')}</span></div>
 </section>
 <section className="storage-section storage-disk" aria-labelledby="storage-disk-title"><h3 id="storage-disk-title">{w('Data storage','Хранение данных')}</h3><p>{w('Prepared files speed up backtests. Candle history is stored separately.','Подготовленные файлы ускоряют бэктесты. История свечей хранится отдельно.')}</p>
 <div className="storage-usage"><div><span><Database aria-hidden="true"/>{w('Prepared data','Подготовленные данные')}</span><strong>{bytes(data.used_bytes)}</strong></div>
 {usage!==null&&<progress max="100" value={usage} aria-label={w('Prepared data disk usage','Использование места подготовленными данными')}/>}
 <div className="storage-hint"><span>{w('Current limit','Текущий лимит')}: {bytes(data.budget_bytes)}</span><span>{w('Free on disk','Свободно на диске')}: {bytes(data.free_bytes)}</span></div></div>
 <form id="storage-settings-form" onSubmit={e=>{e.preventDefault();if(!dirty||!validBudget||!schedule)return;void run({...command('configure'),budget_bytes:Number(budget),schedule},true);}}>
 <fieldset disabled={disabled} className="storage-resource"><label htmlFor="storage-budget"><span><HardDrive aria-hidden="true"/>{w('Prepared data disk limit','Лимит для подготовленных данных')}</span><output htmlFor="storage-budget">{bytes(budget?Number(budget):null)}</output></label>
 <input id="storage-budget" type="range" min="0" max={budgetOptions.length-1} step="1" value={budgetIndex} aria-valuetext={bytes(budget?Number(budget):null)} onChange={e=>{edited.current=true;setBudget(String(budgetOptions[Number(e.target.value)]));setOutcome('');}}/>
 <div className="storage-range-labels"><span>{bytes(budgetOptions[0])}</span><span>{bytes(budgetOptions.at(-1))}</span></div></fieldset></form>
 <div className="storage-cleanup-entry"><button type="button" aria-expanded={cleanupOpen} aria-controls="storage-cleanup" disabled={disabled} onClick={()=>{setCleanupOpen(!cleanupOpen);setResult(null);}}>{w('Review unused files','Посмотреть, что можно очистить')}</button><span className="storage-hint">{w('Candle history and files in use are protected.','История свечей и используемые файлы сохранятся.')}</span></div>
 {cleanupOpen&&<div id="storage-cleanup" className="storage-cleanup"><p>{w('Choose an instrument to review its unused prepared files before deletion.','Выберите инструмент: перед удалением покажем объём его неиспользуемых файлов.')}</p>
 <fieldset disabled={disabled}><div className="storage-grid">
 <Field name="cleanup-exchange" label={w('Exchange','Биржа')}><select id="cleanup-exchange" value={target.exchange} onChange={e=>{setTarget({...target,exchange:e.target.value});setResult(null);}}><option value="binance">Binance</option><option value="bybit">Bybit</option></select></Field>
 <Field name="cleanup-market" label={w('Market','Рынок')}><select id="cleanup-market" value={target.market_type} onChange={e=>{setTarget({...target,market_type:e.target.value});setResult(null);}}><option value="spot">Spot</option><option value="futures">Futures</option></select></Field>
 <Field name="cleanup-symbol" label={w('Symbol','Инструмент')}><input id="cleanup-symbol" placeholder="BTCUSDT" value={target.symbol} maxLength={64} onChange={e=>{setTarget({...target,symbol:e.target.value.toUpperCase().trim()});setResult(null);}}/></Field></div>
 <button type="button" disabled={!target.symbol.trim()} onClick={()=>void run({...command('cleanup_preview'),coordinate:target,slot:'slot_a'})}>{w('Preview cleanup','Проверить очистку')}</button>
 {result?.state==='preview'&&<div className="storage-cleanup-preview" role="group" aria-label={w('Confirm cleanup','Подтверждение очистки')}><p>{target.exchange} / {target.market_type} / {target.symbol}</p><p>{w('Unused files eligible for deletion','Можно удалить неиспользуемые файлы')}: <strong>{bytes(result.eligible_bytes)}</strong></p><div className="work-actions"><button type="button" disabled={!validToken} onClick={()=>void run({...command('cleanup_confirm'),preview_id:result.preview_id!})}>{w('Delete reviewed files','Удалить проверенные файлы')}</button><button type="button" onClick={()=>setResult(null)}>{w('Cancel cleanup','Отменить очистку')}</button></div>{!validToken&&<p>{w('Preview expired. Review the files again.','Проверка устарела. Проверьте файлы повторно.')}</p>}</div>}
 </fieldset></div>}
 </section>
 </>}
 {!blocked&&<WorkError error={failure}/>}
 <footer className="storage-footer"><span role="status">{pending?w('Saving…','Выполняется…'):unknown?w('Result unknown','Результат неизвестен'):outcome==='configured'?w('Settings saved','Настройки сохранены'):outcome==='cleaned'?w('Cleanup completed','Очистка завершена'):result?.state==='relocation_required'?w('Storage relocation requires a separate operation.','Для переноса хранилища нужна отдельная операция.'):dirty?w('Unsaved changes','Есть несохранённые изменения'):''}</span><div className="work-actions"><button type="button" disabled={!dirty||pending||blocked} onClick={()=>{edited.current=false;setBudget(String(data?.budget_bytes??''));setFailure(null);setOutcome('');}}>{w('Cancel','Отмена')}</button><button type="submit" form="storage-settings-form" className="primary" disabled={disabled||!dirty||!validBudget||!schedule}>{w('Save changes','Сохранить изменения')}</button></div></footer>
 </div>;
}

const GIB=1024**3;
export function formatStorageSize(value:number|null|undefined,locale='en') {
 if(value==null)return locale==='ru'?'Недоступно':'Unavailable';
 const units=locale==='ru'?['Б','КБ','МБ','ГБ','ТБ']:['B','KB','MB','GB','TB'];
 const index=value>0?Math.min(4,Math.floor(Math.log(value)/Math.log(1024))):0;
 return `${new Intl.NumberFormat(locale,{maximumFractionDigits:1}).format(value/1024**index)} ${units[index]}`;
}

function StorageDraftGuard({dirty}:{dirty:boolean}) {
 const w=useWords(),prompted=useRef(false);
 const blocker=useBlocker(({currentLocation,nextLocation})=>{
  const current=new URLSearchParams(currentLocation.search),next=new URLSearchParams(nextLocation.search);
  current.delete('storage-unresolved');next.delete('storage-unresolved');current.sort();next.sort();
  return dirty&&(currentLocation.pathname!==nextLocation.pathname||current.toString()!==next.toString());
 });
 useEffect(()=>{if(blocker.state!=='blocked'){prompted.current=false;return;}if(prompted.current)return;prompted.current=true;
  if(window.confirm(w('Discard unsaved changes?','Отменить несохранённые изменения?')))blocker.proceed();else blocker.reset();
 },[blocker,w]);
 useEffect(()=>{const beforeUnload=(e:BeforeUnloadEvent)=>{if(dirty){e.preventDefault();e.returnValue='';}};window.addEventListener('beforeunload',beforeUnload);return()=>window.removeEventListener('beforeunload',beforeUnload);},[dirty]);
 return null;
}
