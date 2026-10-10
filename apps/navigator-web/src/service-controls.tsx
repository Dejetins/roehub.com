import {useEffect,useRef,useState} from 'react';
import {useQuery} from '@tanstack/react-query';
import {z} from 'zod';
import {ApiError,requestJson} from './api';
import {readStorageAccess} from './storage-api';
import {ReadFeedback,useWords,WorkError} from './workspace-ui';

const resultSchema=z.object({operation_id:z.string(),state:z.enum(['accepted','running','succeeded','failed','unknown','rejected']),detail_code:z.string().nullable().optional()});
const receiptSchema=z.object({key:z.string().uuid(),id:z.string().uuid(),action:z.enum(['start','stop']),profile:z.string(),service:z.string()});
type Receipt=z.infer<typeof receiptSchema>;
// Same UUIDv5 contract as admin._operation_id; permits lookup after a lost response.
export async function operationId(org:string,key:string){
 const namespace=Uint8Array.from(org.replaceAll('-','').match(/../g)!.map(x=>parseInt(x,16)));
 const name=new TextEncoder().encode(`admin-operation:${key}`),input=new Uint8Array(namespace.length+name.length);
 input.set(namespace);input.set(name,namespace.length);
 const hash=new Uint8Array(await crypto.subtle.digest('SHA-1',input)).slice(0,16);
 hash[6]=(hash[6]&15)|80;hash[8]=(hash[8]&63)|128;
 const hex=Array.from(hash,x=>x.toString(16).padStart(2,'0')).join('');
 return `${hex.slice(0,8)}-${hex.slice(8,12)}-${hex.slice(12,16)}-${hex.slice(16,20)}-${hex.slice(20)}`;
}
export function ServiceControls({subject,org,profile,service,commands,onChange}:{subject:string;org:string;profile:string;service:string;commands:readonly string[];onChange:()=>void}){
 const w=useWords(),storageKey=`roehub:service-operation:${subject}:${org}:${service}`;
 const [saved]=useState(()=>{try{const raw=sessionStorage.getItem(storageKey);return {receipt:raw?receiptSchema.parse(JSON.parse(raw)):null,error:false};}catch{return {receipt:null,error:true};}});
 const [receipt,setReceipt]=useState<Receipt|null>(saved.receipt);
 const [failure,setFailure]=useState<unknown>(null),[pending,setPending]=useState(false),[message,setMessage]=useState('');
 const lock=useRef(false),alive=useRef(true);
 useEffect(()=>{alive.current=true;return()=>{alive.current=false;};},[]);
 const access=useQuery({queryKey:['private',subject,'service-access',org],queryFn:({signal})=>readStorageAccess(org,signal),retry:false,refetchInterval:15000});
 const allowed=access.isSuccess&&access.data.organization_id===org&&access.data.installation_owner&&access.data.permissions.includes('operations.execute');
 const recent=allowed&&access.data?.recent_auth;
 const base=`/api/v1/admin/organizations/${encodeURIComponent(org)}/operations`;
 const result=useQuery({queryKey:['private',subject,'service-operation',org,receipt?.id],enabled:allowed&&!!receipt,retry:false,gcTime:0,
  queryFn:({signal})=>requestJson(`${base}/${receipt!.id}`,resultSchema,{signal,idempotencyKey:receipt!.key}).then(r=>r.data),
  refetchInterval:q=>q.state.data&&['succeeded','failed','rejected'].includes(q.state.data.state)?false:3000});
 function remember(value:Receipt|null){if(value)sessionStorage.setItem(storageKey,JSON.stringify(value));else sessionStorage.removeItem(storageKey);setReceipt(value);}
 useEffect(()=>{if(allowed&&receipt&&result.data&&['succeeded','failed','rejected'].includes(result.data.state)){
  setMessage(result.data.state==='succeeded'?w('Operation completed. Refreshing service status.','Операция завершена. Обновляем состояние сервиса.'):w('Operation failed. Check service status before trying again.','Операция не выполнена. Проверьте состояние сервиса.'));
  try{sessionStorage.removeItem(storageKey);setReceipt(null);}catch{setFailure(new Error('Operation recovery storage unavailable'));}
  onChange();
 }},[result.data,allowed]);
 async function submit(action:string){
  if(lock.current||receipt||saved.error||!recent||!commands.includes(action))return;
  lock.current=true;setPending(true);setFailure(null);setMessage('');
  let receiptSaved=false;
  try{
   const key=crypto.randomUUID(),id=await operationId(org,key);
   if(!alive.current)return;
   remember({key,id,action:action as 'start'|'stop',profile,service});receiptSaved=true;
   const response=await requestJson(base,resultSchema,{method:'POST',idempotencyKey:key,body:{action,profile,services:[service]}});
   if(!alive.current)return;
   if(['succeeded','failed','rejected'].includes(response.data.state)){
    remember(null);setMessage(response.data.state==='succeeded'?w('Operation completed.','Операция завершена.'):w('Operation failed.','Операция не выполнена.'));onChange();
   }
  }catch(error){if(alive.current){setFailure(error);if(receiptSaved&&error instanceof ApiError&&error.outcome==='failed'&&error.status!==null&&error.status<500)remember(null);}}
  finally{lock.current=false;if(alive.current)setPending(false);}
 }
 async function retrySame(){
  if(!receipt||lock.current||!recent||receipt.service!==service||receipt.profile!==profile||!commands.includes(receipt.action))return;
  lock.current=true;setPending(true);setFailure(null);
  try{await requestJson(base,resultSchema,{method:'POST',idempotencyKey:receipt.key,body:{action:receipt.action,profile:receipt.profile,services:[receipt.service]}});if(alive.current)await result.refetch();}
  catch(error){if(alive.current)setFailure(error);}finally{lock.current=false;if(alive.current)setPending(false);}
 }
 async function reconcile(){
  if(!receipt||lock.current||!allowed)return;lock.current=true;setPending(true);setFailure(null);
  try{await requestJson(`${base}/${receipt.id}:reconcile`,resultSchema,{method:'POST',idempotencyKey:receipt.key});if(alive.current)await result.refetch();}
  catch(error){if(alive.current)setFailure(error);}finally{lock.current=false;if(alive.current)setPending(false);}
 }
 return <div className="work-section-actions">
  <ReadFeedback pending={access.isFetching} initial={!access.data} error={access.error}/>
  {saved.error&&<p role="alert">{w('Operation recovery data is unavailable; controls are locked.','Данные восстановления операции недоступны; управление заблокировано.')}</p>}
  {allowed&&<><div className="work-actions">{commands.includes('start')&&<button disabled={!recent||pending||!!receipt||saved.error||access.isFetching} onClick={()=>void submit('start')}>{w('Start service','Запустить сервис')}</button>}{commands.includes('stop')&&<button disabled={!recent||pending||!!receipt||saved.error||access.isFetching} onClick={()=>void submit('stop')}>{w('Stop service','Остановить сервис')}</button>}</div>
   {!recent&&<p>{w('Sign in again to manage services.','Войдите заново, чтобы управлять сервисами.')}</p>}
   {receipt&&<p role="status">{w('Checking the existing operation; a new command is blocked.','Проверяем текущую операцию; новая команда заблокирована.')} {result.error instanceof ApiError&&result.error.status===404&&<button disabled={pending||!recent} onClick={()=>void retrySame()}>{w('Retry the same request','Повторить тот же запрос')}</button>} {result.data?.state==='unknown'&&<button disabled={pending} onClick={()=>void reconcile()}>{w('Check outcome','Сверить результат')}</button>}</p>}
   {message&&<p role="status">{message}</p>}{(failure||result.error)&&<WorkError error={failure||result.error}/>}</>}
 </div>;
}
