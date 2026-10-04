import {useRef,useState} from 'react';
import {z} from 'zod';
import {useLocation} from 'react-router';
import {requestJson} from './api';
import {useWords} from './workspace-ui';

const optionsSchema=z.object({challenge_id:z.string(),publicKey:z.object({challenge:z.string(),rpId:z.string().optional(),timeout:z.number().optional(),userVerification:z.enum(['required','preferred','discouraged']).optional(),allowCredentials:z.array(z.object({id:z.string(),type:z.literal('public-key')})).optional()})});
const decode=(value:string)=>Uint8Array.from(atob(value.replaceAll('-','+').replaceAll('_','/')),c=>c.charCodeAt(0));
const encode=(value:ArrayBuffer)=>btoa(String.fromCharCode(...new Uint8Array(value))).replaceAll('+','-').replaceAll('/','_').replaceAll('=','');

/** Existing passkey step-up, with session rotation and double-submit CSRF. */
export function RecentAuth({onVerified}:{onVerified:()=>void}){
 const location=useLocation(),login=`/login?${new URLSearchParams({next:location.pathname+location.search})}`,reauth=`/logout?${new URLSearchParams({next:login})}`;
 const w=useWords(),busy=useRef(false),[pending,setPending]=useState(false),[failed,setFailed]=useState(false);
 async function verify(){if(busy.current)return;busy.current=true;setPending(true);setFailed(false);try{
  const {data}=await requestJson('/api/auth/local/recent-auth/options',optionsSchema,{method:'POST',csrf:true});
  const credential=await navigator.credentials.get({publicKey:{...data.publicKey,challenge:decode(data.publicKey.challenge),allowCredentials:data.publicKey.allowCredentials?.map(c=>({...c,id:decode(c.id)}))}}) as PublicKeyCredential|null;
  if(!credential)throw new Error('not_verified');
  const response=credential.response as AuthenticatorAssertionResponse;
  await requestJson('/api/auth/local/recent-auth/complete',z.object({authenticated:z.literal(true)}),{method:'POST',csrf:true,body:{challenge_id:data.challenge_id,credential:{id:credential.id,rawId:encode(credential.rawId),type:credential.type,authenticatorAttachment:credential.authenticatorAttachment,clientExtensionResults:credential.getClientExtensionResults(),response:{clientDataJSON:encode(response.clientDataJSON),authenticatorData:encode(response.authenticatorData),signature:encode(response.signature),userHandle:response.userHandle?encode(response.userHandle):null}}}});
  onVerified();
 }catch{setFailed(true);}finally{busy.current=false;setPending(false);}}
 return <div className="work-reauth"><button type="button" disabled={pending} onClick={()=>void verify()}>{pending?w('Confirm on your device…','Подтвердите на устройстве…'):w('Confirm with passkey','Подтвердить ключом доступа')}</button><a className="work-link-button" href={reauth}>{w('Sign in again','Войти заново')}</a>{failed&&<p className="notice error" role="alert">{w('Verification was not completed. Use an existing passkey or sign in again, then check the saved state.','Проверка не завершена. Используйте существующий ключ доступа или войдите заново, затем проверьте сохранённое состояние.')}</p>}</div>;
}
