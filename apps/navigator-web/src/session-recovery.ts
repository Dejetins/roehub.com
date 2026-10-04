import type {QueryClient} from '@tanstack/react-query';
import {ApiError} from './api';

export function temporarySessionError(error:unknown):error is ApiError {
 return error instanceof ApiError&&(error.status===null||error.status===408||error.status===429||(error.status??0)>=500);
}
/** Retry reads only; authentication denial, malformed replies and changed identity stop. */
export function sessionReadInterval(error:unknown,failures=1):number|false {
 if(!error)return 30000;
 if(!temporarySessionError(error))return false;
 return Math.max((error.retryAfterSeconds??0)*1000,Math.min(30000,2000*2**Math.min(Math.max(failures-1,0),4)));
}

/** Restore only active transient reads after re-verifying identity. */
export function restoreTemporaryReads(client:QueryClient){
    const timers=new Set<ReturnType<typeof setTimeout>>();
    const restore=(key:readonly unknown[])=>{
      const query=client.getQueryCache().find({queryKey:key,exact:true});
      const error=query?.state.error;
      if(!query?.isActive()||!temporarySessionError(error))return;
      const delay=Math.max(0,query.state.errorUpdatedAt+(error.retryAfterSeconds??0)*1000-Date.now());
      if(delay>0){
        const timer=setTimeout(()=>{timers.delete(timer);restore(key);},delay);
        timers.add(timer);return;
      }
      void client.refetchQueries({queryKey:key,exact:true,type:'active'}, {cancelRefetch:false});
    };
    for(const query of client.getQueryCache().findAll()){
      if(query.queryKey[0]!=='session'&&temporarySessionError(query.state.error))restore(query.queryKey);
    }
    return()=>timers.forEach(clearTimeout);
}
