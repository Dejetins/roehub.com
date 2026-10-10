import '@testing-library/jest-dom/vitest';
import {webcrypto} from 'node:crypto';
import {afterEach,expect,it,vi} from 'vitest';
import {cleanup,fireEvent,render,screen,waitFor} from '@testing-library/react';
import {QueryClientProvider} from '@tanstack/react-query';
import {I18nextProvider} from 'react-i18next';
import {ServiceControls,operationId} from './service-controls';
import {createQueryClient} from './query-client';
import {createI18n} from './i18n';
const org='08f28fa7-7c6b-4a73-9fd6-d39aab49c83e';
function mount(owner=true){
 const requests:{key:string,body:any}[]=[];
 vi.stubGlobal('crypto',webcrypto);
 vi.stubGlobal('fetch',vi.fn(async(url,options)=>{
  if(String(url).endsWith('/snapshot'))return Response.json({organization_id:org,installation_owner:owner,recent_auth:true,permissions:['operations.execute']});
  if(options.method==='POST'){
   requests.push({key:options.headers['Idempotency-Key'],body:JSON.parse(options.body)});
   if(requests.length===1)throw new TypeError('offline before delivery');
   return Response.json({operation_id:await operationId(org,requests[0].key),state:'succeeded'});
  }
  if(requests.length>1)return Response.json({operation_id:await operationId(org,requests[0].key),state:'succeeded'});
  return Response.json({error:{code:'admin.operation_not_found'}},{status:404});
 }));
 render(<I18nextProvider i18n={createI18n('en')}><QueryClientProvider client={createQueryClient()}><ServiceControls subject="owner" org={org} profile="trading" service="backtest-artifact-publisher" commands={['start','stop']} onChange={()=>{}}/></QueryClientProvider></I18nextProvider>);
 return requests;
}
afterEach(()=>{cleanup();vi.unstubAllGlobals();sessionStorage.clear();});
it('does not offer lifecycle commands to organization-only owner',async()=>{mount(false);await waitFor(()=>expect(fetch).toHaveBeenCalled());expect(screen.queryByRole('button',{name:'Stop service'})).not.toBeInTheDocument();});
it('recovers an undelivered command using the exact original key and body',async()=>{
 const requests=mount();fireEvent.click(await screen.findByRole('button',{name:'Stop service'}));
 const retry=await screen.findByRole('button',{name:'Retry the same request'});
 expect(screen.getByRole('button',{name:'Start service'})).toBeDisabled();
 fireEvent.click(retry);await waitFor(()=>expect(requests).toHaveLength(2));
 expect(requests[1]).toEqual(requests[0]);
 await screen.findByText(/Operation completed/);
});
it('fails closed on corrupted persisted recovery state',async()=>{
 sessionStorage.setItem(`roehub:service-operation:owner:${org}:backtest-artifact-publisher`,'invalid');
 mount();expect(await screen.findByRole('button',{name:'Start service'})).toBeDisabled();
 expect(screen.getByRole('alert')).toHaveTextContent('controls are locked');
});
