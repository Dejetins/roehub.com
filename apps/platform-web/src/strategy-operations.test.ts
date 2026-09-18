import {afterEach,expect,test,vi} from 'vitest';
import {sendStrategyCommand} from './strategy-operations-api';
import {pendingOperation,retainOperation,releaseOperation,clearOperationRecovery} from './operation-recovery';
const id='54f2d6f4-7cb7-4d7b-9469-437b2bb77362',key='8cda6c70-5012-4e3f-99cc-c771ba5ac74e';
afterEach(()=>{vi.unstubAllGlobals();clearOperationRecovery();});
test('DELETE accepts the actual 204 empty response',async()=>{const fetch=vi.fn().mockResolvedValue(new Response(null,{status:204}));vi.stubGlobal('fetch',fetch);expect((await sendStrategyCommand(id,'delete',key)).status).toBe('accepted');expect(fetch).toHaveBeenCalledTimes(1);});
test('pending identity survives reload storage and is scoped to subject and strategy',()=>{retainOperation({subject:'alice',strategy:id,key});expect(pendingOperation('alice',id)?.key).toBe(key);expect(pendingOperation('bob',id)).toBeUndefined();releaseOperation('bob',id);expect(pendingOperation('alice',id)?.key).toBe(key);releaseOperation('alice',id);expect(pendingOperation('alice',id)).toBeUndefined();});
test('accepted manual intent is not mistaken for a confirmed fill',async()=>{vi.stubGlobal('fetch',vi.fn().mockResolvedValue(new Response(JSON.stringify({status:'accepted',intent_id:key,outcome_reason:'pending_dispatch',duplicate:false,paper_order_state:null}),{status:200})));const reply=await sendStrategyCommand(id,'manual-entry',key,100);expect('terminal' in reply&&reply.terminal).toBe(false);});
