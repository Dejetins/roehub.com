import {z} from 'zod';
const record=z.object({subject:z.string(),strategy:z.uuid(),key:z.uuid(),intent:z.uuid().optional(),command:z.enum(['run','stop','restart','delete','manual-entry','manual-exit']).optional(),amount:z.number().positive().optional(),price:z.number().positive().optional(),createdAt:z.number().optional(),runId:z.string().nullable().optional()});
const schema=z.array(record).max(100);
const storageKey='roehub.strategy-commands.pending.v1';
let records:z.infer<typeof schema>=[];
function read(){try{const raw=sessionStorage.getItem(storageKey);if(raw&&raw.length<30000)records=schema.parse(JSON.parse(raw));}catch{/* Retain validated in-memory lock. */}}
function write(){try{sessionStorage.setItem(storageKey,JSON.stringify(records));}catch{/* In-memory lock remains. */}}
export function pendingOperation(subject:string,strategy:string){read();return records.find(r=>r.subject===subject&&r.strategy===strategy);}
export function retainOperation(value:z.infer<typeof record>){read();records=records.filter(r=>r.subject===value.subject&&r.strategy!==value.strategy);records.push(record.parse(value));write();}
export function releaseOperation(subject:string,strategy:string){read();records=records.filter(r=>r.subject!==subject||r.strategy!==strategy);write();}
export function clearOperationRecovery(){records=[];try{sessionStorage.removeItem(storageKey);}catch{/* No private response is stored. */}}
