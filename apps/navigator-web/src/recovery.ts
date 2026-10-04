import { z } from 'zod';
import { researchSchema, type Research } from './builder-api';
export const RECOVERY_KEY='roehub.backtests.create-recovery.v1';
const maxBytes=256_000;
export const recoverySchema=z.object({ operation: z.literal('create-backtest'), body: researchSchema, key: z.uuid(), createdAt: z.number().int().positive().max(8_640_000_000_000_000),
  subject: z.string().min(1), organization: z.uuid().nullable(), resultId: z.uuid().nullable() });
export type Recovery=z.infer<typeof recoverySchema>;
let memory: Recovery|null=null;
let storageAvailable=true;
export function storageWorks() {
  try { const probe=`${RECOVERY_KEY}.probe`; sessionStorage.setItem(probe,'1'); sessionStorage.removeItem(probe); storageAvailable=true; }
  catch { storageAvailable=false; }
  return storageAvailable;
}
export function clearRecovery() {
  memory=null;
  try { sessionStorage.removeItem(RECOVERY_KEY); } catch { storageAvailable=false; }
}
export function loadRecovery(subject: string): Recovery|null {
  try {
    const raw=sessionStorage.getItem(RECOVERY_KEY);
    if (raw) {
      if (raw.length>maxBytes) { clearRecovery(); return null; }
      const parsed=recoverySchema.safeParse(JSON.parse(raw));
      if (!parsed.success) { clearRecovery(); return null; }
      memory=parsed.data;
    }
  } catch { storageAvailable=false; }
  if (memory && memory.subject!==subject) clearRecovery();
  return memory;
}
export function saveRecovery(record: Recovery) {
  const clean=recoverySchema.parse(record); const serialized=JSON.stringify(clean);
  if (serialized.length>maxBytes) throw new Error('recoveryTooLarge');
  memory=clean;
  try { sessionStorage.setItem(RECOVERY_KEY,serialized); }
  catch { storageAvailable=false; }
  return storageAvailable;
}
export function freezeAttempt(body: Research, subject: string): Recovery {
  return recoverySchema.parse({operation:'create-backtest',body,key:crypto.randomUUID(),createdAt:Date.now(),subject,organization:null,resultId:null});
}
/** A stored org is never a server precondition. No caller can enable replay in this API version. */
export function recoveryReason(record: Recovery, subject: string|null, organization: string|null, retentionSeconds: number|null, now=Date.now()) {
  if (!subject || subject!==record.subject) return 'identityUnknown';
  if (!organization || !record.organization) return 'scopeUnknown';
  if (organization!==record.organization) return 'scopeChanged';
  if (retentionSeconds===null || !Number.isFinite(retentionSeconds) || retentionSeconds<=0) return 'retentionUnknown';
  if (now<record.createdAt || now-record.createdAt>=retentionSeconds*1000) return 'retentionExpired';
  return 'serverScopeUnbound';
}
