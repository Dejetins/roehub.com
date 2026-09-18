import {afterEach,it,expect,vi} from 'vitest';
import {readResearchSource,readResearchVariant,sourceSchema,profileFields} from './strategy-insights-api';
const id='00000000-0000-4000-8000-000000000001',job='00000000-0000-4000-8000-000000000002';
const reply=(data:unknown)=>new Response(JSON.stringify(data),{headers:{'Content-Type':'application/json'}});
afterEach(()=>vi.unstubAllGlobals());
it('requires a complete persisted source pair and strips unrelated metadata',()=>{
 expect(sourceSchema.safeParse({strategy_id:id,source_job_id:job,source_variant_key:null}).success).toBe(false);
 expect(sourceSchema.parse({strategy_id:id,source_job_id:job,source_variant_key:'v',private_payload:'no'})).not.toHaveProperty('private_payload');
});
it('rejects a research origin bound to a different strategy',async()=>{
 vi.stubGlobal('fetch',vi.fn().mockResolvedValue(reply({strategy_id:job,source_job_id:job,source_variant_key:'v'})));
 await expect(readResearchSource(id,new AbortController().signal)).rejects.toMatchObject({kind:'invalid-response'});
});
it('reads the existing variant DTO without demanding a nonexistent job field and rejects wrong variant',async()=>{
 const data={variant_key:'v',variant_hash:'hash',rank:1,summary_metrics:{trade_count:5},canonical_variant_params:{execution:{direction_mode:'long_only',fee_rate:0.001}}};
 vi.stubGlobal('fetch',vi.fn().mockResolvedValueOnce(reply(data)).mockResolvedValueOnce(reply({...data,variant_key:'wrong'})));
 await expect(readResearchVariant(job,'v',new AbortController().signal)).resolves.toMatchObject({data:{variant_key:'v',summary_metrics:{trade_count:5}}});
 await expect(readResearchVariant(job,'v',new AbortController().signal)).rejects.toMatchObject({kind:'invalid-response'});
});
it('accepts finite Decimal wire values including exponent notation but rejects nonnumeric and infinite values',()=>{
 expect(profileFields.sizing_value.parse('0E-8')).toBe(0);
 expect(profileFields.sizing_value.parse('123.45')).toBe(123.45);
 expect(profileFields.sizing_value.safeParse('NaN').success).toBe(false);
 expect(profileFields.sizing_value.safeParse('1e999').success).toBe(false);
});
