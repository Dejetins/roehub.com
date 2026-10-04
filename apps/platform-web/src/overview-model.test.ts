import {describe,it,expect} from 'vitest';
import {AS_OF,START,DAY,builtins,portfolio,readBaskets} from './overview-model';
const now=Date.parse(AS_OF);
describe('Overview deterministic fixture invariants',()=>{
 it('reconciles equity, P&L and linked TWR contributions for every builtin and a cross-exchange basket',()=>{
  for(const b of [...builtins,{id:'custom-test',members:['atlas','ether','atlas']}])for(const mode of ['live','paper'] as const)for(const start of [START,now-25*DAY,now-DAY]){
   const p=portfolio(b.members,b.id,mode,start,now);
   expect(p.rows.reduce((n,s)=>n+s.current,0)+p.cash).toBeCloseTo(p.current.equity,7);
   expect(p.rows.reduce((n,s)=>n+s.pnl,0)).toBeCloseTo(p.pnl,7);
   expect(p.rows.reduce((n,s)=>n+s.contribution,0)).toBeCloseTo(p.returns,7);
   expect(p.rows.length).toBe(new Set(b.members).size);
  }
 });
 it('keeps exchange scopes additive, without duplicating total cash',()=>{
  const total=portfolio(builtins[0]!.members,'total','live',START,now);
  const exchanges=builtins.slice(1,3).map(b=>portfolio(b.members,b.id,'live',START,now));
  expect(exchanges.reduce((n,p)=>n+p.current.equity,0)).toBeCloseTo(total.current.equity,8);
 });
 it('excludes external deposits from time-weighted performance',()=>{
  const p=portfolio(builtins[0]!.members,'total','live',START,now);
  const before=p.all[119]!,deposit=p.all[120]!;
  expect(deposit.flow).toBe(5000);
  expect(deposit.wealth/before.wealth-1).toBeCloseTo((deposit.equity-before.equity-5000)/before.equity,12);
  expect(p.points.at(-1)!.wealth).toBeCloseTo(1+p.returns/100,12);
 });
 it('keeps current capital and exposure independent of historical period',()=>{
  const a=portfolio(['atlas','delta'],'custom-a','live',START,now);
  const b=portfolio(['atlas','delta'],'custom-a','live',now-7*DAY,now-DAY);
  expect(a.current).toEqual(b.current);expect(a.gross).toBe(b.gross);expect(a.pnl).not.toBe(b.pnl);
  expect(a.cash).toBe(0);expect(a.gross).toBeGreaterThan(a.net);
 });
 it('keeps separate paper and live histories',()=>{
  const live=portfolio(['atlas'],'BTCUSDT','live',START,now);
  const paper=portfolio(['atlas'],'BTCUSDT','paper',START,now);
  expect(paper.current.equity).toBeLessThan(live.current.equity);
  expect(paper.returns).not.toBe(live.returns);
 });
 it('chains the monthly return to selected-period TWR',()=>{
  for(const start of [START,now-45*DAY,now-25*DAY]){
   const p=portfolio(builtins[0]!.members,'total','live',start,now);
   expect(p.monthly.reduce<number>((wealth,r)=>wealth*(1+(r??0)/100),1)).toBeCloseTo(1+p.returns/100,10);
  }
 });
 it('rejects invalid saved membership and deduplicates valid baskets',()=>{
  localStorage.setItem('test-overview',JSON.stringify({selected:'missing',baskets:[{id:'custom-a',name:'Test',members:['atlas','atlas']},{id:'custom-b',name:'Bad',members:['unknown']}]}));
  expect(readBaskets('test-overview')).toEqual({selected:'missing',baskets:[{id:'custom-a',name:'Test',members:['atlas']}]});
  localStorage.removeItem('test-overview');
 });
});
