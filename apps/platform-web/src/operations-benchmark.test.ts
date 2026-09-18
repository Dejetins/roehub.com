import {expect,it} from 'vitest';
import {buyAndHold} from './operations-benchmark';
it('compares the same starting capital and limits the benchmark to the equity period',()=>{
 expect(buyAndHold([{timestamp:'2026-01-01T00:00:00Z',value:1000},{timestamp:'2026-01-03T00:00:00Z',value:1050}],[{timestamp:'2026-01-01T00:00:00Z',close:100},{timestamp:'2026-01-02T00:00:00Z',close:110},{timestamp:'2026-01-04T00:00:00Z',close:200}])).toEqual([{timestamp:'2026-01-01T00:00:00Z',value:1000},{timestamp:'2026-01-02T00:00:00Z',value:1100}]);
});
it('does not invent a starting price',()=>{
 expect(buyAndHold([{timestamp:'2026-01-01T00:00:00Z',value:1000}],[{timestamp:'2026-01-02T00:00:00Z',close:100}])).toEqual([]);
});
