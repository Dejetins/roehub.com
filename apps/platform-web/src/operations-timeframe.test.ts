import {expect,it} from 'vitest';
import {rollupCandles} from './operations-timeframe';
it('rolls OHLC into UTC buckets without inventing gap candles',()=>{
 const c=(timestamp:string,open:number,high:number,low:number,close:number)=>({timestamp,open,high,low,close});
 expect(rollupCandles([c('2026-09-12T00:15:00Z',11,15,9,14),c('2026-09-12T00:00:00Z',10,12,8,11),c('2026-09-12T02:00:00Z',20,21,19,20)],3600000)).toEqual([c('2026-09-12T00:00:00.000Z',10,15,8,14),c('2026-09-12T02:00:00.000Z',20,21,19,20)]);
});
