import {expect,it} from 'vitest';
import {chartTimes,periodStart,periodZoom} from './chart-period';
it('uses UTC calendar months and clamps leap and month-end dates',()=>{
 expect(new Date(periodStart(Date.parse('2024-03-31T12:00:00Z'),'1m')).toISOString()).toBe('2024-02-29T12:00:00.000Z');
 expect(new Date(periodStart(Date.parse('2024-02-29T12:00:00Z'),'1y')).toISOString()).toBe('2023-02-28T12:00:00.000Z');
 expect(periodStart(Date.parse('2026-03-29T00:00:00Z'),'1w')).toBe(Date.parse('2026-03-22T00:00:00Z'));
});
it('never interprets trade indices or unordered coordinates as dates',()=>{
 expect(chartTimes([1,2,3])).toEqual([]);
 expect(chartTimes(['0','1'])).toEqual([]);
 expect(chartTimes(['2026-03-02','2026-03-01'])).toEqual([]);
});
it('retains the opening equity observation without resampling and resets value zoom for All',()=>{
 const times=chartTimes(['2026-03-01','2026-03-03','2026-03-05','2026-03-07']);
 expect(periodZoom(times,{start:Date.parse('2026-03-04'),end:Date.parse('2026-03-06')})).toMatchObject({startValue:1,endValue:2});
 expect(periodZoom(times,null)).toEqual({start:0,end:100,rangeMode:['percent','percent']});
});
