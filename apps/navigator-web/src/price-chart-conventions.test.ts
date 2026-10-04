import {describe,it,expect} from 'vitest';
import {candleIndex,candleZoom,executionMarker} from './price-chart-conventions';
describe('shared candle interaction conventions',()=>{
 it('places an intra-candle execution on its containing bar',()=>{
  const times=[0,900000,1800000];
  expect(candleIndex(times,new Date(950000).toISOString())).toBe(1);
  expect(candleIndex(times,new Date(-1).toISOString())).toBe(-1);
 });
 it('opens the last 150 bars and preserves a user range',()=>{
  expect(candleZoom(300)).toMatchObject({start:50,end:100,filterMode:'filter',zoomOnMouseWheel:'ctrl'});
  expect(candleZoom(30)).toMatchObject({start:0,end:100});
  expect(candleZoom(300,{start:65,end:90})).toMatchObject({start:65,end:90});
 });
 it('distinguishes execution action without changing its meaning by direction',()=>{
  expect(executionMarker(false)).toMatchObject({symbol:'triangle',symbolRotate:0,itemStyle:{color:'#80baff'}});
  expect(executionMarker(true)).toMatchObject({symbol:'triangle',symbolRotate:180,itemStyle:{color:'#f2cc74'}});
 });
});
