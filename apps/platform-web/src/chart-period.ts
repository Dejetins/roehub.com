import {useLayoutEffect, type RefObject} from 'react';
import type {EChartsType} from 'echarts/core';

export const chartPeriods=['1d','1w','1m','3m','1y','all'] as const;
export type ChartPeriod=typeof chartPeriods[number];
export type TimeWindow={start:number;end:number};
export type PeriodView={range:TimeWindow|null;onZoom:(range:TimeWindow)=>void};
const day=86_400_000;
export function periodStart(end:number,period:ChartPeriod):number {
 if(period==='all')return -Infinity;
 if(period==='1d'||period==='1w')return end-(period==='1d'?1:7)*day;
 const date=new Date(end),dayOfMonth=date.getUTCDate();
 date.setUTCDate(1);
 date.setUTCMonth(date.getUTCMonth()-(period==='1m'?1:period==='3m'?3:12));
 const lastDay=new Date(Date.UTC(date.getUTCFullYear(),date.getUTCMonth()+1,0)).getUTCDate();
 date.setUTCDate(Math.min(dayOfMonth,lastDay));
 return date.getTime();
}
// Numeric report coordinates are trade indices, not Unix timestamps.
export function chartTimes(values:(string|number)[]):number[] {
 const times=values.map(value=>typeof value==='string'&&/^\d{4}-\d{2}-\d{2}/.test(value)?Date.parse(value):NaN);
 return times.length&&times.every((time,i)=>Number.isFinite(time)&&(i===0||time>=times[i-1]!))?times:[];
}
export function periodZoom(times:number[],range:TimeWindow|null) {
 if(!range||!times.length)return {start:0,end:100,rangeMode:['percent','percent']};
 // Include the preceding observation: realized equity is unchanged until the next exit.
 let first=0,last=times.length-1;
 while(first+1<times.length&&times[first+1]!<=range.start)first++;
 while(last>first&&times[last]!>range.end)last--;
 return {startValue:first,endValue:last,rangeMode:['value','value']};
}
export function usePeriodZoom(instance:RefObject<EChartsType|null>,times:number[],view?:PeriodView) {
 useLayoutEffect(()=>{
  const chart=instance.current;if(!chart||!view||!times.length)return;
  const onZoom=()=>{
   const option=chart.getOption()?.dataZoom as {startValue?:number;endValue?:number;start?:number;end?:number}[]|undefined;
   const zoom=option?.[0];if(!zoom)return;
   const index=(value:number|undefined,percent:number)=>Math.max(0,Math.min(times.length-1,Math.round(value??percent/100*(times.length-1))));
   view.onZoom({start:times[index(zoom.startValue,zoom.start??0)]!,end:times[index(zoom.endValue,zoom.end??100)]!});
  };
  chart.on?.('datazoom',onZoom);
  return ()=>{chart.off?.('datazoom',onZoom);};
 },[instance,times,view]);
}
