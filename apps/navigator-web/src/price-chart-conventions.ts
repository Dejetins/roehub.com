/** Shared Backtests/Strategies candle navigation and execution markers. */
export function candleIndex(times:number[],timestamp:string) {
 const time=Date.parse(timestamp);if(!Number.isFinite(time)||!times.length||time<times[0]!)return -1;
 let lo=0,hi=times.length;while(lo<hi){const mid=(lo+hi)>>>1;if(times[mid]!<=time)lo=mid+1;else hi=mid;}return lo-1;
}
export function candleZoom(count:number, saved?:{start:number;end:number}|null) {
 return {type:'inside' as const,filterMode:'filter' as const,zoomOnMouseWheel:'ctrl' as const,
   start:saved?.start??Math.max(0,(1-150/Math.max(1,count))*100),end:saved?.end??100};
}
export function executionMarker(exit:boolean) {
 return {symbol:'triangle' as const,symbolSize:12,symbolRotate:exit?180:0,
   itemStyle:{color:exit?'#f2cc74':'#80baff'}};
}
