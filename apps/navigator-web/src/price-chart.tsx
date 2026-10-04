import {chartTimes,periodZoom,usePeriodZoom,type PeriodView} from './chart-period';
import {useChartInstance} from './chart-instance';
import {candleIndex,candleZoom,executionMarker} from './price-chart-conventions';
import {useLayoutEffect,useMemo,useRef,useState} from 'react';
import {useTranslation} from 'react-i18next';
import {z} from 'zod';
import {use} from 'echarts/core';
import {CandlestickChart,ScatterChart} from 'echarts/charts';
import {GridComponent,TooltipComponent,DataZoomComponent} from 'echarts/components';
import {CanvasRenderer} from 'echarts/renderers';
import {ApiError,requestJson} from './api';
import {chartDate} from './results';
import {useResultRead} from './result-read';
import * as api from './results-api';
use([CandlestickChart,ScatterChart,GridComponent,TooltipComponent,DataZoomComponent,CanvasRenderer]);
const candle=z.object({time:z.iso.datetime(),open:z.number().finite(),high:z.number().finite(),low:z.number().finite(),close:z.number().finite()});
const candlesSchema=z.object({job_id:z.uuid(),variant_key:api.variantKeySchema,timeframe:z.string(),source_bars:z.number().int().nonnegative(),group_size:z.number().int().positive(),candles:z.array(candle).max(60000)});
type Trade=z.infer<typeof api.tradeSchema>;
export {candleIndex} from './price-chart-conventions';
export function usePriceRead(job:string,variant:string,subject:string,now:number,timeframe:string,enabled:boolean) {
 const prices=useResultRead(subject,[job,variant,'candles',timeframe],async signal=>{const r=await requestJson(`${api.variantPath(job,variant)}/candles?timeframe=${timeframe}&max_bars=60000`,candlesSchema,{signal});api.assertSource(r.data,job,variant);if(r.data.timeframe!==timeframe)throw new ApiError('invalid-response',200,'failed');return r;},now,enabled);
 const trades=useResultRead(subject,[job,variant,'price-trades'],async signal=>{
   const items:Trade[]=[];let total=0;
   for(let page=1;page<=100;page++){
     const r=await api.readResult(job,variant,`trades?page=${page}&page_size=100`,api.tradesSchema,signal);
     if('materialization' in r.data)return {data:{...r.data,items:[] as Trade[],total:0,pending:true},status:202,retryAfterSeconds:r.retryAfterSeconds};
     if(r.data.pagination.page!==page)throw new ApiError('invalid-response',200,'failed');
     items.push(...r.data.items);total=r.data.pagination.total;
     if(!r.data.pagination.has_next)break;
   }
   return {data:{items,total,pending:false},status:200,retryAfterSeconds:0};
 },now,enabled);
 return {prices,trades};
}
export function PriceChart({data,entries,timeframe,periodView,markers=true}:{markers?:boolean;periodView?:PeriodView;data:z.infer<typeof candlesSchema>;entries:Trade[];timeframe:string}) {
 const {t,i18n}=useTranslation();const ref=useRef<HTMLDivElement>(null);const zoomTimeframe=useRef(timeframe);const zoom=useRef<{start:number;end:number}|null>(null);const instance=useChartInstance(ref);
 const periodTimes=useMemo(()=>chartTimes(data.candles.map(c=>c.time)),[data]);
 usePeriodZoom(instance,periodTimes,periodView);
 useLayoutEffect(()=>{
   if(!ref.current)return;
   const chart=instance.current;if(!chart)return;
   const saved=chart.getOption?.()?.dataZoom as {start:number;end:number}[]|undefined;
   if(zoomTimeframe.current!==timeframe){zoom.current=null;zoomTimeframe.current=timeframe;}else if(saved?.[0])zoom.current={start:saved[0].start,end:saved[0].end};
   const times=data.candles.map(c=>Date.parse(c.time));
   const n=(v:number)=>v.toLocaleString(i18n.language,{maximumFractionDigits:2});
   const date=(v:string)=>chartDate(v,i18n.language);
   const points=(exit:boolean)=>(entries??[]).flatMap(trade=>{
     const x=candleIndex(times,exit?trade.exit_timestamp:trade.entry_timestamp),price=exit?trade.exit_price:trade.entry_price;
     return x<0||price==null?[]:[{value:[x,price],trade,symbolRotate:exit?180:0}];
   });
   chart.setOption({animation:false,grid:{left:72,right:18,top:18,bottom:62},
     tooltip:{trigger:'axis',renderMode:'richText',confine:true,backgroundColor:'#171d23',borderColor:'#39424b',textStyle:{color:'#e6e9ee'},axisPointer:{type:'cross'},formatter:(raw:any)=>{
       const rows=Array.isArray(raw)?raw:[raw],bar=data.candles[rows.find((r:any)=>r.seriesType==='candlestick')?.dataIndex??rows[0]?.data?.value?.[0]];
       if(!bar)return '';return `${date(bar.time)}\nO ${n(bar.open)} · H ${n(bar.high)}\nL ${n(bar.low)} · C ${n(bar.close)}`+rows.filter((r:any)=>r.data?.trade).map((r:any)=>`\n${r.seriesName} #${r.data.trade.trade_index} · ${r.data.trade.side} · ${n(r.data.value[1])}`).join('');
     }},
     xAxis:{type:'category',data:data.candles.map(c=>c.time),axisLabel:{formatter:date,hideOverlap:true,color:'#b1b8c0'},axisPointer:{label:{formatter:(p:any)=>date(p.value)}}},
     yAxis:{scale:true,axisLabel:{color:'#b1b8c0'},splitLine:{lineStyle:{color:'#293139'}}},
     dataZoom:[{...candleZoom(data.candles.length,zoom.current),...(periodView?{start:undefined,end:undefined,...periodZoom(periodTimes,periodView.range)}:{})},{type:'slider',bottom:4,height:18,labelFormatter:(_:number,v:string)=>date(v),textStyle:{color:'#b1b8c0'}}],
     series:[{name:t('results.price'),type:'candlestick',barMinWidth:2,data:data.candles.map(c=>[c.open,c.close,c.low,c.high]),itemStyle:{color:'#65d69b',color0:'#ed958b',borderColor:'#65d69b',borderColor0:'#ed958b'}},
       ...(markers?[{name:t('results.entry'),type:'scatter',...executionMarker(false),data:points(false)},{name:t('results.exit'),type:'scatter',...executionMarker(true),data:points(true)}]:[])]},{replaceMerge:['series']});

 },[data,entries,markers,i18n.language,t,timeframe,periodView]);
 return <div className="result-detail-view price-view" data-ready={!!data} data-motion-content>
 {data&&!data.candles.length&&<p>{t('results.empty')}</p>}
 <div ref={ref} className="result-chart price-chart" role="img" aria-label={t('results.price')}/><p className="chart-help muted">{t('results.priceLegend')}</p>
 </div>;
}
