import {useEffect,useRef,useState} from 'react';
import {useTranslation} from 'react-i18next';
import {z} from 'zod';
import {init,use} from 'echarts/core';
import {CandlestickChart,ScatterChart} from 'echarts/charts';
import {GridComponent,TooltipComponent,DataZoomComponent} from 'echarts/components';
import {CanvasRenderer} from 'echarts/renderers';
import {ApiError,requestJson} from './api';
import {ReadError,isRestricted} from './library';
import {chartDate,useResultRead} from './results';
import * as api from './results-api';
use([CandlestickChart,ScatterChart,GridComponent,TooltipComponent,DataZoomComponent,CanvasRenderer]);
const candle=z.object({time:z.iso.datetime(),open:z.number().finite(),high:z.number().finite(),low:z.number().finite(),close:z.number().finite()});
const candlesSchema=z.object({job_id:z.uuid(),variant_key:api.variantKeySchema,timeframe:z.string(),source_bars:z.number().int().nonnegative(),group_size:z.number().int().positive(),candles:z.array(candle).max(60000)});
type Trade=z.infer<typeof api.tradeSchema>;
export function candleIndex(times:number[],timestamp:string) {
 const time=Date.parse(timestamp);if(!Number.isFinite(time)||!times.length||time<times[0]!)return -1;
 let lo=0,hi=times.length;while(lo<hi){const mid=(lo+hi)>>>1;if(times[mid]!<=time)lo=mid+1;else hi=mid;}return lo-1;
}
export function PriceView({job,variant,subject,now}:{job:string;variant:string;subject:string;now:number}) {
 const {t,i18n}=useTranslation();const ref=useRef<HTMLDivElement>(null);const [markers,setMarkers]=useState(true);const [timeframe,setTimeframe]=useState('15m');const zoomTimeframe=useRef('15m');const zoom=useRef<{start:number;end:number}|null>(null);
 const prices=useResultRead(subject,[job,variant,'candles',timeframe],async signal=>{const r=await requestJson(`${api.variantPath(job,variant)}/candles?timeframe=${timeframe}&max_bars=60000`,candlesSchema,{signal});api.assertSource(r.data,job,variant);if(r.data.timeframe!==timeframe)throw new ApiError('invalid-response',200,'failed');return r;},now);
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
 },now);
 const tradeReply=trades.data?.data;
 const failed=tradeReply?.pending&&'status' in tradeReply&&!['queued','running','pending'].includes(String(tradeReply.status));
 const data=prices.data?.data,entries=isRestricted(trades.error)?undefined:trades.data?.data.items;
 useEffect(()=>{
   if(!ref.current||!data?.candles.length||isRestricted(prices.error))return;
   const element=ref.current,chart=init(element),times=data.candles.map(c=>Date.parse(c.time));
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
     dataZoom:[{type:'inside',filterMode:'filter',zoomOnMouseWheel:'ctrl',start:zoom.current?.start??Math.max(0,(1-150/data.candles.length)*100),end:zoom.current?.end??100},{type:'slider',bottom:4,height:18,labelFormatter:(_:number,v:string)=>date(v),textStyle:{color:'#b1b8c0'}}],
     series:[{name:t('results.price'),type:'candlestick',barMinWidth:2,data:data.candles.map(c=>[c.open,c.close,c.low,c.high]),itemStyle:{color:'#65d69b',color0:'#ed958b',borderColor:'#65d69b',borderColor0:'#ed958b'}},
       ...(markers?[{name:t('results.entry'),type:'scatter',symbol:'triangle',symbolSize:12,itemStyle:{color:'#80baff'},data:points(false)},{name:t('results.exit'),type:'scatter',symbol:'triangle',symbolSize:12,itemStyle:{color:'#f2cc74'},data:points(true)}]:[])]});
   const resize=new ResizeObserver(()=>chart.resize());resize.observe(element);return()=>{resize.disconnect();const saved=chart.getOption?.().dataZoom as {start:number;end:number}[]|undefined;if(saved?.[0]&&zoomTimeframe.current===timeframe)zoom.current={start:saved[0].start,end:saved[0].end};chart.dispose();};
 },[data,entries,markers,i18n.language,t,prices.error,timeframe]);
 return <div className="result-detail-view price-view" data-motion-content><div className="chart-toolbar"><p className="chart-help muted">{data&&`${data.timeframe} · ${data.candles.length} ${t('results.candles')}`}{data&&data.group_size>1&&` · ${t('results.candleGrouping',{count:data.group_size})}`}</p><label className="chart-timeframe">{t('results.chartTimeframe')}<select value={timeframe} onChange={e=>{zoomTimeframe.current=e.target.value;zoom.current=null;setTimeframe(e.target.value);}}>{['1m','5m','15m','30m','1h','4h','1d'].map(tf=><option key={tf}>{tf}</option>)}</select></label><label className="trade-toggle"><input type="checkbox" checked={markers} onChange={e=>setMarkers(e.target.checked)}/>{t('results.priceTrades')}</label></div>
 <ReadError error={prices.error}/><ReadError error={trades.error}/>{prices.isPending&&<p role="status">{t('results.loading')}</p>}{(trades.isPending||(trades.data?.data.pending&&!failed))&&<p role="status">{t('results.loadingTrades')}</p>}{failed&&<p role="status" className="notice">{t('results.failed')}</p>}{trades.data&&trades.data.data.total>(entries?.length??0)&&<p className="notice">{t('results.tradeLimit',{count:entries?.length??0,total:trades.data.data.total})}</p>}
 {data&&!data.candles.length&&<p>{t('results.empty')}</p>}
 <div ref={ref} className="result-chart price-chart" role="img" aria-label={t('results.price')}/><p className="chart-help muted">{t('results.priceLegend')}</p>
 <button disabled={!prices.canRefresh||!trades.canRefresh} onClick={()=>{void prices.refetch();void trades.refetch();}}>{t('results.refresh')}</button></div>;
}
