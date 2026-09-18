import {Eye,EyeOff} from 'lucide-react';
import {ChartDisplayMenu} from './chart-display-menu';
import {chartIntervals,rollupCandles} from './operations-timeframe';
import {buyAndHold} from './operations-benchmark';
import {useLayoutEffect,useRef,useState} from 'react';
import {useTranslation} from 'react-i18next';
import {init,use} from 'echarts/core';
import {CandlestickChart,LineChart,ScatterChart} from 'echarts/charts';
import {GridComponent,TooltipComponent,DataZoomComponent,MarkLineComponent} from 'echarts/components';
import {CanvasRenderer} from 'echarts/renderers';
import {candleIndex,candleZoom,executionMarker} from './price-chart-conventions';
import type {Operations} from './strategy-operations-api';
use([CandlestickChart,LineChart,ScatterChart,GridComponent,TooltipComponent,DataZoomComponent,MarkLineComponent,CanvasRenderer]);
type Candle={timestamp:string;open:number|null;high:number|null;low:number|null;close:number|null};
export function OperationsChart({candles,sourceTimeframe,data,view,selected,onSelect,reason,label}:{candles:Candle[];sourceTimeframe:string;data:Operations;view:string;selected:string|null;onSelect:(id:string)=>void;reason:(r:string)=>string;label:string}){
 const {t,i18n}=useTranslation();
 const [timeframe,setTimeframe]=useState(sourceTimeframe);
 const sourceInterval=chartIntervals[sourceTimeframe]??0;
 const selectedInterval=chartIntervals[timeframe]??sourceInterval;
 const available=Object.keys(chartIntervals).filter(tf=>sourceInterval>0&&chartIntervals[tf]!>=sourceInterval&&chartIntervals[tf]!%sourceInterval===0);
 const [benchmark,setBenchmark]=useState(true);
 const comparison=buyAndHold(data.equity,candles);
 const [markers,setMarkers]=useState(true);
 const [levels,setLevels]=useState({entry:true,stopLoss:true,takeProfit:true});
 const zoom=useRef<Record<string,{start:number;end:number}>>({});
 const ref=useRef<HTMLDivElement>(null),callback=useRef(onSelect);callback.current=onSelect;
 useLayoutEffect(()=>{
   if(!ref.current)return;
   const chart=init(ref.current),muted='#b1b8c0';
   const bars=(timeframe===sourceTimeframe?candles:rollupCandles(candles,selectedInterval)).filter(c=>[c.open,c.close,c.low,c.high].every(v=>v!=null));
   const times=bars.map(c=>Date.parse(c.timestamp));
   const date=(value:string|number)=>new Date(value).toISOString().slice(0,16).replace('T',' ');
   const number=(value:number)=>value.toLocaleString(i18n.language,{maximumFractionDigits:6});
   const points=view==='equity'?data.equity:data.drawdown;
   const fills=data.trades.flatMap(trade=>trade.fills.flatMap(fill=>{
     const x=candleIndex(times,fill.time);
     // No attachment to an unrelated last candle when the execution is outside coverage.
     const interval=selectedInterval;
     if(x<0||!interval||Date.parse(fill.time)>=times[times.length-1]!+interval)return [];
     return [{...fill,trade:trade.trade_id,value:[x,fill.price]}];
   }));
   const marks=(exit:boolean)=>fills.filter(f=>(f.action==='exit')===exit).map(f=>({
     value:f.value,trade:f.trade,fill:f,...executionMarker(exit),
     symbolSize:selected===f.trade?16:12,
     label:{show:selected===f.trade,formatter:reason(f.reason),position:exit?'top':'bottom',color:muted,textBorderWidth:0},
   }));
   const lines=[...(data.current_price==null?[]:[{yAxis:data.current_price,name:reason('mark')}]),
     ...(levels.entry?data.trades:[]).filter(trade=>trade.remaining_quantity>0).map(trade=>({yAxis:trade.entry,name:reason('entry')})),
     ...(!levels.stopLoss||data.stop_loss==null?[]:[{yAxis:data.stop_loss,name:'SL'}]),...(!levels.takeProfit||data.take_profit==null?[]:[{yAxis:data.take_profit,name:'TP'}])];
   chart.setOption({animation:false,backgroundColor:'transparent',grid:{left:72,right:18,top:18,bottom:62},
     tooltip:{trigger:'axis',renderMode:'richText',confine:true,backgroundColor:'#171d23',borderColor:'#39424b',textStyle:{color:'#e6e9ee'},axisPointer:{type:'cross'},formatter:(raw:any)=>{
       const rows=Array.isArray(raw)?raw:[raw];
       if(view!=='price'){const p=rows[0];return p?`${date(new Date(p.value[0]).toISOString())} UTC`+rows.map((row:any)=>`\n${row.seriesName}: ${number(row.value[1])}`).join(''):'';}
       const index=rows.find((r:any)=>r.seriesType==='candlestick')?.dataIndex??rows[0]?.data?.value?.[0];
       const bar=bars[index];
       if(!bar)return '';
       return `${date(bar.timestamp)} UTC\nO ${number(bar.open!)} · H ${number(bar.high!)}\nL ${number(bar.low!)} · C ${number(bar.close!)}`+
         (markers?fills.filter(f=>f.value[0]===index):[]).map(f=>`\n${t(f.action==='exit'?'results.exit':'results.entry')} · ${reason(f.reason)} · ${number(f.quantity)} @ ${number(f.price)}\n${date(f.time)} UTC`).join('');
     }},
     xAxis:view==='price'?{type:'category',data:bars.map(c=>c.timestamp),axisLabel:{formatter:(value:string|number)=>date(value).slice(0,10),hideOverlap:true,color:muted},axisPointer:{label:{formatter:(p:any)=>date(p.value)}}}:{type:'time',axisLabel:{formatter:(value:string|number)=>date(value).slice(0,10),color:muted,hideOverlap:true}},
     yAxis:{type:'value',scale:true,axisLabel:{color:muted},splitLine:{lineStyle:{color:'#293139'}}},
     dataZoom:[view==='price'?candleZoom(bars.length,zoom.current[view==='price'?timeframe:view]):{type:'inside',filterMode:'filter',zoomOnMouseWheel:'ctrl',...zoom.current[view==='price'?timeframe:view]},
       {type:'slider',bottom:4,height:18,...zoom.current[view==='price'?timeframe:view],...(view==='price'?{labelFormatter:(_:number,value:string)=>date(value)}:{}),textStyle:{color:muted}}],
     series:view==='price'?[
       {name:t('results.price'),type:'candlestick',barMinWidth:2,data:bars.map(c=>[c.open,c.close,c.low,c.high]),itemStyle:{color:'#65d69b',color0:'#ed958b',borderColor:'#65d69b',borderColor0:'#ed958b'},
         markLine:{symbol:'none',silent:true,label:{position:'insideEndTop',formatter:'{b}',color:muted,textBorderWidth:0,backgroundColor:'#171d23',padding:[2,4],fontSize:10},data:lines,lineStyle:{type:'dashed',color:'#a58aff'}}},
       ...(markers?[{name:t('results.entry'),type:'scatter',...executionMarker(false),data:marks(false),z:8},
         {name:t('results.exit'),type:'scatter',...executionMarker(true),data:marks(true),z:8}]:[]),
     ]:[{name:label,type:'line',showSymbol:false,data:points.map(p=>[Date.parse(p.timestamp),p.value]),lineStyle:{color:view==='drawdown'?'#ed958b':'#a58aff',width:2},areaStyle:{opacity:.08,color:'#a58aff'}},...(view==='equity'&&benchmark&&comparison.length?[{name:'Buy & Hold',type:'line',showSymbol:false,data:comparison.map(p=>[Date.parse(p.timestamp),p.value]),lineStyle:{color:'#d4b36a',width:2,type:'dashed'}}]:[])],useUTC:true});
   chart.on('datazoom',()=>{const current=(chart.getOption().dataZoom as {start:number;end:number}[])[0];if(current)zoom.current[view==='price'?timeframe:view]={start:current.start,end:current.end};});
   chart.on('click',(p:any)=>{if(p.data?.trade)callback.current(p.data.trade);});
   const observer=new ResizeObserver(()=>chart.resize());observer.observe(ref.current);
   return()=>{observer.disconnect();chart.dispose();};
 },[candles,data,view,selected,reason,label,markers,levels,t,i18n.language,benchmark,timeframe,sourceTimeframe,selectedInterval]);
 return <>{view==='drawdown'&&<div className="chart-toolbar operations-mini-toolbar operations-toolbar-spacer" aria-hidden="true"/>}{view==='equity'&&<div className="chart-toolbar operations-mini-toolbar"><button type="button" className="benchmark-toggle" aria-pressed={benchmark} title={i18n.language.startsWith('ru')?(benchmark?'Скрыть Buy & Hold':'Показать Buy & Hold'):(benchmark?'Hide Buy & Hold':'Show Buy & Hold')} onClick={()=>setBenchmark(value=>!value)}>{benchmark?<Eye size={14} aria-hidden="true"/>:<EyeOff size={14} aria-hidden="true"/>}Buy &amp; Hold</button></div>}{view==='price'&&<div className="chart-toolbar operations-mini-toolbar"><ChartDisplayMenu label={i18n.language.startsWith('ru')?'Отображение графика':'Chart display'} options={[
 {label:i18n.language.startsWith('ru')?'Сделки':'Trades',checked:markers,onChange:setMarkers},
 ...(['entry','stopLoss','takeProfit'] as const).map(key=>({label:reason(key),checked:levels[key],onChange:(checked:boolean)=>setLevels(current=>({...current,[key]:checked}))})),
 ]}/><ChartDisplayMenu label={t('results.chartTimeframe')} value={timeframe} options={[...new Set([sourceTimeframe,...available])].map(tf=>({label:tf,checked:tf===timeframe,onChange:()=>setTimeframe(tf)}))}/></div>}
   <div ref={ref} className="operations-chart" role="img" aria-label={label}/>
   </>;
}
