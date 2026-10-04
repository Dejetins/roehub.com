import {useChartInstance} from './chart-instance';
import {useLayoutEffect,useRef} from 'react';
import {use} from 'echarts/core';
import {LineChart} from 'echarts/charts';
import {GridComponent,TooltipComponent,DataZoomComponent,MarkPointComponent} from 'echarts/components';
import {CanvasRenderer} from 'echarts/renderers';
import {motionDuration} from './motion';
import type {Point} from './overview-model';
use([LineChart,GridComponent,TooltipComponent,DataZoomComponent,MarkPointComponent,CanvasRenderer]);
export function OverviewChart({points,view,label,locale,compare}:{points:Point[];view:string;label:string;locale:string;compare?:{name:string;values:number[]}[]}) {
 const host=useRef<HTMLDivElement>(null);
 const instance=useChartInstance(host);
 useLayoutEffect(()=>{
   const chart=instance.current;if(!chart)return;
   const value=(p:Point)=>view==='equity'?p.equity:view==='return'?(p.wealth-1)*100:p.drawdown;
   chart.setOption({animation:motionDuration()>0,animationDuration:0,animationDurationUpdate:Math.min(200,motionDuration()),useUTC:true,
    textStyle:{fontFamily:'-apple-system, BlinkMacSystemFont, Inter, sans-serif'},
    grid:{left:72,right:18,top:18,bottom:62},
    tooltip:{trigger:'axis',renderMode:'richText',confine:true,backgroundColor:'#171d23',borderColor:'#39424b',textStyle:{color:'#e6e9ee'},valueFormatter:(v:number)=>new Intl.NumberFormat(locale,{maximumFractionDigits:2}).format(v)+(view==='equity'?' USDT':'%')},
    xAxis:{type:'time',axisLabel:{color:'#b1b8c0',hideOverlap:true,formatter:(v:number)=>new Intl.DateTimeFormat(locale,{month:'short',day:'numeric',timeZone:'UTC'}).format(v)},axisLine:{lineStyle:{color:'#353d45'}},splitLine:{show:false}},
    yAxis:{type:'value',scale:true,axisLabel:{color:'#b1b8c0',formatter:(v:number)=>view==='equity'?`${(v/1000).toFixed(1)}k`:`${v.toFixed(1)}%`},splitLine:{lineStyle:{color:'#293139'}}},
    dataZoom:[{type:'inside',filterMode:'none'},{type:'slider',bottom:5,height:18,textStyle:{color:'#b1b8c0'},borderColor:'#353d45',fillerColor:'#9a78ff18',handleStyle:{color:'#9a78ff'}}],
    series:[{id:'portfolio',name:label,type:'line',showSymbol:false,data:points.map(p=>[p.time,value(p)]),lineStyle:{width:2,color:view==='drawdown'?'#ed958b':'#a58aff'},areaStyle:{color:view==='drawdown'?'#ed958b':'#a58aff',opacity:.09},
      markPoint:{symbol:'circle',symbolSize:8,itemStyle:{color:'#d4b36a'},label:{show:false},data:view==='equity'?points.filter(p=>p.flow).map(p=>({name:locale==='ru'?'Пополнение':'Deposit',coord:[p.time,p.equity],value:p.flow})):[]}},
      ...(view==='return'?(compare??[]).map((s,i)=>({id:s.name,name:s.name,type:'line',showSymbol:false,data:points.map((p,j)=>[p.time,s.values[j]]),lineStyle:{width:1.5,color:['#65d69b','#d4b36a','#82b6ee','#ed958b'][i],type:'dashed'}})):[])]
   },{replaceMerge:['series']});
 },[points,view,label,locale,compare]);
 return <div ref={host} className="overview-chart" role="img" aria-label={label}/>;
}
