/** Deterministic visual fixtures. Not a financial service or real account valuation. */
export const AS_OF = '2026-09-26T12:00:00Z';
export const START = Date.parse('2026-01-01T12:00:00Z');
export const DAY = 86_400_000;
export const strategies = [
  {id:'atlas',name:'Atlas Momentum',exchange:'Binance',instrument:'BTCUSDT',account:'Main · USDT-M',capital:42000,rate:.0011,phase:1,side:'Long',exposure:38000},
  {id:'ether',name:'Ether Swing',exchange:'Bybit',instrument:'ETHUSDT',account:'Trading · Linear',capital:28000,rate:.0007,phase:4,side:'Long',exposure:23500},
  {id:'delta',name:'Delta Neutral',exchange:'Binance',instrument:'ETHUSDT',account:'Main · USDT-M',capital:22000,rate:.00035,phase:9,side:'Short',exposure:18000},
  {id:'sol',name:'Solana Breakout',exchange:'Bybit',instrument:'SOLUSDT',account:'Trading · Linear',capital:18000,rate:-.0001,phase:13,side:'Long',exposure:12500},
] as const;
export type Strategy = typeof strategies[number];
export type Basket = {id:string;name:string;members:string[]};
export type Mode = 'live'|'paper';
export type Point = {time:number;equity:number;wealth:number;drawdown:number;flow:number};
export const builtins = [
  {id:'total',name:'Total',members:strategies.map(s=>s.id)},
  ...['Binance','Bybit'].map(exchange=>({id:exchange,name:exchange,members:strategies.filter(s=>s.exchange===exchange).map(s=>s.id)})),
  ...['BTCUSDT','ETHUSDT','SOLUSDT'].map(instrument=>({id:instrument,name:instrument,members:strategies.filter(s=>s.instrument===instrument).map(s=>s.id)})),
];
export function history(s:Pick<Strategy,'capital'|'rate'|'phase'>,mode:Mode) {
  let equity=s.capital*(mode==='paper'?.25:1);
  return Array.from({length:269},(_,i)=>{
    if(i) equity*=1+s.rate+(Math.sin(i*.48+s.phase)*.0035+Math.cos(i*.12+s.phase)*.002)*(mode==='paper'?1.2:1);
    return equity;
  });
}
export function portfolio(members:string[],scope:string,mode:Mode,from:number,to:number) {
  const selected=strategies.filter(s=>members.includes(s.id));
  const accountBacked=['total','Binance','Bybit'].includes(scope);
  const cash=accountBacked?(scope==='total'?10000:5000)*(mode==='paper'?.25:1):0;
  const series=selected.map(s=>({s,values:history(s,mode)}));
  let wealth=1,peak=1,previous=0;
  const all:Point[]=Array.from({length:269},(_,i)=>{
    const flow=accountBacked&&i===120?(scope==='total'?5000:2500)*(mode==='paper'?.25:1):0;
    const equity=series.reduce((sum,row)=>sum+row.values[i]!,0)+cash+(accountBacked&&i>=120?(scope==='total'?5000:2500)*(mode==='paper'?.25:1):0);
    if(i&&previous)wealth*=1+(equity-previous-flow)/previous;
    peak=Math.max(peak,wealth);previous=equity;
    return {time:START+i*DAY,equity,wealth,drawdown:(wealth/peak-1)*100,flow};
  });
  const begin=Math.max(0,Math.min(268,Math.floor((from-START)/DAY)));
  const end=Math.max(begin,Math.min(268,Math.floor((to-START)/DAY)));
  const first=all[begin]!,last=all[end]!,current=all.at(-1)!;
  const pnl=series.reduce((sum,row)=>sum+row.values[end]!-row.values[begin]!,0);
  const returns=first.equity?(last.wealth/first.wealth-1)*100:0;
  // Link daily contribution amounts by prior portfolio wealth. This reconciles
  // with aggregate TWR for these explicit end-of-day-flow fixture assumptions.
  const rows=series.map(({s,values})=>{
    let contribution=0;
    for(let i=begin+1;i<=end;i++){
      const daily=(values[i]!-values[i-1]!)/all[i-1]!.equity;
      contribution+=daily*(last.wealth/all[i]!.wealth)*100;
    }
    let high=values[begin]!,dd=0;
    for(let i=begin;i<=end;i++){high=Math.max(high,values[i]!);dd=Math.min(dd,(values[i]!/high-1)*100);}
    return {...s,current:values.at(-1)!,pnl:values[end]!-values[begin]!,returns:(values[end]!/values[begin]!-1)*100,contribution,dd,exposure:s.exposure*(mode==='paper'?.25:1)};
  });
  const points=all.slice(begin,end+1).map(p=>({...p,wealth:first.wealth?p.wealth/first.wealth:1}));
  const monthly=Array.from({length:9},(_,month)=>{
    const start=Math.max(begin,Math.round((Date.UTC(2026,month,1,12)-START)/DAY));
    const stop=Math.min(end,Math.round((Date.UTC(2026,month+1,1,12)-START)/DAY)-1);
    if(start>stop)return null;
    const base=all[Math.max(begin,start-1)]!;
    return (all[stop]!.wealth/base.wealth-1)*100;
  });
  return {rows,all,points,current,pnl,returns,cash:cash+(accountBacked?(scope==='total'?5000:2500)*(mode==='paper'?.25:1):0),accountBacked,monthly,maxDD:Math.min(...points.map(p=>p.drawdown)),gross:rows.reduce((sum,s)=>sum+s.exposure,0),net:rows.reduce((sum,s)=>sum+(s.side==='Long'?s.exposure:-s.exposure),0)};
}
export function readBaskets(key:string):{baskets:Basket[];selected:string} {
  try {
    const data=JSON.parse(localStorage.getItem(key)??'null');
    if(!data||!Array.isArray(data.baskets))return {baskets:[],selected:'total'};
    const baskets=data.baskets.filter((b:Basket)=>typeof b.id==='string'&&b.id.startsWith('custom-')&&typeof b.name==='string'&&Array.isArray(b.members)&&b.members.every(id=>strategies.some(s=>s.id===id))).map((b:Basket)=>({...b,members:[...new Set(b.members)]}));
    return {baskets,selected:typeof data.selected==='string'?data.selected:'total'};
  }catch{return {baskets:[],selected:'total'};}
}
