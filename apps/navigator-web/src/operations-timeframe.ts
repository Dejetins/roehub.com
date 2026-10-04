export const chartIntervals:Record<string,number>={'1m':60000,'5m':300000,'15m':900000,'30m':1800000,'1h':3600000,'4h':14400000,'1d':86400000};
type Candle={timestamp:string;open:number|null;high:number|null;low:number|null;close:number|null};
export function rollupCandles(candles:Candle[],interval:number):Candle[]{
 const groups=new Map<number,Candle>();
 for(const c of [...candles].sort((a,b)=>Date.parse(a.timestamp)-Date.parse(b.timestamp))){
  if([c.open,c.high,c.low,c.close].some(v=>v==null))continue;
  const bucket=Math.floor(Date.parse(c.timestamp)/interval)*interval;
  const old=groups.get(bucket);
  if(old){old.high=Math.max(old.high!,c.high!);old.low=Math.min(old.low!,c.low!);old.close=c.close;}
  else groups.set(bucket,{...c,timestamp:new Date(bucket).toISOString()});
 }
 return [...groups.values()];
}
