type Point={timestamp:string;value:number};
export function buyAndHold(equity:Point[],candles:{timestamp:string;close:number|null}[]):Point[]{
 const start=equity[0],end=equity.at(-1);
 if(!start||!end||start.value<=0)return [];
 const bars=candles.filter(c=>c.close!=null&&c.close>0).sort((a,b)=>Date.parse(a.timestamp)-Date.parse(b.timestamp));
 const anchor=[...bars].reverse().find(c=>Date.parse(c.timestamp)<=Date.parse(start.timestamp));
 if(!anchor)return [];
 return [{timestamp:start.timestamp,value:start.value},...bars.filter(c=>Date.parse(c.timestamp)>Date.parse(start.timestamp)&&Date.parse(c.timestamp)<=Date.parse(end.timestamp)).map(c=>({timestamp:c.timestamp,value:start.value*c.close!/anchor.close!}))];
}
