import {ChevronRight} from 'lucide-react';

const exchangeLogos:Record<string,string>={binance:new URL('./assets/exchanges/binance.ico',import.meta.url).href,bybit:new URL('./assets/exchanges/bybit.svg',import.meta.url).href};
export const exchangeName=(name:string)=>name==='binance'?'Binance':name==='bybit'?'Bybit':name;

export function ExchangeLinkContent({name}:{name:string}){
 return <><span className={`data-exchange-mark ${name}`} aria-hidden="true"><img src={exchangeLogos[name]} className={`data-exchange-logo ${name}`} alt=""/></span>{name!=='bybit'&&exchangeName(name)}<ChevronRight/></>;
}
