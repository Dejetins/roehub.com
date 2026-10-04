import {useEffect,useRef,useState} from 'react';
import {useTranslation} from 'react-i18next';
import {useResultRead} from './result-read';
import {readResult,tradesSchema} from './results-api';
import {ApiError} from './api';
import {ReadError,isRestricted} from './library';
const PAGE=100,ROW=40;
const columns=['trade_index','entry_timestamp','exit_timestamp','side','entry_price','exit_price','quantity','net_pnl_quote','return_pct','fee_quote','exit_reason','equity_after'] as const;
export function tradeWindow(scrollTop:number,total:number){return Math.min(Math.max(0,Math.floor((scrollTop-32)/(PAGE*ROW))),Math.max(0,Math.ceil(total/PAGE)-1));}
/** Windowed server reads: at most two rendered pages, with bounded shared query cache. */
export function TradesScroll({subject,job,variant,now}:{subject:string;job:string;variant:string;now:number}){
 const {t,i18n}=useTranslation(),root=useRef<HTMLDivElement>(null),[page,setPage]=useState(0);
 function usePage(index:number,enabled=true){const suffix=`trades?page=${index+1}&page_size=${PAGE}`;return useResultRead(subject,[job,variant,suffix],async signal=>{const response=await readResult(job,variant,suffix,tradesSchema,signal);if(response.status===200&&('pagination' in response.data)&&(response.data.pagination.page!==index+1||response.data.pagination.page_size!==PAGE))throw new ApiError('invalid-response',200,'failed');return response;},now,enabled);}
 const first=usePage(0),total=first.data?.status===200&&'pagination' in first.data.data?first.data.data.pagination.total:0;
 const current=usePage(page,page>0&&total>0),next=usePage(page+1,(page+1)*PAGE<total);
 const active=page===0?first:current;
 const denied=[first,current,next].some(q=>isRestricted(q.error));
 useEffect(()=>{const scroller=root.current?.closest('.navigator-table-body');if(!scroller)return;scroller.scrollTop=0;},[]);
 useEffect(()=>{const scroller=root.current?.closest('.navigator-table-body');if(!scroller)return;const update=()=>setPage(tradeWindow(scroller.scrollTop,total));scroller.addEventListener('scroll',update,{passive:true});update();return()=>scroller.removeEventListener('scroll',update);},[total]);
 const rows=denied?[]:[...(active.data?.status===200&&'items' in active.data.data?active.data.data.items:[]),...(active.data?.status===200&&'items' in active.data.data&&next.data?.status===200&&'items' in next.data.data&&(page+1)*PAGE<total?next.data.data.items:[])];
 const failed=[first,active,next].find(query=>query.error);
 const error=failed?.error??null;
 const pending=first.isFetching||active.isFetching||next.isFetching||first.data?.status===202||active.data?.status===202;
 return <div ref={root} className="trades-scroll" aria-busy={pending}><ReadError error={error}/>{error&&!isRestricted(error)&&<button disabled={!failed?.canRefresh} onClick={()=>void failed?.refetch()}>{t('refresh')}</button>}
 <table aria-label={t('results.trades')} aria-rowcount={total+1}><thead><tr>{columns.map(key=><th key={key}>{t(`results.columns.${key}`)}</th>)}</tr></thead><tbody>
 {page>0&&<tr aria-hidden="true" className="trade-spacer"><td colSpan={columns.length} style={{height:page*PAGE*ROW}}/></tr>}
 {rows.map((row,index)=><tr key={row.trade_index} aria-rowindex={page*PAGE+index+2}>{columns.map(key=><td key={key}>{typeof row[key]==='number'?(row[key] as number).toLocaleString(i18n.language,{maximumFractionDigits:6}):row[key]??'—'}</td>)}</tr>)}
 {!rows.length&&<tr><td colSpan={columns.length}>{pending?t('results.loading'):t('results.empty')}</td></tr>}
 {total>page*PAGE+rows.length&&<tr aria-hidden="true" className="trade-spacer"><td colSpan={columns.length} style={{height:(total-page*PAGE-rows.length)*ROW}}/></tr>}
 </tbody></table>{pending&&<span className="trade-load-status" role="status">{t('results.loading')}</span>}</div>;
}
