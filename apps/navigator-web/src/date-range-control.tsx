import {useId,useRef,useState} from 'react';
import {CalendarDays} from 'lucide-react';
import {useTranslation} from 'react-i18next';
import type {TimeWindow} from './chart-period';
export const utcDate=(time:number)=>new Date(time).toISOString().slice(0,10);
/** Inclusive calendar dates, clipped to the available history (UTC). */
export function dateWindow(from:string,to:string,bounds:TimeWindow):TimeWindow|null {
 const start=Date.parse(`${from}T00:00:00Z`),end=Date.parse(`${to}T00:00:00Z`)+86400000-1;
 if(!Number.isFinite(start)||!Number.isFinite(end)||from>to||from<utcDate(bounds.start)||to>utcDate(bounds.end))return null;
 return {start:Math.max(start,bounds.start),end:Math.min(end,bounds.end)};
}
export function DateRangeControl({bounds,value,active,onApply}:{bounds:TimeWindow|null;value:TimeWindow|null;active:boolean;onApply:(range:TimeWindow)=>void}){
 const {i18n}=useTranslation(),ru=i18n.language.startsWith('ru');
 const tr=(en:string,ruText:string)=>ru?ruText:en;
 const id=useId(),popup=useRef<HTMLDivElement>(null),trigger=useRef<HTMLButtonElement>(null);
 const [draft,setDraft]=useState({from:'',to:''}),[error,setError]=useState(false);
 const rangeLabel=value?new Intl.DateTimeFormat(i18n.language,{day:'numeric',month:'short',...(new Date(value.start).getUTCFullYear()!==new Date(value.end).getUTCFullYear()?{year:'numeric' as const}:{}),timeZone:'UTC'}).formatRange(value.start,value.end):'';
 const close=()=>{popup.current?.hidePopover();trigger.current?.focus();};
 const open=()=>{
  if(!bounds)return;
  const range=value??bounds;
  setDraft({from:utcDate(Math.max(bounds.start,range.start)),to:utcDate(Math.min(bounds.end,range.end))});setError(false);
  const box=trigger.current!.getBoundingClientRect();
  const panel=popup.current!;panel.style.left=`${Math.max(8,Math.min(box.left,window.innerWidth-308))}px`;
  panel.style.top=`${Math.max(8,Math.min(box.bottom+8,window.innerHeight-250))}px`;
  panel.showPopover();panel.querySelector('input')?.focus();
 };
 return <div className="date-range-control view-switch">
  <button ref={trigger} type="button" disabled={!bounds} aria-haspopup="dialog" aria-controls={id} aria-pressed={active} onClick={open}><CalendarDays size={14} aria-hidden="true"/>{active&&value?rangeLabel:tr('Date range','Период')}</button>
  <div ref={popup} id={id} popover="auto" role="dialog" aria-label={tr('Date range','Период')} className="date-range-popup" onKeyDown={e=>{if(e.key==='Escape'){e.preventDefault();close();}}}>
   <form onSubmit={e=>{e.preventDefault();const range=bounds&&dateWindow(draft.from,draft.to,bounds);if(!range){setError(true);popup.current?.querySelector('input')?.focus();return;}onApply(range);close();}}>
    <div className="date-range-fields"><label>{tr('From','С')}<input type="date" required min={bounds?utcDate(bounds.start):undefined} max={bounds?utcDate(bounds.end):undefined} value={draft.from} aria-invalid={error} onInput={e=>{const value=e.currentTarget.value;setDraft(current=>({...current,from:value}));}}/></label><label>{tr('To','По')}<input type="date" required min={draft.from||undefined} max={bounds?utcDate(bounds.end):undefined} value={draft.to} aria-invalid={error} onInput={e=>{const value=e.currentTarget.value;setDraft(current=>({...current,to:value}));}}/></label></div>
    {error&&<p role="alert">{tr('Choose dates within the available history.','Выберите даты в пределах доступной истории.')}</p>}
    <div className="actions"><button type="button" onClick={close}>{tr('Cancel','Отмена')}</button><button type="submit" className="primary">{tr('Apply','Применить')}</button></div>
   </form>
  </div>
 </div>;
}
