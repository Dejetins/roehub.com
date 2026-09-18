import {useEffect,useId,useRef,useState} from 'react';
import {Check,ChevronDown,ListChecks} from 'lucide-react';

type DisplayOption={label:string;checked:boolean;onChange:(checked:boolean)=>void};
/** Shared chart popup; the entire option row is one focusable control. */
export function ChartDisplayMenu({label,options,value,multiple=false}:{label:string;options:DisplayOption[];value?:string;multiple?:boolean}){
 const [open,setOpen]=useState(false);
 const root=useRef<HTMLDivElement>(null),trigger=useRef<HTMLButtonElement>(null);
 const id=useId();
 useEffect(()=>{
  if(!open)return;
  root.current?.querySelector<HTMLElement>('.chart-display-option[aria-checked=true],.chart-display-option')?.focus();
  const dismiss=(event:PointerEvent)=>{if(event.target instanceof Node&&!root.current?.contains(event.target))setOpen(false);};
  document.addEventListener('pointerdown',dismiss);
  return()=>document.removeEventListener('pointerdown',dismiss);
 },[open]);
 return <div className="chart-display-menu" ref={root} onBlur={event=>{if(!event.currentTarget.contains(event.relatedTarget))setOpen(false);}} onKeyDown={event=>{
  if(event.key==='Escape'&&open){event.preventDefault();event.stopPropagation();setOpen(false);trigger.current?.focus();}
  if(open&&['ArrowDown','ArrowUp','Home','End'].includes(event.key)){
   event.preventDefault();const items=[...event.currentTarget.querySelectorAll<HTMLButtonElement>('.chart-display-option')];const current=items.indexOf(document.activeElement as HTMLButtonElement);
   const next=event.key==='Home'?0:event.key==='End'?items.length-1:(current+(event.key==='ArrowUp'?-1:1)+items.length)%items.length;items[next]?.focus();if(value&&!multiple)options[next]?.onChange(true);
  }
 }}>
  <button ref={trigger} type="button" className={`chart-display-trigger${value?' chart-timeframe-trigger':''}`} aria-label={label} title={label} aria-expanded={open} aria-controls={id} onClick={()=>setOpen(value=>!value)}>{value?<>{value}<ChevronDown size={12} aria-hidden="true"/></>:<ListChecks size={17} aria-hidden="true"/>}</button>
  {open&&<div id={id} className="chart-display-popup" role={value&&!multiple?'radiogroup':'group'} aria-label={label}>{options.map(option=><button type="button" role={value&&!multiple?'radio':'checkbox'} aria-checked={option.checked} className="chart-display-option" key={option.label} onClick={()=>{
   option.onChange(!option.checked);
   if(value&&!multiple){setOpen(false);trigger.current?.focus();}
  }}>
   <span className="chart-display-check" data-checked={option.checked} aria-hidden="true"><Check size={14}/></span><span>{option.label}</span>
  </button>)}</div>}
 </div>;
}
