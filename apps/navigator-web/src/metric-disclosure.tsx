import {useEffect,useId,useRef,useState,type ReactNode} from 'react';
import {ChevronDown} from 'lucide-react';

export function MetricDisclosure({label,value,valueClass,disclosureLabel,children}:{label:string;value:string;valueClass:string;disclosureLabel:string;children:ReactNode}){
 const [open,setOpen]=useState(false);
 const root=useRef<HTMLDivElement>(null),trigger=useRef<HTMLButtonElement>(null);
 const id=useId();
 useEffect(()=>{
  if(!open)return;
  const outside=(event:PointerEvent)=>{if(event.target instanceof Node&&!root.current?.contains(event.target))setOpen(false);};
  const escape=(event:KeyboardEvent)=>{if(event.key==='Escape'){setOpen(false);trigger.current?.focus();}};
  document.addEventListener('pointerdown',outside);
  document.addEventListener('keydown',escape);
  return()=>{document.removeEventListener('pointerdown',outside);document.removeEventListener('keydown',escape);};
 },[open]);
 return <div className="overview-metric metric-disclosure" ref={root}>
  <span>{label}</span><strong className={valueClass}>{value}</strong>
  <button type="button" ref={trigger} className="metric-disclosure-trigger" aria-label={disclosureLabel} title={disclosureLabel} aria-expanded={open} aria-controls={id} onClick={()=>setOpen(value=>!value)}><ChevronDown size={16} aria-hidden="true"/></button>
  {open&&<div id={id} className="metric-disclosure-popup" role="region" aria-label={disclosureLabel}>{children}</div>}
 </div>;
}
