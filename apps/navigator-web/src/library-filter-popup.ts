import {useEffect,useRef,useState} from 'react';
/** Shared Library-bound top-layer filter geometry and dismissal state. */
export function useLibraryFilterPopup(){
 const [open,setOpen]=useState(false),trigger=useRef<HTMLButtonElement>(null),popup=useRef<HTMLElement>(null);
 function position(){const panel=popup.current,button=trigger.current,library=button?.closest('.navigator-library');if(!panel||!button||!library)return;
 const bounds=library.getBoundingClientRect(),top=Math.max(8,button.getBoundingClientRect().bottom+8);
 Object.assign(panel.style,{left:`${bounds.left+12}px`,width:`${Math.max(0,bounds.width-24)}px`,top:`${top}px`,maxHeight:`${Math.max(0,Math.min(bounds.bottom-12,window.innerHeight-8)-top)}px`});}
 function close(){popup.current?.hidePopover();trigger.current?.focus();}
 function toggle(){const panel=popup.current;if(!panel)return;if(panel.matches(':popover-open')){close();return;}position();panel.showPopover();panel.querySelector<HTMLElement>('input,button')?.focus();}
 useEffect(()=>{if(!open)return;const library=trigger.current?.closest('.navigator-library'),observer=new ResizeObserver(position);if(library)observer.observe(library);window.addEventListener('resize',position);window.addEventListener('scroll',position,true);return()=>{observer.disconnect();window.removeEventListener('resize',position);window.removeEventListener('scroll',position,true);};},[open]);
 return {open,setOpen,trigger,popup,toggle,close};
}
