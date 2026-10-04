import {useLayoutEffect, useRef, type RefObject} from 'react';
import {init} from 'echarts/core';

/** Canvas lifetime follows its DOM host, never its data, locale, selection or polling. */
export function useChartInstance(host:RefObject<HTMLDivElement|null>) {
  const instance=useRef<ReturnType<typeof init>|null>(null);
  useLayoutEffect(()=>{
    if(!host.current)return;
    const chart=init(host.current);instance.current=chart;
    const resize=new ResizeObserver(()=>chart.resize());resize.observe(host.current);
    return()=>{resize.disconnect();chart.dispose();instance.current=null;};
  },[host]);
  return instance;
}
