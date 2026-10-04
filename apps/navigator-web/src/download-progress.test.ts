import {expect,it} from 'vitest';
import {DownloadRate,downloadPercent,downloadState} from './download-progress';
import type {WorkJob} from './market-data-api';

const job=(units:number,extra:Partial<WorkJob>={}):WorkJob=>({job_id:'a',kind:'candle_ingestion',market_id:1,symbols:['BTCUSDT'],timeframe:'1m',start_at:null,end_at:null,state:'running',attempt:1,completed_units:units,total_units:12000,rows_read:units,rows_written:units,error_code:null,created_at:'2024-01-01Z',updated_at:'2024-01-01Z',started_at:'2024-01-01Z',finished_at:null,can_cancel:true,can_retry:false,attempts:[],...extra});
it('keeps yielded work downloading, but preserves initial queue and real terminal/cancel states',()=>{
 expect(downloadState(job(0,{state:'queued',started_at:null}))).toBe('queued');
 expect(downloadState(job(0,{state:'queued'}))).toBe('downloading');
 expect(downloadState(job(60,{state:'queued',started_at:null}))).toBe('downloading');
 for(const state of ['failed','cancelled','cancel_requested','succeeded'])expect(downloadState(job(60,{state}))).toBe(state);
 expect(downloadPercent(job(11999))).toBe(99.99);
 expect(downloadPercent(job(12000))).toBe(99.99);
 expect(downloadPercent(job(12000,{state:'succeeded'}))).toBe(100);
});
it('warms up, includes pauses in speed, smooths ETA and limits changes to ten-second intervals',()=>{
 const rate=new DownloadRate();
 expect(rate.observe(job(0),0).kind).toBe('estimating');
 expect(rate.observe(job(600),10000).kind).toBe('estimating');
 const first=rate.observe(job(600,{state:'queued'}),20000);
 expect(first).toEqual({kind:'ready',seconds:380}); // 600 units / 20s, including the pause.
 expect(rate.observe(job(1200),22000)).toBe(first);
 const next=rate.observe(job(1200),30000);
 expect(next.seconds).toBeCloseTo(10800/34); // smoothed 30 -> 40 units/s.
});
it('does not count down through a stall or read failure, and resets on retry',()=>{
 const rate=new DownloadRate();rate.observe(job(0),0);rate.observe(job(600),20000);
 expect(rate.observe(job(600),80000)).toEqual({kind:'stalled'});
 expect(rate.observe(job(600),82000,true)).toEqual({kind:'unavailable'});
 expect(rate.observe(job(660),84000).kind).toBe('estimating');
 expect(rate.observe(job(0,{attempt:2,state:'queued',started_at:null}),86000).kind).toBe('hidden');
 expect(rate.observe(job(0,{attempt:2}),88000).kind).toBe('estimating');
 expect(rate.observe(job(12000,{attempt:2,state:'succeeded'}),90000).kind).toBe('hidden');
});
it('uses recent throughput rather than the lifetime average of a long job',()=>{
 const rate=new DownloadRate();
 for(let at=0;at<=120000;at+=2000)rate.observe(job(at/1000*20),at);
 let eta;
 for(let at=122000;at<=400000;at+=2000)eta=rate.observe(job(2400+(at-120000)/1000*2),at);
 expect(Math.abs(eta!.seconds!-(12000-2960)/2)).toBeLessThan(5);
});

it('freezes ETA on pause/wait and recalculates without including downtime after resume',()=>{
 const rate=new DownloadRate();rate.observe(job(0),0);expect(rate.observe(job(600),20000).kind).toBe('ready');
 expect(rate.observe(job(600,{state:'paused'}),30000).kind).toBe('paused');
 expect(rate.observe(job(600,{state:'paused'}),600000).kind).toBe('paused');
 expect(rate.observe(job(600,{progress_epoch:1}),601000).kind).toBe('estimating');
 expect(rate.observe(job(1200,{progress_epoch:1}),621000).seconds).toBe(360);
 expect(rate.observe(job(1200,{state:'retry_wait',progress_epoch:2}),630000).kind).toBe('waiting');
 expect(rate.observe(job(1200,{progress_epoch:2}),900000).kind).toBe('estimating');
});

it('shows real queue competition and storage cooldown without an ETA that keeps counting',()=>{
 const rate=new DownloadRate();rate.observe(job(0),0);rate.observe(job(600),20000);
 const waiting=job(600,{state:'queued',queue_waiting:true});
 expect(downloadState(waiting)).toBe('queued');
 expect(rate.observe(waiting,30000).kind).toBe('waiting');
 expect(rate.observe(job(1200),90000).kind).toBe('ready');
 expect(downloadState(job(600,{state:'queued',storage_retry_at:'2026-10-04T12:00:00Z'}))).toBe('queued');
});


it('measures fair queue cycles even when polling misses their brief running states',()=>{
 const rate=new DownloadRate();
 rate.observe(job(0,{state:'queued',queue_waiting:true,started_at:null}),0);
 for(let at=2000;at<600000;at+=2000)rate.observe(job(0,{state:'queued',queue_waiting:true}),at);
 rate.observe(job(1440,{state:'queued',queue_waiting:true}),600000);
 const eta=rate.observe(job(1440),602000);
 expect(eta.kind).toBe('ready');
 expect(eta.seconds).toBeCloseTo((12000-1440)/(1440/602));
});
