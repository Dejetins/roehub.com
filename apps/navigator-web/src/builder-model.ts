import { type Bounds, type Catalog, type Defaults, type Research, researchSchema } from './builder-api';
export type Issue = { path: string; code: string; message: string };
/** Calendar dates represent UTC day boundaries, independent of browser timezone. */
export function dateBoundary(value: string): string {
  return /^\d{4}-\d{2}-\d{2}$/.test(value) ? `${value}T00:00:00Z` : '';
}
export function firstWindow(d: Defaults, id: string) {
  const s = d.indicator_param_specs[id]?.params.window;
  return s?.mode === 'explicit' ? s.values[0] : s?.start;
}
export function initialResearch(d: Defaults, c: Catalog): Research {
  const draft = c.config_draft;
  return { strategy_name: '', coordinates: { ...draft.coordinates }, timeframe: draft.timeframe,
    time_range: { start: dateBoundary(draft.time_range.start.slice(0,10)), end: dateBoundary(draft.time_range.end.slice(0,10)) },
    indicators: draft.indicators.map(i => ({ ...i, window: { start: firstWindow(d, i.indicator_id) ?? 0, stop: firstWindow(d, i.indicator_id) ?? 0, step: 1 } })),
    risk: { ...draft.risk }, execution: { ...d.execution_defaults, direction_mode: draft.execution.direction_mode }, ranking: { ...d.ranking_default },
    top_n: d.top_n_default, quality_constraints: { ...d.quality_constraints_default } };
}
/** Explicit local demonstration preset; never substitutes normal server defaults. */
export function syntheticResearch(d: Defaults, c: Catalog): Research {
  const b = initialResearch(d,c);
  return {...b, strategy_name:'Synthetic demo · EMA 5–50',
    coordinates:{exchange:'binance',market_type:'spot',symbol:'BTCUSDT'}, timeframe:'15m',
    time_range:{start:'2026-02-27T00:00:00Z',end:'2026-03-29T00:00:00Z'},
    indicators:[{indicator_id:'ma.ema',sources:['close'],window:{start:5,stop:50,step:5}}],
    risk:{mode:'tp_sl_grid',tp:{enabled:true,start_pct:1,stop_pct:2,step_pct:1},sl:{enabled:true,start_pct:1,stop_pct:2,step_pct:1}}, execution:{...b.execution,direction_mode:'long_only',initial_cash_quote:10000,fee_rate:0.00075,slippage_rate:0.0001,sizing:{mode:'all_in'}},
    ranking:{primary_metric:'total_return_pct',direction:'desc'},top_n:10};
}
/** Inactive mode fields are draft state, never submitted compute inputs. */
export function prepareResearch(b: Research): Research {
  const side=(s:Research['risk']['tp'])=>s?.enabled?{enabled:true,start_pct:s.start_pct,stop_pct:s.stop_pct,step_pct:s.step_pct}:{enabled:false};
  const risk=b.risk.mode==='none'?{mode:'none'}:{mode:b.risk.mode,tp:side(b.risk.tp),sl:side(b.risk.sl)};
  const original=b.execution.sizing;const sizing:Research['execution']['sizing']={mode:original.mode};
  if(original.mode==='fixed_quote')sizing.quote_amount=original.quote_amount;
  if(original.mode.startsWith('fixed_equity_pct'))sizing.equity_pct=original.equity_pct;
  if(original.mode.endsWith('min_quote'))sizing.min_quote=original.min_quote;
  if(original.mode.endsWith('max_quote'))sizing.max_quote=original.max_quote;
  return researchSchema.parse({...b,risk,execution:{...b.execution,sizing}});
}
export function computationIdentity(body: Research) {
  const { strategy_name: _label, ...rest } = researchSchema.parse(body);
  return JSON.stringify(rest);
}
export function normalizedLabel(value: string) { return Array.from(value.trim().split(/\s+/).join(' ')).slice(0, 96).join(''); }
export function fieldPath(path: string) { return path.replace(/^body\./, '').replace(/\[(\d+)\]/g, '.$1'); }
export function validateResearch(b: Research, d: Defaults, c: Catalog, bounds?: Bounds): Issue[] {
  const issues: Issue[] = [];
  const add = (path: string, code: string) => issues.push({ path, code, message: code });
  const choice = (path: string, v: string, allowed: readonly string[]) => { if (!allowed.includes(v)) add(path, 'unsupported'); };
  const positive = (path: string, n: number, integer = false) => { if (!Number.isFinite(n) || n <= 0 || (integer && !Number.isSafeInteger(n))) add(path, integer ? 'positiveInteger' : 'positive'); };
  for (const [key, options] of Object.entries({ exchange: c.instrument_universe.markets, market_type: c.instrument_universe.market_types, symbol: c.instrument_universe.symbols })) {
    choice(`coordinates.${key}`, b.coordinates[key as keyof Research['coordinates']], options.filter(o => o.status === 'available').map(o => o.value));
  }
  if (!['ready','degraded'].includes(c.instrument_universe.state) || !['ready','degraded'].includes(c.indicator_catalog.state)) add('coordinates', 'catalogUnavailable');
  choice('timeframe', b.timeframe, d.supported_timeframes);
  for (const side of ['start','end'] as const) {
    const value=b.time_range[side]; const date=new Date(value);
    if (!/^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(\.\d{1,3})?Z$/.test(value) || !Number.isFinite(date.getTime()) || date.toISOString().slice(0,19) !== value.slice(0,19)) add(`time_range.${side}`, 'utc');
  }
  if (!(Date.parse(b.time_range.start) < Date.parse(b.time_range.end))) add('time_range.end','halfOpen');
  if (!bounds || bounds.state !== 'ready' || !bounds.max_end || JSON.stringify(bounds.coordinates) !== JSON.stringify(b.coordinates)) add('time_range','boundsUnavailable');
  else if (Date.parse(b.time_range.end) > Date.parse(bounds.max_end)) add('time_range.end','dateBounds');
  if (!b.indicators.length || b.indicators.length > d.guardrails.max_indicator_arity) add('indicators','arity');
  let rows=0, combinations=1;
  b.indicators.forEach((i,index) => {
    const path=`indicators.${index}`;
    choice(`${path}.indicator_id`,i.indicator_id,d.supported_indicator_ids.filter(id=>c.indicator_catalog.items.some(r=>r.indicator_id===id && r.status==='available')));
    const sources=d.indicator_sources[i.indicator_id] ?? [];
    if ((sources.length > 0 && !i.sources.length) || i.sources.some(s=>!sources.includes(s))) add(`${path}.sources`,'sources');
    const w=i.window; for (const k of ['start','stop','step'] as const) positive(`${path}.window.${k}`,w[k],true);
    const count=Math.floor((w.stop-w.start)/w.step)+1; const spec=d.indicator_param_specs[i.indicator_id]?.params.window;
    if (!spec || w.start>w.stop || !Number.isSafeInteger(count) || count<1) add(`${path}.window`,'grid');
    else if (spec.mode==='explicit') {
      if (count>spec.values.length || !Number.isSafeInteger(w.start) || !Number.isSafeInteger(w.step) || !Array.from({length:Math.min(count,spec.values.length+1)},(_,j)=>w.start+j*w.step).every(v=>spec.values.includes(v))) add(`${path}.window`,'grid');
    } else if (w.start<spec.start || w.start+(count-1)*w.step>spec.stop_incl || (w.start-spec.start)%spec.step!==0 || (count>1 && w.step%spec.step!==0)) add(`${path}.window`,'grid');
    const row=count*Math.max(1,new Set(i.sources).size); rows+=row; combinations*=row;
  });
  if (rows>d.guardrails.max_indicator_rows || combinations>d.guardrails.max_candidate_combinations) add('indicators','cost');
  choice('risk.mode',b.risk.mode,d.risk_modes);
  if (b.risk.mode==='tp_sl_grid') {
    if (!b.risk.tp?.enabled && !b.risk.sl?.enabled) add('risk','riskEmpty');
    let cells=1;
    for (const side of ['tp','sl'] as const) {
      const s=b.risk[side]; if (!s?.enabled) continue;
      const levels=d.hit_times_grid[`${side}_levels_pct`];
      const a=s.start_pct ?? NaN, end=s.stop_pct ?? NaN, step=s.step_pct ?? NaN;
      for (const [k,v] of Object.entries({start_pct:a,stop_pct:end,step_pct:step})) positive(`risk.${side}.${k}`,v);
      const count=Math.floor(Number(((end-a)/step).toPrecision(12)))+1;
      if (!Number.isSafeInteger(count) || count<1 || count>levels.length || a>end || !Array.from({length:Math.max(0,Math.min(count || 0,levels.length+1))},(_,j)=>Number((a+j*step).toPrecision(12))).every(v=>levels.includes(v))) add(`risk.${side}`,'riskGrid');
      cells*=count;
    }
    if (cells>d.guardrails.max_tp_sl_cells) add('risk','cost');
  }
  const e=b.execution;
  choice('execution.direction_mode',e.direction_mode,d.direction_modes.filter(v=>d.direction_market_compatibility.markets[b.coordinates.market_type]?.allowed_direction_modes.includes(v)));
  positive('execution.initial_cash_quote',e.initial_cash_quote);
  for (const k of ['fee_rate','slippage_rate'] as const) if (!Number.isFinite(e[k]) || e[k]<0) add(`execution.${k}`,'nonnegative');
  choice('execution.sizing.mode',e.sizing.mode,d.sizing_modes);
  if (e.sizing.mode==='fixed_quote') positive('execution.sizing.quote_amount',e.sizing.quote_amount ?? NaN);
  if (e.sizing.mode.startsWith('fixed_equity_pct')) { positive('execution.sizing.equity_pct',e.sizing.equity_pct ?? NaN); if ((e.sizing.equity_pct ?? 0)>100) add('execution.sizing.equity_pct','percent'); }
  for (const k of ['min_quote','max_quote'] as const) if (e.sizing.mode.endsWith(k)) positive(`execution.sizing.${k}`,e.sizing[k] ?? NaN);
  choice('ranking.primary_metric',b.ranking.primary_metric,d.ranking_metrics); choice('ranking.direction',b.ranking.direction,['asc','desc']);
  if (b.ranking.primary_metric==='total_return_pct_net_of_funding' && (b.coordinates.market_type!=='futures' || e.funding.mode!=='include_when_futures')) add('ranking.primary_metric','fundingMetric');
  positive('top_n',b.top_n,true); if (b.top_n>d.guardrails.max_top_n) add('top_n','topN');
  return issues;
}
