import { z } from 'zod';
import { requestJson } from './api';
const strings = z.array(z.string());
const numbers = z.array(z.number());
export const coordinatesSchema = z.object({ exchange: z.string(), market_type: z.string(), symbol: z.string() });
const windowSchema = z.object({ start: z.number(), stop: z.number(), step: z.number() });
const fundingSchema = z.object({ mode: z.string(), coverage_policy: z.string() });
const sizingSchema = z.object({ mode: z.string(), quote_amount: z.number().optional(), equity_pct: z.number().optional(), min_quote: z.number().optional(), max_quote: z.number().optional() });
export const executionSchema = z.object({ direction_mode: z.string(), initial_cash_quote: z.number(), fee_rate: z.number(), slippage_rate: z.number(), sizing: sizingSchema,
  funding: fundingSchema, profit_lock: z.object({ enabled: z.boolean(), safe_profit_percent: z.number().optional() }), close_on_end: z.boolean() });
const sideSchema = z.object({ enabled: z.boolean(), start_pct: z.number().optional(), stop_pct: z.number().optional(), step_pct: z.number().optional(), levels_pct: numbers.optional() });
const riskSchema = z.object({ mode: z.string(), tp: sideSchema.optional(), sl: sideSchema.optional() });
const rankingSchema = z.object({ primary_metric: z.string(), direction: z.string(), effective_primary_metric: z.string().optional() });
const qualitySchema = z.object({ min_closed_trades: z.number().optional(), min_closed_trades_policy: z.string().optional(), base_trades_per_year_at_1h: z.number().optional(), min_trades_per_year: z.number().optional(), max_trades_per_year: z.number().optional() });
/** This exact allowlist is used both on the wire and in bounded recovery. */
export const researchSchema = z.object({ strategy_name: z.string().optional(), coordinates: coordinatesSchema, timeframe: z.string(),
  time_range: z.object({ start: z.string(), end: z.string() }), indicators: z.array(z.object({ indicator_id: z.string(), sources: strings, window: windowSchema })),
  risk: riskSchema, execution: executionSchema, ranking: rankingSchema, top_n: z.number(), quality_constraints: qualitySchema });
export type Research = z.infer<typeof researchSchema>;
const gridSpec = z.union([z.object({ mode: z.literal('explicit'), values: numbers }), z.object({ mode: z.literal('range'), start: z.number(), stop_incl: z.number(), step: z.number() })]);
export const defaultsSchema = z.object({ supported_timeframes: strings.nonempty(), risk_modes: strings.nonempty(), direction_modes: strings.nonempty(), sizing_modes: strings.nonempty(), ranking_metrics: strings.nonempty(),
  ranking_default: rankingSchema, top_n_default: z.number(), quality_constraints_default: qualitySchema,
  guardrails: z.object({ max_top_n: z.number(), max_indicator_arity: z.number(), max_indicator_rows: z.number(), max_candidate_combinations: z.number(), max_tp_sl_cells: z.number() }),
  execution_defaults: executionSchema, supported_indicator_ids: strings.nonempty(), indicator_sources: z.record(z.string(), strings),
  indicator_param_specs: z.record(z.string(), z.object({ params: z.object({ window: gridSpec.optional() }) })),
  hit_times_grid: z.object({ timeframe: z.string(), tp_levels_pct: numbers, sl_levels_pct: numbers }),
  direction_market_compatibility: z.object({ markets: z.record(z.string(), z.object({ allowed_direction_modes: strings, funding_default: fundingSchema })) }) });
export type Defaults = z.infer<typeof defaultsSchema>;
const option = z.object({ value: z.string(), label: z.string(), status: z.enum(['available', 'disabled']) });
export const metadataSchema = z.object({ artifact_asof_date: z.string().optional(), published_at_utc: z.string().optional(), artifact_slot_generation: z.number().optional(),
  funding_coverage_status: z.string().nullable().optional(), funding_expected_event_count: z.number().nullable().optional(), funding_missing_event_count: z.number().nullable().optional() });
export const boundsSchema = z.object({ state: z.enum(['ready', 'unavailable']), coordinates: coordinatesSchema, default_end: z.string().nullable(), max_end: z.string().nullable(), artifact_metadata: metadataSchema.nullable().optional() });
export type Bounds = z.infer<typeof boundsSchema>;
export const builderCatalogSchema = z.object({ generated_at: z.string(),
  sources: z.array(z.object({ name: z.string(), status: z.string(), generated_at: z.string().nullable() })),
  config_draft: z.object({ coordinates: coordinatesSchema, timeframe: z.string(), time_range: z.object({ start: z.string(), end: z.string() }),
    indicators: z.array(z.object({ indicator_id: z.string(), sources: strings })), risk: riskSchema, execution: z.object({ direction_mode: z.string() }) }),
  instrument_universe: z.object({ state: z.string(), markets: z.array(option), market_types: z.array(option), symbols: z.array(option) }),
  indicator_catalog: z.object({ state: z.string(), items: z.array(z.object({ indicator_id: z.string(), label: z.string(), status: z.string() })) }) });
export type Catalog = z.infer<typeof builderCatalogSchema>;
const issue = z.object({ path: z.string(), code: z.string(), message: z.string() });
export const preflightSchema = z.object({ normalized_request: researchSchema, request_hash: z.string(), result_config_hash: z.string(), artifact_metadata: metadataSchema,
  cost_estimate: z.object({ indicator_rows: z.number(), candidate_combinations: z.number(), tp_sl_cells: z.number(), cost_class: z.string(), scheduling_class: z.string().optional() }),
  warnings: z.array(issue), errors: z.array(issue), funding_readiness: z.object({ status: z.string(), coverage_policy: z.string(), coverage_ratio: z.number().nullable(), rows_count: z.number(), expected_event_count: z.number(), missing_event_count: z.number(), warning_codes: strings }),
  direction_market_compatibility: z.object({ compatible: z.boolean(), market_type: z.string(), direction_mode: z.string() }) });
export type Preflight = z.infer<typeof preflightSchema>;
export const createdSchema = z.object({ job_id: z.uuid(), organization_id: z.uuid(), request_hash: z.string(), result_config_hash: z.string(), state: z.string() });
export const readDefaults = (signal?: AbortSignal) => requestJson('/api/backtests/runtime-defaults', defaultsSchema, { signal }).then(r => r.data);
export const readBuilderCatalog = (signal?: AbortSignal, coordinates?: Pick<Research['coordinates'], 'exchange'|'market_type'>) => requestJson(`/api/ui/backtests/workstation${coordinates ? `?${new URLSearchParams({instrument_exchange:coordinates.exchange,instrument_market_type:coordinates.market_type})}` : ''}`, builderCatalogSchema, { signal }).then(r => r.data);
export const readBounds = (coordinates: Research['coordinates'], signal?: AbortSignal) => requestJson(`/api/ui/backtests/artifact-date-bounds?${new URLSearchParams(coordinates)}`, boundsSchema, { signal }).then(r => r.data);
export const preflightRequest = (body: Research, signal?: AbortSignal) => requestJson('/api/backtests/preflight', preflightSchema, { method: 'POST', body, signal }).then(r => r.data);
export const createJob = (body: Research, key: string, signal?: AbortSignal) => requestJson('/api/backtests/jobs', createdSchema, { method: 'POST', body, idempotencyKey: key, signal });
