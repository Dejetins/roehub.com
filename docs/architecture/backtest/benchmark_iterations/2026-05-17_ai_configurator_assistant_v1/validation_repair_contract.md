# Iteration 05 Validation/Repair/Load Gate Contract

Дата: 2026-05-18.

## Authority

Backend validation is the only authority for a loadable `/backtests` config.

`load_action.enabled=true` is allowed only when:

- final pipeline status is `ready`;
- `validated_config` is present;
- model JSON passed output safety, schema validation, catalog validation, artifact checks, and `BacktestPreflightService`;
- the final response is not an unsupported, clarification, policy-blocked, security-review, or failed state.

Frontend/client text must not infer `load_action` from `assistant_message`.

## Validation Path

Pipeline order:

1. deterministic input/security gate;
2. context/catalog resolve;
3. LM Studio generate through `POST /v1/chat/completions`;
4. safe JSON parse as one plain object;
5. output safety gate;
6. assistant envelope JSON schema validation;
7. config schema and allowed catalog validation;
8. artifact period/indicator checks;
9. `/backtests` preflight validation-only check;
10. one repair attempt when the failure is repairable;
11. terminal state.

Unsafe text, HTML/links, private path/secret leakage, prompt injection, and
`auto_run_backtest_attempt` are blocked before a loadable config can exist.

## Repair Contract

Repair attempts are fixed at `repair_attempts: 1`.

The repair attempt:

- uses the same backend adapter/runtime as generation;
- sends a dedicated repair prompt;
- includes only previous JSON draft, validation errors, and compact trusted context;
- does not start or enqueue core backtest jobs;
- does not silently convert unsupported symbols/indicators into another supported config.

Unsupported symbols, unsupported indicators, unavailable artifact coverage, and
no-window indicators that are not loadable through the current execution contract return
`needs_clarification` or `unsupported_request` without `load_action.enabled`.

## Indicator Audit

Current default audit result: `32/40` preflight-valid.

The 8 excluded/hidden model-facing indicators are:

- `momentum.stoch`;
- `structure.candle_stats`;
- `structure.candle_stats_atr_norm`;
- `structure.pivots`;
- `trend.psar`;
- `volatility.tr`;
- `volume.ad_line`;
- `volume.obv`.

Reason: `unsupported_window_axis`. They are hidden from `TRUSTED_CONTEXT_JSON.allowed_values.indicators`
until the no-window runtime/form/prepare-pools contract is loadable end to end.

## Contract Impact

| Surface | Classification | Notes |
| --- | --- | --- |
| Public API contract | compatible-change | `load_action` remains present; backend now returns state/reason/config from backend readiness instead of text inference. |
| Port contract | compatible-change | `BacktestConfigAgentGateway` adds `run_repair_config_session` for one same-runtime repair attempt. |
| DTO schema | compatible-change | Conversation run status can expose `input_too_large` and `security_review`; load action config remains nullable. |
| Persisted schema | none | No migration or table shape change. |
| Config schema | none | `repair_attempts: 1` remains the configured runtime value. |
| Request hash/cache identity | none | Core `/backtests` preflight request normalization remains the identity authority. |
| Browser-visible behavior | compatible-change | Apply/load button remains backend-gated; no UI implementation in this iteration. |

