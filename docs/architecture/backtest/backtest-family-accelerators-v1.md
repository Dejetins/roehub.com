# Backtest Family Accelerators v1

Документ фиксирует foundation Milestone E (`EPIC E1 + E2`) для proposal-only family accelerators
в `hybrid_family`.

## Status

- Status: proposal-only foundation; no concrete family plugin is shipped in this document.
- Scope:
  - typed contracts for future family accelerators,
  - deterministic registry selection,
  - per-run circuit breaker and warning semantics,
  - additive execution-profile budget surface.
- Explicitly out of scope:
  - live `hybrid_family` routing in `RunBacktestUseCase`,
  - public selector changes in `POST /backtests`,
  - any family-specific backtest engine,
  - any concrete MA-family or other plugin implementation.

## Files

- `src/trading/contexts/backtest/application/services/v2/family_plugins/contracts_v2.py`
- `src/trading/contexts/backtest/application/services/v2/family_plugins/registry_v2.py`
- `src/trading/contexts/backtest/application/services/v2/family_plugins/circuit_breaker_v2.py`
- `src/trading/contexts/backtest/application/services/v2/execution_profile_v2.py`
- `src/trading/contexts/backtest/adapters/outbound/config/backtest_runtime_config.py`
- `apps/api/dto/backtest_runtime_defaults.py`

## Core rule

Family accelerators are a `proposal` layer only.

They may suggest:

- `row shortlist`
- `pair shortlist`
- `proxy score`

They may not:

- replace the shared exact Stage B scorer,
- introduce a family-specific runtime pipeline,
- change winner semantics,
- bypass the universal conservative fallback path.

The exact scorer remains the canonical source of truth on retained survivors.

## Planning context

The plugin context is intentionally narrow:

- it references the already prepared `BacktestArtifactRuntimePlanV2`,
- it carries optional explicit requested profile metadata,
- it exposes normalized indicator ids,
- it exposes one typed `family_plugin_budget_ms`.

No second planning stack is introduced.

## Deterministic selection

Registry lookup depends only on startup-validated stable inputs:

- resolved execution profile mode,
- `family_plugin_enabled`,
- deterministic indicator-family literal derived from runtime-plan indicator ids.

Current family resolution rule is explicit:

- if all runtime-plan indicator ids share the same prefix before the first `.`, that prefix is the
  candidate family literal,
- otherwise the request is considered mixed-family and remains on the universal path.

This keeps family selection internal and reviewable without adding public API metadata.

## Failure handling

Failure handling is explicit and reusable:

- timeout uses `family_plugin_budget_ms`, which must stay `<= planning_budget_ms`,
- timeout -> `warning + universal fallback`,
- error -> `warning + universal fallback`,
- open breaker -> `warning + universal fallback`,
- repeated failures open a per-run `circuit breaker`,
- once open, the breaker stays open for the rest of that run.

The breaker is run-local only. No cross-run cache or persistence is introduced.

## Runtime/profile surface

`ExecutionProfileV2` now publishes:

- `planning_budget_ms`
- `family_plugin_budget_ms`

`family_plugin_budget_ms` is additive discovery/config surface only in this milestone.

Runtime defaults now expose the same field in `contracts.execution.available_execution_profiles[]`
so browser/debug tooling sees the same typed profile contract used by config and planner layers.

## Why no concrete plugin yet

Milestone E ships only the foundation because the first concrete plugin needs separate rollout
evidence and benchmark gates.

The next plugin candidate may be MA-family, but it must still:

- plug into the shared registry,
- return proposal-only output,
- respect `family_plugin_budget_ms`,
- degrade to the universal path on timeout/error/open breaker.
