# Backtest Family Accelerators v1

Документ фиксирует foundation Milestone E (`EPIC E1 + E2`) и первый concrete `MA-family`
plugin (`EPIC E3`) для proposal-only family accelerators в `hybrid_family`.

## Status

- Status: proposal-only foundation plus the first shipped `MA-family` plugin.
- Scope:
  - typed contracts for future family accelerators,
  - deterministic registry selection,
  - per-run circuit breaker and warning semantics,
  - additive execution-profile budget surface,
  - the first concrete `ma.` plugin wired through the shared runtime path.
- Explicitly out of scope:
  - public selector changes in `POST /backtests`,
  - any family-specific backtest engine,
  - adaptive selector routing,
  - non-MA plugins.

## Files

- `src/trading/contexts/backtest/application/services/v2/family_plugins/contracts_v2.py`
- `src/trading/contexts/backtest/application/services/v2/family_plugins/registry_v2.py`
- `src/trading/contexts/backtest/application/services/v2/family_plugins/circuit_breaker_v2.py`
- `src/trading/contexts/backtest/application/services/v2/family_plugins/ma_family_plugin_v2.py`
- `src/trading/contexts/backtest/application/services/v2/hierarchical_shortlist_builder_v2.py`
- `src/trading/contexts/backtest/application/use_cases/run_backtest.py`
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

## First plugin: MA-family

The first shipped plugin is `MA-family`.

Implementation facts:

- plugin id: `ma.family.v1`
- registration: shared `FamilyPluginRegistryV2`
- execution profile: internal-only `hybrid_family`
- applicability: only pure canonical `ma.` indicator sets from
  `src/trading/contexts/indicators/domain/definitions/ma.py`
- proposal shapes used in v1:
  - `row shortlist`
  - `proxy score`

Heuristic shape:

- the plugin reads only shared runtime-plan metadata;
- it samples deterministic `window` anchors per MA indicator;
- it keeps `source` deterministic instead of creating a second runtime stack;
- it expands retained compute anchors back into exact Stage A indexes;
- it remains a `proposal layer` and never bypasses the exact scorer.

This is intentionally narrow.
It is not a special backtest engine.
It is not a per-family runtime fork.

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

Additional E3 rule:

- canonical `ma.` definitions are the source of truth for the first plugin;
- unknown `ma.` ids are not treated as implicitly supported.

## Failure handling

Failure handling is explicit and reusable:

- timeout uses `family_plugin_budget_ms`, which must stay `<= planning_budget_ms`,
- timeout -> `warning + universal fallback`,
- error -> `warning + universal fallback`,
- open breaker -> `warning + universal fallback`,
- missing plugin -> `warning + universal fallback`,
- mixed-family / registry non-applicability -> `warning + universal fallback`,
- repeated failures open a per-run `circuit breaker`,
- once open, the breaker stays open for the rest of that run.

The breaker is run-local only. No cross-run cache or persistence is introduced.

`hybrid_family` reuses the same shared hierarchical shortlist runtime as `hybrid_conservative`.

That means:

- successful plugin proposals become reduced exact runtime plans,
- fallback still uses the existing universal conservative shortlist behavior,
- exact Stage B remains canonical in every path.

## Runtime/profile surface

`ExecutionProfileV2` now publishes:

- `planning_budget_ms`
- `family_plugin_budget_ms`

`family_plugin_budget_ms` is the typed plugin budget for the proposal layer.

Runtime defaults now expose the same field in `contracts.execution.available_execution_profiles[]`
so browser/debug tooling sees the same typed profile contract used by config and planner layers.

Rollout remains conservative:

- `dev`: `hybrid_family` runtime gates are enabled, but adaptive-selector rollout for the
  candidate remains `shadow`;
- `test`: `hybrid_family` runtime gates are enabled for internal coverage and perf-smoke
  evidence, while selector rollout remains `shadow`;
- `prod`: `hybrid_family` runtime gates are enabled for internal/manual evaluation and shadow
  recommendations, but committed selector rollout still keeps it non-live by default.

There is still no public `POST /backtests` selector for `hybrid_family`.
The live path stays internal-only through the existing requested-profile override or shadow
recommendation/debug surfaces.

## Warning/debug surface

Reduced hybrid plans now carry compact internal debug metadata for reviewability:

- proposal-layer source (`universal` vs `family_plugin`),
- registry resolution status for `hybrid_family`,
- optional plugin warning payload,
- optional successful plugin proposal payload.

This keeps warning behavior explicit without widening the public request contract.

## Future work

Future plugins must still:

- plug into the shared registry,
- return proposal-only output,
- respect `family_plugin_budget_ms`,
- degrade to the universal path on timeout/error/open breaker.
