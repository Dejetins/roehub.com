# Backtest Adaptive Selector v1

Status: Milestone F / EPIC F1 foundation  
Scope: deterministic planning-time execution-profile selection for the shared backtest runtime  
Related documents:
- `docs/architecture/roadmap/backtest-runtime-acceleration-plan-v1.md`
- `docs/architecture/backtest/backtest-runtime-acceleration-benchmarks-v1.md`
- `docs/architecture/backtest/backtest-family-accelerators-v1.md`
- `docs/architecture/roadmap/backtest-engine-vnext-parity-corrective-plan-v2.md`

## Purpose

The adaptive selector is one explicit cost model that chooses the effective execution profile from
planning-time evidence already available before runtime execution starts.

It exists to keep automatic selection:

- deterministic,
- cheap on the hot path,
- reviewable through one typed policy surface,
- compatible with the existing shared runtime orchestration surface.

This document does not add a new public `POST /backtests` field.
The browser still does not choose `execution_profile_mode`.

Remaining parity closure authority does not live in this selector document.
`v2` umbrella master-plan requires planner/selector topology split so canonical no-risk parity
classification must not be silently reduced to hybrid rollout semantics.

## Policy modes

The selector exposes one typed rollout policy:

- `disabled`: automatic behavior stays on the existing conservative exact-only fallback.
- `shadow`: the selector computes a recommendation but the executed profile remains the exact-only
  fallback.
- `opt_in`: automatic selection still keeps exact execution, but internal requested hybrid
  overrides are now explicitly allowed for controlled live evaluation.
- `active`: the selector may execute the recommended profile when it is valid.

The policy lives in `backtest.execution_profiles.adaptive_selector` inside
`configs/<env>/backtest.yaml` and is startup-validated by
`src/trading/contexts/backtest/adapters/outbound/config/backtest_runtime_config.py`.

Milestone F1 keeps `dev`, `test`, and `prod` on `disabled`.
F2 may promote environments separately without redesigning the selector contract.

## F2 rollout state

Milestone F / EPIC F2 keeps rollout explicit on the startup-validated config surface in
`backtest.execution_profiles.adaptive_selector`.

Committed env defaults after F2:

| Env | Selector mode | `hybrid_conservative` candidate cap | `hybrid_family` candidate cap | Effect |
|---|---|---|---|---|
| `dev` | `active` | `active` | `shadow` | large runs may execute `hybrid_conservative`; pure `ma.` runs may still be recommended for `hybrid_family` without switching live execution |
| `test` | `shadow` | `active` | `shadow` | selector recommendations stay inspectable, executed profile remains exact fallback |
| `prod` | `shadow` | `active` | `shadow` | prod remains conservative by default and reversible through config |

The candidate cap is the narrow additive F2 extension that keeps `hybrid_family` narrower than
`hybrid_conservative` without redesigning the global selector contract.

The next explicit rollout phase after committed prod `shadow` is `opt_in`:

- `prod=opt_in` keeps automatic selection recommendation-only;
- but internal non-public requested `execution_profile_mode=hybrid_*` overrides now become
  explicitly sanctioned for controlled live evaluation;
- `active` still remains the later phase where selective defaulting may execute automatically.

This does not widen the public `POST /backtests` contract and does not let the browser choose
profiles.

## Deterministic evidence

The cost model may use only planning-time evidence already available before execution:

- `grid cardinality`
- estimated `stage_a` work
- estimated `stage_b` work
- estimated memory bytes
- `runtime mode` derived by the planner as `sync_inline` vs background-capable
- indicator-family / `plugin availability` evidence needed for `hybrid_family`

The selector must not:

- read benchmark fixture files on the request path,
- depend on wall-clock measurements,
- create side effects,
- bypass the shared exact scorer.

## Candidate rules

Requested internal `execution_profile_mode` overrides keep precedence over automatic selection.
When there is no explicit override, the selector applies this order:

1. conservative exact fallback from the existing execution-profile catalog,
2. policy-gated hybrid promotion when the cost model and rollout gates both allow it,
3. exact-only fallback when policy is `disabled` or evidence is ambiguous.

### Exact fallback

The conservative fallback still comes from the existing ordered execution-profile catalog:

- `exact_small`
- `exact_parallel`

Existing exact launch budgets remain authoritative for exact fallback and sync/background routing.

### Hybrid promotion

Hybrid candidates remain additive and reuse the same downstream runtime path:

- `hybrid_conservative`
- `hybrid_family`

F2 adds one more rule before a hybrid candidate becomes live:

- the env-level selector mode and the candidate-specific rollout cap are combined, and the more
  conservative mode wins.

`hybrid_conservative` may be recommended only when:

- the profile runtime gate is live,
- the profile launch budget allows the request,
- the cost model exceeds the configured threshold count.

`hybrid_family` may be recommended only when all of the following are true:

- the profile runtime gate is live,
- family-plugin routing is enabled for that profile,
- the request resolves to one deterministic indicator family,
- the current registry resolves one live family plugin for that family/profile pair,
- the cost model exceeds the stricter family thresholds.

If the request is mixed-family, unsupported, or the plugin is unavailable, `hybrid_family` is not
selected automatically.

## Shared runtime boundary

The adaptive selector does not create a second engine.

All selected profiles still flow through the same shared runtime orchestration:

- exact profiles stay on the canonical exact pipeline,
- `hybrid_conservative` stays on the shared shortlist runtime,
- `hybrid_family` stays on the shared shortlist runtime plus proposal-only family plugin layer,
- the exact Stage B scorer remains canonical.

For internal debug/testing only, the built runtime plan may now carry the selector decision
payload (`effective_profile` vs `recommended_profile`) so shadow recommendations are inspectable
without changing the public API.

## Benchmark caveat

`exact_baseline` remains a benchmark evidence anchor, not a runtime default.

Specifically:

- benchmark `exact_baseline` continues to map to `exact_parallel`,
- the active runtime default remains `exact_small`,
- Milestone F1 does not silently realign those two concepts.

That distinction stays explicit in docs, config, and tests so later rollout decisions remain
reviewable.
