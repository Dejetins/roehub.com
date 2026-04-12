# Backtest Engine vNext

Canonical redesign document for the `backtest-engine-vnext` shape after Milestone G / H closure aligned the active launch, detail, and worker contracts.

Status: canonical redesign baseline after Milestone G / H closure  
Implementation state: active `summary-only` launch, `on-demand` variant detail, and shared sync/worker runtime are aligned with the shipped v2 runtime/API/worker contracts  
Canonical experimental anchor: `tests/notebook_tests/new_engine/01_run_322_btcusdt_1h_artifact_probe.ipynb`

## 1. Role of this document

This document is the canonical redesign source of truth for the next backtest engine shape.

It exists to:

- move redesign planning onto the approved `01_run_322...` notebook anchor;
- describe the staged engine shape in implementation-ready terms for future follow-up work;
- freeze the agreed vocabulary for launch shape, internal exact payloads, and shared runtime boundaries;
- prevent later implementation prompts from mixing current production behavior with target redesign intent.

This document does not replace the shipped runtime/API/worker contracts as the source of active behavior.

Current shipped behavior remains governed by:

- [Backtest Runtime Kernels V2](/Users/daniildegtyarev/Projects/roehub.com/docs/architecture/backtest/backtest-runtime-kernels-v2.md)
- [Backtest API v1 — `POST /backtests`](/Users/daniildegtyarev/Projects/roehub.com/docs/architecture/backtest/backtest-api-post-backtests-v1.md)
- [Backtest Job Runner v2 -- canonical claimed background worker for persisted runs](/Users/daniildegtyarev/Projects/roehub.com/docs/architecture/backtest/backtest-job-runner-v2.md)

If this document and an active runtime/API/worker contract appear to conflict, the current shipped behavior wins until a later cutover prompt updates the active contract explicitly.

## 2. Canonical redesign anchor

The only canonical experimental anchor for the redesign is:

- `tests/notebook_tests/new_engine/01_run_322_btcusdt_1h_artifact_probe.ipynb`

That notebook is the approved source for the universal engine patterns that the redesign carries forward:

- `trade-list-first design`
- `prefilter before exact path`
- `hit-time tables`
- `fast monotone TP/SL kernel`
- `reference-vs-fast self-check`

The historical notebook `tests/notebook_tests/06_backtest_compute.ipynb` may still be cited only as historical background. It is not the canonical redesign anchor for new prompts, new handoff docs, or future cutover work.

## 2.1 Notebook parity benchmark authority

The redesign now has one frozen benchmark authority for notebook parity closure:

- committed corpus:
  `tests/perf_smoke/contexts/backtest/fixtures/backtest_notebook_parity_benchmark_corpus_v1.json`
- executable harness:
  `tests/perf_smoke/contexts/backtest/test_backtest_notebook_parity_perf_smoke_v1.py`

Canonical classes:

- `NR2`
- `RG-TTR`
- `RG-ALT`

Required runtime-shape fields for benchmark evidence:

- `wall_clock_seconds`
- `cpu_time_seconds`
- `peak_rss_bytes`
- `numba_threads_used`
- `max_python_processes_seen`
- `stage_b_execution_mode`
- `stage_b_process_fallback_threshold`
- `exact_replay_count`

Normalization rules:

- every backend-vs-notebook comparison uses `equal thread budget`
- `numba_threads_used` must match exactly
- host and artifact slot must match exactly
- sync and worker measurements are comparable only when the same host, artifact slot,
  `stage_b_execution_mode`, and `stage_b_process_fallback_threshold` are preserved

Accepted runtime-shape answer:

- `NR2`: no-risk runs resolve on the Stage A no-risk terminal path,
  `stage_b_execution_mode = bypassed_no_risk`, `single-process default`,
  `wall_clock_ratio <= 1.18`, `peak_rss_ratio <= 1.35`
- `RG-TTR`: risk-grid runs keep Stage B `in_process` as the `single-process default`,
  preserve `finalist-only exact replay` with `exact_replay_count <= 64`,
  and hold `wall_clock_ratio <= 1.18`
- `RG-ALT`: alternative ranking metrics remain correctness-first and must not regress runtime by
  more than `10%`

This benchmark surface is internal-only, but it is no longer only additive: it is the blocking
review surface for notebook parity. Maintainers should treat
`tests/perf_smoke/contexts/backtest/test_backtest_notebook_parity_perf_smoke_v1.py`
as the acceptance authority, and rollout remains incomplete whenever its frozen gates or captured
live benchmark measurements fail.

## 3. Target architecture summary

The target engine keeps one deterministic staged runtime with one final exact authority.

Target shape:

1. Stage A performs row prefilter and combo proxy narrowing on the request timeframe instead of
   exact-first breadth scoring across the full grid.
2. Stage A builds compact trade candidates from signal-timeline decisions, and no-risk classes
   finalize here through the no-risk terminal path without entering generic Stage B.
3. Stage B expands shortlisted risk-grid candidates across the published TP/SL grid using cheap
   artifact-backed kernels, with finalist-only exact replay for the retained winners.
4. The exact scorer remains the final authority for winners, persisted summary ordering, and on-demand detail.

Non-negotiable redesign rules:

- Sync launch and claimed background execution must use one shared engine path.
- Heuristic or approximate layers may narrow the candidate space, but they must not replace the exact Stage B authority.
- The redesign remains generic for N-indicator search and must not become pair-specific notebook orchestration.
- Public launch remains `summary-only`.
- Full trades remain on-demand only for an explicitly requested variant.

## 4. Canonical vocabulary for vNext

| Term | Canonical meaning in vNext |
|---|---|
| `signal timeline` | The request timeframe timeline used to aggregate signals and define strategy state changes. |
| `execution timeline` | The canonical `1m` execution timeline used for exact risk exit resolution. |
| `compact trade representation` | Internal exact payload derived from staged signal decisions and used by the exact engine. It is not a default user-facing trades output. |
| `trade-list-first design` | Engine pattern where staged trade candidates are built before exhaustive risk replay. |
| `prefilter before exact path` | Required narrowing step before bounded exact evaluation. |
| `summary-only` | Launch, persistence, and top-row flows return summary payloads only; reports and full trades are not eagerly materialized. |
| `trades on demand` | Variant-level report/trades materialization happens only through explicit detail/report requests after a run exists. |

## 5. Current -> Target replacement matrix

| Surface | Current active contract | Target vNext direction |
|---|---|---|
| Redesign anchor | Active runtime docs still describe shipped semantics through current v2 contracts, with older notebook history still present in the doc set | `01_run_322...` becomes the canonical redesign anchor for new engine prompts and architecture reasoning |
| Stage A hot path | Current production runtime preserves the shipped artifact-backed Stage A boundary and exact production behavior | Stage A stops being the exact-first breadth authority for the full grid and becomes a staged narrowing layer |
| Search flow | Current system already uses exact Stage B over artifact-backed inputs | vNext makes `prefilter before exact path` the canonical engine shape |
| Trade intermediate form | Current runtime uses compact trade concepts inside the v2 kernels | vNext makes `compact trade representation` an explicit internal exact payload in the redesign vocabulary |
| Launch result shape | Active launch and persisted top rows are `summary-only` | vNext keeps launch `summary-only` and makes this an explicit product rule, not an accidental optimization |
| Trade details | Full detail/trades are loaded through explicit detail/report flows | vNext keeps full trades on-demand only and rejects eager top-row trade materialization |
| Sync vs worker execution | Active sync and claimed background execution already share the artifact-backed runtime family | vNext keeps one shared planner and one shared engine for sync and job-runner execution |
| Ranking knobs | Active public surface keeps only `primary_metric` with deterministic tie-break | vNext keeps `primary_metric` as the steady-state public ranking selector |
| Warmup input | Active public contract derives warmup internally from effective indicator requirements | vNext keeps warmup internal to runtime planning and execution |
| Trade eager detail knob | Active public launch/detail docs keep full trades on-demand only and do not expose `top_trades_n` | vNext keeps `top_trades_n` out of the public launch/detail surface because launch remains `summary-only` |

## 6. Public and runtime decisions fixed by this redesign

### 6.1 Launch remains `summary-only`

The target launch shape does not change into a full-report launch.

Rules:

- sync launch remains `summary-only`;
- background execution remains `summary-only`;
- persisted top rows remain `summary-only`;
- full report bodies and full trades are materialized only via explicit variant-level detail/report operations.

This means `trade-list-first design` is an internal engine architecture choice, not a promise that default launch responses include trade logs.

### 6.2 `compact trade representation` is internal only

The engine may build and pass a compact trade representation through the exact path as an internal deterministic payload.

It must be treated as:

- internal exact engine state;
- compatible with bounded exact replay and exact metrics computation;
- distinct from any user-facing full trades list or report body.

It must not be described as the default launch output.

### 6.3 Shared sync and worker engine contract

The target architecture keeps one shared runtime stack for:

- sync launch;
- claimed background execution through `backtest-job-runner`;
- variant-level on-demand detail/report generation.

Implications:

- no separate scoring engine for the worker;
- no separate planner policy for background runs;
- no separate correctness authority for detail versus launch;
- one exact Stage B scorer remains the final authority in every execution mode.

### 6.4 Public launch vocabulary reset

The target public surface keeps:

- `direction_mode`
- `sizing_mode`
- `primary_metric`
- `top_k` / top-N launch semantics
- ordinary execution and risk settings

The target public surface removes:

- `secondary_metric`
- `warmup_bars`
- `top_trades_n`

These fields are not part of the target public surface.

Rules behind that reset:

- `primary_metric` remains the single user-facing ranking selector;
- deterministic tie-break remains mandatory;
- `warmup_bars` becomes an internal derived value only;
- `top_trades_n` disappears because launch remains `summary-only` and full trades are on-demand only.

## 7. Target staged engine shape

### 7.1 Stage A: bounded narrowing

Stage A in vNext is responsible for:

- reading request-timeframe signal data from artifacts;
- building strategy state on the `signal timeline`;
- producing internal trade candidates through `trade-list-first design`;
- computing cheap narrowing signals before the exact path;
- sending only shortlisted candidates into the bounded exact path.

Stage A in vNext is not allowed to become:

- an exact-first breadth scorer for the entire grid;
- a separate authority that can overrule the final exact scorer;
- a pair-specific orchestration path coupled to one historical notebook shape.

### 7.2 Stage B: exact authority

Stage B in vNext is responsible for:

- using `hit-time tables` on the `1m` execution timeline;
- evaluating shortlisted candidates on the published TP/SL artifact grid;
- using a `fast monotone TP/SL kernel` for exact grid search;
- preserving a `reference-vs-fast self-check` discipline for correctness-sensitive work;
- producing the final exact ranking and exact summary metrics.

The redesign explicitly keeps exact Stage B as the source of truth.

## 8. Dependencies and rollout assumptions

The following dependencies are part of the target architecture even though this prompt does not implement them:

- Wider TP/SL artifact grids are a required dependency for full rollout. The redesign should rely on canonical published grids rather than ad hoc runtime scans for the production exact path.
- `signal_features` remain optional in the first redesign wave. They may act as an accelerator surface later, but the first vNext rollout must not depend on them as a mandatory public or artifact contract.

ASSUMPTION: later implementation prompts will update the active runtime/API/worker docs only when the underlying runtime code and public contract changes are ready to ship together.

## 9. Historical scope boundaries for Milestone A / EPIC A1

Milestone A introduced only the documentation foundation for the redesign.

It does not:

- change runtime code;
- change DTOs, persisted payloads, or UI behavior;
- change the active `POST /backtests` contract yet;
- update the old notebook-derived implementation;
- add new user-facing filters or new launch fields;
- cut sync or worker execution over to a new engine path in code.

## 10. Related documents

- Current runtime contract: [Backtest Runtime Kernels V2](/Users/daniildegtyarev/Projects/roehub.com/docs/architecture/backtest/backtest-runtime-kernels-v2.md)
- Current sync launch contract: [Backtest API v1 — `POST /backtests`](/Users/daniildegtyarev/Projects/roehub.com/docs/architecture/backtest/backtest-api-post-backtests-v1.md)
- Current claimed worker contract: [Backtest Job Runner v2 -- canonical claimed background worker for persisted runs](/Users/daniildegtyarev/Projects/roehub.com/docs/architecture/backtest/backtest-job-runner-v2.md)
- Approved implementation roadmap: [План переустройства backtest engine vNext](/Users/daniildegtyarev/Projects/roehub.com/docs/architecture/roadmap/backtest-engine-vnext-implementation-plan-v1.md)
