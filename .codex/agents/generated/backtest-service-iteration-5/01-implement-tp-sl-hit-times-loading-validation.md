---
prompt_name: backtest_service_iteration_5_tp_sl_hit_times_loading_validation
repo: roehub.com
branch: current
scope: "Iteration 5: implement artifact-backed TP/SL hit-times loading and request grid validation before TP/SL exact scoring."

language:
  implementation: python
  agent_report: ru

context_sources:
  always_read:
    - path: .codex/AGENTS.md
      why: "repo engineering contract and safety invariants"
    - path: docs/architecture/backtest/backtest-service-artifact-runtime-v1.ru.md
      why: "Iteration 5 target contract and benchmark gate"
    - path: docs/architecture/backtest/benchmark_iterations/README.md
      why: "benchmark evidence manifest and stage naming rules"
    - path: docs/architecture/backtest/benchmark_iterations/2026-04-26_engine_test_btcusdt_15m/benchmark_results.json
      why: "canonical risk-on target values for load_hit_times and tp_sl_grid_validation"
  task_entrypoints:
    - path: src/trading/contexts/backtest/application/services/v2/benchmark_accounting.py
      why: "canonical/service-only stage accounting and risk-on stage order"
      inspect_symbols:
        - CANONICAL_STAGE_ORDER
        - canonical_required_stages
    - path: src/trading/contexts/backtest/application/services/v2/prepare_pools.py
      why: "existing artifact context, mmap loading, request slice, and timing patterns"
      inspect_symbols:
        - BacktestPreparePoolsService
        - BacktestPreparePoolsRuntimeArrays
    - path: src/trading/contexts/backtest/application/services/v2/preflight.py
      why: "existing public request normalization and config-level TP/SL validation semantics"
      inspect_symbols:
        - BACKTEST_ERROR_TP_SL_GRID_NOT_COVERED
        - BacktestPreflightService
    - path: src/trading/contexts/backtest_artifacts/application/services/v2/contracts.py
      why: "typed hit_times/15m artifact contracts, filenames, dtypes, and existing slice DTOs"
      inspect_symbols:
        - HIT_TIMES_TIMEFRAME_LITERAL_V2
        - ArtifactHitTimesArraysV2
        - StageBHitTimesSliceV2
        - ArtifactHitTimesManifestDocumentV2
  conditional_bundles:
    artifact_loader_adapter:
      read_when: "adding or using hit-times loading through an artifact array loader port"
      paths:
        - src/trading/contexts/backtest/application/ports/artifact_arrays.py
        - src/trading/contexts/backtest_artifacts/application/services/v2/artifact_manifest_loader.py
        - src/trading/contexts/backtest_artifacts/adapters/outbound/artifacts_fs/path_builder.py
    previous_iteration_patterns:
      read_when: "building the Iteration 5 benchmark runner or summary writer"
      paths:
        - scripts/backtest/run_iteration_4_2_exact_scoring_benchmark.py
        - scripts/backtest/run_iteration_4_3_heap_update_benchmark.py
        - scripts/backtest/run_iteration_4_4_top_result_proxy_fill_benchmark.py
        - scripts/backtest/run_iteration_4_7_memory_cleanup_smoke.py
    unit_test_patterns:
      read_when: "adding service/DTO tests for hit-times grid validation and cleanup"
      paths:
        - tests/unit/contexts/backtest/application/services/v2/artifact_testkit_v2.py
        - tests/unit/contexts/backtest/application/services/v2/test_backtest_preflight_service.py
        - tests/unit/contexts/backtest/application/services/v2/test_artifact_manifest_validator_v2.py
  consult_if_needed:
    - path: docs/architecture/backtest/benchmark_iterations/2026-05-01_iteration_4_2_exact_scoring_self_check/benchmark_summary.md
      read_when: "copying accepted Mac Studio benchmark evidence style"
    - path: docs/architecture/backtest/benchmark_iterations/2026-05-01_iteration_4_7_memory_cleanup/benchmark_summary.md
      read_when: "copying service hygiene cleanup evidence style"

style_references:
  - docs/architecture/backtest/benchmark_iterations/README.md
  - docs/architecture/backtest/benchmark_iterations/2026-05-01_iteration_4_2_exact_scoring_self_check/benchmark_summary.md

hard_requirements:
  hit_times_path_is_15m: true
  compare_only_notebook_compatible_stages: true
  preserve_no_risk_acceptance: true
  deterministic_grid_not_covered_422: true
  cleanup_heavy_arrays_on_failure: true
  macstudio_acceptance_required: true
  max_implementation_attempts: 2

task_toggles:
  implement_hit_times_load: true
  implement_tp_sl_grid_validation: true
  implement_tp_sl_exact_scoring: false
  implement_persistence_or_public_api: false
  update_benchmark_manifest_if_needed: true

skill_routing:
  - skill: backend-performance-evidence
    use_when: "defining or reporting Iteration 5 benchmark comparison against canonical notebook stages"
    timing: before implementation and during verification
    reason: "load_hit_times and tp_sl_grid_validation are benchmark-gated stages"
  - skill: contract-impact-analysis
    use_when: "changing DTOs, application ports, benchmark JSON schema, or deterministic error payloads"
    timing: before implementation
    reason: "Iteration 5 touches request validation, artifact loading boundaries, and error contracts"
  - skill: backend-quality-gates
    use_when: "running targeted lint/type/test gates"
    timing: during verification
    reason: "Roehub backend gates are uv-based"

target_envs:
  - local-dev
  - macstudio

required_literals:
  - "hit_times/15m"
  - "load_hit_times"
  - "tp_sl_grid_validation"
  - "backtest.tp_sl_grid_not_covered"
  - "2.0..25.0"
  - "step 0.5"
  - "historical_prefix_compatible"

non_goals:
  - "Do not implement `tp_sl_exact_scoring`; that belongs to Iteration 6."
  - "Do not implement job persistence, public variant identity, or lazy trades; those belong to later iterations."
  - "Do not change no-risk Iteration 4 scoring behavior except where shared types/imports require compatible updates."
  - "Do not use legacy `hit_times/1m` path or old execution-profile vocabulary."

final_report_format:
  language: ru
  sections:
    - "Что сделано"
    - "Stage contract"
    - "Benchmark / Mac Studio"
    - "Проверки"
    - "Contract impact"
    - "Ограничения / следующий шаг"

quality_gates:
  - cmd: "uv run ruff check src/trading/contexts/backtest tests/unit/contexts/backtest scripts/backtest"
    expect: "passes, or a narrower justified target passes if unrelated existing files fail"
  - cmd: "uv run pyright"
    expect: "passes"
  - cmd: "uv run pytest -q tests/unit/contexts/backtest/application/services/v2"
    expect: "passes or targeted subset with unrelated failures isolated"
  - cmd: "uv run python -m tools.docs.generate_docs_index --check"
    expect: "passes if docs or benchmark summaries change"

expected_primary_touches:
  - "src/trading/contexts/backtest/application/dto/<new tp/sl hit-times dto>.py"
  - "src/trading/contexts/backtest/application/services/v2/<new tp/sl hit-times service>.py"
  - "src/trading/contexts/backtest/application/ports/artifact_arrays.py"
  - "tests/unit/contexts/backtest/application/services/v2/<new hit-times service tests>.py"
  - "scripts/backtest/run_iteration_5_tp_sl_hit_times_benchmark.py"
  - "docs/architecture/backtest/benchmark_iterations/YYYY-MM-DD_iteration_5_tp_sl_hit_times_loading_validation/"

possible_secondary_touches:
  - "src/trading/contexts/backtest/application/dto/__init__.py"
  - "src/trading/contexts/backtest/application/services/v2/__init__.py"
  - "src/trading/contexts/backtest/application/services/v2/benchmark_accounting.py"
  - "tests/unit/contexts/backtest/application/services/v2/artifact_testkit_v2.py"
  - "docs/architecture/backtest/benchmark_iterations/README.md"

safety_notes:
  - "The current live contract is `hit_times/15m`; do not revive old `hit_times/1m` runtime wording."
  - "Config-level preflight coverage is not enough; Iteration 5 must validate the actual artifact grid values and manifest-backed arrays."
  - "Keep `load_hit_times` and `tp_sl_grid_validation` as separate measured stages."
  - "The executor has only 2 implementation attempts; after the second failed corrective cycle, stop and report the blocker with evidence."
---

# Task

Implement Iteration 5: artifact-backed TP/SL grid validation and hit-times subset loading for risk-on backtests.

Done means:

- service code can validate a requested TP/SL grid against the actual published `hit_times/15m` artifact grid;
- service code can load and materialize the requested subset of `long_tp`, `long_sl`, `short_tp`, and `short_sl` hit-time tables into bounded contiguous arrays for Iteration 6 kernels;
- stage telemetry records `tp_sl_grid_validation` separately from `load_hit_times`;
- deterministic grid coverage failures produce `422 backtest.tp_sl_grid_not_covered` semantics before scoring;
- `hit_times_manifest_hash` is recorded in the result/evidence;
- benchmark evidence is written under `docs/architecture/backtest/benchmark_iterations/<date>_iteration_5_tp_sl_hit_times_loading_validation/`;
- no TP/SL exact scoring is implemented in this iteration.

## Context / Current State

Completed service iterations:

- Iteration 1: request normalization, preflight, artifact context.
- Iteration 2: artifact arrays, slicing, `prepare_pools_core`.
- Iteration 3: combo planning contexts.
- Iteration 4: no-risk exact scoring and top-K path, including memory cleanup smoke.

Iteration 5 starts risk-on support but stops before scoring. It owns only:

- `tp_sl_grid_validation`;
- `load_hit_times`;
- hit-times manifest hash evidence;
- success/failure grid coverage evidence;
- cleanup of per-job hit-times arrays when validation/load fails.

The canonical benchmark target uses:

- runtime target path: `hit_times/15m`;
- TP/SL grid: `2.0..25.0` inclusive, step `0.5`;
- risk mode: `tp_sl_grid`;
- arity 1..7 and both direction modes in the canonical JSON;
- `historical_prefix_compatible` artifact policy when full manifest hash differs but the request slice/grid is compatible.

## Requirements (Must)

- Use `hit_times/15m`, never legacy `hit_times/1m` paths.
- Validate against actual artifact `tp_values.f32.npy` and `sl_values.f32.npy`, not only runtime config defaults.
- Interpret request TP/SL values as human percentages and map them to decimal artifact levels with bounded float tolerance.
- A requested TP or SL value must match exactly one artifact level after tolerance; zero matches or multiple matches must fail deterministically.
- For grid-not-covered failure, return or raise through the existing error path with code `backtest.tp_sl_grid_not_covered`.
- Load only the requested subset:
  - `long_tp.u32.npy`;
  - `long_sl.u32.npy`;
  - `short_tp.u32.npy`;
  - `short_sl.u32.npy`.
- Materialized subset arrays must be contiguous and scoped to one job/request.
- Record `load_hit_times` timing separately from `tp_sl_grid_validation`.
- Record `hit_times_manifest_hash` and enough grid evidence to prove target grid `2.0..25.0 step 0.5` is covered.
- Preserve benchmark accounting rules from Iteration 4.6: service-only telemetry must not be compared as a canonical stage.
- Add focused unit tests for:
  - successful subset selection;
  - missing TP level;
  - missing SL level;
  - duplicate/ambiguous tolerance match if representable with a small fixture;
  - manifest hash propagation;
  - cleanup/compact result behavior on failed validation or failed load.
- Add a Mac Studio benchmark runner for Iteration 5.

## Requirements (Should)

- Reuse existing typed contracts from `backtest_artifacts.application.services.v2.contracts` where they fit, especially hit-times manifest/array/slice types.
- Prefer adding a small application service and DTOs rather than growing `prepare_pools.py`.
- Keep the application port boundary explicit. If `BacktestArtifactArrayLoader` is extended with hit-times methods, classify the port change as compatible additive.
- Keep benchmark JSON machine-readable and similar to accepted Iteration 4 records:
  - `schema`;
  - `host`;
  - `git_commit`;
  - `artifact_manifest_hash`;
  - `hit_times_manifest_hash`;
  - `artifact_policy`;
  - `stage_pass`;
  - `pass`;
  - per-run stage timings and ratios.

## Requirements (Nice-to-have)

- Provide a small helper for generating the Markdown summary from benchmark JSON.
- Include service-only RSS evidence for validation/load failure paths if cheap to measure.

# Context Acquisition Protocol

Read only in this order and stop once sufficient:

1. `.codex/AGENTS.md`
2. `docs/architecture/backtest/backtest-service-artifact-runtime-v1.ru.md`
3. `docs/architecture/backtest/benchmark_iterations/README.md`
4. canonical benchmark JSON
5. task entrypoints
6. only conditional bundles required by touched contracts or failing gates

Pre-implementation reading target:

- `<= 8 files`
- `<= ~35k-50k tokens`

Stop reading once:

- DTO/service/port touch points are bounded;
- validation semantics are unambiguous;
- benchmark stage boundaries are clear;
- no unresolved public API or persistence ambiguity remains.

Expand context only for:

- artifact loader adapter ambiguity;
- failing quality gates;
- benchmark threshold conflicts;
- unclear error mapping.

# Reading Manifest

Use the front-matter `context_sources` as the canonical reading map.

Do not broad-read all backtest artifact publisher code unless the artifact loader contract is unclear.

# Work Plan (Agent Should Follow)

Skill routing:

- Use `backend-performance-evidence` before implementation to define comparable stage boundaries and during verification to report benchmark evidence.
- Use `contract-impact-analysis` before changing DTOs, ports, benchmark JSON, or error payload behavior.
- Use `backend-quality-gates` during verification.

Implementation steps:

1. Locate the cleanest application boundary for TP/SL hit-times validation/loading.
2. Define compact DTOs for:
   - requested TP/SL levels;
   - resolved TP/SL indexes;
   - materialized hit-times subset arrays;
   - timing/telemetry;
   - manifest hash and grid evidence.
3. Implement `tp_sl_grid_validation`:
   - human-percent request values to decimal levels;
   - tolerance-bounded exact-one matching;
   - deterministic failure details;
   - target grid evidence.
4. Implement `load_hit_times`:
   - manifest hash read/propagation;
   - mmap/open actual artifact arrays;
   - selected row copy into contiguous arrays;
   - dtype/shape/sentinel validation at service boundary.
5. Add cleanup/release boundary so failed validation/load does not leave heavy arrays retained through result DTOs.
6. Add focused unit tests.
7. Add `scripts/backtest/run_iteration_5_tp_sl_hit_times_benchmark.py` following accepted Iteration 4 evidence style.
8. Run local gates.
9. Publish/deploy/sync to Mac Studio and run benchmark acceptance.

Mac Studio benchmark pipeline:

1. Local code passes focused gates.
2. Commit and push the implementation branch/commit.
3. Ensure `/opt/roehub/app` on Mac Studio contains the pushed code. If it is a runtime copy rather than a git checkout, record that explicitly.
4. Run the Iteration 5 benchmark command on Mac Studio.
5. Write JSON and Markdown evidence under:

```text
docs/architecture/backtest/benchmark_iterations/YYYY-MM-DD_iteration_5_tp_sl_hit_times_loading_validation/
```

6. Compare `load_hit_times` and `tp_sl_grid_validation` only against canonical notebook-compatible targets.
7. Record request grid coverage success and deterministic failure evidence.
8. Run docs index check/regeneration if benchmark Markdown is added.

# Acceptance Criteria (Definition of Done)

- Local service tests pass.
- No-risk Iteration 4 tests are not regressed.
- Mac Studio benchmark evidence exists.
- `load_hit_times` passes 90% target comparison for risk-on canonical rows.
- `tp_sl_grid_validation` passes 90% target comparison for risk-on canonical rows.
- Failure evidence includes at least one missing TP or SL level producing `backtest.tp_sl_grid_not_covered`.
- Evidence records target grid `2.0..25.0` inclusive, step `0.5`.
- Evidence records `hit_times_manifest_hash`.
- Cleanup evidence shows failed validation/load does not retain heavy hit-time arrays through returned DTOs or telemetry.
- Final report states whether any artifact full-hash mismatch was accepted only under `historical_prefix_compatible`.

# Implementation Constraints

## Stage Boundaries

- `tp_sl_grid_validation`: grid request parsing, percent-to-decimal conversion, level lookup, coverage success/failure.
- `load_hit_times`: manifest/hash work, mmap/open, dtype/shape validation, selected subset copy, fee-adjusted TP/SL outcome precompute only if already required by the Iteration 5 plan.
- Do not include TP/SL exact scoring, heap update, result assembly, persistence, or lazy trades in Iteration 5 timings.

## Contracts

- Public API: no new public endpoint work unless needed only for error mapping compatibility.
- DTO schema: additive internal DTOs are allowed.
- Application port: additive hit-times methods are allowed if needed; classify as compatible-change.
- Persisted schema: none.
- Config schema: none unless a missing tolerance/default is unavoidable; classify explicitly before changing.
- Benchmark evidence schema: compatible additive fields only.

## Performance

- Compare only equivalent workload/stage boundaries.
- Do not compare service-only artifact context/open overhead to notebook `load_hit_times` unless the canonical notebook target includes the same work.
- Capture warmup separately from measured runtime.

# Files To Indicate (Expected Touched Areas)

- `src/trading/contexts/backtest/application/dto/<new tp/sl hit-times dto>.py`
- `src/trading/contexts/backtest/application/services/v2/<new tp/sl hit-times service>.py`
- `src/trading/contexts/backtest/application/ports/artifact_arrays.py`
- `src/trading/contexts/backtest/application/dto/__init__.py`
- `src/trading/contexts/backtest/application/services/v2/__init__.py`
- `tests/unit/contexts/backtest/application/services/v2/<new hit-times tests>.py`
- `scripts/backtest/run_iteration_5_tp_sl_hit_times_benchmark.py`
- `docs/architecture/backtest/benchmark_iterations/YYYY-MM-DD_iteration_5_tp_sl_hit_times_loading_validation/benchmark_results.json`
- `docs/architecture/backtest/benchmark_iterations/YYYY-MM-DD_iteration_5_tp_sl_hit_times_loading_validation/benchmark_summary.md`

# Non-Goals

- TP/SL exact scoring kernels.
- TP/SL self-check.
- Risk-on heap/top-K.
- Public job API/persistence.
- Lazy trades.
- UI chart payload.
- Rewriting artifact publisher or hit-times precompute algorithm.

# Quality Gates (Must Run And Pass)

Use focused commands first:

```bash
uv run ruff check src/trading/contexts/backtest tests/unit/contexts/backtest scripts/backtest
uv run pyright
uv run pytest -q tests/unit/contexts/backtest/application/services/v2
```

If benchmark Markdown or docs index changes:

```bash
python -m tools.docs.generate_docs_index
python -m tools.docs.generate_docs_index --check
```

Mac Studio acceptance evidence is required for completion. Local tests alone are not enough.

# Final Output: Report Format (Strict)

Respond in Russian with:

1. `Что сделано`
2. `Stage contract`
3. `Benchmark / Mac Studio`
4. `Проверки`
5. `Contract impact`
6. `Ограничения / следующий шаг`

Include exact paths changed, commands run, Mac Studio evidence path, pass/fail table for `load_hit_times` and `tp_sl_grid_validation`, and any residual blocker after the second implementation attempt.
