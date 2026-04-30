---
prompt_name: backtest_service_iteration_4_6_benchmark_runner_accounting
repo: roehub.com
branch: current
scope: "Iteration 4.6: implement/fix benchmark runner accounting so notebook-compatible totals and service-only telemetry are separated."

language:
  implementation: python
  agent_report: ru

context_sources:
  always_read:
    - path: .codex/AGENTS.md
      why: "repo contract"
    - path: docs/architecture/backtest/backtest-service-artifact-runtime-v1.ru.md
      why: "benchmark accounting contract"
    - path: docs/architecture/backtest/benchmark_iterations/README.md
      why: "benchmark manifest"
  task_entrypoints:
    - path: docs/architecture/backtest/benchmark_iterations/2026-04-26_engine_test_btcusdt_15m/benchmark_results.json
      why: "canonical stage keys"
      inspect_symbols:
        - runs
        - timers
    - path: src/trading/contexts/backtest/application/services/v2/no_risk_exact.py
      why: "stage telemetry producer"
      inspect_symbols:
        - BacktestNoRiskExactScoringService
    - path: docs/architecture/backtest/benchmark_iterations/2026-04-27_iteration_3_combo_planning_contexts/benchmark_results.json
      why: "accepted service benchmark pattern"
      inspect_symbols:
        - service_raw
        - comparisons
  conditional_bundles:
    runner_code:
      read_when: "locating existing benchmark runner or smoke scripts"
      paths:
        - scripts
        - tests/perf_smoke/contexts/backtest
    docs_index:
      read_when: "docs index changes are made"
      paths:
        - docs/architecture/README.md
        - docs/INDEX.md
  consult_if_needed:
    - path: docs/architecture/backtest/benchmark_iterations/2026-04-27_iteration_4_no_risk_exact_scoring_notebook_topk/benchmark_summary.md
      read_when: "checking previous accounting failure"

style_references:
  - docs/architecture/backtest/benchmark_iterations/README.md

hard_requirements:
  preserve_canonical_stage_order: true
  no_new_canonical_stage_without_target: true
  separate_service_total: true
  macstudio_pipeline_documented: true
  max_implementation_attempts: 2

task_toggles:
  implement_runner_accounting: true
  update_benchmark_manifest_if_needed: true
  implement_scoring_algorithms: false

skill_routing:
  - skill: backend-performance-evidence
    use_when: "designing or validating benchmark comparison"
    timing: before implementation
    reason: "comparability and target/stage boundaries"
  - skill: contract-impact-analysis
    use_when: "changing benchmark JSON/schema fields"
    timing: before implementation
    reason: "benchmark evidence contract compatibility"
  - skill: backend-quality-gates
    use_when: "running local checks"
    timing: during verification
    reason: "lint/type/test gates"

target_envs:
  - local-dev
  - macstudio

required_literals:
  - "total_without_warmup"
  - "service_total_without_warmup"
  - "prepare_pools_core"
  - "benchmark_top_k = 5"
  - "request.top_n = 100"

non_goals:
  - "Do not implement no-risk algorithms here."
  - "Do not add cleanup as a canonical benchmark stage."
  - "Do not compare service_total_without_warmup against canonical total."

final_report_format:
  language: ru
  sections:
    - "Что сделано"
    - "Accounting rules"
    - "Mac Studio pipeline"
    - "Проверки"
    - "Contract impact"

quality_gates:
  - cmd: "uv run ruff check scripts tests src/trading/contexts/backtest docs/architecture/backtest"
    expect: "passes or scope-targeted alternative justified"
  - cmd: "uv run pyright"
    expect: "passes"
  - cmd: "uv run pytest -q tests/unit/contexts/backtest tests/perf_smoke/contexts/backtest"
    expect: "passes or targeted subset with unrelated failures isolated"

expected_primary_touches:
  - "docs/architecture/backtest/benchmark_iterations/README.md"
  - "benchmark runner script or test under scripts/ or tests/perf_smoke/contexts/backtest"

possible_secondary_touches:
  - "docs/architecture/backtest/backtest-service-artifact-runtime-v1.ru.md"
  - "docs/architecture/backtest/benchmark_iterations/YYYY-MM-DD_iteration_4_6_benchmark_runner_accounting/"

safety_notes:
  - "This prompt owns accounting, not scoring."
  - "The ordered canonical stage list must match current canonical JSON; service-only telemetry stays separate."
  - "The canonical notebook JSON uses timer key `prepare_pools`; service evidence may expose `prepare_pools_core`. Runner must normalize this alias before stage comparison."
  - "The executor has only 2 implementation attempts; after the second failed corrective cycle, stop and report the blocker."
---

# Task

Implement or fix Iteration 4.6 benchmark runner accounting so future Iteration 4 acceptance compares equal processes.

Done means:

- runner computes `total_without_warmup` only from notebook-compatible stages;
- runner records `service_total_without_warmup` separately;
- runner records `request.top_n=100`, `benchmark_top_k=5`, `sample_warmup_top_k=1`, `top_results_count=5`, heap capacity;
- service-only telemetry does not become a canonical benchmark stage;
- Mac Studio pipeline is executable and evidence is written according to manifest.

## Context / Current State

- Previous failed Iteration 4 record compared canonical `total_without_warmup` with `service_total_without_warmup`.
- Current plan says cleanup evidence is service hygiene, not a canonical stage.
- Canonical JSON has current timer keys; do not invent new canonical target fields.
- Canonical notebook JSON uses `timers.prepare_pools`; service-side comparison name is `prepare_pools_core`. Treat this as an explicit alias, not as an unknown stage.

## Requirements (Must)

- Preserve canonical ordered stage list:
  - `service_warmup`;
  - `numba_warmup`;
  - `sample_warmup`;
  - `total_without_warmup`;
  - `load_hit_times`;
  - `tp_sl_grid_validation`;
  - `prepare_pools_core` as service alias for notebook `prepare_pools`;
  - `build_exact_context`;
  - `build_proxy_context`;
  - `combo_iteration`;
  - `proxy_filter`;
  - `self_check`;
  - `exact_scoring`;
  - `tp_sl_exact_scoring`;
  - `heap_update`;
  - `top_result_proxy_fill`.
- Treat canonical JSON `total` as historical/no-risk alias of `total_without_warmup`, not a separate stage.
- Normalize notebook `prepare_pools` to service `prepare_pools_core` before comparing stage presence, ratios, or summary rows.
- Keep service-only telemetry separate:
  - `artifact_context_resolve`;
  - `artifact_array_open`;
  - `request_slice_prepare`;
  - `prepare_pools_total`;
  - `service_total_without_warmup`;
  - `top_result_assembly`;
  - `tp_sl_full_metrics_second_pass`;
  - `persist_top_n_io`;
  - `lazy_trades_compute`;
  - `lazy_trades_cache_hit`;
  - memory cleanup evidence.
- Fail closed if runner sees an unknown canonical stage after explicit alias normalization, or if a required target is missing.

## Requirements (Should)

- Provide a small local validation test that loads canonical JSON and asserts stage/accounting rules.
- Keep benchmark result schema backward-compatible with accepted records where practical.

## Requirements (Nice-to-have)

- Add a helper to generate summary Markdown from JSON without hand-editing numbers.

# Context acquisition protocol

Read only in order: repo contract, current architecture doc, benchmark manifest, canonical JSON, runner code. Do not broad-read all scripts unless locating runner is necessary.

Pre-implementation reading target:

- `<= 8 files`
- `<= ~35k-50k tokens`

Expand only for runner discovery, schema ambiguity, or failing tests.

# Reading manifest

Use front-matter `context_sources`; do not create a broad mandatory reading list.

# Work plan (agent should follow)

Skill routing:

- `backend-performance-evidence`: use for benchmark comparability.
- `contract-impact-analysis`: use for JSON/schema changes.
- `backend-quality-gates`: use for local checks.

1. Locate existing benchmark runner or create the smallest runner/accounting helper needed.
2. Encode canonical stage list and service-only telemetry separation.
3. Add tests validating accounting against canonical JSON, including `prepare_pools` to `prepare_pools_core` alias normalization.
4. Update benchmark manifest only if needed and keep it consistent with architecture doc.
5. Run local gates.
6. Run Mac Studio pipeline with a small validation benchmark if feasible.

Mac Studio benchmark pipeline:

1. Commit local changes after local gates pass.
2. Push branch/commit to remote.
3. SSH to `macstudio`.
4. In `/opt/roehub/app`, fetch and pull the pushed commit.
5. Verify commit SHA.
6. Run the benchmark accounting validation/runner on Mac Studio.
7. Save evidence under `docs/architecture/backtest/benchmark_iterations/YYYY-MM-DD_iteration_4_6_benchmark_runner_accounting/`.
8. Record that canonical `total_without_warmup` and service-only `service_total_without_warmup` are separate.

# Acceptance criteria (Definition of Done)

- Accounting test proves `service_total_without_warmup` is not compared to canonical `total_without_warmup`.
- Accounting test proves notebook `prepare_pools` is normalized to service `prepare_pools_core`.
- No cleanup field is present in canonical stage list.
- Manifest and architecture doc agree.
- Mac Studio evidence exists for runner/accounting validation, or final report explains why it is pending.
- Local gates pass.

# Implementation constraints

## Determinism & ordering

- Preserve canonical stage order.
- Do not sort stages alphabetically for comparison if order matters in summaries.

## API / contracts

- Public API: none.
- DTO schema: none unless benchmark DTO exists.
- Persisted schema: none.
- Config schema: none.
- Benchmark evidence schema: compatible-change if additive fields are added; document it.

# Files to indicate (expected touched areas)

- benchmark runner script/test if present or newly created
- `docs/architecture/backtest/benchmark_iterations/README.md`
- `docs/architecture/backtest/backtest-service-artifact-runtime-v1.ru.md` only if a real inconsistency remains
- benchmark evidence folder if run

# Non-goals

- scorer implementation;
- heap/proxy optimization;
- public API;
- persistence;
- lazy trades implementation.

# Quality gates (must run and pass)

- `uv run ruff check <changed-python-files>`
- `uv run pyright`
- `uv run pytest -q <targeted-runner-or-backtest-tests>`
- `uv run python -m tools.docs.generate_docs_index --check` if docs index changes
- Mac Studio runner/accounting validation if available.

# Final output: report format (strict)

Report in Russian:

1. Accounting changes.
2. Canonical vs service-only field list.
3. Tests.
4. Mac Studio evidence path or pending reason.
5. Contract impact classification.
6. Any remaining runner gaps.
