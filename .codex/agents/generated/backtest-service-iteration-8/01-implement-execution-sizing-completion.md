---
prompt_name: backtest_service_iteration_8_execution_sizing_completion
repo: roehub.com
branch: current
scope: "Iteration 8: complete execution/sizing semantics in compiled service paths for no-risk and TP/SL backtests."

language:
  implementation: python
  agent_report: ru

context_sources:
  always_read:
    - path: .codex/AGENTS.md
      why: "repo engineering contract, prompt precedence, delivery rules, and merge/deploy expectations"
    - path: docs/architecture/backtest/backtest-service-artifact-runtime-v1.ru.md
      why: "canonical Iteration 8 contract, sizing matrix, execution defaults, and benchmark gate"
    - path: docs/architecture/backtest/benchmark_iterations/README.md
      why: "benchmark evidence manifest, Mac Studio acceptance rules, and service-only accounting"
    - path: docs/architecture/backtest/benchmark_iterations/2026-04-26_engine_test_btcusdt_15m/benchmark_results.json
      why: "canonical sizing_smoke evidence and arity/risk/direction regression target"
    - path: docs/architecture/backtest/benchmark_iterations/2026-05-01_iteration_7_job_orchestration_persistence/benchmark_summary.md
      why: "latest accepted job orchestration/persistence evidence and Iteration 8 precondition"
  task_entrypoints:
    - path: src/trading/contexts/backtest/application/services/v2/preflight.py
      why: "public request normalization already accepts all v1 sizing modes, profit_lock, and close_on_end"
      inspect_symbols:
        - BACKTEST_SIZING_MODES_V1
        - BacktestPreflightService
        - _normalize_sizing
        - _normalize_profit_lock
    - path: src/trading/contexts/backtest/application/dto/runtime_preflight.py
      why: "runtime defaults and public execution default shape"
      inspect_symbols:
        - BacktestExecutionDefaults
        - BacktestRuntimeDefaults
    - path: src/trading/contexts/backtest/application/services/v2/no_risk_exact.py
      why: "accepted no-risk compiled execution path currently limited to all_in/fixed_quote"
      inspect_symbols:
        - BacktestNoRiskExactScoringService
        - _ExecutionSettings
        - _execution_settings_from_normalized
    - path: src/trading/contexts/backtest/application/services/v2/tp_sl_exact.py
      why: "accepted TP/SL compiled execution path reuses no-risk execution settings and must keep risk semantics stable"
      inspect_symbols:
        - BacktestTpSlExactScoringService
        - TP_SL_FULL_METRICS_SECOND_PASS_STAGE_NAME
    - path: tests/unit/contexts/backtest/application/services/v2/test_no_risk_exact_scoring_service.py
      why: "current no-risk execution/direction/sizing test patterns"
    - path: tests/unit/contexts/backtest/application/services/v2/test_tp_sl_exact_scoring_service.py
      why: "current TP/SL execution/direction/sizing test patterns"
  conditional_bundles:
    iteration_7_acceptance:
      read_when: "before implementation; if accepted Iteration 7 evidence is missing or failed, stop and report the precondition blocker"
      paths:
        - docs/architecture/backtest/benchmark_iterations/2026-05-01_iteration_7_job_orchestration_persistence/benchmark_summary.md
        - docs/architecture/backtest/benchmark_iterations/2026-05-01_iteration_7_job_orchestration_persistence/benchmark_results.json
    canonical_notebook_algorithm:
      read_when: "sizing/profit_lock semantics are ambiguous or service/reference parity fails"
      paths:
        - tests/notebook_tests/engine_test/btcusdt_15m_research_engine.ipynb
        - docs/architecture/backtest/benchmark_iterations/2026-04-26_engine_test_btcusdt_15m/benchmark_summary.md
    job_orchestration_surface:
      read_when: "execution/sizing changes affect persisted job summaries, variant identity, or API defaults"
      paths:
        - .codex/agents/generated/backtest-service-iteration-7/01-implement-job-orchestration-persistence.md
        - apps/api/routes/backtests.py
        - apps/api/dto/backtests.py
        - src/trading/contexts/backtest/application/services/v2/top_result_assembly.py
    benchmark_runner_patterns:
      read_when: "adding Iteration 8 benchmark runner and evidence writer"
      paths:
        - scripts/backtest/run_iteration_6_tp_sl_exact_scoring_benchmark.py
        - scripts/backtest/run_iteration_7_job_orchestration_persistence_benchmark.py
        - scripts/backtest/validate_benchmark_accounting.py
    legacy_execution_reference:
      read_when: "legacy account/profit-lock model helps clarify mechanics after active v2 code and notebook have been checked"
      paths:
        - src/trading/contexts/backtest/domain/entities/execution_v1.py
        - src/trading/contexts/backtest/domain/value_objects/execution_v1.py
      warning: "Use only as reference. Public v1 contract is the runtime document and active preflight DTOs, not legacy `strategy_compound` vocabulary."
    delivery_skill_reference:
      read_when: "all local gates and Mac Studio acceptance pass and the implementation is ready to merge/deploy"
      paths:
        - /Users/daniildegtyarev/.codex/skills/publish-ci-deploy/SKILL.md
      instruction: "Use only for the final delivery path. Do not preload it during implementation."
  consult_if_needed:
    - path: tests/unit/contexts/backtest/application/services/v2/test_backtest_preflight_service.py
      read_when: "public request/defaults validation changes"
    - path: tests/unit/apps/api/test_backtests_routes.py
      read_when: "runtime-defaults response changes"
    - path: docs/architecture/backtest/benchmark_iterations/2026-05-01_iteration_6_tp_sl_exact_scoring_full_metrics/benchmark_summary.md
      read_when: "checking risk-on exact-scoring regression envelope"

style_references:
  - .codex/promt_template.md
  - .codex/agents/generated/backtest-service-iteration-7/01-implement-job-orchestration-persistence.md
  - docs/architecture/backtest/benchmark_iterations/README.md

hard_requirements:
  iteration_7_acceptance_required_before_implementation: true
  all_public_sizing_modes_supported_in_compiled_path: true
  profit_lock_supported_for_every_sizing_mode: true
  close_on_end_false_supported: true
  preserve_no_risk_and_tp_sl_semantics: true
  macstudio_acceptance_required: true
  merge_main_and_pull_after_success: true
  max_implementation_attempts: 2

task_toggles:
  implement_all_in_regression: true
  implement_fixed_quote_regression: true
  implement_fixed_equity_pct: true
  implement_fixed_equity_pct_min_quote: true
  implement_fixed_equity_pct_max_quote: true
  implement_profit_lock_for_all_modes: true
  implement_close_on_end_false: true
  update_runtime_defaults_if_needed: true
  update_variant_identity_inputs_if_needed: true
  implement_lazy_trades: false
  implement_ui: false
  publish_merge_deploy_after_success: true

skill_routing:
  - skill: contract-impact-analysis
    use_when: "changing public execution defaults, request normalization, variant identity inputs, persisted summaries, or benchmark evidence schema"
    timing: before implementation and before final report
    reason: "Iteration 8 changes result-affecting execution semantics and may affect request hashes and variant hashes"
  - skill: numba
    use_when: "changing compiled scoring/accounting kernels or typed execution settings used by no-risk and TP/SL exact services"
    timing: during implementation and optimization
    reason: "equity-percent sizing and profit_lock must run in compiled hot paths without degrading accepted benchmark stages"
  - skill: backend-performance-evidence
    use_when: "building or reporting sizing smoke, regression benchmark, CPU/RSS, and Mac Studio evidence"
    timing: during benchmark verification
    reason: "Iteration 8 acceptance requires benchmark/regression evidence, especially for first compiled parity of equity-percent modes"
  - skill: backend-quality-gates
    use_when: "running targeted lint, type, unit, and regression tests"
    timing: during verification
    reason: "Roehub backend gates are uv-based"
  - skill: publish-ci-deploy
    use_when: "all implementation gates and Mac Studio acceptance pass, and the branch is ready for merge/deploy"
    timing: after verification
    reason: "user requires merge to main, local pull, Mac Studio pull, deploy verification, and post-deploy evidence"

target_envs:
  - local-dev
  - github-actions
  - macstudio

required_literals:
  - "all_in"
  - "fixed_quote"
  - "fixed_equity_pct"
  - "fixed_equity_pct_min_quote"
  - "fixed_equity_pct_max_quote"
  - "profit_lock"
  - "safe_profit_percent"
  - "close_on_end"
  - "close_on_end = false"
  - "sizing_smoke"
  - "first compiled parity point"
  - "historical_prefix_compatible"

non_goals:
  - "Do not implement lazy trades detail, cache, or chart payloads; that belongs to Iteration 9."
  - "Do not implement UI integration; that belongs to Iteration 10."
  - "Do not change public job API vocabulary from Iteration 7."
  - "Do not broaden into new risk modes or new indicators catalog work."
  - "Do not use legacy `strategy_compound` or execution-profile vocabulary as the public v1 contract."
  - "Do not use legacy `hit_times/1m` paths."

final_report_format:
  language: ru
  sections:
    - "Что сделано"
    - "Execution / sizing contract"
    - "Benchmark / Mac Studio"
    - "Проверки"
    - "Delivery / merge"
    - "Contract impact"
    - "Ограничения / следующий шаг"

quality_gates:
  - cmd: "uv run ruff check apps/api src/trading/contexts/backtest tests/unit/apps/api tests/unit/contexts/backtest scripts/backtest"
    expect: "passes, or a narrower justified target passes if unrelated existing files fail"
  - cmd: "uv run pyright"
    expect: "passes"
  - cmd: "uv run pytest -q tests/unit/contexts/backtest/application/services/v2/test_no_risk_exact_scoring_service.py tests/unit/contexts/backtest/application/services/v2/test_tp_sl_exact_scoring_service.py tests/unit/contexts/backtest/application/services/v2/test_backtest_preflight_service.py tests/unit/apps/api/test_backtests_routes.py"
    expect: "passes; include any new Iteration 8 test file explicitly"
  - cmd: "uv run pytest -q -ra"
    expect: "passes before merge/deploy"
  - cmd: "uv run python scripts/backtest/validate_benchmark_accounting.py --out docs/architecture/backtest/benchmark_iterations/<iteration_8_dir>/local_accounting_validation.json"
    expect: "passes after the Iteration 8 runner writes local accounting evidence"
  - cmd: "uv run python -m tools.docs.generate_docs_index --check"
    expect: "passes if docs or benchmark summaries change"

expected_primary_touches:
  - "src/trading/contexts/backtest/application/services/v2/<shared execution sizing helper>.py"
  - "src/trading/contexts/backtest/application/services/v2/no_risk_exact.py"
  - "src/trading/contexts/backtest/application/services/v2/tp_sl_exact.py"
  - "src/trading/contexts/backtest/application/dto/no_risk_exact.py"
  - "src/trading/contexts/backtest/application/dto/tp_sl_exact.py"
  - "tests/unit/contexts/backtest/application/services/v2/<new execution sizing tests>.py"
  - "tests/unit/contexts/backtest/application/services/v2/test_no_risk_exact_scoring_service.py"
  - "tests/unit/contexts/backtest/application/services/v2/test_tp_sl_exact_scoring_service.py"
  - "scripts/backtest/run_iteration_8_execution_sizing_benchmark.py"
  - "docs/architecture/backtest/benchmark_iterations/YYYY-MM-DD_iteration_8_execution_sizing_completion/"

possible_secondary_touches:
  - "src/trading/contexts/backtest/application/services/v2/preflight.py"
  - "src/trading/contexts/backtest/application/dto/runtime_preflight.py"
  - "apps/api/dto/backtests.py"
  - "apps/api/routes/backtests.py"
  - "src/trading/contexts/backtest/application/services/v2/top_result_assembly.py"
  - "src/trading/contexts/backtest/domain/value_objects/variant_identity.py"
  - "docs/architecture/backtest/backtest-service-artifact-runtime-v1.ru.md"
  - "docs/architecture/README.md"

safety_notes:
  - "Do not start implementation unless Iteration 7 accepted evidence exists. Iteration 8 depends on persisted job/top-result behavior."
  - "Current preflight accepts every public sizing mode, but accepted compiled scorers historically supported only `all_in` and `fixed_quote`; closing that gap is the main task."
  - "Canonical sizing_smoke evidence has compiled parity for `all_in` and `fixed_quote`; equity-percent modes are reference-only and must become service first compiled parity evidence."
  - "Beware field-name drift: public preflight normalizes fixed quote as `quote_amount`, while older scorer code may read `fixed_quote`."
  - "After all checks and Mac Studio acceptance pass, use the repo delivery path to merge to `main`, then pull `main` locally and on Mac Studio."
  - "The executor has only 2 implementation attempts. After the second failed corrective cycle, stop and report the blocker with exact evidence."
---

# Task

Implement Iteration 8: complete execution and sizing support in the compiled backtest service.

Done means:

- all public sizing modes are supported by the compiled no-risk and TP/SL service paths;
- `profit_lock` works for every sizing mode;
- `close_on_end = false` is supported and covered by service-level correctness tests;
- `all_in` and `fixed_quote` accepted behavior does not regress;
- `fixed_equity_pct`, `fixed_equity_pct_min_quote`, and `fixed_equity_pct_max_quote` have service-level first compiled parity evidence;
- no-risk and TP/SL semantics remain stable across execution settings;
- Mac Studio benchmark/correctness evidence is written under `docs/architecture/backtest/benchmark_iterations/<date>_iteration_8_execution_sizing_completion/`;
- if all checks and acceptance gates pass, the work is merged to `main`, `main` is pulled locally, and `main` is pulled on Mac Studio.

## Context / Current State

Precondition:

- Iteration 7 must already be accepted with Mac Studio benchmark evidence.
- If accepted Iteration 7 evidence is missing or failed, stop before implementation and report that Iteration 8 is blocked.

Context ledger from previous accepted iterations:

- completed:
  - Iteration 1: public request normalization already validates all v1 sizing modes, `profit_lock`, and `close_on_end`.
  - Iteration 4: no-risk exact scoring supports accepted compiled path and benchmark stages.
  - Iteration 6: TP/SL exact scoring supports accepted compiled path and full metrics.
  - Iteration 7: job orchestration and summary persistence are accepted.
- open_items:
  - compiled scoring path still needs public sizing parity for equity-percent modes;
  - `profit_lock` must be verified for every sizing mode;
  - `close_on_end = false` must be implemented and tested in no-risk and TP/SL paths.
- contract_changes:
  - execution settings are result-affecting and must remain part of canonical request/variant identity;
  - runtime defaults may need compatible DTO updates only if the public default shape changes.
- touched_paths:
  - likely shared execution helper plus no-risk and TP/SL scorer services;
  - tests and benchmark runner;
  - docs status/evidence only after acceptance.
- risks:
  - field-name drift between `quote_amount` and older `fixed_quote` helper code;
  - equity-percent sizing can silently use initial cash instead of current equity after each trade;
  - profit locking can be double-counted or accidentally reinvested;
  - close-on-end false can corrupt metrics if open positions are force-closed anyway.
- next_focus:
  - implement one shared execution/sizing contract used by no-risk and TP/SL;
  - add correctness tests for every sizing mode and profit lock state;
  - add Mac Studio benchmark evidence and regression envelope;
  - deliver through merge/main/pull only after acceptance.

Additional context:

- Public sizing modes:
  - `all_in`;
  - `fixed_quote`;
  - `fixed_equity_pct`;
  - `fixed_equity_pct_min_quote`;
  - `fixed_equity_pct_max_quote`.
- Public profit lock:
  - `profit_lock.enabled`;
  - `profit_lock.safe_profit_percent`.
- Current canonical `sizing_smoke` has compiled parity for `all_in` and `fixed_quote`.
- Equity-percent modes in canonical evidence are reference-only and must become first compiled parity points in service implementation.
- Public request field is `fixed_quote` mode with `quote_amount`. Do not require users to send old internal `fixed_quote`.

## Requirements (Must)

- Read context using the protocol below and stop early once sufficient.
- Verify Iteration 7 acceptance evidence before implementation.
- Implement only the scoped change described in this prompt.
- Preserve all accepted benchmark and runtime behavior from Iterations 1..7.
- Add or update targeted tests for sizing, profit lock, close-on-end, no-risk, TP/SL, and request normalization aliases.
- Keep public API/request semantics deterministic and backward-compatible.

Execution/sizing behavior:

- Support these sizing modes in compiled no-risk and TP/SL paths:
  - `all_in`: use all currently investable quote.
  - `fixed_quote`: use configured `quote_amount`, capped by currently investable quote where the reference semantics require it.
  - `fixed_equity_pct`: use a percentage of current equity for each entry.
  - `fixed_equity_pct_min_quote`: use current-equity percentage but not less than configured `min_quote`, subject to available/investable quote semantics from the notebook/reference.
  - `fixed_equity_pct_max_quote`: use current-equity percentage but not more than configured `max_quote`.
- Use public normalized request fields:
  - `quote_amount` for `fixed_quote`;
  - `equity_pct` for equity-percent modes;
  - `min_quote`;
  - `max_quote`.
- Do not require legacy internal field names in public requests.
- Profit lock must be a sizing overlay, not a separate sizing mode:
  - disabled: all realized equity remains available for reinvestment;
  - enabled: configured share of realized profit moves to safe/non-reinvested quote according to notebook/reference semantics;
  - `safe_quote` must be reported in smoke evidence where available.
- `close_on_end = true` must preserve accepted behavior.
- `close_on_end = false` must avoid forced final close and must produce deterministic metrics for open-at-end scenarios.
- Fees and slippage remain decimal rates from the public request.
- Direction modes remain:
  - `long_only`;
  - `long_short_reversal`.

Implementation shape:

- Prefer extracting shared execution/sizing settings and compiled helpers from `no_risk_exact.py` into a small v2 module instead of letting `tp_sl_exact.py` import private no-risk helpers.
- Keep hot-path helpers array/scalar oriented and Numba-compatible.
- Preserve accepted stage boundaries:
  - sizing logic belongs inside existing exact/full-metrics calculations;
  - do not add public API/persistence assembly into exact-scoring timers;
  - record any new service-only telemetry separately.
- Update variant identity inputs if and only if current variant hashing omits result-affecting sizing/profit/close-on-end fields.

Correctness and regression:

- Add synthetic unit tests for each sizing mode with and without profit lock.
- Add tests where equity changes after a profitable and losing trade, proving `fixed_equity_pct` uses current equity rather than initial cash.
- Add min/max clamp tests:
  - percentage below `min_quote`;
  - percentage above `max_quote`;
  - available quote below requested amount, if reference semantics cap to available quote.
- Add `close_on_end = true/false` tests for no-risk and TP/SL.
- Add regression tests proving `all_in` and `fixed_quote` still match existing accepted expectations.
- Add TP/SL tests proving best-cell selection and full metrics remain stable across sizing/profit-lock settings.

Benchmark/evidence:

- Add `scripts/backtest/run_iteration_8_execution_sizing_benchmark.py`.
- Write evidence to:
  - `docs/architecture/backtest/benchmark_iterations/YYYY-MM-DD_iteration_8_execution_sizing_completion/benchmark_results.json`;
  - `docs/architecture/backtest/benchmark_iterations/YYYY-MM-DD_iteration_8_execution_sizing_completion/benchmark_summary.md`.
- Evidence must include:
  - canonical `sizing_smoke` comparison for every sizing mode and profit-lock state;
  - first compiled parity evidence for equity-percent modes;
  - close-on-end true/false service-level correctness evidence;
  - no-risk and TP/SL regression checks;
  - arity/risk/direction benchmark regression envelope against current canonical evidence;
  - CPU/RSS metrics for the sizing smoke and regression runs;
  - commit, request hash, artifact hashes, artifact compatibility policy, and environment metadata.

Delivery:

- If and only if local gates, Mac Studio evidence, benchmark/correctness acceptance, docs checks, and CI all pass:
  - use `publish-ci-deploy` for the delivery path;
  - push the branch;
  - open/update PR if needed;
  - watch CI to completion;
  - merge into `main`;
  - pull `main` on the local machine in `/Users/daniildegtyarev/Projects/roehub.com`;
  - pull `main` on Mac Studio in `/Users/daniildegtyarev/Projects/roehub.com`;
  - verify deployed/runtime surface on Mac Studio.
- Do not merge or pull/deploy if any required gate is red, skipped, or ambiguous.

## Requirements (Should)

- Keep DTO/request semantics aligned with existing public preflight normalization.
- Keep the implementation shared between no-risk and TP/SL where that reduces drift.
- Prefer explicit small typed settings over generic dict access inside hot loops.
- Keep benchmark runner output shape close to previous iteration summaries.
- Record reference-only vs compiled-parity status explicitly in the JSON.
- Include failure diagnostics by `{risk_mode, sizing_mode, profit_lock, close_on_end, direction_mode}`.

## Requirements (Nice-to-have)

- Add a small table in `benchmark_summary.md` comparing ending equity, safe quote, total return, and trade count for every sizing smoke row.
- Add a local developer smoke command that can run a tiny synthetic fixture without Mac Studio artifacts.

# Context acquisition protocol

Read only in this order and stop once sufficient:

1. `.codex/AGENTS.md`
2. Iteration 8 section plus sizing matrix, risk notes, and test matrix sections of `docs/architecture/backtest/backtest-service-artifact-runtime-v1.ru.md`
3. accepted Iteration 7 evidence folder; stop if missing or failed
4. canonical JSON `sizing_smoke` evidence
5. task entrypoints
6. only the conditional bundle(s) required by touched contracts, failing checks, or parity ambiguity
7. consult-if-needed references only for blockers, ambiguity, or conflict resolution

Do not eagerly preload all listed sources.

Pre-implementation reading target:

- `<= 10 files`
- `<= ~50k-65k tokens`

Stop reading once all of the following are true:

- execution/sizing contract is identified;
- no-risk and TP/SL touched files are bounded;
- benchmark evidence shape is implementable;
- no unresolved public API, request hash, or variant identity ambiguity remains.

Expand context only for:

- notebook/reference sizing ambiguity;
- Numba typing/performance blocker;
- failing parity;
- request/defaults contract conflict;
- Mac Studio benchmark/deploy failure.

# Reading manifest

Use the front-matter `context_sources` as the canonical reading map.

Read with this intent:

- `always_read`:
  - repository rules;
  - current runtime target;
  - benchmark evidence contract;
  - accepted Iteration 7 precondition.
- `task_entrypoints`:
  - public request/defaults normalization;
  - accepted compiled scorer paths;
  - current tests.
- `conditional_bundles`:
  - read only when the stated condition applies.
- `consult_if_needed`:
  - read only for blockers, ambiguity, or conflict resolution.

Do not convert this manifest into a broad mandatory reading list.

# Work plan (agent should follow)

Skill routing for this task:

- `contract-impact-analysis`: use before implementation and before final report; owns request/defaults compatibility, request hash, variant identity, and persisted summary impact.
- `numba`: use during implementation and optimization; owns compiled helper design and Numba typing/performance issues.
- `backend-performance-evidence`: use during benchmark verification; owns sizing smoke, regression envelope, CPU/RSS evidence, and Mac Studio acceptance claims.
- `backend-quality-gates`: use during verification; owns uv-based lint, type, and test gates.
- `publish-ci-deploy`: use only after all implementation and Mac Studio acceptance gates pass; owns push, PR/CI, merge to `main`, local pull, Mac Studio pull, deploy verification, and post-deploy evidence.

Implementation sequence:

1. Verify Iteration 7 accepted evidence exists. Stop if missing.
2. Read bounded context and classify contract impact before code changes.
3. Inspect canonical `sizing_smoke` rows and normalize reference aliases such as `pct` to public `equity_pct` only inside benchmark fixture adaptation.
4. Design shared execution/sizing settings for no-risk and TP/SL.
5. Implement compiled support for all sizing modes and profit-lock overlay.
6. Implement or fix `close_on_end = false` semantics in no-risk and TP/SL paths.
7. Update tests for all modes and regression cases.
8. Add Iteration 8 benchmark runner and evidence writer.
9. Run local gates and fix introduced failures.
10. Run Mac Studio benchmark/correctness evidence.
11. If accepted, update the main runtime document status for Iteration 8 and docs index.
12. If all gates are green, use `publish-ci-deploy` to push, PR, watch CI, merge to `main`, pull locally, pull on Mac Studio, and verify production/runtime health.
13. If any gate fails after two implementation attempts, stop and report exact blockers.

# Benchmark and Mac Studio pipeline

Acceptance benchmark and post-merge verification must use Mac Studio evidence.

Implementation benchmark path:

```bash
export PATH="/opt/homebrew/bin:$PATH"
cd /Users/daniildegtyarev/Projects/roehub.com
git pull --ff-only
uv run python scripts/backtest/run_iteration_8_execution_sizing_benchmark.py \
  --out-dir docs/architecture/backtest/benchmark_iterations/$(date +%F)_iteration_8_execution_sizing_completion
```

If `/opt/roehub/app` is the runtime surface being benchmarked, record that explicitly. Do not run `git pull` in `/opt/roehub/app`; it is a deployed runtime copy, not the repository checkout.

After all checks pass and the branch is ready:

```bash
# Use publish-ci-deploy to perform the actual push/PR/CI/merge/deploy flow.
# After merge:
cd /Users/daniildegtyarev/Projects/roehub.com
git checkout main
git pull --ff-only

ssh macstudio 'cd /Users/daniildegtyarev/Projects/roehub.com && git checkout main && git pull --ff-only'
ssh macstudio 'cd /Users/daniildegtyarev/Projects/roehub.com && bash scripts/macos/smoke_prod.sh'
```

If Mac Studio auth or remote state is broken, treat that as part of the delivery task and fix it through `publish-ci-deploy` rather than reporting only the first failure.

# Acceptance criteria (Definition of Done)

- Iteration 7 accepted evidence exists before implementation.
- Compiled no-risk path supports all public sizing modes.
- Compiled TP/SL path supports all public sizing modes.
- Profit lock works for every sizing mode.
- `close_on_end = true` accepted behavior is preserved.
- `close_on_end = false` has deterministic service-level correctness tests.
- `all_in` and `fixed_quote` compiled parity remains passing.
- `fixed_equity_pct`, `fixed_equity_pct_min_quote`, and `fixed_equity_pct_max_quote` have first compiled parity evidence against notebook/reference fixtures.
- No-risk and TP/SL summaries remain stable across execution settings where expected.
- Request/defaults/API contract remains compatible.
- Variant identity includes all result-affecting execution/sizing fields or an explicit no-change proof is documented.
- Iteration 8 benchmark evidence exists under the accepted folder.
- Mac Studio evidence passes.
- Local full gates pass before merge.
- CI passes before merge.
- The branch is merged to `main`.
- Local checkout at `/Users/daniildegtyarev/Projects/roehub.com` is on updated `main`.
- Mac Studio checkout at `/Users/daniildegtyarev/Projects/roehub.com` is on updated `main`.
- Mac Studio production/runtime smoke passes after merge/deploy.

# Implementation constraints

- Keep diffs scoped to Iteration 8.
- Do not silently change public request fields.
- Do not remove any public sizing mode from runtime defaults.
- Do not introduce legacy public modes such as `strategy_compound`.
- Do not use process-local or global mutable state for account/sizing calculations.
- Do not mix Iteration 9 lazy trades into this work.
- Do not mark Iteration 8 as accepted from local tests alone.
- Do not merge if any required gate is missing, red, or ambiguous.

# Files to indicate (expected touched areas)

Expected primary files:

- `src/trading/contexts/backtest/application/services/v2/<shared execution sizing helper>.py`
- `src/trading/contexts/backtest/application/services/v2/no_risk_exact.py`
- `src/trading/contexts/backtest/application/services/v2/tp_sl_exact.py`
- `src/trading/contexts/backtest/application/dto/no_risk_exact.py`
- `src/trading/contexts/backtest/application/dto/tp_sl_exact.py`
- `tests/unit/contexts/backtest/application/services/v2/<new execution sizing tests>.py`
- `tests/unit/contexts/backtest/application/services/v2/test_no_risk_exact_scoring_service.py`
- `tests/unit/contexts/backtest/application/services/v2/test_tp_sl_exact_scoring_service.py`
- `scripts/backtest/run_iteration_8_execution_sizing_benchmark.py`
- `docs/architecture/backtest/benchmark_iterations/YYYY-MM-DD_iteration_8_execution_sizing_completion/benchmark_results.json`
- `docs/architecture/backtest/benchmark_iterations/YYYY-MM-DD_iteration_8_execution_sizing_completion/benchmark_summary.md`

Possible secondary files:

- `src/trading/contexts/backtest/application/services/v2/preflight.py`
- `src/trading/contexts/backtest/application/dto/runtime_preflight.py`
- `apps/api/dto/backtests.py`
- `apps/api/routes/backtests.py`
- `src/trading/contexts/backtest/application/services/v2/top_result_assembly.py`
- `src/trading/contexts/backtest/domain/value_objects/variant_identity.py`
- `docs/architecture/backtest/backtest-service-artifact-runtime-v1.ru.md`
- `docs/architecture/README.md`

# Non-goals

- No lazy trades compute/cache endpoint implementation.
- No UI chart payload implementation.
- No new public job endpoints.
- No new risk modes.
- No broad roadmap cleanup.
- No legacy `strategy_compound` public mode.
- No legacy `hit_times/1m` runtime path.

# Quality gates (must run and pass)

Run local gates before claiming implementation completion:

```bash
uv run ruff check apps/api src/trading/contexts/backtest tests/unit/apps/api tests/unit/contexts/backtest scripts/backtest
uv run pyright
uv run pytest -q tests/unit/contexts/backtest/application/services/v2/test_no_risk_exact_scoring_service.py tests/unit/contexts/backtest/application/services/v2/test_tp_sl_exact_scoring_service.py tests/unit/contexts/backtest/application/services/v2/test_backtest_preflight_service.py tests/unit/apps/api/test_backtests_routes.py
uv run pytest -q -ra
```

After benchmark evidence is written:

```bash
uv run python scripts/backtest/validate_benchmark_accounting.py \
  --out docs/architecture/backtest/benchmark_iterations/<iteration_8_dir>/local_accounting_validation.json
uv run python -m tools.docs.generate_docs_index --check
```

Run the Mac Studio benchmark gate before claiming acceptance:

```bash
export PATH="/opt/homebrew/bin:$PATH"
uv run python scripts/backtest/run_iteration_8_execution_sizing_benchmark.py \
  --out-dir docs/architecture/backtest/benchmark_iterations/$(date +%F)_iteration_8_execution_sizing_completion
```

Run delivery only after all gates pass:

```bash
# Use publish-ci-deploy skill for the actual flow.
# Required terminal state:
# - branch merged to main
# - local checkout pulled to main
# - Mac Studio checkout pulled to main
# - Mac Studio smoke passed
```

If a command cannot be run, state why, classify the risk, and do not claim that gate as passed.

# Contract impact report

Include this classification in the final report:

- Public API:
- Ports:
- DTO schema:
- Persisted schema:
- Config schema:
- Request/cache/persistence identity:
- Benchmark evidence schema:
- Runtime artifact contract:
- Delivery/deploy surface:

Use one of:

- `none`;
- `compatible-change`;
- `breaking-change`;
- `unknown`.

# Failure/blocker behavior

You have only 2 implementation attempts.

An attempt is a full cycle of implementation, local gates, and Mac Studio benchmark or equivalent blocker evidence. If the second attempt still fails acceptance:

- stop;
- do not broaden scope into Iteration 9 or Iteration 10;
- do not merge;
- do not hide failed benchmark rows or correctness failures;
- report:
  - implementation commit;
  - changed files;
  - exact failed `{risk_mode, sizing_mode, profit_lock, close_on_end, direction_mode}` rows;
  - canonical/reference value, service value, and diff;
  - artifact hashes and compatibility policy;
  - whether the failure is request normalization, sizing semantics, profit lock, close-on-end, no-risk regression, TP/SL regression, Numba typing/performance, CI/deploy, or Mac Studio environment;
  - the smallest next investigation step.

# Final output: report format (strict)

Use Russian.

## Что сделано

- Concise implementation summary.

## Execution / sizing contract

- Supported modes.
- Profit lock behavior.
- Close-on-end behavior.
- Shared helper / compiled path notes.

## Benchmark / Mac Studio

- Evidence directory.
- Commit.
- Artifact hashes.
- Sizing smoke table summary.
- Regression envelope summary.
- Failed rows first if any.

## Проверки

- Commands run and results.
- Commands not run and why.

## Delivery / merge

- Branch / PR.
- CI status.
- Merge status.
- Local `main` pull status.
- Mac Studio `main` pull status.
- Mac Studio smoke/deploy verification.

## Contract impact

- Public API:
- Ports:
- DTO schema:
- Persisted schema:
- Config schema:
- Request/cache/persistence identity:
- Benchmark evidence schema:
- Runtime artifact contract:
- Delivery/deploy surface:

## Ограничения / следующий шаг

- Remaining risks.
- If accepted and merged, state that Iteration 9 is next.
- If not accepted, state the blocker and stop.
