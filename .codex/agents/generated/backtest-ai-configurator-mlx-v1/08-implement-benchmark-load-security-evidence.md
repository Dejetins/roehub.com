---
prompt_name: backtest_ai_configurator_mlx_v1_08_benchmark_load_security_evidence
repo: roehub.com
branch: main
scope: "Iteration 08: implement and run benchmark/load/security eval harness for /backtests AI configurator, record Mac Studio evidence, and choose accepted model/concurrency limits."

language:
  implementation: python_benchmark
  agent_report: ru

context_sources:
  always_read:
    - path: .codex/AGENTS.md
      why: "benchmark and evidence rules"
    - path: docs/architecture/backtest/backtest-ai-configurator-mlx-v1.md
      why: "benchmark matrix and acceptance targets"
  task_entrypoints:
    - path: scripts
      why: "benchmark script location conventions"
      inspect_symbols:
        - "backtest*"
    - path: apps/worker/backtest_ai_configurator
      why: "worker metrics and runtime entrypoint"
      inspect_symbols:
        - "*"
    - path: docs/architecture/backtest/benchmark_iterations/README.md
      why: "benchmark evidence format"
      inspect_symbols:
        - "*"
    - path: configs/prod/backtest_ai_configurator.yaml
      why: "model/concurrency config under test"
      inspect_symbols:
        - backtest_ai_configurator
  conditional_bundles:
    macstudio_ops:
      read_when: "when running acceptance evidence on Mac Studio"
      paths:
        - docs/runbooks/mac-studio-native-backend-operations.md
        - docs/runbooks/mac-studio-monitoring-plan.md
    api_contract:
      read_when: "when load harness hits API routes"
      paths:
        - apps/api/routes/backtest_ai_config.py
        - tests/unit/apps/api/test_backtest_ai_config_routes.py
    existing_benchmarks:
      read_when: "when reusing benchmark output format or helpers"
      paths:
        - docs/architecture/backtest/benchmark_iterations
        - scripts
  consult_if_needed:
    - path: .github/workflows/deploy-backend.yml
      read_when: "if benchmark requires deployed runtime sync"
    - path: .codex/agents/.context/promt_manager_state.yaml
      read_when: "only to check for a newer executor handoff; ignore if stale/unrelated"

style_references:
  - path: docs/architecture/backtest/benchmark_iterations/README.md
    purpose: "benchmark summary style"
  - path: scripts
    purpose: "repo script style"

hard_requirements:
  depends_on_iteration_07: true
  macstudio_acceptance_required: true
  local_smoke_not_acceptance: true
  security_eval_required: true
  no_load_generator_on_macstudio: true
  model_concurrency_evidence_required: true

task_toggles:
  implement_load_harness: true
  implement_security_eval_harness: true
  run_scenarios_s1_s5_s10_s50_s100: true
  record_benchmark_summary: true
  publish_or_deploy: false

skill_routing:
  - skill: backend-performance-evidence
    use_when: "designing/running load, latency, throughput, memory or benchmark evidence"
    timing: "before implementation and during verification"
    reason: "performance evidence and comparability"
  - skill: backend-quality-gates
    use_when: "running harness unit tests, ruff, pyright"
    timing: "during verification"
    reason: "benchmark tooling quality gates"
  - skill: production-risk-review
    use_when: "before final report if benchmark suggests production rollout"
    timing: "before ship"
    reason: "capacity/security rollout risk"

target_envs:
  - local-dev
  - mac-studio
  - external-load-generator

required_literals:
  - "S1"
  - "S5"
  - "S10"
  - "S50"
  - "S100"
  - "final valid config rate"
  - "security eval mix"
  - "memory_pressure"
  - "vm_stat"
  - "active_generations"

non_goals:
  - "Do not claim acceptance from local-only tests."
  - "Do not run the load generator on Mac Studio."
  - "Do not roll out to paid users in this iteration."
  - "Do not increase concurrency without Mac Studio evidence."

final_report_format:
  language: ru
  sections:
    - "Что реализовано"
    - "Benchmark evidence"
    - "Accepted runtime settings"
    - "Security eval"
    - "Блокеры rollout"

quality_gates:
  - cmd: "uv run pytest -q tests/unit/scripts/test_backtest_ai_config_load_harness.py tests/unit/contexts/backtest/application/ai_configurator"
    expect: "passes if tests exist for harness"
  - cmd: "uv run ruff check scripts apps/worker src/trading/contexts/backtest tests/unit"
    expect: "passes"
  - cmd: "uv run pyright"
    expect: "passes"
  - cmd: "python -m tools.docs.generate_docs_index --check"
    expect: "passes if benchmark docs changed"
  - cmd: "git diff --check"
    expect: "passes"

expected_primary_touches:
  - "scripts/backtest_ai/run_configurator_load_test.py"
  - "scripts/backtest_ai/run_configurator_security_eval.py"
  - "tests/unit/scripts/test_backtest_ai_config_load_harness.py"
  - "docs/architecture/backtest/benchmark_iterations/"
  - "configs/prod/backtest_ai_configurator.yaml"

possible_secondary_touches:
  - "docs/runbooks/mac-studio-monitoring-plan.md"
  - "docs/architecture/backtest/backtest-ai-configurator-mlx-v1.md"
  - "apps/worker/backtest_ai_configurator/"

safety_notes:
  - "Mac Studio benchmark evidence is the acceptance gate."
  - "Benchmark output must include commit SHA, model_id, model_path hash, context, tokens, active_generations, latency, queue, valid rate, memory pressure."
  - "Security eval must require 0 unauthorized actions and 0 leakage."
---

# Task

Implement Iteration 08 of the `/backtests` AI Configurator: benchmark/load/security evaluation harness and Mac Studio evidence collection. Run S1/S5/S10/S50/S100 scenarios, record evidence under `docs/architecture/backtest/benchmark_iterations/`, and choose accepted model/concurrency/context settings.

Done means:

- a reusable load harness exists for API pipeline scenarios;
- a reusable security eval harness exists for prompt-injection/off-topic/output-injection/resource-abuse cases;
- benchmark prompts cover create/edit/repair/suggest safer/unsupported/off-topic cases;
- Mac Studio evidence is recorded for S1/S5/S10/S50/S100 or exact blockers are documented;
- accepted runtime settings are recorded: model id/path hash, context window, max output tokens, active generations, queue limits;
- rollout is blocked if Mac Studio evidence is missing or fails targets.

## Context / Current State

Context ledger:

- completed:
  - Iteration 07 should provide operable worker, metrics, health and ops files.
- open_items:
  - public rollout is not allowed until benchmark/security evidence is accepted.
- contract_changes:
  - benchmark artifacts and possibly config defaults after evidence.
- risks:
  - local smoke being mistaken for Mac Studio acceptance;
  - load generator consuming Mac Studio inference resources;
  - benchmark evidence missing commit/config identity.
- next_focus:
  - quantitative acceptance for production rollout.

## Requirements (Must)

- Use `backend-performance-evidence` before designing/running benchmark.
- Do not claim acceptance from local-only tests.
- Load generator must run outside Mac Studio or be clearly classified as non-acceptance if local constraints force a smoke.
- Scenarios must include S1, S5, S10, S50 and S100 online users.
- Prompt mix must include safe supported prompts, unsupported timeframe/indicator/symbol, edit, repair, suggest safer, off-topic and security attack prompts.
- Capture p50/p95/p99 total latency, queue wait, LLM latency, valid config rate, repair rate, quota/capacity responses, worker restarts, RSS/memory pressure/swap.
- Security eval requires 0 unauthorized actions, 0 private/system leakage, 0 rendered HTML/script, friendly blocked messages.
- Record evidence with commit SHA, model id, model path hash, active generations, context window, max output tokens, Mac Studio host status.
- If Mac Studio or model is unavailable, stop public rollout and report blocker.

## Requirements (Should)

- Keep harness dependency dev-only; do not add Locust/k6 as production dependency.
- Prefer `httpx.AsyncClient` script unless existing benchmark tooling suggests otherwise.
- Include JSON and Markdown summary output.
- Use conservative `active_generations=1` baseline before testing higher concurrency.

## Requirements (Nice-to-have)

- Add a quick local fake-worker mode for CI smoke of harness logic.
- Include Prometheus query snippets for metrics collection.

# Context acquisition protocol

Read only in this order and stop once sufficient:

1. `.codex/AGENTS.md`
2. `.codex/agents/.context/promt_manager_state.yaml` or latest state snapshot, if available and relevant
3. latest executor final report, if available
4. task entrypoints
5. only conditional bundles required by Mac Studio/runbook/API ambiguity
6. consult-if-needed references only for blockers

Do not eagerly preload all listed sources.

Pre-implementation reading target:

- `<= 8 files`
- `<= ~35k-50k tokens`

Stop reading once benchmark target, endpoint shape, metric sources and evidence format are clear.

# Reading manifest

Use the front-matter `context_sources` as the canonical reading map.

Read with this intent:

- `always_read`: repo rules and benchmark acceptance contract.
- `task_entrypoints`: scripts area, worker metrics, benchmark doc format, config.
- `conditional_bundles`: Mac Studio ops, API contract or existing benchmark style only when needed.
- `consult_if_needed`: deploy workflow only if sync/deploy is needed.

Do not convert this manifest into a broad mandatory reading list.

# Work plan (agent should follow)

Skill routing for this task:

- `backend-performance-evidence`: use before implementation and during measurement; owns evidence comparability.
- `backend-quality-gates`: use during harness test/lint/type verification.
- `production-risk-review`: use before final report if recommending rollout.

1. Define benchmark/eval plan and evidence schema.
2. Implement load harness and security eval harness.
3. Add tests/smoke for harness argument parsing/result aggregation.
4. Run local fake smoke to prove harness mechanics.
5. Run Mac Studio S1/S5/S10/S50/S100 only when runtime is deployed/available and load generator is off-host.
6. Record Markdown/JSON benchmark summary under benchmark iterations.
7. Update config defaults only if evidence supports the change.
8. Report accepted settings or rollout blocker.

# Acceptance criteria (Definition of Done)

- Harness can run S1/S5/S10/S50/S100 with realistic think times.
- Security eval pack includes direct injection, fake turns, system extraction, secrets, encoded instructions, HTML/script injection, auto-run attempts and huge prompts.
- Benchmark summary contains identity/config/hardware/metrics.
- Accepted settings are evidence-backed.
- Missing Mac Studio evidence is reported as blocker, not success.

# Implementation constraints

## Measurement

- Separate local smoke from Mac Studio acceptance.
- Compare only equivalent model/config/workload runs.
- Capture cold/warm state when relevant.

## Operations

- Do not run load generator on Mac Studio.
- Do not alter production quotas/concurrency without explicit accepted evidence.

## Security

- Security eval prompts must be synthetic and must not include real secrets.

# Files to indicate (expected touched areas)

Expected primary touches:

- `scripts/backtest_ai/run_configurator_load_test.py`
- `scripts/backtest_ai/run_configurator_security_eval.py`
- `tests/unit/scripts/test_backtest_ai_config_load_harness.py`
- `docs/architecture/backtest/benchmark_iterations/`

Possible secondary touches:

- `configs/prod/backtest_ai_configurator.yaml`
- `docs/runbooks/mac-studio-monitoring-plan.md`
- `docs/architecture/backtest/backtest-ai-configurator-mlx-v1.md`

# Non-goals

- No paid-tier rollout.
- No UI redesign.
- No new model provider.
- No production dependency on Locust/k6 unless explicitly approved.

# Quality gates (must run and pass)

- `uv run pytest -q tests/unit/scripts/test_backtest_ai_config_load_harness.py tests/unit/contexts/backtest/application/ai_configurator`
- `uv run ruff check scripts apps/worker src/trading/contexts/backtest tests/unit`
- `uv run pyright`
- `python -m tools.docs.generate_docs_index --check`
- `git diff --check`

Mac Studio benchmark commands and results must be included when run. If not run, state exact blocker and do not mark production acceptance complete.

# Final output: report format (strict)

Report in Russian with:

- `Что реализовано`: harness/eval/docs/config changes.
- `Benchmark evidence`: scenarios run, commands, host, config, metrics.
- `Accepted runtime settings`: model/concurrency/context/tokens/queue.
- `Security eval`: pass/fail and false-positive notes.
- `Блокеры rollout`: anything preventing public rollout.
