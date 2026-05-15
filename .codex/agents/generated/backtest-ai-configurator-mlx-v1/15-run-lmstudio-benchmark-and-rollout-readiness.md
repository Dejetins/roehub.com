---
prompt_name: backtest_ai_configurator_lmstudio_v1_15_benchmark_rollout_readiness
repo: roehub.com
branch: main
scope: "Iteration 15: rerun gated LM Studio benchmark/security acceptance on Mac Studio and decide internal rollout readiness for /backtests AI Configurator."

language:
  implementation: python_benchmark_ops
  agent_report: ru

context_sources:
  always_read:
    - path: .codex/AGENTS.md
      why: "benchmark and delivery contract"
    - path: docs/architecture/backtest/benchmark_iterations/2026-05-13_lmstudio_serving_recovery/cleanup_and_contract_sync.md
      why: "required cleanup evidence"
    - path: docs/architecture/backtest/benchmark_iterations/2026-05-13_lmstudio_serving_recovery/security_pipeline_readiness.md
      why: "required security and pipeline evidence"
  task_entrypoints:
    - path: scripts/backtest_ai/run_configurator_load_test.py
      why: "load benchmark harness"
      inspect_symbols:
        - "*"
    - path: scripts/backtest_ai/run_configurator_security_eval.py
      why: "security eval harness"
      inspect_symbols:
        - "*"
    - path: configs/prod/backtest_ai_configurator.yaml
      why: "runtime settings under test"
      inspect_symbols:
        - backtest_ai_configurator
    - path: docs/architecture/backtest/benchmark_iterations/2026-05-13_lmstudio_serving_recovery
      why: "accepted serving/readiness evidence"
      inspect_symbols:
        - "*"
  conditional_bundles:
    macstudio_ops:
      read_when: "before remote benchmark or verification"
      paths:
        - docs/runbooks/mac-studio-native-backend-operations.md
        - docs/runbooks/mac-studio-monitoring-plan.md
    api_auth:
      read_when: "if benchmark auth/session is missing"
      paths:
        - apps/api/routes/backtest_ai_config.py
        - tests/unit/apps/api/test_backtest_ai_config_routes.py
  consult_if_needed:
    - path: docs/architecture/backtest/backtest-ai-configurator-mlx-v1.md
      read_when: "if acceptance targets conflict"

style_references:
  - docs/architecture/backtest/benchmark_iterations/2026-05-12_iteration_08_ai_configurator_load_security/load_benchmark_summary.md
  - docs/architecture/backtest/benchmark_iterations/2026-05-12_iteration_08_ai_configurator_load_security/security_eval_summary.md

documentation_continuity:
  old_current_docs:
    - docs/architecture/backtest/benchmark_iterations/2026-05-12_iteration_08_ai_configurator_load_security/README.md
    - docs/architecture/backtest/backtest-ai-configurator-mlx-v1.md
  new_doc_artifact: docs/architecture/backtest/benchmark_iterations/2026-05-13_lmstudio_serving_recovery/lmstudio_load_benchmark_acceptance.md
  canonical_shape: "benchmark markdown and JSON evidence"
  docs_gate: "uv run python -m tools.docs.generate_docs_index --check"

hard_requirements:
  depends_on_iteration_14_accepted: true
  macstudio_benchmark_acceptance_required: true
  sequential_scenario_gating_required: true
  security_eval_required: true
  port_conflict_preflight_required: true
  no_local_smoke_acceptance: true
  no_paid_rollout: true
  publish_ci_deploy_required: true
  macstudio_sync_required: true

task_toggles:
  run_serving_preflight: true
  run_s1_first: true
  run_s5_s10_after_s1: true
  run_s50_s100_after_s10: true
  record_accepted_settings: true
  enable_internal_rollout_only_if_passed: true

skill_routing:
  - skill: backend-performance-evidence
    use_when: "running and evaluating benchmark scenarios"
    timing: "before implementation and during verification"
    reason: "benchmark acceptance quality"
  - skill: production-risk-review
    use_when: "before readiness verdict or internal rollout"
    timing: "before ship"
    reason: "security/capacity rollout risk"
  - skill: backend-quality-gates
    use_when: "running harness tests and local gates"
    timing: "during verification"
    reason: "test gates"
  - skill: publish-ci-deploy
    use_when: "after evidence/docs/config changes and acceptance checks pass"
    timing: "final delivery step"
    reason: "ship evidence/config and verify Mac Studio"

target_envs:
  - local-dev
  - external-load-generator
  - mac-studio-prod
  - github-actions

required_literals:
  - "S1"
  - "S5"
  - "S10"
  - "S50"
  - "S100"
  - "final valid config rate"
  - "LM Studio"
  - "gemma-4-e2b-it-4bit"
  - "unauthorized actions: 0"
  - "memory_pressure"
  - "vm_stat"
  - "POST /v1/chat/completions"
  - "choices[0].message.content"
  - "JSON Schema type values must be strings"
  - "do not use type: [\"string\", \"null\"]"
  - "accepted: true/false"
  - "blocking_reason"
  - "next_prompt_allowed"

non_goals:
  - "Do not edit old prompt files 01-09."
  - "Do not run load generator on Mac Studio."
  - "Do not roll out to paid users."
  - "Do not accept benchmark if serving preflight fails."

final_report_format:
  language: ru
  sections:
    - "Readiness verdict"
    - "Benchmark evidence"
    - "Security eval"
    - "Accepted runtime settings"
    - "Доставка и Mac Studio"

quality_gates:
  - cmd: "uv run pytest -q tests/unit/scripts/test_backtest_ai_config_load_harness.py tests/unit/contexts/backtest/application/ai_configurator"
    expect: "passes"
  - cmd: "uv run ruff check scripts apps/worker src/trading/contexts/backtest tests/unit"
    expect: "passes"
  - cmd: "uv run pyright"
    expect: "passes"
  - cmd: "uv run python -m tools.docs.generate_docs_index --check"
    expect: "passes"
  - cmd: "git diff --check"
    expect: "passes"

expected_primary_touches:
  - scripts/backtest_ai/run_configurator_load_test.py
  - scripts/backtest_ai/run_configurator_security_eval.py
  - docs/architecture/backtest/benchmark_iterations/2026-05-13_lmstudio_serving_recovery/
  - configs/prod/backtest_ai_configurator.yaml

possible_secondary_touches:
  - tests/unit/scripts/test_backtest_ai_config_load_harness.py
  - docs/architecture/backtest/backtest-ai-configurator-mlx-v1.md
  - docs/runbooks/mac-studio-monitoring-plan.md

safety_notes:
  - "Benchmark acceptance starts only after serving and one-job pipeline preflight pass."
  - "S50/S100 are downstream of green S1/S5/S10."
  - "Local fake-worker smoke is never acceptance evidence."
---

# Task

Run the final gated benchmark and security acceptance for the LM Studio-backed `/backtests` AI Configurator.

This prompt starts only after Iteration 14 cleanup evidence exists and passed. It must not repeat the Iteration 08 mistake: benchmark starts only after serving preflight and one supported real job are green on Mac Studio.

Done means:

- LM Studio serving preflight passes on Mac Studio;
- one supported real `/backtests/ai-config` job reaches `ready`;
- security eval has unauthorized actions 0;
- S1 passes before S5/S10;
- S5/S10 pass before S50/S100;
- accepted runtime settings are recorded or rollout remains blocked;
- evidence is written in JSON and Markdown.

## Context / Current State

Context ledger:

- completed:
  - LM Studio serving, adapter, lifecycle, security, pipeline readiness and cleanup should be accepted.
- open_items:
  - failed Iteration 08 is historical and not acceptance.
  - accepted model/concurrency/context settings still need benchmark evidence.
- contract_changes:
  - benchmark may update internal config defaults or keep feature disabled.
- risks:
  - starting higher scenarios after failed S1;
  - load generator running on Mac Studio;
  - accepting local smoke as production evidence.
- next_focus:
  - quantitative internal rollout readiness.

## Requirements (Must)

- Stop if Iteration 14 cleanup evidence is missing or blocked.
- Run serving preflight first: configured `base_url` port is free or owned by LM Studio, LM Studio daemon/server/model loaded/structured generation. The structured generation probe must use `POST /v1/chat/completions`, `response_format.type=json_schema`, string-only JSON Schema `type` values, and parse `choices[0].message.content` as JSON.
- Run one supported real API job and require `ready` before S1.
- Run security eval and require unauthorized actions 0 before S1.
- Apply explicit benchmark thresholds; do not use vague "architecture target" wording:
  - all scenarios: `final valid config rate >= 98%`;
  - all scenarios: `unauthorized actions = 0`, `private/system leakage = 0`, `rendered HTML/script = 0`;
  - all scenarios: `queue timeout rate <= 2%`;
  - all scenarios: `unexpected HTTP 5xx rate <= 1%` and every 5xx has a classified cause;
  - supported prompt repair rate: `<= 25%` unless a higher rate is documented as model-quality blocker;
  - S1/S5/S10: `p95 total latency <= 45s` and `p95 queue wait <= 15s`;
  - S50/S100: `p95 total latency <= 90s` and `p95 queue wait <= 45s`;
  - worker/LM Studio restarts during a scenario: `0`;
  - memory: no sustained macOS critical memory pressure, no sustained swap-in/out after warmup, and swap growth `<= 1 GiB` per scenario unless marked capacity-blocked.
- Run scenarios sequentially:
  - S1 first;
  - S5/S10 only after S1 passes;
  - S50/S100 only after S5/S10 pass.
- Load generator must not run on Mac Studio.
- Capture p50/p95/p99 total latency, queue wait, LLM latency, final valid config rate, repair rate, capacity/quota, worker restarts, RSS, memory pressure, swap.
- Record commit SHA, config SHA, model id, model path hash, context, max output tokens, `active_generations`, LM Studio loaded instance, host status.
- If any gate fails, stop and record blocker; do not continue to larger scenarios.
- Markdown and JSON evidence must include explicit machine-readable gate fields: `accepted: true/false`, `blocking_reason: null|string`, and `next_prompt_allowed: true/false`.
- Run `publish-ci-deploy` after evidence/config/docs changes and gates pass.

## Requirements (Should)

- Keep benchmark auth/session handling documented and redacted.
- Keep accepted settings conservative for MVP.
- Prefer internal/admin rollout verdict over paid-user rollout.
- Include rollback: disable feature flag, stop worker, stop LM Studio runtime if needed.

## Requirements (Nice-to-have)

- Include small Grafana/Prometheus query snippets if metrics were useful.
- Include before/after `memory_pressure` snapshots.

# Context acquisition protocol

Read only in this order and stop once sufficient:

1. `.codex/AGENTS.md`
2. Iteration 14 cleanup evidence
3. Iteration 13 security/pipeline readiness evidence
4. task entrypoints
5. Mac Studio ops bundle only before remote benchmark
6. API auth bundle only if benchmark auth blocks execution
7. architecture doc only if targets conflict

Do not preload all previous benchmark folders.

Reading budget: max 10 repo files plus the required evidence files.

# Reading manifest

- `always_read`: repo contract, cleanup evidence, security/pipeline evidence.
- `task_entrypoints`: load harness, security harness, prod config, accepted evidence folder.
- `conditional_bundles`: Mac Studio ops and API auth only when needed.
- `consult_if_needed`: architecture doc only for threshold conflicts.

Stop reading once benchmark preflight, scenario sequence and evidence target are clear.

# Work plan (agent should follow)

1. Verify Iteration 14 and Iteration 13 evidence.
2. Run local harness unit gates.
3. Verify Mac Studio LM Studio runtime and worker readiness.
4. Run one real supported job to `ready`.
5. Run security eval; stop if unauthorized actions > 0.
6. Run S1; stop if it fails acceptance.
7. Run S5/S10; stop if either fails acceptance.
8. Run S50/S100 only if smaller scenarios pass.
9. Record accepted settings or blockers.
10. Use `publish-ci-deploy`.

# Acceptance criteria (Definition of Done)

- Serving preflight passes on Mac Studio.
- Configured LM Studio port preflight passes on Mac Studio and is recorded in evidence.
- One supported real job reaches `ready`.
- Security eval unauthorized actions: 0.
- S1/S5/S10/S50/S100 evidence is recorded or execution stops at first failed gate with blocker.
- Threshold verdict is explicit for every executed scenario:
  - `final valid config rate >= 98%`;
  - `queue timeout rate <= 2%`;
  - `unexpected HTTP 5xx rate <= 1%`;
  - S1/S5/S10 `p95 total latency <= 45s` and `p95 queue wait <= 15s`;
  - S50/S100 `p95 total latency <= 90s` and `p95 queue wait <= 45s`;
  - worker/LM Studio restarts `0`;
  - no sustained critical memory pressure and swap growth `<= 1 GiB` per scenario.
- Security leakage metrics are all zero: unauthorized actions, private/system leakage, rendered HTML/script.
- Accepted runtime settings are explicitly recorded.
- Evidence contains top-level gate markers: `accepted`, `blocking_reason`, and `next_prompt_allowed`; rollout may proceed only when `accepted=true` and `next_prompt_allowed=true`.
- `publish-ci-deploy` reaches `deployed`, or exact `green-pr`/`blocked` state is recorded.

# Implementation constraints

## Benchmark discipline

- Do not run load generator on Mac Studio.
- Do not continue to larger scenarios after a failed smaller gate.
- Do not classify local fake-worker runs as acceptance.

## Documentation

- Create `lmstudio_load_benchmark_acceptance.md` and matching JSON.
- Mark failed scenarios clearly.
- Run docs index check.

## Rollout

- Paid-tier rollout remains out of scope.
- Internal/admin enablement is allowed only if all acceptance gates pass and feature flag remains controlled.

# Files to indicate (expected touched areas)

Expected primary touches:

- `docs/architecture/backtest/benchmark_iterations/2026-05-13_lmstudio_serving_recovery/lmstudio_load_benchmark_acceptance.md`
- `docs/architecture/backtest/benchmark_iterations/2026-05-13_lmstudio_serving_recovery/lmstudio_load_benchmark_acceptance.json`
- `configs/prod/backtest_ai_configurator.yaml`

Possible secondary touches:

- `scripts/backtest_ai/run_configurator_load_test.py`
- `scripts/backtest_ai/run_configurator_security_eval.py`
- `tests/unit/scripts/test_backtest_ai_config_load_harness.py`
- `docs/architecture/backtest/backtest-ai-configurator-mlx-v1.md`

# Non-goals

- No paid rollout.
- No LM Studio UI embedding.
- No new runtime architecture.
- No old prompt edits.

# Quality gates (must run and pass)

- `uv run pytest -q tests/unit/scripts/test_backtest_ai_config_load_harness.py tests/unit/contexts/backtest/application/ai_configurator`
- `uv run ruff check scripts apps/worker src/trading/contexts/backtest tests/unit`
- `uv run pyright`
- `uv run python -m tools.docs.generate_docs_index --check`
- `git diff --check`
- Mac Studio serving preflight, one-job ready smoke, security eval, and gated S1/S5/S10/S50/S100 benchmark.

If any Mac Studio acceptance gate fails, stop at that gate and report blocker.

# Final output: report format (strict)

Report in Russian with:

- `Readiness verdict`: accepted/internal-only/blocked and why.
- `Benchmark evidence`: scenario table, valid rate, latency, queue, memory.
- `Security eval`: unauthorized actions, leakage, rendered HTML/script.
- `Accepted runtime settings`: model, context, tokens, concurrency, queue limits.
- `Evidence marker`: `accepted`, `blocking_reason`, `next_prompt_allowed`.
- `Доставка и Mac Studio`: publish-ci-deploy state, CI, Mac Studio sync/smoke.
