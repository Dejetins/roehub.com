---
prompt_name: backtest_ai_configurator_mlx_v1_07_observability_ops_training_export
repo: roehub.com
branch: main
scope: "Iteration 07: implement metrics, structured logs, health/readiness, training export/redaction, launchd/Monit service files, Prometheus target, and runbook updates."

language:
  implementation: python_macos_ops
  agent_report: ru

context_sources:
  always_read:
    - path: .codex/AGENTS.md
      why: "repo and ops safety rules"
    - path: docs/architecture/backtest/backtest-ai-configurator-mlx-v1.md
      why: "observability and Monit contract"
  task_entrypoints:
    - path: apps/worker/backtest_ai_configurator
      why: "worker runtime to expose health/metrics"
      inspect_symbols:
        - "*"
    - path: infra/macos/prometheus/prometheus.prod.yml
      why: "Prometheus target source of truth"
      inspect_symbols:
        - scrape_configs
    - path: infra/scripts/monit
      why: "Monit snippet and launchctl wrapper style"
      inspect_symbols:
        - "*.monitrc"
    - path: scripts/macos/bootstrap_native_prod.sh
      why: "native prod install path"
      inspect_symbols:
        - "*"
  conditional_bundles:
    launchd_reload:
      read_when: "when adding plist or reload service list"
      paths:
        - infra/macos/launchd
        - scripts/macos/reload_launchd_services.sh
    runbooks:
      read_when: "when documenting native service or monitoring changes"
      paths:
        - docs/runbooks/mac-studio-native-backend-operations.md
        - docs/runbooks/mac-studio-monitoring-plan.md
    metrics_patterns:
      read_when: "when choosing metrics helper/export style"
      paths:
        - apps/worker
        - apps/scheduler
        - tests/unit/apps/scheduler
  consult_if_needed:
    - path: .github/workflows/deploy-backend.yml
      read_when: "if bootstrap/deploy path needs updating"
    - path: .codex/agents/.context/promt_manager_state.yaml
      read_when: "only to check for a newer executor handoff; ignore if stale/unrelated"

style_references:
  - path: infra/scripts/monit/roehub-market-data.monitrc
    purpose: "Monit process supervision style"
  - path: infra/macos/launchd/com.roehub.market-data-ws-worker.plist
    purpose: "LaunchAgent style"

hard_requirements:
  depends_on_iteration_06: true
  health_ready_metrics_required: true
  prometheus_target_required: true
  monit_control_required: true
  launchd_autostart_required: true
  training_export_redaction_required: true
  no_public_model_runtime: true
  publish_ci_deploy_required: true
  main_branch_deployment_required: true
  macstudio_sync_required: true

task_toggles:
  implement_metrics: true
  implement_health_endpoints: true
  implement_training_export: true
  implement_launchd: true
  implement_monit: true
  update_runbooks: true
  run_macstudio_service_install: false

skill_routing:
  - skill: contract-impact-analysis
    use_when: "adding config, ops files, metrics names or training export behavior"
    timing: "before implementation"
    reason: "operations, monitoring and data export contracts"
  - skill: backend-quality-gates
    use_when: "running worker/export/metrics tests and lint/type gates"
    timing: "during verification"
    reason: "backend and ops code quality"
  - skill: production-risk-review
    use_when: "before final report for Monit/launchd/training-data risk"
    timing: "before ship"
    reason: "production service and privacy risk"
  - skill: publish-ci-deploy
    use_when: "after implementation and local gates pass, deliver this iteration to main, sync Mac Studio, and run post-deploy verification"
    timing: "final delivery step"
    reason: "required end-to-end Roehub GitHub CI, main deployment, Mac Studio sync and smoke"

target_envs:
  - local-dev
  - unit-tests
  - mac-studio-prod-after-deploy
  - github-actions
  - mac-studio-prod

required_literals:
  - "com.roehub.backtest-ai-configurator-worker"
  - "roehub_backtest_ai_configurator_worker"
  - "127.0.0.1:9205"
  - "/health/live"
  - "/health/ready"
  - "/metrics"
  - "backtest-ai-configurator-worker"

non_goals:
  - "Do not run production deployment unless explicitly asked."
  - "Do not publish or merge."
  - "Do not add non-MLX runtime."
  - "Do not expose raw audit rows through public API."

final_report_format:
  language: ru
  sections:
    - "Что реализовано"
    - "Ops/monitoring contract"
    - "Training export"
    - "Проверки"
    - "Доставка и Mac Studio"
    - "Следующая итерация"

quality_gates:
  - cmd: "uv run pytest -q tests/unit/apps/worker/test_backtest_ai_configurator_worker.py tests/unit/contexts/backtest/application/ai_configurator tests/unit/apps/api/test_backtest_ai_config_routes.py"
    expect: "passes"
  - cmd: "uv run ruff check apps/worker apps/api src/trading/contexts/backtest tests/unit"
    expect: "passes"
  - cmd: "uv run pyright"
    expect: "passes"
  - cmd: "python -m tools.docs.generate_docs_index --check"
    expect: "passes if docs changed"
  - cmd: "git diff --check"
    expect: "passes"

expected_primary_touches:
  - "apps/worker/backtest_ai_configurator/"
  - "src/trading/contexts/backtest/application/ai_configurator/"
  - "infra/macos/launchd/com.roehub.backtest-ai-configurator-worker.plist"
  - "infra/scripts/monit/roehub-backtest-ai-configurator.monitrc"
  - "infra/macos/prometheus/prometheus.prod.yml"
  - "docs/runbooks/mac-studio-native-backend-operations.md"
  - "docs/runbooks/mac-studio-monitoring-plan.md"

possible_secondary_touches:
  - "scripts/macos/bootstrap_native_prod.sh"
  - "scripts/macos/reload_launchd_services.sh"
  - "tests/unit/apps/worker/test_backtest_ai_configurator_worker.py"
  - "tests/unit/contexts/backtest/application/ai_configurator/"

safety_notes:
  - "Monit controls service via launchctl_service_control.sh."
  - "Prometheus target is loopback 127.0.0.1:9205 only."
  - "Training export must scrub secrets/private topology and select only labeled safe rows."
---

# Task

Implement Iteration 07 of the `/backtests` AI Configurator: production observability and operations scaffolding. Add worker health/readiness/metrics, structured logging, training export/redaction, launchd plist, Monit snippet, Prometheus target and runbook updates.

Done means:

- worker exposes `GET /health/live`, `GET /health/ready`, and `GET /metrics` on `127.0.0.1:9205`;
- readiness checks config, model registry/path/runtime connection, Postgres queue/audit access, queue loop and drain mode without running heavy generation;
- Prometheus metrics include job counts, inflight, queue depth, active generations, stage durations, LLM latency, total latency, validation failures, repair attempts, security decisions, quota/capacity rejections, applied count, model reload and process metrics;
- launchd plist starts worker with `RunAtLoad=true` and `KeepAlive=true`;
- Monit snippet can start/stop/restart via `launchctl_service_control.sh` and restarts on failed readiness/metrics;
- Prometheus prod config scrapes `backtest-ai-configurator-worker`;
- training export produces scrubbed, labeled rows and excludes secrets/private infra/raw debug dumps;
- runbooks mention service, commands and monitoring target.

## Context / Current State

Context ledger:

- completed:
  - Iteration 06 should provide end-to-end local user flow and worker runtime.
- open_items:
  - benchmark/load evidence is later;
  - production deploy/install should only happen when explicitly requested.
- contract_changes:
  - new operations files, metrics names, health endpoints and training export path.
- risks:
  - service starts automatically before smoke is accepted;
  - Monit/launchd conflict or restart storm;
  - training export leaks private data.
- next_focus:
  - operable service ready for benchmark and rollout.

## Requirements (Must)

- Verify worker runtime exists; if not, stop and report blocker.
- Add health endpoints and metrics endpoint.
- Add structured logs without raw prompts/model output outside restricted audit.
- Add training export with redaction and quality labels.
- Add launchd plist `com.roehub.backtest-ai-configurator-worker`.
- Add Monit snippet `roehub-backtest-ai-configurator.monitrc`.
- Add Prometheus target `backtest-ai-configurator-worker` on `127.0.0.1:9205`.
- Update bootstrap/reload scripts only if required by current native service pattern.
- Update runbooks and docs index if docs changed.
- Do not run production install/deploy unless explicitly asked.

## Requirements (Should)

- Keep metrics labels bounded; avoid user ids or prompt text as labels.
- Include drain mode in readiness.
- Keep training export as an admin/ops command or internal use case, not public endpoint.
- Include smoke commands in runbook.

## Requirements (Nice-to-have)

- Add alert rule snippets if repository has a current rules file for Mac Studio.
- Add a metrics unit test that asserts metric names exist after a fake job.

# Context acquisition protocol

Read only in this order and stop once sufficient:

1. `.codex/AGENTS.md`
2. `.codex/agents/.context/promt_manager_state.yaml` or latest state snapshot, if available and relevant
3. latest executor final report, if available
4. task entrypoints
5. only conditional bundles required by launchd/runbook/metrics ambiguity
6. consult-if-needed references only for blockers

Do not eagerly preload all listed sources.

Pre-implementation reading target:

- `<= 8 files`
- `<= ~35k-50k tokens`

Stop reading once worker health surface, Monit style, Prometheus target and export boundary are clear.

# Reading manifest

Use the front-matter `context_sources` as the canonical reading map.

Read with this intent:

- `always_read`: repo rules and ops contract.
- `task_entrypoints`: worker, Prometheus, Monit and bootstrap entrypoints.
- `conditional_bundles`: launchd, runbooks, metrics patterns only when needed.
- `consult_if_needed`: deploy workflow only if bootstrap path changes.

Do not convert this manifest into a broad mandatory reading list.

# Work plan (agent should follow)

Skill routing for this task:

- `contract-impact-analysis`: use before implementation; owns metrics/config/ops/export compatibility.
- `backend-quality-gates`: use during verification; owns tests/lint/type checks.
- `production-risk-review`: use before final report; owns launchd/Monit/privacy risk.

1. Verify current worker and metrics/ops patterns.
2. Add health/readiness/metrics in worker.
3. Add structured logs and bounded metric labels.
4. Add training export/redaction path with tests.
5. Add launchd plist, Monit snippet, Prometheus target and script updates if needed.
6. Update runbooks and docs index.
7. Run gates and report explicit deploy/install status.

# Acceptance criteria (Definition of Done)

- `/health/live`, `/health/ready`, `/metrics` work in unit/local smoke.
- Metrics names match architecture contract and have bounded labels.
- Training export excludes secrets/private topology/model base URL/raw tracebacks.
- Launchd plist and Monit snippet pass syntax/lint checks where available.
- Prometheus target is added without removing existing targets.
- Runbooks include commands for Monit summary/status/restart, launchctl print, curl health/metrics.

- `publish-ci-deploy` terminal state is `deployed`, or `green-pr`/`blocked` is reported with exact blocker evidence.

# Implementation constraints

## Determinism & ordering

- Readiness must not run heavy model generation.
- Metrics should be deterministic in tests after fake job events.

## Operations

- Use existing user-level LaunchAgent pattern.
- Do not add Docker/Colima runtime.
- Do not force production service into reload baseline before smoke if local policy requires staged enablement; document the chosen path.

## Data safety

- Training export must scrub or exclude secrets/private topology.
- No user prompt text in Prometheus labels or general logs.

# Files to indicate (expected touched areas)

Expected primary touches:

- `apps/worker/backtest_ai_configurator/`
- `src/trading/contexts/backtest/application/ai_configurator/`
- `infra/macos/launchd/com.roehub.backtest-ai-configurator-worker.plist`
- `infra/scripts/monit/roehub-backtest-ai-configurator.monitrc`
- `infra/macos/prometheus/prometheus.prod.yml`
- `docs/runbooks/mac-studio-native-backend-operations.md`
- `docs/runbooks/mac-studio-monitoring-plan.md`

Possible secondary touches:

- `scripts/macos/bootstrap_native_prod.sh`
- `scripts/macos/reload_launchd_services.sh`
- `tests/unit/apps/worker/test_backtest_ai_configurator_worker.py`
- `docs/architecture/README.md`

# Non-goals

- No public rollout.
- No Mac Studio load benchmark.
- No UI redesign.
- No remote model fallback.

# Quality gates (must run and pass)

- `uv run pytest -q tests/unit/apps/worker/test_backtest_ai_configurator_worker.py tests/unit/contexts/backtest/application/ai_configurator tests/unit/apps/api/test_backtest_ai_config_routes.py`
- `uv run ruff check apps/worker apps/api src/trading/contexts/backtest tests/unit`
- `uv run pyright`
- `python -m tools.docs.generate_docs_index --check`
- `git diff --check`

If production service install is not run, state that clearly. Do not imply Mac Studio service is live.

Required delivery step: after the quality gates above pass, invoke `publish-ci-deploy` as the final step. The expected terminal state for this prompt is `deployed`: intended files committed and pushed, GitHub Actions green, revision shipped to `main`, `/opt/roehub/app` on `macstudio` pulled to that revision, the relevant production services reloaded through the repository runbook, and `bash scripts/macos/smoke_prod.sh` passed. If the skill reaches `green-pr` because a human merge/approval is required, or `blocked` because of missing auth, unrelated dirty scope, external CI, Mac Studio access, or production verification failure, report that exact state and do not claim deployment.

# Final output: report format (strict)

Report in Russian with:

- `Что реализовано`: health, metrics, export, ops files, docs.
- `Ops/monitoring contract`: ports, labels, Monit/launchd/Prometheus changes.
- `Training export`: what is included/excluded and tests.
- `Проверки`: exact commands and results.
- `Доставка и Mac Studio`: publish-ci-deploy terminal state, main/PR SHA, CI result, Mac Studio pull/reload/smoke evidence, or exact blocker.
- `Следующая итерация`: benchmark/load/security evidence.
