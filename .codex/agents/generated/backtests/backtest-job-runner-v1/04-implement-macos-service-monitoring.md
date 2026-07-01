---
prompt_name: backtest_job_runner_v1_04_implement_macos_service_monitoring
repo: roehub.com
branch: main
scope: "R4: add Mac Studio launchd service, reload/bootstrap integration, Prometheus target and runbook updates for `backtest-job-runner`."

language:
  implementation: bash_plist_yaml_docs
  agent_report: ru

context_sources:
  always_read:
    - path: .codex/AGENTS.md
      why: "repo contract"
    - path: docs/architecture/backtest/backtest-job-runner-production-plan-v1.md
      why: "R4 source of truth"
    - path: .codex/agents/.context/promt_manager_state.yaml
      why: "latest compact state if present"
  task_entrypoints:
    - path: scripts/macos/bootstrap_native_prod.sh
      why: "prod install surface"
      inspect_symbols:
        - install loop
    - path: scripts/macos/reload_launchd_services.sh
      why: "prod/test reload surface"
      inspect_symbols:
        - prod_services
        - test_services
        - collect_worker_services
    - path: infra/macos/launchd/com.roehub.market-data-ws-worker.plist
      why: "existing worker launchd pattern"
      inspect_symbols:
        - ProgramArguments
    - path: infra/macos/prometheus/prometheus.prod.yml
      why: "prod Prometheus target list"
      inspect_symbols:
        - scrape_configs
  conditional_bundles:
    test_profile:
      read_when: "adding test launchd/prometheus target"
      paths:
        - scripts/macos/bootstrap_native_test.sh
        - infra/macos/prometheus/prometheus.test.yml
        - infra/macos/launchd/com.roehub.test.market-data-ws-worker.plist
    monit:
      read_when: "Monit supervision is added or runbook mentions it"
      paths:
        - infra/scripts/monit
        - docs/runbooks/mac-studio-monitoring-plan.md
    runbooks:
      read_when: "operator docs need updates"
      paths:
        - docs/runbooks/mac-studio-native-backend-operations.md
        - docs/runbooks/mac-studio-monitoring-plan.md
  consult_if_needed:
    - path: apps/worker/backtest_job_runner/main/main.py
      read_when: "CLI args/env for launchd are unclear"
    - path: docs/runbooks/keycloak-local-setup-and-ops.md
      read_when: "launchd bootstrap pattern is unclear"

style_references: []

hard_requirements:
  launchd_prod_service_required: true
  reload_must_not_remove_new_runner: true
  metrics_target_9204_required: true
  logs_paths_required: true
  no_runtime_enable_without_tests: true
  runbook_update_required: true

task_toggles:
  implement_launchd: true
  implement_prometheus_target: true
  implement_monit_optional: true
  implement_worker_code: false
  publish_after_success: true

skill_routing:
  - skill: production-risk-review
    use_when: "reviewing launchd/reload/monitoring diff before final report"
    timing: "before ship"
    reason: "production service lifecycle risk"
  - skill: contract-impact-analysis
    use_when: "changing config, runtime workflow, monitoring targets or runbooks"
    timing: "before final report"
    reason: "ops contract impact"
  - skill: backend-quality-gates
    use_when: "running bash/plist/docs and focused tests"
    timing: "during verification"
    reason: "ops change verification"
  - skill: publish-ci-deploy
    use_when: "all local gates pass and publish_after_success is true"
    timing: "before ship"
    reason: "direct-main delivery and Mac Studio sync"

target_envs:
  - local-dev
  - mac-studio
  - github-actions

required_literals:
  - "com.roehub.backtest-job-runner"
  - "com.roehub.test.backtest-job-runner"
  - "ROEHUB_BACKTEST_RUNNER_METRICS_PORT=9204"
  - "127.0.0.1:9204"
  - "/Users/daniildegtyarev/Library/Logs/roehub/backtest-job-runner.out.log"
  - "scripts/macos/reload_launchd_services.sh prod"

non_goals:
  - "Do not change `/backtests` UI."
  - "Do not add a second detail-runner process."
  - "Do not process production backlog in this prompt."
  - "Do not claim production smoke; R5 owns smoke/load acceptance."

final_report_format:
  language: ru
  sections:
    - "Intent"
    - "Scope"
    - "Design"
    - "Contract impact"
    - "Tests"
    - "Runtime evidence"
    - "Risks"
    - "Handoff"
    - "Publish/deploy"

quality_gates:
  - cmd: "plutil -lint infra/macos/launchd/com.roehub.backtest-job-runner.plist infra/macos/launchd/com.roehub.test.backtest-job-runner.plist"
    expect: "passes if both plists are added"
  - cmd: "bash -n scripts/macos/bootstrap_native_prod.sh scripts/macos/bootstrap_native_test.sh scripts/macos/reload_launchd_services.sh"
    expect: "passes"
  - cmd: "uv run pytest -q tests/unit/apps/worker/backtest_job_runner tests/unit/contexts/backtest/application/use_cases/test_backtest_job_worker_use_case.py"
    expect: "passes or exact pre-existing failure classification"
  - cmd: "uv run ruff check apps/worker tests/unit/apps/worker"
    expect: "passes"
  - cmd: "uv run python -m tools.docs.generate_docs_index --check"
    expect: "passes"

expected_primary_touches:
  - "infra/macos/launchd/com.roehub.backtest-job-runner.plist"
  - "infra/macos/launchd/com.roehub.test.backtest-job-runner.plist"
  - "scripts/macos/bootstrap_native_prod.sh"
  - "scripts/macos/bootstrap_native_test.sh"
  - "scripts/macos/reload_launchd_services.sh"
  - "infra/macos/prometheus/prometheus.prod.yml"
  - "infra/macos/prometheus/prometheus.test.yml"
  - "docs/runbooks/mac-studio-native-backend-operations.md"
  - "docs/runbooks/mac-studio-monitoring-plan.md"

possible_secondary_touches:
  - "infra/scripts/monit/*backtest*"
  - "docs/architecture/backtest/backtest-job-runner-production-plan-v1.md"
  - "tests/unit/apps/worker/backtest_job_runner/**"

safety_notes:
  - "The reload script currently removes legacy backtest-job-runner plists; do not let it remove the new static service."
  - "R5 owns actual production smoke after deployment."
---

# Task

Implement Mac Studio service and monitoring configuration for `backtest-job-runner`.

Done means:

- launchd plists exist for prod and test profiles;
- bootstrap/reload scripts install and reload them safely;
- Prometheus target `127.0.0.1:9204` exists for prod and `127.0.0.1:19204` for test if test profile is added;
- runbooks explain start/stop/logs/metrics/smoke boundaries;
- local plist/script/docs gates pass.

## Context / Current State

The runner plan says `launchd` owns the process. Current `reload_launchd_services.sh` removes legacy `backtest-job-runner.*` plists and does not include a new static runner service. This prompt must replace that legacy exclusion with explicit static service management.

## Requirements (Must)

- Add `infra/macos/launchd/com.roehub.backtest-job-runner.plist`.
- Add test plist if the repository test profile is maintained for adjacent workers.
- Update `bootstrap_native_prod.sh` and test bootstrap if applicable.
- Update `reload_launchd_services.sh` so the new runner service is static and not removed by legacy cleanup.
- Add Prometheus scrape target for prod `backtest-job-runner` at `127.0.0.1:9204`.
- Add test target at `127.0.0.1:19204` if test plist is added.
- Add or update runbook steps for logs, metrics, launchctl print, restart/reload and rollback.
- Keep Monit optional unless existing monitoring convention requires it.

## Requirements (Should)

- Use the same environment loading and `/opt/roehub/app` working directory pattern as other services.
- Use `ROEHUB_BACKTEST_RUNNER_*` env vars and `--metrics-port` consistently.
- Keep labels and metric targets low-cardinality.

## Requirements (Nice-to-have)

- Add a small script/test assertion that prevents future reload scripts from deleting the new static runner service.

# Context acquisition protocol

Read only in this order and stop once sufficient:

1. `.codex/AGENTS.md`
2. `.codex/agents/.context/promt_manager_state.yaml` or latest state snapshot, if available
3. latest executor final report, if available
4. task entrypoints
5. only the conditional bundle(s) required by touched contracts or failing checks
6. consult-if-needed references only for blockers, ambiguity, or conflicts

Do not eagerly preload all listed sources.

Pre-implementation reading target:

- `<= 8 files`
- `<= ~45k tokens`

Stop reading once plist, reload, Prometheus and runbook changes are bounded.

# Reading manifest

Use the front-matter `context_sources` as the canonical reading map.

# Work plan (agent should follow)

Skill routing for this task:

- `production-risk-review`: use before ship for launchd/reload risk.
- `contract-impact-analysis`: use before final report for config/runtime/monitoring impacts.
- `backend-quality-gates`: use during verification.
- `publish-ci-deploy`: use before ship only after all gates pass.

1. Inspect existing launchd/reload/bootstrap patterns.
2. Add plists and script entries.
3. Add Prometheus targets.
4. Update runbooks.
5. Run plist/bash/docs/focused tests.
6. Publish via direct-main flow if fully green.

# Acceptance criteria (Definition of Done)

- `plutil -lint` passes for new plists.
- `bash -n` passes for changed scripts.
- New runner is part of static reload baseline.
- New runner is not caught by legacy removal.
- Prometheus target names and ports match the runner plan.
- Runbooks document the operational path.

# Implementation constraints

- Do not bootstrap production service during local implementation unless the prompt explicitly enters delivery mode after all gates.
- Do not include secrets in plist.
- Keep repo checkout and runtime bundle surfaces distinct in final report.

# Files to indicate (expected touched areas)

Use `expected_primary_touches` and `possible_secondary_touches` from front matter.

# Non-goals

See front matter `non_goals`.

# Quality gates (must run and pass)

Run the `quality_gates` commands from front matter. If a gate fails, classify it as `introduced`, `required-path pre-existing`, `unrelated pre-existing`, `environmental`, or `flaky`.

# Final output: report format (strict)

Report in Russian using the `final_report_format.sections` order. Include exact launchd labels, metrics ports, changed scripts, and delivery state.
