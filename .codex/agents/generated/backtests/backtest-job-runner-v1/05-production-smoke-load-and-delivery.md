---
prompt_name: backtest_job_runner_v1_05_production_smoke_load_and_delivery
repo: roehub.com
branch: main
scope: "R5: final production smoke/load evidence and direct-main delivery for `backtest-job-runner-v1`."

language:
  implementation: verification_deploy_ops
  agent_report: ru

context_sources:
  always_read:
    - path: .codex/AGENTS.md
      why: "repo contract"
    - path: docs/architecture/backtest/backtest-job-runner-production-plan-v1.md
      why: "R5 source of truth"
    - path: .codex/agents/.context/promt_manager_state.yaml
      why: "latest compact state if present"
  task_entrypoints:
    - path: scripts/macos/smoke_prod.sh
      why: "existing production smoke baseline"
      inspect_symbols:
        - main checks
    - path: apps/worker/backtest_job_runner/main/main.py
      why: "runner process entrypoint"
      inspect_symbols:
        - main
    - path: scripts/macos/reload_launchd_services.sh
      why: "Mac Studio service reload path"
      inspect_symbols:
        - prod_services
    - path: infra/macos/prometheus/prometheus.prod.yml
      why: "metrics target verification"
      inspect_symbols:
        - backtest-job-runner
  conditional_bundles:
    runner_smoke_script:
      read_when: "dedicated runner smoke script exists or must be added"
      paths:
        - scripts/backtest/run_stage_8_5_create_path_load_smoke.py
        - scripts/backtest
    api_contract:
      read_when: "creating controlled job through API requires endpoint details"
      paths:
        - apps/api/routes/backtests.py
        - apps/api/dto/backtests.py
        - apps/web/templates/pages/backtests.html
    runbooks:
      read_when: "operator instructions or rollback path are missing"
      paths:
        - docs/runbooks/mac-studio-native-backend-operations.md
        - docs/runbooks/mac-studio-monitoring-plan.md
  consult_if_needed:
    - path: docs/architecture/backtest/backtest-service-artifact-runtime-v1.ru.md
      read_when: "controlled BTCUSDT 15m smoke config is unclear"
    - path: docs/architecture/apps/web/web-ui-backend-implementation-plan-v1.md
      read_when: "Web UI result-state smoke is required"

style_references: []

hard_requirements:
  direct_main_delivery_required: true
  controlled_smoke_job_required: true
  missing_artifacts_are_blocker: true
  old_queued_jobs_not_primary_acceptance: true
  metrics_smoke_required: true
  api_responsiveness_smoke_required: true
  lazy_detail_cache_miss_and_hit_required: true

task_toggles:
  implementation_changes_allowed: false
  fix_introduced_failures_allowed: true
  add_smoke_script_if_missing: true
  publish_after_success: true

skill_routing:
  - skill: publish-ci-deploy
    use_when: "local gates pass and delivery begins"
    timing: "before ship"
    reason: "Git/GitHub/CI/Mac Studio deployment"
  - skill: backend-quality-gates
    use_when: "running local and production smoke-related gates"
    timing: "during verification"
    reason: "test and lint evidence"
  - skill: backend-performance-evidence
    use_when: "running load smoke, RSS or endpoint latency evidence"
    timing: "during verification"
    reason: "runner host capacity evidence"
  - skill: root-cause-debugging
    use_when: "runner job does not leave queued/running or service fails"
    timing: "if blocker"
    reason: "production defect localization"
  - skill: contract-impact-analysis
    use_when: "fixes alter API/status/config/runtime behavior"
    timing: "before final report"
    reason: "release contract classification"

target_envs:
  - local-dev
  - github-actions
  - mac-studio
  - production

required_literals:
  - "BTCUSDT"
  - "15m"
  - "queued -> running -> succeeded"
  - "127.0.0.1:9204/metrics"
  - "backtest_runner_tasks_claimed_total"
  - "backtest_runner_last_success_unixtime"
  - "publish-ci-deploy"
  - "git pull --ff-only"

non_goals:
  - "Do not accept a failed/missing-artifact job as successful runner smoke."
  - "Do not use an old stuck queued job as primary acceptance."
  - "Do not broaden feature scope beyond fixes required for R5 blockers."
  - "Do not claim deploy success before Mac Studio smoke evidence."

final_report_format:
  language: ru
  sections:
    - "Intent"
    - "Scope"
    - "Local gates"
    - "CI/deploy"
    - "Mac Studio service state"
    - "Production smoke"
    - "Performance"
    - "Contract impact"
    - "Risks"
    - "Handoff"
    - "Publish/deploy"

quality_gates:
  - cmd: "uv run pytest -q tests/unit/apps/api/test_backtests_routes.py tests/unit/contexts/backtest/application/use_cases/test_backtest_job_worker_use_case.py tests/unit/apps/worker/backtest_job_runner"
    expect: "passes"
  - cmd: "uv run ruff check apps/api apps/worker src/trading/contexts/backtest tests/unit/apps/api tests/unit/apps/worker tests/unit/contexts/backtest"
    expect: "passes"
  - cmd: "uv run pyright"
    expect: "passes or unrelated pre-existing failures classified"
  - cmd: "bash -n scripts/macos/bootstrap_native_prod.sh scripts/macos/reload_launchd_services.sh scripts/macos/smoke_prod.sh"
    expect: "passes"
  - cmd: "uv run python -m tools.docs.generate_docs_index --check"
    expect: "passes if docs changed"

expected_primary_touches:
  - "scripts/backtest/run_backtest_job_runner_prod_smoke.py if missing"
  - "scripts/macos/smoke_prod.sh if runner smoke is integrated"
  - "docs/runbooks/mac-studio-native-backend-operations.md if smoke/runbook is incomplete"

possible_secondary_touches:
  - "apps/worker/backtest_job_runner/** only for introduced blocker fixes"
  - "apps/api/routes/backtests.py only for introduced blocker fixes"
  - "infra/macos/launchd/com.roehub.backtest-job-runner.plist only for service blocker fixes"
  - "infra/macos/prometheus/prometheus.prod.yml only for target blocker fixes"

safety_notes:
  - "Inspect production backlog before enabling runner; do not let old jobs become the primary smoke."
  - "Missing artifacts/config is a blocker for acceptance, not success."
---

# Task

Perform final production smoke/load verification and direct-main delivery for `backtest-job-runner-v1`.

Done means:

- local gates are green;
- code is pushed to `origin/main`;
- CI/deploy are green;
- Mac Studio checkout and runtime are synchronized;
- launchd runner is loaded and running;
- controlled BTCUSDT 15m smoke job reaches `succeeded`;
- lazy detail cache miss materializes and second read is cache hit;
- metrics and logs are verified.

## Context / Current State

Generic `smoke_prod.sh` is not enough for runner acceptance. The runner plan requires dedicated evidence: controlled job, lease/progress fields, terminal top variants, lazy materialization, metrics, logs and API responsiveness.

Existing queued jobs may be inspected, cancelled or released by explicit operator decision, but they are not the primary acceptance smoke.

## Requirements (Must)

- Verify worktree scope before staging/publishing. Preserve unrelated changes.
- Run local gates before push.
- Use `publish-ci-deploy` direct-main flow: no PR branch, no draft PR.
- Monitor GitHub Actions/deploy to green.
- Sync Mac Studio with `git pull --ff-only` from actual repo checkout.
- Reload services using the repository runbook path.
- Verify `launchctl print` for `com.roehub.backtest-job-runner`.
- Verify `curl http://127.0.0.1:9204/metrics`.
- Create a controlled production smoke job using real artifacts, normally BTCUSDT 15m with bounded `top_n`.
- Observe `queued -> running -> succeeded`, `locked_by`, `started_at`, `heartbeat_at`, `lease_expires_at`, `top_variants > 0`.
- Open one top `variant_key`, trigger lazy detail, observe cache miss materialization and cached second read.
- Verify API/auth/dashboard remain responsive.
- Verify logs contain no secrets/full request payload/full trades payload.

## Requirements (Should)

- Run a small create/status burst to prove API remains responsive while runner is active.
- Capture RSS/load evidence before and after smoke if feasible.
- Record exact commit SHA and service label state.

## Requirements (Nice-to-have)

- Add a reusable smoke script if the smoke is still manual and scriptable.

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

Expand only for production blockers, failing CI, missing smoke scripts or unclear rollback.

# Reading manifest

Use the front-matter `context_sources` as the canonical reading map.

# Work plan (agent should follow)

Skill routing for this task:

- `publish-ci-deploy`: use for the full Git/GitHub/CI/Mac Studio deployment.
- `backend-quality-gates`: use during local verification.
- `backend-performance-evidence`: use for load/RSS/latency evidence.
- `root-cause-debugging`: use if runner job remains queued/running or service fails.
- `contract-impact-analysis`: use if blocker fixes alter contracts.

1. Inspect local status and intended diff.
2. Run local gates.
3. Add/fix only required smoke tooling or introduced blockers.
4. Publish through direct-main flow.
5. Sync/reload Mac Studio.
6. Run runner production smoke.
7. Run metrics/log/API responsiveness checks.
8. Report evidence and residual risks.

# Acceptance criteria (Definition of Done)

- Local gates pass.
- CI/deploy green.
- Mac Studio service loaded/running.
- Metrics endpoint returns runner metrics.
- Controlled job succeeds with top variants.
- Lazy materialization succeeds and second read is cache hit.
- API remains responsive.
- Final report includes exact commands, commit SHA, host/path, service label and smoke ids.

# Implementation constraints

- Fix only introduced failures or missing smoke tooling.
- Do not use destructive git commands.
- Do not stage unrelated local changes.
- Do not report success until delivery and production smoke are complete.

# Files to indicate (expected touched areas)

Use `expected_primary_touches` and `possible_secondary_touches` from front matter.

# Non-goals

See front matter `non_goals`.

# Quality gates (must run and pass)

Run the `quality_gates` commands from front matter. If a gate fails, classify it as `introduced`, `required-path pre-existing`, `unrelated pre-existing`, `environmental`, or `flaky`.

# Final output: report format (strict)

Report in Russian using the `final_report_format.sections` order. Separate local test evidence, CI/deploy evidence, Mac Studio runtime evidence, production smoke evidence, inference and assumptions.
