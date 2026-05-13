---
prompt_name: backtest_runner_compute_remediation_v1_04_production_delivery_and_full_use_case_acceptance
repo: roehub.com
branch: main
scope: "P0: deliver the remediated API/runner compute path to production and prove the UI-created backtest use case on Mac Studio."

language:
  implementation: verification_deploy_ops
  agent_report: ru

context_sources:
  always_read:
    - path: .codex/AGENTS.md
      why: "repo engineering contract"
    - path: docs/architecture/backtest/benchmark_iterations/README.md
      why: "benchmark acceptance policy"
    - path: docs/architecture/backtest/backtest-job-runner-production-plan-v1.md
      why: "runner production smoke contract"
  task_entrypoints:
    - path: scripts/macos/smoke_prod.sh
      why: "existing production smoke baseline"
      inspect_symbols:
        - main checks
    - path: scripts/macos/reload_launchd_services.sh
      why: "Mac Studio service reload path"
      inspect_symbols:
        - prod_services
    - path: infra/macos/prometheus/prometheus.prod.yml
      why: "Prometheus target verification"
      inspect_symbols:
        - backtest-job-runner
    - path: scripts/backtest/run_api_runner_benchmark_parity.py
      why: "new API-runner benchmark acceptance script"
      inspect_symbols:
        - main
  conditional_bundles:
    latest_benchmark_evidence:
      read_when: "before delivery; identify the new accepted benchmark folder from prompt 03"
      paths:
        - docs/architecture/backtest/benchmark_iterations
    api_and_ui_smoke:
      read_when: "when creating a UI-equivalent job or checking result state through web/API"
      paths:
        - apps/api/routes/backtests.py
        - apps/api/dto/backtests.py
        - apps/web/templates/pages/backtests.html
    macos_operations:
      read_when: "when syncing/reloading Mac Studio services or debugging deployment"
      paths:
        - docs/runbooks/mac-studio-native-backend-operations.md
        - docs/runbooks/mac-studio-monitoring-plan.md
        - infra/macos/launchd/com.roehub.backtest-job-runner.plist
        - infra/scripts/monit/roehub-backtest-job-runner.monitrc
    github_ci:
      read_when: "when publishing, watching CI, or fixing deploy failures"
      paths:
        - .github/workflows
  consult_if_needed:
    - path: docs/architecture/backtest/backtest-service-artifact-runtime-v1.ru.md
      read_when: "controlled benchmark fixture or lazy detail contract is unclear"
    - path: docs/architecture/apps/web/web-ui-backend-implementation-plan-v1.md
      read_when: "browser-visible UI verification is required"

style_references:
  - .codex/promt_template.md

hard_requirements:
  direct_main_delivery_required: true
  local_gates_before_publish_required: true
  ci_green_required: true
  macstudio_sync_required: true
  git_pull_ff_only_required: true
  launchd_runner_loaded_required: true
  monit_must_not_kill_compute_required: true
  benchmark_acceptance_required: true
  controlled_ui_like_job_required: true
  lazy_detail_cache_miss_and_hit_required: true
  metrics_smoke_required: true
  mixed_light_heavy_scheduler_smoke_required: true
  preflight_heavy_classification_required: true
  light_candidate_refinement_required: true
  heavy_fifo_required: true
  light_parallelism_cap_required: true

task_toggles:
  implementation_changes_allowed: false
  fix_introduced_failures_allowed: true
  publish_after_success: true
  run_browser_if_ui_visible: true
  run_macstudio_benchmark: true
  run_mixed_scheduler_smoke: true

skill_routing:
  - skill: publish-ci-deploy
    use_when: "local gates and benchmark prerequisites are ready for delivery"
    timing: before ship
    reason: "owns Git/GitHub/CI/Mac Studio deployment and post-deploy verification"
  - skill: backend-performance-evidence
    use_when: "running final benchmark, load smoke, CPU/RSS, or endpoint latency evidence"
    timing: during verification
    reason: "acceptance depends on measured host performance"
  - skill: root-cause-debugging
    use_when: "CI, service reload, benchmark, runner, metrics, or UI-created job fails"
    timing: if blocker
    reason: "fix real blockers rather than reporting partial delivery"
  - skill: backend-quality-gates
    use_when: "running local and production-related backend gates"
    timing: during verification
    reason: "Roehub backend gates are uv-based"
  - skill: browser-qa-evidence
    use_when: "verifying browser-visible `/backtests` job create/result flow"
    timing: during verification
    reason: "the user use case starts from UI"
  - skill: contract-impact-analysis
    use_when: "any delivery blocker fix changes API/status/config/runtime behavior"
    timing: before final report
    reason: "final report must classify compatibility"

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
  - "benchmark_results.json"
  - "benchmark_summary.md"
  - "scheduling_class"
  - "light_candidate"
  - "light"
  - "heavy"
  - "estimated_combinations_upper_bound"
  - "ROEHUB_BACKTEST_LIGHT_CONCURRENCY"
  - "ROEHUB_BACKTEST_HEAVY_CONCURRENCY"

non_goals:
  - "Do not claim deploy success before Mac Studio benchmark and smoke evidence."
  - "Do not use old queued jobs as primary acceptance."
  - "Do not accept failed/missing-artifact jobs as success."
  - "Do not claim scheduler safety without mixed light/heavy smoke evidence."
  - "Do not broaden into new UI design or indicator features."
  - "Do not silently skip browser-visible verification if the UI flow is available."

final_report_format:
  language: ru
  sections:
    - "Intent"
    - "Scope"
    - "Local gates"
    - "CI/deploy"
    - "Mac Studio service state"
    - "Benchmark acceptance"
    - "UI/API production smoke"
    - "Performance"
    - "Contract impact"
    - "Risks"
    - "Handoff"
    - "Publish/deploy"

quality_gates:
  - cmd: "uv run pytest -q tests/unit/apps/api/test_backtests_routes.py tests/unit/apps/worker/backtest_job_runner tests/unit/contexts/backtest/application/services/v2 tests/unit/contexts/backtest/application/use_cases/test_backtest_job_worker_use_case.py"
    expect: "passes"
  - cmd: "uv run ruff check apps/api apps/worker src/trading/contexts/backtest tests/unit/apps/api tests/unit/apps/worker tests/unit/contexts/backtest scripts/backtest"
    expect: "passes"
  - cmd: "uv run pyright"
    expect: "passes or unrelated pre-existing failures classified"
  - cmd: "bash -n scripts/macos/bootstrap_native_prod.sh scripts/macos/reload_launchd_services.sh scripts/macos/smoke_prod.sh"
    expect: "passes"
  - cmd: "uv run python -m tools.docs.generate_docs_index --check"
    expect: "passes"

expected_primary_touches:
  - "none unless fixing introduced delivery blockers"

possible_secondary_touches:
  - "scripts/backtest/run_api_runner_benchmark_parity.py only for introduced benchmark blocker fixes"
  - "scripts/backtest/run_backtest_job_runner_prod_smoke.py only for introduced smoke blocker fixes"
  - "scripts/macos/smoke_prod.sh only for smoke integration fixes"
  - "infra/macos/launchd/com.roehub.backtest-job-runner.plist only for service blocker fixes"
  - "infra/scripts/monit/roehub-backtest-job-runner.monitrc only for supervision blocker fixes"
  - "docs/runbooks/mac-studio-native-backend-operations.md only for handoff/rollback gaps"

safety_notes:
  - "Inspect dirty worktree scope before staging; preserve unrelated changes."
  - "If unrelated local changes make direct-main unsafe, stop and report the exact blocker."
  - "Mac Studio runtime path and repository checkout may differ; verify the deployed runtime bundle."
---

# Task

Deliver the remediated backtest API/runner compute path to production and prove the full user use case: a user starts a backtest from UI/API, the job runs through the production runner on Mac Studio, compute uses the corrected engine path, and results/lazy details are available.

Done means:

- local gates pass;
- changes are delivered to `origin/main` using the repository delivery flow;
- CI/deploy is green;
- Mac Studio repo checkout and runtime bundle are synchronized;
- `git pull --ff-only` succeeds where applicable;
- `com.roehub.backtest-job-runner` is loaded/running under launchd;
- Monit does not kill live compute because of metrics scrape behavior;
- final benchmark acceptance exists in the new benchmark folder;
- controlled UI/API-like `BTCUSDT 15m` job reaches `queued -> running -> succeeded`;
- obvious heavy requests are classified as `heavy` by preflight before compute;
- `light_candidate` requests are refined after prepare before exact scoring;
- mixed scheduler smoke proves bounded `light` parallelism and FIFO `heavy` processing;
- one lazy detail cache miss materializes and second read is cache hit;
- metrics and logs are verified.

## Context / Current State

This is the final delivery prompt. It assumes the preceding implementation prompts have already created the process-isolated runner boundary, hot-path compute remediation, and API-runner benchmark evidence. If those prerequisites are absent, stop and report the blocker rather than doing a partial production claim.

Acceptance is Mac Studio evidence, not local tests alone. Old queued jobs can be inspected, but they are not primary acceptance. Primary acceptance must create a controlled job and record exact evidence.

## Requirements (Must)

- Read context using the protocol below and stop early once sufficient.
- Verify worktree scope before staging or publishing.
- Preserve unrelated dirty files.
- Run local gates before publishing.
- Use `publish-ci-deploy` for the end-to-end delivery flow.
- Watch CI/deploy to green and fix introduced failures.
- Sync Mac Studio checkout and runtime; verify actual deployed code identity.
- Reload services through repository scripts/runbook.
- Verify `launchctl print` for `com.roehub.backtest-job-runner`.
- Verify `curl http://127.0.0.1:9204/metrics` and required metrics names.
- Run or verify the new API-runner benchmark folder with `benchmark_results.json`, `benchmark_summary.md`, and accounting validation.
- Create a controlled `BTCUSDT 15m` UI/API-like job and observe `queued -> running -> succeeded`.
- Verify `top_variants > 0` and expected top-N behavior.
- Run a mixed scheduler smoke: multiple controlled light jobs plus multiple controlled heavy jobs.
- Verify an obvious heavy request enters the heavy lane from preflight classification using `estimated_combinations_upper_bound` or equivalent evidence.
- Verify a `light_candidate` request is confirmed as `light` after prepare before exact scoring.
- Verify no more than configured light jobs run concurrently.
- Verify heavy jobs run FIFO by `created_at ASC, job_id ASC`.
- Verify light jobs do not starve an older queued heavy job.
- Verify production does not overlap light jobs with an active heavy job unless the benchmark explicitly accepted that mode.
- Trigger lazy detail cache miss and verify materialization then cache hit.
- Verify API/auth/dashboard remain responsive during or after compute.
- Verify logs contain no secrets, full request payloads, or full trades payloads.

## Requirements (Should)

- Capture process/thread CPU evidence and RSS before/during/after the controlled job.
- Capture Prometheus target state for `backtest-job-runner`.
- Capture active child counts and metrics by scheduling class if available.
- Capture browser-visible `/backtests` evidence if the UI route is available and credentials are known.
- Record rollback commands and service state in the final handoff.

## Requirements (Nice-to-have)

- Run a secondary heavy-load smoke for 5 indicators if the primary acceptance passes and runtime budget allows.
- Attach concise browser screenshot references if Browser/Playwright evidence is gathered.

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
- `<= ~35k-50k tokens`

Stop reading once all of the following are true:

- changed contracts are identified,
- touched files are bounded,
- acceptance criteria are implementable without ambiguity,
- no unresolved public API or persistence-contract ambiguity remains.

Expand context only for blockers, failing quality gates, unclear contracts, benchmark threshold conflicts, or architecture conflicts that affect correctness.

# Reading manifest

Use the front-matter `context_sources` as the canonical reading map.

Read with this intent:

- `always_read`: repository rules, benchmark acceptance policy, runner production plan;
- `task_entrypoints`: production smoke, reload, Prometheus, and benchmark script;
- `conditional_bundles`: read only when the stated condition applies;
- `consult_if_needed`: read only for blockers, ambiguity, or conflict resolution.

Do not convert this manifest into a broad mandatory reading list.

# Work plan (agent should follow)

Skill routing for this task:

- `publish-ci-deploy`: use for delivery, CI watch, Mac Studio sync, deploy, and post-deploy verification.
- `backend-performance-evidence`: use for benchmark, CPU/RSS, and latency evidence.
- `root-cause-debugging`: use if any gate, service, benchmark, or UI/API smoke fails.
- `backend-quality-gates`: use for local gates.
- `browser-qa-evidence`: use if browser-visible `/backtests` flow is available.
- `contract-impact-analysis`: use before final report if any blocker fix changes runtime/API/config behavior.

1. Verify prerequisite implementation and benchmark evidence exist.
2. Inspect worktree and isolate intended changes only.
3. Run local gates.
4. Publish via `publish-ci-deploy`, watch CI/deploy, and fix introduced blockers.
5. Sync Mac Studio and verify deployed runtime identity.
6. Reload services and verify launchd/Monit/Prometheus state.
7. Run final benchmark acceptance and controlled UI/API production smoke.
8. Run mixed scheduler smoke for light parallelism and heavy FIFO.
9. Verify lazy detail cache miss/hit, metrics, logs, and API responsiveness.
10. Produce final Russian report with exact evidence and residual risks.

# Acceptance criteria (Definition of Done)

- Local gates pass or unrelated pre-existing failures are clearly classified.
- CI/deploy is green.
- Mac Studio has the intended commit/runtime files.
- Runner service is loaded/running.
- Metrics endpoint and Prometheus target are healthy.
- New benchmark folder is present and accepted.
- Controlled `BTCUSDT 15m` job reaches `succeeded`.
- Preflight heavy classification is verified for an obvious heavy request.
- Light-candidate refinement is verified before exact scoring.
- Mixed scheduler smoke passes: bounded light concurrency, heavy FIFO, no heavy-heavy parallelism, no heavy starvation.
- Lazy detail cache miss and hit are verified.
- CPU/RSS evidence demonstrates the host is used as expected for the accepted workload.
- Final report includes rollback/handoff and contract impact.

# Implementation constraints

## Determinism & ordering

- Do not process old queued jobs as primary acceptance.
- Do not change benchmark output after the fact without rerunning or labeling the correction.
- Do not enable `light=3` in production unless benchmark evidence explicitly accepts it.

## API / contracts

- Public API compatibility target: no breaking changes.
- Any emergency fix must be classified before final report.

## Delivery safety

- Use non-destructive git commands.
- Do not revert unrelated user changes.
- Do not claim production success before Mac Studio evidence.

# Files to indicate (expected touched areas)

Expected primary touches:

- none unless fixing introduced delivery blockers

Possible secondary touches:

- `scripts/backtest/run_api_runner_benchmark_parity.py`
- `scripts/backtest/run_backtest_job_runner_prod_smoke.py`
- `scripts/macos/smoke_prod.sh`
- `infra/macos/launchd/com.roehub.backtest-job-runner.plist`
- `infra/scripts/monit/roehub-backtest-job-runner.monitrc`
- `docs/runbooks/mac-studio-native-backend-operations.md`

# Non-goals

- Do not add new product features.
- Do not redesign the UI.
- Do not skip benchmark acceptance.
- Do not use local-only evidence as production acceptance.

# Quality gates (must run and pass)

- `uv run pytest -q tests/unit/apps/api/test_backtests_routes.py tests/unit/apps/worker/backtest_job_runner tests/unit/contexts/backtest/application/services/v2 tests/unit/contexts/backtest/application/use_cases/test_backtest_job_worker_use_case.py`
- `uv run ruff check apps/api apps/worker src/trading/contexts/backtest tests/unit/apps/api tests/unit/apps/worker tests/unit/contexts/backtest scripts/backtest`
- `uv run pyright`
- `bash -n scripts/macos/bootstrap_native_prod.sh scripts/macos/reload_launchd_services.sh scripts/macos/smoke_prod.sh`
- `uv run python -m tools.docs.generate_docs_index --check`

# Final output: report format (strict)

Write the final report in Russian with these sections:

- `Intent`
- `Scope`
- `Local gates`
- `CI/deploy`
- `Mac Studio service state`
- `Benchmark acceptance`
- `UI/API production smoke`
- `Performance`
- `Contract impact`
- `Risks`
- `Handoff`
- `Publish/deploy`

Include exact commit SHA, benchmark folder path, Mac Studio commands/evidence, service state, and any unresolved risks.
