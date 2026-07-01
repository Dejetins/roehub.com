---
prompt_name: backtest_runner_compute_remediation_v1_01_process_isolated_runner_boundary
repo: roehub.com
branch: current
scope: "P0: restore a production-safe compute boundary by making `backtest-job-runner` a responsive parent process that launches isolated per-job compute children."

language:
  implementation: python
  agent_report: ru

context_sources:
  always_read:
    - path: .codex/AGENTS.md
      why: "repo engineering contract"
    - path: docs/architecture/backtest/backtest-job-runner-production-plan-v1.md
      why: "runner target state and queue contract"
    - path: .codex/agents/.context/promt_manager_state.yaml
      why: "latest compact state if present"
  task_entrypoints:
    - path: apps/worker/backtest_job_runner/wiring/modules/backtest_job_runner.py
      why: "current runner loop, metrics, and service composition"
      inspect_symbols:
        - BacktestJobRunnerApp
        - BacktestJobRunnerRuntimeConfig
    - path: src/trading/contexts/backtest/application/use_cases/backtest_job_worker.py
      why: "claim, heartbeat, progress, and terminal write use case"
      inspect_symbols:
        - BacktestJobWorkerUseCase
        - _LeaseHeartbeat
    - path: src/trading/contexts/backtest/application/services/v2/job_orchestration.py
      why: "current in-process full backtest executor"
      inspect_symbols:
        - BacktestRuntimeJobOrchestrationService
        - execute
    - path: infra/scripts/monit/roehub-backtest-job-runner.monitrc
      why: "current healthcheck can restart active compute"
      inspect_symbols:
        - metrics healthcheck
  conditional_bundles:
    api_create_boundary:
      read_when: "if API enqueue semantics or job ownership need confirmation"
      paths:
        - apps/api/routes/backtests.py
        - src/trading/contexts/backtest/application/use_cases/backtest_jobs.py
    macos_service_files:
      read_when: "when changing launchd, reload, Prometheus, or Monit behavior"
      paths:
        - infra/macos/launchd/com.roehub.backtest-job-runner.plist
        - scripts/macos/reload_launchd_services.sh
        - scripts/macos/bootstrap_native_prod.sh
        - infra/macos/prometheus/prometheus.prod.yml
        - docs/runbooks/mac-studio-native-backend-operations.md
    tests_existing_runner:
      read_when: "when adding or updating runner boundary tests"
      paths:
        - tests/unit/apps/worker/backtest_job_runner
        - tests/unit/contexts/backtest/application/use_cases/test_backtest_job_worker_use_case.py
    benchmark_contract:
      read_when: "if process boundary changes benchmark or telemetry accounting"
      paths:
        - docs/architecture/backtest/benchmark_iterations/README.md
        - docs/architecture/backtest/backtest-service-artifact-runtime-v1.ru.md
  consult_if_needed:
    - path: docs/architecture/backtest/benchmark_iterations/2026-05-02_iteration_8_execution_sizing_completion/benchmark_summary.md
      read_when: "canonical 2026-05-02 acceptance context is unclear"
    - path: docs/architecture/backtest/benchmark_iterations/2026-05-02_iteration_9_lazy_trades_detail/benchmark_summary.md
      read_when: "lazy detail materialization semantics are affected"

style_references:
  - .codex/promt_template.md

hard_requirements:
  api_must_remain_enqueue_only: true
  runner_parent_must_stay_responsive_during_compute: true
  child_process_per_full_job_required: true
  child_process_must_exit_after_one_full_job: true
  full_job_memory_release_boundary_required: true
  replace_parent_in_process_executor_required: true
  parent_must_not_construct_full_compute_graph_in_production_wiring: true
  old_sync_runner_loop_must_be_replaced_or_narrowed: true
  active_docs_must_remove_parent_in_process_compute_claims: true
  docs_cleanup_required_for_replaced_runner_paths: true
  parent_must_remain_terminal_commit_owner: true
  child_must_not_write_terminal_job_state: true
  scheduler_classifies_full_jobs_before_compute: true
  preflight_must_classify_obvious_heavy_jobs: true
  preflight_conservative_upper_bound_required: true
  light_candidate_must_be_confirmed_after_prepare: true
  light_jobs_may_run_parallel_with_bounded_concurrency: true
  heavy_jobs_must_use_exclusive_host_slot: true
  heavy_jobs_claim_order_fifo_by_created_at_job_id: true
  unknown_cost_jobs_default_to_heavy: true
  metrics_endpoint_must_not_depend_on_compute_child_responsiveness: true
  monit_must_not_kill_live_compute_on_metrics_timeout: true
  preserve_claim_lease_terminal_commit_semantics: true
  user_job_ownership_must_be_preserved: true

task_toggles:
  implementation_changes_allowed: true
  add_child_compute_entrypoint: true
  update_macos_supervision_config: true
  add_light_heavy_scheduler_policy: true
  update_runbook_if_needed: true
  run_macstudio_smoke: false
  publish_after_success: true
  publish_via_github_yeet: true
  direct_main_push_after_local_gates: true
  merge_to_main_in_this_prompt: true
  deploy_to_macstudio_in_this_prompt: true

skill_routing:
  - skill: architecture-design
    use_when: "defining parent/child process boundary, ownership, IPC, and rollback behavior"
    timing: before implementation
    reason: "this changes runtime service boundaries without changing public API vocabulary"
  - skill: root-cause-debugging
    use_when: "runner, lease, heartbeat, child exit, or Monit behavior does not match expected state"
    timing: if blocker
    reason: "the current failure mode is production runtime regression under heavy compute"
  - skill: backend-quality-gates
    use_when: "running focused tests, lint, and type checks for runner/control-plane changes"
    timing: during verification
    reason: "Roehub backend gates are uv-based"
  - skill: contract-impact-analysis
    use_when: "changing job states, metrics names, config keys, persistence fields, or lazy detail status behavior"
    timing: before final report
    reason: "process isolation must remain a compatible runtime workflow change"
  - skill: github:yeet
    use_when: "scoped implementation/docs changes pass local gates and are ready to publish"
    timing: after local gates and before host delivery
    reason: "safe scope inspection, intentional commit, direct main push, and no unrelated staging"
  - skill: publish-ci-deploy
    use_when: "direct main push is complete and updates must be delivered to Mac Studio"
    timing: after `github:yeet` publish
    reason: "CI/deploy watch, Mac Studio sync, service reload, and production smoke"

target_envs:
  - local-dev
  - mac-studio
  - production

required_literals:
  - "queued -> running -> succeeded"
  - "backtest-job-runner"
  - "127.0.0.1:9204/metrics"
  - "backtest_runner_tasks_claimed_total"
  - "backtest_runner_last_success_unixtime"
  - "child process"
  - "disposable"
  - "RSS"
  - "retained_rss_delta"
  - "BacktestJobWorkerUseCase.run_next"
  - "BacktestRunnerTaskScheduler.run_next"
  - "build_backtest_job_runner_app"
  - "BacktestRuntimeJobOrchestrationService"
  - "docs cleanup"
  - "sync_inline"
  - "terminal owner"
  - "finish_with_top_variants"
  - "at-most-one terminal commit"
  - "scheduling_class"
  - "light_candidate"
  - "light"
  - "heavy"
  - "estimated_combinations_upper_bound"
  - "ROEHUB_BACKTEST_LIGHT_CONCURRENCY"
  - "ROEHUB_BACKTEST_HEAVY_CONCURRENCY"
  - "github:yeet"
  - "git push origin main"
  - "origin/main"
  - "publish-ci-deploy"
  - "git pull --ff-only"

non_goals:
  - "Do not move long-running compute back into `com.roehub.api`."
  - "Do not accept increasing Monit timeout as the primary fix."
  - "Do not implement the ordinal/Numba hot-path rewrite in this prompt unless required by tests."
  - "Do not run heavy full jobs in parallel on Mac Studio v1."
  - "Do not run light jobs concurrently with an active heavy job until a separate benchmark proves safe host sharing."
  - "Do not change public jobs API DTO vocabulary unless a blocker proves it necessary."
  - "Do not run final Mac Studio benchmark acceptance; that belongs to the benchmark prompt."
  - "Do not implement lazy trades cache-miss child execution or cache-hit bounded readers here; that belongs to prompt 03."
  - "Do not stop at local gates; direct `main`/`origin/main` delivery and Mac Studio smoke are required after gates pass."

final_report_format:
  language: ru
  sections:
    - "Intent"
    - "Process boundary"
    - "Removed/replaced paths"
    - "Terminal ownership"
    - "Docs cleanup"
    - "Runner/Monit behavior"
    - "Tests"
    - "Contract impact"
    - "Risks"
    - "Publish"
    - "CI/deploy"
    - "Mac Studio delivery"
    - "Next prompt"

quality_gates:
  - cmd: "uv run pytest -q tests/unit/apps/worker/backtest_job_runner tests/unit/contexts/backtest/application/use_cases/test_backtest_job_worker_use_case.py"
    expect: "passes"
  - cmd: "uv run ruff check apps/worker src/trading/contexts/backtest tests/unit/apps/worker tests/unit/contexts/backtest/application/use_cases"
    expect: "passes"
  - cmd: "uv run pyright"
    expect: "passes or unrelated pre-existing failures classified"
  - cmd: "bash -n scripts/macos/bootstrap_native_prod.sh scripts/macos/reload_launchd_services.sh scripts/macos/smoke_prod.sh"
    expect: "passes if shell files changed"
  - cmd: "uv run python -m tools.docs.generate_docs_index --check"
    expect: "passes if docs changed"
  - cmd: "gh --version && gh auth status"
    expect: "passes before `github:yeet` publication, or publish blocker is reported"

expected_primary_touches:
  - "apps/worker/backtest_job_runner/**"
  - "src/trading/contexts/backtest/application/use_cases/backtest_job_worker.py"
  - "tests/unit/apps/worker/backtest_job_runner/**"
  - "infra/scripts/monit/roehub-backtest-job-runner.monitrc"

possible_secondary_touches:
  - "infra/macos/launchd/com.roehub.backtest-job-runner.plist"
  - "scripts/macos/reload_launchd_services.sh"
  - "scripts/macos/bootstrap_native_prod.sh"
  - "infra/macos/prometheus/prometheus.prod.yml"
  - "docs/runbooks/mac-studio-native-backend-operations.md"
  - "docs/runbooks/mac-studio-monitoring-plan.md"
  - "docs/architecture/backtest/backtest-job-runner-production-plan-v1.md"
  - "docs/architecture/apps/web/web-ui-backend-implementation-plan-v1.md"
  - "src/trading/contexts/backtest/application/ports/backtest_job_repositories.py"

safety_notes:
  - "Preserve unrelated dirty worktree changes; stage only scoped files if publishing later."
  - "Do not log secrets, full request payloads, or full trades payloads."
  - "If child process isolation requires a persistence contract change, stop and classify before broad edits."
---

# Task

Implement the first remediation step for backtest production compute: make `backtest-job-runner` a responsive control-plane parent that launches an isolated child process for each full backtest job.

Done means:

- `POST /api/backtests/jobs` still only creates a queued job and does not execute full compute.
- the runner parent owns claim, heartbeat, progress, metrics, child supervision, and terminal commit coordination;
- full compute runs in a separate child process for the claimed job;
- every full-job child process is disposable and exits after exactly one full job;
- the runner parent enforces a two-lane scheduler: bounded parallel `light` jobs and FIFO/exclusive `heavy` jobs;
- parent metrics at `127.0.0.1:9204/metrics` remain responsive while a child is busy;
- Monit no longer restarts live compute because the metrics endpoint temporarily blocks;
- tests prove claim/finish/fail/reclaim semantics still preserve `at-most-one terminal commit`.

## Context / Current State

The current API create boundary is already mostly correct: it creates a queued job and calls the execution trigger. The regression is in the production execution topology: the runner currently composes `BacktestRuntimeJobOrchestrationService` in the same process as lease, heartbeat, metrics, and Monit health.

Observed production failure mode: a heavy job can consume the process enough that `/metrics` times out; Monit then restarts `backtest-job-runner` even though compute is alive. This creates lease/reclaim churn and makes CPU/load behavior look broken from the UI.

This prompt does not optimize the numerical algorithm. It creates the process boundary needed so the later hot-path and benchmark prompts can run safely on Mac Studio.

This prompt owns full-job disposable compute only. Lazy trades cache-miss disposable compute and memory-bounded cache-hit reads are handled by prompt 03.

## Method-level replacement / cleanup map

This prompt must replace the old production execution path, not build a second path beside it.

Required current-method decisions:

- `apps/worker/backtest_job_runner/wiring/modules/backtest_job_runner.py::build_backtest_job_runner_app`
  - Replace production parent wiring that constructs `BacktestRuntimeJobOrchestrationService` directly in the runner parent.
  - The parent may construct repositories, preflight, lease, metrics, scheduler, and child supervisor only.
  - Full compute service graph construction belongs in the child entrypoint/wiring.
  - A direct in-process executor may remain only in focused tests or explicitly named local-dev diagnostic code, never in production service composition.
- `src/trading/contexts/backtest/application/use_cases/backtest_job_worker.py::BacktestJobWorkerUseCase.run_next`
  - Keep claim/heartbeat/progress/terminal semantics.
  - Replace the parent-side call to an in-process compute executor with a child-supervision executor contract.
  - Do not let this method call `BacktestRuntimeJobOrchestrationService.execute` in the parent.
  - Keep the parent as the terminal owner: parent calls `finish_with_top_variants` after child success/failure is interpreted.
  - Child must not directly write `succeeded`/`failed` terminal job state.
- `apps/worker/backtest_job_runner/wiring/modules/backtest_job_runner.py::BacktestRunnerTaskScheduler.run_next`
  - Replace or narrow the synchronous “claim one task and block until complete” loop if it prevents light-job concurrency.
  - The parent must be able to launch/reap children and keep metrics responsive while children run.
  - If a blocking helper remains, it must not be the production scheduling loop for full jobs.
- `apps/worker/backtest_job_runner/wiring/modules/backtest_job_runner.py::BacktestJobRunnerApp.run`
  - Replace active-task accounting that is updated only after a blocking `worker.run_next()` returns.
  - Parent metrics must reflect active child state while children are running, and `/metrics` must remain responsive.
- `src/trading/contexts/backtest/application/services/v2/job_orchestration.py::BacktestRuntimeJobOrchestrationService.execute`
  - Keep as the child-only canonical compute service unless a narrower child service is introduced.
  - Child output must be bounded structured result data for parent terminalization, not direct production terminal writes.
  - Update misleading cleanup metadata: `worker_recycle_required: False` must not remain as production evidence that process recycle is unnecessary.
  - Any `gc.collect()` cleanup may remain as best-effort child-local cleanup, but acceptance must not rely on it for parent memory release.
- Exports/imports in `apps/worker/backtest_job_runner/wiring/modules/__init__.py` and service `__all__`
  - Remove or narrow exports that make obsolete parent in-process runner APIs look like active production entrypoints.

Definition of “removed” for this prompt:

- The old in-process full compute path must be absent from production runner wiring.
- It is acceptable to keep canonical compute classes as child-only implementation, unit-test helpers, or direct benchmark baselines if their module/docstring/name makes that boundary explicit.
- It is not acceptable to leave both old parent in-process execution and new child execution selectable by accident through production defaults.

## Documentation cleanup / drift map

Docs must be updated in the same change when code paths are replaced. Do not leave active docs that describe the removed parent in-process runner path as current behavior.

Required doc decisions:

- `docs/architecture/backtest/backtest-job-runner-production-plan-v1.md`
  - Replace statements that frame v1 memory growth mitigation as `MAX_JOBS_PER_PROCESS`/launchd restart only.
  - Describe the new parent/child disposable full-job process boundary and parent-owned metrics/lease behavior.
  - Update sequence diagrams or flow text if they imply `backtest-job-runner` executes full compute inside the parent process.
- `docs/architecture/apps/web/web-ui-backend-implementation-plan-v1.md`
  - Remove or rewrite active TODO/status text that says `sync_inline` or `BacktestRuntimeJobOrchestrationService` remains the production API/runner path after this remediation lands.
  - Historical notes may remain only if clearly marked historical and not current acceptance criteria.
- `docs/runbooks/mac-studio-native-backend-operations.md` and `docs/runbooks/mac-studio-monitoring-plan.md`
  - Update operations/runbook text if service behavior, restart expectations, active child process observation, or memory-release evidence changes.
- `docs/architecture/backtest/README.md` or architecture index files
  - Update status only if they still classify the old runner path as residual/untrusted after successful delivery.

Required docs verification:

- Run `rg -n "sync_inline|BacktestRuntimeJobOrchestrationService|MAX_JOBS_PER_PROCESS|in-process|parent process" docs/architecture docs/runbooks` and classify every remaining hit as historical, child-only, direct-benchmark-only, or updated current-state text.
- Run `uv run python -m tools.docs.generate_docs_index --check` if any Markdown docs change.

Target Mac Studio v1 scheduling policy:

- full jobs get an initial preflight scheduling decision before compute;
- preflight must classify obvious heavy jobs as `heavy` using a conservative upper bound;
- preflight may classify bounded small jobs only as `light_candidate`, not final `light`;
- `light_candidate` jobs must be refined after prepare/basic stages using actual row counts;
- if refined cost exceeds light thresholds, the job is promoted/requeued to `heavy` before exact scoring;
- unknown or unclassified cost defaults to `heavy`;
- `light` jobs may run in parallel with bounded child concurrency, default `2`, configurable up to `3` only after benchmark evidence;
- `heavy` jobs use an exclusive host slot: at most one active heavy child per Mac Studio host;
- v1 should not run light jobs while a heavy job is active unless a later benchmark explicitly proves safe host sharing;
- heavy job claim order is FIFO by `created_at ASC, job_id ASC`;
- new light jobs must not starve an older queued heavy job; add an anti-starvation rule such as age threshold or max consecutive light claims before a heavy claim.

## Requirements (Must)

- Read context using the protocol below and stop early once sufficient.
- Preserve public jobs API behavior and DTO vocabulary unless a blocker proves a compatible additive change is required.
- Keep `BacktestJobWorkerUseCase` semantics equivalent at the application boundary: claim, heartbeat, progress, execute, terminal write.
- Add a child compute entrypoint that can execute one full job by `job_id` under the same production config and artifact context as the current in-process runner.
- Ensure the full-job child exits after one job regardless of success or failure, so OS memory release is tied to process exit rather than Python `gc.collect()`.
- Remove the old production parent wiring that directly composes and runs `BacktestRuntimeJobOrchestrationService`.
- Update active architecture/runbook docs so they no longer describe parent in-process compute or `MAX_JOBS_PER_PROCESS` as the primary memory-release strategy.
- Add or reuse a scheduling classification field/metadata for full jobs. If persistence is needed, make it additive and contract-classified.
- Persist or expose the preflight scheduling decision so the runner can route obvious heavy jobs without running prepare first.
- Treat preflight `heavy` as final for v1 scheduling; do not auto-demote it after prepare unless a later prompt explicitly adds safe reclassification evidence.
- Allow `light_candidate` to use a light slot only through prepare/basic stages. It must be confirmed as `light` before exact scoring.
- Requeue/promote `light_candidate` to `heavy` when post-prepare actual combinations exceed configured light thresholds.
- Add parent-side slot accounting for `light` and `heavy` full-job children.
- Enforce `ROEHUB_BACKTEST_LIGHT_CONCURRENCY` default `2` and `ROEHUB_BACKTEST_HEAVY_CONCURRENCY` default `1`.
- Ensure heavy jobs are claimed FIFO and run exclusively on the host.
- Ensure light jobs cannot starve queued heavy jobs.
- Ensure only the parent exposes the runner metrics endpoint.
- Ensure child exit success/failure is translated into the same terminal job behavior as the current worker path.
- Ensure parent remains the only terminal state owner for full jobs. Child may compute and return bounded result payload, but must not independently call `finish_with_top_variants`.
- Guard terminal write by the existing lease/lock semantics or an equally strict ownership token.
- Update Monit/service configuration so metrics scrape failure does not kill an active compute child.
- Add focused tests for parent/child success, child failure, child timeout/exit, and lease ownership behavior.

## Requirements (Should)

- Prefer a simple `subprocess`-based child model over a complex pool; do not introduce a long-lived compute worker pool for full jobs in this prompt.
- Keep IPC boring: pass `job_id` and config through argv/env, write structured child result or rely on guarded DB terminalization.
- Keep scheduling policy explicit and config-driven; avoid hard-coding commercial tier names into the runner loop.
- Add structured logs for child start, child exit, duration, and error class without high-cardinality metrics labels.
- Keep cancellation cooperative at stage/chunk boundaries unless a later prompt adds hard kill semantics.

## Requirements (Nice-to-have)

- Add a local fake child mode for tests to avoid running full numerical compute.
- Emit a parent metric that distinguishes parent alive from child active.
- Emit low-cardinality metrics for active child count by scheduling class.

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

- `always_read`: repository rules, runner plan, compact prior state;
- `task_entrypoints`: current runner loop, worker use case, executor, and Monit health;
- `conditional_bundles`: read only when the stated condition applies;
- `consult_if_needed`: read only for blockers, ambiguity, or conflict resolution.

Do not convert this manifest into a broad mandatory reading list.

# Work plan (agent should follow)

Skill routing for this task:

- `architecture-design`: use before implementation for the parent/child boundary and rollback behavior.
- `root-cause-debugging`: use if runner, child, lease, or Monit behavior cannot be reproduced or localized.
- `backend-quality-gates`: use during verification for uv-based tests/lint/type checks.
- `contract-impact-analysis`: use before final report if any externally relied-on behavior changes.
- `github:yeet`: use after local gates for scope inspection, selective staging, commit, and direct push to `main`/`origin/main`.
- `publish-ci-deploy`: use after direct main push to watch CI/deploy, sync Mac Studio with `git pull --ff-only`, reload services, and smoke production.

1. Verify current API create remains enqueue-only.
2. Design the minimal parent/child process contract and record assumptions in code comments or nearby docs only where useful.
3. Define the minimal preflight scheduling decision: `heavy` for conservative upper-bound overflow, `light_candidate` only for bounded small jobs. Unknown cost must classify as `heavy`.
4. Implement a disposable child compute entrypoint for one full job and a parent runner path that supervises it.
5. Implement post-prepare refinement: confirm `light_candidate` as `light`, or promote/requeue it to `heavy` before exact scoring.
6. Implement parent-side slot accounting: light concurrency default `2`, heavy concurrency `1`, no heavy-heavy parallelism, and no light-heavy overlap in v1 unless benchmarked.
7. Keep metrics and heartbeat in the parent process and make active-child state visible without job/user labels.
8. Adjust Monit/service behavior so launchd/Monit supervise parent liveness, not child compute responsiveness.
9. Add focused unit tests around success/failure/reclaim/lease/scheduling behavior.
10. Run required quality gates and classify any unrelated pre-existing failures.
11. If gates pass and the scoped diff is isolated, use `github:yeet` discipline to stage only intended files, commit, and push directly to `main`/`origin/main`.
12. Use `publish-ci-deploy` to watch CI/deploy, sync Mac Studio with `git pull --ff-only`, reload the affected services, and run the required smoke checks.

# Acceptance criteria (Definition of Done)

- API create path still persists `queued` and returns without executing full compute.
- Preflight classifies obvious heavy jobs as `heavy` using `estimated_combinations_upper_bound` or an equivalent conservative work estimate.
- Preflight classifies possible light jobs as `light_candidate`, not final `light`.
- Post-prepare refinement confirms `light_candidate` as `light` or promotes/requeues it to `heavy` before exact scoring.
- Parent runner can claim a queued job and launch exactly one child for that job.
- The full-job child process exits after one job, and parent evidence distinguishes parent retained memory from child peak memory.
- Parent runner can run two light children concurrently by default, while respecting configured cap.
- Parent runner never runs more than one heavy child on Mac Studio v1.
- Parent runner does not overlap light children with an active heavy child unless a separate benchmark-gated flag enables it.
- Heavy jobs are claimed FIFO by `created_at ASC, job_id ASC`.
- A stream of new light jobs cannot indefinitely starve an older queued heavy job.
- Child success reaches `succeeded` through the existing terminal persistence path.
- Child failure reaches `failed` with bounded error data and no secret/full payload logging.
- Full-job terminal writes happen in the parent only; child does not write terminal job state.
- Parent metrics endpoint is independent from child compute path.
- Monit config no longer restarts active compute solely because `/metrics` scrape times out.
- Tests cover child success, child failure, lease ownership, and no compute in API.
- Tests or static assertions prove production runner wiring does not construct the full compute service graph in the parent.
- Grep/static check proves `BacktestRuntimeJobOrchestrationService` is reachable from child entrypoint/direct benchmark/test code only, not from production parent service composition.
- Docs cleanup evidence proves active docs no longer describe the removed parent in-process compute path as current production behavior.
- Scoped changes are committed and pushed directly to `main`/`origin/main` after local gates, unless worktree scope is mixed or `gh`/GitHub auth is blocked.
- Mac Studio is synchronized from `main` with `git pull --ff-only`, affected services are reloaded, and smoke evidence is reported.

# Implementation constraints

## Determinism & ordering

- Preserve FIFO claim ordering unless a stronger existing repository rule says otherwise.
- Preserve heavy FIFO ordering by `created_at ASC, job_id ASC`.
- Preserve stable job, request hash, artifact pin, and variant identity semantics.

## API / contracts

- Public API compatibility target: `compatible-change`.
- Do not rename existing job states.
- If any DTO/persistence/config change is required for `scheduling_class`, `estimated_combinations_upper_bound`, queue priority, or slot accounting, classify it explicitly.

## Runtime / operations

- Parent process must remain the launchd service process.
- Child process must be disposable after one job.
- Default Mac Studio v1 slot policy: `light=2`, `heavy=1`, `light_heavy_overlap=false`.
- `light=3` is allowed only as a configured value after benchmark evidence proves safe CPU/RSS/latency behavior.
- Logs must be useful for production diagnosis without leaking secrets or large payloads.

# Files to indicate (expected touched areas)

Expected primary touches:

- `apps/worker/backtest_job_runner/**`
- `src/trading/contexts/backtest/application/use_cases/backtest_job_worker.py`
- `tests/unit/apps/worker/backtest_job_runner/**`
- `infra/scripts/monit/roehub-backtest-job-runner.monitrc`

Possible secondary touches:

- `infra/macos/launchd/com.roehub.backtest-job-runner.plist`
- `scripts/macos/reload_launchd_services.sh`
- `scripts/macos/bootstrap_native_prod.sh`
- `docs/runbooks/mac-studio-native-backend-operations.md`

# Non-goals

- Do not implement ordinal chunking or Numba kernel changes here.
- Do not run final production benchmark acceptance here.
- Do not move compute back into API.
- Do not process old queued jobs as acceptance evidence.
- Do not stop at local gates without direct `main`/`origin/main` delivery and Mac Studio smoke.
- Do not use `git add -A` when the worktree contains unrelated changes; follow `github:yeet` selective staging rules.

# Quality gates (must run and pass)

- `uv run pytest -q tests/unit/apps/worker/backtest_job_runner tests/unit/contexts/backtest/application/use_cases/test_backtest_job_worker_use_case.py`
- `uv run ruff check apps/worker src/trading/contexts/backtest tests/unit/apps/worker tests/unit/contexts/backtest/application/use_cases`
- `uv run pyright`
- `bash -n scripts/macos/bootstrap_native_prod.sh scripts/macos/reload_launchd_services.sh scripts/macos/smoke_prod.sh` if shell files changed
- `uv run python -m tools.docs.generate_docs_index --check` if docs changed
- `gh --version` and `gh auth status` before `github:yeet` publish

# Final output: report format (strict)

Write the final report in Russian with these sections:

- `Intent`
- `Process boundary`
- `Runner/Monit behavior`
- `Tests`
- `Contract impact`
- `Risks`
- `Publish`
- `CI/deploy`
- `Mac Studio delivery`
- `Next prompt`

Include exact commands run, pass/fail status, main push commit SHA, Mac Studio delivery evidence, and any pre-existing unrelated failures.
