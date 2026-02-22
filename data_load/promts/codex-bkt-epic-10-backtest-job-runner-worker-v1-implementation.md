---
# =========================
# Prompt Front-Matter (YAML)
# =========================
prompt_name: codex-bkt-epic-10-backtest-job-runner-worker-v1-implementation
repo: Dejetins/roehub.com
branch: main
scope: "Implement BKT-EPIC-10 end-to-end: Backtest job-runner worker v1 (claim/lease loop, streaming staged execution, progress snapshots, cancel handling, finalizing, and observability)"
language:
  implementation: python
  agent_report: ru
primary_docs:
  - docs/architecture/backtest/backtest-job-runner-worker-v1.md
  - docs/architecture/backtest/backtest-jobs-storage-pg-state-machine-v1.md
  - docs/architecture/roadmap/milestone-5-epics-v1.md
  - docs/architecture/roadmap/base_milestone_plan.md
  - docs/architecture/backtest/backtest-grid-builder-staged-runner-guards-v1.md
  - docs/architecture/backtest/backtest-reporting-metrics-table-v1.md
  - docs/architecture/backtest/backtest-api-post-backtests-v1.md
  - .codex/AGENTS.md
style_references:
  - src/trading/contexts/market_data/application/ports/stores/canonical_candle_reader.py
  - src/trading/contexts/market_data/application/ports/stores/raw_kline_writer.py
  - src/trading/contexts/market_data/application/ports/stores/enabled_instrument_reader.py
  - apps/worker/strategy_live_runner/main/main.py
  - apps/worker/strategy_live_runner/wiring/modules/strategy_live_runner.py
  - tests/unit/apps/test_strategy_live_runner_main.py
  - tests/unit/contexts/backtest/adapters/test_postgres_backtest_job_repositories.py
hard_requirements:
  keep_fail_fast_on_startup: true
  do_not_break_public_imports: true
  deterministic_ordering_everywhere: true
  stable_exports_via_init: true
  add_or_update_unit_tests_as_needed: true
  docstrings_must_link_to_docs_and_related_files: true
  maintain_existing_variant_key_v1_semantics: true
task_toggles:
  read_codex_agents_first: true
  if_agents_conflict_follow_main_prompt: true
  implement_new_backtest_job_runner_entrypoint: true
  implement_worker_wiring_module: true
  implement_streaming_orchestrator_for_jobs: true
  do_not_use_backtest_staged_runner_as_primary_jobs_engine: true
  implement_claim_poll_loop_with_lease: true
  implement_stage_a_streaming_shortlist: true
  persist_stage_a_shortlist_snapshot: true
  implement_stage_b_streaming_top_k_heap: true
  snapshot_trigger_is_or_condition: true
  implement_cancel_checks_on_batch_boundaries: true
  implement_lease_lost_fail_fast: true
  reclaim_policy_restart_attempt_from_beginning: true
  finalizing_compute_details_only_for_persisted_k: true
  persist_report_table_md_only_for_succeeded: true
  persist_trades_only_for_top_trades_n: true
  add_worker_prometheus_metrics: true
  update_init_exports_if_new_modules_added: true
  add_unit_tests_for_main_wiring_and_orchestrator: true
  do_not_implement_jobs_http_endpoints_epic_11: true
  update_docs_index_if_any_md_changed: true
target_envs:
  - dev
  - test
  - prod
required_literals:
  - backtest-job-runner
  - backtest.jobs.enabled
  - claim_poll_seconds
  - lease_seconds
  - heartbeat_seconds
  - snapshot_seconds
  - snapshot_variants_step
  - stage_a
  - stage_b
  - finalizing
  - succeeded
  - failed
  - cancelled
  - locked_by
  - lease_expires_at
  - FOR UPDATE SKIP LOCKED
  - report_table_md
  - top_trades_n
  - STRATEGY_PG_DSN
non_goals:
  - "Do not implement jobs HTTP endpoints from BKT-EPIC-11."
  - "Do not introduce a generic platform-wide jobs framework outside backtest context."
  - "Do not implement cursor-level Stage-B resume/checkpointing; restart attempt from beginning is required for v1."
  - "Do not change sync POST /backtests contracts or response shape."
  - "Do not store traceback in DB payloads."
final_report_format:
  language: ru
  sections:
    - "Результат"
    - "Изменённые файлы"
    - "Ключевые решения"
    - "Как проверить"
    - "Риски / что дальше"
quality_gates:
  - cmd: "uv run ruff check ."
    expect: "exit code 0"
  - cmd: "uv run pyright"
    expect: "exit code 0"
  - cmd: "uv run pytest -q"
    expect: "exit code 0"
  - cmd: "python -m tools.docs.generate_docs_index"
    expect:
      - "run only if any .md file was added/changed"
      - "docs/architecture/README.md updated accordingly"
expected_touched_paths:
  - apps/worker/backtest_job_runner/**
  - src/trading/contexts/backtest/application/use_cases/**
  - src/trading/contexts/backtest/application/services/**
  - src/trading/contexts/backtest/application/ports/**
  - src/trading/contexts/backtest/adapters/outbound/config/backtest_runtime_config.py
  - src/trading/contexts/backtest/adapters/outbound/persistence/postgres/**
  - src/trading/contexts/backtest/**/__init__.py
  - tests/unit/apps/**
  - tests/unit/contexts/backtest/**
  - docs/architecture/backtest/backtest-job-runner-worker-v1.md
  - docs/architecture/README.md
safety_notes:
  - "Preserve all EPIC-09 storage invariants and lease-owner conditional write semantics."
  - "Keep deterministic ordering and tie-break rules exactly as documented."
  - "Keep report_table_md NULL for non-succeeded jobs."
  - "Use structured logs without leaking secrets or raw tracebacks into DB/API payloads."
---
 Task
Implement the full BKT-EPIC-10 worker described in:
- `docs/architecture/backtest/backtest-job-runner-worker-v1.md`
Done means the repository has a production-ready async worker path for backtest jobs with deterministic behavior and tests.
 Context / Current State
- BKT-EPIC-09 is already implemented:
  - jobs schema and repositories exist (`backtest_jobs`, `backtest_job_top_variants`, `backtest_job_stage_a_shortlist`)
  - claim supports `FOR UPDATE SKIP LOCKED`
  - lease-guarded writes and deterministic pagination already exist.
- Sync path (`RunBacktestUseCase` + `BacktestStagedRunnerV1`) exists and is optimized for small sync requests.
- There is currently no worker under `apps/worker/backtest_job_runner/**`.
- Existing worker patterns exist in:
  - `apps/worker/strategy_live_runner/**`
  - `apps/worker/market_data_ws/**`
 Requirements (Must)
- Read `.codex/AGENTS.md` before any implementation. If instructions conflict, follow this prompt.
- Implement new worker process:
  - `apps/worker/backtest_job_runner/main/main.py`
  - `apps/worker/backtest_job_runner/wiring/modules/backtest_job_runner.py`
- Implement disabled semantics:
  - if `backtest.jobs.enabled=false` then log disabled state and exit with code `0`.
- Implement claim/poll flow:
  - poll every `claim_poll_seconds`
  - claim via `BacktestJobLeaseRepository.claim_next(...)`
  - process one claimed job at a time.
- Implement EPIC-10 execution flow with a **separate streaming orchestrator**:
  - do not use current `BacktestStagedRunnerV1` as the primary jobs engine for large jobs.
  - Stage A: streaming shortlist (base variants only, deterministic ranking by `total_return_pct DESC`, tie-break `base_variant_key ASC`).
  - Persist Stage A shortlist via `BacktestJobResultsRepository.save_stage_a_shortlist(...)`.
  - Stage B: streaming expanded scoring over risk axes with bounded running top-K structure.
  - Ranking in Stage B must be deterministic (`total_return_pct DESC`, tie-break `variant_key ASC`).
- Snapshot cadence must use OR semantics:
  - persist top snapshot when either:
    - elapsed time >= `snapshot_seconds`, OR
    - processed variants delta >= `snapshot_variants_step`.
- Progress semantics:
  - `stage_a`: units are base variants
  - `stage_b`: units are expanded variants
  - `finalizing`: fixed one-step progress (`processed_units=0`, `total_units=1` or equivalent deterministic fixed step)
- Cancel semantics:
  - check `cancel_requested_at` at batch boundaries
  - transition to `cancelled` (not `failed`)
  - stop further computation and writes except terminal transition write.
- Lease lost semantics:
  - on lease-guarded write failure (`None`/`False` from repository), stop processing immediately (fail-fast for that job).
- Reclaim semantics:
  - if job is reclaimed after lease expiry, restart attempt from beginning (Stage A -> Stage B -> finalizing).
- Finalizing semantics for succeeded jobs only:
  - compute details only for `persisted_k = min(request.top_k, backtest.jobs.top_k_persisted_default)`
  - build and persist `report_table_md` for all persisted rows
  - persist trades only for `rank <= top_trades_n`
  - keep `report_table_md` NULL for `running|cancelled|failed`.
- Failure semantics:
  - persist `last_error` and RoehubError-like `last_error_json {code,message,details}` for failed jobs
  - traceback must remain only in logs.
- Add/update classes and protocols with docstrings linking docs + related files.
- Update relevant `__init__.py` exports for all new modules.
- Add/adjust unit tests as needed.
- Keep all new files ruff/pyright clean and aligned with repository style references.
 Requirements (Should)
- Reuse existing EPIC-09 repository contracts without widening public API unless strictly necessary.
- Keep orchestration decomposition explicit (small classes/functions for claim loop, stage execution, snapshot policy, finalizing).
- Keep worker runtime deterministic and easy to test using fakes/stubs.
 Requirements (Nice-to-have)
- Add explicit helper(s) for snapshot cadence decision to keep logic testable.
- Add targeted unit tests for edge scenarios:
  - cancel arrives between batches,
  - lease lost during snapshot write,
  - reclaim leads to deterministic final output equivalence.
 Required reading (do first)
1) `.codex/AGENTS.md`  
2) `docs/architecture/backtest/backtest-job-runner-worker-v1.md`  
3) `docs/architecture/backtest/backtest-jobs-storage-pg-state-machine-v1.md`  
4) `docs/architecture/roadmap/milestone-5-epics-v1.md`  
5) `src/trading/contexts/backtest/application/ports/backtest_job_repositories.py`  
6) `src/trading/contexts/backtest/adapters/outbound/persistence/postgres/backtest_job_lease_repository.py`  
7) `src/trading/contexts/backtest/adapters/outbound/persistence/postgres/backtest_job_results_repository.py`  
8) `src/trading/contexts/backtest/application/services/staged_runner_v1.py`  
9) `src/trading/contexts/backtest/application/services/close_fill_scorer_v1.py`  
10) `src/trading/contexts/backtest/adapters/outbound/config/backtest_runtime_config.py`  
11) `apps/worker/strategy_live_runner/main/main.py`  
12) `apps/worker/strategy_live_runner/wiring/modules/strategy_live_runner.py`  
13) `src/trading/contexts/market_data/application/ports/stores/canonical_candle_reader.py`
 Work plan (agent should follow)
1) Read required docs/code and map reusable EPIC-09 + sync backtest components.  
2) Design worker runtime decomposition (entrypoint, wiring, orchestrator, snapshot policy, metrics hooks).  
3) Implement `backtest_job_runner` main entrypoint with config loading, disabled-mode exit, signal handling, and async loop.  
4) Implement wiring module to build dependencies (runtime config, compute, candle feed, repositories, orchestrator).  
5) Implement streaming orchestration for Stage A/Stage B/finalizing with deterministic ranking and bounded top-K memory behavior.  
6) Implement cancel checks, lease lost fail-fast behavior, reclaim restart-from-beginning policy, and robust failed-state persistence.  
7) Implement observability primitives (Prometheus metrics + structured logs with job_id/attempt/stage/event).  
8) Update `__init__.py` exports and docstrings with docs/related-file links for new classes/protocols.  
9) Add/adjust unit tests for worker main, wiring, orchestrator behavior, snapshot cadence, cancel, lease-loss, and finalizing policy.  
10) Run quality gates and fix all lint/type/test issues.
 Acceptance criteria (Definition of Done)
- New worker process starts and runs with valid config.
- `backtest.jobs.enabled=false` path exits with code `0` and does not claim jobs.
- Claim loop processes jobs via repository contracts without double execution semantics.
- Stage A and Stage B progress updates are persisted with documented stage unit semantics.
- Snapshot cadence uses OR condition (`snapshot_seconds` OR `snapshot_variants_step`).
- Cancel moves running job to `cancelled` without marking `failed`.
- Lease lost stops further job writes immediately.
- Reclaimed job restarts from beginning and eventually reaches terminal state.
- Succeeded finalizing persists `report_table_md` for persisted rows and trades only for `top_trades_n`.
- Non-succeeded snapshots keep `report_table_md` as NULL.
- Existing sync `POST /backtests` behavior remains unchanged.
- `uv run ruff check .`, `uv run pyright`, and `uv run pytest -q` pass.
 Implementation constraints
 Determinism & ordering
- Keep deterministic sort contracts exactly:
  - Stage A: `total_return_pct DESC`, tie-break `base_variant_key ASC`
  - Stage B: `total_return_pct DESC`, tie-break `variant_key ASC`
- Do not depend on dict iteration order; normalize/sort where needed.
- Use deterministic hash/key behavior and keep existing variant-key semantics unchanged.
- Keep all timestamps UTC-aware in domain/application logic.
 API / contracts (if relevant)
- Do not add or change jobs HTTP endpoints (EPIC-11 scope).
- Do not break existing EPIC-09 repository contracts unless absolutely necessary; if changed, update all adapters/tests/docs accordingly.
- Keep RoehubError-like failure payload shape (`code`, `message`, `details`) for persisted failed jobs.
- Do not change sync `RunBacktestUseCase` request/response contracts.
 Documentation
- Every new class/protocol must include docstrings with links to architecture docs and related files.
- Update related architecture docs only if implementation reveals a necessary contract correction.
- If any `.md` is added or modified, run:
  - `python -m tools.docs.generate_docs_index`
 Tests
- Add unit tests for:
  - worker main disabled mode (`exit 0`)
  - wiring fail-fast behavior (missing required env/config)
  - orchestrator stage progress semantics
  - snapshot OR cadence
  - cancel at batch boundaries
  - lease lost fail-fast stop
  - finalizing persisted_k/trades policy.
- Prefer deterministic fakes/stubs; avoid external services/network/time randomness.
- Keep tests concise and behavior-focused (contract-level assertions).
 Files to indicate (expected touched areas)
- `apps/worker/backtest_job_runner/main/main.py`
- `apps/worker/backtest_job_runner/main/__init__.py`
- `apps/worker/backtest_job_runner/wiring/modules/backtest_job_runner.py`
- `apps/worker/backtest_job_runner/wiring/modules/__init__.py`
- `apps/worker/backtest_job_runner/wiring/__init__.py`
- `apps/worker/backtest_job_runner/__init__.py`
- `src/trading/contexts/backtest/application/use_cases/**`
- `src/trading/contexts/backtest/application/services/**`
- `src/trading/contexts/backtest/application/ports/**`
- `src/trading/contexts/backtest/adapters/outbound/config/backtest_runtime_config.py`
- `src/trading/contexts/backtest/**/__init__.py`
- `tests/unit/apps/**`
- `tests/unit/contexts/backtest/**`
- `docs/architecture/backtest/backtest-job-runner-worker-v1.md`
- `docs/architecture/README.md`
 Non-goals
- Implementing jobs HTTP endpoints and transport DTOs (EPIC-11).
- Building generic platform jobs infrastructure outside backtest context.
- Implementing precise Stage-B cursor checkpoint resume.
- Implementing retention/cleanup policies for jobs/results tables.
- Introducing strict latency SLA assertions in tests.
 Quality gates (must run and pass)
- `uv run ruff check .`
- `uv run pyright`
- `uv run pytest -q`
 Final output: report format (strict)
Your final message MUST be in Russian and follow exactly:
1) **Результат**
2) **Изменённые файлы**
3) **Ключевые решения**
4) **Как проверить**
5) **Риски / что дальше**