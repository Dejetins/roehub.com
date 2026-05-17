---
prompt_name: backtest_ai_configurator_mlx_v1_01_storage_quota_contracts
repo: roehub.com
branch: main
status: superseded
do_not_execute: true
scope: "Iteration 01: implement durable storage, DTOs, repository ports, quota accounting, idempotency, and lease contracts for /backtests AI configurator without enabling MLX or UI."
superseded_reason: "Retired as an executable prompt on 2026-05-17 during LM Studio tools cleanup. The implemented storage/quota foundation is retained, but this old prompt pack must not be rerun."
replacement_direction: "Use the forthcoming tool-based LM Studio prompt pack while preserving the existing storage, quota, idempotency and lease foundation."

language:
  implementation: python_fastapi_postgres
  agent_report: ru

context_sources:
  always_read:
    - path: .codex/AGENTS.md
      why: "repo contract and skill routing"
    - path: docs/architecture/backtest/backtest-ai-configurator-mlx-v1.md
      why: "AI configurator source of truth"
  task_entrypoints:
    - path: src/trading/contexts/backtest/application/dto/runtime_preflight.py
      why: "current backtest request/default DTO style"
      inspect_symbols:
        - BacktestRuntimeDefaults
        - BacktestValidationIssue
    - path: src/trading/contexts/backtest/application/use_cases/backtest_jobs.py
      why: "nearby durable job/use-case patterns"
      inspect_symbols:
        - BacktestJobsUseCase
    - path: src/trading/contexts/backtest/adapters/outbound/persistence/postgres
      why: "Postgres repository patterns for backtest context"
      inspect_symbols:
        - "*Repository"
    - path: apps/api/wiring/modules/backtest.py
      why: "composition root pattern for backtest services"
      inspect_symbols:
        - build_backtest_use_cases
  conditional_bundles:
    migrations:
      read_when: "before adding Alembic migration"
      paths:
        - alembic/versions
        - alembic/env.py
    paid_level:
      read_when: "when resolving subscription tier/paid level source"
      paths:
        - src
        - apps/api
    existing_job_storage:
      read_when: "if repository or idempotency behavior is unclear"
      paths:
        - src/trading/contexts/backtest/adapters/outbound/persistence/postgres
        - tests/unit/contexts/backtest
  consult_if_needed:
    - path: docs/architecture/backtest/backtest-job-runner-production-plan-v1.md
      read_when: "if lease or worker queue semantics conflict"
    - path: .codex/agents/.context/promt_manager_state.yaml
      read_when: "only to check for a newer executor handoff; ignore if stale/unrelated"

style_references:
  - path: src/trading/contexts/backtest/application/dto/runtime_preflight.py
    purpose: "dataclass DTO style and as_mapping patterns"
  - path: src/trading/contexts/backtest/adapters/outbound/persistence/postgres
    purpose: "repository and row mapping style"

hard_requirements:
  docs_source_of_truth: true
  no_mlx_runtime: true
  no_ui_enablement: true
  no_existing_backtest_hash_change: true
  durable_storage_required: true
  idempotency_required: true
  owner_scope_required: true
  publish_ci_deploy_required: true
  main_branch_deployment_required: true
  macstudio_sync_required: true

task_toggles:
  implement_storage: true
  implement_quota: true
  implement_api_routes: false
  implement_worker: false
  implement_mlx: false
  implement_ui: false

skill_routing:
  - skill: contract-impact-analysis
    use_when: "before adding migrations, DTOs, config, quota or idempotency semantics"
    timing: "before implementation"
    reason: "persistence, DTO and config compatibility"
  - skill: backend-quality-gates
    use_when: "running focused unit tests, ruff or pyright"
    timing: "during verification"
    reason: "backend correctness gates"
  - skill: root-cause-debugging
    use_when: "a repository/migration/test failure appears"
    timing: "if blocker"
    reason: "localize failing gate before changing scope"
  - skill: publish-ci-deploy
    use_when: "after implementation and local gates pass, deliver this iteration to main, sync Mac Studio, and run post-deploy verification"
    timing: "final delivery step"
    reason: "required end-to-end Roehub GitHub CI, main deployment, Mac Studio sync and smoke"

target_envs:
  - local-dev
  - unit-tests
  - github-actions
  - mac-studio-prod

required_literals:
  - "backtest_ai_config_jobs"
  - "backtest_ai_config_events"
  - "backtest_ai_config_llm_attempts"
  - "backtest_ai_quota_events"
  - "idempotency_key"
  - "lease_expires_at"
  - "quota_charged"

non_goals:
  - "Do not add browser-visible routes in this iteration."
  - "Do not call MLX, mlx_lm.server, or any model runtime."
  - "Do not enable ai_configurator_state.enabled."
  - "Do not change /backtests/jobs request_hash semantics."

final_report_format:
  language: ru
  sections:
    - "Что реализовано"
    - "Контракты и совместимость"
    - "Проверки"
    - "Доставка и Mac Studio"
    - "Остаточные риски"
    - "Следующая итерация"

quality_gates:
  - cmd: "uv run ruff check src/trading/contexts/backtest apps/api tests/unit/contexts/backtest tests/unit/apps/api"
    expect: "passes or failure is classified"
  - cmd: "uv run pyright"
    expect: "passes or failure is classified"
  - cmd: "uv run pytest -q tests/unit/contexts/backtest/application/ai_configurator tests/unit/apps/api/test_backtest_ai_config_storage.py"
    expect: "new focused tests pass"
  - cmd: "git diff --check"
    expect: "passes"

expected_primary_touches:
  - "alembic/versions/*backtest_ai_config*.py"
  - "src/trading/contexts/backtest/application/ai_configurator/"
  - "src/trading/contexts/backtest/application/ports/"
  - "src/trading/contexts/backtest/adapters/outbound/persistence/postgres/"
  - "tests/unit/contexts/backtest/application/ai_configurator/"
  - "tests/unit/apps/api/test_backtest_ai_config_storage.py"

possible_secondary_touches:
  - "apps/api/wiring/modules/backtest.py"
  - "apps/api/dto/backtests.py"
  - "configs/prod/backtest_ai_configurator.yaml"

safety_notes:
  - "Persist user prompts only in AI audit tables; do not leak them through normal list/read APIs."
  - "Additive schema only. Existing backtest job tables and hashes must remain unchanged."
  - "If PaidLevel source is unclear, implement quota as a port with deterministic test fake and report the integration point."
---

# Task

Implement Iteration 01 of the `/backtests` AI Configurator: durable storage contracts, application DTOs, repository ports/adapters, quota event accounting, idempotency, and lease recovery primitives.

Done means:

- migrations define the AI configurator tables and indexes from the architecture plan;
- application-layer DTOs and repository ports exist under the backtest bounded context;
- Postgres adapters can create/read/update/claim jobs and append events/attempts/quota events;
- idempotent create semantics are represented at the repository/service boundary;
- quota windows for `5h` and `week` are implemented or cleanly abstracted behind a port;
- focused tests prove idempotency, owner scope, lease recovery, quota accounting and retention query shape.

## Context / Current State

Context ledger:

- completed:
  - architecture source exists at `docs/architecture/backtest/backtest-ai-configurator-mlx-v1.md`;
  - current `/backtests/jobs` path is already durable queued and must not be changed;
  - current `/backtests` runtime defaults and validation live in `BacktestPreflightService`.
- open_items:
  - no AI configurator storage, API, worker or UI exists yet;
  - exact tier quota values are config defaults until product finalizes them;
  - this iteration must not call a model.
- contract_changes:
  - additive AI-only tables and DTOs;
  - no change to existing public `/backtests/jobs` request hash or job persistence.
- touched_paths:
  - expected new `src/trading/contexts/backtest/application/ai_configurator/*`;
  - expected new migration and Postgres repository adapter.
- risks:
  - double charging quota if idempotency is not modeled now;
  - stale leases creating duplicate work if claim/update semantics are weak;
  - raw prompt/audit data leaking through normal APIs later.
- next_focus:
  - durable storage foundation for API shell and worker iterations.

Additional context:

- Required job states include `queued`, `preparing_catalog`, `generating`, `validating`, `repairing`, `ready`, `needs_clarification`, `blocked_by_policy`, `input_too_large`, `security_review`, `failed`.
- Required tables are `backtest_ai_config_jobs`, `backtest_ai_config_events`, `backtest_ai_config_llm_attempts`, `backtest_ai_quota_events`.

## Requirements (Must)

- Read context using the protocol below and stop early once sufficient.
- Implement only the storage/quota/contracts foundation.
- Additive migrations only; no destructive or broad schema changes.
- Preserve existing `/backtests/jobs` behavior and request identity semantics.
- Store prompt/audit data only in AI-specific tables.
- Include `owner_user_id` on all user-owned rows.
- Include `idempotency_key`, `user_prompt_hash`, `current_config_hash`, `catalog_snapshot_hash`, `runtime_defaults_hash`, `system_prompt_version`, `model_id`, `model_path_hash`, lease fields, attempt fields, `quota_charged`, timestamps and feedback fields where applicable.
- Add indexes for owner/state/lease/idempotency/retention queries.
- Provide repository methods for create-or-get-by-idempotency, get by owner/job, claim next job with lease, heartbeat, append event, append attempt, mark terminal, record feedback, and quota event writes.
- Tests must prove owner isolation for read/update operations.

## Requirements (Should)

- Prefer dataclass DTOs and ports consistent with the backtest application layer.
- Keep quota values config-driven, with safe defaults from the architecture doc.
- Make retention cleanup queryable but do not implement a scheduler unless local patterns make it trivial.
- Use JSONB for raw audit payloads and typed DTOs at application boundaries.

## Requirements (Nice-to-have)

- Include helper methods for prompt/config hashes using canonical JSON where useful.
- Add a small test fixture factory for AI config jobs to help later iterations.

# Context acquisition protocol

Read only in this order and stop once sufficient:

1. `.codex/AGENTS.md`
2. `.codex/agents/.context/promt_manager_state.yaml` or latest state snapshot, if available and relevant
3. latest executor final report, if available
4. task entrypoints
5. only the conditional bundle(s) required by touched contracts or failing checks
6. consult-if-needed references only for blockers, ambiguity, or conflicts

Do not eagerly preload all listed sources.

Pre-implementation reading target:

- `<= 8 files`
- `<= ~35k-50k tokens`

Stop reading once changed contracts are identified, touched files are bounded, acceptance criteria are implementable, and no unresolved persistence-contract ambiguity remains.

Expand context only for blockers, failing quality gates, unclear migration conventions, or contract conflicts.

# Reading manifest

Use the front-matter `context_sources` as the canonical reading map.

Read with this intent:

- `always_read`: repository rules and AI configurator architecture.
- `task_entrypoints`: current DTO, use-case, repository and wiring style.
- `conditional_bundles`: migrations, paid level and existing storage only when needed.
- `consult_if_needed`: broader plans only for conflicts.

Do not convert this manifest into a broad mandatory reading list.

# Work plan (agent should follow)

Skill routing for this task:

- `contract-impact-analysis`: use before implementation; owns persistence/config/idempotency compatibility.
- `backend-quality-gates`: use during verification; owns ruff/pyright/pytest triage.
- `root-cause-debugging`: use only if a gate fails or storage behavior is inconsistent.

1. Identify the current migration/repository patterns and the smallest additive schema.
2. Add AI config DTOs, status literals/enums, repository ports, and Postgres row mapping.
3. Add migrations and indexes for the four AI config tables.
4. Implement quota and idempotent create/claim/lease operations with deterministic timestamps where tests need them.
5. Add focused tests for storage, idempotency, lease recovery, owner scope and quota windows.
6. Run quality gates and report contract classification.

# Acceptance criteria (Definition of Done)

- New schema is additive and includes required tables/indexes.
- Repository tests pass for create/get, idempotency, owner isolation, claim lease, expired lease reclaim, terminal state and quota event accounting.
- No existing backtest job tests are broken.
- Existing `/backtests/jobs` public route and request hash semantics are untouched.
- Final report states exact migrations, DTOs, ports, adapters and tests added.

- `publish-ci-deploy` terminal state is `deployed`, or `green-pr`/`blocked` is reported with exact blocker evidence.

# Implementation constraints

## Determinism & ordering

- Keep claim ordering deterministic: oldest eligible queued/expired job first unless local patterns require a stronger priority field.
- Use database time or injected clock consistently; do not mix hidden wall-clock behavior in tests.

## API / contracts

- Do not expose raw `backtest_ai_config_llm_attempts` through public API in this iteration.
- Do not add browser-visible endpoints yet.

## Data safety

- Do not store secrets in config defaults or tests.
- Tests may use synthetic prompts only.
- Raw audit rows must be owner-scoped and restricted by repository methods.

## Performance

- Add indexes required by the architecture doc.
- Do not benchmark in this iteration.

# Files to indicate (expected touched areas)

Expected primary touches:

- `alembic/versions/*backtest_ai_config*.py`
- `src/trading/contexts/backtest/application/ai_configurator/`
- `src/trading/contexts/backtest/application/ports/`
- `src/trading/contexts/backtest/adapters/outbound/persistence/postgres/`
- `tests/unit/contexts/backtest/application/ai_configurator/`

Possible secondary touches:

- `apps/api/wiring/modules/backtest.py`
- `apps/api/dto/backtests.py`
- `configs/prod/backtest_ai_configurator.yaml`

# Non-goals

- No API routes.
- No SSE.
- No fake worker.
- No MLX/model adapter.
- No Web UI changes.
- No production launchd/Monit changes.

# Quality gates (must run and pass)

- `uv run ruff check src/trading/contexts/backtest apps/api tests/unit/contexts/backtest tests/unit/apps/api`
- `uv run pyright`
- `uv run pytest -q tests/unit/contexts/backtest/application/ai_configurator tests/unit/apps/api/test_backtest_ai_config_storage.py`
- `git diff --check`

If a gate cannot run, classify it as introduced, required-path pre-existing, unrelated pre-existing, environmental, or flaky.

Required delivery step: after the quality gates above pass, invoke `publish-ci-deploy` as the final step. The expected terminal state for this prompt is `deployed`: intended files committed and pushed, GitHub Actions green, revision shipped to `main`, `/opt/roehub/app` on `macstudio` pulled to that revision, the relevant production services reloaded through the repository runbook, and `bash scripts/macos/smoke_prod.sh` passed. If the skill reaches `green-pr` because a human merge/approval is required, or `blocked` because of missing auth, unrelated dirty scope, external CI, Mac Studio access, or production verification failure, report that exact state and do not claim deployment.

# Final output: report format (strict)

Report in Russian with:

- `Что реализовано`: storage, DTO, port, adapter, migration summary.
- `Контракты и совместимость`: public API, persisted schema, config, request hash.
- `Проверки`: exact commands and results.
- `Доставка и Mac Studio`: publish-ci-deploy terminal state, main/PR SHA, CI result, Mac Studio pull/reload/smoke evidence, or exact blocker.
- `Остаточные риски`: unresolved tier source, retention scheduler, or env constraints.
- `Следующая итерация`: API shell + fake worker readiness.
