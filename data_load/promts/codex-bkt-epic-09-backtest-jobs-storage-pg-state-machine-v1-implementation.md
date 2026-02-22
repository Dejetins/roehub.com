---
# =========================
# Prompt Front-Matter (YAML)
# =========================
prompt_name: codex-bkt-epic-09-backtest-jobs-storage-pg-state-machine-v1-implementation
repo: Dejetins/roehub.com
branch: main
scope: "Implement BKT-EPIC-09 end-to-end: Backtest Jobs v1 storage in Postgres (schema, state machine, repositories, deterministic SQL contracts, jobs runtime config, and config hash)"
language:
  implementation: python
  agent_report: ru
primary_docs:
  - docs/architecture/backtest/backtest-jobs-storage-pg-state-machine-v1.md
  - docs/architecture/roadmap/milestone-5-epics-v1.md
  - docs/architecture/roadmap/base_milestone_plan.md
  - docs/architecture/backtest/backtest-api-post-backtests-v1.md
  - docs/architecture/backtest/backtest-grid-builder-staged-runner-guards-v1.md
  - docs/architecture/backtest/backtest-reporting-metrics-table-v1.md
  - docs/architecture/api/api-errors-and-422-payload-v1.md
  - .codex/AGENTS.md
style_references:
  - src/trading/contexts/market_data/application/ports/stores/canonical_candle_reader.py
  - src/trading/contexts/market_data/application/ports/stores/raw_kline_writer.py
  - src/trading/contexts/strategy/adapters/outbound/persistence/postgres/strategy_run_repository.py
  - tests/unit/contexts/strategy/adapters/test_postgres_strategy_repositories.py
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
  implement_alembic_migration_for_backtest_jobs: true
  implement_backtest_job_domain_state_machine: true
  implement_backtest_job_repository_ports: true
  implement_postgres_job_adapters_with_explicit_sql: true
  implement_deterministic_keyset_pagination: true
  implement_claim_with_skip_locked: true
  enforce_lease_owner_conditional_writes: true
  implement_replace_snapshot_write_policy: true
  enforce_saved_mode_spec_snapshot_required: true
  extend_backtest_runtime_config_jobs_section: true
  strict_required_jobs_config_keys: true
  implement_backtest_runtime_config_hash_result_affecting_only: true
  include_jobs_top_k_persisted_default_in_runtime_hash: true
  update_init_exports_if_new_modules_added: true
  add_unit_tests_for_state_machine_sql_and_config_hash: true
  do_not_implement_worker_loop_epic_10: true
  do_not_implement_jobs_api_endpoints_epic_11: true
  update_docs_index_if_any_md_changed: true
target_envs:
  - dev
  - test
  - prod
required_literals:
  - backtest_jobs
  - backtest_job_top_variants
  - backtest_job_stage_a_shortlist
  - queued
  - running
  - succeeded
  - failed
  - cancelled
  - stage_a
  - stage_b
  - finalizing
  - FOR UPDATE SKIP LOCKED
  - STRATEGY_PG_DSN
  - POSTGRES_DSN
  - backtest.jobs.top_k_persisted_default
non_goals:
  - "Do not implement worker execution loop, heartbeat loop orchestration, or metrics emission (EPIC-10)."
  - "Do not implement new jobs API endpoints (EPIC-11)."
  - "Do not introduce generic platform-wide jobs framework outside backtest context."
  - "Do not implement retention/cleanup policies for old jobs/results."
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
  - cmd: "python -m tools.docs.generate_docs_index --check"
    expect:
      - "exit code 0"
      - "run only if any .md file was added/changed"
expected_touched_paths:
  - alembic/versions/*_backtest_jobs_v1.py
  - src/trading/contexts/backtest/domain/**
  - src/trading/contexts/backtest/application/ports/**
  - src/trading/contexts/backtest/adapters/outbound/persistence/postgres/**
  - src/trading/contexts/backtest/adapters/outbound/config/backtest_runtime_config.py
  - src/trading/contexts/backtest/adapters/outbound/config/__init__.py
  - src/trading/contexts/backtest/adapters/outbound/__init__.py
  - src/trading/contexts/backtest/adapters/__init__.py
  - src/trading/contexts/backtest/application/ports/__init__.py
  - src/trading/contexts/backtest/application/__init__.py
  - src/trading/contexts/backtest/domain/__init__.py
  - configs/dev/backtest.yaml
  - configs/test/backtest.yaml
  - configs/prod/backtest.yaml
  - tests/unit/contexts/backtest/**
safety_notes:
  - "Preserve existing sync POST /backtests behavior and public contracts from Milestone 4."
  - "Keep deterministic ordering in every list/select/sort operation; always add explicit ORDER BY in SQL reads."
  - "Keep report_table_md NULL for cancelled/failed/running snapshots; store best-so-far ranking rows only."
  - "Traceback must stay in logs only; last_error_json in DB must be RoehubError-like payload."
---

# Task

Implement the full storage/domain/config layer defined by:

- `docs/architecture/backtest/backtest-jobs-storage-pg-state-machine-v1.md`

for BKT-EPIC-09.

Done means the codebase includes:

1) Alembic migration(s) for all required backtest jobs tables + constraints/indexes.
2) Domain/application contracts for job lifecycle, states/stages, and repository ports.
3) Postgres adapters implementing deterministic SQL rules, claim semantics, lease-safe conditional writes, and keyset list pagination.
4) Runtime config extension `backtest.jobs.*` with strict fail-fast validation and deterministic `backtest_runtime_config_hash` logic.
5) Updated exports (`__init__.py`) and unit tests for schema contracts, repository SQL behavior, and config validation/hash behavior.

Your final report MUST be in Russian and follow the strict section order defined in front-matter.

## Context / Current State

- Milestone 4 sync backtest exists and is working.
- Backtest jobs storage is not implemented yet.
- Backtest runtime config currently has warmup/top_k/preselect + execution/reporting, but no jobs section.
- Backtest adapter folders for persistence/progress exist but have no implementation files for jobs.
- Existing repository style uses:
  - explicit SQL adapters,
  - deterministic ordering,
  - strict dataclass/protocol contracts,
  - fail-fast config loaders.

## Requirements (Must)

- Read `.codex/AGENTS.md` before coding. If there is any conflict, follow this prompt.
- Implement schema for:
  - `backtest_jobs`
  - `backtest_job_top_variants`
  - `backtest_job_stage_a_shortlist`
- Enforce state/stage literals and invariants from the architecture doc.
- Enforce saved/template consistency in storage:
  - `mode=saved` requires `spec_hash` + `spec_payload_json`.
- Enforce lifecycle rule: `queued -> failed` is not allowed by domain contract.
- Add repository ports:
  - `BacktestJobRepository`
  - `BacktestJobLeaseRepository`
  - `BacktestJobResultsRepository`
- Implement Postgres adapters with explicit SQL (no ORM):
  - claim query must use `FOR UPDATE SKIP LOCKED`
  - claim FIFO order: `created_at ASC, job_id ASC`
  - keyset list order: `created_at DESC, job_id DESC` + cursor `{created_at, job_id}`
  - conditional running writes by active lease owner: `(job_id, locked_by, lease_expires_at > now())`
- Implement snapshot write policy in results repository:
  - replace whole top-K snapshot in one transaction (delete old + insert new).
- Keep best-so-far ranking rows for cancelled/failed jobs, but keep `report_table_md` as NULL for non-succeeded jobs.
- Extend `backtest.yaml` loader and config dataclasses with strict required `backtest.jobs.*` keys:
  - enabled
  - top_k_persisted_default
  - max_active_jobs_per_user
  - claim_poll_seconds
  - lease_seconds
  - heartbeat_seconds
  - parallel_workers
  - optional: snapshot_seconds and/or snapshot_variants_step
- Add deterministic `backtest_runtime_config_hash` helper that includes only result-affecting fields and MUST include `backtest.jobs.top_k_persisted_default`.
- Add/adjust unit tests to cover:
  - config loader strict validation for jobs section
  - deterministic hash behavior
  - SQL ordering/claim semantics (including presence of SKIP LOCKED)
  - lifecycle/state invariants and core repository behaviors
- Add docstrings with links to docs/related files for all new classes/protocols.
- Update all relevant `__init__.py` exports for new modules.

## Requirements (Should)

- Reuse existing style patterns from market_data/strategy ports and postgres adapters.
- Keep repository methods small and explicit; avoid hidden side effects.
- Keep exception mapping deterministic and aligned with RoehubError semantics.

## Requirements (Nice-to-have)

- Add concise helper constructors/parsers for cursor payloads to keep pagination logic easy to test.
- Add SQL shape tests that assert deterministic ORDER BY clauses explicitly.

# Required reading (do first)

1) `.codex/AGENTS.md`
2) `docs/architecture/backtest/backtest-jobs-storage-pg-state-machine-v1.md`
3) `docs/architecture/roadmap/milestone-5-epics-v1.md`
4) `docs/architecture/roadmap/base_milestone_plan.md`
5) `src/trading/contexts/backtest/adapters/outbound/config/backtest_runtime_config.py`
6) `src/trading/contexts/strategy/adapters/outbound/persistence/postgres/strategy_run_repository.py`
7) `src/trading/contexts/market_data/application/ports/stores/canonical_candle_reader.py`
8) `tests/unit/contexts/strategy/adapters/test_postgres_strategy_repositories.py`

# Work plan (agent should follow)

1) Design and implement domain value objects/entities/errors for backtest job state machine and invariants.
2) Introduce application ports for jobs repositories (core, lease, results).
3) Add Alembic migration with all tables, constraints, indexes.
4) Implement Postgres gateways/adapters with explicit deterministic SQL.
5) Extend backtest runtime config (`jobs` section) and add deterministic config-hash helper.
6) Update public exports in `__init__.py` files.
7) Add/extend unit tests for config, domain invariants, and repository SQL/behavior.
8) Run quality gates and fix all issues.

# Acceptance criteria (Definition of Done)

- Migration exists and can be applied/downgraded without contract violations.
- Repository ports + Postgres adapters exist and satisfy deterministic SQL rules.
- Claim query uses `FOR UPDATE SKIP LOCKED` and FIFO ordering.
- List pagination is keyset-based with deterministic order and cursor contract.
- Runtime config loader fails fast when required `backtest.jobs.*` keys are missing/invalid.
- `backtest_runtime_config_hash` is deterministic and includes `jobs.top_k_persisted_default` while excluding operational-only fields.
- Saved-mode storage contract enforces required strategy snapshot fields.
- Unit tests pass and cover newly introduced contracts.
- `uv run ruff check .`, `uv run pyright`, `uv run pytest -q` all pass.

# Implementation constraints

## Determinism & ordering

- Use explicit ORDER BY in every SQL read path.
- For equal sort keys, always add deterministic tie-break by `job_id` or `variant_key` as specified.
- Use canonical JSON hashing (`sort_keys`, compact separators, ensure_ascii=true).

## API / contracts (if relevant)

- Do not change existing sync `POST /backtests` request/response contracts.
- Do not add jobs API endpoints in this epic.
- Keep RoehubError compatibility for persisted `last_error_json` shape (`code/message/details`).

## Documentation

- Add doc/related-file links in docstrings for all new classes/protocols.
- If any `.md` files are added/changed, run:
  - `python -m tools.docs.generate_docs_index`

## Tests

- Follow existing repository test style for fake gateways and deterministic SQL assertions.
- Add tests only where they validate new contracts/invariants; avoid noisy duplication.

# Files to indicate (expected touched areas)

- `alembic/versions/*_backtest_jobs_v1.py`
- `src/trading/contexts/backtest/domain/**`
- `src/trading/contexts/backtest/application/ports/**`
- `src/trading/contexts/backtest/adapters/outbound/persistence/postgres/**`
- `src/trading/contexts/backtest/adapters/outbound/config/backtest_runtime_config.py`
- `configs/dev/backtest.yaml`
- `configs/test/backtest.yaml`
- `configs/prod/backtest.yaml`
- `tests/unit/contexts/backtest/**`

# Non-goals

- Worker loop orchestration (poll/heartbeat runtime loop, metrics loop) from EPIC-10.
- Jobs HTTP endpoints and transport DTOs from EPIC-11.
- Generic reusable jobs module for non-backtest contexts.
- Retention/cleanup of old jobs and result rows.

# Quality gates (must run and pass)

- `uv run ruff check .`
- `uv run pyright`
- `uv run pytest -q`

# Final output: report format (strict)

Your final message MUST be in Russian and follow exactly:

1) **Результат**

2) **Изменённые файлы**

3) **Ключевые решения**

4) **Как проверить**

5) **Риски / что дальше**
