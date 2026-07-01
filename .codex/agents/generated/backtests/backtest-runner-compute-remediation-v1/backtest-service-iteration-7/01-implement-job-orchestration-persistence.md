---
prompt_name: backtest_service_iteration_7_job_orchestration_persistence
repo: roehub.com
branch: current
scope: "Iteration 7: implement job orchestration, public job API contracts, top-result assembly, and summary persistence for no-risk and TP/SL backtests."

language:
  implementation: python
  agent_report: ru

context_sources:
  always_read:
    - path: .codex/AGENTS.md
      why: "repo engineering contract, delivery rules, and merge/deploy expectations"
    - path: docs/architecture/backtest/backtest-service-artifact-runtime-v1.ru.md
      why: "canonical Iteration 7 contract, public API vocabulary, variant identity, progress mapping, and persistence gate"
    - path: docs/architecture/backtest/benchmark_iterations/README.md
      why: "benchmark evidence manifest, service-only stage accounting, and Mac Studio acceptance rules"
    - path: docs/architecture/backtest/benchmark_iterations/2026-04-26_engine_test_btcusdt_15m/benchmark_results.json
      why: "canonical fixture and top-row parity target for end-to-end job runs"
  task_entrypoints:
    - path: apps/api/routes/backtests.py
      why: "current public backtests API shell; Iteration 7 adds jobs endpoints here"
      inspect_symbols:
        - build_backtests_router
    - path: apps/api/dto/backtests.py
      why: "current public DTO response style for runtime-defaults and preflight"
      inspect_symbols:
        - BacktestRuntimeDefaultsResponse
        - BacktestPreflightResponse
    - path: apps/api/wiring/modules/backtest.py
      why: "current API wiring for backtest services, artifact config, defaults, and auth dependency"
      inspect_symbols:
        - build_backtests_router
    - path: src/trading/contexts/backtest/application/ports/backtest_job_repositories.py
      why: "existing job repository and lease ports for create/get/list/cancel/progress/finish"
      inspect_symbols:
        - BacktestJobRepository
        - BacktestJobLeaseRepository
        - BacktestJobListQuery
    - path: src/trading/contexts/backtest/domain/entities/backtest_job.py
      why: "existing job lifecycle aggregate, state/stage model, artifact pin, progress, and request hash invariants"
      inspect_symbols:
        - BacktestJob
        - BacktestJobState
        - BacktestJobStage
        - BacktestJobArtifactPin
    - path: src/trading/contexts/backtest/domain/entities/backtest_job_results.py
      why: "existing summary-only top variant storage contract and SHA-only storage-key constraint"
      inspect_symbols:
        - BacktestJobTopVariant
    - path: src/trading/contexts/backtest/adapters/outbound/persistence/postgres/backtest_job_repository.py
      why: "existing Postgres adapter for job and top-variant persistence"
      inspect_symbols:
        - PostgresBacktestJobRepository
        - create_with_top_variants
  conditional_bundles:
    iteration_6_acceptance:
      read_when: "before implementation; if accepted Iteration 6 evidence is missing, stop and report the precondition blocker"
      paths:
        - docs/architecture/backtest/benchmark_iterations/
      instruction: "Find the latest folder matching `*_iteration_6_tp_sl_exact_scoring_full_metrics` and read its `benchmark_summary.md` plus `benchmark_results.json`."
    accepted_runtime_services:
      read_when: "building the end-to-end job runner or top-result assembly for no-risk and TP/SL"
      paths:
        - src/trading/contexts/backtest/application/services/v2/preflight.py
        - src/trading/contexts/backtest/application/services/v2/prepare_pools.py
        - src/trading/contexts/backtest/application/services/v2/combo_planning.py
        - src/trading/contexts/backtest/application/services/v2/no_risk_exact.py
        - src/trading/contexts/backtest/application/services/v2/tp_sl_hit_times.py
        - src/trading/contexts/backtest/application/services/v2/tp_sl_exact.py
        - src/trading/contexts/backtest/application/services/v2/benchmark_accounting.py
    postgres_schema:
      read_when: "changing persistence behavior, adding idempotency storage, or deciding whether a migration is required"
      paths:
        - alembic/versions/20260222_0003_backtest_jobs_v1.py
        - alembic/versions/20260326_0004_backtest_job_artifact_pin_v1.py
        - alembic/versions/20260329_0005_backtest_persisted_run_storage_v1.py
        - alembic/versions/20260411_0006_backtest_execution_profile_metadata_v1.py
        - alembic/versions/20260418_0009_backtest_execution_profile_metadata_parity_v1.py
    route_and_api_tests:
      read_when: "adding public endpoints or DTOs"
      paths:
        - tests/unit/apps/api/test_backtests_routes.py
        - tests/unit/contexts/backtest/domain/entities/test_backtest_job_entities.py
    benchmark_runner_patterns:
      read_when: "adding Iteration 7 end-to-end benchmark and service-only budget evidence"
      paths:
        - scripts/backtest/run_iteration_4_7_memory_cleanup_smoke.py
        - scripts/backtest/run_iteration_5_tp_sl_hit_times_benchmark.py
        - scripts/backtest/validate_benchmark_accounting.py
    delivery_skill_reference:
      read_when: "all local gates and Mac Studio acceptance pass and the implementation is ready to merge/deploy"
      paths:
        - /Users/daniildegtyarev/.codex/skills/publish-ci-deploy/SKILL.md
      instruction: "Use only for the final delivery path. Do not preload it during implementation."
  consult_if_needed:
    - path: docs/architecture/backtest/README.md
      read_when: "resolving public API vocabulary or superseded docs conflicts"
    - path: docs/architecture/backtest/benchmark_iterations/2026-05-01_iteration_5_tp_sl_hit_times_loading_validation/benchmark_summary.md
      read_when: "copying artifact compatibility and summary style"
    - path: src/trading/contexts/backtest/domain/value_objects/variant_identity.py
      read_when: "storage SHA identity helpers are needed for variant_hash/indicator_variant_hash"
    - path: src/trading/contexts/indicators/application/dto/variant_key.py
      read_when: "indicator-only canonical hash construction is needed"

style_references:
  - .codex/promt_template.md
  - .codex/agents/generated/backtest-service-iteration-6/01-implement-tp-sl-exact-scoring-full-metrics.md
  - docs/architecture/backtest/benchmark_iterations/README.md

hard_requirements:
  iteration_6_acceptance_required_before_implementation: true
  public_api_uses_jobs_vocabulary: true
  persist_summary_only_top_n: true
  readable_public_variant_key: true
  stable_storage_variant_hash: true
  map_public_identity_to_storage_identity_explicitly: true
  no_lazy_trades_implementation: true
  macstudio_acceptance_required: true
  merge_main_and_pull_after_success: true
  max_implementation_attempts: 2

task_toggles:
  implement_top_result_assembly: true
  implement_persist_top_n_io: true
  implement_job_create_status_top_list_cancel: true
  implement_idempotency: true
  implement_request_guardrails: true
  implement_progress_mapping: true
  implement_authz_ownership_checks: true
  implement_worker_cleanup_boundary: true
  implement_lazy_trades: false
  implement_sizing_iteration_8: false
  publish_merge_deploy_after_success: true

skill_routing:
  - skill: contract-impact-analysis
    use_when: "designing or changing public API DTOs, persistence rows, variant identity, idempotency, migrations, or cache/request keys"
    timing: before implementation and before final report
    reason: "Iteration 7 crosses API, storage, identity, and compatibility boundaries"
  - skill: backend-quality-gates
    use_when: "running targeted lint, type, unit, and API contract tests"
    timing: during verification
    reason: "Roehub backend gates are uv-based"
  - skill: backend-performance-evidence
    use_when: "measuring `top_result_assembly`, `persist_top_n_io`, end-to-end job runtime, cleanup, or benchmark evidence"
    timing: during benchmark verification
    reason: "Iteration 7 adds service-only performance budgets and end-to-end job benchmark evidence"
  - skill: publish-ci-deploy
    use_when: "all implementation gates and Mac Studio acceptance pass, and the branch is ready for merge/deploy"
    timing: after verification
    reason: "user requires merge to main, local pull, Mac Studio pull, deploy verification, and post-deploy evidence"

target_envs:
  - local-dev
  - github-actions
  - macstudio

required_literals:
  - "POST /backtests/jobs"
  - "GET /backtests/jobs"
  - "GET /backtests/jobs/{job_id}"
  - "GET /backtests/jobs/{job_id}/top"
  - "GET /backtests/jobs/{job_id}/variants/{variant_key}"
  - "POST /backtests/jobs/{job_id}/cancel"
  - "variant_key"
  - "variant_hash"
  - "indicator_variant_hash"
  - "top_result_assembly"
  - "persist_top_n_io"
  - "service_total_without_warmup"
  - "historical_prefix_compatible"

non_goals:
  - "Do not implement lazy trades detail or chart payloads; that belongs to Iteration 9."
  - "Do not implement Iteration 8 sizing/execution completion work."
  - "Do not change accepted exact-scoring kernels except for adapting their top rows into service DTOs."
  - "Do not expose raw storage SHA as the public route `variant_key`."
  - "Do not use old `POST /backtests` or runs vocabulary as v1 canonical API."
  - "Do not use legacy `hit_times/1m` paths or old execution-profile vocabulary in new public contracts."

final_report_format:
  language: ru
  sections:
    - "Что сделано"
    - "API / Persistence contract"
    - "Benchmark / Mac Studio"
    - "Проверки"
    - "Delivery / merge"
    - "Contract impact"
    - "Ограничения / следующий шаг"

quality_gates:
  - cmd: "uv run ruff check apps/api src/trading/contexts/backtest tests/unit/apps/api tests/unit/contexts/backtest scripts/backtest"
    expect: "passes, or a narrower justified target passes if unrelated existing files fail"
  - cmd: "uv run pyright"
    expect: "passes"
  - cmd: "uv run pytest -q tests/unit/apps/api/test_backtests_routes.py tests/unit/contexts/backtest/domain/entities/test_backtest_job_entities.py tests/unit/contexts/backtest/application"
    expect: "passes; include any new repository/adapter test file explicitly if persistence adapters change"
  - cmd: "uv run pytest -q -ra"
    expect: "passes before merge/deploy"
  - cmd: "uv run python scripts/backtest/validate_benchmark_accounting.py --out docs/architecture/backtest/benchmark_iterations/<iteration_7_dir>/local_accounting_validation.json"
    expect: "passes after the Iteration 7 runner writes local accounting evidence"
  - cmd: "uv run python -m tools.docs.generate_docs_index --check"
    expect: "passes if docs or benchmark summaries change"

expected_primary_touches:
  - "apps/api/routes/backtests.py"
  - "apps/api/dto/backtests.py"
  - "apps/api/wiring/modules/backtest.py"
  - "src/trading/contexts/backtest/application/dto/<new job orchestration dto>.py"
  - "src/trading/contexts/backtest/application/use_cases/<new backtest jobs use cases>.py"
  - "src/trading/contexts/backtest/application/services/v2/<new top_result_assembly service>.py"
  - "src/trading/contexts/backtest/application/services/v2/<new job orchestration service>.py"
  - "scripts/backtest/run_iteration_7_job_orchestration_persistence_benchmark.py"
  - "tests/unit/apps/api/<new or updated backtests route tests>.py"
  - "tests/unit/contexts/backtest/<new job orchestration/persistence tests>.py"
  - "docs/architecture/backtest/benchmark_iterations/YYYY-MM-DD_iteration_7_job_orchestration_persistence/"

possible_secondary_touches:
  - "src/trading/contexts/backtest/application/ports/backtest_job_repositories.py"
  - "src/trading/contexts/backtest/adapters/outbound/persistence/postgres/backtest_job_repository.py"
  - "src/trading/contexts/backtest/adapters/outbound/persistence/postgres/backtest_job_lease_repository.py"
  - "src/trading/contexts/backtest/domain/entities/backtest_job.py"
  - "src/trading/contexts/backtest/domain/entities/backtest_job_results.py"
  - "src/trading/contexts/backtest/domain/value_objects/variant_identity.py"
  - "alembic/versions/<new additive migration if required>.py"
  - "docs/architecture/backtest/backtest-service-artifact-runtime-v1.ru.md"
  - "docs/architecture/README.md"

safety_notes:
  - "Do not start implementation unless Iteration 6 accepted evidence exists. Iteration 7 depends on accepted no-risk and TP/SL top rows."
  - "Existing `BacktestJobTopVariant.variant_key` is SHA-only storage identity. Public readable `variant_key` must be mapped explicitly, preferably through payload/read-model fields unless an additive migration is justified."
  - "Top-N persistence is summary-only. Do not persist trades or chart payloads."
  - "After all checks and Mac Studio acceptance pass, use the repo delivery path to merge to `main`, then pull `main` locally and on Mac Studio."
  - "The executor has only 2 implementation attempts. After the second failed corrective cycle, stop and report the blocker with exact evidence."
---

# Task

Implement Iteration 7: job orchestration and persistence for the Backtest Service Artifact Runtime v1.

Done means:

- public job API endpoints exist for create/status/list/top/variant/cancel/defaults/preflight;
- no-risk and TP/SL top rows from accepted runtime services are assembled into persisted/API read-model rows;
- `top_result_assembly` creates public readable `variant_key`, stable `variant_hash`, optional `indicator_variant_hash`, canonical variant params, and API/read-model DTOs;
- persistence writes canonical request snapshot, artifact metadata, and summary-only top-N rows;
- public route identity maps safely to storage identity;
- idempotency, guardrails, ownership/authz, progress mapping, and cancellation semantics are tested;
- benchmark evidence records `top_result_assembly`, `persist_top_n_io`, and end-to-end job runtime on Mac Studio;
- if all checks and acceptance gates pass, the work is merged to `main`, `main` is pulled locally, and `main` is pulled on Mac Studio.

## Context / Current State

Precondition:

- Iteration 6 must already be accepted with Mac Studio benchmark evidence.
- If no accepted Iteration 6 evidence folder exists under `docs/architecture/backtest/benchmark_iterations/*_iteration_6_tp_sl_exact_scoring_full_metrics/`, stop before implementation and report that Iteration 7 is blocked.

Context ledger from previous accepted iterations:

- completed:
  - Iteration 1: request normalization, preflight, artifact context, runtime defaults.
  - Iteration 2: artifact arrays, slicing, `prepare_pools_core`.
  - Iteration 3: combo planning contexts.
  - Iteration 4: no-risk exact scoring/top-K path and cleanup evidence.
  - Iteration 5: `hit_times/15m` TP/SL grid validation and hit-times subset loading.
  - Iteration 6: TP/SL exact scoring and full metrics, required as a precondition.
- open_items:
  - public job lifecycle API is not complete;
  - top-result assembly is service-only and not part of notebook benchmark;
  - persisted top-N summary rows must be produced without trades/detail payloads;
  - public readable `variant_key` must be separated from SHA-only storage identity.
- contract_changes:
  - Iteration 7 adds public endpoints and persistence behavior;
  - Iteration 7 may add DTO/read-model fields and possibly an additive migration;
  - Iteration 7 must preserve old storage compatibility or explicitly migrate it.
- touched_paths:
  - current API shell has only `/backtests/runtime-defaults` and `/backtests/preflight`;
  - existing domain/repository code already has `BacktestJob`, `BacktestJobTopVariant`, and Postgres job repositories.
- risks:
  - raw storage SHA might accidentally leak as public `variant_key`;
  - persistence might store trades or heavy payloads, violating summary-only v1;
  - job lifecycle and cleanup might leave heavy compute objects retained after terminal persistence;
  - route vocabulary might drift back to old `runs` or `POST /backtests` names.
- next_focus:
  - build application use cases for job create/status/list/top/variant/cancel;
  - assemble top rows from no-risk and TP/SL results into read-model rows;
  - persist top-N summary atomically with terminal job state;
  - add API contract tests and Mac Studio end-to-end evidence;
  - merge/deploy through the repo delivery path only after acceptance.

Additional context:

- External v1 vocabulary is `jobs`, not `runs`.
- `POST /backtests/jobs` always creates a persisted job unless it is an idempotent replay.
- UI uses the same public API. Do not create an internal/private UI shortcut.
- `variant_key` is public and readable within one job. `variant_hash` is stable SHA-256 over canonical variant params without `job_id`.
- Existing storage can still require `BacktestJobTopVariant.variant_key` to be a 64-char SHA. Treat that as storage identity, not public API identity.

## Requirements (Must)

- Read context using the protocol below and stop early once sufficient.
- Verify Iteration 6 acceptance evidence before implementation.
- Implement only the scoped change described in this prompt.
- Preserve all accepted benchmark and runtime behavior from Iterations 1..6.
- Add or update targeted tests for API contracts, persistence, idempotency, ownership, progress, and cleanup.
- Keep public API, persistence, and identity behavior deterministic.

Public API:

- Add/complete these endpoints:
  - `POST /backtests/jobs`;
  - `GET /backtests/jobs`;
  - `GET /backtests/jobs/{job_id}`;
  - `GET /backtests/jobs/{job_id}/top`;
  - `GET /backtests/jobs/{job_id}/variants/{variant_key}`;
  - `POST /backtests/jobs/{job_id}/cancel`;
  - preserve existing `GET /backtests/runtime-defaults`;
  - preserve existing `POST /backtests/preflight`.
- Use authenticated current user for every job endpoint.
- Return deterministic error payloads:
  - `401 auth.required`;
  - `403 backtest.forbidden`;
  - `404 backtest.not_found`;
  - `409 backtest.idempotency_key_conflict`;
  - `409 backtest.job_not_cancellable`;
  - `422 backtest.invalid_request`;
  - `422 backtest.request_too_expensive`;
  - `429 backtest.rate_limited`;
  - `503 backtest.artifacts_unavailable`;
  - `503 backtest.queue_saturated`.
- Do not use old `POST /backtests` as the canonical create endpoint.
- Do not expose `runs` vocabulary in new public v1 contracts.

Job create/orchestration:

- `POST /backtests/jobs` must:
  - normalize and validate the request through the accepted preflight path;
  - enforce resource guardrails and per-user active job limits;
  - compute canonical `request_hash`;
  - pin selected artifact metadata;
  - create a persisted job snapshot;
  - run or enqueue the accepted runtime pipeline according to current repo architecture;
  - make terminal top-N summary available through `/top`.
- If implementation runs inline in v1, it must still persist lifecycle transitions as job state:
  - `queued`;
  - `running`;
  - `succeeded` / `failed` / `cancelled`.
- If implementation uses queue/lease worker paths, writes must be lease-owner guarded.
- Cancellation must be idempotent:
  - active jobs become or request `cancelled`;
  - terminal jobs return current terminal state;
  - cancellation must not delete already committed top-N summaries.

Idempotency:

- Without `Idempotency-Key`, every valid request creates a new job.
- With `Idempotency-Key`, replay of the same canonical request by the same user within TTL returns the original job.
- Reuse of the same key with a different canonical request returns deterministic `409 backtest.idempotency_key_conflict`.
- Idempotency state must be persistent or derived from durable storage. Do not implement production idempotency as process-local memory only.

Top-result assembly:

- Implement `top_result_assembly` for no-risk and TP/SL top rows.
- Assembly must produce:
  - public readable `variant_key`;
  - stable `variant_hash`;
  - optional `indicator_variant_hash`;
  - canonical variant params;
  - compact readable params;
  - summary metrics;
  - `best_tp_pct` / `best_sl_pct`;
  - links/actions for lazy detail, without implementing lazy detail itself.
- Public `variant_key` format should follow the runtime doc target:

```text
job_<job_short>__<readable_slug>__vh_<variant_hash_short>
```

- Same variant params in two jobs may share `variant_hash` but must have different public `variant_key`.
- Public route lookup must resolve `{job_id, variant_key}` to exactly one persisted top row owned by the current user.
- Direct lookup by raw storage SHA must not be a public v1 contract.

Persistence:

- Persist only top-N summary rows, not trades or chart payloads.
- Persist:
  - canonical request snapshot;
  - `request_hash`;
  - engine/config hashes;
  - artifact slot metadata;
  - requested `top_n`;
  - ranking metric;
  - progress state;
  - terminal result summary;
  - summary top rows.
- Existing `BacktestJobTopVariant.variant_key` may require SHA-only storage identity. Preferred compatibility path:
  - store `variant_hash` in the storage `variant_key` field;
  - store public readable key and readable params in `payload_json` or an additive read-model field;
  - API response maps storage identity back to public `variant_key` plus `variant_hash`.
- If an additive migration is needed, it must be backward-compatible and tested.
- `report_table_md` and `trades_json` must remain null in summary-only contract.

Progress:

- API consumers use `state` for lifecycle and `progress.pipeline_stage` for UI detail.
- Persisted coarse stages can remain:
  - `stage_a`;
  - `stage_b`;
  - `finalizing`.
- API progress must expose canonical pipeline stage names from the runtime doc.
- Benchmark records use canonical notebook/service timer names, not legacy persisted stage names.

Benchmark/evidence:

- Implement or update a runner:
  - `scripts/backtest/run_iteration_7_job_orchestration_persistence_benchmark.py`.
- Write evidence to:
  - `docs/architecture/backtest/benchmark_iterations/YYYY-MM-DD_iteration_7_job_orchestration_persistence/benchmark_results.json`;
  - `docs/architecture/backtest/benchmark_iterations/YYYY-MM-DD_iteration_7_job_orchestration_persistence/benchmark_summary.md`.
- Record service-only stages:
  - `top_result_assembly`;
  - `persist_top_n_io`;
  - `service_total_without_warmup`.
- Record end-to-end job benchmark for current canonical no-risk and TP/SL fixtures.
- Record persisted top-N summary hash/parity evidence.
- Record repeated end-to-end cleanup evidence:
  - after terminal persistence and result availability, the worker must not retain heavy compute objects;
  - if retained RSS exceeds threshold, worker recycle behavior must be recorded.

Delivery:

- If and only if local gates, API contract tests, Mac Studio benchmark evidence, and post-implementation docs checks all pass:
  - use `publish-ci-deploy` for the delivery path;
  - push the branch;
  - open/update PR if needed;
  - watch CI to completion;
  - merge into `main`;
  - pull `main` on the local machine in `/Users/daniildegtyarev/Projects/roehub.com`;
  - pull `main` on Mac Studio in `/Users/daniildegtyarev/Projects/roehub.com`;
  - verify deployed/runtime surface on Mac Studio.
- Do not merge or pull/deploy if any required gate is red, skipped, or ambiguous.

## Requirements (Should)

- Prefer a minimal compatibility implementation over a broad schema refactor.
- Keep API DTOs explicit and small; avoid leaking domain aggregates directly.
- Keep read models sorted deterministically:
  - jobs list: newest first via keyset pagination;
  - top rows: `rank ASC`.
- Prefer summary JSON/hash parity evidence over large payload comparisons.
- Keep route tests independent from live Postgres by using fakes where possible.
- Add repository/adapter tests for SQL serialization if persistence behavior changes.
- Keep lazy trades links present but non-operative until Iteration 9, or point to a not-implemented contract only if the existing API error style supports it.

## Requirements (Nice-to-have)

- Add an explicit public contract fixture for one no-risk job and one TP/SL job.
- Add a small benchmark summary table that separates notebook-compatible runtime, service-only overhead, and persistence IO.

# Context acquisition protocol

Read only in this order and stop once sufficient:

1. `.codex/AGENTS.md`
2. Iteration 7 section plus API, variant identity, progress mapping, and service-only stage sections of `docs/architecture/backtest/backtest-service-artifact-runtime-v1.ru.md`
3. benchmark manifest and canonical JSON target
4. accepted Iteration 6 evidence folder; stop if missing or failed
5. task entrypoints
6. only the conditional bundle(s) required by touched contracts, migrations, failing checks, or route ambiguity
7. consult-if-needed references only for blockers, ambiguity, or conflict resolution

Do not eagerly preload all listed sources.

Pre-implementation reading target:

- `<= 12 files`
- `<= ~60k tokens`

Stop reading once all of the following are true:

- public endpoint surface is bounded;
- persistence and identity compatibility strategy is selected;
- touched files are bounded;
- benchmark and delivery acceptance criteria are implementable;
- no unresolved migration/idempotency/authz ambiguity remains.

Expand context only for:

- migration or schema conflict;
- public/storage identity ambiguity;
- failing API/persistence tests;
- Mac Studio benchmark/deploy failure;
- route wiring ambiguity.

# Reading manifest

Use the front-matter `context_sources` as the canonical reading map.

Read with this intent:

- `always_read`:
  - repository rules;
  - runtime target;
  - benchmark evidence contract;
  - canonical fixture.
- `task_entrypoints`:
  - current API shell;
  - job persistence ports/adapters;
  - storage/domain identity constraints.
- `conditional_bundles`:
  - read only when the stated condition applies.
- `consult_if_needed`:
  - read only for blockers, ambiguity, or conflict resolution.

Do not convert this manifest into a broad mandatory reading list.

# Work plan (agent should follow)

Skill routing for this task:

- `contract-impact-analysis`: use before implementation and before final report; owns API, DTO, persistence, idempotency, variant identity, and migration compatibility.
- `backend-quality-gates`: use during verification; owns uv-based lint, type, unit, and API contract gates.
- `backend-performance-evidence`: use during benchmark verification; owns service-only budget, end-to-end job benchmark, and cleanup evidence.
- `publish-ci-deploy`: use only after all implementation and Mac Studio acceptance gates pass; owns push, PR/CI, merge to `main`, local pull, Mac Studio pull, deploy verification, and post-deploy evidence.

Implementation sequence:

1. Verify Iteration 6 accepted evidence exists. Stop if missing.
2. Read bounded context and classify contract impact before code changes.
3. Decide storage identity compatibility:
   - preferred: storage `variant_key = variant_hash`, public key in payload/read model;
   - additive migration only if necessary and justified.
4. Implement top-result assembly for accepted no-risk and TP/SL service result shapes.
5. Implement application use cases for create/status/list/top/variant/cancel.
6. Implement idempotency and guardrails using durable storage or existing durable request metadata.
7. Wire public API DTOs/routes with ownership/authz checks.
8. Wire persistence repositories/adapters and migrations only as required.
9. Add API, domain, use-case, repository, idempotency, authz, and cleanup tests.
10. Add Iteration 7 benchmark runner and evidence writer.
11. Run local gates and fix introduced failures.
12. Run Mac Studio benchmark and write evidence.
13. If accepted, update the main runtime doc status for Iteration 7 and docs index.
14. If all gates are green, use `publish-ci-deploy` to push, PR, watch CI, merge to `main`, pull locally, pull on Mac Studio, and verify production/runtime health.
15. If any gate fails after two implementation attempts, stop and report exact blockers.

# Benchmark and Mac Studio pipeline

Acceptance benchmark and post-merge verification must use Mac Studio evidence.

Implementation benchmark path:

```bash
export PATH="/opt/homebrew/bin:$PATH"
cd /Users/daniildegtyarev/Projects/roehub.com
git pull --ff-only
uv run python scripts/backtest/run_iteration_7_job_orchestration_persistence_benchmark.py \
  --out-dir docs/architecture/backtest/benchmark_iterations/$(date +%F)_iteration_7_job_orchestration_persistence
```

If `/opt/roehub/app` is the runtime surface being benchmarked, record that explicitly. Do not run `git pull` in `/opt/roehub/app`; it is a deployed runtime copy, not the repository checkout.

After all checks pass and the branch is ready:

```bash
# Use publish-ci-deploy to perform the actual push/PR/CI/merge/deploy flow.
# After merge:
cd /Users/daniildegtyarev/Projects/roehub.com
git checkout main
git pull --ff-only

ssh macstudio 'cd /Users/daniildegtyarev/Projects/roehub.com && git checkout main && git pull --ff-only'
ssh macstudio 'cd /Users/daniildegtyarev/Projects/roehub.com && bash scripts/macos/smoke_prod.sh'
```

If Mac Studio auth or remote state is broken, treat that as part of the delivery task and fix it through `publish-ci-deploy` rather than reporting only the first failure.

# Acceptance criteria (Definition of Done)

- Iteration 6 accepted evidence exists before implementation.
- Public endpoints exist and match the runtime document.
- `POST /backtests/jobs` creates persisted jobs and supports idempotent replay/conflict behavior.
- `GET /backtests/jobs` returns user-scoped keyset history.
- `GET /backtests/jobs/{job_id}` returns status/progress/artifact metadata.
- `GET /backtests/jobs/{job_id}/top` returns persisted summary-only top rows.
- `GET /backtests/jobs/{job_id}/variants/{variant_key}` resolves public variant key to exactly one owned persisted top row.
- `POST /backtests/jobs/{job_id}/cancel` is idempotent and preserves committed summaries.
- Public readable `variant_key` and stable `variant_hash` are both present in API read models.
- Existing SHA-only storage identity is either preserved through adapter mapping or migrated additively with tests.
- Summary top rows do not persist trades or chart payloads.
- API contract tests cover success and error paths.
- Idempotency replay/conflict evidence exists.
- Ownership/authz failure evidence exists.
- `top_result_assembly`, `persist_top_n_io`, `service_total_without_warmup`, and cleanup evidence are recorded.
- Mac Studio benchmark evidence exists and passes the Iteration 7 gate.
- Local full gates pass before merge.
- CI passes before merge.
- The branch is merged to `main`.
- Local checkout at `/Users/daniildegtyarev/Projects/roehub.com` is on updated `main`.
- Mac Studio checkout at `/Users/daniildegtyarev/Projects/roehub.com` is on updated `main`.
- Mac Studio production/runtime smoke passes after merge/deploy.

# Implementation constraints

- Keep diffs scoped to Iteration 7.
- Do not silently change public API contracts.
- Do not leak raw storage SHA as public route `variant_key`.
- Do not persist trades/detail/chart payloads.
- Do not implement lazy trades.
- Do not add broad migrations unless the adapter-mapping path is impossible or unsafe.
- Do not compare service-only stages against notebook timers.
- Do not mark Iteration 7 as accepted from local tests alone.
- Do not merge if any required gate is missing, red, or ambiguous.

# Files to indicate (expected touched areas)

Expected primary files:

- `apps/api/routes/backtests.py`
- `apps/api/dto/backtests.py`
- `apps/api/wiring/modules/backtest.py`
- `src/trading/contexts/backtest/application/dto/<new job orchestration dto>.py`
- `src/trading/contexts/backtest/application/use_cases/<new backtest jobs use cases>.py`
- `src/trading/contexts/backtest/application/services/v2/<new top_result_assembly service>.py`
- `src/trading/contexts/backtest/application/services/v2/<new job orchestration service>.py`
- `scripts/backtest/run_iteration_7_job_orchestration_persistence_benchmark.py`
- `tests/unit/apps/api/<new or updated backtests route tests>.py`
- `tests/unit/contexts/backtest/<new job orchestration/persistence tests>.py`
- `docs/architecture/backtest/benchmark_iterations/YYYY-MM-DD_iteration_7_job_orchestration_persistence/benchmark_results.json`
- `docs/architecture/backtest/benchmark_iterations/YYYY-MM-DD_iteration_7_job_orchestration_persistence/benchmark_summary.md`

Possible secondary files:

- `src/trading/contexts/backtest/application/ports/backtest_job_repositories.py`
- `src/trading/contexts/backtest/adapters/outbound/persistence/postgres/backtest_job_repository.py`
- `src/trading/contexts/backtest/adapters/outbound/persistence/postgres/backtest_job_lease_repository.py`
- `src/trading/contexts/backtest/domain/entities/backtest_job.py`
- `src/trading/contexts/backtest/domain/entities/backtest_job_results.py`
- `src/trading/contexts/backtest/domain/value_objects/variant_identity.py`
- `alembic/versions/<new additive migration if required>.py`
- `docs/architecture/backtest/backtest-service-artifact-runtime-v1.ru.md`
- `docs/architecture/README.md`

# Non-goals

- No lazy trades compute/cache endpoint implementation.
- No UI chart payload implementation.
- No Iteration 8 sizing/execution expansion.
- No broad roadmap cleanup.
- No legacy `runs` API.
- No old `POST /backtests` create endpoint.
- No legacy `hit_times/1m` runtime path.

# Quality gates (must run and pass)

Run local gates before claiming implementation completion:

```bash
uv run ruff check apps/api src/trading/contexts/backtest tests/unit/apps/api tests/unit/contexts/backtest scripts/backtest
uv run pyright
uv run pytest -q tests/unit/apps/api/test_backtests_routes.py tests/unit/contexts/backtest/domain/entities/test_backtest_job_entities.py tests/unit/contexts/backtest/application
uv run pytest -q -ra
```

After benchmark evidence is written:

```bash
uv run python scripts/backtest/validate_benchmark_accounting.py \
  --out docs/architecture/backtest/benchmark_iterations/<iteration_7_dir>/local_accounting_validation.json
uv run python -m tools.docs.generate_docs_index --check
```

Run the Mac Studio benchmark gate before claiming acceptance:

```bash
export PATH="/opt/homebrew/bin:$PATH"
uv run python scripts/backtest/run_iteration_7_job_orchestration_persistence_benchmark.py \
  --out-dir docs/architecture/backtest/benchmark_iterations/$(date +%F)_iteration_7_job_orchestration_persistence
```

Run delivery only after all gates pass:

```bash
# Use publish-ci-deploy skill for the actual flow.
# Required terminal state:
# - branch merged to main
# - local checkout pulled to main
# - Mac Studio checkout pulled to main
# - Mac Studio smoke passed
```

If a command cannot be run, state why, classify the risk, and do not claim that gate as passed.

# Contract impact report

Include this classification in the final report:

- Public API:
- Ports:
- DTO schema:
- Persisted schema:
- Config schema:
- Request/cache/persistence identity:
- Benchmark evidence schema:
- Runtime artifact contract:
- Delivery/deploy surface:

Use one of:

- `none`;
- `compatible-change`;
- `breaking-change`;
- `unknown`.

# Failure/blocker behavior

You have only 2 implementation attempts.

An attempt is a full cycle of implementation, local gates, and Mac Studio benchmark or equivalent blocker evidence. If the second attempt still fails acceptance:

- stop;
- do not broaden scope into Iteration 8 or Iteration 9;
- do not merge;
- do not hide failed benchmark rows or API failures;
- report:
  - implementation commit;
  - changed files;
  - exact failed endpoint/test/benchmark rows;
  - artifact hashes and compatibility policy;
  - whether the failure is API contract, persistence, identity mapping, idempotency, authz, cleanup, CI/deploy, or service-only performance;
  - the smallest next investigation step.

# Final output: report format (strict)

Use Russian.

## Что сделано

- Concise implementation summary.

## API / Persistence contract

- Endpoints implemented.
- Persistence strategy.
- Public `variant_key` to storage `variant_hash` mapping.
- Idempotency and guardrails.

## Benchmark / Mac Studio

- Evidence directory.
- Commit.
- Artifact hashes.
- Service-only stage table summary.
- End-to-end job result summary.
- Failed rows first if any.

## Проверки

- Commands run and results.
- Commands not run and why.

## Delivery / merge

- Branch / PR.
- CI status.
- Merge status.
- Local `main` pull status.
- Mac Studio `main` pull status.
- Mac Studio smoke/deploy verification.

## Contract impact

- Public API:
- Ports:
- DTO schema:
- Persisted schema:
- Config schema:
- Request/cache/persistence identity:
- Benchmark evidence schema:
- Runtime artifact contract:
- Delivery/deploy surface:

## Ограничения / следующий шаг

- Remaining risks.
- If accepted and merged, state that Iteration 8 is next.
- If not accepted, state the blocker and stop.
