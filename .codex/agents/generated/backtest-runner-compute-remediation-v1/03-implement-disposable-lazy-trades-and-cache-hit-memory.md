---
prompt_name: backtest_runner_compute_remediation_v1_03_disposable_lazy_trades_and_cache_hit_memory
repo: roehub.com
branch: current
scope: "P0: make lazy trades cache-miss compute disposable and make cache-hit/detail reads memory-bounded so API does not load full trades detail into memory."

language:
  implementation: python
  agent_report: ru

context_sources:
  always_read:
    - path: .codex/AGENTS.md
      why: "repo engineering contract"
    - path: docs/architecture/backtest/backtest-job-runner-production-plan-v1.md
      why: "runner/lazy materialization queue contract"
    - path: .codex/agents/.context/promt_manager_state.yaml
      why: "latest compact state if present"
  task_entrypoints:
    - path: src/trading/contexts/backtest/application/services/v2/lazy_trades_detail.py
      why: "current lazy trades cache miss recompute and cache hit read model assembly"
      inspect_symbols:
        - BacktestLazyTradesDetailService
        - read_cached
        - execute
        - _recompute_payload
    - path: src/trading/contexts/backtest/adapters/outbound/cache_fs/lazy_trades_cache.py
      why: "current JSON cache read/write loads full payload"
      inspect_symbols:
        - LocalFileBacktestLazyTradesCache
        - read
        - write
    - path: src/trading/contexts/backtest/application/use_cases/lazy_trades_materialization_worker.py
      why: "current lazy materialization claim/heartbeat/terminal task path"
      inspect_symbols:
        - BacktestLazyTradesMaterializationWorkerUseCase
        - run_next
        - _execute
    - path: src/trading/contexts/backtest/application/use_cases/backtest_jobs.py
      why: "public cache-hit endpoints, ownership checks, result series/stats/trades/CSV behavior"
      inspect_symbols:
        - BacktestJobsUseCase
        - trades
        - paginated_trades
        - trades_csv
  conditional_bundles:
    runner_parent_child:
      read_when: "when integrating lazy materialization with the process-isolated runner parent from prompt 01"
      paths:
        - apps/worker/backtest_job_runner/wiring/modules/backtest_job_runner.py
        - apps/worker/backtest_job_runner/main/main.py
        - src/trading/contexts/backtest/application/use_cases/backtest_job_worker.py
    lazy_materialization_persistence:
      read_when: "when changing task status, cache status, lease, or terminal behavior"
      paths:
        - src/trading/contexts/backtest/application/ports/lazy_trades_materialization_repositories.py
        - src/trading/contexts/backtest/adapters/outbound/persistence/postgres/backtest_lazy_trades_materialization_repository.py
        - tests/unit/contexts/backtest/application/use_cases/test_lazy_trades_materialization_worker_use_case.py
    api_dto_contract:
      read_when: "when changing public lazy detail, paginated trades, stats, series, or CSV behavior"
      paths:
        - apps/api/routes/backtests.py
        - apps/api/dto/backtests.py
        - apps/web/templates/pages/backtests.html
    benchmark_memory_policy:
      read_when: "when defining memory evidence fields or acceptance thresholds"
      paths:
        - docs/architecture/backtest/benchmark_iterations/README.md
        - docs/architecture/backtest/benchmark_iterations/2026-05-01_iteration_4_7_memory_cleanup/benchmark_summary.md
        - docs/architecture/backtest/benchmark_iterations/2026-05-02_iteration_9_lazy_trades_detail/benchmark_summary.md
  consult_if_needed:
    - path: docs/architecture/backtest/backtest-service-artifact-runtime-v1.ru.md
      read_when: "lazy detail semantic parity, cache identity, or artifact fields are unclear"
    - path: docs/architecture/apps/web/web-ui-backend-implementation-plan-v1.md
      read_when: "browser-visible result/detail behavior is affected"

style_references:
  - .codex/promt_template.md

hard_requirements:
  lazy_cache_miss_child_process_required: true
  child_process_per_lazy_materialization_required: true
  child_process_must_exit_after_one_lazy_task: true
  api_cache_hit_must_not_load_full_trades_detail: true
  full_detail_json_read_text_cache_hit_for_api_forbidden: true
  paginated_trades_must_be_memory_bounded: true
  trades_csv_must_be_streaming_or_chunked: true
  lazy_cache_identity_must_remain_stable: true
  lazy_cache_miss_and_hit_contract_must_remain_compatible: true
  owner_access_checks_must_remain_before_cache_read: true
  parent_metrics_must_not_depend_on_lazy_child_responsiveness: true
  lazy_materialization_terminal_semantics_must_be_preserved: true
  replace_lazy_worker_in_process_execute_required: true
  replace_backtest_jobs_full_detail_loader_required: true
  replace_result_series_full_detail_builders_on_api_path_required: true
  old_monolithic_cache_read_must_not_remain_primary_api_path: true
  active_docs_must_remove_in_process_lazy_cache_miss_and_full_detail_cache_hit: true
  docs_cleanup_required_for_lazy_cache_replacement: true
  memory_release_evidence_required_in_benchmark_prompt: true

task_toggles:
  implementation_changes_allowed: true
  add_lazy_child_compute_entrypoint: true
  refactor_lazy_cache_storage: true
  add_chunked_or_streaming_cache_reader: true
  update_api_read_paths_if_needed: true
  update_tests: true
  run_macstudio_memory_smoke: false
  publish_after_success: true
  publish_via_github_yeet: true
  direct_main_push_after_local_gates: true
  merge_to_main_in_this_prompt: true
  deploy_to_macstudio_in_this_prompt: true

skill_routing:
  - skill: architecture-design
    use_when: "choosing the lazy child-process boundary, cache file layout, and public read-path compatibility strategy"
    timing: before implementation
    reason: "this changes runtime and storage boundaries while preserving public API behavior"
  - skill: backend-performance-evidence
    use_when: "designing memory-bounded reads, cache-hit memory checks, or Mac Studio memory evidence fields"
    timing: before implementation and during verification
    reason: "the goal is provable memory release and bounded API memory use"
  - skill: root-cause-debugging
    use_when: "lazy materialization hangs, cache identity drifts, cache hit loads too much memory, or child exit handling fails"
    timing: if blocker
    reason: "memory regressions must be localized instead of hidden behind broader refactors"
  - skill: backend-quality-gates
    use_when: "running focused lazy detail, cache, API, worker, lint, and type gates"
    timing: during verification
    reason: "Roehub backend gates are uv-based"
  - skill: contract-impact-analysis
    use_when: "changing cache schema, lazy detail DTOs, endpoint response behavior, task statuses, or config keys"
    timing: before final report
    reason: "cache-hit behavior is externally visible through results/detail endpoints"
  - skill: github:yeet
    use_when: "local gates pass and scoped lazy-memory changes should be published for review"
    timing: after verification and before host delivery
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
  - "lazy_trades_compute"
  - "lazy_trades_cache_hit"
  - "child process"
  - "cache miss"
  - "cache hit"
  - "trades_cache"
  - "result_contains_heavy_references"
  - "worker_recycle_required"
  - "BacktestLazyTradesMaterializationWorkerUseCase.run_next"
  - "BacktestLazyTradesMaterializationWorkerUseCase._execute"
  - "BacktestLazyTradesDetailService.execute"
  - "BacktestLazyTradesDetailService.read_cached"
  - "LocalFileBacktestLazyTradesCache.read"
  - "BacktestJobsUseCase.trades"
  - "build_paginated_trades_read_model"
  - "build_trades_csv"
  - "docs cleanup"
  - "monolithic cache"
  - "RSS"
  - "retained_rss_delta"
  - "vmmap"
  - "physical footprint"
  - "queued -> running -> completed"
  - "127.0.0.1:9204/metrics"
  - "github:yeet"
  - "git push origin main"
  - "origin/main"
  - "publish-ci-deploy"
  - "git pull --ff-only"

non_goals:
  - "Do not change scoring semantics or lazy trades parity."
  - "Do not move lazy cache-miss recompute back into API."
  - "Do not keep using full-file JSON cache hits for API detail/trades/stat endpoints."
  - "Do not require browser/UI redesign for this runtime fix."
  - "Do not run final Mac Studio benchmark acceptance in this prompt; prompt 04 owns acceptance evidence."
  - "Do not use the single 140+ second heaviest benchmark job in any required benchmark or smoke check."
  - "Do not stop at local gates; direct `main`/`origin/main` delivery and Mac Studio smoke are required after gates pass."

final_report_format:
  language: ru
  sections:
    - "Intent"
    - "Lazy child boundary"
    - "Cache-hit memory surface"
    - "Removed/replaced paths"
    - "Docs cleanup"
    - "API compatibility"
    - "Tests"
    - "Contract impact"
    - "Risks"
    - "Publish"
    - "CI/deploy"
    - "Mac Studio delivery"
    - "Next prompt"

quality_gates:
  - cmd: "uv run pytest -q tests/unit/contexts/backtest/application/use_cases/test_lazy_trades_materialization_worker_use_case.py tests/unit/apps/api/test_backtests_routes.py tests/unit/apps/worker/backtest_job_runner"
    expect: "passes; include new focused tests explicitly"
  - cmd: "uv run ruff check apps/api apps/worker src/trading/contexts/backtest tests/unit/apps/api tests/unit/apps/worker tests/unit/contexts/backtest"
    expect: "passes"
  - cmd: "uv run pyright"
    expect: "passes or unrelated pre-existing failures classified"
  - cmd: "uv run python -m tools.docs.generate_docs_index --check"
    expect: "passes if docs changed"
  - cmd: "gh --version && gh auth status"
    expect: "passes before `github:yeet` publication, or publish blocker is reported"

expected_primary_touches:
  - "src/trading/contexts/backtest/application/services/v2/lazy_trades_detail.py"
  - "src/trading/contexts/backtest/adapters/outbound/cache_fs/lazy_trades_cache.py"
  - "src/trading/contexts/backtest/application/use_cases/lazy_trades_materialization_worker.py"
  - "src/trading/contexts/backtest/application/use_cases/backtest_jobs.py"
  - "tests/unit/contexts/backtest/application/use_cases/test_lazy_trades_materialization_worker_use_case.py"
  - "tests/unit/apps/api/test_backtests_routes.py"

possible_secondary_touches:
  - "apps/worker/backtest_job_runner/**"
  - "src/trading/contexts/backtest/application/ports/lazy_trades_cache.py"
  - "src/trading/contexts/backtest/application/ports/lazy_trades_materialization_repositories.py"
  - "apps/api/dto/backtests.py"
  - "docs/architecture/backtest/backtest-job-runner-production-plan-v1.md"
  - "docs/architecture/backtest/backtest-service-artifact-runtime-v1.ru.md"
  - "docs/architecture/backtest/backtest-service-artifact-runtime-v1.md"
  - "docs/architecture/apps/web/web-ui-backend-implementation-plan-v1.md"
  - "docs/runbooks/mac-studio-native-backend-operations.md"

safety_notes:
  - "Preserve unrelated dirty worktree changes; stage only scoped files if publishing later."
  - "Do not log secrets, full request payloads, full trades payloads, or cache files."
  - "If chunked cache migration needs backward compatibility, support old cache reads only in a memory-bounded way or treat old cache as miss."
---

# Task

Make lazy trades computation and cache-hit reads compatible with the new disposable-compute requirement.

Done means:

- lazy trades cache miss/materialization runs in a one-task child process and that child exits after writing the cache/result;
- the runner parent retains claim, heartbeat, metrics, child supervision, and terminal materialization status behavior;
- API cache-hit paths do not load the entire trades detail payload into memory;
- paginated trades, result series/statistics, and CSV export read cache data through bounded/chunked/streaming paths;
- public API ownership checks and response semantics remain compatible;
- prompt 04 can prove memory release on Mac Studio with explicit RSS/physical-footprint evidence.

## Context / Current State

The existing lazy trades design has the right request boundary for cache miss: API returns a materialization state and does not recompute inline. The remaining problem is the runtime and cache memory boundary.

Current risk points:

- `BacktestLazyTradesDetailService.execute` performs cache miss recompute in the same process that called it.
- lazy recompute opens artifact arrays, prepares pools, builds detail payload, builds a Python list of trades, writes JSON cache, and returns a read model.
- `LocalFileBacktestLazyTradesCache.read` uses full-file text read plus `json.loads`, so a cache hit can load the entire detail payload into API memory.
- endpoints for paginated trades, series/statistics, and CSV can indirectly force full detail materialization before slicing or formatting.
- `gc.collect()` and dropping Python references are not a reliable OS memory-release boundary for NumPy/Numba/macOS. A child process exit is the required hard boundary.

This prompt must not change lazy trades semantics. It changes where compute runs and how cache data is read.

## Method-level replacement / cleanup map

This prompt must replace old full-detail cache and in-process lazy compute paths, not leave them as production alternatives.

Required current-method decisions:

- `src/trading/contexts/backtest/application/use_cases/lazy_trades_materialization_worker.py::BacktestLazyTradesMaterializationWorkerUseCase.run_next`
  - Keep claim, heartbeat, lease, `finish_completed`, and `finish_failed` semantics.
  - Replace parent-side execution of lazy recompute with lazy child supervision.
  - Parent should mark completion from child result/cache metadata, not from a full `BacktestLazyTradesDetailReadModel` kept in parent memory.
- `src/trading/contexts/backtest/application/use_cases/lazy_trades_materialization_worker.py::BacktestLazyTradesMaterializationWorkerUseCase._execute`
  - Remove or narrow the direct call to `self.lazy_trades_service.execute(...)` from the production worker path.
  - If a helper remains, it must be child-entrypoint-only or test-only and named/documented that way.
- `src/trading/contexts/backtest/application/services/v2/lazy_trades_detail.py::BacktestLazyTradesDetailService.execute`
  - Split responsibilities if needed: cache miss materialization compute may remain here only as child-only code.
  - It must not be used by API cache-hit endpoints or by the runner parent for recompute.
  - It should write bounded cache artifacts and return bounded metadata required for terminal task update.
- `src/trading/contexts/backtest/application/services/v2/lazy_trades_detail.py::BacktestLazyTradesDetailService.read_cached`
  - Replace full-detail read-model construction on cache hit.
  - Cache probing/status can remain, but public API read paths must use bounded cache reader methods.
- `src/trading/contexts/backtest/adapters/outbound/cache_fs/lazy_trades_cache.py::LocalFileBacktestLazyTradesCache.read`
  - Remove as the primary API cache-hit path if it performs `path.read_text()` plus `json.loads()` for the full payload.
  - Replace with explicit bounded methods such as `read_metadata`, `read_summary`, `read_page`, `read_series`, `read_stats`, and `iter_csv_rows`.
  - A legacy monolithic reader may exist only as a migration/miss helper and must not be called by public API endpoints.
- `src/trading/contexts/backtest/adapters/outbound/cache_fs/lazy_trades_cache.py::LocalFileBacktestLazyTradesCache.write`
  - Replace single monolithic JSON payload writes with atomic bundle/chunk writes.
  - Suggested shape: metadata/summary JSON plus trades as chunk-readable data and precomputed bounded series/stats artifacts where useful.
- `src/trading/contexts/backtest/application/ports/lazy_trades_cache.py::BacktestLazyTradesCache`
  - Replace or extend the port so API code can ask for bounded views without receiving full `payload`.
  - The old `read() -> payload` contract must not remain the only cache interface.
- `src/trading/contexts/backtest/application/use_cases/backtest_jobs.py::BacktestJobsUseCase.trades`
  - Stop using this as the shared full-detail loader for page/stats/series/CSV endpoints.
  - It may remain only as a detail-status/summary endpoint if the public contract needs it, and it must not force full trades materialization in API memory.
- `BacktestJobsUseCase.variant_series`, `monthly_stats`, `symbol_stats`, `paginated_trades`, `trades_csv`
  - Replace calls that first invoke `self.trades(...)` and then build views from `detail.trades`.
  - Each endpoint must use the bounded cache-reader method matching its response.
- `src/trading/contexts/backtest/application/services/v2/result_series.py::build_result_series_read_model`, `build_monthly_stats_read_model`, `build_symbol_stats_read_model`, `build_paginated_trades_read_model`, `build_trades_csv`
  - Do not leave these full-detail builders on public API cache-hit paths if they require `BacktestLazyTradesDetailReadModel.trades`.
  - Replace with bounded/chunked builder variants, or mark the old builders child-only/test-only/direct-object helpers and verify they are not used by API read paths.

Definition of “removed” for this prompt:

- The old monolithic cache file format can be treated as expired/miss or migrated, but public API endpoints must not keep a permanent full-load fallback.
- The old in-process lazy recompute can remain only in child code or tests, not in the runner parent or API.
- Static tests or grep assertions must fail if API page/stat/CSV paths call the old full-detail loader.

## Documentation cleanup / drift map

Docs must be updated in the same change when lazy cache miss execution and cache-hit storage/read paths are replaced.

Required doc decisions:

- `docs/architecture/backtest/backtest-job-runner-production-plan-v1.md`
  - Replace text that says one runner parent process performs lazy materialization compute in-process.
  - Describe lazy cache miss as disposable child process execution with parent-owned lease/heartbeat/terminal status.
  - Replace “cache hit in API only” with “cache hit in API through bounded/chunked readers only”.
- `docs/architecture/backtest/backtest-service-artifact-runtime-v1.ru.md` and `.md`
  - Update lazy trades sections to describe bounded cache layout/readers, chunked/page/stat/CSV access, and no public full-detail cache-hit load.
  - Mark old monolithic JSON cache payload behavior as historical/migration-only if mentioned.
- `docs/architecture/apps/web/web-ui-backend-implementation-plan-v1.md`
  - Remove or rewrite active status text that says lazy trades cache miss can compute in API use case or that result/stat endpoints may rely on full trades detail loading.
- `docs/runbooks/mac-studio-native-backend-operations.md`
  - Add/update operator notes for lazy trades cache layout, safe cache cleanup, and child process observation if operationally relevant.

Required docs verification:

- Run `rg -n "lazy trades cache miss|lazy_trades_compute|lazy_trades_cache_hit|full trades|monolithic|json.loads|read_text|compute in API|cache hit in API" docs/architecture docs/runbooks` and classify every remaining hit as historical, bounded-reader current behavior, migration-only, or removed.
- Run `uv run python -m tools.docs.generate_docs_index --check` if Markdown docs change.

## Requirements (Must)

- Read context using the protocol below and stop early once sufficient.
- Preserve public ownership checks before any cache read or detail access.
- Preserve lazy cache key identity: job id, public variant key, storage variant hash, request hash, engine params hash, and artifact manifest hash must remain part of the cache identity.
- Add a lazy materialization child compute entrypoint that executes one materialization task or one `(job_id, variant_key)` under production config.
- Ensure lazy cache miss compute runs in a disposable child process and the child exits after one task.
- Keep runner parent metrics responsive while lazy child compute is active.
- Preserve lazy materialization terminal semantics: claimed/running/completed/failed/cancelled, heartbeat, lease, retry, and bounded error payloads.
- Do not write full trades payloads into logs, metrics labels, or task errors.
- Replace full-file JSON cache hit for API reads with a memory-bounded cache layout or reader.
- Remove or narrow old full-detail loaders from public API read paths instead of adding bounded readers beside them.
- Ensure `paginated_trades` reads only the requested page plus bounded metadata.
- Ensure `trades_csv` streams or chunk-reads data; it must not build the full CSV source in API memory.
- Ensure result series/statistics either read precomputed bounded artifacts or compute from chunked trades without materializing the whole detail.
- Treat old monolithic cache entries as either backward-compatible misses or bounded migration inputs. Do not keep a permanent full-load API path for old cache files.
- Add focused tests proving cache hit does not call the full-detail loader for paginated/stat/CSV paths.
- Add focused tests proving lazy child success/failure maps to the same materialization terminal behavior.
- Add focused tests proving ownership denial happens before cache file read.
- Update active architecture/UI/runbook docs so they no longer describe in-process lazy cache miss or full-detail cache-hit reads as current behavior.

## Requirements (Should)

- Prefer a simple file layout: metadata/summary JSON plus trades as JSONL, NDJSON, Parquet, Arrow, or another chunk-readable format already reasonable for the repo.
- Use atomic temp-file writes and final rename for every cache artifact.
- Keep cache schema version explicit.
- Add low-cardinality telemetry for cache hit/miss/read mode without file path, user id, request hash, or variant hash labels.
- Keep old cache cleanup/migration simple; expired old cache can be deleted or ignored as miss.

## Requirements (Nice-to-have)

- Add cache reader helpers that can return only summary, only chart series, only one page, or CSV chunks.
- Add a local fake lazy child mode for unit tests.
- Add operator docs for clearing only lazy trades cache safely.

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
- no unresolved public API, cache identity, or persistence-contract ambiguity remains.

Expand context only for blockers, failing quality gates, unclear contracts, benchmark threshold conflicts, or architecture conflicts that affect correctness.

# Reading manifest

Use the front-matter `context_sources` as the canonical reading map.

Read with this intent:

- `always_read`: repository rules, runner/lazy queue plan, compact prior state;
- `task_entrypoints`: lazy compute, cache storage, materialization worker, API read paths;
- `conditional_bundles`: read only when the stated condition applies;
- `consult_if_needed`: read only for blockers, ambiguity, or conflict resolution.

Do not convert this manifest into a broad mandatory reading list.

# Work plan (agent should follow)

Skill routing for this task:

- `architecture-design`: use before implementation for child-process/cache-layout/API compatibility decisions.
- `backend-performance-evidence`: use before implementation for memory boundedness and benchmark evidence design.
- `root-cause-debugging`: use if child execution, cache identity, or endpoint memory behavior fails.
- `backend-quality-gates`: use during verification for uv-based tests/lint/type checks.
- `contract-impact-analysis`: use before final report if cache schema, DTO, status, or config behavior changes.
- `github:yeet`: use after local gates for scope inspection, selective staging, commit, and direct push to `main`/`origin/main`.
- `publish-ci-deploy`: use after direct main push to watch CI/deploy, sync Mac Studio with `git pull --ff-only`, reload services, and smoke production.

1. Confirm the current lazy cache miss and cache hit paths.
2. Design the minimal lazy child-process contract and parent supervision behavior.
3. Design a chunked or bounded cache layout that preserves cache key identity and public response compatibility.
4. Implement lazy cache miss materialization through disposable child execution.
5. Refactor cache hit read paths so API endpoints never load full trades detail into memory.
6. Add tests for child terminal behavior, bounded read behavior, owner checks before cache read, and backward compatibility/miss handling.
7. Run focused gates and classify any unrelated failures.
8. Leave Mac Studio memory-release benchmark acceptance to prompt 04, but make the implementation expose enough evidence hooks for it.
9. Inspect `git status -sb` and the scoped diff before publication.
10. If gates pass and the worktree scope is clean enough, use `github:yeet` discipline to stage only intended files, commit, and push directly to `main`/`origin/main`.
11. Use `publish-ci-deploy` to watch CI/deploy, sync Mac Studio with `git pull --ff-only`, reload the affected services, and run the required smoke checks.
12. If the worktree is mixed, `gh` is missing, or `gh auth status` fails, stop publication and report the exact blocker without publishing.

# Acceptance criteria (Definition of Done)

- Lazy cache miss/materialization runs through a one-task child process.
- Child process exits after one lazy materialization task.
- Runner parent remains the metrics/heartbeat/control-plane process.
- Materialization task reaches `queued -> running -> completed` on success.
- Child failure reaches `failed` with bounded error data.
- Cache key identity is unchanged or explicitly compatibility-classified.
- API cache hit for paginated trades does not load full detail/trades payload.
- API cache hit for CSV does not load full detail/trades payload.
- API cache hit for series/statistics does not load full detail/trades payload.
- Ownership denial is checked before cache file read.
- Tests prove bounded read APIs are used by public read paths.
- Tests or static assertions prove `BacktestJobsUseCase.paginated_trades`, `variant_series`, `monthly_stats`, `symbol_stats`, and `trades_csv` do not call the old full-detail `trades()` loader for cache-hit reads.
- Tests or static assertions prove `LocalFileBacktestLazyTradesCache.read` full-payload loading is not used by public API cache-hit paths.
- Prompt 04 can measure cache miss child RSS and API cache-hit retained RSS separately.
- Docs cleanup evidence proves active docs no longer describe in-process lazy cache miss or full-detail cache-hit loading as current behavior.
- Scoped changes are committed and pushed directly to `main`/`origin/main` after local gates, or a precise publication blocker is reported.
- Mac Studio is synchronized from `main` with `git pull --ff-only`, affected services are reloaded, and smoke evidence is reported.

# Implementation constraints

## API / contracts

- Public API compatibility target: `compatible-change`.
- Do not remove existing endpoint routes.
- Do not rename lazy materialization states unless a blocker proves it necessary.
- If public lazy detail response shape must change, stop and classify it as potential `breaking-change` before implementation.

## Runtime / operations

- Parent process must remain the launchd service process.
- Lazy child process must be disposable after one task.
- Do not make a persistent worker pool for lazy detail in this prompt.
- Do not let cache hit invoke recompute.
- Do not let cache hit read unbounded full cache files in API.

## Performance / memory

- The hard memory-release guarantee comes from child exit, not from `gc.collect()`.
- API cache-hit memory must be bounded by page/chunk size, not by total trades count.
- Avoid high-cardinality metrics labels.

## Delivery safety

- Push directly to `main`/`origin/main` only after local gates pass and scoped staging is confirmed.
- Do not use `git add -A` in a mixed worktree; stage only intended scoped files.
- CI watch, Mac Studio sync, service reload, and smoke evidence are part of this prompt's delivery step.

# Files to indicate (expected touched areas)

Expected primary touches:

- `src/trading/contexts/backtest/application/services/v2/lazy_trades_detail.py`
- `src/trading/contexts/backtest/adapters/outbound/cache_fs/lazy_trades_cache.py`
- `src/trading/contexts/backtest/application/use_cases/lazy_trades_materialization_worker.py`
- `src/trading/contexts/backtest/application/use_cases/backtest_jobs.py`
- `tests/unit/contexts/backtest/application/use_cases/test_lazy_trades_materialization_worker_use_case.py`
- `tests/unit/apps/api/test_backtests_routes.py`

Possible secondary touches:

- `apps/worker/backtest_job_runner/**`
- `src/trading/contexts/backtest/application/ports/lazy_trades_cache.py`
- `src/trading/contexts/backtest/application/ports/lazy_trades_materialization_repositories.py`
- `apps/api/dto/backtests.py`
- `docs/architecture/backtest/backtest-job-runner-production-plan-v1.md`

# Non-goals

- Do not change numerical compute semantics.
- Do not optimize ordinal/Numba full-job scoring here.
- Do not make cache miss synchronous in API.
- Do not run final Mac Studio acceptance here.
- Do not use the single 140+ second heaviest benchmark job in any required check.
- Do not stop at local gates without direct `main`/`origin/main` delivery and Mac Studio smoke.

# Quality gates (must run and pass)

- `uv run pytest -q tests/unit/contexts/backtest/application/use_cases/test_lazy_trades_materialization_worker_use_case.py tests/unit/apps/api/test_backtests_routes.py tests/unit/apps/worker/backtest_job_runner`
- `uv run ruff check apps/api apps/worker src/trading/contexts/backtest tests/unit/apps/api tests/unit/apps/worker tests/unit/contexts/backtest`
- `uv run pyright`
- `uv run python -m tools.docs.generate_docs_index --check` if docs changed
- `gh --version && gh auth status` before `github:yeet` publication

# Final output: report format (strict)

Write the final report in Russian with these sections:

- `Intent`
- `Lazy child boundary`
- `Cache-hit memory surface`
- `API compatibility`
- `Tests`
- `Contract impact`
- `Risks`
- `Publish`
- `CI/deploy`
- `Mac Studio delivery`
- `Next prompt`

Include exact commands run, pass/fail status, any pre-existing unrelated failures, main push commit SHA, and Mac Studio delivery evidence.
