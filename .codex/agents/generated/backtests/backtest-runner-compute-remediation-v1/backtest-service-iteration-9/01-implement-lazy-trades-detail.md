---
prompt_name: backtest_service_iteration_9_lazy_trades_detail
repo: roehub.com
branch: current
scope: "Iteration 9: implement lazy trades detail lookup, deterministic one-variant recompute, 48h cache, and chart-ready API payload."

language:
  implementation: python
  agent_report: ru

context_sources:
  always_read:
    - path: .codex/AGENTS.md
      why: "repo engineering contract, prompt precedence, delivery rules, and merge/deploy expectations"
    - path: docs/architecture/backtest/backtest-service-artifact-runtime-v1.ru.md
      why: "canonical Iteration 9 contract, lazy trades cache topology, variant identity, and benchmark gate"
    - path: docs/architecture/backtest/benchmark_iterations/README.md
      why: "benchmark evidence manifest, service-only stages, Mac Studio acceptance rules, and cache identity requirements"
    - path: docs/architecture/backtest/benchmark_iterations/2026-05-02_iteration_8_execution_sizing_completion/benchmark_summary.md
      why: "latest accepted execution/sizing evidence and Iteration 9 precondition"
    - path: docs/architecture/backtest/benchmark_iterations/2026-05-02_iteration_8_execution_sizing_completion/benchmark_results.json
      why: "latest accepted request/artifact/hash context and regression envelope before lazy trades"
  task_entrypoints:
    - path: apps/api/routes/backtests.py
      why: "public backtest routes; add POST /backtests/jobs/{job_id}/variants/{variant_key}/trades here"
      inspect_symbols:
        - build_backtests_router
    - path: apps/api/dto/backtests.py
      why: "API response DTOs for job/top/variant payloads; add lazy trades detail response shape here if API DTOs own it"
    - path: src/trading/contexts/backtest/application/use_cases/backtest_jobs.py
      why: "user-scoped job/variant lookup, ownership checks, and public variant key resolution"
      inspect_symbols:
        - BacktestJobsUseCase
        - variant
    - path: src/trading/contexts/backtest/application/ports/backtest_job_repositories.py
      why: "repository contract for public variant lookup and any cache metadata additions"
      inspect_symbols:
        - BacktestJobRepository
        - get_top_variant_by_public_key
    - path: src/trading/contexts/backtest/application/dto/backtest_jobs.py
      why: "top variant read-model, public variant_key vs storage variant_hash mapping, links/actions"
      inspect_symbols:
        - BacktestJobTopVariantReadModel
        - build_top_variant_read_model
    - path: src/trading/contexts/backtest/application/services/v2/top_result_assembly.py
      why: "accepted source for public variant_key, variant_hash, canonical_variant_params, and lazy_trades link metadata"
  conditional_bundles:
    iteration_8_acceptance:
      read_when: "before implementation; if accepted Iteration 8 evidence is missing, failed, or contradictory, stop and report the precondition blocker"
      paths:
        - docs/architecture/backtest/benchmark_iterations/2026-05-02_iteration_8_execution_sizing_completion/benchmark_summary.md
        - docs/architecture/backtest/benchmark_iterations/2026-05-02_iteration_8_execution_sizing_completion/benchmark_results.json
    recompute_runtime_services:
      read_when: "implementing deterministic one-variant recompute from persisted request + selected variant params"
      paths:
        - src/trading/contexts/backtest/application/services/v2/preflight.py
        - src/trading/contexts/backtest/application/services/v2/prepare_pools.py
        - src/trading/contexts/backtest/application/services/v2/combo_planning.py
        - src/trading/contexts/backtest/application/services/v2/no_risk_exact.py
        - src/trading/contexts/backtest/application/services/v2/tp_sl_exact.py
        - src/trading/contexts/backtest/application/services/v2/benchmark_accounting.py
    artifact_context_and_loader:
      read_when: "recompute needs pinned artifact metadata, historical-prefix-compatible artifact loading, or hit_times/15m subset loading"
      paths:
        - src/trading/contexts/backtest/application/ports/artifact_context.py
        - src/trading/contexts/backtest/application/ports/artifact_arrays.py
        - src/trading/contexts/backtest/adapters/outbound/artifacts_fs/artifact_context_resolver.py
        - src/trading/contexts/backtest/adapters/outbound/artifacts_fs/artifact_array_loader.py
        - src/trading/contexts/backtest_artifacts/application/services/v2/artifact_manifest_loader.py
        - tests/unit/contexts/backtest/application/services/v2/artifact_testkit_v2.py
    cache_storage:
      read_when: "adding cache port/adapter and 48h local object/file cache implementation"
      paths:
        - apps/api/wiring/modules/backtest.py
        - src/trading/contexts/backtest/application/dto/runtime_preflight.py
        - src/trading/contexts/backtest/application/services/v2/preflight.py
    api_and_route_tests:
      read_when: "adding route, API DTO, authz/ownership behavior, or error mapping"
      paths:
        - tests/unit/apps/api/test_backtests_routes.py
        - tests/unit/contexts/backtest/application/use_cases/test_backtest_jobs_use_case.py
    persistence_tests:
      read_when: "changing repository ports, Postgres adapter, or cache metadata persistence"
      paths:
        - tests/unit/contexts/backtest/domain/entities/test_backtest_job_entities.py
        - tests/unit/contexts/backtest/application/services/v2/test_top_result_assembly.py
        - src/trading/contexts/backtest/adapters/outbound/persistence/postgres/backtest_job_repository.py
    benchmark_runner_patterns:
      read_when: "adding Iteration 9 benchmark runner and evidence writer"
      paths:
        - scripts/backtest/run_iteration_7_job_orchestration_persistence_benchmark.py
        - scripts/backtest/run_iteration_8_execution_sizing_benchmark.py
        - scripts/backtest/validate_benchmark_accounting.py
    canonical_notebook_algorithm:
      read_when: "trade reconstruction semantics are ambiguous or service/reference parity fails"
      paths:
        - tests/notebook_tests/engine_test/btcusdt_15m_research_engine.ipynb
        - docs/architecture/backtest/benchmark_iterations/2026-04-26_engine_test_btcusdt_15m/benchmark_summary.md
        - docs/architecture/backtest/benchmark_iterations/2026-04-26_engine_test_btcusdt_15m/benchmark_results.json
    delivery_skill_reference:
      read_when: "all local gates and Mac Studio acceptance pass and the implementation is ready to merge/deploy"
      paths:
        - /Users/daniildegtyarev/.codex/skills/publish-ci-deploy/SKILL.md
      instruction: "Use only for the final delivery path. Do not preload it during implementation."
  consult_if_needed:
    - path: src/trading/contexts/backtest/application/services/v2/execution_sizing.py
      read_when: "existing shared sizing helper exists after Iteration 8 and trades recompute must reuse it"
    - path: tests/unit/contexts/backtest/application/services/v2/test_no_risk_exact_scoring_service.py
      read_when: "lazy trade rows need exact no-risk entry/exit parity fixtures"
    - path: tests/unit/contexts/backtest/application/services/v2/test_tp_sl_exact_scoring_service.py
      read_when: "lazy trade rows need TP/SL exit reason, best-cell, or close-on-end parity fixtures"

style_references:
  - .codex/promt_template.md
  - .codex/agents/generated/backtest-service-iteration-8/01-implement-execution-sizing-completion.md
  - docs/architecture/backtest/benchmark_iterations/README.md

hard_requirements:
  iteration_8_acceptance_required_before_implementation: true
  public_endpoint_required: "POST /backtests/jobs/{job_id}/variants/{variant_key}/trades"
  resolve_variant_by_public_key_only: true
  deterministic_one_variant_recompute: true
  cache_ttl_hours: 48
  cache_key_must_include:
    - job_id
    - variant_key
    - variant_hash
    - request_hash
    - engine_params_hash
    - artifact_manifest_hash
  cache_failure_must_not_break_successful_recompute: true
  summary_top_rows_must_remain_summary_only: true
  macstudio_acceptance_required: true
  merge_main_and_pull_after_success: true
  max_implementation_attempts: 2

task_toggles:
  implement_variant_lookup: true
  implement_lazy_trades_recompute: true
  implement_lazy_trades_cache_port: true
  implement_local_file_cache_adapter: true
  implement_chart_ready_payload: true
  implement_ownership_authz: true
  implement_cache_failure_fallback: true
  implement_benchmark_runner: true
  implement_ui: false
  publish_merge_deploy_after_success: true

skill_routing:
  - skill: contract-impact-analysis
    use_when: "adding the public trades endpoint, API DTOs, cache key, cache metadata, request/hash identity, or repository port methods"
    timing: before implementation and before final report
    reason: "Iteration 9 crosses public API, cache identity, persistence/read-model, and variant identity boundaries"
  - skill: backend-performance-evidence
    use_when: "building or reporting lazy_trades_compute/cache_hit benchmarks, CPU/RSS evidence, and Mac Studio acceptance"
    timing: during benchmark verification
    reason: "Iteration 9 acceptance is service-only performance evidence, not canonical notebook total comparison"
  - skill: backend-quality-gates
    use_when: "running targeted lint, type, unit, API, persistence, and regression tests"
    timing: during verification
    reason: "Roehub backend gates are uv-based"
  - skill: root-cause-debugging
    use_when: "lazy recompute, cache identity, ownership, or benchmark acceptance fails after the first implementation attempt"
    timing: only for a concrete failure/blocker
    reason: "failure diagnosis must isolate root cause before the second and final attempt"
  - skill: publish-ci-deploy
    use_when: "all implementation gates and Mac Studio acceptance pass, and the branch is ready for merge/deploy"
    timing: after verification
    reason: "user requires merge to main, local pull, Mac Studio pull, deploy verification, and post-deploy evidence"

target_envs:
  - local-dev
  - github-actions
  - macstudio

required_literals:
  - "POST /backtests/jobs/{job_id}/variants/{variant_key}/trades"
  - "lazy_trades_compute"
  - "lazy_trades_cache_hit"
  - "/opt/roehub/state/backtest/trades_cache"
  - "variant_key"
  - "variant_hash"
  - "request_hash"
  - "engine_params_hash"
  - "artifact_manifest_hash"
  - "historical_prefix_compatible"
  - "summary-only"

non_goals:
  - "Do not implement UI integration or candle chart rendering; that belongs to Iteration 10."
  - "Do not change the completed top-N summary persistence contract."
  - "Do not store full trades in `backtest_job_top_variants.trades_json`."
  - "Do not expose raw storage SHA lookup as a public endpoint."
  - "Do not broaden into new risk modes, indicator catalog expansion, or new sizing modes."
  - "Do not replace local cache with shared object storage unless the port abstraction makes the replacement possible later."
  - "Do not use legacy `hit_times/1m` paths."

final_report_format:
  language: ru
  sections:
    - "Что сделано"
    - "Lazy trades contract"
    - "Cache / identity"
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
  - cmd: "uv run pytest -q tests/unit/apps/api/test_backtests_routes.py tests/unit/contexts/backtest/application/use_cases/test_backtest_jobs_use_case.py tests/unit/contexts/backtest/application/services/v2 tests/unit/contexts/backtest/domain/entities/test_backtest_job_entities.py"
    expect: "passes; include any new Iteration 9 test file explicitly"
  - cmd: "uv run pytest -q -ra"
    expect: "passes before merge/deploy"
  - cmd: "uv run python scripts/backtest/run_iteration_9_lazy_trades_benchmark.py --out-dir docs/architecture/backtest/benchmark_iterations/<iteration_9_dir>"
    expect: "developer smoke can run locally when artifacts are available; Mac Studio run is acceptance"
  - cmd: "uv run python -m tools.docs.generate_docs_index --check"
    expect: "passes if docs or benchmark summaries change"

expected_primary_touches:
  - "apps/api/routes/backtests.py"
  - "apps/api/dto/backtests.py"
  - "apps/api/wiring/modules/backtest.py"
  - "src/trading/contexts/backtest/application/use_cases/backtest_jobs.py"
  - "src/trading/contexts/backtest/application/dto/backtest_jobs.py"
  - "src/trading/contexts/backtest/application/ports/backtest_job_repositories.py"
  - "src/trading/contexts/backtest/application/ports/<lazy trades cache port>.py"
  - "src/trading/contexts/backtest/application/services/v2/<lazy trades detail service>.py"
  - "src/trading/contexts/backtest/adapters/outbound/<local trades cache adapter>.py"
  - "tests/unit/apps/api/test_backtests_routes.py"
  - "tests/unit/contexts/backtest/application/<new lazy trades tests>.py"
  - "scripts/backtest/run_iteration_9_lazy_trades_benchmark.py"
  - "docs/architecture/backtest/benchmark_iterations/YYYY-MM-DD_iteration_9_lazy_trades_detail/"

possible_secondary_touches:
  - "src/trading/contexts/backtest/application/services/v2/no_risk_exact.py"
  - "src/trading/contexts/backtest/application/services/v2/tp_sl_exact.py"
  - "src/trading/contexts/backtest/application/services/v2/execution_sizing.py"
  - "src/trading/contexts/backtest/application/services/v2/benchmark_accounting.py"
  - "src/trading/contexts/backtest/adapters/outbound/persistence/postgres/backtest_job_repository.py"
  - "src/trading/contexts/backtest/application/dto/runtime_preflight.py"
  - "docs/architecture/backtest/backtest-service-artifact-runtime-v1.ru.md"
  - "docs/architecture/README.md"

safety_notes:
  - "Do not start implementation unless Iteration 8 accepted evidence exists. Iteration 9 depends on completed execution/sizing semantics."
  - "Inbound API route accepts public readable `variant_key`; storage row identity remains SHA-only `variant_hash` unless an explicit migration is implemented."
  - "Lazy recompute must operate on exactly one selected variant, not re-run the full job/grid."
  - "Cache miss is normal and must be recompute-safe. Corrupt cache, read failure, or write failure must not break a successful recompute response."
  - "Keep full trades out of top-N summary rows. Use a distinct cache port/adapter and, if needed, small bounded metadata."
  - "After all checks and Mac Studio acceptance pass, use the repo delivery path to merge to `main`, then pull `main` locally and on Mac Studio."
  - "The executor has only 2 implementation attempts. After the second failed corrective cycle, stop and report the blocker with exact evidence."
---

# Task

Implement Iteration 9: lazy trades detail for the backtest service.

Done means:

- public API exposes `POST /backtests/jobs/{job_id}/variants/{variant_key}/trades`;
- route uses only public readable `variant_key` and resolves it to exactly one persisted top-N row inside `job_id`;
- ownership is enforced before variant detail access;
- service recomputes exact trades for one selected variant using persisted job request snapshot, selected variant params, stored artifact metadata, and historical-prefix-compatible artifacts;
- service returns summary + trades + chart-ready overlay payload;
- result is cached for 48h using a deterministic cache key;
- cache hit returns quickly and records `lazy_trades_cache_hit`;
- cache miss recomputes and records `lazy_trades_compute`;
- local cache failure does not break a successful recompute response;
- Mac Studio benchmark evidence is written under `docs/architecture/backtest/benchmark_iterations/<date>_iteration_9_lazy_trades_detail/`;
- if all checks and acceptance gates pass, the work is merged to `main`, `main` is pulled locally, and `main` is pulled on Mac Studio.

## Context / Current State

Precondition:

- Iteration 8 must already be accepted with Mac Studio benchmark evidence:
  `docs/architecture/backtest/benchmark_iterations/2026-05-02_iteration_8_execution_sizing_completion/`.
- If accepted Iteration 8 evidence is missing, failed, or contradicts the runtime document, stop before implementation and report that Iteration 9 is blocked.

Context ledger from previous accepted iterations:

- completed:
  - Iteration 1: request normalization, defaults, guardrails, and artifact context.
  - Iteration 2: prepare pools.
  - Iteration 3: combo planning contexts.
  - Iteration 4: no-risk exact scoring, heap/top-result proxy fill, result shape/hash parity, accounting.
  - Iteration 5: risk-on TP/SL data path.
  - Iteration 6: TP/SL exact scoring and full metrics.
  - Iteration 7: job orchestration, public/storage variant identity, summary-only top-N persistence.
  - Iteration 8: execution/sizing completion.
- open_items:
  - trades for a top variant are not returned by default;
  - UI/API needs an on-demand "show trades" endpoint;
  - full trades must be computed lazily and cached, not persisted in top-N summary rows.
- contract_changes:
  - new public endpoint and response DTO;
  - new cache port/adapter and cache identity;
  - optional small cache metadata persistence only if bounded and justified.
- touched_paths:
  - API route/DTO/wiring;
  - `BacktestJobsUseCase`;
  - lazy trades detail service;
  - cache port/adapter;
  - tests and benchmark runner.
- risks:
  - resolving raw storage SHA as public key would violate the v1 `variant_key` contract;
  - recomputing the whole job/grid would be too slow and wrong for the endpoint;
  - storing full trades in `backtest_job_top_variants.trades_json` would break the summary-only contract;
  - cache identity missing artifact or engine hashes can return stale trades;
  - local file cache can fail independently from recompute and must be non-fatal.

Important current contract:

- Public `variant_key` is the readable route/UI key.
- `variant_hash` is the stable SHA-256 of canonical variant params and remains the storage identity where legacy schema expects SHA-only keys.
- `BacktestJobTopVariant.variant_key` is SHA-only storage identity.
- Public readable key is carried in `payload_json.public_variant_key`.
- API responses expose both `variant_key` and `variant_hash`.
- Lazy trades cache key must include:
  - `job_id`;
  - public `variant_key`;
  - `variant_hash`;
  - `request_hash`;
  - `engine_params_hash`;
  - `artifact_manifest_hash`.

## Requirements (Must)

- Read context using the protocol below and stop early once sufficient.
- Verify Iteration 8 acceptance evidence before implementation.
- Implement only the scoped change described in this prompt.
- Preserve all accepted benchmark and runtime behavior from Iterations 1..8.
- Add the public endpoint:

```http
POST /backtests/jobs/{job_id}/variants/{variant_key}/trades
```

- Use the same authentication/current-user dependency style as the existing backtest routes.
- Enforce ownership before returning variant detail:
  - missing job: `404 backtest.not_found`;
  - job owned by another user: `403 backtest.forbidden`;
  - public variant key not found inside job: `404 backtest.not_found`.
- Resolve variant only through public readable `variant_key`.
- Do not add public lookup by raw storage SHA.
- Validate that the resolved row has a coherent public `variant_key` and `variant_hash`.
- If payload `public_variant_key` or `variant_hash` contradicts the route/storage row, fail deterministically with a 409-style conflict or an existing compatible Roehub error code; document the choice.

Lazy recompute:

- Recompute exact trades for one selected variant only.
- Use persisted job request snapshot and artifact metadata from the job row.
- Use selected variant params from the top row payload:
  - `canonical_variant_params`;
  - risk mode and best TP/SL cell when applicable;
  - execution settings from the persisted request;
  - sizing, profit lock, direction mode, close-on-end, fees, and slippage from the accepted Iteration 8 contract.
- Read current artifact data by historical-prefix invariant and stored artifact metadata.
- Use `hit_times/15m` for TP/SL; do not use legacy `hit_times/1m`.
- Return trades in deterministic order.
- Return full summary metrics in the detail payload.
- The recompute path must be isolated from the full job/top-N pipeline:
  - no Cartesian grid iteration over unrelated variants;
  - no top-N heap update;
  - no persistence of new top rows.

Payload shape:

- Return a backend chart-ready payload, not UI rendering.
- Include at minimum:
  - `job_id`;
  - public `variant_key`;
  - `variant_hash`;
  - `request_hash`;
  - `engine_params_hash`;
  - `artifact_manifest_hash`;
  - `summary_metrics`;
  - `canonical_variant_params`;
  - `readable_params`;
  - `trades`;
  - `chart_overlay`;
  - `cache`;
  - `timing`.
- Trade rows must be explicit and stable enough for UI overlay:
  - entry timestamp/bar index;
  - exit timestamp/bar index;
  - side/direction;
  - entry price;
  - exit price;
  - quantity/notional where available;
  - return/profit metrics;
  - fee/slippage where available;
  - exit reason (`signal`, `take_profit`, `stop_loss`, `close_on_end`, or equivalent stable enum).
- `chart_overlay` should be derived from trades and contain candle-overlay-friendly markers/segments, but do not implement chart rendering.
- If exact candle payload design is not final, keep the DTO minimal and versioned enough for Iteration 10 to adapt without breaking the endpoint.

Cache:

- Implement a cache port in the application layer.
- Implement a local object/file cache adapter with default root:
  `/opt/roehub/state/backtest/trades_cache`.
- Use TTL 48h by default.
- Cache key must include all required identity fields listed above.
- Cache payload must be deterministic JSON or another documented stable format.
- Use atomic writes.
- Treat missing, expired, corrupt, or unreadable cache entries as cache misses.
- If cache read/write/delete fails but recompute succeeds, return the recomputed response and record a warning/telemetry flag.
- Do not store full trades in `BacktestJobTopVariant.trades_json`.
- Postgres JSONB storage is allowed only for small bounded metadata if needed; do not store unbounded trade payloads in the main DB.
- Preserve v1 topology:
  - single API/worker host or sticky local cache semantics are acceptable;
  - cache miss is normal and deterministic recompute-safe;
  - port boundary must allow replacing local cache with shared object storage before multi-host scale-out.

Benchmark/evidence:

- Add `scripts/backtest/run_iteration_9_lazy_trades_benchmark.py`.
- Write evidence to:
  - `docs/architecture/backtest/benchmark_iterations/YYYY-MM-DD_iteration_9_lazy_trades_detail/benchmark_results.json`;
  - `docs/architecture/backtest/benchmark_iterations/YYYY-MM-DD_iteration_9_lazy_trades_detail/benchmark_summary.md`.
- Evidence must include:
  - commit;
  - host;
  - Python/Numba versions where available;
  - request hash;
  - engine/config hash;
  - artifact manifest hash;
  - hit-times manifest hash for TP/SL;
  - artifact compatibility policy;
  - selected public `variant_key`;
  - selected `variant_hash`;
  - risk mode;
  - sizing mode;
  - direction mode;
  - close-on-end;
  - cache root;
  - cache key;
  - cache TTL;
  - `lazy_trades_compute`;
  - `lazy_trades_cache_hit`;
  - CPU/RSS metrics;
  - trade count;
  - summary/trade parity checks against the selected persisted top row.
- `lazy_trades_compute` and `lazy_trades_cache_hit` are service-only gates.
- Do not include lazy trades timers in `total_without_warmup`.
- Compare `lazy_trades_cache_hit` against the first accepted service baseline only after this iteration establishes that baseline. For this iteration, enforce absolute budget and correctness, and record values clearly.

Tests:

- Add tests for:
  - cache miss recomputes and writes cache;
  - cache hit returns cached payload and does not recompute;
  - cache read failure plus successful recompute returns success;
  - cache write failure plus successful recompute returns success with warning/telemetry;
  - ownership failure;
  - public variant key not found;
  - route refuses/does not resolve raw storage SHA as public key;
  - variant key/hash mismatch failure;
  - no-risk trade detail payload;
  - TP/SL trade detail payload with TP/SL exit reason;
  - close-on-end true/false where fixtures can cover it;
  - sizing/profit-lock fields preserved from Iteration 8 semantics.

Delivery:

- If and only if local gates, Mac Studio evidence, benchmark/correctness acceptance, docs checks, and CI all pass:
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

- Keep the public endpoint idempotent in effect: repeated requests for the same cached/recomputed identity should return the same payload unless artifacts outside the immutable prefix are irrelevant.
- Keep response shape explicit rather than passing raw internal dicts through the API.
- Keep cache-key construction in one tested helper.
- Keep lazy trades service independent from FastAPI route code.
- Keep benchmark runner output shape close to previous accepted iteration summaries.
- Include cache status in telemetry: `miss`, `hit`, `expired`, `read_failed`, `write_failed`.
- Prefer reusing accepted exact scorer/accounting functions over duplicating trade semantics.

## Requirements (Nice-to-have)

- Add a local developer smoke fixture that can validate cache behavior without full Mac Studio artifacts.
- Add cache cleanup/prune helper if it is small and safe, but do not make cleanup required for endpoint success.

# Context acquisition protocol

Read only in this order and stop once sufficient:

1. `.codex/AGENTS.md`
2. Iteration 9 section plus variant identity, lazy trades, benchmark stages, guardrails, and cache-risk sections of `docs/architecture/backtest/backtest-service-artifact-runtime-v1.ru.md`
3. accepted Iteration 8 evidence folder; stop if missing or failed
4. task entrypoints
5. conditional bundles required by touched contracts, failing checks, or parity ambiguity
6. consult-if-needed references only for blockers, ambiguity, or conflict resolution

Do not eagerly preload all listed sources.

Pre-implementation reading target:

- `<= 12 files`
- `<= ~60k-75k tokens`

Stop reading once all of the following are true:

- public endpoint contract is identified;
- ownership/variant lookup path is identified;
- one-variant recompute path is identified;
- cache port/adapter touch set is bounded;
- benchmark evidence shape is implementable;
- no unresolved public API, persistence, or cache-identity ambiguity remains.

Expand context only for:

- trade reconstruction ambiguity;
- variant key/hash mismatch;
- cache identity/persistence contract conflict;
- failing route/use-case tests;
- Mac Studio benchmark/deploy failure.

# Reading manifest

Use the front-matter `context_sources` as the canonical reading map.

Read with this intent:

- `always_read`:
  - repository rules;
  - current runtime target;
  - benchmark evidence contract;
  - accepted Iteration 8 precondition.
- `task_entrypoints`:
  - public API route/DTO;
  - job/variant ownership and lookup;
  - public/storage variant identity mapping.
- `conditional_bundles`:
  - read only when the stated condition applies.
- `consult_if_needed`:
  - read only for blockers, ambiguity, or conflict resolution.

Do not convert this manifest into a broad mandatory reading list.

# Work plan (agent should follow)

Skill routing for this task:

- `contract-impact-analysis`: use before implementation and before final report; owns public endpoint compatibility, DTO schema, cache identity, repository port impact, and persisted summary contract.
- `backend-performance-evidence`: use during benchmark verification; owns lazy trades compute/cache-hit measurements, CPU/RSS evidence, Mac Studio acceptance claims, and service-only stage separation.
- `backend-quality-gates`: use during verification; owns uv-based lint, type, API, unit, and regression gates.
- `root-cause-debugging`: use only after a concrete failure; owns isolating failed recompute/cache/ownership/benchmark behavior before the second attempt.
- `publish-ci-deploy`: use only after all implementation and Mac Studio acceptance gates pass; owns push, PR/CI, merge to `main`, local pull, Mac Studio pull, deploy verification, and post-deploy evidence.

Implementation sequence:

1. Verify Iteration 8 accepted evidence exists. Stop if missing.
2. Read bounded context and classify contract impact before code changes.
3. Confirm current public/storage variant mapping:
   - public route key in `payload_json.public_variant_key`;
   - storage SHA in `BacktestJobTopVariant.variant_key`;
   - API `variant_hash` maps to storage SHA.
4. Design lazy trades DTO/read-model and cache identity helper.
5. Add cache port and local file adapter.
6. Add lazy trades service that:
   - owns public variant lookup;
   - validates identity;
   - checks cache;
   - recomputes exactly one variant on miss;
   - writes cache best-effort;
   - returns chart-ready payload.
7. Add API route and wiring.
8. Add route/use-case/service/cache tests.
9. Add Iteration 9 benchmark runner and evidence writer.
10. Run local gates and fix introduced failures.
11. Run Mac Studio benchmark/correctness evidence.
12. If accepted, update the main runtime document status for Iteration 9 and docs index.
13. If all gates are green, use `publish-ci-deploy` to push, PR, watch CI, merge to `main`, pull locally, pull on Mac Studio, and verify production/runtime health.
14. If any gate fails after two implementation attempts, stop and report exact blockers.

# Benchmark and Mac Studio pipeline

Acceptance benchmark and post-merge verification must use Mac Studio evidence.

Implementation benchmark path:

```bash
export PATH="/opt/homebrew/bin:$PATH"
cd /Users/daniildegtyarev/Projects/roehub.com
git pull --ff-only
uv run python scripts/backtest/run_iteration_9_lazy_trades_benchmark.py \
  --out-dir docs/architecture/backtest/benchmark_iterations/$(date +%F)_iteration_9_lazy_trades_detail
```

The runner must measure at least:

- cold cache/miss path: `lazy_trades_compute`;
- warm cache path: `lazy_trades_cache_hit`;
- correctness parity between returned summary and persisted selected top row;
- cache identity fields and TTL.

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

- Iteration 8 accepted evidence exists before implementation.
- `POST /backtests/jobs/{job_id}/variants/{variant_key}/trades` exists and is wired.
- Endpoint enforces authentication and ownership.
- Endpoint resolves only public readable `variant_key`.
- Raw storage SHA lookup is not exposed as public v1 behavior.
- Variant key/hash mismatch is detected and fails deterministically.
- Lazy recompute processes exactly one selected variant.
- No-risk lazy trades payload passes correctness tests.
- TP/SL lazy trades payload passes correctness tests.
- Execution/sizing/profit-lock/close-on-end semantics from Iteration 8 are preserved.
- Cache key includes `job_id`, `variant_key`, `variant_hash`, `request_hash`, `engine_params_hash`, and `artifact_manifest_hash`.
- Cache miss recomputes and writes cache.
- Cache hit returns cached payload.
- Cache failure with successful recompute returns success and records warning/telemetry.
- Full trades are not stored in summary top rows.
- `lazy_trades_compute` and `lazy_trades_cache_hit` evidence exists.
- Lazy trades timers are not included in `total_without_warmup`.
- Mac Studio evidence passes.
- Local full gates pass before merge.
- CI passes before merge.
- The branch is merged to `main`.
- Local checkout at `/Users/daniildegtyarev/Projects/roehub.com` is on updated `main`.
- Mac Studio checkout at `/Users/daniildegtyarev/Projects/roehub.com` is on updated `main`.
- Mac Studio production/runtime smoke passes after merge/deploy.

# Implementation constraints

- Keep diffs scoped to Iteration 9.
- Do not silently change public job/top/variant response fields from Iteration 7/8.
- Do not persist full trades in `backtest_job_top_variants.trades_json`.
- Do not create a public endpoint that accepts `variant_hash` instead of public `variant_key`.
- Do not recompute the full top-N job to answer one lazy trades request.
- Do not compare lazy trades against canonical notebook `total_without_warmup`.
- Do not mark Iteration 9 as accepted from local tests alone.
- Do not merge if any required gate is missing, red, or ambiguous.

# Files to indicate (expected touched areas)

Expected primary files:

- `apps/api/routes/backtests.py`
- `apps/api/dto/backtests.py`
- `apps/api/wiring/modules/backtest.py`
- `src/trading/contexts/backtest/application/use_cases/backtest_jobs.py`
- `src/trading/contexts/backtest/application/dto/backtest_jobs.py`
- `src/trading/contexts/backtest/application/ports/backtest_job_repositories.py`
- `src/trading/contexts/backtest/application/ports/<lazy trades cache port>.py`
- `src/trading/contexts/backtest/application/services/v2/<lazy trades detail service>.py`
- `src/trading/contexts/backtest/adapters/outbound/<local trades cache adapter>.py`
- `tests/unit/apps/api/test_backtests_routes.py`
- `tests/unit/contexts/backtest/application/<new lazy trades tests>.py`
- `scripts/backtest/run_iteration_9_lazy_trades_benchmark.py`
- `docs/architecture/backtest/benchmark_iterations/YYYY-MM-DD_iteration_9_lazy_trades_detail/benchmark_results.json`
- `docs/architecture/backtest/benchmark_iterations/YYYY-MM-DD_iteration_9_lazy_trades_detail/benchmark_summary.md`

Possible secondary files:

- `src/trading/contexts/backtest/application/services/v2/no_risk_exact.py`
- `src/trading/contexts/backtest/application/services/v2/tp_sl_exact.py`
- `src/trading/contexts/backtest/application/services/v2/execution_sizing.py`
- `src/trading/contexts/backtest/application/services/v2/benchmark_accounting.py`
- `src/trading/contexts/backtest/adapters/outbound/persistence/postgres/backtest_job_repository.py`
- `src/trading/contexts/backtest/application/dto/runtime_preflight.py`
- `docs/architecture/backtest/backtest-service-artifact-runtime-v1.ru.md`
- `docs/architecture/README.md`

# Non-goals

- No UI integration or browser chart rendering.
- No candle chart component work.
- No new risk modes.
- No new sizing modes.
- No indicator catalog expansion.
- No broad roadmap cleanup.
- No shared object storage implementation unless absolutely necessary for a tested port boundary.
- No legacy `hit_times/1m` runtime path.

# Quality gates (must run and pass)

Run local gates before claiming implementation completion:

```bash
uv run ruff check apps/api src/trading/contexts/backtest tests/unit/apps/api tests/unit/contexts/backtest scripts/backtest
uv run pyright
uv run pytest -q tests/unit/apps/api/test_backtests_routes.py tests/unit/contexts/backtest/application/use_cases/test_backtest_jobs_use_case.py tests/unit/contexts/backtest/application/services/v2 tests/unit/contexts/backtest/domain/entities/test_backtest_job_entities.py
uv run pytest -q -ra
```

Run local benchmark/developer evidence when artifacts are available:

```bash
uv run python scripts/backtest/run_iteration_9_lazy_trades_benchmark.py \
  --out-dir docs/architecture/backtest/benchmark_iterations/<iteration_9_dir>
```

After docs or benchmark summaries are written:

```bash
uv run python -m tools.docs.generate_docs_index --check
```

Run the Mac Studio benchmark gate before claiming acceptance:

```bash
export PATH="/opt/homebrew/bin:$PATH"
uv run python scripts/backtest/run_iteration_9_lazy_trades_benchmark.py \
  --out-dir docs/architecture/backtest/benchmark_iterations/$(date +%F)_iteration_9_lazy_trades_detail
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
- do not broaden scope into Iteration 10;
- do not merge;
- do not hide failed benchmark rows, cache failures, authz failures, or correctness failures;
- report:
  - implementation commit;
  - changed files;
  - exact failed route/use-case/service/cache/benchmark scenario;
  - selected public `variant_key` and `variant_hash`;
  - request hash, engine params hash, artifact manifest hash;
  - expected value, actual value, and diff where applicable;
  - whether the failure is public variant lookup, ownership, cache identity, recompute semantics, no-risk trades, TP/SL trades, sizing/profit-lock carryover, benchmark runner, CI/deploy, or Mac Studio environment;
  - the smallest next investigation step.

# Final output: report format (strict)

Use Russian.

## Что сделано

- Concise implementation summary.

## Lazy trades contract

- Endpoint.
- Lookup semantics.
- Recompute semantics.
- Payload shape.

## Cache / identity

- Cache root.
- TTL.
- Cache key fields.
- Cache miss/hit/failure behavior.

## Benchmark / Mac Studio

- Evidence directory.
- Commit.
- Artifact hashes.
- Selected variants.
- `lazy_trades_compute`.
- `lazy_trades_cache_hit`.
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
- If accepted and merged, state that Iteration 10 is next.
- If not accepted, state the blocker and stop.
