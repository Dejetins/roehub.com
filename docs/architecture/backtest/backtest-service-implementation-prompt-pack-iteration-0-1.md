# Backtest Service Implementation Prompt Pack — Iteration 0/1

Статус: executable prompt pack для запуска первых двух итераций реализации
`Backtest Service Artifact Runtime v1`.

Этот pack не заменяет архитектурный документ. Source of truth для контрактов,
алгоритма, стадий и benchmark policy остается
`docs/architecture/backtest/backtest-service-artifact-runtime-v1.ru.md`.

## Как использовать

1. Запустить prompt `Iteration 0` и закрыть documentation/benchmark harness gate.
2. Сохранить итоговый отчет исполнителя и benchmark/evidence record.
3. Запустить prompt `Iteration 1`, добавив в него ссылку на итоговый отчет
   `Iteration 0`, если он уже есть.
4. Не считать `Iteration 1` accepted без Mac Studio benchmark record. Локальные
   smoke checks допустимы только как developer evidence.

## Общие инварианты для обоих prompts

- Писать итоговый отчет на русском.
- Использовать bounded reading: не загружать notebook и весь backtest code заранее.
- `hit_times/15m` является целевой моделью; `1m/5m` runtime request не поддерживаются.
- UI и внешние клиенты используют один public API; separate internal API не проектируется.
- `variant_key` в public API не должен закрепляться как SHA-only identity.
- Benchmark evidence для acceptance запускается только на `Mac Studio`.

## Prompt: Iteration 0

```markdown
---
prompt_name: backtest-service-artifact-runtime-v1-iteration-0-docs-benchmark-harness
repo: roehub.com
branch: main
scope: "Закрыть Iteration 0: source-of-truth документацию, benchmark harness contract и навигацию для Backtest Service Artifact Runtime v1."

language:
  implementation: docs
  agent_report: ru

context_sources:
  always_read:
    - path: .codex/AGENTS.md
      why: "repo contract and skill routing"
    - path: .codex/agents/.context/promt_manager_state.yaml
      why: "latest compact state, if present"
    - path: docs/architecture/backtest/backtest-service-artifact-runtime-v1.ru.md
      why: "canonical target contract"

  task_entrypoints:
    - path: docs/architecture/backtest/backtest-service-artifact-runtime-v1.ru.md
      why: "Iteration 0 scope and benchmark policy"
      inspect_symbols:
        - "Статус"
        - "Политика бенчмарков"
        - "План внедрения"
        - "Итерация 0"
    - path: docs/architecture/backtest/benchmark_iterations/README.md
      why: "benchmark record template"
      inspect_symbols:
        - "Шаблон записи"
        - "Runtime Metrics Without Warmup"
        - "Service-Only Overhead"
    - path: docs/architecture/backtest/benchmark_iterations/2026-04-26_engine_test_btcusdt_15m/benchmark_results.json
      why: "canonical numeric target values"
      inspect_symbols:
        - "methodology"
        - "request_hash"
        - "runs"
        - "sizing_smoke"
    - path: docs/architecture/backtest/benchmark_iterations/2026-04-26_engine_test_btcusdt_15m/benchmark_summary.md
      why: "human-readable benchmark summary"
      inspect_symbols:
        - "Methodology"
        - "Run Matrix Summary"
        - "Stage Metrics"
        - "Sizing Smoke"

  conditional_bundles:
    docs_navigation:
      read_when: "updating source-of-truth links or docs index"
      paths:
        - docs/architecture/backtest/README.md
        - docs/architecture/README.md
        - docs/INDEX.md

    current_code_contracts:
      read_when: "documenting current compatibility gaps against code"
      paths:
        - src/trading/contexts/backtest/domain/value_objects/variant_identity.py
        - src/trading/contexts/backtest/domain/entities/backtest_job.py
        - src/trading/contexts/backtest/application/ports/backtest_job_repositories.py

  consult_if_needed:
    - path: tests/notebook_tests/engine_test/btcusdt_15m_research_engine.ipynb
      read_when: "benchmark method names or stage semantics are ambiguous"
    - path: docs/architecture/backtest/deep-research-report_for_Backtest Service Artifact Runtime v1.md
      read_when: "need prior research rationale for a current doc decision"

style_references:
  - docs/architecture/backtest/backtest-service-artifact-runtime-v1.ru.md
  - docs/architecture/backtest/benchmark_iterations/README.md

hard_requirements:
  no_service_runtime_code_changes: true
  russian_docs_by_default: true
  benchmark_acceptance_macstudio_only: true
  no_legacy_source_of_truth_revival: true
  runtime_doc_is_sole_implementation_source: true
  keep_existing_user_changes: true

task_toggles:
  update_docs_navigation: true
  update_benchmark_template_if_stale: true
  verify_current_backtest_entrypoints: true
  create_iteration_0_evidence_record_if_needed: true

skill_routing:
  - skill: architecture-review
    use_when: "checking docs/source-of-truth drift"
    timing: before implementation
    reason: "owns selective docs-sync and architecture drift classification"
  - skill: backend-performance-evidence
    use_when: "updating benchmark template or evidence requirements"
    timing: during verification
    reason: "owns benchmark baseline, metrics, and performance evidence semantics"
  - skill: contract-impact-analysis
    use_when: "changing public API, variant identity, progress, config, or persistence wording"
    timing: during investigation
    reason: "owns boundary compatibility and contract drift classification"
  - skill: backend-quality-gates
    use_when: "running docs/static checks and triaging failures"
    timing: during verification
    reason: "owns local quality gate execution and failure reporting"

target_envs:
  - local-dev
  - macstudio

required_literals:
  - "tests/notebook_tests/engine_test/btcusdt_15m_research_engine.ipynb"
  - "docs/architecture/backtest/benchmark_iterations/2026-04-26_engine_test_btcusdt_15m/benchmark_results.json"
  - "hit_times/15m"
  - "risk.mode"
  - "variant_key"
  - "variant_hash"
  - "total_without_warmup"

non_goals:
  - "Implement service runtime code."
  - "Restore old deleted legacy prompt packs or runtime docs."
  - "Run or edit scoring kernels."
  - "Define UI chart payload details."
  - "Create a separate internal API."

final_report_format:
  language: ru
  sections:
    - "Изменения"
    - "Контрактные решения"
    - "Проверки"
    - "Benchmark/Evidence"
    - "Следующий шаг"

quality_gates:
  - cmd: "python -m tools.docs.generate_docs_index"
    expect: "updates generated docs index if needed"
  - cmd: "python -m tools.docs.generate_docs_index --check"
    expect: "passes"
  - cmd: "python - <<'PY'\nimport json\np='docs/architecture/backtest/benchmark_iterations/2026-04-26_engine_test_btcusdt_15m/benchmark_results.json'\nwith open(p) as f: d=json.load(f)\nrequired={'methodology','request_hash','artifact_manifest_hash','hit_times_manifest_hash','runs','sizing_smoke'}\nmissing=required-set(d)\nassert not missing, missing\nassert len(d['runs']) == 28, len(d['runs'])\nPY"
    expect: "canonical benchmark JSON has required keys and 28 runs"
  - cmd: "rg -n \"hit_times/1m|execution profile|POST /backtests\\b|\\bruns\\b\" docs/INDEX.md docs/architecture/README.md docs/architecture/backtest/README.md docs/architecture/backtest/backtest-service-artifact-runtime-v1*.md docs/architecture/backtest/benchmark_iterations/README.md || true"
    expect: "matches in current entrypoints are either absent or explicitly classified as superseded/compatibility"
  - cmd: "git diff --check"
    expect: "passes"

expected_primary_touches:
  - "docs/architecture/backtest/backtest-service-artifact-runtime-v1.ru.md"
  - "docs/architecture/backtest/benchmark_iterations/README.md"
  - "docs/architecture/backtest/README.md"
  - "docs/architecture/README.md"
  - "docs/INDEX.md"

possible_secondary_touches:
  - "docs/architecture/backtest/benchmark_iterations/<date>_iteration_0_docs_benchmark_harness/"
  - "docs/architecture/backtest/backtest-service-artifact-runtime-v1.md"

safety_notes:
  - "Do not revert unrelated user changes in .codex, docs, or code."
  - "Do not read roadmap docs as implementation context; use the runtime document as the sole target contract."
  - "Benchmark numbers must come from the canonical JSON, not from manual summary edits."
---

# Task

Закрыть `Iteration 0` для `Backtest Service Artifact Runtime v1`: привести
документацию, benchmark harness contract и навигацию к состоянию, в котором
реализацию Iteration 1 можно запускать без конкурирующих source-of-truth.

Done means:

- текущий runtime-документ явно является canonical target для нового сервиса;
- benchmark folder и template используют актуальные stage names из current notebook evidence;
- устаревшие формулировки `POST /backtests`, `runs`, `hit_times/1m`,
  `execution profile` не выглядят как активный v1 contract;
- docs navigation указывает на актуальные документы;
- checks из `Quality gates` пройдены или отклонения явно описаны.

## Context / Current State

Context ledger from the previous iteration:

- completed:
  - canonical notebook выбран: `tests/notebook_tests/engine_test/btcusdt_15m_research_engine.ipynb`;
  - canonical benchmark evidence выбран: `2026-04-26_engine_test_btcusdt_15m`;
  - public v1 API построен вокруг `/backtests/jobs`, `/backtests/preflight`, `/backtests/runtime-defaults`;
  - internal API на v1 не нужен.
- open_items:
  - проверить, что benchmark iteration template не содержит старый пятистадийный runtime словарь как target;
  - проверить docs navigation на stale links к удаленным backtest prompt/docs;
  - зафиксировать Iteration 0 evidence record, если в репозитории его еще нет.
- contract_changes:
  - `risk.mode` заменяет public `execution profile`;
  - `hit_times/15m` является target path;
  - public `variant_key` человекочитаемый и уникальный within job, storage identity отдельно через `variant_hash`;
  - persisted coarse stages могут оставаться `stage_a/stage_b/finalizing`, API progress использует canonical pipeline stage names.
- touched_paths:
  - expected docs-only changes.
- risks:
  - старые docs/code могут честно подсказывать неверную v1 модель;
  - benchmark record template может требовать устаревшие timer names;
  - current domain code может хранить SHA-only `variant_key`.
- next_focus:
  - подготовить Iteration 1 к request normalization и artifact context без scoring.

Additional context:

- Acceptance benchmark запускается только на `Mac Studio`.
- Local docs/static checks полезны, но не заменяют benchmark evidence.

## Requirements (Must)

- Read context using the protocol below and stop early once sufficient.
- Implement only the scoped docs/harness changes described in this prompt.
- Preserve all explicitly protected contracts and invariants.
- Add or update targeted docs when needed.
- Keep the implementation deterministic and reviewable.
- Do not change service runtime code in Iteration 0.
- Do not resurrect deleted legacy documents as source of truth.
- If benchmark template mentions old timers such as `count_trades` as active target,
  update it to the current timer list from the runtime document and benchmark JSON.
- If stale docs navigation points to removed files, replace it with current canonical
  backtest docs.
- Record every unresolved contradiction as an explicit open item, not as silent ambiguity.

## Requirements (Should)

- Prefer the Russian runtime document as the human-readable canonical doc.
- Keep English code identifiers, API routes, config keys, metric names and file paths unchanged.
- Do not read or patch roadmap docs in this iteration unless the user explicitly asks for roadmap cleanup.

## Requirements (Nice-to-have)

- Add a compact Iteration 0 evidence record under `benchmark_iterations/` if useful.
- Add a short command contract for future Mac Studio benchmark runs if an actual command already exists.

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

- changed docs/contracts are identified;
- touched files are bounded;
- acceptance criteria are implementable without ambiguity;
- no unresolved public API, benchmark, or persistence-contract ambiguity remains.

Expand context only for:

- blockers;
- failing quality gates;
- unclear contracts;
- benchmark threshold conflicts;
- architecture conflicts that affect correctness.

# Reading manifest

Use the front-matter `context_sources` as the canonical reading map.

Read with this intent:

- `always_read`: repository rules, compact prior state, canonical target contract;
- `task_entrypoints`: benchmark policy and canonical evidence;
- `conditional_bundles`: navigation, superseded vocabulary, or current compatibility gaps;
- `consult_if_needed`: notebook or prior research only for blockers.

Do not convert this manifest into a broad mandatory reading list.

# Work plan (agent should follow)

Skill routing for this task:

- `architecture-review`: use before implementation when checking docs/source-of-truth drift; owns selective docs-sync.
- `backend-performance-evidence`: use during verification when changing benchmark template/evidence language; owns metric and baseline semantics.
- `contract-impact-analysis`: use during investigation when touching API, variant identity, progress, config, or persistence wording; owns compatibility classification.
- `backend-quality-gates`: use during verification for docs/static checks and failure triage.

1. Inspect the runtime document and current benchmark iteration template.
2. Compare benchmark template timer names with the current required canonical timers.
3. Inspect current backtest docs navigation only if links are stale or new references are needed.
4. Patch the smallest set of docs needed for source-of-truth closure.
5. Run docs index generation/checks and benchmark JSON shape check.
6. Record unresolved issues only if they block Iteration 1.

# Acceptance criteria (Definition of Done)

- `docs/architecture/backtest/backtest-service-artifact-runtime-v1.ru.md` remains the clear canonical target.
- `benchmark_iterations/README.md` no longer presents stale stage names as the active service target.
- `docs/architecture/backtest/README.md`, `docs/architecture/README.md`, and `docs/INDEX.md` do not point users to missing legacy backtest files as canonical entrypoints.
- The canonical benchmark JSON is readable and contains 28 runs.
- `variant_key` / `variant_hash`, `risk.mode`, progress mapping, and `hit_times/15m` remain documented.
- All quality gates pass, or every failure is classified as pre-existing vs introduced.

# Implementation constraints

- Use `apply_patch` for manual edits.
- Keep docs in Russian unless preserving code identifiers or paths.
- Do not edit notebooks.
- Do not create service runtime modules.
- Do not read or rewrite roadmap docs in this iteration.
- Preserve unrelated user changes.

# Files to indicate (expected touched areas)

Expected primary touched areas are listed in front matter.

If you touch code contract files, explain why that was necessary.
If you do not touch optional areas, say so in the final report.

# Non-goals

- No API route implementation.
- No persistence migration.
- No request normalization implementation.
- No scorer/kernel implementation.
- No UI work.
- No Mac Studio benchmark run unless explicitly available and requested during this iteration.

# Quality gates (must run and pass)

Run the commands listed in front-matter `quality_gates`.

If `python -m tools.docs.generate_docs_index` modifies generated docs, keep the
generated update and rerun `--check`.

# Final output: report format (strict)

Your final message MUST be in Russian and follow exactly:

1. **Изменения**
   - Files changed and why.

2. **Контрактные решения**
   - Source-of-truth, benchmark, API vocabulary, and identity decisions affected.

3. **Проверки**
   - Commands run and result.

4. **Benchmark/Evidence**
   - What evidence exists, what was checked locally, and whether Mac Studio acceptance evidence exists.

5. **Следующий шаг**
   - Whether Iteration 1 can start, plus any blockers.
```

## Prompt: Iteration 1

```markdown
---
prompt_name: backtest-service-artifact-runtime-v1-iteration-1-request-normalization-artifact-context
repo: roehub.com
branch: main
scope: "Реализовать Iteration 1: request normalization, artifact context resolver, runtime-defaults/preflight API shell без scoring/job execution."

language:
  implementation: python
  agent_report: ru

context_sources:
  always_read:
    - path: .codex/AGENTS.md
      why: "repo contract and delivery rules"
    - path: .codex/agents/.context/promt_manager_state.yaml
      why: "latest compact state, if present"
    - path: docs/architecture/backtest/backtest-service-artifact-runtime-v1.ru.md
      why: "canonical v1 contract"

  task_entrypoints:
    - path: docs/architecture/backtest/backtest-service-artifact-runtime-v1.ru.md
      why: "Iteration 1 API and request contract"
      inspect_symbols:
        - "Публичный API v1"
        - "Lifecycle stage: нормализация request"
        - "Итерация 1"
        - "Ресурсные guardrails"
    - path: configs/prod/indicators.yaml
      why: "indicator catalog and source/window defaults"
      inspect_symbols:
        - "defaults"
        - "inputs.source"
        - "params.window"
    - path: configs/prod/backtest_artifacts.yaml
      why: "artifact root, hit-times grid, slot policy"
      inspect_symbols:
        - "artifact_root"
        - "hit_times_grid"
        - "slot_policy"
    - path: src/trading/contexts/backtest_artifacts/application/services/v2/contracts.py
      why: "trusted artifact constants and manifest contracts"
      inspect_symbols:
        - "CURRENT_ARTIFACT_POINTER_FILENAME_V2"
        - "HIT_TIMES_TIMEFRAME_LITERAL_V2"
        - "HIT_TIMES_*"
        - "ROOT_ARTIFACT_MANIFEST_REQUIRED_KEYS_V2"

  conditional_bundles:
    existing_backtest_domain:
      read_when: "creating or changing backtest DTOs, identity, job, or errors"
      paths:
        - src/trading/contexts/backtest/domain/value_objects/variant_identity.py
        - src/trading/contexts/backtest/domain/entities/backtest_job.py
        - src/trading/contexts/backtest/domain/errors/backtest_errors.py
        - src/trading/contexts/backtest/application/ports/backtest_job_repositories.py

    api_route_patterns:
      read_when: "adding public API routes or wiring"
      paths:
        - apps/api/routes/indicators.py
        - apps/api/routes/market_data_reference.py
        - apps/api/common/errors.py
        - apps/api/wiring/modules/indicators.py
        - apps/api/wiring/modules/market_data_reference.py

    artifact_adapter_patterns:
      read_when: "implementing filesystem artifact context resolver"
      paths:
        - src/trading/contexts/backtest_artifacts/adapters/outbound/artifacts_fs/path_builder.py
        - src/trading/contexts/backtest_artifacts/adapters/outbound/artifacts_fs/current_pointer_writer.py
        - src/trading/contexts/backtest_artifacts/application/services/v2/artifact_manifest_loader.py
        - src/trading/contexts/backtest_artifacts/application/services/v2/artifact_manifest_validator.py

    test_patterns:
      read_when: "creating unit/API tests"
      paths:
        - tests/unit/apps/api/test_indicators_wiring_module.py
        - tests/unit/apps/api/test_market_data_reference_routes.py
        - tests/unit/contexts/backtest/application/services/test_signals_from_indicators_v1.py
        - tests/unit/contexts/backtest/adapters/test_indicators_yaml_defaults_provider.py

    persistence_contracts:
      read_when: "if implementation touches jobs/top rows/idempotency storage"
      paths:
        - alembic/versions/20260222_0003_backtest_jobs_v1.py
        - alembic/versions/20260326_0004_backtest_job_artifact_pin_v1.py
        - src/trading/contexts/backtest/adapters/outbound/persistence/postgres/backtest_job_repository.py

  consult_if_needed:
    - path: docs/architecture/backtest/benchmark_iterations/2026-04-26_engine_test_btcusdt_15m/benchmark_results.json
      read_when: "canonical request hash or fixture details are needed"
    - path: docs/architecture/backtest/benchmark_iterations/2026-04-26_engine_test_btcusdt_15m/benchmark_summary.md
      read_when: "human-readable fixture details are needed"
    - path: tests/notebook_tests/engine_test/btcusdt_15m_research_engine.ipynb
      read_when: "request normalization method semantics are ambiguous"

style_references:
  - src/trading/contexts/backtest_artifacts/application/services/v2/contracts.py
  - apps/api/routes/indicators.py
  - apps/api/wiring/modules/indicators.py

hard_requirements:
  no_scoring_or_kernel_work: true
  no_job_execution_worker: true
  public_api_shell_only: true
  reject_1m_5m_runtime_requests: true
  no_internal_api: true
  preserve_public_variant_key_contract: true
  benchmark_acceptance_macstudio_only: true
  keep_existing_user_changes: true

task_toggles:
  implement_request_normalization: true
  implement_artifact_context_resolver: true
  implement_runtime_defaults_endpoint: true
  implement_preflight_endpoint: true
  implement_guardrail_cost_estimate: true
  implement_canonical_request_hash: true
  implement_targeted_tests: true

skill_routing:
  - skill: contract-impact-analysis
    use_when: "designing DTOs, API payloads, defaults, errors, hashes, cache identity, or config keys"
    timing: before implementation
    reason: "owns public API/config/request-hash compatibility"
  - skill: architecture-design
    use_when: "module boundaries or ports/adapters placement are ambiguous"
    timing: during investigation
    reason: "owns target dependency direction and bounded-context placement"
  - skill: backend-performance-evidence
    use_when: "measuring preflight/artifact resolver timings or reporting benchmark readiness"
    timing: during verification
    reason: "owns performance evidence and Mac Studio acceptance distinction"
  - skill: backend-quality-gates
    use_when: "running pytest, ruff, pyright, docs checks, or triaging failures"
    timing: during verification
    reason: "owns backend verification gates"
  - skill: root-cause-debugging
    use_when: "a failing existing test or route behaves unexpectedly"
    timing: if blocker
    reason: "owns failure investigation before fixes"

target_envs:
  - local-dev
  - macstudio

required_literals:
  - "POST /backtests/preflight"
  - "GET /backtests/runtime-defaults"
  - "POST /backtests/jobs"
  - "hit_times/15m"
  - "risk.mode"
  - "none"
  - "tp_sl_grid"
  - "long_only"
  - "long_short_reversal"
  - "total_return_pct"
  - "request_hash"
  - "artifact_manifest_hash"
  - "hit_times_manifest_hash"

non_goals:
  - "Create jobs or execute workers."
  - "Implement prepare_pools, combo planning, exact scoring, heap/top-N, or lazy trades."
  - "Persist top-N summaries."
  - "Add UI."
  - "Add a private/internal API."
  - "Support runtime requests for 1m or 5m."
  - "Change artifact publisher semantics."

final_report_format:
  language: ru
  sections:
    - "Изменения"
    - "Контрактное влияние"
    - "Проверки"
    - "Benchmark/Evidence"
    - "Остаточные риски"

quality_gates:
  - cmd: "uv run pytest -q tests/unit/contexts/backtest"
    expect: "passes, or failures classified as pre-existing with evidence"
  - cmd: "uv run pytest -q tests/unit/apps/api/test_backtests_routes.py"
    expect: "passes after creating targeted backtests route tests"
  - cmd: "uv run ruff check src/trading/contexts/backtest apps/api tests/unit/contexts/backtest tests/unit/apps/api"
    expect: "passes"
  - cmd: "uv run pyright"
    expect: "passes, or unrelated existing failures are explicitly classified"
  - cmd: "python -m tools.docs.generate_docs_index --check"
    expect: "passes if docs were touched"
  - cmd: "git diff --check"
    expect: "passes"

expected_primary_touches:
  - "src/trading/contexts/backtest/domain/value_objects/"
  - "src/trading/contexts/backtest/application/dto/"
  - "src/trading/contexts/backtest/application/services/v2/"
  - "src/trading/contexts/backtest/application/ports/"
  - "src/trading/contexts/backtest/adapters/outbound/artifacts_fs/"
  - "apps/api/routes/backtests.py"
  - "apps/api/wiring/modules/backtest.py"
  - "tests/unit/contexts/backtest/"
  - "tests/unit/apps/api/test_backtests_routes.py"

possible_secondary_touches:
  - "apps/api/routes/__init__.py"
  - "apps/api/main/app.py"
  - "apps/api/wiring/modules/__init__.py"
  - "configs/test/backtest_artifacts.yaml"
  - "configs/dev/backtest_artifacts.yaml"
  - "docs/architecture/backtest/benchmark_iterations/<date>_iteration_1_request_normalization_artifact_context/"

safety_notes:
  - "Do not accept filesystem paths from request payloads."
  - "Do not silently change SHA-only storage fields into public readable variant keys."
  - "If persistence schema must change, stop and report the migration need unless explicitly scoped."
  - "If not running on Mac Studio, do not claim benchmark acceptance."
---

# Task

Реализовать `Iteration 1` для нового backtest service: strict request
normalization, artifact context resolver, cost estimate, canonical
`request_hash`, `GET /backtests/runtime-defaults` и `POST /backtests/preflight`.

Эта итерация не запускает scoring и не создает persisted jobs. Она готовит
публичный contract shell, по которому UI/API смогут построить валидный request и
получить deterministic validation результат до создания job.

Done means:

- service умеет нормализовать request по v1 contract;
- artifact context resolver читает trusted config/current pointer/manifests без user-supplied paths;
- runtime defaults отражают supported public contract и guardrails;
- preflight возвращает normalized request, request hash, artifact metadata, cost estimate, warnings/errors;
- targeted tests покрывают success/failure контракты;
- Iteration 1 evidence/benchmark record создан, если запуск идет на `Mac Studio`.

## Context / Current State

Context ledger from the previous iteration:

- completed:
  - Iteration 0 должен был закрыть docs/source-of-truth и benchmark harness drift;
  - canonical algorithm: `tests/notebook_tests/engine_test/btcusdt_15m_research_engine.ipynb`;
  - canonical benchmark target: `2026-04-26_engine_test_btcusdt_15m/benchmark_results.json`;
  - public API vocabulary: `jobs`, not `runs`;
  - public risk vocabulary: `risk.mode`, not `execution profile`.
- open_items:
  - request normalization and artifact context are not implemented yet;
  - service does not yet expose `/backtests/preflight` or `/backtests/runtime-defaults`;
  - current domain may still assume SHA-only `variant_key`.
- contract_changes:
  - request `coordinates` shape is `{exchange, market_type, symbol}`;
  - time range semantics are `[start, end)`;
  - timeframe public support for this runtime is `15m`; reject `1m` and `5m`;
  - indicator request uses `indicator_id` + `sources` + `window` range;
  - TP/SL grid comes from request and must be covered by published `hit_times/15m`;
  - execution defaults come from config when fields are omitted;
  - `top_n`, arity, rows, combinations and TP/SL cells are guardrailed.
- touched_paths:
  - expected code additions in backtest domain/application/adapters and API.
- risks:
  - broad `configs/prod/indicators.yaml` expansion can create combinatorial requests;
  - artifact resolver must not trust request paths;
  - old storage/domain vocabulary can leak into the public contract.
- next_focus:
  - Iteration 2 will load arrays and implement `prepare_pools`; do not pull that work into Iteration 1.

Additional context:

- Guardrail defaults from the runtime document:
  `max_active_jobs_per_user=1`, `max_queued_jobs_per_user=3`,
  `max_active_jobs_global=1`, `max_top_n=100`,
  `max_indicator_arity=10`, `max_indicator_rows=1000`,
  `max_candidate_combinations=300000`, `max_tp_sl_cells=2209`,
  `lazy_trades_rate_limit=30/10min`, `job_queue_timeout_seconds=300`,
  `job_wall_timeout_seconds=900`, `lazy_trades_timeout_seconds=30`.
- The canonical acceptance benchmark covers arity `1..7`; arity `8..10` are public-contract allowed only inside cost guardrails.

## Requirements (Must)

- Read context using the protocol below and stop early once sufficient.
- Implement only the scoped change described in this prompt.
- Preserve all explicitly protected contracts and invariants.
- Add or update targeted tests where needed.
- Update related exports / nearby docs when required.
- Keep the implementation deterministic and reviewable.
- Validate `coordinates.exchange`, `coordinates.market_type`, and `coordinates.symbol`.
- Validate `timeframe == "15m"` for this runtime. Reject `1m`, `5m`, and unsupported values with deterministic 422.
- Validate `[start, end)` by 15m `open_time` semantics at request level; detailed array slicing is Iteration 2.
- Validate indicator ids against `configs/prod/indicators.yaml`.
- Validate `sources` and `window` ranges against the catalog. Support one source or many sources.
- Resolve execution defaults for `direction_mode`, `fee_rate`, `slippage_rate`,
  `initial_cash_quote`, `sizing`, `profit_lock`, and `close_on_end`.
- Validate sizing modes:
  `all_in`, `fixed_quote`, `fixed_equity_pct`,
  `fixed_equity_pct_min_quote`, `fixed_equity_pct_max_quote`.
- Validate risk modes `none` and `tp_sl_grid`.
- For `tp_sl_grid`, validate request TP/SL levels against configured/published grid coverage where manifest data is available.
- Calculate deterministic canonical `request_hash` from normalized result-affecting request fields.
- Calculate result-affecting config hash if config defaults affect the result.
- Estimate indicator rows, candidate combinations after planned row cap/prefilter rules, TP/SL cells, and cost class.
- Expose `GET /backtests/runtime-defaults`.
- Expose `POST /backtests/preflight`.
- Return deterministic error payloads for invalid request, unsupported grid, artifacts unavailable, and request too expensive.

## Requirements (Should)

- Keep request/default models independent of FastAPI where practical.
- Keep artifact resolver behind an application port.
- Use existing artifact contracts/constants instead of duplicating filenames.
- Keep preflight advisory: `POST /backtests/jobs` in later iterations must repeat authoritative validation.
- Include `artifact_slot`, `artifact_slot_generation`, `artifact_manifest_hash`,
  `artifact_asof_date`, and `hit_times_manifest_hash` when available.
- Compare canonical request hash to benchmark fixture when the exact canonical request can be reconstructed.

## Requirements (Nice-to-have)

- Add a small local timing smoke for preflight/artifact resolver and record it as non-acceptance if not on Mac Studio.
- Add warnings in preflight when request is valid but close to guardrail limits.

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

- changed contracts are identified;
- touched files are bounded;
- acceptance criteria are implementable without ambiguity;
- no unresolved public API or persistence-contract ambiguity remains.

Expand context only for:

- blockers;
- failing quality gates;
- unclear contracts;
- benchmark threshold conflicts;
- architecture conflicts that affect correctness.

# Reading manifest

Use the front-matter `context_sources` as the canonical reading map.

Read with this intent:

- `always_read`: repository rules, prior state, canonical v1 contract;
- `task_entrypoints`: request/API/catalog/artifact context contracts;
- `conditional_bundles`: current domain, API, artifact adapters, tests, persistence only if touched;
- `consult_if_needed`: benchmark or notebook only for hash/fixture ambiguity.

Do not convert this manifest into a broad mandatory reading list.

# Work plan (agent should follow)

Skill routing for this task:

- `contract-impact-analysis`: use before implementation for DTOs/API/defaults/errors/hash/config; owns public contract compatibility.
- `architecture-design`: use during investigation only if module/port placement is ambiguous; owns dependency direction.
- `backend-performance-evidence`: use during verification for preflight/artifact resolver timing and benchmark acceptance language.
- `backend-quality-gates`: use during verification for pytest/ruff/pyright/docs checks.
- `root-cause-debugging`: use only if a failing existing test or route needs root-cause investigation.

1. Identify the minimal current API and wiring patterns for adding `backtests` routes.
2. Define request/default/normalized DTOs and validation errors in the backtest bounded context.
3. Define an artifact context port and filesystem adapter that uses trusted config/current pointer/manifests.
4. Implement runtime defaults service and preflight use case.
5. Wire `GET /backtests/runtime-defaults` and `POST /backtests/preflight`.
6. Add unit tests for validation, defaults, hash determinism, guardrails, and artifact resolver.
7. Add API route tests for success and deterministic failure cases.
8. Run quality gates.
9. If running on `Mac Studio`, create Iteration 1 benchmark/evidence record; otherwise report local smoke only as non-acceptance evidence.

# Acceptance criteria (Definition of Done)

- `GET /backtests/runtime-defaults` returns supported timeframe `15m`, risk modes, direction modes, sizing modes, ranking metrics, top-N defaults and guardrails.
- `POST /backtests/preflight` accepts a valid canonical-style request and returns normalized effective request, `request_hash`, artifact metadata, cost estimate and no blocking errors.
- Invalid indicator id, invalid source, invalid window range, unsupported timeframe, reversed/empty time range, unsupported sizing, unsupported direction mode and request-too-expensive each have deterministic tests.
- `tp_sl_grid` validates TP/SL cell count and grid coverage where artifact/config data is present.
- Preflight never creates a job, never persists top-N, and never runs scoring.
- Public contract does not expose SHA-only `variant_key` as the only v1 identity.
- Artifact resolver never consumes paths from request payloads.
- Targeted tests and static checks pass or failures are explicitly classified.
- Iteration 1 benchmark record exists when executed on `Mac Studio`; otherwise final report states acceptance benchmark is pending.

# Implementation constraints

- Use `apply_patch` for manual edits.
- Preserve DDD / ports-and-adapters direction.
- Domain/application code must not import FastAPI.
- API routes must not contain business rules beyond transport validation/mapping.
- Do not add dependencies unless unavoidable and justified.
- Do not edit notebooks.
- Do not implement Numba kernels or scoring in this iteration.
- Do not create persistence migrations unless the task becomes impossible without one; if so, stop and report the migration need.

# Files to indicate (expected touched areas)

Expected primary and possible secondary touched areas are listed in front matter.

If the implementation chooses different paths, explain the reason and how the
dependency direction remains aligned with the runtime document.

# Non-goals

- No `/backtests/jobs` creation implementation yet, except route names may appear in docs/default links if harmless.
- No `prepare_pools`.
- No `build_exact_context`, `combo_iteration`, `proxy_filter`, or `exact_scoring`.
- No `heap_update` or persisted top-N.
- No lazy trades.
- No UI.
- No separate internal API.

# Quality gates (must run and pass)

Run the commands listed in front-matter `quality_gates`.

If a listed path does not exist until this iteration creates it, create the
targeted tests first and then run the command. If `uv run pyright` has unrelated
pre-existing failures, include enough output to prove they are unrelated.

For performance evidence:

- Record local preflight/artifact resolver timing as developer evidence.
- Only record acceptance benchmark pass/fail if running on `Mac Studio`.
- Benchmark/evidence record path should be:
  `docs/architecture/backtest/benchmark_iterations/<date>_iteration_1_request_normalization_artifact_context/`.

# Final output: report format (strict)

Your final message MUST be in Russian and follow exactly:

1. **Изменения**
   - Files changed and what behavior was added.

2. **Контрактное влияние**
   - public API, DTO, port, persisted schema, config schema, request hash/cache identity, benchmark gate impact as yes/no.

3. **Проверки**
   - Commands run and result.

4. **Benchmark/Evidence**
   - Local timing, Mac Studio acceptance status, request/hash/artifact evidence.

5. **Остаточные риски**
   - Anything blocking Iteration 2 or production acceptance.
```
