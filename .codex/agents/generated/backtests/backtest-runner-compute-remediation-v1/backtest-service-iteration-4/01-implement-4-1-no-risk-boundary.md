---
prompt_name: backtest_service_iteration_4_1_no_risk_boundary
repo: roehub.com
branch: current
scope: "Iteration 4.1: create the no-risk exact scoring DTO/service boundary without implementing measured kernels."

language:
  implementation: python
  agent_report: ru

context_sources:
  always_read:
    - path: .codex/AGENTS.md
      why: "repo contract and verification rules"
    - path: docs/architecture/backtest/backtest-service-artifact-runtime-v1.ru.md
      why: "Iteration 4.1 contract"
  task_entrypoints:
    - path: src/trading/contexts/backtest/application/dto/prepare_pools.py
      why: "prepared pool DTO shape"
      inspect_symbols:
        - BacktestPreparePoolsResult
        - PreparedIndicatorPool
    - path: src/trading/contexts/backtest/application/dto/combo_planning.py
      why: "combo planning DTO shape"
      inspect_symbols:
        - BacktestComboPlanningResult
        - BacktestProxyFilterResult
    - path: src/trading/contexts/backtest/application/services/v2/prepare_pools.py
      why: "existing v2 service style"
      inspect_symbols:
        - BacktestPreparePoolsService
    - path: src/trading/contexts/backtest/application/services/v2/combo_planning.py
      why: "existing v2 stage style"
      inspect_symbols:
        - BacktestComboPlanningService
  conditional_bundles:
    tests:
      read_when: "adding unit coverage"
      paths:
        - tests/unit/contexts/backtest/application/services/v2/test_prepare_pools_service.py
        - tests/unit/contexts/backtest/application/services/v2/test_combo_planning_service.py
    benchmark_manifest:
      read_when: "recording service smoke or memory cleanup evidence"
      paths:
        - docs/architecture/backtest/benchmark_iterations/README.md
  consult_if_needed:
    - path: tests/notebook_tests/engine_test/btcusdt_15m_research_engine.ipynb
      read_when: "boundary ambiguity around notebook top result shape"
    - path: docs/architecture/backtest/benchmark_iterations/2026-04-26_engine_test_btcusdt_15m/benchmark_results.json
      read_when: "benchmark field names are needed"

style_references:
  - src/trading/contexts/backtest/application/services/v2/prepare_pools.py
  - src/trading/contexts/backtest/application/services/v2/combo_planning.py

hard_requirements:
  no_public_api_changes: true
  no_persistence_changes: true
  no_canonical_benchmark_stage_added: true
  result_must_not_retain_heavy_refs: true
  max_implementation_attempts: 2

task_toggles:
  implement_dto_boundary: true
  implement_service_shell: true
  implement_exact_kernels: false
  run_macstudio_acceptance: false
  run_macstudio_smoke_evidence: true

skill_routing:
  - skill: contract-impact-analysis
    use_when: "adding DTOs, exports, or telemetry fields"
    timing: before implementation
    reason: "protect API, persistence, and benchmark contracts"
  - skill: backend-quality-gates
    use_when: "running local backend checks"
    timing: during verification
    reason: "triage ruff, pyright, and pytest failures"

target_envs:
  - local-dev
  - macstudio

required_literals:
  - "risk.mode='none'"
  - "benchmark_top_k"
  - "request_top_n"
  - "top_results_count"
  - "memory cleanup evidence"

non_goals:
  - "Do not implement exact scoring kernels in 4.1."
  - "Do not implement heap_update or top_result_proxy_fill in 4.1."
  - "Do not add public variant_key, variant_hash, persistence, or API read models."
  - "Do not add cleanup as a canonical benchmark stage."

final_report_format:
  language: ru
  sections:
    - "Что сделано"
    - "Контрактное влияние"
    - "Проверки"
    - "Benchmark / memory evidence"
    - "Ограничения и следующий шаг"

quality_gates:
  - cmd: "uv run ruff check src/trading/contexts/backtest/application/dto src/trading/contexts/backtest/application/services/v2 tests/unit/contexts/backtest/application/services/v2"
    expect: "passes"
  - cmd: "uv run pyright"
    expect: "passes"
  - cmd: "uv run pytest -q tests/unit/contexts/backtest/application/services/v2"
    expect: "passes or unrelated failures explicitly isolated"

expected_primary_touches:
  - "src/trading/contexts/backtest/application/dto/no_risk_exact.py"
  - "src/trading/contexts/backtest/application/services/v2/no_risk_exact.py"
  - "tests/unit/contexts/backtest/application/services/v2/test_no_risk_exact_scoring_service.py"

possible_secondary_touches:
  - "src/trading/contexts/backtest/application/dto/__init__.py"
  - "src/trading/contexts/backtest/application/services/v2/__init__.py"

safety_notes:
  - "This is a boundary-only prompt; keep heavy computation out."
  - "Returned DTOs must be compact and must not keep prepared pools, contexts, arrays, or candidate buffers alive."
  - "The executor has only 2 implementation attempts; after the second failed corrective cycle, stop and report the blocker."
---

# Task

Implement Iteration 4.1 from `docs/architecture/backtest/backtest-service-artifact-runtime-v1.ru.md`: create the no-risk exact scoring boundary and compact result DTOs, without implementing measured exact scoring kernels, heap maintenance, or proxy fill.

Done means:

- internal no-risk DTOs/config/telemetry exist and follow current v2 style;
- a service shell can validate `risk.mode='none'`, receive prepared pools + combo planning result + normalized request, and return/raise deterministic boundary-level outcomes;
- returned results contain only compact top result placeholders/telemetry/self-check summary and do not retain heavy per-job references;
- exports and targeted tests are updated.

## Context / Current State

Context ledger:

- completed:
  - Iterations 1-3 are accepted.
  - `prepare_pools` and `combo_planning` v2 services exist.
  - Current canonical benchmark uses `request.top_n=100`, `benchmark_top_k=5`, sample warmup `top_k=1`.
- open_items:
  - no-risk exact service file is absent after revert;
  - Iteration 4 failed previously because exact, heap, proxy fill, shape, and benchmark accounting were mixed.
- contract_changes:
  - none allowed for public API, persistence, request hash, or variant identity.
- risks:
  - accidentally retaining prepared pools or NumPy arrays in result DTOs;
  - adding cleanup as a fake canonical benchmark stage.
- next_focus:
  - establish a safe boundary for later 4.2-4.7 work.

## Requirements (Must)

- Implement only Iteration 4.1 boundary.
- Add DTOs for no-risk config, telemetry, compact result, and price/execution context only as needed by later stages.
- Preserve `request_top_n` separately from `benchmark_top_k`.
- Ensure result objects do not store prepared pools, combo planning context, score arrays, eval/trade matrices, combo chunks, or full candidate sets.
- Add tests proving risk-mode rejection and compact result/telemetry behavior.
- Do not add or modify persistence schemas, public routes, or variant identity fields.

## Requirements (Should)

- Mirror naming and dataclass style from existing v2 DTOs.
- Keep service shell boring and small; later prompts will fill algorithms.
- Prefer immutable tuples/mappings for public result surfaces where consistent with repo style.

## Requirements (Nice-to-have)

- Add a tiny reference-retention test using weak references only if it stays simple.

# Context acquisition protocol

Read only in this order and stop once sufficient:

1. `.codex/AGENTS.md`
2. latest compact state or executor report, if available
3. task entrypoints above
4. conditional bundles only when needed for tests or manifest wording
5. consult-if-needed files only for ambiguity

Do not eagerly preload all listed sources.

Pre-implementation reading target:

- `<= 8 files`
- `<= ~35k-50k tokens`

Stop reading once changed contracts are identified, touched files are bounded, and acceptance criteria are implementable without ambiguity.

# Reading manifest

Use the front-matter `context_sources` as the canonical reading map. Do not convert it into a broad mandatory reading list.

# Work plan (agent should follow)

Skill routing:

- `contract-impact-analysis`: use before implementation if adding DTO fields or exports; owns compatibility review.
- `backend-quality-gates`: use during verification for local gates.

1. Inspect current v2 DTO/service style.
2. Add no-risk DTOs and service shell with deterministic rejection for non-`none` risk mode.
3. Add exports only where local patterns require them.
4. Add targeted tests for config/telemetry, top-k separation, and compact-result behavior.
5. Run local quality gates.
6. If a smoke runner exists, run Mac Studio service smoke evidence; do not label it canonical `>=90%` acceptance.

Mac Studio smoke evidence pipeline:

1. Commit scoped local changes after local gates pass.
2. Push branch/commit to the remote repository.
3. SSH to `macstudio`.
4. In `/opt/roehub/app`, fetch and pull the pushed commit with fast-forward semantics.
5. Verify Mac Studio checkout commit equals the pushed commit.
6. Run the smallest available smoke that proves compact no-risk result shape and no retained heavy references.
7. Save evidence under `docs/architecture/backtest/benchmark_iterations/YYYY-MM-DD_iteration_4_1_no_risk_boundary/` if evidence is produced.
8. Mark evidence as service hygiene smoke, not canonical benchmark acceptance.

# Acceptance criteria (Definition of Done)

- No-risk boundary exists and imports cleanly.
- `request_top_n` and `benchmark_top_k` are separate telemetry concepts.
- Result DTOs are compact and do not retain heavy compute references.
- Non-`risk.mode='none'` request is rejected deterministically.
- Local ruff, pyright, and targeted pytest pass or unrelated failures are clearly isolated.
- Service smoke evidence exists if a runner exists; otherwise the final report says smoke evidence is pending and why.

# Implementation constraints

## Determinism & ordering

- Keep telemetry field names deterministic.
- Do not introduce public/storage identity generation.

## API / contracts

- Public API contract: none.
- DTO schema: compatible internal addition only.
- Persisted schema: none.
- Config schema: none.
- Request hash/cache identity: none.

## Benchmark / performance

- No Mac Studio canonical acceptance is required for 4.1.
- If recording smoke evidence, use `docs/architecture/backtest/benchmark_iterations/README.md`; mark it service hygiene evidence, not canonical `>=90%` benchmark evidence.
- Do not mark Iteration 4.1 accepted as a benchmarked compute stage; this sub-iteration only creates the safe boundary for measured stages.

# Files to indicate (expected touched areas)

- `src/trading/contexts/backtest/application/dto/no_risk_exact.py`
- `src/trading/contexts/backtest/application/services/v2/no_risk_exact.py`
- `tests/unit/contexts/backtest/application/services/v2/test_no_risk_exact_scoring_service.py`
- nearby `__init__.py` exports if needed

# Non-goals

- exact scoring kernels;
- self-check implementation;
- heap update;
- top result proxy fill;
- benchmark runner;
- persistence/API identity.

# Quality gates (must run and pass)

- `uv run ruff check src/trading/contexts/backtest/application/dto src/trading/contexts/backtest/application/services/v2 tests/unit/contexts/backtest/application/services/v2`
- `uv run pyright`
- `uv run pytest -q tests/unit/contexts/backtest/application/services/v2`

# Final output: report format (strict)

Report in Russian:

1. Что сделано.
2. Какие файлы изменены.
3. Contract impact classification: public API, DTO, persisted schema, config, request/cache identity, benchmark gate.
4. Проверки и результаты.
5. Что остается для 4.2.
