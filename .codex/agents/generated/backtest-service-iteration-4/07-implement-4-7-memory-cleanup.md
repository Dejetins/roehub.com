---
prompt_name: backtest_service_iteration_4_7_memory_cleanup
repo: roehub.com
branch: current
scope: "Iteration 4.7: add memory cleanup evidence and bounded per-job reference lifecycle for no-risk run."

language:
  implementation: python
  agent_report: ru

context_sources:
  always_read:
    - path: .codex/AGENTS.md
      why: "repo contract"
    - path: docs/architecture/backtest/backtest-service-artifact-runtime-v1.ru.md
      why: "memory cleanup contract"
    - path: docs/architecture/backtest/benchmark_iterations/README.md
      why: "memory evidence manifest"
  task_entrypoints:
    - path: src/trading/contexts/backtest/application/services/v2/no_risk_exact.py
      why: "no-risk per-job objects"
      inspect_symbols:
        - BacktestNoRiskExactScoringService
    - path: src/trading/contexts/backtest/application/services/v2/prepare_pools.py
      why: "prepared arrays and pools"
      inspect_symbols:
        - BacktestPreparePoolsService
    - path: src/trading/contexts/backtest/application/services/v2/combo_planning.py
      why: "combo planning contexts"
      inspect_symbols:
        - BacktestComboPlanningService
  conditional_bundles:
    orchestration:
      read_when: "worker/job orchestration boundary exists and is touched"
      paths:
        - src/trading/contexts/backtest/application/ports/staged_runner.py
        - src/trading/contexts/backtest/application/services
    tests:
      read_when: "adding repeated-run or reference-retention tests"
      paths:
        - tests/unit/contexts/backtest/application/services/v2/test_no_risk_exact_scoring_service.py
        - tests/unit/contexts/backtest/application/services/v2/test_prepare_pools_service.py
  consult_if_needed:
    - path: docs/runbooks/mac-studio-native-backend-operations.md
      read_when: "worker recycle or Mac Studio process behavior is unclear"

style_references:
  - docs/architecture/backtest/benchmark_iterations/README.md

hard_requirements:
  cleanup_not_canonical_stage: true
  no_90_percent_without_baseline: true
  repeated_run_memory_smoke: true
  max_implementation_attempts: 2

task_toggles:
  implement_cleanup_boundary: true
  implement_memory_evidence: true
  implement_scoring_changes: false

skill_routing:
  - skill: backend-performance-evidence
    use_when: "recording RSS/CPU/memory evidence"
    timing: during verification
    reason: "memory evidence is service-specific, not canonical benchmark"
  - skill: production-risk-review
    use_when: "reviewing cleanup/recycle behavior"
    timing: before ship
    reason: "avoid breaking artifact cache or worker lifecycle"
  - skill: backend-quality-gates
    use_when: "running local checks"
    timing: during verification
    reason: "lint/type/test gates"

target_envs:
  - local-dev
  - macstudio

required_literals:
  - "memory cleanup evidence"
  - "rss_before"
  - "rss_peak"
  - "rss_after_cleanup"
  - "retained_rss_delta"

non_goals:
  - "Do not add cleanup as a canonical benchmark stage."
  - "Do not compare cleanup duration to notebook target."
  - "Do not change scoring semantics."
  - "Do not change public API."

final_report_format:
  language: ru
  sections:
    - "Что сделано"
    - "Memory evidence"
    - "Проверки"
    - "Contract impact"
    - "Ограничения"

quality_gates:
  - cmd: "uv run ruff check src/trading/contexts/backtest/application/services/v2 tests/unit/contexts/backtest/application/services/v2"
    expect: "passes"
  - cmd: "uv run pyright"
    expect: "passes"
  - cmd: "uv run pytest -q tests/unit/contexts/backtest/application/services/v2"
    expect: "passes or unrelated failures isolated"

expected_primary_touches:
  - "src/trading/contexts/backtest/application/services/v2/no_risk_exact.py"
  - "tests/unit/contexts/backtest/application/services/v2/test_no_risk_exact_scoring_service.py"

possible_secondary_touches:
  - "worker/orchestration files if current implementation has a job scope boundary"
  - "docs/architecture/backtest/benchmark_iterations/YYYY-MM-DD_iteration_4_7_memory_cleanup/"

safety_notes:
  - "Artifact mmap/cache handles may remain in bounded runtime cache; per-job arrays must not be retained."
  - "RSS may not immediately return to OS on macOS; use retained RSS and recycle evidence, not naive expectations."
  - "The executor has only 2 implementation attempts; after the second failed corrective cycle, stop and report the blocker."
---

# Task

Implement Iteration 4.7 memory cleanup evidence and bounded per-job reference lifecycle for no-risk runs.

Done means:

- no-risk result objects do not retain heavy arrays or contexts;
- per-job heavy objects are released after result availability;
- repeated-run memory smoke records `rss_before`, `rss_peak`, `rss_after_cleanup`, `retained_rss_delta`;
- cleanup evidence is documented as service hygiene, not canonical benchmark stage.

## Context / Current State

- Current plan explicitly says memory cleanup is not a canonical stage.
- Artifact mmap/cache handles can be retained by bounded artifact cache; per-job arrays cannot.
- Mac Studio evidence is needed for realistic RSS behavior.

## Requirements (Must)

- Add cleanup boundary where current service/orchestration shape supports it.
- If no worker/job boundary exists yet, add service-level evidence proving compact result and no retained local references; do not invent full job orchestration here.
- Do not add cleanup to canonical stage list or benchmark timer comparison.
- Add repeated-run smoke: same heavy no-risk request at least 3 times in one worker/process lifecycle where feasible.
- Record:
  - cleanup duration;
  - `rss_before`;
  - `rss_peak`;
  - `rss_after_cleanup`;
  - `retained_rss_delta`;
  - repeated run count;
  - monotonic retained RSS growth boolean;
  - worker recycle if applicable.

## Requirements (Should)

- Use `try/finally` around scoring path where orchestration exists.
- Prefer deleting strong references and clearing containers over relying on `gc.collect()`.
- Use `gc.collect()` only as fallback evidence after references are removed.

## Requirements (Nice-to-have)

- Add a small reference-retention unit test for result DTOs if reliable.

# Context acquisition protocol

Read only in order: repo contract, runtime doc, manifest, service files, conditional orchestration if touched. Do not broad-read worker code unless needed.

Pre-implementation reading target:

- `<= 8 files`
- `<= ~35k-50k tokens`

Expand only for worker boundary ambiguity or memory evidence blocker.

# Reading manifest

Use front-matter `context_sources`; do not duplicate as broad reading list.

# Work plan (agent should follow)

Skill routing:

- `backend-performance-evidence`: use for memory evidence and RSS reporting.
- `production-risk-review`: use before push if touching worker lifecycle/recycle.
- `backend-quality-gates`: use for local gates.

1. Identify where per-job heavy references live.
2. Ensure result DTOs remain compact.
3. Add cleanup boundary or service-level cleanup evidence without adding canonical stage.
4. Add local tests/smoke.
5. Run local gates.
6. Run Mac Studio memory evidence pipeline.

Mac Studio evidence pipeline:

1. Commit local changes after gates pass.
2. Push branch/commit to remote.
3. SSH to `macstudio`.
4. In `/opt/roehub/app`, fetch and pull the pushed commit.
5. Verify commit SHA.
6. Run repeated-run memory smoke for no-risk workload.
7. Save evidence under `docs/architecture/backtest/benchmark_iterations/YYYY-MM-DD_iteration_4_7_memory_cleanup/`.
8. Mark evidence as service hygiene, not canonical `>=90%` benchmark acceptance.

# Acceptance criteria (Definition of Done)

- Result object does not retain heavy per-job arrays/contexts.
- Repeated-run memory smoke exists or final report explains why current architecture lacks worker lifecycle to test.
- No monotonic retained RSS growth, or worker recycle is proven before next heavy job.
- No canonical benchmark stage order changed.
- Local gates pass.

# Implementation constraints

## Determinism & ordering

- Cleanup must not change result ordering or metrics.

## API / contracts

- Public API: none.
- DTO schema: no breaking change.
- Persistence: none.
- Config: compatible-change only if adding recycle threshold.
- Benchmark gate: compatible service hygiene evidence only.

## Benchmark / memory

- Do not claim `>=90%` target for cleanup; no notebook baseline exists.
- Report RSS limitations on macOS allocator explicitly.

# Files to indicate (expected touched areas)

- `src/trading/contexts/backtest/application/services/v2/no_risk_exact.py`
- related tests
- orchestration file only if existing boundary requires it
- memory evidence folder if run

# Non-goals

- scoring changes;
- benchmark runner accounting;
- persistence/API endpoints;
- lazy trades;
- canonical stage additions.

# Quality gates (must run and pass)

- `uv run ruff check src/trading/contexts/backtest/application/services/v2 tests/unit/contexts/backtest/application/services/v2`
- `uv run pyright`
- `uv run pytest -q tests/unit/contexts/backtest/application/services/v2`
- Mac Studio repeated-run memory evidence if feasible.

# Final output: report format (strict)

Report in Russian:

1. Cleanup boundary / retained-reference changes.
2. Memory evidence path and key numbers.
3. Local checks.
4. Contract impact classification.
5. Known limitations.
