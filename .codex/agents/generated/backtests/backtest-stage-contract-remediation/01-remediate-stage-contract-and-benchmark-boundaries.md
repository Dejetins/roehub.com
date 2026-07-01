---
prompt_name: backtest-stage-contract-and-iteration-2-benchmark-remediation
repo: roehub.com
branch: main
scope: "Доработать stage contract backtest runtime, разделить notebook-compatible compute stages и service overhead, затем исправить Iteration 2 prepare_pools implementation/benchmark boundaries."

language:
  implementation: python
  agent_report: ru

context_sources:
  always_read:
    - path: .codex/AGENTS.md
      why: "repo contract, safety invariants, verification rules"
    - path: .codex/agents/.context/promt_manager_state.yaml
      why: "compact coordination state, if relevant to current branch"
    - path: docs/architecture/backtest/backtest-service-artifact-runtime-v1.ru.md
      why: "canonical v1 target document that must be corrected before further iterations"
    - path: docs/architecture/backtest/benchmark_iterations/2026-04-26_iteration_2_prepare_pools/benchmark_summary.md
      why: "latest real Mac Studio Iteration 2 failure evidence"

  task_entrypoints:
    - path: docs/architecture/backtest/backtest-service-artifact-runtime-v1.ru.md
      why: "stage contract, benchmark policy, and iteration plan to update"
      inspect_symbols:
        - "Измеряемая стадия бенчмарка: `prepare_pools`"
        - "Итерация 2: artifact arrays и `prepare_pools`"
        - "Политика бенчмарков"
        - "Гейт бенчмарка"
    - path: src/trading/contexts/backtest/application/services/v2/prepare_pools.py
      why: "current service implementation mixes artifact context, mmap loading, slicing, and notebook-core prepare_pools timing"
      inspect_symbols:
        - "BacktestPreparePoolsService.execute"
        - "prepare_indicator_pools"
        - "prepare_indicator_pool"
        - "prefilter_indicator_rows"
        - "build_signal_segments"
    - path: src/trading/contexts/backtest/adapters/outbound/artifacts_fs/artifact_array_loader.py
      why: "current filesystem loader resolves context, hashes manifest, opens mmap arrays, and loads signal manifests"
      inspect_symbols:
        - "FilesystemBacktestArtifactArrayLoader.resolve_context"
        - "load_price_arrays"
        - "load_mapping_arrays"
        - "load_signal_matrix"
        - "_load_npy_mmap"
    - path: docs/architecture/backtest/benchmark_iterations/2026-04-26_iteration_2_prepare_pools/benchmark_results.json
      why: "machine-readable timing subsegments and pass/fail evidence"

  conditional_bundles:
    notebook_contract:
      read_when: "when validating exact baseline scope and method boundaries"
      paths:
        - tests/notebook_tests/engine_test/btcusdt_15m_research_engine.ipynb
        - docs/architecture/backtest/benchmark_iterations/2026-04-26_engine_test_btcusdt_15m/benchmark_results.json
        - docs/architecture/backtest/benchmark_iterations/2026-04-26_engine_test_btcusdt_15m/benchmark_summary.md

    iteration_1_runtime_context:
      read_when: "when deciding where artifact context resolution belongs after the split"
      paths:
        - src/trading/contexts/backtest/application/services/v2/preflight.py
        - src/trading/contexts/backtest/application/dto/runtime_preflight.py
        - src/trading/contexts/backtest/application/ports/artifact_context.py
        - src/trading/contexts/backtest/adapters/outbound/artifacts_fs/artifact_context_resolver.py

    dto_ports_and_exports:
      read_when: "when adding internal runtime DTOs, ports, cache handles, or timing structures"
      paths:
        - src/trading/contexts/backtest/application/dto/prepare_pools.py
        - src/trading/contexts/backtest/application/ports/artifact_arrays.py
        - src/trading/contexts/backtest/application/services/v2/__init__.py
        - src/trading/contexts/backtest/application/dto/__init__.py
        - src/trading/contexts/backtest/application/ports/__init__.py

    tests_and_benchmark_runner:
      read_when: "when updating tests or the Iteration 2 benchmark runner/evidence writer"
      paths:
        - tests/unit/contexts/backtest/application/services/v2/test_prepare_pools_service.py
        - tests/unit/contexts/backtest/adapters/outbound/artifacts_fs/test_artifact_array_loader.py
        - tests/unit/contexts/backtest/application/services/v2/artifact_testkit_v2.py
        - docs/architecture/backtest/benchmark_iterations/README.md

  consult_if_needed:
    - path: docs/architecture/backtest/deep-research-report_for_btcusdt_15m_research_engine.md
      read_when: "if notebook stage semantics are still ambiguous after inspecting named cells"
    - path: docs/architecture/backtest/backtest-service-artifact-runtime-v1.md
      read_when: "only if the English version must stay synchronized with the Russian canonical document"

style_references:
  - .codex/agents/generated/backtest-service-iteration-2/01-implement-prepare-pools.md
  - src/trading/contexts/backtest/application/services/v2/preflight.py
  - src/trading/contexts/backtest/application/services/v2/prepare_pools.py
  - src/trading/contexts/backtest/adapters/outbound/artifacts_fs/artifact_array_loader.py

hard_requirements:
  runtime_doc_ru_is_canonical: true
  fix_stage_contract_before_code_changes: true
  separate_notebook_comparable_compute_from_service_overhead: true
  no_public_api_breakage: true
  no_internal_api: true
  no_roadmap_docs_as_source_of_truth: true
  no_notebook_edits: true
  benchmark_acceptance_macstudio_only: true
  preserve_existing_user_changes: true

task_toggles:
  update_runtime_plan: true
  refactor_prepare_pools_stage_boundary: true
  add_or_update_timing_dtos: true
  update_benchmark_runner_or_evidence_shape: true
  add_targeted_tests: true
  optionally_optimize_segment_build_after_boundary_fix: true

skill_routing:
  - skill: architecture-review
    use_when: "before implementation, while updating the canonical runtime document and checking for contradictory stage wording"
    timing: before implementation
    reason: "owns architecture docs sync and source-of-truth consistency"
  - skill: contract-impact-analysis
    use_when: "when changing DTOs, ports, timing payloads, benchmark JSON shape, cache identity, or any API-visible field"
    timing: during investigation
    reason: "owns compatibility and boundary impact"
  - skill: backend-performance-evidence
    use_when: "when defining benchmark scopes, interpreting Iteration 2 failure evidence, or reporting performance claims"
    timing: before implementation and during verification
    reason: "owns baseline selection, comparable measurement, and evidence discipline"
  - skill: numba
    use_when: "only if optimizing @njit/prange kernels or segment-build code after the stage-boundary fix"
    timing: during implementation
    reason: "owns Numba-specific performance and typing concerns"
  - skill: backend-quality-gates
    use_when: "when running pytest, ruff, pyright, docs index, or classifying failures"
    timing: during verification
    reason: "owns backend quality gate execution and triage"
  - skill: root-cause-debugging
    use_when: "if a focused test, benchmark runner, or existing Iteration 1 behavior fails unexpectedly"
    timing: if blocker
    reason: "owns root-cause investigation before fixes"

target_envs:
  - local-dev
  - macstudio

required_literals:
  - "artifact_context_resolve"
  - "artifact_array_open"
  - "request_slice_prepare"
  - "prepare_pools_core"
  - "prepare_pools_total"
  - "artifact_manifest_load"
  - "artifact_array_mmap_load"
  - "time_range_slice"
  - "signal_row_selection"
  - "row_prefilter"
  - "segment_build"
  - "canonical notebook prepare_pools"
  - "notebook-compatible"
  - "service overhead"
  - "np.load(..., mmap_mode=\"r\")"
  - "[start, end)"

non_goals:
  - "Do not implement Iteration 3 combo planning, proxy contexts, exact scoring, heap/top-K, lazy trades, job workers, persistence, or UI."
  - "Do not change public API routes or add an internal API."
  - "Do not edit notebooks; use them as canonical reference evidence only."
  - "Do not rewrite legacy roadmap docs or use them as v1 implementation source."
  - "Do not mark Iteration 2 accepted unless the corrected comparable benchmark passes on Mac Studio."
  - "Do not hide the previous failed benchmark; preserve it as evidence of the bad stage boundary."

final_report_format:
  language: ru
  sections:
    - "Изменения"
    - "Stage Contract"
    - "Benchmark/Evidence"
    - "Проверки"
    - "Остаточные риски"

quality_gates:
  - cmd: "uv run pytest -q tests/unit/contexts/backtest/application/services/v2/test_prepare_pools_service.py tests/unit/contexts/backtest/adapters/outbound/artifacts_fs/test_artifact_array_loader.py"
    expect: "passes after updating targeted tests; if filenames differ, run the exact created/updated test files"
  - cmd: "uv run pytest -q tests/unit/contexts/backtest/application/services/v2 tests/unit/contexts/backtest/adapters/outbound/artifacts_fs"
    expect: "passes, or unrelated existing failures are explicitly classified"
  - cmd: "uv run ruff check src/trading/contexts/backtest tests/unit/contexts/backtest"
    expect: "passes"
  - cmd: "uv run pyright"
    expect: "passes, or unrelated existing failures are explicitly classified"
  - cmd: "uv run python -m tools.docs.generate_docs_index --check"
    expect: "passes if Markdown docs were touched"
  - cmd: "git diff --check"
    expect: "passes"

expected_primary_touches:
  - "docs/architecture/backtest/backtest-service-artifact-runtime-v1.ru.md"
  - "src/trading/contexts/backtest/application/services/v2/prepare_pools.py"
  - "src/trading/contexts/backtest/application/dto/prepare_pools.py"
  - "src/trading/contexts/backtest/adapters/outbound/artifacts_fs/artifact_array_loader.py"
  - "tests/unit/contexts/backtest/application/services/v2/test_prepare_pools_service.py"
  - "tests/unit/contexts/backtest/adapters/outbound/artifacts_fs/test_artifact_array_loader.py"
  - "docs/architecture/backtest/benchmark_iterations/2026-04-26_iteration_2_prepare_pools/"

possible_secondary_touches:
  - "docs/architecture/backtest/benchmark_iterations/README.md"
  - "docs/architecture/backtest/backtest-service-artifact-runtime-v1.md"
  - "src/trading/contexts/backtest/application/ports/artifact_arrays.py"
  - "src/trading/contexts/backtest/application/services/v2/__init__.py"
  - "src/trading/contexts/backtest/application/dto/__init__.py"
  - "src/trading/contexts/backtest/application/ports/__init__.py"
  - "tests/unit/contexts/backtest/application/services/v2/artifact_testkit_v2.py"

safety_notes:
  - "The previous Iteration 2 benchmark failure is real evidence. Do not delete or rewrite it as passing."
  - "The fix is not to lower the target. The fix is to compare equivalent scopes and separately measure service overhead."
  - "Keep artifact paths trusted-config/current-pointer/manifest based; never accept user-supplied filesystem paths."
  - "If not running on Mac Studio, record only local developer evidence and state that acceptance was not run."
---

# Task

Доработать stage contract и Iteration 2 runtime так, чтобы benchmark pipeline
сравнивал только сопоставимые участки алгоритма и больше не смешивал
notebook-core compute target с service overhead.

Текущий Mac Studio benchmark Iteration 2 показал `0 / 28` pass, но разбор
показал primary root cause: service-level `prepare_pools` включает
`artifact_manifest_load`, повторное mmap opening и request slicing, тогда как
canonical notebook `prepare_pools` timer замеряет уже прогретые/opened arrays и
оборачивает только `prepare_indicator_pools(...)`.

Done means:

- `docs/architecture/backtest/backtest-service-artifact-runtime-v1.ru.md`
  явно разделяет notebook-compatible stages и service overhead stages;
- Iteration 2 benchmark gate больше не требует сравнивать manifest/mmap/slice
  overhead с canonical notebook `prepare_pools`;
- runtime code exposes a notebook-compatible `prepare_pools_core` timing scope;
- artifact context resolution, array opening, and request slicing are measured
  separately and remain visible in evidence;
- tests prevent future regressions where unrelated service overhead is silently
  folded into a notebook-comparable stage;
- previous failed benchmark remains preserved as evidence and is described as a
  stage-boundary failure, not as an algorithmic correctness failure.

## Context / Current State

Context ledger:

- completed:
  - Iteration 0/1 are complete.
  - Iteration 2 prepare-pools implementation exists.
  - Real Mac Studio benchmark exists at
    `docs/architecture/backtest/benchmark_iterations/2026-04-26_iteration_2_prepare_pools/`.
  - Canonical notebook baseline remains
    `tests/notebook_tests/engine_test/btcusdt_15m_research_engine.ipynb`.
- open_items:
  - strict service-level `prepare_pools` benchmark failed `0 / 28`;
  - current runtime document says `prepare_pools` includes artifact root resolve,
    manifest reads, mmap loading, slicing, row extraction, prefilter, and segment build;
  - current benchmark compares that strict service scope to canonical notebook
    `prepare_pools`, which is not the same scope;
  - core-only service subsegments are much closer to target and passed `18 / 28`
    tuple diagnostics.
- measured failure facts:
  - actual service `prepare_pools` median: about `0.08177s`;
  - accepted service median target: about `0.00716s`;
  - median `artifact_manifest_load`: about `0.0668s`;
  - median `artifact_array_mmap_load`: about `0.0067s`;
  - `artifact_manifest_load` dominates the strict service timer.
- contract_changes:
  - define `prepare_pools_core` as the notebook-compatible target scope;
  - define `artifact_context_resolve`, `artifact_array_open`, and
    `request_slice_prepare` as service overhead/preparation scopes measured
    separately;
  - define `prepare_pools_total` only as an aggregate service telemetry value,
    not the value compared directly to canonical notebook `prepare_pools`.
- risks:
  - if the document is not corrected first, future iterations may keep comparing
    non-equivalent stages and reject valid implementations;
  - changing timing DTO shape may affect tests or future benchmark JSON consumers;
  - caching/pinning artifact arrays must preserve artifact identity and immutable
    prefix assumptions.
- next_focus:
  - update stage contract;
  - refactor measured boundaries;
  - update tests/benchmark evidence shape;
  - rerun local gates and prepare Mac Studio acceptance instructions if not on
    Mac Studio.

Additional stable facts:

- Notebook opens price/signal/mapping arrays before the timed prepare-pools call.
- Notebook computes `time_selector_15m`, `signal_returns_15m`, and execution
  mapping before the timed prepare-pools call.
- Notebook `prepare_pools` timer wraps `prepare_indicator_pools(...)`.
- Service currently calls `resolve_context(...)` and opens arrays inside
  `BacktestPreparePoolsService.execute(...)`.

## Requirements (Must)

- Read context using the protocol below and stop early once sufficient.
- Update `backtest-service-artifact-runtime-v1.ru.md` before implementation so
  the stage contract is unambiguous.
- Preserve the previous failed Iteration 2 benchmark files as historical
  evidence. Do not rewrite them into a pass.
- Make benchmark terminology explicit:
  - `artifact_context_resolve`: current pointer / slot manifest identity /
    manifest hash validation and typed context creation;
  - `artifact_array_open`: mmap handle opening and manifest-backed dtype/shape
    validation for prices, mappings, and signals;
  - `request_slice_prepare`: `[start, end)` 15m slicing, returns, and execution
    mapping derivation;
  - `prepare_pools_core`: signal row selection/extraction, row prefilter, and
    compressed segment build, equivalent to notebook `prepare_indicator_pools`;
  - `prepare_pools_total`: aggregate telemetry only.
- Update the Iteration 2 benchmark gate so the 90% canonical notebook ratio
  applies to `prepare_pools_core`, not to `prepare_pools_total`.
- Keep `artifact_context_resolve`, `artifact_array_open`, and
  `request_slice_prepare` measured and reported separately with CPU/memory
  evidence, but do not compare them to canonical notebook `prepare_pools`.
- Refactor code so the notebook-compatible prepare-pools path can be called with
  already resolved/opened runtime inputs.
- Keep a compatibility facade if needed, but its timing must make scope explicit.
- Add or update tests that fail if context resolve or mmap opening is counted as
  `prepare_pools_core`.
- Preserve row metadata/order hash semantics and indicator pool correctness.
- Run targeted quality gates and report exact commands/results.

## Requirements (Should)

- Prefer a small internal DTO/service split over broad rewrites.
- Cache or pin runtime artifact context/array handles at job/runtime scope when
  doing so is compatible with existing ports and tests.
- If practical after the stage-boundary fix, remove the redundant
  `change_count` scan in `build_signal_segments` by reusing the value already
  computed by `fused_row_prefilter_stats`.
- Keep benchmark JSON backwards-readable where possible by adding explicit new
  fields rather than removing old fields without a migration note.
- Add a short note in the benchmark iteration summary explaining why the old
  strict result failed and what the corrected measurement boundary is.

## Requirements (Nice-to-have)

- Add a small helper that computes notebook-compatible core time from
  subsegments for diagnostic comparison.
- Add comments only where they clarify the stage boundary and prevent future
  benchmark misuse.

# Context acquisition protocol

Read only in this order and stop once sufficient:

1. `.codex/AGENTS.md`
2. `.codex/agents/.context/promt_manager_state.yaml`
3. `docs/architecture/backtest/backtest-service-artifact-runtime-v1.ru.md`
4. latest failed Iteration 2 benchmark summary/results
5. task entrypoints
6. conditional bundles only if the touched contract or failing checks require them
7. consult-if-needed references only for blockers, ambiguity, or conflicts

Do not eagerly preload all listed sources.

Pre-implementation reading target:

- `<= 8 files`
- `<= ~35k-50k tokens`

Stop reading once all of the following are true:

- the corrected stage contract is clear;
- touched files are bounded;
- acceptance criteria are implementable without ambiguity;
- no unresolved public API or persistence-contract ambiguity remains.

Expand context only for:

- blockers;
- failing quality gates;
- benchmark threshold conflicts;
- timing/DTO compatibility questions;
- uncertainty about notebook timer scope.

# Reading manifest

Use the front-matter `context_sources` as the canonical reading map.

Read with this intent:

- `always_read`: repo rules, current canonical runtime doc, latest failed
  benchmark evidence;
- `task_entrypoints`: exact document/code locations that caused the mismatch;
- `conditional_bundles`: load only when needed for tests, DTOs, ports, or
  notebook semantics;
- `consult_if_needed`: load only for ambiguity or blocker resolution.

Do not convert this manifest into a broad mandatory reading list.

# Work plan (agent should follow)

Skill routing for this task:

- `architecture-review`: use before implementation to update and self-check the
  canonical runtime document; owns docs consistency.
- `contract-impact-analysis`: use when modifying timing DTOs, ports, benchmark
  JSON shape, or any compatibility-sensitive boundary.
- `backend-performance-evidence`: use before implementation to restate the
  baseline/scope, and during verification to report comparable measurements.
- `numba`: use only if touching JIT kernels or segment-build hot path.
- `backend-quality-gates`: use during verification.
- `root-cause-debugging`: use if unexpected failures appear.

1. Re-read the failed benchmark evidence and restate the root cause in one
   sentence before editing code.
2. Update `backtest-service-artifact-runtime-v1.ru.md`:
   - replace the current broad `prepare_pools` definition with explicit
     sub-stages;
   - state that notebook 90% ratio applies only to `prepare_pools_core`;
   - state that service overhead is measured separately and may have its own
     absolute budget later;
   - update Iteration 2 benchmark gate accordingly.
3. Inspect current `prepare_pools.py` and `artifact_array_loader.py` call graph.
4. Design the smallest internal split that allows:
   - resolving/pinning artifact context once;
   - opening/caching mmap handles outside core;
   - computing request slice outside core;
   - calling `prepare_pools_core` with already prepared runtime inputs.
5. Implement the split with clear timing fields.
6. Update tests:
   - core timing excludes manifest/mmap/slice overhead;
   - total timing still includes aggregate telemetry when using the facade;
   - row metadata/order hash and pool shapes remain stable;
   - loader behavior remains strict about dtype/shape/manifest identity.
7. Update benchmark evidence writer/summary shape if present in repo. If the
   actual runner is outside repo, document exactly what must change.
8. Optionally optimize redundant segment build scan only after the corrected
   stage-boundary tests pass.
9. Run quality gates.
10. Final report must distinguish:
    - implemented facts;
    - performance evidence;
    - Mac Studio acceptance status;
    - any remaining unverified assumptions.

# Acceptance criteria (Definition of Done)

- The canonical Russian runtime document no longer contradicts itself about
  what `prepare_pools` means.
- The document says explicitly that `prepare_pools_core` is the
  notebook-compatible comparison target.
- Code exposes a clear core path whose timer excludes artifact context resolve,
  mmap opening, and request slicing.
- Service overhead is still measurable and visible.
- Targeted tests cover the boundary split.
- Existing Iteration 1 behavior is not broken.
- If Mac Studio benchmark is run, a new/updated evidence record clearly reports:
  - `prepare_pools_core`;
  - `artifact_context_resolve`;
  - `artifact_array_open`;
  - `request_slice_prepare`;
  - `prepare_pools_total`;
  - pass/fail based on core ratio only.
- If Mac Studio benchmark is not run, final report says acceptance benchmark was
  not run and gives the exact command/instructions needed.

# Implementation constraints

- Use `apply_patch` for manual edits.
- Do not revert user changes.
- Keep diffs narrow and local to the stage contract, prepare-pools boundary,
  artifact array loading/timing DTOs, tests, and benchmark docs.
- Do not edit notebooks.
- Do not add dependencies unless absolutely required and justified.
- Do not silently change public API contracts.
- Do not change database schema.
- Do not hide or delete failing evidence from
  `2026-04-26_iteration_2_prepare_pools`.

# Files to indicate (expected touched areas)

Expected primary touches:

- `docs/architecture/backtest/backtest-service-artifact-runtime-v1.ru.md`
- `src/trading/contexts/backtest/application/services/v2/prepare_pools.py`
- `src/trading/contexts/backtest/application/dto/prepare_pools.py`
- `src/trading/contexts/backtest/adapters/outbound/artifacts_fs/artifact_array_loader.py`
- `tests/unit/contexts/backtest/application/services/v2/test_prepare_pools_service.py`
- `tests/unit/contexts/backtest/adapters/outbound/artifacts_fs/test_artifact_array_loader.py`

Possible secondary touches:

- `docs/architecture/backtest/benchmark_iterations/README.md`
- `docs/architecture/backtest/backtest-service-artifact-runtime-v1.md`
- `src/trading/contexts/backtest/application/ports/artifact_arrays.py`
- `src/trading/contexts/backtest/application/services/v2/__init__.py`
- `src/trading/contexts/backtest/application/dto/__init__.py`
- `src/trading/contexts/backtest/application/ports/__init__.py`
- `tests/unit/contexts/backtest/application/services/v2/artifact_testkit_v2.py`
- `docs/architecture/backtest/benchmark_iterations/2026-04-26_iteration_2_prepare_pools/benchmark_summary.md`

# Non-goals

- Iteration 3+ compute implementation.
- Exact scoring, combo planning, proxy filtering, heap/top-K, lazy trades.
- Public API route changes.
- Internal API.
- DB migrations.
- UI.
- Notebook edits.
- Roadmap cleanup.

# Quality gates (must run and pass)

Run the most focused gates first:

```bash
uv run pytest -q tests/unit/contexts/backtest/application/services/v2/test_prepare_pools_service.py tests/unit/contexts/backtest/adapters/outbound/artifacts_fs/test_artifact_array_loader.py
```

Then run the broader impacted backend gates:

```bash
uv run pytest -q tests/unit/contexts/backtest/application/services/v2 tests/unit/contexts/backtest/adapters/outbound/artifacts_fs
uv run ruff check src/trading/contexts/backtest tests/unit/contexts/backtest
uv run pyright
uv run python -m tools.docs.generate_docs_index --check
git diff --check
```

If any gate fails:

- classify whether it is introduced or pre-existing;
- do not ignore introduced failures;
- include exact failing command and root-cause summary in the final report.

# Final output: report format (strict)

Write the final report in Russian with these sections:

## Изменения

- concise list of code/doc changes;
- mention exact files changed.

## Stage Contract

- state the final stage split;
- state which metric is notebook-comparable;
- state which metrics are service overhead.

## Benchmark/Evidence

- report whether Mac Studio benchmark was run;
- if run, include pass/fail and key timing numbers;
- if not run, provide exact acceptance command/instruction and say it remains pending.

## Проверки

- list commands run and outcomes;
- classify any failures.

## Остаточные риски

- list only real remaining risks or say none known.
