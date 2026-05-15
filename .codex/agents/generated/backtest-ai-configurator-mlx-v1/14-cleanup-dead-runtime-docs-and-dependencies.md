---
prompt_name: backtest_ai_configurator_lmstudio_v1_14_cleanup_docs_dead_runtime
repo: roehub.com
branch: main
scope: "Iteration 14: remove stale MLX runtime code/docs/process assumptions, normalize docs/dependencies/config, and leave only the accepted LM Studio path as current architecture."

language:
  implementation: python_docs_ops_cleanup
  agent_report: ru

context_sources:
  always_read:
    - path: .codex/AGENTS.md
      why: "cleanup and contract rules"
    - path: docs/architecture/backtest/benchmark_iterations/2026-05-13_lmstudio_serving_recovery/security_pipeline_readiness.md
      why: "required previous readiness evidence"
    - path: docs/architecture/backtest/backtest-ai-configurator-mlx-v1.md
      why: "main stale architecture doc"
  task_entrypoints:
    - path: src/trading/contexts/backtest/adapters/outbound/llm
      why: "runtime adapter code"
      inspect_symbols:
        - "*"
    - path: configs/prod/backtest_ai_configurator.yaml
      why: "runtime and queue defaults"
      inspect_symbols:
        - backtest_ai_configurator
    - path: docs/architecture/backtest/benchmark_iterations/2026-05-12_iteration_08_ai_configurator_load_security
      why: "failed evidence to mark historical"
      inspect_symbols:
        - "*"
    - path: infra
      why: "launchd, Monit, Prometheus references"
      inspect_symbols:
        - "*backtest*ai*"
  conditional_bundles:
    tests:
      read_when: "when deleting or renaming runtime code"
      paths:
        - tests/unit/contexts/backtest/application/ai_configurator
        - tests/unit/apps/worker/test_backtest_ai_configurator_worker.py
    docs_index:
      read_when: "when docs references move or are removed"
      paths:
        - docs/architecture/README.md
        - docs/architecture/backtest/README.md
        - docs/runbooks
    dependency_inventory:
      read_when: "always before final cleanup verdict, and whenever imports or runtime packages changed"
      paths:
        - pyproject.toml
        - uv.lock
  consult_if_needed:
    - path: pyproject.toml
      read_when: "if dependency inventory shows an ambiguous dependency owner"

style_references:
  - docs/architecture/backtest/benchmark_iterations/2026-05-13_lmstudio_serving_recovery/
  - docs/runbooks/mac-studio-native-backend-operations.md

documentation_continuity:
  old_current_docs:
    - docs/architecture/backtest/backtest-ai-configurator-mlx-v1.md
    - docs/architecture/backtest/benchmark_iterations/2026-05-12_iteration_08_ai_configurator_load_security/README.md
  new_doc_artifact: docs/architecture/backtest/benchmark_iterations/2026-05-13_lmstudio_serving_recovery/cleanup_and_contract_sync.md
  canonical_shape: "cleanup report with current/historical/deleted classification"
  docs_gate: "uv run python -m tools.docs.generate_docs_index --check"

hard_requirements:
  depends_on_iteration_13_accepted: true
  remove_dead_mlx_runtime_code: true
  rewrite_stale_docs_required: true
  no_huge_unaccepted_limits: true
  no_dead_processes_required: true
  zero_current_stale_reference_gate_required: true
  dependency_inventory_required: true
  publish_ci_deploy_required: true
  macstudio_sync_required: true

task_toggles:
  cleanup_code: true
  cleanup_docs: true
  cleanup_config: true
  cleanup_ops_files: true
  run_macstudio_dead_process_check: true
  run_benchmark: false

skill_routing:
  - skill: production-risk-review
    use_when: "before deleting code, configs, docs or process files"
    timing: "before implementation"
    reason: "avoid deleting live contracts"
  - skill: contract-impact-analysis
    use_when: "classifying deleted/renamed runtime contracts"
    timing: "before implementation"
    reason: "compatibility and rollback"
  - skill: backend-quality-gates
    use_when: "running tests, lint and type checks"
    timing: "during verification"
    reason: "cleanup verification"
  - skill: publish-ci-deploy
    use_when: "after cleanup and real host checks pass"
    timing: "final delivery step"
    reason: "ship cleanup and sync Mac Studio"

target_envs:
  - local-dev
  - unit-tests
  - mac-studio-prod
  - github-actions

required_literals:
  - "runtime: lm_studio"
  - "LMStudioOpenAICompatibleAdapter"
  - "historical failure evidence"
  - "gemma-4-e2b-it-4bit"
  - "active_generations: 1"
  - "max_queue_size"
  - "mlx_lm.server"
  - "MLXOpenAICompatibleAdapter"
  - "mlx_lm_server"
  - "MLX generate"
  - "MLX repair"
  - "POST /v1/chat/completions"
  - "JSON Schema type values must be strings"
  - "do not use type: [\"string\", \"null\"]"
  - "accepted: true/false"
  - "blocking_reason"
  - "next_prompt_allowed"
  - "current_active_hit_count: 0"

non_goals:
  - "Do not edit old prompt files 01-09."
  - "Do not delete historical benchmark evidence."
  - "Do not hide failed Iteration 08 evidence."
  - "Do not run load benchmark."

final_report_format:
  language: ru
  sections:
    - "Что удалено/переписано"
    - "Текущий runtime contract"
    - "Документация и зависимости"
    - "Проверки"
    - "Доставка и Mac Studio"

quality_gates:
  - cmd: "uv run pytest -q tests/unit/contexts/backtest/application/ai_configurator tests/unit/apps/worker/test_backtest_ai_configurator_worker.py tests/unit/apps/api/test_backtest_ai_config_routes.py"
    expect: "passes"
  - cmd: "uv run ruff check apps/api apps/worker src/trading/contexts/backtest scripts tests/unit"
    expect: "passes"
  - cmd: "uv run pyright"
    expect: "passes"
  - cmd: "uv run python -m tools.docs.generate_docs_index --check"
    expect: "passes"
  - cmd: "git diff --check"
    expect: "passes"

expected_primary_touches:
  - src/trading/contexts/backtest/adapters/outbound/llm/
  - configs/prod/backtest_ai_configurator.yaml
  - docs/architecture/backtest/backtest-ai-configurator-mlx-v1.md
  - docs/architecture/backtest/benchmark_iterations/2026-05-12_iteration_08_ai_configurator_load_security/
  - docs/architecture/backtest/benchmark_iterations/2026-05-13_lmstudio_serving_recovery/
  - infra/

possible_secondary_touches:
  - pyproject.toml
  - docs/architecture/README.md
  - docs/architecture/backtest/README.md
  - docs/runbooks/
  - tests/unit/

safety_notes:
  - "Do not delete old prompts or historical evidence; classify them as historical."
  - "Do not remove code that is still imported by tests or runtime."
  - "Mac Studio process checks are evidence; do not kill unrelated services."
---

# Task

Clean up the dead/stale runtime path after LM Studio serving, adapter and pipeline readiness have been accepted.

This prompt starts only after Iteration 13 security and real pipeline readiness evidence exists and passed. The goal is to remove old assumptions and leave the codebase, docs, config, process files and evidence in a clean, current state before benchmark rerun.

Done means:

- production code/docs no longer describe `mlx_lm.server` as current target runtime;
- stale adapter names/files are removed or renamed unless they are historical evidence;
- current config uses accepted LM Studio runtime names and conservative limits;
- failed Iteration 08 evidence remains but is clearly historical/non-acceptance;
- no dead launchd/Monit process files remain for a rejected runtime path;
- docs/dependencies are updated and docs index passes.

## Context / Current State

Context ledger:

- completed:
  - LM Studio serving, adapter, service lifecycle and pipeline readiness should be accepted.
- open_items:
  - old docs still mention `mlx_lm.server` and `MLXOpenAICompatibleAdapter`.
  - production config may contain temporary huge queue/concurrency values from failed benchmark.
  - historical failed benchmark must stay visible but not treated as current target.
- contract_changes:
  - current runtime contract becomes LM Studio only for MVP.
- risks:
  - deleting useful audit/evidence;
  - leaving dead names that mislead future agents;
  - changing external API behavior during cleanup.
- next_focus:
  - make the repository clean enough for benchmark acceptance.

## Requirements (Must)

- Stop if Iteration 13 evidence is missing or blocked.
- Search current production code/docs/config/ops files for stale `mlx_lm.server`, `MLXOpenAICompatibleAdapter`, `mlx_lm_server`, and rejected-runtime process references.
- Run the exact zero-current-stale-reference check and classify every hit:
  - `rg -n "mlx_lm\\.server|MLXOpenAICompatibleAdapter|mlx_lm_server|MLX generate|MLX repair" src apps configs infra scripts docs tests`
  - allowed hits: old prompts are outside this command; historical failed evidence is allowed only when classified as historical/non-current;
  - acceptance fails if any current production code, current config, current ops file, current runbook, or current architecture doc still presents the rejected MLX serving path as active.
  - evidence must report `current_active_hit_count: 0`.
- Do not edit old prompt files 01-09; they are historical prompt artifacts.
- Do not delete historical benchmark evidence; mark it as failed/non-acceptance if needed.
- Remove dead code only after proving it is not imported or needed.
- Replace temporary giant queue/concurrency literals with conservative accepted/internal defaults, or keep feature disabled and document why values are not public defaults.
- Update docs and dependencies so current instructions point to LM Studio and
  preserve the accepted API rule: `POST /v1/chat/completions`,
  `response_format.type=json_schema`, prompt text in `messages[].content`,
  parse `choices[0].message.content` as JSON, and do not emit JSON Schema
  nullable union `type: ["string", "null"]`.
- Produce dependency inventory: inspect `pyproject.toml`, `uv.lock`, and current imports; remove dependencies/imports added only for the failed `mlx_lm.server` path if they are unused, or document why each retained dependency is still needed.
- Verify Mac Studio has no monitored/running stale `mlx_lm.server` service for this feature.
- Markdown and JSON evidence must include explicit machine-readable gate fields: `accepted: true/false`, `blocking_reason: null|string`, and `next_prompt_allowed: true/false`.
- Run `publish-ci-deploy` after cleanup gates pass.

## Requirements (Should)

- Produce a classification table: current / historical / deleted / intentionally retained.
- Keep cleanup diffs narrow and evidence-backed.
- Prefer renaming tests to LM Studio rather than leaving misleading MLX names.
- Keep public API compatibility unchanged.

## Requirements (Nice-to-have)

- Add a grep-based cleanup check script or documented command.
- Add a short rollback note: restore previous commit, disable worker, stop LM Studio runtime.

# Context acquisition protocol

Read only in this order and stop once sufficient:

1. `.codex/AGENTS.md`
2. Iteration 13 readiness evidence
3. task entrypoints
4. tests bundle only for renamed/deleted code
5. docs index bundle only when docs references change
6. dependency file only if imports/dependencies change

Do not scan the whole repo unless targeted `rg` finds stale references outside expected areas.

Reading budget: max 12 files plus targeted `rg` output.

# Reading manifest

- `always_read`: repo contract, readiness evidence, main architecture doc.
- `task_entrypoints`: runtime adapters, prod config, failed evidence, infra references.
- `conditional_bundles`: tests, docs index, runbooks only when touched.
- `consult_if_needed`: dependency file only if imports/deps change.

Stop reading once stale-current references are classified and cleanup scope is bounded.

# Work plan (agent should follow)

1. Verify Iteration 13 evidence.
2. Run targeted `rg` for stale runtime names in current code/docs/config/infra.
3. Classify each hit as current, historical, old prompt, or delete/rename.
4. Remove/rename dead code and stale docs/config/process references.
5. Update docs/evidence with current LM Studio contract.
6. Run local gates and docs index.
7. Verify Mac Studio has no stale rejected runtime process/Monit service.
8. Use `publish-ci-deploy`.

# Acceptance criteria (Definition of Done)

- No current production code imports stale MLX adapter names.
- Zero-current-stale-reference gate passes: every `rg` hit is classified, and there are no current active references to `mlx_lm.server`, `MLXOpenAICompatibleAdapter`, `mlx_lm_server`, `MLX generate`, or `MLX repair` outside historical/non-current evidence.
- No current architecture/runbook text instructs use of `mlx_lm.server` for this MVP runtime.
- Old prompts and historical failed evidence remain untouched or explicitly historical.
- Dependency inventory exists and confirms `pyproject.toml`, `uv.lock`, and imports contain no unused dependency from the rejected MLX-serving implementation.
- Config limits are conservative or documented as internal-only/disabled.
- Mac Studio check finds no stale rejected runtime service for this feature.
- Evidence contains top-level gate markers: `accepted`, `blocking_reason`, and `next_prompt_allowed`; downstream prompts may proceed only when `accepted=true` and `next_prompt_allowed=true`.
- `publish-ci-deploy` reaches `deployed`, or exact `green-pr`/`blocked` state is recorded.

# Implementation constraints

## Cleanup boundaries

- Do not change `/backtests/jobs` behavior.
- Do not alter public API response shape except internal diagnostic fields already hidden from users.
- Do not delete training/audit data.

## Documentation

- Update old/current docs and create `cleanup_and_contract_sync.md`.
- Run docs index check.

# Files to indicate (expected touched areas)

Expected primary touches:

- `src/trading/contexts/backtest/adapters/outbound/llm/`
- `configs/prod/backtest_ai_configurator.yaml`
- `docs/architecture/backtest/backtest-ai-configurator-mlx-v1.md`
- `docs/architecture/backtest/benchmark_iterations/2026-05-13_lmstudio_serving_recovery/cleanup_and_contract_sync.md`
- `infra/`

Possible secondary touches:

- `tests/unit/contexts/backtest/application/ai_configurator/`
- `tests/unit/apps/worker/test_backtest_ai_configurator_worker.py`
- `docs/runbooks/`
- `pyproject.toml`

# Non-goals

- No benchmark run.
- No UI redesign.
- No paid rollout.
- No deletion of historical evidence or old prompts.

# Quality gates (must run and pass)

- `uv run pytest -q tests/unit/contexts/backtest/application/ai_configurator tests/unit/apps/worker/test_backtest_ai_configurator_worker.py tests/unit/apps/api/test_backtest_ai_config_routes.py`
- `uv run ruff check apps/api apps/worker src/trading/contexts/backtest scripts tests/unit`
- `uv run pyright`
- `uv run python -m tools.docs.generate_docs_index --check`
- `git diff --check`
- `sh -lc 'rg -n "mlx_lm\\.server|MLXOpenAICompatibleAdapter|mlx_lm_server|MLX generate|MLX repair" src apps configs infra scripts docs tests || true'` with every hit classified and zero current-active hits. A no-hit `rg` exit code is acceptable only when the evidence records `current_active_hit_count: 0`.
- Mac Studio stale-process check for rejected runtime path and LM Studio runtime ready check.

If cleanup exposes a live dependency on stale code, stop and report blocker instead of deleting broadly.

# Final output: report format (strict)

Report in Russian with:

- `Что удалено/переписано`: files/symbols/docs.
- `Текущий runtime contract`: accepted LM Studio path and limits.
- `Документация и зависимости`: old docs updated, new cleanup evidence.
- `Zero-current-stale-reference gate`: exact `rg` output classification and current-active hit count.
- `Проверки`: commands and results.
- `Доставка и Mac Studio`: publish-ci-deploy state and Mac Studio checks.
