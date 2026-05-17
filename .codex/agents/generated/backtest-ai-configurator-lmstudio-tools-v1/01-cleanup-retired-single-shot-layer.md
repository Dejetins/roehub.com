---
prompt_name: backtest_ai_configurator_lmstudio_tools_v1_01_cleanup_retired_single_shot_layer
repo: roehub.com
branch: main
scope: "Inventory and surgically clean any remaining retired single-shot /backtests AI Configurator runtime and prompt contracts while preserving the API/storage/validator foundation."

language:
  implementation: python_docs_cleanup
  agent_report: ru

context_sources:
  always_read:
    - path: .codex/AGENTS.md
      why: "repo contract and safety rules"
    - path: docs/architecture/backtest/backtest-ai-configurator-mlx-v1.md
      why: "current reset document and retained foundation"
    - path: docs/architecture/backtest/benchmark_iterations/2026-05-13_lmstudio_serving_recovery/single_shot_contract_retirement.md
      why: "retirement evidence and cleanup boundary"
  task_entrypoints:
    - path: src/trading/contexts/backtest/application/ai_configurator
      why: "current AI Configurator application boundary"
      inspect_symbols:
        - "*"
    - path: src/trading/contexts/backtest/adapters/outbound
      why: "runtime adapters and config loading boundary"
      inspect_symbols:
        - "*"
    - path: apps/api/wiring/modules/backtest.py
      why: "API composition root"
      inspect_symbols:
        - build_backtest_use_cases
    - path: apps/worker/backtest_ai_configurator
      why: "worker composition and readiness behavior"
      inspect_symbols:
        - "*"
  conditional_bundles:
    docs_cleanup:
      read_when: "when stale single-shot wording is found in docs or runbooks"
      paths:
        - docs/architecture/backtest/benchmark_iterations/2026-05-13_lmstudio_serving_recovery/README.md
        - docs/runbooks/mac-studio-native-backend-operations.md
        - docs/runbooks/mac-studio-monitoring-plan.md
    prompt_tombstones:
      read_when: "when old generated prompts still look executable"
      paths:
        - .codex/agents/generated/backtest-ai-configurator-mlx-v1
  consult_if_needed:
    - path: tests/unit/contexts/backtest/application/ai_configurator
      read_when: "if cleanup changes break focused tests"
    - path: tests/unit/apps/api/test_backtest_ai_config_routes.py
      read_when: "if API wiring behavior is affected"

documentation_continuity:
  old_current_docs:
    - docs/architecture/backtest/backtest-ai-configurator-mlx-v1.md
    - docs/architecture/backtest/benchmark_iterations/2026-05-13_lmstudio_serving_recovery/single_shot_contract_retirement.md
  new_doc_artifact: docs/architecture/backtest/benchmark_iterations/2026-05-17_lmstudio_tools_contract/cleanup_readiness.md
  canonical_shape: "benchmark iteration Markdown + matching JSON gate markers"
  docs_gate: "uv run python -m tools.docs.generate_docs_index --check"

hard_requirements:
  surgical_cleanup_only: true
  preserve_api_storage_validator_base: true
  feature_disabled_until_tool_agent: true
  no_commit_revert_range: true
  no_model_runtime_acceptance: true

task_toggles:
  inventory_wrong_layer: true
  delete_or_replace_wrong_contracts: true
  update_docs_retired_status: true
  leave_runtime_blocked: true

skill_routing:
  - skill: production-risk-review
    use_when: "before deleting or replacing runtime contracts"
    timing: "during investigation"
    reason: "avoid removing retained API/storage/validator foundation"
  - skill: contract-impact-analysis
    use_when: "when changing ports, config runtime literals, API DTOs, or docs that describe current behavior"
    timing: "before implementation and final report"
    reason: "classify cleanup as intentional runtime contract break while preserving public shell"
  - skill: backend-quality-gates
    use_when: "running focused and broad Python gates"
    timing: "during verification"
    reason: "triage cleanup fallout"

target_envs:
  - local-dev
  - github-actions
  - mac-studio-prod

required_literals:
  - "single-shot prompt/blob contract retired"
  - "BacktestConfigLLMGateway"
  - "LMStudioOpenAICompatibleAdapter"
  - "TRUSTED_CAPABILITIES"
  - "runtime: lm_studio_tools"
  - "tool_agent_pending"
  - "accepted: false"
  - "blocking_reason"
  - "next_prompt_allowed"

non_goals:
  - "Do not implement the new tool-agent pipeline in this prompt."
  - "Do not roll back unrelated commits."
  - "Do not delete validator, catalog resolver, artifact coverage, indicator bounds, quota, storage, status, or SSE shell."
  - "Do not run Mac Studio load benchmark."

final_report_format:
  language: ru
  sections:
    - "Что очищено"
    - "Что сохранено"
    - "Runtime state"
    - "Проверки"
    - "Следующий prompt"

quality_gates:
  - cmd: "uv run pytest -q tests/unit/scripts/test_backtest_ai_config_load_harness.py tests/unit/contexts/backtest/application/ai_configurator tests/unit/apps/api/test_backtest_ai_config_routes.py tests/unit/apps/worker/test_backtest_ai_configurator_worker.py"
    expect: "passes"
  - cmd: "uv run ruff check scripts apps/worker src/trading/contexts/backtest tests/unit"
    expect: "passes"
  - cmd: "uv run pyright"
    expect: "passes"
  - cmd: "uv run python -m tools.docs.generate_docs_index --check"
    expect: "passes"
  - cmd: "git diff --check"
    expect: "passes"

expected_primary_touches:
  - src/trading/contexts/backtest/application/ai_configurator
  - src/trading/contexts/backtest/adapters/outbound
  - apps/api/wiring/modules/backtest.py
  - apps/worker/backtest_ai_configurator
  - docs/architecture/backtest/backtest-ai-configurator-mlx-v1.md
  - docs/architecture/backtest/benchmark_iterations/2026-05-17_lmstudio_tools_contract/cleanup_readiness.md
  - docs/architecture/backtest/benchmark_iterations/2026-05-17_lmstudio_tools_contract/cleanup_readiness.json

possible_secondary_touches:
  - .codex/agents/generated/backtest-ai-configurator-mlx-v1
  - docs/runbooks/mac-studio-native-backend-operations.md
  - docs/runbooks/mac-studio-monitoring-plan.md
  - configs/prod/backtest_ai_configurator.yaml
  - tests/unit/contexts/backtest/application/ai_configurator

safety_notes:
  - "This prompt prepares the repo for the new pack; it must leave the feature blocked."
  - "Local fake-worker tests are regression evidence only, never production acceptance."
---

# Task

Clean any remaining wrong single-shot `/backtests` AI Configurator runtime and
prompt contracts. This is not a git revert. It is a surgical cleanup that keeps
the useful foundation and removes or retires only the wrong runtime path.

Done means:

- active code no longer exposes single-shot prompt/blob gateway or adapter contracts;
- public API/storage/quota/status/SSE shell remains intact;
- validator, catalog resolver, artifact coverage, indicator bounds and security gates remain intact;
- docs say the old single-shot contract is retired;
- feature remains disabled/blocked until the tool-agent pipeline exists;
- cleanup evidence is written in Markdown and JSON.

## Context / Current State

Context ledger:

- completed:
  - single-shot LM Studio prompt/blob contract was identified as the wrong path;
  - old accepted evidence is historical only;
  - target runtime direction is LM Studio OpenAI-compatible tools through backend-owned executors.
- open_items:
  - new tool-agent contract is not implemented yet;
  - old prompts/docs may still contain executable-looking instructions;
  - stale ignored scratch artifacts or bytecode may still contain old names.
- contract_changes:
  - runtime contract intentionally breaks away from `generate_config` / `repair_config`;
  - `runtime: lm_studio_tools` is a blocked placeholder, not accepted production runtime.
- risks:
  - accidentally deleting retained validator/catalog/security foundation;
  - leaving old prompt pack runnable;
  - treating local smoke or historical benchmark as acceptance.
- next_focus:
  - clean boundary for the next design prompt.

## Requirements (Must)

- Inventory wrong-layer artifacts before editing.
- Remove or replace only the retired single-shot runtime/prompt contracts:
  `BacktestConfigLLMGateway`, single-shot `generate_config`/`repair_config`
  semantics, full prompt envelope builders, model-visible `TRUSTED_CAPABILITIES`
  blobs, adapter smoke based only on `messages + response_format`, and docs that
  present that path as current.
- Preserve retained foundation: job storage, quota, idempotency, API routes,
  status/SSE shell, validator, catalog resolver, artifact coverage, indicator
  bounds, security gates and final backend validation.
- Keep production feature disabled/blocked and report the exact blocker.
- Create `cleanup_readiness.md` and `cleanup_readiness.json` with top-level
  `accepted`, `blocking_reason`, and `next_prompt_allowed`.
- If any old executable contract remains in active code, do not proceed to the
  next prompt; record blocker.

## Requirements (Should)

- Mark old generated prompts as `do_not_execute: true` tombstones instead of
  leaving runnable stale instructions.
- Remove ignored scratch artifacts that contain full old prompt/data blobs.
- Keep docs concise and explicit about what is historical vs current.

## Requirements (Nice-to-have)

- Include a short grep inventory table in the cleanup evidence.

# Context acquisition protocol

Read only in this order and stop once sufficient:

1. `.codex/AGENTS.md`
2. reset/retirement docs listed in `always_read`
3. task entrypoints
4. conditional docs/prompt bundles only if stale hits are found
5. tests only when needed for failures

Reading budget: max 10 repo files plus directories searched by `rg`.

# Reading manifest

Use the front-matter `context_sources` as the canonical reading map.

# Work plan (agent should follow)

1. Run inventory searches for old symbols and old acceptance wording.
2. Classify each hit as active code, historical doc, tombstone prompt, ignored scratch, or stale bytecode.
3. Remove/replace active wrong contracts only.
4. Preserve retained backend gates and API shell.
5. Update docs/evidence.
6. Run focused gates.
7. Report blocker or clean handoff to Prompt 02.

# Acceptance criteria (Definition of Done)

- Active `src`, `apps`, `scripts`, `tests`, and `configs` have no old
  single-shot gateway/adapter/prompt-blob contract.
- Historical docs may mention old literals only as retired/historical evidence.
- `runtime: lm_studio_tools` remains blocked/pending.
- `cleanup_readiness.md/json` exist and are indexed.
- All required quality gates pass.

# Implementation constraints

## Cleanup

- Do not delete storage, quota, API routes, status/SSE, validator, catalog
  resolver, artifact coverage, indicator bounds or security gates.
- Do not implement tool calls in this prompt.

## Documentation

- Old current docs must no longer describe single-shot runtime as current.
- New cleanup evidence must use machine-readable gate markers.

# Files to indicate (expected touched areas)

Use the front-matter expected touches.

# Non-goals

- No new tool-agent implementation.
- No Mac Studio load benchmark.
- No paid rollout.

# Quality gates (must run and pass)

Use the front-matter `quality_gates`.

# Final output: report format (strict)

Report in Russian using the front-matter sections.
