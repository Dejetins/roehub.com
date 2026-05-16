---
prompt_name: backtest_ai_configurator_lmstudio_v1_16_auto_intent_chat_contract
repo: roehub.com
branch: main
scope: "Iteration 16: redesign /backtests AI Configurator from explicit mode buttons to one auto-intent chat contract."

language:
  implementation: architecture_docs
  agent_report: ru

context_sources:
  always_read:
    - path: .codex/AGENTS.md
      why: "repo contract, browser-visible verification and documentation rules"
    - path: docs/architecture/backtest/backtest-ai-configurator-mlx-v1.md
      why: "current architecture source with explicit mode selector and AI configurator contract"
    - path: .codex/agents/generated/backtest-ai-configurator-mlx-v1/06-implement-web-ui-integration.md
      why: "old UI prompt that introduced explicit CREATE/EDIT/EXPLAIN/REPAIR/SAFER mode selector"
  task_entrypoints:
    - path: apps/web/templates/pages/backtests.html
      why: "current AI panel markup and mode buttons"
      inspect_symbols:
        - data-ai-mode
        - backtests-ai-modes
    - path: apps/web/dist/js/pages/backtests.js
      why: "current selected-mode state and AI payload"
      inspect_symbols:
        - AI_DEFAULT_MODE
        - currentAiPayload
        - selectAiMode
    - path: apps/api/dto/backtest_ai_config.py
      why: "public API request shape currently requiring mode"
      inspect_symbols:
        - BacktestAiConfigCreateRequest
    - path: src/trading/contexts/backtest/application/ai_configurator/dto.py
      why: "domain job mode and state contract"
      inspect_symbols:
        - BacktestAiConfigMode
        - BacktestAiConfigJob
  conditional_bundles:
    prompt_profiles:
      read_when: "when defining target prompt/intent profile responsibilities"
      paths:
        - src/trading/contexts/backtest/application/ai_configurator/services/prompt_profiles.py
        - src/trading/contexts/backtest/application/ai_configurator/services/catalog.py
        - src/trading/contexts/backtest/application/ai_configurator/services/validator.py
        - src/trading/contexts/backtest/application/ai_configurator/services/security.py
    ui_state_contract:
      read_when: "when defining backend-to-web state payload"
      paths:
        - apps/api/wiring/modules/ui_backtests.py
        - tests/unit/apps/api/test_ui_backtests_routes.py
        - tests/unit/apps/web/test_backtests_ai_configurator.py
  consult_if_needed:
    - path: docs/architecture/apps/web/web-ui-design-manifest-v1.md
      read_when: "only if visual/control style conflicts"
    - path: docs/architecture/backtest/benchmark_iterations/2026-05-13_lmstudio_serving_recovery/security_pipeline_readiness.md
      read_when: "only if existing accepted security evidence exists and conflicts with this redesign"

style_references:
  - docs/architecture/backtest/backtest-ai-configurator-mlx-v1.md
  - docs/architecture/backtest/benchmark_iterations/2026-05-13_lmstudio_serving_recovery/

documentation_continuity:
  old_current_docs:
    - docs/architecture/backtest/backtest-ai-configurator-mlx-v1.md
  new_doc_artifact: docs/architecture/backtest/backtest-ai-configurator-auto-intent-chat-v1.md
  canonical_shape: "architecture decision plus rollout plan in Russian"
  docs_gate: "uv run python -m tools.docs.generate_docs_index --check"

hard_requirements:
  remove_user_selected_mode_cjm: true
  single_chat_auto_intent_required: true
  startup_message_platform_locale_required: true
  model_reply_language_matches_user_request_required: true
  no_ai_backtest_execution_capability_required: true
  trusted_capabilities_boundary_required: true
  no_general_backtesting_chat_scope_creep_required: true
  docs_update_required: true
  publish_ci_deploy_required: true
  macstudio_sync_required: true

task_toggles:
  design_only: true
  change_runtime_code: false
  change_browser_ui_code: false
  write_architecture_doc: true
  update_current_architecture_doc: true

skill_routing:
  - skill: architecture-design
    use_when: "defining target UI/backend contract and migration path"
    timing: "before writing docs"
    reason: "this is a target-state design and contract change"
  - skill: contract-impact-analysis
    use_when: "classifying API, DTO, storage and browser-visible compatibility"
    timing: "before finalizing the design"
    reason: "mode removal affects request, UI state and persisted job semantics"
  - skill: backend-quality-gates
    use_when: "running docs index and formatting checks"
    timing: "during verification"
    reason: "docs gates"
  - skill: publish-ci-deploy
    use_when: "after docs changes and gates pass"
    timing: "final delivery step"
    reason: "ship architecture docs and sync Mac Studio"

target_envs:
  - local-dev
  - github-actions
  - mac-studio-prod

required_literals:
  - "single chat"
  - "auto intent"
  - "CREATE / EDIT / EXPLAIN / REPAIR / SAFER"
  - "Load configuration"
  - "Загрузить конфигурацию"
  - "model replies in the language of the user request"
  - "startup message uses platform locale"
  - "AI cannot run backtests"
  - "TRUSTED_CAPABILITIES"
  - "externalized_runtime_capabilities"
  - "model does not read repository source code"
  - "every model answer remains validator-gated"
  - "mode selector removed"
  - "JSON Schema type values must be strings"
  - "do not use type: [\"string\", \"null\"]"
  - "accepted: true/false"
  - "blocking_reason"
  - "next_prompt_allowed"

non_goals:
  - "Do not implement backend or UI code in this prompt."
  - "Do not edit old prompt files 01-15."
  - "Do not remove audit/training storage."
  - "Do not add any model execution path."

final_report_format:
  language: ru
  sections:
    - "Целевая картина"
    - "Контрактное влияние"
    - "Документация"
    - "Проверки"
    - "Доставка и Mac Studio"

quality_gates:
  - cmd: "uv run python -m tools.docs.generate_docs_index --check"
    expect: "passes"
  - cmd: "git diff --check"
    expect: "passes"

expected_primary_touches:
  - docs/architecture/backtest/backtest-ai-configurator-mlx-v1.md
  - docs/architecture/backtest/backtest-ai-configurator-auto-intent-chat-v1.md

possible_secondary_touches:
  - docs/architecture/backtest/README.md
  - docs/architecture/README.md

safety_notes:
  - "The model never receives authority to run backtests; it may only return validated configs or explanations."
  - "Startup assistant message is trusted UI copy, not model output."
  - "Mode selector removal is a UX and API contract change; classify it explicitly."
---

# Task

Design the new `/backtests` AI Configurator contract: replace the explicit mode-selector CJM with a single chat that resolves user intent automatically.

This prompt is design/docs only. It must define the target UI/backend behavior before implementation prompts change code.

Done means:

- current docs no longer describe the explicit mode selector as the desired target UX;
- a new Russian architecture document defines the single-chat auto-intent contract;
- old `CREATE / EDIT / EXPLAIN / REPAIR / SAFER` buttons are classified as legacy UI to remove, not to hide;
- backend target contract is explicit enough for implementation prompts 17-19;
- backend target contract preserves the accepted LM Studio structured-output
  rule: request body is JSON, model prompt is text in `messages[].content`, and
  any `response_format.json_schema` sent to LM Studio uses string-only `type`
  values with no nullable union arrays;
- evidence markers are present: `accepted`, `blocking_reason`, `next_prompt_allowed`.

## Context / Current State

Context ledger:

- completed:
  - Iteration 06 introduced `/backtests` AI UI with explicit user-selected modes.
  - Backend currently expects a `mode` value in AI job creation.
  - Current UI has mode buttons: `CREATE`, `EDIT`, `EXPLAIN`, `REPAIR`, `SAFER`.
- open_items:
  - User-selected mode CJM is not acceptable; the chat should infer intent from natural language.
  - Startup message should use the user's platform-selected language.
  - Model response language should follow the language of the user's request, not platform UI language.
  - Model must not have any ability to run backtests.
- contract_changes:
  - Public browser UX changes from mode selection to one chat.
  - Backend request mode should become optional/deprecated or replaced by server-side resolved intent.
  - AI state payload should expose capabilities/onboarding, not mode buttons.
- risks:
  - breaking existing internal clients without a compatibility plan;
  - leaving stale `data-ai-mode` UI code while visually hiding buttons;
  - implying the model can run backtests;
  - mixing UI locale and response language policy.
- next_focus:
  - freeze target UI/backend contract before code changes.

## Requirements (Must)

- Define target UI:
  - one AI chat input;
  - no visible mode selector;
  - no `CREATE / EDIT / EXPLAIN / REPAIR / SAFER` radio/button row;
  - trusted startup assistant message shown by default;
  - startup message localized by platform language (`ru` or `en`);
  - status timeline remains stage-based, no chain-of-thought;
  - `Load configuration` / `Загрузить конфигурацию` only appears for validated config returned by backend.
- Define target backend:
  - server resolves intent automatically from user message and current form/config context;
  - supported intents: `create_config`, `edit_current_config`, `explain_current_config`, `repair_invalid_config`, `suggest_safer_config`, `needs_clarification`, `unsupported`;
  - old user-selected mode is removed from the browser contract;
  - compatibility handling for old `mode` field is explicitly decided and classified;
  - storage can keep existing `mode` column only as a resolved intent/compatibility field if that avoids unsafe migration churn.
- Define prompt/data boundary:
  - model does not read repository source code, database state, raw artifact
    manifests, private paths, runtime config, logs, or platform internals;
  - backend builds sanitized `TRUSTED_CAPABILITIES` from external/runtime
    sources and passes only that capability object to the model;
  - capabilities include only backend-executable indicators, allowed
    `indicators.yaml` window bounds/explicit values, and artifact publisher
    period coverage;
  - system prompt and additive security gates may be supplied by operator-owned
    absolute JSON files outside the repository.
- Explicitly decide chat scope: the single chat is an AI configurator chat for
  `/backtests` config creation/edit/explanation/repair/safety suggestions, not
  an unrestricted educational backtesting assistant. If a future general
  `discuss_backtest_config` mode is desired, it must use a separate output
  schema with no `config`, no load action, and the same input/output security
  gates.
- Define language policy:
  - startup UI message uses platform-selected locale;
  - model assistant response uses the language detected from the user request;
  - if request language is ambiguous, prefer platform locale but ask a concise clarification only when needed.
- Define safety:
  - model cannot start, schedule, enqueue, or auto-run backtests;
  - AI may only return explanations, clarification, or validated config drafts that the user may apply manually;
  - every model answer remains schema/security validated before it is shown or
    treated as loadable; no raw model text is a browser contract;
  - existing backtest run button remains the only job-creation control.
- Update old/current docs and create the new architecture doc.
- Run `publish-ci-deploy` after docs gates pass.

## Requirements (Should)

- Keep the target UX compact and consistent with the existing terminal workstation style.
- Prefer `capabilities` / `onboarding` payload over `modes` in `ai_configurator_state`.
- Define a compatibility window for backend accepting legacy `mode` if needed, but make clear it is not user-facing.
- Include contract-impact table with `none`, `compatible-change`, `breaking-change`, or `unknown`.

## Requirements (Nice-to-have)

- Include a small before/after wireflow diagram in Mermaid.
- Include example RU and EN startup messages.

# Context acquisition protocol

Read only in this order and stop once sufficient:

1. `.codex/AGENTS.md`
2. current architecture doc and Iteration 06 prompt
3. task entrypoints for current UI/API shape
4. conditional bundles only for prompt/security or UI-state details
5. consult-if-needed docs only for conflicts

Do not preload all old prompts or all benchmark artifacts.

Reading budget: max 10 repo files.

# Reading manifest

- `always_read`: repo contract, current architecture, old UI prompt.
- `task_entrypoints`: current template, JS, API request DTO, domain DTO.
- `conditional_bundles`: prompt profiles and UI state only when needed.
- `consult_if_needed`: design manifest and latest accepted evidence only for conflicts.

Stop reading once the target contract, migration boundaries and docs write set are clear.

# Work plan (agent should follow)

1. Verify current mode-selector contract in docs/code.
2. Decide and document the target auto-intent contract.
3. Classify contract impact for UI, API request DTO, storage and tests.
4. Update `backtest-ai-configurator-mlx-v1.md` so explicit mode selector is no longer current target UX.
5. Create `backtest-ai-configurator-auto-intent-chat-v1.md`.
6. Run docs gates.
7. Use `publish-ci-deploy`.

# Acceptance criteria (Definition of Done)

- New architecture doc exists and is in Russian.
- Current architecture doc no longer presents explicit mode buttons as target UX.
- Old mode buttons are explicitly marked as legacy to remove.
- Language policy is explicit:
  - startup message uses platform locale;
  - model answers in request language.
- Safety policy is explicit: AI cannot run backtests and only returns validated configs/explanations/clarifications.
- Compatibility decision for old `mode` field is recorded.
- Evidence markers are included in the new doc or delivery evidence:
  - `accepted: true/false`;
  - `blocking_reason: null|string`;
  - `next_prompt_allowed: true/false`.
- `publish-ci-deploy` reaches `deployed`, or exact `green-pr`/`blocked` state is recorded.

# Implementation constraints

## Documentation

- Use one consistent architecture-doc shape.
- Update old/current docs in the same change as the new doc.
- Run docs index check.

## Compatibility

- Do not silently break existing API callers. If the old `mode` field becomes optional/deprecated, document it.
- Do not plan DB migrations unless clearly required.

# Files to indicate (expected touched areas)

Expected primary touches:

- `docs/architecture/backtest/backtest-ai-configurator-mlx-v1.md`
- `docs/architecture/backtest/backtest-ai-configurator-auto-intent-chat-v1.md`

Possible secondary touches:

- `docs/architecture/backtest/README.md`
- `docs/architecture/README.md`

# Non-goals

- No runtime code changes.
- No browser UI implementation.
- No model-serving changes.
- No old prompt edits.

# Quality gates (must run and pass)

- `uv run python -m tools.docs.generate_docs_index --check`
- `git diff --check`

Required delivery step: after gates pass, invoke `publish-ci-deploy` as the final step. The expected terminal state is `deployed`; if the skill reaches `green-pr` or `blocked`, report the exact blocker.

# Final output: report format (strict)

Report in Russian with:

- `Целевая картина`: new UI/backend contract.
- `Контрактное влияние`: API/DTO/storage/UI classification.
- `Документация`: old docs updated, new doc path.
- `Проверки`: commands and results.
- `Доставка и Mac Studio`: publish-ci-deploy state, CI, Mac Studio sync/smoke evidence or blocker.
