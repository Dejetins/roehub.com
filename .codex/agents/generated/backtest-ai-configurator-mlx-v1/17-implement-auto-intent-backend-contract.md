---
prompt_name: backtest_ai_configurator_lmstudio_v1_17_auto_intent_backend_contract
repo: roehub.com
branch: main
scope: "Iteration 17: implement backend auto-intent contract and remove user-selected mode as the server authority for /backtests AI jobs."

language:
  implementation: python_backend
  agent_report: ru

context_sources:
  always_read:
    - path: .codex/AGENTS.md
      why: "repo contract and compatibility rules"
    - path: docs/architecture/backtest/backtest-ai-configurator-auto-intent-chat-v1.md
      why: "required target contract from Iteration 16"
    - path: docs/architecture/backtest/backtest-ai-configurator-mlx-v1.md
      why: "current architecture doc to keep synchronized"
  task_entrypoints:
    - path: apps/api/dto/backtest_ai_config.py
      why: "AI job create request/response DTO"
      inspect_symbols:
        - BacktestAiConfigCreateRequest
        - BacktestAiConfigJobResponse
    - path: apps/api/routes/backtest_ai_config.py
      why: "job create route currently forwards payload.mode"
      inspect_symbols:
        - post_backtest_ai_config_job
    - path: src/trading/contexts/backtest/application/ai_configurator/jobs.py
      why: "job creation, mode normalization and idempotency"
      inspect_symbols:
        - BacktestAiConfigJobsUseCase
        - _normalize_mode
    - path: src/trading/contexts/backtest/application/ai_configurator/services/prompt_profiles.py
      why: "prompt profile selection currently uses mode"
      inspect_symbols:
        - backtest_ai_prompt_profile_for_mode
        - build_generate_prompt_envelope
  conditional_bundles:
    security_and_language:
      read_when: "when implementing intent/language detection and safety gates"
      paths:
        - src/trading/contexts/backtest/application/ai_configurator/services/security.py
        - tests/unit/contexts/backtest/application/ai_configurator/
    persistence:
      read_when: "if storage mode/intent fields or audit shape need migration/compatibility work"
      paths:
        - src/trading/contexts/backtest/adapters/outbound/persistence/postgres/backtest_ai_config_repository.py
        - alembic/versions
    ui_state:
      read_when: "when removing modes from backend workstation payload"
      paths:
        - apps/api/wiring/modules/ui_backtests.py
        - tests/unit/apps/api/test_ui_backtests_routes.py
  consult_if_needed:
    - path: docs/architecture/backtest/benchmark_iterations/2026-05-13_lmstudio_serving_recovery/lmstudio_adapter_acceptance.md
      read_when: "only if LM Studio structured output contract conflicts with prompt changes"

style_references:
  - src/trading/contexts/backtest/application/ai_configurator/
  - tests/unit/contexts/backtest/application/ai_configurator/

documentation_continuity:
  old_current_docs:
    - docs/architecture/backtest/backtest-ai-configurator-mlx-v1.md
    - docs/architecture/backtest/backtest-ai-configurator-auto-intent-chat-v1.md
  new_doc_artifact: docs/architecture/backtest/benchmark_iterations/2026-05-15_ai_configurator_auto_intent_ux/backend_auto_intent_acceptance.md
  canonical_shape: "implementation evidence markdown plus JSON marker"
  docs_gate: "uv run python -m tools.docs.generate_docs_index --check"

hard_requirements:
  depends_on_iteration_16_accepted: true
  backend_auto_intent_required: true
  user_selected_mode_not_authoritative: true
  legacy_mode_compatibility_explicit: true
  model_reply_language_matches_user_request_required: true
  no_ai_backtest_execution_capability_required: true
  publish_ci_deploy_required: true
  macstudio_sync_required: true

task_toggles:
  implement_backend: true
  implement_ui: false
  add_tests: true
  update_docs: true
  run_real_lmstudio_e2e: false

skill_routing:
  - skill: contract-impact-analysis
    use_when: "changing request DTO, domain mode/intent, persistence or API response behavior"
    timing: "before implementation"
    reason: "compatibility and rollout safety"
  - skill: backend-quality-gates
    use_when: "running backend/API/unit/type/lint gates"
    timing: "during verification"
    reason: "backend verification"
  - skill: production-risk-review
    use_when: "before final report for security, prompt and no-auto-run trust boundary"
    timing: "before ship"
    reason: "LLM trust boundary"
  - skill: publish-ci-deploy
    use_when: "after local backend gates pass"
    timing: "final delivery step"
    reason: "ship backend contract and verify Mac Studio"

target_envs:
  - local-dev
  - unit-tests
  - github-actions
  - mac-studio-prod

required_literals:
  - "auto intent"
  - "create_config"
  - "edit_current_config"
  - "explain_current_config"
  - "repair_invalid_config"
  - "suggest_safer_config"
  - "needs_clarification"
  - "unsupported"
  - "model replies in the language of the user request"
  - "AI cannot run backtests"
  - "mode is optional/deprecated"
  - "POST /v1/chat/completions"
  - "choices[0].message.content"
  - "JSON Schema type values must be strings"
  - "do not use type: [\"string\", \"null\"]"
  - "accepted: true/false"
  - "blocking_reason"
  - "next_prompt_allowed"

non_goals:
  - "Do not implement browser UI changes in this prompt."
  - "Do not remove audit/training data."
  - "Do not run S1/S5/S10/S50/S100 benchmark."
  - "Do not give model any tool/API permission to run backtests."
  - "Do not edit old prompt files 01-16."

final_report_format:
  language: ru
  sections:
    - "Что реализовано"
    - "Backend contract"
    - "Безопасность и язык ответа"
    - "Проверки"
    - "Доставка и Mac Studio"

quality_gates:
  - cmd: "uv run pytest -q tests/unit/contexts/backtest/application/ai_configurator tests/unit/apps/api/test_backtest_ai_config_routes.py tests/unit/apps/api/test_ui_backtests_routes.py"
    expect: "passes"
  - cmd: "uv run ruff check apps/api src/trading/contexts/backtest tests/unit"
    expect: "passes"
  - cmd: "uv run pyright"
    expect: "passes"
  - cmd: "uv run python -m tools.docs.generate_docs_index --check"
    expect: "passes if docs changed"
  - cmd: "git diff --check"
    expect: "passes"

expected_primary_touches:
  - apps/api/dto/backtest_ai_config.py
  - apps/api/routes/backtest_ai_config.py
  - apps/api/wiring/modules/ui_backtests.py
  - src/trading/contexts/backtest/application/ai_configurator/
  - tests/unit/contexts/backtest/application/ai_configurator/
  - tests/unit/apps/api/test_backtest_ai_config_routes.py
  - tests/unit/apps/api/test_ui_backtests_routes.py
  - docs/architecture/backtest/benchmark_iterations/2026-05-15_ai_configurator_auto_intent_ux/

possible_secondary_touches:
  - src/trading/contexts/backtest/adapters/outbound/persistence/postgres/backtest_ai_config_repository.py
  - alembic/versions/
  - docs/architecture/backtest/backtest-ai-configurator-mlx-v1.md
  - docs/architecture/backtest/backtest-ai-configurator-auto-intent-chat-v1.md

safety_notes:
  - "The AI backend must never call `/api/backtests/jobs` or enqueue backtest jobs."
  - "Intent detection is a backend concern; the browser must not be trusted to choose the operation."
  - "Model output remains untrusted until schema, output and business validation pass."
---

# Task

Implement the backend auto-intent contract for `/backtests` AI Configurator.

This prompt starts only after Iteration 16 design evidence exists and is accepted. It changes backend/API/domain behavior so the user's explicit mode selection is no longer the authority. The backend resolves intent from the user's message plus current `/backtests` form state.

Done means:

- job creation no longer requires browser-selected `mode` as authoritative input;
- backend resolves one of the target intents;
- legacy `mode` compatibility is explicit and tested if retained;
- prompt profiles and security gates use resolved intent, not a UI radio button;
- model response language policy is enforced in prompt/envelope tests;
- AI still cannot run backtests;
- docs/evidence are updated and delivered.

## Context / Current State

Context ledger:

- completed:
  - Iteration 16 should define target single-chat auto-intent contract.
  - Current backend DTO requires `mode`.
  - Current jobs use case normalizes `mode`.
- open_items:
  - Need server-side intent resolver.
  - Need request language detection/policy for assistant response language.
  - Need compatibility decision for old `mode`.
- contract_changes:
  - `mode` becomes optional/deprecated, replaced by server resolved intent.
  - `ai_configurator_state.modes` should no longer be needed for browser UI.
  - storage may retain old field as resolved intent if no migration is needed.
- risks:
  - breaking idempotency if mode disappears from pending job hash;
  - over-trusting LLM intent classification;
  - ambiguous prompts turning into loadable configs;
  - model implying it can run a backtest.
- next_focus:
  - backend contract before browser UI deletion.

## Requirements (Must)

- Stop if Iteration 16 evidence is missing or not accepted.
- Add or update backend intent contract:
  - `create_config`;
  - `edit_current_config`;
  - `explain_current_config`;
  - `repair_invalid_config`;
  - `suggest_safer_config`;
  - `needs_clarification`;
  - `unsupported`.
- `BacktestAiConfigCreateRequest.mode` must become optional/deprecated or be replaced by a new auto-intent field according to the Iteration 16 decision.
- Browser/user-selected mode must not be authoritative. If a legacy `mode` value is sent, backend may map it into a hint only if documented and tested.
- Add deterministic intent resolver before LLM call. It may use domain terms, current-config state and clear user phrases; unclear cases must return `needs_clarification` rather than guessing a loadable config.
- Prompt profile selection must use resolved intent.
- Any LM Studio `response_format.json_schema` emitted or changed in this prompt
  must keep every schema `type` value as a string. Do not emit nullable union
  arrays such as `type: ["string", "null"]`; use empty string or explicit
  status/boolean fields for optional model-output fields.
- Prompt/system policy must state:
  - model replies in the language of the user request;
  - AI cannot run backtests;
  - AI can only explain, ask clarification, or return a validated config that the user may load.
- Preserve one-attempt repair and existing validation gates.
- Update `ai_configurator_state` payload away from `modes` toward `capabilities`/`onboarding` if this is part of the Iteration 16 contract.
- Update docs and create backend evidence with `accepted`, `blocking_reason`, `next_prompt_allowed`.
- Run `publish-ci-deploy` after gates pass.

## Requirements (Should)

- Keep old storage schema if a migration is not needed; document any retained `mode` column as resolved intent/compatibility.
- Add tests for RU and EN prompts proving response-language instruction follows request language.
- Add tests for off-topic, prompt-injection, "run this backtest", explain-only and safer-config prompts.
- Keep idempotency stable and explicit when old mode is absent.

## Requirements (Nice-to-have)

- Include an `intent_confidence` internal diagnostic if useful, but do not expose confusing scores to users.

# Context acquisition protocol

Read only in this order and stop once sufficient:

1. `.codex/AGENTS.md`
2. Iteration 16 architecture doc/evidence
3. task entrypoints
4. conditional bundles only for touched security/persistence/UI-state contracts
5. consult-if-needed LM Studio evidence only for payload conflicts

Do not preload all old benchmark artifacts.

Reading budget: max 12 repo files plus tests touched by failures.

# Reading manifest

- `always_read`: repo contract, target auto-intent doc, current architecture doc.
- `task_entrypoints`: API DTO/route, jobs use case, prompt profiles.
- `conditional_bundles`: security/language, persistence, UI state only if touched.
- `consult_if_needed`: LM Studio adapter evidence only for structured-output conflicts.

Stop reading once DTO/domain/prompt/test write set is bounded.

# Work plan (agent should follow)

1. Verify Iteration 16 accepted evidence.
2. Run contract-impact analysis for API, idempotency, storage and UI state.
3. Implement resolved intent contract and legacy compatibility.
4. Update prompt profiles/security gates to use resolved intent and request-language response policy.
5. Add targeted tests.
6. Update docs/evidence.
7. Run local gates.
8. Use `publish-ci-deploy`.

# Acceptance criteria (Definition of Done)

- API accepts the new auto-intent request shape without requiring user-selected mode.
- Legacy `mode` behavior is either removed with documented breakage or accepted as a compatibility hint with tests.
- Backend resolves safe supported prompts into expected intents.
- Ambiguous/off-topic/injection prompts do not produce loadable ready configs.
- Prompt/envelope tests prove model is instructed to answer in the language of the user request.
- Tests prove AI cannot run/enqueue/start backtests.
- Docs/evidence include `accepted`, `blocking_reason`, `next_prompt_allowed`.
- `publish-ci-deploy` reaches `deployed`, or exact `green-pr`/`blocked` state is recorded.

# Implementation constraints

## Backend

- Keep `BacktestConfigLLMGateway` port stable unless a narrow contract review proves change is required.
- Do not add public routes unless required.
- Do not expose raw model drafts/prompts.

## Security

- Treat model output as untrusted.
- Do not let intent resolver bypass input/output/security gates.
- Do not add any tool-calling or API-calling capability for the model.

## Documentation

- Update current docs and create `backend_auto_intent_acceptance.md`.
- Run docs index check.

# Files to indicate (expected touched areas)

Expected primary touches:

- `apps/api/dto/backtest_ai_config.py`
- `apps/api/routes/backtest_ai_config.py`
- `apps/api/wiring/modules/ui_backtests.py`
- `src/trading/contexts/backtest/application/ai_configurator/`
- `tests/unit/contexts/backtest/application/ai_configurator/`
- `tests/unit/apps/api/test_backtest_ai_config_routes.py`
- `tests/unit/apps/api/test_ui_backtests_routes.py`
- `docs/architecture/backtest/benchmark_iterations/2026-05-15_ai_configurator_auto_intent_ux/backend_auto_intent_acceptance.md`

Possible secondary touches:

- `src/trading/contexts/backtest/adapters/outbound/persistence/postgres/backtest_ai_config_repository.py`
- `alembic/versions/`
- `docs/architecture/backtest/backtest-ai-configurator-mlx-v1.md`
- `docs/architecture/backtest/backtest-ai-configurator-auto-intent-chat-v1.md`

# Non-goals

- No browser UI removal in this prompt.
- No benchmark run.
- No model-serving changes.
- No automatic backtest execution.
- No old prompt edits.

# Quality gates (must run and pass)

- `uv run pytest -q tests/unit/contexts/backtest/application/ai_configurator tests/unit/apps/api/test_backtest_ai_config_routes.py tests/unit/apps/api/test_ui_backtests_routes.py`
- `uv run ruff check apps/api src/trading/contexts/backtest tests/unit`
- `uv run pyright`
- `uv run python -m tools.docs.generate_docs_index --check`
- `git diff --check`

Required delivery step: after gates pass, invoke `publish-ci-deploy` as the final step. The expected terminal state is `deployed`; if `green-pr` or `blocked`, report exact blocker and do not claim deployment.

# Final output: report format (strict)

Report in Russian with:

- `Что реализовано`: intent resolver, DTO/API/prompt changes.
- `Backend contract`: new request shape, legacy mode handling, idempotency/storage impact.
- `Безопасность и язык ответа`: no-run proof, request-language response policy.
- `Проверки`: commands and results.
- `Доставка и Mac Studio`: publish-ci-deploy state, CI, Mac Studio sync/smoke evidence or blocker.
