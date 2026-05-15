---
prompt_name: backtest_ai_configurator_lmstudio_v1_18_single_chat_ui_onboarding
repo: roehub.com
branch: main
scope: "Iteration 18: remove explicit AI mode buttons from /backtests and implement localized single-chat onboarding UI."

language:
  implementation: fastapi_jinja_plain_js_css
  agent_report: ru

context_sources:
  always_read:
    - path: .codex/AGENTS.md
      why: "browser-visible verification rules"
    - path: docs/architecture/backtest/backtest-ai-configurator-auto-intent-chat-v1.md
      why: "target UX contract"
    - path: docs/architecture/backtest/benchmark_iterations/2026-05-15_ai_configurator_auto_intent_ux/backend_auto_intent_acceptance.md
      why: "required backend auto-intent acceptance"
  task_entrypoints:
    - path: apps/web/templates/pages/backtests.html
      why: "current AI panel markup with mode buttons"
      inspect_symbols:
        - backtests-ai-modes
        - data-ai-mode
        - data-ai-prompt
    - path: apps/web/dist/js/pages/backtests.js
      why: "current selected mode state and AI payload"
      inspect_symbols:
        - AI_DEFAULT_MODE
        - state.ai.mode
        - currentAiPayload
        - selectAiMode
    - path: apps/web/dist/css/pages/backtests.css
      why: "mode button styles and chat panel layout"
      inspect_symbols:
        - backtests-ai-modes
        - backtests-ai-mode
    - path: tests/unit/apps/web/test_backtests_ai_configurator.py
      why: "browser-visible asset contract tests"
      inspect_symbols:
        - "*"
  conditional_bundles:
    ui_state:
      read_when: "when consuming ai_configurator_state payload"
      paths:
        - apps/api/wiring/modules/ui_backtests.py
        - tests/unit/apps/api/test_ui_backtests_routes.py
    locale_strings:
      read_when: "when changing RU/EN copy"
      paths:
        - apps/web
    browser_route:
      read_when: "when route/template data attributes change"
      paths:
        - tests/unit/apps/web/test_app_routes.py
        - tests/unit/apps/api/test_ui_backtests_routes.py
  consult_if_needed:
    - path: docs/architecture/apps/web/web-ui-design-manifest-v1.md
      read_when: "only if visual style conflicts"

style_references:
  - apps/web/templates/pages/backtests.html
  - apps/web/dist/js/pages/backtests.js
  - apps/web/dist/css/pages/backtests.css

documentation_continuity:
  old_current_docs:
    - docs/architecture/backtest/backtest-ai-configurator-mlx-v1.md
    - docs/architecture/backtest/backtest-ai-configurator-auto-intent-chat-v1.md
  new_doc_artifact: docs/architecture/backtest/benchmark_iterations/2026-05-15_ai_configurator_auto_intent_ux/single_chat_ui_acceptance.md
  canonical_shape: "browser-visible implementation evidence markdown plus screenshots/notes"
  docs_gate: "uv run python -m tools.docs.generate_docs_index --check"

hard_requirements:
  depends_on_iteration_17_accepted: true
  remove_mode_selector_ui_required: true
  remove_mode_selector_js_state_required: true
  localized_startup_message_required: true
  request_language_response_policy_visible_required: true
  no_ai_backtest_execution_capability_required: true
  browser_qa_required: true
  publish_ci_deploy_required: true
  macstudio_sync_required: true

task_toggles:
  implement_ui: true
  implement_backend: false
  update_tests: true
  update_docs: true
  run_browser_qa: true

skill_routing:
  - skill: contract-impact-analysis
    use_when: "removing browser-visible controls and request payload fields"
    timing: "before implementation"
    reason: "browser/API compatibility"
  - skill: browser-qa-evidence
    use_when: "after UI implementation"
    timing: "during verification"
    reason: "visible /backtests behavior must be verified in browser"
  - skill: backend-quality-gates
    use_when: "running web/API unit tests and lint/type gates"
    timing: "during verification"
    reason: "local gates"
  - skill: publish-ci-deploy
    use_when: "after local and browser gates pass"
    timing: "final delivery step"
    reason: "ship UI changes and verify production host"

target_envs:
  - local-dev
  - browser
  - github-actions
  - mac-studio-prod

required_literals:
  - "backtests-ai-modes"
  - "backtests-ai-mode"
  - "data-ai-mode"
  - "CREATE"
  - "EDIT"
  - "EXPLAIN"
  - "REPAIR"
  - "SAFER"
  - "Load configuration"
  - "Загрузить конфигурацию"
  - "AI cannot run backtests"
  - "startup message uses platform locale"
  - "model replies in the language of the user request"
  - "accepted: true/false"
  - "blocking_reason"
  - "next_prompt_allowed"

non_goals:
  - "Do not implement backend intent resolver in this prompt."
  - "Do not add a JSON editor."
  - "Do not embed LM Studio UI."
  - "Do not auto-run backtests."
  - "Do not edit old prompt files 01-17."

final_report_format:
  language: ru
  sections:
    - "Что реализовано"
    - "Browser-visible contract"
    - "Локализация и язык ответа"
    - "Browser QA evidence"
    - "Доставка и Mac Studio"

quality_gates:
  - cmd: "uv run pytest -q tests/unit/apps/web tests/unit/apps/api/test_ui_backtests_routes.py tests/unit/apps/api/test_backtest_ai_config_routes.py"
    expect: "passes"
  - cmd: "uv run ruff check apps/web apps/api tests/unit/apps"
    expect: "passes"
  - cmd: "uv run pyright"
    expect: "passes"
  - cmd: "uv run python -m tools.docs.generate_docs_index --check"
    expect: "passes if docs changed"
  - cmd: "git diff --check"
    expect: "passes"

expected_primary_touches:
  - apps/web/templates/pages/backtests.html
  - apps/web/dist/js/pages/backtests.js
  - apps/web/dist/css/pages/backtests.css
  - tests/unit/apps/web/test_backtests_ai_configurator.py
  - tests/unit/apps/api/test_ui_backtests_routes.py
  - docs/architecture/backtest/benchmark_iterations/2026-05-15_ai_configurator_auto_intent_ux/

possible_secondary_touches:
  - apps/api/wiring/modules/ui_backtests.py
  - tests/unit/apps/web/test_app_routes.py
  - docs/architecture/backtest/backtest-ai-configurator-mlx-v1.md
  - docs/architecture/backtest/backtest-ai-configurator-auto-intent-chat-v1.md

safety_notes:
  - "Remove old mode controls from markup/JS/CSS/tests; do not hide them with CSS only."
  - "Startup message is trusted localized UI text; assistant/model text remains safely rendered as text."
  - "AI load action fills the form only; existing run button remains the only job creation action."
---

# Task

Implement the single-chat `/backtests` AI Configurator UI.

This prompt starts only after Iteration 17 backend auto-intent acceptance exists and passed. It removes the old explicit mode-selection UI and replaces it with a localized onboarding/startup assistant message plus one prompt input.

Done means:

- `CREATE / EDIT / EXPLAIN / REPAIR / SAFER` UI controls are removed from DOM, JS state, CSS and tests;
- AI prompt payload no longer sends a user-selected mode;
- startup message is shown by default in the platform-selected language;
- UI explains that the assistant can prepare/applyable configs but cannot run backtests;
- model/assistant text is still rendered as inert text;
- browser QA proves the new UI on `/backtests`;
- docs/evidence are updated and delivered.

## Context / Current State

Context ledger:

- completed:
  - Iteration 17 should make backend auto-intent accepted.
  - Current live UI has a 5-button mode row.
- open_items:
  - Remove mode row from markup/CSS/JS/tests.
  - Add localized trusted onboarding message.
  - Remove `state.ai.mode` from browser payload and feedback if no longer needed.
- contract_changes:
  - Browser no longer sends user-selected `mode`.
  - `ai_configurator_state.modes` is not a UI contract.
  - Startup copy is localized by platform locale.
- risks:
  - old hidden controls still affecting keyboard/navigation/tests;
  - backend still depending on `mode`;
  - user thinking AI can run backtests;
  - startup message being treated as model output.
- next_focus:
  - browser-visible UX fix.

## Requirements (Must)

- Stop if Iteration 17 backend acceptance is missing or blocked.
- Remove old mode selector from `apps/web/templates/pages/backtests.html`:
  - no `backtests-ai-modes`;
  - no `backtests-ai-mode`;
  - no `data-ai-mode`;
  - no visible `CREATE`, `EDIT`, `EXPLAIN`, `REPAIR`, `SAFER` AI mode controls.
- Remove old mode-state JS:
  - no `AI_DEFAULT_MODE` for user-selected mode;
  - no `state.ai.mode`;
  - no `selectAiMode`;
  - `currentAiPayload` must not send user-selected `mode`;
  - feedback payload must not depend on user-selected mode.
- Remove stale mode CSS selectors if no longer used.
- Add trusted default startup/onboarding message:
  - shown before first user message;
  - localized from platform language (`ru` or `en`);
  - not model-generated;
  - says AI can create/edit/explain/repair/suggest safer configs for `/backtests`;
  - says AI cannot run backtests and user must press existing run button manually.
- Preserve language policy in UI copy: model replies in the language of the user's request.
- Preserve status timeline and safe text rendering.
- `Load configuration` / `Загрузить конфигурацию` remains only for backend `ready` with validated config.
- Update tests to assert old mode controls are gone and onboarding exists.
- Run browser QA on `/backtests`, including console/network check.
- Run `publish-ci-deploy` after gates pass.

## Requirements (Should)

- Keep the AI panel compact and work-focused.
- If `ai_configurator_state.capabilities` exists, render concise capability copy from trusted locale strings only.
- If feature is disabled, still do not show mode buttons.
- Keep responsive layout stable and avoid text overflow.

## Requirements (Nice-to-have)

- Include screenshots in evidence if browser QA tooling makes this easy.

# Context acquisition protocol

Read only in this order and stop once sufficient:

1. `.codex/AGENTS.md`
2. Iteration 17 backend acceptance
3. target auto-intent architecture doc
4. task entrypoints
5. conditional bundles only for UI state, locale or route tests
6. design manifest only for style conflicts

Do not preload unrelated web UI pages.

Reading budget: max 10 repo files plus test files touched by failures.

# Reading manifest

- `always_read`: repo contract, target UX doc, backend acceptance.
- `task_entrypoints`: template, JS, CSS, web tests.
- `conditional_bundles`: UI state and locale tests only when touched.
- `consult_if_needed`: design manifest only for conflicts.

Stop reading once UI write set and test expectations are clear.

# Work plan (agent should follow)

1. Verify Iteration 17 accepted evidence.
2. Remove old mode selector markup, JS, CSS and tests.
3. Add localized trusted onboarding message.
4. Update AI payload and feedback payload to no longer send user-selected mode.
5. Update tests.
6. Run local gates.
7. Run real browser QA on `/backtests`.
8. Write evidence.
9. Use `publish-ci-deploy`.

# Acceptance criteria (Definition of Done)

- `rg -n "data-ai-mode|backtests-ai-modes|backtests-ai-mode|selectAiMode|state\\.ai\\.mode|AI_DEFAULT_MODE" apps/web/templates/pages/backtests.html apps/web/dist/js/pages/backtests.js apps/web/dist/css/pages/backtests.css tests/unit/apps/web/test_backtests_ai_configurator.py` returns no current-active references. If any retained reference exists, it is classified and justified.
- `/backtests` shows one chat input and no mode selector.
- Startup message appears by default in RU when platform locale is RU and EN when platform locale is EN.
- UI copy states AI cannot run backtests.
- Model/assistant output is rendered as text, not HTML.
- `Load configuration` appears only for validated ready config.
- Browser QA evidence includes screenshot or explicit visual check, console clean, network clean, no auto-run.
- Evidence includes `accepted`, `blocking_reason`, `next_prompt_allowed`.
- `publish-ci-deploy` reaches `deployed`, or exact `green-pr`/`blocked` state is recorded.

# Implementation constraints

## UI

- Do not hide old mode buttons with CSS; remove the controls and event handling.
- Do not add decorative cards or landing copy.
- Do not add visible chain-of-thought.

## API

- Do not call `/api/backtests/jobs` from AI flow.
- Do not stream raw JSON/model output to browser.

## Documentation

- Update current docs and create `single_chat_ui_acceptance.md`.
- Run docs index check.

# Files to indicate (expected touched areas)

Expected primary touches:

- `apps/web/templates/pages/backtests.html`
- `apps/web/dist/js/pages/backtests.js`
- `apps/web/dist/css/pages/backtests.css`
- `tests/unit/apps/web/test_backtests_ai_configurator.py`
- `tests/unit/apps/api/test_ui_backtests_routes.py`
- `docs/architecture/backtest/benchmark_iterations/2026-05-15_ai_configurator_auto_intent_ux/single_chat_ui_acceptance.md`

Possible secondary touches:

- `apps/api/wiring/modules/ui_backtests.py`
- `tests/unit/apps/web/test_app_routes.py`
- `docs/architecture/backtest/backtest-ai-configurator-mlx-v1.md`
- `docs/architecture/backtest/backtest-ai-configurator-auto-intent-chat-v1.md`

# Non-goals

- No backend intent implementation.
- No runtime/model-serving work.
- No benchmark run.
- No old prompt edits.
- No AI auto-run.

# Quality gates (must run and pass)

- `uv run pytest -q tests/unit/apps/web tests/unit/apps/api/test_ui_backtests_routes.py tests/unit/apps/api/test_backtest_ai_config_routes.py`
- `uv run ruff check apps/web apps/api tests/unit/apps`
- `uv run pyright`
- `uv run python -m tools.docs.generate_docs_index --check`
- `git diff --check`
- Browser QA with available browser surface: `/backtests`, RU and EN startup message, no mode selector, console/network clean, no auto-run.

Required delivery step: after gates pass, invoke `publish-ci-deploy` as the final step. The expected terminal state is `deployed`; if `green-pr` or `blocked`, report exact blocker and do not claim deployment.

# Final output: report format (strict)

Report in Russian with:

- `Что реализовано`: removed old controls, new onboarding, payload changes.
- `Browser-visible contract`: no mode selector, no auto-run, safe rendering.
- `Локализация и язык ответа`: RU/EN startup message and request-language policy.
- `Browser QA evidence`: target, screenshot/visual assertions, console/network.
- `Доставка и Mac Studio`: publish-ci-deploy state, CI, Mac Studio sync/smoke evidence or blocker.
