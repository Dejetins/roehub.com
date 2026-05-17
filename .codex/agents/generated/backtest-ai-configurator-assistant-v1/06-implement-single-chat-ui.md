---
prompt_name: backtest_ai_configurator_assistant_v1_06_single_chat_ui
repo: roehub.com
branch: main
scope: "Implement browser-visible single-chat UI for /backtests AI assistant with no mode buttons and backend-gated Apply."

language:
  implementation: jinja_js_css_python
  agent_report: ru

context_sources:
  always_read:
    - path: .codex/AGENTS.md
      why: "repo contract"
    - path: docs/architecture/backtest/backtest-ai-configurator-assistant-v1.md
      why: "UI contract and Iteration 06"
    - path: docs/architecture/backtest/benchmark_iterations/2026-05-17_ai_configurator_assistant_v1/implementation_progress.md
      why: "Iteration 05 human-readable gate"
    - path: docs/architecture/backtest/benchmark_iterations/2026-05-17_ai_configurator_assistant_v1/implementation_progress.json
      why: "Iteration 05 gate"
  task_entrypoints:
    - path: apps/web/templates/pages/backtests.html
      why: "AI block markup"
    - path: apps/web/dist/js/pages/backtests.js
      why: "AI chat behavior"
    - path: apps/web/locales/en.json
      why: "English startup copy"
    - path: apps/web/locales/ru.json
      why: "Russian startup copy"
  conditional_bundles:
    api_contract:
      read_when: "message/load endpoints are unclear"
      paths:
        - docs/architecture/backtest/benchmark_iterations/2026-05-17_ai_configurator_assistant_v1/conversation_api_contract.md
        - apps/api/routes/backtest_ai_config.py
    web_tests:
      read_when: "updating browser tests"
      paths:
        - tests/unit/apps/web/
        - tests/e2e/

hard_requirements:
  previous_iteration_accepted_required: true
  remove_mode_buttons: true
  startup_message_platform_language: true
  assistant_answer_user_prompt_language: true
  no_chain_of_thought_visible: true
  apply_config_only_backend_ready: true
  direct_main_publish_after_acceptance: true

task_toggles:
  implement_ui: true
  implement_backend_api: false

skill_routing:
  - skill: browser-qa-evidence
    use_when: "verifying /backtests UI, responsive layout, Apply behavior, and console/network"
    timing: "during verification"
    reason: "browser-visible behavior requires runtime evidence"
  - skill: contract-impact-analysis
    use_when: "changing browser API payloads, load action, or locale behavior"
    timing: "before final report"
    reason: "browser/API compatibility"
  - skill: backend-quality-gates
    use_when: "running web/API tests"
    timing: "during verification"
    reason: "regression coverage"
  - skill: publish-ci-deploy
    use_when: "local/browser/Mac Studio UI evidence passes and marker accepted=true"
    timing: "before final report"
    reason: "publish accepted changes and verify accepted commit on Mac Studio"

target_envs: [local-dev, mac-studio, browser]

required_literals:
  - "Apply configuration"
  - "Применить конфигурацию"
  - "New chat"
  - "History"
  - "queued"
  - "preparing_context"
  - "generating"
  - "validating"
  - "repairing"
  - "ready"

non_goals:
  - "Do not expose model reasoning or chain-of-thought."
  - "Do not add separate mode buttons."
  - "Do not let Apply run a backtest."

final_report_format:
  language: ru
  sections: ["Что изменено", "UI contract", "Browser QA", "Проверки", "Mac Studio", "Delivery"]

quality_gates:
  - cmd: "uv run pytest -q tests/unit/apps/web tests/unit/apps/api"
    expect: "focused tests pass"
  - cmd: "uv run ruff check apps/web apps/api tests/unit/apps/web tests/unit/apps/api"
    expect: "passes"
  - cmd: "uv run pyright"
    expect: "passes or unrelated pre-existing failures classified"
  - cmd: "uv run python -m tools.docs.generate_docs_index --check"
    expect: "passes"

expected_primary_touches:
  - apps/web/templates/pages/backtests.html
  - apps/web/dist/js/pages/backtests.js
  - apps/web/locales/en.json
  - apps/web/locales/ru.json
  - tests/unit/apps/web/
  - docs/architecture/backtest/benchmark_iterations/2026-05-17_ai_configurator_assistant_v1/ui_acceptance.md
  - docs/architecture/backtest/benchmark_iterations/2026-05-17_ai_configurator_assistant_v1/iteration_06_ui.md
  - docs/architecture/backtest/benchmark_iterations/2026-05-17_ai_configurator_assistant_v1/iteration_06_ui.json
  - docs/architecture/backtest/benchmark_iterations/2026-05-17_ai_configurator_assistant_v1/implementation_progress.md
  - docs/architecture/backtest/benchmark_iterations/2026-05-17_ai_configurator_assistant_v1/implementation_progress.json

safety_notes:
  - "Apply loads values into the form only; it does not submit preflight or run optimization."
  - "Status chips are allowed; hidden reasoning is not."
---

# Task

Implement Iteration 06: replace mode-based AI block with a single chat assistant UI.

## Requirements (Must)

- Stop if Iteration 05 is not accepted.
- Also stop if the previous iteration accepted commit is not recorded as pushed to `origin/main` and verified on Mac Studio in its evidence/progress marker.
- Remove Create/Edit/Explain/Repair/Safer mode row from UI/current locales/tests.
- Add default assistant startup message in platform-selected language.
- Add `New chat` and history UI inside the existing AI panel: list conversations, show title, last message time, terminal status, and allow opening an existing conversation.
- Submit one chat input; backend classifies intent.
- Show stage statuses only: queued, preparing_context, generating, validating, repairing, ready/error.
- Do not show model reasoning.
- Show `Apply configuration` only for backend `ready` with validated config/load action.
- Applying config fills current `/backtests` form and does not run backtest/preflight automatically.
- Explicit indicator params must use discrete controls/chips/selects when the catalog has explicit values; no-window indicators must not show synthetic `from/to/step` fields.
- Browser QA must cover desktop and narrow viewport.
- Create UI evidence and progress updates.
- After accepted evidence, use `publish-ci-deploy`; sync/verify accepted commit on Mac Studio.
- Delivery contract: use `publish-ci-deploy` in explicit direct-main mode only. Do not create a feature branch, draft PR, or PR branch. Stage only scoped files, commit on `main`, and push to `origin/main` only after all gates pass and evidence has `accepted=true`; wait for relevant main CI/deploy; then pull/sync the exact commit on Mac Studio and run the iteration-specific smoke. If direct main push, CI, or Mac Studio verification cannot be completed, set `accepted=false`, `next_iteration_allowed=false`, and report the blocker.

# Acceptance criteria (Definition of Done)

- Browser screenshots/evidence show no mode buttons.
- Browser screenshots/evidence show `New chat` and history UI.
- Startup copy respects platform language; assistant response language follows user prompt in integration smoke where backend supports it.
- Apply appears only after backend ready.
- Explicit/no-window indicator controls match catalog axis semantics.
- Console/network has no unexpected errors.
- Evidence JSON has `accepted=true`, `next_iteration_allowed=true`, `pushed_to_main=true`, and `macstudio_verified=true` after delivery.

# Final output: report format (strict)

Report in Russian with screenshots/evidence paths, tests, Mac Studio browser/API smoke, and delivery status.
