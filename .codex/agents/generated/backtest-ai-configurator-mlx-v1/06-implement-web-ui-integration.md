---
prompt_name: backtest_ai_configurator_mlx_v1_06_web_ui_integration
repo: roehub.com
branch: main
scope: "Iteration 06: enable /backtests AI chat panel, mode selector, SSE/polling status, safe assistant rendering, load-configuration action, feedback, and RU/EN data notice."

language:
  implementation: fastapi_jinja_plain_js_css
  agent_report: ru

context_sources:
  always_read:
    - path: .codex/AGENTS.md
      why: "browser-visible verification rules"
    - path: docs/architecture/backtest/backtest-ai-configurator-mlx-v1.md
      why: "UI and load-action contract"
  task_entrypoints:
    - path: apps/web/templates/pages/backtests.html
      why: "AI block and current form controls"
      inspect_symbols:
        - data-backtests-root
        - data-ai-log
    - path: apps/web/dist/js/pages/backtests.js
      why: "current form payload and apply draft helpers"
      inspect_symbols:
        - buildRequestPayload
        - seedConfigDraft
    - path: apps/api/wiring/modules/ui_backtests.py
      why: "ai_configurator_state and workstation payload"
      inspect_symbols:
        - _build_config_draft
        - _build_indicator_catalog
    - path: apps/api/routes/backtest_ai_config.py
      why: "AI API contract to consume from browser"
      inspect_symbols:
        - "*"
  conditional_bundles:
    locale_strings:
      read_when: "when adding RU/EN labels, notices or messages"
      paths:
        - apps/web
    browser_tests:
      read_when: "when updating app route or asset tests"
      paths:
        - tests/unit/apps/web
        - tests/unit/apps/api/test_ui_backtests_routes.py
    ui_assets:
      read_when: "when CSS changes are needed for AI panel"
      paths:
        - apps/web/dist/css
        - apps/web/dist/js/core
  consult_if_needed:
    - path: docs/architecture/apps/web/web-ui-design-manifest-v1.md
      read_when: "if visual/control style conflicts"
    - path: .codex/agents/.context/promt_manager_state.yaml
      read_when: "only to check for a newer executor handoff; ignore if stale/unrelated"

style_references:
  - path: apps/web/dist/js/pages/backtests.js
    purpose: "plain JS module style"
  - path: apps/web/templates/pages/backtests.html
    purpose: "existing workstation layout"

hard_requirements:
  depends_on_iteration_05: true
  no_react_or_vite: true
  ai_only_on_backtests: true
  no_auto_backtest_job: true
  safe_text_rendering: true
  browser_qa_required: true
  ru_en_required: true

task_toggles:
  implement_ui: true
  enable_ai_configurator_state: true
  implement_sse_or_polling: true
  implement_load_configuration: true
  implement_feedback: true
  implement_mlx: false
  implement_ops: false

skill_routing:
  - skill: contract-impact-analysis
    use_when: "flipping ai_configurator_state.enabled or changing workstation/browser-visible behavior"
    timing: "before implementation"
    reason: "browser-visible and DTO compatibility"
  - skill: backend-quality-gates
    use_when: "running API/web unit tests and type/lint checks"
    timing: "during verification"
    reason: "backend and web unit gates"
  - skill: browser-qa-evidence
    use_when: "after local server is running and UI changes are implemented"
    timing: "during verification"
    reason: "visible /backtests behavior requires browser evidence"
  - skill: playwright
    use_when: "capturing browser evidence if Browser plugin is unavailable"
    timing: "during verification"
    reason: "automated browser QA fallback"

target_envs:
  - local-dev
  - browser
  - unit-tests

required_literals:
  - "/api/backtests/ai-config/jobs"
  - "/api/backtests/ai-config/jobs/{job_id}/events"
  - "Загрузить конфигурацию"
  - "Load configuration"
  - "queued"
  - "preparing_catalog"
  - "generating"
  - "validating"
  - "repairing"
  - "ready"

non_goals:
  - "Do not auto-run a backtest job from AI output."
  - "Do not stream raw model JSON to the browser."
  - "Do not render assistant text as HTML."
  - "Do not convert the page to React/Vite/SPA."
  - "Do not implement production launchd/Monit."

final_report_format:
  language: ru
  sections:
    - "Что реализовано"
    - "Browser-visible contract"
    - "Проверки"
    - "Browser QA evidence"
    - "Следующая итерация"

quality_gates:
  - cmd: "uv run pytest -q tests/unit/apps/web tests/unit/apps/api/test_ui_backtests_routes.py tests/unit/apps/api/test_backtest_ai_config_routes.py"
    expect: "passes"
  - cmd: "uv run ruff check apps/web apps/api src/trading/contexts/backtest tests/unit/apps"
    expect: "passes"
  - cmd: "uv run pyright"
    expect: "passes"
  - cmd: "git diff --check"
    expect: "passes"

expected_primary_touches:
  - "apps/web/templates/pages/backtests.html"
  - "apps/web/dist/js/pages/backtests.js"
  - "apps/web/dist/css/pages/backtests.css"
  - "apps/api/wiring/modules/ui_backtests.py"
  - "tests/unit/apps/web/test_backtests_ai_configurator.py"
  - "tests/unit/apps/api/test_ui_backtests_routes.py"

possible_secondary_touches:
  - "apps/web/dist/js/core/sse.js"
  - "apps/web/dist/js/core/poller.js"
  - "apps/web/translations"
  - "tests/unit/apps/web/test_app_routes.py"

safety_notes:
  - "Use textContent or equivalent safe rendering for assistant messages."
  - "Load button label comes from trusted locale text, not model output."
  - "AI load action fills current form only; existing run button remains the only job creation action."
---

# Task

Implement Iteration 06 of the `/backtests` AI Configurator: browser UI integration inside the existing `/backtests` workstation. Add chat input, mode selection, status timeline, SSE/polling updates, typewriter-style final assistant message, safe `Load configuration` action, feedback, and RU/EN data notice.

Done means:

- `ai_configurator_state.enabled` can be true by config/wiring when backend AI routes are available;
- AI panel lets user choose/create modes or otherwise send mode with prompt;
- UI posts prompt/current config/idempotency key to `/api/backtests/ai-config/jobs`;
- UI listens to SSE events or falls back to polling;
- UI shows observable stages only, never chain-of-thought;
- final assistant text is rendered safely as text;
- load button appears only for `status=ready` and backend `load_action.enabled=true`;
- clicking load fills the existing `/backtests` form and does not create a job;
- feedback event records `applied=true`;
- RU and EN data notice tells users prompts/outputs may be saved to improve the configurator.

## Context / Current State

Context ledger:

- completed:
  - Iteration 05 should provide backend AI routes and worker runtime behind feature/config.
- open_items:
  - production operations, Prometheus/Monit and load testing are later.
- contract_changes:
  - browser-visible AI block becomes enabled;
  - existing manual form/run flow remains unchanged.
- risks:
  - assistant text XSS if rendered as HTML;
  - accidental auto-submit if load action calls existing create job flow;
  - multi-symbol prompt might not fit single-symbol form.
- next_focus:
  - user-visible integration and browser evidence.

## Requirements (Must)

- Preserve FastAPI SSR + Jinja2 + plain JS stack.
- Keep AI only inside `/backtests`.
- Send current form snapshot using existing `buildRequestPayload` semantics.
- Apply validated config through existing form setters/dropdown option validation.
- Never call `/api/backtests/jobs` from AI flow.
- Render assistant-controlled text as text, never `innerHTML`.
- Show status events like `queued`, `preparing_catalog`, `generating`, `validating`, `repairing`, `ready`.
- Show no raw JSON/model output in chat.
- Add data notice in RU and EN.
- Add tests for AI panel enabled/disabled state, route payload attributes, safe rendering or asset literals, and no auto-run.
- Run browser QA on `/backtests` after implementation.

## Requirements (Should)

- Use SSE when available; fallback polling is acceptable and should not overlap requests.
- Add cancel/retry only if backend already supports it; otherwise do not invent scope.
- Keep text concise and utilitarian, matching existing workstation style.
- Disable submit while a job is active for the same prompt.

## Requirements (Nice-to-have)

- Persist `applied=false` when user dismisses/replaces a ready config, if easy.
- Add lightweight client-side idempotency key generation.

# Context acquisition protocol

Read only in this order and stop once sufficient:

1. `.codex/AGENTS.md`
2. `.codex/agents/.context/promt_manager_state.yaml` or latest state snapshot, if available and relevant
3. latest executor final report, if available
4. task entrypoints
5. only conditional bundles required by locale/tests/UI assets
6. consult-if-needed references only for visual/control conflicts

Do not eagerly preload all listed sources.

Pre-implementation reading target:

- `<= 8 files`
- `<= ~35k-50k tokens`

Stop reading once AI panel markup, JS apply flow, route contract and locale strings are clear.

# Reading manifest

Use the front-matter `context_sources` as the canonical reading map.

Read with this intent:

- `always_read`: repo rules and UI contract.
- `task_entrypoints`: template, JS, workstation payload and API routes.
- `conditional_bundles`: locale, tests and CSS only when needed.
- `consult_if_needed`: design manifest only for style conflicts.

Do not convert this manifest into a broad mandatory reading list.

# Work plan (agent should follow)

Skill routing for this task:

- `contract-impact-analysis`: use before implementation; owns browser-visible behavior and workstation DTO.
- `backend-quality-gates`: use during unit verification.
- `browser-qa-evidence`: use after implementation with a real browser target.
- `playwright`: use if Browser plugin is unavailable for browser evidence.

1. Verify backend AI route shape and current form payload/apply helpers.
2. Add/enable AI panel controls, mode handling, data notice and safe message renderer.
3. Add job create/status/event client logic with SSE or polling fallback.
4. Implement load-configuration action and feedback call.
5. Add unit tests for UI assets/routes and API integration assumptions.
6. Start local server if required and run browser QA for RU/EN, ready state, load action, no auto-run and no console/network errors.
7. Run quality gates and report evidence.

# Acceptance criteria (Definition of Done)

- `/backtests` shows enabled AI configurator when backend state enables it.
- Safe prompt can move through visible statuses and show final assistant message.
- Ready response displays `Загрузить конфигурацию`/`Load configuration`.
- Load action updates symbol/timeframe/risk/execution/indicator fields supported by current form.
- No backtest job is created until user presses existing run button.
- Malicious assistant text sample is rendered as inert text.
- RU and EN notice exists.
- Browser QA evidence includes console/network check.

# Implementation constraints

## Determinism & ordering

- Client polling/SSE must avoid overlapping requests.
- Stale responses must not overwrite a newer active job.

## API / contracts

- Browser-visible path is `/api/backtests/ai-config/*`.
- Do not add a second `/api` prefix to backend route paths.

## Accessibility and rendering

- Keep aria-live/status feedback for async messages.
- Use stable dimensions for the AI panel to avoid layout jumps.

# Files to indicate (expected touched areas)

Expected primary touches:

- `apps/web/templates/pages/backtests.html`
- `apps/web/dist/js/pages/backtests.js`
- `apps/web/dist/css/pages/backtests.css`
- `apps/api/wiring/modules/ui_backtests.py`
- `tests/unit/apps/web/test_backtests_ai_configurator.py`
- `tests/unit/apps/api/test_ui_backtests_routes.py`

Possible secondary touches:

- `apps/web/dist/js/core/sse.js`
- `apps/web/dist/js/core/poller.js`
- `apps/web/translations`
- `tests/unit/apps/web/test_app_routes.py`

# Non-goals

- No MLX/runtime changes.
- No Monit/Prometheus.
- No benchmark harness.
- No generic AI page.
- No automatic backtest job creation.

# Quality gates (must run and pass)

- `uv run pytest -q tests/unit/apps/web tests/unit/apps/api/test_ui_backtests_routes.py tests/unit/apps/api/test_backtest_ai_config_routes.py`
- `uv run ruff check apps/web apps/api src/trading/contexts/backtest tests/unit/apps`
- `uv run pyright`
- `git diff --check`
- Browser QA with available browser surface: `/backtests`, RU and EN states, console/network clean, no auto-run.

If browser QA cannot run, state the exact blocker and do not claim browser-visible behavior works.

# Final output: report format (strict)

Report in Russian with:

- `Что реализовано`: UI controls, job flow, load action, feedback, notice.
- `Browser-visible contract`: status rendering, no auto-run, safe rendering.
- `Проверки`: exact commands and results.
- `Browser QA evidence`: browser target, steps, console/network result.
- `Следующая итерация`: observability, training export and ops.
