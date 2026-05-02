---
prompt_name: web_ui_backend_v1_08_backtests_history_configurator
repo: roehub.com
branch: current
scope: "Этап 8: разделить backtests на историю и конфигуратор, добавить presets/counters при необходимости."

language:
  implementation: python_fastapi_jinja_css_js_alembic
  agent_report: ru

context_sources:
  always_read:
    - path: .codex/AGENTS.md
      why: "contracts, DDD, gates, browser verification"
    - path: docs/architecture/apps/web/web-ui-backend-implementation-plan-v1.md
      why: "Этап 8 source of truth"
    - path: docs/architecture/apps/web/web-ui-design-manifest-v1.md
      why: "backtest configurator visual rules"
  task_entrypoints:
    - path: apps/api/routes/backtests.py
      why: "existing public jobs/preflight/runtime-defaults API"
    - path: apps/api/dto/backtests.py
      why: "current backtest DTO surface"
    - path: src/trading/contexts/backtest/application/use_cases/backtest_jobs.py
      why: "job create/list/cancel/idempotency behavior"
    - path: apps/web/templates/backtests.html
      why: "current combined page replacement target"
    - path: apps/web/dist/backtest_ui.js
      why: "current combined JS replacement target"
  conditional_bundles:
    presets_persistence:
      read_when: "if implementing `backtest_presets`"
      paths:
        - alembic/versions
        - apps/migrations/main.py
        - src/trading/contexts/backtest/application/ports
    market_indicators:
      read_when: "if configurator reference dropdown behavior is unclear"
      paths:
        - apps/api/routes/market_data_reference.py
        - apps/api/routes/indicators.py
    tests:
      read_when: "when adding API/web tests"
      paths:
        - tests/unit/apps/api/test_backtests_routes.py
        - tests/unit/apps/web/test_app_routes.py
  consult_if_needed:
    - path: docs/architecture/backtest/backtest-service-artifact-runtime-v1.ru.md
      read_when: "if request hash, jobs vocabulary, or lazy result boundaries are ambiguous"

hard_requirements:
  split_backtests_page: true
  history_cursor_pagination: true
  configurator_uses_runtime_defaults: true
  preflight_advisory_create_authoritative: true
  idempotency_key_required_for_create: true
  no_full_results_or_trades_on_configurator: true
  backtest_presets_owner_scoped_if_implemented: true
  browser_qa_required: true

task_toggles:
  implement_history_page: true
  implement_configurator_page: true
  implement_presets_optional: true
  implement_job_events_optional: true
  implement_runtime_hardening: false
  publish_after_success: true

skill_routing:
  - skill: contract-impact-analysis
    use_when: "touching `/backtests/jobs`, idempotency, request hash/cache identity, presets schema, DTOs, browser-visible backtest defaults"
    timing: "before implementation and final report"
    reason: "backtest jobs and request hash are core public contracts"
  - skill: backend-quality-gates
    use_when: "running backtest API/web/migration tests, ruff, pyright"
    timing: "during verification"
    reason: "stage has API, web, and optional migration surfaces"
  - skill: browser-qa-evidence
    use_when: "verifying `/backtests`, `/backtests/new`, create/preflight/cancel/history flow, console/network, responsive layout"
    timing: "after backend tests"
    reason: "backtest configurator/history are browser-visible"
  - skill: playwright
    use_when: "capturing screenshots/snapshots"
    timing: "during browser QA"
    reason: "plan requires Playwright CLI"
  - skill: publish-ci-deploy
    use_when: "all tests, browser QA, idempotency/request-hash checks pass"
    timing: "after verification"
    reason: "full Roehub delivery chain after complete success"

target_envs:
  - local-dev
  - browser
  - github-actions

required_literals:
  - "/backtests"
  - "/backtests/new"
  - "/api/backtests/runtime-defaults"
  - "/api/backtests/preflight"
  - "/api/backtests/jobs"
  - "Idempotency-Key"
  - "backtest_presets"
  - "request_hash"

non_goals:
  - "Do not implement results/statistics page; Stage 9 owns it."
  - "Do not harden sync_inline runtime; Stage 8.5 owns it."
  - "Do not change canonical backtest request hashing."
  - "Do not load full results/trades on configurator."

final_report_format:
  - "Summary: что сделано"
  - "Files changed: пути и назначение"
  - "Contracts: classification and API/schema/UI impact"
  - "Verification: команды, Playwright evidence, результаты"
  - "Publish/deploy: terminal state publish-ci-deploy или причина пропуска"
  - "Risks / follow-up: остаточные риски"

quality_gates:
  - cmd: "uv run pytest -q tests/unit/apps/api/test_backtests_routes.py tests/unit/apps/api/test_ui_backtests_routes.py tests/unit/apps/web/test_app_routes.py"
    expect: "passes; create focused tests if missing"
  - cmd: "uv run ruff check apps/api apps/web src/trading/contexts/backtest tests/unit/apps/api tests/unit/apps/web"
    expect: "passes for touched paths"
  - cmd: "uv run pyright"
    expect: "passes"
  - cmd: "python -m tools.docs.generate_docs_index --check"
    expect: "passes if docs changed"

expected_primary_touches:
  - "apps/api/routes/ui_backtests.py"
  - "apps/api/dto/ui_backtests.py"
  - "apps/api/wiring/modules/ui_backtests.py"
  - "apps/api/routes/backtests.py"
  - "src/trading/contexts/backtest/**"
  - "alembic/versions/*.py"
  - "apps/web/templates/pages/backtests_history.html"
  - "apps/web/templates/pages/backtests_run.html"
  - "apps/web/templates/fragments/backtests/*"
  - "apps/web/dist/js/pages/backtests_history.js"
  - "apps/web/dist/js/pages/backtests_run.js"
  - "apps/web/dist/css/pages/backtests.css"
  - "tests/unit/apps/api/test_ui_backtests_routes.py"
  - "tests/unit/apps/web/test_app_routes.py"

possible_secondary_touches:
  - "apps/api/main/app.py"
  - "apps/api/wiring/modules/__init__.py"
  - "docs/architecture/backtest/backtest-service-artifact-runtime-v1.ru.md"

safety_notes:
  - "Presets belong to Alembic/application DB unless a separate design decision moves them to identity DB."
  - "`POST /api/backtests/jobs` is browser path; backend route is `/backtests/jobs`."
  - "Preflight is advisory. Create repeats validation."
---

# Task

Implement Stage 8 backtests history and configurator split.

Done means:

- `/backtests` shows only history/list with cursor pagination;
- `/backtests/new` shows configurator using runtime defaults/reference endpoints;
- create flow sends `Idempotency-Key`;
- invalid requests never create jobs;
- presets/counters exist only if implemented through proper contracts;
- old combined `backtests.html` and `backtest_ui.js` are no longer long-term dependencies for these routes;
- browser evidence exists.

## Context / Current State

- Current `/backtests` combines form, jobs, selected job and trades.
- Backend already has runtime-defaults, preflight, jobs, top, cancel.
- Backtest create still may execute inline; Stage 8.5 hardens runtime path separately.

## Requirements (Must)

- Preserve jobs vocabulary and public API compatibility.
- Preserve canonical request hash/cache identity.
- Keep full results/trades out of history/configurator first paint.
- Use cursor pagination for history.
- Add tests for preflight valid/invalid, idempotency replay/conflict, cancel UX, presets if implemented.
- Use `publish-ci-deploy` only after full success.

## Requirements (Should)

- Keep browser polling no-overlap and hidden-tab aware.
- Use shared JS core and design components.
- Keep request drafts safe and owner-scoped.

## Requirements (Nice-to-have)

- Add `GET /api/ui/backtests/counters` if it simplifies toolbar badges without bloating history payload.

# Context acquisition protocol

Read `.codex/AGENTS.md`, plan Stage 8, backtest artifact runtime doc if needed, then task entrypoints. Expand into persistence only if presets are implemented.

Reading budget: keep pre-implementation reading to the smallest sufficient set; default target `<= 8 files`, `<= ~45k tokens` unless this prompt states a tighter number.
Stop reading when touched files, contract surfaces, and acceptance gates are bounded enough to implement safely.
Do not eager-load all `context_sources`, `conditional_bundles`, or `consult_if_needed` files at startup.

# Reading manifest

Use front matter `context_sources`.

# Work plan (agent should follow)

1. Split routes/templates for history and configurator.
2. Define optional UI backtest DTOs for presets/counters.
3. Implement backend additions only when required.
4. Implement page JS/CSS with bounded polling and no full results payloads.
5. Add focused tests.
6. Run browser QA and quality gates.
7. Use `publish-ci-deploy` only after complete success.

# Acceptance criteria (Definition of Done)

- `/backtests` shows only history and paginates.
- `/backtests/new` builds valid request from runtime defaults/reference endpoints.
- Invalid request never creates job.
- Duplicate submit with same idempotency key returns same job.
- Cancel is UX-idempotent.
- History remains responsive with large job count.
- Full results/trades are not loaded on configurator.

# Implementation constraints

## API / contracts

- Public API contract: `compatible-change`.
- DTO schema: `compatible-change`.
- Persisted schema: `compatible-change` if presets are added.
- Request hash/cache identity: `none`.

# Files to indicate (expected touched areas)

Use front matter touched paths.

# Non-goals

- Backtest results page.
- Runtime worker hardening.
- AI configurator.

# Quality gates (must run and pass)

```bash
uv run pytest -q tests/unit/apps/api/test_backtests_routes.py tests/unit/apps/api/test_ui_backtests_routes.py tests/unit/apps/web/test_app_routes.py
uv run ruff check apps/api apps/web src/trading/contexts/backtest tests/unit/apps/api tests/unit/apps/web
uv run pyright
python -m tools.docs.generate_docs_index --check
```

Playwright CLI:

```bash
export CODEX_HOME="${CODEX_HOME:-$HOME/.codex}"
export PWCLI="$CODEX_HOME/skills/playwright/scripts/playwright_cli.sh"
"$PWCLI" open http://127.0.0.1:8010/backtests
"$PWCLI" snapshot
"$PWCLI" screenshot --filename output/playwright/backtests-history-desktop.png
"$PWCLI" open http://127.0.0.1:8010/backtests/new
"$PWCLI" snapshot
"$PWCLI" screenshot --filename output/playwright/backtests-run-desktop.png
```

# Final output: report format (strict)

Report in Russian: `Intent`, `Scope`, `Design`, `Contract impact`, `Tests`, `Runtime evidence`, `Risks`, `Handoff`, `Publish/deploy`.
