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

style_references:
  design_manifest:
    path: docs/architecture/apps/web/web-ui-design-manifest-v1.md
    purpose: "визуальный source of truth для токенов, тем, layouts, density и accessibility"
  external_reference_root:
    path: /Users/daniildegtyarev/Projects/roehub_web_ui
    purpose: "reference screenshots/assets; inspect only stage-relevant pages"
  default_palette: terminal-orange
  theme_variants:
    - terminal-orange
    - graphite
    - matrix-green
    - high-contrast
  invariant_financial_colors: true
  default_locale: en
  secondary_locale: ru
  language_switch_required: true

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

package_contract:
  depends_on:
    - "01-shell-auth-register accepted"
    - "02-design-system-js-core accepted"
  owns:
    - "apps/api/routes/ui_backtests.py"
    - "apps/api/dto/ui_backtests.py"
    - "apps/api/wiring/modules/ui_backtests.py"
    - "apps/api/routes/backtests.py additive history/configurator behavior only"
    - "apps/web/templates/pages/backtests_history.html"
    - "apps/web/templates/pages/backtests_run.html"
    - "apps/web/templates/fragments/backtests/**"
    - "apps/web/dist/js/pages/backtests_history.js"
    - "apps/web/dist/js/pages/backtests_run.js"
    - "apps/web/dist/css/pages/backtests.css"
    - "tests/unit/apps/api/test_ui_backtests_routes.py"
  forbidden:
    - "backtest runtime worker hardening"
    - "backtest results/statistics page"
    - "AI configurator"
    - "canonical request hash changes"
  integration_points:
    - "existing /api/backtests/jobs/preflight/defaults routes"
    - "Idempotency-Key semantics"
    - "backtest_presets migration if implemented"
    - "apps/api/main.py route include"
  handoff:
    - "split history/configurator pages and safe request draft flow"

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
  - "Intent: что реализовано и почему это нужно пользователю"
  - "Scope: bounded capability, routes, modules, files, owns/forbidden compliance"
  - "Design: use cases, DTO, ports/adapters, migrations, JS modules, template fragments"
  - "Contract impact: public API, port, DTO, persisted schema, config, cache/request identity, browser-visible behavior, performance risk"
  - "Tests: exact commands, cwd, results, focused/lint/type/migration gates"
  - "Docs: updated docs or explicit reason no docs changed"
  - "Performance: touched hot paths, payload/latency/RSS/load checks, or explicit none"
  - "Runtime evidence: Playwright/browser, tests, inference, assumptions clearly separated"
  - "Risks: edge cases, migration/rollback, pre-existing/environmental/flaky failures"
  - "Handoff: stable exports, route includes, helpers, endpoint contracts for next agents"
  - "Publish/deploy: publish-ci-deploy terminal state; if successful, include main merge, local sync, Mac Studio git pull, impacted service restart/reload, and smoke verification evidence; otherwise exact reason it was skipped"

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
If `.codex/agents/.context/promt_manager_state.yaml` or a latest executor final report for this pack exists, read only its completed/open_items/risks/handoff summary before task entrypoints; skip this step if absent.

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

## Agent package boundaries

- Treat `package_contract.owns` as the write allow-list for this prompt.
- Do not edit `package_contract.forbidden` areas. If an implementation truly needs one, stop and report the required integration point instead of broadening scope silently.
- Keep shared integration edits small and explicit: route includes, DTO exports, CSS tokens, JS core APIs, migration chain, edge config.
- In final report, state whether the diff stayed inside `owns`; list any integration-point edits separately.

## API endpoint specification checklist

Before coding any new endpoint or browser-visible API addition, write the local contract in the implementation notes/tests with:

- `method/path`: browser-visible `/api/...` path and actual backend router path without duplicate `/api` prefix;
- `owner scope`: current user/account resolution and authorization check;
- `request DTO`: required/optional fields, defaults, validation, idempotency key, size limits;
- `response DTO`: shape, nullable fields, enums, links, timestamps, pagination;
- `status codes`: expected `200/201/204/400/401/403/404/409/422/429/500/503` semantics where applicable;
- `error payload`: compatible `RoehubError` envelope, field errors, retryability/correlation id when available;
- `pagination`: cursor/keyset/page semantics, max limit, stable ordering, or explicit `none`;
- `cache identity`: request hash/cache key/persistence identity impact or explicit `none`;
- `compatibility`: `none`, `compatible-change`, `breaking-change`, or `unknown` with migration/deprecation notes.

## Browser runtime evidence checklist

For every browser-visible change, collect and report runtime evidence:

- desktop screenshot, normally around `1440x1000`;
- mobile screenshot, normally around `390x844`;
- `snapshot` after the key state;
- console errors absent;
- failed same-origin network requests absent except expected auth redirects;
- auth state/protected route behavior verified when the page is protected;
- theme switcher changes base/accent/state but not financial colors;
- primary workflow has no overlapping requests;
- chart/canvas/SVG pages include a nonblank check;
- final report separates observed browser evidence, automated test evidence, inference, and assumptions.

## Gate failure classification

- Classify every failing gate as `introduced`, `required-path pre-existing`, `unrelated pre-existing`, `environmental`, or `flaky`.
- Do not run `publish-ci-deploy` with unresolved `introduced` failures or missing required browser/performance evidence.
- If a failure is pre-existing or environmental, include exact command, failure summary, and why it does or does not block this stage.

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

# i18n / language contract

The Web UI v1 is multilingual. Every prompt in this pack must preserve this contract:

- default locale is `en`; secondary locale is `ru`;
- any new user-visible copy introduced by this stage must have both `en` and `ru` strings through the shared locale catalog/helper;
- do not localize routes, `/api/*` paths, DTO fields, enum values, market symbols, strategy ids, `job_id`, `variant_key`, config keys, or metric identifiers;
- rendered pages must keep `<html lang>` and root `data-locale` aligned with the selected locale;
- the language switcher must remain available from shell/account controls and must not compete with primary navigation;
- browser QA for any stage that adds or changes visible copy must include default `en` evidence and either `ru` locale-switch evidence or an explicit blocker;
- final report must state i18n impact: locale keys/catalogs touched, fallback behavior, and whether language-switch evidence was collected.

# publish-ci-deploy terminal delivery contract

When all stage DoD, gates, browser evidence, and performance evidence required by this prompt pass, and `publish_after_success` is true, run `publish-ci-deploy` to the natural terminal state. A successful terminal state for this prompt means more than PR creation, green CI, or deploy workflow completion. It must include, when the agent has authority and no external blocker remains:

- branch/PR merged into `main`, or exact blocker why merge is outside current authority;
- local checkout synchronized with `origin/main`;
- Mac Studio repository checkout synchronized with `origin/main` using `git pull --ff-only` from the actual repo checkout, normally `/Users/daniildegtyarev/Projects/roehub.com`;
- deployed runtime updated through the repository deploy/runbook path, keeping the repo checkout and runtime bundle as separate surfaces when they differ;
- impacted services restarted only when touched-path impact requires it; if impact is unclear, use the standard prod reload path from `publish-ci-deploy`;
- post-restart smoke verification completed;
- final report names exact commands, host/paths used, commit SHA on `main`, restarted services, smoke result, or exact blocker.

Do not report successful publish/deploy while merge to `main`, Mac Studio git pull, required service restart/reload, or smoke verification remains pending.

# Final output: report format (strict)

Report in Russian with these exact sections:

- `Intent`: что реализовано и почему это нужно пользователю.
- `Scope`: bounded capability, routes, modules, files, and `owns`/`forbidden` compliance.
- `Design`: use cases, DTO, ports/adapters, migrations, JS modules, template fragments.
- `Contract impact`: classify public API, port, DTO, persisted schema, config, request hash/cache identity, browser-visible behavior, performance risk.
- `Tests`: exact commands, cwd, result, focused gates, lint/type gates, migration gates.
- `Docs`: docs changed, docs index result, or explicit reason docs were not changed.
- `Performance`: hot path impact, payload/latency/RSS/load checks, or explicit `none`.
- `Runtime evidence`: Playwright/browser evidence, automated test evidence, inference, assumptions.
- `Risks`: edge cases, migration/rollback risks, pre-existing/environmental/flaky failures.
- `Handoff`: stable exports, route includes, shared helpers, endpoint contracts for next agents.
- `Publish/deploy`: whether `publish-ci-deploy` ran, terminal state, or exact reason it was skipped.
