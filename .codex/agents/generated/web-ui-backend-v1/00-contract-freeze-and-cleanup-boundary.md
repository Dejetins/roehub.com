---
prompt_name: web_ui_backend_v1_00_contract_freeze_and_cleanup_boundary
repo: roehub.com
branch: main
scope: "Этап 0: зафиксировать контракты, route/endpoint map и границы удаления старого Web UI перед реализацией."

language:
  implementation: docs_inventory_python
  agent_report: ru

context_sources:
  always_read:
    - path: .codex/AGENTS.md
      why: "нормативный engineering contract, DDD, browser verification, contract impact, delivery rules"
    - path: docs/architecture/apps/web/web-ui-backend-implementation-plan-v1.md
      why: "source of truth для этапов, контрактов, API notation и параллелизации"
    - path: docs/architecture/apps/web/web-ui-design-manifest-v1.md
      why: "визуальный source of truth и правила тем"
  task_entrypoints:
    - path: apps/web/main/app.py
      why: "текущие web routes, protected gate, local /api proxy"
      inspect_symbols:
        - create_app
        - _register_routes
        - _render_protected_page
    - path: apps/web/templates
      why: "текущие templates, которые нужно классифицировать как replace/move/delete"
    - path: apps/web/dist
      why: "текущие CSS/JS assets, которые нужно классифицировать как replace/move/delete"
    - path: tests/unit/apps/web/test_app_routes.py
      why: "текущие route smoke patterns"
  conditional_bundles:
    api_surface:
      read_when: "если endpoint map неясен из плана или web routes"
      paths:
        - apps/api/main/app.py
        - apps/api/routes/backtests.py
        - apps/api/routes/strategies.py
        - apps/api/routes/identity.py
    edge_contract:
      read_when: "если меняется или уточняется same-origin /api contract"
      paths:
        - docs/runbooks/web-ui-gateway-same-origin.md
        - infra/caddy/Caddyfile.vps
  consult_if_needed:
    - path: docs/web-ui+backend-plan-deep-research.md
      read_when: "только если текущий план противоречит observed code"

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
  do_not_implement_page_ui: true
  classify_current_web_files: true
  freeze_route_map: true
  freeze_endpoint_map: true
  preserve_api_prefix_notation: true
  freeze_auth_modal_register_page_contract: true
  freeze_branded_controls_contract: true
  freeze_data_source_refresh_contract: true
  current_migrations_gap_inventory_required: true
  favicon_no_incidental_404_invariant: true
  user_badge_partial_route_not_public_contract: true
  physical_delete_only_after_route_replacement: true
  freeze_reference_fidelity_map: true
  update_docs_index_if_docs_change: true

task_toggles:
  write_docs_inventory: true
  edit_runtime_code: false
  delete_files: false
  publish_after_success: true

package_contract:
  depends_on:
    - "none"
  owns:
    - "docs/architecture/apps/web/web-ui-backend-implementation-plan-v1.md"
    - "docs/architecture/apps/web/web-ui-design-manifest-v1.md"
    - "docs/architecture/README.md"
  forbidden:
    - "apps/web/** runtime implementation"
    - "apps/api/** runtime implementation"
    - "src/** runtime implementation"
  integration_points:
    - "route map"
    - "endpoint map"
    - "reference fidelity map"
    - "auth modal/register page contract"
    - "branded controls contract"
    - "data source/refresh/autorefresh contract"
    - "cleanup inventory"
  handoff:
    - "accepted route/endpoint map and cleanup boundaries for Stage 1-2"

skill_routing:
  - skill: architecture-review
    use_when: "перед правками и при сравнении плана с текущими routes/assets"
    timing: "before implementation"
    reason: "этап является review-first фиксацией фактов и границ"
  - skill: contract-impact-analysis
    use_when: "если route map, endpoint map, browser-visible defaults или API prefix contract уточняются"
    timing: "before final report"
    reason: "контрактные поверхности должны быть классифицированы до передачи агентам"
  - skill: backend-quality-gates
    use_when: "при запуске docs index или web route tests"
    timing: "during verification"
    reason: "нужно отделить introduced/pre-existing failures"
  - skill: publish-ci-deploy
    use_when: "только если docs inventory завершен на 100%, все проверки прошли, diff содержит только scope этапа"
    timing: "after verification"
    reason: "полная Roehub delivery chain: publish, CI, deploy, post-deploy verification"

target_envs:
  - local-dev
  - github-actions

required_literals:
  - "replace"
  - "move"
  - "delete"
  - "/api/... browser-visible"
  - "terminal-orange"
  - "site.css"
  - "strategy_ui.js"
  - "backtest_ui.js"
  - "GET /favicon.ico"
  - "ANY /api/{upstream_path:path}"
  - "MOUNT /assets/*"
  - "GET /_partial/user_badge"
  - "protected_page.html"
  - "general_page.png -> /"
  - "personal_dashboard.png -> /dashboard"
  - "personal_settings.png -> /settings"
  - "strategy_statistic.png -> /strategies"
  - "stategy_backtest.png -> /backtests"
  - "login modal"
  - "register page"
  - "branded dropdown"
  - "autorefresh"
  - "retry_after_seconds"

non_goals:
  - "Не реализовывать новый UI, backend endpoints или миграции."
  - "Не удалять старые файлы физически на этом этапе."
  - "Не менять production edge routing."

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
  - "Publish/deploy: direct-main publish-ci-deploy terminal state; if successful, include direct push to origin/main, main CI/deploy monitoring, local main sync, Mac Studio git pull, impacted service restart/reload, and smoke verification evidence; otherwise exact blocker or reason it was skipped"

quality_gates:
  - cmd: "python -m tools.docs.generate_docs_index --check"
    expect: "passes"
  - cmd: "uv run pytest -q tests/unit/apps/web/test_app_routes.py"
    expect: "passes if route tests are touched or route inventory claims are changed"

expected_primary_touches:
  - "docs/architecture/apps/web/web-ui-backend-implementation-plan-v1.md"
  - "docs/architecture/README.md"

possible_secondary_touches:
  - ".codex/PLANS.md"
  - "tests/unit/apps/web/test_app_routes.py"

safety_notes:
  - "Этот этап должен дать stable handoff для следующих агентов, а не начать реализацию."
  - "Не stage unrelated local changes."
  - "Если найдено противоречие между `.codex/AGENTS.md` и планом, остановись и зафиксируй conflict."
  - "`GET /favicon.ico` должен оставаться без incidental browser 404: `204` совместим, позже можно заменить на static/versioned asset."
  - "`GET /_partial/user_badge` не переносится как public contract; Stage 1 должен заменить его shell component/fragment."
  - "Физическое удаление legacy files допускается только в owning stage, который уже заменил соответствующий route/page/asset."
---

# Task

Зафиксировать границы реализации Web UI v1 до старта coding-этапов.

Done means:

- текущие `apps/web` routes, templates и assets классифицированы как `replace`, `move` или `delete`;
- `GET /favicon.ico`, `ANY /api/{upstream_path:path}`, `MOUNT /assets/*` и `GET /_partial/user_badge` имеют явный target decision;
- `apps/web/templates/protected_page.html` классифицирован как `delete`, но без физического удаления на Stage 0;
- route map и endpoint map из плана сверены с текущим кодом;
- reference map закреплен ровно на 5 визуальных страницах: `general_page.png -> /`, `personal_dashboard.png -> /dashboard`, `personal_settings.png -> /settings`, `strategy_statistic.png -> /strategies`, `stategy_backtest.png -> /backtests`;
- `/monitoring` зафиксирован как compatibility/ops route only без canonical PNG в v1 map;
- `/api/...` нотация подтверждена как browser-visible contract, без добавления `/api` prefix в backend routers;
- `/_partial/user_badge` зафиксирован как non-public legacy partial route, который заменяется shell component/fragment;
- login зафиксирован как branded modal/deep-link modal state, а registration как отдельная `/register` page;
- branded dropdown/listbox/menu controls зафиксированы как обязательные; visible native select не является целевым UX;
- текущие migrations/read-model gaps для live dashboard/reference panels зафиксированы;
- manual refresh/autorefresh/rate-limit contract зафиксирован для live-data страниц;
- список handoff-инвариантов готов для этапов 1-2;
- docs index проходит.

## Context / Current State

- Новый план и design manifest уже существуют в `docs/architecture/apps/web/`.
- Текущий UI будет утилизирован и не сохраняется как legacy skin.
- `apps/web` сейчас содержит SSR routes, static `/assets`, встроенный local `/api/*` proxy и protected gate через `/api/auth/current-user`.
- Production edge снимает `/api` prefix перед backend API.

## Requirements (Must)

- Работай в review-first режиме: сначала факты, затем узкие docs-правки.
- Добавь или уточни в плане только то, что нужно для безопасной передачи этапов агентам.
- Не меняй runtime behavior.
- Не удаляй старые templates/assets физически.
- Не превращай `/_partial/user_badge` в стабильный публичный контракт.
- Не допускай возврата browser 404 noise для `GET /favicon.ico`.
- Классифицируй contract impact: public API, browser-visible behavior, persisted schema, config.
- Проверь текущие identity SQL migrations, Alembic migrations и ClickHouse DDL настолько, чтобы не заявлять несуществующие live-data источники.
- Зафиксируй, какие reference panels требуют новых read-model/persistence additions или typed degraded state.
- Если все проверки прошли и этап завершен на 100%, запусти delivery chain через `publish-ci-deploy`. Если нет, не публикуй и отчитай blocker.

## Requirements (Should)

- Сохрани документ на русском языке, технические identifiers оставь как есть.
- Держи diff минимальным.
- Зафиксируй open questions только если они реально блокируют следующий этап.

## Requirements (Nice-to-have)

- Добавить короткую таблицу `current file -> target decision`, если ее еще нет или она недостаточно явная.

# Context acquisition protocol

Read only in this order and stop once sufficient:

1. `.codex/AGENTS.md`
2. `docs/architecture/apps/web/web-ui-backend-implementation-plan-v1.md`
3. `docs/architecture/apps/web/web-ui-design-manifest-v1.md`
4. task entrypoints
5. conditional bundles only when the stated condition applies

Pre-implementation reading target: `<= 8 files`, `<= ~35k tokens`.

Stop reading once changed contracts are identified, touched files are bounded, and no route/API ambiguity remains.

Reading budget: keep pre-implementation reading to the smallest sufficient set; default target `<= 8 files`, `<= ~45k tokens` unless this prompt states a tighter number.
Stop reading when touched files, contract surfaces, and acceptance gates are bounded enough to implement safely.
Do not eager-load all `context_sources`, `conditional_bundles`, or `consult_if_needed` files at startup.
If `.codex/agents/.context/promt_manager_state.yaml` or a latest executor final report for this pack exists, read only its completed/open_items/risks/handoff summary before task entrypoints; skip this step if absent.

# Reading manifest

Use the front matter `context_sources` as the canonical reading map. Do not turn it into a repo-wide crawl.

# Work plan (agent should follow)

1. Inspect current web routes/templates/assets.
2. Compare them with the target route/file structure in the plan.
3. Record replace/move/delete decisions and handoff boundaries.
4. Update only the plan or directly related docs if gaps are found.
5. Run docs index check and relevant route tests if touched.
6. If and only if DoD is fully satisfied and all checks pass, use `publish-ci-deploy` for publish/CI/deploy/post-deploy verification.

# Acceptance criteria (Definition of Done)

- Every current web route has a target decision.
- Every current top-level template/asset has a replace/move/delete decision.
- Stage 1 and Stage 2 can start without public API or persistence ambiguity.
- Docs index check passes.
- Final report includes `Intent`, `Scope`, `Design`, `Contract impact`, `Tests`, `Docs`, `Performance`, `Runtime evidence`, `Risks`, `Handoff`.

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

- Do not alter public API behavior.
- Do not change same-origin production routing.

## Browser-visible behavior

- No browser claims are allowed unless runtime evidence exists; this stage is primarily docs/inventory.

# Files to indicate (expected touched areas)

- `docs/architecture/apps/web/web-ui-backend-implementation-plan-v1.md`
- `docs/architecture/README.md` if docs index changes

# Non-goals

- New CSS/JS implementation.
- New backend routes.
- File deletion.
- UI screenshots.

# Quality gates (must run and pass)

```bash
python -m tools.docs.generate_docs_index --check
```

Run focused route tests only if route tests or route behavior are touched:

```bash
uv run pytest -q tests/unit/apps/web/test_app_routes.py
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

# publish-ci-deploy direct-main delivery contract

When all stage DoD, gates, browser evidence, and performance evidence required by this prompt pass, and `publish_after_success` is true, run `publish-ci-deploy` in direct-main mode. For this prompt pack, do not create a delivery branch, draft PR, or PR-based merge path. Work is published directly to `main` only after local gates pass.

A successful terminal state for this prompt means more than local green or a pushed commit. It must include, when the agent has authority and no external blocker remains:

- executor is on an up-to-date `main`, or has stopped with an exact blocker explaining why direct-main publish is unsafe;
- only intended scope is staged and committed; unrelated local changes are preserved and not staged;
- mandatory local gates for the stage pass before push;
- commit is pushed directly to `origin/main`;
- GitHub Actions and deploy workflow for `main` are monitored to green; failing checks are inspected and fixed if attributable to this diff, otherwise reported as blocker;
- local checkout is synchronized with `origin/main` after the push/deploy flow;
- Mac Studio repository checkout is synchronized with `origin/main` using `git pull --ff-only` from the actual repo checkout, normally `/Users/daniildegtyarev/Projects/roehub.com`;
- deployed runtime is updated through the repository deploy/runbook path, keeping the repo checkout and runtime bundle as separate surfaces when they differ;
- impacted services are restarted only when touched-path impact requires it; if impact is unclear, use the standard prod reload path from `publish-ci-deploy`;
- post-restart smoke verification is completed;
- final report names exact commands, host/paths used, commit SHA on `main`, CI/deploy status, restarted services, smoke result, or exact blocker.

Do not report successful publish/deploy while direct push to `origin/main`, main CI/deploy monitoring, Mac Studio git pull, required service restart/reload, or smoke verification remains pending.

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
