---
prompt_name: web_ui_backend_v1_09_backtests_results
repo: roehub.com
branch: main
scope: "Этап 9: harden/complete existing backtest result API/state внутри /backtests workstation, without duplicating already implemented summary/chart/stat/trades endpoints."

language:
  implementation: python_fastapi_jinja_css_js
  agent_report: ru

context_sources:
  always_read:
    - path: .codex/AGENTS.md
      why: "contracts, performance, browser verification"
    - path: docs/architecture/apps/web/web-ui-backend-implementation-plan-v1.md
      why: "Этап 9 source of truth"
    - path: docs/architecture/backtest/backtest-service-artifact-runtime-v1.ru.md
      why: "jobs, variant_key, lazy trades, summary-only contract"
    - path: docs/architecture/backtest/backtest-job-runner-production-plan-v1.md
      why: "production lazy trades materialization and no heavy API cache-miss recompute contract"
    - path: docs/architecture/apps/web/web-ui-design-manifest-v1.md
      why: "result/statistics visual rules"
  task_entrypoints:
    - path: apps/api/routes/backtests.py
      why: "current job/top/variant/trades routes"
    - path: apps/api/routes/ui_backtests.py
      why: "current `/api/ui/backtests/workstation` bounded workstation read model"
    - path: apps/api/dto/backtests.py
      why: "current response DTOs"
    - path: apps/api/dto/ui_backtests.py
      why: "current Stage 8 workstation DTOs; must remain bounded without full trades"
    - path: src/trading/contexts/backtest/application/use_cases/backtest_jobs.py
      why: "variant lookup and ownership"
    - path: src/trading/contexts/backtest/application/use_cases/backtest_job_worker.py
      why: "current Stage 8.5 worker claim/execute/finish boundary"
    - path: apps/web/templates/pages/backtests.html
      why: "backtest workstation page target"
  conditional_bundles:
    canonical_reference:
      read_when: "when verifying result state still fits the backtest workstation reference"
      paths:
        - /Users/daniildegtyarev/Projects/roehub_web_ui/stategy_backtest.png
    lazy_trades:
      read_when: "when implementing GET paginated trades over lazy detail/cache"
      paths:
        - src/trading/contexts/backtest/application/services/v2/lazy_trades_detail.py
        - tests/unit/apps/api/test_backtests_routes.py
    chart_downsampling:
      read_when: "when adding equity/drawdown/monthly series endpoints"
      paths:
        - src/trading/contexts/backtest/application/services/v2
    browser_charts:
      read_when: "when implementing canvas/SVG chart helpers"
      paths:
        - apps/web/dist/js/charts
  consult_if_needed:
    - path: docs/architecture/backtest/benchmark_iterations/2026-05-02_iteration_9_lazy_trades_detail/benchmark_summary.md
      read_when: "if lazy trades accepted behavior is ambiguous"

style_references:
  design_manifest:
    path: docs/architecture/apps/web/web-ui-design-manifest-v1.md
    purpose: "визуальный source of truth для токенов, тем, layouts, density и accessibility"
  external_reference_root:
    path: /Users/daniildegtyarev/Projects/roehub_web_ui
    purpose: "reference screenshots/assets; canonical `/backtests` page remains stategy_backtest.png"
  canonical_reference:
    route: /backtests
    path: /Users/daniildegtyarev/Projects/roehub_web_ui/stategy_backtest.png
    fidelity: "result state must stay inside the reference-shaped backtest workstation; no separate sixth page"
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
  readiness_prerequisites_green_required: true
  readiness_gates_include_ui_backtests: true
  inventory_current_result_methods_first: true
  do_not_reimplement_existing_result_routes: true
  harden_existing_result_methods_before_ui_expansion: true
  selected_result_state_required: true
  no_full_trades_initial_payload: true
  server_pagination_trades_required: true
  chart_points_bounded: true
  public_variant_key_only: true
  storage_identity_split_preserved: true
  no_full_trades_in_top_rows: true
  manual_refresh_required: true
  autorefresh_rate_limit_required: true
  chart_nonblank_browser_check_required: true
  no_separate_result_page_required: true
  reject_generic_result_cards: true

task_toggles:
  verify_existing_summary_endpoints: true
  verify_existing_chart_stat_trades_endpoints: true
  implement_missing_result_endpoints_only_if_absent: true
  harden_lazy_materialization_status_contract: true
  keep_current_summary_only_ui_until_runner_ready: true
  implement_result_ui_expansion_only_after_materialization_ready: true
  publish_after_success: true

package_contract:
  depends_on:
    - "08-backtests-workstation accepted; current `/backtests`, `/backtests/new`, `/backtests/{job_id}` shell/alias readiness pinned by `tests/unit/apps/web/test_app_routes.py`"
    - "08.5-backtest-runtime-hardening accepted; current queued create and `BacktestJobWorkerUseCase` readiness pinned by focused API/use-case tests"
  owns:
    - "apps/api/routes/backtests.py result endpoints only"
    - "apps/api/dto/backtests.py result DTO additions only"
    - "src/trading/contexts/backtest/application/services/v2/* result/lazy detail helpers"
    - "apps/web/templates/pages/backtests.html result state only"
    - "apps/web/dist/js/pages/backtests.js result state only"
    - "apps/web/dist/js/charts/**"
    - "apps/web/dist/css/pages/backtests.css result sections"
    - "tests/unit/apps/api/test_backtests_routes.py result assertions"
  forbidden:
    - "backtests workstation flow rewrites"
    - "worker runtime hardening"
    - "AI configurator"
    - "full trades in top rows"
  integration_points:
    - "public variant_key mapping"
    - "lazy trades cache/materialization"
    - "bounded chart point limits"
    - "refresh/autorefresh helper and retry_after_seconds"
    - "CSV export route"
  handoff:
    - "bounded results/statistics endpoints and `/backtests` result-state evidence"

skill_routing:
  - skill: contract-impact-analysis
    use_when: "adding result endpoints, DTOs, CSV export, pagination, cache keys, variant lookup semantics"
    timing: "before implementation and final report"
    reason: "result APIs are public and cache/persistence sensitive"
  - skill: backend-quality-gates
    use_when: "running API/service/web tests, ruff, pyright"
    timing: "during verification"
    reason: "backend route and service gates"
  - skill: backend-performance-evidence
    use_when: "making chart/trades payload, downsampling, cache materialization, or CSV performance claims"
    timing: "during performance verification"
    reason: "result endpoints can become heavy"
  - skill: browser-qa-evidence
    use_when: "verifying `/backtests` selected result state, variant switch, nonblank charts, table pagination, CSV link, console/network"
    timing: "after backend tests"
    reason: "results are browser-visible"
  - skill: playwright
    use_when: "capturing Playwright evidence"
    timing: "during browser QA"
    reason: "plan requires Playwright CLI"
  - skill: publish-ci-deploy
    use_when: "all endpoint tests, browser QA, nonblank chart evidence, and bounded payload checks pass"
    timing: "after verification"
    reason: "full Roehub delivery chain after complete success"

target_envs:
  - local-dev
  - browser
  - github-actions

required_literals:
  - "/backtests"
  - "/backtests?job_id="
  - "stategy_backtest.png"
  - "/api/backtests/jobs/{job_id}/summary"
  - "/api/backtests/jobs/{job_id}/variants/{variant_key}"
  - "POST /api/backtests/jobs/{job_id}/variants/{variant_key}/trades"
  - "/api/backtests/jobs/{job_id}/variants/{variant_key}/equity"
  - "/api/backtests/jobs/{job_id}/variants/{variant_key}/drawdown"
  - "/api/backtests/jobs/{job_id}/variants/{variant_key}/monthly-stats"
  - "/api/backtests/jobs/{job_id}/variants/{variant_key}/symbol-stats"
  - "GET /api/backtests/jobs/{job_id}/variants/{variant_key}/trades?page="
  - "/api/backtests/jobs/{job_id}/variants/{variant_key}/trades.csv"
  - "/api/ui/backtests/workstation"
  - "BacktestJobWorkerUseCase"
  - "background_auto"
  - "execution_trigger"
  - "variant_key"
  - "variant_hash"
  - "summary-only"
  - "retry_after_seconds"
  - "refresh_status"
  - "backtest_lazy_trades_materializations"
  - "renderBacktestSeries"

non_goals:
  - "Do not duplicate or rewrite already existing result routes just because older prompt text says implement them."
  - "Do not change canonical request hash."
  - "Do not store full trades in top variant rows."
  - "Do not accept raw storage SHA as public route key."
  - "Do not create a separate sixth results page or `backtests_result.html` layout."
  - "Do not replace the backtest workstation reference with generic result cards."
  - "Do not let result refresh/autorefresh fetch unbounded trades or trigger compute."

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
  - cmd: "uv run pytest -q tests/unit/apps/api/test_backtests_routes.py tests/unit/apps/api/test_ui_backtests_routes.py tests/unit/apps/web/test_app_routes.py tests/unit/contexts/backtest/application/use_cases/test_backtest_job_worker_use_case.py"
    expect: "passes; preserves Stage 8 workstation, Stage 8.5 queued runtime, public variant-key, lazy trades, and result endpoint coverage"
  - cmd: "uv run ruff check apps/api apps/web src/trading/contexts/backtest tests/unit/apps/api tests/unit/apps/web tests/unit/contexts/backtest"
    expect: "passes for touched paths"
  - cmd: "uv run pyright"
    expect: "passes"
  - cmd: "python -m tools.docs.generate_docs_index --check"
    expect: "passes if docs changed"

expected_primary_touches:
  - "apps/api/routes/backtests.py"
  - "apps/api/dto/backtests.py"
  - "src/trading/contexts/backtest/application/services/v2/*"
  - "src/trading/contexts/backtest/application/use_cases/backtest_jobs.py"
  - "apps/web/templates/pages/backtests.html"
  - "apps/web/dist/js/pages/backtests.js"
  - "apps/web/dist/js/charts/*"
  - "apps/web/dist/css/pages/backtests.css"
  - "tests/unit/apps/api/test_backtests_routes.py"
  - "tests/unit/apps/api/test_ui_backtests_routes.py"
  - "tests/unit/apps/web/test_app_routes.py"
  - "tests/unit/contexts/backtest/application/use_cases/test_backtest_job_worker_use_case.py"

possible_secondary_touches:
  - "src/trading/contexts/backtest/application/ports/lazy_trades_cache.py"
  - "docs/architecture/backtest/backtest-service-artifact-runtime-v1.ru.md"

safety_notes:
  - "GET /trades returns paginated rows; POST /trades may remain lazy materialization/cache warm path."
  - "Chart endpoints must downsample to bounded points."
  - "Unknown public `variant_key` returns 404."
  - "The `/backtests` page body must remain reference-shaped against stategy_backtest.png while adding selected result state."
  - "Manual refresh/autorefresh for selected result state must be no-overlap, bounded and rate-limit aware."
---

# Task

Harden and complete Stage 9 backtest result state and bounded result APIs inside
`/backtests`, starting from the current code rather than the older greenfield prompt
assumption.

Done means:

- `/backtests` can open selected job/result state, normally via query/deep-link such as `/backtests?job_id=...`;
- `/backtests/{job_id}`, if preserved, redirects/aliases to the same workstation state;
- page body remains reference-shaped against `stategy_backtest.png`;
- current result routes are inventoried before edits, and already implemented routes are hardened rather than duplicated;
- result state loads summary and one selected job/variant without all trades;
- existing chart/stat/trades endpoints are bounded/downsampled/paginated and safe under cache/materialization behavior;
- CSV export remains separate, owner-scoped and bounded by cache/materialization policy;
- manual refresh/autorefresh for selected result state is bounded and respects server retry windows;
- charts are nonblank in browser evidence;
- variant lookup uses public `variant_key`.

## Context / Current State

- Current API already has job/top/variant, POST lazy trades, result summary, equity,
  drawdown, monthly stats, symbol stats, paginated GET trades and CSV endpoints in
  `apps/api/routes/backtests.py`.
- Current result DTOs already exist in `apps/api/dto/backtests.py` and
  `src/trading/contexts/backtest/application/services/v2/result_series.py`.
- Current `/api/ui/backtests/workstation` read model is the Stage 8 bounded workstation
  payload and must not be expanded with full trades by Stage 9 initial render.
- Current create path is queued: `POST /api/backtests/jobs` returns `queued`/`background_auto` semantics and enqueues through `execution_trigger`; `BacktestJobWorkerUseCase` owns claim/execute/finish/fail.
- Current public result-adjacent lookup uses readable public `variant_key`; raw storage `variant_hash` is not accepted as the public route key.
- Current Web UI consumes only `GET /api/backtests/jobs/{job_id}/summary` for variant
  expansion and renders CSV links. It intentionally does not call `/equity`,
  `/drawdown`, `/monthly-stats`, `/symbol-stats`, or `GET /trades?page=...`; tests
  assert `renderBacktestSeries` and `/trades?page=` are absent from current JS.
- Summary top rows must stay lightweight.
- Lazy trades details are separate from top variant persistence.
- Cache-miss lazy trades/result methods must align with
  `docs/architecture/backtest/backtest-job-runner-production-plan-v1.md`: no heavy
  sync cache-miss recompute in the production API request path.

## Requirements (Must)

- Keep result UI inside `/backtests`; do not add a separate `backtests_result.html` page.
- Preserve `/backtests` reference shape from `stategy_backtest.png` while adding selected job/result state.
- Verify existing result summary, equity, drawdown, monthly, symbol stats, paginated
  trades and CSV endpoints before coding. Implement only endpoints that are truly absent.
- Harden existing result/stat/trades methods so cache miss returns materialization/status
  behavior instead of blocking API on heavy lazy recompute in production.
- Preserve public/storage identity split.
- Keep initial page payload bounded.
- Manual refresh/autorefresh must not fetch all trades, must not trigger full compute,
  and must respect `retry_after_seconds`.
- Add tests for 404, pagination, downsampling bounds, CSV auth/ownership.
- Preserve and run the readiness tests for Stage 8 workstation, Stage 8.5 queued runtime, public `variant_key` split, and lazy trades POST boundary before adding result-state behavior.
- Run browser nonblank chart evidence only if this prompt actually connects charts into
  the current UI; otherwise report charts as backend-only and not browser-visible in this pass.
- Use `publish-ci-deploy` only after all gates pass.

## Requirements (Should)

- Keep point limits configurable or centralized.
- Reuse lazy trades cache/materialization.
- Use shared chart helpers.

## Requirements (Nice-to-have)

- Add table sorting only if it remains server-owned and bounded.

# Context acquisition protocol

Read `.codex/AGENTS.md`, plan Stage 9, design manifest backtests sections, `stategy_backtest.png`, backtest runtime doc, then task entrypoints. Expand only for lazy trades or chart downsampling implementation.

Reading budget: keep pre-implementation reading to the smallest sufficient set; default target `<= 8 files`, `<= ~45k tokens` unless this prompt states a tighter number.
Stop reading when touched files, contract surfaces, and acceptance gates are bounded enough to implement safely.
Do not eager-load all `context_sources`, `conditional_bundles`, or `consult_if_needed` files at startup.
If `.codex/agents/.context/promt_manager_state.yaml` or a latest executor final report for this pack exists, read only its completed/open_items/risks/handoff summary before task entrypoints; skip this step if absent.

# Reading manifest

Use front matter `context_sources`.

# Work plan (agent should follow)

1. Re-open `stategy_backtest.png` and confirm the current `/backtests` workstation must not be rewritten.
2. Inventory current result endpoints/methods/tests before edits: summary, variant, equity, drawdown, monthly stats, symbol stats, paginated trades, CSV, POST lazy trades.
3. Compare current Web UI expectations: summary endpoint template, variant endpoint template, CSV links, no `renderBacktestSeries`, no `/trades?page=` JS call.
4. Harden backend services/routes/tests around materialization/cache-status, pagination, downsampling, owner scope and public `variant_key`; implement missing endpoints only if absent.
5. Connect additional UI panels/charts/tables only if materialization/status contract is ready and acceptance evidence can be collected without drifting the workstation.
6. Add nonblank chart/browser QA only for actually connected browser-visible charts.
7. Run performance smoke for heavy endpoints where feasible.
8. Run gates.
9. Use `publish-ci-deploy` only after complete success.

# Acceptance criteria (Definition of Done)

- `/backtests?job_id=...` opens result state; `/backtests/{job_id}`, if kept, aliases to it.
- `/backtests` remains reference-shaped against `stategy_backtest.png`.
- Loading/result state does not fetch all trades.
- Current variant expansion fetches bounded summary; future chart/stat expansion fetches one selected variant only after materialization/status readiness.
- Manual refresh/autorefresh refreshes one selected job/variant state without overlapping requests.
- Trades table uses server pagination if connected in this pass.
- CSV export is separate from table paging and owner-scoped.
- Canvas/SVG charts are nonblank if connected in this pass.
- Multi-year series respects point limits.
- Financial colors remain invariant.
- Generic result cards or separate sixth page layout are not acceptable.
- Existing readiness gates stay green: `/api/ui/backtests/workstation` remains bounded without full trades, `/backtests` aliases keep selected-job markers, queued create remains non-inline, worker success/failure remains covered, and raw `variant_hash` remains rejected as public route key.

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
- Persisted schema: expected `none` unless cache metadata moves to DB.
- Request hash/cache identity: `none` or additive cache key metadata only.

# Files to indicate (expected touched areas)

Use front matter touched paths.

# Non-goals

- Configurator/history.
- Worker hardening.
- AI assistant.

# Quality gates (must run and pass)

```bash
uv run pytest -q tests/unit/apps/api/test_backtests_routes.py tests/unit/apps/api/test_ui_backtests_routes.py tests/unit/apps/web/test_app_routes.py tests/unit/contexts/backtest/application/use_cases/test_backtest_job_worker_use_case.py
uv run ruff check apps/api apps/web src/trading/contexts/backtest tests/unit/apps/api tests/unit/apps/web tests/unit/contexts/backtest
uv run pyright
python -m tools.docs.generate_docs_index --check
```

Playwright CLI:

```bash
export CODEX_HOME="${CODEX_HOME:-$HOME/.codex}"
export PWCLI="$CODEX_HOME/skills/playwright/scripts/playwright_cli.sh"
"$PWCLI" open http://127.0.0.1:8010/backtests?job_id=<job_id>
"$PWCLI" snapshot
"$PWCLI" screenshot --filename output/playwright/backtests-result-state-desktop.png
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
