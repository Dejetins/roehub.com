---
prompt_name: backtest_service_iteration_10_ui_integration
repo: roehub.com
branch: current
scope: "Iteration 10: integrate backtest public API into apps/web UI, including job progress, top-N results, lazy show trades, and chart overlay rendering."

language:
  implementation: python_jinja_plain_js_css
  agent_report: ru

context_sources:
  always_read:
    - path: .codex/AGENTS.md
      why: "repo engineering contract, prompt precedence, browser-visible verification, and delivery rules"
    - path: docs/architecture/backtest/backtest-service-artifact-runtime-v1.ru.md
      why: "canonical Iteration 10 contract and accepted public API vocabulary"
    - path: docs/architecture/backtest/benchmark_iterations/2026-05-02_iteration_9_lazy_trades_detail/benchmark_summary.md
      why: "latest accepted lazy trades evidence and Iteration 10 precondition"
    - path: docs/architecture/apps/web/web-ui-skeleton-ssr-htmx-auth-v1.md
      why: "canonical apps/web shape: SSR/Jinja/HTMX facade over same-origin /api proxy"
    - path: docs/architecture/apps/web/web-strategy-ui-crud-builder-delete-v1.md
      why: "current web UI conventions for protected routes, templates, assets, and API-driven JS"
  task_entrypoints:
    - path: apps/web/main/app.py
      why: "protected page routing, login gate, /api proxy, and template context"
      inspect_symbols:
        - create_app
        - _register_routes
        - _render_protected_page
    - path: apps/web/templates/base.html
      why: "top navigation, shared layout, and asset blocks"
    - path: apps/web/dist/site.css
      why: "existing app styling, panels, tables, state badges, and responsive controls"
    - path: apps/api/dto/backtests.py
      why: "public response DTOs for jobs, top variants, and lazy trades detail"
      inspect_symbols:
        - BacktestJobResponse
        - BacktestTopVariantResponse
        - BacktestLazyTradesDetailResponse
    - path: apps/api/routes/backtests.py
      why: "public API endpoints that browser UI must call through same-origin /api proxy"
    - path: tests/unit/apps/web/test_app_routes.py
      why: "current SSR route smoke patterns and protected-page hooks"
  conditional_bundles:
    iteration_9_acceptance:
      read_when: "before implementation; if accepted Iteration 9 evidence is missing, failed, or contradictory, stop and report the precondition blocker"
      paths:
        - docs/architecture/backtest/benchmark_iterations/2026-05-02_iteration_9_lazy_trades_detail/benchmark_summary.md
        - docs/architecture/backtest/benchmark_iterations/2026-05-02_iteration_9_lazy_trades_detail/benchmark_results.json
    backtest_api_contract:
      read_when: "building JS request/response adapters or handling API error states"
      paths:
        - apps/api/dto/backtests.py
        - tests/unit/apps/api/test_backtests_routes.py
        - src/trading/contexts/backtest/application/dto/backtest_jobs.py
        - src/trading/contexts/backtest/application/dto/runtime_preflight.py
    existing_strategy_ui_patterns:
      read_when: "copying established plain-JS page initialization, table rendering, error normalization, or fetch helpers"
      paths:
        - apps/web/templates/strategies_list.html
        - apps/web/templates/strategy_builder.html
        - apps/web/templates/strategy_details.html
        - apps/web/dist/strategy_ui.js
    browser_qa_runtime:
      read_when: "running browser-visible QA or debugging local web/API proxy behavior"
      paths:
        - apps/web/main/main.py
        - apps/api/main/main.py
        - docs/runbooks/web-ui-gateway-same-origin.md
    deployment_reference:
      read_when: "all local gates and browser QA pass and the implementation is ready to merge/deploy"
      paths:
        - /Users/daniildegtyarev/.codex/skills/publish-ci-deploy/SKILL.md
        - scripts/macos/smoke_prod.sh
      instruction: "Use only for the final delivery path. Do not preload it during implementation."
  consult_if_needed:
    - path: src/trading/contexts/backtest/application/services/v2/lazy_trades_detail.py
      read_when: "chart_overlay or trades payload shape is ambiguous from API DTO/tests"
    - path: docs/architecture/backtest/benchmark_iterations/README.md
      read_when: "UI status text needs to distinguish service benchmark evidence from browser QA evidence"
    - path: docs/architecture/backtest/backtest-service-artifact-runtime-v1.md
      read_when: "Russian document is ambiguous; English file is companion/reference only, not source of truth"

style_references:
  - .codex/promt_template.md
  - .codex/agents/generated/backtest-service-iteration-9/01-implement-lazy-trades-detail.md
  - docs/architecture/apps/web/web-strategy-ui-crud-builder-delete-v1.md

hard_requirements:
  iteration_9_acceptance_required_before_implementation: true
  use_existing_public_api_only: true
  protected_backtests_page_required: true
  show_job_progress_required: true
  show_top_n_required: true
  show_trades_lazy_action_required: true
  chart_overlay_rendering_required: true
  same_origin_api_proxy_required: true
  browser_visible_qa_required: true
  merge_main_and_pull_after_success: true
  max_implementation_attempts: 2

task_toggles:
  implement_backtests_nav_link: true
  implement_backtests_ssr_page: true
  implement_backtest_ui_asset: true
  implement_runtime_defaults_loading: true
  implement_preflight_and_create_job: true
  implement_job_history_and_status_polling: true
  implement_top_n_table: true
  implement_show_trades: true
  implement_chart_overlay: true
  implement_browser_qa_evidence: true
  implement_backend_api_changes: false
  implement_new_chart_dependency: false
  publish_merge_deploy_after_success: true

skill_routing:
  - skill: contract-impact-analysis
    use_when: "changing browser-visible defaults, protected routes, API path literals, DTO assumptions, or adding any backend compatibility shim"
    timing: before implementation and before final report
    reason: "Iteration 10 binds UI behavior to the public backtest API and auth/proxy contracts"
  - skill: browser-qa-evidence
    use_when: "verifying the rendered backtests UI, create/preflight/job/top/show-trades flow, chart overlay, responsive layout, and console/network cleanliness"
    timing: during verification after local unit gates pass
    reason: "Iteration 10 is browser-visible and cannot be accepted from source inspection alone"
  - skill: backend-quality-gates
    use_when: "running uv-based lint, type, web route, API route, and regression tests"
    timing: during verification
    reason: "Roehub web and API are Python/FastAPI codepaths with uv-based gates"
  - skill: root-cause-debugging
    use_when: "browser flow, API proxy, auth gate, chart rendering, or route tests fail after the first implementation attempt"
    timing: only for a concrete failure/blocker
    reason: "failure diagnosis must isolate root cause before the second and final attempt"
  - skill: publish-ci-deploy
    use_when: "all implementation gates, browser QA, and local verification pass, and the branch is ready for merge/deploy"
    timing: after verification
    reason: "user requires merge to main, local pull, Mac Studio pull, deploy verification, and post-deploy evidence"

target_envs:
  - local-dev
  - github-actions
  - macstudio
  - browser

required_literals:
  - "/backtests"
  - "/api/backtests/runtime-defaults"
  - "/api/backtests/preflight"
  - "/api/backtests/jobs"
  - "/api/backtests/jobs/{job_id}/top"
  - "/api/backtests/jobs/{job_id}/variants/{variant_key}"
  - "/api/backtests/jobs/{job_id}/variants/{variant_key}/trades"
  - "backtest_chart_overlay_v1"
  - "data-backtest-page"
  - "show trades"
  - "variant_key"
  - "variant_hash"

non_goals:
  - "Do not introduce React/Vite/Next/npm build tooling; current apps/web is Python SSR + Jinja + plain JS/CSS."
  - "Do not create a new internal API for UI; use the same public backtest API through same-origin /api proxy."
  - "Do not change accepted backend scoring, persistence, lazy trades, or benchmark behavior."
  - "Do not add a charting dependency unless existing repo policy already supports it and the dependency is explicitly justified."
  - "Do not render tens of thousands of trades as individual DOM rows at once."
  - "Do not implement new backend candle endpoints unless a small compatible addition is proven necessary and approved by contract-impact-analysis."
  - "Do not use legacy `runs`, `POST /backtests`, execution-profile, or `hit_times/1m` vocabulary."

final_report_format:
  language: ru
  sections:
    - "Что сделано"
    - "UI flow"
    - "API / contract"
    - "Browser QA"
    - "Проверки"
    - "Delivery / merge"
    - "Contract impact"
    - "Ограничения / следующий шаг"

quality_gates:
  - cmd: "uv run ruff check apps/web apps/api src/trading/contexts/backtest tests/unit/apps/web tests/unit/apps/api"
    expect: "passes, or a narrower justified target passes if unrelated existing files fail"
  - cmd: "uv run pyright"
    expect: "passes"
  - cmd: "uv run pytest -q tests/unit/apps/web/test_app_routes.py tests/unit/apps/web/test_api_client.py tests/unit/apps/web/test_security.py tests/unit/apps/api/test_backtests_routes.py"
    expect: "passes; include any new Iteration 10 web tests explicitly"
  - cmd: "uv run pytest -q -ra"
    expect: "passes before merge/deploy"
  - cmd: "uv run python -m tools.docs.generate_docs_index --check"
    expect: "passes if docs or architecture status changes"

expected_primary_touches:
  - "apps/web/main/app.py"
  - "apps/web/templates/base.html"
  - "apps/web/templates/backtests.html"
  - "apps/web/dist/backtest_ui.js"
  - "apps/web/dist/site.css"
  - "tests/unit/apps/web/test_app_routes.py"
  - "tests/unit/apps/web/test_backtest_ui_asset.py"
  - "docs/architecture/backtest/backtest-service-artifact-runtime-v1.ru.md"

possible_secondary_touches:
  - "apps/api/dto/backtests.py"
  - "apps/api/routes/backtests.py"
  - "tests/unit/apps/api/test_backtests_routes.py"
  - "docs/architecture/README.md"

safety_notes:
  - "Do not start implementation unless Iteration 9 accepted evidence exists."
  - "The web app must remain an HTML facade over JSON API. It must not import trading use-cases or repositories directly."
  - "All browser API calls must go through same-origin `/api/*`, not direct backend host URLs."
  - "Current lazy trades payload may contain many trades: render a chart/summary and a paginated or capped table, never one DOM node per trade for the full payload."
  - "If the accepted lazy trades payload lacks OHLC candles, render the best available price/time trade overlay from `chart_overlay` and trades, and document any true candle-data gap instead of inventing an unplanned backend API."
  - "After all checks and browser QA pass, use the repo delivery path to merge to `main`, then pull `main` locally and on Mac Studio."
  - "The executor has only 2 implementation attempts. After the second failed corrective cycle, stop and report the blocker with exact evidence."
---

# Task

Implement Iteration 10: UI integration for the backtest service.

Done means:

- `apps/web` exposes a protected backtests page;
- the page uses the existing public API through same-origin `/api/*`;
- user can create/preflight a backtest request from UI controls;
- user can see job history, job status/progress, and terminal top-N summary;
- user can click `show trades` for a top variant;
- UI calls the lazy trades endpoint and renders trades on a chart/overlay surface;
- trades are also inspectable through a bounded/paginated table;
- browser-visible QA evidence exists;
- if all checks and acceptance gates pass, the work is merged to `main`, `main` is pulled locally, and `main` is pulled on Mac Studio.

## Context / Current State

Precondition:

- Iteration 9 must already be accepted with Mac Studio benchmark evidence:
  `docs/architecture/backtest/benchmark_iterations/2026-05-02_iteration_9_lazy_trades_detail/`.
- If accepted Iteration 9 evidence is missing, failed, or contradicts the runtime document, stop before implementation and report that Iteration 10 is blocked.

Current web architecture:

- `apps/web` is a separate FastAPI SSR app.
- Templates live under `apps/web/templates`.
- Static assets live under `apps/web/dist`.
- There is no `package.json`, no Vite, no React, and no npm build pipeline.
- Existing interactive UI uses plain JavaScript (`apps/web/dist/strategy_ui.js`) and Jinja templates.
- Browser API calls go to same-origin `/api/*`; `apps/web` proxies them to the API upstream.
- Protected pages use `_render_protected_page`.

Accepted public backtest API:

- `GET /backtests/runtime-defaults`
- `POST /backtests/preflight`
- `POST /backtests/jobs`
- `GET /backtests/jobs`
- `GET /backtests/jobs/{job_id}`
- `GET /backtests/jobs/{job_id}/top`
- `GET /backtests/jobs/{job_id}/variants/{variant_key}`
- `POST /backtests/jobs/{job_id}/variants/{variant_key}/trades`
- `POST /backtests/jobs/{job_id}/cancel`

From browser, these must be called as:

- `/api/backtests/runtime-defaults`
- `/api/backtests/preflight`
- `/api/backtests/jobs`
- `/api/backtests/jobs/{job_id}`
- `/api/backtests/jobs/{job_id}/top`
- `/api/backtests/jobs/{job_id}/variants/{variant_key}`
- `/api/backtests/jobs/{job_id}/variants/{variant_key}/trades`
- `/api/backtests/jobs/{job_id}/cancel`

Important API identity contract:

- UI route/action key is public readable `variant_key`.
- Stable SHA identity is `variant_hash`.
- UI must never call lazy trades using raw `variant_hash` as the route key.

## Requirements (Must)

- Read context using the protocol below and stop early once sufficient.
- Verify Iteration 9 acceptance evidence before implementation.
- Implement only the scoped UI integration described in this prompt.
- Preserve accepted backend API and benchmark behavior from Iterations 1..9.
- Use current `apps/web` stack:
  - FastAPI SSR route;
  - Jinja templates;
  - existing `/api/*` proxy;
  - plain JS asset in `apps/web/dist`;
  - CSS in `apps/web/dist/site.css`.
- Do not introduce a Node/npm frontend toolchain.

Routing and layout:

- Add protected backtests UI route:
  - preferred route: `GET /backtests`;
  - optional detail route or query handling may be added if it reduces UI complexity without duplicating pages.
- Add Backtests navigation in `apps/web/templates/base.html`.
- Add template `apps/web/templates/backtests.html`.
- Add static asset `apps/web/dist/backtest_ui.js`.
- Use stable data hooks, at minimum:
  - `data-backtest-page`;
  - `data-api-defaults-path="/api/backtests/runtime-defaults"`;
  - `data-api-preflight-path="/api/backtests/preflight"`;
  - `data-api-jobs-path="/api/backtests/jobs"`;
  - `data-api-job-path-template="/api/backtests/jobs/{job_id}"`;
  - `data-api-top-path-template="/api/backtests/jobs/{job_id}/top"`;
  - `data-api-variant-path-template="/api/backtests/jobs/{job_id}/variants/{variant_key}"`;
  - `data-api-trades-path-template="/api/backtests/jobs/{job_id}/variants/{variant_key}/trades"`;
  - `data-api-cancel-path-template="/api/backtests/jobs/{job_id}/cancel"`.

Backtest request UI:

- Load runtime defaults on page start.
- Provide practical controls for v1 public request:
  - coordinates: `exchange`, `market_type`, `symbol`;
  - timeframe, defaulting to `15m`;
  - half-open period `[start, end)`;
  - indicators list with `indicator_id`, `source`, and window range/grid;
  - risk mode: `none` or `tp_sl_grid`;
  - TP/SL grid start/stop/step when risk mode is `tp_sl_grid`;
  - ranking metric/order;
  - `top_n`;
  - execution settings: fees, slippage, initial cash, sizing mode, fixed quote/equity params, profit lock, direction mode, close-on-end.
- Keep controls dense, operational, and consistent with existing web UI.
- Do not create a marketing/landing page.
- Use runtime defaults to populate select options and defaults where possible.
- Provide a preflight action that calls `/api/backtests/preflight` and displays normalized request, cost estimate, warnings, and validation errors.
- Provide a create job action that calls `/api/backtests/jobs`.
- Use `Idempotency-Key` only if the UI explicitly generates a stable retry key for the same in-flight submission; otherwise allow the API's default "new job" behavior.

Jobs, progress, and top-N:

- Show recent jobs using `GET /api/backtests/jobs`.
- Allow selecting a job from the list or the newly created job.
- Poll `GET /api/backtests/jobs/{job_id}` while job state is queued/running.
- Display progress:
  - state;
  - `progress.pipeline_stage`;
  - percent;
  - processed/total units;
  - timestamps;
  - request hash/config hash.
- Stop polling when state is terminal.
- On succeeded jobs, call `GET /api/backtests/jobs/{job_id}/top`.
- Render top-N summary table:
  - rank;
  - readable params;
  - key metrics;
  - best TP/SL where present;
  - public `variant_key`;
  - compact `variant_hash`;
  - `show trades` action.
- The default sort already comes from API/job result. Do not invent a frontend-only ranking that conflicts with the backend result.

Lazy trades and chart:

- `show trades` must call:

```http
POST /api/backtests/jobs/{job_id}/variants/{variant_key}/trades
```

- Use the top row public `variant_key`, not `variant_hash`.
- Display lazy trades metadata:
  - cache status;
  - timing;
  - trade count;
  - selected variant params;
  - summary metrics.
- Render chart overlay from `chart_overlay.schema == "backtest_chart_overlay_v1"`.
- Render entry/exit markers and trade segments from `chart_overlay.markers` and `chart_overlay.segments`.
- If OHLC candle arrays are not available in the accepted API payload, render a price/time trade overlay chart from trade entry/exit timestamps and prices, and document the candle-data gap as a follow-up. Do not add an unplanned backend candle endpoint by default.
- Prefer `<canvas>` or inline SVG for the chart to avoid adding dependencies.
- The chart must have stable dimensions and responsive constraints.
- Do not render all trades as separate DOM nodes when payload is large. Use:
  - aggregate chart rendering;
  - capped/paginated table;
  - summary counts;
  - "show first N / next page" behavior.
- Provide clear loading, empty, cache-hit, error, and retry states.

Error handling:

- Handle at least:
  - `401`/login redirect behavior through existing protected page;
  - `403 backtest.forbidden`;
  - `404 backtest.not_found`;
  - `409` idempotency or variant conflict if surfaced;
  - `422 backtest.invalid_request`;
  - `429 backtest.rate_limited`;
  - `503` retryable artifact/service errors;
  - API proxy `502`.
- Keep error messages useful without leaking raw stack traces.

Accessibility and UX:

- Use semantic buttons, labels, tables, and `aria-live` for progress/status updates.
- The chart must have text fallback or summary for screen readers.
- UI text must fit in controls at mobile and desktop widths.
- Match existing Roehub web styling; keep the interface compact and work-focused.
- Avoid nested cards and marketing-style hero layouts.

Delivery:

- If and only if local gates, browser QA, docs checks, CI, and deploy checks pass:
  - use `publish-ci-deploy` for the delivery path;
  - push the branch;
  - open/update PR if needed;
  - watch CI to completion;
  - merge into `main`;
  - pull `main` on the local machine in `/Users/daniildegtyarev/Projects/roehub.com`;
  - pull `main` on Mac Studio in `/Users/daniildegtyarev/Projects/roehub.com`;
  - verify deployed/runtime surface on Mac Studio.
- Do not merge or pull/deploy if any required gate is red, skipped, or ambiguous.

## Requirements (Should)

- Keep JS page code modular inside `backtest_ui.js`: API helpers, state, render functions, chart renderer.
- Reuse strategy UI helper patterns only where they fit; do not couple backtest UI to strategy UI state.
- Use `AbortController` or equivalent stale-request guards for polling and show-trades requests when practical.
- Keep poll interval modest, for example 1-2 seconds, and stop on terminal state or page unload.
- Add a small "load sample request" action only if it accelerates manual QA without hiding the real request controls.
- Keep route tests focused on SSR hooks and protected routing.
- Keep browser QA artifacts/screenshots under an ignored output directory when tooling supports it.

## Requirements (Nice-to-have)

- Remember the last selected job in `sessionStorage`.
- Allow copying `job_id`, `variant_key`, and `variant_hash`.
- Add a compact chart legend for long/short, entry/exit, TP/SL/signal close.

# Context acquisition protocol

Read only in this order and stop once sufficient:

1. `.codex/AGENTS.md`
2. Iteration 10 section plus public API/lazy trades sections of `docs/architecture/backtest/backtest-service-artifact-runtime-v1.ru.md`
3. accepted Iteration 9 evidence folder; stop if missing or failed
4. current `apps/web` architecture docs and route/template/asset entrypoints
5. backtest API DTO/routes and tests only to confirm payload shape and paths
6. conditional bundles required by touched contracts, failing checks, or browser QA ambiguity
7. consult-if-needed references only for blockers, ambiguity, or conflict resolution

Do not eagerly preload all listed sources.

Pre-implementation reading target:

- `<= 12 files`
- `<= ~60k-75k tokens`

Stop reading once all of the following are true:

- protected route/template/asset touch set is identified;
- public API calls and payload shapes are identified;
- chart payload assumptions are identified;
- test and browser QA plan is implementable;
- no unresolved public API or web/auth/proxy ambiguity remains.

Expand context only for:

- browser flow failure;
- auth/proxy behavior ambiguity;
- chart payload mismatch;
- API contract mismatch;
- failing unit or browser QA gates;
- Mac Studio deploy failure.

# Reading manifest

Use the front-matter `context_sources` as the canonical reading map.

Read with this intent:

- `always_read`:
  - repository rules;
  - current runtime target;
  - accepted Iteration 9 precondition;
  - web architecture contract.
- `task_entrypoints`:
  - web route/template/asset entrypoints;
  - API DTO/routes for backtest public contract;
  - current web route tests.
- `conditional_bundles`:
  - read only when the stated condition applies.
- `consult_if_needed`:
  - read only for blockers, ambiguity, or conflict resolution.

Do not convert this manifest into a broad mandatory reading list.

# Work plan (agent should follow)

Skill routing for this task:

- `contract-impact-analysis`: use before implementation and before final report; owns public route/API/proxy/defaults compatibility and any backend DTO assumptions.
- `browser-qa-evidence`: use after local tests pass; owns real browser screenshots/flow verification, responsive checks, console/network errors, and chart visibility.
- `backend-quality-gates`: use during verification; owns uv-based lint, type, unit, API, and web route gates.
- `root-cause-debugging`: use only after a concrete failing route, JS, API, browser, or deploy gate.
- `publish-ci-deploy`: use only after all implementation, tests, and browser QA pass; owns push, PR/CI, merge to `main`, local pull, Mac Studio pull, deploy verification, and post-deploy evidence.

Implementation sequence:

1. Verify Iteration 9 accepted evidence exists. Stop if missing.
2. Read bounded context and classify contract impact before code changes.
3. Add protected web route and template for `/backtests`.
4. Add Backtests nav link.
5. Add `backtest_ui.js` with runtime defaults loading, request builder, preflight, create job, jobs list, polling, top-N rendering, show-trades, and chart overlay rendering.
6. Add CSS for dense controls, progress, top table, chart area, and paginated trades table.
7. Add/update web route tests for protected route, nav link, data hooks, and asset inclusion.
8. Add/update asset smoke tests for required API literals, chart schema handling, and "do not use variant_hash as route key" guard where practical.
9. Run focused local gates and fix introduced failures.
10. Run browser-visible QA with real rendered page and mocked or real API responses sufficient to exercise the flow.
11. If accepted, update the main runtime document status for Iteration 10 and docs index if needed.
12. If all gates are green, use `publish-ci-deploy` to push, PR, watch CI, merge to `main`, pull locally, pull on Mac Studio, and verify production/runtime health.
13. If any gate fails after two implementation attempts, stop and report exact blockers.

# Browser QA pipeline

The web app can be started locally as:

```bash
WEB_API_BASE_URL=http://127.0.0.1:8010 WEB_API_UPSTREAM_URL=http://127.0.0.1:8000 \
  uv run python -m apps.web.main.main --host 127.0.0.1 --port 8010
```

The API app can be started locally as:

```bash
uv run python -m apps.api.main.main --host 127.0.0.1 --port 8000
```

Browser QA must verify the strongest feasible runtime surface:

- If a real authenticated local or Mac Studio session is available, use it.
- If real auth/data is not available locally, use a controlled mock/stub API surface or test app override for browser QA and state the limitation.
- At minimum, capture evidence for:
  - `/backtests` protected/login behavior;
  - rendered controls and data hooks;
  - runtime defaults/preflight/create-job flow with mocked or real API;
  - progress/top-N rendering;
  - `show trades` lazy request;
  - chart overlay nonblank rendering;
  - responsive desktop and mobile layout;
  - no serious console errors;
  - no failed network requests except intentionally mocked/blocked auth cases.

Suggested viewports:

- desktop: `1440x1000`;
- mobile: `390x844`.

# Delivery and Mac Studio pipeline

After all checks pass and the branch is ready:

```bash
# Use publish-ci-deploy to perform the actual push/PR/CI/merge/deploy flow.
# After merge:
cd /Users/daniildegtyarev/Projects/roehub.com
git checkout main
git pull --ff-only

ssh macstudio 'cd /Users/daniildegtyarev/Projects/roehub.com && git checkout main && git pull --ff-only'
ssh macstudio 'cd /Users/daniildegtyarev/Projects/roehub.com && bash scripts/macos/smoke_prod.sh'
```

If `/opt/roehub/app` is checked during deploy verification, record that explicitly. Do not run `git pull` in `/opt/roehub/app`; it is a deployed runtime copy, not the repository checkout.

If Mac Studio auth, remote state, or deploy state is broken, treat that as part of the delivery task and fix it through `publish-ci-deploy` rather than reporting only the first failure.

# Acceptance criteria (Definition of Done)

- Iteration 9 accepted evidence exists before implementation.
- `/backtests` protected web page exists.
- Base navigation includes Backtests.
- Page uses same-origin `/api/backtests/*` paths only.
- Runtime defaults load and populate controls.
- Preflight action displays normalized result, warnings/errors, and cost estimate.
- Create job action posts a valid public request.
- Jobs list renders from `GET /api/backtests/jobs`.
- Job progress polling renders state, pipeline stage, percent, and processed/total units.
- Top-N table renders from `GET /api/backtests/jobs/{job_id}/top`.
- `show trades` calls the lazy endpoint using public `variant_key`.
- UI never uses raw `variant_hash` as route key for trades.
- Lazy trades response renders summary, cache/timing metadata, chart overlay, and bounded/paginated trades table.
- Chart/overlay surface is nonblank for a fixture with trades.
- Large trade payloads do not create one DOM row per trade for the full payload.
- Error states are visible and recoverable.
- Protected route/login behavior is preserved.
- Route/unit tests pass.
- Browser-visible QA evidence exists.
- Local full gates pass before merge.
- CI passes before merge.
- The branch is merged to `main`.
- Local checkout at `/Users/daniildegtyarev/Projects/roehub.com` is on updated `main`.
- Mac Studio checkout at `/Users/daniildegtyarev/Projects/roehub.com` is on updated `main`.
- Mac Studio production/runtime smoke passes after merge/deploy.

# Implementation constraints

- Keep diffs scoped to Iteration 10.
- Do not change accepted backtest backend semantics unless a small compatible UI-support fix is strictly necessary.
- Do not add a new frontend framework or build chain.
- Do not bypass `/api/*` same-origin proxy.
- Do not import `src/trading/**` directly from `apps/web`.
- Do not expose secrets, auth tokens, or raw backend URLs in rendered HTML.
- Do not mark browser-visible behavior as working without browser/runtime evidence.
- Do not merge if any required gate is missing, red, or ambiguous.

# Files to indicate (expected touched areas)

Expected primary files:

- `apps/web/main/app.py`
- `apps/web/templates/base.html`
- `apps/web/templates/backtests.html`
- `apps/web/dist/backtest_ui.js`
- `apps/web/dist/site.css`
- `tests/unit/apps/web/test_app_routes.py`
- `tests/unit/apps/web/test_backtest_ui_asset.py`
- `docs/architecture/backtest/backtest-service-artifact-runtime-v1.ru.md`

Possible secondary files:

- `apps/api/dto/backtests.py`
- `apps/api/routes/backtests.py`
- `tests/unit/apps/api/test_backtests_routes.py`
- `docs/architecture/README.md`

# Non-goals

- No backend scorer, persistence, lazy trades, or benchmark changes unless required by a proven compatibility bug.
- No separate internal UI API.
- No React/Vite/Next/npm migration.
- No new broad design system rewrite.
- No new multi-host cache/storage work.
- No legacy `runs`, `POST /backtests`, execution-profile, or `hit_times/1m` paths.

# Quality gates (must run and pass)

Run local gates before claiming implementation completion:

```bash
uv run ruff check apps/web apps/api src/trading/contexts/backtest tests/unit/apps/web tests/unit/apps/api
uv run pyright
uv run pytest -q tests/unit/apps/web/test_app_routes.py tests/unit/apps/web/test_api_client.py tests/unit/apps/web/test_security.py tests/unit/apps/api/test_backtests_routes.py
uv run pytest -q -ra
```

After docs are changed:

```bash
uv run python -m tools.docs.generate_docs_index --check
```

Run browser-visible QA before claiming acceptance:

```bash
# Use browser-qa-evidence with the strongest available runtime surface.
# Record URL, viewport(s), screenshots/artifacts, console status, network status, and tested flow.
```

Run delivery only after all gates pass:

```bash
# Use publish-ci-deploy skill for the actual flow.
# Required terminal state:
# - branch merged to main
# - local checkout pulled to main
# - Mac Studio checkout pulled to main
# - Mac Studio smoke passed
```

If a command cannot be run, state why, classify the risk, and do not claim that gate as passed.

# Contract impact report

Include this classification in the final report:

- Public API:
- Web routes:
- Browser-visible defaults:
- DTO assumptions:
- Auth/proxy behavior:
- Persisted schema:
- Config schema:
- Benchmark evidence schema:
- Delivery/deploy surface:

Use one of:

- `none`;
- `compatible-change`;
- `breaking-change`;
- `unknown`.

# Failure/blocker behavior

You have only 2 implementation attempts.

An attempt is a full cycle of implementation, local gates, and browser QA or equivalent blocker evidence. If the second attempt still fails acceptance:

- stop;
- do not broaden scope into backend refactors or a new frontend framework;
- do not merge;
- do not hide failed route tests, browser errors, network errors, chart blanks, or accessibility/layout failures;
- report:
  - implementation commit;
  - changed files;
  - exact failed UI/API/browser scenario;
  - API path and payload involved;
  - screenshot/artifact path if available;
  - browser console/network evidence;
  - whether the failure is auth/proxy, request builder, preflight, job creation, polling, top-N, lazy trades, chart rendering, large payload handling, CSS/responsive layout, CI/deploy, or Mac Studio environment;
  - the smallest next investigation step.

# Final output: report format (strict)

Use Russian.

## Что сделано

- Concise implementation summary.

## UI flow

- Routes.
- Main controls.
- Job/progress/top-N behavior.
- Show-trades behavior.
- Chart behavior.

## API / contract

- Public API paths used.
- Variant identity behavior.
- Any backend/API changes, if any.

## Browser QA

- Runtime surface.
- URL(s).
- Viewports.
- Screenshots/artifacts.
- Console/network status.
- Tested scenarios.

## Проверки

- Commands run and results.
- Commands not run and why.

## Delivery / merge

- Branch / PR.
- CI status.
- Merge status.
- Local `main` pull status.
- Mac Studio `main` pull status.
- Mac Studio smoke/deploy verification.

## Contract impact

- Public API:
- Web routes:
- Browser-visible defaults:
- DTO assumptions:
- Auth/proxy behavior:
- Persisted schema:
- Config schema:
- Benchmark evidence schema:
- Delivery/deploy surface:

## Ограничения / следующий шаг

- Remaining risks.
- If accepted and merged, state that Iteration 10 completes the planned v1 UI integration scope.
- If not accepted, state the blocker and stop.
