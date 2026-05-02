---
prompt_name: web_ui_backend_v1_09_backtests_results
repo: roehub.com
branch: current
scope: "Этап 9: backtest result page, summary/chart/stat endpoints, paginated trades, CSV export."

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
    - path: docs/architecture/apps/web/web-ui-design-manifest-v1.md
      why: "result/statistics visual rules"
  task_entrypoints:
    - path: apps/api/routes/backtests.py
      why: "current job/top/variant/trades routes"
    - path: apps/api/dto/backtests.py
      why: "current response DTOs"
    - path: src/trading/contexts/backtest/application/use_cases/backtest_jobs.py
      why: "variant lookup and ownership"
    - path: src/trading/contexts/backtest/application/ports/lazy_trades_cache.py
      why: "lazy trades cache identity"
    - path: apps/web/templates/pages/backtests_result.html
      why: "result page target"
  conditional_bundles:
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

hard_requirements:
  direct_result_url_required: true
  no_full_trades_initial_payload: true
  server_pagination_trades_required: true
  chart_points_bounded: true
  public_variant_key_only: true
  storage_identity_split_preserved: true
  no_full_trades_in_top_rows: true
  chart_nonblank_browser_check_required: true

task_toggles:
  implement_summary_endpoints: true
  implement_chart_endpoints: true
  implement_paginated_trades_get: true
  implement_csv_export: true
  implement_result_page: true
  publish_after_success: true

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
    use_when: "verifying result page, variant switch, nonblank charts, table pagination, CSV link, console/network"
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
  - "/backtests/{job_id}"
  - "/api/backtests/jobs/{job_id}/summary"
  - "/api/backtests/jobs/{job_id}/variants/{variant_key}/equity"
  - "/api/backtests/jobs/{job_id}/variants/{variant_key}/trades"
  - "variant_key"
  - "variant_hash"
  - "summary-only"

non_goals:
  - "Do not change canonical request hash."
  - "Do not store full trades in top variant rows."
  - "Do not accept raw storage SHA as public route key."
  - "Do not implement configurator/history flows here."

final_report_format:
  - "Summary: что сделано"
  - "Files changed: пути и назначение"
  - "Contracts: classification and API/schema/UI impact"
  - "Verification: команды, Playwright evidence, результаты"
  - "Publish/deploy: terminal state publish-ci-deploy или причина пропуска"
  - "Risks / follow-up: остаточные риски"

quality_gates:
  - cmd: "uv run pytest -q tests/unit/apps/api/test_backtests_routes.py tests/unit/apps/web/test_app_routes.py"
    expect: "passes; add focused result endpoint tests"
  - cmd: "uv run ruff check apps/api apps/web src/trading/contexts/backtest tests/unit/apps/api tests/unit/apps/web"
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
  - "apps/web/templates/pages/backtests_result.html"
  - "apps/web/dist/js/pages/backtests_result.js"
  - "apps/web/dist/js/charts/*"
  - "apps/web/dist/css/pages/backtests.css"
  - "tests/unit/apps/api/test_backtests_routes.py"
  - "tests/unit/apps/web/test_app_routes.py"

possible_secondary_touches:
  - "src/trading/contexts/backtest/application/ports/lazy_trades_cache.py"
  - "docs/architecture/backtest/backtest-service-artifact-runtime-v1.ru.md"

safety_notes:
  - "GET /trades returns paginated rows; POST /trades may remain lazy materialization/cache warm path."
  - "Chart endpoints must downsample to bounded points."
  - "Unknown public `variant_key` returns 404."
---

# Task

Implement Stage 9 backtest results page and bounded result APIs.

Done means:

- `/backtests/{job_id}` opens directly;
- page loads summary and one selected variant without all trades;
- chart endpoints are bounded/downsampled;
- trades table uses server pagination;
- CSV export is separate;
- charts are nonblank in browser evidence;
- variant lookup uses public `variant_key`.

## Context / Current State

- Current API has job/top/variant and POST lazy trades endpoint.
- Summary top rows must stay lightweight.
- Lazy trades details are separate from top variant persistence.

## Requirements (Must)

- Add result summary, equity, drawdown, monthly, symbol stats, paginated trades and CSV endpoints as compatible additions.
- Preserve public/storage identity split.
- Keep initial page payload bounded.
- Add tests for 404, pagination, downsampling bounds, CSV auth/ownership.
- Run browser nonblank chart evidence.
- Use `publish-ci-deploy` only after all gates pass.

## Requirements (Should)

- Keep point limits configurable or centralized.
- Reuse lazy trades cache/materialization.
- Use shared chart helpers.

## Requirements (Nice-to-have)

- Add table sorting only if it remains server-owned and bounded.

# Context acquisition protocol

Read `.codex/AGENTS.md`, plan Stage 9, backtest runtime doc, then task entrypoints. Expand only for lazy trades or chart downsampling implementation.

Reading budget: keep pre-implementation reading to the smallest sufficient set; default target `<= 8 files`, `<= ~45k tokens` unless this prompt states a tighter number.
Stop reading when touched files, contract surfaces, and acceptance gates are bounded enough to implement safely.
Do not eager-load all `context_sources`, `conditional_bundles`, or `consult_if_needed` files at startup.

# Reading manifest

Use front matter `context_sources`.

# Work plan (agent should follow)

1. Specify result endpoint DTOs and pagination.
2. Implement backend services/routes/tests.
3. Implement result page, JS charts/table, CSV action.
4. Add nonblank chart/browser QA.
5. Run performance smoke for heavy endpoints where feasible.
6. Run gates.
7. Use `publish-ci-deploy` only after complete success.

# Acceptance criteria (Definition of Done)

- Result page opens by URL.
- Loading page does not fetch all trades.
- Variant switch fetches one variant's summary/chart endpoints.
- Trades table uses server pagination.
- CSV export is separate from table paging.
- Canvas/SVG charts are nonblank.
- Multi-year series respects point limits.
- Financial colors remain invariant.

# Implementation constraints

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
uv run pytest -q tests/unit/apps/api/test_backtests_routes.py tests/unit/apps/web/test_app_routes.py
uv run ruff check apps/api apps/web src/trading/contexts/backtest tests/unit/apps/api tests/unit/apps/web
uv run pyright
python -m tools.docs.generate_docs_index --check
```

Playwright CLI:

```bash
export CODEX_HOME="${CODEX_HOME:-$HOME/.codex}"
export PWCLI="$CODEX_HOME/skills/playwright/scripts/playwright_cli.sh"
"$PWCLI" open http://127.0.0.1:8010/backtests/<job_id>
"$PWCLI" snapshot
"$PWCLI" screenshot --filename output/playwright/backtests-result-desktop.png
```

# Final output: report format (strict)

Report in Russian: `Intent`, `Scope`, `Design`, `Contract impact`, `Tests`, `Runtime evidence`, `Performance`, `Risks`, `Handoff`, `Publish/deploy`.
