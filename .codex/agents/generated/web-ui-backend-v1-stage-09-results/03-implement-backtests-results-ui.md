---
prompt_name: web_ui_backend_v1_stage09_03_backtests_results_ui
repo: roehub.com
branch: main
scope: "Stage 09 UI integration: connect `/backtests` result panels to hardened backend endpoints without changing the current workstation visual model."

language:
  implementation: jinja_css_plain_js
  agent_report: ru

context_sources:
  always_read:
    - path: .codex/AGENTS.md
      why: "repo contract and browser verification rules"
    - path: docs/architecture/apps/web/web-ui-backend-implementation-plan-v1.md
      why: "Stage 09 UI contract"
    - path: docs/architecture/apps/web/web-ui-design-manifest-v1.md
      why: "current Web UI visual/source-of-truth rules"
    - path: docs/architecture/backtest/backtest-service-artifact-runtime-v1.ru.md
      why: "result/statistics semantics"
    - path: .codex/agents/generated/web-ui-backend-v1/09-implement-backtests-results.md
      why: "parent prompt"
  task_entrypoints:
    - path: apps/web/templates/pages/backtests.html
      why: "current /backtests workstation"
    - path: apps/web/dist/js/pages/backtests.js
      why: "current client state and result calls"
    - path: apps/web/dist/css/pages/backtests.css
      why: "current visual model"
    - path: apps/web/dist/js
      why: "shared JS helpers/charts if already present"
    - path: apps/api/routes/backtests.py
      why: "endpoint contract to consume"
    - path: apps/api/dto/backtests.py
      why: "response shapes"
    - path: tests/unit/apps/web/test_app_routes.py
      why: "web route/asset assertions"

style_references:
  design_manifest:
    path: docs/architecture/apps/web/web-ui-design-manifest-v1.md
    purpose: "canonical current UI tokens/components"
  canonical_backtests_reference:
    path: /Users/daniildegtyarev/Projects/roehub_web_ui/stategy_backtest.png
    purpose: "canonical reference for /backtests functional workstation"
  current_ui_primary_source:
    route: "https://roehub.com/backtests"
    purpose: "current deployed UI is source of truth; do not replace with generic cards"

hard_requirements:
  current_ui_is_source_of_truth: true
  reference_shaped_after_header: true
  no_generic_card_redesign: true
  backend_contract_must_be_ready_first: true
  no_fake_unavailable_data: true
  charts_nonblank_if_connected: true
  browser_qa_required: true

task_toggles:
  implement_backend_api: false
  implement_web_ui: true
  implement_charts: true
  publish_after_success: true

package_contract:
  depends_on:
    - "01 materialization/status contract accepted"
    - "02 result/statistics endpoints accepted"
  owns:
    - "apps/web/templates/pages/backtests.html"
    - "apps/web/dist/js/pages/backtests.js"
    - "apps/web/dist/css/pages/backtests.css"
    - "apps/web/dist/js/charts/** if a chart helper is needed"
    - "tests/unit/apps/web/test_app_routes.py"
  forbidden:
    - "apps/api/** unless fixing a tiny endpoint mismatch discovered during UI integration"
    - "src/trading/contexts/**"
    - "dashboard/settings/strategies pages"
    - "global shell/auth redesign"
  integration_points:
    - "current result summary endpoint"
    - "variant endpoint"
    - "equity/drawdown/monthly/symbol/trades/CSV endpoints"
    - "materialization pending/degraded status"
  handoff:
    - "browser-visible result panel behavior"
    - "chart helper contract"
    - "manual refresh/status behavior for result data"

skill_routing:
  - skill: prompt-manager
    use_when: "executing generated prompt"
    timing: "startup and final report"
    reason: "prompt-pack discipline"
  - skill: browser-qa-evidence
    use_when: "verifying `/backtests` UI with real browser"
    timing: "after implementation"
    reason: "browser-visible result panels"
  - skill: playwright
    use_when: "capturing snapshots/screenshots and chart nonblank evidence"
    timing: "browser QA"
    reason: "Stage 09 requires Playwright evidence"
  - skill: contract-impact-analysis
    use_when: "changing browser-visible behavior or endpoint consumption"
    timing: "before final report"
    reason: "UI/API contract boundary"
  - skill: backend-quality-gates
    use_when: "running web/API focused gates"
    timing: "verification"
    reason: "route and JS tests"
  - skill: publish-ci-deploy
    use_when: "all gates and browser evidence pass"
    timing: "after verification"
    reason: "direct-main delivery"

target_envs:
  - local-dev
  - browser
  - github-actions
  - mac-studio

required_literals:
  - "renderBacktestSeries"
  - "/api/backtests/jobs/{job_id}/summary"
  - "/api/backtests/jobs/{job_id}/variants/{variant_key}"
  - "/api/backtests/jobs/{job_id}/variants/{variant_key}/equity"
  - "/api/backtests/jobs/{job_id}/variants/{variant_key}/drawdown"
  - "/api/backtests/jobs/{job_id}/variants/{variant_key}/monthly-stats"
  - "/api/backtests/jobs/{job_id}/variants/{variant_key}/symbol-stats"
  - "GET /api/backtests/jobs/{job_id}/variants/{variant_key}/trades?page="
  - "/api/backtests/jobs/{job_id}/variants/{variant_key}/trades.csv"

non_goals:
  - "Do not redesign the whole `/backtests` page."
  - "Do not add React/Vite/build tooling."
  - "Do not hide unavailable backend data by inventing values."
  - "Do not change auth/session behavior."

quality_gates:
  - cmd: "uv run pytest -q tests/unit/apps/web/test_app_routes.py tests/unit/apps/api/test_backtests_routes.py"
    expect: "passes"
  - cmd: "uv run ruff check apps/web apps/api tests/unit/apps/web tests/unit/apps/api"
    expect: "passes"
  - cmd: "python -m tools.docs.generate_docs_index --check"
    expect: "passes if docs changed"
---

# Task

Implement the browser UI part of Stage 09 for backtest results, using the already hardened backend endpoints.

The page must stay aligned with the current `/backtests` workstation model and canonical `stategy_backtest.png` reference. Do not replace it with generic cards or a new layout system.

## Requirements (Must)

- Before UI edits, verify that all required backend endpoints from prompt 02 exist and have tests.
- Connect result panels for selected job/variant to:
  - summary;
  - variant detail;
  - equity series;
  - drawdown series;
  - monthly stats;
  - symbol stats;
  - paginated trades;
  - CSV export.
- Add chart rendering only through plain JS modules; no new build toolchain.
- Implement `renderBacktestSeries` or equivalent shared helper if missing.
- Show typed `pending/materializing/degraded/unavailable` states when backend data is not ready.
- Add manual refresh for result data. Respect backend rate-limit/polling hints if present.
- Keep current theme and locale behavior. Do not regress `en`/`ru` copy.
- Browser QA must include desktop and mobile, console/network checks, and nonblank chart evidence if charts are rendered.

## Requirements (Should)

- Keep result requests deduplicated: avoid overlapping polling/manual refresh requests for the same job/variant.
- Keep table pagination server-driven.
- Use stable dimensions for charts/tables so loading states do not shift layout.

# Context Acquisition Protocol

Read `.codex/AGENTS.md`, design manifest `/backtests` rules, parent Stage 09 prompt, current template/JS/CSS, then API route/DTO only as needed to consume the contract.

Reading budget: default `<= 12 files`, `<= ~70k tokens`.

# Work Plan

1. Inspect current `/backtests` UI state model and endpoint templates.
2. Verify backend endpoint readiness from prompt 02.
3. Add or refine result panel markup without replacing workstation structure.
4. Add JS data loaders, status handling, manual refresh and chart/table rendering.
5. Add/adjust web route/asset tests.
6. Run local server and Playwright QA.
7. Run focused gates.
8. If all gates pass, use `publish-ci-deploy` direct-main flow.

# Acceptance Criteria

- Current `/backtests` visual model remains intact.
- Selected job/variant can load result data from backend endpoints.
- Empty/pending/degraded/materializing states are explicit and not fake.
- Trades are server-paginated.
- CSV export link still works.
- Charts are nonblank when data exists.
- No console errors and no unexpected same-origin failed requests.
- Theme and language switches remain functional.

# Browser Runtime Evidence Checklist

Collect and report:

- desktop screenshot around `1440x1000`;
- mobile screenshot around `390x844`;
- Playwright snapshot after selected result state;
- console errors absent;
- failed same-origin network requests absent except expected auth redirects;
- manual refresh evidence;
- chart/canvas/SVG nonblank evidence when charts render;
- `en` default evidence and `ru` switch evidence or explicit blocker.

# publish-ci-deploy Direct-Main Delivery Contract

When all DoD, gates and browser evidence pass, run `publish-ci-deploy` in direct-main mode. Do not create a branch or PR. Publish directly to `main`, monitor CI/deploy, sync Mac Studio with `git pull --ff-only`, restart impacted services if needed, and smoke `/backtests` in production.

Do not report successful publish/deploy while any of those steps remains pending.

# Final Output: Report Format

Report in Russian with these exact sections:

- `Intent`
- `Scope`
- `Design`
- `Contract impact`
- `Tests`
- `Docs`
- `Performance`
- `Runtime evidence`
- `Risks`
- `Handoff`
- `Publish/deploy`
