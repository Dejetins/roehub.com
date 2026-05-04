---
prompt_name: web_ui_backend_v1_12_capacity_load_validation
repo: roehub.com
branch: main
scope: "Этап 12: capacity/load validation текущего backend host для Web UI/API."

language:
  implementation: python_docs_ops
  agent_report: ru

context_sources:
  always_read:
    - path: .codex/AGENTS.md
      why: "performance evidence, gates, delivery rules"
    - path: docs/architecture/apps/web/web-ui-backend-implementation-plan-v1.md
      why: "Этап 12 source of truth"
    - path: docs/runbooks/web-ui-gateway-same-origin.md
      why: "local/prod same-origin topology"
  task_entrypoints:
    - path: tools/load
      why: "target location for lightweight capacity harness if missing"
    - path: apps/web/main/app.py
      why: "web host and API proxy route surface"
    - path: apps/api/main/app.py
      why: "API metrics and route surface"
    - path: apps/api/monitoring.py
      why: "existing Prometheus HTTP metrics"
  conditional_bundles:
    deploy_host:
      read_when: "when running or documenting current host/VPS/Mac Studio capacity"
      paths:
        - infra/caddy/Caddyfile.vps
        - .github/workflows/deploy-web.yml
        - docs/runbooks/mac-studio-native-backend-operations.md
    scenario_endpoints:
      read_when: "when implementing scenarios"
      paths:
        - apps/api/routes/backtests.py
        - apps/api/routes/strategies.py
        - apps/api/routes/identity.py
  consult_if_needed:
    - path: docs/architecture/apps/web/web-ui-design-manifest-v1.md
      read_when: "if browser-visible first-paint constraints are ambiguous"

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
  capacity_report_required: true
  no_external_load_dependencies_without_justification: true
  no_secrets_in_report: true
  green_yellow_red_classification_required: true
  host_class_commit_config_required: true
  p50_p95_p99_error_rss_cpu_required: true
  refresh_autorefresh_scenarios_required: true
  exchange_rate_limit_smoke_required: true
  publish_after_success: true

task_toggles:
  implement_lightweight_harness_if_missing: true
  implement_runbook_optional: true
  run_read_mostly_scenarios: true
  run_refresh_autorefresh_scenarios: true
  run_exchange_limiter_smoke: true
  run_controlled_create_burst_after_8_5: true
  publish_after_success: true

package_contract:
  depends_on:
    - "main page packages implemented or target scenario explicitly skipped"
    - "11-security-performance-delivery-hardening accepted or coordinated final hardening window"
    - "08.5-backtest-runtime-hardening accepted before create-burst scenario"
  owns:
    - "tools/load/web_capacity_smoke.py"
    - "tests/unit/tools/test_web_capacity_smoke.py"
    - "docs/runbooks/web-ui-capacity-smoke.md"
    - "docs/architecture/apps/web/web-ui-backend-implementation-plan-v1.md capacity notes only"
    - "docs/architecture/README.md"
  forbidden:
    - "feature implementation in apps/web/apps/api"
    - "unbounded production stress tests"
    - "secret/cookie/token output in reports"
  integration_points:
    - "Prometheus HTTP metrics"
    - "same-origin local/prod topology"
    - "capacity report artifact path"
    - "green/yellow/red rollout notes"
    - "refresh/autorefresh coalescing and retry window evidence"
    - "exchange-bound limiter evidence"
  handoff:
    - "measured host capacity, limits, and rollout mitigations"

skill_routing:
  - skill: backend-performance-evidence
    use_when: "designing/running/reporting load/capacity tests and classifying green/yellow/red"
    timing: "before and during measurement"
    reason: "capacity claims require comparable measurements and evidence"
  - skill: backend-quality-gates
    use_when: "running harness tests/lint/type checks"
    timing: "during verification"
    reason: "new tool/runbook must be gated"
  - skill: contract-impact-analysis
    use_when: "capacity profile settings, rollout gates, config defaults, or public docs become delivery contracts"
    timing: "before final report"
    reason: "capacity thresholds can become rollout/config contracts"
  - skill: publish-ci-deploy
    use_when: "harness/report/docs/tests pass, no secrets are recorded, and capacity findings are committed as scoped artifacts"
    timing: "after verification"
    reason: "full Roehub delivery chain after complete success"

target_envs:
  - local-dev
  - current-host
  - github-actions
  - macstudio

required_literals:
  - "tools/load/web_capacity_smoke.py"
  - "green"
  - "yellow"
  - "red"
  - "p50"
  - "p95"
  - "p99"
  - "RSS"
  - "CPU"
  - "retry_after_seconds"
  - "autorefresh"
  - "exchange limiter"
  - "no secrets"

non_goals:
  - "Do not add external load-test dependency unless separately justified."
  - "Do not run destructive flows outside isolated test account."
  - "Do not claim production capacity without recording host/config/dataset."
  - "Do not publish cookies/tokens in reports."

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
  - cmd: "uv run python tools/load/web_capacity_smoke.py --help"
    expect: "passes if harness is added"
  - cmd: "uv run ruff check tools/load tests"
    expect: "passes for touched paths"
  - cmd: "uv run pyright"
    expect: "passes if typed Python touched"
  - cmd: "python -m tools.docs.generate_docs_index --check"
    expect: "passes if docs/runbooks changed"

expected_primary_touches:
  - "tools/load/web_capacity_smoke.py"
  - "docs/runbooks/web-ui-capacity-smoke.md"
  - "docs/architecture/apps/web/web-ui-backend-implementation-plan-v1.md"

possible_secondary_touches:
  - "tests/unit/tools/test_web_capacity_smoke.py"
  - "docs/architecture/README.md"

safety_notes:
  - "No secrets, cookies, tokens, env dumps in capacity report."
  - "Read-mostly by default; destructive create bursts only controlled and isolated."
  - "If scenario is yellow/red, record mitigation before public rollout."
  - "Do not run early against incomplete page packages; skip each unavailable scenario with an explicit accepted-stage blocker."
---

# Task

Implement Stage 12 capacity/load validation for the current backend host.

Done means:

- lightweight capacity harness exists or an existing one is used;
- scenarios cover shell/assets, dashboard summary, settings reads, strategies dashboard/SSE, backtests workstation/results/trades, and controlled preflight/create burst after Stage 8.5;
- scenarios cover manual refresh bursts, autorefresh interval presets, hidden-tab/no-overlap behavior, backend `retry_after_seconds`, stale/degraded source responses and exchange-bound limiter smoke;
- report records host class, commit, config, dataset/cache state, concurrency/duration;
- report records p50/p95/p99, error rate, payload size, RSS, CPU, DB/Redis latency signs, active SSE connections where applicable;
- each area is classified `green`, `yellow`, or `red`;
- yellow/red scenarios include mitigation before public rollout.

## Context / Current State

- Plan targets current backend host capacity, including 1 vCPU / 2 GB class constraints.
- Existing API has Prometheus HTTP metrics.
- This is measurement/evidence work, not feature implementation.
- This stage runs after the main page packages are implemented; unavailable scenarios must be skipped with explicit blocker notes, not silently approximated.

## Requirements (Must)

- Use `backend-performance-evidence`.
- Do not add Node/load-test runtime.
- Prefer Python + already available `httpx` if adding harness.
- Record exact commands.
- Never record secrets/cookies/tokens.
- Do not validate against unfinished placeholder pages as if they were final flows.
- Do not simulate exchange-bound refresh by bypassing backend limiter/cache; use mock/fake adapter or controlled test account path.
- Include refresh/autorefresh storm/coalescing scenario for `/dashboard`, `/strategies` and `/backtests` where implemented.
- Use `publish-ci-deploy` only after harness/report/docs/tests pass.

## Requirements (Should)

- Keep scenarios read-mostly by default.
- Make auth bootstrap configurable without hardcoding credentials.
- Output machine-readable JSON plus readable summary if practical.

## Requirements (Nice-to-have)

- Add a runbook for local/prod capacity smoke.

# Context acquisition protocol

Read `.codex/AGENTS.md`, plan Stage 12, same-origin runbook, then entrypoints. Expand into endpoint routes only for scenario implementation.

Reading budget: keep pre-implementation reading to the smallest sufficient set; default target `<= 8 files`, `<= ~45k tokens` unless this prompt states a tighter number.
Stop reading when touched files, contract surfaces, and acceptance gates are bounded enough to implement safely.
Do not eager-load all `context_sources`, `conditional_bundles`, or `consult_if_needed` files at startup.
If `.codex/agents/.context/promt_manager_state.yaml` or a latest executor final report for this pack exists, read only its completed/open_items/risks/handoff summary before task entrypoints; skip this step if absent.

# Reading manifest

Use front matter `context_sources`.

# Work plan (agent should follow)

1. Design minimal harness and report schema.
2. Implement or reuse harness.
3. Add safe scenario set and auth/session handling without secret logging.
4. Run local smoke.
5. Run current-host capacity assessment if environment is available and target page packages are accepted.
6. Classify results and record mitigations.
7. Run gates.
8. Use `publish-ci-deploy` only after complete success.

# Acceptance criteria (Definition of Done)

- Capacity report contains commands, host class, commit and config.
- No first-paint endpoint transfers unbounded data.
- Polling/SSE loops do not overlap requests under latency.
- Manual refresh bursts do not create parallel fan-out to exchanges.
- Autorefresh preserves selected intervals and respects server retry windows.
- Backtest create path does not execute full compute in API request path.
- p95/RSS/error trends are recorded.
- Known limits are in rollout notes and handoff.

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

- Public API contract: `none` if measurement only.
- Config schema: `compatible-change` if capacity profile settings are added.
- Performance risk: measured evidence.
- Rollout gates: `compatible-change` if capacity report becomes pre-ship gate.

# Files to indicate (expected touched areas)

Use front matter touched paths.

# Non-goals

- Feature implementation.
- Unbounded stress tests against production users.
- Secret-bearing reports.

# Quality gates (must run and pass)

```bash
uv run python tools/load/web_capacity_smoke.py --help
uv run ruff check tools/load tests
uv run pyright
python -m tools.docs.generate_docs_index --check
```

Example capacity smoke:

```bash
uv run python tools/load/web_capacity_smoke.py \
  --base-url http://127.0.0.1:8010 \
  --api-base-url http://127.0.0.1:8000 \
  --profile local-smoke \
  --duration-s 60 \
  --concurrency 10 \
  --scenario dashboard,strategies_live,backtests_workstation
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
