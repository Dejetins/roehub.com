---
prompt_name: web_ui_backend_v1_10_ai_backtest_configurator
repo: roehub.com
branch: current
scope: "Этап 10: AI-assisted draft config for backtest configurator, gated by explicit AI backend design decision."

language:
  implementation: python_fastapi_jinja_css_js
  agent_report: ru

context_sources:
  always_read:
    - path: .codex/AGENTS.md
      why: "security, contracts, skill routing"
    - path: docs/architecture/apps/web/web-ui-backend-implementation-plan-v1.md
      why: "Этап 10 source of truth and AI open question"
    - path: docs/architecture/apps/web/web-ui-design-manifest-v1.md
      why: "configurator visual rules"
  task_entrypoints:
    - path: apps/web/templates/pages/backtests_run.html
      why: "configurator page target"
    - path: apps/web/dist/js/pages/backtests_run.js
      why: "manual configurator integration point"
    - path: apps/api/routes
      why: "API route location for AI routes if design decision exists"
  conditional_bundles:
    ai_design_decision:
      read_when: "always before implementation; if absent, stop and report blocker or create only design artifact if explicitly allowed"
      paths:
        - docs/architecture/apps/web/web-ui-backend-implementation-plan-v1.md
        - docs/web-ui+backend-plan-deep-research.md
    backtest_validation:
      read_when: "when validating AI draft against manual config"
      paths:
        - apps/api/routes/backtests.py
        - src/trading/contexts/backtest/application/services/v2/preflight.py
  consult_if_needed:
    - path: docs/architecture/identity/identity-keycloak-auth-model-v1.md
      read_when: "if auth/session data handling is ambiguous"

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

hard_requirements:
  explicit_ai_backend_design_decision_required: true
  ai_can_only_create_draft: true
  ai_cannot_create_job_directly: true
  manual_apply_preflight_submit_required: true
  no_secrets_in_prompt_or_session: true
  rate_limit_required: true
  cancellation_required: true
  browser_qa_required_if_implemented: true

task_toggles:
  implement_ai_routes_if_decision_exists: true
  implement_ui_draft_assistant: true
  implement_direct_job_creation_by_ai: false
  publish_after_success: true

package_contract:
  depends_on:
    - "08-backtests-history-configurator accepted"
    - "08.5-backtest-runtime-hardening accepted or public rollout blocker documented"
    - "explicit AI backend design decision accepted"
  owns:
    - "apps/api/routes/ai_backtest_config.py"
    - "apps/api/dto/ai_backtest_config.py"
    - "apps/api/wiring/modules/ai_backtest_config.py"
    - "apps/web/dist/js/pages/backtests_ai.js"
    - "apps/web/dist/js/pages/backtests_run.js AI integration only"
    - "apps/web/templates/pages/backtests_run.html AI zone only"
    - "tests/unit/apps/api/test_ai_backtest_config_routes.py"
    - "docs/architecture/apps/web/ai-backtest-configurator-v1.md if design artifact is needed"
  forbidden:
    - "direct /api/backtests/jobs call from AI"
    - "exchange/session/API secrets in prompt/session data"
    - "provider implementation without accepted design decision"
    - "manual validation/preflight bypass"
  integration_points:
    - "manual backtests_run.js apply/preflight/submit flow"
    - "AI provider adapter boundary"
    - "redaction/rate-limit/cancellation contract"
  handoff:
    - "AI draft-only assistant with explicit user apply/preflight/submit"

skill_routing:
  - skill: architecture-design
    use_when: "no explicit AI backend provider/storage/redaction/rate-limit design decision exists"
    timing: "before implementation"
    reason: "AI trust boundary must be designed before code"
  - skill: contract-impact-analysis
    use_when: "adding AI routes, DTOs, prompt/session storage, config, redaction, rate limits, browser defaults"
    timing: "before implementation and final report"
    reason: "AI crosses API, privacy, config, and browser-visible contracts"
  - skill: backend-quality-gates
    use_when: "running AI route/validation tests, ruff, pyright"
    timing: "during verification"
    reason: "backend routes and validation must be tested"
  - skill: browser-qa-evidence
    use_when: "verifying AI draft UI, invalid draft errors, cancellation, no auto-submit"
    timing: "after backend tests"
    reason: "AI configurator is browser-visible"
  - skill: playwright
    use_when: "capturing browser evidence"
    timing: "during browser QA"
    reason: "plan requires Playwright CLI"
  - skill: publish-ci-deploy
    use_when: "explicit AI design exists, all security/tests/browser checks pass, no secret leakage"
    timing: "after verification"
    reason: "full Roehub delivery chain after complete success"

target_envs:
  - local-dev
  - browser
  - github-actions

required_literals:
  - "/api/ai/backtest-config/chat"
  - "/api/ai/backtest-config/stream/{session_id}"
  - "/api/ai/backtest-config/validate"
  - "draft config"
  - "preflight"
  - "no secrets"

non_goals:
  - "Do not implement AI without explicit backend design decision."
  - "Do not let AI call `/api/backtests/jobs` directly."
  - "Do not send exchange keys, session cookies, API keys, or raw private audit logs to AI provider."
  - "Do not bypass manual validation/preflight/submit."

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
  - "Publish/deploy: terminal state publish-ci-deploy or exact reason it was skipped"

quality_gates:
  - cmd: "uv run pytest -q tests/unit/apps/api/test_ai_backtest_config_routes.py tests/unit/apps/web/test_app_routes.py tests/unit/apps/api/test_backtests_routes.py"
    expect: "passes if implementation proceeds; create focused tests"
  - cmd: "uv run ruff check apps/api apps/web src tests/unit/apps/api tests/unit/apps/web"
    expect: "passes for touched paths"
  - cmd: "uv run pyright"
    expect: "passes"
  - cmd: "python -m tools.docs.generate_docs_index --check"
    expect: "passes if docs changed"

expected_primary_touches:
  - "apps/api/routes/ai_backtest_config.py"
  - "apps/api/dto/ai_backtest_config.py"
  - "apps/api/wiring/modules/ai_backtest_config.py"
  - "apps/web/dist/js/pages/backtests_ai.js"
  - "apps/web/dist/js/pages/backtests_run.js"
  - "apps/web/templates/pages/backtests_run.html"
  - "tests/unit/apps/api/test_ai_backtest_config_routes.py"

possible_secondary_touches:
  - "docs/architecture/apps/web/ai-backtest-configurator-v1.md"
  - "docs/architecture/apps/web/web-ui-backend-implementation-plan-v1.md"

safety_notes:
  - "If no design decision exists, do not implement code. Stop with blocker or produce the minimal design artifact only if explicitly allowed by the prompt/user."
  - "AI output is untrusted and must pass same validation as manual config."
---

# Task

Implement Stage 10 AI-assisted backtest draft configurator only if the required AI backend design decision exists.

Done means:

- AI can produce a draft config only;
- user must explicitly apply draft, run preflight, and submit job;
- invalid AI draft shows deterministic validation errors;
- stream cancellation works;
- no secrets/private tokens are sent to AI routes/provider;
- browser evidence exists if implemented.

## Context / Current State

- Stage 10 is intentionally gated by open question: provider, storage, redaction and rate limits require explicit design decision.
- Backtest configurator and validation path must already exist.

## Requirements (Must)

- Before code, verify explicit AI backend design decision exists.
- If missing, stop and report blocker. Do not improvise provider/storage/security.
- AI endpoints must not call job creation.
- AI output must pass normal validation/preflight.
- Add tests for redaction, invalid draft, rate limit/cancellation, no auto-submit.
- Use `publish-ci-deploy` only after full success.

## Requirements (Should)

- Keep prompt/session payload minimal.
- Keep UI draft diff inspectable before apply.
- Rate-limit and cancel long provider calls.

## Requirements (Nice-to-have)

- Add audit event for applying AI draft if audit infrastructure exists.

# Context acquisition protocol

Read `.codex/AGENTS.md`, plan Stage 10, then check for explicit AI backend design decision. Stop if absent.

Reading budget: keep pre-implementation reading to the smallest sufficient set; default target `<= 8 files`, `<= ~45k tokens` unless this prompt states a tighter number.
Stop reading when touched files, contract surfaces, and acceptance gates are bounded enough to implement safely.
Do not eager-load all `context_sources`, `conditional_bundles`, or `consult_if_needed` files at startup.
If `.codex/agents/.context/promt_manager_state.yaml` or a latest executor final report for this pack exists, read only its completed/open_items/risks/handoff summary before task entrypoints; skip this step if absent.

# Reading manifest

Use front matter `context_sources`.

# Work plan (agent should follow)

1. Verify AI design decision exists.
2. If absent, stop with blocker and do not implement code.
3. If present, classify contracts/security.
4. Implement routes/DTOs/services and UI integration.
5. Add tests and browser QA.
6. Run gates.
7. Use `publish-ci-deploy` only after complete success.

# Acceptance criteria (Definition of Done)

- AI draft appears without launching job.
- Invalid AI draft shows validation errors.
- Stream cancellation works.
- Secrets/session/API key data are absent from AI requests.
- User remains in control of apply/preflight/submit.

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
- Security risk requires explicit review/design evidence.

# Files to indicate (expected touched areas)

Use front matter touched paths.

# Non-goals

- Direct job creation by AI.
- Secret access.
- Provider implementation without design decision.

# Quality gates (must run and pass)

```bash
uv run pytest -q tests/unit/apps/api/test_ai_backtest_config_routes.py tests/unit/apps/web/test_app_routes.py tests/unit/apps/api/test_backtests_routes.py
uv run ruff check apps/api apps/web src tests/unit/apps/api tests/unit/apps/web
uv run pyright
python -m tools.docs.generate_docs_index --check
```

Browser QA if implemented:

```bash
export CODEX_HOME="${CODEX_HOME:-$HOME/.codex}"
export PWCLI="$CODEX_HOME/skills/playwright/scripts/playwright_cli.sh"
"$PWCLI" open http://127.0.0.1:8010/backtests/new
"$PWCLI" snapshot
"$PWCLI" screenshot --filename output/playwright/backtests-ai-draft-desktop.png
```

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
