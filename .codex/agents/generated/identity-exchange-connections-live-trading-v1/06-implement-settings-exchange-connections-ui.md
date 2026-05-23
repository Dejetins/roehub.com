---
prompt_name: identity_exchange_connections_v1_06_settings_exchange_connections_ui
repo: roehub.com
branch: main
scope: "Stage 6: complete `/settings` UI for exchange connections with real status, environment, permissions default `read`, rotate/disable/validate flows, and browser secret-leak QA."

language:
  implementation: python_fastapi_jinja_js_css_tests_docs
  agent_report: ru

context_sources:
  always_read:
    - path: .codex/AGENTS.md
      why: "repo UI/security contract"
    - path: docs/architecture/identity/identity-exchange-connections-live-trading-v1.md
      why: "Stage 6 source of truth"
    - path: docs/architecture/identity/exchange-connections-stage-reports/identity-exchange-connections-live-trading-v1-iteration-ledger.md
      why: "shared iteration ledger and next-stage handoff facts"
    - path: docs/architecture/identity/exchange-connections-stage-reports/05-binance-bybit-validation.md
      why: "accepted Stage 5 evidence"
  task_entrypoints:
    - path: apps/web/templates/pages/settings.html
      why: "settings page shell"
      inspect_symbols:
        - settings page layout
        - account fragments
    - path: apps/web/templates/fragments/account/exchange_keys.html
      why: "exchange connections UI fragment"
      inspect_symbols:
        - exchange key form
        - table rows
    - path: apps/web/dist/js/pages/settings.js
      why: "settings interactions and current hardcoded trade/status"
      inspect_symbols:
        - exchange key submit
        - status rendering
        - password cleanup
    - path: tests/unit/apps/web/test_app_routes.py
      why: "web route regression tests"
      inspect_symbols:
        - settings route tests
        - fragment tests
  conditional_bundles:
    api_contract:
      read_when: "UI needs endpoint fields or errors"
      paths:
        - apps/api/routes/ui_account.py
        - tests/unit/apps/api/test_ui_account_routes.py
    browser_qa:
      read_when: "running runtime browser verification"
      paths:
        - docs/architecture/apps/web/web-ui-design-manifest-v1.md
        - apps/web/dist/css/pages/settings.css
    exchange_control:
      read_when: "status model or validation action fields are unclear"
      paths:
        - src/trading/contexts/exchange_control
  consult_if_needed:
    - path: docs/architecture/identity/identity-exchange-keys-storage-2fa-gate-policy-v2.md
      read_when: "secret-safe UI behavior is ambiguous"

style_references:
  - docs/architecture/apps/web/web-ui-design-manifest-v1.md

documentation_continuity:
  old_current_docs:
    - "docs/architecture/identity/identity-exchange-connections-live-trading-v1.md"
    - "docs/architecture/identity/identity-exchange-keys-storage-2fa-gate-policy-v2.md"
    - "docs/architecture/apps/web/web-ui-design-manifest-v1.md"
  new_doc_artifact: "docs/architecture/identity/exchange-connections-stage-reports/06-settings-ui.md"
  canonical_shape: "stage report with Markdown evidence tables: workflow, viewport, expected result, observed result, artifact"
  docs_gate: "python -m tools.docs.generate_docs_index --check"

iteration_ledger:
  path: "docs/architecture/identity/exchange-connections-stage-reports/identity-exchange-connections-live-trading-v1-iteration-ledger.md"
  update_required: true
  required_sections:
    - "Stage status"
    - "Facts for next stages"
    - "Contracts and migrations"
    - "Publish / deploy handoff"

hard_requirements:
  iteration_ledger_update_required: true
  github_yeet_after_validation_required: true
  previous_stage_must_be_accepted: true
  default_permissions_read_required: true
  hardcoded_trade_forbidden: true
  synthetic_status_latency_forbidden: true
  hardcoded_account_limits_forbidden: true
  password_inputs_clear_after_submit_or_failure: true
  browser_secret_grep_required: true
  mobile_layout_required: true

task_toggles:
  implement_ui: true
  implement_api_read_model_adjustments_if_needed: true
  run_browser_qa: true
  github_yeet_after_validation: true

skill_routing:
  - skill: contract-impact-analysis
    use_when: "changing browser-visible defaults, DTO fields, or account limits"
    timing: "before implementation and final report"
    reason: "UI defaults and API fields are contracts"
  - skill: backend-quality-gates
    use_when: "running web/API tests, ruff, pyright"
    timing: "during verification"
    reason: "UI changes touch API and tests"
  - skill: browser-qa-evidence
    use_when: "verifying `/settings` interactions, responsive layout, console/network, secret leakage"
    timing: "after focused tests"
    reason: "settings flow is browser-visible and security-sensitive"
  - skill: playwright
    use_when: "capturing screenshots or browser snapshots"
    timing: "during browser QA"
    reason: "browser evidence artifact collection"

  - skill: github:yeet
    use_when: "stage implementation, validation, stage report, and iteration ledger update are complete"
    timing: "before final report"
    reason: "user requires each validated iteration to be pushed/deployed through GitHub draft PR handoff"

target_envs:
  - local-dev
  - browser

required_literals:
  - "/settings"
  - "permissions"
  - "read"
  - "trade"
  - "environment"
  - "testnet"
  - "mainnet"
  - "valid_readonly"
  - "valid_trade_enabled"
  - "api_secret"

non_goals:
  - "Do not add exchange execution UI."
  - "Do not add order placement controls."
  - "Do not accept hardcoded `trade` permissions."
  - "Do not display real API key/secret/passphrase/ciphertext/HMAC."

final_report_format:
  language: ru
  sections:
    - "Что реализовано"
    - "UI contract"
    - "Browser QA"
    - "Проверки"
    - "Stage 7 readiness"

quality_gates:
  - cmd: "uv run pytest -q tests/unit/apps/web/test_app_routes.py tests/unit/apps/api/test_ui_account_routes.py"
    expect: "passes"
  - cmd: "uv run ruff check apps/web apps/api tests/unit/apps/web tests/unit/apps/api"
    expect: "passes for touched paths"
  - cmd: "uv run pyright apps/api tests/unit/apps/api"
    expect: "passes if typed API code changed"
  - cmd: "python -m tools.docs.generate_docs_index --check"
    expect: "passes after Markdown changes"

  - cmd: "gh --version && gh auth status"
    expect: "GitHub CLI is installed/authenticated before github:yeet; otherwise publish handoff is blocked"

expected_primary_touches:
  - "apps/web/templates/fragments/account/exchange_keys.html"
  - "apps/web/dist/js/pages/settings.js"
  - "apps/web/dist/css/pages/settings.css"
  - "tests/unit/apps/web/test_app_routes.py"
  - "tests/unit/apps/api/test_ui_account_routes.py"
  - "docs/architecture/identity/exchange-connections-stage-reports/06-settings-ui.md"

possible_secondary_touches:
  - "apps/web/templates/pages/settings.html"
  - "apps/api/routes/ui_account.py"
  - "src/trading/contexts/exchange_control/**"
  - "docs/architecture/README.md"

safety_notes:
  - "Browser artifacts must be grep-checked for secret-like markers."
  - "Do not use in-app explanatory text to describe functionality unless it is actual UI copy needed for the workflow."
---

# Task

Complete the `/settings` exchange connections UI for key storage and validation management.

Done means:

- UI uses real backend status, not synthetic latency/status;
- environment and permissions are explicit controls;
- permissions default is `read`;
- `trade` is opt-in only and never hardcoded;
- validate/rotate/disable flows exist;
- secret inputs are cleared after submit and failure;
- browser QA proves desktop/mobile behavior and no secret leakage.

## Context / Current State

Stage 5 must be accepted. If Stage 5 evidence is missing or blocked, stop.

Known current issues from the architecture audit: `/settings` has synthetic exchange status/latency, and JS may hardcode `permissions: "trade"`. This stage must remove those behaviors.

## Requirements (Must)

- Update the iteration ledger with stage status, evidence paths, changed contracts, migrations/config/env, blockers, and facts required by following stages.
- After validation and ledger update, run `github:yeet`: inspect mixed worktree, stage only intended changes, commit, push branch, and open a draft PR. Record branch, commit, PR URL, and deploy/runtime status in the ledger and final report.
- Replace synthetic status/latency with backend connection status.
- Add explicit environment selection.
- Add permissions selector with default `read`.
- Ensure selected permissions value goes to backend.
- Remove hardcoded `trade` from JS submit path.
- Add validate/rotate/disable UI flows.
- Add IP allowlist guidance from backend/runbook state where available.
- Replace hardcoded account limits/counts with backend read model.
- Remove hardcoded `exchange_connections_used=0`, `api_keys_used=0`, and any fake latency/status values from the UI/account read model.
- Clear password inputs after submit and failure.
- Create Stage 6 report with browser QA evidence.

## Requirements (Should)

- Keep UI dense, operational, and consistent with existing settings design.
- Use existing component/style conventions.

## Requirements (Nice-to-have)

- Add accessible status labels without exposing raw exchange errors.

# Context acquisition protocol

Read only in this order and stop once sufficient:

1. `.codex/AGENTS.md`
2. Stage 5 report
3. architecture document Stage 6
4. task entrypoints
5. conditional bundles only for API/browser/style ambiguity

Pre-implementation reading target:

- `<= 8 files`
- `<= ~45k tokens`

Stop reading once UI form, JS submit path, API payload, status render, tests, and browser QA path are bounded.

# Reading manifest

Use front-matter `context_sources` as the canonical reading map. Do not preload unrelated pages.

# Work plan (agent should follow)

Skill routing for this task:

- `contract-impact-analysis`: use for browser-visible defaults and API DTO fields.
- `backend-quality-gates`: use during verification.
- `browser-qa-evidence`: use after focused tests for real `/settings` verification.
- `playwright`: use during browser QA if screenshots/snapshots are needed.

1. Confirm Stage 5 accepted.
2. Update UI template and JS payload/status handling.
3. Adjust API read model only if needed for real counts/status/actions.
4. Add focused web/API tests.
5. Run browser QA and create Stage 6 report.

After the stage-specific implementation and validation steps:

- Update the iteration ledger with stage status, evidence, blockers, and next-stage facts.
- Run `github:yeet` for targeted staging, commit, push, and draft PR. Do not stage unrelated user changes.

# Acceptance criteria (Definition of Done)

- Iteration ledger is updated with facts required by the next stage.
- `github:yeet` publish/deploy handoff is completed after validation, or the stage is marked blocked with the exact reason.
- Authenticated `/settings` opens.
- Add key request is write-only and secret-safe.
- Default permissions value is `read`.
- Hardcoded `trade` is absent from submit path.
- Error path clears secret inputs.
- List shows masked key, status, and last validation.
- Validate/rotate/disable flows work.
- Mobile layout is coherent.
- Browser artifacts contain no secret markers.
- `rg` confirms hardcoded `permissions: "trade"`, synthetic latency/status, and hardcoded account limit counters are absent from the touched UI path.

# Implementation constraints

## Determinism & ordering

- Keep DOM IDs/data attributes stable where tests depend on them.
- Do not create layout shifts from status or button labels.

## API / contracts

- Browser-visible defaults are contract-affecting.
- Do not expose raw validation error bodies.

## Documentation

- Update the iteration ledger before running `github:yeet`; this is the canonical cross-stage handoff document.
- Create Stage 6 report.
- Update architecture only if implementation deviates.
- Review old/current docs listed in `documentation_continuity.old_current_docs`; if they describe stale behavior as current, update them in the same change, otherwise state that no stale text was found.
- Use Markdown tables for browser workflows, desktop/mobile viewports, network checks, console checks, secret cleanup, and artifact grep evidence.
- Run docs-index check after Markdown changes.

## Tests

- Add unit tests for settings render and payload behavior where available.
- Browser verification must include desktop and mobile.

# Files to indicate (expected touched areas)

Primary touches:

- `apps/web/templates/fragments/account/exchange_keys.html`
- `apps/web/dist/js/pages/settings.js`
- `apps/web/dist/css/pages/settings.css`
- `tests/unit/apps/web/test_app_routes.py`
- `tests/unit/apps/api/test_ui_account_routes.py`
- `docs/architecture/identity/exchange-connections-stage-reports/06-settings-ui.md`

Possible secondary touches:

- `apps/web/templates/pages/settings.html`
- `apps/api/routes/ui_account.py`
- `src/trading/contexts/exchange_control/**`
- `docs/architecture/README.md`

# Non-goals

- Marketing page.
- Execution UI.
- Order placement.
- Strategy assignment.
- Full design-system rewrite.

# Quality gates (must run and pass)

- `gh --version && gh auth status`
- `uv run pytest -q tests/unit/apps/web/test_app_routes.py tests/unit/apps/api/test_ui_account_routes.py`
- `uv run ruff check apps/web apps/api tests/unit/apps/web tests/unit/apps/api`
- `uv run pyright apps/api tests/unit/apps/api` if typed API code changed
- `python -m tools.docs.generate_docs_index --check`
- `! rg -n 'permissions: "trade"|exchange_connections_used=0|api_keys_used=0|128 ms|needsAttention' apps/web apps/api`
- Browser QA: authenticated `/settings`, add error path, secret cleanup, status/list, validate/rotate/disable, mobile layout, artifact grep.

# Final output: report format (strict)

Your final message MUST be in Russian and follow exactly:

Your final message MUST include `github:yeet` branch, commit, draft PR URL, and deploy/runtime status.

1. **Что реализовано**
2. **UI contract**
3. **Browser QA**
4. **Проверки**
5. **Stage 7 readiness**
