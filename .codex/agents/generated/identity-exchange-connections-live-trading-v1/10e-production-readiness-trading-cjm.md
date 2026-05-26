---
prompt_name: identity_exchange_connections_v1_10e_production_readiness_trading_cjm
repo: roehub.com
branch: main
scope: "Stage 10E: prove production readiness for trading-only /settings CJM with authenticated Playwright, readonly rejection, optional trade-ready credentials, metrics, audit, docs, and direct-main verification."

language:
  implementation: qa_docs_runtime
  agent_report: ru

context_sources:
  always_read:
    - path: .codex/AGENTS.md
      why: "repo contract, browser/runtime evidence, delivery rules"
    - path: docs/architecture/identity/identity-exchange-connections-live-trading-v1.md
      why: "Stage 10 readiness source of truth"
    - path: docs/architecture/identity/exchange-connections-stage-reports/identity-exchange-connections-live-trading-v1-iteration-ledger.md
      why: "must confirm 10D accepted and update 10E"
    - path: docs/architecture/identity/exchange-connections-stage-reports/10d-non-trading-active-reclassification.md
      why: "accepted data reclassification evidence"
  task_entrypoints:
    - path: apps/web/templates/fragments/account/exchange_keys.html
      why: "settings exchange panel"
      inspect_symbols: ["settings.exchange"]
    - path: apps/web/dist/js/pages/settings.js
      why: "browser connect/re-check/disconnect/history behavior"
      inspect_symbols: ["renderExchangeKeys"]
    - path: apps/api/routes/ui_account.py
      why: "account facade endpoints"
      inspect_symbols: ["exchange_connections"]
    - path: src/trading/contexts/exchange_control/adapters/inbound/http/app.py
      why: "runtime metrics/capabilities"
      inspect_symbols: ["metrics", "capabilities"]
  conditional_bundles:
    runtime_ops:
      read_when: "collecting Mac Studio/OpenBao/Prometheus/Monit evidence"
      paths:
        - docs/runbooks/exchange-secret-management.md
        - infra/macos
        - scripts
    tests:
      read_when: "readiness blocker needs scoped repair"
      paths:
        - tests/unit/contexts/exchange_control
        - tests/unit/apps/api/test_ui_account_routes.py
        - tests/unit/apps/web/test_app_routes.py
        - tests/unit/apps/migrations
  consult_if_needed:
    - path: docs/architecture/identity/exchange-connections-stage-reports/09e-lifecycle-production-readiness.md
      read_when: "previous Playwright login/evidence pattern is needed"

documentation_continuity:
  old_current_docs:
    - "docs/architecture/identity/identity-exchange-connections-live-trading-v1.md"
    - "docs/architecture/identity/exchange-connections-stage-reports/identity-exchange-connections-live-trading-v1-iteration-ledger.md"
  new_doc_artifact: "docs/architecture/identity/exchange-connections-stage-reports/10e-trading-cjm-production-readiness.md"
  canonical_shape: "readiness report with evidence matrix, blockers, and residual risks"
  docs_gate: "python -m tools.docs.generate_docs_index --check"

stage_execution_ledger:
  path: "docs/architecture/identity/exchange-connections-stage-reports/identity-exchange-connections-live-trading-v1-iteration-ledger.md"
  plan_doc: "docs/architecture/identity/identity-exchange-connections-live-trading-v1.md"
  current_stage: "10E"
  required_update: true
  template: .codex/agents/stage_execution_ledger_template.md

hard_requirements:
  previous_stage_10d_must_be_accepted: true
  authenticated_playwright_required: true
  readonly_key_rejected_or_not_active_required: true
  active_only_ready_for_trading_required: true
  no_permissions_selector_required: true
  active_history_only_required: true
  metrics_audit_db_evidence_required: true
  direct_main_push_after_validation_required: true
  no_stage_branch_or_pr: true
  no_secret_leakage: true
  no_trading_execution: true
  concrete_runtime_calls_required: true
  tests_are_not_acceptance: true

task_toggles:
  implementation_changes_allowed: false
  scoped_bugfixes_allowed_if_readiness_blocked: true
  browser_qa_required: true
  runtime_evidence_required: true
  publish_after_success: true
  target_branch: main

skill_routing:
  - skill: browser-qa-evidence
    use_when: "proving /settings CJM and visible browser behavior"
    timing: "during verification"
    reason: "Stage 10E is browser-readiness"
  - skill: playwright
    use_when: "running authenticated production/local browser proof"
    timing: "during verification"
    reason: "user requires e2e proof"
  - skill: backend-quality-gates
    use_when: "running tests/lint/type/docs or fixing scoped blocker"
    timing: "during verification"
    reason: "readiness needs local gates too"
  - skill: contract-impact-analysis
    use_when: "a scoped bugfix changes API, DTO, persistence, config, or browser defaults"
    timing: "before any bugfix finalization"
    reason: "readiness fixes must not silently shift contracts"
  - skill: publish-ci-deploy
    use_when: "readiness evidence/report/ledger are complete"
    timing: "after validation"
    reason: "direct-main delivery and post-deploy verification required"

target_envs: ["production-browser", "mac-studio", "prometheus", "postgres"]

required_literals:
  - "Ready for trading"
  - "read_only_not_supported"
  - "Connect and validate"
  - "Disconnect"
  - "Re-check"
  - "ROEHUB_E2E_BYBIT_MAINNET_TRADE_API_KEY"
  - "ROEHUB_E2E_BYBIT_TESTNET_TRADE_API_KEY"

non_goals:
  - "Do not place orders."
  - "Do not add exchange-execution."
  - "Do not physically delete records."
  - "Do not paste or commit credentials."

final_report_format:
  language: ru
  sections: ["Вердикт", "Production evidence", "Browser/Playwright", "Проверки", "Direct-main delivery", "Residual risk"]

quality_gates:
  - cmd: "uv run pytest -q tests/unit/contexts/exchange_control tests/unit/apps/api/test_ui_account_routes.py tests/unit/apps/web/test_app_routes.py tests/unit/apps/migrations"
    expect: "passes"
  - cmd: "uv run ruff check src/trading/contexts/exchange_control apps/api apps/web tests/unit/contexts/exchange_control tests/unit/apps/api tests/unit/apps/web tests/unit/apps/migrations"
    expect: "passes"
  - cmd: "uv run pyright src/trading/contexts/exchange_control apps/api tests/unit/contexts/exchange_control tests/unit/apps/api"
    expect: "passes"
  - cmd: "python -m tools.docs.generate_docs_index --check"
    expect: "passes"
  - cmd: "authenticated Playwright readonly rejection/not-active flow"
    expect: "passes"
  - cmd: "authenticated Playwright trade-ready flow when env credentials exist"
    expect: "passes or explicit blocked/partial with missing env evidence"
  - cmd: "concrete API/DB/audit/metrics/Prometheus/Monit/OpenBao calls"
    expect: "passes; tests alone are not acceptance"

runtime_acceptance:
  required: true
  commands:
    - cmd: "authenticated Playwright readonly rejection/not-active flow"
      expect: "no read/trade selector; readonly key produces read_only_not_supported/not-active; Active does not contain the attempt"
    - cmd: "curl -fsS \"$ROEHUB_BASE_URL/api/ui/account/exchange-connections?status=active\" -H \"Cookie: $ROEHUB_SESSION_COOKIE\" | jq -e 'all(.items[]; .connection_readiness == \"ready_for_trading\" and .effective_capability == \"trading\")'"
      expect: "public Active API is trading-ready only"
    - cmd: "psql \"$ROEHUB_PG_DSN\" -c \"SELECT label, status, status_reason, permission_summary_json FROM exchange_connections WHERE status='active' ORDER BY created_at DESC LIMIT 20;\""
      expect: "DB active rows are trading-ready by Stage 10 metadata; no secret-bearing columns selected"
    - cmd: "psql \"$ROEHUB_PG_DSN\" -c \"SELECT event_type, target_id, metadata_json, created_at FROM identity_audit_events WHERE event_type IN ('exchange_connection_auto_validated','exchange_connection_rejected','exchange_connection_reclassified','exchange_connection_disabled') ORDER BY created_at DESC LIMIT 20;\""
      expect: "audit evidence exists and is redacted"
    - cmd: "curl -fsS http://127.0.0.1:9205/metrics | rg 'exchange_connection_auto_validation_total|exchange_connection_trading_readiness_total|exchange_connection_reclassification_total'"
      expect: "Stage 10 metrics exist"
    - cmd: "curl -fsS 'http://127.0.0.1:9090/api/v1/query?query=up%7Bjob%3D%22exchange-control%22%7D' | jq"
      expect: "Prometheus sees exchange-control as up"
    - cmd: "monit summary | rg 'roehub_exchange_control|roehub_openbao'"
      expect: "Monit services are OK on Mac Studio"
    - cmd: "curl -fsS \"$OPENBAO_ADDR/v1/sys/health\" | jq '{sealed,initialized}'"
      expect: "OpenBao initialized and unsealed; no tokens printed"
  acceptance_rule: "Stage 10E cannot be accepted from tests alone. Missing trade-enabled credentials must be recorded as partial/blocked for trading-ready proof."

expected_primary_touches:
  - "docs/architecture/identity/exchange-connections-stage-reports/10e-trading-cjm-production-readiness.md"
  - "docs/architecture/identity/exchange-connections-stage-reports/identity-exchange-connections-live-trading-v1-iteration-ledger.md"
  - "docs/architecture/identity/identity-exchange-connections-live-trading-v1.md"
  - "docs/architecture/README.md"

possible_secondary_touches:
  - "apps/web/templates/fragments/account/exchange_keys.html"
  - "apps/web/dist/js/pages/settings.js"
  - "apps/api/routes/ui_account.py"
  - "src/trading/contexts/exchange_control"
  - "tests/unit/contexts/exchange_control"
  - "tests/unit/apps/api/test_ui_account_routes.py"
  - "tests/unit/apps/web/test_app_routes.py"

safety_notes:
  - "Use host-local env or approved secret channel for test credentials; never write values into repo, reports, screenshots, traces, or logs."
  - "Full trading-ready proof requires trade-enabled credentials but still must not place orders."
---

# Task

Execute Stage 10E production readiness for trading-only exchange connection CJM.

Done means:

- Stage 10A-10D are accepted in the ledger;
- authenticated `/settings` Playwright proves no read/trade selector, Active/History only, Connect and validate, Re-check, Disconnect;
- readonly key path is rejected or not active and does not occupy Active/limits;
- Active list contains only `Ready for trading`;
- trade-ready proof runs if env-backed trade-enabled credentials exist; otherwise full trading-ready acceptance is explicitly blocked/partial with missing env evidence;
- API, DB, audit, metrics, Prometheus/Monit/OpenBao evidence are captured when applicable;
- secret/artifact grep passes;
- Stage 10E report and ledger are updated and delivered on `main`.

## Context / Current State

- Stage 10E is an evidence/readiness stage.
- Scoped bugfixes are allowed only when readiness evidence proves a blocker.
- No order placement is allowed.

## Requirements (Must)

- Stop if 10D is not accepted.
- Run local gates.
- Run authenticated Playwright against the correct target environment.
- Prove UI:
  - no permissions selector;
  - mainnet default;
  - testnet advanced/dev control;
  - Active/History only;
  - Connect and validate;
  - Re-check;
  - Disconnect.
- Prove readonly key is not active/success.
- Prove Active list contains only `ready_for_trading`.
- If env-backed trade credentials exist, prove connect -> active ready-for-trading without placing orders.
- If env-backed trade credentials do not exist, do not claim full trading-ready production success; mark that part blocked/partial.
- Capture metrics/audit/DB evidence.
- Run secret artifact grep.
- Create Stage 10E report and update ledger.

## Requirements (Should)

- Include evidence matrix: local gates, browser, API, DB/audit, metrics/ops, docs/direct-main.
- Keep screenshots/traces redacted.
- Verify password manager does not offer API key/secret as Roehub login.

## Requirements (Nice-to-have)

- Include before/after CJM screenshots if safe.

# Context acquisition protocol

Read `.codex/AGENTS.md`, Stage 10 plan, ledger, 10D report, then task entrypoints. Pre-evidence reading target: `<= 12 files`.

# Reading manifest

Use front matter as canonical. Do not inspect execution/order code unless a no-order grep fails.

# Work plan (agent should follow)

1. Confirm `main` and fast-forward pull.
2. Verify 10D accepted.
3. Run focused local gates.
4. Run authenticated Playwright readonly rejection/not-active flow.
5. Run trade-ready flow only when env-backed credentials exist.
6. Collect API/DB/audit/metrics/ops evidence.
7. Run secret/no-order grep.
8. Fix scoped blockers only if necessary.
9. Create Stage 10E report and update ledger.
10. Direct-main delivery and post-deploy verification.

# Acceptance criteria (Definition of Done)

- Browser proof matches Stage 10 CJM.
- Read-only key does not become active.
- Active list is trading-ready only.
- No order/execution path exists.
- Secrets absent from artifacts/reports/log snippets.
- Docs index passes.
- Ledger records final status, evidence, missing trade-credential blocker if any, and direct-main delivery.

# Implementation constraints

## Documentation

- Create `docs/architecture/identity/exchange-connections-stage-reports/10e-trading-cjm-production-readiness.md`.
- Update ledger before final output.

# Files to indicate (expected touched areas)

Use front matter expected touches.

# Non-goals

Trading execution, physical delete, new exchange support, broad refactor.

# Quality gates (must run and pass)

- `uv run pytest -q tests/unit/contexts/exchange_control tests/unit/apps/api/test_ui_account_routes.py tests/unit/apps/web/test_app_routes.py tests/unit/apps/migrations`
- `uv run ruff check src/trading/contexts/exchange_control apps/api apps/web tests/unit/contexts/exchange_control tests/unit/apps/api tests/unit/apps/web tests/unit/apps/migrations`
- `uv run pyright src/trading/contexts/exchange_control apps/api tests/unit/contexts/exchange_control tests/unit/apps/api`
- `python -m tools.docs.generate_docs_index --check`
- Authenticated Playwright readonly rejection/not-active flow
- Optional authenticated Playwright trade-ready flow when env credentials exist
- Runtime API/DB/audit/metrics evidence
- Secret/no-order grep

# Final output: report format (strict)

Your final message MUST be in Russian and follow exactly:

1) **Вердикт**
2) **Production evidence**
3) **Browser/Playwright**
4) **Проверки**
5) **Direct-main delivery**
6) **Residual risk**
