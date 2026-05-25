---
prompt_name: identity_exchange_connections_v1_09e_production_readiness_lifecycle_hardening
repo: roehub.com
branch: main
scope: "Stage 09E: prove production readiness for exchange-connection lifecycle hardening with authenticated Playwright, API, DB, audit, metrics, docs, and direct-main verification."

language:
  implementation: qa_docs_runtime
  agent_report: ru

context_sources:
  always_read:
    - path: .codex/AGENTS.md
      why: "repo contract, browser/runtime evidence rules, direct-main delivery"
    - path: docs/architecture/identity/identity-exchange-connections-live-trading-v1.md
      why: "Stage 09 readiness source of truth"
    - path: docs/architecture/identity/exchange-connections-stage-reports/identity-exchange-connections-live-trading-v1-iteration-ledger.md
      why: "must confirm 09D accepted and update 09E"
    - path: docs/architecture/identity/exchange-connections-stage-reports/09d-e2e-cleanup-controlled-backfill.md
      why: "cleanup/backfill handoff before readiness"
  task_entrypoints:
    - path: apps/web/templates/fragments/account/exchange_keys.html
      why: "settings exchange panel rendered to user"
      inspect_symbols: ["settings.exchange"]
    - path: apps/web/dist/js/pages/settings.js
      why: "browser lifecycle actions and filters"
      inspect_symbols: ["renderExchangeKeys", "connect", "disable", "archive"]
    - path: apps/api/routes/ui_account.py
      why: "public authenticated account facade"
      inspect_symbols: ["exchange_connections", "archive"]
    - path: src/trading/contexts/exchange_control/adapters/inbound/http/app.py
      why: "internal runtime metrics/capabilities"
      inspect_symbols: ["metrics", "capabilities"]
  conditional_bundles:
    runtime_ops:
      read_when: "checking Mac Studio deployment, launchd, Monit, Prometheus, or OpenBao"
      paths:
        - docs/runbooks/exchange-secret-management.md
        - infra/macos
        - scripts
    tests:
      read_when: "local quality gate failures require focused fixes"
      paths:
        - tests/unit/contexts/exchange_control
        - tests/unit/apps/api/test_ui_account_routes.py
        - tests/unit/apps/web/test_app_routes.py
        - tests/unit/apps/migrations
    reports:
      read_when: "readiness evidence conflicts with earlier accepted stages"
      paths:
        - docs/architecture/identity/exchange-connections-stage-reports/09a-lifecycle-domain-persistence.md
        - docs/architecture/identity/exchange-connections-stage-reports/09b-api-ui-list-archive.md
        - docs/architecture/identity/exchange-connections-stage-reports/09c-permission-semantics.md
  consult_if_needed:
    - path: docs/architecture/identity/exchange-connections-stage-reports/08-settings-production-repair.md
      read_when: "production origin, schema repair, or Playwright login mechanics are unclear"

documentation_continuity:
  old_current_docs:
    - "docs/architecture/identity/identity-exchange-connections-live-trading-v1.md"
    - "docs/architecture/identity/exchange-connections-stage-reports/identity-exchange-connections-live-trading-v1-iteration-ledger.md"
  new_doc_artifact: "docs/architecture/identity/exchange-connections-stage-reports/09e-lifecycle-production-readiness.md"
  canonical_shape: "readiness report with evidence matrix and residual risks"
  docs_gate: "python -m tools.docs.generate_docs_index --check"

stage_execution_ledger:
  path: "docs/architecture/identity/exchange-connections-stage-reports/identity-exchange-connections-live-trading-v1-iteration-ledger.md"
  plan_doc: "docs/architecture/identity/identity-exchange-connections-live-trading-v1.md"
  current_stage: "09E"
  required_update: true
  template: .codex/agents/stage_execution_ledger_template.md

hard_requirements:
  previous_stage_09d_must_be_accepted: true
  direct_main_push_after_validation_required: true
  no_stage_branch_or_pr: true
  authenticated_playwright_required: true
  create_validate_disable_archive_assert_hidden_required: true
  metrics_audit_db_evidence_required: true
  no_secret_leakage: true
  no_trading_execution: true

task_toggles:
  implementation_changes_allowed: false
  scoped_bugfixes_allowed_if_readiness_blocked: true
  browser_qa_required: true
  runtime_evidence_required: true
  publish_after_success: true
  target_branch: main

skill_routing:
  - skill: browser-qa-evidence
    use_when: "proving authenticated /settings lifecycle behavior"
    timing: "during verification"
    reason: "Stage 09 readiness is browser-visible"
  - skill: playwright
    use_when: "running authenticated create -> validate -> disable -> archive -> assert hidden flow"
    timing: "during verification"
    reason: "the user requires Playwright proof for working state"
  - skill: backend-quality-gates
    use_when: "running focused tests/lint/type/docs gates or fixing a readiness blocker"
    timing: "during verification"
    reason: "readiness cannot rely only on manual browser evidence"
  - skill: contract-impact-analysis
    use_when: "a scoped bugfix changes API, DTO, persistence, config, or browser-visible defaults"
    timing: "before any bugfix finalization"
    reason: "readiness fixes must not silently shift contracts"
  - skill: publish-ci-deploy
    use_when: "readiness report, ledger, or scoped bugfix is complete"
    timing: "after validation"
    reason: "Stage 09E requires direct-main delivery and post-deploy verification"

target_envs: ["production-browser", "mac-studio", "prometheus", "postgres"]

required_literals:
  - "create -> validate -> disable -> archive -> assert hidden"
  - "exchange_connection_archive_total"
  - "exchange_connection_cleanup_total"
  - "exchange_permission_mismatch_total"
  - "exchange_connection_archived"
  - "permission_mismatch"

non_goals:
  - "Do not implement a new feature unless it is a scoped readiness blocker fix."
  - "Do not add trading execution, order placement, or exchange-execution."
  - "Do not use real secrets in docs, prompts, reports, screenshots, traces, logs, or shell output."
  - "Do not physically delete records."

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
  - cmd: "authenticated Playwright: create -> validate -> disable -> archive -> assert hidden"
    expect: "passes with trace/screenshot paths and secret grep"
  - cmd: "runtime metrics/audit/DB evidence"
    expect: "shows archive/cleanup/mismatch counters, archive audit event, archived row hidden from default list"

expected_primary_touches:
  - "docs/architecture/identity/exchange-connections-stage-reports/09e-lifecycle-production-readiness.md"
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
  - "Use test/smoke credentials only through approved secret channels or environment variables; never paste them into files or reports."
  - "Screenshots and traces must not contain visible full API secret values."
  - "If a readiness blocker requires code changes, keep the fix scoped and update the report/ledger with the exact blocker and evidence."
---

# Task

Execute Stage 09E production readiness for exchange-connection lifecycle hardening.

Done means:

- Stage 09A-09D are accepted in the ledger;
- authenticated browser proof covers create -> validate -> disable -> archive -> assert hidden;
- API, DB, audit, metrics, Prometheus/Monit/OpenBao-relevant runtime evidence are captured when applicable;
- secret/ciphertext/HMAC/token/cookie leakage checks pass for artifacts and reports;
- Stage 09E readiness report and ledger are updated;
- readiness docs are delivered directly to `main` after validation.

## Context / Current State

- Stage 09E is primarily an evidence/readiness stage, not a feature stage.
- Scoped bugfixes are allowed only if readiness evidence proves a blocker.
- Trading execution remains out of scope.
- User-visible correctness depends on the real `/settings` flow, not only unit tests.

## Requirements (Must)

- Read context using the protocol below and stop once sufficient.
- Stop if Stage 09D is not accepted in the ledger.
- Run authenticated Playwright against the correct target environment.
- Prove the lifecycle flow:
  - create a test/e2e connection with a safe label prefix;
  - validate or record deterministic skip/failure according to live validation env;
  - disable;
  - archive;
  - assert hidden from default UI/API list;
  - assert visible only through explicit archive/history path if supported.
- Verify status/permission display does not show `permission_mismatch` or readonly mismatch as normal successful trade readiness.
- Capture DB evidence for lifecycle timestamps and audit evidence for `exchange_connection_archived`.
- Capture metrics evidence for archive/cleanup/mismatch metrics.
- Run artifact/log grep for secret-like markers without printing secrets.
- Create Stage 09E report and update ledger after validation and before final output.

## Requirements (Should)

- Include compact evidence tables: local gates, browser evidence, API evidence, DB/audit evidence, metrics/ops evidence, docs/direct-main evidence.
- Prefer deterministic test labels and cleanup them through disable/archive before finishing.
- If external exchange validation env is absent, record skip policy and prove lifecycle still works without pretending live exchange validation passed.

## Requirements (Nice-to-have)

- Include a short residual-risk table for any external validation, browser, deployment, or monitoring evidence that could not be collected.

# Context acquisition protocol

Read only in this order:

1. `.codex/AGENTS.md`
2. Stage 09 section in the plan
3. iteration ledger
4. Stage 09D report
5. task entrypoints
6. conditional bundles only when needed

Pre-implementation reading target: `<= 12 files` before running evidence collection.

# Reading manifest

Use front matter as the canonical reading map. Do not inspect unrelated trading execution code unless a no-order grep fails.

# Work plan (agent should follow)

1. Confirm current branch is `main` and pull `origin/main` fast-forward.
2. Verify Stage 09D accepted.
3. Run focused local quality gates.
4. Run authenticated Playwright lifecycle proof.
5. Collect API, DB, audit, metrics, Prometheus/Monit and OpenBao relevant evidence.
6. Run no-secret and no-order grep checks over artifacts/logs/docs.
7. If a readiness blocker appears, either fix narrowly with tests or mark 09E blocked.
8. Create Stage 09E report and update ledger.
9. Direct-main commit/push and CI/deploy/post-deploy verification.

# Acceptance criteria (Definition of Done)

- Playwright evidence proves create -> validate/skip -> disable -> archive -> assert hidden.
- Default list/API does not show archived e2e connection.
- Explicit history/archive path shows archived record when supported.
- Metrics and audit evidence are present and bounded.
- No secret-like values are present in artifacts or docs.
- No order placement/execution path was added.
- Docs index passes.
- Ledger records final Stage 09 status, evidence, blockers/residual risk, and direct-main delivery.

# Implementation constraints

## Readiness

- Do not claim production readiness from local tests alone.
- If production/browser/runtime checks are unavailable, mark readiness blocked and record the missing evidence.

## Documentation

- Create `docs/architecture/identity/exchange-connections-stage-reports/09e-lifecycle-production-readiness.md`.
- Update the shared iteration ledger after validation and before final response.
- Keep all evidence secret-free and concise.

# Files to indicate (expected touched areas)

Use front matter expected touches.

# Non-goals

- Any trading execution behavior.
- Physical delete.
- New exchange support.
- Broad refactor.

# Quality gates (must run and pass)

- `uv run pytest -q tests/unit/contexts/exchange_control tests/unit/apps/api/test_ui_account_routes.py tests/unit/apps/web/test_app_routes.py tests/unit/apps/migrations`
- `uv run ruff check src/trading/contexts/exchange_control apps/api apps/web tests/unit/contexts/exchange_control tests/unit/apps/api tests/unit/apps/web tests/unit/apps/migrations`
- `uv run pyright src/trading/contexts/exchange_control apps/api tests/unit/contexts/exchange_control tests/unit/apps/api`
- `python -m tools.docs.generate_docs_index --check`
- Authenticated Playwright: `create -> validate -> disable -> archive -> assert hidden`
- Runtime evidence: API, DB, audit, metrics/Prometheus/Monit as applicable
- Secret/artifact grep

# Final output: report format (strict)

Your final message MUST be in Russian and follow exactly:

1) **Вердикт**
2) **Production evidence**
3) **Browser/Playwright**
4) **Проверки**
5) **Direct-main delivery**
6) **Residual risk**
