---
prompt_name: identity_exchange_connections_v1_11_strategy_binding_guard
repo: roehub.com
branch: main
scope: "Stage 11: implement strategy-to-exchange-connection usage registry and guard Disconnect/Archive while active trading strategy bindings exist, without adding exchange execution or order placement."

language:
  implementation: python_fastapi_postgres_js
  agent_report: ru

context_sources:
  always_read:
    - path: .codex/AGENTS.md
      why: "repo contract, DDD boundaries, runtime evidence, direct-main delivery rules"
    - path: docs/architecture/identity/identity-exchange-connections-live-trading-v1.md
      why: "Stage 11 source of truth"
    - path: docs/architecture/identity/exchange-connections-stage-reports/identity-exchange-connections-live-trading-v1-iteration-ledger.md
      why: "must confirm 10E accepted and update Stage 11 handoff/evidence"
    - path: docs/architecture/identity/exchange-connections-stage-reports/10e-trading-cjm-production-readiness.md
      why: "accepted Stage 10 readiness evidence; Stage 11 must not start without it"
  task_entrypoints:
    - path: src/trading/contexts/exchange_control/application/connections.py
      why: "exchange connection lifecycle commands that must call the usage guard before disable/archive"
      inspect_symbols: ["disable", "archive", "rotate"]
    - path: src/trading/contexts/exchange_control/adapters/outbound/postgres_connections.py
      why: "Postgres-backed exchange connection persistence and lifecycle mutation boundary"
      inspect_symbols: ["ExchangeConnection", "Postgres"]
    - path: src/trading/contexts/strategy/domain/entities/strategy.py
      why: "existing owner-scoped Strategy aggregate; binding must reference strategies without changing execution semantics"
      inspect_symbols: ["Strategy"]
    - path: src/trading/contexts/strategy/application/ports/repositories/strategy_repository.py
      why: "strategy repository ownership lookup pattern"
      inspect_symbols: ["StrategyRepository"]
    - path: apps/api/routes/ui_account.py
      why: "account exchange-connection facade, DTO mapping, CSRF/recent-auth mutation surface"
      inspect_symbols: ["exchange_connections"]
    - path: apps/web/templates/fragments/account/exchange_keys.html
      why: "/settings exchange connection visible state and Disconnect action"
      inspect_symbols: ["exchange"]
  conditional_bundles:
    strategy_api:
      read_when: "choosing the correct strategy configuration API or adding a minimal binding route/tool"
      paths:
        - apps/api/routes/strategies.py
        - src/trading/contexts/strategy/application/use_cases
        - src/trading/contexts/strategy/adapters/outbound/persistence/postgres/strategy_repository.py
        - docs/architecture/strategy/strategy-api-immutable-crud-clone-run-control-v1.md
    web_behavior:
      read_when: "updating /settings row rendering, network calls, or action labels"
      paths:
        - apps/web/dist/js/pages/settings.js
        - tests/unit/apps/web/test_app_routes.py
        - tests/unit/apps/api/test_ui_account_routes.py
    migrations:
      read_when: "adding strategy_exchange_bindings persistence"
      paths:
        - migrations/postgres/0008_exchange_connections_v1.sql
        - alembic/versions/20260215_0001_strategy_storage_v1.py
        - tests/unit/apps/migrations
    runtime_ops:
      read_when: "collecting post-deploy runtime, metrics, Prometheus, Monit, or Mac Studio evidence"
      paths:
        - docs/runbooks/exchange-secret-management.md
        - infra/macos
        - scripts
  consult_if_needed:
    - path: docs/architecture/identity/exchange-connections-stage-reports/09e-lifecycle-production-readiness.md
      read_when: "previous lifecycle Playwright and cleanup evidence pattern is needed"
    - path: docs/architecture/identity/exchange-connections-stage-reports/10c-settings-trading-cjm-ui.md
      read_when: "current /settings Active/History CJM evidence is needed"

documentation_continuity:
  old_current_docs:
    - "docs/architecture/identity/identity-exchange-connections-live-trading-v1.md"
    - "docs/architecture/identity/exchange-connections-stage-reports/identity-exchange-connections-live-trading-v1-iteration-ledger.md"
    - "docs/architecture/strategy/strategy-api-immutable-crud-clone-run-control-v1.md"
  new_doc_artifact: "docs/architecture/identity/exchange-connections-stage-reports/11-strategy-binding-guard.md"
  canonical_shape: "stage report with scope, contract impact, runtime evidence matrix, blockers, and next-stage handoff"
  docs_gate: "python -m tools.docs.generate_docs_index --check"

stage_execution_ledger:
  path: "docs/architecture/identity/exchange-connections-stage-reports/identity-exchange-connections-live-trading-v1-iteration-ledger.md"
  plan_doc: "docs/architecture/identity/identity-exchange-connections-live-trading-v1.md"
  current_stage: "11"
  required_update: true
  template: .codex/agents/stage_execution_ledger_template.md

hard_requirements:
  previous_stage_10e_must_be_accepted: true
  no_exchange_execution: true
  no_order_placement: true
  no_strategy_runner_changes_unless_required_for_imports: true
  stable_connection_id_binding_required: true
  disconnect_archive_guard_required: true
  deterministic_conflict_required: true
  rotate_allowed_for_used_connection: true
  owner_scoping_required: true
  csrf_recent_auth_preserved: true
  metrics_audit_required: true
  browser_evidence_required: true
  concrete_runtime_calls_required: true
  tests_are_not_acceptance: true
  direct_main_push_after_validation_required: true
  no_stage_branch_or_pr: true
  no_secret_leakage: true

task_toggles:
  implementation_changes_allowed: true
  migration_allowed: true
  browser_ui_changes_allowed: true
  internal_admin_or_test_tool_allowed_for_binding_acceptance: true
  publish_after_success: true
  target_branch: main

skill_routing:
  - skill: architecture-design
    use_when: "choosing the bounded-context ownership for strategy_exchange_bindings before implementation"
    timing: "before implementation"
    reason: "Stage 11 crosses strategy and exchange-control boundaries"
  - skill: contract-impact-analysis
    use_when: "finalizing API/DTO/persistence/config/browser behavior changes"
    timing: "before finalizing implementation"
    reason: "disconnect/archive will now return deterministic 409 when a connection is in use"
  - skill: backend-quality-gates
    use_when: "running focused tests/lint/type/docs and triaging failures"
    timing: "during verification"
    reason: "backend and migration gates are required quality gates"
  - skill: browser-qa-evidence
    use_when: "proving /settings shows usage and blocks/explains Disconnect"
    timing: "during verification"
    reason: "Stage 11 has browser-visible behavior"
  - skill: playwright
    use_when: "running authenticated browser evidence"
    timing: "during verification"
    reason: "user requires end-to-end browser proof"
  - skill: publish-ci-deploy
    use_when: "Stage 11 validation is complete"
    timing: "after validation"
    reason: "direct-main push, CI/deploy watch, and post-deploy verification are required"

target_envs: ["local-runtime", "mac-studio", "postgres", "prometheus", "production-browser"]

required_literals:
  - "exchange_connection_in_use"
  - "Cannot disconnect"
  - "used_by_strategies_count"
  - "active_strategy_bindings_count"
  - "/api/ui/strategies/{strategy_id}/exchange-bindings"
  - "/api/ui/strategies/{strategy_id}/exchange-bindings/{binding_id}/disable"
  - "strategy_exchange_binding_created"
  - "exchange_connection_disconnect_blocked"
  - "exchange_connection_usage_guard_total"

non_goals:
  - "Do not add exchange-execution."
  - "Do not place, simulate, or submit orders."
  - "Do not create a signal-to-execution contract."
  - "Do not reactivate archived exchange connections."
  - "Do not physically delete exchange connections, bindings, credentials, or audit rows."
  - "Do not store API secrets in strategy binding rows."

final_report_format:
  language: ru
  sections: ["Вердикт", "Что изменено", "Runtime evidence", "Проверки", "Contract impact", "Direct-main delivery", "Residual risk"]

quality_gates:
  - cmd: "uv run pytest -q tests/unit/contexts/exchange_control tests/unit/contexts/strategy tests/unit/apps/api/test_ui_account_routes.py tests/unit/apps/web/test_app_routes.py tests/unit/apps/migrations"
    expect: "passes"
  - cmd: "uv run ruff check src/trading/contexts/exchange_control src/trading/contexts/strategy apps/api apps/web tests/unit/contexts/exchange_control tests/unit/contexts/strategy tests/unit/apps/api tests/unit/apps/web tests/unit/apps/migrations"
    expect: "passes"
  - cmd: "uv run pyright src/trading/contexts/exchange_control src/trading/contexts/strategy apps/api tests/unit/contexts/exchange_control tests/unit/contexts/strategy tests/unit/apps/api"
    expect: "passes"
  - cmd: "python -m tools.docs.generate_docs_index --check"
    expect: "passes"
  - cmd: "runtime/API/DB/browser/metrics acceptance calls listed below"
    expect: "passes; tests alone are not acceptance"

runtime_acceptance:
  required: true
  acceptance_rule: "Stage 11 cannot be accepted from unit tests, direct DB inserts, or fake/in-memory validation alone. It must prove the guard through real API/tool calls, DB evidence, browser behavior, audit, and metrics."
  commands:
    - cmd: "Create or locate an owner-scoped active trading-ready exchange connection via the accepted Stage 10 flow; do not print or store secrets."
      expect: "connection_id is active and ready_for_trading"
    - cmd: "curl -i -X POST \"$ROEHUB_BASE_URL/api/ui/strategies/$STRATEGY_ID/exchange-bindings\" -H \"Origin: $ROEHUB_BASE_URL\" -H \"Cookie: $ROEHUB_SESSION_COOKIE\" -H \"X-CSRF-Token: $ROEHUB_CSRF_TOKEN\" --data \"{\\\"exchange_connection_id\\\":\\\"$CONNECTION_ID\\\",\\\"usage_mode\\\":\\\"trading\\\"}\""
      expect: "HTTP 200/201; response includes binding_id, binding_status=active, usage_mode=trading, owner_user_id implicitly matches strategy and connection owner"
    - cmd: "curl -fsS \"$ROEHUB_BASE_URL/api/ui/strategies/$STRATEGY_ID/exchange-bindings\" -H \"Cookie: $ROEHUB_SESSION_COOKIE\" | CONNECTION_ID=\"$CONNECTION_ID\" jq -e '.items[] | select(.exchange_connection_id == env.CONNECTION_ID and .binding_status == \"active\")'"
      expect: "binding is observable through strategy binding API; not seeded only by DB insert"
    - cmd: "curl -fsS \"$ROEHUB_BASE_URL/api/ui/account/exchange-connections?status=active\" -H \"Cookie: $ROEHUB_SESSION_COOKIE\" | jq '.items[] | {connection_id, used_by_strategies_count, active_strategy_bindings_count}'"
      expect: "target connection exposes used_by_strategies_count or active_strategy_bindings_count >= 1"
    - cmd: "curl -i -X POST \"$ROEHUB_BASE_URL/api/ui/account/exchange-connections/$CONNECTION_ID/disable\" -H \"Origin: $ROEHUB_BASE_URL\" -H \"Cookie: $ROEHUB_SESSION_COOKIE\" -H \"X-CSRF-Token: $ROEHUB_CSRF_TOKEN\""
      expect: "HTTP 409 with error code exchange_connection_in_use while active trading binding exists"
    - cmd: "curl -i -X POST \"$ROEHUB_BASE_URL/api/ui/account/exchange-connections/$CONNECTION_ID/archive\" -H \"Origin: $ROEHUB_BASE_URL\" -H \"Cookie: $ROEHUB_SESSION_COOKIE\" -H \"X-CSRF-Token: $ROEHUB_CSRF_TOKEN\""
      expect: "HTTP 409 with error code exchange_connection_in_use while active trading binding exists"
    - cmd: "Rotate the same connection with env-backed valid credentials when available."
      expect: "rotate is not blocked merely because the connection has an active strategy binding; missing credentials produce blocked/partial evidence, not fake success"
    - cmd: "curl -i -X POST \"$ROEHUB_BASE_URL/api/ui/strategies/$STRATEGY_ID/exchange-bindings/$BINDING_ID/disable\" -H \"Origin: $ROEHUB_BASE_URL\" -H \"Cookie: $ROEHUB_SESSION_COOKIE\" -H \"X-CSRF-Token: $ROEHUB_CSRF_TOKEN\""
      expect: "HTTP 200/204; binding no longer active"
    - cmd: "curl -i -X POST \"$ROEHUB_BASE_URL/api/ui/account/exchange-connections/$CONNECTION_ID/disable\" -H \"Origin: $ROEHUB_BASE_URL\" -H \"Cookie: $ROEHUB_SESSION_COOKIE\" -H \"X-CSRF-Token: $ROEHUB_CSRF_TOKEN\""
      expect: "Disconnect succeeds after binding release; target connection disappears from Active"
    - cmd: "psql \"$ROEHUB_PG_DSN\" -c \"SELECT binding_id, owner_user_id, strategy_id, exchange_connection_id, usage_mode, binding_status, created_at, disabled_at, archived_at FROM strategy_exchange_bindings ORDER BY created_at DESC LIMIT 20;\""
      expect: "DB evidence shows binding lifecycle without secret-bearing columns"
    - cmd: "psql \"$ROEHUB_PG_DSN\" -c \"SELECT event_type, target_id, metadata_json, created_at FROM identity_audit_events WHERE event_type IN ('strategy_exchange_binding_created','strategy_exchange_binding_disabled','strategy_exchange_binding_archived','exchange_connection_disconnect_blocked') ORDER BY created_at DESC LIMIT 20;\""
      expect: "audit events exist and are redacted"
    - cmd: "curl -fsS http://127.0.0.1:9205/metrics | rg 'exchange_connection_usage_guard_total|strategy_exchange_binding_total|exchange_connection_active_strategy_bindings'"
      expect: "Stage 11 metrics exist with bounded labels"
    - cmd: "authenticated Playwright /settings proof"
      expect: "used connection shows usage count and blocks/explains Disconnect; after binding release, Disconnect succeeds and row hides from Active"

expected_primary_touches:
  - "src/trading/contexts/exchange_control/application/connections.py"
  - "src/trading/contexts/exchange_control/adapters/outbound/postgres_connections.py"
  - "src/trading/contexts/strategy"
  - "apps/api/routes/ui_account.py"
  - "apps/web/templates/fragments/account/exchange_keys.html"
  - "apps/web/dist/js/pages/settings.js"
  - "migrations/postgres"
  - "tests/unit/contexts/exchange_control"
  - "tests/unit/contexts/strategy"
  - "tests/unit/apps/api/test_ui_account_routes.py"
  - "tests/unit/apps/web/test_app_routes.py"
  - "docs/architecture/identity/exchange-connections-stage-reports/11-strategy-binding-guard.md"
  - "docs/architecture/identity/exchange-connections-stage-reports/identity-exchange-connections-live-trading-v1-iteration-ledger.md"

possible_secondary_touches:
  - "apps/api/dto/ui_account.py"
  - "apps/api/wiring/modules"
  - "src/trading/contexts/exchange_control/adapters/inbound/http/app.py"
  - "src/trading/contexts/strategy/application/use_cases"
  - "src/trading/contexts/strategy/application/ports/repositories"
  - "src/trading/contexts/strategy/adapters/outbound/persistence/postgres"
  - "docs/architecture/identity/identity-exchange-connections-live-trading-v1.md"
  - "docs/architecture/strategy/strategy-api-immutable-crud-clone-run-control-v1.md"
  - "docs/architecture/README.md"

safety_notes:
  - "Never write secrets, cookies, tokens, ciphertext, HMAC, raw exchange bodies, or real API credentials into code, docs, reports, logs, screenshots, or ledger."
  - "If usage registry/guard storage is unavailable, lifecycle mutation must fail closed rather than silently disconnect a used connection."
  - "Binding rows must contain strategy/connection identifiers and lifecycle state only; no API key material."
---

# Task

Implement Stage 11 strategy binding guard for exchange connections.

Done means:

- Stage 10E is accepted in the ledger before implementation starts;
- an owner-scoped strategy-to-exchange-connection usage registry exists;
- active `usage_mode=trading` bindings block `Disconnect`/backend disable and archive with deterministic `409 exchange_connection_in_use`;
- rotation is still allowed for a used connection because `connection_id` is stable and credential version is replaceable;
- `/settings` shows usage count and explains why an in-use connection cannot be disconnected;
- binding release lets disconnect succeed;
- audit, metrics, DB evidence, browser evidence, docs, and ledger are complete;
- changes are delivered directly to `main` after validation.

## Context / Current State

Context ledger from the previous iteration:

- completed:
  - Stage 09 accepted lifecycle `active -> disabled -> archived`, archive endpoint, history/default-active semantics, audit and metrics.
  - Stage 10 plan makes `/settings` trading-only: no read/trade selector, auto-validation on create/rotate, Active/History only, `Disconnect` instead of user-facing `Disable`.
  - Exchange execution and order placement remain explicitly out of scope.
- open_items:
  - A user must not be able to disconnect a key currently selected by a strategy.
  - There is no accepted signal-to-execution contract, so Stage 11 must solve only configuration lifecycle safety.
- contract_changes:
  - Additive usage count/read-model fields are allowed.
  - Disconnect/archive now may return deterministic `409 exchange_connection_in_use`.
  - Persistence may add `strategy_exchange_bindings` or a semantically equivalent owner-scoped usage registry.
- risks:
  - A direct DB-only binding seed would not prove the product contract.
  - Putting the binding inside execution runtime would create a premature dependency on a module that does not exist.
  - Secret-bearing data in binding/audit/metrics would violate the custody boundary.
- next_focus:
  - Implement usage registry and guard.
  - Prove runtime behavior with API/tool calls, DB, browser, audit, and metrics.
  - Update docs/ledger and deliver through direct-main workflow.

## Requirements (Must)

- Read context using the protocol below and stop early once sufficient.
- Stop if Stage 10E is not accepted in the ledger, unless the task is explicitly changed to repair Stage 10E.
- Implement only Stage 11.
- Do not add exchange-execution, execution intents, risk engine, order ledger, order submission, or fake order simulation.
- Model binding as strategy configuration:
  - owner-scoped;
  - references stable `strategy_id` and `exchange_connection_id`;
  - `usage_mode=trading` for v1;
  - lifecycle/status supports at least active and disabled/paused; archived if needed for history.
- Enforce owner matching between strategy, connection, and current user.
- Block disable/disconnect and archive while active trading bindings exist.
- Return deterministic conflict:
  - HTTP status `409`;
  - error code `exchange_connection_in_use`;
  - user-facing message equivalent to "Cannot disconnect. This exchange account is used by N active strategies. Pause or reassign strategies first."
- Keep rotate allowed for used connections, subject to accepted Stage 10 trading-ready validation.
- Add bounded metrics and redacted audit events.
- Update `/settings` to show usage count and block or clearly explain Disconnect when in use.
- Primary binding API contract:
  - `GET /api/ui/strategies/{strategy_id}/exchange-bindings`;
  - `POST /api/ui/strategies/{strategy_id}/exchange-bindings`;
  - `POST /api/ui/strategies/{strategy_id}/exchange-bindings/{binding_id}/disable`.
- If these HTTP endpoints cannot be safely added in Stage 11, a local/admin command tool is allowed only as a documented fallback. The report must explain the deviation and still prove lifecycle guard through real account lifecycle API calls. A direct DB insert is never acceptance.
- Preserve CSRF, same-origin, and recent-auth mutation gates.
- Add or update targeted tests, but do not treat tests as acceptance.
- Create `docs/architecture/identity/exchange-connections-stage-reports/11-strategy-binding-guard.md`.
- Update the plan and iteration ledger after validation and before final report.
- Run docs index gate if Markdown docs changed.
- Deliver successful Stage 11 directly on `main`; do not create a stage branch or draft PR.

## Requirements (Should)

- Prefer `strategy_exchange_bindings` as the table/read-model name unless an existing strategy persistence convention makes another name clearly better.
- Keep binding repository/application ports near the strategy/exchange boundary rather than leaking ORM records into routes.
- Make lifecycle guard fail closed if binding read-model cannot be checked.
- Use DTO fields that are easy for UI to display:
  - `used_by_strategies_count` or `active_strategy_bindings_count`;
  - optional compact strategy references only if they are safe and owner-scoped.
- Use bounded metric labels only: no `user_id`, `connection_id`, `strategy_id`, API key, or raw error labels.

## Requirements (Nice-to-have)

- If existing strategy UI has a suitable configuration surface, expose binding there. Otherwise keep Stage 11 binding creation as minimal API/tool surface for acceptance and future UI work.
- Provide a small ops/admin command for controlled binding cleanup if test bindings need release after e2e.

# Context acquisition protocol

Read only in this order and stop once sufficient:

1. `.codex/AGENTS.md`
2. Stage 11 section in `docs/architecture/identity/identity-exchange-connections-live-trading-v1.md`
3. iteration ledger and Stage 10E report
4. task entrypoints
5. conditional bundles only for touched contracts or failing checks
6. consult-if-needed references only for blockers, ambiguity, or conflicts

Do not eagerly preload all listed sources.

Pre-implementation reading target:

- `<= 10 files`
- `<= ~45k-60k tokens`

Stop reading once:

- binding ownership location is chosen;
- touched API/DTO/persistence contracts are identified;
- lifecycle guard insertion points are clear;
- browser behavior can be implemented and verified;
- runtime acceptance commands are executable.

# Reading manifest

Always read:

- `.codex/AGENTS.md`
- `docs/architecture/identity/identity-exchange-connections-live-trading-v1.md`
- `docs/architecture/identity/exchange-connections-stage-reports/identity-exchange-connections-live-trading-v1-iteration-ledger.md`
- `docs/architecture/identity/exchange-connections-stage-reports/10e-trading-cjm-production-readiness.md`

Primary implementation entrypoints:

- `src/trading/contexts/exchange_control/application/connections.py`
- `src/trading/contexts/exchange_control/adapters/outbound/postgres_connections.py`
- `src/trading/contexts/strategy/domain/entities/strategy.py`
- `src/trading/contexts/strategy/application/ports/repositories/strategy_repository.py`
- `apps/api/routes/ui_account.py`
- `apps/web/templates/fragments/account/exchange_keys.html`

# Work plan (agent should follow)

1. Confirm Stage 10E is accepted and direct-main delivered.
2. Decide binding ownership and repository shape using current strategy/exchange-control boundaries.
3. Add persistence/model/port for strategy exchange bindings.
4. Add application guard used by disable/archive lifecycle commands.
5. Add the primary strategy binding API endpoints or explicitly documented fallback tool to create/list/release bindings for real acceptance.
6. Add account DTO usage count and `/settings` UI behavior.
7. Add audit events and metrics.
8. Add focused tests for domain/application/API/UI/migration behavior.
9. Run local quality gates.
10. Run runtime acceptance calls: create binding, block disconnect/archive, release binding, disconnect succeeds, DB/audit/metrics/browser evidence.
11. Create Stage 11 report and update ledger/plan.
12. Publish directly to `main`, watch CI/deploy, and record post-deploy evidence.

# Acceptance criteria (Definition of Done)

- Stage 10E accepted before Stage 11 starts.
- Active trading binding exists and is observable through API/tool/read-model.
- Default Active account list exposes usage count for the connection.
- Disable/Disconnect returns `409 exchange_connection_in_use` while active binding exists.
- Archive returns `409 exchange_connection_in_use` while active binding exists.
- Rotate remains allowed for used connection when valid env-backed credentials are available; if credentials are missing, that proof is `blocked/partial`, not fake success.
- Binding release changes status and allows disconnect.
- `/settings` authenticated Playwright proves usage count and blocked/explained Disconnect.
- DB evidence shows binding lifecycle without secrets.
- Audit events are written and redacted.
- Metrics exist and have bounded labels.
- No orders, execution module, execution intent, risk engine, or physical delete were added.
- Stage report and ledger are updated.
- Direct-main delivery, CI/deploy, and post-deploy runtime evidence are recorded.

# Implementation constraints

## Documentation

- Update the old/current docs listed in `documentation_continuity`.
- Create `docs/architecture/identity/exchange-connections-stage-reports/11-strategy-binding-guard.md`.
- Keep the Stage 11 report in Russian.
- Keep raw credentials, cookies, tokens, ciphertext, HMAC, and raw provider responses out of docs and ledger.
- Run `python -m tools.docs.generate_docs_index --check`.

## Contract

- Classify impact across API/DTO, persistence, config/env, browser behavior, runtime/ops, rollback.
- Treat `409 exchange_connection_in_use` as an intentional compatible conflict condition, not an unhandled error.
- Keep existing lifecycle states and Stage 10 readiness semantics intact.

## Runtime

- Tests/lint/type checks are required quality gates but are not acceptance.
- Runtime calls must be recorded in the Stage 11 report with sanitized commands/results.
- If a required runtime dependency is missing, mark the stage `blocked` or `partial` in the ledger and do not push as accepted.

# Files to indicate (expected touched areas)

Likely:

- `src/trading/contexts/exchange_control/application/connections.py`
- `src/trading/contexts/exchange_control/adapters/outbound/postgres_connections.py`
- `src/trading/contexts/strategy/**`
- `apps/api/routes/ui_account.py`
- `apps/api/dto/ui_account.py`
- `apps/web/templates/fragments/account/exchange_keys.html`
- `apps/web/dist/js/pages/settings.js`
- `migrations/postgres/**`
- `tests/unit/contexts/exchange_control/**`
- `tests/unit/contexts/strategy/**`
- `tests/unit/apps/api/test_ui_account_routes.py`
- `tests/unit/apps/web/test_app_routes.py`
- `tests/unit/apps/migrations/**`
- `docs/architecture/identity/identity-exchange-connections-live-trading-v1.md`
- `docs/architecture/identity/exchange-connections-stage-reports/identity-exchange-connections-live-trading-v1-iteration-ledger.md`
- `docs/architecture/identity/exchange-connections-stage-reports/11-strategy-binding-guard.md`

# Non-goals

- Exchange execution.
- Order placement.
- Strategy runner behavior changes.
- Signal-to-execution architecture.
- Risk engine.
- Physical deletion.
- Reactivation of archived connections.
- Read-only exchange key product workflows.

# Quality gates (must run and pass)

Run focused gates first:

```bash
uv run pytest -q tests/unit/contexts/exchange_control tests/unit/contexts/strategy tests/unit/apps/api/test_ui_account_routes.py tests/unit/apps/web/test_app_routes.py tests/unit/apps/migrations
uv run ruff check src/trading/contexts/exchange_control src/trading/contexts/strategy apps/api apps/web tests/unit/contexts/exchange_control tests/unit/contexts/strategy tests/unit/apps/api tests/unit/apps/web tests/unit/apps/migrations
uv run pyright src/trading/contexts/exchange_control src/trading/contexts/strategy apps/api tests/unit/contexts/exchange_control tests/unit/contexts/strategy tests/unit/apps/api
python -m tools.docs.generate_docs_index --check
```

Then run concrete runtime acceptance. Do not replace this with tests.

# Final output: report format (strict)

Write the final report in Russian:

## Вердикт

- `accepted`, `blocked`, or `partial`.
- One sentence why.

## Что изменено

- Key files and behavior.

## Runtime evidence

- API/tool calls.
- Browser/Playwright.
- DB evidence.
- Audit/metrics.
- Prometheus/Monit/deploy evidence when applicable.

## Проверки

- Commands run and result.
- Explicitly say that tests are quality gates, not acceptance.

## Contract impact

- API/DTO.
- Persistence.
- Config/env.
- Browser.
- Ops/runtime.
- Rollback.

## Direct-main delivery

- Commit SHA.
- Push status.
- CI/deploy status.
- Post-deploy evidence.

## Residual risk

- Missing credentials, blocked runtime, or follow-up work.
