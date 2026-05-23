---
prompt_name: identity_exchange_connections_v1_05_binance_bybit_validation_no_orders
repo: roehub.com
branch: main
scope: "Stage 5: implement Binance/Bybit credential validation adapters without order placement."

language:
  implementation: python_fastapi_http_tests_docs
  agent_report: ru

context_sources:
  always_read:
    - path: .codex/AGENTS.md
      why: "repo contract and secret safety"
    - path: docs/architecture/identity/identity-exchange-connections-live-trading-v1.md
      why: "Stage 5 source of truth"
    - path: docs/architecture/identity/exchange-connections-stage-reports/identity-exchange-connections-live-trading-v1-iteration-ledger.md
      why: "shared stage execution ledger and direct-main delivery handoff facts"
    - path: docs/architecture/identity/exchange-connections-stage-reports/04-connections-credential-versions-backfill.md
      why: "accepted Stage 4 evidence"
  task_entrypoints:
    - path: src/trading/contexts/exchange_control
      why: "connection model, secret resolver, metrics"
      inspect_symbols:
        - validation use cases
        - credential resolver
        - metrics
    - path: apps/api/routes/ui_account.py
      why: "validate endpoint surface"
      inspect_symbols:
        - exchange connection validate route
        - response mapping
    - path: tests/unit/contexts/exchange_control
      why: "adapter and validation unit coverage"
      inspect_symbols:
        - validation tests
        - sanitized error tests
    - path: tests/unit/apps/api/test_ui_account_routes.py
      why: "API contract regression tests"
      inspect_symbols:
        - exchange connection route tests
  conditional_bundles:
    exchange_docs:
      read_when: "implementing adapter request/response mapping"
      paths:
        - docs/architecture/identity/identity-exchange-connections-live-trading-v1.md
    web_status:
      read_when: "validation read model affects `/settings` payload"
      paths:
        - apps/web/templates/fragments/account/exchange_keys.html
        - apps/web/dist/js/pages/settings.js
        - tests/unit/apps/web/test_app_routes.py
    ops_metrics:
      read_when: "validation metrics or alerts are changed"
      paths:
        - infra/macos/prometheus/prometheus.prod.yml
        - docs/runbooks/mac-studio-monitoring-plan.md
  consult_if_needed:
    - path: docs/runbooks/exchange-secret-management.md
      read_when: "credential decrypt/Transit use is unclear"

style_references: []

documentation_continuity:
  old_current_docs:
    - "docs/architecture/identity/identity-exchange-connections-live-trading-v1.md"
    - "docs/runbooks/exchange-secret-management.md"
  new_doc_artifact: "docs/architecture/identity/exchange-connections-stage-reports/05-binance-bybit-validation.md"
  canonical_shape: "stage report with Markdown evidence tables: exchange, scenario, env vars, expected status, observed status, blocker"
  docs_gate: "python -m tools.docs.generate_docs_index --check"

stage_execution_ledger:
  path: "docs/architecture/identity/exchange-connections-stage-reports/identity-exchange-connections-live-trading-v1-iteration-ledger.md"
  plan_doc: "docs/architecture/identity/identity-exchange-connections-live-trading-v1.md"
  current_stage: "05"
  update_required: true
  update_timing: "after validation, before direct-main push and final report"
  direct_main_delivery_required: true

hard_requirements:
  iteration_ledger_update_required: true
  previous_stage_must_be_accepted: true
  stage2_process_required: true
  stage3_transit_acl_required: true
  stage4_connection_model_required: true
  no_order_placement_required: true
  validation_feature_flag_required: true
  external_validation_skip_policy_required: true
  readonly_live_validation_required_for_acceptance: true
  no_secret_leak_required: true
  stage_execution_ledger_update_required: true
  direct_main_push_after_validation_required: true
  feature_branch_per_stage_forbidden: true
  draft_pr_forbidden: true
  work_on_main_from_start_required: true

task_toggles:
  implement_validation_adapters: true
  implement_sanitized_errors: true
  implement_validation_metrics: true
  implement_live_validation_optional_gate: true
  publish_after_success: true
  direct_main_push_after_validation: true
  target_branch: main
  draft_pr_after_success: false

skill_routing:
  - skill: publish-ci-deploy
    use_when: "stage implementation, validation, stage report, and ledger update are complete"
    timing: "after validation and before final report"
    reason: "user requires direct push to main after accepted validation, with CI/deploy follow-through"
  - skill: contract-impact-analysis
    use_when: "adding validation statuses, errors, DTO fields, config flags, or metrics"
    timing: "before implementation and final report"
    reason: "validation changes public read model and operational contracts"
  - skill: backend-quality-gates
    use_when: "running adapter/API/web tests, ruff, pyright"
    timing: "during verification"
    reason: "backend validation requires deterministic gates"
  - skill: root-cause-debugging
    use_when: "live exchange validation fails unexpectedly"
    timing: "if blocker"
    reason: "classify external, config, credential, or implementation failures"


target_envs:
  - local-dev
  - optional-external-testnet

required_literals:
  - "ROEHUB_EXCHANGE_VALIDATION_LIVE"
  - "ROEHUB_TEST_BINANCE_READONLY_API_KEY"
  - "ROEHUB_TEST_BINANCE_READONLY_API_SECRET"
  - "ROEHUB_TEST_BYBIT_READONLY_API_KEY"
  - "ROEHUB_TEST_BYBIT_READONLY_API_SECRET"
  - "valid_readonly"
  - "valid_trade_enabled"
  - "invalid_credentials"
  - "invalid_permissions"
  - "invalid_ip_restriction"
  - "unsupported_account_mode"
  - "skipped_external_validation"

non_goals:
  - "Do not place, cancel, amend, or reconcile orders."
  - "Do not add private user stream order/fill handling."
  - "Do not implement strategy signal-to-execution transport."
  - "Do not require trade-enabled keys in default CI."
  - "Do not use CCXT for production validation."

final_report_format:
  language: ru
  sections:
    - "Что реализовано"
    - "Validation contract"
    - "External validation"
    - "Проверки"
    - "Stage 6 readiness"
    - "Direct-main delivery"

quality_gates:
  - cmd: "uv run pytest -q tests/unit/contexts/exchange_control tests/unit/apps/api/test_ui_account_routes.py tests/unit/apps/web/test_app_routes.py"
    expect: "passes; live validation skipped unless env flag is set"
  - cmd: "uv run ruff check src/trading/contexts/exchange_control apps/api apps/web tests/unit/contexts/exchange_control tests/unit/apps/api tests/unit/apps/web"
    expect: "passes"
  - cmd: "uv run pyright src/trading/contexts/exchange_control apps/api tests/unit/contexts/exchange_control tests/unit/apps/api"
    expect: "passes"
  - cmd: "! rg -n \"/order|createOrder|submit_order|place_order\" src/trading/contexts/exchange_control"
    expect: "no order placement implementation in exchange_control"
  - cmd: "python -m tools.docs.generate_docs_index --check"
    expect: "passes after Markdown changes"
  - cmd: "curl -fsS -X POST \"$ROEHUB_BASE_URL/api/ui/account/exchange-connections/$CONNECTION_ID/validate\" -H \"Origin: $ROEHUB_BASE_URL\" -H \"Cookie: $ROEHUB_SESSION_COOKIE\" -H \"X-CSRF-Token: $ROEHUB_CSRF_TOKEN\""
    expect: "validates readonly Binance and Bybit env-backed connections when ROEHUB_EXCHANGE_VALIDATION_LIVE=1; otherwise Stage 5 is blocked for production acceptance"
  - cmd: 'test "$(git branch --show-current)" = main'
    expect: "passes before direct-main push; otherwise stop and do not create a stage branch"
  - cmd: "gh --version && gh auth status"
    expect: "GitHub CLI is installed/authenticated for CI/deploy inspection after pushing main"

  - cmd: "gh --version && gh auth status"
    expect: "GitHub CLI is installed/authenticated for CI/deploy inspection after pushing main"

expected_primary_touches:
  - "docs/architecture/identity/exchange-connections-stage-reports/identity-exchange-connections-live-trading-v1-iteration-ledger.md"
  - "src/trading/contexts/exchange_control/**"
  - "apps/api/routes/ui_account.py"
  - "tests/unit/contexts/exchange_control/**"
  - "tests/unit/apps/api/test_ui_account_routes.py"
  - "docs/architecture/identity/exchange-connections-stage-reports/05-binance-bybit-validation.md"

possible_secondary_touches:
  - "apps/web/templates/fragments/account/exchange_keys.html"
  - "apps/web/dist/js/pages/settings.js"
  - "tests/unit/apps/web/test_app_routes.py"
  - "infra/macos/prometheus/prometheus.prod.yml"
  - "docs/architecture/README.md"

safety_notes:
  - "Live validation must be opt-in and must report skipped state honestly."
  - "Sanitize raw exchange errors before logging, API responses, audit, and metrics."
---

# Task

Implement Binance and Bybit credential validation without any trading action.

Done means:

- Binance and Bybit validation adapters verify credential permissions and metadata;
- validation status is persisted and visible through account API;
- metrics and audit events reflect validation results;
- live validation is opt-in through env vars and skip policy;
- no order path exists in `exchange_control`.

## Context / Current State

Stage 5 has hard prerequisites: Stage 2 `exchange-control`, Stage 3 Transit ACL, and Stage 4 connection/credential model. If any report is missing or blocked, stop.

The architecture rejects CCXT for production validation and requires native exchange endpoints. Use direct HTTP/native SDK only for metadata/permission validation, not order placement. Before implementing adapter mapping, verify current official Binance/Bybit documentation or native SDK docs; do not rely on stale memory for endpoint fields.

## Requirements (Must)

- Before making changes, verify the current branch is `main` and `git pull --ff-only origin main` succeeds; if not, stop and mark the stage blocked instead of creating a side branch.
- Update the shared stage execution ledger after validation and before delivery; include stage status, evidence, blockers, compatibility/rollback notes, CI/deploy status, and facts next stages must know.
- After all required validation passes, deliver directly to `main`: stay/switch to `main`, run `git pull --ff-only origin main`, stage only scoped files, commit on `main`, push `origin main`, and follow CI/deploy status. Do not create a per-stage branch or draft PR.
- Add Binance validation for API restrictions/permissions.
- Add Bybit validation for API key information.
- Normalize statuses: `valid_readonly`, `valid_trade_enabled`, `invalid_credentials`, `invalid_permissions`, `invalid_ip_restriction`, `unsupported_account_mode`.
- Add sanitized error mapping.
- Add validation metrics and audit events.
- Implement env contract and skip policy:
  - `ROEHUB_EXCHANGE_VALIDATION_LIVE=1` enables live validation.
  - readonly Binance and Bybit env vars are required for production acceptance.
  - trade-enabled validation is optional/manual and never places orders.
- If readonly Binance and Bybit env-backed validation cannot run, Stage 5 must finish as blocked for production acceptance; `skipped_external_validation` is valid evidence only for local CI, not acceptance of this stage.
- Create Stage 5 report.

## Requirements (Should)

- Use deterministic fake exchange clients in unit tests.
- Ensure validation adapters are behind explicit config flags.

## Requirements (Nice-to-have)

- Include rate-limit header normalization if the exchange response exposes it in tests.

# Context acquisition protocol

Read only in this order and stop once sufficient:

1. `.codex/AGENTS.md`
2. Stage 4 report and prerequisite reports if needed
3. architecture document Stage 5
4. task entrypoints
5. conditional bundles only for UI/metrics surfaces touched

Pre-implementation reading target:

- `<= 8 files`
- `<= ~50k tokens`

Stop reading once adapter interfaces, status mapping, API endpoint, tests, and skip policy are bounded.

# Reading manifest

Use front-matter `context_sources` as the canonical reading map. Do not preload strategy or execution modules.

# Work plan (agent should follow)

0. Verify the local checkout is on `main`, run `git pull --ff-only origin main`, and confirm there are no unrelated changes in scope. Stop if this cannot be proven.
Skill routing for this task:

- `contract-impact-analysis`: use for status/error/config/metrics/API contracts.
- `backend-quality-gates`: use during verification.
- `root-cause-debugging`: use only if live external validation fails unexpectedly.

1. Confirm prerequisite stages are accepted.
2. Implement fake-client tests and adapter interfaces first.
3. Implement Binance/Bybit validation adapters without order methods.
4. Wire validate endpoint/status/audit/metrics.
5. Run gates, required readonly external validation when production acceptance is requested, secret grep, and Stage 5 report.

After stage-specific verification:

- update `docs/architecture/identity/exchange-connections-stage-reports/identity-exchange-connections-live-trading-v1-iteration-ledger.md` with accepted/blocked status, evidence, changed contracts, blockers, next-stage facts, and direct-main delivery status;
- perform direct-main delivery only after successful validation: confirm the current branch is `main`, fast-forward from `origin/main`, stage only scoped files, commit, push `origin main`, and watch CI/deploy status;
- if `main` cannot fast-forward, GitHub auth is unavailable, local gates fail, or unrelated worktree changes cannot be isolated, stop and mark the stage blocked in the ledger; do not create a stage branch or draft PR as a workaround.

# Acceptance criteria (Definition of Done)

- Iteration ledger is updated with facts required by the next stage.
- Invalid key returns `invalid_credentials`.
- Readonly key returns `valid_readonly`.
- Trade-enabled key returns `valid_trade_enabled` only as informational.
- Withdrawal/transfer permission returns `invalid_permissions` or policy disable.
- Missing IP restriction for mainnet returns configured warning or `invalid_ip_restriction`.
- Unsupported account mode returns `unsupported_account_mode`.
- Raw exchange error body is not exposed.
- `rg` finds no order placement implementation in `exchange_control`.
- Stage report cites the official docs/native SDK source used for Binance and Bybit validation mapping.
- Shared ledger `docs/architecture/identity/exchange-connections-stage-reports/identity-exchange-connections-live-trading-v1-iteration-ledger.md` is updated with stage status, evidence, blockers, next-stage facts, and direct-main delivery status.
- Direct-main push to `origin/main` is completed after validation and CI/deploy status is recorded, or the stage is blocked with the exact reason.
- No per-stage branch and no draft PR are created for this stage.

# Implementation constraints

## Determinism & ordering

- Unit tests must not depend on live exchange availability.
- Live validation must be skipped in default CI unless explicitly enabled.
- Stage acceptance requires readonly Binance and Bybit env-backed validation; otherwise report `blocked_external_validation_credentials_missing`.

## API / contracts

- Validation statuses are public read-model contracts.
- Do not add order submit/cancel routes or adapter methods.

## Documentation

- Update the shared stage execution ledger before direct-main delivery; it is the canonical cross-stage handoff document.
- Record direct-main delivery evidence in the ledger: commit SHA, `git push origin main` result, CI/deploy status, runtime status when applicable, or exact blocker.
- Create Stage 5 report.
- Update architecture only if implementation deviates.
- Review old/current docs listed in `documentation_continuity.old_current_docs`; if they describe stale behavior as current, update them in the same change, otherwise state that no stale text was found.
- Use Markdown tables for invalid, readonly, optional trade-enabled, IP restriction, unsupported account mode, exchange error redaction, metrics, audit, and no-order evidence.
- Run docs-index check after Markdown changes.

## Tests

- Cover fake-client scenarios for all required statuses and sanitized errors.
- Cover API endpoint response and audit/metrics where feasible.

# Files to indicate (expected touched areas)

Primary touches:

- `docs/architecture/identity/exchange-connections-stage-reports/identity-exchange-connections-live-trading-v1-iteration-ledger.md`
- `src/trading/contexts/exchange_control/**`
- `apps/api/routes/ui_account.py`
- `tests/unit/contexts/exchange_control/**`
- `tests/unit/apps/api/test_ui_account_routes.py`
- `docs/architecture/identity/exchange-connections-stage-reports/05-binance-bybit-validation.md`

Possible secondary touches:

- `apps/web/templates/fragments/account/exchange_keys.html`
- `apps/web/dist/js/pages/settings.js`
- `tests/unit/apps/web/test_app_routes.py`
- `infra/macos/prometheus/prometheus.prod.yml`
- `docs/architecture/README.md`

# Non-goals

- Trading execution.
- Live trading canary.
- User streams for orders/fills.
- Portfolio accounting.
- Replacing strategy signal flow.

# Quality gates (must run and pass)

- `test "$(git branch --show-current)" = main`
- `gh --version && gh auth status`
- `uv run pytest -q tests/unit/contexts/exchange_control tests/unit/apps/api/test_ui_account_routes.py tests/unit/apps/web/test_app_routes.py`
- `uv run ruff check src/trading/contexts/exchange_control apps/api apps/web tests/unit/contexts/exchange_control tests/unit/apps/api tests/unit/apps/web`
- `uv run pyright src/trading/contexts/exchange_control apps/api tests/unit/contexts/exchange_control tests/unit/apps/api`
- `! rg -n "/order|createOrder|submit_order|place_order" src/trading/contexts/exchange_control`
- `python -m tools.docs.generate_docs_index --check`
- `curl -fsS -X POST "$ROEHUB_BASE_URL/api/ui/account/exchange-connections/$CONNECTION_ID/validate" -H "Origin: $ROEHUB_BASE_URL" -H "Cookie: $ROEHUB_SESSION_COOKIE" -H "X-CSRF-Token: $ROEHUB_CSRF_TOKEN"`
- `rg -n "$ROEHUB_TEST_BINANCE_READONLY_API_SECRET|$ROEHUB_TEST_BYBIT_READONLY_API_SECRET|TEST_PASSPHRASE|api_secret|passphrase" logs output .playwright-cli || true`

# Final output: report format (strict)

Your final message MUST be in Russian and follow exactly:

Your final message MUST include direct-main commit SHA, `git push origin main` status, CI/deploy status, and deploy/runtime status.

1. **Что реализовано**
2. **Validation contract**
3. **External validation**
4. **Проверки**
5. **Stage 6 readiness**
6. **Direct-main delivery**
