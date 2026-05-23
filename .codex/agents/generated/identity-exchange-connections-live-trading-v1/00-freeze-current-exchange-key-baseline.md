---
prompt_name: identity_exchange_connections_v1_00_freeze_current_exchange_key_baseline
repo: roehub.com
branch: main
scope: "Stage 0: freeze current `/api/exchange-keys` and `/settings` exchange-key baseline before implementation."

language:
  implementation: python_fastapi_jinja_js_sql_docs
  agent_report: ru

context_sources:
  always_read:
    - path: .codex/AGENTS.md
      why: "repo contract and gates"
    - path: docs/architecture/identity/identity-exchange-connections-live-trading-v1.md
      why: "source architecture and stage gates"
    - path: .codex/agents/.context/promt_manager_state.yaml
      why: "latest compact state if present"
  task_entrypoints:
    - path: src/trading/contexts/identity/adapters/inbound/api/routes/exchange_keys.py
      why: "current exchange-key API contract"
      inspect_symbols:
        - create_exchange_key
        - list_exchange_keys
        - delete_exchange_key
    - path: migrations/postgres/0003_identity_exchange_keys_v1.sql
      why: "current persisted market_type contract"
      inspect_symbols:
        - identity_exchange_keys
        - market_type
    - path: apps/api/routes/ui_account.py
      why: "current account facade and mutation guard"
      inspect_symbols:
        - _enforce_same_origin
        - account_summary
    - path: apps/web/templates/fragments/account/exchange_keys.html
      why: "current settings exchange-key UI"
      inspect_symbols:
        - exchange key form
        - exchange key table
  conditional_bundles:
    browser_surface:
      read_when: "if runtime `/settings` or browser artifact evidence is collected"
      paths:
        - apps/web/templates/pages/settings.html
        - apps/web/dist/js/pages/settings.js
        - tests/unit/apps/web/test_app_routes.py
    api_tests:
      read_when: "if focused API tests fail or need coverage updates"
      paths:
        - tests/unit/apps/api/test_identity_exchange_keys_routes.py
        - tests/unit/apps/api/test_ui_account_routes.py
    storage_policy:
      read_when: "if secret response/storage behavior is ambiguous"
      paths:
        - docs/architecture/identity/identity-exchange-keys-storage-2fa-gate-policy-v2.md
        - migrations/postgres/0004_identity_exchange_keys_v2.sql
  consult_if_needed:
    - path: docs/architecture/identity/identity-keycloak-auth-model-v1.md
      read_when: "auth/session behavior is unclear"

style_references: []

documentation_continuity:
  old_current_docs:
    - "docs/architecture/identity/identity-exchange-connections-live-trading-v1.md"
    - "docs/architecture/identity/identity-exchange-keys-storage-2fa-gate-policy-v2.md"
  new_doc_artifact: "docs/architecture/identity/exchange-connections-stage-reports/00-baseline-current-state.md"
  canonical_shape: "stage report with Markdown evidence tables: route matrix, storage evidence, UI evidence, secret grep, blockers"
  docs_gate: "python -m tools.docs.generate_docs_index --check"

hard_requirements:
  implementation_changes_allowed: false
  baseline_report_required: true
  no_secret_leak_required: true
  market_type_v1_must_remain_spot_futures: true
  do_not_design_execution: true

task_toggles:
  code_changes_allowed: false
  docs_patch_allowed_if_stage_table_or_index_drift_found: true
  runtime_calls_required_when_env_available: true
  publish_after_success: false

skill_routing:
  - skill: architecture-review
    use_when: "checking current docs/code/API drift without implementation"
    timing: "during investigation"
    reason: "baseline freeze and drift classification"
  - skill: contract-impact-analysis
    use_when: "classifying current API, DTO, schema, UI defaults"
    timing: "before final report"
    reason: "baseline is a compatibility contract"
  - skill: backend-quality-gates
    use_when: "running focused pytest, ruff, docs-index checks"
    timing: "during verification"
    reason: "baseline evidence must be reproducible"

target_envs:
  - local-dev

required_literals:
  - "/api/exchange-keys"
  - "/settings"
  - "identity_exchange_keys"
  - "spot"
  - "futures"
  - "api_secret"
  - "passphrase"
  - "ciphertext"
  - "hmac"

non_goals:
  - "Do not implement new exchange_connections tables."
  - "Do not change API behavior unless required only for docs-index drift."
  - "Do not add Binance/Bybit validation."
  - "Do not implement exchange-execution, order placement, risk engine, or order ledger."

final_report_format:
  language: ru
  sections:
    - "Что проверено"
    - "Текущий контракт"
    - "Секреты и утечки"
    - "Проверки"
    - "Блокеры к Stage 1"

quality_gates:
  - cmd: "uv run pytest -q tests/unit/apps/api/test_identity_exchange_keys_routes.py tests/unit/apps/api/test_ui_account_routes.py tests/unit/apps/web/test_app_routes.py"
    expect: "passes or exact pre-existing failure classification"
  - cmd: "python -m tools.docs.generate_docs_index --check"
    expect: "passes"
  - cmd: "rg -n \"spot|futures|linear|inverse\" migrations/postgres/0003_identity_exchange_keys_v1.sql src/trading/contexts/identity apps/api/routes apps/web/templates/fragments/account/exchange_keys.html apps/web/dist/js/pages/settings.js"
    expect: "confirms v1 contract and any drift is reported"

expected_primary_touches:
  - "docs/architecture/identity/exchange-connections-stage-reports/00-baseline-current-state.md"

possible_secondary_touches:
  - "docs/architecture/identity/identity-exchange-connections-live-trading-v1.md only if doc/index drift is found"
  - "docs/architecture/README.md if docs index changes"

safety_notes:
  - "This prompt is evidence-only unless a narrow documentation drift fix is required."
  - "Never print or persist real API secrets in reports, logs, screenshots, or artifacts."
---

# Task

Freeze the current exchange-key baseline for `/api/exchange-keys` and `/settings` before implementation of Exchange Control v1.

Done means:

- the current API/UI/storage contract is captured in a stage report;
- secret fields are proven absent from responses and artifacts;
- `market_type` v1 is confirmed as `spot|futures`;
- Stage 1 can start with a concrete baseline, or blockers are listed.

## Context / Current State

The source architecture document defines Exchange Control v1 as key storage, validation, rotation, audit, metrics, and operational control. Trading execution is explicitly out of scope.

Current expected facts to verify:

- `identity_exchange_keys` exists and stores Binance/Bybit key metadata plus encrypted blobs.
- `/api/exchange-keys` is the current compatibility surface.
- `/settings` already contains an exchange-key UI fragment.
- The current `market_type` contract is `spot|futures`, not `linear|inverse`.

## Requirements (Must)

- Keep this stage evidence-only unless a narrow documentation drift correction is required.
- Identify exact current request/response fields for `POST`, `GET`, and `DELETE /api/exchange-keys`.
- Confirm that responses do not include `api_secret`, `passphrase`, ciphertext, fingerprint, or HMAC.
- Confirm duplicate/delete behavior and current deterministic errors.
- Confirm current `market_type` validation from schema, API DTO, and UI.
- Create `docs/architecture/identity/exchange-connections-stage-reports/00-baseline-current-state.md`.
- If Markdown docs change, run the docs-index gate.

## Requirements (Should)

- Include a compact Markdown table in the stage report for route, behavior, current evidence, and gap.
- Include exact commands and outputs summarized enough for review.

## Requirements (Nice-to-have)

- Collect a local curl smoke if a local authenticated session is available.

# Context acquisition protocol

Read only in this order and stop once sufficient:

1. `.codex/AGENTS.md`
2. `.codex/agents/.context/promt_manager_state.yaml` if present
3. the architecture document and task entrypoints
4. conditional bundles only if a gate fails or behavior is ambiguous
5. consult-if-needed references only for blockers

Do not eagerly preload all listed files.

Pre-implementation reading target:

- `<= 8 files`
- `<= ~35k tokens`

Stop reading once current routes, schema, UI surface, secret behavior, and `market_type` contract are identified.

# Reading manifest

Use the front-matter `context_sources` as the canonical reading map. Do not duplicate it into a broad mandatory reading list.

# Work plan (agent should follow)

Skill routing for this task:

- `architecture-review`: use during investigation for docs/code drift and baseline evidence.
- `contract-impact-analysis`: use before final report to classify the frozen baseline.
- `backend-quality-gates`: use during verification for focused tests and docs-index checks.

1. Inspect the architecture document Stage 0 and the current API/schema/UI entrypoints.
2. Run the focused test and docs-index gates.
3. If a runtime session is available, run the GET acceptance call and secret grep.
4. Write the Stage 0 evidence report as a Markdown table-led document.
5. Report whether Stage 1 is unblocked.

# Acceptance criteria (Definition of Done)

- Stage report exists at `docs/architecture/identity/exchange-connections-stage-reports/00-baseline-current-state.md`.
- Report includes route matrix, storage evidence, UI evidence, secret grep, and `market_type` contract.
- No implementation behavior changed.
- Quality gates pass or failures are classified as introduced, pre-existing, environmental, or skipped with reason.

# Implementation constraints

## Determinism & ordering

- Keep evidence ordered by route, then storage, then UI.
- Do not use production secrets or real exchange credentials.

## API / contracts

- Do not change public or persisted contracts in this stage.
- If a drift is found, document it instead of silently fixing behavior.

## Documentation

- The stage report is the required new documentation artifact.
- Update the architecture document only if its current text contradicts verified code.
- Review old/current docs listed in `documentation_continuity.old_current_docs`; if they describe stale behavior as current, update them in the same change, otherwise state that no stale text was found.
- Use Markdown tables for the stage report evidence matrix; do not rely on screenshots or wide rendered tables as the primary readable artifact.
- Run `python -m tools.docs.generate_docs_index --check` after Markdown changes.

## Tests

- Use targeted tests only.
- Do not broaden to unrelated repository gates unless a local change requires it.

# Files to indicate (expected touched areas)

Primary touches:

- `docs/architecture/identity/exchange-connections-stage-reports/00-baseline-current-state.md`

Possible secondary touches:

- `docs/architecture/identity/identity-exchange-connections-live-trading-v1.md`
- `docs/architecture/README.md`

# Non-goals

- New storage model.
- Security hardening implementation.
- OpenBao/Vault setup.
- Binance/Bybit validation.
- Trading execution.

# Quality gates (must run and pass)

- `uv run pytest -q tests/unit/apps/api/test_identity_exchange_keys_routes.py tests/unit/apps/api/test_ui_account_routes.py tests/unit/apps/web/test_app_routes.py`
- `python -m tools.docs.generate_docs_index --check`
- `rg -n "TEST_SECRET|TEST_API_SECRET|TEST_PASSPHRASE" logs output .playwright-cli || true`

# Final output: report format (strict)

Your final message MUST be in Russian and follow exactly:

1. **Что проверено**
2. **Текущий контракт**
3. **Секреты и утечки**
4. **Проверки**
5. **Блокеры к Stage 1**
