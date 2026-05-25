---
prompt_name: identity_exchange_connections_v1_08_repair_settings_exchange_origin_schema_e2e
repo: roehub.com
branch: main
scope: "Stage 08 repair: make production /settings exchange-key add flow work through the public edge, fix account settings schema drift, and prove the authenticated browser flow with Playwright."

language:
  implementation: python_ops_browser_docs
  agent_report: ru

context_sources:
  always_read:
    - path: .codex/AGENTS.md
      why: "repo contract, direct-main delivery rules, browser verification rules"
    - path: docs/architecture/identity/identity-exchange-connections-live-trading-v1.md
      why: "source plan and scope boundaries for exchange connections v1"
    - path: docs/architecture/identity/exchange-connections-stage-reports/identity-exchange-connections-live-trading-v1-iteration-ledger.md
      why: "shared stage handoff ledger; Stage 08 must supersede the Stage 07 readiness gap"
    - path: docs/architecture/identity/exchange-connections-stage-reports/07-production-readiness.md
      why: "latest accepted readiness claim that must be corrected with production browser evidence"
  task_entrypoints:
    - path: src/trading/contexts/identity/adapters/inbound/api/csrf.py
      why: "same-origin decision logic for browser mutations"
      inspect_symbols:
        - same_origin_rejection_reason
        - _expected_origin_sources
        - _matches_expected_origin
    - path: apps/api/routes/ui_account.py
      why: "account settings API facade and exchange-connection mutation routes"
      inspect_symbols:
        - _enforce_same_origin_mutation
        - create_exchange_connection
        - list_integrations
        - get_profile
    - path: src/trading/contexts/identity/adapters/outbound/persistence/postgres/account_settings_repository.py
      why: "Postgres repository that produced production 500s for account settings"
      inspect_symbols:
        - list_integrations
        - get_profile
    - path: infra/caddy/Caddyfile.vps
      why: "repository source of truth for public edge same-origin proxy headers"
      inspect_symbols:
        - X-Forwarded-Host
        - X-Forwarded-Proto
        - /api/*
    - path: apps/web/templates/fragments/account/exchange_keys.html
      why: "settings exchange connection form and secret input autocomplete behavior"
      inspect_symbols:
        - api_key
        - api_secret
        - permissions
    - path: apps/web/dist/js/pages/settings.js
      why: "browser submit path for /api/ui/account/exchange-connections"
      inspect_symbols:
        - exchange-connections
        - connect key
        - rotate
        - disable
  conditional_bundles:
    migration_and_schema:
      read_when: "account settings API still returns 500 or schema drift is confirmed"
      paths:
        - migrations/postgres/0006_identity_account_settings_v1.sql
        - apps/migrations/bootstrap_main.py
        - tests/unit/apps/migrations
        - tests/unit/apps/api/test_ui_account_routes.py
    web_gateway_docs:
      read_when: "runtime edge config or same-origin proxy behavior is changed"
      paths:
        - docs/runbooks/web-ui-gateway-same-origin.md
        - docs/architecture/apps/gateway/nginx-gateway-same-origin-ui-api-v1.md
        - docs/runbooks/keycloak-local-setup-and-ops.md
    delivery_ops:
      read_when: "deploying or verifying production VPS/Mac Studio runtime"
      paths:
        - docs/runbooks/mac-studio-native-backend-operations.md
        - docs/runbooks/mac-studio-backend-operations.md
        - scripts/macos/smoke_prod.sh
        - infra/caddy/Caddyfile.vps
    browser_tests:
      read_when: "adding or updating browser-visible settings tests"
      paths:
        - tests/unit/apps/web/test_app_routes.py
        - docs/architecture/identity/exchange-connections-stage-reports/06-settings-ui.md
  consult_if_needed:
    - path: docs/architecture/identity/exchange-connections-stage-reports/06-settings-ui.md
      read_when: "Stage 6 browser evidence or UI contract is unclear"
    - path: docs/runbooks/exchange-secret-management.md
      read_when: "exchange-control, OpenBao, or secret-redaction boundaries become relevant"
    - path: infra/scripts
      read_when: "deployment validation should gain a stable scripted check instead of ad hoc commands"

style_references:
  - docs/architecture/identity/exchange-connections-stage-reports/07-production-readiness.md
  - docs/architecture/identity/exchange-connections-stage-reports/06-settings-ui.md

documentation_continuity:
  old_current_docs:
    - "docs/architecture/identity/identity-exchange-connections-live-trading-v1.md"
    - "docs/architecture/identity/exchange-connections-stage-reports/07-production-readiness.md"
    - "docs/runbooks/web-ui-gateway-same-origin.md"
    - "docs/runbooks/keycloak-local-setup-and-ops.md"
  new_doc_artifact: "docs/architecture/identity/exchange-connections-stage-reports/08-settings-production-repair.md"
  canonical_shape: "stage report with Markdown evidence tables: issue, root cause, fix, validation command, observed evidence, verdict, residual risk"
  docs_gate: "python -m tools.docs.generate_docs_index --check"

stage_execution_ledger:
  path: "docs/architecture/identity/exchange-connections-stage-reports/identity-exchange-connections-live-trading-v1-iteration-ledger.md"
  plan_doc: "docs/architecture/identity/identity-exchange-connections-live-trading-v1.md"
  current_stage: "08"
  update_required: true
  update_timing: "after local and production Playwright validation, before direct-main push and final report"
  template: ".codex/agents/stage_execution_ledger_template.md"
  direct_main_delivery_required: true

hard_requirements:
  root_cause_before_fix_required: true
  production_playwright_evidence_required: true
  authenticated_keycloak_browser_flow_required: true
  same_origin_must_remain_fail_closed: true
  schema_drift_must_be_fixed_or_explicitly_blocked: true
  password_manager_hardening_required: true
  no_real_exchange_credentials_required: true
  no_live_orders_required: true
  secret_artifact_grep_required: true
  documentation_continuity_required: true
  stage_execution_ledger_update_required: true
  direct_main_push_after_validation_required: true
  feature_branch_per_stage_forbidden: true
  draft_pr_forbidden: true

task_toggles:
  implementation_changes_allowed: true
  runtime_vps_edge_check_required: true
  runtime_mac_studio_backend_check_required: true
  add_or_update_tests_required: true
  run_playwright_required: true
  publish_after_success: true
  direct_main_push_after_validation: true
  target_branch: main
  draft_pr_after_success: false

skill_routing:
  - skill: root-cause-debugging
    use_when: "investigating the 403 csrf_origin_mismatch, account API 500s, or browser password-manager prompt"
    timing: "before implementation and whenever a fix hypothesis changes"
    reason: "this is a production regression/repair stage; reproduce and localize before editing"
  - skill: contract-impact-analysis
    use_when: "touching CSRF semantics, account API DTOs, migrations, Caddy config, env/config defaults, cookies, or browser-visible defaults"
    timing: "during design and before final report"
    reason: "same-origin, schema, and settings UI behavior are externally relied-upon contracts"
  - skill: backend-quality-gates
    use_when: "running focused pytest, ruff, pyright, and docs-index checks"
    timing: "during verification"
    reason: "backend/schema repairs need deterministic local evidence"
  - skill: browser-qa-evidence
    use_when: "verifying /settings exchange connection flow, console/network behavior, screenshots, and secret artifact safety"
    timing: "during verification"
    reason: "the reported defect is browser-visible and cannot be accepted by unit tests alone"
  - skill: playwright
    use_when: "performing the authenticated public https://roehub.com/settings browser proof"
    timing: "during verification after local fixes and deploy"
    reason: "the user explicitly requires working state to be proven in Playwright"
  - skill: publish-ci-deploy
    use_when: "local gates, docs, ledger, and production Playwright validation are complete"
    timing: "after validation and before final report"
    reason: "this staged rollout uses direct-main push, CI/deploy observation, and post-deploy verification"

target_envs:
  - local-dev
  - vps-edge
  - mac-studio
  - production-browser

required_literals:
  - "Mutation origin is not allowed"
  - "csrf_origin_mismatch"
  - "X-Forwarded-Host"
  - "X-Forwarded-Proto"
  - "https://roehub.com/settings"
  - "/api/ui/account/exchange-connections"
  - "/api/ui/account/profile"
  - "/api/ui/account/integrations"
  - "integration_key"
  - "autocomplete"
  - "playwright"
  - "output/playwright"

non_goals:
  - "Do not implement signal-to-execution, exchange-execution, order placement, order ledger, or strategy signal delivery."
  - "Do not use real exchange API credentials for this repair; use dummy credentials only."
  - "Do not weaken CSRF/same-origin fail-closed behavior to make the browser flow pass."
  - "Do not bypass SSH host-key safety or disable host-key checking blindly."
  - "Do not store smoke account credentials, cookies, API keys, API secrets, tokens, ciphertext, or raw provider responses in repo/docs/logs."
  - "Do not create a per-stage branch or draft PR."

final_report_format:
  language: ru
  sections:
    - "Вердикт"
    - "Root cause и исправления"
    - "Playwright evidence"
    - "Docs и ledger"
    - "Direct-main delivery"
    - "Residual risks"

quality_gates:
  - cmd: "uv run pytest -q tests/unit/apps/api/test_ui_account_routes.py tests/unit/apps/web/test_app_routes.py tests/unit/apps/migrations"
    expect: "passes; include any new focused tests for same-origin headers, schema repair, and autocomplete/password-manager hardening"
  - cmd: "uv run ruff check apps/api apps/web src/trading/contexts/identity tests/unit/apps/api tests/unit/apps/web tests/unit/apps/migrations"
    expect: "passes"
  - cmd: "uv run pyright apps/api src/trading/contexts/identity tests/unit/apps/api"
    expect: "passes"
  - cmd: "python -m tools.docs.generate_docs_index --check"
    expect: "passes after docs are updated"
  - cmd: "curl -fsS https://roehub.com/__edge_id"
    expect: "returns vps-edge"
  - cmd: "command -v npx >/dev/null 2>&1"
    expect: "passes before Playwright CLI use"
  - cmd: "rg -n \"TEST_SECRET|TEST_API_SECRET|TEST_PASSPHRASE|dummy-secret|dummy_api_secret\" logs output .playwright-cli || true"
    expect: "no real or test secret values are present in artifacts; if dummy literals are intentionally used, record why they are safe"
  - cmd: "test \"$(git branch --show-current)\" = main"
    expect: "passes before direct-main delivery; otherwise stop and do not create a stage branch"
  - cmd: "git pull --ff-only origin main"
    expect: "passes before final scoped staging and direct push"
  - cmd: "gh --version && gh auth status"
    expect: "GitHub CLI is installed/authenticated for CI/deploy inspection after pushing main"

expected_primary_touches:
  - "src/trading/contexts/identity/adapters/inbound/api/csrf.py"
  - "apps/api/routes/ui_account.py"
  - "src/trading/contexts/identity/adapters/outbound/persistence/postgres/account_settings_repository.py"
  - "migrations/postgres"
  - "infra/caddy/Caddyfile.vps"
  - "apps/web/templates/fragments/account/exchange_keys.html"
  - "apps/web/dist/js/pages/settings.js"
  - "tests/unit/apps/api/test_ui_account_routes.py"
  - "tests/unit/apps/web/test_app_routes.py"
  - "tests/unit/apps/migrations"
  - "docs/architecture/identity/exchange-connections-stage-reports/08-settings-production-repair.md"
  - "docs/architecture/identity/exchange-connections-stage-reports/identity-exchange-connections-live-trading-v1-iteration-ledger.md"

possible_secondary_touches:
  - "docs/architecture/identity/identity-exchange-connections-live-trading-v1.md"
  - "docs/runbooks/web-ui-gateway-same-origin.md"
  - "docs/runbooks/keycloak-local-setup-and-ops.md"
  - "docs/runbooks/mac-studio-native-backend-operations.md"
  - "docs/architecture/README.md"
  - "scripts/macos/smoke_prod.sh"
  - "infra/scripts"

safety_notes:
  - "The previous Stage 07 readiness claim is incomplete for the real browser add-key flow; this Stage 08 repairs and records that gap."
  - "The smoke Keycloak account may be provided by the user or a secure operator channel, but credentials must never be committed, documented, printed in final reports, or saved in Playwright artifacts."
  - "Use dummy exchange credentials only. If a dummy connection is created, disable or delete it through supported UI/API after proving the flow and record cleanup evidence."
  - "If VPS SSH reports a changed host key, stop until the expected host fingerprint is verified by a trusted source. Do not use StrictHostKeyChecking=no."
  - "Keep `read` as the safe permission default. `trade` must remain explicit opt-in."
---

# Task

Repair the production `/settings` exchange connection add-key flow and prove it in an authenticated Playwright browser session against `https://roehub.com/settings`.

Done means:

- submitting a dummy Bybit or Binance API key/secret from `/settings` no longer returns `Mutation origin is not allowed`;
- `/api/ui/account/profile` and `/api/ui/account/integrations` no longer return production 500s during the settings load;
- API key and API secret inputs no longer look like the site's login/password fields to browser/password-manager heuristics;
- same-origin/CSRF remains fail-closed for true cross-origin mutations;
- the fixed behavior is validated by targeted local tests and real Playwright evidence against production;
- all affected docs, Stage 08 report, and the shared iteration ledger are updated;
- after successful validation, the change is delivered directly to `main` and CI/deploy/post-deploy evidence is recorded.

## Context / Current State

Context ledger from the previous investigation:

- completed:
  - Production authenticated browser reproduction reached `https://roehub.com/settings` with the smoke Keycloak account.
  - Existing exchange connection row visible in the table: `exchange_name=binance`, label/account `binance_1`, masked API `****efsd`, status `disabled`, validation `skipped external validation`, permissions `trade`, market `futures`, environment `mainnet`, IP allowlist `unknown`.
  - A dummy Bybit `spot/mainnet/trade` create attempt posted to `https://roehub.com/api/ui/account/exchange-connections` and returned HTTP `403`.
  - Response body was `{"error":{"code":"forbidden","message":"Mutation origin is not allowed","details":{"reason":"csrf_origin_mismatch"}}}`.
  - Playwright request headers for the failed create showed `referer: https://roehub.com/settings`, `content-type: application/json`, and no visible `origin` header.
  - Console/network also showed HTTP `500` for `/api/ui/account/profile` and `/api/ui/account/integrations`.
  - Production log evidence for integrations showed `psycopg.errors.UndefinedColumn: column "integration_key" does not exist` in `account_settings_repository.py`.
  - Screenshot evidence exists at `output/playwright/settings-exchange-origin-repro-2026-05-24.png`.
- open_items:
  - Capture the exact `/api/ui/account/profile` production stack trace before fixing schema drift.
  - Verify actual VPS `/etc/caddy/Caddyfile`, not only the repository copy.
  - Fix the browser/password-manager prompt that treats exchange API key/secret as Roehub login/password credentials.
  - Prove the final working state in Playwright after deploy.
- contract_changes:
  - Public settings endpoints should remain compatible except for bug fixes and fail-closed security behavior.
  - Any schema repair must be additive/idempotent and safe on already-correct databases.
  - CSRF behavior must not be weakened; the browser flow should pass because public edge headers are correct, not because checks are skipped.
- touched_paths:
  - Expected touches are listed in front matter and must stay scoped.
- risks:
  - The public edge is `VPS Caddy`; `curl -fsS https://roehub.com/__edge_id` returned `vps-edge`.
  - Local repository `infra/caddy/Caddyfile.vps` contains `header_up X-Forwarded-Host {host}` and `header_up X-Forwarded-Proto {scheme}` for `/api/*`, but the deployed Mac Studio copy under `/opt/roehub/app/infra/caddy/Caddyfile.vps` was observed without those lines. The actual VPS Caddy runtime config still needs verification.
  - SSH to the VPS previously reported `REMOTE HOST IDENTIFICATION HAS CHANGED` with fingerprint `SHA256:MQPcAz0ewaAU5IvqU1AMJ1ba+NCjoF4gY7u9hgpP+lY`; treat this as a security-sensitive blocker until verified.
- next_focus:
  - Root-cause the public-edge forwarded-host/proto mismatch and production account-settings schema drift.
  - Add missing automated coverage so local tests would catch this class of defect.
  - Deploy and prove the real browser flow with Playwright, including network request/response evidence.

Additional context:

- A local CSRF helper check showed:
  - `referer_only_internal_host csrf_origin_mismatch`
  - `referer_with_forwarded_host None`
  - `origin_with_forwarded_host None`
- This means a Referer-only browser mutation is allowed when the backend receives the public forwarded host/proto context, and rejected when it only sees the internal upstream host.
- Stage 06 accepted the browser UI locally, but Stage 08 must prove the public production flow.
- Stage 07 accepted production readiness, but did not prove this exact authenticated create-key browser flow.

## Requirements (Must)

- Read context using the protocol below and stop early once sufficient.
- Implement only this repair stage.
- Keep `exchange-control` and secret custody boundaries intact; `apps/api` must not gain direct decrypt/native exchange SDK paths.
- Preserve CSRF fail-closed behavior:
  - true cross-origin mutation remains rejected with `csrf_origin_mismatch`;
  - missing browser origin can pass only when same-origin Referer and trusted forwarded public host/proto prove same-origin;
  - do not set `fail_closed_without_origin=False` in new sensitive paths unless there is an explicit tested browser rationale.
- Verify the real runtime edge:
  - confirm `https://roehub.com/__edge_id` is `vps-edge`;
  - inspect the actual active VPS Caddy config for `/api/*`;
  - ensure `X-Forwarded-Host` and `X-Forwarded-Proto` are forwarded to the API backend;
  - validate and reload Caddy safely if config changes are needed.
- Handle the VPS SSH host-key warning safely:
  - verify the expected host fingerprint through a trusted source before modifying `known_hosts`;
  - do not bypass host-key checks;
  - if fingerprint cannot be verified, stop and record a blocker.
- Fix production account-settings schema drift:
  - capture the `/api/ui/account/profile` stack trace before editing;
  - inspect production DB schema for account profile, preferences, integrations, notifications, sessions, audit, exchange connection tables touched by `/settings`;
  - add idempotent migration/repair coverage for missing `integration_key` and any confirmed profile/integrations columns;
  - ensure bootstrap/deploy applies the repair.
- Fix browser/password-manager heuristics for exchange API credentials:
  - API key/API secret fields must use stable names/ids/labels that do not mimic site login credentials;
  - add appropriate `autocomplete` and password-manager ignore attributes where supported by existing UI conventions;
  - clear secret inputs after success and failure;
  - do not add visible instructional copy just to explain browser behavior.
- Keep UI permission behavior safe:
  - default permission remains `read`;
  - `trade` remains explicit opt-in and must not be hardcoded in the form or request payload.
- Use dummy exchange credentials only.
- If the browser flow creates a dummy connection, disable or delete it through supported UI/API before finishing and record cleanup evidence without printing secret values.
- Create `docs/architecture/identity/exchange-connections-stage-reports/08-settings-production-repair.md` using the same evidence-table style as Stage 06/07.
- Update:
  - `docs/architecture/identity/identity-exchange-connections-live-trading-v1.md` with Stage 08 repair/gate status or plan adjustment;
  - `docs/architecture/identity/exchange-connections-stage-reports/identity-exchange-connections-live-trading-v1-iteration-ledger.md` with Stage 08 status, evidence, blocker/resolution, and what future stages must know;
  - `docs/runbooks/web-ui-gateway-same-origin.md` with the production Caddy forwarded-header requirement and verification commands;
  - Keycloak/browser E2E runbook docs if smoke-account Playwright acceptance procedure needs to be documented.
- Run docs index check after Markdown docs change.
- After local + production validation, deliver directly to `main`: `git pull --ff-only origin main`, scoped staging, commit, `git push origin main`, then inspect CI/deploy and repeat production smoke/Playwright where required by the deployment path.

## Requirements (Should)

- Add a small deploy/runtime drift check that would catch missing Caddy `X-Forwarded-Host`/`X-Forwarded-Proto` before a future deploy is marked healthy.
- Add tests for the exact Referer-only browser mutation case behind a forwarded public host.
- Add tests asserting the account exchange form secret fields have non-login autocomplete/password-manager-hardening attributes.
- Keep production repair SQL/migrations reversible by normal forward migration practice; avoid hand-edited database state that is not captured in repo.
- Keep Playwright artifacts under `output/playwright/` with timestamped names and sanitized data.

## Requirements (Nice-to-have)

- Add a concise troubleshooting table to the Stage 08 report mapping symptoms to checks:
  - `Mutation origin is not allowed`;
  - `/api/ui/account/profile` 500;
  - `/api/ui/account/integrations` 500;
  - browser save-password prompt after exchange key submit.
- If practical, add a one-command smoke script for the non-secret parts of `/settings` production readiness.

# Context acquisition protocol

Read only in this order and stop once sufficient:

1. `.codex/AGENTS.md`
2. The Stage 08 prompt front matter and this context section
3. The shared iteration ledger
4. Stage 07 report, then Stage 06 report only if UI/browser contract is unclear
5. Task entrypoints
6. Conditional bundles required by confirmed touched contracts or failing checks
7. Consult-if-needed references only for blockers, ambiguity, or conflicts

Do not eagerly preload all listed sources.

Pre-implementation reading target:

- `<= 10 files`
- `<= ~45k-60k tokens`

Stop reading once all of the following are true:

- root-cause hypotheses are testable,
- touched files are bounded,
- public API/schema/runtime contracts are identified,
- Playwright acceptance path is clear,
- no unresolved safety ambiguity remains around VPS SSH host identity.

Expand context only for:

- blockers,
- failing quality gates,
- unknown schema drift,
- runtime deploy uncertainty,
- Caddy config conflicts,
- architecture/security contract conflicts that affect correctness.

# Reading manifest

Use the front-matter `context_sources` as the canonical reading map.

Read with this intent:

- `always_read`:
  - repository rules,
  - source plan,
  - stage ledger,
  - latest readiness claim that Stage 08 supersedes
- `task_entrypoints`:
  - CSRF logic,
  - account settings routes,
  - account settings persistence,
  - public-edge Caddy config,
  - settings exchange form and submit code
- `conditional_bundles`:
  - read only when the stated condition applies
- `consult_if_needed`:
  - read only for blockers, ambiguity, or conflict resolution

Do not convert this manifest into a broad mandatory reading list.

# Work plan (agent should follow)

Skill routing for this task:

- `root-cause-debugging`: use before implementation and whenever evidence contradicts a hypothesis; owns defect reproduction/localization.
- `contract-impact-analysis`: use when touching CSRF, schema, runtime config, UI defaults, or public account APIs; owns compatibility classification.
- `backend-quality-gates`: use for focused pytest/ruff/pyright/docs gates.
- `browser-qa-evidence` + `playwright`: use for authenticated production `/settings` proof; owns screenshots, network/console evidence, and artifact hygiene.
- `publish-ci-deploy`: use after validation; owns direct-main delivery, CI/deploy observation, and post-deploy runtime verification.

1. Confirm repository state:
   - current branch is `main`;
   - worktree state is understood;
   - previous uncommitted unrelated changes are not reverted.
2. Reproduce/collect current evidence:
   - re-run or confirm the production browser defect with Playwright if still present;
   - capture current `/api/ui/account/profile` and `/api/ui/account/integrations` errors from logs;
   - confirm public edge identity and runtime Caddy config, handling SSH host-key safety correctly.
3. Localize root causes:
   - prove whether the 403 is caused by missing forwarded public host/proto at the VPS edge, backend CSRF logic, or another header/path issue;
   - prove which database columns/migrations are missing for account settings;
   - inspect why browser/password-manager heuristics see exchange key fields as login/password fields.
4. Implement scoped repairs:
   - edge config/deploy drift prevention for forwarded headers;
   - idempotent account-settings schema repair/migration and tests;
   - UI input hardening and tests;
   - any minimal account API fixes required by the captured stack traces.
5. Run focused local gates.
6. Deploy/deliver according to the direct-main stage contract only after local gates pass:
   - update docs and ledger first;
   - commit/push directly to `main`;
   - inspect CI/deploy;
   - run post-deploy smoke and Playwright proof.
7. Final Playwright acceptance:
   - log in through Keycloak with a smoke account obtained from a secure channel;
   - open `https://roehub.com/settings`;
   - verify profile, integrations, limits, exchange connections calls do not return 500;
   - verify permission default is `read` on a fresh form;
   - submit a dummy Bybit or Binance connection;
   - assert the create request is not rejected with `Mutation origin is not allowed` or `csrf_origin_mismatch`;
   - capture create request headers and response body/status;
   - verify no browser console errors except explicitly expected non-fatal entries;
   - verify secret fields clear after submit;
   - disable/delete any dummy connection created;
   - capture a final screenshot and request summary under `output/playwright/`.
8. Update Stage 08 report and iteration ledger with:
   - root cause;
   - changed files/contracts;
   - local gates;
   - Caddy/runtime evidence;
   - Playwright evidence paths;
   - dummy connection cleanup evidence;
   - CI/deploy evidence;
   - residual risks and what future stages must know.

# Acceptance criteria (Definition of Done)

- Local tests include the exact Referer-only + forwarded public host/proto scenario and true cross-origin rejection.
- Local tests or template assertions cover exchange API key/secret field autocomplete/password-manager hardening.
- Account settings migration/schema tests prove `identity_user_integrations.integration_key` and any other confirmed missing settings columns are present or repaired idempotently.
- Production public edge forwards enough trusted host/proto context for `/settings` same-origin mutations to pass without weakening backend CSRF checks.
- `/api/ui/account/profile` and `/api/ui/account/integrations` return successful responses during authenticated production settings load.
- Playwright evidence proves the real authenticated add-key flow no longer returns:
  - `Mutation origin is not allowed`;
  - `csrf_origin_mismatch`;
  - account settings 500s.
- Playwright evidence captures request/response/network/console summary and a final screenshot.
- Secret artifact grep is clean; no smoke password, cookies, API key/secret, tokens, ciphertext, or raw provider responses are committed or reported.
- Any dummy exchange connection created during proof is disabled/deleted before final report.
- Documentation is updated:
  - Stage 08 report created;
  - source plan and iteration ledger updated;
  - same-origin runbook updated;
  - Keycloak/browser E2E runbook updated if needed.
- Docs index check passes.
- Direct-main delivery evidence is recorded after successful validation.

# Implementation constraints

## Determinism & ordering

- Keep changes scoped and reviewable.
- Do not reorder stage history except to add Stage 08 repair/supersession notes.
- Prefer idempotent migrations/repair SQL over manual database edits.
- Preserve stable user IDs, connection IDs, credential version semantics, API response field names, and audit event naming unless a test-covered compatibility decision requires an additive change.

## API / contracts

- Same-origin mutation behavior may be fixed, but not weakened.
- Public browser path remains `/api/ui/account/exchange-connections`.
- Legacy exchange key and exchange-control boundaries remain unchanged unless the captured root cause proves a directly related defect.
- If API/DTO/persistence/config behavior changes, classify each dimension as `none`, `compatible-change`, `breaking-change`, or `unknown` in the Stage 08 report.

## Security

- Never print or persist smoke credentials, session cookies, exchange API keys, API secrets, provider response bodies, Transit tokens, ciphertext, or HMACs.
- Do not add real Binance/Bybit SDK calls or external validation calls in this repair unless already part of existing validate flow and explicitly needed for cleanup.
- Do not leave dummy trade-capable credentials active.
- Do not disable CORS/CSRF protections globally.
- Do not bypass SSH host-key verification.

## Documentation

- Update only directly relevant docs.
- The new Stage 08 report must use the evidence-table style of Stage 06/07.
- The iteration ledger must record Stage 08 status, direct-main delivery, production Playwright evidence, and future-stage handoff facts.
- The source plan must clearly state that Stage 07 readiness was followed by a Stage 08 production-browser repair gate.
- `docs/runbooks/web-ui-gateway-same-origin.md` must include production Caddy forwarded-header requirements and verification commands.
- Do not write secrets, raw cookies, raw exchange errors, or raw provider responses into docs.
- Run the docs-index check after Markdown changes.

## Tests

- Add/update deterministic focused tests for the changed behavior.
- Do not rely only on unit tests; final acceptance requires Playwright against production after deploy.
- If production access is blocked by VPS host-key verification, record the blocker and do not claim accepted.

## Playwright

- Use the repository/global `playwright` skill workflow.
- Use a smoke Keycloak account obtained from the user or secure operator channel; do not write credentials into files, docs, command history snippets, or final report.
- Prefer the Playwright CLI wrapper when available:

```bash
export CODEX_HOME="${CODEX_HOME:-$HOME/.codex}"
PWCLI="$CODEX_HOME/skills/playwright/playwright-cli.js"
test -f "$PWCLI"
npx playwright --version
```

- Required Playwright evidence:
  - successful login/callback to `https://roehub.com/settings`;
  - network request status/body for `/api/ui/account/profile`;
  - network request status/body for `/api/ui/account/integrations`;
  - network request headers/status/body for `POST /api/ui/account/exchange-connections`;
  - screenshot before/after add flow;
  - console summary;
  - cleanup/disable/delete evidence for any dummy connection.

# Files to indicate (expected touched areas)

Primary touches:

- `src/trading/contexts/identity/adapters/inbound/api/csrf.py`
- `apps/api/routes/ui_account.py`
- `src/trading/contexts/identity/adapters/outbound/persistence/postgres/account_settings_repository.py`
- `migrations/postgres`
- `infra/caddy/Caddyfile.vps`
- `apps/web/templates/fragments/account/exchange_keys.html`
- `apps/web/dist/js/pages/settings.js`
- `tests/unit/apps/api/test_ui_account_routes.py`
- `tests/unit/apps/web/test_app_routes.py`
- `tests/unit/apps/migrations`
- `docs/architecture/identity/exchange-connections-stage-reports/08-settings-production-repair.md`
- `docs/architecture/identity/exchange-connections-stage-reports/identity-exchange-connections-live-trading-v1-iteration-ledger.md`

Possible secondary touches:

- `docs/architecture/identity/identity-exchange-connections-live-trading-v1.md`
- `docs/runbooks/web-ui-gateway-same-origin.md`
- `docs/runbooks/keycloak-local-setup-and-ops.md`
- `docs/runbooks/mac-studio-native-backend-operations.md`
- `docs/architecture/README.md`
- `scripts/macos/smoke_prod.sh`
- `infra/scripts`

# Non-goals

- Do not implement live trading execution.
- Do not place orders.
- Do not add signal ingestion or strategy-to-order routing.
- Do not change native Binance/Bybit validation behavior unless directly required for dummy connection cleanup.
- Do not replace OpenBao/Vault, exchange-control, Keycloak, or the account settings architecture.
- Do not add CCXT.
- Do not perform a broad UI redesign.
- Do not create a stage branch or draft PR.

# Quality gates (must run and pass)

- `uv run pytest -q tests/unit/apps/api/test_ui_account_routes.py tests/unit/apps/web/test_app_routes.py tests/unit/apps/migrations`
- `uv run ruff check apps/api apps/web src/trading/contexts/identity tests/unit/apps/api tests/unit/apps/web tests/unit/apps/migrations`
- `uv run pyright apps/api src/trading/contexts/identity tests/unit/apps/api`
- `python -m tools.docs.generate_docs_index --check`
- `curl -fsS https://roehub.com/__edge_id`
- `command -v npx >/dev/null 2>&1`
- Playwright authenticated production flow for `https://roehub.com/settings`, including screenshots, network request details, console summary, and dummy connection cleanup.
- `rg -n "TEST_SECRET|TEST_API_SECRET|TEST_PASSPHRASE|dummy-secret|dummy_api_secret" logs output .playwright-cli || true`
- `test "$(git branch --show-current)" = main`
- `git pull --ff-only origin main`
- `gh --version && gh auth status`
- After push: inspect GitHub CI/deploy runs, then repeat the production browser acceptance if deploy changed runtime code/config.

If a gate cannot run, classify it as:

- introduced failure,
- required-path pre-existing failure,
- unrelated pre-existing failure,
- environmental blocker,
- flaky/inconclusive.

Do not mark Stage 08 accepted unless the Playwright production acceptance and required runtime/schema evidence pass.

# Final output: report format (strict)

Your final message MUST be in Russian and follow exactly:

1) **Вердикт**

State accepted / blocked / partial, with one sentence explaining why.

2) **Root cause и исправления**

List the confirmed root causes and the scoped fixes. Include contract-impact classification for API/DTO, persistence, config/runtime, UI/browser defaults, and docs.

3) **Playwright evidence**

Include the production URL, what account class was used without credentials, request/response evidence summary, screenshot/artifact paths, and dummy connection cleanup status.

4) **Docs и ledger**

List updated docs and what was recorded in Stage 08 report/iteration ledger.

5) **Direct-main delivery**

Include commit SHA, push status, CI/deploy run status, and post-deploy smoke/Playwright status.

6) **Residual risks**

List only real residual risks or write `None`.
