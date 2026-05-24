# Stage 6: Settings UI

Дата проверки: 2026-05-24.

Статус: accepted locally; direct-main delivery evidence is recorded in the shared
iteration ledger after push/CI/deploy observation.

Stage 6 completes the `/settings` exchange connection workstation flow for key
storage and validation management. The browser UI now uses the account
exchange-connections backend read model, sends explicit `environment` and
`permissions`, defaults permissions to `read`, and exposes validate, rotate and
disable actions without rendering raw secrets or exchange error bodies.

## Scope

| Area | Expected result | Observed result | Evidence |
|---|---|---|---|
| Exchange list source | `/settings` reads backend connection status, not legacy/synthetic UI rows. | Settings now uses `/api/ui/account/exchange-connections`; table renders `status`, `validation_status`, `ip_restriction_status`, masked `api_key`, and `last_validated_at`. | `apps/web/templates/pages/settings.html`; `apps/web/dist/js/pages/settings.js`; focused tests |
| Add form defaults | `permissions` defaults to `read`; `trade` is opt-in only. | Initial browser state reports `defaultPermissions=read`; request capture shows first create sends `permissions=read`, second create sends `permissions=trade` only after explicit selector change. | Browser QA payload capture |
| Environment control | User explicitly chooses `mainnet` or `testnet`. | Initial browser state reports `defaultEnvironment=mainnet`; opt-in trade create sends `environment=testnet` after selector change. | Browser QA payload capture |
| Secret handling | Add and rotate forms are write-only and clear password inputs on failure and success. | Failure path returned deterministic 422 and cleared password fields; success path and rotate path also cleared password fields. | Browser QA secret cleanup checks; artifact grep |
| Management actions | Validate, rotate, and disable flows exist. | Browser flow validated a connection to `valid_readonly`, rotated to masked key suffix `****3333`, and disabled the selected connection after typed confirmation. | Browser QA workflow table |
| Limits | Account counts are backend-derived, not hardcoded template/read-model counters. | Template no longer renders fixed used values; `/ui/account/limits` derives exchange/API-key used counts from the backend exchange connection list. | `apps/web/templates/fragments/account/limits.html`; `apps/api/routes/ui_account.py`; route tests |

## Browser Workflow Evidence

| Workflow | Expected result | Observed result | Artifact |
|---|---|---|---|
| Authenticated `/settings` load | Settings opens without redirect and account API calls return 200. | Fresh browser context with session cookie opened `/settings`; account profile, limits, integrations, notifications, preferences, exchange-connections, sessions and audit requests returned 200. | Playwright requests output |
| Invalid add request | Error path clears secret inputs and does not display raw secret values. | Empty API key request returned 422 and all password inputs were empty after failure. | Playwright run-code result; artifact grep |
| Read-only add request | Create request sends `permissions=read`, `environment=mainnet`, and table shows masked key/status. | Request capture recorded `permissions=read`, `environment=mainnet`; table showed masked key and backend validation status. | Playwright run-code result |
| Trade opt-in add request | `trade` is sent only after explicit selector change. | Request capture recorded `permissions=trade`, `environment=testnet` only for the opt-in create. | Playwright run-code result |
| Validate | Validation action refreshes backend status. | Validate action returned 200 and table showed `valid readonly` plus last validation timestamp. | Playwright requests output |
| Rotate | Rotate form sends write-only credentials and refreshes masked key. | Rotate returned 200 and table showed masked key suffix `****3333`; rotate secret fields were not visible in artifacts. | Playwright requests output; screenshots |
| Disable | Disable requires typed confirmation and refreshes status. | Disable returned 200 after `DISABLE` confirmation and table showed `disabled`; disabled row rotate/disable actions became unavailable. | Playwright requests output; screenshots |

## Viewport Evidence

| Viewport | Expected result | Observed result | Artifact |
|---|---|---|---|
| Desktop 1440x960 | Dense settings workstation remains coherent; exchange form/table do not overlap. | Exchange panel uses separate grid rows for heading, form and table; submit is clickable after environment/permissions selection. | `output/playwright/stage6-settings-desktop.png` |
| Mobile 390x844 | Settings panels stack coherently and exchange panel remains visible. | `mobileRootVisible=true`, `mobileExchangeVisible=true`; screenshot captured stacked workstation. | `output/playwright/stage6-settings-mobile.png` |

## Console And Network Evidence

| Check | Expected result | Observed result | Artifact |
|---|---|---|---|
| Authenticated load console | No unexpected console errors. | Fresh authenticated load had 0 console errors and 0 warnings. | Playwright console output |
| Error-path console | Validation failure may log expected 422 only. | After intentional invalid add request, console contained one expected 422 for `/api/ui/account/exchange-connections`. | Playwright console output |
| Network status | Mutations hit account exchange-connections endpoints and refresh the list. | Create returned 201; validate/rotate/disable returned 200; list refreshes returned 200. | Playwright requests output |
| Payload semantics | Create payload carries explicit permission/environment values. | Sanitized browser capture recorded read/mainnet first create, trade/testnet opt-in create, and no permission/environment values on validate/rotate/disable action routes. | Playwright run-code result |

## Secret Artifact Grep

| Scope | Command | Expected result | Observed result |
|---|---|---|---|
| Browser artifacts | `rg -n 'STAGE6_BROWSER_SECRET\|STAGE6_SECRET\|STAGE6_BROWSER_PASSPHRASE\|api_secret\|passphrase\|TEST_PASSPHRASE' output/playwright/stage6-settings-desktop.png output/playwright/stage6-settings-mobile.png .playwright-cli \|\| true` | No matches. | Passed; no matches. |
| UI/API static guard | `rg -n 'permissions: "trade"\|exchange_connections_used=0\|api_keys_used=0\|128 ms\|needsAttention' apps/web apps/api \|\| true` | No matches. | Passed; no matches. |

## Quality Gates

| Gate | Expected result | Observed result |
|---|---|---|
| Branch | `test "$(git branch --show-current)" = main` | Passed. |
| GitHub auth | `gh --version && gh auth status` | Passed; GitHub CLI authenticated. |
| Focused tests | `uv run pytest -q tests/unit/apps/web/test_app_routes.py tests/unit/apps/api/test_ui_account_routes.py` | Passed: 44 passed, 3 pre-existing httpx cookie deprecation warnings. |
| Lint | `uv run ruff check apps/web apps/api tests/unit/apps/web tests/unit/apps/api` | Passed. |
| Type check | `uv run pyright apps/api tests/unit/apps/api` | Passed: 0 errors. |
| Docs index | `python -m tools.docs.generate_docs_index --check` | To be run after this report and ledger update. |

## Contract Impact Classification

| Dimension | Classification | Reason |
|---|---|---|
| Public API contract | `compatible-change` | `/ui/account/limits` response shape is unchanged, but exchange/API-key used counts now come from exchange-control read model instead of fixed placeholders. |
| Browser-visible behavior | `compatible-change` | `/settings` default permission is explicit `read`; `trade` remains available only as an explicit selector value. |
| DTO schema | `none` | Existing account/exchange DTO fields are reused. |
| Persisted schema | `none` | No migration or persisted shape changed. |
| Config schema | `none` | No runtime config keys changed. |
| Request hash/cache identity | `none` | No cache keys or request identity semantics changed. |
| Runtime/ops | `none` | Stage 6 changes web/API UI surfaces only; no Mac Studio service config changed. |

## Rollback Notes

| Scenario | Rollback path | Data impact |
|---|---|---|
| UI regression | Revert Stage 6 web template/JS/CSS changes to previous `/settings` exchange fragment. | No schema rollback; stored exchange connections remain intact. |
| Limits read-model issue | Revert `/ui/account/limits` count derivation to previous response behavior, or gate the exchange-control list dependency. | No data mutation. |
| Browser action issue | Disable the UI buttons by reverting settings JS action handlers; public account endpoints remain available for API clients. | No data mutation beyond user-triggered create/rotate/disable already audited by backend. |

## Stage 7 Handoff Facts

- Stage 7 can start from a browser-verified `/settings` UI that sends
  explicit `permissions` and `environment`.
- `read` is the only default permission; `trade` is opt-in and should remain a
  higher-risk capability in future production-readiness work.
- UI-visible connection status now comes from backend fields:
  `status`, `validation_status`, `ip_restriction_status`, and
  `last_validated_at`.
- Browser artifact grep passed for Stage 6 sentinel markers and secret field
  names.
- The settings UI still uses existing session/recent-auth/API route gates; Stage
  7 should verify production runtime auth and direct Mac Studio behavior after
  deployment.
