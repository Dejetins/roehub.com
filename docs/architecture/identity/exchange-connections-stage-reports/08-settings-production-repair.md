# Stage 8: Settings Production Repair

Дата проверки: 2026-05-24.

Статус: accepted. Stage 8 supersedes the incomplete Stage 7 readiness claim for
the concrete production-browser `/settings` add-key flow.

Scope was limited to public edge same-origin context, account-settings schema
drift, exchange credential form password-manager hardening, runtime evidence,
and documentation continuity. Trading execution, live orders and direct secret
custody in `apps/api` remained out of scope.

## Verdict

| Area | Expected result | Observed evidence | Verdict | Residual risk |
|---|---|---|---|---|
| Root cause | Explain production `Mutation origin is not allowed`, account settings 500s and browser credential-form prompt risk before fixing. | Confirmed three defects: public-origin context was lost through the VPS -> Tailscale Serve hop; production account settings tables had legacy columns; exchange credential forms needed form-level password-manager ignore hints. A fourth runtime blocker was found after deploy: OpenBao restarted sealed, so Transit encrypt returned 503 until operator unseal. | Accepted. | None. |
| Local repair | Keep CSRF fail-closed, repair schema idempotently and harden form defaults. | Focused tests cover Referer-only browser mutation with trusted public forwarded context, edge-owned `X-Roehub-Forwarded-*` headers, and true cross-origin rejection with `csrf_origin_mismatch`; migration tests cover legacy `integration_key` repair; web tests cover `autocomplete`/password-manager hardening. | Accepted. | None. |
| Runtime repair | Public edge and Mac Studio runtime must match the production browser path. | Deploy Web `26374257112` validated active VPS Caddy with `X-Forwarded-Host`, `X-Forwarded-Proto`, `X-Roehub-Forwarded-Host`, `X-Roehub-Forwarded-Proto`; Deploy Backend `26374257097` applied schema bootstrap; OpenBao was unsealed and Transit ACL smoke passed. | Accepted. | None. |
| Browser acceptance | Prove authenticated `https://roehub.com/settings` add flow with dummy credentials only and cleanup. | Playwright accepted artifact shows settings GETs 200, default permission `read`, POST `/api/ui/account/exchange-connections` returned 201, secret inputs cleared, dummy connection disabled with 200, console warnings/errors 0. | Accepted. | None. |

## Issue / Root Cause / Fix

| Issue | Root cause | Fix | Validation command | Observed evidence | Verdict |
|---|---|---|---|---|---|
| `POST /api/ui/account/exchange-connections` returned `403` with `Mutation origin is not allowed` and `csrf_origin_mismatch`. | Browser POST had same-origin `Referer: https://roehub.com/settings` but no visible `Origin`; standard VPS `X-Forwarded-*` headers were not enough because the VPS -> Tailscale Serve hop rewrote backend-visible public context. | Kept CSRF fail-closed; added trusted edge-owned `X-Roehub-Forwarded-Host` / `X-Roehub-Forwarded-Proto`; added deploy-web drift checks for all four Caddy headers. | `uv run pytest -q tests/unit/apps/api/test_ui_account_routes.py` | Referer-only public same-origin mutations pass only with trusted public forwarded context; true cross-origin requests remain rejected with `csrf_origin_mismatch`. First post-standard-header Playwright still failed with 403; second edge-owned-header deploy passed. | Accepted. |
| `/api/ui/account/profile` returned production 500. | Production `identity_user_profile_overrides` still had legacy `display_name`, while current repository selected `username`, `email`, `telegram_discord`. | `0006_identity_account_settings_v1.sql` additively adds current columns and maps `display_name -> username` without dropping old columns. | Mac Studio schema introspection and `uv run pytest -q tests/unit/apps/migrations` | Post-deploy authenticated settings load returned `/api/ui/account/profile` 200. | Accepted. |
| `/api/ui/account/integrations` returned production 500. | Production `identity_integrations` still had legacy `provider/enabled/settings_json` and primary key `(owner_user_id, provider)` while current code expects `integration_key`, `mode`, `webhook_url_masked`, and `(owner_user_id, integration_key)`. | `0006` additively repairs current columns, maps legacy providers, and switches the primary key when needed. | Mac Studio schema introspection and `uv run pytest -q tests/unit/apps/migrations` | Post-deploy authenticated settings load returned `/api/ui/account/integrations` 200 and integration rows include `integration_key`. | Accepted. |
| Browser/password manager treated exchange credential form like site credentials. | Add/rotate forms had secret-looking credential inputs but lacked form-level password-manager ignore hints. | Add and rotate forms now use non-login field names, `autocomplete="off"`, `data-lpignore`, `data-1p-ignore`, `data-bwignore`, and `data-form-type="other"`; secret fields clear after success and failure. | `uv run pytest -q tests/unit/apps/web/test_app_routes.py`; Playwright artifact field attrs. | Playwright artifact records `exchange_public_token` / `exchange_private_token`, `autocomplete=off`, all ignore attrs, and zero-length secret fields after submit. | Accepted. |
| POST advanced past CSRF/recent-auth but returned `503 exchange_control_unavailable`. | OpenBao file-storage runtime was sealed after deploy/restart, so `exchange-control` Transit encrypt raised `ExchangeSecretCipherError: transit request failed with status 503`. | Used existing host-local `/opt/roehub/bin/provision_openbao_transit_stage3a.sh` to unseal without printing keys; reran Transit ACL smoke. | `bash /opt/roehub/bin/smoke_openbao_transit_acl.sh` | `openbao_sealed=False`, `exchange_control_encrypt=ok`, `apps_api_decrypt_denied=403`; subsequent Playwright POST returned 201 and cleanup disable returned 200. | Accepted. |

## Runtime Evidence

| Check | Command / source | Observed evidence | Verdict |
|---|---|---|---|
| Public edge identity | `curl -fsS https://roehub.com/__edge_id` | Returned `vps-edge`. | Pass. |
| Active VPS Caddy config | Deploy Web `26374257112` | Pinned ED25519 fingerprint matched `SHA256:MQPcAz0ewaAU5IvqU1AMJ1ba+NCjoF4gY7u9hgpP+lY`; active `/etc/caddy/Caddyfile` contained standard and `X-Roehub-Forwarded-*` headers; `caddy validate` and reload passed. | Pass. |
| Mac Studio schema bootstrap | Deploy Backend `26374257097` plus Postgres introspection | `0006_identity_account_settings_v1.sql` applied; `identity_integrations` has `integration_key`, `mode`, `webhook_url_masked` and PK `(owner_user_id, integration_key)`; profile overrides have `username`, `email`, `telegram_discord`. | Pass. |
| Mac Studio runtime smoke | `bash scripts/macos/smoke_prod.sh` on target runtime | Backend smoke passed after deploy; expected unauthenticated API 401 remained intact; metrics and Tailscale checks passed. | Pass. |
| OpenBao / Transit | Host-local unseal/provision script and ACL smoke | `openbao_unsealed=ok`, `openbao_sealed=False`, `exchange_control_encrypt=ok`, `apps_api_decrypt_denied=403`. | Pass. |
| Secret artifact safety | `rg -n "TEST_SECRET\|TEST_API_SECRET\|TEST_PASSPHRASE\|dummy-secret\|dummy_api_secret" logs output .playwright-cli 2>/dev/null \|\| true` plus sanity grep for smoke credential/dummy prefixes. | No matches. | Pass. |

## Quality Gates

| Gate | Observed result |
|---|---|
| `uv run pytest -q tests/unit/apps/api/test_ui_account_routes.py tests/unit/apps/web/test_app_routes.py tests/unit/apps/migrations` | Passed: `65 passed, 3 warnings`. |
| `uv run ruff check apps/api apps/web src/trading/contexts/identity tests/unit/apps/api tests/unit/apps/web tests/unit/apps/migrations` | Passed. |
| `uv run pyright apps/api src/trading/contexts/identity tests/unit/apps/api` | Passed: `0 errors`. |
| `python -m tools.docs.generate_docs_index --check` | Passed after final report and docs-index update. |
| `command -v npx >/dev/null 2>&1 && npx playwright --version` | Passed; Playwright `1.60.0`. |
| `test "$(git branch --show-current)" = main` and `git pull --ff-only origin main` | Passed before direct-main push. |
| `gh --version && gh auth status` | Passed. |

## Playwright Evidence

| Workflow | Required evidence | Observed evidence | Verdict |
|---|---|---|---|
| Authenticated load | `https://roehub.com/settings` opens with smoke Keycloak account; settings APIs return no 500s. | `output/playwright/settings-stage08-production-sanitized-20260524T222715Z.json` records 200 for `/api/ui/account/profile`, `/api/ui/account/integrations`, `/api/ui/account/limits`, `/api/ui/account/exchange-connections`. | Pass. |
| Add dummy connection | Submit dummy Binance/Bybit credentials to `/api/ui/account/exchange-connections`; response is not `Mutation origin is not allowed` and not `csrf_origin_mismatch`. | Same artifact records POST `https://roehub.com/api/ui/account/exchange-connections`, request `referer=https://roehub.com/settings`, response status 201, permission `read`, validation `skipped_external_validation`; API key/secret redacted. | Pass. |
| Secret handling | Secret fields clear after submit; artifacts contain no dummy or real secret values. | Same artifact records `secretLengthsAfterSubmit.api_key=0` and `api_secret=0`; secret grep and sanity grep had no matches. | Pass. |
| Cleanup | Disable/delete any dummy connection created by the proof. | Same artifact records cleanup POST status 200 and post-cleanup list status `disabled`, `status_reason=user_disabled`. | Pass. |
| Browser console and screenshot | Capture console summary and final screenshot. | `output/playwright/settings-stage08-console-20260524T222742Z.txt` shows `Errors: 0, Warnings: 0`; scoped screenshot is `output/playwright/settings-stage08-exchange-panel-only-2026-05-24T22-28-34-016Z.png`. | Pass. |

## Contract Impact Classification

| Dimension | Classification | Reason |
|---|---|---|
| Public API / DTO | `none` | Route names and request/response fields are unchanged; behavior is a bug fix for production same-origin context and schema availability. |
| Persistence | `compatible-change` | `0006` additively repairs legacy account-settings columns and primary-key shape required by current repository code; old columns are preserved. |
| Config / runtime | `compatible-change` | Deploy workflow now requires pinned VPS ED25519 fingerprint and validates standard plus `X-Roehub-Forwarded-*` Caddy forwarded-origin headers before reload. Public URLs and upstreams are unchanged. |
| UI/browser defaults | `compatible-change` | Exchange credential fields keep the same submitted DTO mapping but add password-manager ignore metadata; default permission remains `read`. |
| Secret custody | `none` | `apps/api` still calls `exchange-control`; no decrypt, Transit, native exchange SDK, or order placement path is added. |
| Trading execution | `none` | No signal-to-execution, order placement, order ledger, or live order path is introduced. |
| Docs | `compatible-change` | Stage 8 supersession, edge header contract, OpenBao unseal handoff and Playwright evidence are documented. |

## Troubleshooting

| Symptom | First check | Expected repair signal |
|---|---|---|
| `Mutation origin is not allowed` | Inspect active VPS Caddy `/api/*` block for `X-Forwarded-Host`, `X-Forwarded-Proto`, `X-Roehub-Forwarded-Host`, and `X-Roehub-Forwarded-Proto`. | All four headers are forwarded and Referer-only `https://roehub.com/settings` mutations pass through the VPS -> Tailscale Serve path. |
| `/api/ui/account/profile` 500 | Check `identity_user_profile_overrides` columns and API logs for `username`. | Current columns `username`, `email`, `telegram_discord` exist. |
| `/api/ui/account/integrations` 500 | Check `identity_integrations` columns and primary key. | `integration_key`, `mode`, `webhook_url_masked` exist and primary key is `(owner_user_id, integration_key)`. |
| Browser save-password prompt after exchange key submit | Inspect `/settings` add and rotate forms for non-login names and password-manager ignore attributes. | Forms and inputs have ignore attributes; secret inputs clear after submit. |
| `exchange_control_unavailable` during add-key | Check OpenBao health on Mac Studio. | `/v1/sys/health` reports `sealed=false`; Transit ACL smoke returns `exchange_control_encrypt=ok`. |

## Direct-Main Delivery Evidence

| Item | Evidence | Result |
|---|---|---|
| Branch | `test "$(git branch --show-current)" = main` | Passed; no stage branch or draft PR. |
| Fast-forward | `git pull --ff-only origin main` | Passed before scoped staging and direct push. |
| Implementation commits | `0b77d6e1 Repair settings exchange production flow`; `7a7d40b3 Preserve public origin through Tailscale edge hop`. | Pushed to `origin/main`. |
| CI / deploy for first repair | CI `26372497555`, Deploy Backend `26372572180`, Publish App Image `26372572170`, Deploy Web `26372595761` / `26372591033`. | Succeeded; schema repair applied; standard Caddy headers deployed; first Playwright proved edge-owned header copy was still required. |
| CI / deploy for edge-hop repair | CI `26374180601`, Deploy Backend `26374257097`, Publish App Image `26374257127`, Deploy Web `26374257112`. | Succeeded; active VPS Caddy validated with standard and `X-Roehub-Forwarded-*` headers; backend deploy and schema bootstrap succeeded. |
| Post-deploy runtime | Mac Studio smoke, OpenBao unseal/Transit ACL smoke, production Playwright. | Passed; final accepted Playwright artifact stored under `output/playwright/`. |
