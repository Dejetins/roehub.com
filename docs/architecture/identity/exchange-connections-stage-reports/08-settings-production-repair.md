# Stage 8: Settings Production Repair

Дата проверки: 2026-05-24.

Статус: in progress; local repair gates passed, production deploy and
authenticated Playwright acceptance pending direct-main delivery.

Stage 8 supersedes the incomplete Stage 7 readiness claim for the concrete
production-browser `/settings` add-key flow. Scope is limited to the public edge
same-origin context, account-settings schema drift, exchange credential form
password-manager hardening, runtime evidence, and documentation continuity.

## Verdict

| Area | Expected result | Observed evidence | Verdict | Residual risk |
|---|---|---|---|---|
| Root cause | Explain why production returned `Mutation origin is not allowed` and account settings 500s. | Mac Studio deployed copy of `infra/caddy/Caddyfile.vps` lacked standard forwarded headers; after VPS Caddy was corrected, Playwright still proved `csrf_origin_mismatch` because the VPS -> Tailscale Serve hop did not preserve the public host in backend-visible standard `X-Forwarded-*` context. API logs also showed missing account settings columns: `username`, `integration_key`, `autorefresh_preset`, `summary`. | Confirmed. | Final production acceptance still pending the second deploy with `X-Roehub-Forwarded-*`. |
| Local repair | Keep CSRF fail-closed and repair legacy schema without manual DB edits. | Focused tests cover standard forwarded public same-origin, edge-owned `X-Roehub-Forwarded-*` public origin, Referer-only browser mutation, and cross-origin rejection with `csrf_origin_mismatch`; `0006` now repairs legacy profile/integration table shapes idempotently. | Passed locally. | Production DB repair passed first deploy; final browser acceptance pending second edge/backend deploy. |
| Browser hardening | API key/API secret fields should not look like site login/password fields. | Add and rotate forms use non-login field names, `type="text"`, `autocomplete="off"`, password-manager ignore attributes, and form-level `data-form-type="other"`. | Passed locally. | Final browser run must confirm no save-password prompt. |
| Production acceptance | Prove authenticated `https://roehub.com/settings` add flow. | Pending after direct-main deploy. | Pending. | Stage remains not accepted until Playwright evidence passes. |

## Issue / Root Cause / Fix

| Issue | Root cause | Fix | Validation command | Observed evidence | Verdict |
|---|---|---|---|---|---|
| `POST /api/ui/account/exchange-connections` returned `403` with `Mutation origin is not allowed` and `csrf_origin_mismatch`. | Browser POST carried `Referer: https://roehub.com/settings` but no visible `Origin`; after standard VPS Caddy headers were deployed, the backend still saw upstream context through Tailscale Serve instead of the public host. | Keep CSRF logic fail-closed; add exact Referer-only tests for both standard `X-Forwarded-*` and edge-owned `X-Roehub-Forwarded-*`; deploy-web checks that `/api/*` Caddy config contains all four forwarded-origin headers. | `uv run pytest -q tests/unit/apps/api/test_ui_account_routes.py` | Passed inside the focused suite: Referer-only standard and edge-forwarded public host accepted; Referer-only cross-origin rejected with `csrf_origin_mismatch`. | Local pass; second production deploy pending. |
| `/api/ui/account/profile` returned production 500. | Production `identity_user_profile_overrides` had legacy columns `display_name`, `created_at`, `updated_at`; repository selected `username`, `email`, `telegram_discord`. | `0006_identity_account_settings_v1.sql` additively adds current columns, maps `display_name -> username`, and sets a safe default for legacy `created_at`. | Mac Studio log grep and schema introspection; focused migration static test. | Logs captured `psycopg.errors.UndefinedColumn: column "username" does not exist`; schema introspection confirmed legacy shape. | Local repair ready; production bootstrap pending. |
| `/api/ui/account/integrations` returned production 500. | Production `identity_integrations` had legacy `provider/enabled/settings_json` shape and primary key `(owner_user_id, provider)`; repository expects `integration_key`, `mode`, `webhook_url_masked` and upsert on `(owner_user_id, integration_key)`. | `0006` adds current columns, maps legacy providers to safe current keys, switches primary key to `(owner_user_id, integration_key)` when needed, and preserves old columns as inert compatibility baggage. | Mac Studio log grep and schema introspection; focused migration static test. | Logs captured `psycopg.errors.UndefinedColumn: column "integration_key" does not exist`; constraints showed legacy primary key. | Local repair ready; production bootstrap pending. |
| Browser/password manager treated exchange credential form like site credentials. | Credential inputs were already renamed away from `api_key`/`api_secret`, but form-level ignore hints were absent. | Add form-level `data-lpignore`, `data-1p-ignore`, `data-bwignore`, and `data-form-type="other"` to add and rotate forms; keep secret inputs cleared on success and failure. | `uv run pytest -q tests/unit/apps/web/test_app_routes.py` | Static route test asserts non-login names and form/input password-manager hints. | Local pass; production browser pending. |

## Runtime Evidence

| Check | Command / source | Observed evidence | Verdict |
|---|---|---|---|
| Public edge identity | `curl -fsS https://roehub.com/__edge_id` | Returned `vps-edge`. | Pass. |
| Mac Studio deployed Caddy source copy | `ssh macstudio 'cd /opt/roehub/app && sed -n "1,80p" infra/caddy/Caddyfile.vps'` | `/api/*` block lacked `header_up X-Forwarded-Host {host}` and `header_up X-Forwarded-Proto {scheme}` before Stage 8 repair. | Confirmed deployed-source drift. |
| First VPS Caddy deploy | Deploy Web `26372595761` | Pinned ED25519 host-key check passed; active `/etc/caddy/Caddyfile` contained standard `X-Forwarded-Host` and `X-Forwarded-Proto`; Caddy validate/reload and public edge smoke passed. | Standard headers present, but Playwright still failed through Tailscale Serve. |
| Tailscale Serve hop | Authenticated Playwright POST after first deploy | Request had `Referer: https://roehub.com/settings`, no visible `Origin`, and response remained `403 csrf_origin_mismatch`. | Confirms backend-visible public context needs edge-owned `X-Roehub-Forwarded-*` copy. |
| Production API logs | `ssh macstudio 'tail ... api.err.log | grep ...'` | Captured stack traces for missing `username`, `integration_key`, `autorefresh_preset`, and `summary`; latest open items are profile/integrations. | Confirms schema drift. |
| Production schema | `information_schema.columns` and `pg_constraint` on Mac Studio Postgres. | `identity_user_profile_overrides` had `display_name`; `identity_integrations` had `provider`, `enabled`, `settings_json`, primary key `(owner_user_id, provider)`. | Confirms repair target. |
| VPS SSH host key | Local known_hosts for `155.212.170.144`. | ED25519 fingerprint was pinned locally; local root SSH auth was not available. GitHub deploy workflow now checks the pinned fingerprint before trusting the host key. | Local direct VPS inspection blocked; deploy path hardened. |

## Quality Gates

| Gate | Expected result | Observed result |
|---|---|---|
| Focused tests | `uv run pytest -q tests/unit/apps/api/test_ui_account_routes.py tests/unit/apps/web/test_app_routes.py tests/unit/apps/migrations` | Passed: `65 passed, 3 warnings`. |
| Ruff | `uv run ruff check apps/api apps/web src/trading/contexts/identity tests/unit/apps/api tests/unit/apps/web tests/unit/apps/migrations` | Passed. |
| Pyright | `uv run pyright apps/api src/trading/contexts/identity tests/unit/apps/api` | Passed: `0 errors`. |
| Scratch SQL execution | Mac Studio scratch DB using production Postgres. | Not run: production DB role cannot create scratch database. | Environmental limitation; production bootstrap will be the execution gate. |
| Docs index | `python -m tools.docs.generate_docs_index --check` | Passed after updating `docs/architecture/README.md`. |
| Edge id | `curl -fsS https://roehub.com/__edge_id` | Returned `vps-edge`. |
| Playwright prerequisite | `command -v npx >/dev/null 2>&1 && npx playwright --version` | Passed; Playwright `1.60.0`. |
| Secret artifact grep | `rg -n "TEST_SECRET\|TEST_API_SECRET\|TEST_PASSPHRASE\|dummy-secret\|dummy_api_secret" logs output .playwright-cli 2>/dev/null \|\| true` | Passed; no matches. |

## Playwright Evidence

| Workflow | Required evidence | Observed evidence | Verdict |
|---|---|---|---|
| Authenticated load | `https://roehub.com/settings` opens with smoke Keycloak account; `/api/ui/account/profile` and `/api/ui/account/integrations` return 200. | Pending post-deploy Playwright. | Pending. |
| Add dummy connection | Submit dummy Binance or Bybit credentials to `/api/ui/account/exchange-connections`; response is not `Mutation origin is not allowed` and not `csrf_origin_mismatch`. | Pending post-deploy Playwright. | Pending. |
| Secret handling | Secret fields clear after submit; artifacts contain no dummy or real secret values. | Pending post-deploy Playwright and secret grep. | Pending. |
| Cleanup | Disable/delete any dummy connection created by the proof. | Pending post-deploy Playwright. | Pending. |

## Contract Impact Classification

| Dimension | Classification | Reason |
|---|---|---|
| Public API / DTO | `none` | Route names and request/response fields are unchanged; behavior is a bug fix for same-origin production context and schema availability. |
| Persistence | `compatible-change` | `0006` additively repairs legacy account-settings columns and primary-key shape required by current repository code; old columns are not dropped. |
| Config / runtime | `compatible-change` | Deploy workflow now requires pinned VPS ED25519 fingerprint and validates standard plus `X-Roehub-Forwarded-*` Caddy forwarded-origin headers before reload. Public URLs and upstreams are unchanged. |
| UI/browser defaults | `compatible-change` | Exchange credential fields keep the same submitted DTO mapping but add password-manager ignore metadata; default permission remains `read`. |
| Secret custody | `none` | `apps/api` still calls `exchange-control`; no decrypt, Transit, native exchange SDK, or order placement path is added. |
| Trading execution | `none` | No signal-to-execution, order placement, order ledger, or live order path is introduced. |

## Troubleshooting

| Symptom | First check | Expected repair signal |
|---|---|---|
| `Mutation origin is not allowed` | Inspect active VPS Caddy `/api/*` block for `X-Forwarded-Host`, `X-Forwarded-Proto`, `X-Roehub-Forwarded-Host`, and `X-Roehub-Forwarded-Proto`. | All four headers are forwarded and Referer-only `https://roehub.com/settings` mutations pass through the VPS -> Tailscale Serve path. |
| `/api/ui/account/profile` 500 | Check `identity_user_profile_overrides` columns and API logs for `username`. | Current columns `username`, `email`, `telegram_discord` exist. |
| `/api/ui/account/integrations` 500 | Check `identity_integrations` columns and primary key. | `integration_key`, `mode`, `webhook_url_masked` exist and primary key is `(owner_user_id, integration_key)`. |
| Browser save-password prompt after exchange key submit | Inspect `/settings` add and rotate forms for non-login names and password-manager ignore attributes. | Forms and inputs have ignore attributes; secret inputs clear after submit. |

## Direct-Main Delivery Evidence

| Item | Evidence | Result |
|---|---|---|
| Branch | `test "$(git branch --show-current)" = main` | Pending final pre-push check. |
| Fast-forward | `git pull --ff-only origin main` | Pending before scoped staging. |
| Commit | Pending. | Pending. |
| Push | Pending. | Pending. |
| CI/deploy | Pending. | Pending. |
| Post-deploy smoke/Playwright | Pending. | Pending. |
