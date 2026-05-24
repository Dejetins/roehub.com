# Stage 3C: Exchange-control Internal Command API

Дата проверки: 2026-05-24.

Статус: accepted; direct-main delivered; Mac Studio runtime evidence complete.
Stage 3A and Stage 3B evidence are accepted; Stage 3C adds the local-only
authenticated internal API boundary and `apps/api` outbound client contract
before schema/backfill or exchange validation.

## Scope

Stage 3C adds `GET /internal/v1/capabilities`, service-to-service auth,
request headers, `apps/api` client/config, deterministic tests and
no-direct-import evidence. It does not add exchange connection tables, backfill,
create/rotate/disable business handlers, Binance/Bybit validation, UI changes
or order execution.

## Prerequisite Evidence

| Boundary | Endpoint / call | Expected result | Observed result | Blocker |
|---|---|---|---|---|
| Stage 3A | `docs/architecture/identity/exchange-connections-stage-reports/03a-openbao-vault-runtime-provisioning.md` | OpenBao/Vault Transit runtime is accepted before Stage 3C. | Accepted: runtime health, Transit key, ACL, Monit, Prometheus and recovery evidence exist. | None |
| Stage 3B | `docs/architecture/identity/exchange-connections-stage-reports/03b-transit-application-integration.md` | Application secret boundary is accepted before Stage 3C. | Accepted: `ExchangeSecretCipher`, Transit adapter, fail-closed config, redaction and repeated ACL evidence exist. | None |
| Branch gate | `test "$(git branch --show-current)" = main` | Work starts on `main`. | Passed before implementation: current branch is `main`. | None |
| Fast-forward gate | `git pull --ff-only origin main` | Local checkout can fast-forward from `origin/main`. | Passed before implementation: `Already up to date.` | None |

## Internal API Evidence

| Boundary | Endpoint / call | Expected result | Observed result | Blocker |
|---|---|---|---|---|
| `exchange-control` local-only namespace | `GET /internal/v1/capabilities` | Additive internal endpoint on existing `127.0.0.1:9205` service. | Added under `create_exchange_control_app`; production bind remains `127.0.0.1:9205`. | None |
| Capabilities response | Authenticated local curl with `X-Request-Id: stage-3c-smoke` | Compact, secret-free response with service identity, contract version, capabilities, error model and timeout policy. | Passed locally: returned `service=exchange-control`, `service_identity=exchange-control`, `contract_version=internal-v1`, `retry_policy=no_implicit_retry`. | None |
| No business handlers | Source inspection and tests | No create/rotate/disable/validate command handlers are implemented yet. | Only capabilities smoke exists; future command capabilities are marked Stage 4/5 pending. | None |
| Secret-safe payload | `test_internal_capabilities_are_secret_safe` | No token, `api_secret` or passphrase appears in response body. | Passed. | None |

## Service Auth Evidence

| Boundary | Endpoint / call | Expected result | Observed result | Blocker |
|---|---|---|---|---|
| Auth token | `ROEHUB_EXCHANGE_CONTROL_INTERNAL_API_TOKEN` | Product `exchange-control` config requires internal token; value is host-local only. | Covered by `ExchangeControlRuntimeConfig` and focused tests. | None |
| Required service header | `X-Roehub-Internal-Service: apps/api` | Missing/wrong service identity is denied. | Passed in `test_internal_capabilities_require_service_auth_and_headers`: HTTP `403`. | None |
| Required request id | `X-Request-Id` | Missing request id is denied with sanitized error. | Passed in focused tests: HTTP `400` and code `request_id_required`. | None |
| Missing auth runtime call | `curl -i http://127.0.0.1:9205/internal/v1/capabilities -H "X-Roehub-Internal-Service: apps/api"` | Missing token denied with `401`/`403`. | Passed locally: HTTP `401`, code `internal_auth_required`. | None |
| Invalid token | Focused test with wrong bearer token | Invalid token is denied with `403`. | Passed: HTTP `403`, code `internal_auth_denied`. | None |

## Runtime And Delivery Evidence

| Boundary | Endpoint / call | Expected result | Observed result | Blocker |
|---|---|---|---|---|
| Direct-main commit | `git push origin main` | Scoped Stage 3C files are committed and pushed directly to `main`; no stage branch or draft PR. | Passed: commit `54e99b54452f` pushed to `origin/main` (`e37f35fe..54e99b54`). | None |
| CI | GitHub Actions run `26362342903` | Repository checks pass after push. | Passed: `changes`, `static`, `migrations`, test shards and aggregate `ci` completed successfully. | None |
| Deploy | GitHub deploy runs | App image, web and backend deploys complete successfully. | Passed: Publish App Image `26362381051`, Deploy Web `26362381077`, Deploy Backend `26362381076`. | None |
| Mac Studio readiness | `curl -fsS http://127.0.0.1:9205/health/ready` on target runtime | `exchange-control` remains ready after deploy. | Passed after deploy. | None |
| Mac Studio capabilities | Authenticated `curl -fsS http://127.0.0.1:9205/internal/v1/capabilities ...` with host-local token | Authenticated internal call returns sanitized capabilities. | Passed after deploy: `service=exchange-control`, `service_identity=exchange-control`, `contract_version=internal-v1`, `retry_policy=no_implicit_retry`. | None |
| Mac Studio missing auth | `curl -i http://127.0.0.1:9205/internal/v1/capabilities -H "X-Roehub-Internal-Service: apps/api"` | Missing token is denied with `401`/`403`. | Passed after deploy: HTTP `401`, code `internal_auth_required`. | None |
| OpenBao recovery after deploy reload | `/opt/roehub/bin/provision_openbao_transit_stage3a.sh` | OpenBao is unsealed and Transit/key/policies remain present without printing secrets. | Passed: `openbao_unsealed=ok`, `transit_mount=already`, `transit_key=roehub-exchange-credentials`, tokens reused. | None |
| OpenBao health and ACL | `/v1/sys/health`; `/opt/roehub/bin/smoke_openbao_transit_acl.sh` | Accepted Stage 3A runtime remains usable after Stage 3C deploy. | Passed: health `sealed=false`; ACL smoke `exchange_control_encrypt=ok`, `apps_api_decrypt_denied=403`. | None |
| Monit | `monit validate`; `monit summary \| grep -Ei "exchange_control\|openbao"` | Supervised services are visible and OK. | Passed after validation refresh: `roehub_openbao OK`; `roehub_exchange_control OK`. | None |
| Deployed bundle | Local and `/opt/roehub/app` SHA-256 for Stage 3C files | Target runtime runs the pushed source. | Passed: hashes match for `apps/api/exchange_control_client.py`, `apps/api/main/app.py`, exchange-control HTTP app and API launchd plist. | None |

## `apps/api` Client Evidence

| Boundary | Endpoint / call | Expected result | Observed result | Blocker |
|---|---|---|---|---|
| Client port | `apps/api/exchange_control_client.py` | `apps/api` has an outbound client/port for `exchange-control`. | Added `ExchangeControlClient` protocol, HTTP client, deterministic fake and sanitized client errors. | None |
| Base URL config | `ROEHUB_EXCHANGE_CONTROL_INTERNAL_BASE_URL` | Product config has a stable local base URL contract. | `com.roehub.api.plist` defaults it to `http://127.0.0.1:9205`; no token value is committed. | None |
| Fail-closed public route gate | `test_exchange_control_client_config_fails_closed_when_public_routes_enabled` | If future public exchange connection routes are enabled, missing base URL/token fails startup wiring. | Passed for missing base URL and missing token; configured client builds with both values. | None |
| Header contract | `test_exchange_control_http_client_sends_internal_auth_headers` | Client sends bearer auth, `X-Roehub-Internal-Service: apps/api` and `X-Request-Id`. | Passed with deterministic `httpx.MockTransport`. | None |
| Sanitized failures | `test_exchange_control_http_client_sanitizes_failures` | Upstream body/token text is not surfaced in internal client error. | Passed: status is retained, token text is not included. | None |

## No-direct-import Evidence

| Boundary | Endpoint / call | Expected result | Observed result | Blocker |
|---|---|---|---|---|
| `apps/api` grep | `rg -n "ExchangeSecretCipher\|decrypt\|openbao\|vault\|binance\|bybit\|pybit\|api_secret\|passphrase" apps/api \|\| true` | No direct Transit/decrypt/native exchange adapter imports in `apps/api`; literal non-secret market labels may be justified. | Passed for Stage 3C boundary: no `ExchangeSecretCipher`, `decrypt`, `openbao`, `vault`, `pybit`, `api_secret` or `passphrase` matches. Existing `binance` literals are UI/backtest market option strings in `apps/api/wiring/modules/ui_backtests.py`. | None |
| Public API behavior | Focused UI account tests | Existing account public routes remain unchanged. | Passed: account route tests still pass. | None |

## Quality Gates

| Gate | Expected result | Observed result | Blocker |
|---|---|---|---|
| `uv run pytest -q tests/unit/contexts/exchange_control tests/unit/apps/api/test_ui_account_routes.py` | Focused internal API/client tests pass. | Passed: `21 passed`. | None |
| `uv run ruff check apps/api apps/exchange_control src/trading/contexts/exchange_control tests/unit/contexts/exchange_control tests/unit/apps/api` | Lint passes. | Passed: `All checks passed!`. | None |
| `uv run pyright apps/api apps/exchange_control src/trading/contexts/exchange_control tests/unit/contexts/exchange_control tests/unit/apps/api` | Type check passes. | Passed: `0 errors, 0 warnings, 0 informations`. | None |
| `plutil -lint infra/macos/launchd/com.roehub.api.plist infra/macos/launchd/com.roehub.exchange-control.plist` | launchd plists remain valid. | Passed: both `OK`. | None |
| `python -m tools.docs.generate_docs_index --check` | Docs index is current after Markdown changes. | Passed after docs index update. | None |

## Contract Impact Classification

| Dimension | Classification | Reason |
|---|---|---|
| Public API contract | `none` | Existing `apps/api` public routes and payloads are unchanged. |
| Internal API contract | `compatible-change` | Additive local-only `GET /internal/v1/capabilities` contract is introduced for `apps/api -> exchange-control`. |
| Port / client contract | `compatible-change` | `apps/api` gains an outbound `ExchangeControlClient` port and deterministic fake for future Stage 4/5 wiring. |
| DTO schema | `none` | No public DTO or persisted business DTO changes. |
| Persisted schema | `none` | No database migration, table or backfill is introduced. |
| Config schema | `compatible-change` | New internal env contract: `ROEHUB_EXCHANGE_CONTROL_INTERNAL_API_TOKEN`, `ROEHUB_EXCHANGE_CONTROL_INTERNAL_BASE_URL`, optional timeout and future public route enablement flag. |
| Request hash / cache / persistence identity | `none` | No cache keys, request hashes or persistence identity semantics are changed. |
| Ops / runtime | `compatible-change` | Existing supervised `exchange-control` service exposes an authenticated internal namespace on the same local-only listener. |

## Stage 4 / 5 Handoff Facts

- Stage 4 must call `exchange-control` through `ExchangeControlClient`; it must
  not import Transit/decrypt/native exchange adapters into `apps/api`.
- Internal service auth uses `ROEHUB_EXCHANGE_CONTROL_INTERNAL_API_TOKEN`;
  values remain host-local in `/Users/daniildegtyarev/.config/roehub/roehub.env`.
- Internal base URL is `ROEHUB_EXCHANGE_CONTROL_INTERNAL_BASE_URL`, defaulted in
  Mac Studio API launchd to `http://127.0.0.1:9205`.
- Required headers are `Authorization: Bearer <token>`,
  `X-Roehub-Internal-Service: apps/api`, and `X-Request-Id`.
- Error envelope is sanitized as `roe_internal_error_v1`; upstream bodies and
  token values are not surfaced by the client.
- Timeout policy is short default timeout `2.0` seconds and
  `no_implicit_retry`; future mutating create/rotate/disable commands must use
  explicit idempotency keys.
- Stage 4 may now add schema/backfill from this accepted internal command
  boundary. Stage 5 still waits for Stage 4 acceptance.
