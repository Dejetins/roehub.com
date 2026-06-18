# Stage 5: Binance/Bybit Validation

Дата проверки: 2026-05-24.

Статус: accepted; direct-main delivery and Mac Studio runtime evidence complete.

Stage 2, Stage 3A, Stage 3B, Stage 3C and Stage 4 are accepted. Stage 5
implemented the validation boundary, deterministic status mapping, local API
flow, metrics and audit events. Mac Studio host-local env contains
`ROEHUB_EXCHANGE_VALIDATION_LIVE=1` and readonly Binance/Bybit validation env
vars. Both live readonly metadata smokes now pass as `valid_readonly`; no raw
API keys, secrets or exchange error bodies were printed or committed.

## Scope

Stage 5 adds native Binance/Bybit credential validation without order
placement. Validation is routed through `apps/api -> exchange-control`; the
public API remains a facade and does not import Transit decrypt, credential
resolver or native exchange adapter code. This stage does not place, cancel,
amend or reconcile orders.

## Prerequisite Evidence

| Exchange | Scenario | Env vars | Expected status | Observed status | Blocker |
|---|---|---|---|---|---|
| Stage 3A | OpenBao/Vault Transit runtime | `OPENBAO_ADDR`, Transit tokens | Accepted prerequisite. | Accepted in `03a-openbao-vault-runtime-provisioning.md`. | None |
| Stage 3B | Transit application integration | Stage 3A env contract | Accepted prerequisite. | Accepted in `03b-transit-application-integration.md`. | None |
| Stage 3C | Internal command API | `ROEHUB_EXCHANGE_CONTROL_INTERNAL_API_TOKEN`, `ROEHUB_EXCHANGE_CONTROL_INTERNAL_BASE_URL` | Accepted prerequisite. | Accepted in `03c-exchange-control-internal-command-api.md`. | None |
| Stage 4 | Persisted connection model | `IDENTITY_PG_DSN` on target runtime | Accepted prerequisite. | Accepted in `04-connections-credential-versions-backfill.md`. | None |

## Validation Contract

| Exchange | Scenario | Env vars | Expected status | Observed status | Blocker |
|---|---|---|---|---|---|
| Binance mainnet | API key permission mapping | Native signed `GET /sapi/v1/account/apiRestrictions` | Map read-only, trade, transfer/withdrawal, IP restriction and unsupported mode to normalized statuses. | Implemented in `BinanceExchangeCredentialValidator`; unit mapping covers `valid_readonly`, `valid_trade_enabled`, `invalid_permissions`, `invalid_ip_restriction`, `unsupported_account_mode`. | None |
| Binance Demo Spot | API key permission mapping | Native signed `GET https://demo-api.binance.com/api/v3/account`; for requested trade, safe `POST /api/v3/order/test` probe | Demo Spot keys from `demo.binance.com` must validate against the documented Spot Demo endpoint, not legacy `testnet.binance.vision` or SAPI `apiRestrictions`. | Implemented in `BinanceExchangeCredentialValidator`; unit mapping covers read/trade/mismatch and endpoint routing. | None |
| Binance Demo Futures | API key permission mapping | Native signed `GET https://demo-fapi.binance.com/fapi/v2/account` | Futures demo keys must validate against the documented Futures Demo endpoint. | Implemented in `BinanceExchangeCredentialValidator`; unit routing covers `demo-fapi.binance.com`. | None |
| Bybit | API key information mapping | Native signed `GET /v5/user/query-api` | Map `readOnly`, per-market `permissions`, `ips` and account mode to normalized statuses. Spot trade requires `SpotTrade`; futures trade requires `ContractTrade` with `Order` + `Position`, `DerivativesTrade`, or `OptionsTrade` for USDC contracts. | Implemented in `BybitExchangeCredentialValidator`; unit mapping covers status classes and per-market Spot/Futures permission buckets. | None for host-local readonly credential smoke. |
| Invalid credentials | Native HTTP/auth rejection | Test fake payloads | Return `invalid_credentials` without raw exchange body. | Unit mapping returns `invalid_credentials`; raw `retMsg` is not copied into reason. | None for deterministic evidence. |
| API facade | `POST /api/ui/account/exchange-connections/{connection_id}/validate` | Roehub session, same-origin headers | Route through `ExchangeControlClient` only. | Implemented and tested with deterministic fake client; response contains no secret fields. | None |
| Internal boundary | `POST /internal/v1/exchange-connections/{connection_id}/validate` | Internal token headers | `exchange-control` owns decrypt and native exchange calls. | Implemented; default dev config returns `skipped_external_validation`; post-deploy Mac Studio internal validation returned `valid_readonly` for Binance and Bybit. | None |
| Metrics | `/metrics` | None | Increment bounded labels only: exchange/result/reason. | `exchange_connection_validation_total` and `exchange_connection_status` updated by internal validate flow. | None for deterministic evidence. |
| Audit | Account audit events | None | Emit secret-free validation event. | `exchange_connection_validated` writes `exchange` and `validation_status` only. | None for deterministic evidence. |

Sources verified before implementation:

- Binance mainnet: `GET /sapi/v1/account/apiRestrictions`, response fields
  `ipRestrict`, `enableReading`, withdrawal/transfer flags and trading flags.
- Binance Demo Spot: REST base URL `https://demo-api.binance.com/api`, signed
  `GET /api/v3/account` for read validation, and side-effect-free signed
  `POST /api/v3/order/test` for trade permission probing.
- Binance Demo Futures: REST base URL `https://demo-fapi.binance.com`, signed
  `GET /fapi/v2/account` for read/trade capability.
- Binance: signed request security requires timestamp plus HMAC signature.
- Bybit: `GET /v5/user/query-api`, response fields `readOnly`, `secret=""`
  and `permissions`.
- Bybit market-scoped validation: `Spot`/`SpotTrade` proves Roehub `spot`;
  `ContractTrade` with `Order` + `Position`, `Derivatives`/`DerivativesTrade`
  or `Options`/`OptionsTrade` proves Roehub `futures`. Native `linear` /
  `inverse` are provider categories only and are not persisted as Roehub
  `market_type` values.
- Bybit: `X-BAPI-*` signed request rule uses timestamp, API key, recv window
  and query string for GET requests.

## Bybit Omni-Key Correction

Дата проверки: `2026-06-19`.

Bybit V5 allows one physical API key to cover multiple product buckets. Roehub
still keeps execution/readiness market-scoped by concrete `exchange_connection_id`,
so validation must prove the selected market rather than infer futures readiness
from an active spot row.

The Bybit validator now stores additive market-level evidence in
`permission_summary_json`:

- `bybit_permissions`: sanitized permission bucket names and values from
  `/v5/user/query-api`;
- `bybit_market_support.spot`: true only when `SpotTrade` is present;
- `bybit_market_support.futures`: true when the documented derivatives/contract
  trade bucket is present;
- `market_type`: the Roehub market row being validated.

If a key is writable but lacks the selected market permission, validation returns
`permission_mismatch` with `validation_reason=bybit_<market>_trade_permission_missing`.
`readOnly=1` for requested trade remains `permission_mismatch` with
`requested_trade_but_exchange_readonly`. Transfer permissions remain
`invalid_permissions`; mainnet without IP restrictions remains
`invalid_ip_restriction`; testnet without IP restrictions remains allowed but is
shown as `not_restricted_testnet`.

## External Validation Evidence

| Exchange | Scenario | Env vars | Expected status | Observed status | Blocker |
|---|---|---|---|---|---|
| Binance | Read-only live credential | `ROEHUB_EXCHANGE_VALIDATION_LIVE=1`, `ROEHUB_TEST_BINANCE_READONLY_API_KEY`, `ROEHUB_TEST_BINANCE_READONLY_API_SECRET` | `valid_readonly`. | Mac Studio host-local smoke returned `valid_readonly`; sanitized permission summary: reading enabled, IP restricted, trading/withdrawal/transfer flags disabled. | None |
| Bybit | Read-only live credential | `ROEHUB_EXCHANGE_VALIDATION_LIVE=1`, `ROEHUB_TEST_BYBIT_READONLY_API_KEY`, `ROEHUB_TEST_BYBIT_READONLY_API_SECRET` | `valid_readonly`. | Mac Studio host-local smoke returned `valid_readonly`; sanitized summary: `readOnly=true`, IP restricted, supported account mode. | None |
| Local/CI skip | Live flag absent | No live env vars | `skipped_external_validation`, not acceptance. | Implemented; internal validate test returns `skipped_external_validation` with `live_validation_disabled`. | None |

## Quality Gates

| Exchange | Scenario | Env vars | Expected status | Observed status | Blocker |
|---|---|---|---|---|---|
| Local tests | `uv run pytest -q tests/unit/contexts/exchange_control tests/unit/apps/api/test_ui_account_routes.py` | None | Pass. | Passed: `30 passed`. | None |
| Lint | `uv run ruff check src/trading/contexts/exchange_control apps/api tests/unit/contexts/exchange_control tests/unit/apps/api` | None | Pass. | Passed. | None |
| Type check | `uv run pyright src/trading/contexts/exchange_control apps/api tests/unit/contexts/exchange_control tests/unit/apps/api` | None | Pass. | Passed: `0 errors`. | None |
| No order path | `! rg -n "/order\|createOrder\|submit_order\|place_order" src/trading/contexts/exchange_control` | None | No matches. | Passed. | None |
| Full Stage 5 tests | `uv run pytest -q tests/unit/contexts/exchange_control tests/unit/apps/api/test_ui_account_routes.py tests/unit/apps/web/test_app_routes.py` | None | Pass. | Passed: `59 passed`, 3 pre-existing httpx cookie deprecation warnings. | None |
| Full Stage 5 lint | `uv run ruff check src/trading/contexts/exchange_control apps/api apps/web tests/unit/contexts/exchange_control tests/unit/apps/api tests/unit/apps/web` | None | Pass. | Passed. | None |
| Full Stage 5 type check | `uv run pyright src/trading/contexts/exchange_control apps/api tests/unit/contexts/exchange_control tests/unit/apps/api` | None | Pass. | Passed: `0 errors`. | None |
| Docs index | `python -m tools.docs.generate_docs_index --check` | None | Pass. | Passed after adding report and ledger updates. | None |
| Branch/auth | `test "$(git branch --show-current)" = main`; `gh --version && gh auth status` | None | Main branch and GitHub auth available. | Passed: branch `main`; GitHub CLI authenticated. | None |
| `apps/api` no-direct-import grep | `rg -n "ExchangeSecretCipher\|decrypt\|openbao\|vault\|binance\|bybit\|pybit\|api_secret\|passphrase" apps/api \|\| true` | None | No direct secret/decrypt/native exchange adapter imports; allowed request-forwarding literals documented. | No `ExchangeSecretCipher`, `decrypt`, `openbao`, `vault` or `pybit` imports. Matches are request DTO/client forwarding fields and existing `binance`/`bybit` literals. | None |
| Artifact secret grep | Sentinel-safe grep over `logs output .playwright-cli` for live secret env names plus `TEST_PASSPHRASE`, `api_secret`, `passphrase` | No live secret markers. | Passed with sentinel values; `logs` directory absent and no matches. | None |
| Live public validation curl | `curl -fsS -X POST "$ROEHUB_BASE_URL/api/ui/account/exchange-connections/$CONNECTION_ID/validate" ...` | `ROEHUB_BASE_URL`, session cookie, CSRF token, `CONNECTION_ID`, live exchange env vars | `valid_readonly` for readonly Binance and Bybit env-backed connections. | Public session/CSRF env is not available in the execution shell; route behavior is covered by deterministic facade tests, and live exchange acceptance is covered by host-local Mac Studio readonly metadata smoke. | None |
| Mac Studio direct credential smoke | Host-local env plus signed Binance/Bybit readonly metadata calls | No secret values printed; values loaded from `/Users/daniildegtyarev/.config/roehub/roehub.env` | Binance and Bybit both return `valid_readonly`. | Passed: Binance returned `valid_readonly` with reading enabled, IP restricted and dangerous permission flags disabled; Bybit returned `valid_readonly` with `readOnly=true`, IP restricted and supported account mode. | None |
| Stage 3C preflight | `curl -fsS http://127.0.0.1:9205/internal/v1/capabilities ... stage-5-preflight` | Internal token and running Mac Studio target service | Internal boundary reachable. | Passed on Mac Studio: `/health/ready` returned ready and capabilities returned `contract_version=internal-v1` with `exchange_connections.validate`. | None |
| Post-deploy internal validation | `POST /internal/v1/exchange-connections/{connection_id}/validate` after creating temporary smoke connections | Host-local internal token, Transit/OpenBao env, readonly Binance/Bybit env | Binance and Bybit return `valid_readonly`; temporary connections are disabled after smoke. | Passed on Mac Studio: Binance `valid_readonly`, `ip_restriction_status=restricted`; Bybit `valid_readonly`, `ip_restriction_status=restricted`; both smoke connections disabled. | None |
| Metrics | `curl -fsS http://127.0.0.1:9205/metrics | grep exchange_connection_validation_total` | None | Bounded metrics include exchange/result/reason only. | Passed: `exchange_connection_validation_total{exchange="binance",reason="readonly_permission_detected",result="valid_readonly"} 2.0`; Bybit same labels with value `1.0`. | None |

## Contract Impact Classification

| Dimension | Classification | Reason |
|---|---|---|
| Public API contract | `compatible-change` | Adds `POST /api/ui/account/exchange-connections/{connection_id}/validate` and additive validation fields. |
| Internal API/client contract | `compatible-change` | Adds `POST /internal/v1/exchange-connections/{connection_id}/validate` and `ExchangeControlClient.validate_connection`. |
| DTO schema | `compatible-change` | Adds validation status/reason/IP/last-validated fields to existing exchange connection response. |
| Persisted schema | `none` | Reuses Stage 4 `permission_summary_json`, `ip_restriction_status` and `last_validated_at`; no new migration. |
| Config schema | `compatible-change` | Adds explicit `ROEHUB_EXCHANGE_VALIDATION_LIVE`; default disabled returns skip status. |
| Metrics/audit | `compatible-change` | Uses existing bounded validation metrics and Stage 1 audit event names. |
| Request hash/cache identity | `none` | No cache or request-hash behavior changed. |

## Rollback Notes

| Exchange | Scenario | Env vars | Expected status | Observed status | Blocker |
|---|---|---|---|---|---|
| Feature disable | Unset `ROEHUB_EXCHANGE_VALIDATION_LIVE` | None | Runtime returns `skipped_external_validation`. | Implemented default behavior. | None |
| API rollback | Revert Stage 5 validate route/client/adapter commit | None | Create/list/rotate/disable from Stage 4 remain available. | Compatible because no schema migration was added. | None |
| Emergency validation stop | Keep service running, disable live flag | None | No external exchange calls; existing stored connections remain readable. | Implemented default behavior. | None |

## Stage 6 Handoff Facts

- Stage 6 can start after Stage 5 direct-main delivery and post-deploy runtime
  evidence are recorded.
- Public connection responses now include `validation_status`,
  `validation_reason`, `ip_restriction_status` and `last_validated_at`.
- `skipped_external_validation` is valid local/CI evidence only; product
  acceptance requires live env-backed validation.
- `apps/api` remains a facade and must not import native exchange adapters,
  Transit decrypt code or credential resolver code.
- No order placement implementation exists in `exchange-control`.

## Direct-Main Delivery Evidence

| Item | Evidence | Result |
|---|---|---|
| Implementation commit | `d526940579fb71cb3bc7fcd81f5b7c2e3b1518ce` | Pushed to `origin/main`. |
| Runtime persistence fix | `ab8f28e9bec258776fd36d74e48c08a80135b4fe` | Pushed to `origin/main`; fixes Postgres `jsonb_build_object` nullable parameter casts discovered during post-deploy smoke. |
| CI | GitHub Actions `26367259189` and `26367582869` | Success. |
| Deploy | Deploy Backend `26367298732`, `26367658704`; Publish App Image `26367298740`, `26367658694`; Deploy Web `26367301918`, `26367663544` | Success. |
| Runtime | Mac Studio `exchange-control` restarted after fix deploy; `/health/ready` ready; OpenBao unsealed; Transit ACL smoke passed; internal validate smoke passed. | Accepted. |
