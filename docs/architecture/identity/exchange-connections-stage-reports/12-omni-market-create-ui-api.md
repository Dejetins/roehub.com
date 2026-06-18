# Stage 12: Omni-market create UI/API

Статус: `accepted`.

Direct-main delivery, CI/deploy and production browser/runtime evidence are complete.

## Scope

Stage 12 добавляет совместимый первый этап omni-channel ключа:

- public `/api/ui/account/exchange-connections` сохраняет legacy `market_type`;
- request может дополнительно передать `market_types: ["spot", "futures"]`;
- один plaintext API key/secret внутри create request валидируется по каждому
  выбранному market;
- результатом остаются отдельные market-scoped `exchange_connections`;
- execution/readiness/strategy binding продолжают работать только с конкретным
  `exchange_connection_id`.

Второй этап, который выносит secret material из
`exchange_credential_versions.connection_id` в отдельный credential object, не
входит в Stage 12. На этом этапе ciphertext может быть продублирован для
нескольких market-scoped rows.

## Implementation

| Area | Change |
|---|---|
| API DTO | `CreateExchangeConnectionRequest.market_types` добавлен как optional `spot|futures` list. |
| API response | Create response сохраняет top-level connection fields и добавляет optional `items[]` + `market_results[]` для multi-market create. |
| API route | `apps/api` fan-out вызывает существующий `ExchangeControlClient.create_connection` отдельно для каждого market. |
| Secret boundary | Plaintext используется только в текущем create request; response не содержит secret/ciphertext/HMAC. |
| UI | `/settings` add-key form заменяет single-market dropdown на `Spot`/`Futures` checkboxes. |
| UI environment | Mainnet/Testnet теперь видимый segmented control, без скрытого `details` advanced block. |
| UI validation result | После multi-market create показывается per-market validation/readiness result. |
| Compatibility | Старый single-market request/response остается рабочим; legacy `permissions` default остается `read`, но `/settings` отправляет `trade` как product intent. |

## Contract Impact

| Dimension | Classification | Note |
|---|---|---|
| Public API | `compatible-change` | Adds optional `market_types[]`, optional response `items[]` and `market_results[]`; does not remove `market_type` or top-level connection fields. |
| DTO schema | `compatible-change` | Additive fields only. Invalid `linear`/`inverse` remain rejected. |
| Persistence | `compatible-change` | No schema migration. Multi-market create creates existing market-scoped rows. |
| Port contract | `none` | `ExchangeControlClient.create_connection` remains single-market; fan-out lives in account facade. |
| Execution/readiness | `none` | Execution still receives a concrete market-scoped `exchange_connection_id`. |
| Secret boundary | `compatible-change` | No new secret-returning field. Stage 12 still duplicates ciphertext when the same physical key is used for multiple markets. |
| Browser-visible behavior | `workflow change` | Add-key form uses market checkboxes and visible Mainnet/Testnet control; table stays market-scoped. |
| Logs/metrics/audit | `compatible-change` | Existing auto-validation audit records are written per created market row; no secret labels. |

## Local Evidence

| Check | Result |
|---|---|
| API/web focused unit tests | `uv run pytest -q tests/unit/apps/api/test_ui_account_routes.py tests/unit/apps/web/test_app_routes.py` passed (`57 passed`, warnings only from existing httpx cookie deprecation). |
| Python syntax | `python -m py_compile apps/api/dto/ui_account.py apps/api/routes/ui_account.py` passed. |
| JS syntax | `node --check apps/web/dist/js/pages/settings.js` passed. |
| Ruff | `uv run ruff check apps/api/dto/ui_account.py apps/api/routes/ui_account.py apps/web/dist/js/pages/settings.js tests/unit/apps/api/test_ui_account_routes.py tests/unit/apps/web/test_app_routes.py` passed. |
| Pyright | `uv run pyright apps/api tests/unit/apps/api` passed. |
| Docs index | `python -m tools.docs.generate_docs_index --check` passed. |
| Local browser QA | `npx --package=@playwright/test playwright test --config=output/playwright/playwright.stage12.config.js` passed against a local web app with mock API transport. Evidence file `output/playwright/settings-stage12-omni-market-create-local.json` records visible Spot/Futures controls, `market_type=spot`, `market_types=[spot,futures]`, `permissions=trade`, response `items=[spot,futures]`, per-market readiness `ready_for_trading`, no console errors, and no secret/ciphertext/HMAC in the response. Screenshot: `output/playwright/settings-stage12-omni-market-create-local.png`. |
| Secret artifact grep | `rg -n "BROWSER_SECRET_VALUE|BROWSERKEY1234|ciphertext|hmac" output/playwright/settings-stage12-omni-market-create-local.*` returned no matches after sanitizing the QA payload artifact. |

## Production Evidence

| Check | Result |
|---|---|
| Direct-main delivery | Implementation commit `5696e21240ee6a65832e221d6fd36802777a4f05` pushed to `main`. |
| CI | GitHub Actions CI run `27728371649` succeeded for `5696e21240ee6a65832e221d6fd36802777a4f05`. |
| Deploy Backend | Run `27728426127` succeeded. The deploy job synced backend source to `/opt/roehub/app`, built the Python runtime, ran DB bootstrap/migrations, reloaded production launchd services, recovered OpenBao Transit, and ran backend smoke. |
| Deploy Web | Runs `27728426087` and `27728460775` succeeded for the shipped SHA. |
| Publish App Image | Run `27728426106` succeeded for the shipped SHA. |
| Mac Studio runtime via deploy runner | Deploy Backend `27728426127` proved `exchange-control`, `exchange-execution`, `strategy-live-runner`, `api`, OpenBao and exporters were bootstrapped by launchd; OpenBao recovery reported `openbao_health=ok`, `exchange_control_encrypt=ok`, `apps_api_decrypt_denied=403`, `openbao_recovery=ok`; backend smoke returned expected unauthenticated `401` for `/auth/current-user` and required metrics ports passed. |
| Authenticated production browser smoke | `output/playwright/settings-stage12-prod-auth.json` passed against `https://roehub.com/settings#api` using the smoke Keycloak account. It verified visible checked `Spot` and `Futures` checkboxes, visible Mainnet/Testnet segmented control, Mainnet default selected, Testnet toggles selected, Mainnet toggles back selected, `consoleErrorCount=0`, `requestFailureCount=0`. No exchange key was created in this smoke. |
| Secret artifact grep | `rg -n "SmokeE2E|password|api_secret|BROWSER_SECRET|KEYCLOAK|roehub_session|AUTH_SESSION|code=|state=" output/playwright` returned no matches. |
| Screenshot limitation | Production headless screenshot capture timed out after fonts loaded, so the accepted production browser artifact is JSON state evidence rather than a PNG. Local browser QA did capture the add-key modal screenshot. |

## Follow-up: Market Availability Display

Дата проверки: `2026-06-18`.

Production API evidence for `smoke_e2e_keycloak` showed active `binance_testnet`
only as a futures-scoped connection: `exchange=binance`, `label=binance_testnet`,
masked key `****RcSh`, `market_type=futures`, `environment=testnet`,
`validation_status=valid_trade_enabled`, `connection_readiness=ready_for_trading`.
`status=all` also showed no Binance testnet spot row for that same label/masked
key; there is one active futures row and one disabled futures duplicate.

The UI follow-up keeps the market-scoped execution model unchanged and changes
only the `/settings` table display: the former single `Market` value is now a
`Markets` availability cell with separate `Spot` and `Futures` lines for the
same display-safe `(exchange, account/name, masked API key, environment)` group.
The current row market is highlighted so row-level actions still target a single
market-scoped binding. Missing market rows render as `Not connected`; ready rows
render as compact `Ready` because the full readiness remains in the row `Status`
column. This makes the current fact explicit:
`binance_testnet` is ready for futures in Roehub, while spot still needs its own
market-scoped connection row before execution/readiness can use it.

## Handoff

The next credential-management stage should introduce a shared credential object
only after a separate migration plan covers rotation, duplicate fingerprints,
rollback, and how existing `exchange_credential_versions.connection_id` rows map
to the new credential identity. That stage must not weaken market-scoped
execution guards.
