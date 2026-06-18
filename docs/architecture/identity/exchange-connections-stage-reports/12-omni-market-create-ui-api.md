# Stage 12: Omni-market create UI/API

Статус: `implemented-local`; production runtime evidence pending direct-main delivery.

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

## Required Production Evidence Before Acceptance

Stage 12 is not accepted until direct-main delivery plus runtime evidence prove:

1. Authenticated `/settings` opens and displays market checkboxes plus visible
   Mainnet/Testnet segmented control without layout overlap.
2. Browser/network create payload includes `market_type` and `market_types[]`
   when both markets are checked.
3. API response for a multi-market create returns `items[]` and
   `market_results[]`, and no plaintext secret/ciphertext/HMAC.
4. Active list shows market-scoped rows, not a universal execution connection.
5. Runtime smoke, exchange-control health, Prometheus/Monit/OpenBao checks remain
   green after deploy.

## Handoff

The next credential-management stage should introduce a shared credential object
only after a separate migration plan covers rotation, duplicate fingerprints,
rollback, and how existing `exchange_credential_versions.connection_id` rows map
to the new credential identity. That stage must not weaken market-scoped
execution guards.
