# Stage 04: BTCUSDT Market Readiness

Статус: `accepted`

## Pre-Start

User required before start: nothing

Stage `03` проверен в ledger: `accepted`; следующий stage разрешен.

## Scope

Только `BTCUSDT`. Non-`BTCUSDT` artifacts не проверяются и не provision-ятся в этом stage.

## Concrete Planned File List Before Editing

| File | Planned action | Reason |
|---|---:|---|
| `src/trading/contexts/market_data/application/dto/reference_api.py` | modify | Add BTCUSDT readiness read-model DTOs. |
| `src/trading/contexts/market_data/application/ports/stores/btcusdt_market_readiness_reader.py` | create | Add storage port for BTCUSDT market/reference readiness matrix. |
| `src/trading/contexts/market_data/application/ports/stores/__init__.py` | modify | Export new port. |
| `src/trading/contexts/market_data/application/use_cases/btcusdt_market_readiness.py` | create | Compose reference readiness with Redis stream freshness. |
| `src/trading/contexts/market_data/application/use_cases/__init__.py` | modify | Export new use-case and constants. |
| `src/trading/contexts/market_data/adapters/outbound/persistence/clickhouse/btcusdt_market_readiness_reader.py` | create | Read latest `BTCUSDT` reference rows from ClickHouse. |
| `src/trading/contexts/market_data/adapters/outbound/persistence/clickhouse/__init__.py` | modify | Export new ClickHouse adapter. |
| `apps/api/dto/market_data_reference.py` | modify | Add API response payload for readiness matrix. |
| `apps/api/dto/__init__.py` | modify | Export new API DTOs where needed by routes. |
| `apps/api/routes/market_data_reference.py` | modify | Add auth-only BTCUSDT readiness endpoint. |
| `apps/api/wiring/modules/market_data_reference.py` | modify | Wire ClickHouse + Redis readiness dependencies. |
| `apps/api/dto/ui_strategies_dashboard.py` | modify | Add `/strategies` dashboard market-readiness panel response. |
| `apps/api/wiring/modules/ui_strategies_dashboard.py` | modify | Load BTCUSDT readiness for `/ui/strategies/dashboard`. |
| `apps/web/templates/pages/strategies.html` | modify | Render market readiness on `/strategies`. |
| `apps/web/dist/js/pages/strategies.js` | modify | Populate BTCUSDT market readiness fields. |
| `apps/web/dist/css/pages/strategies.css` | modify | Style market readiness state consistently with existing panels. |
| `apps/web/locales/en.json` | modify | Add English labels for market readiness panel. |
| `apps/web/locales/ru.json` | modify | Add Russian labels for market readiness panel. |
| `src/trading/contexts/market_data/adapters/outbound/clients/rest_instrument_metadata_source.py` | modify if root-cause evidence requires it | Preserve exchange metadata fields needed for BTCUSDT precision/min-notional readiness. |
| `configs/dev/whitelist.csv` | modify if root-cause evidence requires it | Add only missing `BTCUSDT` market_id `4` if Bybit futures is absent from whitelist. |
| `configs/prod/whitelist.csv` | modify if root-cause evidence requires it | Add only missing `BTCUSDT` market_id `4` if Bybit futures is absent from whitelist. |
| `tests/unit/contexts/market_data/application/use_cases/test_btcusdt_market_readiness.py` | create | Cover readiness state matrix logic. |
| `tests/unit/contexts/market_data/adapters/test_clickhouse_btcusdt_market_readiness_reader.py` | create | Cover ClickHouse query/row mapping. |
| `tests/unit/contexts/market_data/adapters/test_rest_instrument_metadata_source.py` | modify if root-cause evidence requires it | Cover exchange metadata parser fallbacks. |
| `tests/unit/apps/api/test_market_data_reference_routes.py` | modify | Cover new auth-only API endpoint. |
| `tests/unit/apps/api/test_ui_strategy_dashboard_routes.py` | modify | Cover dashboard payload includes market readiness. |
| `tests/unit/apps/web/test_app_routes.py` | modify | Guard browser asset/template wiring. |
| `docs/architecture/live_execution/strategy-producer-paper-testnet-trading-v1-stage-reports/04-btcusdt-market-readiness.md` | create | Stage evidence report. |
| `docs/architecture/live_execution/strategy-producer-paper-testnet-trading-v1-stage-reports/strategy-producer-paper-testnet-trading-v1-stage-ledger.md` | modify | Stage ledger handoff. |
| `docs/architecture/README.md` | modify if generated index requires it | Docs index for new stage report; existing dirty ML entries are pre-existing and not part of this stage. |

## Freshness Thresholds And Keys

| Surface | Threshold / key |
|---|---|
| Redis stream prefix | `md.candles.1m` |
| BTCUSDT stream keys | `md.candles.1m.{exchange}:{market_type}:BTCUSDT` |
| Stream freshness threshold | `180s` default stale threshold from `RedisMarketDataReadinessReader` unless runtime config overrides later. |
| Reference table | ClickHouse `market_data.ref_market` + `market_data.ref_instruments` latest rows. |
| Precision/min-notional readiness | `price_step`, `qty_step`, and `min_notional` must be present and positive for a market row to be ready. |

## File Manifest

| Created | Modified | Deleted | Reason | Contract impact |
|---|---|---|---|---|
| `src/trading/contexts/market_data/application/ports/stores/btcusdt_market_readiness_reader.py`; `src/trading/contexts/market_data/application/use_cases/btcusdt_market_readiness.py`; `src/trading/contexts/market_data/adapters/outbound/persistence/clickhouse/btcusdt_market_readiness_reader.py`; `tests/unit/contexts/market_data/application/use_cases/test_btcusdt_market_readiness.py`; `tests/unit/contexts/market_data/adapters/test_clickhouse_btcusdt_market_readiness_reader.py`; this report | `src/trading/contexts/market_data/application/dto/reference_api.py`; `src/trading/contexts/market_data/application/ports/stores/__init__.py`; `src/trading/contexts/market_data/application/use_cases/__init__.py`; `src/trading/contexts/market_data/adapters/outbound/persistence/clickhouse/__init__.py`; `apps/api/dto/market_data_reference.py`; `apps/api/dto/__init__.py`; `apps/api/routes/market_data_reference.py`; `apps/api/wiring/modules/market_data_reference.py`; `apps/api/dto/ui_strategies_dashboard.py`; `apps/api/wiring/modules/ui_strategies_dashboard.py`; `apps/web/templates/pages/strategies.html`; `apps/web/dist/js/pages/strategies.js`; `apps/web/dist/css/pages/strategies.css`; `apps/web/locales/en.json`; `apps/web/locales/ru.json`; `tests/unit/apps/api/test_market_data_reference_routes.py`; `tests/unit/apps/api/test_ui_strategy_dashboard_routes.py`; `tests/unit/apps/web/test_app_routes.py`; `docs/architecture/README.md` Stage 04 index hunk only | none | Add explicit BTCUSDT market-readiness API/UI matrix with Redis stream freshness and ClickHouse reference precision/min-notional evidence. | compatible-change: additive endpoint and additive dashboard response field/panel. |
| none | `src/trading/contexts/market_data/adapters/outbound/clients/rest_instrument_metadata_source.py`; `tests/unit/contexts/market_data/adapters/test_rest_instrument_metadata_source.py` | none | Root-cause fix for missing BTCUSDT precision/min-notional fields from current Binance futures and Bybit spot payload shapes. | compatible-change: preserves additional exchange metadata fields; no schema break. |
| none | `configs/dev/whitelist.csv`; `configs/prod/whitelist.csv` | none | Add missing Bybit futures `BTCUSDT` whitelist row (`market_id=4`) so existing sync/enrich/live-feed jobs can create/prove the required artifact. | compatible-change: BTCUSDT-only allowlist expansion for an already configured market. |

Files outside prompt expected paths: `configs/dev/whitelist.csv` and `configs/prod/whitelist.csv` are touched because runtime evidence proved Bybit futures `BTCUSDT` was missing from the whitelist while the plan requires Binance/Bybit spot/futures readiness for BTCUSDT only. `src/trading/contexts/market_data/adapters/outbound/clients/rest_instrument_metadata_source.py` is touched because current exchange payloads contained the missing BTCUSDT readiness fields under already-supported provider schemas.

## BTCUSDT Readiness Matrix

Baseline target-runtime evidence before delivery:

| Exchange | Market | Redis stream | Reference row | Precision/min-notional | Stage status |
|---|---|---|---|---|---|
| Binance | spot | exists, length `10082`, last id `1781652420000-0` | `ENABLED`, tradable | `price_step=0.01`, `qty_step=0.00001`, `min_notional=5` | ready baseline |
| Binance | futures | exists, length `10083`, last id `1781652420000-0` | `ENABLED`, tradable | `price_step=0.1`, `qty_step=0.001`, `min_notional=NULL` before fix | fixed parser fallback for `MIN_NOTIONAL.notional`; needs post-deploy sync/enrich proof |
| Bybit | spot | exists, length `10083`, last id `1781652420000-0` | `ENABLED`, tradable | `price_step=0.1`, `qty_step=NULL`, `min_notional=5` before fix | fixed parser fallback for `lotSizeFilter.basePrecision`; needs post-deploy sync/enrich proof |
| Bybit | futures | `ERR no such key` before fix | market exists, no BTCUSDT instrument row | missing before fix | fixed whitelist row; needs post-deploy sync/enrich/live-feed proof |

Root-cause hypothesis confirmed:

| Symptom | Root cause | Evidence | Fix |
|---|---|---|---|
| Binance futures `min_notional=NULL` | Binance USD-M futures `MIN_NOTIONAL` filter uses `notional`, while parser only checked `minNotional`. | Live `https://fapi.binance.com/fapi/v1/exchangeInfo` BTCUSDT payload contained `{"filterType":"MIN_NOTIONAL","notional":"50"}`. | Parser accepts `MIN_NOTIONAL.notional` and `NOTIONAL.notional`. |
| Bybit spot `qty_step=NULL` | Bybit spot instrument payload uses `lotSizeFilter.basePrecision`, while parser only checked `qtyStep`. | Live `https://api.bybit.com/v5/market/instruments-info?category=spot&symbol=BTCUSDT` payload contained `basePrecision=0.000001` and no `qtyStep`. | Parser falls back to `basePrecision`. |
| Bybit futures instrument/stream missing | Prod/dev whitelist lacked `4,BTCUSDT,1`, so sync/live-feed had no enabled Bybit futures BTCUSDT instrument to publish. | `configs/prod/whitelist.csv` and `configs/dev/whitelist.csv` had BTCUSDT rows for `1/2/3`, while `configs/test/whitelist.csv` already had `4,BTCUSDT,1`; ClickHouse and Redis lacked Bybit futures BTCUSDT. | Add `4,BTCUSDT,1` to dev/prod whitelist. |

## Evidence

### Local gates

| Command | Result |
|---|---|
| `uv run ruff check src/trading/contexts/market_data src/trading/contexts/strategy apps tests` | passed |
| `uv run pyright src/trading/contexts/market_data src/trading/contexts/strategy apps tests` | passed, `0 errors` |
| `uv run pytest -q tests/unit/contexts/market_data tests/unit/contexts/strategy tests/unit/apps` | passed, `501 passed, 3 warnings` |
| `python -m tools.docs.generate_docs_index --check` | passed |
| focused tests for readiness/API/dashboard/parser | passed before broad gates |

### Target-runtime baseline

| Surface | Command / call | Result |
|---|---|---|
| Redis streams | `XINFO STREAM md.candles.1m.binance:spot:BTCUSDT`, `binance:futures`, `bybit:spot`, `bybit:futures` on Mac Studio Redis `127.0.0.1:6379` | first three streams existed with lengths around `10082-10083`; `bybit:futures` returned `ERR no such key` before fix. |
| Reference rows | ClickHouse `market_data.ref_market` + `market_data.ref_instruments` latest BTCUSDT query | rows existed for Binance spot/futures and Bybit spot; Bybit futures market existed but BTCUSDT instrument fields were empty; Binance futures `min_notional` and Bybit spot `qty_step` were missing before fix. |
| API pre-deploy | `GET http://127.0.0.1:8000/api/market-data/btcusdt-readiness` | `404`, expected before deploying the new endpoint. |

### Delivery / post-deploy

| Surface | Evidence | Result |
|---|---|---|
| Main delivery | Direct `main` commit `39ff9f8d` (`Add BTCUSDT market readiness`) pushed to `origin/main`; `git ls-remote origin refs/heads/main` returned `39ff9f8d89cedb6b08e1482c18a62ed9d595b66d`. | pass |
| CI | GitHub Actions CI run `27655281690` for `39ff9f8d` completed successfully. | pass |
| Deploy Backend | Run `27655488294` completed successfully; backend deploy smoke step passed. | pass |
| Deploy Web | Run `27655488312` completed successfully. | pass |
| Publish App Image | Run `27655488320` completed successfully; Docker cache reservation warning was non-fatal. | pass |
| Mac Studio host sync | `/Users/daniildegtyarev/Projects/roehub.com` fast-forwarded to `39ff9f8d`; `## main...origin/main`. | pass |
| Mac Studio smoke | `/opt/roehub/app/scripts/macos/smoke_prod.sh` exited `0`: API, Redis `PONG`, launchd services and Tailscale backend were healthy. | pass |
| Runtime sync/enrich | One-shot scheduler sync/enrich on `/opt/roehub/app` completed: `sync_and_enrich_ok`; then `com.roehub.market-data-scheduler` and `com.roehub.market-data-ws-worker` were restarted and running. | pass |
| Active smoke sessions | Temporary browser/API smoke sessions revoked; active smoke-session count `0`. | pass |

Post-deploy ClickHouse latest BTCUSDT reference rows:

| Market | Status | Tradable | Base/quote | price_step | qty_step | min_notional | updated_at |
|---|---:|---:|---|---:|---:|---:|---|
| Binance spot (`1`) | `ENABLED` | `1` | `BTC/USDT` | `0.01` | `0.00001` | `5.0` | `2026-03-17 22:19:33.328000` |
| Binance futures (`2`) | `ENABLED` | `1` | `BTC/USDT` | `0.1` | `0.001` | `50.0` | `2026-06-16 23:41:52.706000` |
| Bybit spot (`3`) | `ENABLED` | `1` | `BTC/USDT` | `0.1` | `0.000001` | `5.0` | `2026-06-16 23:41:52.706000` |
| Bybit futures (`4`) | `ENABLED` | `1` | `BTC/USDT` | `0.1` | `0.001` | `5.0` | `2026-06-16 23:41:52.706000` |

Post-deploy Redis stream proof after worker restart:

| Stream | Length | Last id | Last observed candle |
|---|---:|---|---|
| `md.candles.1m.binance:spot:BTCUSDT` | `10082` | `1781653440000-0` | `2026-06-16T23:44:00Z` / ingested `2026-06-16T23:45:00.130Z` |
| `md.candles.1m.binance:futures:BTCUSDT` | `10083` | `1781653440000-0` | `2026-06-16T23:44:00Z` / ingested `2026-06-16T23:45:00.282Z` |
| `md.candles.1m.bybit:spot:BTCUSDT` | `10083` | `1781653440000-0` | `2026-06-16T23:44:00Z` / ingested `2026-06-16T23:45:00.649Z` |
| `md.candles.1m.bybit:futures:BTCUSDT` | `2` | `1781653440000-0` | `2026-06-16T23:44:00Z` / ingested `2026-06-16T23:45:00.839Z` |

Authenticated production API proof:

- `GET https://roehub.com/api/market-data/btcusdt-readiness` with temporary smoke session returned four `items`, all with `readiness_state=ready`, `reference_state=ready`, and `stream_state=ready`.
- Artifact: `output/playwright/stage04-btcusdt-api-readiness.json`.

Authenticated browser proof:

- Playwright against `https://roehub.com/strategies` with a temporary smoke session rendered `.strategies-market-readiness` with `data-readiness="ready"`, visible `BTCUSDT`, `Binance`, `Bybit`, and `ready` text.
- Network log included `GET https://roehub.com/api/ui/strategies/dashboard?refresh=initial&state=all => 200` and refresh `200`.
- Screenshot: `output/playwright/stage04-btcusdt-market-readiness.png`.
- DOM/network artifact: `output/playwright/stage04-btcusdt-market-readiness.json`, `output/playwright/stage04-btcusdt-market-readiness.network.txt`.

## Blockers

None. The Stage `04` blocker was resolved by production sync/enrich, market-data worker restart, fresh Bybit futures BTCUSDT stream proof, authenticated API readiness proof, and authenticated browser proof.

## Handoff

Stage `05` can rely on accepted BTCUSDT readiness for Binance/Bybit spot/futures:

- All four required BTCUSDT streams are present and fresh under `md.candles.1m.*`.
- ClickHouse reference rows have enabled/tradable status plus positive `price_step`, `qty_step`, and `min_notional`.
- Authenticated API/UI surfaces expose the readiness matrix and fail closed through explicit readiness states if a dependency degrades.
- Non-BTCUSDT market readiness remains out of scope for this stage.
