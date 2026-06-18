---
doc: rl-trading-agent-platform-v1-stage-04a-binance-futures-universe-whitelist
stage: "04A"
status: accepted
plan: docs/architecture/ml/rl-trading-agent-platform-v1.md
ledger: docs/architecture/ml/rl-trading-agent-platform-v1-stage-reports/rl-trading-agent-platform-v1-stage-ledger.md
collected_at: "2026-06-19"
---

# Stage 04A: Binance Futures Universe And Whitelist

Статус: `accepted`.

User required before start: nothing unless a listed prerequisite is not accepted or a required credential/dataset/runtime source is unavailable; never ask for secrets in chat.

Stage `04A` зафиксировал текущую train-compatible вселенную Binance USD-M Futures для RL v1: стартовая выборка взята из observed HF `train_data.npz` (`309` unique train symbols), затем пересечена с публичным Binance Futures `exchangeInfo` по правилам `status=TRADING`, `contractType=PERPETUAL`, `quoteAsset=USDT`. Свечи не backfilled в этом stage.

## Scope

Входит:

- проверить prerequisite Stage `04`;
- записать prompt path/hash и конкретный planned file list до правок;
- добавить deterministic resolver для Stage `04A`;
- обновить только `binance:futures` rows в `configs/prod/whitelist.csv`;
- создать sanitized universe/exclusion/source-window manifest на Mac Studio;
- выполнить whitelist -> `market_data.ref_instruments` sync и Binance Futures-only enrichment на Mac Studio;
- обновить stage report и ledger.

Не входит:

- candle backfill, feature slabs, sessionized datasets или model training;
- Binance spot, Bybit spot, Bybit futures training scope;
- synthetic mapping для delisted/renamed/non-USDT/quarterly symbols;
- API/UI/schema migration, exchange private/account endpoints, paper/testnet/live/mainnet submit;
- GitHub branch/PR/main delivery.

## Методология Анализа

| Поле | Значение |
|---|---|
| Уровень глубины | `стандартный анализ`, deterministic universe/data-quality reconciliation. |
| Тип задачи | Сверка HF train symbols с текущей биржевой metadata и операторское onboarding в reference data. |
| Выбранная методология | Source inventory reconciliation + fail-closed exclusion classification + runtime reference-data verification. |
| Простое объяснение метода | Берем только train-split symbols, оставляем только текущие Binance USD-M Futures USDT perpetual contracts, все остальные symbols не repair-им и не backfill-им. |
| Бизнес-язык | Stage защищает будущую загрузку данных от попыток качать несуществующие, delisted, не-USDT или quarterly контракты, чтобы следующий stage работал по точному списку. |
| Единица анализа | `binance:futures` symbol. |
| Основные метрики | HF candidate count, Binance metadata count, accepted/excluded counts, whitelist added rows, ref/enrichment coverage counts and hashes. |
| Прокси-метрики | `exchangeInfo.onboardDate` как lower-bound source-window policy до отдельной lifecycle table. |
| Период анализа | HF train-compatible required source window starts at `2020-01-13T22:30:00Z`; per-symbol lower bound is max of that timestamp and exchange onboard time. |
| Основные фильтры | `status=TRADING`, `contractType=PERPETUAL`, `quoteAsset=USDT`. |
| Риски интерпретации | Binance metadata is current-state metadata, not point-in-time historical lifecycle. Delisted symbols stay excluded for v1 refresh. |
| Проверки перед выводом | Focused unit tests, ruff, pyright, public Binance Futures metadata fetch, Mac Studio ClickHouse sync/enrichment evidence, idempotent sync rerun. |
| Вопросы до расчетов | Нет; Stage `04` accepted, HF NPZ existed on Mac Studio, public metadata required no private credentials. |

## File Manifest

Planned concrete file list before implementation edits:

- `configs/prod/whitelist.csv`
- `scripts/rl_trading/resolve_stage04a_binance_futures_universe.py`
- `tests/unit/scripts/rl_trading/test_resolve_stage04a_binance_futures_universe.py`
- `docs/architecture/ml/rl-trading-agent-platform-v1-stage-reports/04a-binance-futures-universe-whitelist.md`
- `docs/architecture/ml/rl-trading-agent-platform-v1-stage-reports/rl-trading-agent-platform-v1-stage-ledger.md`
- `docs/architecture/README.md` only if docs index regeneration is required

| Created | Modified | Deleted | Reason | Contract impact |
|---|---|---|---|---|
| `scripts/rl_trading/resolve_stage04a_binance_futures_universe.py` | - | - | Deterministic Stage `04A` resolver, whitelist append helper, sanitized manifest renderer. | `compatible-change` opt-in operator script |
| `tests/unit/scripts/rl_trading/test_resolve_stage04a_binance_futures_universe.py` | - | - | Focused coverage for exchange filter, source lower bound, sanitized symbol list and whitelist append semantics. | `compatible-change` test-only |
| `docs/architecture/ml/rl-trading-agent-platform-v1-stage-reports/04a-binance-futures-universe-whitelist.md` | - | - | Stage `04A` report. | `compatible-change` docs/report only |
| - | `configs/prod/whitelist.csv` | - | Add accepted `binance:futures` symbols for RL v1 current-trading universe; preserve unrelated rows and disabled `PEPEUSDT`. | `compatible-change` production config contents; schema unchanged |
| - | `docs/architecture/ml/rl-trading-agent-platform-v1-stage-reports/rl-trading-agent-platform-v1-stage-ledger.md` | - | Record Stage `04A` acceptance and Stage `04B` handoff. | `compatible-change` docs/ledger only |
| - | `docs/architecture/README.md` | - | Generated docs index only if required by docs check. | `compatible-change` docs index only |

Outside expected paths: none.

## Prompt Evidence

| Field | Value |
|---|---|
| Prompt path | `.codex/agents/generated/rl-trading-agent-platform-v1/04a-binance-futures-universe-whitelist.md` |
| Prompt sha256 | `80cbd412dc8d66191eab16dc9926f9303ee4923028cea7b8ae5796f6325cb3a1` |
| Ledger state before implementation | Stage `04` accepted; `current_stage=04A`; Stage `04A` pending |
| Required prerequisite | Stage `04` accepted |
| Delivery state | Repository changes are `local-only`; no branch, PR, merge, main delivery or app deploy. Mac Studio `market_data.ref_instruments` was onboarded from a staged whitelist artifact under `/opt/roehub/state/rl_trading/`. |
| Large/runtime artifacts | Runtime summaries live under `/opt/roehub/state/rl_trading/stage04a_binance_futures_universe/`; no raw provider payloads, raw NPZ arrays, secrets or credentials are committed. |

## Universe Evidence

Public metadata source: Binance USD-M Futures public `GET /fapi/v1/exchangeInfo` via the existing configured `binance:futures` base URL. Private/account endpoints were not used.

| Metric | Value |
|---|---:|
| HF train candidate symbols | `309` |
| Binance Futures metadata rows | `793` |
| Current `TRADING` + `PERPETUAL` + `USDT` rows | `528` |
| Accepted train-compatible symbols | `215` |
| Excluded symbols | `94` |
| Whitelist rows added for `market_id=2` | `209` |

Hashes:

| Artifact / list | sha256 |
|---|---|
| Accepted symbols list | `114b0314a062415cef95fdabffe84b5ff5f2e4d0333f574bb356ef71bc4223fd` |
| Excluded symbols list | `4780f83bffacbc0993add8abf7e564e9bbea28f5ebca44a8c8f9cc1cf2416524` |
| Current Binance Futures USDT perpetual symbol list | `a921fb1faea04411d3a07f596dafb804ba8be85ec7a8686e92dce37a30bce85c` |
| Mac Studio universe manifest | `70bba53e5eed3d04084b2e8104f03dc4d28b5fd7b61f8b5ff77919258b38cd14` |

Runtime manifest:

`/opt/roehub/state/rl_trading/stage04a_binance_futures_universe/stage04a_universe_manifest.json`

This manifest contains the full accepted list, full excluded list with reason, and per-symbol Stage `04B` source lower bound policy:

```text
source_lower_bound_utc = max(2020-01-13T22:30:00Z, exchangeInfo.onboardDate)
```

Accepted symbols:

```text
1000BONKUSDT
1000FLOKIUSDT
1000LUNCUSDT
1000PEPEUSDT
1000RATSUSDT
1000SHIBUSDT
1000XECUSDT
1INCHUSDT
AAVEUSDT
ACEUSDT
ACHUSDT
ADAUSDT
AEVOUSDT
AGLDUSDT
ALGOUSDT
ALICEUSDT
ALTUSDT
ANKRUSDT
APEUSDT
API3USDT
APTUSDT
ARBUSDT
ARKMUSDT
ARKUSDT
ARPAUSDT
ARUSDT
ASTRUSDT
ATOMUSDT
AUCTIONUSDT
AVAXUSDT
AXLUSDT
AXSUSDT
BANANAUSDT
BANDUSDT
BATUSDT
BBUSDT
BCHUSDT
BEAMXUSDT
BELUSDT
BICOUSDT
BIGTIMEUSDT
BLURUSDT
BNBUSDT
BNTUSDT
BOMEUSDT
BSVUSDT
BTCDOMUSDT
BTCUSDT
C98USDT
CAKEUSDT
CELOUSDT
CELRUSDT
CFXUSDT
CHRUSDT
CHZUSDT
CKBUSDT
COMPUSDT
COTIUSDT
CRVUSDT
CTKUSDT
CTSIUSDT
CVXUSDT
CYBERUSDT
DASHUSDT
DODOXUSDT
DOGEUSDT
DOTUSDT
DUSKUSDT
DYDXUSDT
DYMUSDT
EDUUSDT
EGLDUSDT
ENAUSDT
ENJUSDT
ENSUSDT
ETCUSDT
ETHFIUSDT
ETHUSDT
ETHWUSDT
FETUSDT
FILUSDT
FLOWUSDT
GALAUSDT
GASUSDT
GLMUSDT
GMTUSDT
GMXUSDT
GRTUSDT
GTCUSDT
GUSDT
HBARUSDT
HFTUSDT
HOTUSDT
ICPUSDT
ICXUSDT
IDUSDT
ILVUSDT
IMXUSDT
INJUSDT
IOSTUSDT
IOTAUSDT
IOTXUSDT
IOUSDT
JASMYUSDT
JOEUSDT
JTOUSDT
JUPUSDT
KASUSDT
KAVAUSDT
KNCUSDT
KSMUSDT
LDOUSDT
LINKUSDT
LISTAUSDT
LITUSDT
LPTUSDT
LQTYUSDT
LSKUSDT
LTCUSDT
LUNA2USDT
MAGICUSDT
MANAUSDT
MANTAUSDT
MASKUSDT
MAVUSDT
MEMEUSDT
METISUSDT
MINAUSDT
MOVRUSDT
MTLUSDT
NEARUSDT
NEOUSDT
NFPUSDT
NMRUSDT
NOTUSDT
OGNUSDT
ONDOUSDT
ONEUSDT
ONGUSDT
ONTUSDT
OPUSDT
ORDIUSDT
PENDLEUSDT
PEOPLEUSDT
PIXELUSDT
POLYXUSDT
PORTALUSDT
POWRUSDT
PYTHUSDT
QNTUSDT
QTUMUSDT
RAREUSDT
RENDERUSDT
REZUSDT
RIFUSDT
RLCUSDT
RONINUSDT
ROSEUSDT
RSRUSDT
RUNEUSDT
RVNUSDT
SAGAUSDT
SANDUSDT
SEIUSDT
SFPUSDT
SKLUSDT
SNXUSDT
SOLUSDT
SSVUSDT
STEEMUSDT
STGUSDT
STORJUSDT
STRKUSDT
STXUSDT
SUIUSDT
SUNUSDT
SUPERUSDT
SUSHIUSDT
SYNUSDT
TAOUSDT
THETAUSDT
TIAUSDT
TLMUSDT
TNSRUSDT
TONUSDT
TRBUSDT
TRXUSDT
TUSDT
TWTUSDT
UMAUSDT
UNIUSDT
USTCUSDT
VANRYUSDT
VETUSDT
WAXPUSDT
WIFUSDT
WLDUSDT
WOOUSDT
WUSDT
XAIUSDT
XLMUSDT
XMRUSDT
XRPUSDT
XTZUSDT
XVGUSDT
XVSUSDT
YFIUSDT
YGGUSDT
ZECUSDT
ZENUSDT
ZETAUSDT
ZILUSDT
ZKUSDT
ZROUSDT
ZRXUSDT
```

Excluded symbols: all entries below have reason `excluded_not_currently_trading_or_not_usdt_perpetual`.

```text
1000BONKUSDC
1000PEPEUSDC
1000SHIBUSDC
ADABUSD
AGIXBUSD
AGIXUSDT
AIUSDT
ALPHAUSDT
AMBUSDT
ANTUSDT
APTBUSD
ATAUSDT
AUDIOUSDT
BAKEUSDT
BALUSDT
BCHUSDC
BLZUSDT
BNBBUSD
BNBUSDC
BNXUSDT
BOMEUSDC
BTCBUSD
BTCUSDC
BTCUSDT_230929
COMBOUSDT
DARUSDT
DEFIUSDT
DENTUSDT
DGBUSDT
DODOBUSD
DOGEBUSD
DOGEUSDC
ENAUSDC
EOSUSDT
ETHBUSD
ETHFIUSDC
ETHUSDC
ETHUSDT_230929
FLMUSDT
FOOTBALLUSDT
FTMBUSD
FTMUSDT
FXSUSDT
GALABUSD
GALUSDT
HIFIUSDT
HIGHUSDT
HOOKUSDT
IDEXUSDT
KEYUSDT
KLAYUSDT
LDOBUSD
LINAUSDT
LINKUSDC
LRCUSDT
LTCBUSD
MATICBUSD
MATICUSDT
MDTUSDT
MKRUSDT
MYROUSDT
NKNUSDT
NTRNUSDT
OCEANUSDT
OMGUSDT
OMNIUSDT
OMUSDT
ORDIUSDC
OXTUSDT
PERPUSDT
PHBUSDT
RADUSDT
RDNTUSDT
REEFUSDT
RENUSDT
RNDRUSDT
SOLBUSD
SOLUSDC
STMXUSDT
SUIUSDC
SXPUSDT
SYSUSDT
TOKENUSDT
TOMOUSDT
TRUUSDT
TRXBUSD
UNFIUSDT
VOXELUSDT
WAVESUSDT
WIFUSDC
WLDUSDC
XEMUSDT
XRPBUSD
XRPUSDC
```

## Whitelist And Ref Evidence

`configs/prod/market_data.yaml` confirms `binance:futures` is `market_id=2`.

Whitelist result:

| Metric | Value |
|---|---:|
| Total whitelist rows after update | `277` |
| `market_id=2` rows | `216` |
| `market_id=2` enabled rows | `215` |
| `market_id=2` disabled rows | `1` |
| `market_id=2` duplicate symbols | `0` |
| Disabled `market_id=2` symbol preserved | `PEPEUSDT` |

Initial Mac Studio sync command used the staged whitelist at:

`/opt/roehub/state/rl_trading/stage04a_binance_futures_universe/whitelist_stage04a.csv`

Initial sync result:

```json
{"ref_market_inserted":0,"ref_instruments_rows_total":276,"ref_instruments_upserted":209,"ref_instruments_skipped_unchanged":67,"enabled_count":272,"disabled_count":4}
```

The sync emitted a warning for an existing duplicate whitelist key `(1, LTCUSDT)` from the pre-existing spot rows. This stage did not change or clean unrelated spot whitelist history. A second sync rerun after enrichment was idempotent:

```json
{"ref_market_inserted":0,"ref_instruments_rows_total":276,"ref_instruments_upserted":0,"ref_instruments_skipped_unchanged":276,"enabled_count":272,"disabled_count":4}
```

Futures-only enrichment evidence:

| Field | Value |
|---|---:|
| Instruments considered | `215` |
| Markets considered | `1` |
| Enrichment rows upserted | `209` |
| Missing metadata symbols | `0` |
| Latest `market_id=2` symbols | `216` |
| Latest enabled accepted symbols | `215` |
| Latest disabled symbols | `1` |
| Missing accepted symbols | `0` |
| Extra enabled symbols | `0` |
| `base_asset` coverage | `215` |
| `quote_asset=USDT` coverage | `215` |
| `price_step` coverage | `215` |
| `qty_step` coverage | `215` |
| `min_notional` coverage | `215` |

Runtime evidence artifact:

`/opt/roehub/state/rl_trading/stage04a_binance_futures_universe/stage04a_ref_enrichment_evidence.json`

sha256: `20559fbd67ae69de5c54c243399525494cb95ff665ab9328c40863aae346e4ad`

## Stage 04B Handoff

Stage `04B` must consume the manifest from:

`/opt/roehub/state/rl_trading/stage04a_binance_futures_universe/stage04a_universe_manifest.json`

Handoff contract:

- exact accepted universe: `215` `binance:futures` symbols, hash `114b0314a062415cef95fdabffe84b5ff5f2e4d0333f574bb356ef71bc4223fd`;
- exact exclusions: `94` symbols, reason `excluded_not_currently_trading_or_not_usdt_perpetual`, hash `4780f83bffacbc0993add8abf7e564e9bbea28f5ebca44a8c8f9cc1cf2416524`;
- source lower-bound policy: `max(required_source_window_start_utc, exchangeInfo.onboardDate)`;
- no excluded symbol is scheduled for backfill;
- no Binance spot, Bybit spot or Bybit futures branch is opened for v1 training;
- no candle backfill was performed in Stage `04A`.

Stage `04B` is allowed to start because the accepted whitelist/ref/enrichment evidence exists. Stage `04B` still owns historical backfill start/coverage evidence and must not treat this Stage `04A` acceptance as candle coverage.

## Service Calls And Runtime Writes

| Surface | Coverage |
|---|---|
| Public metadata call | Binance USD-M Futures public `GET /fapi/v1/exchangeInfo`; auth `N/A`; private/account endpoints `N/A`; order submit `N/A`. |
| Timeout/retry/error behavior | Local resolver uses configured Binance Futures base URL and timeout; enrichment uses existing `RestInstrumentMetadataSource` / `RequestsHttpClient` config. Failures propagate and would block acceptance instead of falling back to guessed symbols. |
| ClickHouse writes | Existing `sync-instruments` and `EnrichRefInstrumentsFromExchangeUseCase` wrote only `market_data.ref_market`/`market_data.ref_instruments`; no schema migration. |
| Idempotency | Initial sync inserted `209`; sync rerun inserted `0`. Enrichment only upserted changed/missing metadata for the accepted `market_id=2` symbols. |
| Unknown-state handling | Missing metadata count was `0`; if nonzero, Stage `04A` would stay blocked because Stage `04B` would not have complete filter/step/min-notional evidence. |
| Redaction boundary | Host-local env was sourced on Mac Studio without printing credentials; docs store only counts, hashes, paths and sanitized symbol lists. |
| Alerts/runbooks | `N/A`; no alert routing, Monit, launchd, scheduler interval or runbook action changed. |
| Browser/UI | `N/A`; prompt disabled browser runtime verification and no browser-visible behavior changed. |

## Contract Impact

| Surface | Classification | Reason |
|---|---|---|
| Public API contract | `none` | No routes or payloads changed. |
| Port contract | `none` | No existing port/protocol signature changed. |
| DTO schema | `none` | No wire DTO changed. |
| Persisted schema | `none` | No migration or table schema changed. |
| Persisted data semantics | `compatible-change` | `market_data.ref_instruments` latest state now includes the accepted `binance:futures` universe and enrichment rows. |
| Config schema/defaults | `compatible-change` | `configs/prod/whitelist.csv` contents changed; CSV schema and market ids are unchanged. |
| Request hash / cache key / persistence identity | `none` | No request/cache identity changed; `(market_id, symbol)` ref identity is preserved. |
| Service-call auth/timeout/retry/error semantics | `none` | Existing public Binance metadata source and ClickHouse writers were reused. |
| External side effects / idempotency / unknown-state semantics | `compatible-change` | Public metadata read and ClickHouse reference-data writes only; no exchange private/account or order side effects. Sync rerun proved no additional writes. |
| Logs/metrics/traces/audit/ledger/report/redaction | `compatible-change` | Adds sanitized manifest/evidence hashes and stage ledger/report; no secrets or raw provider payloads committed. |
| Alert/runbook semantics | `none` | No alert, Monit, launchd or runbook behavior changed. |
| Benchmark / rollout gate impact | `compatible-change` | Stage `04B` is unblocked only for the accepted symbol list and lower-bound policy. |
| Performance hot path | `none` | No live inference, scheduler loop or execution hot path changed. |
| Browser-visible behavior | `none` | Browser runtime verification disabled by prompt; no UI changed. |

## Quality Gates

| Gate | Result |
|---|---|
| `shasum -a 256 .codex/agents/generated/rl-trading-agent-platform-v1/04a-binance-futures-universe-whitelist.md` | passed; `80cbd412dc8d66191eab16dc9926f9303ee4923028cea7b8ae5796f6325cb3a1` |
| `uv run pytest -q tests/unit/scripts/rl_trading/test_resolve_stage04a_binance_futures_universe.py` | passed; `4 passed` |
| `uv run pytest -q tests/unit/contexts/market_data/adapters/test_whitelist_csv_loader.py tests/unit/contexts/market_data/application/use_cases/test_enrich_ref_instruments_from_exchange.py tests/unit/scripts/rl_trading/test_resolve_stage04a_binance_futures_universe.py` | passed; `10 passed` |
| `uv run ruff check scripts/rl_trading/resolve_stage04a_binance_futures_universe.py tests/unit/scripts/rl_trading/test_resolve_stage04a_binance_futures_universe.py` | passed |
| `uv run ruff check configs src/trading/contexts/market_data scripts/rl_trading tests/unit/contexts/market_data tests/unit/scripts/rl_trading` | passed |
| `uv run pyright scripts/rl_trading/resolve_stage04a_binance_futures_universe.py tests/unit/scripts/rl_trading/test_resolve_stage04a_binance_futures_universe.py` | passed; `0 errors` |
| Mac Studio public Binance Futures metadata resolver | passed; accepted/excluded hashes matched local evidence |
| Mac Studio whitelist -> `ref_instruments` sync | passed; first run inserted `209`, rerun inserted `0` |
| Mac Studio futures-only enrichment | passed; `215/215` accepted symbols enriched with no missing metadata |

## Cold Self-Review

Mode: `cold self-review fallback`.

Reason: the `multi_agent_v1.spawn_agent` tool exists, but its tool contract says not to spawn subagents unless the user explicitly asks for subagents, delegation or parallel agent work. No such user instruction was present, so an independent subagent pass was not allowed by the active tool contract.

Result: `Release`.

Reviewed lenses:

- prerequisite continuity: Stage `04` accepted before Stage `04A`;
- prompt traceability: path/hash recorded;
- planned file list and actual manifest aligned;
- accepted/excluded universe is based on HF train split `309`, not all-split `478`;
- exclusion reason literal `excluded_not_currently_trading_or_not_usdt_perpetual` is present;
- `binance:futures` only: no spot/Bybit branch opened;
- no candle backfill or model training performed;
- Mac Studio `ref_instruments` sync/enrichment evidence exists and tests-only acceptance was not used;
- source-window lower-bound policy is handed off to Stage `04B`;
- service-call, auth, timeout/error, redaction and N/A surfaces are explicit;
- delivery state is explicit as repository `local-only` plus Mac Studio reference-data write evidence.

No required fixes remained after this review.

## Residual Risks

- Binance `exchangeInfo` is current-state metadata. It does not prove historical point-in-time lifecycle for delisted symbols; those remain excluded by design in v1.
- Stage `04A` does not prove candle coverage. Stage `04B` must prove ingestion start and later coverage/gap metrics before Stage `04C`.
- Repository changes are local-only until a separate publish path creates a branch/PR and delivery evidence.
