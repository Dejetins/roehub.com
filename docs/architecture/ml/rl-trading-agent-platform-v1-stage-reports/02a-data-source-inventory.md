---
doc: rl-trading-agent-platform-v1-stage-02a-data-source-inventory
stage: "02A"
status: accepted
plan: docs/architecture/ml/rl-trading-agent-platform-v1.md
ledger: docs/architecture/ml/rl-trading-agent-platform-v1-stage-reports/rl-trading-agent-platform-v1-stage-ledger.md
collected_at: "2026-06-17"
---

# Stage 02A: Data Source Inventory

Статус: `accepted`.

Stage `02A` фиксирует инвентаризацию источников данных для RL Trading Agent Platform v1. Этот stage не принимает feature-contract, feature-mask, trainable/backtestable activation matrix, action/state/reward contract и не открывает paper/testnet/live execution.

Amendment `2026-06-17`: после дополнительной проверки полные HF NPZ payloads были скачаны во временный путь `/tmp/roehub_hf_open_rl_trading_binance_dataset` и inspected locally. Raw arrays не внесены в git и не сохранены в docs. Эта поправка уточняет symbol counts, NPZ format, channel order, training-source scope и backfill windows; статус Stage `02A` остается `accepted`, но downstream Stage `02B` обязан использовать уточненные факты.

User required before start: nothing unless a listed prerequisite is not accepted or a required credential/dataset/runtime source is unavailable; never ask for secrets in chat.

## Scope

Входит:

- проверить prerequisite Stage `01`;
- собрать source-backed inventory по HF dataset, Roehub ClickHouse, backtest artifacts, exchange/market coverage, lifecycle/gaps и raw feature gaps;
- сверить текущий classic strategy producer blocker state для Stage `05`, а также зависимости Stage `07`/`09`;
- записать prompt path/hash, file manifest, evidence, contract impact и next-stage handoff.

Не входит:

- обучение модели, импорт HF dataset в Roehub или сохранение полных NPZ payloads в git/runtime artifact store; amendment used a temporary `/tmp` download for inspection only;
- изменение API, DTO, storage schema, config, runtime services, exchange paths или UI;
- feature-mask/trainable/blocked decision для market branches;
- mainnet, testnet или paper execution;
- запись runtime ML artifacts в `/opt/roehub/state/rl_trading/`.

## Методология анализа

| Поле | Значение |
|---|---|
| Уровень глубины | `стандартный анализ`, inventory/data-quality audit без causal/business conclusion. |
| Тип задачи | Data source inventory и missing-data audit для будущих ML/dataset stages. |
| Выбранная методология | Source inventory + coverage reconciliation + missing-data classification. |
| Простое объяснение метода | Сравниваем заявленные источники, фактические schemas/manifests/counts и gaps, затем явно отделяем observed facts от решений следующего stage. |
| Бизнес-язык | Проверяем, на каких рынках у Roehub уже есть данные для будущей RL-стратегии, а где нужны дообогащение, repair или отдельное решение до запуска обучения. |
| Единица анализа | `exchange_name + market_type + symbol`, HF split, artifact root, feature field. |
| Основные метрики | Row/session counts, unique tickers, first/last candle, `volume_quote_rows`, `trades_count_rows`, inferred missing minutes, artifact `ohlcv` shape/hash. |
| Прокси-метрики | Lifecycle coverage через first/last observed candle; точных listing/delisting timestamps в текущем `market_data` schema нет. |
| Риски интерпретации | HF metadata card и NPZ `_keys_map_` отличаются по train count; lifecycle gaps cannot prove exchange listing windows; Bybit `trades_count` gap is source/schema-level in current ingestion. |
| Проверки перед выводом | Mac Studio ClickHouse read-only queries, HF metadata byte-range read plus later full temporary NPZ inspection, artifact manifest read, docs index check. |
| Вопросы до расчетов | Нет; stage descriptive-only and does not make activation decisions. |

## File Manifest

| Created | Modified | Deleted | Reason | Contract impact |
|---|---|---|---|---|
| `docs/architecture/ml/rl-trading-agent-platform-v1-stage-reports/02a-data-source-inventory.md` | - | - | Stage `02A` source-backed data inventory report. | `compatible-change` docs/report only |
| - | `docs/architecture/ml/rl-trading-agent-platform-v1-stage-reports/rl-trading-agent-platform-v1-stage-ledger.md` | - | Mark Stage `02A` accepted, record evidence/handoff, open Stage `02B`. | `compatible-change` docs/ledger only |
| - | `docs/architecture/README.md` | - | Docs index regeneration after adding Stage `02A` report; generator also indexed pre-existing unrelated workspace file `docs/architecture/live_execution/strategy-producer-paper-testnet-trading-v1-stage-reports/06-supervised-strategy-producer.md`. | `compatible-change` docs index only |

Amendment file manifest on `2026-06-17`:

| Created | Modified | Deleted | Reason | Contract impact |
|---|---|---|---|---|
| - | `docs/architecture/ml/rl-trading-agent-platform-v1.md` | - | Restrict v1 training-source scope to `binance:futures` and record source windows. | `compatible-change` docs/plan only |
| - | `docs/architecture/ml/rl-trading-agent-platform-v1-stage-reports/02a-data-source-inventory.md` | - | Add full NPZ inspection results, clarify `30/33` overlap, and record backfill windows. | `compatible-change` docs/report only |
| - | `docs/architecture/ml/rl-trading-agent-platform-v1-stage-reports/rl-trading-agent-platform-v1-stage-ledger.md` | - | Align current handoff facts with Binance Futures-only training scope. | `compatible-change` docs/ledger only |
| - | `.codex/agents/generated/rl-trading-agent-platform-v1/02b-feature-live-feed-contract.md`, `.codex/agents/generated/rl-trading-agent-platform-v1/04-hf-reproducibility.md`, `.codex/agents/generated/rl-trading-agent-platform-v1/05-roehub-dataset-builder-v1.md`, `.codex/agents/generated/rl-trading-agent-platform-v1/06-dataset-qa-session-extractor.md`, `.codex/agents/generated/rl-trading-agent-platform-v1/07-d3qn-per-training-runner.md`, `.codex/agents/generated/rl-trading-agent-platform-v1/08-roehub-backtest-evaluation.md` | - | Keep future stage prompts aligned with Binance Futures-only training source. Historical executed prompt `02a-data-source-inventory.md` is intentionally not changed. | `compatible-change` prompt/docs only |

Planned concrete file list before edits:

- `docs/architecture/ml/rl-trading-agent-platform-v1-stage-reports/02a-data-source-inventory.md`
- `docs/architecture/ml/rl-trading-agent-platform-v1-stage-reports/rl-trading-agent-platform-v1-stage-ledger.md`
- `docs/architecture/README.md` only if `python -m tools.docs.generate_docs_index --check` requires index regeneration

Outside expected paths: `docs/architecture/README.md` is in `docs_to_update` and is justified as the generated architecture docs index. The pre-existing `06-supervised-strategy-producer.md` file was not created or edited by this stage.

## Prompt Evidence

| Field | Value |
|---|---|
| Prompt path | `.codex/agents/generated/rl-trading-agent-platform-v1/02a-data-source-inventory.md` |
| Prompt sha256 | `e875a20c92121eb94107124975628f636b6055000086d93e71c3231144eb6de4` |
| Ledger state before implementation | Stage `01` accepted; `current_stage=02A`; Stage `02A` pending |
| Required prerequisite | Stage `01` accepted |
| Delivery state | `local-only`; no branch, PR, main delivery, deploy, runtime service, schema, API, UI, exchange, or ML artifact change |
| Large artifacts | HF NPZ raw arrays were later downloaded only to `/tmp/roehub_hf_open_rl_trading_binance_dataset` for local inspection; nothing was written to git or `/opt/roehub/state/rl_trading/` by this amendment |

## Source Evidence

| Source | Evidence collected | Boundary |
|---|---|---|
| HF dataset card/article/repo | `ResearchRL/open-rl-trading-binance-dataset` and the source article/repo describe Binance Futures minute-level subsets, sessions with shape `(150, 7)`, and periods Train/Validation/Test/Backtest. Full inspection corrected the article-compatible channel order to `open`, `high`, `volume_weighted_average`, `low`, `close`, `volume`, `num_trades`. | Public source plus local temporary inspection; no raw arrays committed. |
| HF NPZ full local inspection | Full `train_data.npz`, `val_data.npz`, `test_data.npz`, `backtest_data.npz` were downloaded to `/tmp/roehub_hf_open_rl_trading_binance_dataset` and inspected with NumPy for `_keys_map_`, array shapes, dtypes, per-split unique symbols and hashes. | Temporary local files only; no candle arrays committed. |
| Mac Studio ClickHouse | Read-only queries through `/opt/clickhouse/clickhouse client` against `market_data`. | No mutation; no provider secrets. |
| Backtest artifacts | Read-only `current.yaml` and active slot `manifest.yaml` under `/opt/roehub/state/backtest_artifacts/v2`. | Manifest summaries and hashes only. |
| Code entrypoints | `canonical_candle_reader.py`, `artifact_array_loader.py`, `artifact_manifest_loader.py`, REST/WS clients and raw kline schemas. | Static contract evidence only. |

## HF Dataset Inventory

HF source: `https://huggingface.co/datasets/ResearchRL/open-rl-trading-binance-dataset`.

| Split file | Card samples | Observed `_keys_map_` sessions | Observed period | Notes |
|---|---:|---:|---|---|
| `train_data.npz` | `24,104` | `24,086` | `2020-01-14 14:28` to `2024-08-30 18:33` | Observed metadata is 18 sessions below card count. |
| `val_data.npz` | `1,377` | `1,377` | `2024-09-01 06:02` to `2024-11-30 22:46` | Matches card count. |
| `test_data.npz` | `3,400` | `3,400` | `2024-12-01 00:16` to `2025-02-28 22:53` | Matches card count. |
| `backtest_data.npz` | `3,186` | `3,186` | `2025-03-01 00:15` to `2025-05-31 22:47` | Matches card count. |
| Total | `32,067` from card sum | `32,049` observed | `2020-01-14 14:28` to `2025-05-31 22:47` | Use observed metadata count for reproducibility until Stage `04` imports full dataset. |

Observed unique HF symbols: `478`.

Full NPZ inspection amendment:

| Split file | Observed sessions | Unique symbols | Shape / dtype summary | sha256 |
|---|---:|---:|---|---|
| `train_data.npz` | `24,086` | `309` | `24,086` arrays shaped `(150, 7)`; `24,085` `float64`, `1` `float32` | `1c5cdf179777f0a68a81da915749f50d97826282e1419a5314a67b170e9cb14d` |
| `val_data.npz` | `1,377` | `280` | all arrays shaped `(150, 7)`; `float64` | `1e1e347bd4f842680f8a1781bc1e51f790f5e5865796e9ef3bd69548e20c51f4` |
| `test_data.npz` | `3,400` | `362` | all arrays shaped `(150, 7)`; `float64` | `ff72d998fbf7d507b3db46e543aae324bece368a50ad043c057217ec2c744b1b` |
| `backtest_data.npz` | `3,186` | `321` | all arrays shaped `(150, 7)`; `float64` | `dce732fda8fe1d33e92617d12f0defa3e202013617b91bb34df4d0b65aa023ee` |

NPZ format: each split stores session arrays under `fetcher_N` keys plus a pickled `_keys_map_` dict mapping `fetcher_N` to `(symbol, signal_datetime_utc)`. The train split has `309` unique symbols, not `33`. The earlier `30/33` fact below is only Roehub `ref_instruments` overlap across the current local reference universe; it is not the size of the HF training universe.

Full observed HF symbol list:

```text
1000000MOGUSDT, 1000BONKUSDC, 1000BONKUSDT, 1000CATUSDT, 1000FLOKIUSDT,
1000LUNCUSDT, 1000PEPEUSDC, 1000PEPEUSDT, 1000RATSUSDT, 1000SHIBUSDC,
1000SHIBUSDT, 1000XECUSDT, 1000XUSDT, 1INCHUSDT, AAVEUSDT, ACEUSDT,
ACHUSDT, ACTUSDT, ACXUSDT, ADABUSD, ADAUSDC, ADAUSDT, AERGOUSDT,
AEROUSDT, AEVOUSDT, AGIXBUSD, AGIXUSDT, AGLDUSDT, AGTUSDT, AI16ZUSDT,
AIOTUSDT, AIUSDT, AIXBTUSDT, AKTUSDT, ALCHUSDT, ALGOUSDT, ALICEUSDT,
ALPHAUSDT, ALPINEUSDT, ALTUSDT, AMBUSDT, ANIMEUSDT, ANKRUSDT, ANTUSDT,
APEUSDT, API3USDT, APTBUSD, APTUSDT, ARBUSDT, ARCUSDT, ARKMUSDT,
ARKUSDT, ARPAUSDT, ARUSDT, ASRUSDT, ASTRUSDT, ATAUSDT, ATHUSDT,
ATOMUSDT, AUCTIONUSDT, AUDIOUSDT, AVAAIUSDT, AVAUSDT, AVAXUSDC,
AVAXUSDT, AXLUSDT, AXSUSDT, B2USDT, BABYUSDT, BAKEUSDT, BALUSDT,
BANANAUSDT, BANDUSDT, BANKUSDT, BANUSDT, BATUSDT, BBUSDT, BCHUSDC,
BCHUSDT, BEAMXUSDT, BELUSDT, BERAUSDT, BICOUSDT, BIDUSDT, BIGTIMEUSDT,
BIOUSDT, BLURUSDT, BLZUSDT, BMTUSDT, BNBBUSD, BNBUSDC, BNBUSDT,
BNTUSDT, BNXUSDT, BOMEUSDC, BOMEUSDT, BRETTUSDT, BROCCOLI714USDT,
BROCCOLIF3BUSDT, BRUSDT, BSVUSDT, BSWUSDT, BTCBUSD, BTCDOMUSDT,
BTCUSDC, BTCUSDT, BTCUSDT_230929, BUSDT, C98USDT, CAKEUSDT, CATIUSDT,
CELOUSDT, CELRUSDT, CETUSUSDT, CFXUSDT, CGPTUSDT, CHESSUSDT,
CHILLGUYUSDT, CHRUSDT, CHZUSDT, CKBUSDT, COMBOUSDT, COMPUSDT,
COOKIEUSDT, COSUSDT, COTIUSDT, COWUSDT, CRVUSDC, CRVUSDT, CTKUSDT,
CTSIUSDT, CVCUSDT, CVXUSDT, CYBERUSDT, DARUSDT, DASHUSDT, DEEPUSDT,
DEFIUSDT, DEGENUSDT, DEGOUSDT, DENTUSDT, DEXEUSDT, DFUSDT, DGBUSDT,
DIAUSDT, DODOBUSD, DODOXUSDT, DOGEBUSD, DOGEUSDC, DOGEUSDT, DOLOUSDT,
DOTUSDT, DRIFTUSDT, DUSDT, DUSKUSDT, DYDXUSDT, DYMUSDT, EDUUSDT,
EGLDUSDT, EIGENUSDT, ENAUSDC, ENAUSDT, ENJUSDT, ENSUSDT, EOSUSDT,
EPICUSDT, EPTUSDT, ETCUSDT, ETHBUSD, ETHFIUSDC, ETHFIUSDT, ETHUSDC,
ETHUSDT, ETHUSDT_230929, ETHUSDT_250926, ETHWUSDT, FARTCOINUSDT,
FETUSDT, FHEUSDT, FIDAUSDT, FILUSDC, FILUSDT, FIOUSDT, FISUSDT,
FLMUSDT, FLOWUSDT, FLUXUSDT, FOOTBALLUSDT, FORMUSDT, FORTHUSDT,
FTMBUSD, FTMUSDT, FXSUSDT, GALABUSD, GALAUSDT, GALUSDT, GASUSDT,
GHSTUSDT, GLMUSDT, GMTUSDT, GMXUSDT, GOATUSDT, GPSUSDT, GRASSUSDT,
GRIFFAINUSDT, GRTUSDT, GTCUSDT, GUNUSDT, GUSDT, HAEDALUSDT, HBARUSDT,
HEIUSDT, HFTUSDT, HIFIUSDT, HIGHUSDT, HIPPOUSDT, HIVEUSDT, HOOKUSDT,
HOTUSDT, HUMAUSDT, HYPERUSDT, ICPUSDT, ICXUSDT, IDEXUSDT, IDUSDT,
ILVUSDT, IMXUSDT, INITUSDT, INJUSDT, IOSTUSDT, IOTAUSDT, IOTXUSDT,
IOUSDT, IPUSDT, JASMYUSDT, JELLYJELLYUSDT, JOEUSDT, JSTUSDT, JTOUSDT,
JUPUSDT, KAIAUSDT, KAITOUSDC, KAITOUSDT, KASUSDT, KAVAUSDT, KDAUSDT,
KERNELUSDT, KEYUSDT, KLAYUSDT, KMNOUSDT, KNCUSDT, KOMAUSDT, KSMUSDT,
LAYERUSDT, LDOBUSD, LDOUSDT, LINAUSDT, LINKUSDC, LINKUSDT, LISTAUSDT,
LITUSDT, LOKAUSDT, LPTUSDT, LQTYUSDT, LRCUSDT, LSKUSDT, LTCBUSD,
LTCUSDC, LTCUSDT, LUMIAUSDT, LUNA2USDT, MAGICUSDT, MANAUSDT,
MANTAUSDT, MASKUSDT, MATICBUSD, MATICUSDT, MAVIAUSDT, MAVUSDT,
MBOXUSDT, MDTUSDT, MELANIAUSDT, MEMEUSDT, METISUSDT, MEUSDT, MEWUSDT,
MILKUSDT, MINAUSDT, MKRUSDT, MLNUSDT, MOCAUSDT, MOODENGUSDT,
MORPHOUSDT, MOVEUSDT, MOVRUSDT, MTLUSDT, MUBARAKUSDT, MYROUSDT,
NEARUSDT, NEIROETHUSDT, NEOUSDC, NEOUSDT, NFPUSDT, NILUSDT, NKNUSDT,
NMRUSDT, NOTUSDT, NTRNUSDT, NXPCUSDT, OBOLUSDT, OCEANUSDT, OGNUSDT,
OMGUSDT, OMNIUSDT, OMUSDT, ONDOUSDT, ONEUSDT, ONGUSDT, ONTUSDT,
OPUSDT, ORCAUSDT, ORDIUSDC, ORDIUSDT, OXTUSDT, PARTIUSDT, PENDLEUSDT,
PENGUUSDT, PEOPLEUSDT, PERPUSDT, PHAUSDT, PHBUSDT, PIPPINUSDT,
PIXELUSDT, PLUMEUSDT, PNUTUSDC, PNUTUSDT, POLUSDT, POLYXUSDT,
PONKEUSDT, POPCATUSDT, PORTALUSDT, POWRUSDT, PROMPTUSDT, PUNDIXUSDT,
PYTHUSDT, QNTUSDT, QTUMUSDT, QUICKUSDT, RADUSDT, RAREUSDT, RAYSOLUSDT,
RDNTUSDT, REDUSDT, REEFUSDT, REIUSDT, RENDERUSDT, RENUSDT, REZUSDT,
RIFUSDT, RLCUSDT, RNDRUSDT, RONINUSDT, ROSEUSDT, RPLUSDT, RSRUSDT,
RUNEUSDT, RVNUSDT, SAFEUSDT, SAGAUSDT, SANDUSDT, SANTOSUSDT, SCRTUSDT,
SCRUSDT, SEIUSDT, SFPUSDT, SHELLUSDT, SIGNUSDT, SIRENUSDT, SKLUSDT,
SKYAIUSDT, SLERFUSDT, SNXUSDT, SOLBUSD, SOLUSDC, SOLUSDT, SOLVUSDT,
SONICUSDT, SOONUSDT, SOPHUSDT, SPXUSDT, SSVUSDT, STEEMUSDT, STGUSDT,
STMXUSDT, STORJUSDT, STOUSDT, STRKUSDT, STXUSDT, SUIUSDC, SUIUSDT,
SUNUSDT, SUPERUSDT, SUSDT, SUSHIUSDT, SWARMSUSDT, SWELLUSDT, SXPUSDT,
SXTUSDT, SYNUSDT, SYRUPUSDT, SYSUSDT, TAOUSDT, THETAUSDT, THEUSDT,
TIAUSDC, TIAUSDT, TLMUSDT, TNSRUSDT, TOKENUSDT, TOMOUSDT, TONUSDT,
TRBUSDT, TRUMPUSDC, TRUMPUSDT, TRUUSDT, TRXBUSD, TRXUSDT, TSTUSDT,
TURBOUSDT, TUSDT, TUTUSDT, TWTUSDT, UMAUSDT, UNFIUSDT, UNIUSDT,
USTCUSDT, USUALUSDT, UXLINKUSDT, VANAUSDT, VANRYUSDT, VELODROMEUSDT,
VETUSDT, VICUSDT, VINEUSDT, VIRTUALUSDT, VOXELUSDT, VVVUSDT, WALUSDT,
WAVESUSDT, WAXPUSDT, WCTUSDT, WIFUSDC, WIFUSDT, WLDUSDC, WLDUSDT,
WOOUSDT, WUSDT, XAIUSDT, XCNUSDT, XEMUSDT, XLMUSDT, XMRUSDT, XRPBUSD,
XRPUSDC, XRPUSDT, XTZUSDT, XVGUSDT, XVSUSDT, YFIUSDT, YGGUSDT,
ZECUSDT, ZENUSDT, ZEREBROUSDT, ZETAUSDT, ZILUSDT, ZKUSDT, ZROUSDT,
ZRXUSDT
```

## Roehub Market Coverage And HF Comparison

Current `market_data.ref_market` / `ref_instruments` snapshot on Mac Studio:

| Exchange | Market type | `ref_instruments` symbols | Tradable | HF matches in ref | HF matches tradable | Ref symbols not in HF |
|---|---|---:|---:|---:|---:|---|
| `binance` | `futures` | 7 | 6 | 6 | 6 | `PEPEUSDT` |
| `binance` | `spot` | 32 | 31 | 30 | 30 | `PEPEUSDT`, `SHIBUSDT` |
| `bybit` | `futures` | 5 | 4 | 4 | 4 | `PEPEUSDT` |
| `bybit` | `spot` | 23 | 22 | 19 | 19 | `HYPEUSDT`, `MNTUSDT`, `PEPEUSDT`, `SHIBUSDT` |
| Distinct symbols | all | 34 | 33 | 30 | 30 | `HYPEUSDT`, `MNTUSDT`, `PEPEUSDT`, `SHIBUSDT` |

HF-matched Roehub symbols by branch:

| Branch | Symbols |
|---|---|
| `binance:futures` | `BTCUSDT`, `ICPUSDT`, `NEARUSDT`, `TAOUSDT`, `WLDUSDT`, `ZECUSDT` |
| `binance:spot` | `AAVEUSDT`, `ADAUSDT`, `ALGOUSDT`, `APTUSDT`, `ATOMUSDT`, `AVAXUSDT`, `BCHUSDT`, `BNBUSDT`, `BTCUSDT`, `DOGEUSDT`, `DOTUSDT`, `ENAUSDT`, `ETCUSDT`, `ETHUSDT`, `HBARUSDT`, `ICPUSDT`, `LINKUSDT`, `LTCUSDT`, `NEARUSDT`, `QNTUSDT`, `SOLUSDT`, `SUIUSDT`, `TAOUSDT`, `TONUSDT`, `TRXUSDT`, `UNIUSDT`, `WLDUSDT`, `XLMUSDT`, `XRPUSDT`, `ZECUSDT` |
| `bybit:futures` | `BTCUSDT`, `ICPUSDT`, `NEARUSDT`, `WLDUSDT` |
| `bybit:spot` | `AAVEUSDT`, `ADAUSDT`, `ALGOUSDT`, `APTUSDT`, `ATOMUSDT`, `BTCUSDT`, `DOGEUSDT`, `DOTUSDT`, `ENAUSDT`, `ETCUSDT`, `ICPUSDT`, `LTCUSDT`, `NEARUSDT`, `QNTUSDT`, `SOLUSDT`, `SUIUSDT`, `TONUSDT`, `UNIUSDT`, `WLDUSDT` |

Interpretation boundary: these are inventory overlaps only. Stage `02A` does not decide whether any overlap is trainable, blocked, feature-mask, or research-only.

Training-source scope amendment: the accepted v1 training source is `binance:futures` only. Binance spot, Bybit spot, and Bybit futures are not training sources in this cycle and must be recorded downstream as `blocked_not_training_source_v1`, not as Bybit enrich/feature-mask training branches.

## ClickHouse Schema And Coverage

Current `market_data` tables matching the inventory scope:

- `canonical_candles_1m`
- `canonical_candles_1m_stats`
- `raw_binance_klines_1m`
- `raw_bybit_klines_1m`
- `ref_market`
- `ref_instruments`

No `market_data` table was found for funding, mark price, index price, leverage tiers, or separate exchange filters beyond `ref_instruments`.

Canonical aggregate coverage:

| Exchange | Market type | Rows | Symbols | First observed candle | Last observed candle | `volume_quote_rows` | `trades_count_rows` | `vwap` computable rows |
|---|---|---:|---:|---|---|---:|---:|---:|
| `binance` | `futures` | 14,524,225 | 6 | `2019-09-08 17:57` | `2026-06-17 19:33` | 14,524,225 | 14,524,225 | 14,499,852 |
| `binance` | `spot` | 99,683,637 | 31 | `2017-08-17 04:00` | `2026-06-17 19:33` | 99,683,637 | 99,683,637 | 96,743,865 |
| `bybit` | `futures` | 9,765,829 | 4 | `2020-03-25 10:36` | `2026-06-17 19:33` | 9,765,829 | 0 | 9,531,325 |
| `bybit` | `spot` | 46,627,907 | 22 | `2021-07-05 12:00` | `2026-06-17 19:33` | 46,627,907 | 0 | 40,312,897 |

Current-day tail snapshot at query time:

| Exchange | Market type | Current-day rows | Current-day symbols | Max observed current-day candle |
|---|---:|---:|---:|
| `binance` | `futures` | 7,068 | 6 | `2026-06-17 19:37` |
| `binance` | `spot` | 36,518 | 31 | `2026-06-17 19:37` |
| `bybit` | `futures` | 4,712 | 4 | `2026-06-17 19:37` |
| `bybit` | `spot` | 24,738 | 21 | `2026-06-17 19:37` |

Bybit spot has 22 canonical symbols overall, but only 21 had current-day rows in this snapshot. `TONUSDT` last observed candle was `2026-06-16 08:03`.

## Feature Field Inventory

Article-compatible 7-channel baseline:

| Feature | Roehub observed source | Current inventory status |
|---|---|---|
| `open` | `canonical_candles_1m.open`; artifact `ohlcv[:, 0]` | Present in ClickHouse and artifacts. |
| `high` | `canonical_candles_1m.high`; artifact `ohlcv[:, 1]` | Present in ClickHouse and artifacts. |
| `low` | `canonical_candles_1m.low`; artifact `ohlcv[:, 2]` | Present in ClickHouse and artifacts. |
| `close` | `canonical_candles_1m.close`; artifact `ohlcv[:, 3]` | Present in ClickHouse and artifacts. |
| `volume` | `canonical_candles_1m.volume_base`; artifact `ohlcv[:, 4]` | Present in ClickHouse and artifacts. |
| `vwap` / `volume_weighted_average` | `volume_quote / volume_base` when `volume_base > 0` and `volume_quote` is not null | Computable for many rows; zero-volume policy remains a later feature-contract decision. |
| `num_trades` | `canonical_candles_1m.trades_count` | Present for Binance rows; absent for all current Bybit canonical rows. |

Raw-source explanation:

| Source | Observed fields | Impact |
|---|---|---|
| `raw_binance_klines_1m` | `quote_asset_volume`, `number_of_trades`, taker volumes. | Binance raw and canonical can carry `volume_quote` and `trades_count`. |
| `raw_bybit_klines_1m` | `turnover`; no trade-count column. | Bybit canonical rows get `volume_quote` from turnover but no `trades_count`. |
| Bybit REST/WS clients | Current code maps Bybit `volume_quote` from `turnover` and sets `trades_count=None`. | Bybit `num_trades` remains an inventory gap, but Bybit is not a v1 training source. Do not open a Bybit enrich/feature-mask training branch in Stage `02B`. |
| Redis live candle publisher | Publishes `volume_quote`; no observed `trades_count` payload field. | Stage `02B` must decide live-feed parity for `num_trades`; Stage `02A` makes no decision. |

Futures-only missing data for article-compatible production backtest:

| Required data | Current inventory |
|---|---|
| Funding-rate history | No `market_data` funding table found. |
| Mark/index price history | No `market_data` mark/index price table found. |
| Filters / min notional / precision / qty step | `ref_instruments` has current `min_notional`, `qty_step`, `price_step`, not point-in-time history. |
| Leverage / margin tiers | No `market_data` leverage-tier table found. Current classic producer Stage `05` account projection can read current account leverage/margin, but this is not historical dataset coverage. |
| Instrument lifecycle listing/delisting windows | No explicit listing/delisting timestamp columns in `ref_instruments`; only current `status`, `is_tradable`, and first/last observed candles are available. |

## Backtest Artifact Inventory

Production artifact root from config: `/opt/roehub/state/backtest_artifacts/v2`.

Current artifact roots:

| Artifact root | Active slot | Generation | `asof_date` | Manifest hash | 1m bars | 1m `ohlcv` shape | Notes |
|---|---|---:|---|---|---:|---|---|
| `binance/futures/BTCUSDT` | `slot_a` | 1 | `2026-06-16` | `0cd6537e0b5ef70415e99915e93ff8ad46630010033029a0abf7162da43f96a3` | 3,561,231 | `[3561231, 5]` | Prices for `1m`, `15m`, `30m`, `1h`, `2h`, `4h`, `6h`, `8h`, `1d`, `2d`, `3d`; 10 mappings; 3 signals; hit-times present. |
| `binance/spot/BTCUSDT` | `slot_a` | 11 | `2026-06-16` | `af411c1c42280b624078865c27d7d8a59397e852638e6f39a9a27029a0c53436` | 4,635,952 | `[4635952, 5]` | Same timeframe/mapping/signal/hit-times families. |
| `bybit/spot/BTCUSDT` | `slot_a` | 1 | `2026-06-16` | `2e47d8ca04652957e8cd55acdb85848f5063d18431fe68af619b28d22ef449a2` | 2,602,559 | `[2602559, 5]` | Same timeframe/mapping/signal/hit-times families. |

No current artifact root was found for `bybit/futures/BTCUSDT`, and no non-BTCUSDT artifact roots were found under the production v2 artifact root.

Artifact contract facts:

- `FilesystemBacktestArtifactArrayLoader.load_price_arrays` validates `ohlcv` as `float32`, rank 2, exactly 5 fields.
- `YamlBacktestArtifactLoaderV2` loads strict `current.yaml` and slot `manifest.yaml` from deterministic paths and validates manifest hashes/metadata.
- Existing artifact arrays are a fast OHLCV source, not a complete 7-channel RL source. Later stages must augment or materialize separate RL feature artifacts.

## Lifecycle And Gap Inventory

Lifecycle evidence:

| Source | Observed lifecycle fields | Limitation |
|---|---|---|
| `ref_instruments` | `status`, `is_tradable`, `base_asset`, `quote_asset`, `price_step`, `qty_step`, `min_notional`, `updated_at`. | Current-state only; no listing/delisting timestamps or point-in-time filter history. |
| `canonical_candles_1m` | First/last observed candle per branch/symbol. | Proxy for available history, not proof of exchange lifecycle. |
| Artifacts | Active root `asof_date` and manifest coverage/hash. | Only three BTCUSDT roots and 5-column OHLCV arrays. |

Inferred internal continuity gaps from first to last observed candle, using `expected_minutes - uniqExact(ts_open)` per symbol:

| Branch | Symbols | Unique minutes | Expected minutes from first to last | Inferred missing minutes | Duplicate physical rows |
|---|---:|---:|---:|---:|---:|
| `binance:futures` | 6 | 14,524,255 | 14,524,256 | 1 | 0 |
| `binance:spot` | 31 | 99,683,792 | 99,779,850 | 96,058 | 0 |
| `bybit:futures` | 4 | 9,765,849 | 9,765,849 | 0 | 0 |
| `bybit:spot` | 22 | 46,628,012 | 46,638,170 | 10,158 | 0 |

Largest inferred symbol gaps:

| Branch | Symbol | Inferred missing minutes | First observed | Last observed |
|---|---|---:|---|---|
| `bybit:spot` | `SHIBUSDT` | 10,155 | `2021-10-29 07:27` | `2026-06-17 19:39` |
| `binance:spot` | `BTCUSDT` | 8,632 | `2017-08-17 04:00` | `2026-06-17 19:39` |
| `binance:spot` | `ETHUSDT` | 8,632 | `2017-08-17 04:00` | `2026-06-17 19:39` |
| `binance:spot` | `BNBUSDT` | 8,213 | `2017-11-06 03:54` | `2026-06-17 19:39` |
| `binance:spot` | `LTCUSDT` | 8,164 | `2017-12-13 03:32` | `2026-06-17 19:39` |

Repair paths already present:

| Gap class | Existing Roehub path | Status in this stage |
|---|---|---|
| Tail/historical canonical gaps | `RestCatchUp1mUseCase`, `SchedulerBackfillPlanner`, `AsyncRestFillQueue`, `apps/cli/commands/rest_catchup_1m.py`, market-data scheduler metrics. | Exists; Stage `02A` did not run a repair. |
| Raw/canonical duplicates | `docs/runbooks/clickhouse-partition-dedup.md`; `scripts/macos/clickhouse_partition_dedup.sh`. | Exists; Stage `02A` observed duplicate physical rows as `0` in the gap proxy query. |
| Artifact rebuild/publish | `docs/runbooks/backtest-artifacts-rebuild.md`; `apps/cli/commands/backtest_artifact_publish.py`; production config `configs/prod/backtest_artifacts.yaml`. | Exists for artifact roots; Stage `02A` did not rebuild artifacts. |
| Bybit `trades_count` | No current raw Bybit field or adapter mapping for trade count. | Future/non-training inventory gap only; not a v1 training repair path because Bybit is `blocked_not_training_source_v1`. |
| Futures funding/mark/index/leverage history | No current `market_data` tables found. | Open evidence/gap; later stage must define source or approximation before futures activation. |
| Point-in-time lifecycle/filter history | No listing/delisting/filter-history tables found. | Open evidence/gap; current `ref_instruments` is not enough for survivorship-bias proof. |

## Dataset Supplement And Backfill Windows

Can Roehub supplement the HF-compatible dataset in the same format: yes, for `binance:futures`. Binance raw/canonical data contains `open`, `high`, `low`, `close`, `volume_base`, `volume_quote`, and `trades_count`; `volume_weighted_average` can be derived from `volume_quote / volume_base` when `volume_base > 0` with a zero-volume policy defined in Stage `02B`. The article-compatible NPZ writer can mirror the HF structure: `fetcher_N` arrays shaped `(150, 7)` plus `_keys_map_`.

Current Roehub Binance Futures coverage for the accepted training source:

| Symbol | Ref status | Tradable | First candle | Last candle at amendment query | Inferred missing minutes | Required fields |
|---|---|---:|---|---|---:|---|
| `BTCUSDT` | `ENABLED` | 1 | `2019-09-08 17:57` | `2026-06-17 20:31` | `1` (`2019-09-08 19:00`, outside current HF-compatible windows) | `volume_quote`, `trades_count`, derived `vwap` present |
| `ICPUSDT` | `ENABLED` | 1 | `2022-09-27 02:30` | `2026-06-17 20:31` | `0` | `volume_quote`, `trades_count`, derived `vwap` present |
| `NEARUSDT` | `ENABLED` | 1 | `2020-10-15 08:00` | `2026-06-17 20:31` | `0` | `volume_quote`, `trades_count`, derived `vwap` present |
| `TAOUSDT` | `ENABLED` | 1 | `2024-04-11 14:30` | `2026-06-17 20:31` | `0` | `volume_quote`, `trades_count`, derived `vwap` present |
| `WLDUSDT` | `ENABLED` | 1 | `2023-07-24 12:00` | `2026-06-17 20:31` | `0` | `volume_quote`, `trades_count`, derived `vwap` present |
| `ZECUSDT` | `ENABLED` | 1 | `2020-02-05 08:01` | `2026-06-17 20:31` | `0` | `volume_quote`, `trades_count`, derived `vwap` present |
| `PEPEUSDT` | `DISABLED` | 0 | no usable rows | no usable rows | n/a | not in accepted current training universe |

For the current six-symbol Roehub Binance Futures universe, no candle backfill is needed for the HF-compatible/post-HF source windows below, except that each symbol's listing start naturally limits early windows. If the selected universe is expanded toward the current HF Binance Futures breadth, current Binance Futures `exchangeInfo` shows `342` HF symbols are currently trading USDT perpetual contracts; Roehub currently has `6`, so the maximal HF-current expansion would require adding/backfilling `336` additional Binance Futures symbols before session extraction.

Source windows to load/build for article-compatible sessions:

| Dataset segment | Signal window | Required source candle window |
|---|---|---|
| HF train-compatible | `[2020-01-14T00:00:00Z, 2024-08-31T00:00:00Z)` | `[2020-01-13T22:30:00Z, 2024-08-31T01:00:00Z)` |
| HF validation-compatible | `[2024-09-01T00:00:00Z, 2024-12-01T00:00:00Z)` | `[2024-08-31T22:30:00Z, 2024-12-01T01:00:00Z)` |
| HF test-compatible | `[2024-12-01T00:00:00Z, 2025-03-01T00:00:00Z)` | `[2024-11-30T22:30:00Z, 2025-03-01T01:00:00Z)` |
| HF backtest-compatible | `[2025-03-01T00:00:00Z, 2025-06-01T00:00:00Z)` | `[2025-02-28T22:30:00Z, 2025-06-01T01:00:00Z)` |
| Post-HF extension from current Mac Studio snapshot | `[2025-06-01T00:00:00Z, 2026-06-17T19:32:00Z]` | `[2025-05-31T22:30:00Z, 2026-06-17T20:32:00Z)` |

The source window expands the signal window by `pre_signal_len=90` minutes and `post_signal_len=60` minutes. The post-HF endpoint is tied to the observed last closed Binance Futures candle `2026-06-17 20:31`; Stage `06` must recompute this endpoint at build time.

Existing Roehub load paths: `RestFillRange1mUseCase` and scheduler `historical_backfill`/`scheduler_bootstrap` can fill explicit REST ranges with dedup/slicing and `RestInstrumentHistoryStartSource`; `apps/cli/commands/rest_catchup_1m.py` is tail/gap catch-up after seed, and `apps/cli/commands/backfill_1m.py` is file/parquet based, not the direct REST initial historical loader.

## Classic Producer Dependency Recheck

Current `strategy-producer-paper-testnet-trading-v1` ledger state:

| Stage | Status | Inventory relevance |
|---|---|---|
| `01` | `accepted` | Baseline foundation exists. |
| `02` | `accepted` | Backtest launch UI is accepted for classic strategy producer. |
| `03` | `accepted` | Scenario matrix/readiness foundation exists. |
| `04` | `accepted` | BTCUSDT market readiness accepted. |
| `05` | `blocked` | New Binance futures testnet connection validates and reads account-state, but BTCUSDT futures is `cross`/`20x` and USDT free balance is `0`; blockers are `insufficient_balance`, `margin_mode_mismatch`, `leverage_mismatch`. |
| `07` | `pending` | Required before RL paper integration Stage `15`. |
| `09` | `pending` | Required before RL testnet integration Stage `16`. |

This is a changed nuance versus older memory/context: the legacy/non-Transit custody blocker is resolved for the new Binance futures connection, but Stage `05` remains blocked on futures testnet account funding/config state. RL data/model/UI/monitor-only stages can continue when their own prerequisites pass; RL paper/testnet execution stages remain gated.

## Contract Impact

| Surface | Classification | Reason |
|---|---|---|
| Public API contract | `none` | No API changed. |
| Port contract | `none` | No port/interface changed. |
| DTO schema | `none` | No DTO changed. |
| Persisted schema | `none` | No migration or storage schema changed. |
| Config schema/defaults | `none` | No config changed. |
| Request hash / cache key / persistence identity | `none` | No runtime identity changed. |
| Benchmark / rollout gate | `none` | Inventory only; no benchmark threshold changed. |
| External side effects | `none` | Read-only HF metadata, ClickHouse queries, and filesystem manifest reads only. |
| Browser-visible behavior | `none` | No UI changed; browser runtime verification is out of scope and disabled by prompt. |
| Logs/metrics/traces/audit/ledger/report semantics | `compatible-change` | Adds Stage `02A` report/ledger evidence only. |
| Performance hot path | `none` | No runtime code or hot path changed. |

## Quality Gates

| Gate | Result |
|---|---|
| `shasum -a 256 .codex/agents/generated/rl-trading-agent-platform-v1/02a-data-source-inventory.md` | passed; hash recorded above |
| HF metadata byte-range inventory | passed; `_keys_map_` read for four NPZ files; no raw arrays committed |
| HF full temporary NPZ inspection amendment | passed; four NPZ files downloaded to `/tmp/roehub_hf_open_rl_trading_binance_dataset`, inspected with NumPy, hashes/counts/channel order recorded; no raw arrays committed |
| Mac Studio ClickHouse read-only coverage queries | passed; schema, aggregate coverage, gaps, and source tables recorded |
| Mac Studio artifact manifest read | passed; three BTCUSDT roots recorded |
| `python -m tools.docs.generate_docs_index --check` | passed after docs index regeneration; generated README reflects both this Stage `02A` report and the pre-existing unrelated Stage `06` report present in the worktree |

## Cold Self-Review

Mode: `cold self-review fallback`.

Reason: subagent tooling is present, but the active platform tool instruction allows spawning subagents only when the user explicitly asks for delegation/subagents. To satisfy the repository cold-head intent without violating the higher-priority tool rule, this stage used a local read-only cold review.

Result: `Release`.

Review checklist:

- Stage `02A` prerequisite Stage `01` is accepted.
- Prompt path/hash recorded.
- File manifest includes created/modified/deleted/outside expected paths.
- HF, ClickHouse, artifact, lifecycle, raw gaps and classic producer blocker evidence are separated by source.
- No feature-mask/trainable/blocked branch decision is hidden in this inventory.
- No secrets, provider payload dumps, raw arrays, model checkpoints or runtime ML artifacts are committed.
- Docs-only contract impact is classified as `compatible-change`; runtime/API/storage/config impact is `none`.
- Amendment facts distinguish HF train split size (`309` unique symbols) from Roehub `30/33` reference overlap.
- Full NPZ payloads are documented as temporary `/tmp` inspection only, not git/runtime artifacts.
- Plan, ledger, and future prompts align on `binance:futures` as the only v1 training source.
- Backfill windows are stated as signal windows plus `90` minute pre-signal and `60` minute post-signal source expansion.

## Blockers And Residual Risks

| Item | Status | Next action |
|---|---|---|
| Bybit `trades_count` | Not a v1 training blocker | Bybit branches are `blocked_not_training_source_v1`; reopen only through a separate accepted plan. |
| Futures funding/mark/index/leverage history | Open evidence gap | Stage `02B`/`05` must define source or explicit approximation before any futures activation. |
| Lifecycle listing/delisting windows | Open evidence gap | Stage `02B`/`06` must avoid survivorship bias by adding lifecycle source or blocking/inventory-marking unavailable windows. |
| Artifact coverage | Partial | Stage `05` can reuse existing BTCUSDT artifacts for OHLCV only; non-BTCUSDT and Bybit futures artifacts are absent in current production root. |
| Classic producer Stage `05` | Blocked | Operator must fund Binance Futures Testnet USDT and set BTCUSDT isolated `1x`; RL Stage `15`/`16` remain gated by classic Stage `07`/`09`. |

## Next-Stage Handoff

Stage `02B` is allowed to start from this local accepted inventory.

It must know:

- `ResearchRL/open-rl-trading-binance-dataset` full NPZ inspection has `478` unique symbols across all splits and `32,049` observed sessions; train split has `24,086` observed sessions and `309` unique symbols. The public card train count is `24,104`, so Stage `04` must record the 18-session mismatch.
- Roehub current distinct ref universe has `34` symbols, `33` tradable, and `30` symbols overlapping HF; this is a Roehub reference overlap fact, not a train-split pair count.
- V1 training source is `binance:futures` only. Current Roehub Binance Futures tradable universe is `BTCUSDT`, `ICPUSDT`, `NEARUSDT`, `TAOUSDT`, `WLDUSDT`, `ZECUSDT`; these have the article-compatible candle fields in ClickHouse through the current snapshot.
- Binance rows currently have `volume_quote` and `trades_count`; Bybit rows have `volume_quote` but no `trades_count`, and Bybit is `blocked_not_training_source_v1`.
- Existing artifacts are only `binance/spot/BTCUSDT`, `binance/futures/BTCUSDT`, and `bybit/spot/BTCUSDT`; each artifact price array is 5-column `ohlcv`, not article-compatible 7-channel features.
- `market_data` currently lacks funding, mark/index price, leverage-tier, point-in-time filters and explicit listing/delisting lifecycle tables.
- Existing repair paths cover canonical gap/tail backfill, scheduler/REST historical backfill, dedup and artifact rebuild, but not Bybit trade counts or futures metadata history.
- Classic producer Stage `05` is still blocked, but the current blocker is Binance Futures Testnet account funding/config (`insufficient_balance`, `margin_mode_mismatch`, `leverage_mismatch`), not missing readiness code and not the old active legacy-ciphertext issue for the new connection.
- Stage `02B` must produce the `binance:futures` feature/live-feed contract, Binance Futures metadata gate, and training-source matrix. Non-training branches must be recorded as `blocked_not_training_source_v1`.
