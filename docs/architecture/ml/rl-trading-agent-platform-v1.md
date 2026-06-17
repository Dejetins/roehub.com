# RL Trading Agent Platform v1

Статус: architecture plan для staged внедрения платформенного RL/ML trading agent на базе концепции D3QN/PER из `YuriyKolesnikov/rl-trading-binance`, но с посадкой в текущую архитектуру Roehub. Документ не является реализацией и не открывает mainnet execution сам по себе.

## Цель

Построить полный production-ready путь RL-стратегии:

1. доказать воспроизводимость подхода на внешнем HF dataset;
2. собрать Roehub-native dataset из `market_data` и backtest artifacts;
3. обучать общую платформенную модель на Mac Studio с CPU/GPU/MPS evidence;
4. вести model registry, dataset lineage, reproducibility hashes и per-ticker calibration;
5. показывать RL/ML стратегию в отдельной вкладке `/strategies`;
6. запускать сигналы в `monitor_only`, затем `paper`, `testnet`, и только после доказательств переходить к bounded `mainnet live`;
7. ограничивать количество активных live tickers по тарифу backend-лимитами без полноценного billing.

## Контекст

### Проверенные факты на момент дизайна

| Область | Наблюдение |
|---|---|
| Execution boundary | В Roehub уже есть `live_execution`, `ml_agent_decision`, source events, intents, risk gate, Redis dispatch, `exchange-execution`, order/fill/reconciliation ledgers. |
| Classic strategy producer | `docs/architecture/live_execution/strategy-producer-paper-testnet-trading-v1.md` является отдельным циклом для запусков classic strategies в `paper`/`testnet`; ledger сейчас `current_stage=05`, Stage `05` blocked на Binance futures testnet credential custody (`legacy/non-Transit ciphertexts`, fail-closed exchange-control). RL execution stages `15`/`16` blocked до classic Stage `07`/`09` after Stage `05` repair. |
| Market coverage | На Mac Studio `market_data.ref_market` содержит Binance/Bybit × spot/futures. |
| Canonical candles | `market_data.canonical_candles_1m` содержит `open/high/low/close`, `volume_base`, `volume_quote`, `trades_count`, taker volumes, source/ingestion metadata. |
| Artifact arrays | Текущий backtest artifact loader грузит `ohlcv.f32.npy` как 5 колонок OHLCV; этого недостаточно для полного 7-feature формата статьи без augment/enrich. |
| Data completeness snapshot | Binance spot/futures имеют `volume_quote` и `trades_count`; Bybit spot/futures имеют `volume_quote`, но `trades_count` сейчас отсутствует в canonical rows. Это требует Bybit enrich или feature-mask/model branch. |
| Runtime hardware | Единственный ML runtime target — Mac Studio M2 Max 64GB. PyTorch/MPS надо ставить в отдельный `uv` env/dependency group; основной API runtime не должен получать `torch`. |

### Связь с существующим plan-doc

RL-план не дублирует `strategy-producer-paper-testnet-trading-v1`.

| Уже делает classic strategy producer plan | Что делает RL-план |
|---|---|
| User launch из `/backtests` для classic variants | Отдельная RL/ML вкладка на `/strategies`, model/policy selection и active ticker slots. |
| Paper/testnet execution rails, source events, intents, testnet orders | Генерирует `ml_agent_decision` и использует те же rails после готовности classic producer/execution stages. |
| Supervised producer для classic strategy signals | Отдельный ML inference producer из-за PyTorch/MPS lifecycle; он не вызывает exchange SDK и не владеет секретами. |
| Manual entry/exit, journal, notification outbox base | RL signal journal/read-models должны быть совместимы и переиспользовать outcome/outbox паттерн. |
| 24h paper/testnet acceptance | RL получает свои 24h/7d gates поверх того же money boundary. |

## Бизнес-Смысл

Пользователь получает не инструмент обучения моделей, а готовую платформенную RL/ML стратегию. Он выбирает доступные тикеры в отдельной вкладке `/strategies`, настраивает risk/sizing policy и режим (`monitor_only`, `paper`, `testnet`, позже `live`). Количество активных live tickers ограничивается тарифом:

| План | Активные live tickers для RL strategy |
|---|---:|
| `Free` | 1 |
| `Pro` | 5 |
| `Premium` | 20 |
| `Enterprise` | custom/backend override |

Слот считается по активному live-режиму, а не по monitor/paper/testnet. Пользователь может свободно менять тикеры; освобождение слота происходит при остановке/выключении live ticker. Anti-rotation блокировка не вводится в v1, только audit и backend rate limit на частоту изменений при необходимости.

Current identity source of truth is not these product labels. Existing code exposes `paid_level` values `base|free|pro|ultra`; Keycloak/provider claims are not canonical entitlement source. Stage `12` must implement RL entitlement mapping explicitly:

| Current source | Product label | Default RL live ticker slots | Notes |
|---|---:|---:|---|
| `free` | `Free` | 1 | User-accepted v1 free plan. |
| `pro` | `Pro` | 5 | User-accepted v1 pro plan. |
| `ultra` | `Premium` | 20 | Product label differs from current enum; do not add `premium` enum without separate identity contract change. |
| `base` | internal/base | 0 until Stage `12` proves it should map to `Free` | Fail closed to avoid accidentally granting live RL slots from ambiguous legacy/base state. |
| per-account RL override | `Enterprise` | custom | Additive backend override table/use-case; no billing integration required in v1. |

## Охват

Входит:

- Binance и Bybit;
- `spot` и `futures`;
- общий платформенный RL model family;
- per-ticker/per-market calibration package: thresholds, normalization/calibration stats, optional ticker head only after evidence;
- HF dataset import как reproducibility baseline;
- Roehub-native dataset как acceptance basis;
- Mac Studio-only training/inference runtime;
- model registry и local artifact store;
- `/strategies` отдельная вкладка для RL/ML strategies;
- backend entitlement limits без billing/payment integration;
- staged rollout до mainnet live с отдельным approval gate.

Не входит:

- обучение пользователем своих моделей;
- cloud/S3/model hosting;
- прямой exchange SDK или secrets внутри ML worker;
- mainnet submit до завершения monitor/paper/testnet доказательств;
- замена `live_execution` или `exchange-execution`;
- полноценная биллинговая система;
- auto-config биржевого аккаунта без отдельного approval.

## Execution Boundary

Решение v1: каждый RL decision сначала становится durable `ml_agent_decision` source event. В `monitor_only` он завершается как `no_intent`. В `paper`/`testnet`/`live` intent создается только после user risk/sizing policy, tariff/quota check, account readiness и `live_execution` risk gate.

## RL Action, State И Reward Contract

V1 сохраняет семантику external repo, но переводит ее в Roehub money-boundary contract.

| Action id | Model action | Roehub meaning | Intent behavior |
|---:|---|---|---|
| `0` | `hold` | No-op для текущего RL strategy run. Если позиция этой стратегии уже открыта, она остается открытой; если позиции нет, стратегия остается flat. | Никогда не создает order intent. Пишется только decision/source-event/status. |
| `1` | `open_long` | Request to open long for this RL strategy run and ticker. | Может создать order intent только после risk/sizing/quota/account/readiness/risk-gate checks. |
| `2` | `open_short` | Request to open short for this RL strategy run and ticker. | Может создать order intent только если market/account supports short branch; spot-short без margin product блокируется как unsupported. |
| `3` | `close` | Request to close only the position owned by this RL strategy run for this exchange/market/ticker. | Не закрывает позиции пользователя от других стратегий или manual trades. |

Ownership invariant: несколько стратегий одного пользователя могут одновременно работать по одному ticker. Позиция, PnL, close/exit и risk limits считаются в scope `owner_user_id + strategy_run_id + exchange_name + market_type + symbol`. RL close action не имеет права закрыть позицию другой стратегии, даже если ticker совпадает.

V1 не добавляет pyramiding/scaling поверх логики external repo: `open_long`/`open_short` создают intent только когда в scope этой RL strategy позиции нет. Повторный same-side `open_*` при уже открытой позиции и opposite-side `open_*` до `close` превращаются в `no_intent` с audit reason `strategy_position_already_open`. `close` без strategy-owned позиции превращается в `no_intent` с reason `no_strategy_position`.

Training reward v1 повторяет external repo:

- reward считается как `pnl_change / initial_balance - inaction_penalty`;
- opening action учитывает fee как отрицательный `pnl_change`;
- closing action реализует trade PnL минус fee;
- `hold` при flat получает inaction penalty;
- `hold` при открытой позиции удерживает позицию и не получает mark-to-market reward;
- на последнем шаге training episode открытая позиция должна быть принудительно закрыта, а попытка открыть новую позицию в последний шаг превращается в `hold`;
- training reward не переписывается под Roehub risk score до отдельного accepted research stage.

Backtest/live distinction:

- offline backtest/evaluation считает realized trade outcomes, fees, slippage/funding policy и scorecard metrics;
- live execution не вычисляет reward как источник истины для денег; live outcome берется из order/fill/reconciliation ledgers;
- paper/testnet/live outcome может использоваться для monitoring, drift и evaluation ledgers;
- user-specific paper/testnet/live outcomes не попадают в platform-wide retraining dataset в v1. Использование таких outcomes для обучения требует отдельного accepted governance contract: source scope, redaction, owner isolation, opt-out/consent/product policy, no PII/secrets и lineage marker `source=platform_live_outcome`.

## Latency Budget

Safety boundary не должен добавлять заметную задержку в live execution path. Целевой runtime path держит модель загруженной в памяти, live feature window обновляет на закрытых 1m candles, а `source_event` и optional `intent` записывает коротким durable write path без тяжелой ML/ClickHouse работы в момент исполнения.

Начальные target budgets являются рабочими ориентирами до Stage `13`/`17` measurements:

| Segment | Target before evidence | Rule |
|---|---:|---|
| Candle close -> feature window ready | p95 <= 250 ms | No full ClickHouse scan on hot path; repair only on gap/degraded state. |
| Feature window -> model decision | p95 <= 100 ms | Model preloaded; no checkpoint load per decision. |
| Decision -> `source_event` persisted | p95 <= 50 ms | Durable audit insert must stay lightweight. |
| `source_event` -> accepted/rejected `intent` | p95 <= 100 ms | Risk/sizing/quota checks must be local and bounded. |
| Accepted `intent` -> Redis dispatch | p95 <= 100 ms | Existing dispatch path; no exchange call in producer. |
| Signal decision -> exchange submit attempt | p95 <= 750 ms before provider latency | Stage `16`/`17` must record real testnet evidence. |

If measurements exceed budget, the fix is not to remove audit/risk gates; the fix is to remove hot-path blocking work, cache readiness/quota/model state, or split slow repair/enrichment into degraded async paths.

## Mac Studio Resource Isolation

Mac Studio M2 Max 64GB является единственным ML host, поэтому training не должен деградировать live inference или existing backtest/runtime workers.

| Area | V1 rule |
|---|---|
| Training schedule | Manual/scheduled training windows are disabled during Stage `20` live canary unless an operator explicitly enables a bounded window with resource evidence. |
| Inference | Active inference keeps model preloaded, has a memory/RSS cap, health metric, restart policy and degraded mode when model cannot be loaded. |
| MPS | Stage `03`/`07` benchmark CPU vs MPS; accepted config records device policy and fallback. MPS training cannot starve inference. |
| CPU threads | Trainer, backtest/evaluator and inference process set bounded CPU/thread counts; Stage `17` records p95 latency and RSS under concurrent load. |
| Backtest jobs | RL training cannot consume the whole host while backtest/artifact publisher jobs are running; scheduler either serializes heavy jobs or enforces concurrency limits. |
| Disk | Training blocks before disk watermark; artifact cleanup cannot delete referenced accepted artifacts. |
| Evidence | Stage reports include resource usage (`rss`, CPU, MPS availability, wall-clock, queue lag) and explicit go/no-go for the next mode. |

## Целевая Архитектура

```mermaid
flowchart LR
    UI["/strategies RL/ML tab"]
    API["apps/api"]
    Entitlements["identity / entitlement limits"]
    MarketData["market_data canonical + Redis live feed"]
    Artifacts["backtest_artifacts .npy/.manifest"]
    Dataset["rl dataset builder"]
    Trainer["rl_trading trainer on Mac Studio"]
    Registry["model registry + local artifact store"]
    Inference["rl_trading inference producer"]
    LiveExec["live_execution ingress/risk gate"]
    Redis["Redis execution streams"]
    ExchangeExec["exchange-execution"]
    Exchanges["Binance / Bybit"]
    Metrics["Prometheus / Monit"]

    UI --> API
    API --> Entitlements
    API --> Registry
    MarketData --> Dataset
    Artifacts --> Dataset
    Dataset --> Trainer
    Trainer --> Registry
    Registry --> Inference
    MarketData --> Inference
    Inference --> LiveExec
    LiveExec --> Redis
    Redis --> ExchangeExec
    ExchangeExec --> Exchanges
    Trainer --> Metrics
    Inference --> Metrics
    ExchangeExec --> Metrics
```

## Направление Зависимостей

| Слой | Может зависеть от | Не должен зависеть от |
|---|---|---|
| `rl_trading` domain/application | model policy, feature contract, calibration, dataset/model metadata, ports | FastAPI, ClickHouse driver, Redis client, exchange SDK, API secrets |
| Dataset builder adapters | ClickHouse canonical reader, artifact array loader, HF importer | execution intents/orders |
| Trainer app | PyTorch, NumPy, dataset artifacts, model registry writer | web UI, exchange SDK, secrets |
| Inference producer | model registry reader, Redis/canonical market feed, entitlement/risk config, `live_execution` ACL | raw exchange credentials, direct order adapters |
| `live_execution` | `ml_agent_decision` source event, risk context, order model | ML internal tensors/checkpoints |
| API/UI | model/read-model/entitlement use cases | PyTorch model execution, plaintext secrets |

## Контексты И Модули

Плановая target structure:

| Area | Planned path / artifact | Role |
|---|---|---|
| Domain/application | `src/trading/contexts/rl_trading/` | Feature contracts, dataset/model metadata, policy decisions, calibration, quota decisions. |
| Trainer app | `apps/worker/rl_trading_trainer/` | Offline training/evaluation jobs on Mac Studio. |
| Inference producer | `apps/worker/rl_trading_inference/` | Supervised runtime producer for monitor/paper/testnet/live decisions. |
| API routes | `apps/api/routes/ui_rl_strategies.py` or extension under strategies routes | UI read models, active ticker config, model registry summaries. |
| UI tab | `apps/web/templates/pages/strategies.html`, `apps/web/dist/js/pages/strategies.js`, CSS/locales | Separate RL/ML tab on `/strategies`. |
| Migrations | `alembic/versions/...rl_trading...py` | Additive Postgres metadata tables. |
| Local artifact store | `/opt/roehub/state/rl_trading/` | Datasets, checkpoints, training/eval reports, calibration packs. |
| Docs/ledger | `docs/architecture/ml/rl-trading-agent-platform-v1-stage-reports/` | Stage reports and iteration ledger. |

## Данные И Feature Contract

### Minimal article-compatible features

Для полного соответствия статье и HF baseline нужен 7-feature набор:

| Feature | Roehub source | Status |
|---|---|---|
| `open` | `canonical_candles_1m.open` or `ohlcv[:, 0]` | available |
| `high` | `canonical_candles_1m.high` or `ohlcv[:, 1]` | available |
| `low` | `canonical_candles_1m.low` or `ohlcv[:, 2]` | available |
| `close` | `canonical_candles_1m.close` or `ohlcv[:, 3]` | available |
| `volume` | `canonical_candles_1m.volume_base` or `ohlcv[:, 4]` | available |
| `vwap` | `volume_quote / volume_base`, fallback close if zero-volume policy accepts | available when `volume_quote` exists |
| `num_trades` | `canonical_candles_1m.trades_count` | available for Binance, missing for current Bybit rows |

Для Roehub production dataset дополнительно нужны:

- `exchange_name`, `market_type`, `symbol`, `instrument_key`;
- `ts_open`, `ts_close`;
- fee/slippage/funding config used by training/backtest reward;
- instrument filters: min notional, precision, qty step;
- instrument lifecycle windows: listing start/end, delisting/inactive state, and exchange availability gaps, so train/backtest splits do not get survivorship bias;
- futures-only inputs: funding-rate history, mark/index price source, margin/leverage tier metadata, and liquidation-risk assumptions;
- dataset split manifest and leak-check report;
- source table/artifact hashes and query bounds;
- feature availability mask for markets where `trades_count` missing.

### Dataset strategy

1. HF dataset используется только как external reproducibility baseline.
2. Acceptance model quality строится на Roehub-native dataset.
3. Existing `.npy` artifact arrays можно использовать как fast OHLCV source, но v1 RL feature dataset обязан augment-ить `vwap` и `num_trades` из ClickHouse или materialize separate RL feature artifacts.
4. Для Bybit v1 есть два допустимых пути:
   - `preferred`: enrich Bybit `trades_count` из внешнего historical source;
   - `fallback`: отдельная feature-mask/model-calibration ветка без `num_trades`, если backtest evidence показывает приемлемое качество.
5. Futures activation нельзя считать полной, пока Stage `02A`/`02B`/`05` не докажут funding/fee/slippage/contract-spec coverage или явно не зафиксируют accepted approximation для раннего research-only этапа.
6. Splits строятся по времени и instrument lifecycle, а не только по активным сегодня symbols; delisted/inactive intervals должны попадать в inventory как known missing/known unavailable, если исторические данные недоступны.

### Futures Metadata Gate

Futures branch получает статус `trainable`/`backtestable` только если Stage `02A`/`02B` доказывают point-in-time coverage или явно ограничивают branch как `research_only_approximation`.

| Required input | Acceptance rule |
|---|---|
| Funding | Point-in-time funding history by exchange/market/symbol, with exact timestamp alignment and missing-period policy. |
| Fees/slippage | Maker/taker fee policy, commission tier assumption, spread/slippage/liquidity cap used by simulator and scorecard. |
| Mark/index price | Mark/index source or explicit approximation; liquidation/risk assumptions cannot use close price silently. |
| Instrument filters | `price_step`, `qty_step`, precision, min notional and contract size must be non-null or branch blocks. |
| Leverage/margin | Leverage tier, isolated/cross assumption, maintenance margin and liquidation approximation must be versioned in feature/backtest contract. |
| Acceptance status | Missing field can produce only `blocked`, `feature-mask`, or `research_only_approximation`; it cannot silently become production `trainable`. |

### Train/live feature parity

Feature generation должен быть общим контрактом, а не двумя похожими реализациями.

| Rule | Requirement |
|---|---|
| Shared builder | Dataset builder и live inference producer используют один feature-contract модуль или один набор pure functions для channel order, normalization, action-history extras и metadata. |
| Golden windows | Stage `05`/`13` создают golden fixtures: один и тот же candle window из ClickHouse/artifact должен давать одинаковый feature vector в offline dataset и live inference path. |
| Tolerance | Для `float32` normalized features acceptance через `np.testing.assert_allclose(..., rtol=1e-6, atol=1e-6)`; если MPS/NumPy path требует иной tolerance, Stage report обязан доказать причину и зафиксировать ее в feature contract. |
| Source ordering | Channel order фиксируется как `open, high, vwap, low, close, volume, num_trades` для article-compatible mode. Любая альтернативная ветка (`feature-mask`) получает отдельный `feature_contract_hash`. |
| Live gaps | Live inference читает Redis closed candles для hot path и допускает read-only ClickHouse canonical repair при gaps по паттерну strategy live-runner; full ClickHouse scan на hot path запрещен. |
| Drift guard | Feature stats в live сравниваются с promotion baseline; drift создает retraining candidate task/alert, но не меняет active model автоматически. |

Live feed feature gate:

| Decision path | Requirement |
|---|---|
| Preferred | Stage `02B`/`05` расширяет live feature window так, чтобы `trades_count` был доступен на hot path вместе с `open/high/low/close/volume_base/volume_quote`; `vwap` считается из `volume_quote / volume_base` только при валидном `volume_base > 0`. |
| Fallback | Если `num_trades` недоступен для exchange/market branch, Stage `02B` создает отдельный `feature-mask` contract/model branch without `num_trades`; такой branch не совместим по hash с article-compatible model. |
| Block | Market branch блокируется как `blocked`, если нельзя доказать train/live feature parity без тяжелого ClickHouse repair на обычном hot path. |
| Stage `13` acceptance | `monitor_only` inference не accepted, пока Redis/live feature window и offline dataset fixture не дают идентичный feature vector для одного candle window в пределах tolerance. |

### Session Extraction Policy

Stage `06` должен сначала воспроизвести подход статьи/repo максимально близко, затем адаптировать его под Roehub-native universe только после evidence.

| Area | V1 decision |
|---|---|
| Initial universe | Начинаем с HF tickers и тех, что доступны в `market_data`; расширение universe фиксируется отдельным Stage `02A`/`06` решением. |
| Window shape | Article-compatible default: `full_seq_len=150`, `pre_signal_len=90`, `post_signal_len=60`, `agent_history_len=30`, `agent_session_len=10`; demo config может использовать shorter path только как explicit research mode. |
| High-volatility rule | Сначала повторить repo/article extractor criteria максимально близко; если точный алгоритм не восстановлен из article/HF artifacts, Stage `06` фиксирует observed proxy и сравнение распределений с HF dataset. |
| Overlap | Overlapping sessions разрешены внутри одного split для увеличения sample count. Между train/val/test/backtest запрещен leakage: time-based split, instrument lifecycle bounds и embargo не меньше максимального `full_seq_len` вокруг split boundary. |
| Listing/delisting | Session extractor не строит окна вне instrument lifecycle. Missing lifecycle metadata блокирует market branch или помечает его `feature-mask/blocked` в activation matrix. |
| Keys | Session key включает `exchange_name`, `market_type`, `symbol`, `instrument_key`, `signal_ts_open`, `split`, `feature_contract_hash`. |
| Audit | Stage report сохраняет counts by split/ticker/market, rejected-window reasons, overlap rate, gap rate и distribution comparison with HF baseline. |

## Model Registry И Local Artifact Store

Локальное хранение на Mac Studio:

```text
/opt/roehub/state/rl_trading/
  datasets/{dataset_id}/
    manifest.json
    train.npz
    val.npz
    test.npz
    backtest.npz
    feature_stats.npz
    leak_check_report.json
  training_runs/{run_id}/
    config.json
    metrics.json
    logs.jsonl
    resource_usage.json
    promotion_decision.json
  models/{model_version_id}/
    checkpoint.pt
    model_config.json
    normalization.npz
    calibration.json
    metrics.json
    hashes.json
    rollback_manifest.json
  qval_cache/{model_version_id}/
```

Postgres хранит metadata, paths, hashes, owner/scope, status, metrics summary и activation state. Checkpoints и datasets не хранятся в БД и не попадают в git.

### Registry state machine

Stage `09` обязан реализовать state machine и invariant tests, а не только таблицы metadata.

| Entity | Allowed states |
|---|---|
| `dataset_version` | `building -> qa_failed | accepted | missing_artifact | superseded` |
| `training_run` | `planned -> running -> failed | completed | rejected | candidate` |
| `model_version` | `candidate -> rejected | accepted_champion | rollback_candidate | missing_artifact` |
| `calibration_pack` | `candidate -> accepted | rejected | superseded | missing_artifact` |
| `activation` | `inactive -> shadow -> monitor_only -> paper -> testnet -> live -> paused -> rolled_back` |

Required invariants:

- only one `accepted_champion` can be active per platform model family and feature contract scope;
- runtime inference can load only `accepted_champion` or explicitly selected rollback version;
- `candidate`, `rejected`, `failed`, `qa_failed`, `building` and `missing_artifact` cannot be runtime-loaded;
- activation requires matching `feature_contract_hash`, `dataset_hash`, `model_version_id`, `calibration_pack_hash` and exchange/market activation matrix;
- rollback changes activation pointer but does not delete current/rejected artifacts;
- `missing_artifact` blocks activation and requires restore/rollback evidence before mode can advance.

### Artifact operations

Основной путь v1: `/opt/roehub/state/rl_trading/`. Cloud/S3 storage не входит в v1. Local backup/restore добавляется отдельным Stage `09B`; если нет второго диска, Stage `19`/`21` обязаны явно принять residual single-host disk risk.

| Area | V1 rule |
|---|---|
| Atomic writes | Все datasets/checkpoints/reports пишутся во временный path, затем atomic rename; Postgres metadata обновляется только после hash validation. |
| Retention | Accepted champions, active calibration packs, rollback manifests и source manifests хранятся без автоматического удаления. Rejected/incomplete training runs удаляются через configurable `rejected_run_retention_days`; Stage `09` должен запретить prod startup без явного значения. |
| Disk quota | Stage `09` вводит local quota/watermark metrics for `/opt/roehub/state/rl_trading/`; new training run blocks before disk exhaustion. |
| Cleanup | Cleanup удаляет только artifacts со статусом `rejected|incomplete|superseded` after retention and only when no active metadata reference exists. |
| Metadata after missing artifact | Если Postgres ссылается на missing artifact, model/dataset status становится `missing_artifact`, activation blocks, UI показывает degraded reason. |
| Backup | Stage `09B` создает local backup path for accepted champion, calibration pack, source manifests and registry metadata dump. Если backup path на том же physical disk, это только corruption/operator-error protection, не disaster recovery. |
| Restore drill | Stage `09B` выполняет restore drill в отдельный restore path, hash validation после restore и rollback to previous local accepted champion. |
| Git/docs hygiene | Datasets/checkpoints/log dumps never enter git/docs; docs contain only sanitized summaries and hashes. |

### Checkpoint security

PyTorch checkpoints считаются trusted local artifacts, а не пользовательским upload.

| Rule | Requirement |
|---|---|
| No user upload | V1 не принимает model/checkpoint upload от пользователя. Все checkpoints создаются только Roehub trainer service. |
| Hash before load | Inference producer загружает checkpoint только после Postgres metadata lookup, file existence check и sha256 validation. |
| Accepted states only | Load разрешен только для `accepted_champion` или explicitly selected rollback version. Candidate/rejected/incomplete checkpoints не загружаются в runtime inference. |
| Safer load | Где поддерживается установленной PyTorch версией, использовать `torch.load(..., weights_only=True)` или эквивалентный safe loading path; fallback должен быть зафиксирован в Stage `03`/`09` evidence. |
| Path safety | Registry paths canonicalized under `/opt/roehub/state/rl_trading/`; symlink/path traversal outside store blocks load. |
| Audit | Every model load writes audit/metric with model_version_id, calibration_id, checkpoint hash, code/config hash, device, result. |

## Training, Retraining И Promotion Lifecycle

Training/retraining является платформенным offline-процессом, пользователь его не запускает и не настраивает напрямую.

Минимальный v1 lifecycle:

1. `dataset_version` создается из ClickHouse/artifacts с deterministic manifest, hashes, split policy и feature availability mask.
2. `training_run` создает candidate model от frozen config. Для дообучения допустимы два режима:
   - `full_retrain`: полный replay на новом dataset_version;
   - `fine_tune`: продолжение от accepted champion checkpoint только если config/hash compatibility явно подтверждена.
3. `candidate` никогда не активируется автоматически. Promotion требует:
   - положительный Roehub backtest после fees/slippage/funding policy;
   - per-ticker/per-market calibration report;
   - latency/resource evidence на Mac Studio;
   - rollback_manifest;
   - stage report + ledger update.
4. `champion` модель platform-wide. Per-ticker поведение задается calibration/weights/thresholds/head metadata, но не пользовательскими training jobs.
5. `challenger` может работать только в `monitor_only` shadow mode, пока Stage `13+` не подтвердит signal parity, latency и drift evidence.
6. Drift monitoring сравнивает live feature distribution, action distribution, skipped-action reasons и paper/testnet/live outcomes с promotion baseline. Drift сам по себе не обновляет модель; он создает retraining candidate task.
7. Rollback всегда переводит активную модель/калибровку на предыдущий accepted champion без удаления audit trail.

Retraining triggers/cadence v1:

| Trigger | Behavior |
|---|---|
| Manual operator/admin action | Доступно сразу через host-local backend/admin command; RL/ML UI action control для authorized operators доставляется в Stage `11` только после server-side operator/admin authorization guard. |
| Scheduled retraining | Доступно сразу как disabled-by-default supervised schedule; включается config flag и пишет planned run в registry before execution. |
| Drift trigger | Drift создает alert/task и может автоматически запустить candidate training run, но не может auto-promote модель. |
| New dataset version | Создает retraining candidate task after dataset QA acceptance. |
| Promotion approval | Auto-approval допустим после всех gates, но activation champion требует operator/admin confirmation action and audit row. |
| Rollback | Доступен сразу через host-local backend/admin command + runbook; RL/ML UI rollback control доставляется в Stage `11` только после server-side operator/admin authorization guard; rollback не удаляет rejected/current artifacts. |

Operator authority v1:

| Surface | Rule |
|---|---|
| Stage `10A` backend command | Host-local CLI/runbook on Mac Studio under operator shell access. It may call application use cases directly and must write operator/source/audit metadata. |
| Stage `10A` web/API mutation | Not exposed to normal owner-scoped users unless Stage `10A` proves an existing server-side admin primitive. If no primitive exists, the API mutation remains internal-only until Stage `11`. |
| Stage `11` UI controls | UI action controls for retrain/promote/rollback require server-side operator/admin authorization, not only hidden buttons or frontend state. If no admin guard is implemented, controls render disabled/read-only with a blocked reason. |
| Ordinary user scope | Users can configure their own RL ticker activations and risk/sizing policy within entitlement limits, but cannot train, promote, rollback, or activate platform champion models. |
| Audit | Every retrain/promotion/rollback action records operator identity/source, model hash, calibration hash, reason, previous champion, next champion and rollback reference. |

### Promotion scorecard

Stage `08`/`10A` не принимает candidate только по одному числу PnL.

V1 разделяет уровни допуска:

| Level | Meaning | Gate |
|---|---|---|
| `research_candidate` | Модель может быть сохранена и анализироваться offline. | Stage `08`: positive PnL after costs плюс полный scorecard и sanity baselines. |
| `promotion_grade_candidate` | Модель может идти в `monitor_only`/`paper`/`testnet` pipeline. | Stage `10A`: versioned numeric threshold profile approved by operator/admin confirmation. |
| `live_candidate` | Модель может попасть в bounded mainnet canary. | Stage `19`: отдельный go/no-go review, incident drills, legal/product checklist and explicit approval. |

Stage `10A` обязан сохранить numeric threshold profile рядом с promotion decision. Конкретные числа задаются в stage report/config после evidence, но сам profile обязателен и должен включать минимум:

| Threshold area | Required field |
|---|---|
| Trade sufficiency | `min_total_trades`, `min_trades_per_split`, `min_trades_per_active_ticker_market` or explicit low-trade rejection. |
| Drawdown | `max_drawdown_pct` and worst-period loss limit. |
| Stability | per ticker/month/market pass/fail, not only aggregate PnL. |
| Concentration | max share of total PnL from one ticker, one month/period or one session bucket. |
| Baseline sanity | random/no-trade/simple heuristic cannot materially dominate candidate without blocking investigation. |
| Cost realism | fees, slippage and funding must already be inside net PnL. |
| Uncertainty | bootstrap/confidence interval or explicit uncertainty band for key metrics. |

Mandatory scorecard fields:

| Area | Required scorecard fields |
|---|---|
| Profitability | realized PnL after fee/slippage/funding policy, per split and aggregate. |
| Risk | max drawdown, downside distribution, loss streak, worst ticker/session, exposure time. |
| Trade sufficiency | total trades, trades by ticker/market/direction, skipped/blocked action counts. |
| Stability | metrics by ticker, market type, exchange, month/period and volatility bucket. |
| Out-of-sample | separate validation/test/backtest periods with source bounds and no leakage proof. |
| Overfit checks | train vs val/test/backtest gap, action distribution shift, suspicious concentration by ticker/session. |
| Runtime | CPU/MPS inference latency, memory/RSS, model load time, feature generation latency. |
| Operations | artifact hashes, rollback manifest, rejected/accepted reason, operator confirmation if promoted. |

### ML sanity baselines

Sanity baselines не являются пользовательским benchmark и не меняют RL policy. Они нужны, чтобы поймать ошибки симулятора, reward shaping, feature leakage или backtest accounting.

Stage `08` должен сохранять рядом с candidate evaluation:

- `no_trade/hold` baseline;
- random valid-action baseline with fixed seed;
- simple threshold heuristic baseline from price/volatility movement;
- optional external repo CNN classifier baseline только если Stage `04` воспроизвел его без scope creep.

Baseline outputs хранятся в model registry/evaluation artifact как diagnostic metadata. Candidate promotion не требует “обыграть baseline” как business rule на первом этапе, но аномалия вроде random/no-trade materially better than candidate становится blocker для promotion investigation.

### Simulator/accounting parity

RL backtest должен доказывать совместимость не только с reward simulation, но и с Roehub accounting/execution ledgers.

Required parity ladder:

1. Stage `08` creates a deterministic decision-sequence fixture: same candle window, same feature vectors, same model/action sequence.
2. Offline simulator computes realized PnL, fees, slippage/funding and position transitions from that sequence.
3. Roehub paper/execution accounting model replays the same accepted intents without exchange submit and reconciles within documented tolerance.
4. Stage `15` completes paper parity by running the same or equivalent decision sequence through paper execution ledgers and comparing outcome to Stage `08` simulator/accounting artifact.

If offline simulator outcome and paper/execution ledger accounting diverge beyond tolerance, the model cannot advance beyond research candidate regardless of PnL.

Обязательные reproducibility поля:

- `dataset_id`, `dataset_hash`, source bounds, source query hash;
- `code_commit_sha`, dirty-state flag for research runs;
- `training_config_hash`, `feature_contract_hash`;
- `random_seed`, Python/PyTorch/NumPy versions, device (`cpu|mps`);
- checkpoint sha256;
- metrics summary and full report path;
- calibration pack hash;
- accepted/rejected status with reason.

## Risk/Sizing Policy

Пользователь настраивает risk/sizing, но модель не получает право обходить `live_execution`.

Minimum v1 controls:

- quote allocation per ticker/position;
- max position notional;
- max daily loss;
- max trades per day;
- max concurrent positions;
- long-only / short-only / long-short;
- futures leverage cap;
- stop-loss/take-profit/trailing policy as platform-side risk layer;
- emergency stop/kill switch;
- per-ticker enable/disable;
- mode: `monitor_only|paper|testnet|live`.

Баланс доходность/risk должен быть регулируемым через policy profile: `conservative`, `balanced`, `aggressive`, плюс advanced custom fields. Модель предлагает decision; risk/sizing policy переводит decision в order model или блокирует.

TP/SL/trailing v1 contract:

- `stop-loss`, `take-profit` and `trailing` are platform-side synthetic risk/exits, not native exchange OCO/TP/SL/trailing order fields;
- current `ExecutionOrderModelRequest` accepts only simple `market|limit` order intents for execution; advanced order fields are rejected by the existing execution boundary;
- synthetic exits create separate close intents only after strategy-owned position lookup, risk gate and idempotency checks;
- executor must not pass `take_profit`, `stop_loss`, `trailing`, `oco`, `amend_replace` or multi-leg fields into the exchange order model until a separate live-execution contract explicitly supports them.

Safe modes and incident actions:

| Mode | Meaning |
|---|---|
| `NORMAL` | Decisions may become intents after all risk/readiness checks. |
| `REDUCE_ONLY` | No new exposure; only closes/reductions for RL-owned positions are allowed. |
| `CANCEL_ALL` | Cancel pending RL-owned intents/orders where exchange state is known or reconciled. |
| `FLATTEN` | Close only RL-owned positions when configured and safe; never close other strategies/manual positions. |
| `HALT` | Stop new decisions/intents and require operator review. |

Kill switch drill before mainnet:

- global, user and per-strategy kill switches;
- stop new decisions and intents;
- cancel pending RL-owned intents/orders where safe;
- optional close of RL-owned position only when configured;
- reconcile unknown exchange state before retry/cancel/close;
- UI shows terminal/degraded reason and audit trail.

Stage `18` must prove this drill before Stage `19` can start.

## Тарифы И Enforcement

Backend-only v1 без billing:

| Contract | Rule |
|---|---|
| Plan source | Existing account/profile plan value first; if unavailable, default `Pro` only in dev/test, not silent prod entitlement. |
| Counted unit | Distinct active live RL ticker: `(owner_user_id, exchange_name, market_type, symbol)`. |
| Counted modes | Only `live`. `monitor_only`, `paper`, `testnet` do not consume live ticker slots. |
| Entitlement source | Existing Roehub identity `paid_level` plus optional RL entitlement override table; no browser/provider claim is trusted as canonical. |
| Default mapping | `free=1`, `pro=5`, `ultra=20`, `base=0/fail-closed until Stage 12 decision`, override=`custom`. |
| Free changes | User may freely stop one ticker and start another; audit event records change. |
| Enforcement point | API/use-case before activating live ticker and inference producer before creating live intent. |
| UI | `/strategies` RL tab shows used/limit and blocked reason before submit. |
| Bypass prevention | Database uniqueness for active ticker rows plus transactional quota check. |

## Сервисные Обращения

| Caller | Callee | Стиль | Contract | Timeout / retry | Failure behavior |
|---|---|---|---|---|---|
| UI | `apps/api` | HTTP | RL tab config, model registry summaries, active ticker CRUD, mode changes | Existing frontend timeout; no hidden retry for mode/live activation | Show stable blocked reason |
| API | RL application use cases | In-process | Validate ticker slots, policy config, model availability | DB transaction timeout; idempotency for activate/deactivate | Return typed error, no partial activation |
| Dataset builder | ClickHouse canonical | Read-only SQL | Feature extraction from `market_data.canonical_candles_1m` | Bounded chunking; no concurrent-session misuse | Stage blocked if coverage or feature availability missing |
| Dataset builder | Artifact arrays | Filesystem mmap | Fast OHLCV source from pinned manifests | Fail-fast hash/manifest validation | No fallback to unpinned arrays |
| Trainer | Model registry | Filesystem + Postgres metadata | Write dataset/run/model artifacts and hashes | Atomic write temp->rename; metadata after file hash | Failed run stays rejected/incomplete |
| Host-local operator command / authorized UI | Model registry and promotion use cases | CLI first; guarded HTTP later | Retrain, promote, rollback candidate/champion state | Idempotent by run/model/calibration hash and confirmation id | No activation if operator authority is missing or ambiguous |
| Inference producer | Redis/canonical market feed | Async/read | Closed 1m candles and gap repair | Backoff; no busy loop | Signal generation pauses with observable degraded reason |
| Inference producer | `live_execution` ACL | In-process/service ACL | `ml_agent_decision` source event and optional intent | Dedupe by source decision key before retry | Unknown write state requires lookup before retry |
| `live_execution` | Redis/exchange execution | Existing async path | Accepted intents only | Existing retry/DLQ/reconciliation | No blind exchange retry after unknown provider state |

## Ошибки, Retry И Idempotency

| Risk | Rule |
|---|---|
| Duplicate RL decision | Idempotency key includes `model_version_id`, `calibration_id`, `owner_user_id`, `exchange`, `market_type`, `symbol`, `bar_ts_open`, `action`. |
| Duplicate live activation | Transactional active ticker uniqueness plus idempotent activate request. |
| Unknown source event write | Read by idempotency key before retry. |
| Unknown exchange submit | Existing `exchange-execution` rule applies: reconcile/provider lookup before retry. |
| Feature gap | Stage blocks dataset/model activation; no implicit fill unless feature contract defines it. |
| Missing model/calibration | Inference emits degraded status, no intent. |
| Quota exceeded | API blocks activation; inference blocks live intent if active quota state is inconsistent. |
| PyTorch/MPS failure | Trainer falls back only when benchmark stage accepts CPU fallback; inference uses last accepted model or pauses. |

## Логирование И Redaction

Можно логировать:

- model/version/calibration ids;
- dataset ids and hashes;
- non-sensitive ticker/mode/status/reason;
- aggregate metrics and latency;
- decision category and confidence bucket.

Нельзя логировать:

- API secrets, tokens, cookies, OpenBao/transit data;
- raw signed provider payloads;
- raw exchange credentials;
- user PII beyond stable internal owner id where needed;
- unredacted provider order payloads in stage reports;
- full model tensors/checkpoint binary content in logs/docs.

## Monitoring И Alerts

| Signal | Severity | Required evidence |
|---|---|---|
| Trainer job failed | warning | run status, error class, resource usage |
| Inference producer down | critical for enabled runtime | Monit/launchd + Prometheus `up` |
| No live market feed for active ticker | warning/critical by mode | freshness metric and UI degraded state |
| Model registry hash mismatch | critical | registry validation failure, no activation |
| Model artifact missing/corrupt | critical for active model | hash check failure, UI degraded state, rollback path |
| Quota enforcement mismatch | critical | API and producer disagree on active live slots |
| Mainnet intent attempted before approval | critical | counter must stay zero before mainnet stage |
| Unknown unreconciled order | critical | existing execution runbook behavior |
| MPS/CPU benchmark regression | warning | wall-clock/RSS/device report |
| RL artifact disk quota high | warning/critical | local store watermark and blocked new training runs |
| Drift trigger fired | warning | drift task/alert, no auto-promotion |
| Feature/schema parity violation | critical | train/live golden fixture mismatch, activation blocked |
| Model action anomaly | warning/critical | action distribution, entropy/confidence bucket and skipped-action reasons vs promotion baseline |
| Trading quality anomaly | warning/critical | exposure, leverage, turnover, fill rate, reject rate, realized slippage and funding |
| Risk limit breach | critical | drawdown, daily loss, VaR/CVaR proxy, kill-switch/safe-mode state |
| Governance drift | warning | active model/calibration/config version, last approval, rollback pointer and operator audit |
| Training/inference resource contention | warning/critical | inference p95/RSS/CPU/MPS regression while training/backtest jobs are active |

## План Внедрения

Stages are grouped so data/model work can proceed before classic strategy producer delivery unblocks, while execution stages depend on the existing paper/testnet plan.

| Stage | Name | Purpose | Dependency | Acceptance evidence |
|---|---|---|---|---|
| `01` | Baseline and plan freeze | Freeze current ML/execution/data/user constraints, create ledger. | none | plan + ledger, ClickHouse feature snapshot, dirty-worktree note, docs index. |
| `02A` | Data source inventory | Inventory HF tickers, Roehub ClickHouse coverage, artifact arrays, exchange/market coverage, instrument lifecycle, raw gaps and current classic producer blocker state. | `01` | SQL coverage, HF ticker manifest, artifact manifests, lifecycle/gap report, classic Stage `05` blocker recorded, no feature contract decisions hidden. |
| `02B` | Feature and live-feed contract | Freeze article-compatible vs feature-mask branches, channel order, missing fields, futures metadata gate and Redis/live feature hot-path decision. | `02A` | feature contract hash, Binance/Bybit × spot/futures activation matrix `trainable|blocked|feature-mask|research_only_approximation`, live `trades_count`/feature-mask/block decision, no hot-path full ClickHouse scan. |
| `02C` | Action/state/reward contract | Freeze RL environment semantics, action/state/reward mapping, position ownership, close scope and external-repo-compatible fixtures. | `02B` | action/reward/state contract tests planned, no-pyramiding/no-cross-strategy-close fixtures, reward compatibility notes. |
| `03` | Mac Studio ML environment | Create isolated `uv` ML env with PyTorch, CPU/MPS smoke, resource isolation policy and no API runtime dependency. | `02C` | `torch.backends.mps.is_available`, CPU/MPS microbenchmark, RSS/thread report, accepted device/fallback policy. |
| `04` | External repo/HF reproducibility | Import external repo concept safely and reproduce a small HF train/eval/backtest baseline. | `03` | dataset hash, run config hash, metrics, no vendored code without attribution/license note. |
| `05` | Roehub dataset builder v1 | Build raw feature slabs/manifests and golden fixtures from canonical/artifacts for branches allowed by Stage `02B`; explicitly block incomplete branches. Stage `05` does not emit final accepted trainable/sessionized datasets. | `04` | raw slab manifests, feature stats, deterministic rebuild hash, offline/live feature golden fixtures, live-feed feature parity decision implemented, no accepted sessionized training artifact yet. |
| `06` | Dataset QA and session extractor | Implement high-volatility session extraction and data QA; emit accepted sessionized train/val/test/backtest datasets. | `05` | sessionized dataset hashes, session counts, gap report, machine-readable leakage/embargo report, no look-ahead/survivorship-bias proof, reproducible split. |
| `07` | D3QN/PER training runner | Port/adapt D3QN/PER training into Roehub ML app. | `04`,`06` | focused tests, training smoke, CPU/MPS performance evidence, accepted/rejected run records. |
| `08` | Roehub backtest/evaluation harness | Evaluate model decisions with Roehub fee/slippage/funding/risk/backtest semantics, diagnostic sanity baselines and simulator/accounting parity. | `07` | research candidate may be accepted only with positive PnL after costs, scorecard, sanity baseline artifacts, drawdown/stability report and offline simulator/accounting parity fixture; promotion-grade not granted here. |
| `09` | Model registry and activation gates | Persist datasets/models/calibrations with hashes, registry state machine, artifact lifecycle, checkpoint security and candidate/champion activation lifecycle. | `07`,`08` | registry state-machine invariant tests, API/use-case tests, corrupt/missing hash block, safe checkpoint load evidence, retention/quota config, activation/deactivation audit. |
| `09B` | Local artifact backup and restore drill | Backup accepted champion/calibration/source manifests/registry metadata locally and prove restore/rollback drill before runtime activation. | `09` | backup manifest, registry metadata dump, restore to separate path, hash validation after restore, rollback to previous accepted champion; residual single-host disk risk recorded. |
| `10` | Per-ticker calibration | Create per-ticker/per-market calibration thresholds, weights, or heads. | `08`,`09B` | calibration report per ticker, no global-only threshold activation unless accepted. |
| `10A` | Retraining and promotion lifecycle | Add full-retrain/fine-tune command path, manual + scheduled triggers, hard promotion approval contract, candidate/champion gates, drift trigger, rollback manifest, host-local operator command/runbook and internal application/API contract. | `08`,`09B`,`10` | deterministic rerun, schedule disabled-by-default proof, numeric threshold profile, candidate no-auto-activation proof, host-local rollback command/internal API test, promotion-grade report. |
| `11` | RL tab UI skeleton | Add `/strategies` RL/ML tab for model status, ticker slots, modes, risk config and authorized operator controls for retraining/rollback actions; extend the reusable strategy signal/outcome read model instead of creating an RL-only signal panel. | `09B`,`10A` | browser QA, API read models, authorized rollback UI control test, delivery-neutral signal/outcome read model, no live side effects. |
| `12` | Backend entitlements | Enforce active live ticker quotas from current `paid_level` (`base/free/pro/ultra`) plus optional RL override for Enterprise/custom. | `11` | transactional quota tests, UI blocked reason, audit rows, unknown/ambiguous level fail-closed tests. |
| `13` | Monitor-only inference producer | Supervised inference producer emits `ml_agent_decision` source events with `no_intent` and proves train/live feature parity. | `10`,`10A`,`11`,`12` | Monit/Prometheus, Redis/canonical feed with ClickHouse repair only on gaps, DB source events, Redis/live vs offline golden feature parity, common UI signal/outcome journal for `source_type=ml_agent_decision`. |
| `14` | User risk/sizing policy | Add configurable risk/sizing profiles for RL ticker activation, including synthetic platform-side exits. | `13` | API/UI tests, invalid policy blocks, TP/SL/trailing as synthetic exits only, no exchange-native advanced order fields, no exchange submit. |
| `15` | Paper RL integration | Convert accepted decisions to paper intents/orders through existing execution path and reconcile simulator vs paper accounting. | classic producer Stage `07` accepted; RL `14` | paper ledger proof, PnL accounting, simulator/accounting/paper parity within tolerance, duplicate/idempotency evidence. |
| `16` | Testnet RL integration | Run Binance/Bybit spot/futures testnet with safe guards and no mainnet. | classic producer Stage `09` accepted; RL `15` | real testnet orders/fills/reconciliation, unsupported branch blocks. |
| `17` | Multi-ticker runtime/load gate | Prove quotas, inference scheduling, market feed, rate limits across tariff-like ticker counts. | `16` | p95 latency, CPU/GPU/RSS, Redis lag, no DLQ growth. |
| `18` | 24h/7d RL soak and incident drills | 24h minimum, 7d preferred monitor/paper/testnet soak for selected tickers plus incident drill before mainnet review. | `17` | logs, metrics, UI, DB ledgers, model drift report, kill switch, pause, rollback, missing artifact, stale feed and unknown-state drill evidence. |
| `19` | Mainnet readiness architecture review | Separate go/no-go review for real-money live enablement, product/legal/support readiness and backup risk. | `18` | signed-off risk register, mainnet guard diff, rollback/runbook, disclaimer/risk disclosure/support policy/mainnet enablement policy/operator approval trail, backup evidence or explicit no-backup exception. |
| `20` | Bounded mainnet canary | Tiny-capital live canary on approved exchange/market/ticker count. | `19` explicit approval | real-money canary evidence, stop/close/reconcile proof, no quota bypass. |
| `21` | Product rollout | Enable tariff-limited live RL for broader users. | `20` accepted; backup/support/legal/product gates closed | rollout metrics, support/runbook, alert stability, artifact backup or signed residual-risk exception. |
| `22` | Final docs/prompt closure | Close docs, ledger, prompt pack, runbooks and delivery state. | all required prior stages | docs index, CI/deploy/host sync, final go/no-go record. |

## Планируемые Файлы И Артефакты По Stages

| Stage | Primary planned artifacts |
|---|---|
| `01` | This plan, stage ledger, Stage `01` report, docs index. |
| `02A` | Data inventory report, ClickHouse/HF/artifact coverage report, exchange/market/instrument lifecycle/gap report, classic producer blocker note. |
| `02B` | `src/trading/contexts/rl_trading/...feature_contract...`, feature contract note/tests plan, futures metadata gate report, live-feed `trades_count|feature-mask|blocked` decision, exchange/market activation matrix. |
| `02C` | Action/reward/state contract note, external-repo-compatible fixtures plan, no-pyramiding and strategy-owned close tests plan. |
| `03` | ML dependency group/config, Mac Studio env runbook, benchmark/resource isolation report. |
| `04` | HF import scripts/adapters, external license/attribution note, reproducibility report. |
| `05` | Dataset builder, raw feature slabs, feature parity fixtures, raw manifests and deterministic rebuild tests. |
| `06` | Session extractor, accepted sessionized train/val/test/backtest datasets, leak-check tests, overlap/embargo reports, `/opt/roehub/state/rl_trading/datasets/*` runtime artifacts. |
| `07`-`10A` | Trainer app, model code, simulator/accounting parity fixtures, sanity baseline evaluator, registry state machine/metadata, artifact retention/quota controls, checkpoint security, calibration/promotion artifacts, host-local rollback command/runbook, tests, metrics reports. |
| `09B` | Local backup path, backup manifest, registry metadata dump, restore drill report and rollback evidence. |
| `11`-`12` | API DTO/routes/read models, UI tab assets/locales, server-side operator/admin guard for model action controls, entitlement use cases/migrations/tests. |
| `13`-`18` | Inference producer app, launchd/Monit/Prometheus config, source event integration, paper/testnet evidence reports, simulator-paper parity report, incident drill reports. |
| `19`-`21` | Mainnet readiness doc, product/legal/support checklist, runbooks, canary evidence, rollback procedures, backup/residual-risk decision. |
| `22` | Final ledger, prompt pack closure, docs/runbook index updates. |

## Затрагиваемая Документация

| Document | Action |
|---|---|
| `docs/architecture/ml/rl-trading-agent-platform-v1.md` | Source of truth for this cycle. |
| `docs/architecture/ml/rl-trading-agent-platform-v1-stage-reports/rl-trading-agent-platform-v1-stage-ledger.md` | Update every stage. |
| `docs/architecture/live_execution/strategy-producer-paper-testnet-trading-v1.md` | Read as dependency; update only if RL discovers a required shared contract change. |
| `docs/architecture/live_execution/live-execution-universal-order-gateway-v1.md` | Read as accepted money-boundary foundation; update only for real contract drift. |
| `docs/architecture/market_data/*` | Update if RL feature artifacts change market data contracts. |
| `docs/architecture/backtest/*` and `docs/runbooks/backtest-artifacts-rebuild.md` | Update if RL dataset builder extends artifact publisher semantics. |
| `docs/architecture/identity/*` | Update if entitlement/plan limits become identity-owned contracts. |
| `docs/runbooks/mac-studio-native-backend-operations.md` | Update when ML trainer/inference services are supervised. |
| `docs/runbooks/rl-trading-operations.md` (planned) | Create before Stage `18`/`19` for incident drills, safe modes, restore/rollback and mainnet operator actions. |
| Product/legal/support readiness note (planned) | Create or link before Stage `19` for risk disclosure, support policy, mainnet enablement policy and approval trail. |
| `docs/architecture/README.md` | Regenerate/check after Markdown changes. |

## Журнал Выполнения Stages

Ledger path:

```text
docs/architecture/ml/rl-trading-agent-platform-v1-stage-reports/rl-trading-agent-platform-v1-stage-ledger.md
```

Rules:

- every stage updates the ledger before final status;
- every stage report includes `Created / Modified / Deleted / Reason / Contract impact`;
- every implementation stage must start from or create an executor prompt under `.codex/agents/generated/rl-trading-agent-platform-v1/`, and the stage report must record prompt path/hash;
- Stage `02A` is startable only from the generated executor prompt under `.codex/agents/generated/rl-trading-agent-platform-v1/`, which names required context/files, includes a strict file manifest and passed prompt-pack cold-head review after required fixes;
- no secrets, cookies, raw provider payloads, model binaries, or large local logs in docs;
- runtime artifacts live under `/opt/roehub/state/rl_trading/`, not git;
- dependent execution stages cannot start until their classic producer prerequisites are accepted;
- stage acceptance requires real-boundary evidence when API, persistence, browser, runtime, performance, exchange, or Mac Studio surfaces are touched;
- code/runtime stage acceptance must state delivery state explicitly: `local-only`, `published-to-branch/draft-pr`, `delivered-to-main`, and/or `deployed-on-macstudio`. Runtime/live stages cannot be accepted as complete from local gates alone when host sync or supervised service state is part of the stage;
- direct `git push origin main` is not the default delivery path. If a stage is published to GitHub, use `github:yeet`: inspect scope, stage only intended files, create one `codex/*` branch only when starting from `main`/default branch, open a draft PR, mark `delivered-to-main` only after PR merge plus SHA evidence on `origin/main`, and delete only the temporary `codex/*` branch created for that stage after the successful PR/test/deploy path. If a suitable task branch already exists, do not create a redundant branch;
- mainnet stages require explicit approval after Stage `19`; earlier stages must keep mainnet submit blocked.

## Architecture Review Gate

Cold-head review mode for this plan: explicit architecture-review subagent plus main-agent follow-up on 2026-06-17.

Subagent verdict: `Release after fixes`.

Applied fixes:

- identity tier mapping reconciled with current `base|free|pro|ultra` contract;
- prompt/delivery handoff added before executable stage work;
- training/retraining promotion lifecycle made explicit;
- classic producer current status refreshed: accepted through Stage `04`, Stage `05` currently blocked on Binance futures testnet credential custody;
- Stage `02A`/`02B` data and feature manifest expectations tightened.
- follow-up lifecycle contracts added: action/reward/state, strategy-scoped close ownership, train/live feature parity, session extraction policy, promotion scorecard, sanity baselines, artifact operations, checkpoint security, retraining cadence and staged rollback controls.
- dataset artifact ownership split: Stage `05` raw feature slabs/golden fixtures, Stage `06` accepted sessionized datasets.
- operator authority grounded: Stage `10A` starts host-local, Stage `11` web controls require server-side operator/admin guard.
- reusable signal/outcome UI contract extended for `source_type=ml_agent_decision`.
- external completeness review gaps integrated: Stage `02A/02B/02C` decomposition, live-feed feature gate, futures metadata gate, registry state machine, Stage `09B` local backup/restore, promotion-grade threshold profile, synthetic exits, simulator/accounting/paper parity, Mac Studio resource isolation, incident drills, live-outcome governance and product/legal mainnet gate.

Current architecture-plan verdict after fixes: `Release for staged planning`.

Prompt-pack readiness on 2026-06-17: `.codex/agents/generated/rl-trading-agent-platform-v1/` was generated for Stage `01` repair plus Stage `02A`-`22` execution. Independent cold-head review initially returned `Block` on one High finding: prompts used an ambiguous combined worker topology while the plan fixes separate `apps/worker/rl_trading_trainer/` and `apps/worker/rl_trading_inference/` apps. The prompt pack was fixed to preserve the planned topology, non-browser prompt auth over-routing was trimmed, browser QA surfaces were clarified for UI-affecting prompts, and local follow-up checks passed. Execution remains gated by each stage prompt, ledger prerequisites and real-boundary evidence.

## Риски И Открытые Вопросы

| Risk | Handling |
|---|---|
| Bybit lacks `trades_count` in current canonical rows | Stage `02B` must choose enrich, live-feed hot-path support, feature-mask branch or blocked branch before Bybit model activation. |
| Classic strategy producer Stage `05` is blocked | RL data/model/UI/monitor-only work may proceed, but RL paper/testnet/live stages depend on classic Stage `05` repair and accepted classic Stage `07`/`09`. |
| External repo is demo, not production module | Treat as research input; port concepts with attribution, do not blindly vendor code. |
| Positive backtest alone is not production promotion | Stage `08` accepts only research candidates; Stage `10A` requires numeric promotion-grade threshold profile before paper/testnet/live progression. |
| MPS support may be incomplete for chosen ops | Stage `03`/`07` benchmark CPU vs MPS and define accepted fallback. |
| Futures funding/contract metadata may be incomplete | Stage `02A`/`02B` must inventory funding, mark/index, filters, leverage tiers and explicitly block or mark `research_only_approximation` futures branches before Stage `05`/`08`. |
| Retraining can silently change live behavior | Stage `10A` requires candidate/champion gates, no auto-activation, rollback manifest and drift-triggered retraining task instead of in-place live mutation. |
| Action/reward semantics can drift from external repo | Stage `02C` records Roehub action/state/reward contract and Stage `07` tests it against external-repo-compatible fixtures before training acceptance. |
| Train/live features can diverge | Stage `05`/`13` require golden feature parity fixtures and shared feature builder contract before monitor/paper/testnet activation. |
| Overlapping sessions can leak across splits | Overlap is allowed only inside a split; Stage `06` must prove time embargo and lifecycle-aware split boundaries. |
| Local artifacts remain single-host risk | Stage `09B` must add local backup/restore drill; Stage `19`/`21` must either prove backup path is sufficient or explicitly accept residual single-host disk risk. |
| PyTorch checkpoint loading is a trust boundary | Stage `09` requires no user upload, sha256 validation, accepted-state-only load, path canonicalization and `weights_only` evidence or documented fallback. |
| Prompt pack can drift from plan | Stage executors must use `.codex/agents/generated/rl-trading-agent-platform-v1/*`, record prompt path/hash in every stage report and keep the ledger as the source of truth for current stage status. |
| Operator/admin auth primitive may be absent | Stage `10A` uses host-local command/runbook first; Stage `11` cannot enable web action controls until a server-side operator/admin guard exists. |
| Live feed schema may not carry all article-compatible features | Stage `02B`/`05` must choose `trades_count` hot-path support, feature-mask branch, or blocked branch; Stage `13` cannot pass without Redis/live/offline golden parity. |
| TP/SL/trailing can be confused with exchange-native order fields | RL v1 treats them as synthetic platform-side exits and must not pass advanced order fields into current execution order model. |
| Training can starve inference on the only ML host | Stage `03`/`07`/`17` must prove resource isolation, bounded CPU/MPS/RSS and no inference degradation under training/backtest load. |
| User live outcomes can leak into platform training | V1 blocks user-specific paper/testnet/live outcomes from platform-wide retraining until a separate governance/consent/redaction/lineage contract is accepted. |
| Mainnet needs product/legal/support readiness | Stage `19` must include disclaimer, risk disclosure, support policy, mainnet enablement policy and operator approval trail. |
| Current identity tiers differ from product labels | Stage `12` must map `base/free/pro/ultra` to RL entitlements explicitly and keep `base` fail-closed unless product evidence says it is Free. |
| User risk preferences can make model unsafe | Risk/sizing policy is platform-side and fail-closed; model confidence alone never authorizes live order. |
| Free ticker switching can be abused | v1 allows free changes; audit/rate-limit can be added if real abuse appears. |
| Mainnet pressure before evidence | Stage `19` is mandatory architecture review; Stage `20` requires explicit approval and bounded capital. |
