# Mainnet Real-Money Trading v1

Статус: `draft plan`. Документ описывает отдельный цикл внедрения real-money
mainnet trading для Roehub. Это не продолжение `Strategy Producer
Paper/Testnet Trading v1` и не Stage `18` старого
`live-execution-universal-order-gateway-v1`; оба принятых цикла используются
как foundation.

> Связь с greenfield-платформой: Stage `16` плана
> `roehub-self-hosted-oss-platform-v1` реализует только persisted
> owner/recent-auth policy, двойной submit guard, атомарный claim,
> reconcile-before-retry, allowlist и неизменяемый аудит. Он не принимает ни
> один этап этого отдельного плана, не включает нативный `mainnet` и не
> разрешает реальные заявки. Любая будущая real-money canary по-прежнему
> требует собственных prerequisites и явного пользовательского разрешения.

Связанные execution artifacts:

| Артефакт | Путь |
|---|---|
| `plan_doc` | `docs/architecture/live_execution/mainnet-real-money-trading-v1.md` |
| `prompt_pack_dir` | `.codex/agents/generated/mainnet-real-money-trading-v1/` |
| `stage_ledger` | `docs/architecture/live_execution/mainnet-real-money-trading-v1-stage-reports/mainnet-real-money-trading-v1-stage-ledger.md` |
| `execution_mode` | `goal_driven` поверх `plan_doc + prompt_pack_dir + stage_ledger`; `GOAL.md` не требуется |

## Цель

Безопасно включить работу с реальными Binance/Bybit mainnet API для
`spot` и `futures`, доказать реальные market orders малым капиталом и получить
измеримый путь:

```text
candle close -> StrategySignal -> ExecutionSourceEvent -> ExecutionIntent
-> risk accepted -> Redis dispatch -> exchange-execution read
-> native adapter submit -> exchange ack -> DB order persisted
-> first fill -> reconciliation matched -> user/operator notification
```

Критерий успеха v1: Roehub умеет автоматически, но под контролем агента,
исполнить реальные mainnet market orders по Binance/Bybit `spot` и `futures`
с жесткими лимитами, немедленным закрытием canary-позиций, Prometheus/Grafana
метриками, user/operator alerts, reconciliation и безопасным rollback.

## Бизнес-Смысл

Roehub переходит от доказанного `paper/testnet` контура к реальному
money-moving режиму. Ошибка в этом цикле может привести к потере денег,
незакрытым позициям, неизвестному состоянию ордера или повторному submit.
Поэтому каждый stage имеет go/no-go gate: если доказательство не собрано,
stage блокируется, а следующий stage не стартует.

## Уже Доказанный Foundation

| Область | Текущее состояние |
|---|---|
| Exchange keys custody | OpenBao/Transit + `exchange-control` используются как custody boundary; raw secrets не должны попадать в API/UI/логи. |
| Strategy producer | `apps/worker/strategy_live_runner` принят для `paper,testnet`; сейчас deliberately blocks `live/mainnet`. |
| Universal execution path | `ExecutionSourceEvent`, `ExecutionIntent`, risk gate, Redis dispatch, `exchange-execution`, order/fill/reconciliation ledgers уже приняты в testnet cycle. |
| Exchange execution service | `apps/exchange_execution` supervised через launchd/Monit/Prometheus; сейчас adapter mode `disabled/testnet`, mainnet submit hard-blocked. |
| Native adapters | Binance/Bybit native adapters доказаны на testnet; real mainnet enablement не доказан. |
| Market data live tail | 1m live-tail repair принят; `12.4` доказал signal-path latency/dedup для `paper/testnet`. |
| Notifications | `execution_notification_outbox` и admin alert/runbook слой есть; реальная user delivery через Telegram зависит от хостовой доступности `api.telegram.org`. |

## Hard Go/No-Go Decisions

| Decision | Значение v1 | Go/no-go |
|---|---|---|
| Биржи | Binance и Bybit | Обе обязательны для closure. |
| Markets | `spot` и `futures` | Обязательны Binance spot, Binance futures, Bybit spot, Bybit futures. |
| Symbol | `BTCUSDT` | Другие символы вне scope. |
| Order type | Только `market` | `limit`, OCO, trailing, TP/SL, amend/replace, batch orders вне scope. |
| Spot short | Unsupported | Spot canary только long buy -> sell close. |
| Futures direction | Long и short | Только isolated `1x`, default `isolated 1x`, user может задать futures параметры. |
| Futures config | Platform может менять margin/leverage/position config только как явная stage-команда до order submit | Скрытая auto-config внутри submit запрещена. |
| First canary max notional | `15 USDT` на один order | Любой order выше cap блокируется. |
| Budget | Заявлено `20 USDT` на каждый рынок и `60 USDT` всего | До submit нужен explicit capital allocation manifest; пока действует более строгий global cap `60 USDT` и per-order cap `15 USDT`. |
| Canary close | Обязательно немедленное закрытие | Spot buy закрывается sell; futures open закрывается reduce-only market. |
| Canary lifecycle | Отдельная пара `open_order -> close_order` | `cancel_after_submit` не является close proof и не засчитывается для mainnet. |
| Kill switch semantics | `kill_switch_active=true` означает аварийный stop | Текущий drift вокруг `kill_switch_open` должен быть исправлен до caps/submit stages. |
| Private stream / fallback | Private order/fill stream обязателен или должен быть заменен явно ограниченным REST polling fallback | Testnet auth probe не считается полноценным mainnet private stream proof. |
| Futures reconciliation | Fill/position reconciliation отделен от funding reconciliation | Short-lived canary не должен блокироваться только потому, что funding event еще не наступил. |
| Strategy mode | Автоматический режим под присмотром агента | Нет ручного подтверждения каждого order; есть scoped allowlist, kill switch и hard caps. |
| Sustained observation | Минимальная длительность | Не 6h; достаточно доказать факт signal -> execution на каждом required market. |
| Telegram/user alert | В scope как обязательный user-alert proof | Stage execution blocked до явного подтверждения пользователя, что проблема Telegram на хосте решена. VLESS/VPN настройка вне scope. |
| Mainnet approval | По stages, не общий вечный флаг | Каждый stage открывает только следующий bounded action. |

## Adoption Диагностики GPT-5.5 Pro

Все пункты диагностики приняты. Нет рекомендаций, с которыми этот план
осознанно не согласен.

| Пункт диагностики | Как учтен в плане |
|---|---|
| `kill_switch_open` имеет опасный semantic drift | Добавлен Stage `03`: переименование/инверсия в `kill_switch_active` или `execution_enabled`, обновление reason codes, docs, tests и runtime evidence. |
| Auto-close отсутствует как модель исполнения | Добавлен Stage `07`: durable `open_order -> close_order` lifecycle, linkage, spot close by filled base qty, futures reduce-only close, terminal states and failure handling. |
| Futures mainnet config не встроен в submit path | Stage `08` теперь требует pre-submit read-back guard: no open orders, no existing position, isolated `1x` or user-selected safe config, fresh config snapshot. |
| Risk/caps слишком абстрактны | Stage `04` теперь требует persisted `mainnet_capital_manifest`, `mainnet_canary_scope`, order/open-exposure/gross-notional/daily-loss caps, one-shot approval token and audit math. |
| Budget conflict blocks canaries | Stage `04` формализует `per_order_notional_cap`, `max_open_exposure_total`, `max_open_exposure_per_exchange_market`, `max_gross_submitted_notional_per_stage`, fees/slippage reserve. |
| Private stream readiness не равен mainnet private stream | Добавлен Stage `09`: полноценный private stream lifecycle или явно принятый REST polling fallback with stricter limits. |
| Futures funding blocks short canary reconciliation | Stage `09` разделяет order/fill, position and funding reconciliation; `funding_not_due_yet` не блокирует canary closure if fill/position matched. |
| Strategy live-mode contract не описан | Stage `17` теперь проектирует sizing, profile binding, canary token propagation, fan-out guard, stop-after-first-canary and restart/dedup semantics before strategy orders. |
| Stage `07` слишком крупный | Real ops canaries split into Stages `10`-`16`, one exchange/market/direction row at a time with stop/reconcile/alert/no residual state after every row. |
| Metrics names use `_bucket` base names | Metrics section now uses base histogram names without `_bucket`; Prometheus-generated `_bucket/_sum/_count` series are only query outputs. Units are seconds. |
| Stale UI/code reason codes | Stage `00` must inventory stale copy, and Stage `03` must repair reason semantics before risk/caps stages can accept. |

## Что Требуется От Пользователя

Эти действия не выполняются агентом и должны быть явно подтверждены в ledger.
Если действие не выполнено, соответствующий stage фиксирует `blocked`.

| Требование | Нужно до stage | Как подтвердить без раскрытия секретов |
|---|---:|---|
| Сказать буквально, что Telegram blocker решен | `01` | Сообщение пользователя: `Telegram blocker resolved for mainnet plan` или русская формулировка `проблема Telegram решена`; затем runtime readiness без раскрытия токенов. |
| Настроить VLESS/VPN для обращений Mac Studio к `api.telegram.org` | `01` | Это вне scope плана; stage только проверяет доступность/готовность. |
| Подключить Binance mainnet API key без withdrawals | `02` | Через `/settings`; report пишет только connection id, exchange, market, masked key suffix. |
| Подключить Bybit mainnet API key без withdrawals | `02` | То же. |
| Включить IP allowlist на публичный IP Mac Studio для mainnet ключей | `02` | Runtime validation/readiness доказывает restricted IP / trade-ready status; raw IP/keys не пишутся в docs. |
| Пополнить реальные балансы | `02` | Минимум достаточно для canary: план исходит из `15 USDT` на order и stricter total cap `60 USDT`; точные balances не публикуются, только pass/fail buckets. |
| Подтвердить capital allocation manifest | `04` | Нужно устранить неоднозначность `20 USDT` на каждый рынок vs `60 USDT` всего. До этого stage не может отправлять реальные orders. |
| Подтвердить futures policy | `08` | Default `isolated 1x`; любые другие user-selected параметры должны быть явно записаны и пройти read-back. |
| Подтвердить first real-money canary window | `10` | Пользователь разрешает sequential bounded canary matrix по `15 USDT`, auto-close, no manual per-order approval. |
| Подтвердить strategy-driven mainnet canary window | `18` | Пользователь разрешает scoped automatic strategy execution под наблюдением агента. |

## Scope

Входит:

| Область | Решение v1 |
|---|---|
| Mainnet credentials | Use existing exchange connection flow and custody boundary; raw credentials не передаются агенту. |
| Binance/Bybit spot/futures | Readiness, account projection, market orders, status/fill/reconciliation. |
| Futures account config | Explicit pre-submit command: set/read isolated `1x` by default, block if mismatch after read-back. |
| Auto-close lifecycle | Durable open/close order pair, close linkage, reduce-only futures close, spot close by actual filled quantity, close failure handling. |
| Mainnet risk | Per-order, open exposure, gross submitted notional, per-market, per-user, per-strategy caps; daily loss cap; fees/slippage reserve; kill switch. |
| Private stream / reconciliation | Private order/fill stream lifecycle or explicit REST polling fallback; separate order/fill, position and funding reconciliation. |
| User alerts | Outbox + real user-alert delivery proof only after Telegram host blocker resolved. |
| Metrics | Dedicated Prometheus metrics for mainnet latency, slippage, unknown state, reconciliation, exposure, caps, alerts. |
| UI | Mainnet blocked/ready/live status and user-visible warnings on `/settings` and `/strategies`. |
| Validation | Real runtime calls: API, DB, Redis, Prometheus, Monit, browser, exchange mainnet canaries. Tests are gates only. |

Не входит:

| Не входит | Причина |
|---|---|
| Настройка VLESS/VPN на host | Пользователь прямо исключил это из scope. |
| Mainnet beyond BTCUSDT | Следующий отдельный plan/extension. |
| Spot margin/borrow short | Нет отдельного margin product. |
| Advanced orders | Не нужны для v1 market-order canary. |
| Portfolio allocator | Достаточно строгих caps/reservations. |
| ML agent real-money activation | Только compatibility contract; ML activation отдельным планом. |
| Долгий 6h/24h money soak | В v1 нужен минимальный proof факта signal execution per market. |
| Advanced private stream optimization | Fast execution streams and latency optimization beyond safe mainnet proof are later work. |

## Целевая Архитектура

```mermaid
flowchart LR
    User["User / operator"]
    UI["/settings / /strategies"]
    API["apps/api"]
    Producer["strategy-live-runner"]
    LE["live_execution context"]
    Redis["Redis execution.requests.v1"]
    EE["exchange-execution"]
    EC["exchange-control + OpenBao"]
    EX["Binance / Bybit mainnet"]
    DB["Postgres ledgers"]
    Metrics["Prometheus / Grafana"]
    Alerts["notification outbox + Telegram user alert"]

    User --> UI
    UI --> API
    API --> Producer
    Producer --> LE
    LE --> DB
    LE --> Redis
    Redis --> EE
    EE --> EC
    EE --> EX
    EE --> DB
    LE --> Alerts
    EE --> Alerts
    Producer --> Metrics
    LE --> Metrics
    EE --> Metrics
```

Правило зависимости: `strategy-live-runner` не получает raw secrets и не
вызывает exchange SDK. Он создает `StrategySignal` и `ExecutionSourceEvent`.
Единственная money-moving boundary остается `exchange-execution`.

## Service Calls

| Caller | Callee | Contract | Auth / custody | Timeout / retry | Unknown-state rule |
|---|---|---|---|---|---|
| UI | `apps/api` | Mainnet readiness, launch, dashboard, manual stop/kill | User session + recent-auth where needed | No blind retry for money actions | UI shows pending/unknown, not success |
| `apps/api` | `live_execution` | Source event, intent, risk/cap state | App identity + user ownership | Idempotency by source key | Read durable source/intent before retry |
| `live_execution` | Redis | Dispatch accepted intent | Internal Redis ACL | Retry budget/backpressure/DLQ | DB remains source of truth |
| `exchange-execution` | `exchange-control` | Resolve secret for scoped connection | Service identity; decrypt only in execution service | Bounded timeout; no submit if unavailable | Block submit, readiness degraded |
| `exchange-execution` | Binance/Bybit REST | Server time, account state, futures config, market submit, close submit, order status, fills | Native signed exchange API | Per-exchange limiter; retry only safe failure classes | If provider state unknown, query/reconcile before any retry |
| `exchange-execution` | Binance/Bybit private stream or REST fallback | Order/fill/position updates | Native signed exchange API; no raw secret in logs | Stream reconnect/keepalive/backfill or bounded polling | Stale stream/fallback blocks submit unless stage explicitly allows degraded no-submit mode |
| Notification worker | Telegram API | User/operator alert delivery | Host-local bot config; no tokens in reports | Provider backoff; no blind replay for unknown critical delivery | Unknown delivery blocks closure until inspected |

Official provider docs that must be rechecked by relevant stages:

| Provider | Документ |
|---|---|
| Binance Spot limits | `https://developers.binance.com/docs/binance-spot-api-docs/rest-api/limits` |
| Binance USD-M Futures general info | `https://developers.binance.com/docs/derivatives/usds-margined-futures/general-info` |
| Bybit V5 rate limits | `https://bybit-exchange.github.io/docs/v5/rate-limit` |
| Bybit V5 create order | `https://bybit-exchange.github.io/docs/v5/order/create-order` |

## Error, Retry, Idempotency

| Сценарий | Правило |
|---|---|
| Duplicate strategy signal | Existing `signal_id` / source idempotency key returns existing rows, no duplicate order. |
| Binance HTTP `429` / `418` | Respect provider headers/backoff; no aggressive retry; alert if limiter is insufficient. |
| Binance futures `503` unknown execution status | Treat as `unknown`; first private stream/order-status lookup, then reconciliation, no blind second submit. |
| Bybit async order ack | Ack means accepted, final status comes from websocket/status/reconciliation; do not mark filled without fill facts. |
| Private stream stale | Block submit if private stream proof is required and stale/missing. |
| Private stream unavailable but REST fallback accepted | Submit can proceed only if Stage `09` accepted explicit fallback with stricter polling/reconciliation thresholds and alerting. |
| Redis message pending | Do not ack until durable state transition is written. |
| Unknown order | Kill new dispatch for affected scope, alert user/operator, reconcile from provider before retry/cleanup. |
| Auto-close required | Every real canary creates a durable open/close pair; `cancel_after_submit` is not accepted as close evidence for market orders. |
| Spot close | Close quantity is based on actual filled base quantity minus dust/min-notional constraints; residual dust must be audited. |
| Futures close | Close order must be reduce-only in one-way/net mode; if reduce-only is unavailable or rejected, stage blocks and opens critical incident. |
| Partial fill | Close only confirmed filled quantity; unresolved partial/unknown fill stops the scope until provider status is reconciled. |
| Auto-close failed | Critical alert, engage kill switch for affected scope, stop strategy scope, reconcile position, do not continue strategy canary. |
| Futures funding pending | Does not block canary closure when order/fill and position are matched; funding remains separate ledger concern with `funding_not_due_yet` or equivalent reason. |
| Notification unknown | Mainnet closure blocked until delivery row inspected or replay policy followed. |

## Mainnet Metrics

Новые/расширяемые метрики должны быть добавлены в Prometheus rules и
`docs/runbooks/prod-dashboard-metrics-reference-ru.md`.

| Metric | Labels | Что измеряет |
|---|---|---|
| `mainnet_execution_latency_seconds` | `segment`, `exchange`, `market_type`, `source_type` | Histogram base name для p50/p95/p99 по сегментам полного пути. Prometheus создаст `_bucket`, `_sum`, `_count`. |
| `mainnet_signal_to_fill_latency_seconds` | `exchange`, `market_type`, `direction` | Histogram base name для времени от сигнала до первого fill. |
| `mainnet_exchange_submit_latency_seconds` | `exchange`, `market_type`, `order_type` | Histogram base name для native adapter submit latency. |
| `mainnet_order_slippage_bps` | `exchange`, `market_type`, `direction` | Slippage в bps по canary/fill facts. |
| `mainnet_reconciliation_lag_seconds` | `exchange`, `market_type`, `reconciliation_type` | Histogram base name для времени до matched order/fill, position or funding reconciliation. |
| `mainnet_unknown_state_total` | `exchange`, `market_type`, `reason` | Unknown provider/execution states. |
| `mainnet_risk_rejections_total` | `reason`, `exchange`, `market_type` | Mainnet risk/cap/futures-config blocks. |
| `mainnet_gross_submitted_notional_usd` | `exchange`, `market_type`, `stage_scope` | Gross submitted notional including close orders. |
| `mainnet_open_exposure_usd` | `exchange`, `market_type`, `user_scope` | Текущая открытая экспозиция в mainnet. |
| `mainnet_daily_loss_usd` | `exchange`, `market_type`, `user_scope` | Реализованный/оценочный дневной loss для caps. |
| `mainnet_kill_switch_state` | `scope`, `reason` | Глобальный/per-user/per-strategy/per-exchange stop state. |
| `mainnet_private_stream_age_seconds` | `exchange`, `market_type` | Freshness private order/fill stream. |
| `mainnet_auto_close_total` | `exchange`, `market_type`, `direction`, `status` | Open/close pair outcomes. |
| `mainnet_user_alert_delivery_total` | `event_type`, `status`, `channel` | User alert delivery outcomes. |

Метрики не должны иметь high-cardinality labels вроде raw order id, user id,
API key suffix или provider payload.

## Alerts And Runbooks

| Alert | Severity | Owner | Trigger | Action |
|---|---|---|---|---|
| `MainnetTradingKillSwitchActive` | critical | live-execution | Kill switch active for mainnet scope | Stop dispatch, show UI blocked, do not clear without operator note. |
| `MainnetUnknownOrderState` | critical | live-execution | Unknown provider/order/reconciliation state | Stop affected strategy/exchange, reconcile provider state before retry. |
| `MainnetAutoCloseFailed` | critical | exchange-execution | Canary close failed or position remains open | Stop all mainnet canaries, reconcile/close manually if needed. |
| `MainnetLatencyHigh` | warning/critical | live-execution | p99 segment threshold exceeded | Pause expansion; inspect Redis, adapter, provider, host resources. |
| `MainnetSlippageHigh` | warning/critical | live-execution | Slippage exceeds configured bps threshold | Stop canary expansion; inspect spread/liquidity/order sizing. |
| `MainnetRiskCapNearLimit` | warning | live-execution | Exposure/loss approaches cap | Stop new orders for scope if threshold crossed. |
| `MainnetPrivateStreamStale` | critical | exchange-execution | Stream age above threshold | Block submit, reconnect/backfill/reconcile. |
| `MainnetRestFallbackActive` | warning | exchange-execution | Stage accepted REST fallback instead of private stream | Keep canary scope small; verify reconciliation lag thresholds. |
| `MainnetUserAlertDeliveryFailed` | critical | notifications | Critical trade alert unknown/failed | Block closure and inspect delivery route/provider. |
| `MainnetTelegramUnavailable` | critical | notifications | Host cannot reach Telegram while mainnet enabled | Keep mainnet disabled or open kill switch. |

## Stage Plan

| Stage | Назначение | User required before start | Acceptance / go-no-go |
|---|---|---|---|
| `00` | Baseline hard-block and stale-copy manifest | Nothing. | Current paper/testnet/gateway foundation reconciled; mainnet hard-blocks listed; stale reason/copy drift inventoried; prompt pack checked; no runtime mutation. |
| `01` | User prerequisite and Telegram gate | Literal user confirmation that Telegram problem is solved; host Telegram readiness source available. | Runtime proves Telegram readiness without secrets; mainnet execution remains blocked if not proven. |
| `02` | Mainnet exchange connections read-only readiness | Binance/Bybit mainnet keys connected, IP allowlist enabled, balances funded. | Real read-only Binance/Bybit spot/futures readiness, trade permission/no withdrawal/IP restriction/balance buckets; no order submit. |
| `03` | Kill-switch semantics and stale reason repair | Nothing beyond accepted `02`. | `kill_switch_active` / `execution_enabled` semantics fixed or explicitly mapped; stale `stage19_20`, `stage05`, `stage13` reason/copy drift repaired; tests/docs/runbooks updated; no order submit. |
| `04` | Mainnet risk, caps, capital manifest and approval schema | Capital allocation manifest resolves `20 per market` vs `60 total`. | Durable manifest/scope/order/open-exposure/gross/daily-loss/fee-reserve caps, one-shot approval token, risk audit math, UI blocked/ready states; no order submit. |
| `05` | Mainnet metrics, alerts, dashboard and user-alert contract | Telegram gate accepted. | Prometheus base histogram names in seconds, dashboard metrics reference, notification outbox/user delivery contract, runbooks updated and smoke-proven; no order submit. |
| `06` | Mainnet adapter enablement behind fail-closed no-submit mode | Mainnet readiness/caps/alerts accepted. | `exchange-execution` can run in mainnet-capable no-submit mode; mainnet submit remains blocked without scoped one-shot canary approval; no real order. |
| `07` | Open/close order-pair lifecycle and auto-close model | Nothing beyond accepted `06`. | Durable open/close pair model, close linkage, spot filled-qty close, futures reduce-only close, terminal states, partial-fill and close-failure handling proven without mainnet submit. |
| `08` | Futures account config and market-order guard | User approves default or selected futures config. | Binance/Bybit futures set/read back isolated `1x` or user-selected safe config; no open orders/positions preflight; fresh config snapshot linked to canary scope; market-order guard; no strategy order yet. |
| `09` | Private stream or REST fallback and reconciliation semantics | Nothing beyond accepted `08`. | Mainnet private order/fill stream lifecycle or explicit REST polling fallback accepted; order/fill, position and funding reconciliation separated; stale stream/fallback blocks submit as designed. |
| `10` | Real mainnet ops canary: Binance spot long | User approves bounded canary window. | Binance spot buy market `<=15 USDT`, close sell by filled base qty, fills/reconciliation/alert/latency/slippage, no residual exposure. |
| `11` | Real mainnet ops canary: Bybit spot long | Stage `10 accepted`. | Bybit spot buy market `<=15 USDT`, close sell by filled base qty, fills/reconciliation/alert/latency/slippage, no residual exposure. |
| `12` | Real mainnet ops canary: Binance futures long | Stage `11 accepted`. | Binance futures long open/close reduce-only market `<=15 USDT`, isolated `1x`, matched fill/position reconciliation, funding not blocking. |
| `13` | Real mainnet ops canary: Binance futures short | Stage `12 accepted`. | Binance futures short open/close reduce-only market `<=15 USDT`, isolated `1x`, matched fill/position reconciliation, funding not blocking. |
| `14` | Real mainnet ops canary: Bybit futures long | Stage `13 accepted`. | Bybit futures long open/close reduce-only market `<=15 USDT`, isolated `1x`, matched fill/position reconciliation, funding not blocking. |
| `15` | Real mainnet ops canary: Bybit futures short | Stage `14 accepted`. | Bybit futures short open/close reduce-only market `<=15 USDT`, isolated `1x`, matched fill/position reconciliation, funding not blocking. |
| `16` | Ops canary matrix closure | Stage `15 accepted`. | Matrix-wide no residual orders/positions, no unknown/retry/DLQ growth, alerts delivered, metrics complete, budget/gross/fee audit recorded. |
| `17` | Strategy producer live-mode contract and no-order enablement | Stage `16 accepted`. | Live-mode contract defines sizing, connection binding, canary token propagation, fan-out guard, stop-after-first-canary, restart/dedup semantics; no strategy-driven real order yet. |
| `18` | Strategy-driven mainnet canaries per market | User approves scoped automatic strategy window. | One real strategy signal per required market surface executes automatically under allowlist/caps, auto-closes, records candle-to-fill-to-alert latency, and stops after first canary per scope. |
| `19` | Closure, cleanup and go/no-go record | Nothing unless residual open position/unknown state needs operator action. | No unexpected open orders/positions, no unknown/DLQ/retry growth, metrics/dashboard/alerts verified, ledgers updated, mainnet expansion remains disabled outside accepted scopes. |

## Validation Ladder

| Surface | Minimum proof |
|---|---|
| Local gates | Focused `ruff`, `pyright`, `pytest` for touched areas; docs index for Markdown. |
| API | Real authenticated calls through current production routes or internal local-only endpoints where appropriate. |
| DB | SQL evidence for source events, intents, orders, fills, reconciliation, risk, caps, notifications. |
| Redis | `XINFO`, `XPENDING`, retry/DLQ checks before/after canaries. |
| Exchange | Real Binance/Bybit mainnet read-only calls; canary stages perform real orders only within caps. |
| Browser | `/settings` and `/strategies` proof with no credential leakage in DOM/screenshots. |
| Metrics | Prometheus scrape/query evidence for new mainnet metrics and alerts. |
| Runtime | Mac Studio launchd/Monit/health/readiness proof after changed code reaches `main` and deploy/sync is complete. |
| Performance | p50/p95/p99 latency per segment from durable timestamps; no reasoned estimate as acceptance. |
| User alerts | Delivery proof for mainnet trade/incident notifications; if Telegram unavailable, stage blocked. |
| Canary row | After every real canary row: provider status/fills, close proof, exposure zero, Redis pending `0`, no DLQ/retry growth, no unknown state, alert delivered. |

Tests-only acceptance запрещена для всех stages этого плана.

## Planned Files And Artifacts

| Area | Expected touches |
|---|---|
| Plan / ledger | `docs/architecture/live_execution/mainnet-real-money-trading-v1.md`, `docs/architecture/live_execution/mainnet-real-money-trading-v1-stage-reports/mainnet-real-money-trading-v1-stage-ledger.md`, stage reports. |
| Prompt pack | `.codex/agents/generated/mainnet-real-money-trading-v1/*.md`. |
| API / DTO / UI | `apps/api/**`, `apps/web/**` only when stages add mainnet status/actions. |
| Domain/application | `src/trading/contexts/live_execution/**`, `src/trading/contexts/strategy/**`, identity/exchange-control adapters as needed. |
| Runtime | `apps/exchange_execution/**`, `apps/worker/strategy_live_runner/**`. |
| Config/infra | `configs/prod/**`, `infra/macos/launchd/**`, `infra/scripts/monit/**`, `infra/macos/prometheus/**`. |
| Migrations | `alembic/versions/**` for new risk/cap/approval, open/close pair, futures config audit, reconciliation and alert persistence. |
| Runbooks | `docs/runbooks/exchange-execution.md`, `docs/runbooks/strategy-live-worker.md`, `docs/runbooks/prod-dashboard-metrics-reference-ru.md`, `docs/runbooks/notifications-admin-alerts.md`. |
| Evidence | Stage reports under `docs/architecture/live_execution/mainnet-real-money-trading-v1-stage-reports/`; runtime artifacts under `/opt/roehub/state/live_execution/mainnet-real-money-trading-v1/` when created by execution stages. |

## Rollback And Recovery

| Сценарий | Rollback |
|---|---|
| Before real order | Keep mainnet submit disabled; revert config/feature flags; no ledger cleanup needed. |
| After canary order filled and closed | Leave ledger rows immutable; disable mainnet scope; verify no open orders/positions. |
| Auto-close failed | Open kill switch, stop strategy producer mainnet scope, reconcile provider position, perform operator/manual close if needed, record incident. |
| Unknown provider state | Do not replay; query provider order/status/fill, reconcile, then decide. |
| Telegram delivery failed | Keep mainnet expansion blocked; inspect delivery rows/provider health; follow notification replay policy. |
| Metrics/alerts missing | Do not accept stage; keep next stage blocked until Prometheus/Grafana/runbook evidence exists. |

## Open Decisions / Blockers

| Blocker | Почему важно | Resolution required |
|---|---|---|
| Telegram host access to `api.telegram.org` | User alerts are mandatory for real-money v1. | User says the issue is solved; Stage `01` proves readiness. |
| Capital allocation wording conflict | `20 USDT` per market across four market surfaces conflicts with `60 USDT` total. | Stage `04` must record explicit allocation manifest before any submit. |
| Kill switch semantic drift | Existing `kill_switch_open` wording can mean either allow or stop depending on context. | Stage `03` must fix or explicitly map semantics before caps and submit gates. |
| Auto-close lifecycle missing | Market order can be filled before cancel; close requires an opposite/reduce-only order. | Stage `07` must add and prove open/close pair lifecycle before any real order. |
| Private stream / fallback decision | Testnet auth probe is not enough for mainnet order/fill proof. | Stage `09` must accept full private stream lifecycle or explicit REST polling fallback with thresholds. |
| Futures reconciliation semantics | Funding can be pending while fill/position already match. | Stage `09` must split order/fill, position and funding reconciliation. |
| Exact strategy for signal canary | A real strategy signal must happen with minimal observation time. | Stage `18` selects/creates a canary strategy/run that can produce a bounded signal on live candles without fake exchange side effects. |
| Telegram real delivery channel | Outbox exists, but actual provider reachability depends on host/VPN. | Stage `04` cannot accept without real delivery readiness/canary proof. |

## Definition Of Done For The Plan

The plan is completed only when:

- all stages `00`-`19` are `accepted` in the stage ledger;
- Binance/Bybit `spot/futures` mainnet canary matrix is completed with real orders and auto-close;
- every ops canary row is accepted independently before the next row starts;
- at least one strategy-driven real signal executes per required market surface;
- no unexpected open order/position remains;
- no unexplained retry/DLQ/unknown/reconciliation debt remains;
- Prometheus metrics and alert rules cover mainnet latency/slippage/reconciliation/user-alert state;
- user alert delivery is proven after Telegram blocker resolution;
- docs, runbooks, prompt pack, ledger and docs index are synchronized;
- mainnet remains disabled outside explicitly accepted scoped allowlists/caps.
