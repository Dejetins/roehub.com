# RL Trading Operations Runbook

This runbook covers the RL Trading Agent Platform v1 operational boundary for monitor-only technical runs and incident drills.

Current production rule: mainnet submit is blocked until Stage `19` readiness
review and explicit later approval. Stage `18` is accepted only as
`monitor_only_technical_soak`.

## Safety Boundary

| Rule | Requirement |
|---|---|
| Exchange access | RL trainer/inference code must not call exchange SDKs or resolve exchange credentials directly. |
| Execution path | Any future paper/testnet/live action must pass through existing `live_execution` and `exchange-execution` boundaries. |
| Monitor-only | `ml_agent_decision` must remain `no_intent`; no DB order state, Redis dispatch write, provider call or exchange submit is allowed. |
| Secrets | Do not print or store passwords, cookies, tokens, API keys, signed requests, raw provider payloads or checkpoint tensors in docs, logs, traces or screenshots. |
| Artifacts | Runtime ML artifacts live under `/opt/roehub/state/rl_trading/`; do not commit datasets/checkpoints/log dumps. |

## Stage 18 Monitor-Only Command

Run from the authoritative Mac Studio checkout:

```bash
cd /Users/daniildegtyarev/Projects/roehub.com
uv run python scripts/rl_trading/stage18_rl_soak_incident_drills.py \
  --config configs/prod/rl_trading_ml_runtime.yaml \
  --candidate-manifest /opt/roehub/state/rl_trading/evaluation_runs/stage08m_supervised_warm_start_candidate_scorecard_v1/stage08m_supervised_warm_start_fe2fe3c5257fd9992c55/stage08m_supervised_warm_start_candidate_manifest.json \
  --output-root /opt/roehub/state/rl_trading \
  --run-id stage18_manual_<UTC_TS> \
  --iterations-per-scenario 1 \
  --allow-fixture-manifest-hash \
  --ui-evidence-json '<sanitized browser evidence JSON>'
```

Expected accepted scope:

| Field | Expected |
|---|---|
| `mode` | `monitor_only_technical_soak` |
| `max_tickers` | `<=20` |
| `quality_claims.model_quality` | `false` |
| `quality_claims.trading_edge` | `false` |
| `quality_claims.mainnet_readiness` | `false` |
| `stage19_handoff.stage19_mainnet_readiness_allowed` | `false` |

## Incident Drills

| Drill | Expected result |
|---|---|
| Kill switch | Runtime remains fail-closed; no dispatch/order side effect. |
| Pause | Monitor-only processing can be paused without order-state mutation. |
| Rollback | Registry/model transition remains dry-run/fail-closed unless an accepted state exists. |
| Missing artifact | Missing candidate/checkpoint/artifact blocks activation and keeps safe state. |
| Stale feed | Stale feed blocks runtime readiness; no provider submit. |
| Unknown state | Unknown registry/runtime state blocks until explicit reconciliation. |

Every drill must record `exchange_side_effect=none` and `order_state=not_visible`
or equivalent fail-closed evidence.

## Browser Evidence

Browser QA may use either production-safe authenticated proof when the stage
requires it, or a local sanitized harness when the UI code is unchanged and the
goal is to prove visible safe/degraded state.

Minimum visible evidence:

| Surface | Required observation |
|---|---|
| `/strategies` RL/ML tab | Safe/degraded state is visible. |
| Active mode | `monitor_only`. |
| Operator controls | Retraining and rollback controls disabled unless a later accepted operator/admin guard exists. |
| Source outcomes | `ml_agent_decision` appears as `no_intent` only. |
| Console/network | `0` console warnings/errors and dashboard `200` response. |

## Response To Unsafe Signals

| Symptom | Action |
|---|---|
| Redis execution/retry/DLQ grows during monitor-only run | Stop the run; mark the stage blocked; inspect dispatch boundaries before rerun. |
| Any order/intention state appears | Treat as stage-blocking contract breach; preserve sanitized IDs only; do not continue soak. |
| Feed lag exceeds the accepted threshold | Keep monitor-only blocked/degraded; check live candle stream freshness before rerun. |
| Artifact/checkpoint hash mismatch | Do not load the model; run missing-artifact or rollback drill and record the blocker. |
| Unknown registry state | Fail closed and require explicit reconciliation before any later mode. |
| Browser shows enabled operator controls unexpectedly | Treat as UI safety breach; block stage handoff until server-side guard state is proven. |

## Handoff Rules

Stage `18` can hand off only technical evidence. It cannot open Stage `19` while
Stage `08N` has:

- `stage19_mainnet_readiness_allowed=false`;
- `stage20_mainnet_canary_allowed=false`;
- `stage21_product_rollout_allowed=false`.

Future Stage `19+` work requires separate accepted readiness evidence and
explicit approval. Do not infer mainnet readiness from monitor-only safety
checks.

## Stage 08K Monitor-Only Runtime

Постоянный исследовательский процесс использует DQN-кандидат Stage `08K`, но
не меняет его продуктовый статус. Кандидат остаётся `research_monitor_only` и
не открывает `paper`, `testnet`, `live` или Stage `19+`.

Зафиксированный контракт первого запуска:

| Поле | Значение |
|---|---|
| Модель | `stage08k_roehub_native_best_3e033951` |
| Checkpoint SHA-256 | `3e0339514d808a34a20d36a3e7e4035c5e722097046c2fc817bb5a4b93a03199` |
| Политика | `stage08k_long_only_hold_1m_monitor_v1` |
| Направление | Только `open_long`; `open_short` преобразуется в `hold` |
| Закрытие | Виртуальное закрытие по следующей закрытой минутной свече |
| Комиссия | Binance Futures taker `0.0005` на каждую сторону |
| Проскальзывание | `0.00025` на каждую сторону |
| Funding | Не моделируется; результат имеет исследовательский статус |
| Исполнение | Всегда `ml_agent_decision -> no_intent`; `intent_id=null` |

Процесс запускается службой `com.roehub.rl-trading-inference` и публикует
`/health/live`, `/health/ready` и `/metrics` на `127.0.0.1:9213`.

```bash
launchctl print gui/$(id -u)/com.roehub.rl-trading-inference
curl -fsS http://127.0.0.1:9213/health/ready
curl -fsS http://127.0.0.1:9213/metrics | grep '^rl_trading_inference_'
```

Структурированные журналы:

- `/Users/daniildegtyarev/Library/Logs/roehub/rl-trading-inference.out.log`;
- `/Users/daniildegtyarev/Library/Logs/roehub/rl-trading-inference.err.log`.

Prometheus хранит метрики, а не текст журналов. В labels запрещены идентификаторы
пользователя, стратегии и тикера. Детализация по инструменту хранится только в
существующем PostgreSQL-журнале `execution_source_events`.

Порядок расширения allowlist:

1. `one_ticker_1h`: один тикер и минимум один час чистого runtime evidence.
2. `five_ticker_24h`: пять тикеров только после принятого первого окна.
3. `twenty_ticker_7d`: двадцать тикеров только после принятого суточного окна.

Каждая фаза обязана подтвердить `0` execution intents, `0` orders, отсутствие
роста `execution.requests.v1`, `execution.requests.retry.v1` и
`execution.requests.dlq.v1`, а также отсутствие
`rl_trading_inference_safety_breaches_total`.

## Stage 08K Monitor-Only Alert Actions

| Сигнал | Действие оператора |
|---|---|
| `RlTradingInferenceDown` | Проверить `launchctl`, Monit и оба файла журналов. Не включать другие режимы. |
| `RlTradingInferenceNotReady` | Проверить SHA-256 модели, evaluation manifest, normalization stats и доступность host-local env. Не обходить artifact gate. |
| `RlTradingInferenceFeedLag` | Проверить `md.candles.1m.*`, market-data worker и Redis pending entries. |
| `RlTradingInferenceErrors` | Найти bounded `operation/reason`; сообщение не подтверждается до успешной записи состояния и source event. |
| `RlTradingInferenceSafetyBreach` | Немедленно остановить службу и проверить `execution_source_events`, `execution_intents`, orders и Redis execution streams. |

Остановка и повторный запуск:

```bash
launchctl bootout gui/$(id -u) /Users/daniildegtyarev/Library/LaunchAgents/com.roehub.rl-trading-inference.plist
launchctl bootstrap gui/$(id -u) /Users/daniildegtyarev/Library/LaunchAgents/com.roehub.rl-trading-inference.plist
```

Незавершённые виртуальные позиции сохраняются в
`/opt/roehub/state/rl_trading/monitor_state/stage08k_long_only_hold_1m_monitor_v1.json`.
Удалять этот файл во время активной фазы нельзя. Повторная обработка безопасна:
source events имеют детерминированные idempotency keys, а Redis message
подтверждается только после успешной записи source event и локального состояния.
