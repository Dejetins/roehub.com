# Strategy live worker / supervised producer runbook

Статус: active для Stage `06` strategy-producer paper/testnet cycle.

Сервис `com.roehub.strategy-live-runner` переиспользует `apps.worker.strategy_live_runner.main.main` как supervised strategy producer. Он читает live market-data streams, пишет strategy signal journal, а source events в `live_execution` создает только через существующий execution producer port. Прямой exchange SDK, raw credentials и mainnet/live mode в этом процессе запрещены.

## Runtime Surface

| Surface | Value |
|---|---|
| launchd label | `com.roehub.strategy-live-runner` |
| Monit process | `roehub_strategy_live_runner` |
| Working directory | `/opt/roehub/app` |
| Config | `/opt/roehub/app/configs/prod/strategy.yaml` |
| Health | `http://127.0.0.1:9207/health/live` |
| Readiness | `http://127.0.0.1:9207/health/ready` |
| Metrics | `http://127.0.0.1:9207/metrics` |
| Logs | `~/Library/Logs/roehub/strategy-live-runner.{out,err}.log` |

## Producer Controls

Default repo config is fail-closed:

| Control | Default | Runtime env override |
|---|---:|---|
| Admin switch | `strategy.producer.enabled=false` | `ROEHUB_EXECUTION_STRATEGY_PRODUCER_ENABLED=true` |
| Allow all | `strategy.producer.allow_all=false` | `ROEHUB_STRATEGY_PRODUCER_ALLOW_ALL=true` |
| Allowed modes | `paper,testnet` only | `ROEHUB_STRATEGY_PRODUCER_ALLOWED_MODES=paper,testnet` |
| User allowlist | empty | `ROEHUB_STRATEGY_PRODUCER_ALLOWED_USER_IDS=<uuid>[,<uuid>]` |
| Strategy allowlist | empty | `ROEHUB_STRATEGY_PRODUCER_ALLOWED_STRATEGY_IDS=<uuid>[,<uuid>]` |

Do not add `live`, `mainnet`, `monitor_only`, exchange names, API keys or secrets to producer config. Stage `06` permits source-event production only for `paper`/`testnet` strategy profiles and only when the admin switch plus allowlist pass.

## Commands

```bash
launchctl print gui/$(id -u)/com.roehub.strategy-live-runner
monit status roehub_strategy_live_runner
curl -fsS http://127.0.0.1:9207/health/ready
curl -fsS http://127.0.0.1:9207/metrics | rg '^(strategy_producer_|strategy_live_runner_)'
```

Controlled restart:

```bash
/opt/homebrew/etc/monit.d/scripts/launchctl_service_control.sh stop com.roehub.strategy-live-runner /Users/daniildegtyarev/Library/LaunchAgents/com.roehub.strategy-live-runner.plist
/opt/homebrew/etc/monit.d/scripts/launchctl_service_control.sh start com.roehub.strategy-live-runner /Users/daniildegtyarev/Library/LaunchAgents/com.roehub.strategy-live-runner.plist
```

## Stage 06 Alert Actions

| Alert / symptom | Action |
|---|---|
| `/health/ready` fails | Keep producer admin switch disabled; inspect `strategy-live-runner.err.log`, Postgres DSN, Redis, ClickHouse and config parsing. |
| `/metrics` absent | Check launchd loaded state, Monit process match, port `9207`, and whether another process already binds the port. |
| `strategy_live_runner_iteration_errors_total` increases | Compare error logs with Postgres/Redis/ClickHouse health; do not broaden allowlists until the loop is stable. |
| `producer_mode_not_allowed` increases | Verify no `live`/mainnet profile is being routed through Stage `06`; keep allowed modes to `paper,testnet`. |
| `producer_allowlist_missing` increases | This is expected until a specific smoke user or strategy is allowlisted; add only scoped UUIDs through host env or config. |

## Stage 13 Notification And Runbook Alert Actions

Stage `13` uses `execution_notification_outbox` as a delivery-neutral fact ledger. Telegram/email delivery is outside this runbook: operators must inspect outbox rows, source events, intents, order ledgers, reconciliation rows, and producer metrics without copying secrets, cookies, DSNs, tokens, exchange keys, raw provider payloads, or session values.

| Alert / outbox event | Severity | Owner | Escalation | Operator action |
|---|---|---|---|---|
| `StrategyProducerExecutionRejected`; `producer_signal_rejected`, `producer_order_rejected`, legacy `producer_rejected` | warning | `strategy-producer` | Keep `paper,testnet` allowlists scoped and do not replay until the source event, risk audit, order guard, and config change are understood. | Query the matching `execution_notification_outbox` row by sanitized ids; inspect `execution_source_events`, `execution_intents`, `execution_risk_audit_events`, and `execution_orders` if an order row exists. |
| `StrategyProducerCriticalIncidentNotification`; `producer_kill_switch` | critical | `strategy-producer` | Stop broader fan-out and keep mainnet disabled until the kill-switch source and latest producer config are confirmed. | Check `/health/ready`, host env allowlists, recent manual/admin actions, and risk audit rows; restart only through managed `launchd`/`Monit` after an operator decision. |
| `StrategyProducerCriticalIncidentNotification`; `producer_unknown`, `producer_reconciliation_pending` | critical | `strategy-producer` | Do not blindly retry unknown order states. Reconcile durable order state and provider/testnet state first. | Inspect `execution_orders`, `execution_order_events`, `execution_reconciliation_runs`, Redis pending/DLQ, and exchange-execution logs before any replay. |
| `StrategyProducerRunStateNotification`; `producer_manual_exit`, `producer_strategy_stopped`, `producer_strategy_restarted` | warning | `strategy-producer` | Treat unexpected stop/restart loops as an incident; expected operator actions require no replay. | Confirm the strategy id/run id, action source, and current run state; compare with producer metrics and browser/API status. |
| `StrategyProducerRunStateNotification`; `producer_fill`, `producer_terminal`, `producer_soak_succeeded` | warning | `strategy-producer` | Verify the event belongs to the current test window and is not a duplicate. | Check source event/outbox dedupe keys, latest fills/reconciliation rows, and Stage `12`/`13` evidence before closing the action. |
| `StrategyProducerCriticalIncidentNotification`; `producer_soak_failed`, `producer_resource_threshold_breached` | critical | `strategy-producer` | Stop load expansion, preserve collector artifacts, and do not mark soak/load gates accepted until resource evidence is explained. | Inspect Stage `11`/`12` harness artifacts, Prometheus resource snapshots, Redis lag, retry/DLQ, Monit status, and process RSS/CPU rows. |

Non-destructive dry-run:

1. Use `/api/ui/execution/notifications` or an in-memory/local test client with `source_type=ops_test`.
2. Emit one dry-run event with a reason such as `stage13_runbook_dry_run` and labels limited to `stage`, `surface`, and `drill`.
3. Verify the outbox/API response contains the expected `event_type`, `severity`, redacted labels, `status=pending`, and no delivery-channel side effect.
4. Run Prometheus rule validation before deployment and record the rule names/severities in the stage report.
5. Do not send Telegram/email or mutate production exchange state as part of this dry-run.

## Acceptance Evidence Checklist

| Evidence | Command / source |
|---|---|
| launchd loaded | `launchctl print gui/$(id -u)/com.roehub.strategy-live-runner` |
| Monit controlled | `monit status roehub_strategy_live_runner` |
| Health/readiness | `curl -fsS :9207/health/live`, `curl -fsS :9207/health/ready` |
| Metrics bounded | `curl -fsS :9207/metrics`; confirm producer metrics use only `mode`, `outcome`, `reason`, `scope` labels. |
| Disabled switch blocks | `strategy_producer_admin_enabled 0` and `strategy_producer_skipped_strategies_total{reason="producer_disabled"}` after a controlled signal probe. |
| Missing allowlist blocks | `strategy_producer_skipped_strategies_total{reason="producer_allowlist_missing"}` with admin enabled and empty allowlists. |
| Allowlisted paper/testnet creates source event | Existing execution source-event table/metrics show a `strategy_signal` source event for the scoped user/strategy. |
