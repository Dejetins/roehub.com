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
