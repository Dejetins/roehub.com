# Backtest AI Configurator Assistant v1 — Ops Runbook

Дата: 2026-05-18.

Статус: Iteration 07 operations contract.

## Services

Production на Mac Studio состоит из двух локальных границ:

- `roehub_lmstudio_backtest_ai_runtime` — Monit `check program`, one-shot launchd ensure `com.roehub.lmstudio-backtest-ai-runtime`, loopback-only LM Studio runtime.
- `roehub_backtest_ai_configurator_worker` — launchd service `com.roehub.backtest-ai-configurator-worker`, Monit-managed worker and metrics process.

LM Studio не должен слушать публичный интерфейс. Source of truth для runtime:
`configs/prod/backtest_ai_configurator.yaml`, `model.runtime: lm_studio`,
`model.base_url: http://127.0.0.1:8080`.

## Readiness

Worker readiness endpoint:

```text
http://127.0.0.1:9205/health/ready
```

Ready только если выполнены все проверки:

- worker loop started and not in drain mode;
- config loaded;
- configured model path exists;
- PostgreSQL queue audit query succeeds;
- LM Studio smoke confirms loaded model plus lightweight generation.

LM Studio smoke:

```bash
/opt/roehub/app/scripts/macos/smoke_lmstudio_backtest_ai_runtime.sh \
  --config /opt/roehub/app/configs/prod/backtest_ai_configurator.yaml \
  --json
```

Smoke проверяет port preflight, `lms ps --json`, `/api/v1/models` loaded instance
и lightweight `POST /v1/chat/completions` с `response_format.type=json_schema`.
`/v1/models` alone is insufficient.

## Health and Metrics

Worker endpoints:

```bash
curl -fsS http://127.0.0.1:9205/health/live
curl -fsS http://127.0.0.1:9205/health/ready
curl -fsS http://127.0.0.1:9205/metrics | rg 'backtest_ai_config_|process_resident_memory_bytes'
```

Required metric families:

- `backtest_ai_config_jobs_total{status,intent,tier,model_id}`;
- `backtest_ai_config_jobs_inflight{intent,model_id}`;
- `backtest_ai_config_queue_depth{priority}`;
- `backtest_ai_config_queue_wait_seconds_bucket{intent,tier,model_id}`;
- `backtest_ai_config_total_latency_seconds_bucket{intent,tier,model_id}`;
- `backtest_ai_config_llm_latency_seconds_bucket{model_id,attempt_kind}`;
- `backtest_ai_config_validation_failures_total{code}`;
- `backtest_ai_config_repair_attempts_total{result,model_id}`;
- `backtest_ai_config_security_decisions_total{decision,flag}`;
- `backtest_ai_config_load_action_total{result}`;
- `backtest_ai_config_high_load_responses_total{reason}`;
- `backtest_ai_config_conversations_total{status}`;
- `backtest_ai_config_messages_total{role,intent}`;
- `backtest_ai_config_model_loaded{model_id}`;
- `backtest_ai_config_model_reload_total{result,model_id}`.

Prometheus target: `backtest-ai-configurator-worker` at `127.0.0.1:9205`.

## Monit Commands

```bash
/opt/homebrew/opt/monit/bin/monit -t -c /opt/homebrew/etc/monitrc
/opt/homebrew/opt/monit/bin/monit -c /opt/homebrew/etc/monitrc reload
/opt/homebrew/opt/monit/bin/monit -c /opt/homebrew/etc/monitrc summary | rg 'roehub_(lmstudio_backtest_ai_runtime|backtest_ai_configurator_worker)'
/opt/homebrew/opt/monit/bin/monit -c /opt/homebrew/etc/monitrc stop roehub_backtest_ai_configurator_worker
/opt/homebrew/opt/monit/bin/monit -c /opt/homebrew/etc/monitrc start roehub_backtest_ai_configurator_worker
/opt/homebrew/opt/monit/bin/monit -c /opt/homebrew/etc/monitrc restart roehub_backtest_ai_configurator_worker
```

Acceptance requires two stop/start/restart cycles without restart loop:

```bash
for i in 1 2; do
  /opt/homebrew/opt/monit/bin/monit -c /opt/homebrew/etc/monitrc stop roehub_backtest_ai_configurator_worker
  sleep 3
  /opt/homebrew/opt/monit/bin/monit -c /opt/homebrew/etc/monitrc start roehub_backtest_ai_configurator_worker
  sleep 10
  curl -fsS http://127.0.0.1:9205/health/live
  curl -fsS http://127.0.0.1:9205/health/ready
  curl -fsS http://127.0.0.1:9205/metrics | rg 'backtest_ai_config_jobs_total|backtest_ai_config_model_loaded'
  /opt/homebrew/opt/monit/bin/monit -c /opt/homebrew/etc/monitrc restart roehub_backtest_ai_configurator_worker
  sleep 10
  curl -fsS http://127.0.0.1:9205/health/ready
done
```

No restart loop means `monit summary` keeps both services in `Status ok` /
`Running` or `Accessible`, and does not show `unmonitor`.

## Autostart

Autostart is provided by user launchd plists with `RunAtLoad=true`:

- `infra/macos/launchd/com.roehub.lmstudio-backtest-ai-runtime.plist`;
- `infra/macos/launchd/com.roehub.backtest-ai-configurator-worker.plist`.

Bootstrap/reload:

```bash
bash /opt/roehub/app/scripts/macos/bootstrap_native_prod.sh
bash /opt/roehub/app/scripts/macos/reload_launchd_services.sh prod
```

After reboot/login, verify:

```bash
launchctl print gui/$(id -u)/com.roehub.lmstudio-backtest-ai-runtime | grep -E 'state =|last exit code ='
launchctl print gui/$(id -u)/com.roehub.backtest-ai-configurator-worker | grep -E 'state =|pid =|last exit code ='
```
