# LM Studio Service Lifecycle - Mac Studio

Service lifecycle gate for `/backtests` AI Configurator LM Studio serving.

This iteration proves production-operable local deployment on Mac Studio:
automatic startup, idempotent ensure, loaded-model readiness, Monit control and
post-reload recovery. It is not an S1/S5/S10/S50/S100 benchmark.

## Gate Verdict

- accepted: false
- blocking_reason: implementation pending Mac Studio deployment verification
- next_prompt_allowed: false
- host: `MacStudioDaniil`
- runtime: `lm_studio`
- launchd label: `com.roehub.lmstudio-backtest-ai-runtime`
- Monit check: `roehub_lmstudio_backtest_ai_runtime`
- model identifier: `gemma-4-e2b-it-4bit`
- config source: `base_url from configs/prod/backtest_ai_configurator.yaml`

## Implemented Control Plane

- Ensure command:
  `/opt/roehub/app/scripts/macos/ensure_lmstudio_backtest_ai_runtime.sh --config /opt/roehub/app/configs/prod/backtest_ai_configurator.yaml --json`
- Smoke command:
  `/opt/roehub/app/scripts/macos/smoke_lmstudio_backtest_ai_runtime.sh --config /opt/roehub/app/configs/prod/backtest_ai_configurator.yaml --json`
- Absolute `lms` path:
  `/Users/daniildegtyarev/.lmstudio/bin/lms`
- Startup commands:
  - `lms daemon up`
  - `lms server start --port 8080 --bind 127.0.0.1`
  - `lms load gemma-4-e2b-it --identifier gemma-4-e2b-it-4bit --context-length 8192 --parallel 1`

The port is not blindly hardcoded by the script: it is resolved from
`configs/prod/backtest_ai_configurator.yaml` model `base_url`; `8080` is the
current configured value.

## Restart-Storm Avoidance

The selected safe design is:

- `launchd` label `com.roehub.lmstudio-backtest-ai-runtime` is a one-shot
  `RunAtLoad` ensure job with `KeepAlive=false`;
- Monit check `roehub_lmstudio_backtest_ai_runtime` is a `check program` that
  calls the idempotent ensure/smoke path;
- Monit `restart` runs script-level stop/start, not a long-running process
  restart expectation.

This is intentional because `lms server start` starts a background server and
returns. A launchd `KeepAlive=true` wrapper around that command would risk a
successful-exit restart loop.

## Readiness Contract

`/v1/models is not readiness`.

The runtime smoke accepts readiness only when all checks pass:

- port preflight confirms the configured loopback port is not owned by another
  service and is not bound publicly;
- `lms ps --json` shows loaded identifier `gemma-4-e2b-it-4bit` with context
  `8192` and `parallel=1`;
- `GET /api/v1/models` shows the loaded instance;
- `POST /v1/chat/completions` returns structured JSON through
  `response_format.type=json_schema`;
- all JSON Schema `type` values are strings; do not use
  `type: ["string", "null"]`;
- `choices[0].message.content` is parsed as JSON and includes
  `accepted=true`, string `blocking_reason`, and `next_prompt_allowed=true`.

The smoke uses a bounded retry loop for transient post-restart HTTP connection
closures while LM Studio is reattaching the server/model. Port conflict and
public-bind failures are not retried.

## Verification To Record After Deploy

```bash
bash scripts/macos/bootstrap_native_prod.sh
bash scripts/macos/reload_launchd_services.sh prod
/opt/homebrew/opt/monit/bin/monit -t -c /opt/homebrew/etc/monitrc
/opt/homebrew/opt/monit/bin/monit -c /opt/homebrew/etc/monitrc reload
launchctl print gui/$(id -u)/com.roehub.lmstudio-backtest-ai-runtime
/opt/homebrew/opt/monit/bin/monit -c /opt/homebrew/etc/monitrc status roehub_lmstudio_backtest_ai_runtime
/opt/roehub/app/scripts/macos/smoke_lmstudio_backtest_ai_runtime.sh --config /opt/roehub/app/configs/prod/backtest_ai_configurator.yaml --json
```

Two stop/start/restart cycles must be recorded before this gate can be accepted.

## Machine-Readable Summary

See `lmstudio_service_lifecycle.json` in this directory.
