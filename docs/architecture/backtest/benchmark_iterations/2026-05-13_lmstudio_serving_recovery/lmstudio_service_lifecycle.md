# LM Studio Service Lifecycle - Mac Studio

Historical service lifecycle gate for `/backtests` AI Configurator LM Studio
serving.

Supersession note: on 2026-05-16 the single-shot structured-generation
readiness contract was retired. The lifecycle evidence below remains useful for
LM Studio daemon/server/model operations, but it is not current AI Configurator
acceptance and must not unlock rollout.

This iteration proves production-operable local deployment on Mac Studio:
automatic startup, idempotent ensure, loaded-model readiness, Monit control and
post-reload recovery. It is not an S1/S5/S10/S50/S100 benchmark.

## Current Gate Verdict

- accepted: false
- blocking_reason: superseded by single-shot contract retirement
- next_prompt_allowed: true
- historical_accepted_at_collection_time: true
- host: `MacStudioDaniil`
- timestamp UTC: `2026-05-15T22:17:35Z`
- deployed runtime commit: `5e43906023997eba44c199bc4ce16c76eb65fc6a`
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

## Retired Readiness Contract

`/v1/models is not readiness`.

At collection time, the runtime smoke accepted readiness only when all checks
passed:

- port preflight confirms the configured loopback port is not owned by another
  service and is not bound publicly;
- `lms ps --json` shows loaded identifier `gemma-4-e2b-it-4bit` with context
  `8192` and `parallel=1`;
- `GET /api/v1/models` shows the loaded instance;
- retired: `POST /v1/chat/completions` returns structured JSON through
  `response_format.type=json_schema`;
- all JSON Schema `type` values are strings; do not use
  `type: ["string", "null"]`;
- `choices[0].message.content` is parsed as JSON and includes
  `accepted=true`, string `blocking_reason`, and `next_prompt_allowed=true`.

The current smoke no longer performs the structured-generation probe. It checks
only LM Studio lifecycle/model-loaded state; tool-agent acceptance must be
introduced by the new contract.

## Mac Studio Verification

```bash
bash scripts/macos/bootstrap_native_prod.sh
bash scripts/macos/reload_launchd_services.sh prod
/opt/homebrew/opt/monit/bin/monit -t -c /opt/homebrew/etc/monitrc
/opt/homebrew/opt/monit/bin/monit -c /opt/homebrew/etc/monitrc reload
launchctl print gui/$(id -u)/com.roehub.lmstudio-backtest-ai-runtime
/opt/homebrew/opt/monit/bin/monit -c /opt/homebrew/etc/monitrc status roehub_lmstudio_backtest_ai_runtime
/opt/roehub/app/scripts/macos/smoke_lmstudio_backtest_ai_runtime.sh --config /opt/roehub/app/configs/prod/backtest_ai_configurator.yaml --json
```

Recorded evidence:

| Check | Result |
| --- | --- |
| `git pull --ff-only origin main` on Mac Studio checkout | `5e43906023997eba44c199bc4ce16c76eb65fc6a` |
| `bootstrap_native_prod.sh` + `reload_launchd_services.sh prod` | installed and bootstrapped `com.roehub.lmstudio-backtest-ai-runtime` |
| Monit syntax/reload | `Control file syntax OK`, `Reinitializing monit daemon` |
| Clean cycle 1 stop | `lms server status --json --quiet` returned `{"running":false,"port":8080}` |
| Clean cycle 1 start smoke | `accepted=True`, `next_prompt_allowed=True`, `attempts=1`, listener `127.0.0.1:8080 (LISTEN)` |
| Clean cycle 1 restart smoke | `accepted=True`, `next_prompt_allowed=True`, `attempts=1`, listener `127.0.0.1:8080 (LISTEN)` |
| Clean cycle 2 stop | `lms server status --json --quiet` returned `{"running":false,"port":8080}` |
| Clean cycle 2 start smoke | `accepted=True`, `next_prompt_allowed=True`, `attempts=1`, listener `127.0.0.1:8080 (LISTEN)` |
| Clean cycle 2 restart smoke | `accepted=True`, `next_prompt_allowed=True`, `attempts=1`, listener `127.0.0.1:8080 (LISTEN)` |
| Monit final status | `roehub_lmstudio_backtest_ai_runtime` `OK`, `last exit value 0`, `monitoring status Monitored` |
| launchd final status | `state = not running`, `last exit code = 0`; expected for one-shot `KeepAlive=false` ensure job |
| `lms ps --json` | `identifier=gemma-4-e2b-it-4bit`, `contextLength=8192`, `parallel=1`, `status=idle` |
| Worker readiness after companion smoke | Historical: `/health/ready` returned `status=ready`; superseded for tool-agent rollout |
| Production smoke | `bash /opt/roehub/app/scripts/macos/smoke_prod.sh` passed |

No restart storm was observed: Monit stayed `Monitored`, final status was `OK`,
final `last exit value` was `0`, and the launchd one-shot exited with code `0`
instead of being kept alive.

## Machine-Readable Summary

See `lmstudio_service_lifecycle.json` in this directory.
