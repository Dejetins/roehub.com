# Iteration 07 Ops/Monit/Metrics

Дата: 2026-05-18.

Статус: accepted.

## Предварительный gate

Iteration 06 проверен перед началом:

- `implementation_progress.json`: `06-ui.accepted=true`;
- `next_iteration_allowed=true`;
- `iteration_06_ui.json`: `pushed_to_main=true`;
- `macstudio_verified=true`;
- recorded accepted commit: `4ce800699de3d7b1fbf49485b1498e0410c1fda6`;
- current `origin/main` содержит Iteration 06 delivery evidence commit `90e88351`.

## Что изменено

- Worker readiness больше не использует placeholder `assistant_v1_runtime_pending`.
- `/health/ready` требует LM Studio loaded model plus lightweight generation smoke:
  `lms ps --json`, `/api/v1/models` loaded instance и `POST /v1/chat/completions`
  с `response_format.type=json_schema`.
- Worker config получил readiness smoke timeout/cache env keys and production
  launchd exports.
- Monit worker check now probes `/health/live`, `/health/ready`, and `/metrics`.
- Prometheus metrics surface expanded for request/status, queue wait, LLM latency
  by attempt kind, validation/repair, load_action, security, high-load responses,
  conversations/messages, and model loaded state.
- Added `ops_runbook.md` for Mac Studio lifecycle and acceptance commands.
- Updated Mac Studio monitoring docs to state that `/v1/models` alone is not readiness.

## Локальные проверки

Completed locally before delivery:

```text
uv run pytest -q tests/unit/apps/worker tests/unit/contexts/backtest/application/ai_configurator
```

Result: `84 passed`.

```text
uv run ruff check apps/worker src/trading/contexts/backtest infra tests/unit/apps/worker
```

Result: passed.

```text
uv run pyright
```

Result: passed, `0 errors`.

```text
uv run python -m tools.docs.generate_docs_index --check
```

Result: passed.

## Mac Studio acceptance

Mac Studio verification completed after direct-main deploy.

- host: `MacStudioDaniil`;
- repo path: `/Users/daniildegtyarev/projects/roehub.com`;
- repo commit: `1890236f0a05499e91ad3197c7dcd844971bbd97`;
- runtime path: `/opt/roehub/app`;
- runtime worker file hashes match repo for `modules.py` and `observability.py`;
- `scripts/macos/smoke_prod.sh`: passed;
- LM Studio smoke: accepted, loaded `gemma-4-e2b-it-4bit`, `lms ps --json` matched context `8192` and parallel `1`;
- lightweight generation smoke: `POST /v1/chat/completions`, `response_format=json_schema`, accepted;
- `/health/live`: status `live`;
- `/health/ready`: status `ready`, `lmstudio_loaded_generation_smoke=true`;
- `/metrics`: scraped required `backtest_ai_config_*` series including `load_action`, `high_load`, and `model_loaded`;
- Prometheus: `up{job="backtest-ai-configurator-worker"} == 1`;
- Monit: `roehub_backtest_ai_configurator_worker` `OK`, `Monitored`, `on reboot start`;
- Monit: `roehub_lmstudio_backtest_ai_runtime` `OK`, `Monitored`, `on reboot start`;
- lifecycle: two stop/start/restart cycles passed without restart loop.

## Contract Impact

| Surface | Classification | Notes |
| --- | --- | --- |
| Public API contract | compatible-change | Adds/strengthens worker-local operational endpoints semantics for `/health/ready`; `/health/live` and `/metrics` remain scrapeable. |
| Browser-visible behavior | none | No `/backtests` UI behavior changed. |
| DTO schema | none | User API DTOs unchanged. |
| Persisted schema | none | No migrations or storage shape changes. |
| Config schema | compatible-change | Adds ignored/documented prod `worker` ops keys and launchd envs; existing runtime loader remains backward-compatible. |
| Request hash/cache identity | none | Backtest request identity and AI job hashes unchanged. |

## Delivery

Direct-main delivery completed.

- implementation commit: `1890236f0a05499e91ad3197c7dcd844971bbd97`;
- pushed to `origin/main`: true;
- `CI`: success, run `26044314216`;
- `Publish App Image`: success, run `26044314218`;
- `Deploy Backend`: success, run `26044314673`;
- `Deploy Web`: success, run `26044361400`.

Acceptance marker:

- accepted: true;
- pushed to `origin/main`: true;
- Mac Studio verified: true;
- next iteration allowed: true.
