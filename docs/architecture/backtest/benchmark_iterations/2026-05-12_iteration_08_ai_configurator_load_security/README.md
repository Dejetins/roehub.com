# Backtest AI Configurator Iteration 08 - Load and Security Evidence

This folder records the reusable Iteration 08 benchmark harness and current
rollout status for the `/backtests` AI configurator.

## Scope

- Implemented:
  - reusable API pipeline load harness:
    `scripts/backtest_ai/run_configurator_load_test.py`;
  - reusable security eval harness:
    `scripts/backtest_ai/run_configurator_security_eval.py`;
  - shared prompt mix, S1/S5/S10/S50/S100 scenario definitions, metrics
    aggregation, Mac Studio snapshot collection and deterministic fake-worker
    smoke support in `scripts/backtest_ai/configurator_benchmark_common.py`;
  - unit coverage for header parsing, scenario selection, percentile
    aggregation, final valid config rate, quota/capacity friendliness and
    security failure classification.
- Not in scope:
  - paid-user rollout;
  - increasing production concurrency;
  - treating local fake-worker output as acceptance evidence.

## Benchmark Plan

Load harness command shape:

```bash
uv run python scripts/backtest_ai/run_configurator_load_test.py \
  --base-url https://<prod-or-private-api-host>/api \
  --header 'Authorization: Bearer <redacted>' \
  --all-scenarios \
  --macstudio-host macstudio \
  --metrics-url http://127.0.0.1:9205/metrics \
  --out-dir docs/architecture/backtest/benchmark_iterations/2026-05-12_iteration_08_ai_configurator_load_security
```

Security eval command shape:

```bash
uv run python scripts/backtest_ai/run_configurator_security_eval.py \
  --base-url https://<prod-or-private-api-host>/api \
  --header 'Authorization: Bearer <redacted>' \
  --out-dir docs/architecture/backtest/benchmark_iterations/2026-05-12_iteration_08_ai_configurator_load_security
```

Local fake-worker smoke command shape:

```bash
uv run python scripts/backtest_ai/run_configurator_load_test.py \
  --fake-worker --scenario S1 --duration-scale 0.001 --max-requests-per-scenario 3

uv run python scripts/backtest_ai/run_configurator_security_eval.py \
  --fake-worker
```

## Required Mac Studio Evidence

Acceptance evidence is still required for:

- S1;
- S5;
- S10;
- S50;
- S100;
- `security eval mix`;
- `final valid config rate`;
- `active_generations`;
- `memory_pressure`;
- `vm_stat`;
- worker RSS/swap/restart behavior.

The load generator must run off Mac Studio. Mac Studio is only the inference and
worker host under test.

## Current Rollout Decision

Rollout status: blocked until Mac Studio benchmark evidence is recorded.

Current blocker evidence:

- `macstudio_blocker.md`;
- `macstudio_blocker.json`;
- local fake-worker security eval observed 3 unauthorized load-action cases:
  `secrets_env_vars`, `output_script_injection`, `auto_run_backtest_attempt`.

Accepted runtime settings:

- model id: not accepted yet;
- model path hash: not accepted yet;
- context window: not accepted yet;
- max output tokens: not accepted yet;
- active generations: not accepted yet;
- queue limits: not accepted yet.

Current production config keeps AI configurator disabled. Existing high queue
and concurrency limit literals are not accepted public defaults; they require
replacement only after Mac Studio evidence supports concrete values.
