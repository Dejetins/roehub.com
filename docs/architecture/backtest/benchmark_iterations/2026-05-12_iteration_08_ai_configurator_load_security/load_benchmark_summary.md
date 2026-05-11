# Backtest AI Configurator Iteration 08 Load Benchmark

API pipeline benchmark harness for `/backtests` AI configurator scenarios.

## Version

- Branch: main
- Commit: 388cd0f602f0641fcd4419ea0f44e4145531caaa
- Config: configs/prod/backtest_ai_configurator.yaml
- Config SHA256: 5d0f77e7d891c9fdc9d437a87dabab858b55eb1d48c3fbe532fe41934fd5e255
- Model id: gemma-4-e2b-it-4bit
- Model path hash: 0b04fef753d2c1eeb2a4908ed08b9427368dffbc18bae65ccf4cfec4d57ee7d0
- context_window: 8192
- max_output_tokens: 1024
- active_generations: 1
- queue limits: {'max_queue_size': 1000000000000, 'max_active_generations': 1000000000000, 'request_timeout_sec': 90, 'queue_timeout_sec': 180}

## Scenario Metrics

| scenario | requests | p50_total_ms | p95_total_ms | p99_total_ms | p95_queue_ms | final valid config rate | repair_rate | quota_capacity |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| S1 | 3 | 3.341 | 6.934 | 7.253 | 0.603 | 1.000 | 0.000 | 0 |

## Host Evidence

- Load generator host is recorded in JSON.
- `memory_pressure` and `vm_stat` snapshots are recorded in JSON when `--macstudio-host` is used.
- `active_generations` is recorded from config identity; live worker metric snapshots are recorded when `--metrics-url` is used.

## Rollout Decision

- Accepted: False
- Reason: rollout blocked
- Blockers: local fake-worker smoke is not Mac Studio acceptance evidence, missing required scenarios: ['S10', 'S100', 'S5', 'S50'], Mac Studio host snapshots were not collected
