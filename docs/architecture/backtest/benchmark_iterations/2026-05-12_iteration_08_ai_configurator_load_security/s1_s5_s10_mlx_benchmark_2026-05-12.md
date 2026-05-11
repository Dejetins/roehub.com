# Backtest AI Configurator Iteration 08 Load Benchmark

API pipeline benchmark harness for `/backtests` AI configurator scenarios.

## Version

- Branch: main
- Commit: bd56cdb96bb961266d0bc5008f9c69d14f09b974
- Config: configs/prod/backtest_ai_configurator.yaml
- Config SHA256: 57e9cd392e7d5c7a43f1b1dba63846b2ca98be9615cd08d979816640d86ff46f
- Model id: gemma-4-e2b-it-4bit
- Model path hash: 0b04fef753d2c1eeb2a4908ed08b9427368dffbc18bae65ccf4cfec4d57ee7d0
- context_window: 8192
- max_output_tokens: 256
- active_generations: 1
- queue limits: {'max_queue_size': 1000000000000, 'max_active_generations': 1000000000000, 'request_timeout_sec': 240, 'queue_timeout_sec': 300}

## Scenario Metrics

Run scope: S1/S5/S10 only per operator request on 2026-05-12; S50/S100 were not run in
this pass. Each scenario used `--max-requests-per-scenario 10`, so this is bounded
Mac Studio evidence, not full public rollout acceptance.

Command shape:

```bash
uv run python scripts/backtest_ai/run_configurator_load_test.py \
  --base-url http://127.0.0.1:18080 \
  --header 'Cookie: roehub_session_id=<redacted>' \
  --scenario S1 --scenario S5 --scenario S10 \
  --max-requests-per-scenario 10 \
  --poll-interval-seconds 1 \
  --job-timeout-seconds 300 \
  --http-timeout-seconds 60 \
  --metrics-url http://127.0.0.1:19205/metrics \
  --macstudio-host macstudio \
  --json-name s1_s5_s10_mlx_benchmark_2026-05-12.json \
  --markdown-name s1_s5_s10_mlx_benchmark_2026-05-12.md
```

Runtime note: `mlx_lm.server` 0.31.3 could not generate from the supplied
`gemma-4-e2b-it-4bit` checkpoint (`ValueError: Received 140 parameters not in model`).
The run below used LM Studio's local OpenAI-compatible server bound to
`127.0.0.1:8080`, with the same model folder loaded as identifier
`gemma-4-e2b-it-4bit`, context 8192 and parallel 1.

| scenario | requests | p50_total_ms | p95_total_ms | p99_total_ms | p95_queue_ms | final valid config rate | repair_rate | quota_capacity |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| S1 | 10 | 2889.406 | 7297.588 | 8408.448 | 4752.733 | 0.000 | 0.000 | 0 |
| S5 | 10 | 3342.448 | 6128.007 | 6511.159 | 5722.344 | 0.000 | 0.000 | 0 |
| S10 | 10 | 11839.434 | 17833.917 | 18213.501 | 16944.346 | 0.000 | 0.000 | 0 |

Status/error breakdown from persisted Mac Studio jobs:

| state | last_error | message | count |
| --- | --- | --- | --- |
| blocked_by_policy | blocked_by_policy |  | 3 |
| failed | worker_runtime_error | MLX generate request returned HTTP 400 | 20 |
| failed | worker_runtime_error | MLX repair request returned HTTP 400 | 1 |
| failed | worker_runtime_error | Object of type mappingproxy is not JSON serializable | 6 |

## Host Evidence

- Load generator host is recorded in JSON.
- Load generator host: `MacBook-Pro-Daniil.local`, macOS 15.7.4 arm64.
- Mac Studio model host status: worker readiness was `ready`; LM Studio model
  `gemma-4-e2b-it-4bit` loaded with context 8192 and parallel 1.
- `vm_stat`/`memory_pressure`: swap remained 0 before and after the run.
- `memory_pressure` and `vm_stat` snapshots are recorded in JSON when `--macstudio-host` is used.
- `active_generations` is recorded from config identity; live worker metric snapshots are recorded when `--metrics-url` is used.

## Rollout Decision

- Accepted: False
- Reason: rollout blocked
- Blockers: missing required scenarios: ['S100', 'S50'], S1 final valid config rate below 98%, S5 final valid config rate below 98%, S10 final valid config rate below 98%
