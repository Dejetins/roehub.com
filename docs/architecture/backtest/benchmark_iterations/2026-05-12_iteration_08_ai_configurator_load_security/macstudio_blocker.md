# Backtest AI Configurator Iteration 08 - Mac Studio Blocker

Captured at `2026-05-12T00:39:41+0300` local time
(`2026-05-11T21:39:41Z`) for commit
`388cd0f602f0641fcd4419ea0f44e4145531caaa`.

## Host Status

- Host: `MacStudioDaniil`
- Checkout: `/Users/daniildegtyarev/Projects/roehub.com`
- Checkout state: `main...origin/main`
- Worker launchd state: `running`
- Worker pid: `8464`
- Worker last exit code: `never exited`

## Blocking Checks

API admission probe:

```bash
ssh macstudio 'curl -sS -i http://127.0.0.1:8000/backtests/ai-config/jobs \
  -H "content-type: application/json" \
  --data "{\"mode\":\"create\",\"locale\":\"en\",\"message\":\"Create BTCUSDT RSI config\"}" | head -40'
```

Result: `401 auth.required`. The load harness needs an authenticated benchmark
session or a dedicated internal benchmark auth surface before API pipeline
acceptance can be measured.

Worker readiness probe:

```bash
ssh macstudio 'curl -sS -i http://127.0.0.1:9205/health/ready'
```

Result: `503 Service Unavailable`.

Ready payload:

```json
{
  "checks": {
    "config_loaded": true,
    "drain_mode": true,
    "model_path": true,
    "model_registry": true,
    "postgres_queue_audit": true,
    "queue_loop": true,
    "runtime_connection": false
  },
  "model_id": "gemma-4-e2b-it-4bit",
  "model_path_configured": true,
  "status": "not_ready"
}
```

Because `runtime_connection=false`, S1/S5/S10/S50/S100 would not measure
production MLX inference throughput. They were not run.

## Host Metrics Snapshot

- `process_resident_memory_bytes`: `170000384`
- `backtest_ai_config_active_generations{model_id="unknown"}`: `0`
- `backtest_ai_config_model_loaded{model_id="gemma-4-e2b-it-4bit"}`: `1`
- `vm_stat` page size: `16384`
- `vm_stat` pages free: `2865927`
- `memory_pressure` physical bytes: `68719476736`
- `memory_pressure` swapins/swapouts: `0/0`

## Scenario Status

| Scenario | Status |
|---|---|
| S1 | not run - blocked |
| S5 | not run - blocked |
| S10 | not run - blocked |
| S50 | not run - blocked |
| S100 | not run - blocked |
| security eval mix | local fake-worker only, not acceptance |

## Rollout Decision

Rollout is blocked.

Blockers:

- Mac Studio worker `/health/ready` is `503` because `runtime_connection=false`;
- authenticated benchmark API session is unavailable in this run;
- S1/S5/S10/S50/S100 Mac Studio acceptance scenarios were not run;
- local fake-worker security eval observed 3 unauthorized load-action cases.
