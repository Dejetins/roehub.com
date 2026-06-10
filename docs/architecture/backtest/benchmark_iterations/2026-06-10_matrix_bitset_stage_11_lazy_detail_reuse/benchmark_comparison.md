# Stage 11 lazy detail reuse comparison

## Provenance

- Host: `MacStudioDaniil`
- Base commit: `34fac40074bcf082e86a4396daae3ab6dbdde1a3`
- Candidate: base commit plus scoped Stage 11 dirty patch.
- Candidate patch SHA-256: `c410a8783202c2fcb2cf9008899bb0c0ff05b1011b92e53a5be1f6ce8cc06d92`
- Candidate worktree: `/tmp/roehub-stage11-candidate`
- Baseline worktree: `/tmp/roehub-stage11-baseline`
- Env keys set for benchmark: `ROEHUB_ENV`, `ROEHUB_BACKTEST_ARTIFACTS_CONFIG`
- Env file values were not printed.
- Artifact config path: `configs/prod/backtest_artifacts.yaml`

## Commands

Baseline:

```bash
ROEHUB_ENV=prod \
ROEHUB_BACKTEST_ARTIFACTS_CONFIG=configs/prod/backtest_artifacts.yaml \
/opt/homebrew/bin/uv run python scripts/backtest/run_iteration_9_lazy_trades_benchmark.py \
  --artifact-config configs/prod/backtest_artifacts.yaml \
  --out-dir docs/architecture/backtest/benchmark_iterations/2026-06-10_matrix_bitset_stage_11_lazy_detail_reuse_baseline \
  --cache-root docs/architecture/backtest/benchmark_iterations/2026-06-10_matrix_bitset_stage_11_lazy_detail_reuse_baseline/sidecar_artifacts/lazy_trades_cache \
  --arity 1 \
  --direction-mode long_short_reversal \
  --rows-per-indicator 6 \
  --self-check-n 1
```

Candidate:

```bash
ROEHUB_ENV=prod \
ROEHUB_BACKTEST_ARTIFACTS_CONFIG=configs/prod/backtest_artifacts.yaml \
/opt/homebrew/bin/uv run python scripts/backtest/run_iteration_9_lazy_trades_benchmark.py \
  --artifact-config configs/prod/backtest_artifacts.yaml \
  --out-dir docs/architecture/backtest/benchmark_iterations/2026-06-10_matrix_bitset_stage_11_lazy_detail_reuse \
  --cache-root docs/architecture/backtest/benchmark_iterations/2026-06-10_matrix_bitset_stage_11_lazy_detail_reuse/sidecar_artifacts/lazy_trades_cache \
  --arity 1 \
  --direction-mode long_short_reversal \
  --rows-per-indicator 6 \
  --self-check-n 1
```

## Results

| Risk mode | Baseline miss s | Candidate miss s | Miss delta | Baseline hit s | Candidate hit s | Hit delta | Parity |
|---|---:|---:|---:|---:|---:|---:|---|
| `none` | 2.869005 | 2.855623 | -0.466% | 0.000305 | 0.000299 | -2.090% | pass |
| `tp_sl_grid` | 4.334214 | 4.292836 | -0.955% | 0.000301 | 0.000301 | -0.207% | pass |

Interpretation: Stage 11 preserves selected-variant lazy payload/cache identity
and reuses the sparse trade tape backend for TP/SL lazy materialization, but the
same-workload latency delta is small enough to treat as no material speedup.
The stage is accepted as a safe reuse and parity gate, not as a major latency
acceleration.
