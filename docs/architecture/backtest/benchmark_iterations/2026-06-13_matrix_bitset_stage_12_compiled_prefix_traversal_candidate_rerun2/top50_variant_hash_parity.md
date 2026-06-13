# Stage 12 top-50 variant hash parity

This check compared completed Mac Studio API jobs from:

- baseline:
  `../2026-06-13_matrix_bitset_stage_12_compiled_prefix_traversal_baseline_off/benchmark_results.json`
- candidate:
  `benchmark_results.json`

The comparison fetched `/backtests/jobs/{job_id}/top` for each completed job
using the benchmark smoke-session API path and compared job-id-independent
fields: `rank`, `variant_hash`, and `summary_metrics`.

`variant_key` was intentionally excluded because it embeds the job id and
therefore differs across otherwise identical baseline and candidate jobs.

| Job | Baseline rows | Candidate rows | Stable top-50 equal |
|---|---:|---:|---|
| `none/arity_6/long_only` | `50` | `50` | `true` |
| `none/arity_6/long_short_reversal` | `0` | `0` | `true` |
| `none/arity_7/long_only` | `50` | `50` | `true` |
| `none/arity_7/long_short_reversal` | `0` | `0` | `true` |

Command shape:

```bash
ssh macstudio 'cd /tmp/roehub-stage12-candidate-20260613-compiled-prefix && \
  /opt/homebrew/bin/uv run python <top50 comparison helper>'
```

Secret values were not printed or recorded.
