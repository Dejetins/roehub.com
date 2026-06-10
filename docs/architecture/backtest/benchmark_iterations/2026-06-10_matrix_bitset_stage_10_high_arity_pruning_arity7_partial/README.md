# Stage 10 high-arity pruning partial evidence

This directory records a stopped Mac Studio Stage 10 candidate run for exact-safe
high-arity pruning. The runtime candidate is not accepted.

## Scope

- Rule tested: exact-safe branch-and-bound by monotonic min-closed-trades bound.
- Candidate rule literal: `monotonic_min_closed_trades`.
- Approximate beam remains off; no approximate beam search was implemented.
- Candidate copy: `/tmp/roehub-stage10-candidate` on Mac Studio.
- Source checkout: local `main` at `d00e6afc7445d255533c9b97c5ea40c1167e819c`
  plus scoped diff hash
  `91dc3b6247ebc001288504ed241a4e9ee39fe4db33ee432b81280033b8b57c2e`.
- Env file: `/Users/daniildegtyarev/.config/roehub/roehub.env`.
- Artifact config path: `configs/prod/backtest_artifacts.yaml`.
- Secret values: not recorded.

## Exact-Safety Proof

For a partial indicator prefix, raw consensus at each signal bar is nonzero only
where every selected row in the prefix has the same nonzero direction. Adding
another indicator can only keep that direction or turn the bar to zero; it cannot
create a new nonzero consensus bar outside the partial consensus.

After applying `direction_mode`, every closed trade in both current no-risk and
TP/SL scoring requires an entry on a nonzero consensus bar. Therefore any
descendant of a prefix with fewer nonzero direction-adjusted consensus bars than
`quality_constraints.min_closed_trades` cannot become heap-eligible under the
current quality gate. Pruning that subtree cannot remove a valid top candidate.

This proof is only a quality-gate bound. It is not a score upper bound and does
not justify pruning candidates that still can satisfy `min_closed_trades`.

## Mac Studio Partial Result

Command started:

```bash
ssh macstudio 'cd /tmp/roehub-stage10-candidate && \
  /opt/homebrew/bin/uv run python scripts/backtest/run_api_runner_benchmark_parity.py \
    --env-file /Users/daniildegtyarev/.config/roehub/roehub.env \
    --out-dir docs/architecture/backtest/benchmark_iterations/2026-06-10_matrix_bitset_stage_10_high_arity_pruning_arity7 \
    --stage-10-high-arity-pruning \
    --timeout-seconds 7200 \
    --poll-interval-seconds 0.5 \
    --system-memory-cleanup-wait-seconds 90'
```

The run was stopped after the second arity-7 child remained active for about nine
minutes. The first completed row wrote child evidence:

- `child_process_evidence/full-job-29bc0e24-cf87-45a4-a7be-129e699de168-17028-20260609T234801091812Z.json`
- `child_process_evidence/full-job-result-29bc0e24-cf87-45a4-a7be-129e699de168-20260609T234801093765Z.json`

Completed row counters:

| Metric | Value |
|---|---:|
| `combo_count_planned` | `279,936` |
| `candidates_after_proxy` | `116,640` |
| `exact_candidates` | `116,640` |
| `combo_pruning_pruned_subtrees` | `3,246` |
| `combo_pruning_pruned_candidate_upper_bound` | `163,296` |
| `combo_iteration` | `59.350s` |
| `exact_scoring` | `58.182s` |
| `service_total_without_warmup` | `119.252s` |

## Decision

Stage 10 is rejected for this candidate. The pruning rule is exact-safe but not
accepted: the branch traversal adds a large `combo_iteration` cost, no comparable
baseline-off speedup was completed, the full arity-7 API-runner matrix did not
complete, and arity-10 acceptance is blocked because the current canonical
benchmark fixture contains only seven indicators.

Next Stage 10 attempt should use a tighter exact-safe score or eligibility bound,
or add an approved ten-indicator benchmark fixture before claiming arity-10
evidence.
