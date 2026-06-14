# Stage 15 TP/SL Early-Abandon Rejection Summary

Stage 15 tested an exact-safe optimistic log-return upper bound for
`ranking=total_return_pct desc` on current exact TP/SL scoring. The candidate
was benchmarked from isolated Mac Studio checkout
`/tmp/roehub-stage15-candidate-20260614-tp-sl-early-abandon` at
`80a4391f+dirty-4b7ae7aa`; dirty diff hash:
`4b7ae7aa6315879b48918c08e54d48d21ed9b64f3d28efbcc4de97db3054072f`.

## Decision

Status: `accepted_for_learning`; runtime candidate `rejected`.

The candidate preserved parity and memory cleanup, but it did not prune any
candidate on the mandatory TP/SL rows and did not improve service wall.
Runtime code was removed from the active tree; only evidence and this summary
remain. Stage 16 trade-window reuse telemetry is unblocked as the next
measurement-only TP/SL step.

## A/B Evidence

| Job | Control wall s | Candidate wall s | Wall delta | Control total no-warmup s | Candidate total no-warmup s | Total delta | Control exact s | Candidate exact s | Exact delta | Pruned candidates | Bound ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `tp_sl_grid/arity_6/long_only` | `17.728` | `31.298` | `-76.541%` | `33.831` | `47.539` | `-40.516%` | `15.940` | `16.101` | `-1.011%` | `0` | `13751.296` |
| `tp_sl_grid/arity_6/long_short_reversal` | `15.474` | `15.502` | `-0.180%` | `30.726` | `30.773` | `-0.153%` | `15.084` | `15.109` | `-0.163%` | `0` | `0.000` |

Supporting evidence:

- preflight: `../2026-06-14_matrix_bitset_stage_05_12_production_default_stage15_preflight/`;
- control: `../2026-06-14_matrix_bitset_stage_15_tp_sl_early_abandon_control/`;
- candidate: this directory.

## Contract Impact

Public API contract: `none`.

Port contract: `none` in the active tree.

DTO schema: `none` in the active tree.

Persisted schema: `none`.

Config schema: `none` in the active tree.

Request hash / cache identity / persistence identity: `none`.

Benchmark/report semantics: `compatible-change` through evidence-only stage
records.

Performance risk on verified hot path: candidate rejected; active runtime path
unchanged.
