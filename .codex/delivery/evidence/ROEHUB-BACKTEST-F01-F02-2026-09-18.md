# Backtest F01/F02 correction evidence

## Scope and baseline

Standalone corrective task authorized by the attached `goal-objective.md` on
2026-09-18. Publication and safe local synchronization are authorized; runtime
installation/deployment, F03/F04 and performance work are excluded.

Base: `origin/main` / `191a9f8169dab0639d8fb3456eb73b060eb1d2c4`.
Technical branch: `codex/backtest-short-ranking`, isolated worktree
`/Users/daniildegtyarev/Projects/roehub-f01-f02`.
Original checkout remains `/Users/daniildegtyarev/Projects/roehub.com`.
Original index was empty; original main had no unpublished commits. Existing
API/web/strategy/UI/docs changes and untracked strategies-client artifacts were
not transferred. A pre-synchronization hash inventory covers 846 foreign files.

Read-only audit sources: `report.md`, `audit.json`,
`evidence/counterexamples-v2.json`, `evidence/S06.log`, `evidence/S07.log`,
`work/counterexamples.py`, `work/benchmark.py` under
`~/.local/share/openresearch/files/backtest-engine/backtest-local-audit-20260917T202102Z/`.
All 47 audit-listed backtest v2 source/test hashes match the original checkout;
the audit still applies. Its files were not modified or executed in place.

## Confirmed causes and corrections

F01: preflight accepts futures `short`, but registry and direction-code conversion
rejected it. Kernels and trade-list reconstruction also only distinguished
long-only from reversal. Added a distinct short code (3), negative-consensus
entries, neutral/positive exits, and no long entries in event, streaming,
matrix/prefix and reference/detail paths. Existing financial kernels, sizing,
funding windows and spot-short rejection are preserved.

F02: TP/SL normalized ranking was only consumed by funding reranking; local
selection, heap admission and final order always used descending total return.
Thread the ranking criterion through both truncations, retain the actual score
separately from total return, and compute other ranking metrics for every
candidate's selected cell before truncation in a compiled chunk pass. The canonical max-gross-return
cell contract is unchanged, including when row order is ascending. Funding
retains its existing bounded gross-return pool and posthoc net reranking.

Owned production paths under `src/trading/contexts/backtest/application/services/v2/`:
`combo_planning.py`, `execution_sizing.py`, `job_orchestration.py`,
`no_risk_exact.py`, `tp_sl_exact.py`,
`matrix_backend/no_risk_score.py`, `matrix_backend/prefix_traversal.py`.
Also changed: the new `test_short_and_tp_sl_ranking.py`, its explicit CI shard
in `tools/ci/route_changes.py`, canonical runtime documentation, generated
project map and this evidence record.

## Regression evidence

Command prefix: `PYTHONPATH=src:. NUMBA_NUM_THREADS=1 .venv/bin/python -m pytest`.
The isolated worktree reused the existing Python environment; no dependencies
were installed. Explicit PYTHONPATH makes imports resolve to the candidate.

Initial six-test run on unmodified source: **4 failed, 2 passed**. Failures:
short registry rejection in both risk modes; ascending top-1 and top-2 return
ranking. After correction the same six passed. Independent audit arithmetic:
100 -> 120 gives +20%; 120 -> 90 gives -25%; ascending now selects -25%,
descending +20%. Absolute tolerance 1e-4 percentage points matches the audit's
float32 policy (observed positive result 20.000000298023224).

Final new regression file after PR review: **170 passed**. Coverage:

- preflight -> real planning, warmup, scoring and assembly for both short modes;
  only artifact I/O is replaced by deterministic fixture ports;
- five no-risk backend/arity combinations; neutral/positive exits, no long
  entries, repeated shorts and final close;
- all five sizing modes, fees, nonzero slippage and close-on-end true/false,
  with independently calculated P&L and actual lazy detail builders;
- TP, SL and same-bar SL precedence in supported event/matrix TP/SL paths;
- positive/negative funding and entry-exclusive/exit-inclusive windows;
- all six non-funding ranking metrics, both directions, top-N 1/7/12,
  complete membership/order across block sizes 3/5/4096, stable ties,
  finite independently calculated multitrade Sharpe and infinite ratios;
- maximum-return cell preservation under ascending and alternate metrics;
- nonfinite ratios normalized to null in the serialized source result payload.

Expanded checks:

| Command (same worktree and environment) | Outcome |
| --- | --- |
| `pytest -q tests/unit/contexts/backtest tests/unit/contexts/backtest_artifacts` | 513 passed before the final additional serialization test; affected code unchanged afterward |
| `pytest -q tests/unit/tools/test_ci_route_changes.py tests/unit/contexts/backtest/application/services/v2/test_short_and_tp_sl_ranking.py` | 495 passed before the final additional serialization test |
| `pytest -q tests/unit/contexts/backtest/application/services/v2/test_short_and_tp_sl_ranking.py` | 106 passed, final regression suite |
| `.venv/bin/ruff check .` | passed |
| `PYTHONPATH=src:. .venv/bin/pyright` | 0 errors, 0 warnings |
| `.codex/hooks/tests/run_tests.py` | passed |
| `tools.docs.generate_docs_index --check` | passed |
| `tools.docs.generate_project_map --check` | passed after regeneration without local environment symlink |
| `tools.docs.generate_runbooks --check` | passed |
| `tools.artifacts.generate_schemas --check` | passed |
| `tools.backup.generate_schemas --check` | passed |
| `tools/release/generate_runtime_topology.py --check` | passed |
| `tools/release/runtime_input_inventory.py --check` | passed |
| `tools.jobs.generate_schemas --check` | passed |
| `tools/release/oss_metadata.py --check` | passed |

Intermediate test-authoring errors were corrected: a hit-table fixture used the
entry rather than entry+1 column; an assertion used payload metrics rather than
summary_metrics_json; unsupported fixed_quote+matrix TP/SL combinations were
removed from positive parametrization. Initial lint/type failures were fixed.
One router test command named a nonexistent file (no tests ran); the correct
`test_ci_route_changes.py` then passed. These were not engine baseline failures.

## Compatibility and persistence

| Boundary | Classification | Evidence / transition |
| --- | --- | --- |
| Supported short execution / TP/SL row ranking | `compatible-change` | Restores accepted request behavior; does not add a public enum or change the cell policy |
| Request DTO, normalization and request hash | `none` | No changes to preflight or hashing; spot short remains rejected |
| Variant/canonical parameter identity | `none` | Assembly and identity schema unchanged; corrected shortlist membership can differ |
| Persistence schema, cache key, configuration defaults | `none` | No schema/config/key changes; new jobs get UUIDs and lazy cache keys include job_id |
| Funding/financial model | `none` | Existing trade accounting and bounded-pool approximation retained |
| Telemetry | `compatible-change` | Additive `tp_sl_ranking_metrics` timing for pre-selection alternate-metric work |

Old jobs remain historical and are not rewritten. `backtest_jobs.py` returns the
same old job for an existing Idempotency-Key inside its 86400-second TTL. Rerun
without that key or with a new key to create a new job; no cross-job result-cache
reuse occurs. No deletion, schema migration or incompatible invalidation is
needed. Mixed readers still consume the same schema. No database deployment,
cache mutation or runtime rollout/rollback was tested.

## Review, limits and publication boundary

Cold self-review and independent release review completed. Independent reviewer
found no product-code defect; its test parametrization and finite-Sharpe
requests were addressed. Its generated-map `.venv` finding was fixed by removing
the task-owned environment symlink before final map generation. Final verdict:
**approve**, with the existing model/runtime limitations below.

Known retained model limitation: no-risk uses fill slippage and fees on actual
notionals; TP/SL uses raw prices and `(1-fee_rate)^2`, ignoring slippage. This is
existing behavior, documented rather than changed by this correction.
Alternate-metric TP/SL ranking now needs selected-cell statistics for every
candidate in a compiled pass with constant per-candidate state; default return ranking retains the compact path. No latency or
performance claim, real-market-data proof, HTTP/queue/DB proof or runtime
installation proof is made.

Publication is verified separately on the PR and remote-main revision. The
intended terminal state is `shipped-no-runtime`; this local evidence alone does
not establish publication or deployment.


## PR review follow-up

PR #34, initial revision `84f87d8cf2ec98dd9ad503051c580a92402d4e88`:
GitHub review identified missing short routing through the standard arity 6/7
matrix/prefix gates, and the Python reconstruction cost of alternate-metric
preselection. Both were addressed: routing guards now include short, and the
alternate-metric pass uses Numba with shared input matrices and constant state
per candidate. The requested metric is still computed for every candidate
before truncation; the search space and cell policy are unchanged.

Added actual preflight-to-orchestration arity 6/7 tests and 60 comparisons of
compiled scores against the slow final-detail metric path across all alternate
metrics, directions, close-on-end settings, nonzero time slices, sizing and
profit locking. Final new suite: 170 passed. Focused Pyright: 0 errors;
full Ruff and project-map check passed. No benchmark or speed claim is made.

Initial PR CI passed all backend/static/migration shards. Platform Web failed
builder color contrast (4.25 vs 4.5) and library viewport-change transition
rejection. Those UI paths are unchanged by this task. A comparative baseline
workflow was dispatched at the original main SHA: run `35279370276`.
These initial failures are retained as evidence; final publication depends on
checks for the follow-up revision and is reported separately.


Final local follow-up gate:
`PYTHONPATH=src:. NUMBA_NUM_THREADS=1 /Users/daniildegtyarev/Projects/roehub.com/.venv/bin/python -m pytest -q tests/unit/contexts/backtest tests/unit/contexts/backtest_artifacts tests/unit/tools/test_ci_route_changes.py`
returned **968 passed**. Independent follow-up review: **approve**; reviewer ran
`-k 'compiled_cell_metrics or preflight_to_actual_short'`: 66 passed,
104 deselected. The deselection is the deliberate focused scope, not full-suite
coverage. All 170 new regressions are covered by the 968-test final local gate.
