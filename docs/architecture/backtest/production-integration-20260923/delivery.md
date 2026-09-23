# Backtest production integration — 2026-09-23

Status: local implementation verified; published in PR #37, GitHub checks and final coordinator review in progress.

## Authority and delivery unit

User-authorized integration and publication of accepted S1–S6 work into main,
followed by safe synchronization of the main checkout and deletion of the technical
branch. This is a new delivery specification, not a resumed stage or prompt pack.
Source worktree 224b and its completed ledger remain read-only.

Classification: implementation and authorized publication. Primary workflow:
`publish-ci-deploy`; companions: `backend-quality-gates`, `contract-impact-analysis`,
`production-risk-review`. Evidence boundary: local product dispatch, raw numerical
parity, real child/IPC/database integration and GitHub CI; terminal target
`shipped-no-runtime` because no current production deployment target is selected.

## Owned scope and provenance

`ownership.json` records each of 26 imported product/test/config paths, its source
SHA-256 and accepted S1–S5 manifest. Every hash was checked before copying. Source
baseline: `b4f2ddad9b508680fc6d4ad7dbf51b401b8f50b7`. Research scripts, captures,
archives, source ledgers and virtual environments are excluded. Test-only raw-bit
encoding and score observation helpers are relocated under tests; production
imports no research modules.

## Production decisions

Remove the four diagnostic algorithm flags from the immutable job policy. Normal
worker composition uses cost permutation/inverse restoration, exact prefix-entry
local top-K, integer TP/SL tape and exact prefix lower-bound proof automatically
when their eligibility conditions hold. Keep cost permutation minimum 32 rows,
integer tape minimum 32 bars and each component's 64 MiB ndarray scratch cap.
Keep default 12 threads, job-scoped scratch, separate warmup, one heavy lane and
no nested/concurrent Numba workqueue execution.

Fallbacks remain reachable for insufficient work, unsupported inputs, scheduling,
funding/timeframe, insufficient proof and bounded allocation failures. They retain
required financial behavior; removing them would remove supported interactions.
No alternative unused engine is imported. Original financial kernels, arithmetic,
precision, ranking/identity and lazy detail semantics remain unchanged.

## Acceptance and proof plan

Verify normal platform builder/child execution of the optimized blocks. Run focused
edge/fallback and financial regressions, strict raw IEEE parity separately against
immutable original references for compile, compile-warm, cache-load and load-warm,
and real local child/IPC/DB checks for changed lifecycle boundaries. Reuse existing
dependencies and Postgres image. Keep new evidence compact; no historical multi-hour
performance campaign or full captures. Any new timing is diagnostic and bounded,
not a replacement for accepted historical performance results.

Historical owner acceptance prioritized heavy compute after disclosed 14.8–24.1%
heavy warm improvements and 12.3% lower heavy-only queue throughput. Original failed
small/load/queue gates remain failed. Diagnostic minimum 1 results do not establish
performance of production minimum 32, nor a universal small-task slowdown bound.

Before merge: scoped diff, local gates, PR checks and coordinator independent review.
After merge: current main CI, clean-checkout fast-forward synchronization, local and
remote technical branch deletion. No SSH, secrets or historical runtime host use.

## Compatibility assessment (implementation pending validation)

Resource/config selection changes are intended `compatible-change`: algorithms are
now selected normally; existing thread precedence/default12 and supported inputs
remain. Internal diagnostic flags are removed, with no API/environment exposure.
Public API/DTO, storage schema, result identity, financial semantics and funding
are unchanged within the verified cases and source boundaries below.
Telemetry keeps existing fields and reports proof-based traversal bypass truthfully
with no fabricated compiled-loop timing. Historical acceptance is not production
verification.

## Observed implementation and local proof

The initial import manifest was validated against accepted source hashes before
copying. Removed: four algorithm booleans and their disabled dispatch branches;
unused `sequential_numba_thread_mask` and its self-only test; the research-controller
cleanup test and unused research helper functions. Original numerical kernels remain:
all 31 pre-existing decorated functions in the two exact scorers have identical ASTs
(`financial-kernel-audit.json`). `numba_runtime.py` is unchanged from main.

The existing orchestration selector, not registry defaults alone, determines normal
routing: default `stage_05_and_12_no_risk` selects matrix no-risk for arity 6 and
compiled-prefix no-risk for arity 7. No selector or financial kernel was changed.
The first branch observation used arity 6 and honestly failed four-branch coverage;
`dispatch-initial-incomplete.json` preserves it. Adding the missing arity-7 case
closed coverage without adding unnecessary proof work to event-segment paths.

`dispatch-result.json` observes normal builder calls (no policy injection): cost
permutation 6, successful local admission 4, integer tape 51, proved canonical prefix
enumeration 2; insufficient-work permutation fallback 6. Arity-7 lower bound 170
exceeds minimum 13, with 128 candidates, no traversal and absent compiled-loop timer.
Observers wrap and restore test-process calls; product code imports no observers.

`parity-result.json` and `parity-prefix-result.json`: four fixed cases, **16 exact
comparisons**, separately first compilation, compile-origin warm, new-process cache
load and load-origin warm. Zero mismatches in all evaluated raw metrics, canonical
rows, full returned top results and complete lazy details, including raw IEEE words.
Original closure is `b663a1c1c1ebcfbfc0a44e48be44f8d9dfa5b41009505f92d7435c40e51310c1`.
No financial tolerance or nonfinite normalization is used. The extra prefix case is
128 combinations, 2976 bars, top 5; it adds coverage, not a timing comparison.
`proof-bindings.json` and `parity-import-bindings.json` bind inputs and observed import
hashes to final product files. Complete maps were reconstructed by read-only imports
and exactly matched every saved run digest. The only measured-to-final difference
is removal of the unused thread-mask helper and then its trailing blank line.
Script post-processing/formatting is disclosed; current script hashes are reproduction
bindings, not fabricated pre-execution snapshots. No source224b writes occurred.

Real runner/child/IPC/PostgreSQL proof: `runtime-result.json` and
`runtime-prefix-result.json` execute the standard child module through the actual
scheduler and repositories. All 3 jobs succeeded; **105 persisted rows and 20,538
compared values**, zero differences. Database transaction timestamps are checked
against job completion separately from financial values. The standard arity-7 child
reports `not_run_bound_proved_no_pruning`. The existing PostgreSQL17.5 image was used,
no image pull/install. Both owned containers, scratch and IPC directories were removed.
Failure/lease/cancel/timeout/observation cleanup is also exercised by lifecycle tests;
this does not claim arbitrary production-fault incidence or an actual deployment.

### Required retained paths

| Path | Production reachability and proof |
| --- | --- |
| Unpermuted native scoring | Small rows, single thread, unsupported scheduler/layout, missing scratch, budget/allocation failures; `test_cost_permutation.py` and observed small normal job. Kernels are shared by optimized and fallback calls. |
| Full Python global heap admission | NaN keys/global heap, small n/K, unsupported integer bounds/layout, budget/allocation failures; `test_local_top_k.py`. Local admission feeds the same global heap and preserves its mutation sequence. |
| Original TP/SL interval builder | Funding, non-15m, small histories, unsupported integer layout, budget/allocation failures; `test_integer_trade_tape.py`. Independent interval/cell oracles retained. |
| Compiled prefix traversal | Lower bound insufficient/unknown or scratch allocation/budget failure; `test_prefix_guard.py`. Proof is sufficient, not an approximate pruning rule. |
| Other existing financial backends | Existing selector/requested backend support and funding/detail paths; not replaced financial implementations. No new parallel old-engine copy is shipped. |

### Gate commands and outcomes

Cwd is worktree2705. Existing interpreter
`/Users/daniildegtyarev/Projects/roehub.com/.venv/bin/python` is used with `-I -B` and
explicit `sys.path[:0]=['.','src']`; actual imports select this checkout. No venv copy
or dependency install. The commands reproduce current project CI targets:

- `pytest -q -ra` with `TEST_SHARDS['backtest-scoring'].target`: **1213 passed**,
  13.20s, NUMBA/ROEHUB thread budget12 and workqueue.
- `pytest -q -ra` with `TEST_SHARDS['backtest-use-cases'].target`: **155 passed**,
  15.74s, thread budget1. Includes actual disposable lifecycle/IPC cases.
- `ruff check .`: passed.
- Generated docs/project map/runbooks, artifact/backup/job schemas, runtime topology,
  160-input inventory, OSS metadata and all active hook regressions: passed.
- `pyright --pythonpath /Users/daniildegtyarev/Projects/roehub.com/.venv/bin/python`:
  zero errors/warnings. The missing local `.venv` notice is expected because the
  existing external interpreter is explicitly selected.
- Initial broad verification:1270 passed/2 fixture failures; corrected telemetry
  expectation and excluded research-controller-only test; focused recheck15 passed.
  Full final shard outcomes above supersede fixture failures, without hiding them.

CI explicitly includes the four new component test files. Only `backtest-scoring`
uses12 threads and workqueue because native static ownership is checked at1/2/3/12.
This is correctness on the available runner, not a speed test or hardware capacity
claim; all other shards retain their prior thread count and threading-layer defaults.

## Final compatibility and self-review

Public API/DTO, persisted schemas, request/result identity, financial arithmetic,
precision and funding: `none` within inspected source and exact compared cases.
Job resource configuration/dispatch and additive diagnostic fields:
`compatible-change`; diagnostic flags had no public API/environment consumers.
Existing queued payloads and stored rows require no migration. Rollback retains the
same data representations. Single-lane ownership and default12 are preserved.
Production/host-wide scheduling, universal performance and deployment remain unproved.

Cold self-review (`production-risk-review`): no unresolved material implementation
findings after removing unused helper, restoring meaningful CI routing and closing
normal arity-7 proof coverage. Independent coordinator review and GitHub CI are
required before merge. New compute evidence is correctness-only and bounded; historical
performance acceptance and failed gates are not reclassified.

## Publication tracking

PR: https://github.com/Dejetins/roehub.com/pull/37. Initial product commit
`a241bcde67745833ec94691ec2b2d9666ffe8930`. The existing project-map bot added
`0b335e54f703cc2d33f266254dd003750462084c` to include the final manifest itself
in inventory. That generated-only commit was inspected and fast-forwarded; its
`[skip ci]` message does not count as green verification. A follow-up evidence commit
refreshes map hashes and triggers checks on the actual reviewed head.

### GitHub review follow-up: child threading layer

GitHub review identified that proof envelopes selected workqueue explicitly while
production child envelopes left Numba to choose the platform default. This could
leave cost permutation inactive on Linux omp/tbb. The child environment helper now
defaults to workqueue before Numba import, preserving explicit operator selections
and their exact scheduler fallback. Parent state is unchanged. Fresh-child tests
remove inherited layer values and assert the actual initialized layer; the real
standard-child/DB proof is repeated without a proof-script layer preset. Numerical
kernels and the already verified workqueue arithmetic remain unchanged.


Follow-up gates:123 focused lifecycle/orchestration/permutation tests passed (64.33s).
`runtime-layer-result.json` records an additional real standard child with no input
threading-layer preset: actual workqueue, one enabled permutation and one small
fallback, prefix bypass, 5 DB rows/1036 values exact and owned cleanup complete.
The additive diagnostic map contains only actual layer and integer call counts;
it retains no scratch arrays. Empty/whitespace layer values select the default;
explicit omp/tbb are preserved and disable permutation through existing fallback.
The earlier105-row runtime proof remains valid; this is a targeted extra5-row check.
GitHub's previous ad0d84f5 checks were green, but publication waits for the new fix SHA.

`layer-parity-result.json`: two additional current-source raw comparisons (compile
and compile-origin warm) match the corresponding immutable-original digests exactly.
No new original runs or full campaign; `layer-fix-bindings.json` records all observed
imports and the equivalent post-runtime getattr/style adjustment. Ruff/Pyright pass.
