# F03/F04 correction evidence

## Scope and ownership

Task authority: user goal objective `ccfec022-70c3-4360-93d6-7b065a7a1686`.
Base: `ef0142de9e35b1869f7ff7bdd83e811a3552cfdf` (F01/F02, PR #34).
Remote: `https://github.com/Dejetins/roehub.com`; actual primary branch: `main`.
Initial local `main`, `origin/main` and upstream matched; no unpublished commits
and no staged changes. The original checkout had unrelated tracked/untracked
Strategy/Web work and `.codex/AGENTS.md` edits. None was transferred.

Implementation and checks use the isolated worktree
`/Users/daniildegtyarev/Projects/roehub-f03-f04`, branch
`codex/backtest-accounting-parity`, from verified `origin/main`. Only F03/F04
source, benchmark producers/consumers, regressions, CI test routing, canonical
backtest documentation and their generated project-map changes are owned.
Existing dependencies are reused with an absolute Python path and `PYTHONPATH=src`;
no dependencies or credentials were changed. No service environment file was
loaded, and no API/DB benchmark harness was run against a live installation.

## Causes and corrections

F03: `_stage_timings` added overlapping preparation parent/child timers and
TP/SL aliases. A controlled prepare=3, exact=5, assembly=1, warmup=3, elapsed=12
case returned 17 instead of 9. It now subtracts the measured in-interval warmup
from the actual monotonic pre-cleanup wall interval. Nested diagnostics remain
non-additive. Missing persistence/assembly evidence is not synthesized as zero.
Invalid intervals are rejected. A v2 diagnostic marker declares the boundary;
benchmark validation requires this marker and consistent intervals from the same
latest attempt (no borrowing missing fields across retries).

F04: the harness compared only five metric rows, ignored missing metric keys and
identity, and allowed quality/shadow bypasses. Full API rows now require a complete
trusted reference bound to request/engine/runtime/artifact identities. Exact
ranks/order, semantic parameters/hash, all metric keys and selected cells are
compared. Normal production samples stay at five; an opt-in benchmark parent
writer exports at most 50 compact child rows for separate transport consistency.
Current `/top` is unpaginated. Unexpected pagination is not assessed. Incomplete
or incompatible references stay `not_assessed`; mismatches are `failed`; both
prevent acceptance. Diagnostic-only exit zero never changes the acceptance flag.

Full reference fixtures are independently prescribed expectations. The tests do
not generate an oracle from the candidate. They prove accounting/comparison and
producer/report behavior, not independent financial correctness of every engine
calculation. Existing F01/F02 and funding regressions remain part of verification.

Definitions, normalization policy, reference format and compatibility details:
`docs/architecture/backtest/README.md`, section “Benchmark accounting and full
 top-N evidence (2026-09-18)”. Existing historical records were not rewritten.
The notebook validator's historical schema is explicitly distinct from v2 API
runner evidence, and a regression rejects the latter at the notebook validator.

## Before/after and local validation

All commands below run in the isolated worktree; `PY` denotes the unchanged
`/Users/daniildegtyarev/Projects/roehub.com/.venv/bin/python`.

Before implementation:

`PYTHONPATH=src $PY -m pytest -q tests/unit/scripts/backtest/test_accounting_parity_regressions.py`

Three failures observed: `17.0 != 9.0`, missing metric not detected, equal metrics
with different indicator identity not detected. The same checks pass after the fix.

Controlled orchestration regression uses a mutable fake monotonic clock, retaining
all three GC calls. Pre-assembly diagnostics/GC and warmup GC are included; final
diagnostics cost and final GC are excluded. IPC serialization -> child evidence
writer -> benchmark timing reader preserves the corrected total and definition.

Full-result tests cover 1/7/50 rows, shorter and independently empty results, ties,
ascending/descending reference order, both risk modes, nonfinite/null policy,
fixed numerical tolerance, rank-50 metric corruption, identity, rank, swapped tied
rows, missing/duplicate/extra rows, TP/SL cell corruption without changed return,
missing metric, truncated/legacy/incompatible oracle, incomplete pagination,
quality/shadow bypasses, bounded opt-in evidence and JSON/Markdown/gate/exit status.
A connected assembly/read-model/IPC/evidence/report test includes actual legacy
`source_top_result` samples as well as all 50 rows.

Initial broad test run: 629 passed. Final scoped suite including CI routing:
**1033 passed in 24.63s**. Ruff and pyright: passed, 0 errors. Commands:

- `PYTHONPATH=src $PY -m pytest -q tests/unit/contexts/backtest tests/unit/apps/worker/backtest_job_runner tests/unit/scripts/backtest tests/unit/tools/test_ci_route_changes.py`
- `/Users/daniildegtyarev/Projects/roehub.com/.venv/bin/ruff check .`
- `/Users/daniildegtyarev/Projects/roehub.com/.venv/bin/pyright --venvpath /Users/daniildegtyarev/Projects/roehub.com`
- `PYTHONPATH=src $PY -m tools.docs.generate_docs_index --check`
- `PYTHONPATH=src $PY -m tools.docs.generate_project_map --check`
- `PYTHONPATH=src $PY -m tools.docs.generate_runbooks --check`
- `$PY .codex/hooks/tests/run_tests.py`
- `PYTHONPATH=src $PY -m tools.artifacts.generate_schemas --check`
- `PYTHONPATH=src $PY -m tools.backup.generate_schemas --check`
- `$PY tools/release/generate_runtime_topology.py --check`
- `$PY tools/release/runtime_input_inventory.py --check`
- `PYTHONPATH=src $PY -m tools.jobs.generate_schemas --check`
- `$PY tools/release/oss_metadata.py --check`

Ruff, pyright (0 errors), hooks and all listed drift/schema checks passed before
publication preparation. Project map was regenerated for the new files. CI routing
explicitly includes the new benchmark regression directory in `backtest-scoring`.

## Review and compatibility

Review mode: cold self-review plus one independent read-only release review using
`production-risk-review`. The review found an API/sample identity-shape mismatch
and an aggregate/partial-status contradiction. Both were repaired with regressions;
independent follow-up verdict: passed, no material blockers, 53 focused tests passed.

Public API, persistence schema, ranking, funding and execution defaults: `none`.
Bounded opt-in child evidence: `compatible-change`. Corrected benchmark evidence
and acceptance semantics: intentional in-scope `breaking-change`; v1 sample-only
or invalid timing evidence cannot satisfy v2 acceptance. No funding behavior was
changed and no performance improvement is claimed.

Live HTTP/worker/DB behavior and realistic market-data performance: **not run**.
Prerequisite for that separate proof: explicitly identified disposable installation
and compatible full trusted reference. This does not block deterministic F03/F04
correction. No deployment is authorized or performed. Publication target is
`shipped-no-runtime`; actual PR/revision/CI and safe local fast-forward evidence
belong to the final delivery report, not to an unobserved claim in this record.


## GitHub review follow-up

Automated review of `39f24da7` found that a candidate and oracle could lose the
same required metric. Two regression cases reproduced `passed` / `failed` where
an incomplete oracle required `not_assessed`. Full references now declare risk,
funding inclusion and metric manifest; the manifest is checked against current
metric-name contracts and child profile, and every reference row must be complete.
This does not regenerate any financial baseline: synthetic transport fixtures
independently prescribe values for the complete known metric shape.

First PR Web run `35285477803` failed library.spec.ts:112 on
`Transition was aborted because of invalid state. Viewport size changed`.
No frontend files were changed by this task. No unrelated repair was made;
the new revision triggers its own CI and the original failed run is retained.

Follow-up verification: 67 tests passed with
`PYTHONPATH=src $PY -m pytest -q tests/unit/scripts/backtest tests/unit/contexts/backtest/application/services/v2/test_benchmark_accounting.py`.
Whole-repository Ruff and Pyright (0 errors) passed again; project-map drift check
passed. Existing backend computation was not changed by this follow-up.
