# S04 independent production-risk review

Reviewer: `s04_production_review`, 2026-10-08. Procedure: `production-risk-review`.
This is the final update of the same review, not a recursive review.

## Findings and disposition

No unresolved material implementation findings within the S04 proof boundary.
Review covered the owned product paths in `S04-owned-paths.json` against exact
`S04-before-*.txt` snapshots, new unit/real-infrastructure tests and the NPY runtime
document, with relevant S01/S02 contracts and native consumer/ownership code.
Foreign working-tree changes were not treated as S04 changes. Only this report
was written by the reviewer; no product, ledger or accepted artifact was edited.

Initial review identified proof and contract gaps. Final disposition:

- Sparse native row IDs now pass actual filesystem context/array loading and
  `prepare_artifact_inputs` reuse, beyond direct manifest inspection. Corrupt
  advertised signal bytes fail through the actual reader and slot validator.
- Added cases prove all-excluded real risk publication, independent TP/SL,
  source-independent `volume.obv` scheduled/on-demand parity, frozen futures
  funding/risk parity, unavailable/versioned catalog behavior and exact row
  union without invented Cartesian combinations.
- Actual PostgreSQL reader pins protect a prior generation during shared union
  publication. A following publication needing the pinned slot rejects with
  `artifact_source_pinned`; old bytes and current pointer remain intact.
  Mode-off leaves both unchanged.
- Managed publication now compares frozen root ID/revision and resolved physical
  target with trusted publisher binding, and rejects unsupported compute/funding
  identities before publication. Negative tests preserve the pointer.
- The current Russian NPY runtime document records `grid.row_ids`, empty risk
  axes, source-independent rows, reader-before-writer rollout and coordinator
  integration boundaries.

## Production-risk assessment

Frozen catalog JSON detaches nested mutable mappings; admission selections and
compute/signal defaults do not reread YAML. User deselection and capability
availability remain separate. Explicit editing/rebase retains exclusions and
adds new available IDs, while invalid/removed selected entries reject. Existing
admitted snapshots are not rewritten. Native signal parameters remain
`default-only`, consistent with the existing numerical engine.

Compatible demand union uses exact canonical row sets, common frozen full source
history and independently sorted TP/SL levels. It does not manufacture parameter
cross-products or truncate selections to satisfy the row budget. History remains
anchored to the source origin for warmup/global risk indexing. Existing 0028
physical precheck/private staging/publication/reservations remain authoritative;
S02 queue ownership is not a substitute for a slot reservation.

Optional `grid.row_ids` is sorted, unique and nonnegative, with physical-row-count
validation. Missing IDs retain old canonical-prefix semantics. Native reuse
still validates variant-key hashes and frozen defaults. `ArtifactRequestedRow`
accepts `source=None` only structurally; native request validation checks the
actual canonical grid/rule source. Existing explicit-source mappings serialize
unchanged. This supports source-independent indicators without inventing a
`close` axis or replacing the numerical engine.

The empty-risk reader repair checks each advertised empty axis SHA/dtype/shape
and each empty long/short table SHA/dtype/shape before omitting nonexistent level
coverage from reusable references. It therefore avoids manufacturing a zero-row
recipe reference without turning omission into a corruption fallback. Non-risk
array dimensions remain subject to their original positive-shape contracts.

## Verification

Independently executed final command:

`.venv/bin/python -m pytest tests/unit/contexts/backtest_artifacts/test_derivative_profiles.py -q`

Result: **14 passed in 21.10s**. Includes real native calculations, independent
scalar SMA oracle, sparse prepare/reuse, risk/funding and negative identity cases.
An earlier bounded planner subset also passed (3 tests); the full final run
supersedes that limited evidence.

Inspected executor-generated completed evidence (not independently re-executed
by this reviewer):

- `S04-routed-checks.json`: **473 passed**, exit 0; runtime config, manifests,
  schema-compatible loaders, derived materialization, recipe contracts/replay,
  prepare/preflight, CLI/scheduler wrappers and artifact tests.
- `S04-real-infra-checks.json`: **66 passed**, exit 0; disposable loopback
  PostgreSQL physical lifecycle, management and shared selective publication.
- `S04-ruff-checks.json`: **passed**, exit 0.
- `S04-pyright-checks.json`: **0 errors, 0 warnings**, exit 0.

The real-infrastructure test source was reviewed: it reserves a real database
reader pin, publishes the union to the other slot, verifies original bytes,
then proves pinned-slot refusal. No skipped mandatory infrastructure check is
represented as a pass in the inspected final result.

## Compatibility and remaining boundary

- Current schema-aware consumers: `compatible-change`; new optional row IDs,
  empty one-sided risk inventories and null source for source-independent rows.
  Backward reads and existing explicit-source hashes remain preserved.
- Older strict readers consuming these new formats: `breaking-change` in that
  forward direction. Readers must precede writers; rollback to incompatible
  readers after selective publication remains prohibited.
- New managed profile serialization/catalog: `compatible-change`; full catalog
  snapshot and explicit rebase are new managed contracts. No old queued recipe
  is rewritten. Pre-snapshot imported profiles require explicit later-stage
  coordinator handling, not silently invented catalog evidence.
- Physical ownership primitives: `none`; S04 integrates with unchanged 0028
  ownership. Shared-demand composition and current runtime activation remain
  later-stage concerns.

No full-catalog compute, performance/capacity claim, actual user-runtime
activation, external provider proof or production delivery was attempted or
established. Fixed fixtures prove selected native outputs and integrity; they do
not prove every indicator numerically across every history. S05 must construct
authorized complete demand groups, bind trusted root/engine/source identity and
respect queue budgets/cadence; S04 does not itself dispatch durable jobs.

Review is complete within these limits. Ledger acceptance remains the executor's
responsibility using actual stage evidence and the canonical updater.

### Final schema-1 integrity refinement

Re-read the subsequently added root-schema fence in
`artifact_manifest_validator.py` and `artifact_array_loader.py`. Sparse explicit
row IDs and empty risk families reject under root schema-1; the extensions belong
to schema-2. This preserves old schema-1 positivity/implicit-row guarantees in
both publication validation and the runtime loader. No new finding.

Independently executed after that change:

`.venv/bin/python -m pytest tests/unit/contexts/backtest_artifacts/test_derivative_profiles.py -k partial_derivatives_require_schema2 -q`

Result: **1 passed, 14 deselected in 2.93s**. The 473-test executor broad evidence
above predates this final refinement; the executor's subsequent focused
validator/prepare/new-test run must be recorded separately rather than implying
the earlier broad suite ran against the refinement. Physical ownership behavior
was unchanged by this validation-only follow-up.
