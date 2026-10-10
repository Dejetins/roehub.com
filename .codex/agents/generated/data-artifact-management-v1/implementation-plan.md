# Data artifact management — implementation plan v1

Date: 2026-10-08. Status: **approved** by the owner on 2026-10-08. Product requirements, technical design and the sequential implementation plan are approved. The owner will submit S01–S09 in order; each selected prompt authorizes its in-scope implementation and verification without another unchanged-plan approval. Execution has not started; approval alone does not claim or execute a stage.

## Execution artifacts and authority

- Plan: [implementation-plan.md](implementation-plan.md).
- Prompt pack: this directory; exact stage files are linked below.
- Sole mutable execution journal: [iteration-ledger.md](iteration-ledger.md).
- Mode: `manual_sequential`. Submitting a prompt authorizes one selected stage, its necessary in-scope local implementation and verification, and its report/receipt transition. Stop after returning the next prompt link unless the owner explicitly authorizes continuation. No Goal, branch, worktree, stash or publication is implied.
- S01 is a contract-and-baseline stage; it does not start expensive calculations. Entry readiness and execution authorization are separate. The journal, not this paragraph, owns current stage states.
- The existing pack-local updater from `backtest-npy-input-materialization-v1` and its disposable tests are reused byte-for-byte in this new directory. Their old docstring date records origin only; it grants no authority over the prior pack. No new execution engine is designed. The completed source pack remains untouched. The directory-local implementation resolves only this new journal.
- There is no source-required extra design approval between stages. New material owner decisions use the canonical runner decision packet; routine implementation choices remain delegated.

## Owner approval

On 2026-10-08 the owner explicitly stated: «Окей, план утвержден, подставь там, что это не драфт и я буду запускать тогда.» This approves the complete plan and sequential prompt pack, including the common base/derivative 1/3/7-day cadence and all available indicators selected by default. No known product decision remains pending. Exact schemas, limits and implementation details assigned to S01 remain delegated technical work, not a new owner approval gate.

Plan approval is distinct from execution lifecycle: the journal's `ledger_status=draft` means **approved, execution not started** in this pack. The canonical runner changes it to `active` when S01 is claimed. Do not rename that schema value, claim S01 merely to record approval, or mark implementation accepted before its evidence exists.

## Accepted requirements and traceability

The owner's latest response approved points 1–4, amended point 5, approved the remainder, and requested this triad. “Свичи/свитчи” is understood as candles and “NPI” as NPY. The previous explicit alternative in point 7 is resolved by acceptance of the proposed common 1/3/7-day schedule for both base and derivative artifacts. This is recorded interpretation, not a separate owner quotation.

| ID | Accepted product requirement | Stages | Observable evidence |
|---|---|---|---|
| R01 | Data has Candles / Artifacts modes; shared exchange, market, search and selected instrument survive switching. Preserve the current three-pane Navigator and candle/streaming behavior. | S07,S09 | Browser switching, back/forward, reload and existing candle controls |
| R02 | Persisted candles automatically cause initial base NPY preparation, with no per-pair enable switch, across every supported platform timeframe. Candle ingestion success and NPY readiness are separate. | S01,S03,S05,S09 | Ingestion commit -> durable request -> all-timeframe manifests; restart recovery |
| R03 | Base NPY covers all actually loaded history, including later historical extensions. Closed-bar boundaries, gaps and insufficient history are truthful. | S03,S09 | Known-source ranges across all timeframes, gaps, short history and extension fixtures |
| R04 | Derivatives use on-demand (new policy default) or precompute per instrument, independently of streaming. Base NPY remains mandatory. | S02,S04,S09 | Mode changes, source availability and Backtest preparation fixtures |
| R05 | All available indicators from the Backtests catalog are checked by default. The user opens that catalog and unchecks unwanted indicators. Reuse catalog identities, availability, sources and bounded parameter defaults; no second indicator registry. | S01,S04,S07,S09 | Catalog identity parity; all-selected initial profile; exclusions persisted and applied |
| R06 | A derivative profile includes history range, derivative timeframes, indicator parameter/source selections and TP/SL levels; base NPY timeframe set is not editable. | S02,S04,S07 | Strict profile validation and matching manifest inventory |
| R07 | Automatic refresh every 1, 3 or 7 days, never subdaily; common default and per-instrument overrides. Show timezone, next run and last successful run. | S02,S05,S07 | Clock-controlled persistence, overrides and API/browser values |
| R08 | Initial base preparation and enabling precompute enqueue immediately; later changes are coalesced to the selected schedule. Update now does not move the scheduled anchor. | S03,S05,S09 | Immediate first jobs, no duplicate jobs and unchanged next scheduled occurrence |
| R09 | Separate real progress for base NPY and derivatives; use the existing candle progress styling. Waiting for source, queued, running, ready and error states are visible; unknown totals have no fabricated percentage. | S03–S07,S09 | Measured units, stage/timeframe projection and browser progress |
| R10 | Artifact table shows instrument/market, base readiness, derivative mode/progress, coverage date and refresh policy. Inspector shows range, timeframe detail, size, profile, last/next runs, errors and job history. | S06,S07,S09 | Real API-backed table, inspector and bounded history |
| R11 | Single/bulk Update now, interval, precompute mode and profile actions. Disabling precompute keeps existing files and mandatory base updates. Cleanup is a separate guarded action. | S02,S06–S09 | Idempotent commands, bulk result semantics, mode-off retention |
| R12 | Settings -> Storage -> Backtest artifacts uses one server-side root, automatic per-instrument layout, access/free/used-space, budget and cleanup rules. Reuse installation configuration. Existing-data relocation is a separate explicit operation. | S01,S06,S08,S09 | Installation-owner boundary, path validation, live checks and relocation-required state |
| R13 | Last good publication survives failures. Bounded durable queue, restart recovery, no duplicate builds, no removal/overwrite of pinned inputs; Backtests exposes coverage and extra preparation needs. | S02–S06,S08,S09 | Real PostgreSQL/filesystem concurrency, restart and reader/publisher/cleanup drills |

## Verified current sources and limitations

Paths in this section are repository-root-relative; links resolve from this plan.

- [Repository policy](../../../../AGENTS.md): `apps/platform-web` remains the accepted client. The explicitly selected design target is the existing `apps/navigator-web` candidate at `http://localhost:20120/data`; this pack does not silently replace or migrate the baseline client.
- [Data UI](../../../../apps/navigator-web/src/data-page.tsx), [workspace schemas](../../../../apps/navigator-web/src/data-workspace-api.ts), [workspace API](../../../../apps/api/routes/market_data_workspace.py): current exchange catalog, history, stream state and durable ingestion controls. Several are already modified in the worktree by other work; inspect fresh diffs before each stage and preserve them.
- [Builder API/catalog](../../../../apps/navigator-web/src/builder-api.ts), [UI Backtests routes](../../../../apps/api/routes/ui_backtests.py): runtime-supported timeframes/indicator IDs/defaults and catalog availability are already exposed. S01 reconciles the precise sets; a UI hardcoded list is not authoritative.
- [NPY runtime contract](../../../../docs/architecture/backtest/backtest-service-artifact-runtime-v1.ru.md): base prices/mappings/required funding remain necessary; signals and hit-times may be generated on demand. Existing recipe identities, schema-1/2 readers and physical slot ownership are preserved. Job preparation does not read ClickHouse or publish the current pointer.
- [Publisher](../../../../src/trading/contexts/backtest_artifacts/application/use_cases/publish_backtest_artifacts_v2.py), [config](../../../../src/trading/contexts/backtest_artifacts/adapters/outbound/config/backtest_artifacts_runtime_config.py), [scheduler wiring](../../../../apps/scheduler/backtest_artifact_publisher/wiring/modules/backtest_artifact_publisher.py): native builder, validation/publish and coordinate derivative policy exist. Scheduler currently selects the effective collector universe, not a separate artifact policy. Wiring passes only `enabled` into the fixed daily scheduler; YAML full-rebuild hour/minute must not be portrayed as an already working configurable user schedule.
- [Dev config](../../../../configs/dev/backtest_artifacts.yaml): schedule is disabled; all-supported signals and resource budgets exist. The new feature must not simply enable this schedule for the entire catalog.
- [ArtifactStore contract](../../../../docs/architecture/platform/artifact-store-v1.md): general local CAS/S3 adapters exist but do not automatically migrate path-shaped backtest consumers. S3 integration and replacement of NPY mmap storage are out of scope.
- [Admin API](../../../../apps/api/routes/admin.py), [control-agent](../../../../apps/control_agent/server.py): existing authorization/control boundary to inspect for installation-wide storage settings; organization admin does not automatically have installation-owner authority.
- [Data loading contract](../../../../docs/architecture/apps/web/roehub-data-loading-contract-v1.md): coherent snapshots, subject isolation, bounded visible reads/cache, command eligibility and no refresh flicker apply to both Data and Storage.
- [Current local recovery runbook](../../../../docs/runbooks/navigator-local-market-data-recovery.md): keep current local candle/streaming processes and data intact. [Old rebuild runbook](../../../../docs/runbooks/backtest-artifacts-rebuild.md) is retired and supplies no runtime launch authority.

These are source observations, not proof that the proposed feature exists or is deployed. The reference docs/index and several API/UI/market-data files already have foreign changes. Authoring only creates this isolated pack.

## Approved target decisions

### D1 — one Data surface and one catalog

Use a route query such as `view=artifacts`, defaulting to candles for existing links. Preserve existing `market`, `exchange`, symbol selection and applicable common filters; mode-specific filters have distinct names. The UI calls authenticated application services through DTOs; it does not own schedules, artifact state or filesystem paths. Reuse catalog-backed choices from Backtests. New profiles start with every available supported indicator selected; unavailable entries are visible/disabled with reasons. Persist explicit selected IDs and catalog/profile revisions. A later catalog addition is marked as new and selected by default when editing the profile, but is not silently added to an already admitted job. An empty selected indicator set is valid (no signal precompute); requested TP/SL artifacts remain independent. Exclusions affect proactive work, not whether a later on-demand Backtest may request an indicator.

Alternative: add columns to the existing candle view or create a separate top-level Artifacts section. Both are superseded by the owner's selected mode-switch design. Price/mapping timeframe coverage is derived from the platform's authoritative supported set plus mandatory execution inputs; zero closed bars at a long timeframe is explicit insufficient history, not a fabricated completed dataset or a permanent retry storm.

### D2 — ownership and storage of policy

`backtest_artifacts` owns profiles, build requests, publication status and scheduling rules; `market_data` remains owner of committed candle availability. Ingestion supplies a narrow post-commit notification/port with periodic reconciliation for missed notifications; it never imports artifact compute code into candle hot paths. Existing databases/worker conventions remain a modular application, without a new broker or service fleet.

Persist organization-scoped desired profiles, per-instrument cadence/override versions and durable build requests in PostgreSQL; installation-level defaults/root/budgets require the existing installation-owner boundary. The physical slot mutex remains `(exchange, market_type, symbol, slot)`, not organization or generation. Source coordinates are resolved server-side against authorized catalog access.

Because physical publications are shared, S01 must specify deterministic effective demand: union compatible requested derivative inventories/ranges for the same coordinate and source snapshot, bounded by installation budgets; incompatible compute/funding/schema identities stay separate jobs/are rejected, never merged silently. One organization disabling precompute cannot delete another's useful artifacts. Jobs, policy identities, actor data and private quotas are never exposed across organizations. Each organization's cadence is respected for its requested work; a fresher shared base artifact may be reused without attributing another organization's request. Persist each request's desired watermark/profile identity even when coalescing physical work. Admission must report resource conflicts rather than silently shrink all-indicator selections.

New managed profiles default to on-demand. Do not silently reinterpret legacy full-precompute configurations or existing queued jobs; import/preserve legacy policies explicitly and report migration coverage. Discover only actual stored candle coordinates/accepted ingestion work in bounded pages, never turn the full exchange reference catalog into build or download work.

### D3 — base preparation is a durable consequence of stored candles

Create initial work after a committed finite source range is observable; first preparation does not wait indefinitely for an ongoing stream. Freeze a closed-candle watermark per attempt. Generate canonical prices, all supported rolled prices, mappings and required funding through the existing builder/validator. Full loaded-history coverage can require bounded chunks and checkpoints; an incomplete build cannot publish as fully ready. Represent source gaps, short history and funding unavailability explicitly. Missing history is not downloaded as an implicit artifact side effect. Historical extension and corrections invalidate affected ranges; reuse must honor indicator warmup and existing prefix-proof rules.

Automatic base policy cannot be disabled per pair. A global operational pause/unavailable worker shows delayed work and does not pretend ready. Last good coverage and the next target range are separate. Failure or insufficient disk keeps the old pointer, records a stable reason and leaves incomplete staging owned by the failed attempt. Reconcile before retrying unknown publication outcomes.

### D4 — immutable bounded profiles, reuse and safe publication

A build snapshots effective catalog, parameters, risk levels, source identity, compute versions and root/profile revision. All indicators means the catalog's supported indicators with finite runtime parameter defaults, not an unbounded parameter search or combinations across indicators. Estimate resource demand before admission and expose unknown estimates truthfully. Use existing native precompute logic for both scheduled derivatives and job-scoped materialization; no duplicate numerical implementation. Preserve `precompute/on_demand` semantics and schema-1 compatibility; selected subsets require explicit inventory/validation semantics so omitted signals are allowed and corrupt declared files still fail.

Use private staging, validate actual output, then atomic pointer publication under existing ownership. A base-only refresh may obsolete derivatives; never associate old signals with a changed source prefix without proof. Keep pinned old generations until readers release them. Exclusions and mode changes affect future requests only. Existing job recipes and numerical meaning stay unchanged. Do not promise immutable retention of every historical generation when the current slot protocol does not provide it.

### D5 — cadence and durable progress

Support exact UI presets `1`, `3`, `7` calendar days, minimum one day; reject unsupported/subdaily values. Approved default: one day, installation-selected timezone and local run time; persist an explicit schedule anchor and calculate UTC due instants (including DST tests). A manual run and initial preparation do not move that anchor. A 3/7-day interval intentionally permits base NPY to lag candles; display both watermarks. Coalesce new candles until due; initial/new-coordinate work runs immediately. Enabling precompute is immediate once a usable base exists; otherwise it waits on the identified base prerequisite.

Missed schedules collapse into one newest required target, not one job per missed day. Use transactional idempotency, control versions and physical coordinate reservations. Stable identities include desired source/profile and schedule occurrence; automatic and manual submissions coalesce compatible work without losing tenant-visible request outcomes. Schedule edits version future requests and do not mutate active task snapshots. Use bounded queue/CPU/memory/IO budgets and an explicit fairness policy; S01 resolves concrete limits from current configuration and disposable evidence rather than inventing capacity guarantees. No nested heavy compute pools.

Persist measured progress with phase, timeframe, completed/total units, progress epoch, observation time and last checkpoint. Count work from a frozen plan; mark totals unknown until planned, and never invent elapsed-time percentages. Retries may start a new epoch visibly. Distinguish waiting_for_source, queued, running, retry_wait, ready and error, and operational pauses. Publication success is after validation/pointer commit, not after file generation alone. UI overall base progress aggregates declared measured units while timeframe details retain their own readiness. Separate derivative progress uses its own work plan. Last successful inventory remains visible during refresh/failure.

### D6 — API, storage and security

Proposed API families (names finalized in S01/S06, not claims of existing routes): artifact inventory/detail, profile/defaults, build submit/bulk/history/progress, schedule and installation storage inspect/configure/cleanup. Use strict DTOs and bounded pagination; do not serialize raw repositories or manifests. Reads authorize every coordinate/job; commands enforce same-origin/CSRF, appropriate organization permissions, idempotency and optimistic version checks. Reuse existing bulk limits where justified and return explicit per-item outcomes; never partially apply a supposedly atomic batch.

Storage root is on the server and comes from a trusted installation registry/configuration. Ordinary users see a safe location label and readiness, not privileged paths or host capacity outside their authority. Installation-owner reads may show the actual path and total space. Validating a candidate root uses narrow contained probe files, checks permissions/free space on the worker mount, rejects symlink/path traversal and returns sanitized diagnostics. Do not let a browser-provided absolute path become an arbitrary file-write capability. Route privileged host operations through the existing control boundary; do not grant organization admin installation rights or mint credentials.

First-time empty-root configuration can be committed after validation. A root change with existing artifacts, pins or active jobs returns an explicit `relocation_required` outcome and keeps the old root active. Actual data relocation is a separately authorized workflow, not silently implemented as a path edit. Budget and cleanup policies are editable with privilege checks. Cleanup has separate preview/confirmation, honors the existing ownership/reservation protocol, protects current and pinned inputs, and records sanitized audit data; no TTL-based unsafe reclaim. Automatic mandatory base work reports quota/storage blocking rather than discarding history to fit.

### D7 — UX and Backtests integration

Preserve the current Data geometry, compact controls and candle progress style. Separate table rows for base state and derivative state are columns within the same instrument row, not a second duplicate instrument catalog. Inspector shows timeframe completion, known range/gaps, profile/all-indicators count, observed size, next run and bounded job log. Checkbox catalog supports search, select all, deselect individual items, explicit draft save/cancel and keyboard focus. Bulk edits show mixed values and do not overwrite untouched fields. Do not silently turn stream-on into precompute-on.

The status can read “Ready through 7 Oct; latest update failed.” Retained snapshots must not enable commands for stale identity/revision. Follow existing subject-scoped query/cache rules and local status feedback. Backtests shows actual source date bounds, selected effective coverage, `ready` versus additional materialization and missing inputs, using existing preflight contracts. It must not claim runnable for a requested date beyond the base snapshot. No numerical scoring/ranking changes are requested.

## Compatibility and recoverable rollout

| Boundary | Intended classification | Conditions / recovery |
|---|---|---|
| New API DTOs/routes, Data mode and nullable policy metadata | compatible-change | Existing URLs/candle routes continue; strict old clients unaffected |
| Policy defaults and scheduler selection | breaking-change if legacy values silently change; target compatible-change | Explicit legacy import; keep old scheduler disabled for coordinates owned by new queue; no two writers |
| Manifest/subset publication and Backtest input recipes | compatible-change for current schema-aware readers; unknown for uninspected old readers | Inventory all consumers in S01; deploy readers before writers; preserve hashes and schema-1 reads |
| Physical slots, resource cleanup, privileged root settings | security/concurrency-sensitive compatible-change target | Retain ownership protocol; negative authorization/path tests and independent review |
| New PostgreSQL schema | additive compatible-change target | New versioned migrations, no edits to applied migrations or destructive automatic downgrade |
| Market data candle contents and financial kernels | none | No rewrite of candle schema, trading behavior or formulas |
| Runtime installation/S3/provider behavior | unknown until selected proof | Local proof is not production or S3 readiness |

Roll out persistence and readers before new writers. Keep the new queue behind explicit installation capability/config until local proof; activation must disable the overlapping legacy scheduler using one recorded ownership handoff. Rollback disables new submissions, drains/halts owned work safely and preserves last-good artifacts/schema. Do not roll back to an incompatible reader after new-format publication. No irreversible data deletion/move is required for feature activation. Cleanup proof uses disposable data only.

## Stages and dependencies

| Stage | Outcome | Prompt |
|---|---|---|
| S01 | Executable contracts, source inventory and baseline evidence | [01-contracts-and-baseline.md](01-contracts-and-baseline.md) |
| S02 | Durable policies, requests, ownership and additive persistence | [02-policies-and-durable-jobs.md](02-policies-and-durable-jobs.md) |
| S03 | Automatic all-timeframe base NPY lifecycle | [03-automatic-base-npy.md](03-automatic-base-npy.md) |
| S04 | All-indicators default, exclusions and selective precompute | [04-indicator-profiles.md](04-indicator-profiles.md) |
| S05 | Durable 1/3/7-day scheduling, reconciliation and progress | [05-scheduling-and-progress.md](05-scheduling-and-progress.md) |
| S06 | Authorized API and privileged storage operations | [06-api-and-storage-boundary.md](06-api-and-storage-boundary.md) |
| S07 | Data Candles / Artifacts UI and real progress | [07-data-artifacts-ui.md](07-data-artifacts-ui.md) |
| S08 | Storage settings and Backtests readiness UI | [08-storage-and-backtest-readiness.md](08-storage-and-backtest-readiness.md) |
| S09 | Integrated real-infrastructure and browser proof, operational handoff | [09-integrated-proof.md](09-integrated-proof.md) |

Stages form one sequential chain. Each report is a declared product for the next prompt; absent future reports are not owner input gaps. S01 resolves endpoint/schema names and reversible defaults inside this accepted scope in its report/canonical contracts. Do not edit an accepted plan to insert execution discoveries and invalidate earlier receipts. Material scope changes require a revision under the runner contract.

## Verification and acceptance

Match tests to changed boundaries. Select actual commands from package scripts, `pyproject.toml`, `.github/workflows/ci.yml` and `tools/ci/route_changes.py`; focused tests first, required routed checks afterwards. Never use the production/preview database as a disposable fixture. Existing real-infra entrypoint: `tests/real_infra/backtest/test_artifact_input_lifecycle.py`; inspect its setup before execution. Authoring checks only validate this triad and the reused updater, not product tests.

Minimum proof matrix: R02/R03 all supported timeframe equivalence and source coverage; R05 catalog identity/exclusion/default compatibility; R07/R08 restart/DST/anchor/dedupe; R09 measured progress and failure status; R11 mixed bulk commands/idempotency; R12 cross-tenant and installation-owner authorization plus malicious paths/storage failure; R13 publication crash windows, reader/writer/cleanup contention, old generation pins and immutable job recipes. Compare scheduled versus on-demand numerical outputs on independent fixed fixtures, not identical wrappers around one implementation. Use resource measurements only with the `backend-performance-evidence` procedure; no speed or capacity claims from estimates.

Browser proof uses the explicitly selected Navigator design, preserving actual current Data candle flows and streaming controls. Run mutating scenarios on disposable organization/data/runtime; never rebuild all existing instruments or reseed the user's current preview. Read-only inspection of localhost:20120 is allowed. Later stage authority for local disposable proof does not authorize real exchanges, production credentials, deployments or cleanup of existing artifacts.

Independent review is required once for this design's security/ownership boundaries under AGENTS.md, and at implemented security/migration/concurrency boundaries as described in relevant stage prompts. Reviewers do not recursively request review. Each finding needs disposition; an unavailable required review remains an explicit evidence gap. Runtime missing infrastructure is unavailable proof, not pass or a reason to fabricate receipts.

## Documentation and handoff

When implementation changes the durable contract, update the existing NPY runtime document (preserve normative Russian), Market Data API contract, Navigator README and data-loading contract only where necessary. Installation/root operations belong in the current offline installation runbook; never revive the retired rebuild runbook. Generate docs/project-map indexes only if new canonical docs/components require it; preserve concurrent foreign changes and inspect generated diffs. Stage reports record owned paths/hunks, foreign changes excluded, checks/outcomes, compatibility, proof boundary, residual risks and next input.

Open technical work is assigned, not hidden owner blockers: S01 verifies the supported timeframe/indicator sets, shared-coordinate demand merge, supported consumer versions, installation permission mapping, rollout activation and concrete resource defaults. S03 resolves insufficient-history behavior against existing NPY readers. S06 proves privilege/root checks using the actual installation control seam. S09 proves the selected runtime locally. If a required capability genuinely cannot be implemented within the accepted boundaries, record the narrow gap; do not silently drop a requirement or substitute a mock as proof.
