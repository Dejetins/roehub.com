# Backtest Job Runner v2 -- canonical claimed background worker for persisted runs

This document defines the canonical architecture for the `backtest-job-runner` service in the
artifact-backed backtest v2 runtime.

## Status

- Status: canonical architecture document for the active `backtest-job-runner` runtime.
- This file is the source of truth for the claimed background execution path for persisted runs.
- [`docs/architecture/backtest/backtest-job-runner-worker-v1.md`](/Users/daniildegtyarev/Projects/roehub.com/docs/architecture/backtest/backtest-job-runner-worker-v1.md)
  is historical / compatibility-only and is kept for migration context.
- Public lifecycle vocabulary and persisted run storage remain compatible with runs/history
  contracts, while runtime internals are artifact-backed v2.

## Purpose

`backtest-job-runner` is a dedicated long-lived worker service that claims queued persisted runs
from Postgres and executes them through the shared backtest v2 runtime.

The worker exists so that:

- heavy but valid runs can be launched as `background_auto`;
- those runs remain visible in Backtest History as persisted runs;
- API and UI surfaces are not blocked by long-running execution;
- sync and background execution share the same runtime orchestration surface and the same
  canonical exact scorer.

## Scope

This document covers:

- the worker's role in the backtest v2 architecture;
- startup and fail-fast wiring;
- claim, lease, heartbeat, cancel, and reclaim semantics;
- slot-pinned artifact bootstrap for claimed runs;
- shared runtime planning and `ExecutionProfile` usage;
- persisted progress and summary-only results;
- observability;
- production deployment expectations;
- compatibility boundaries.

This document does not cover:

- the artifact rebuild/publish pipeline;
- sync-inline response assembly;
- lazy variant detail or trades recomputation;
- browser UI rendering beyond the persisted progress contract;
- the detailed adaptive-selector rollout matrix.

## Core architectural decisions

### 1. `backtest-job-runner` is not a separate backtest engine

`backtest-job-runner` is a background execution host for the same canonical v2 runtime used by
the sync path.

The worker must not:

- create a second orchestration surface;
- introduce a separate scoring engine;
- own a separate runtime-profile selection policy;
- redefine progress or rollout semantics.

### 2. There is one claimed background path

The canonical background launch mode is `background_auto`.

`background_manual_legacy` remains only a compatibility-only literal:

- new runs must not be created with `background_manual_legacy`;
- the worker must continue to execute already persisted rows that carry this literal;
- active product and architecture documentation must describe the background path through
  `background_auto`.

### 3. One worker process equals one claim loop and one claimed job at a time

One `backtest-job-runner` instance:

- is a single long-lived process;
- owns one sequential claim loop;
- processes at most one claimed job at any given time.

Queue scalability comes from multiple independent worker processes, not from making one claim
loop concurrent.

### 4. Queue concurrency and runtime parallelism are different concerns

Parallel processing of multiple independent background runs must be configured separately from the
internal compute parallelism of one run.

The canonical queue-concurrency knob is:

- `backtest.jobs.worker_processes`

This setting defines how many independent worker processes may claim background jobs at the same
time.

Important boundaries:

- `worker_processes` controls queue concurrency;
- single-run runtime parallelism remains the responsibility of the shared v2 runtime and the
  resolved `ExecutionProfile`;
- queue concurrency and intra-run compute parallelism must not be collapsed into one parameter.

### 5. The architecture is service-manager-agnostic

This document defines the worker service contract, not a supervisor-specific deployment shape.

The architectural contract requires the worker to be:

- a long-lived supervised service;
- restarted automatically after crashes;
- scalable to `N = worker_processes` instances;
- assigned a unique instance identity;
- observable through logs and metrics.

The concrete implementation may use `launchd`, `systemd`, containers, or another supervisor, but
that choice belongs in runbook and ops documentation rather than this architecture document.

## Role in the system

Production has one canonical persisted-run storage family and one runtime orchestration surface:

- the API creates a persisted run record;
- sync-compatible runs may finish inline;
- heavy but valid runs are classified into `background_auto`;
- `backtest-job-runner` claims queued persisted runs and executes them;
- runs history and detail APIs read from the same persisted storage;
- progress, ETA, and top snapshots are exposed through one persisted run contract.

The worker is part of the production backtest contract. It is not an optional helper process.

## Startup and fail-fast wiring

At startup the worker must:

- load runtime config from `backtest.yaml`;
- load artifact runtime config;
- require `STRATEGY_PG_DSN`;
- construct Postgres repositories for jobs, leases, and results;
- construct the canonical request decoder for persisted `request_json`;
- construct defaults providers and compatibility-only estimate helpers;
- construct artifact loading and `ArtifactSlotResolverV2`;
- construct `BacktestArtifactRuntimePlannerV2` from startup-loaded `execution_profiles` and
  `adaptive_selector_policy`;
- start the metrics endpoint;
- verify that the worker can enter the claim loop without lazily repairing critical dependencies.

Startup is fail-fast. Missing runtime config, invalid artifact config, an empty DSN, inability to
construct runtime dependencies, or invalid worker cardinality must stop the process before the
claim loop begins.

## Configuration contract

The minimum operational contract for the worker includes:

- `backtest.jobs.enabled`
- `backtest.jobs.worker_processes`
- `backtest.jobs.claim_poll_seconds`
- `backtest.jobs.lease_seconds`
- `backtest.jobs.heartbeat_seconds`
- `backtest.jobs.snapshot_seconds`
- `backtest.jobs.snapshot_variants_step`

Normative rules:

- if `backtest.jobs.enabled=false`, the worker may exit with disabled status and exit code `0`;
- if `backtest.jobs.enabled=true`, `worker_processes` must be `>= 1`;
- queue concurrency is defined only through `worker_processes`;
- the service manager must materialize exactly `worker_processes` independent worker instances;
- every instance must have a unique `locked_by` value and unique runtime identity.

If an environment temporarily materializes multiple workers through an external mechanism without
the explicit config knob, that may be treated as transitional, but the canonical contract remains
`backtest.jobs.worker_processes`.

## Claim, lease, and reclaim

The worker loop behaves as follows:

1. Poll via `claim_next(now, locked_by, lease_seconds)`.
2. If no job is available, sleep for `claim_poll_seconds`.
3. If a job is claimed, run one deterministic attempt.
4. During processing, extend the lease through heartbeat writes.
5. If the lease is lost, stop the current attempt immediately.

Required invariants:

- claim is atomic and uses row-lock semantics;
- only one worker may own a claimed job at a time;
- startup must pass an explicit `instance_index` per worker process, defaulting to `0` for a
  single local instance;
- `locked_by` is a stable worker identity in
  `hostname=<hostname>;pid=<pid>;instance_index=<instance_index>` format;
- queued jobs may be cancelled immediately at the storage layer;
- running jobs are cancelled only at batch boundaries;
- reclaim may restart the attempt from the beginning;
- a job must not remain in `running` forever after worker death;
- a worker that loses its lease must stop writing progress, snapshots, or terminal state.

## Source of truth for claimed runs

Claimed worker execution reads only persisted run payloads:

- `job.request_json` is the request source of truth;
- saved-mode snapshot payloads are used only when effective-template semantics must be restored;
- the worker must not reread live saved-strategy state for an already created job;
- the worker must not make runtime decisions based on current browser or UI state.

This keeps claimed runs reproducible even if strategy storage changes after launch.

## Slot-pinned artifact contract

Every claimed background run must already carry pinned artifact identity in the persisted job row:

- `artifact_slot`;
- `artifact_slot_generation`;
- `artifact_asof_date`;
- `artifact_manifest_hash`.

Before runtime work begins, the worker must resolve slot-pinned context through
`resolve_pinned_context(...)`.

Required rules:

- missing pin metadata is a deterministic failure;
- missing pinned artifacts are a deterministic failure;
- drifted pinned artifacts are a deterministic failure;
- fallback to live `current.yaml` discovery for claimed runs is forbidden;
- fallback to a legacy runtime after pinned-artifact bootstrap failure is forbidden.

The worker executes only slot-pinned runs. It is not part of the artifact publish or rebuild
decision path and must not rebuild artifacts itself.

## Shared runtime planning and `ExecutionProfile`

The worker does not own a separate planning policy. It delegates runtime planning to the shared
v2 planner stack.

For every claimed run the worker:

- validates the effective request and template contract;
- applies supported request timeframe rules;
- applies default-only signal-override rules;
- resolves the runtime plan through `BacktestArtifactRuntimePlannerV2`;
- uses startup-loaded `execution_profiles` and `adaptive_selector_policy`;
- executes the selected profile through shared artifact-backed runtime services.

This means:

- `background_auto` is a launch classification, not a separate scoring engine;
- `exact_small`, `exact_parallel`, `hybrid_conservative`, and `hybrid_family` are runtime-profile
  decisions inside the same claimed path;
- browser and public API surfaces still do not choose `execution_profile_mode` directly;
- the worker does not make rollout decisions on its own and instead obeys the shared adaptive
  selector policy.

## Rollout-policy boundary

This document intentionally does not duplicate the detailed adaptive-selector rollout matrix.

It defines only the responsibility boundary:

- the worker uses shared `execution_profiles`;
- the worker uses shared `adaptive_selector_policy`;
- the worker does not own rollout phases;
- the worker does not define `shadow`, `opt_in`, or `active` states by itself;
- the worker executes a runtime plan already resolved by the shared planner and policy layer.

Detailed rollout rules, phase literals, benchmark gates, and promotion criteria belong in
selector, runtime, and config documents rather than the worker architecture document.

## Stage execution model

Claimed execution uses one shared stage model.

### Stage A

The worker builds a deterministic shortlist of base variants through artifact-backed Stage A
services and persists Stage A shortlist metadata for observability and diagnostics.

### Stage B

The worker expands retained candidates across risk dimensions and scores them through the shared
artifact-backed Stage B runtime. The current best rows are persisted as summary-only snapshots on
the configured snapshot cadence.

### Finalizing

Finalizing writes terminal summary rows and performs the terminal state transition.

Persisted background rows remain summary-only:

- the worker does not materialize full report bodies as part of the persisted run contract;
- the worker does not materialize trades payloads as part of the persisted run contract;
- detail and trades remain separate from claimed background execution.

## Persistence and progress/ETA contract

The canonical persisted-run storage family remains:

- `backtest_jobs`
- `backtest_job_top_variants`
- `backtest_job_stage_a_shortlist`

The worker is responsible for persisting:

- lifecycle state and timestamps;
- the current stage name;
- `processed_units` and `total_units`;
- lease and heartbeat fields;
- failure and cancel payloads when present;
- summary-only top rows and Stage A shortlist snapshots.

The worker is not responsible for the final browser ETA.

The worker must persist deterministic progress counters that later power the read model:

- `stage`;
- `processed_units`;
- `total_units`;
- timestamps and heartbeat data.

User-facing `progress_percent` and `eta_seconds` are computed in the runs-history layer from:

- worker counters;
- execution-profile semantics;
- throughput estimates;
- benchmark-backed fallback values when real throughput is not yet defensible.

Responsibility is intentionally split:

- the worker writes execution facts;
- the read model calculates user-facing progress and ETA.

## Observability

The worker publishes process-level metrics and structured logs.

Minimum metrics surface:

- total claimed jobs;
- succeeded jobs;
- failed jobs;
- cancelled jobs;
- lease-lost events;
- job duration;
- stage duration;
- gauge of active claimed jobs.

Operational requirements for metrics:

- each worker instance must expose an observable metrics surface;
- the deployment target must provide either a unique metrics endpoint binding or an equivalent
  aggregation model;
- absence of a visible metrics surface for a live worker is an operational defect.

Structured logs must make it possible to answer:

- whether the worker started enabled or disabled;
- which job was claimed by which worker identity;
- which stage a run is in;
- whether the worker lost its lease;
- whether a failure came from pin drift, runtime error, or cancellation;
- how many worker instances are actually active in production.

## Production deployment contract

If `backtest.jobs.enabled=true`, production must provide a continuously running supervised worker
service.

This means:

- deployment must install or update the worker services;
- deployment must restart the worker services;
- deployment must verify that the required number of instances started;
- deployment must verify that those instances are alive and able to remain in the claim loop.

If `backtest.jobs.enabled=true` but the persistent worker service is not installed or not
running, the deploy is incorrect even if the API process starts successfully.

Recommended post-deploy smoke for the worker must be service-level rather than request-level:

- verify worker instance registration in the supervisor;
- verify that the processes are running;
- verify that each instance publishes logs and metrics;
- verify that the worker does not exit immediately with disabled or error status;
- verify that the number of live worker processes matches `worker_processes`.

Creating a test production job as part of every deploy smoke is not required and is not the
canonical validation method.

## Scalability

The worker architecture must remain ready for future scale:

- on one host by increasing the number of supervised worker processes;
- across multiple hosts through the same storage-based claim and lease model;
- by changing the service manager without rewriting the worker contract;
- without introducing a second queue coordinator outside the Postgres storage contract.

The scaling boundary remains simple:

- the queue scales by the number of worker processes;
- correctness is enforced through storage-level claim and lease rules;
- single-run runtime semantics are defined by the shared planner and shared kernels.

## Compatibility

`background_manual_legacy` remains supported only as a compatibility input for already persisted
rows.

Normative compatibility rules:

- the worker must be able to claim and execute such rows;
- new product flows must not be documented through `background_manual_legacy`;
- new launch contracts must not emit that literal;
- the canonical worker architecture is described only through `background_auto`.

## Non-goals

This document does not introduce:

- a new public launch API surface;
- a browser-controlled `execution_profile_mode`;
- a separate policy layer inside the worker;
- a separate engine for background runs;
- a conflation of benchmark-evidence anchors with active runtime defaults;
- a platform-specific dependency on `launchd` or any other supervisor inside the architecture
  contract.

## Related documents

- [Final Backtest Refactor Plan v2](/Users/daniildegtyarev/Projects/roehub.com/docs/architecture/roadmap/backtest-refactor-final-plan-v2.md)
- [Backtest Runtime Acceleration Plan v1](/Users/daniildegtyarev/Projects/roehub.com/docs/architecture/roadmap/backtest-runtime-acceleration-plan-v1.md)
- [Backtest Job Runner Worker v1](/Users/daniildegtyarev/Projects/roehub.com/docs/architecture/backtest/backtest-job-runner-worker-v1.md)
- [Backtest Runs History API v2](/Users/daniildegtyarev/Projects/roehub.com/docs/architecture/backtest/backtest-runs-history-v2.md)
- [Backtest API v1: `POST /backtests`](/Users/daniildegtyarev/Projects/roehub.com/docs/architecture/backtest/backtest-api-post-backtests-v1.md)
- [Backtest Job Runner Runbook](/Users/daniildegtyarev/Projects/roehub.com/docs/runbooks/backtest-job-runner.md)
- [run_backtest_job_runner_v1.py](/Users/daniildegtyarev/Projects/roehub.com/src/trading/contexts/backtest/application/use_cases/run_backtest_job_runner_v1.py)
- [backtest_job_runner.py](/Users/daniildegtyarev/Projects/roehub.com/apps/worker/backtest_job_runner/wiring/modules/backtest_job_runner.py)
