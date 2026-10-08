---
{
  "schema_version": "stage-prompt/v1",
  "prompt_pack_execution": {
    "plan_doc": "implementation-plan.md",
    "prompt_pack_dir": ".",
    "stage_ledger": "iteration-ledger.md"
  },
  "stage_contract": {
    "id": "S04",
    "title": "Full-job preparation, resource admission, pins and attempt cleanup",
    "prompt_path": "04-worker-lifecycle-and-resources.md",
    "report_path": "reports/S04.md",
    "receipt_dir": "receipts/S04",
    "depends_on": [
      "S03"
    ],
    "expected_touches": [
      "../../../../apps/worker/backtest_job_runner/",
      "../../../../src/trading/contexts/backtest/application/services/v2/job_orchestration.py",
      "../../../../src/trading/contexts/backtest/application/services/v2/job_scratch.py",
      "../../../../src/trading/contexts/backtest/application/services/v2/admission.py",
      "../../../../src/trading/contexts/backtest/application/use_cases/backtest_job_worker.py",
      "../../../../src/trading/contexts/backtest/application/ports/backtest_job_repositories.py",
      "../../../../src/trading/contexts/backtest/adapters/outbound/persistence/postgres",
      "../../../../src/trading/contexts/backtest_artifacts/application/services/v2/artifact_slot_publisher.py",
      "../../../../src/trading/contexts/backtest_artifacts/adapters/outbound/persistence/postgres/backtest_job_repository.py",
      "../../../../tests/unit/apps/worker/backtest_job_runner",
      "../../../../tests/unit/apps/worker/test_backtest_job_runner.py",
      "../../../../tests/unit/contexts/backtest/application/use_cases/test_backtest_job_worker_use_case.py",
      "zone: Disposable Postgres pin and lease integration proof",
      "../../../../src/trading/contexts/backtest_artifacts/adapters/outbound/persistence/postgres/gateway.py"
    ],
    "acceptance_criteria": [
      "Exactly one materialization per attempt serves warmup and scoring; no DB candle read occurs.",
      "Success/failure/cancel/timeout/restart recovery leave no owned orphan files, leaked reservations or live-input deletion.",
      "Real concurrent publisher/full/lazy attempts cannot overwrite pinned source inputs; stale owners cannot reclaim live work.",
      "IPC retains recipe and funding fields; timings distinguish preparation, service, parent cleanup and persistence.",
      "Reader admission and both publisher rebuild/deletion use the shared ownership transaction; uncertain expired owners cannot release live files, and prepared proof is durably acknowledged before scoring."
    ],
    "proof_boundary": "Worker subprocess/resource lifecycle and tested local Postgres concurrency; no production installation claim.",
    "validation": {
      "profile": "roehub-focused-gates-and-prompt-pack-artifacts/v1",
      "checks": [
        "uv run pytest -q tests/unit/apps/worker/backtest_job_runner tests/unit/apps/worker/test_backtest_job_runner.py tests/unit/contexts/backtest/application/use_cases/test_backtest_job_worker_use_case.py",
        "Add tests/unit/apps/worker/backtest_job_runner/test_materialization_lifecycle.py and tests/real_infra/backtest/test_artifact_input_lifecycle.py (planned paths). Inspect current real-infra fixtures first; use disposable Postgres resources.",
        "Execute C07/C08/C14 failure/concurrency drills and report live lease recovery independently from mocked process tests. Run focused ruff and uv run pyright."
      ],
      "requires_user_acceptance": false
    },
    "entry_inputs": [
      {
        "path": "implementation-plan.md",
        "producer_stage": null
      },
      {
        "path": "../../../../apps/worker/backtest_job_runner/wiring/modules/child_process.py",
        "producer_stage": null
      },
      {
        "path": "../../../../apps/worker/backtest_job_runner/wiring/modules/full_job_compute.py",
        "producer_stage": null
      },
      {
        "path": "../../../../src/trading/contexts/backtest/application/services/v2/job_orchestration.py",
        "producer_stage": null
      },
      {
        "path": "reports/S03.md",
        "producer_stage": "S03"
      }
    ]
  }
}
---

# S04 — Full-job preparation, resource admission, pins and attempt cleanup

Relevant implementation-plan sections: 4.3–4.4, 4.5 IPC notes, 6 (C07/C08/C14), 7 timing boundaries.

## Execution authority and state

Read the live ledger before doing anything. It alone owns statuses, claims and decision packets. Execute exactly S04 when the owner submits this prompt, then stop with the next prompt link. The owner sends prompts sequentially; no separate stage acceptance is required. Follow `staged-plan-runner` for lifecycle rules and use this pack's `ledger_update.py` for every execution-state mutation. The former updater gap was resolved by explicit owner instruction on 2026-10-07. Never edit active ledger JSON by hand, steal another executor's claim, or expire a claim automatically.

Once a stage is validly claimed, complete the authorized implementation, verification and in-scope fixes. Routine reversible engineering choices stay delegated. Do not ask the owner to approve an unchanged design or each next stage. A real missing external prerequisite blocks only its dependent proof. A new material owner decision must be distinguished from a failed test, implementation detail or earlier-stage output. No publication, deployment, credential change, destructive existing-artifact cleanup, Goal, branch or worktree is implied.

## Context acquisition

Always read repository `AGENTS.md`, this stage's live ledger record and the implementation plan's authority, invariant, compatibility and relevant stage sections. Refresh current source and ownership before editing. The plan's context table establishes observed baseline, not permission to overwrite concurrent changes. Read the task entrypoints below and only conditional bundles needed for the named change. Earlier reports are evidence; implementation decisions remain constrained by this plan and current user corrections.

## Conditional skills

- `backend-quality-gates`: when Python implementation/gates begin; select checks from current CI and affected boundaries.
- `contract-impact-analysis`: before changing API/DTO, file schema, persistence, config, port, cache or identity guarantees; retain classifications and migration directions in the report.
- `backend-performance-evidence`: for measurements, timing contracts or performance claims; mandatory in S08.
- `root-cause-debugging`: when a concrete failure's cause is unknown; fix only after causal evidence.
- `browser-qa-evidence`: if the implementation changes browser-visible behavior; use `playwright-cli` for terminal browser mechanics only when no user-selected browser overrides it.
- `architecture-design`: only if a newly evidenced invariant requires refining an unsettled design detail; do not reopen accepted npy/materialization/storage choices merely by preference.
- `prompt-manager`: only for an authorized artifact revision; preserve claimed/accepted contracts and use runner reconciliation instead of rewriting live stage IDs.

## Implementation constraints

Use production code and real contract boundaries. Reuse the existing builder, indicator computation, hit-time calculation, validators and mmap loader. No compatibility wrappers, fake slot/hash metadata, npy-to-CanonicalCandleReader adapter, generic helper/provider framework, duplicate numerical pipeline or direct ClickHouse fallback. No benchmark scaffolding or large data in the repository. Do not change financial precision, timeline/sentinel semantics, funding, canonical variant identity, ranking or supported direction policy for speed.

Inspect existing tests before adding a named planned module; extend the current owner when it already proves that boundary. Do not leave TODO-driven production behavior, catch-all silent fallbacks or success-shaped placeholder evidence. Preserve foreign paths/hunks and record necessary outside-expected paths with their reason. Existing diagrams/docs/history do not authorize unrelated work.

## Reporting and transition

Write the stage report at the exact contract path in English. Include actual created/modified/deleted paths, outside-expected changes with reasons, foreign changes excluded, exact commands/results/counts, actual evidence locations, compatibility classifications, remaining risk and next-stage inputs. Never log credentials, tokens, raw provider payloads or environment dumps. Keep large benchmark scripts/data/results outside Git; only bounded aggregate proof belongs in reports.

After required proof passes, use `ledger_update.py accept` with the actual report/evidence and `--checks-passed`. The command generates a new immutable receipt, binds live plan/prompt/report/evidence bytes, validates the successor, and commits the ledger transition under its directory lock. No manual receipt/hash construction is needed. A nonzero command result is not a transition. The next eligible stage is enabled but not started. End this stage's turn with its result and the next prompt link; the owner will submit that prompt separately. No additional acceptance question is required. Never claim a pass for missing proof.

## Task entrypoints and conditional source bundle

- `apps/worker/backtest_job_runner/wiring/modules/child_process.py`
- `apps/worker/backtest_job_runner/wiring/modules/full_job_compute.py`
- `src/trading/contexts/backtest/application/services/v2/job_orchestration.py`
- `apps/worker/backtest_job_runner/wiring/modules/process_observation.py`
- `apps/worker/backtest_job_runner/wiring/modules/compute_resources.py`
- `apps/worker/backtest_job_runner/wiring/modules/backtest_job_runner.py`
- `src/trading/contexts/backtest/application/use_cases/backtest_job_worker.py`
- `src/trading/contexts/backtest_artifacts/application/services/v2/artifact_slot_publisher.py`
- `src/trading/contexts/backtest/application/services/v2/job_scratch.py`

Read `reports/S03.md` produced by S03 before claiming. Its absence now is a deferred dependency, not an owner-input question.

## Required work

1. Use the existing parent-owned temporary child directory for an attempt inputs directory. Wire trusted output ownership through child entrypoints, full_job_compute and orchestration; preflight/public requests must never carry arbitrary paths.

2. Prepare one ready input set per attempt; reuse it for sample warmup, preparation and risk scoring. Close mmap owners before cleanup. Keep RAM scratch release and parent filesystem ownership explicit and coordinated.

3. Account for materialization under the current job timeout, cancellation and CPU-thread policy. Reserve bounded disk capacity before launch at existing admission/scheduling boundaries; release on all terminal paths. Apply configured per-attempt/worker/reserve guards and existing stricter memory/cell budgets.

4. Protect source slots for full and lazy attempts by extending the existing repository/gateway with plan 4.4’s explicit transactional ownership protocol. Cover resolve/admit/revalidate, rebuild and previous-slot deletion; current count-only checks and per-query connections are insufficient.

5. Handle child exception, cancellation, timeout and parent restart/orphan cleanup without deleting live/foreign directories. Verify ownership against job/materialization leases and process observation, not age or PID alone.

6. Extend child_ipc mappings for complete typed input recipe/provenance and existing funding fields. Preserve old IPC decoding only for explicitly supported transitional pairs. Add separate materialization/attempt timers while retaining the existing orchestration_elapsed_v2 measurement definition.

## Verification and acceptance

- uv run pytest -q tests/unit/apps/worker/backtest_job_runner tests/unit/apps/worker/test_backtest_job_runner.py tests/unit/contexts/backtest/application/use_cases/test_backtest_job_worker_use_case.py

- Add tests/unit/apps/worker/backtest_job_runner/test_materialization_lifecycle.py and tests/real_infra/backtest/test_artifact_input_lifecycle.py (planned paths). Inspect current real-infra fixtures first; use disposable Postgres resources.

- Execute C07/C08/C14 failure/concurrency drills and report live lease recovery independently from mocked process tests. Run focused ruff and uv run pyright.

All acceptance criteria in the front-matter contract must be evidenced. Worker subprocess/resource lifecycle and tested local Postgres concurrency; no production installation claim.

## Adjacent-stage handoff

S05 consumes attempt ownership, persisted provenance handoff and leases needed for deterministic replay.

## Resolved slot ownership protocol — mandatory

Implement plan 4.4's explicit protocol in the existing repository/gateway owners. Add BacktestPostgresGateway.transaction() bound to one connection and reserve_artifact_reader/release_artifact_reader/reserve_artifact_writer/complete_artifact_writer plus quarantine/recovery transitions. Use one SELECT FOR UPDATE ownership row per normalized physical coordinate/slot; generation is an expected value, not a disjoint mutex key. Reader admission, job creation and the frozen recipe commit share one transaction. Revalidate metadata under ownership; do not silently replace a stale preflight recipe.

Publisher rebuilding AND previous-slot deletion require writer reservation. A raw active-job COUNT is not exclusion. Lazy replay reserves a source before hashing/mmap. CAS ties worker ownership to attempt/parent incarnation. Expiration or DB loss never makes a live child/writer reclaimable: preserve or quarantine durable ownership until the exact owner is terminated/reaped or proven dead. DB epoch fencing does not itself fence filesystem writes. Build private candidates; finalize under exclusive ownership; reconcile pointer/manifests after proven owner death before releasing quarantine.

Extend child_ipc and the existing observed child-process path for a typed prepared-input event and parent acknowledgement. Parent CAS-persists artifact-prefix-payload/v1 and provenance before scoring begins. On cancellation, parent loss, stale attempt or bounded acknowledgement timeout the child does not score. Include the attestation scan and acknowledgement in inclusive attempt timing.

Add forced-interleaving real Postgres tests for readers arriving after the old count but before BOTH rebuild and previous-slot cleanup, expired leases with live children, and DB session loss. Do not pass these cases using only mocked counts.

Publisher reservation must also lock both coordinate slot rows in deterministic order, serialize publication/destructive cleanup across the coordinate, compare the expected current.yaml identity and target generation before any mutation, and reserve the incremental source as a reader for all reads. Reject stale prechecks and recheck pointer identity before final switch; cleanup must verify its target is not active. Add the C08 two-publisher stale-precheck interleaving. A per-slot mutex alone does not prove the target is still inactive.

## Journal commands for this stage

Run from the repository root. Choose a unique executor ID for this execution and retain it for resuming the same claim; do not reuse another executor's ID. The commands below use placeholders that the executor substitutes, not text the owner must provide.

```sh
python3 .codex/agents/generated/backtest-npy-input-materialization-v1/ledger_update.py claim --stage S04 --executor UNIQUE_EXECUTOR_ID
```

Only after the claim succeeds, perform the stage. Write the English stage report to `reports/S04.md`, including exact checks, results, owned changes and proof limits. Then, only when all required checks actually passed:

```sh
python3 .codex/agents/generated/backtest-npy-input-materialization-v1/ledger_update.py accept --stage S04 --executor SAME_EXECUTOR_ID --checks-passed --evidence reports/S04.md
```

Evidence paths resolve relative to this pack. Supply additional `--evidence` paths when needed. The report itself may serve as bounded check evidence when it records actual commands/results; a bare pass assertion is insufficient. This command does not run tests or establish their meaning.

Use `resume --stage S04 --executor SAME_EXECUTOR_ID` only for your own existing claim. For an actual new owner decision, `needs-input` requires `--reason`, `--resume-condition` and readable `--evidence`; resume then also requires `--resolution` pointing to the recorded answer. For a concrete hard blocker, `block` requires `--reason` and readable `--evidence`. A hard-blocked instance is terminal and needs a separately authored revision under the canonical lifecycle; do not reset it. Routine implementation problems should be diagnosed and fixed within the current claim, not converted into owner approvals. `status` is read-only. Every failed command leaves the prior ledger intact; an unreferenced immutable receipt after interruption is evidence only and must not be treated as acceptance.
