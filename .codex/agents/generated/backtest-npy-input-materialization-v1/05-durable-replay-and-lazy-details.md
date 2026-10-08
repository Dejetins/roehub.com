---
{
  "schema_version": "stage-prompt/v1",
  "prompt_pack_execution": {
    "plan_doc": "implementation-plan.md",
    "prompt_pack_dir": ".",
    "stage_ledger": "iteration-ledger.md"
  },
  "stage_contract": {
    "id": "S05",
    "title": "Durable source recipes, deterministic replay and lazy reconstruction",
    "prompt_path": "05-durable-replay-and-lazy-details.md",
    "report_path": "reports/S05.md",
    "receipt_dir": "receipts/S05",
    "depends_on": [
      "S04"
    ],
    "expected_touches": [
      "../../../../src/trading/contexts/backtest/domain/entities/backtest_job.py",
      "../../../../src/trading/contexts/backtest/application/dto",
      "../../../../src/trading/contexts/backtest/application/use_cases",
      "../../../../src/trading/contexts/backtest/application/services/v2/preflight.py",
      "../../../../src/trading/contexts/backtest/application/services/v2/lazy_trades_detail.py",
      "../../../../src/trading/contexts/backtest/application/services/v2/result_series.py",
      "../../../../src/trading/contexts/backtest/application/ports/lazy_trades_cache.py",
      "../../../../src/trading/contexts/backtest/adapters/outbound/cache_fs",
      "../../../../src/trading/contexts/backtest/adapters/outbound/persistence/postgres",
      "../../../../apps/worker/backtest_job_runner/wiring/modules/lazy_trades_compute.py",
      "../../../../apps/worker/backtest_job_runner/wiring/modules/lazy_trades_child_process.py",
      "../../../../apps/worker/backtest_job_runner/main/lazy_trades_child.py",
      "../../../../tests/unit/contexts/backtest/",
      "zone: Disposable Postgres job provenance and replay verification"
    ],
    "acceptance_criteria": [
      "Post-cleanup and post-restart selected details/trades/series/funding match trusted native results.",
      "Append-only snapshots replay only after original-domain proof; changed history/math versions fail explicitly.",
      "Queued jobs retain original recipe semantics and stale attempts cannot overwrite durable provenance.",
      "Legacy jobs remain readable with established limits; storage policies/paths do not alter request/variant identity.",
      "New successful jobs have durable payload-prefix attestation; header/shape changes from append do not invalidate an identical recorded payload domain."
    ],
    "proof_boundary": "Durable/replay semantics on tested local source snapshots and disposable database; no arbitrary historical-data recovery guarantee.",
    "validation": {
      "profile": "roehub-focused-gates-and-prompt-pack-artifacts/v1",
      "checks": [
        "uv run pytest -q tests/unit/contexts/backtest/application/services/v2/test_lazy_trades_detail_service.py tests/unit/contexts/backtest/adapters/outbound/cache_fs/test_lazy_trades_cache.py tests/unit/contexts/backtest/application/use_cases/test_backtest_jobs_use_case.py tests/unit/contexts/backtest/application/use_cases/test_lazy_trades_materialization_worker_use_case.py",
        "Add tests/unit/contexts/backtest/application/services/v2/test_input_recipe_replay.py and extend tests/real_infra/backtest/test_artifact_input_lifecycle.py for C09–C12, including old rows and funding.",
        "Run focused ruff and uv run pyright; search all artifact pin/cache/metadata serializers to ensure source meanings are consistent."
      ],
      "requires_user_acceptance": false
    },
    "entry_inputs": [
      {
        "path": "implementation-plan.md",
        "producer_stage": null
      },
      {
        "path": "../../../../src/trading/contexts/backtest/application/use_cases/backtest_jobs.py",
        "producer_stage": null
      },
      {
        "path": "../../../../src/trading/contexts/backtest/application/use_cases/backtest_job_worker.py",
        "producer_stage": null
      },
      {
        "path": "../../../../src/trading/contexts/backtest/application/services/v2/lazy_trades_detail.py",
        "producer_stage": null
      },
      {
        "path": "reports/S04.md",
        "producer_stage": "S04"
      }
    ]
  }
}
---

# S05 — Durable source recipes, deterministic replay and lazy reconstruction

Relevant implementation-plan sections: 4.1 identity, 4.5, 5 persisted compatibility, 6 (C09–C12).

## Execution authority and state

Read the live ledger before doing anything. It alone owns statuses, claims and decision packets. Execute exactly S05 when the owner submits this prompt, then stop with the next prompt link. The owner sends prompts sequentially; no separate stage acceptance is required. Follow `staged-plan-runner` for lifecycle rules and use this pack's `ledger_update.py` for every execution-state mutation. The former updater gap was resolved by explicit owner instruction on 2026-10-07. Never edit active ledger JSON by hand, steal another executor's claim, or expire a claim automatically.

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

- `src/trading/contexts/backtest/application/use_cases/backtest_jobs.py`
- `src/trading/contexts/backtest/application/use_cases/backtest_job_worker.py`
- `src/trading/contexts/backtest/application/services/v2/lazy_trades_detail.py`
- `src/trading/contexts/backtest/application/services/v2/preflight.py`
- `src/trading/contexts/backtest/application/ports/lazy_trades_cache.py`
- `src/trading/contexts/backtest/adapters/outbound/cache_fs/lazy_trades_cache.py`
- `src/trading/contexts/backtest/application/use_cases/lazy_trades_materialization_worker.py`
- `apps/worker/backtest_job_runner/wiring/modules/lazy_trades_compute.py`
- `apps/worker/backtest_job_runner/wiring/modules/lazy_trades_child_process.py`
- `src/trading/contexts/backtest/application/services/v2/result_series.py`

Read `reports/S04.md` produced by S04 before claiming. Its absence now is a deferred dependency, not an owner-input question.

## Required work

1. Persist the frozen input recipe at job creation and ready-set provenance under the active attempt/lease before temporary files disappear. Extend all repository/entity/DTO serializers deliberately; no path is durable provenance.

2. Add stored-recipe validation in preflight and make run_next preserve its source/financial meaning when rechecking admission. Do not silently select current defaults/pointer. Nullable recipes retain tested legacy behavior without fabricating missing source fingerprints.

3. Rework _recompute_payload and _tp_sl_detail to prepare only selected rows and the selected risk pair through the same builder/loader. Reuse parent-owned lazy attempt files and cleanup mechanisms.

4. Verify full original source-prefix/domain/funding fingerprint before replay from a newer appended snapshot. Slice to the recorded domain so new future bars cannot change hit-time sentinels or final-bar behavior. Reject rewritten history or unavailable math versions explicitly.

5. Version semantic lazy cache keys by organization, source recipe and selected variant; retention/path changes do not change financial identity. Test details, chart/series and CSV consumers, funding overlays and existing published-source metadata.

6. Keep stale workers from overwriting another attempt provenance/results through the existing lease compare-and-set. Do not write fabricated generated hashes on failure before final validation.

## Verification and acceptance

- uv run pytest -q tests/unit/contexts/backtest/application/services/v2/test_lazy_trades_detail_service.py tests/unit/contexts/backtest/adapters/outbound/cache_fs/test_lazy_trades_cache.py tests/unit/contexts/backtest/application/use_cases/test_backtest_jobs_use_case.py tests/unit/contexts/backtest/application/use_cases/test_lazy_trades_materialization_worker_use_case.py

- Add tests/unit/contexts/backtest/application/services/v2/test_input_recipe_replay.py and extend tests/real_infra/backtest/test_artifact_input_lifecycle.py for C09–C12, including old rows and funding.

- Run focused ruff and uv run pyright; search all artifact pin/cache/metadata serializers to ensure source meanings are consistent.

All acceptance criteria in the front-matter contract must be evidenced. Durable/replay semantics on tested local source snapshots and disposable database; no arbitrary historical-data recovery guarantee.

## Adjacent-stage handoff

S06 may enable on-demand policy only after recipe persistence and lazy recovery work together.

## Resolved replay attestation contract

At creation, store the exact manifest-based source identities with prefix_proof=pending; no fabricated verified prefix is allowed. Before scoring, CAS-persist the worker's canonical payload-prefix attestation and acknowledge its matching attempt. Use this durable attestation for lazy replay. Changed npy headers/shapes after append are allowed only when the recorded payload-domain proof matches; changed consumed candles/mappings/funding fail. Preserve original array origins, shapes of the recorded slices and sentinel domain. A successful new job without attested provenance is invalid.

## Journal commands for this stage

Run from the repository root. Choose a unique executor ID for this execution and retain it for resuming the same claim; do not reuse another executor's ID. The commands below use placeholders that the executor substitutes, not text the owner must provide.

```sh
python3 .codex/agents/generated/backtest-npy-input-materialization-v1/ledger_update.py claim --stage S05 --executor UNIQUE_EXECUTOR_ID
```

Only after the claim succeeds, perform the stage. Write the English stage report to `reports/S05.md`, including exact checks, results, owned changes and proof limits. Then, only when all required checks actually passed:

```sh
python3 .codex/agents/generated/backtest-npy-input-materialization-v1/ledger_update.py accept --stage S05 --executor SAME_EXECUTOR_ID --checks-passed --evidence reports/S05.md
```

Evidence paths resolve relative to this pack. Supply additional `--evidence` paths when needed. The report itself may serve as bounded check evidence when it records actual commands/results; a bare pass assertion is insufficient. This command does not run tests or establish their meaning.

Use `resume --stage S05 --executor SAME_EXECUTOR_ID` only for your own existing claim. For an actual new owner decision, `needs-input` requires `--reason`, `--resume-condition` and readable `--evidence`; resume then also requires `--resolution` pointing to the recorded answer. For a concrete hard blocker, `block` requires `--reason` and readable `--evidence`. A hard-blocked instance is terminal and needs a separately authored revision under the canonical lifecycle; do not reset it. Routine implementation problems should be diagnosed and fixed within the current claim, not converted into owner approvals. `status` is read-only. Every failed command leaves the prior ledger intact; an unreferenced immutable receipt after interruption is evidence only and must not be treated as acceptance.
