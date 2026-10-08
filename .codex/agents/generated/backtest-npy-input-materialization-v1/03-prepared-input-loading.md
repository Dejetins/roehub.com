---
{
  "schema_version": "stage-prompt/v1",
  "prompt_pack_execution": {
    "plan_doc": "implementation-plan.md",
    "prompt_pack_dir": ".",
    "stage_ledger": "iteration-ledger.md"
  },
  "stage_contract": {
    "id": "S03",
    "title": "Unified prepared input resolution, validation and mmap loading",
    "prompt_path": "03-prepared-input-loading.md",
    "report_path": "reports/S03.md",
    "receipt_dir": "receipts/S03",
    "depends_on": [
      "S02"
    ],
    "expected_touches": [
      "../../../../src/trading/contexts/backtest/application/ports/artifact_arrays.py",
      "../../../../src/trading/contexts/backtest/adapters/outbound/artifacts_fs",
      "../../../../src/trading/contexts/backtest/application/services/v2/prepare_pools.py",
      "../../../../src/trading/contexts/backtest/application/services/v2/tp_sl_hit_times.py",
      "../../../../src/trading/contexts/backtest/application/dto/prepare_pools.py",
      "../../../../src/trading/contexts/backtest_artifacts/application/services/v2/artifact_manifest_validator.py",
      "../../../../src/trading/contexts/backtest_artifacts/application/services/v2/contracts.py",
      "../../../../tests/unit/contexts/backtest/adapters/outbound/artifacts_fs",
      "../../../../tests/unit/contexts/backtest/application/services/v2/test_prepare_pools_service.py",
      "../../../../tests/unit/contexts/backtest/application/services/v2/test_tp_sl_hit_times_service.py",
      "../../../../tests/unit/contexts/backtest/application/services/v2/test_artifact_manifest_validator_v2.py"
    ],
    "acceptance_criteria": [
      "N/G/M inputs follow one loader/preparation path with exact row identity and numerical equivalence.",
      "No transient input set uses fake slot/current metadata and no candle dataset is duplicated per job.",
      "Missing supported derivatives materialize; corrupt or incompatible declared data fails before scoring.",
      "Path validation remains strict and native no-risk operation does not require hit-time data."
    ],
    "proof_boundary": "File/port/preparation contract equivalence and negative validation on real local fixtures; authorization is additionally verified at API integration later.",
    "validation": {
      "profile": "roehub-focused-gates-and-prompt-pack-artifacts/v1",
      "checks": [
        "uv run pytest -q tests/unit/contexts/backtest/adapters/outbound/artifacts_fs tests/unit/contexts/backtest/application/services/v2/test_prepare_pools_service.py tests/unit/contexts/backtest/application/services/v2/test_tp_sl_hit_times_service.py tests/unit/contexts/backtest/application/services/v2/test_artifact_manifest_validator_v2.py",
        "Exercise C02–C06 with real files, including mixed sparse rows/levels, corrupt declared files, untrusted roots and unchanged published candles.",
        "Run focused ruff and uv run pyright; search all changed port consumers before declaring internal migration complete."
      ],
      "requires_user_acceptance": false
    },
    "entry_inputs": [
      {
        "path": "implementation-plan.md",
        "producer_stage": null
      },
      {
        "path": "../../../../src/trading/contexts/backtest/application/ports/artifact_arrays.py",
        "producer_stage": null
      },
      {
        "path": "../../../../src/trading/contexts/backtest/adapters/outbound/artifacts_fs/artifact_array_loader.py",
        "producer_stage": null
      },
      {
        "path": "../../../../src/trading/contexts/backtest/application/services/v2/prepare_pools.py",
        "producer_stage": null
      },
      {
        "path": "reports/S02.md",
        "producer_stage": "S02"
      }
    ]
  }
}
---

# S03 — Unified prepared input resolution, validation and mmap loading

Relevant implementation-plan sections: 4.1–4.3, 5 internal port compatibility, 6 (C02–C06).

## Execution authority and state

Read the live ledger before doing anything. It alone owns statuses, claims and decision packets. Execute exactly S03 when the owner submits this prompt, then stop with the next prompt link. The owner sends prompts sequentially; no separate stage acceptance is required. Follow `staged-plan-runner` for lifecycle rules and use this pack's `ledger_update.py` for every execution-state mutation. The former updater gap was resolved by explicit owner instruction on 2026-10-07. Never edit active ledger JSON by hand, steal another executor's claim, or expire a claim automatically.

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

- `src/trading/contexts/backtest/application/ports/artifact_arrays.py`
- `src/trading/contexts/backtest/adapters/outbound/artifacts_fs/artifact_array_loader.py`
- `src/trading/contexts/backtest/application/services/v2/prepare_pools.py`
- `src/trading/contexts/backtest_artifacts/application/services/v2/artifact_manifest_validator.py`
- `src/trading/contexts/backtest/adapters/outbound/artifacts_fs/artifact_context_resolver.py`
- `src/trading/contexts/backtest/application/services/v2/tp_sl_hit_times.py`
- `src/trading/contexts/backtest/application/dto/prepare_pools.py`

Read `reports/S02.md` produced by S02 before claiming. Its absence now is a deferred dependency, not an owner-input question.

## Required work

1. Implement prepare_artifact_inputs on the existing BacktestPreparePoolsService: resolve a recipe against published inventory, preserve valid ready references, send only missing requirements to the shared builder, return one complete BacktestPreparedArtifactSet. Separate absence from corruption and version/source mismatch.

2. Evolve the existing BacktestArtifactArrayLoader port and concrete loader to explicit validated file references. Update all callers of resolve_context/load_price_arrays/load_mapping_arrays/load_signal_matrix/load_signal_rows/load_hit_times_* and funding reads; changing slot_root_path alone is insufficient.

3. Add validate_prepared_inputs to the existing validator; validate request completeness as well as every declared file. Keep schema-1 strictness and publication-policy validation distinct. Constrain references to the trusted pinned base and attempt roots, reject traversal and symlink escapes, and retain allow_pickle=False/mmap read-only.

4. Make open_artifact_arrays and prepare_pools_core honor canonical row IDs separate from physical positions for generated/partial matrices. Ensure mixed row/level references present a coherent logical timeline without copying all candle arrays.

5. Update BacktestTpSlHitTimesService.execute/validate_grid/materialize_subset for explicit complete selected risk coverage; risk none must require no manifest. Preserve scoring algorithms, grid matching tolerance and sentinel/timeline domain.

## Verification and acceptance

- uv run pytest -q tests/unit/contexts/backtest/adapters/outbound/artifacts_fs tests/unit/contexts/backtest/application/services/v2/test_prepare_pools_service.py tests/unit/contexts/backtest/application/services/v2/test_tp_sl_hit_times_service.py tests/unit/contexts/backtest/application/services/v2/test_artifact_manifest_validator_v2.py

- Exercise C02–C06 with real files, including mixed sparse rows/levels, corrupt declared files, untrusted roots and unchanged published candles.

- Run focused ruff and uv run pyright; search all changed port consumers before declaring internal migration complete.

All acceptance criteria in the front-matter contract must be evidenced. File/port/preparation contract equivalence and negative validation on real local fixtures; authorization is additionally verified at API integration later.

## Adjacent-stage handoff

S04 receives the ready-set lifecycle and exact loader signature changes; no unconverted caller may be hidden behind a new adapter.

## Resolved prefix verification requirement

Validate file SHA as physical integrity and artifact-prefix-payload/v1 as source-domain equivalence; never substitute one for the other. Follow plan 4.1's exact descriptor, length prefix, role ordering, endian/axis order and recorded domain rules. Replay from appended files hashes the old recorded slices with their original descriptors. Whole-array scans belong to measured preparation, not cheap preflight. No scoring consumes a pending/unacknowledged attestation.

## Journal commands for this stage

Run from the repository root. Choose a unique executor ID for this execution and retain it for resuming the same claim; do not reuse another executor's ID. The commands below use placeholders that the executor substitutes, not text the owner must provide.

```sh
python3 .codex/agents/generated/backtest-npy-input-materialization-v1/ledger_update.py claim --stage S03 --executor UNIQUE_EXECUTOR_ID
```

Only after the claim succeeds, perform the stage. Write the English stage report to `reports/S03.md`, including exact checks, results, owned changes and proof limits. Then, only when all required checks actually passed:

```sh
python3 .codex/agents/generated/backtest-npy-input-materialization-v1/ledger_update.py accept --stage S03 --executor SAME_EXECUTOR_ID --checks-passed --evidence reports/S03.md
```

Evidence paths resolve relative to this pack. Supply additional `--evidence` paths when needed. The report itself may serve as bounded check evidence when it records actual commands/results; a bare pass assertion is insufficient. This command does not run tests or establish their meaning.

Use `resume --stage S03 --executor SAME_EXECUTOR_ID` only for your own existing claim. For an actual new owner decision, `needs-input` requires `--reason`, `--resume-condition` and readable `--evidence`; resume then also requires `--resolution` pointing to the recorded answer. For a concrete hard blocker, `block` requires `--reason` and readable `--evidence`. A hard-blocked instance is terminal and needs a separately authored revision under the canonical lifecycle; do not reset it. Routine implementation problems should be diagnosed and fixed within the current claim, not converted into owner approvals. `status` is read-only. Every failed command leaves the prior ledger intact; an unreferenced immutable receipt after interruption is evidence only and must not be treated as acceptance.
