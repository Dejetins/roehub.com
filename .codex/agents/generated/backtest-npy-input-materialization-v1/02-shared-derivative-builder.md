---
{
  "schema_version": "stage-prompt/v1",
  "prompt_pack_execution": {
    "plan_doc": "implementation-plan.md",
    "prompt_pack_dir": ".",
    "stage_ledger": "iteration-ledger.md"
  },
  "stage_contract": {
    "id": "S02",
    "title": "Native selected-derivative builder and publication reuse",
    "prompt_path": "02-shared-derivative-builder.md",
    "report_path": "reports/S02.md",
    "receipt_dir": "receipts/S02",
    "depends_on": [
      "S01"
    ],
    "expected_touches": [
      "../../../../src/trading/contexts/backtest_artifacts/application/services/v2/artifact_precompute_runner.py",
      "../../../../src/trading/contexts/backtest_artifacts/application/services/v2/contracts.py",
      "../../../../src/trading/contexts/backtest_artifacts/application/services/v2/hit_times_compute_v2.py",
      "../../../../src/trading/contexts/backtest_artifacts/application/services/v2/signal_chunk_planner_v2.py",
      "../../../../src/trading/contexts/backtest_artifacts/application/services/v2/artifact_precompute_coordinator.py",
      "../../../../src/trading/contexts/backtest_artifacts/application/services/v2/artifact_slot_publisher.py",
      "../../../../tests/unit/contexts/backtest/application/services/v2/test_artifact_precompute_runner_v2.py",
      "../../../../tests/unit/contexts/backtest/application/services/v2/test_hit_times_compute_v2.py",
      "../../../../tests/unit/contexts/backtest/application/services/v2/test_derived_artifact_materialization.py"
    ],
    "acceptance_criteria": [
      "Native publication and selected materialization share the same grid/compute/encoding/writing implementation.",
      "Generated selected arrays match the trusted source-domain baseline exactly, including sparse row IDs, EMA seed history and TP/SL sentinels.",
      "Derived-only mode performs zero canonical candle reads and zero publication-pointer writes; risk none creates zero risk files.",
      "Injected chunk/write failures leave no consumable manifest and respect configured allocation limits."
    ],
    "proof_boundary": "Production builder correctness and file publication atomicity on bounded fixtures, not job lifecycle or year performance.",
    "validation": {
      "profile": "roehub-focused-gates-and-prompt-pack-artifacts/v1",
      "checks": [
        "uv run pytest -q tests/unit/contexts/backtest/application/services/v2/test_artifact_precompute_runner_v2.py tests/unit/contexts/backtest/application/services/v2/test_hit_times_compute_v2.py tests/unit/contexts/backtest/application/services/v2/test_derived_artifact_materialization.py",
        "Use real npy files and production SMA/EMA/WMA calculation for C01/C03/C04. Negative dispatch assertions may use spies; a mocked compute cannot establish parity.",
        "Run focused ruff and uv run pyright; inspect the diff for duplicate compute/writer paths and nested parallelism."
      ],
      "requires_user_acceptance": false
    },
    "entry_inputs": [
      {
        "path": "implementation-plan.md",
        "producer_stage": null
      },
      {
        "path": "../../../../src/trading/contexts/backtest_artifacts/application/services/v2/artifact_precompute_runner.py",
        "producer_stage": null
      },
      {
        "path": "../../../../src/trading/contexts/backtest_artifacts/application/services/v2/hit_times_compute_v2.py",
        "producer_stage": null
      },
      {
        "path": "reports/S01.md",
        "producer_stage": "S01"
      }
    ]
  }
}
---

# S02 — Native selected-derivative builder and publication reuse

Relevant implementation-plan sections: 4.1–4.2, 4.6, 6 (C01–C04), 7 fixture requirements.

## Execution authority and state

Read the live ledger before doing anything. It alone owns statuses, claims and decision packets. Execute exactly S02 when the owner submits this prompt, then stop with the next prompt link. The owner sends prompts sequentially; no separate stage acceptance is required. Follow `staged-plan-runner` for lifecycle rules and use this pack's `ledger_update.py` for every execution-state mutation. The former updater gap was resolved by explicit owner instruction on 2026-10-07. Never edit active ledger JSON by hand, steal another executor's claim, or expire a claim automatically.

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

- `src/trading/contexts/backtest_artifacts/application/services/v2/artifact_precompute_runner.py`
- `src/trading/contexts/backtest_artifacts/application/services/v2/hit_times_compute_v2.py`
- `src/trading/contexts/backtest_artifacts/application/services/v2/signal_rules_engine_v2.py`
- `src/trading/contexts/backtest_artifacts/application/services/v2/signal_chunk_planner_v2.py`
- `src/trading/contexts/backtest_artifacts/application/services/v2/artifact_precompute_coordinator.py`
- `src/trading/contexts/backtest_artifacts/application/services/v2/artifact_slot_publisher.py`
- `tests/unit/contexts/backtest/application/services/v2/artifact_testkit_v2.py`

Read `reports/S01.md` produced by S01 before claiming. Its absence now is a deferred dependency, not an owner-input question.

## Required work

1. Implement the planned materialize_derived operation on BacktestArtifactPrecomputeRunnerV2 using typed requests and an existing published candle snapshot. Move the mandatory canonical_candle_reader check to price-export entry; do not implement a reader adapter around npy or inject fake dependencies.

2. Refactor _materialize_signal_artifact_v2, grid/row planning, chunk computation, _evaluate_signal_matrix_v2 and atomic serialization into a single production implementation called by both scheduled publication and derivative-only materialization. Do not duplicate formulas or import private writer functions into backtest services.

3. Build only requested rows and missing coverage with stable canonical row IDs. Preserve dependencies, source fields, encoding, features needed by selection, dtype and complete seed history. Keep the original source timeline for EMA and hit-time indexes/sentinels; no arbitrary lookback substitute.

4. Reuse materialize_hit_times_from_ohlcv_v2 for selected risk levels under bounded cell/byte budgets. Skip it entirely for risk none. Keep actual publisher 15m semantics and correct touched misleading 1m comments.

5. Write private incomplete outputs, hash/validate closed files, then finalize the real generated manifest. Publication invokes the shared operation but alone owns current.yaml. Preserve bounded chunk scheduling; respect a child job already owning Numba threads and do not nest workqueues/process pools.

## Verification and acceptance

- uv run pytest -q tests/unit/contexts/backtest/application/services/v2/test_artifact_precompute_runner_v2.py tests/unit/contexts/backtest/application/services/v2/test_hit_times_compute_v2.py tests/unit/contexts/backtest/application/services/v2/test_derived_artifact_materialization.py

- Use real npy files and production SMA/EMA/WMA calculation for C01/C03/C04. Negative dispatch assertions may use spies; a mocked compute cannot establish parity.

- Run focused ruff and uv run pyright; inspect the diff for duplicate compute/writer paths and nested parallelism.

All acceptance criteria in the front-matter contract must be evidenced. Production builder correctness and file publication atomicity on bounded fixtures, not job lifecycle or year performance.

## Adjacent-stage handoff

S03 consumes actual generated manifest fixtures, selected-row mappings and error/resource behavior.

## Resolved publication staging requirement

The derivative/materialization refactor must support private candidate staging for publication too. Existing published files must not be mutated via r+ while building an inactive-slot candidate. The publisher finalizes only under the S04 writer reservation; derived-only job construction never owns that publication transition. Keep source arrays immutable while hashing/computing and respect canonical prefix-payload byte order without allocating an unbounded duplicate dataset.

## Journal commands for this stage

Run from the repository root. Choose a unique executor ID for this execution and retain it for resuming the same claim; do not reuse another executor's ID. The commands below use placeholders that the executor substitutes, not text the owner must provide.

```sh
python3 .codex/agents/generated/backtest-npy-input-materialization-v1/ledger_update.py claim --stage S02 --executor UNIQUE_EXECUTOR_ID
```

Only after the claim succeeds, perform the stage. Write the English stage report to `reports/S02.md`, including exact checks, results, owned changes and proof limits. Then, only when all required checks actually passed:

```sh
python3 .codex/agents/generated/backtest-npy-input-materialization-v1/ledger_update.py accept --stage S02 --executor SAME_EXECUTOR_ID --checks-passed --evidence reports/S02.md
```

Evidence paths resolve relative to this pack. Supply additional `--evidence` paths when needed. The report itself may serve as bounded check evidence when it records actual commands/results; a bare pass assertion is insufficient. This command does not run tests or establish their meaning.

Use `resume --stage S02 --executor SAME_EXECUTOR_ID` only for your own existing claim. For an actual new owner decision, `needs-input` requires `--reason`, `--resume-condition` and readable `--evidence`; resume then also requires `--resolution` pointing to the recorded answer. For a concrete hard blocker, `block` requires `--reason` and readable `--evidence`. A hard-blocked instance is terminal and needs a separately authored revision under the canonical lifecycle; do not reset it. Routine implementation problems should be diagnosed and fixed within the current claim, not converted into owner approvals. `status` is read-only. Every failed command leaves the prior ledger intact; an unreferenced immutable receipt after interruption is evidence only and must not be treated as acceptance.
