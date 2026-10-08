---
{
  "schema_version": "stage-prompt/v1",
  "prompt_pack_execution": {
    "plan_doc": "implementation-plan.md",
    "prompt_pack_dir": ".",
    "stage_ledger": "iteration-ledger.md"
  },
  "stage_contract": {
    "id": "S01",
    "title": "Versioned input contracts, persistence expansion and trusted baseline",
    "prompt_path": "01-contracts-and-baseline.md",
    "report_path": "reports/S01.md",
    "receipt_dir": "receipts/S01",
    "depends_on": [],
    "expected_touches": [
      "../../../../src/trading/contexts/backtest_artifacts/application/services/v2/contracts.py",
      "../../../../src/trading/contexts/backtest_artifacts/application/services/v2/artifact_manifest_loader.py",
      "../../../../src/trading/contexts/backtest/application/dto",
      "../../../../src/trading/contexts/backtest/domain/entities/backtest_job.py",
      "../../../../src/trading/contexts/backtest/adapters/outbound/persistence/postgres",
      "../../../../migrations/postgres",
      "../../../../tests/unit/contexts/backtest",
      "../../../../tests/unit/apps/migrations",
      "../../../../docs/architecture/backtest/backtest-service-artifact-runtime-v1.ru.md"
    ],
    "acceptance_criteria": [
      "Old schema-1 manifests and recipe-less jobs remain readable without rewritten hashes or guessed provenance.",
      "New typed contracts and strict schema-2/job-manifest parsers round-trip and reject malformed versions, shapes and identities.",
      "Additive migration is ordered and preserves existing rows; local test proof and unrun infra limits are explicit.",
      "Independent fixture baseline and year baseline source/data strategy are recorded outside Git; no numerical/runtime path is switched on in this stage.",
      "The two-phase canonical prefix-proof format and slot ownership tables are specified and round-trip without forcing payload scans in preflight."
    ],
    "proof_boundary": "Typed/serialized contracts and additive migration behavior on tested fixtures; no generated execution, performance or production migration claim.",
    "validation": {
      "profile": "roehub-focused-gates-and-prompt-pack-artifacts/v1",
      "checks": [
        "uv run pytest -q tests/unit/contexts/backtest/application/services/v2/test_yaml_backtest_artifact_loader_v2.py tests/unit/contexts/backtest/domain/entities/test_backtest_job_entities.py",
        "Extend existing fixture tests and add tests/unit/contexts/backtest/application/services/v2/test_input_recipe_contracts.py for version/identity/round-trip failures. Select migration SQL tests from tests/unit/apps/migrations; inspect the current harness before invoking real Postgres.",
        "Run focused ruff and uv run pyright after implementation. Record any unperformed real migration checks as unavailable, not passed."
      ],
      "requires_user_acceptance": false
    },
    "entry_inputs": [
      {
        "path": "implementation-plan.md",
        "producer_stage": null
      },
      {
        "path": "../../../../src/trading/contexts/backtest_artifacts/application/services/v2/contracts.py",
        "producer_stage": null
      },
      {
        "path": "../../../../src/trading/contexts/backtest/application/dto/runtime_preflight.py",
        "producer_stage": null
      },
      {
        "path": "../../../../src/trading/contexts/backtest/domain/entities/backtest_job.py",
        "producer_stage": null
      },
      {
        "path": "../../../../migrations/postgres/manifest.json",
        "producer_stage": null
      }
    ]
  }
}
---

# S01 — Versioned input contracts, persistence expansion and trusted baseline

Relevant implementation-plan sections: 1–5, 6 (C01/C10–C13), 7 baseline protocol.

## Execution authority and state

Read the live ledger before doing anything. It alone owns statuses, claims and decision packets. Execute exactly S01 when the owner submits this prompt, then stop with the next prompt link. The owner sends prompts sequentially; no separate stage acceptance is required. Follow `staged-plan-runner` for lifecycle rules and use this pack's `ledger_update.py` for every execution-state mutation. The former updater gap was resolved by explicit owner instruction on 2026-10-07. Never edit active ledger JSON by hand, steal another executor's claim, or expire a claim automatically.

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

- `src/trading/contexts/backtest_artifacts/application/services/v2/contracts.py`
- `src/trading/contexts/backtest/application/dto/runtime_preflight.py`
- `src/trading/contexts/backtest/domain/entities/backtest_job.py`
- `migrations/postgres/manifest.json`
- `src/trading/contexts/backtest_artifacts/application/services/v2/artifact_manifest_loader.py`
- `src/trading/contexts/backtest/application/services/v2/preflight.py`
- `src/trading/contexts/backtest/application/services/research_identity.py`
- `src/trading/contexts/backtest/domain/value_objects/variant_identity.py`
- `docs/architecture/backtest/README.md`

## Required work

1. Capture the relevant working-tree baseline and foreign paths. Before changing numerical preparation, record native outputs for the deterministic C01/C03 fixtures. Acquire and freeze the year fixture/reference in external scratch following plan section 7 if local prerequisites permit; otherwise preserve enough baseline source outside the repository to run the same native reference later. A missing independent baseline must not be disguised as candidate-generated reference.

2. Define planned ArtifactCandleSnapshot, ArtifactDerivativeBuildRequest/Result, ArtifactInputFileReference, BacktestPreparedArtifactSet and BacktestInputRecipe in their named existing contract owners. Specify exact row-ID mapping, time origin/domain, source-prefix/rule/default/precision/funding identities, trusted roots and ownership. Distinguish persistent source metadata, semantic recipe identity and physical preparation provenance.

3. Add root schema-2 parsing with optional derivative inventory while retaining strict schema-1 parsing. Add a distinct job-prepared manifest kind/version; never accept it as a publication slot. Update exact-key and dataclass validation together, including serialization round trips.

4. Add nullable recipe/provenance persistence through the next free ordered migration and its manifest entry. Update entity/repository representations needed to round-trip new fields, while preserving old rows and old hash schemes. Do not alter applied migration bytes or enable new runtime writers yet.

5. Introduce a versioned semantic hash contract excluding retention, temporary paths and reused/generated choices. Keep canonical request and variant identities stable. Record schema, identity and rollback directions in the normative runtime document.

## Verification and acceptance

- uv run pytest -q tests/unit/contexts/backtest/application/services/v2/test_yaml_backtest_artifact_loader_v2.py tests/unit/contexts/backtest/domain/entities/test_backtest_job_entities.py

- Extend existing fixture tests and add tests/unit/contexts/backtest/application/services/v2/test_input_recipe_contracts.py for version/identity/round-trip failures. Select migration SQL tests from tests/unit/apps/migrations; inspect the current harness before invoking real Postgres.

- Run focused ruff and uv run pyright after implementation. Record any unperformed real migration checks as unavailable, not passed.

All acceptance criteria in the front-matter contract must be evidenced. Typed/serialized contracts and additive migration behavior on tested fixtures; no generated execution, performance or production migration claim.

## Adjacent-stage handoff

S02 must receive the accepted contract definitions, legacy compatibility fixtures and baseline evidence/source locations.

## Resolved prefix proof and source ownership schema

Implement the exact two-phase proof contract in plan 4.1: bounded manifest-based source_file_identities and prefix_proof=pending are valid at preflight; artifact-prefix-payload/v1 is a canonical little-endian payload/domain attestation computed in the reserved worker path, durably acknowledged before scoring. File SHA and payload-prefix identity are different. Add strict state/serialization tests; no preflight full-array scan.

The additive migration also creates backtest_artifact_slot_ownership and backtest_artifact_slot_readers as specified in plan 4.4. Their physical slot key excludes generation (which is validated inside the row). Record exact owner/attempt/epoch and lifecycle predicates, organization visibility and nullable legacy behavior. Do not claim the current per-query gateway already supplies a shared ownership transaction.

## Journal commands for this stage

Run from the repository root. Choose a unique executor ID for this execution and retain it for resuming the same claim; do not reuse another executor's ID. The commands below use placeholders that the executor substitutes, not text the owner must provide.

```sh
python3 .codex/agents/generated/backtest-npy-input-materialization-v1/ledger_update.py claim --stage S01 --executor UNIQUE_EXECUTOR_ID
```

Only after the claim succeeds, perform the stage. Write the English stage report to `reports/S01.md`, including exact checks, results, owned changes and proof limits. Then, only when all required checks actually passed:

```sh
python3 .codex/agents/generated/backtest-npy-input-materialization-v1/ledger_update.py accept --stage S01 --executor SAME_EXECUTOR_ID --checks-passed --evidence reports/S01.md
```

Evidence paths resolve relative to this pack. Supply additional `--evidence` paths when needed. The report itself may serve as bounded check evidence when it records actual commands/results; a bare pass assertion is insufficient. This command does not run tests or establish their meaning.

Use `resume --stage S01 --executor SAME_EXECUTOR_ID` only for your own existing claim. For an actual new owner decision, `needs-input` requires `--reason`, `--resume-condition` and readable `--evidence`; resume then also requires `--resolution` pointing to the recorded answer. For a concrete hard blocker, `block` requires `--reason` and readable `--evidence`. A hard-blocked instance is terminal and needs a separately authored revision under the canonical lifecycle; do not reset it. Routine implementation problems should be diagnosed and fixed within the current claim, not converted into owner approvals. `status` is read-only. Every failed command leaves the prior ledger intact; an unreferenced immutable receipt after interruption is evidence only and must not be treated as acceptance.
