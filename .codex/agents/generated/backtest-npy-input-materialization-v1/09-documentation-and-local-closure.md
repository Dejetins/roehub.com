---
{
  "schema_version": "stage-prompt/v1",
  "prompt_pack_execution": {
    "plan_doc": "implementation-plan.md",
    "prompt_pack_dir": ".",
    "stage_ledger": "iteration-ledger.md"
  },
  "stage_contract": {
    "id": "S09",
    "title": "Canonical documentation, contract reconciliation and local delivery closure",
    "prompt_path": "09-documentation-and-local-closure.md",
    "report_path": "reports/S09.md",
    "receipt_dir": "receipts/S09",
    "depends_on": [
      "S08"
    ],
    "expected_touches": [
      "../../../../docs/architecture/backtest/README.md",
      "../../../../docs/architecture/backtest/backtest-service-artifact-runtime-v1.ru.md",
      "../../../../docs/architecture/backtest/backtest-service-artifact-runtime-v1.md",
      "../../../../docs/runbooks/offline-release-installation.md",
      "../../../../docs/runbooks/indicators-numba-cache-and-threads.md",
      "../../../../docs/architecture/api/api-errors-and-422-payload-v1.md",
      "../../../../docs/architecture/README.md",
      "../../../../docs/architecture/project-map",
      "zone: Narrow documentation corrections in changed source docstrings"
    ],
    "acceptance_criteria": [
      "Canonical docs describe the implemented contracts and verified operational limits without reviving retired procedures.",
      "All required stage obligations, correctness proofs and year measurements are satisfied and linked; no unassessed required proof is presented as a pass.",
      "Owned/foreign changes and local-vs-production boundaries are explicit; generated indices and final consistency checks pass.",
      "Final receipt has no next stage; the runner completes the ledger only after its actual lifecycle obligations are met."
    ],
    "proof_boundary": "Local implementation/documentation closure with linked verification and benchmark evidence; no Git publication or production deployment.",
    "validation": {
      "profile": "roehub-focused-gates-and-prompt-pack-artifacts/v1",
      "checks": [
        "uv run python -m tools.docs.generate_docs_index --check",
        "uv run python -m tools.docs.generate_project_map --check",
        "Reconcile required affected gates with the last changed code; rerun only after relevant edits or unresolved concerns. Run pack receipt validation against actual live files and immutable evidence under the supported updater before transition."
      ],
      "requires_user_acceptance": false
    },
    "entry_inputs": [
      {
        "path": "implementation-plan.md",
        "producer_stage": null
      },
      {
        "path": "../../../../docs/architecture/backtest/README.md",
        "producer_stage": null
      },
      {
        "path": "../../../../docs/architecture/backtest/backtest-service-artifact-runtime-v1.ru.md",
        "producer_stage": null
      },
      {
        "path": "../../../../docs/runbooks/offline-release-installation.md",
        "producer_stage": null
      },
      {
        "path": "reports/S08.md",
        "producer_stage": "S08"
      }
    ]
  }
}
---

# S09 — Canonical documentation, contract reconciliation and local delivery closure

Relevant implementation-plan sections: 1 authority, 5 rollout, 8 traceability, 9 canonical docs/review.

## Execution authority and state

Read the live ledger before doing anything. It alone owns statuses, claims and decision packets. Execute exactly S09 when the owner submits this prompt, then stop with the next prompt link. The owner sends prompts sequentially; no separate stage acceptance is required. Follow `staged-plan-runner` for lifecycle rules and use this pack's `ledger_update.py` for every execution-state mutation. The former updater gap was resolved by explicit owner instruction on 2026-10-07. Never edit active ledger JSON by hand, steal another executor's claim, or expire a claim automatically.

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

- `docs/architecture/backtest/README.md`
- `docs/architecture/backtest/backtest-service-artifact-runtime-v1.ru.md`
- `docs/runbooks/offline-release-installation.md`
- `docs/architecture/backtest/backtest-service-artifact-runtime-v1.md`
- `docs/architecture/api/api-errors-and-422-payload-v1.md`
- `docs/runbooks/indicators-numba-cache-and-threads.md`
- `docs/architecture/project-map/AGENT_GUIDE.md`

Read `reports/S08.md` produced by S08 before claiming. Its absence now is a deferred dependency, not an owner-input question.

## Required work

1. Reconcile the current implementation with every R01–R16 requirement and C01–C14/W1–W4 proof. All remaining implementation or required-proof gaps must be fixed within scope or reported; documentation alone cannot close missing code.

2. Update each named canonical document according to plan section 9. Preserve normative Russian documents and English companion roles. Describe actual ready-set contracts, no-DB preparation, partial manifests, recipe/funding identity, policy defaults, cleanup/resource limits, read-only preflight, lazy replay and timing interpretation.

3. Document reader-before-writer order and the precise safe rollback boundary. Keep retired runbooks retired and do not name an invented production host or deployment command.

4. Regenerate docs/project-map indices through existing tools when required, preserving concurrent foreign index edits. Scope generation output and never overwrite shared files from stale reads.

5. Perform final focused self-review of the owned change and compatibility matrix, incorporating any policy-required independent review evidence already obtained. Verify that no fake slots/hashes, new compatibility adapters/helper framework, duplicated algorithms, temporary benchmark files or unbounded derivative cache were introduced.

6. Complete the final immutable report/receipt and ledger only when all required stage obligations are satisfied. No commit, PR, merge or deployment is implied by local completion. Final user report is Russian with concise implementation results, year timings, proof limits and owned paths.

## Verification and acceptance

- uv run python -m tools.docs.generate_docs_index --check

- uv run python -m tools.docs.generate_project_map --check

- Reconcile required affected gates with the last changed code; rerun only after relevant edits or unresolved concerns. Run pack receipt validation against actual live files and immutable evidence under the supported updater before transition.

All acceptance criteria in the front-matter contract must be evidenced. Local implementation/documentation closure with linked verification and benchmark evidence; no Git publication or production deployment.

## Adjacent-stage handoff

Final stage: next_stage=null and next_stage_allowed=false. Report completion without inventing a further stage.

## Required operational documentation details

Document the exact slot ownership row key, reader/writer admission order, previous-slot cleanup guard, CAS/epoch versus filesystem fencing limits, quarantine on uncertain liveness and confirmed-owner-death recovery. Document two-phase source proof, the canonical payload-prefix format, cheap preflight and prepared-event persistence acknowledgement. Record that upgrading/draining old publisher and cleanup processes is required before enabling the new protocol.

Document pointer/target revalidation under coordinate publication ownership, deterministic multi-slot lock order and the incremental-source reader reservation. Explain why a stale precheck cannot authorize filesystem mutation.

## Journal commands for this stage

Run from the repository root. Choose a unique executor ID for this execution and retain it for resuming the same claim; do not reuse another executor's ID. The commands below use placeholders that the executor substitutes, not text the owner must provide.

```sh
python3 .codex/agents/generated/backtest-npy-input-materialization-v1/ledger_update.py claim --stage S09 --executor UNIQUE_EXECUTOR_ID
```

Only after the claim succeeds, perform the stage. Write the English stage report to `reports/S09.md`, including exact checks, results, owned changes and proof limits. Then, only when all required checks actually passed:

```sh
python3 .codex/agents/generated/backtest-npy-input-materialization-v1/ledger_update.py accept --stage S09 --executor SAME_EXECUTOR_ID --checks-passed --evidence reports/S09.md
```

Evidence paths resolve relative to this pack. Supply additional `--evidence` paths when needed. The report itself may serve as bounded check evidence when it records actual commands/results; a bare pass assertion is insufficient. This command does not run tests or establish their meaning.

Use `resume --stage S09 --executor SAME_EXECUTOR_ID` only for your own existing claim. For an actual new owner decision, `needs-input` requires `--reason`, `--resume-condition` and readable `--evidence`; resume then also requires `--resolution` pointing to the recorded answer. For a concrete hard blocker, `block` requires `--reason` and readable `--evidence`. A hard-blocked instance is terminal and needs a separately authored revision under the canonical lifecycle; do not reset it. Routine implementation problems should be diagnosed and fixed within the current claim, not converted into owner approvals. `status` is read-only. Every failed command leaves the prior ledger intact; an unreferenced immutable receipt after interruption is evidence only and must not be treated as acceptance.
