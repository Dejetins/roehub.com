---
{
  "schema_version": "stage-prompt/v1",
  "prompt_pack_execution": {
    "plan_doc": "implementation-plan.md",
    "prompt_pack_dir": ".",
    "stage_ledger": "iteration-ledger.md"
  },
  "stage_contract": {
    "id": "S07",
    "title": "Complete correctness, integration, migration and failure-mode evidence",
    "prompt_path": "07-production-correctness-proof.md",
    "report_path": "reports/S07.md",
    "receipt_dir": "receipts/S07",
    "depends_on": [
      "S06"
    ],
    "expected_touches": [
      "../../../../tests/unit/contexts/backtest",
      "../../../../tests/unit/contexts/backtest_artifacts",
      "../../../../tests/unit/apps/worker/backtest_job_runner",
      "../../../../tests/unit/apps/api",
      "../../../../tests/unit/apps/migrations",
      "../../../../tests/real_infra/backtest",
      "zone: Narrow in-scope production fixes revealed by required proof"
    ],
    "acceptance_criteria": [
      "Every C01–C14 row has actual sufficient evidence or a named unresolved blocker; unresolved required proof prevents stage acceptance.",
      "Native/generated/mixed numerical and result identities agree against an independent oracle; no correctness tolerance was relaxed.",
      "Migration, leases, resource bounds, cleanup and organization/path protections are verified at their actual affected boundaries.",
      "Required gates and applicable independent review are satisfied; foreign work and production systems remain untouched.",
      "Forced ownership races cover both rebuild and previous-slot cleanup, live expired owners, and acknowledged prefix proof before scoring."
    ],
    "proof_boundary": "Local production-boundary correctness and operational safety under tested fixtures; throughput/latency and deployed readiness are separate.",
    "validation": {
      "profile": "roehub-focused-gates-and-prompt-pack-artifacts/v1",
      "checks": [
        "Run focused pytest targets selected in S01–S06 plus the new deterministic/integration cases; record exact commands and counts, including skips.",
        "uv run ruff check <owned-python-paths>; uv run pyright; uv run python -m tools.ci.route_changes ci --changed-files <external-owned-path-list>, then the required affected CI shards.",
        "Run isolated real-infra fixtures under the repository setup contract. A zero-test/all-skipped success is not proof. Preserve failures and classify attribution from evidence."
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
        "path": "../../../../.github/workflows/ci.yml",
        "producer_stage": null
      },
      {
        "path": "../../../../tools/ci/route_changes.py",
        "producer_stage": null
      },
      {
        "path": "reports/S06.md",
        "producer_stage": "S06"
      }
    ]
  }
}
---

# S07 — Complete correctness, integration, migration and failure-mode evidence

Relevant implementation-plan sections: 5, 6 C01–C14, 9 validation/review policy.

## Execution authority and state

Read the live ledger before doing anything. It alone owns statuses, claims and decision packets. Execute exactly S07 when the owner submits this prompt, then stop with the next prompt link. The owner sends prompts sequentially; no separate stage acceptance is required. Follow `staged-plan-runner` for lifecycle rules and use this pack's `ledger_update.py` for every execution-state mutation. The former updater gap was resolved by explicit owner instruction on 2026-10-07. Never edit active ledger JSON by hand, steal another executor's claim, or expire a claim automatically.

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
- `.github/workflows/ci.yml`
- `tools/ci/route_changes.py`
- `tests/unit/contexts/backtest/application/services/v2/artifact_testkit_v2.py`
- `tests/unit/apps/worker/backtest_job_runner`
- `tests/real_infra`
- `apps/platform-web/package.json`

Read `reports/S06.md` produced by S06 before claiming. Its absence now is a deferred dependency, not an owner-input question.

## Required work

1. Build a requirement-to-evidence matrix for C01–C14 and R01–R16 using actual outcomes from earlier stages. Fill missing proof with meaningful tests, not tests mirroring implementation. Production computation and real file/DB boundaries must not be replaced by mocks for acceptance.

2. Run real subprocess and disposable Postgres migration, lease/pin race, cancellation, crash, restart, idempotency and source-prefix replay cases. Include 200-coordinate policy inventory without implying 200-symbol execution throughput.

3. Exercise spot plus a deterministic futures funding fixture, long/short/reversal where supported, TP-only/SL-only/both/none, gap policy, canonical sparse rows, sentinel edges and all existing error/precision rules.

4. Run API create/status/complete-top/detail/series contracts and existing accepted platform-client parsing. Browser-visible changes require real browser evidence using browser-qa-evidence and the current browser tool onboarding; screenshots alone are not runtime proof.

5. Use current CI change routing for owned paths and run required backend/type/migration/config gates. Repair authorized in-scope regressions and rerun only affected proof. Do not silently fix foreign changes.

6. Perform focused self-review. Obtain one independent review if actual changes cross a security boundary, irreversible migration, release or material unresolved risk per repository policy. Report an unavailable required review; do not substitute an approval question for completed verification.

## Verification and acceptance

- Run focused pytest targets selected in S01–S06 plus the new deterministic/integration cases; record exact commands and counts, including skips.

- uv run ruff check <owned-python-paths>; uv run pyright; uv run python -m tools.ci.route_changes ci --changed-files <external-owned-path-list>, then the required affected CI shards.

- Run isolated real-infra fixtures under the repository setup contract. A zero-test/all-skipped success is not proof. Preserve failures and classify attribution from evidence.

All acceptance criteria in the front-matter contract must be evidenced. Local production-boundary correctness and operational safety under tested fixtures; throughput/latency and deployed readiness are separate.

## Adjacent-stage handoff

S08 starts only with correct comparable paths and trustworthy source/financial fixtures.

## Review findings now encoded as proof obligations

C08 must force both rebuild and previous-slot deletion races, including a new reader after the legacy count, an expired lease whose child is still alive and DB session loss. C10 must change npy header/shape through append, separately mutate the consumed prefix, and verify that preflight performs no payload scan. Assert durable prepared-event acknowledgement before scoring and stale-owner rejection. Inspect the actual gateway transaction scope; multiple independent SQL calls are not an atomic ownership operation.

Include C08 with two publishers using stale prechecks: the second cannot rebuild/delete a newly active slot, incremental source reads retain reader ownership, and final pointer comparison rejects changed expected identity.

## Journal commands for this stage

Run from the repository root. Choose a unique executor ID for this execution and retain it for resuming the same claim; do not reuse another executor's ID. The commands below use placeholders that the executor substitutes, not text the owner must provide.

```sh
python3 .codex/agents/generated/backtest-npy-input-materialization-v1/ledger_update.py claim --stage S07 --executor UNIQUE_EXECUTOR_ID
```

Only after the claim succeeds, perform the stage. Write the English stage report to `reports/S07.md`, including exact checks, results, owned changes and proof limits. Then, only when all required checks actually passed:

```sh
python3 .codex/agents/generated/backtest-npy-input-materialization-v1/ledger_update.py accept --stage S07 --executor SAME_EXECUTOR_ID --checks-passed --evidence reports/S07.md
```

Evidence paths resolve relative to this pack. Supply additional `--evidence` paths when needed. The report itself may serve as bounded check evidence when it records actual commands/results; a bare pass assertion is insufficient. This command does not run tests or establish their meaning.

Use `resume --stage S07 --executor SAME_EXECUTOR_ID` only for your own existing claim. For an actual new owner decision, `needs-input` requires `--reason`, `--resume-condition` and readable `--evidence`; resume then also requires `--resolution` pointing to the recorded answer. For a concrete hard blocker, `block` requires `--reason` and readable `--evidence`. A hard-blocked instance is terminal and needs a separately authored revision under the canonical lifecycle; do not reset it. Routine implementation problems should be diagnosed and fixed within the current claim, not converted into owner approvals. `status` is read-only. Every failed command leaves the prior ledger intact; an unreferenced immutable receipt after interruption is evidence only and must not be treated as acceptance.
