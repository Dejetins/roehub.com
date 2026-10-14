---
{
  "schema_version": "stage-prompt/v1",
  "prompt_pack_execution": {
    "plan_doc": "implementation-plan.md",
    "prompt_pack_dir": ".",
    "stage_ledger": "iteration-ledger.md"
  },
  "stage_contract": {
    "id": "S08",
    "title": "Full correctness, recovery and browser regression suite",
    "prompt_path": "08-functional-recovery-verification.md",
    "report_path": "reports/S08.md",
    "receipt_dir": "receipts/S08",
    "depends_on": [
      "S07"
    ],
    "expected_touches": [
      "../../../../tests",
      "../../../../tools/qa",
      "../../../../tools/release",
      "../../../../tools/backup",
      "../../../../apps/platform-web/e2e",
      "zone: focused fixes for demonstrated migration defects"
    ],
    "acceptance_criteria": [
      "Real PG semantics, all consumers and supported UI/API journeys pass with CH absent; expected results are not replaced by mocks.",
      "Failure drills prove truthful receipts/cursors/events and bounded recovery after DB restart/disconnect/contention.",
      "Backup/restore, bootstrap, reset invalidation and artifact/backtest parity pass; independent reviewer finds no unresolved material migration issue.",
      "A process crash after committing an old-minute value correction but before publishing its hint is recovered by bounded reconciliation and rebuilds the affected artifact."
    ],
    "proof_boundary": "Functional/failure integration on isolated targets; no long-soak or current-installation delivery claim.",
    "validation": {
      "profile": "prompt-manager/bundled-v1 plus repository boundary-specific gates",
      "checks": [
        "Follow the stage work/acceptance and section6 proof matrix; record actual commands and output paths.",
        "Reread current repository CI routing/package scripts; run only relevant checks, plus changed-boundary real infrastructure/browser/performance evidence.",
        "Run bundled draft/entry and receipt checks for transitions; artifact structure alone is not implementation proof."
      ],
      "requires_user_acceptance": false
    },
    "entry_inputs": [
      {
        "path": "../../../../tests/real_infra/market_data/test_transactional_ingestion_postgres.py",
        "producer_stage": null
      },
      {
        "path": "../../../../tests/real_infra/backtest/test_artifact_dispatch.py",
        "producer_stage": null
      },
      {
        "path": "../../../../apps/migrations/verify_storage_runtime.py",
        "producer_stage": null
      },
      {
        "path": "../../../../tools/release/verify_greenfield_lifecycle.py",
        "producer_stage": null
      },
      {
        "path": "../../../../apps/platform-web/package.json",
        "producer_stage": null
      },
      {
        "path": "implementation-plan.md",
        "producer_stage": null
      },
      {
        "path": "reports/S07.md",
        "producer_stage": "S07"
      }
    ]
  }
}
---

# S08 — Full correctness, recovery and browser regression suite

## Scope and current state

Execute this stage only when the owner selects it or an explicitly selected full-pack run
covers it. Authoring the package did not authorize execution. Read the journal for actual
state/claims; never infer status from this prompt. Read the plan's D1–D7 and relevant
workstream/proof sections. Preserve the accepted PG-only direction and existing provider,
Redis, artifact, API/security and user-settings boundaries. Development market history may
be rebuilt; that is not authority for whole-database/volume deletion or publication.

## Context route

Always read `AGENTS.md`, this prompt, `implementation-plan.md` and `iteration-ledger.md`.
Repository-relative task entrypoints:
- `tests/real_infra/market_data`.
- `tests/real_infra/backtest`.
- `apps/migrations/verify_storage_runtime.py`.
- `tools/release/verify_greenfield_lifecycle.py`.
- `apps/platform-web/package.json`.
Read the predecessor report when declared; it is a deferred output, not an authoring gap.
Use project-map slices for affected callers, then inspect actual source. Refresh dirty-tree
ownership and the initial Appendix inventory; follow caller/config/schema references beyond
literal brand matches. Do not reread unrelated historical packs as current instructions.

## Work

Execute the proof matrix using production adapters and disposable infrastructure. Add meaningful tests for unmapped capabilities; port original tests rather than deleting coverage. Include funding even if unavailable in current local DB, shared-PG audit/recovery failure, duplicate/concurrent ingestion, correction invalidation, resets, old-format rejection and partial migrations. Verify complete fixed-input artifact/backtest results. Test UI history loading/progress, event journal, monitoring and no-history backtest state. Perform one independent review required for destructive migration/security-impacting backup changes; reviewer must not recursively delegate review. Fix owned in-scope defects and rerun affected gates.

Add an end-to-end regression: build an artifact, correct an old candle without changing first/last/count, discard/lose the hint, restart reconciliation, and prove revised artifact/input results. Replay identical data and prove no repeated spurious rebuild.

## Required acceptance

- Real PG semantics, all consumers and supported UI/API journeys pass with CH absent; expected results are not replaced by mocks.
- Failure drills prove truthful receipts/cursors/events and bounded recovery after DB restart/disconnect/contention.
- Backup/restore, bootstrap, reset invalidation and artifact/backtest parity pass; independent reviewer finds no unresolved material migration issue.

- A process crash after committing an old-minute value correction but before publishing its hint is recovered by bounded reconciliation and rebuilds the affected artifact.

## Verification and decision policy

Proof boundary: Functional/failure integration on isolated targets; no long-soak or current-installation delivery claim.
Use section6 and current CI/package routes; retain commands, failures, corrections and
raw sanitized evidence beneath `reports/S08/`. Report unperformed/unavailable checks
honestly; do not mark provider, browser, migration or performance checks passed from mocks.
Do not weaken acceptance silently, invent owner approval or touch unrelated changes.
Resolve routine reversible implementation choices. A real missing material decision or
hard boundary blocker must name the narrow affected work, evidence and resume condition;
continue independent authorized work rather than resetting previous accepted stages.

Skills: `backend-quality-gates` for Python verification; `contract-impact-analysis` when
changing relied-upon DTO/schema/config/backup boundaries; `backend-performance-evidence`
for timing/query/resource claims; `browser-qa-evidence` and the user's chosen browser route
when UI behavior changes; `architecture-review`/`production-risk-review` for the one
policy-required independent review. Load only crossed companions. No publication skill
unless the owner separately grants Git publication, which this prompt does not do.

## Claim, journal and handoff

Before side effects, validate current stage entry with the bundled validator and claim via
`python .codex/agents/generated/postgresql-only-storage-v1/ledger_update.py claim --stage S08 --executor <unique-id>`.
The ID is executor-owned, not an owner approval token. Never steal a claim or edit active
JSON by hand. `status`, `resume`, `accept`, `needs-input` and `block` use that same copied
updater and canonical lifecycle; hard-blocked stages require a revisioned replacement,
not status reset. The updater is bound to this directory; do not use an old pack helper.

Write `reports/S08.md` with owned created/modified/deleted paths, foreign changes excluded,
actual verification results and file links, contracts, data/resource effects, remaining
risk and next-stage inputs. If all required checks pass, accept with `--checks-passed`
and actual `--evidence` files. The updater generates a new immutable receipt and validates
the adjacent prompt before allowing its claim. Do not forge a receipt before evidence.
At the last stage completion, all required obligations must be accepted before ledger
completion. Eligibility of the next row is distinct from continuation authorization.

Final report to the owner is Russian, concise and self-contained, with proof limits and
links. Stop after this stage unless a current explicit full-pack continuation covers the next.
