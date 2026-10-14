---
{
  "schema_version": "stage-prompt/v1",
  "prompt_pack_execution": {
    "plan_doc": "implementation-plan.md",
    "prompt_pack_dir": ".",
    "stage_ledger": "iteration-ledger.md"
  },
  "stage_contract": {
    "id": "S10",
    "title": "Local development cutover, real streams and final acceptance",
    "prompt_path": "10-local-cutover-and-final-audit.md",
    "report_path": "reports/S10.md",
    "receipt_dir": "receipts/S10",
    "depends_on": [
      "S09"
    ],
    "expected_touches": [
      "../../../../tools/installation",
      "../../../../docs/runbooks",
      "zone: identified local installation config and market-only derived state",
      "zone: final migration evidence"
    ],
    "acceptance_criteria": [
      "Current local installation uses PG-only market storage, preserved accounts/settings/secrets/selections, and correct fresh history/artifact generation.",
      "Real selected history advances and consecutive closed candles arrive after bounded disconnect/reconnect; independent SQL/API observations and event logs agree.",
      "No reachable CH dependency/process requirement remains; final docs, setup/backup guidance, gates and known data disposition are complete."
    ],
    "proof_boundary": "Actual identified local installation delivery and real-provider evidence; no remote deployment or Git publication.",
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
        "path": "../../../../tools/installation/local_platform.py",
        "producer_stage": null
      },
      {
        "path": "../../../../docs/runbooks/navigator-local-market-data-recovery.md",
        "producer_stage": null
      },
      {
        "path": "../../../../docs/runbooks/storage-bootstrap-and-migrations.md",
        "producer_stage": null
      },
      {
        "path": "implementation-plan.md",
        "producer_stage": null
      },
      {
        "path": "reports/S09.md",
        "producer_stage": "S09"
      }
    ]
  }
}
---

# S10 — Local development cutover, real streams and final acceptance

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
- `tools/installation/local_platform.py`.
- `docs/runbooks/navigator-local-market-data-recovery.md`.
- `docs/runbooks/storage-bootstrap-and-migrations.md`.
Read the predecessor report when declared; it is a deferred output, not an authoring gap.
Use project-map slices for affected callers, then inspect actual source. Refresh dirty-tree
ownership and the initial Appendix inventory; follow caller/config/schema references beyond
literal brand matches. Do not reread unrelated historical packs as current instructions.

## Work

Execute section5 only after this stage is selected and predecessors accepted. Re-resolve current local runbook/target/volume identities, preserve private settings/accounts and record a concrete market-reset manifest. Quiesce only relevant services; reset history-derived state/generations without deleting the whole PG database. Bootstrap PG-only stack, resync references/funding, refill selected instruments, validate artifacts/backtests and real UI. Observe ≥10 consecutive closed minutes across selected available spot/futures sources and a bounded controlled network reconnect that does not affect unrelated apps. If live exchange access is unavailable, record provider-specific limitation and do not claim live acceptance. Confirm no old workers/CH fallback, final health/resources/log volume, and exact data disposition. Remove only obsolete target CH runtime/config; inert volume deletion is optional. Preserve all benchmark evidence and foreign changes. No commit/push.

## Required acceptance

- Current local installation uses PG-only market storage, preserved accounts/settings/secrets/selections, and correct fresh history/artifact generation.
- Real selected history advances and consecutive closed candles arrive after bounded disconnect/reconnect; independent SQL/API observations and event logs agree.
- No reachable CH dependency/process requirement remains; final docs, setup/backup guidance, gates and known data disposition are complete.

## Verification and decision policy

Proof boundary: Actual identified local installation delivery and real-provider evidence; no remote deployment or Git publication.
Use section6 and current CI/package routes; retain commands, failures, corrections and
raw sanitized evidence beneath `reports/S10/`. Report unperformed/unavailable checks
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
`python .codex/agents/generated/postgresql-only-storage-v1/ledger_update.py claim --stage S10 --executor <unique-id>`.
The ID is executor-owned, not an owner approval token. Never steal a claim or edit active
JSON by hand. `status`, `resume`, `accept`, `needs-input` and `block` use that same copied
updater and canonical lifecycle; hard-blocked stages require a revisioned replacement,
not status reset. The updater is bound to this directory; do not use an old pack helper.

Write `reports/S10.md` with owned created/modified/deleted paths, foreign changes excluded,
actual verification results and file links, contracts, data/resource effects, remaining
risk and next-stage inputs. If all required checks pass, accept with `--checks-passed`
and actual `--evidence` files. The updater generates a new immutable receipt and validates
the adjacent prompt before allowing its claim. Do not forge a receipt before evidence.
At the last stage completion, all required obligations must be accepted before ledger
completion. Eligibility of the next row is distinct from continuation authorization.

Final report to the owner is Russian, concise and self-contained, with proof limits and
links. Stop after this stage unless a current explicit full-pack continuation covers the next.
