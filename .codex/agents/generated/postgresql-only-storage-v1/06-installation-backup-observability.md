---
{
  "schema_version": "stage-prompt/v1",
  "prompt_pack_execution": {
    "plan_doc": "implementation-plan.md",
    "prompt_pack_dir": ".",
    "stage_ledger": "iteration-ledger.md"
  },
  "stage_contract": {
    "id": "S06",
    "title": "PostgreSQL-only installation, backup and operations",
    "prompt_path": "06-installation-backup-observability.md",
    "report_path": "reports/S06.md",
    "receipt_dir": "receipts/S06",
    "depends_on": [
      "S05"
    ],
    "expected_touches": [
      "../../../../src/trading/platform/config",
      "../../../../src/trading/contexts/operations",
      "../../../../apps/migrations",
      "../../../../apps/control_agent",
      "../../../../apps/monitoring",
      "../../../../tools/installation",
      "../../../../tools/release",
      "../../../../tools/backup",
      "../../../../configs",
      "../../../../infra",
      "../../../../schemas",
      "../../../../docs/runbooks"
    ],
    "acceptance_criteria": [
      "base/trading/ml and embedded/external PG profiles generate/bootstrap/run without CH service, image, secret, migration or health requirement.",
      "Config/storage-status/backup owner schemas have explicit compatible version handling; stale/old incompatible inputs fail before side effects.",
      "New-format backup/restore and runtime operations preserve digest/quiesce/security invariants; runbooks and monitoring are PG-only."
    ],
    "proof_boundary": "Generated config/schema and disposable installation/backup runtime proof; no release publication or current local reset.",
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
        "path": "../../../../src/trading/platform/config/installation.py",
        "producer_stage": null
      },
      {
        "path": "../../../../src/trading/platform/config/runtime_topology.py",
        "producer_stage": null
      },
      {
        "path": "../../../../src/trading/contexts/operations/backup_contracts.py",
        "producer_stage": null
      },
      {
        "path": "../../../../tools/installation/local_platform.py",
        "producer_stage": null
      },
      {
        "path": "../../../../tools/release/offline_bundle.py",
        "producer_stage": null
      },
      {
        "path": "../../../../docs/runbooks/installation-backup-restore-upgrade.md",
        "producer_stage": null
      },
      {
        "path": "implementation-plan.md",
        "producer_stage": null
      },
      {
        "path": "reports/S05.md",
        "producer_stage": "S05"
      }
    ]
  }
}
---

# S06 — PostgreSQL-only installation, backup and operations

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
- `src/trading/platform/config/installation.py`.
- `src/trading/platform/config/runtime_topology.py`.
- `src/trading/contexts/operations/backup_contracts.py`.
- `tools/installation/local_platform.py`.
- `tools/release/offline_bundle.py`.
- `docs/runbooks/installation-backup-restore-upgrade.md`.
Read the predecessor report when declared; it is a deferred output, not an authoring gap.
Use project-map slices for affected callers, then inspect actual source. Refresh dirty-tree
ownership and the initial Appendix inventory; follow caller/config/schema references beyond
literal brand matches. Do not reread unrelated historical packs as current instructions.

## Work

Remove CH from source manifests, defaults, compose generators, container builds, entrypoints, DSN resolution, OpenBao slots, capability/status/bootstrap checks and control actions. Version all affected backup/capture/policy schemas instead of silently reducing required owners. Port current verifiers and failure fixtures. Retire CH exporter/query settings/maintenance commands and create meaningful PG operational signals in current topology. Regenerate, never hand-edit generated profiles, schemas, runbooks, SBOM/metadata as applicable. Verify all three profiles and external PG, offline bundle validation and backup/restore on disposable environments; do not publish artifacts.

## Required acceptance

- base/trading/ml and embedded/external PG profiles generate/bootstrap/run without CH service, image, secret, migration or health requirement.
- Config/storage-status/backup owner schemas have explicit compatible version handling; stale/old incompatible inputs fail before side effects.
- New-format backup/restore and runtime operations preserve digest/quiesce/security invariants; runbooks and monitoring are PG-only.

## Verification and decision policy

Proof boundary: Generated config/schema and disposable installation/backup runtime proof; no release publication or current local reset.
Use section6 and current CI/package routes; retain commands, failures, corrections and
raw sanitized evidence beneath `reports/S06/`. Report unperformed/unavailable checks
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
`python .codex/agents/generated/postgresql-only-storage-v1/ledger_update.py claim --stage S06 --executor <unique-id>`.
The ID is executor-owned, not an owner approval token. Never steal a claim or edit active
JSON by hand. `status`, `resume`, `accept`, `needs-input` and `block` use that same copied
updater and canonical lifecycle; hard-blocked stages require a revisioned replacement,
not status reset. The updater is bound to this directory; do not use an old pack helper.

Write `reports/S06.md` with owned created/modified/deleted paths, foreign changes excluded,
actual verification results and file links, contracts, data/resource effects, remaining
risk and next-stage inputs. If all required checks pass, accept with `--checks-passed`
and actual `--evidence` files. The updater generates a new immutable receipt and validates
the adjacent prompt before allowing its claim. Do not forge a receipt before evidence.
At the last stage completion, all required obligations must be accepted before ledger
completion. Eligibility of the next row is distinct from continuation authorization.

Final report to the owner is Russian, concise and self-contained, with proof limits and
links. Stop after this stage unless a current explicit full-pack continuation covers the next.
