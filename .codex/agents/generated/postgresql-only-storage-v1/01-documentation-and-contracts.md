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
    "title": "Documentation, complete inventory and target contracts",
    "prompt_path": "01-documentation-and-contracts.md",
    "report_path": "reports/S01.md",
    "receipt_dir": "receipts/S01",
    "depends_on": [],
    "expected_touches": [
      "../../../../docs/architecture/market_data",
      "../../../../docs/architecture/platform",
      "../../../../docs/architecture/backtest",
      "../../../../docs/architecture/strategy",
      "../../../../docs/runbooks",
      "zone: all canonical documents and dependency inventory"
    ],
    "acceptance_criteria": [
      "Every inventory hit has a disposition and a migration owner; callers without literal CH strings are followed.",
      "Normative documentation distinguishes PG target from current CH runtime and covers D1\u2013D7; historical evidence is preserved.",
      "Query catalogue, resource guardrails, reset manifest shape, versioned contract changes and full stage proof map are recorded."
    ],
    "proof_boundary": "Read-only source audit plus normative documentation amendments; no runtime/schema activation.",
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
        "path": "../../../../AGENTS.md",
        "producer_stage": null
      },
      {
        "path": "../../../../docs/architecture/README.md",
        "producer_stage": null
      },
      {
        "path": "../../../../docs/architecture/project-map/AGENT_GUIDE.md",
        "producer_stage": null
      },
      {
        "path": "../../../../docs/architecture/market_data/market-data-application-ports.md",
        "producer_stage": null
      },
      {
        "path": "../../../../docs/runbooks/storage-bootstrap-and-migrations.md",
        "producer_stage": null
      },
      {
        "path": "implementation-plan.md",
        "producer_stage": null
      }
    ]
  }
}
---

# S01 — Documentation, complete inventory and target contracts

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
- `AGENTS.md`.
- `docs/architecture/README.md`.
- `docs/architecture/project-map/AGENT_GUIDE.md`.
- `docs/architecture/market_data/market-data-application-ports.md`.
- `docs/runbooks/storage-bootstrap-and-migrations.md`.
Read the predecessor report when declared; it is a deferred output, not an authoring gap.
Use project-map slices for affected callers, then inspect actual source. Refresh dirty-tree
ownership and the initial Appendix inventory; follow caller/config/schema references beyond
literal brand matches. Do not reread unrelated historical packs as current instructions.

## Work

Refresh the Appendix inventory against the dirty tree; classify active code, generated outputs, schemas, optional scripts, normative docs, immutable historical evidence and narrow rejection fixtures. Trace provider-specific raw/funding fields and every port/caller, including RL notebooks/scripts and backup owners. Amend current docs with clearly labeled target sections and explicit supersession, preserving historical reports. Record a complete query catalogue and baseline signatures, proposed guardrails, fixed tests and reset/invalidation ownership. Resolve reversible SQL choices from the plan; missing future implementation output is not an owner question. No runtime changes in this stage.

## Required acceptance

- Every inventory hit has a disposition and a migration owner; callers without literal CH strings are followed.
- Normative documentation distinguishes PG target from current CH runtime and covers D1–D7; historical evidence is preserved.
- Query catalogue, resource guardrails, reset manifest shape, versioned contract changes and full stage proof map are recorded.

## Verification and decision policy

Proof boundary: Read-only source audit plus normative documentation amendments; no runtime/schema activation.
Use section6 and current CI/package routes; retain commands, failures, corrections and
raw sanitized evidence beneath `reports/S01/`. Report unperformed/unavailable checks
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
`python .codex/agents/generated/postgresql-only-storage-v1/ledger_update.py claim --stage S01 --executor <unique-id>`.
The ID is executor-owned, not an owner approval token. Never steal a claim or edit active
JSON by hand. `status`, `resume`, `accept`, `needs-input` and `block` use that same copied
updater and canonical lifecycle; hard-blocked stages require a revisioned replacement,
not status reset. The updater is bound to this directory; do not use an old pack helper.

Write `reports/S01.md` with owned created/modified/deleted paths, foreign changes excluded,
actual verification results and file links, contracts, data/resource effects, remaining
risk and next-stage inputs. If all required checks pass, accept with `--checks-passed`
and actual `--evidence` files. The updater generates a new immutable receipt and validates
the adjacent prompt before allowing its claim. Do not forge a receipt before evidence.
At the last stage completion, all required obligations must be accepted before ledger
completion. Eligibility of the next row is distinct from continuation authorization.

Final report to the owner is Russian, concise and self-contained, with proof limits and
links. Stop after this stage unless a current explicit full-pack continuation covers the next.
