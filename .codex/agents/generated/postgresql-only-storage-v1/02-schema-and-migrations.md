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
    "title": "PostgreSQL market schema and lifecycle",
    "prompt_path": "02-schema-and-migrations.md",
    "report_path": "reports/S02.md",
    "receipt_dir": "receipts/S02",
    "depends_on": [
      "S01"
    ],
    "expected_touches": [
      "../../../../migrations/postgres",
      "../../../../apps/migrations",
      "../../../../src/trading/contexts/market_data/adapters/outbound/persistence/postgres",
      "../../../../tests/unit/apps/migrations",
      "../../../../tests/real_infra/market_data"
    ],
    "acceptance_criteria": [
      "Additive migrations create every D2 table/constraint/index and typed raw/funding fields; existing migration checksums remain intact.",
      "Clean/resumed/repeated PG bootstrap and historical/month-rollover partition provisioning pass on a disposable instance.",
      "The app uses bounded non-superuser capabilities; partitions/coverage/frontier generations and invalidation APIs are explicit.",
      "Schema includes durable generation-bound content revision and persisted artifact source-target representation; revision survives restart and is not inferred from bounds alone."
    ],
    "proof_boundary": "Real PG schema/migration/partition tests; this is not production cutover or query-performance proof.",
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
        "path": "../../../../migrations/postgres/manifest.json",
        "producer_stage": null
      },
      {
        "path": "../../../../migrations/clickhouse/manifest.json",
        "producer_stage": null
      },
      {
        "path": "../../../../apps/migrations/storage.py",
        "producer_stage": null
      },
      {
        "path": "../../../../src/trading/contexts/market_data/adapters/outbound/persistence/postgres/gateway.py",
        "producer_stage": null
      },
      {
        "path": "implementation-plan.md",
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

# S02 — PostgreSQL market schema and lifecycle

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
- `migrations/postgres/manifest.json`.
- `migrations/clickhouse/manifest.json`.
- `apps/migrations/storage.py`.
- `src/trading/contexts/market_data/adapters/outbound/persistence/postgres/gateway.py`.
Read the predecessor report when declared; it is a deferred output, not an authoring gap.
Use project-map slices for affected callers, then inspect actual source. Refresh dirty-tree
ownership and the initial Appendix inventory; follow caller/config/schema references beyond
literal brand matches. Do not reread unrelated historical packs as current instructions.

## Work

Implement D2 with PostgreSQL16 native partitioning and constraints, all funding and raw field mappings, exact UTC types and generation metadata. Integrate additive migrations with manifest/bootstrap/status; reserve no guessed migration number. Add serialized bounded partition maintenance with current/next and requested historical month creation. Keep old installation bootable until wiring cutover; do not delete CH infrastructure yet. Preserve existing public control/identity tables. Prove migration restart/checksum/privilege/partition behavior on labeled disposable storage.

## Required acceptance

- Additive migrations create every D2 table/constraint/index and typed raw/funding fields; existing migration checksums remain intact.
- Clean/resumed/repeated PG bootstrap and historical/month-rollover partition provisioning pass on a disposable instance.
- The app uses bounded non-superuser capabilities; partitions/coverage/frontier generations and invalidation APIs are explicit.

- Schema includes durable generation-bound content revision and persisted artifact source-target representation; revision survives restart and is not inferred from bounds alone.

## Verification and decision policy

Proof boundary: Real PG schema/migration/partition tests; this is not production cutover or query-performance proof.
Use section6 and current CI/package routes; retain commands, failures, corrections and
raw sanitized evidence beneath `reports/S02/`. Report unperformed/unavailable checks
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
`python .codex/agents/generated/postgresql-only-storage-v1/ledger_update.py claim --stage S02 --executor <unique-id>`.
The ID is executor-owned, not an owner approval token. Never steal a claim or edit active
JSON by hand. `status`, `resume`, `accept`, `needs-input` and `block` use that same copied
updater and canonical lifecycle; hard-blocked stages require a revisioned replacement,
not status reset. The updater is bound to this directory; do not use an old pack helper.

Write `reports/S02.md` with owned created/modified/deleted paths, foreign changes excluded,
actual verification results and file links, contracts, data/resource effects, remaining
risk and next-stage inputs. If all required checks pass, accept with `--checks-passed`
and actual `--evidence` files. The updater generates a new immutable receipt and validates
the adjacent prompt before allowing its claim. Do not forge a receipt before evidence.
At the last stage completion, all required obligations must be accepted before ledger
completion. Eligibility of the next row is distinct from continuation authorization.

Final report to the owner is Russian, concise and self-contained, with proof limits and
links. Stop after this stage unless a current explicit full-pack continuation covers the next.
