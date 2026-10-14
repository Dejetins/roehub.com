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
    "title": "Atomic batch ingestion and bounded PostgreSQL access",
    "prompt_path": "03-transactional-ingestion.md",
    "report_path": "reports/S03.md",
    "receipt_dir": "receipts/S03",
    "depends_on": [
      "S02"
    ],
    "expected_touches": [
      "../../../../src/trading/contexts/market_data/adapters/outbound/persistence/postgres",
      "../../../../src/trading/contexts/market_data/application/ports/stores",
      "../../../../src/trading/contexts/market_data/application/services",
      "../../../../apps/worker/market_data_ws/wiring",
      "../../../../tests/real_infra/market_data"
    ],
    "acceptance_criteria": [
      "COPY staging, raw persistence, deterministic canonical/funding upsert and coverage deltas commit atomically with truthful receipt semantics.",
      "Duplicate-in-batch, retry, equal/stale version and concurrent WS/REST tests preserve exact canonical/index truth.",
      "Connection/transaction lifetime, finite pools/admission and SQLSTATE retry behavior are bounded; existing recovery batching and provider limits survive.",
      "Canonical value changes atomically advance a durable generation-bound content revision; identical retries do not; commit-before-hint crash cannot hide a correction."
    ],
    "proof_boundary": "Transactional real-DB correctness and concurrency/failure receipts, not provider uptime or full-app migration.",
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
        "path": "../../../../src/trading/contexts/market_data/adapters/outbound/persistence/clickhouse/raw_kline_writer.py",
        "producer_stage": null
      },
      {
        "path": "../../../../src/trading/contexts/market_data/adapters/outbound/persistence/clickhouse/funding_rate_store.py",
        "producer_stage": null
      },
      {
        "path": "../../../../apps/worker/market_data_ws/wiring/persistent_writer.py",
        "producer_stage": null
      },
      {
        "path": "../../../../src/trading/contexts/market_data/adapters/outbound/persistence/postgres/async_stream_recovery_store.py",
        "producer_stage": null
      },
      {
        "path": "implementation-plan.md",
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

# S03 — Atomic batch ingestion and bounded PostgreSQL access

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
- `src/trading/contexts/market_data/adapters/outbound/persistence/clickhouse/raw_kline_writer.py`.
- `src/trading/contexts/market_data/adapters/outbound/persistence/clickhouse/funding_rate_store.py`.
- `apps/worker/market_data_ws/wiring/persistent_writer.py`.
- `src/trading/contexts/market_data/adapters/outbound/persistence/postgres/async_stream_recovery_store.py`.
Read the predecessor report when declared; it is a deferred output, not an authoring gap.
Use project-map slices for affected callers, then inspect actual source. Refresh dirty-tree
ownership and the initial Appendix inventory; follow caller/config/schema references beyond
literal brand matches. Do not reread unrelated historical packs as current instructions.

## Work

Implement D3 and PG ref/funding writers behind current ports. Use bounded COPY batches and staging key deduplication; never construct one giant historical VALUES statement. Preserve provider fields, deterministic version precedence, actual-mask/summary count consistency, and correction-driven artifact invalidation. Introduce only the required transaction/pool seam and explicit overall budget accounting. Verify same-key races, deadlocks/retry boundaries, commit ambiguity and no Redis/recovery acknowledgement before durable write. Keep old adapters available only for not-yet-switched consumers; no final dual-write feature.

Persist the D3 generation-bound content revision in the same canonical transaction. Test a same-minute OHLCV correction with unchanged bounds/count and a crash before its hint; identical data replay must not cause perpetual invalidation.

## Required acceptance

- COPY staging, raw persistence, deterministic canonical/funding upsert and coverage deltas commit atomically with truthful receipt semantics.
- Duplicate-in-batch, retry, equal/stale version and concurrent WS/REST tests preserve exact canonical/index truth.
- Connection/transaction lifetime, finite pools/admission and SQLSTATE retry behavior are bounded; existing recovery batching and provider limits survive.

- Canonical value changes atomically advance a durable generation-bound content revision; identical retries do not; commit-before-hint crash cannot hide a correction.

## Verification and decision policy

Proof boundary: Transactional real-DB correctness and concurrency/failure receipts, not provider uptime or full-app migration.
Use section6 and current CI/package routes; retain commands, failures, corrections and
raw sanitized evidence beneath `reports/S03/`. Report unperformed/unavailable checks
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
`python .codex/agents/generated/postgresql-only-storage-v1/ledger_update.py claim --stage S03 --executor <unique-id>`.
The ID is executor-owned, not an owner approval token. Never steal a claim or edit active
JSON by hand. `status`, `resume`, `accept`, `needs-input` and `block` use that same copied
updater and canonical lifecycle; hard-blocked stages require a revisioned replacement,
not status reset. The updater is bound to this directory; do not use an old pack helper.

Write `reports/S03.md` with owned created/modified/deleted paths, foreign changes excluded,
actual verification results and file links, contracts, data/resource effects, remaining
risk and next-stage inputs. If all required checks pass, accept with `--checks-passed`
and actual `--evidence` files. The updater generates a new immutable receipt and validates
the adjacent prompt before allowing its claim. Do not forge a receipt before evidence.
At the last stage completion, all required obligations must be accepted before ledger
completion. Eligibility of the next row is distinct from continuation authorization.

Final report to the owner is Russian, concise and self-contained, with proof limits and
links. Stop after this stage unless a current explicit full-pack continuation covers the next.
