---
{
  "schema_version": "stage-prompt/v1",
  "prompt_pack_execution": {
    "plan_doc": "implementation-plan.md",
    "prompt_pack_dir": ".",
    "stage_ledger": "iteration-ledger.md"
  },
  "stage_contract": {
    "id": "S04",
    "title": "Optimized readers, coverage index and verified history frontier",
    "prompt_path": "04-queries-coverage-frontier.md",
    "report_path": "reports/S04.md",
    "receipt_dir": "receipts/S04",
    "depends_on": [
      "S03"
    ],
    "expected_touches": [
      "../../../../src/trading/contexts/market_data/adapters/outbound/persistence/postgres",
      "../../../../src/trading/contexts/backtest_artifacts/adapters/outbound/persistence/postgres",
      "../../../../src/trading/contexts/market_data/application/services",
      "../../../../tests/real_infra/market_data",
      "../../../../tests/unit/contexts/backtest_artifacts"
    ],
    "acceptance_criteria": [
      "Every CH reader/verifier/coordinate capability has an equivalent PG implementation and contract tests, including funding/readiness.",
      "All D5 query families have query-plan/row-buffer evidence and result parity; bulk readers bound client memory.",
      "Actual/source-absent/unknown states and exclusive verified frontier remain correct through failed provider reads, leases, late candles, deletion and rebuild.",
      "Bounded source observation/reconciliation carries durable content revision and discovers corrected old candles even when first/last/count are unchanged and hints are lost."
    ],
    "proof_boundary": "Real PG query semantics and plans on representative ranges; final full mixed performance is S09.",
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
        "path": "../../../../src/trading/contexts/market_data/adapters/outbound/persistence/clickhouse/canonical_candle_index_reader.py",
        "producer_stage": null
      },
      {
        "path": "../../../../src/trading/contexts/market_data/adapters/outbound/persistence/clickhouse/history_coverage_reader.py",
        "producer_stage": null
      },
      {
        "path": "../../../../src/trading/contexts/market_data/adapters/outbound/persistence/clickhouse/history_range_verifier.py",
        "producer_stage": null
      },
      {
        "path": "../../../../src/trading/contexts/backtest_artifacts/adapters/outbound/persistence/clickhouse/stored_coordinates.py",
        "producer_stage": null
      },
      {
        "path": "implementation-plan.md",
        "producer_stage": null
      },
      {
        "path": "reports/S03.md",
        "producer_stage": "S03"
      }
    ]
  }
}
---

# S04 — Optimized readers, coverage index and verified history frontier

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
- `src/trading/contexts/market_data/adapters/outbound/persistence/clickhouse/canonical_candle_index_reader.py`.
- `src/trading/contexts/market_data/adapters/outbound/persistence/clickhouse/history_coverage_reader.py`.
- `src/trading/contexts/market_data/adapters/outbound/persistence/clickhouse/history_range_verifier.py`.
- `src/trading/contexts/backtest_artifacts/adapters/outbound/persistence/clickhouse/stored_coordinates.py`.
Read the predecessor report when declared; it is a deferred output, not an authoring gap.
Use project-map slices for affected callers, then inspect actual source. Refresh dirty-tree
ownership and the initial Appendix inventory; follow caller/config/schema references beyond
literal brand matches. Do not reread unrelated historical packs as current instructions.

## Work

Implement D4–D5 completely: canonical DTO/array readers, all index methods, batched catalog/instrument coverage and references/search, funding/universe/readiness, exact artifact observations, bounded verifier, prefix generation and invalidation. Use actual masks only for source bounds. Stream historical data with bounded conversion buffers; do not mirror prototype fetchall. Query plans must show pruning/index scaling for large tables; small summary scans can be justified. Test partial days, internal gaps, no-data/day-missing, exclusive cutoff inside minute, epoch zero, absent→actual and stale lease/generation races.

Extend StoredSourceObservation and persisted source-target comparison with generation/content revision from D3. Prove bounded sweep detection after a lost correction hint without canonical-history scans; do not retain content_unverified as the only change detector.

## Required acceptance

- Every CH reader/verifier/coordinate capability has an equivalent PG implementation and contract tests, including funding/readiness.
- All D5 query families have query-plan/row-buffer evidence and result parity; bulk readers bound client memory.
- Actual/source-absent/unknown states and exclusive verified frontier remain correct through failed provider reads, leases, late candles, deletion and rebuild.

- Bounded source observation/reconciliation carries durable content revision and discovers corrected old candles even when first/last/count are unchanged and hints are lost.

## Verification and decision policy

Proof boundary: Real PG query semantics and plans on representative ranges; final full mixed performance is S09.
Use section6 and current CI/package routes; retain commands, failures, corrections and
raw sanitized evidence beneath `reports/S04/`. Report unperformed/unavailable checks
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
`python .codex/agents/generated/postgresql-only-storage-v1/ledger_update.py claim --stage S04 --executor <unique-id>`.
The ID is executor-owned, not an owner approval token. Never steal a claim or edit active
JSON by hand. `status`, `resume`, `accept`, `needs-input` and `block` use that same copied
updater and canonical lifecycle; hard-blocked stages require a revisioned replacement,
not status reset. The updater is bound to this directory; do not use an old pack helper.

Write `reports/S04.md` with owned created/modified/deleted paths, foreign changes excluded,
actual verification results and file links, contracts, data/resource effects, remaining
risk and next-stage inputs. If all required checks pass, accept with `--checks-passed`
and actual `--evidence` files. The updater generates a new immutable receipt and validates
the adjacent prompt before allowing its claim. Do not forge a receipt before evidence.
At the last stage completion, all required obligations must be accepted before ledger
completion. Eligibility of the next row is distinct from continuation authorization.

Final report to the owner is Russian, concise and self-contained, with proof limits and
links. Stop after this stage unless a current explicit full-pack continuation covers the next.
