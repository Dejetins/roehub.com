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
    "title": "Production-query performance, memory and long mixed-load proof",
    "prompt_path": "09-performance-and-soak.md",
    "report_path": "reports/S09.md",
    "receipt_dir": "receipts/S09",
    "depends_on": [
      "S08"
    ],
    "expected_touches": [
      "../../../../tools/qa",
      "../../../../tests/perf_smoke",
      "zone: SQL and resource tuning within migrated adapters and configuration"
    ],
    "acceptance_criteria": [
      "Full-size production-adapter mixed runs meet documented S01 guardrails or have explicit unresolved blockers; no tiny-data substitute.",
      "Three repeated \u22655-minute normal/stress runs and \u226560-minute soak include existing PG application control traffic and \u226530 large-read samples per regime.",
      "Plans, all result hashes/receipts, row counts, CPU, DB+client memory, pool waits, WAL/checkpoints/autovacuum, disk and failure counts are retained with reproducible inputs."
    ],
    "proof_boundary": "Measured capacity for the stated hardware/dataset/mix only; not universal capacity or provider uptime.",
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
        "path": "../../../../tools/qa/postgres_storage_prototype.py",
        "producer_stage": null
      },
      {
        "path": "../../../../tests/perf_smoke/contexts/indicators/test_compute_numba_perf_smoke.py",
        "producer_stage": null
      },
      {
        "path": "../../../../src/trading/contexts/market_data/adapters/outbound/persistence/postgres/canonical_candle_reader.py",
        "producer_stage": null
      },
      {
        "path": "implementation-plan.md",
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

# S09 — Production-query performance, memory and long mixed-load proof

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
- `tools/qa`.
- `tests/perf_smoke`.
- `src/trading/contexts/market_data/adapters/outbound/persistence/postgres`.
Read the predecessor report when declared; it is a deferred output, not an authoring gap.
Use project-map slices for affected callers, then inspect actual source. Refresh dirty-tree
ownership and the initial Appendix inventory; follow caller/config/schema references beyond
literal brand matches. Do not reread unrelated historical packs as current instructions.

## Work

Use backend-performance-evidence. Recreate the full frozen real dataset or larger equivalent in a disposable PG target; use actual application adapters including auth/job/recovery load on the same server. Record full measurement contract, client RSS and disk/WAL separately. Run repeated normal/stress and long soak from section6, cancellation/recovery/pool saturation, and query plans after representative analyze. Use bounded exports to artifact arrays and verify exact contents. Profile and correct SQL/locks/batch/pool defects; do not hide failures by dropping samples or reducing data. Resource increases are allowed, but comparative baselines and new envelope must be honest. Report disk growth/retention limits and residual scalability risk.

## Required acceptance

- Full-size production-adapter mixed runs meet documented S01 guardrails or have explicit unresolved blockers; no tiny-data substitute.
- Three repeated ≥5-minute normal/stress runs and ≥60-minute soak include existing PG application control traffic and ≥30 large-read samples per regime.
- Plans, all result hashes/receipts, row counts, CPU, DB+client memory, pool waits, WAL/checkpoints/autovacuum, disk and failure counts are retained with reproducible inputs.

## Verification and decision policy

Proof boundary: Measured capacity for the stated hardware/dataset/mix only; not universal capacity or provider uptime.
Use section6 and current CI/package routes; retain commands, failures, corrections and
raw sanitized evidence beneath `reports/S09/`. Report unperformed/unavailable checks
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
`python .codex/agents/generated/postgresql-only-storage-v1/ledger_update.py claim --stage S09 --executor <unique-id>`.
The ID is executor-owned, not an owner approval token. Never steal a claim or edit active
JSON by hand. `status`, `resume`, `accept`, `needs-input` and `block` use that same copied
updater and canonical lifecycle; hard-blocked stages require a revisioned replacement,
not status reset. The updater is bound to this directory; do not use an old pack helper.

Write `reports/S09.md` with owned created/modified/deleted paths, foreign changes excluded,
actual verification results and file links, contracts, data/resource effects, remaining
risk and next-stage inputs. If all required checks pass, accept with `--checks-passed`
and actual `--evidence` files. The updater generates a new immutable receipt and validates
the adjacent prompt before allowing its claim. Do not forge a receipt before evidence.
At the last stage completion, all required obligations must be accepted before ledger
completion. Eligibility of the next row is distinct from continuation authorization.

Final report to the owner is Russian, concise and self-contained, with proof limits and
links. Stop after this stage unless a current explicit full-pack continuation covers the next.
