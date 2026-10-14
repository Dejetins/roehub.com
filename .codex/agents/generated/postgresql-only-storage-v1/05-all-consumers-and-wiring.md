---
{
  "schema_version": "stage-prompt/v1",
  "prompt_pack_execution": {
    "plan_doc": "implementation-plan.md",
    "prompt_pack_dir": ".",
    "stage_ledger": "iteration-ledger.md"
  },
  "stage_contract": {
    "id": "S05",
    "title": "Switch all consumers, contracts and user-visible behavior",
    "prompt_path": "05-all-consumers-and-wiring.md",
    "report_path": "reports/S05.md",
    "receipt_dir": "receipts/S05",
    "depends_on": [
      "S04"
    ],
    "expected_touches": [
      "../../../../apps/api",
      "../../../../apps/cli",
      "../../../../apps/scheduler",
      "../../../../apps/worker",
      "../../../../src/trading/contexts/market_data",
      "../../../../src/trading/contexts/strategy",
      "../../../../src/trading/contexts/backtest_artifacts",
      "../../../../scripts/rl_trading",
      "../../../../apps/platform-web",
      "../../../../apps/navigator-web"
    ],
    "acceptance_criteria": [
      "Every supported application/CLI/worker/script is wired to PG, with no reachable CH query or fallback.",
      "Repair source enums/config/errors/events/metrics are migrated consistently; persisted old values are handled by additive migration or scoped reset.",
      "Artifacts, fixed-input backtests, strategy tails, metadata jobs and supported clients preserve behavior and current security/loading boundaries.",
      "All source-observation consumers and persisted artifact source targets understand the generation/content-revision contract in the same version rollout."
    ],
    "proof_boundary": "Composition-root/port/API/UI integration with real PG; no local installation cutover yet.",
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
        "path": "../../../../apps/cli/wiring/db/clickhouse.py",
        "producer_stage": null
      },
      {
        "path": "../../../../apps/api/wiring/modules/market_data_reference.py",
        "producer_stage": null
      },
      {
        "path": "../../../../apps/scheduler/backtest_artifact_publisher/wiring/modules/backtest_artifact_publisher.py",
        "producer_stage": null
      },
      {
        "path": "../../../../apps/worker/strategy_live_runner/wiring/modules/strategy_live_runner.py",
        "producer_stage": null
      },
      {
        "path": "../../../../src/trading/contexts/market_data/application/dto/live_tail_repair.py",
        "producer_stage": null
      },
      {
        "path": "implementation-plan.md",
        "producer_stage": null
      },
      {
        "path": "reports/S04.md",
        "producer_stage": "S04"
      }
    ]
  }
}
---

# S05 — Switch all consumers, contracts and user-visible behavior

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
- `apps/cli/wiring/db/clickhouse.py`.
- `apps/api/wiring/modules/market_data_reference.py`.
- `apps/scheduler/backtest_artifact_publisher/wiring/modules/backtest_artifact_publisher.py`.
- `apps/worker/strategy_live_runner/wiring/modules/strategy_live_runner.py`.
- `src/trading/contexts/market_data/application/dto/live_tail_repair.py`.
Read the predecessor report when declared; it is a deferred output, not an authoring gap.
Use project-map slices for affected callers, then inspect actual source. Refresh dirty-tree
ownership and the initial Appendix inventory; follow caller/config/schema references beyond
literal brand matches. Do not reread unrelated historical packs as current instructions.

## Work

Switch API, catalog, history scheduler, WS persistent writer, artifact publisher/precompute, strategy tail, funding CLIs, imports and RL scripts to PG. Keep Redis and backtest artifact compute semantics. Replace engine-coupled names and enum values with canonical_store (domain) / postgresql (operations) deliberately; preserve existing journal events and sanitize/rate-limit logging. Migrate old source labels/constraints using new PG migration rather than editing applied SQL. Trace direct SQL/HTTP calls outside adapters. Run all consumers with CH stopped/unreachable in disposable environment; use browser evidence for affected supported surfaces, not candidate promotion.

Update all internal source-observation and artifact-target consumers to the revision contract; invalidate legacy cached targets explicitly instead of treating equal bounds as equal contents.

## Required acceptance

- Every supported application/CLI/worker/script is wired to PG, with no reachable CH query or fallback.
- Repair source enums/config/errors/events/metrics are migrated consistently; persisted old values are handled by additive migration or scoped reset.
- Artifacts, fixed-input backtests, strategy tails, metadata jobs and supported clients preserve behavior and current security/loading boundaries.

- All source-observation consumers and persisted artifact source targets understand the generation/content-revision contract in the same version rollout.

## Verification and decision policy

Proof boundary: Composition-root/port/API/UI integration with real PG; no local installation cutover yet.
Use section6 and current CI/package routes; retain commands, failures, corrections and
raw sanitized evidence beneath `reports/S05/`. Report unperformed/unavailable checks
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
`python .codex/agents/generated/postgresql-only-storage-v1/ledger_update.py claim --stage S05 --executor <unique-id>`.
The ID is executor-owned, not an owner approval token. Never steal a claim or edit active
JSON by hand. `status`, `resume`, `accept`, `needs-input` and `block` use that same copied
updater and canonical lifecycle; hard-blocked stages require a revisioned replacement,
not status reset. The updater is bound to this directory; do not use an old pack helper.

Write `reports/S05.md` with owned created/modified/deleted paths, foreign changes excluded,
actual verification results and file links, contracts, data/resource effects, remaining
risk and next-stage inputs. If all required checks pass, accept with `--checks-passed`
and actual `--evidence` files. The updater generates a new immutable receipt and validates
the adjacent prompt before allowing its claim. Do not forge a receipt before evidence.
At the last stage completion, all required obligations must be accepted before ledger
completion. Eligibility of the next row is distinct from continuation authorization.

Final report to the owner is Russian, concise and self-contained, with proof limits and
links. Stop after this stage unless a current explicit full-pack continuation covers the next.
