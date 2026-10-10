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
    "title": "Rolling ingestion scheduling and shared provider admission",
    "prompt_path": "03-rolling-ingestion-and-provider-limits.md",
    "report_path": "reports/S03.md",
    "receipt_dir": "receipts/S03",
    "depends_on": [
      "S02"
    ],
    "expected_touches": [
      "../../../../apps/scheduler/market_data_scheduler/wiring/modules/market_data_scheduler.py",
      "../../../../src/trading/contexts/market_data/application/services/work_requests.py",
      "../../../../src/trading/contexts/market_data/application/services/history_bounds.py",
      "../../../../src/trading/contexts/market_data/adapters/outbound/clients/common_http/http_client.py",
      "../../../../src/trading/contexts/market_data/adapters/outbound/clients/rest_candle_ingest_source.py",
      "../../../../apps/worker/market_data_ws/wiring/io_process.py",
      "../../../../src/trading/contexts/market_data/adapters/outbound/config/runtime_config.py",
      "../../../../tests/unit/contexts/market_data/application/services/test_work_requests.py",
      "../../../../tests/unit/contexts/market_data/application/services/test_rest_fill_queue.py",
      "../../../../tests/unit/contexts/market_data/application/services/test_full_history.py",
      "../../../../tests/real_infra/market_data/test_work_requests_postgres.py",
      "../../../../docs/runbooks/navigator-local-market-data-recovery.md",
      "../../../../docs/architecture/market_data/market-data-reference-api-v1.md",
      "../../../../docs/architecture/market_data/market-data-application-ports.md",
      "../../../../docs/architecture/platform/roehub-market-data-activation-and-instrument-selection-v1.md",
      "zone: Existing application ports adapters wiring and their focused tests",
      "zone: Existing installation config migration manifests and project map when affected",
      "zone: Reusable bounded performance harness and sanitized reports"
    ],
    "acceptance_criteria": [
      "A completed worker slot refills before an unrelated slow sibling finishes, without changing the configured concurrency/window baseline, violating fairness/range exclusion or recovering a live sibling.",
      "Coordinator ownership survives rolling dispatch; pause/cancel/lease loss/shutdown preserve checkpoints and prevent old workers writing after ownership release. Slow-trickle responses and PG permit-session loss prove absolute HTTP deadlines and confirmed I/O stop before a permit can be reused.",
      "Lifecycle-owned HTTP sessions are reused safely; one PostgreSQL-backed provider admission contract covers all participating request attempts, quota dimensions and cooldowns across scheduler and collector REST processes.",
      "Comparable fixed-workload throughput/idle-time/provider-connection/quota/memory measurements and real-DB failure tests pass; live/real-provider proof boundaries and required docs are explicit."
    ],
    "proof_boundary": "Scoped implementation, comparable local replay/real-database measurements and explicitly observed provider/runtime behavior; no untested scale, production or publication claim.",
    "validation": {
      "profile": "roehub-focused-gates-and-prompt-pack-artifacts/v1",
      "checks": [
        "Run the plan M baseline/candidate comparison and stage-specific proof matrix.",
        "Run focused pytest suites listed in this prompt with existing real-infra opt-ins on disposable targets; record skips as unavailable.",
        "Run repository-selected lint/type gates, applicable docs/map checks and required review; report actual outcomes."
      ],
      "requires_user_acceptance": false
    },
    "entry_inputs": [
      {
        "path": "implementation-plan.md",
        "producer_stage": null
      },
      {
        "path": "../../../../apps/scheduler/market_data_scheduler/wiring/modules/market_data_scheduler.py",
        "producer_stage": null
      },
      {
        "path": "../../../../src/trading/contexts/market_data/application/services/work_requests.py",
        "producer_stage": null
      },
      {
        "path": "../../../../src/trading/contexts/market_data/application/services/history_bounds.py",
        "producer_stage": null
      },
      {
        "path": "../../../../src/trading/contexts/market_data/adapters/outbound/clients/common_http/http_client.py",
        "producer_stage": null
      },
      {
        "path": "../../../../src/trading/contexts/market_data/adapters/outbound/clients/rest_candle_ingest_source.py",
        "producer_stage": null
      },
      {
        "path": "../../../../apps/worker/market_data_ws/wiring/io_process.py",
        "producer_stage": null
      },
      {
        "path": "../../../../src/trading/contexts/market_data/adapters/outbound/config/runtime_config.py",
        "producer_stage": null
      },
      {
        "path": "../../../../docs/runbooks/navigator-local-market-data-recovery.md",
        "producer_stage": null
      },
      {
        "path": "../../../../docs/architecture/market_data/market-data-reference-api-v1.md",
        "producer_stage": null
      },
      {
        "path": "../../../../docs/architecture/market_data/market-data-application-ports.md",
        "producer_stage": null
      },
      {
        "path": "../../../../docs/architecture/platform/roehub-market-data-activation-and-instrument-selection-v1.md",
        "producer_stage": null
      },
      {
        "path": "../../../../tests/unit/contexts/market_data/application/services/test_work_requests.py",
        "producer_stage": null
      },
      {
        "path": "../../../../tests/unit/contexts/market_data/application/services/test_rest_fill_queue.py",
        "producer_stage": null
      },
      {
        "path": "../../../../tests/unit/contexts/market_data/application/services/test_full_history.py",
        "producer_stage": null
      },
      {
        "path": "../../../../tests/real_infra/market_data/test_work_requests_postgres.py",
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

# S03 — Rolling ingestion scheduling and shared provider admission

## Execution contract and context

Read the linked ledger first; mutable status and claims exist only there. Use `staged-plan-runner`
when executing this submitted prompt, `backend-performance-evidence` for the M comparison,
and `backend-quality-gates` when selecting backend checks. Use `contract-impact-analysis`
at changed API/port/config/persistence boundaries; `browser-qa-evidence` only for materially
changed displayed resource/recovery behavior. Apply repository review depth; do not spawn
agents except where the current user or applicable required-review policy authorizes it.

Always read `AGENTS.md`, this pack's `implementation-plan.md` (especially the named D decision,
M protocol and documentation matrix), `iteration-ledger.md`, and the predecessor report
when present. Then read the listed entrypoints and relevant tests/docs. Consult the project
map only for a crossed context or named wiring gap; historical packs are not execution authority.

Read-only entry validation must pass, then win the existing updater claim before side effects.
Never overwrite another claim, rewrite accepted stage contracts, reset the ledger or infer
an execution claim from manual sequential mode. Preserve foreign dirty changes and record
source/runtime hashes before changing owned code. Routine reversible implementation and
measurement choices within the plan are delegated; ask only for a missing material owner
decision, not confirmation of these four accepted outcomes. Respect the selected stage's
scope and current runtime authority.

## Measurement and validation obligations

Capture the stage baseline before implementation using M; the preceding candidate is the
next stage's baseline, not the original overloaded runtime. Reuse/extend the same harness.
Run focused existing and added meaningful tests at the changed boundary. Discover required
lint/type routing in `pyproject.toml`, `.github/workflows/ci.yml` and
`tools/ci/route_changes.py`; use scoped `uv run ruff check` and required `uv run pyright`
as selected there. Attribute any foreign failures using preserved baseline evidence, never
asserting they are pre-existing from location alone. Do not run unrelated suites to inflate proof.

Real-infra suites require their explicit opt-ins and disposable database identities; inspect
fixtures before running. Do not reset user data, clear production caches, create arbitrary
runtime targets or run saturated tests against the active installation. Bounded real-provider
and canonical-progress observations are separate from replay benchmarks. Update the exact
canonical documentation assigned by the matrix and source/config examples alongside code.
No report may present a skipped check, fixture, socket connection or isolated latency as
end-to-end delivery proof.

## Report, journal and next-stage handoff

Write the contract's report path only after collecting evidence, with: scope/owned paths and
foreign exclusions; baseline/candidate source+runtime+fixture/config hashes; commands and
all repetition results; before/after/delta/noise table; correctness/failure outcomes; numerical
performance acceptance status; compatibility classifications; exact documentation sections;
review mode/findings; local rollout/rollback state; unavailable proof and successor inputs.
Sanitized raw results belong under this pack's `reports/<stage-id>/` with immutable unique
names and content hashes. No secrets, environment dumps or raw provider payloads.

Do not accept without required comparable measurement and correctness evidence. First verify
the next prompt and produced entry report. Use `ledger_update.py accept` only with actual
passed required checks and evidence; its receipt hashes do not themselves prove those checks.
Update the journal before the final Russian report and return the next prompt link. Execute
no successor until selected by the user. Use canonical needs-input/block/revision transitions
only for actual material blockers, never as substitutes for ordinary debugging. S04 ends
the pack after all required evidence; no production or publication claim.

## Stage work

Implement D3 only after revalidating S02 workload classifications and budgets. Hold the existing coordinator's session ownership across the rolling pool; serialize claim/checkpoint bookkeeping appropriately. Recover abandoned work before active dispatch or after full quiescence, never on each refill. A lost lease stops admission and drains/cancels owned I/O without orphan writes. Avoid starvation of older jobs and cancellation controls, and prevent history-bounds discovery monopolizing ready work.

Reuse HTTP connections per safe thread/process lifetime; preserve current error classification and body validation. Introduce only the application port, provider policy adapters and additive PostgreSQL quota metadata needed for shared admission. All internal HTTP retries must reacquire the relevant weighted quota, honor shared Retry-After, and never refund a sent attempt. Verify endpoint weights and multi-window limits from primary exchange documentation and tested headers. Explicitly configure shared egress scope; no per-symbol quota fiction or process-local fallback. Coordinate max-concurrency permits and rate reservations without holding SQL transactions during HTTP, premature TTL permit recycling or a new service. Bound waits, poll traffic and PG connections. Test two processes and coordination loss.

Use identical provider emulator schedules and fixed quotas for causal comparison; real provider smoke is a separate bounded correctness check. Do not combine this stage with changes to daily partitions, unbounded backfill windows, live-writer replacement or S04 receipt batching.

## Task entrypoints

- `apps/scheduler/market_data_scheduler/wiring/modules/market_data_scheduler.py`
- `src/trading/contexts/market_data/application/services/work_requests.py`
- `src/trading/contexts/market_data/application/services/history_bounds.py`
- `src/trading/contexts/market_data/adapters/outbound/clients/common_http/http_client.py`
- `src/trading/contexts/market_data/adapters/outbound/clients/rest_candle_ingest_source.py`
- `apps/worker/market_data_ws/wiring/io_process.py`
- `src/trading/contexts/market_data/adapters/outbound/config/runtime_config.py`

## Focused existing tests

- `tests/unit/contexts/market_data/application/services/test_work_requests.py`
- `tests/unit/contexts/market_data/application/services/test_rest_fill_queue.py`
- `tests/unit/contexts/market_data/application/services/test_full_history.py`
- `tests/real_infra/market_data/test_work_requests_postgres.py`

Extend these tests for the new contract; do not copy implementation logic into assertions.

Explicit deadline proof: test a response that continuously trickles bytes beyond the total deadline, plus loss of the PostgreSQL permit session. A requests read timeout or cancelled awaiting future alone is insufficient; owned network I/O must stop before concurrency capacity is reused.
