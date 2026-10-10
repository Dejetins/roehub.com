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
    "title": "Asynchronous batched durable stream recovery",
    "prompt_path": "04-async-batched-recovery.md",
    "report_path": "reports/S04.md",
    "receipt_dir": "receipts/S04",
    "depends_on": [
      "S03"
    ],
    "expected_touches": [
      "../../../../src/trading/contexts/market_data/application/services/insert_buffer.py",
      "../../../../src/trading/contexts/market_data/application/services/rest_fill_queue.py",
      "../../../../src/trading/contexts/market_data/application/ports/stores/stream_recovery_store.py",
      "../../../../src/trading/contexts/market_data/adapters/outbound/persistence/postgres/stream_recovery_store.py",
      "../../../../apps/worker/market_data_ws/wiring/modules/market_data_ws.py",
      "../../../../apps/worker/market_data_ws/wiring/persistent_writer.py",
      "../../../../tests/unit/contexts/market_data/application/services/test_insert_buffer.py",
      "../../../../tests/unit/contexts/market_data/application/services/test_stream_recovery.py",
      "../../../../tests/unit/contexts/market_data/application/services/test_rest_fill_queue.py",
      "../../../../tests/unit/apps/worker/test_persistent_writer.py",
      "../../../../tests/real_infra/market_data/test_stream_recovery_postgres.py",
      "../../../../docs/runbooks/navigator-local-market-data-recovery.md",
      "../../../../docs/architecture/market_data/market-data-reference-api-v1.md",
      "../../../../docs/architecture/market_data/market-data-application-ports.md",
      "zone: Existing application ports adapters wiring and their focused tests",
      "zone: Existing installation config migration manifests and project map when affected",
      "zone: Reusable bounded performance harness and sanitized reports"
    ],
    "acceptance_criteria": [
      "No synchronous per-candle PostgreSQL operation remains on the collector event loop; bounded batch admission uses pooled/lifecycle-owned connections and keeps all accepted work durably recoverable.",
      "Idempotent keys, token/lease checks, failed-entry semantics, cancellation and unknown commit handling survive every D4 crash boundary; actual WS candles cannot be acknowledged by source-absence records.",
      "Same-burst S03/S04 comparisons report event-loop p99, transactions/connections per accepted candle, canonical latency, worker+child memory and post-drain plateau; all required loss/backpressure/restart checks pass. Four-stream and synthetic 100-stream burst runs are required; a 1000-stream run is optional and does not establish unsupported scale when absent.",
      "Four-stage result table, compatibility/rollout/docs and locally observed history/live proof are complete; missing scale/provider/runtime evidence remains explicit, and required recovery-risk review is resolved."
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
        "path": "../../../../src/trading/contexts/market_data/application/services/insert_buffer.py",
        "producer_stage": null
      },
      {
        "path": "../../../../src/trading/contexts/market_data/application/services/rest_fill_queue.py",
        "producer_stage": null
      },
      {
        "path": "../../../../src/trading/contexts/market_data/application/ports/stores/stream_recovery_store.py",
        "producer_stage": null
      },
      {
        "path": "../../../../src/trading/contexts/market_data/adapters/outbound/persistence/postgres/stream_recovery_store.py",
        "producer_stage": null
      },
      {
        "path": "../../../../apps/worker/market_data_ws/wiring/modules/market_data_ws.py",
        "producer_stage": null
      },
      {
        "path": "../../../../apps/worker/market_data_ws/wiring/persistent_writer.py",
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
        "path": "../../../../tests/unit/contexts/market_data/application/services/test_insert_buffer.py",
        "producer_stage": null
      },
      {
        "path": "../../../../tests/unit/contexts/market_data/application/services/test_stream_recovery.py",
        "producer_stage": null
      },
      {
        "path": "../../../../tests/unit/contexts/market_data/application/services/test_rest_fill_queue.py",
        "producer_stage": null
      },
      {
        "path": "../../../../tests/unit/apps/worker/test_persistent_writer.py",
        "producer_stage": null
      },
      {
        "path": "../../../../tests/real_infra/market_data/test_stream_recovery_postgres.py",
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

# S04 — Asynchronous batched durable stream recovery

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

Implement D4 only. Reuse the journal's exact recovery key/token semantics and make batch admission/completion explicit application contracts. Keep a bounded coalescer with a lifecycle-owned async PostgreSQL adapter; synchronous callers may retain the current port with a shared tested transaction implementation. Reserve admission capacity before queueing, acknowledge acceptance only after commit, and do not hold async locks across unbounded waits. Include pending producers/in-flight batches/retries in total capacity, not only the existing write buffer.

Audit every recovery-store call made from async collector paths so moving remember() alone cannot leave retry/complete/health calls blocking the loop. Handle per-caller cancellation separately from shared batch ownership. Reconcile unknown commits by stable keys; never delete a committed receipt because its caller disappeared. Drain or durably defer on shutdown and preserve fenced cleanup. Keep one warmed live writer and exact received-candle repair semantics.

Run the fault matrix around PG commit, CH acknowledgement and token completion on disposable real databases, plus deterministic burst comparisons at supported scales. Confirm new canonical candles on currently selected streams. Conclude with per-stage and combined measurements only where comparable, remaining limits and exact updated docs. Do not invent a fifth implementation iteration or auto-publish.

## Task entrypoints

- `src/trading/contexts/market_data/application/services/insert_buffer.py`
- `src/trading/contexts/market_data/application/services/rest_fill_queue.py`
- `src/trading/contexts/market_data/application/ports/stores/stream_recovery_store.py`
- `src/trading/contexts/market_data/adapters/outbound/persistence/postgres/stream_recovery_store.py`
- `apps/worker/market_data_ws/wiring/modules/market_data_ws.py`
- `apps/worker/market_data_ws/wiring/persistent_writer.py`

## Focused existing tests

- `tests/unit/contexts/market_data/application/services/test_insert_buffer.py`
- `tests/unit/contexts/market_data/application/services/test_stream_recovery.py`
- `tests/unit/contexts/market_data/application/services/test_rest_fill_queue.py`
- `tests/unit/apps/worker/test_persistent_writer.py`
- `tests/real_infra/market_data/test_stream_recovery_postgres.py`

Extend these tests for the new contract; do not copy implementation logic into assertions.

Required scale: compare equivalent four-stream and synthetic 100-stream bursts. A 1000-stream run is optional after disposable capacity preflight; report its absence without claiming support for that scale. Do not waive either required run silently.
