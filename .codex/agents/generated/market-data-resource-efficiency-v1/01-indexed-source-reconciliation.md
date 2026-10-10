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
    "title": "Indexed batched artifact source reconciliation",
    "prompt_path": "01-indexed-source-reconciliation.md",
    "report_path": "reports/S01.md",
    "receipt_dir": "receipts/S01",
    "depends_on": [],
    "expected_touches": [
      "../../../../src/trading/contexts/backtest_artifacts/application/services/base_source_reconciliation.py",
      "../../../../src/trading/contexts/backtest_artifacts/adapters/outbound/persistence/clickhouse/stored_coordinates.py",
      "../../../../src/trading/contexts/market_data/adapters/outbound/persistence/clickhouse/canonical_candle_index_reader.py",
      "../../../../src/trading/contexts/market_data/adapters/outbound/persistence/clickhouse/history_coverage_reader.py",
      "../../../../src/trading/contexts/market_data/application/services/stored_candle_signal.py",
      "../../../../apps/scheduler/backtest_artifact_publisher/wiring/managed_app.py",
      "../../../../tests/unit/contexts/backtest_artifacts/test_automatic_base.py",
      "../../../../tests/unit/contexts/market_data/test_stored_candle_signal.py",
      "../../../../tests/real_infra/market_data/test_full_history_clickhouse.py",
      "../../../../tests/real_infra/backtest/test_automatic_base_lifecycle.py",
      "../../../../docs/architecture/backtest/backtest-service-artifact-runtime-v1.ru.md",
      "../../../../docs/architecture/market_data/market-data-reference-api-v1.md",
      "../../../../docs/runbooks/navigator-local-market-data-recovery.md",
      "zone: Existing application ports adapters wiring and their focused tests",
      "zone: Existing installation config migration manifests and project map when affected",
      "zone: Reusable bounded performance harness and sanitized reports"
    ],
    "acceptance_criteria": [
      "Canonical/index fixtures return identical ordered coordinates and real closed-minute bounds, including empty, duplicate, non-aligned cutoff, source-absence and repaired-index cases.",
      "Normal publisher discovery uses bounded index pages without canonical history scans or per-coordinate bounds fan-out; missed hints converge through the existing reconciliation mechanism.",
      "Comparable pre-S01/candidate sweep measurements include query count, read rows, memory and cadence; live canonical advancement is observed or the required proof remains explicitly incomplete.",
      "D1 contracts, applicable documentation and focused gates are updated and passed; report provides the reusable measurement schema and successor inputs."
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
        "path": "../../../../src/trading/contexts/backtest_artifacts/application/services/base_source_reconciliation.py",
        "producer_stage": null
      },
      {
        "path": "../../../../src/trading/contexts/backtest_artifacts/adapters/outbound/persistence/clickhouse/stored_coordinates.py",
        "producer_stage": null
      },
      {
        "path": "../../../../src/trading/contexts/market_data/adapters/outbound/persistence/clickhouse/canonical_candle_index_reader.py",
        "producer_stage": null
      },
      {
        "path": "../../../../src/trading/contexts/market_data/adapters/outbound/persistence/clickhouse/history_coverage_reader.py",
        "producer_stage": null
      },
      {
        "path": "../../../../src/trading/contexts/market_data/application/services/stored_candle_signal.py",
        "producer_stage": null
      },
      {
        "path": "../../../../apps/scheduler/backtest_artifact_publisher/wiring/managed_app.py",
        "producer_stage": null
      },
      {
        "path": "../../../../docs/architecture/backtest/backtest-service-artifact-runtime-v1.ru.md",
        "producer_stage": null
      },
      {
        "path": "../../../../docs/architecture/market_data/market-data-reference-api-v1.md",
        "producer_stage": null
      },
      {
        "path": "../../../../docs/runbooks/navigator-local-market-data-recovery.md",
        "producer_stage": null
      },
      {
        "path": "../../../../tests/unit/contexts/backtest_artifacts/test_automatic_base.py",
        "producer_stage": null
      },
      {
        "path": "../../../../tests/unit/contexts/market_data/test_stored_candle_signal.py",
        "producer_stage": null
      },
      {
        "path": "../../../../tests/real_infra/market_data/test_full_history_clickhouse.py",
        "producer_stage": null
      },
      {
        "path": "../../../../tests/real_infra/backtest/test_automatic_base_lifecycle.py",
        "producer_stage": null
      }
    ]
  }
}
---

# S01 — Indexed batched artifact source reconciliation

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

Implement D1 only. Establish M baseline before changing query behavior. Extend the existing consumer-owned source observation/page contract and wire a ClickHouse adapter to the exact index; do not import concrete adapters into the reconciler. Keep artifact obligations, identity and policy dispatch unchanged. Audit existing post-write notifications across REST and WS and reuse them, including failure isolation. Eliminate the extra epoch-default probe by correct empty-result semantics. Verify actual minute keys before the exclusive cutoff rather than blindly reading aggregate max timestamps. Make degraded index behavior explicit and retain bounded repair plus periodic index reconciliation. A cache that merely hides canonical scans is not completion.

Add only the small shared comparison harness needed by M. Include a 102-coordinate representative fixture and a larger index-only scale fixture with equivalent semantics. Explain what the runtime query log proves versus fixture parity. Preserve all source/index repair and exchange-absence behavior.

## Task entrypoints

- `src/trading/contexts/backtest_artifacts/application/services/base_source_reconciliation.py`
- `src/trading/contexts/backtest_artifacts/adapters/outbound/persistence/clickhouse/stored_coordinates.py`
- `src/trading/contexts/market_data/adapters/outbound/persistence/clickhouse/canonical_candle_index_reader.py`
- `src/trading/contexts/market_data/adapters/outbound/persistence/clickhouse/history_coverage_reader.py`
- `src/trading/contexts/market_data/application/services/stored_candle_signal.py`
- `apps/scheduler/backtest_artifact_publisher/wiring/managed_app.py`

## Focused existing tests

- `tests/unit/contexts/backtest_artifacts/test_automatic_base.py`
- `tests/unit/contexts/market_data/test_stored_candle_signal.py`
- `tests/real_infra/market_data/test_full_history_clickhouse.py`
- `tests/real_infra/backtest/test_automatic_base_lifecycle.py`

Extend these tests for the new contract; do not copy implementation logic into assertions.
