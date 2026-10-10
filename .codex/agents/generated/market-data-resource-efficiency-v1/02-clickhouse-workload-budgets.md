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
    "title": "ClickHouse workload isolation and resource budgets",
    "prompt_path": "02-clickhouse-workload-budgets.md",
    "report_path": "reports/S02.md",
    "receipt_dir": "receipts/S02",
    "depends_on": [
      "S01"
    ],
    "expected_touches": [
      "../../../../apps/cli/wiring/db/clickhouse.py",
      "../../../../src/trading/contexts/market_data/adapters/outbound/persistence/clickhouse/gateway.py",
      "../../../../tools/installation/local_platform.py",
      "../../../../configs/dev/clickhouse-local-profiles.xml",
      "../../../../apps/api/wiring/modules/market_data_reference.py",
      "../../../../apps/worker/market_data_ws/wiring/persistent_writer.py",
      "../../../../apps/scheduler/market_data_scheduler/wiring/modules/market_data_scheduler.py",
      "../../../../apps/scheduler/backtest_artifact_publisher/wiring/modules/backtest_artifact_publisher.py",
      "../../../../tests/unit/tools/test_local_platform.py",
      "../../../../tests/unit/apps/worker/test_persistent_writer.py",
      "../../../../tests/real_infra/market_data/test_full_history_clickhouse.py",
      "../../../../tests/real_infra/backtest/test_artifact_integrated.py",
      "../../../../docs/runbooks/navigator-local-market-data-recovery.md",
      "../../../../docs/runbooks/offline-release-installation.md",
      "../../../../docs/architecture/platform/roehub-market-data-activation-and-instrument-selection-v1.md",
      "../../../../docs/architecture/market_data/market-data-reference-api-v1.md",
      "../../../../docs/architecture/backtest/backtest-service-artifact-runtime-v1.ru.md",
      "zone: Existing application ports adapters wiring and their focused tests",
      "zone: Existing installation config migration manifests and project map when affected",
      "zone: Reusable bounded performance harness and sanitized reports"
    ],
    "acceptance_criteria": [
      "Trusted stream/interactive/backfill/artifacts class assignment covers all participating direct, pooled and child-process clients with server-enforced cross-process memory/concurrency bounds and no unrestricted default-user bypass.",
      "Generated installation config, private credentials and least-privilege grants support ordered upgrade/rollback; ordinary clients cannot override caps, and MV writes work with scoped grants.",
      "Identical offered-load before/after mixed-load runs report live/API tail latency, memory, class waits/errors and finite background progress without changed global limits, lost candles or hidden dropped work.",
      "Resource-pressure retries preserve durable state, public error compatibility is verified, applicable docs/gates pass, and an independent security-boundary review has no unresolved blocking finding."
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
        "path": "../../../../apps/cli/wiring/db/clickhouse.py",
        "producer_stage": null
      },
      {
        "path": "../../../../src/trading/contexts/market_data/adapters/outbound/persistence/clickhouse/gateway.py",
        "producer_stage": null
      },
      {
        "path": "../../../../tools/installation/local_platform.py",
        "producer_stage": null
      },
      {
        "path": "../../../../configs/dev/clickhouse-local-profiles.xml",
        "producer_stage": null
      },
      {
        "path": "../../../../apps/api/wiring/modules/market_data_reference.py",
        "producer_stage": null
      },
      {
        "path": "../../../../apps/worker/market_data_ws/wiring/persistent_writer.py",
        "producer_stage": null
      },
      {
        "path": "../../../../apps/scheduler/market_data_scheduler/wiring/modules/market_data_scheduler.py",
        "producer_stage": null
      },
      {
        "path": "../../../../apps/scheduler/backtest_artifact_publisher/wiring/modules/backtest_artifact_publisher.py",
        "producer_stage": null
      },
      {
        "path": "../../../../docs/runbooks/navigator-local-market-data-recovery.md",
        "producer_stage": null
      },
      {
        "path": "../../../../docs/runbooks/offline-release-installation.md",
        "producer_stage": null
      },
      {
        "path": "../../../../docs/architecture/platform/roehub-market-data-activation-and-instrument-selection-v1.md",
        "producer_stage": null
      },
      {
        "path": "../../../../docs/architecture/market_data/market-data-reference-api-v1.md",
        "producer_stage": null
      },
      {
        "path": "../../../../docs/architecture/backtest/backtest-service-artifact-runtime-v1.ru.md",
        "producer_stage": null
      },
      {
        "path": "../../../../tests/unit/tools/test_local_platform.py",
        "producer_stage": null
      },
      {
        "path": "../../../../tests/unit/apps/worker/test_persistent_writer.py",
        "producer_stage": null
      },
      {
        "path": "../../../../tests/real_infra/market_data/test_full_history_clickhouse.py",
        "producer_stage": null
      },
      {
        "path": "../../../../tests/real_infra/backtest/test_artifact_integrated.py",
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

# S02 — ClickHouse workload isolation and resource budgets

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

Implement D2 only using the accepted S01 candidate as baseline. Inventory every ClickHouse client in the four workloads, including SELECT, INSERT, command, verifier/repair, artifact build, health and child-process paths. Record intentional admin exclusions. Verify native per-user aggregate memory and concurrent-query semantics on the installed version before choosing measured caps. Bind class-specific settings/identities through composition, not arbitrary SQL rewriting or caller-supplied HTTP labels. Scope totals across replicas of the application processes sharing this ClickHouse server; report the boundary if multiple database servers are introduced.

Use existing installation ownership for profiles and private secret references, preserve existing credential/account state, and prove regenerated/recreated disposable installations retain the settings. Keep global/container limits unchanged. Bounded classified rejection/wait must not become exchange absence or permanent data failure. Exercise >1 low-priority process, materialized views, denied cap overrides, unavailable credentials and rollback. Do not claim CPU/IO reservation from memory limits alone. Independent review covers grants, secret handling, bypasses and downgrade behavior. If local credential activation lacks covering authority, finish code/disposable proof and identify only that activation gap.

## Task entrypoints

- `apps/cli/wiring/db/clickhouse.py`
- `src/trading/contexts/market_data/adapters/outbound/persistence/clickhouse/gateway.py`
- `tools/installation/local_platform.py`
- `configs/dev/clickhouse-local-profiles.xml`
- `apps/api/wiring/modules/market_data_reference.py`
- `apps/worker/market_data_ws/wiring/persistent_writer.py`
- `apps/scheduler/market_data_scheduler/wiring/modules/market_data_scheduler.py`
- `apps/scheduler/backtest_artifact_publisher/wiring/modules/backtest_artifact_publisher.py`

## Focused existing tests

- `tests/unit/tools/test_local_platform.py`
- `tests/unit/apps/worker/test_persistent_writer.py`
- `tests/real_infra/market_data/test_full_history_clickhouse.py`
- `tests/real_infra/backtest/test_artifact_integrated.py`

Extend these tests for the new contract; do not copy implementation logic into assertions.
