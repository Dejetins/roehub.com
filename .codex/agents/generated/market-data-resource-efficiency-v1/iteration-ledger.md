# Market Data resource efficiency — iteration journal v1

The JSON record is the only mutable execution-state source. This authoring delivery has
not executed an optimization. The owner selected four outcomes and measurements;
each submitted prompt selects exactly one iteration, in order, without another approval
of routine implementation choices. No autonomous multi-stage run or publication is implied.

## Artifacts and order

- [Plan](implementation-plan.md).
- [S01 — Indexed reconciliation](01-indexed-source-reconciliation.md).
- [S02 — ClickHouse budgets](02-clickhouse-workload-budgets.md).
- [S03 — Rolling ingestion and exchange limits](03-rolling-ingestion-and-provider-limits.md).
- [S04 — Async recovery journal](04-async-batched-recovery.md).

## Exclusive updater and validation

`ledger_update.py` and its disposable test suite are byte-for-byte reuse of the existing
`backtest-npy-input-materialization-v1` implementation. Its module docstring and code own
the atomicity contract: stable directory flock, validation within the lock, durable
executor identity, fsync and atomic replace. No stale-claim stealing or active-file manual
editing. Tests touch only temporary fixture ledgers. Copying the tool does not select or
modify the old pack. Keep this directory stable during execution.

From repository root, use the current Python 3 runtime:

```sh
python3 .codex/agents/generated/market-data-resource-efficiency-v1/ledger_update.py status
python3 .codex/agents/generated/market-data-resource-efficiency-v1/ledger_update.py claim --stage S01 --executor UNIQUE_EXECUTOR_ID
python3 .codex/agents/generated/market-data-resource-efficiency-v1/ledger_update.py accept --stage S01 --executor SAME_EXECUTOR_ID --checks-passed --evidence reports/S01.md
```

The executor chooses the ID. `accept` requires actual successful checks, writes an immutable
receipt, validates successor inputs and allows the next stage; it does not start it. Never
use `--checks-passed` for missing performance comparisons or unresolved correctness risks.
All four rows use `requires_user_acceptance=false`; numerical SLO acceptance is separate
from stage evidence acceptance. For blockers, use the updater's existing `needs-input` or
`block` contract; hard-blocked rows require a revisioned replacement, not in-place reset.

Validation profile: installed Prompt Manager `prompt-pack-ledger/v1`, `stage-prompt/v1`,
`prompt-pack-receipt/v1`, plus repository checks during implementation. Read-only validator:

```sh
python3 /Users/daniildegtyarev/.codex/skills/prompt-manager/scripts/validate_pack.py --root /Users/daniildegtyarev/Projects/roehub.com --ledger /Users/daniildegtyarev/Projects/roehub.com/.codex/agents/generated/market-data-resource-efficiency-v1/iteration-ledger.md --check draft
python3 /Users/daniildegtyarev/.codex/skills/prompt-manager/scripts/validate_pack.py --root /Users/daniildegtyarev/Projects/roehub.com --ledger /Users/daniildegtyarev/Projects/roehub.com/.codex/agents/generated/market-data-resource-efficiency-v1/iteration-ledger.md --check entry --stage S01
python3 .codex/agents/generated/market-data-resource-efficiency-v1/test_ledger_update.py -v
```

Expected validator exit 0 and JSON `status=pass`, proof boundary
`artifact-structure-and-file-binding`. Expected updater exit 0 and `status=ok`.
Neither result proves optimization, authority for a release or real provider capacity.

## Iteration result fields

Each stage report is linked by its contract below and summarized in its own row's evidence
history after execution. Required fields: baseline/candidate identities; workload and runtime;
primary metric before/after/absolute delta/percent/spread; memory and correctness guardrails;
commands and raw evidence hashes; docs updated; compatibility; review; rollout/rollback;
unavailable proof and next stage. Never overwrite a prior iteration with global latest values.

| Iteration | Primary measurement | Guardrails |
|---|---|---|
| S01 | Rows read and query count per source sweep | Exact bounds/cutoff, hint loss recovery, candle progression |
| S02 | Live/API tail latency under fixed mixed load | Global caps across processes, memory, background progress, grants |
| S03 | Completed windows/s and idle time while work is ready | Fairness, shared provider quota, ownership and no overlap |
| S04 | Event-loop p99 and PG transactions per accepted candle | Durability, cancellation, bounded memory and canonical latency |

All before/after result cells are intentionally absent until measured. The read-only
2026-10-10 observations in the plan are diagnostic context, not stage acceptance.

## Authoring log

- 2026-10-10: created four-stage draft from the owner's selected scope; no claims or runs.
  Current implementation/deployment state remains unchanged. Existing updater reused;
  artifact validation and focused design review are recorded below after they complete.
- 2026-10-10 authoring validation: saved-file `--check draft` and `--check entry --stage S01`
  returned exit 0 / `status=pass`; updater `status` returned `draft`, null current stage,
  four pending rows and only S01 enabled. All nine disposable updater tests passed,
  including competing claims, lock rejection, foreign-owner rejection, receipt binding
  and sequential completion. Current entry paths exist; updater and test bytes match
  the existing source. These are artifact/updater checks, not backend measurements.

- 2026-10-10 independent read-only design/security/recovery review: no Blocker/High.
  Two Medium clarifications were addressed before entry: required 4/100 versus optional
  1000-stream S04 measurements now agree across plan/prompt/contract, and S03 explicitly
  tests absolute HTTP lifetime, slow-trickle responses and PG permit-session loss.
  Focused follow-up checks verify these edits; no implementation or live tests were run.
- 2026-10-10 final follow-up: clarified that automatic PG session-lock release cannot
  release an uncertain HTTP concurrency permit. Re-ran saved-file draft/entry validation
  successfully and all nine updater tests passed again. S01 is entry-ready; all rows
  remain pending and the ledger remains draft. No numerical improvement is claimed.

## Canonical execution record

<!-- prompt-pack-ledger:v1 -->
```json
{
  "schema_version": "prompt-pack-ledger/v1",
  "prompt_pack_execution": {
    "plan_doc": "implementation-plan.md",
    "prompt_pack_dir": ".",
    "stage_ledger": "iteration-ledger.md"
  },
  "execution_mode": "manual_sequential",
  "ledger_status": "completed",
  "current_stage": "S04",
  "claim_capability": {
    "mechanism": "Existing pack-local POSIX directory flock, exclusive persistent executor claim and fsynced atomic replacement; reused unchanged from backtest-npy-input-materialization-v1",
    "evidence": {
      "path": "ledger_update.py",
      "sha256": "dbae4cb40c4b69572bd275e2350fa1af79489b1d7a7df4fe3654d40429130550"
    }
  },
  "stages": [
    {
      "contract": {
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
      },
      "status": "accepted",
      "execution_allowed": false,
      "current_authority": true,
      "executor_claim": "s01-01a126d3-7295-7b53-a09e-97229eacf382",
      "claimed_at": "2026-10-10T17:19:58.226845Z",
      "transition_receipt": "receipts/S01/receipt-49e8873d578647438adb6993ffc19e13.md",
      "decision_packet": null
    },
    {
      "contract": {
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
      },
      "status": "accepted",
      "execution_allowed": false,
      "current_authority": true,
      "executor_claim": "s02-01a126e9-8273-7231-887b-b70303ac077c",
      "claimed_at": "2026-10-10T17:43:56.688592Z",
      "transition_receipt": "receipts/S02/receipt-532ed40d08db475f9f468c928e14967d.md",
      "decision_packet": null
    },
    {
      "contract": {
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
      },
      "status": "accepted",
      "execution_allowed": false,
      "current_authority": true,
      "executor_claim": "s03-01a1273a-025e-7412-a504-da65ae609eea",
      "claimed_at": "2026-10-10T19:11:48.779680Z",
      "transition_receipt": "receipts/S03/receipt-dc1e2658ffe44ccd9528581fc8a7dcc0.md",
      "decision_packet": null
    },
    {
      "contract": {
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
      },
      "status": "accepted",
      "execution_allowed": false,
      "current_authority": true,
      "executor_claim": "s04-20261011-async-recovery",
      "claimed_at": "2026-10-10T21:34:52.417432Z",
      "transition_receipt": "receipts/S04/receipt-33c297169bfc4ad8a089b383928ae2a6.md",
      "decision_packet": null
    }
  ],
  "execution_history": [
    {
      "at": "2026-10-10T17:19:58.226845Z",
      "action": "claim",
      "stage": "S01",
      "executor": "s01-01a126d3-7295-7b53-a09e-97229eacf382",
      "receipt": null,
      "reason": null
    },
    {
      "at": "2026-10-10T17:42:05.988830Z",
      "action": "accept",
      "stage": "S01",
      "executor": "s01-01a126d3-7295-7b53-a09e-97229eacf382",
      "receipt": "receipts/S01/receipt-49e8873d578647438adb6993ffc19e13.md",
      "reason": "S01 focused correctness/gates and comparable source-sweep evidence complete: 207 to 3 queries, 2829312 to 612 rows; actual live advancement observed. Installed publisher remains pre-S01; full Pyright retains exactly 54 baseline diagnostics. S02 may be selected separately."
    },
    {
      "at": "2026-10-10T17:43:56.688592Z",
      "action": "claim",
      "stage": "S02",
      "executor": "s02-01a126e9-8273-7231-887b-b70303ac077c",
      "receipt": null,
      "reason": null
    },
    {
      "at": "2026-10-10T19:10:32.509594Z",
      "action": "accept",
      "stage": "S02",
      "executor": "s02-01a126e9-8273-7231-887b-b70303ac077c",
      "receipt": "receipts/S02/receipt-532ed40d08db475f9f468c928e14967d.md",
      "reason": null
    },
    {
      "at": "2026-10-10T19:11:48.779680Z",
      "action": "claim",
      "stage": "S03",
      "executor": "s03-01a1273a-025e-7412-a504-da65ae609eea",
      "receipt": null,
      "reason": null
    },
    {
      "at": "2026-10-10T21:33:29.070811Z",
      "action": "accept",
      "stage": "S03",
      "executor": "s03-01a1273a-025e-7412-a504-da65ae609eea",
      "receipt": "receipts/S03/receipt-dc1e2658ffe44ccd9528581fc8a7dcc0.md",
      "reason": "Required comparable six-run M4, real-PG/CH and bounded provider correctness passed; migration crash/replay blocker fixed and independently reviewed. Throughput/REST-tail/PG/memory tradeoff resolved explicitly under M.7. Full Pyright remains 54 baseline-equivalent diagnostics and runtime inventory fails on independently reproduced foreign binary; no full release or installed-runtime claim. S04 not executed."
    },
    {
      "at": "2026-10-10T21:34:52.417432Z",
      "action": "claim",
      "stage": "S04",
      "executor": "s04-20261011-async-recovery",
      "receipt": null,
      "reason": null
    },
    {
      "at": "2026-10-10T21:59:58.311163Z",
      "action": "accept",
      "stage": "S04",
      "executor": "s04-20261011-async-recovery",
      "receipt": "receipts/S04/receipt-33c297169bfc4ad8a089b383928ae2a6.md",
      "reason": null
    }
  ]
}
```
