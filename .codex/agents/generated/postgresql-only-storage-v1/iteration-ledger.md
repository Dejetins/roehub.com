# PostgreSQL-only storage — operation journal

The JSON below is the only mutable execution state. This authoring initialized a new draft;
no implementation, migration, reset, benchmark or runtime stage has run. Technical entry
eligibility for S01 does not authorize execution during planning. All stage reports and
receipts are future outputs, not existing proof. Plan and prompts form the linked triad.

Updater: `ledger_update.py`, reused unchanged from the existing resource-efficiency pack.
Never mutate active state manually or use another pack's updater. Stage-specific reports,
checks, changed paths, decisions and immutable receipts belong to each row; no global
latest receipt replaces stage history. The executor follows the installed canonical
`staged-plan-runner/references/ledger-lifecycle-v1.md` semantics.

## Authoring operations (not stage execution)

- 2026-10-13: created plan, ten stage prompts and unclaimed draft journal from the owner request.
- Audited 318 initial path/content/dialect matches plus direct consumers; inventory is in the plan Appendix.
- Reused the existing atomic updater and disposable fixture tests byte-for-byte.
- Independent planning review: one P2 (lost correction hint with unchanged history bounds)
  corrected in D2/D3 and S02–S05/S08; narrow follow-up verdict Ready, no unresolved P1/P2.
- Structural `draft` and S01 `entry` validation passed; nine updater fixture tests passed.
  These checks do not claim implementation, database migration or runtime verification.
- All stages remain pending; no claim/transition receipt or execution history was fabricated.

<!-- prompt-pack-ledger:v1 -->
```json
{
  "schema_version": "prompt-pack-ledger/v1",
  "prompt_pack_execution": {
    "plan_doc": "implementation-plan.md",
    "prompt_pack_dir": ".",
    "stage_ledger": "iteration-ledger.md"
  },
  "execution_mode": "goal_driven",
  "ledger_status": "completed",
  "current_stage": "S10-r1",
  "claim_capability": {
    "mechanism": "Reused unchanged POSIX directory flock, executor-bound claim, receipt validation and fsync atomic ledger replacement; cooperative writers only; no auto-steal.",
    "evidence": {
      "path": "ledger_update.py",
      "sha256": "dbae4cb40c4b69572bd275e2350fa1af79489b1d7a7df4fe3654d40429130550"
    }
  },
  "stages": [
    {
      "contract": {
        "id": "S01",
        "title": "Documentation, complete inventory and target contracts",
        "prompt_path": "01-documentation-and-contracts.md",
        "report_path": "reports/S01.md",
        "receipt_dir": "receipts/S01",
        "depends_on": [],
        "expected_touches": [
          "../../../../docs/architecture/market_data",
          "../../../../docs/architecture/platform",
          "../../../../docs/architecture/backtest",
          "../../../../docs/architecture/strategy",
          "../../../../docs/runbooks",
          "zone: all canonical documents and dependency inventory"
        ],
        "acceptance_criteria": [
          "Every inventory hit has a disposition and a migration owner; callers without literal CH strings are followed.",
          "Normative documentation distinguishes PG target from current CH runtime and covers D1–D7; historical evidence is preserved.",
          "Query catalogue, resource guardrails, reset manifest shape, versioned contract changes and full stage proof map are recorded."
        ],
        "proof_boundary": "Read-only source audit plus normative documentation amendments; no runtime/schema activation.",
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
            "path": "../../../../AGENTS.md",
            "producer_stage": null
          },
          {
            "path": "../../../../docs/architecture/README.md",
            "producer_stage": null
          },
          {
            "path": "../../../../docs/architecture/project-map/AGENT_GUIDE.md",
            "producer_stage": null
          },
          {
            "path": "../../../../docs/architecture/market_data/market-data-application-ports.md",
            "producer_stage": null
          },
          {
            "path": "../../../../docs/runbooks/storage-bootstrap-and-migrations.md",
            "producer_stage": null
          },
          {
            "path": "implementation-plan.md",
            "producer_stage": null
          }
        ]
      },
      "status": "accepted",
      "execution_allowed": false,
      "current_authority": true,
      "executor_claim": "s01-20261013-01a136ec-docs",
      "claimed_at": "2026-10-13T20:21:06.075668Z",
      "transition_receipt": "receipts/S01/receipt-f36956a29b8a413095d7be563259edda.md",
      "decision_packet": null
    },
    {
      "contract": {
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
      },
      "status": "accepted",
      "execution_allowed": false,
      "current_authority": true,
      "executor_claim": "s02-20261013-01a136fb-schema",
      "claimed_at": "2026-10-13T20:37:09.834532Z",
      "transition_receipt": "receipts/S02/receipt-50a61577ff324ec38501e65bef918355.md",
      "decision_packet": null
    },
    {
      "contract": {
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
      },
      "status": "accepted",
      "execution_allowed": false,
      "current_authority": true,
      "executor_claim": "s03-20261013-01a13706-ingestion",
      "claimed_at": "2026-10-13T20:49:43.648373Z",
      "transition_receipt": "receipts/S03/receipt-7c6b7401707e4f629df7b763be16f0b4.md",
      "decision_packet": null
    },
    {
      "contract": {
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
      },
      "status": "accepted",
      "execution_allowed": false,
      "current_authority": true,
      "executor_claim": "s04-20261014-readers-frontier",
      "claimed_at": "2026-10-13T21:19:13.154593Z",
      "transition_receipt": "receipts/S04/receipt-29b327ca5daa497b96a1f3bcc5e9f214.md",
      "decision_packet": null
    },
    {
      "contract": {
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
      },
      "status": "accepted",
      "execution_allowed": false,
      "current_authority": true,
      "executor_claim": "s05-20261014-consumers",
      "claimed_at": "2026-10-13T21:56:58.958888Z",
      "transition_receipt": "receipts/S05/receipt-aa42a38497f048d194b183c6be6eed95.md",
      "decision_packet": null
    },
    {
      "contract": {
        "id": "S06",
        "title": "PostgreSQL-only installation, backup and operations",
        "prompt_path": "06-installation-backup-observability.md",
        "report_path": "reports/S06.md",
        "receipt_dir": "receipts/S06",
        "depends_on": [
          "S05"
        ],
        "expected_touches": [
          "../../../../src/trading/platform/config",
          "../../../../src/trading/contexts/operations",
          "../../../../apps/migrations",
          "../../../../apps/control_agent",
          "../../../../apps/monitoring",
          "../../../../tools/installation",
          "../../../../tools/release",
          "../../../../tools/backup",
          "../../../../configs",
          "../../../../infra",
          "../../../../schemas",
          "../../../../docs/runbooks"
        ],
        "acceptance_criteria": [
          "base/trading/ml and embedded/external PG profiles generate/bootstrap/run without CH service, image, secret, migration or health requirement.",
          "Config/storage-status/backup owner schemas have explicit compatible version handling; stale/old incompatible inputs fail before side effects.",
          "New-format backup/restore and runtime operations preserve digest/quiesce/security invariants; runbooks and monitoring are PG-only."
        ],
        "proof_boundary": "Generated config/schema and disposable installation/backup runtime proof; no release publication or current local reset.",
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
            "path": "../../../../src/trading/platform/config/installation.py",
            "producer_stage": null
          },
          {
            "path": "../../../../src/trading/platform/config/runtime_topology.py",
            "producer_stage": null
          },
          {
            "path": "../../../../src/trading/contexts/operations/backup_contracts.py",
            "producer_stage": null
          },
          {
            "path": "../../../../tools/installation/local_platform.py",
            "producer_stage": null
          },
          {
            "path": "../../../../tools/release/offline_bundle.py",
            "producer_stage": null
          },
          {
            "path": "../../../../docs/runbooks/installation-backup-restore-upgrade.md",
            "producer_stage": null
          },
          {
            "path": "implementation-plan.md",
            "producer_stage": null
          },
          {
            "path": "reports/S05.md",
            "producer_stage": "S05"
          }
        ]
      },
      "status": "accepted",
      "execution_allowed": false,
      "current_authority": true,
      "executor_claim": "s06-20261014-installation",
      "claimed_at": "2026-10-13T22:48:58.179442Z",
      "transition_receipt": "receipts/S06/receipt-2c11b063466c43ad996e088015fe1685.md",
      "decision_packet": null
    },
    {
      "contract": {
        "id": "S07",
        "title": "Remove legacy implementation and reconcile every reference",
        "prompt_path": "07-remove-clickhouse-and-docs-audit.md",
        "report_path": "reports/S07.md",
        "receipt_dir": "receipts/S07",
        "depends_on": [
          "S06"
        ],
        "expected_touches": [
          "../../../../src",
          "../../../../apps",
          "../../../../tools",
          "../../../../scripts",
          "../../../../tests",
          "../../../../migrations/clickhouse",
          "../../../../infra",
          "../../../../configs",
          "../../../../schemas",
          "../../../../pyproject.toml",
          "../../../../uv.lock",
          "../../../../docs"
        ],
        "acceptance_criteria": [
          "CH adapters/DDL/client dependency and executable maintenance/benchmark paths are removed or ported; no product/test-tool runtime import remains.",
          "Every residual inventory match has a narrow documented historical or rejection-test justification; all active normative docs describe PG.",
          "Dependency lock, generated docs/map/schemas/topology and CI semantic coverage mapping are consistent."
        ],
        "proof_boundary": "Repository-wide dependency/reference closure and static/build/import gates, not complete functional performance.",
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
            "path": "../../../../pyproject.toml",
            "producer_stage": null
          },
          {
            "path": "../../../../uv.lock",
            "producer_stage": null
          },
          {
            "path": "../../../../tools/ci/route_changes.py",
            "producer_stage": null
          },
          {
            "path": "../../../../.github/workflows/ci.yml",
            "producer_stage": null
          },
          {
            "path": "../../../../tools/docs/generate_project_map.py",
            "producer_stage": null
          },
          {
            "path": "../../../../tools/qa/postgres_storage_prototype.py",
            "producer_stage": null
          },
          {
            "path": "implementation-plan.md",
            "producer_stage": null
          },
          {
            "path": "reports/S06.md",
            "producer_stage": "S06"
          }
        ]
      },
      "status": "accepted",
      "execution_allowed": false,
      "current_authority": true,
      "executor_claim": "s07-20261014-01a137df-removal",
      "claimed_at": "2026-10-14T00:46:09.314724Z",
      "transition_receipt": "receipts/S07/receipt-640750dabda844f6b7d9bcbb90233d8f.md",
      "decision_packet": null
    },
    {
      "contract": {
        "id": "S08",
        "title": "Full correctness, recovery and browser regression suite",
        "prompt_path": "08-functional-recovery-verification.md",
        "report_path": "reports/S08.md",
        "receipt_dir": "receipts/S08",
        "depends_on": [
          "S07"
        ],
        "expected_touches": [
          "../../../../tests",
          "../../../../tools/qa",
          "../../../../tools/release",
          "../../../../tools/backup",
          "../../../../apps/platform-web/e2e",
          "zone: focused fixes for demonstrated migration defects"
        ],
        "acceptance_criteria": [
          "Real PG semantics, all consumers and supported UI/API journeys pass with CH absent; expected results are not replaced by mocks.",
          "Failure drills prove truthful receipts/cursors/events and bounded recovery after DB restart/disconnect/contention.",
          "Backup/restore, bootstrap, reset invalidation and artifact/backtest parity pass; independent reviewer finds no unresolved material migration issue.",
          "A process crash after committing an old-minute value correction but before publishing its hint is recovered by bounded reconciliation and rebuilds the affected artifact."
        ],
        "proof_boundary": "Functional/failure integration on isolated targets; no long-soak or current-installation delivery claim.",
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
            "path": "../../../../tests/real_infra/market_data/test_transactional_ingestion_postgres.py",
            "producer_stage": null
          },
          {
            "path": "../../../../tests/real_infra/backtest/test_artifact_dispatch.py",
            "producer_stage": null
          },
          {
            "path": "../../../../apps/migrations/verify_storage_runtime.py",
            "producer_stage": null
          },
          {
            "path": "../../../../tools/release/verify_greenfield_lifecycle.py",
            "producer_stage": null
          },
          {
            "path": "../../../../apps/platform-web/package.json",
            "producer_stage": null
          },
          {
            "path": "implementation-plan.md",
            "producer_stage": null
          },
          {
            "path": "reports/S07.md",
            "producer_stage": "S07"
          }
        ]
      },
      "status": "accepted",
      "execution_allowed": false,
      "current_authority": true,
      "executor_claim": "s08-20261014-01a1380b-functional",
      "claimed_at": "2026-10-14T01:35:14.413330Z",
      "transition_receipt": "receipts/S08/receipt-ecf8b5647c8048c29b033b1effa0a068.md",
      "decision_packet": null
    },
    {
      "contract": {
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
          "Three repeated ≥5-minute normal/stress runs and ≥60-minute soak include existing PG application control traffic and ≥30 large-read samples per regime.",
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
      },
      "status": "accepted",
      "execution_allowed": false,
      "current_authority": true,
      "executor_claim": "s09-20261014-01a1382f-performance",
      "claimed_at": "2026-10-14T02:13:43.420618Z",
      "transition_receipt": "receipts/S09/receipt-cf2e6c5240e64f189d66014cada80569.md",
      "decision_packet": null
    },
    {
      "contract": {
        "id": "S10",
        "title": "Local development cutover, real streams and final acceptance",
        "prompt_path": "10-local-cutover-and-final-audit.md",
        "report_path": "reports/S10.md",
        "receipt_dir": "receipts/S10",
        "depends_on": [
          "S09"
        ],
        "expected_touches": [
          "../../../../tools/installation",
          "../../../../docs/runbooks",
          "zone: identified local installation config and market-only derived state",
          "zone: final migration evidence"
        ],
        "acceptance_criteria": [
          "Current local installation uses PG-only market storage, preserved accounts/settings/secrets/selections, and correct fresh history/artifact generation.",
          "Real selected history advances and consecutive closed candles arrive after bounded disconnect/reconnect; independent SQL/API observations and event logs agree.",
          "No reachable CH dependency/process requirement remains; final docs, setup/backup guidance, gates and known data disposition are complete."
        ],
        "proof_boundary": "Actual identified local installation delivery and real-provider evidence; no remote deployment or Git publication.",
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
            "path": "../../../../tools/installation/local_platform.py",
            "producer_stage": null
          },
          {
            "path": "../../../../docs/runbooks/navigator-local-market-data-recovery.md",
            "producer_stage": null
          },
          {
            "path": "../../../../docs/runbooks/storage-bootstrap-and-migrations.md",
            "producer_stage": null
          },
          {
            "path": "implementation-plan.md",
            "producer_stage": null
          },
          {
            "path": "reports/S09.md",
            "producer_stage": "S09"
          }
        ]
      },
      "status": "blocked",
      "execution_allowed": false,
      "current_authority": true,
      "executor_claim": "s10-20261014-01a13933-cutover",
      "claimed_at": "2026-10-14T06:57:53.711221Z",
      "transition_receipt": null,
      "decision_packet": null,
      "reason": "Installed PG-only cutover completed with verified retained-history artifacts/backtest and preserved user state; required real selected-provider history/funding, >=10 consecutive closed minutes and bounded reconnect evidence unavailable: Binance HTTP451, Bybit HTTP403. S10 cannot be accepted; continuation requires provider availability and a revisioned replacement, preserving this evidence.",
      "evidence": [
        "reports/S10.md",
        "reports/S10/provider-http-status.jsonl",
        "reports/S10/installed-sql-final.json",
        "reports/S10/cleanup.json"
      ]
    },
    {
      "contract": {
        "id": "S10-r1",
        "title": "Selected-provider live completion after access repair",
        "prompt_path": "10-r1-live-completion.md",
        "report_path": "reports/S10-r1.md",
        "receipt_dir": "receipts/S10-r1",
        "depends_on": [
          "S09"
        ],
        "expected_touches": [
          "../../../../tools/installation",
          "../../../../docs/runbooks",
          "zone: identified local installation config and market-only derived state",
          "zone: final migration evidence"
        ],
        "acceptance_criteria": [
          "Current local installation uses PG-only market storage, preserved accounts/settings/secrets/selections, and correct fresh history/artifact generation.",
          "Real selected history advances and consecutive closed candles arrive after bounded disconnect/reconnect; independent SQL/API observations and event logs agree.",
          "No reachable CH dependency/process requirement remains; final docs, setup/backup guidance, gates and known data disposition are complete."
        ],
        "proof_boundary": "Actual identified local installation delivery and real-provider evidence; no remote deployment or Git publication.",
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
            "path": "../../../../tools/installation/local_platform.py",
            "producer_stage": null
          },
          {
            "path": "../../../../docs/runbooks/navigator-local-market-data-recovery.md",
            "producer_stage": null
          },
          {
            "path": "../../../../docs/runbooks/storage-bootstrap-and-migrations.md",
            "producer_stage": null
          },
          {
            "path": "implementation-plan.md",
            "producer_stage": null
          },
          {
            "path": "reports/S09.md",
            "producer_stage": "S09"
          },
          {
            "path": "reports/S10.md",
            "producer_stage": null
          },
          {
            "path": "reports/S10-r1/provider-repair.json",
            "producer_stage": null
          }
        ]
      },
      "status": "accepted",
      "execution_allowed": false,
      "current_authority": true,
      "executor_claim": "s10-r1-20261014-01a13933-live",
      "claimed_at": "2026-10-14T08:07:21.253666Z",
      "transition_receipt": "receipts/S10-r1/receipt-0ecb2766a8374a6fb729862c89d704bc.md",
      "decision_packet": null,
      "replaces_stage": "S10",
      "repair_evidence_ref": "reports/S10-r1/provider-repair.json"
    }
  ],
  "authoring_scope": "Owner selected full-pack Goal execution on 2026-10-13 in orchestrator chat 01a136ea-3758-7ab0-9a63-4b9c3b2f3a48. Execute S01-S10 sequentially in separate local Codex chats; orchestrator may send corrections. No Git publication or remote deployment. Preserve all plan safety and evidence requirements.",
  "execution_history": [
    {
      "at": "2026-10-13T20:21:06.075668Z",
      "action": "claim",
      "stage": "S01",
      "executor": "s01-20261013-01a136ec-docs",
      "receipt": null,
      "reason": null
    },
    {
      "at": "2026-10-13T20:35:50.747379Z",
      "action": "accept",
      "stage": "S01",
      "executor": "s01-20261013-01a136ec-docs",
      "receipt": "receipts/S01/receipt-f36956a29b8a413095d7be563259edda.md",
      "reason": null
    },
    {
      "at": "2026-10-13T20:37:09.834532Z",
      "action": "claim",
      "stage": "S02",
      "executor": "s02-20261013-01a136fb-schema",
      "receipt": null,
      "reason": null
    },
    {
      "at": "2026-10-13T20:47:59.879096Z",
      "action": "accept",
      "stage": "S02",
      "executor": "s02-20261013-01a136fb-schema",
      "receipt": "receipts/S02/receipt-50a61577ff324ec38501e65bef918355.md",
      "reason": null
    },
    {
      "at": "2026-10-13T20:49:43.648373Z",
      "action": "claim",
      "stage": "S03",
      "executor": "s03-20261013-01a13706-ingestion",
      "receipt": null,
      "reason": null
    },
    {
      "at": "2026-10-13T21:16:29.386957Z",
      "action": "accept",
      "stage": "S03",
      "executor": "s03-20261013-01a13706-ingestion",
      "receipt": "receipts/S03/receipt-7c6b7401707e4f629df7b763be16f0b4.md",
      "reason": null
    },
    {
      "at": "2026-10-13T21:19:13.154593Z",
      "action": "claim",
      "stage": "S04",
      "executor": "s04-20261014-readers-frontier",
      "receipt": null,
      "reason": null
    },
    {
      "at": "2026-10-13T21:54:54.955726Z",
      "action": "accept",
      "stage": "S04",
      "executor": "s04-20261014-readers-frontier",
      "receipt": "receipts/S04/receipt-29b327ca5daa497b96a1f3bcc5e9f214.md",
      "reason": null
    },
    {
      "at": "2026-10-13T21:56:58.958888Z",
      "action": "claim",
      "stage": "S05",
      "executor": "s05-20261014-consumers",
      "receipt": null,
      "reason": null
    },
    {
      "at": "2026-10-13T22:47:08.260818Z",
      "action": "accept",
      "stage": "S05",
      "executor": "s05-20261014-consumers",
      "receipt": "receipts/S05/receipt-aa42a38497f048d194b183c6be6eed95.md",
      "reason": null
    },
    {
      "at": "2026-10-13T22:48:58.179442Z",
      "action": "claim",
      "stage": "S06",
      "executor": "s06-20261014-installation",
      "receipt": null,
      "reason": null
    },
    {
      "at": "2026-10-14T00:44:00.094998Z",
      "action": "accept",
      "stage": "S06",
      "executor": "s06-20261014-installation",
      "receipt": "receipts/S06/receipt-2c11b063466c43ad996e088015fe1685.md",
      "reason": null
    },
    {
      "at": "2026-10-14T00:46:09.314724Z",
      "action": "claim",
      "stage": "S07",
      "executor": "s07-20261014-01a137df-removal",
      "receipt": null,
      "reason": null
    },
    {
      "at": "2026-10-14T01:30:27.564837Z",
      "action": "migrate-pending-entry-locators",
      "stage": "S08,S09",
      "executor": "root-orchestrator",
      "receipt": null,
      "reason": "File-only locators corrected without changing scope or state",
      "evidence": "reports/pending-entry-input-repair.json"
    },
    {
      "at": "2026-10-14T01:33:23.061923Z",
      "action": "accept",
      "stage": "S07",
      "executor": "s07-20261014-01a137df-removal",
      "receipt": "receipts/S07/receipt-640750dabda844f6b7d9bcbb90233d8f.md",
      "reason": null
    },
    {
      "at": "2026-10-14T01:35:14.413330Z",
      "action": "claim",
      "stage": "S08",
      "executor": "s08-20261014-01a1380b-functional",
      "receipt": null,
      "reason": null
    },
    {
      "at": "2026-10-14T02:11:33.663205Z",
      "action": "accept",
      "stage": "S08",
      "executor": "s08-20261014-01a1380b-functional",
      "receipt": "receipts/S08/receipt-ecf8b5647c8048c29b033b1effa0a068.md",
      "reason": null
    },
    {
      "at": "2026-10-14T02:13:43.420618Z",
      "action": "claim",
      "stage": "S09",
      "executor": "s09-20261014-01a1382f-performance",
      "receipt": null,
      "reason": null
    },
    {
      "at": "2026-10-14T06:54:56.071484Z",
      "action": "accept",
      "stage": "S09",
      "executor": "s09-20261014-01a1382f-performance",
      "receipt": "receipts/S09/receipt-cf2e6c5240e64f189d66014cada80569.md",
      "reason": null
    },
    {
      "at": "2026-10-14T06:57:53.711221Z",
      "action": "claim",
      "stage": "S10",
      "executor": "s10-20261014-01a13933-cutover",
      "receipt": null,
      "reason": null
    },
    {
      "at": "2026-10-14T08:03:23.857182Z",
      "action": "block",
      "stage": "S10",
      "executor": "s10-20261014-01a13933-cutover",
      "receipt": null,
      "reason": "Installed PG-only cutover completed with verified retained-history artifacts/backtest and preserved user state; required real selected-provider history/funding, >=10 consecutive closed minutes and bounded reconnect evidence unavailable: Binance HTTP451, Bybit HTTP403. S10 cannot be accepted; continuation requires provider availability and a revisioned replacement, preserving this evidence."
    },
    {
      "at": "2026-10-14T08:07:21.253666Z",
      "action": "claim-revisioned-replacement",
      "stage": "S10-r1",
      "executor": "s10-r1-20261014-01a13933-live",
      "receipt": null,
      "reason": "Owner requested retry after tunnel change; four real provider HTTP200 responses repair prior access blocker",
      "evidence": "reports/S10-r1/provider-repair.json"
    },
    {
      "at": "2026-10-14T09:23:30.528133Z",
      "action": "accept",
      "stage": "S10-r1",
      "executor": "s10-r1-20261014-01a13933-live",
      "receipt": "receipts/S10-r1/receipt-0ecb2766a8374a6fb729862c89d704bc.md",
      "reason": null
    }
  ]
}
```
