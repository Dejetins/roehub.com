# Backtest NPY input materialization — iteration ledger v1

Created: 2026-10-07. Language: English. The JSON record below is the only mutable execution-state source. Prompt intent lives in the linked plan and exact stage contracts; reports/receipts are evidence. The table and log below summarize authoring only and do not supersede JSON.

## Authority and sequential execution

All product decisions remain settled. The owner's 2026-10-07 follow-up requests an unblocked journal and sequential prompt submission. Each submitted prompt executes one stage, writes its result, enables the next eligible stage and stops with the next prompt link. Every stage has `requires_user_acceptance=false`; no separate acceptance or unchanged-design approval is needed. No production deployment, publication, unrelated work or destructive existing-artifact cleanup is authorized.

## Current readiness

- The former updater gap is resolved by the owner-authorized pack-local `ledger_update.py`.
- All stages remain pending; no stage has been claimed or executed by this unblock operation.
- Only S01 is initially executable. Subsequent stages open after predecessor proof and receipt validation.
- No product decision packet or owner acceptance gate is pending.
- Local platform: Python 3 stdlib on a POSIX filesystem supporting directory flock/fsync/atomic replace (verified on this machine). All writers cooperate through the stable pack-directory lock; do not move the pack while executing.

## Updater ownership and commands

`ledger_update.py` locks the pack directory exclusively, reads the actual current journal, validates the installed schema/entry/evidence bindings, checks durable executor ownership, and commits one atomic fsynced replacement. The short OS lock is released when the command exits; the persistent stage claim remains until its valid transition. A different executor cannot overwrite it. No TTL-based stealing or automatic reset is permitted. Failed validation leaves the prior ledger untouched. Receipts are immutable, uniquely named and written before the journal references them; an orphan receipt after interruption is never accepted state.

Run from the repository root. The executor selects its unique ID; the owner does not provide bookkeeping values.

```sh
python3 .codex/agents/generated/backtest-npy-input-materialization-v1/ledger_update.py status
python3 .codex/agents/generated/backtest-npy-input-materialization-v1/ledger_update.py claim --stage S01 --executor UNIQUE_EXECUTOR_ID
python3 .codex/agents/generated/backtest-npy-input-materialization-v1/ledger_update.py accept --stage S01 --executor SAME_EXECUTOR_ID --checks-passed --evidence reports/S01.md
```

The final command requires a completed report containing actual successful required checks; additional evidence files may be supplied. It does not execute tests or validate their semantic adequacy. It creates the receipt, enables the next stage when entry conditions pass and returns the next allowed ID. Do not run the next stage until its prompt is submitted. Use `resume` only for the same owned claim. `needs-input` and `block` require concrete reasons and readable evidence; they are not substitutes for routine debugging. Hard-blocked stages need canonical revision/reconciliation, never an in-place resume/reset. All claimed/accepted contracts and historical receipts remain immutable.

## Validation profile and verified capability

Serialization remains the installed Prompt Manager `prompt-pack-ledger/v1`, `stage-prompt/v1`, `prompt-pack-receipt/v1`. `claim_capability.evidence` binds the actual updater implementation; its module docstring documents the ownership and atomicity contract.

```sh
python3 .codex/agents/generated/backtest-npy-input-materialization-v1/test_ledger_update.py -v
python3 /Users/daniildegtyarev/.codex/skills/prompt-manager/scripts/validate_pack.py --root /Users/daniildegtyarev/Projects/roehub.com --ledger /Users/daniildegtyarev/Projects/roehub.com/.codex/agents/generated/backtest-npy-input-materialization-v1/iteration-ledger.md --check draft
python3 /Users/daniildegtyarev/.codex/skills/prompt-manager/scripts/validate_pack.py --root /Users/daniildegtyarev/Projects/roehub.com --ledger /Users/daniildegtyarev/Projects/roehub.com/.codex/agents/generated/backtest-npy-input-materialization-v1/iteration-ledger.md --check entry --stage S01
```

The updater internally validates receipts before transitions. Successful output is exit 0 and `status=ok`; the read-only validator requires exit 0 and `status=pass`. Nonzero/malformed/interrupted output is not a pass. The stage validation profile retains repository-selected checks; no check is satisfied merely by the existence of a report or receipt.

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
  "current_stage": "S09",
  "stages": [
    {
      "contract": {
        "id": "S01",
        "title": "Versioned input contracts, persistence expansion and trusted baseline",
        "prompt_path": "01-contracts-and-baseline.md",
        "report_path": "reports/S01.md",
        "receipt_dir": "receipts/S01",
        "depends_on": [],
        "expected_touches": [
          "../../../../src/trading/contexts/backtest_artifacts/application/services/v2/contracts.py",
          "../../../../src/trading/contexts/backtest_artifacts/application/services/v2/artifact_manifest_loader.py",
          "../../../../src/trading/contexts/backtest/application/dto",
          "../../../../src/trading/contexts/backtest/domain/entities/backtest_job.py",
          "../../../../src/trading/contexts/backtest/adapters/outbound/persistence/postgres",
          "../../../../migrations/postgres",
          "../../../../tests/unit/contexts/backtest",
          "../../../../tests/unit/apps/migrations",
          "../../../../docs/architecture/backtest/backtest-service-artifact-runtime-v1.ru.md"
        ],
        "acceptance_criteria": [
          "Old schema-1 manifests and recipe-less jobs remain readable without rewritten hashes or guessed provenance.",
          "New typed contracts and strict schema-2/job-manifest parsers round-trip and reject malformed versions, shapes and identities.",
          "Additive migration is ordered and preserves existing rows; local test proof and unrun infra limits are explicit.",
          "Independent fixture baseline and year baseline source/data strategy are recorded outside Git; no numerical/runtime path is switched on in this stage.",
          "The two-phase canonical prefix-proof format and slot ownership tables are specified and round-trip without forcing payload scans in preflight."
        ],
        "proof_boundary": "Typed/serialized contracts and additive migration behavior on tested fixtures; no generated execution, performance or production migration claim.",
        "validation": {
          "profile": "roehub-focused-gates-and-prompt-pack-artifacts/v1",
          "checks": [
            "uv run pytest -q tests/unit/contexts/backtest/application/services/v2/test_yaml_backtest_artifact_loader_v2.py tests/unit/contexts/backtest/domain/entities/test_backtest_job_entities.py",
            "Extend existing fixture tests and add tests/unit/contexts/backtest/application/services/v2/test_input_recipe_contracts.py for version/identity/round-trip failures. Select migration SQL tests from tests/unit/apps/migrations; inspect the current harness before invoking real Postgres.",
            "Run focused ruff and uv run pyright after implementation. Record any unperformed real migration checks as unavailable, not passed."
          ],
          "requires_user_acceptance": false
        },
        "entry_inputs": [
          {
            "path": "implementation-plan.md",
            "producer_stage": null
          },
          {
            "path": "../../../../src/trading/contexts/backtest_artifacts/application/services/v2/contracts.py",
            "producer_stage": null
          },
          {
            "path": "../../../../src/trading/contexts/backtest/application/dto/runtime_preflight.py",
            "producer_stage": null
          },
          {
            "path": "../../../../src/trading/contexts/backtest/domain/entities/backtest_job.py",
            "producer_stage": null
          },
          {
            "path": "../../../../migrations/postgres/manifest.json",
            "producer_stage": null
          }
        ]
      },
      "status": "accepted",
      "execution_allowed": false,
      "current_authority": true,
      "executor_claim": "codex-s01-20261007-8ea794cf",
      "claimed_at": "2026-10-07T19:23:56.026006Z",
      "transition_receipt": "receipts/S01/receipt-35716927862542b08992d3f44f9a7b82.md",
      "decision_packet": {
        "question": "Required global uv run pyright fails with 28 diagnostics reproduced against pre-S01 sources in foreign market-data tests. Choose explicit scoped-gate exception, authorize minimal foreign type-test repairs, or wait for their owner to repair them; no waiver is inferred.",
        "resume_condition": "Readable owner resolution selects a scoped-pyright exception with the documented baseline evidence, or authorizes minimal foreign repairs followed by a passing global gate, or the foreign owner repairs the diagnostics and global pyright passes. Resume only the same executor claim with recorded resolution evidence.",
        "resolution_evidence": "reports/S01-resolution.md"
      },
      "reason": "Required global uv run pyright fails with 28 diagnostics reproduced against pre-S01 sources in foreign market-data tests. Choose explicit scoped-gate exception, authorize minimal foreign type-test repairs, or wait for their owner to repair them; no waiver is inferred.",
      "evidence": [
        "reports/S01.md"
      ]
    },
    {
      "contract": {
        "id": "S02",
        "title": "Native selected-derivative builder and publication reuse",
        "prompt_path": "02-shared-derivative-builder.md",
        "report_path": "reports/S02.md",
        "receipt_dir": "receipts/S02",
        "depends_on": [
          "S01"
        ],
        "expected_touches": [
          "../../../../src/trading/contexts/backtest_artifacts/application/services/v2/artifact_precompute_runner.py",
          "../../../../src/trading/contexts/backtest_artifacts/application/services/v2/contracts.py",
          "../../../../src/trading/contexts/backtest_artifacts/application/services/v2/hit_times_compute_v2.py",
          "../../../../src/trading/contexts/backtest_artifacts/application/services/v2/signal_chunk_planner_v2.py",
          "../../../../src/trading/contexts/backtest_artifacts/application/services/v2/artifact_precompute_coordinator.py",
          "../../../../src/trading/contexts/backtest_artifacts/application/services/v2/artifact_slot_publisher.py",
          "../../../../tests/unit/contexts/backtest/application/services/v2/test_artifact_precompute_runner_v2.py",
          "../../../../tests/unit/contexts/backtest/application/services/v2/test_hit_times_compute_v2.py",
          "../../../../tests/unit/contexts/backtest/application/services/v2/test_derived_artifact_materialization.py"
        ],
        "acceptance_criteria": [
          "Native publication and selected materialization share the same grid/compute/encoding/writing implementation.",
          "Generated selected arrays match the trusted source-domain baseline exactly, including sparse row IDs, EMA seed history and TP/SL sentinels.",
          "Derived-only mode performs zero canonical candle reads and zero publication-pointer writes; risk none creates zero risk files.",
          "Injected chunk/write failures leave no consumable manifest and respect configured allocation limits."
        ],
        "proof_boundary": "Production builder correctness and file publication atomicity on bounded fixtures, not job lifecycle or year performance.",
        "validation": {
          "profile": "roehub-focused-gates-and-prompt-pack-artifacts/v1",
          "checks": [
            "uv run pytest -q tests/unit/contexts/backtest/application/services/v2/test_artifact_precompute_runner_v2.py tests/unit/contexts/backtest/application/services/v2/test_hit_times_compute_v2.py tests/unit/contexts/backtest/application/services/v2/test_derived_artifact_materialization.py",
            "Use real npy files and production SMA/EMA/WMA calculation for C01/C03/C04. Negative dispatch assertions may use spies; a mocked compute cannot establish parity.",
            "Run focused ruff and uv run pyright; inspect the diff for duplicate compute/writer paths and nested parallelism."
          ],
          "requires_user_acceptance": false
        },
        "entry_inputs": [
          {
            "path": "implementation-plan.md",
            "producer_stage": null
          },
          {
            "path": "../../../../src/trading/contexts/backtest_artifacts/application/services/v2/artifact_precompute_runner.py",
            "producer_stage": null
          },
          {
            "path": "../../../../src/trading/contexts/backtest_artifacts/application/services/v2/hit_times_compute_v2.py",
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
      "executor_claim": "codex-s02-20261007-c4e478d3",
      "claimed_at": "2026-10-07T19:58:25.787008Z",
      "transition_receipt": "receipts/S02/receipt-f18397d4168143fcbf0ae2a731f586f5.md",
      "decision_packet": null
    },
    {
      "contract": {
        "id": "S03",
        "title": "Unified prepared input resolution, validation and mmap loading",
        "prompt_path": "03-prepared-input-loading.md",
        "report_path": "reports/S03.md",
        "receipt_dir": "receipts/S03",
        "depends_on": [
          "S02"
        ],
        "expected_touches": [
          "../../../../src/trading/contexts/backtest/application/ports/artifact_arrays.py",
          "../../../../src/trading/contexts/backtest/adapters/outbound/artifacts_fs",
          "../../../../src/trading/contexts/backtest/application/services/v2/prepare_pools.py",
          "../../../../src/trading/contexts/backtest/application/services/v2/tp_sl_hit_times.py",
          "../../../../src/trading/contexts/backtest/application/dto/prepare_pools.py",
          "../../../../src/trading/contexts/backtest_artifacts/application/services/v2/artifact_manifest_validator.py",
          "../../../../src/trading/contexts/backtest_artifacts/application/services/v2/contracts.py",
          "../../../../tests/unit/contexts/backtest/adapters/outbound/artifacts_fs",
          "../../../../tests/unit/contexts/backtest/application/services/v2/test_prepare_pools_service.py",
          "../../../../tests/unit/contexts/backtest/application/services/v2/test_tp_sl_hit_times_service.py",
          "../../../../tests/unit/contexts/backtest/application/services/v2/test_artifact_manifest_validator_v2.py"
        ],
        "acceptance_criteria": [
          "N/G/M inputs follow one loader/preparation path with exact row identity and numerical equivalence.",
          "No transient input set uses fake slot/current metadata and no candle dataset is duplicated per job.",
          "Missing supported derivatives materialize; corrupt or incompatible declared data fails before scoring.",
          "Path validation remains strict and native no-risk operation does not require hit-time data."
        ],
        "proof_boundary": "File/port/preparation contract equivalence and negative validation on real local fixtures; authorization is additionally verified at API integration later.",
        "validation": {
          "profile": "roehub-focused-gates-and-prompt-pack-artifacts/v1",
          "checks": [
            "uv run pytest -q tests/unit/contexts/backtest/adapters/outbound/artifacts_fs tests/unit/contexts/backtest/application/services/v2/test_prepare_pools_service.py tests/unit/contexts/backtest/application/services/v2/test_tp_sl_hit_times_service.py tests/unit/contexts/backtest/application/services/v2/test_artifact_manifest_validator_v2.py",
            "Exercise C02–C06 with real files, including mixed sparse rows/levels, corrupt declared files, untrusted roots and unchanged published candles.",
            "Run focused ruff and uv run pyright; search all changed port consumers before declaring internal migration complete."
          ],
          "requires_user_acceptance": false
        },
        "entry_inputs": [
          {
            "path": "implementation-plan.md",
            "producer_stage": null
          },
          {
            "path": "../../../../src/trading/contexts/backtest/application/ports/artifact_arrays.py",
            "producer_stage": null
          },
          {
            "path": "../../../../src/trading/contexts/backtest/adapters/outbound/artifacts_fs/artifact_array_loader.py",
            "producer_stage": null
          },
          {
            "path": "../../../../src/trading/contexts/backtest/application/services/v2/prepare_pools.py",
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
      "executor_claim": "codex-s03-20261007-984b7330",
      "claimed_at": "2026-10-07T20:21:02.786055Z",
      "transition_receipt": "receipts/S03/receipt-b61a0495cfc84569bd826daff82f5255.md",
      "decision_packet": null
    },
    {
      "contract": {
        "id": "S04",
        "title": "Full-job preparation, resource admission, pins and attempt cleanup",
        "prompt_path": "04-worker-lifecycle-and-resources.md",
        "report_path": "reports/S04.md",
        "receipt_dir": "receipts/S04",
        "depends_on": [
          "S03"
        ],
        "expected_touches": [
          "../../../../apps/worker/backtest_job_runner/",
          "../../../../src/trading/contexts/backtest/application/services/v2/job_orchestration.py",
          "../../../../src/trading/contexts/backtest/application/services/v2/job_scratch.py",
          "../../../../src/trading/contexts/backtest/application/services/v2/admission.py",
          "../../../../src/trading/contexts/backtest/application/use_cases/backtest_job_worker.py",
          "../../../../src/trading/contexts/backtest/application/ports/backtest_job_repositories.py",
          "../../../../src/trading/contexts/backtest/adapters/outbound/persistence/postgres",
          "../../../../src/trading/contexts/backtest_artifacts/application/services/v2/artifact_slot_publisher.py",
          "../../../../src/trading/contexts/backtest_artifacts/adapters/outbound/persistence/postgres/backtest_job_repository.py",
          "../../../../tests/unit/apps/worker/backtest_job_runner",
          "../../../../tests/unit/apps/worker/test_backtest_job_runner.py",
          "../../../../tests/unit/contexts/backtest/application/use_cases/test_backtest_job_worker_use_case.py",
          "zone: Disposable Postgres pin and lease integration proof",
          "../../../../src/trading/contexts/backtest_artifacts/adapters/outbound/persistence/postgres/gateway.py"
        ],
        "acceptance_criteria": [
          "Exactly one materialization per attempt serves warmup and scoring; no DB candle read occurs.",
          "Success/failure/cancel/timeout/restart recovery leave no owned orphan files, leaked reservations or live-input deletion.",
          "Real concurrent publisher/full/lazy attempts cannot overwrite pinned source inputs; stale owners cannot reclaim live work.",
          "IPC retains recipe and funding fields; timings distinguish preparation, service, parent cleanup and persistence.",
          "Reader admission and both publisher rebuild/deletion use the shared ownership transaction; uncertain expired owners cannot release live files, and prepared proof is durably acknowledged before scoring."
        ],
        "proof_boundary": "Worker subprocess/resource lifecycle and tested local Postgres concurrency; no production installation claim.",
        "validation": {
          "profile": "roehub-focused-gates-and-prompt-pack-artifacts/v1",
          "checks": [
            "uv run pytest -q tests/unit/apps/worker/backtest_job_runner tests/unit/apps/worker/test_backtest_job_runner.py tests/unit/contexts/backtest/application/use_cases/test_backtest_job_worker_use_case.py",
            "Add tests/unit/apps/worker/backtest_job_runner/test_materialization_lifecycle.py and tests/real_infra/backtest/test_artifact_input_lifecycle.py (planned paths). Inspect current real-infra fixtures first; use disposable Postgres resources.",
            "Execute C07/C08/C14 failure/concurrency drills and report live lease recovery independently from mocked process tests. Run focused ruff and uv run pyright."
          ],
          "requires_user_acceptance": false
        },
        "entry_inputs": [
          {
            "path": "implementation-plan.md",
            "producer_stage": null
          },
          {
            "path": "../../../../apps/worker/backtest_job_runner/wiring/modules/child_process.py",
            "producer_stage": null
          },
          {
            "path": "../../../../apps/worker/backtest_job_runner/wiring/modules/full_job_compute.py",
            "producer_stage": null
          },
          {
            "path": "../../../../src/trading/contexts/backtest/application/services/v2/job_orchestration.py",
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
      "executor_claim": "codex-s04-20261007-72ac4f09",
      "claimed_at": "2026-10-07T20:51:54.512397Z",
      "transition_receipt": "receipts/S04/receipt-36b1bd154e4148a79277f592810f1b08.md",
      "decision_packet": null
    },
    {
      "contract": {
        "id": "S05",
        "title": "Durable source recipes, deterministic replay and lazy reconstruction",
        "prompt_path": "05-durable-replay-and-lazy-details.md",
        "report_path": "reports/S05.md",
        "receipt_dir": "receipts/S05",
        "depends_on": [
          "S04"
        ],
        "expected_touches": [
          "../../../../src/trading/contexts/backtest/domain/entities/backtest_job.py",
          "../../../../src/trading/contexts/backtest/application/dto",
          "../../../../src/trading/contexts/backtest/application/use_cases",
          "../../../../src/trading/contexts/backtest/application/services/v2/preflight.py",
          "../../../../src/trading/contexts/backtest/application/services/v2/lazy_trades_detail.py",
          "../../../../src/trading/contexts/backtest/application/services/v2/result_series.py",
          "../../../../src/trading/contexts/backtest/application/ports/lazy_trades_cache.py",
          "../../../../src/trading/contexts/backtest/adapters/outbound/cache_fs",
          "../../../../src/trading/contexts/backtest/adapters/outbound/persistence/postgres",
          "../../../../apps/worker/backtest_job_runner/wiring/modules/lazy_trades_compute.py",
          "../../../../apps/worker/backtest_job_runner/wiring/modules/lazy_trades_child_process.py",
          "../../../../apps/worker/backtest_job_runner/main/lazy_trades_child.py",
          "../../../../tests/unit/contexts/backtest/",
          "zone: Disposable Postgres job provenance and replay verification"
        ],
        "acceptance_criteria": [
          "Post-cleanup and post-restart selected details/trades/series/funding match trusted native results.",
          "Append-only snapshots replay only after original-domain proof; changed history/math versions fail explicitly.",
          "Queued jobs retain original recipe semantics and stale attempts cannot overwrite durable provenance.",
          "Legacy jobs remain readable with established limits; storage policies/paths do not alter request/variant identity.",
          "New successful jobs have durable payload-prefix attestation; header/shape changes from append do not invalidate an identical recorded payload domain."
        ],
        "proof_boundary": "Durable/replay semantics on tested local source snapshots and disposable database; no arbitrary historical-data recovery guarantee.",
        "validation": {
          "profile": "roehub-focused-gates-and-prompt-pack-artifacts/v1",
          "checks": [
            "uv run pytest -q tests/unit/contexts/backtest/application/services/v2/test_lazy_trades_detail_service.py tests/unit/contexts/backtest/adapters/outbound/cache_fs/test_lazy_trades_cache.py tests/unit/contexts/backtest/application/use_cases/test_backtest_jobs_use_case.py tests/unit/contexts/backtest/application/use_cases/test_lazy_trades_materialization_worker_use_case.py",
            "Add tests/unit/contexts/backtest/application/services/v2/test_input_recipe_replay.py and extend tests/real_infra/backtest/test_artifact_input_lifecycle.py for C09–C12, including old rows and funding.",
            "Run focused ruff and uv run pyright; search all artifact pin/cache/metadata serializers to ensure source meanings are consistent."
          ],
          "requires_user_acceptance": false
        },
        "entry_inputs": [
          {
            "path": "implementation-plan.md",
            "producer_stage": null
          },
          {
            "path": "../../../../src/trading/contexts/backtest/application/use_cases/backtest_jobs.py",
            "producer_stage": null
          },
          {
            "path": "../../../../src/trading/contexts/backtest/application/use_cases/backtest_job_worker.py",
            "producer_stage": null
          },
          {
            "path": "../../../../src/trading/contexts/backtest/application/services/v2/lazy_trades_detail.py",
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
      "executor_claim": "codex-s05-20261008-a8136f20",
      "claimed_at": "2026-10-07T22:10:51.593221Z",
      "transition_receipt": "receipts/S05/receipt-0743c7a656704f4eb0cd9a129a1acdd5.md",
      "decision_packet": null
    },
    {
      "contract": {
        "id": "S06",
        "title": "Per-instrument retention policy, preflight readiness and compatible rollout",
        "prompt_path": "06-policy-readiness-and-rollout.md",
        "report_path": "reports/S06.md",
        "receipt_dir": "receipts/S06",
        "depends_on": [
          "S05"
        ],
        "expected_touches": [
          "../../../../src/trading/contexts/backtest_artifacts/adapters/outbound/config/backtest_artifacts_runtime_config.py",
          "../../../../src/trading/contexts/backtest_artifacts/application/use_cases/publish_backtest_artifacts_v2.py",
          "../../../../src/trading/contexts/backtest_artifacts/application/services/v2/artifact_slot_publisher.py",
          "../../../../configs/dev/backtest_artifacts.yaml",
          "../../../../configs/test/backtest_artifacts.yaml",
          "../../../../configs/prod/backtest_artifacts.yaml",
          "../../../../apps/scheduler/backtest_artifact_publisher",
          "../../../../apps/cli/commands/backtest_artifact_publish.py",
          "../../../../apps/api/dto/backtests.py",
          "../../../../apps/api/routes/backtests.py",
          "../../../../apps/api/routes/ui_backtests.py",
          "../../../../apps/api/wiring/modules/backtest.py",
          "../../../../src/trading/contexts/backtest/application/services/v2/preflight.py",
          "../../../../src/trading/contexts/backtest/application/dto/runtime_preflight.py",
          "../../../../tools/release",
          "../../../../configs/installation/generated",
          "../../../../tests/unit/contexts/backtest/adapters/test_backtest_artifacts_runtime_config.py",
          "../../../../tests/unit/apps/api",
          "zone: Existing platform Backtests response compatibility tests"
        ],
        "acceptance_criteria": [
          "Old configs retain precompute behavior; explicit per-instrument on-demand policy creates valid partial publication without requiring derivatives for preflight.",
          "Retention changes do not change financial identity or queued recipe semantics.",
          "Inspected existing clients parse safe responses; idempotent submission and access boundaries remain intact.",
          "Local rollout/rollback evidence distinguishes schema/config/worker directions; no production activation or destructive cleanup occurred."
        ],
        "proof_boundary": "Configuration/API behavior and local migration transition tests; no new Data UI or production rollout.",
        "validation": {
          "profile": "roehub-focused-gates-and-prompt-pack-artifacts/v1",
          "checks": [
            "uv run pytest -q tests/unit/contexts/backtest/adapters/test_backtest_artifacts_runtime_config.py tests/unit/contexts/backtest/application/services/v2/test_backtest_preflight_service.py tests/unit/apps/scheduler/test_backtest_artifact_publisher_app.py tests/unit/apps/cli/test_backtest_artifact_publish_cli.py",
            "Select affected API tests and installation generator checks from current scripts/CI. Exercise C02/C11–C14 and existing platform builder schema tests; verify no response path leak.",
            "Run focused ruff, uv run pyright and generated-config drift checks required by current CI. Use browser-qa-evidence only if a visible client behavior is changed."
          ],
          "requires_user_acceptance": false
        },
        "entry_inputs": [
          {
            "path": "implementation-plan.md",
            "producer_stage": null
          },
          {
            "path": "../../../../src/trading/contexts/backtest_artifacts/adapters/outbound/config/backtest_artifacts_runtime_config.py",
            "producer_stage": null
          },
          {
            "path": "../../../../configs/dev/backtest_artifacts.yaml",
            "producer_stage": null
          },
          {
            "path": "../../../../apps/api/dto/backtests.py",
            "producer_stage": null
          },
          {
            "path": "../../../../apps/api/wiring/modules/backtest.py",
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
      "executor_claim": "codex-s06-20261008-4ef098bc",
      "claimed_at": "2026-10-07T22:46:58.473198Z",
      "transition_receipt": "receipts/S06/receipt-0f9c2b0b055641bfb3a934faff1e888f.md",
      "decision_packet": null
    },
    {
      "contract": {
        "id": "S07",
        "title": "Complete correctness, integration, migration and failure-mode evidence",
        "prompt_path": "07-production-correctness-proof.md",
        "report_path": "reports/S07.md",
        "receipt_dir": "receipts/S07",
        "depends_on": [
          "S06"
        ],
        "expected_touches": [
          "../../../../tests/unit/contexts/backtest",
          "../../../../tests/unit/contexts/backtest_artifacts",
          "../../../../tests/unit/apps/worker/backtest_job_runner",
          "../../../../tests/unit/apps/api",
          "../../../../tests/unit/apps/migrations",
          "../../../../tests/real_infra/backtest",
          "zone: Narrow in-scope production fixes revealed by required proof"
        ],
        "acceptance_criteria": [
          "Every C01–C14 row has actual sufficient evidence or a named unresolved blocker; unresolved required proof prevents stage acceptance.",
          "Native/generated/mixed numerical and result identities agree against an independent oracle; no correctness tolerance was relaxed.",
          "Migration, leases, resource bounds, cleanup and organization/path protections are verified at their actual affected boundaries.",
          "Required gates and applicable independent review are satisfied; foreign work and production systems remain untouched.",
          "Forced ownership races cover both rebuild and previous-slot cleanup, live expired owners, and acknowledged prefix proof before scoring."
        ],
        "proof_boundary": "Local production-boundary correctness and operational safety under tested fixtures; throughput/latency and deployed readiness are separate.",
        "validation": {
          "profile": "roehub-focused-gates-and-prompt-pack-artifacts/v1",
          "checks": [
            "Run focused pytest targets selected in S01–S06 plus the new deterministic/integration cases; record exact commands and counts, including skips.",
            "uv run ruff check <owned-python-paths>; uv run pyright; uv run python -m tools.ci.route_changes ci --changed-files <external-owned-path-list>, then the required affected CI shards.",
            "Run isolated real-infra fixtures under the repository setup contract. A zero-test/all-skipped success is not proof. Preserve failures and classify attribution from evidence."
          ],
          "requires_user_acceptance": false
        },
        "entry_inputs": [
          {
            "path": "implementation-plan.md",
            "producer_stage": null
          },
          {
            "path": "../../../../docs/architecture/backtest/README.md",
            "producer_stage": null
          },
          {
            "path": "../../../../.github/workflows/ci.yml",
            "producer_stage": null
          },
          {
            "path": "../../../../tools/ci/route_changes.py",
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
      "executor_claim": "codex-s07-20261008-c742ab91",
      "claimed_at": "2026-10-07T23:09:30.961673Z",
      "transition_receipt": "receipts/S07/receipt-dbf7758dc5f14098bdf276cc53d5ac43.md",
      "decision_packet": null
    },
    {
      "contract": {
        "id": "S08",
        "title": "BTCUSDT year benchmark of native, generated and mixed NPY inputs",
        "prompt_path": "08-year-native-generated-benchmark.md",
        "report_path": "reports/S08.md",
        "receipt_dir": "receipts/S08",
        "depends_on": [
          "S07"
        ],
        "expected_touches": [
          "../../../../scripts/backtest/full_result_parity.py",
          "../../../../scripts/backtest/run_api_runner_benchmark_parity.py",
          "../../../../src/trading/contexts/backtest/application/services/v2/benchmark_accounting.py",
          "zone: Existing parity comparators only if the new semantic identity requires a tested update"
        ],
        "acceptance_criteria": [
          "W1–W4 use validated real year data and actual production N/G/M file paths with recorded code/config/dataset identities.",
          "Complete selected arrays, top-N and required lazy details agree; independent reference/self-check and resource/cleanup gates pass.",
          "Comparable measured costs and uncertainty are reported, including inclusive generated preparation and disk cost; no invented performance threshold or unsupported speedup claim.",
          "No one-off benchmark script, archive, npy fixture or JIT output remains in the repository; aggregate report is reproducible without pretending external scratch is permanent."
        ],
        "proof_boundary": "Measured local performance/correctness for the specified year workloads and machine; not other instruments, installations, 200-symbol throughput or a latency-SLO pass.",
        "validation": {
          "profile": "roehub-focused-gates-and-prompt-pack-artifacts/v1",
          "checks": [
            "Run the external harness with the frozen production environment and record the literal command/harness hash in reports/S08.md. The command does not exist until the external harness is authored; do not invent an existing repository benchmark entrypoint.",
            "Validate complete top-N using the existing full_result_parity contract with deliberately versioned semantic recipe matching; legacy context validation remains enforced. Run independent self-check for at least two combinations and C03 cases.",
            "Before/after repository status and scratch/database inventory prove no temporary benchmark code/data leaked into Git or the user database. Any comparator changes require focused tests, ruff and pyright as applicable."
          ],
          "requires_user_acceptance": false
        },
        "entry_inputs": [
          {
            "path": "implementation-plan.md",
            "producer_stage": null
          },
          {
            "path": "../../../../docs/architecture/backtest/README.md",
            "producer_stage": null
          },
          {
            "path": "../../../../scripts/backtest/full_result_parity.py",
            "producer_stage": null
          },
          {
            "path": "../../../../scripts/backtest/run_api_runner_benchmark_parity.py",
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
      "executor_claim": "codex-s08-20261008-53ca901e",
      "claimed_at": "2026-10-07T23:36:43.148796Z",
      "transition_receipt": "receipts/S08/receipt-f8be7bfe2eef48648b5c839022d329b2.md",
      "decision_packet": null
    },
    {
      "contract": {
        "id": "S09",
        "title": "Canonical documentation, contract reconciliation and local delivery closure",
        "prompt_path": "09-documentation-and-local-closure.md",
        "report_path": "reports/S09.md",
        "receipt_dir": "receipts/S09",
        "depends_on": [
          "S08"
        ],
        "expected_touches": [
          "../../../../docs/architecture/backtest/README.md",
          "../../../../docs/architecture/backtest/backtest-service-artifact-runtime-v1.ru.md",
          "../../../../docs/architecture/backtest/backtest-service-artifact-runtime-v1.md",
          "../../../../docs/runbooks/offline-release-installation.md",
          "../../../../docs/runbooks/indicators-numba-cache-and-threads.md",
          "../../../../docs/architecture/api/api-errors-and-422-payload-v1.md",
          "../../../../docs/architecture/README.md",
          "../../../../docs/architecture/project-map",
          "zone: Narrow documentation corrections in changed source docstrings"
        ],
        "acceptance_criteria": [
          "Canonical docs describe the implemented contracts and verified operational limits without reviving retired procedures.",
          "All required stage obligations, correctness proofs and year measurements are satisfied and linked; no unassessed required proof is presented as a pass.",
          "Owned/foreign changes and local-vs-production boundaries are explicit; generated indices and final consistency checks pass.",
          "Final receipt has no next stage; the runner completes the ledger only after its actual lifecycle obligations are met."
        ],
        "proof_boundary": "Local implementation/documentation closure with linked verification and benchmark evidence; no Git publication or production deployment.",
        "validation": {
          "profile": "roehub-focused-gates-and-prompt-pack-artifacts/v1",
          "checks": [
            "uv run python -m tools.docs.generate_docs_index --check",
            "uv run python -m tools.docs.generate_project_map --check",
            "Reconcile required affected gates with the last changed code; rerun only after relevant edits or unresolved concerns. Run pack receipt validation against actual live files and immutable evidence under the supported updater before transition."
          ],
          "requires_user_acceptance": false
        },
        "entry_inputs": [
          {
            "path": "implementation-plan.md",
            "producer_stage": null
          },
          {
            "path": "../../../../docs/architecture/backtest/README.md",
            "producer_stage": null
          },
          {
            "path": "../../../../docs/architecture/backtest/backtest-service-artifact-runtime-v1.ru.md",
            "producer_stage": null
          },
          {
            "path": "../../../../docs/runbooks/offline-release-installation.md",
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
      "executor_claim": "codex-s09-20261008-6947",
      "claimed_at": "2026-10-08T00:15:34.824553Z",
      "transition_receipt": "receipts/S09/receipt-5af9af812d9f4e599c6e1b0a319c5acb.md",
      "decision_packet": null
    }
  ],
  "claim_capability": {
    "mechanism": "Pack-local ledger_update.py: exclusive POSIX flock on the stable pack directory; validate and reread inside the lock; persistent executor claim; fsync plus atomic replacement; no automatic claim takeover.",
    "evidence": {
      "path": "ledger_update.py",
      "sha256": "dbae4cb40c4b69572bd275e2350fa1af79489b1d7a7df4fe3654d40429130550"
    }
  },
  "execution_history": [
    {
      "at": "2026-10-07T19:23:56.026006Z",
      "action": "claim",
      "stage": "S01",
      "executor": "codex-s01-20261007-8ea794cf",
      "receipt": null,
      "reason": null
    },
    {
      "at": "2026-10-07T19:44:52.235484Z",
      "action": "needs-input",
      "stage": "S01",
      "executor": "codex-s01-20261007-8ea794cf",
      "receipt": null,
      "reason": "Required global uv run pyright fails with 28 diagnostics reproduced against pre-S01 sources in foreign market-data tests. Choose explicit scoped-gate exception, authorize minimal foreign type-test repairs, or wait for their owner to repair them; no waiver is inferred."
    },
    {
      "at": "2026-10-07T19:52:27.292323Z",
      "action": "resume",
      "stage": "S01",
      "executor": "codex-s01-20261007-8ea794cf",
      "receipt": null,
      "reason": null
    },
    {
      "at": "2026-10-07T19:56:48.449725Z",
      "action": "accept",
      "stage": "S01",
      "executor": "codex-s01-20261007-8ea794cf",
      "receipt": "receipts/S01/receipt-35716927862542b08992d3f44f9a7b82.md",
      "reason": null
    },
    {
      "at": "2026-10-07T19:58:25.787008Z",
      "action": "claim",
      "stage": "S02",
      "executor": "codex-s02-20261007-c4e478d3",
      "receipt": null,
      "reason": null
    },
    {
      "at": "2026-10-07T20:18:12.501806Z",
      "action": "accept",
      "stage": "S02",
      "executor": "codex-s02-20261007-c4e478d3",
      "receipt": "receipts/S02/receipt-f18397d4168143fcbf0ae2a731f586f5.md",
      "reason": null
    },
    {
      "at": "2026-10-07T20:21:02.786055Z",
      "action": "claim",
      "stage": "S03",
      "executor": "codex-s03-20261007-984b7330",
      "receipt": null,
      "reason": null
    },
    {
      "at": "2026-10-07T20:48:15.892657Z",
      "action": "accept",
      "stage": "S03",
      "executor": "codex-s03-20261007-984b7330",
      "receipt": "receipts/S03/receipt-b61a0495cfc84569bd826daff82f5255.md",
      "reason": null
    },
    {
      "at": "2026-10-07T20:51:54.512397Z",
      "action": "claim",
      "stage": "S04",
      "executor": "codex-s04-20261007-72ac4f09",
      "receipt": null,
      "reason": null
    },
    {
      "at": "2026-10-07T22:05:15.325778Z",
      "action": "accept",
      "stage": "S04",
      "executor": "codex-s04-20261007-72ac4f09",
      "receipt": "receipts/S04/receipt-36b1bd154e4148a79277f592810f1b08.md",
      "reason": null
    },
    {
      "at": "2026-10-07T22:10:51.593221Z",
      "action": "claim",
      "stage": "S05",
      "executor": "codex-s05-20261008-a8136f20",
      "receipt": null,
      "reason": null
    },
    {
      "at": "2026-10-07T22:45:30.920784Z",
      "action": "accept",
      "stage": "S05",
      "executor": "codex-s05-20261008-a8136f20",
      "receipt": "receipts/S05/receipt-0743c7a656704f4eb0cd9a129a1acdd5.md",
      "reason": null
    },
    {
      "at": "2026-10-07T22:46:58.473198Z",
      "action": "claim",
      "stage": "S06",
      "executor": "codex-s06-20261008-4ef098bc",
      "receipt": null,
      "reason": null
    },
    {
      "at": "2026-10-07T23:07:49.430504Z",
      "action": "accept",
      "stage": "S06",
      "executor": "codex-s06-20261008-4ef098bc",
      "receipt": "receipts/S06/receipt-0f9c2b0b055641bfb3a934faff1e888f.md",
      "reason": null
    },
    {
      "at": "2026-10-07T23:09:30.961673Z",
      "action": "claim",
      "stage": "S07",
      "executor": "codex-s07-20261008-c742ab91",
      "receipt": null,
      "reason": null
    },
    {
      "at": "2026-10-07T23:34:57.731448Z",
      "action": "accept",
      "stage": "S07",
      "executor": "codex-s07-20261008-c742ab91",
      "receipt": "receipts/S07/receipt-dbf7758dc5f14098bdf276cc53d5ac43.md",
      "reason": null
    },
    {
      "at": "2026-10-07T23:36:43.148796Z",
      "action": "claim",
      "stage": "S08",
      "executor": "codex-s08-20261008-53ca901e",
      "receipt": null,
      "reason": null
    },
    {
      "at": "2026-10-08T00:11:00.648719Z",
      "action": "accept",
      "stage": "S08",
      "executor": "codex-s08-20261008-53ca901e",
      "receipt": "receipts/S08/receipt-f8be7bfe2eef48648b5c839022d329b2.md",
      "reason": null
    },
    {
      "at": "2026-10-08T00:15:34.824553Z",
      "action": "claim",
      "stage": "S09",
      "executor": "codex-s09-20261008-6947",
      "receipt": null,
      "reason": null
    },
    {
      "at": "2026-10-08T00:42:35.041641Z",
      "action": "accept",
      "stage": "S09",
      "executor": "codex-s09-20261008-6947",
      "receipt": "receipts/S09/receipt-5af9af812d9f4e599c6e1b0a319c5acb.md",
      "reason": null
    }
  ]
}
```

## Authoring iteration history

| Iteration | Date | Work and evidence | Outcome |
| --- | --- | --- | --- |
| A01 | 2026-10-07 | Read relevant repository policy, skills, runtime contracts, builder/loader/worker/replay/config/persistence code and current test/CI entrypoints. Inspected baseline HEAD and foreign changes. | Confirmed slot coupling, mandatory hit-times resolution, explicit-path loader changes, worker re-preflight and replay/resource requirements. |
| A02 | 2026-10-07 | Authored implementation plan plus S01–S09 prompts; recorded exact changes, C01–C14 tests and W1–W4 year benchmark. | Accepted product direction encoded; no implementation or measurements claimed; benchmark scratch/harness remains outside Git. |
| A03 | 2026-10-07 | Saved artifact validation: draft check exited 0 with status=pass; source-entrypoint check found all declared current paths; entry check exited 1 because claim capability is absent. | draft_valid; entry_ready=false. No implementation tests or benchmarks were run. |
| A04 | 2026-10-07 | Independent read-only architecture/lifecycle review found two high-priority design gaps: count-only slot coordination and undefined prefix-proof bytes/timing. | Both resolved at plan level; independent focused follow-up confirmed their resolution. |
| A05 | 2026-10-07 | Follow-up identified a medium stale publisher-precheck gap. Added coordinate publication serialization, current-pointer/target revalidation and an incremental-source reader reservation; C08 and S04/S07/S09 synchronized. Focused self-review checked this repair. | No remaining identified semantic blocker in authored scope. Final draft validator and contract/path checks pass; entry_ready remains false solely for the documented updater capability gap. Code/runtime/benchmark proof remains future stage work. |

| A06 | 2026-10-07 | Owner explicitly requested unblocking and sequential prompt submission. Added pack-local atomic updater and nine disposable tests; synchronized all prompts and the plan, preserved prior draft/history and bound updater bytes. | Nine local updater tests passed. Saved draft and S01 entry validators both returned status=pass (exit 0); all nine prompt contracts match the ledger. S01 is entry_ready; no implementation claim or receipt exists. |

Future stage iteration reports belong in the exact report path from each row. Record attempts, command outcomes, decisions, deviations, owned paths, residual risks and immutable receipt links without erasing prior history. Use stage status transitions only through the supported updater and canonical runner lifecycle. Do not reset this ledger after its first claim.
