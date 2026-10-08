---
{
  "schema_version": "stage-prompt/v1",
  "prompt_pack_execution": {
    "plan_doc": "implementation-plan.md",
    "prompt_pack_dir": ".",
    "stage_ledger": "iteration-ledger.md"
  },
  "stage_contract": {
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
  }
}
---

# S06 — Per-instrument retention policy, preflight readiness and compatible rollout

Relevant implementation-plan sections: 4.6, 5, 6 (C02/C11/C12/C13/C14), 9 canonical documentation.

## Execution authority and state

Read the live ledger before doing anything. It alone owns statuses, claims and decision packets. Execute exactly S06 when the owner submits this prompt, then stop with the next prompt link. The owner sends prompts sequentially; no separate stage acceptance is required. Follow `staged-plan-runner` for lifecycle rules and use this pack's `ledger_update.py` for every execution-state mutation. The former updater gap was resolved by explicit owner instruction on 2026-10-07. Never edit active ledger JSON by hand, steal another executor's claim, or expire a claim automatically.

Once a stage is validly claimed, complete the authorized implementation, verification and in-scope fixes. Routine reversible engineering choices stay delegated. Do not ask the owner to approve an unchanged design or each next stage. A real missing external prerequisite blocks only its dependent proof. A new material owner decision must be distinguished from a failed test, implementation detail or earlier-stage output. No publication, deployment, credential change, destructive existing-artifact cleanup, Goal, branch or worktree is implied.

## Context acquisition

Always read repository `AGENTS.md`, this stage's live ledger record and the implementation plan's authority, invariant, compatibility and relevant stage sections. Refresh current source and ownership before editing. The plan's context table establishes observed baseline, not permission to overwrite concurrent changes. Read the task entrypoints below and only conditional bundles needed for the named change. Earlier reports are evidence; implementation decisions remain constrained by this plan and current user corrections.

## Conditional skills

- `backend-quality-gates`: when Python implementation/gates begin; select checks from current CI and affected boundaries.
- `contract-impact-analysis`: before changing API/DTO, file schema, persistence, config, port, cache or identity guarantees; retain classifications and migration directions in the report.
- `backend-performance-evidence`: for measurements, timing contracts or performance claims; mandatory in S08.
- `root-cause-debugging`: when a concrete failure's cause is unknown; fix only after causal evidence.
- `browser-qa-evidence`: if the implementation changes browser-visible behavior; use `playwright-cli` for terminal browser mechanics only when no user-selected browser overrides it.
- `architecture-design`: only if a newly evidenced invariant requires refining an unsettled design detail; do not reopen accepted npy/materialization/storage choices merely by preference.
- `prompt-manager`: only for an authorized artifact revision; preserve claimed/accepted contracts and use runner reconciliation instead of rewriting live stage IDs.

## Implementation constraints

Use production code and real contract boundaries. Reuse the existing builder, indicator computation, hit-time calculation, validators and mmap loader. No compatibility wrappers, fake slot/hash metadata, npy-to-CanonicalCandleReader adapter, generic helper/provider framework, duplicate numerical pipeline or direct ClickHouse fallback. No benchmark scaffolding or large data in the repository. Do not change financial precision, timeline/sentinel semantics, funding, canonical variant identity, ranking or supported direction policy for speed.

Inspect existing tests before adding a named planned module; extend the current owner when it already proves that boundary. Do not leave TODO-driven production behavior, catch-all silent fallbacks or success-shaped placeholder evidence. Preserve foreign paths/hunks and record necessary outside-expected paths with their reason. Existing diagrams/docs/history do not authorize unrelated work.

## Reporting and transition

Write the stage report at the exact contract path in English. Include actual created/modified/deleted paths, outside-expected changes with reasons, foreign changes excluded, exact commands/results/counts, actual evidence locations, compatibility classifications, remaining risk and next-stage inputs. Never log credentials, tokens, raw provider payloads or environment dumps. Keep large benchmark scripts/data/results outside Git; only bounded aggregate proof belongs in reports.

After required proof passes, use `ledger_update.py accept` with the actual report/evidence and `--checks-passed`. The command generates a new immutable receipt, binds live plan/prompt/report/evidence bytes, validates the successor, and commits the ledger transition under its directory lock. No manual receipt/hash construction is needed. A nonzero command result is not a transition. The next eligible stage is enabled but not started. End this stage's turn with its result and the next prompt link; the owner will submit that prompt separately. No additional acceptance question is required. Never claim a pass for missing proof.

## Task entrypoints and conditional source bundle

- `src/trading/contexts/backtest_artifacts/adapters/outbound/config/backtest_artifacts_runtime_config.py`
- `configs/dev/backtest_artifacts.yaml`
- `apps/api/dto/backtests.py`
- `apps/api/wiring/modules/backtest.py`
- `src/trading/contexts/backtest_artifacts/application/use_cases/publish_backtest_artifacts_v2.py`
- `apps/scheduler/backtest_artifact_publisher/wiring/modules/backtest_artifact_publisher.py`
- `apps/cli/commands/backtest_artifact_publish.py`
- `tools/release/generate_installation_config.py`
- `apps/platform-web/src/builder-api.ts`
- `apps/api/routes/ui_backtests.py`

Read `reports/S05.md` produced by S05 before claiming. Its absence now is a deferred dependency, not an owner-input question.

## Required work

1. Implement typed per-coordinate signals/hit_times precompute or on_demand policy in existing configuration. Omitted policy retains current behavior. Keep supported indicators and allowed risk grids independent of stored coverage. No public strategy compute-mode switch is necessary.

2. Update to_precompute_runtime_settings/to_validation_spec and publisher/scheduler/CLI paths so candles and mappings are maintained while derivatives follow retention policy. On-demand preparation never publishes; old files are not destructively deleted during migration.

3. Split semantic result-affecting config identity from retention/execution-storage settings using the S01 versioned contract. Preserve existing jobs/legacy hash decoding and freeze effective policy/recipe where appropriate.

4. Expose bounded input_readiness and estimates through preflight DTOs. Preflight remains read-only and cheap: no derivative compute/write. Preserve old public metadata meaning and required fields, error envelope, access and idempotency.

5. Regenerate installation configurations from current owned source inputs; inspect generator policy and avoid editing generated files independently. Update API/worker composition consistently.

6. Rehearse reader-before-writer enablement and old/new schema/config/queue combinations locally. New schema-aware readers precede partial publications. Record the post-new-write rollback restriction; do not enable an unspecified installation.

## Verification and acceptance

- uv run pytest -q tests/unit/contexts/backtest/adapters/test_backtest_artifacts_runtime_config.py tests/unit/contexts/backtest/application/services/v2/test_backtest_preflight_service.py tests/unit/apps/scheduler/test_backtest_artifact_publisher_app.py tests/unit/apps/cli/test_backtest_artifact_publish_cli.py

- Select affected API tests and installation generator checks from current scripts/CI. Exercise C02/C11–C14 and existing platform builder schema tests; verify no response path leak.

- Run focused ruff, uv run pyright and generated-config drift checks required by current CI. Use browser-qa-evidence only if a visible client behavior is changed.

All acceptance criteria in the front-matter contract must be evidenced. Configuration/API behavior and local migration transition tests; no new Data UI or production rollout.

## Adjacent-stage handoff

S07 receives the enabled local feature path and a complete contract-impact matrix with remaining proof gaps.

## Resolved mixed-version operational constraint

The ownership protocol changes both reader admission and destructive publisher/cleanup paths. Before enabling recipe-aware execution or partial publication in an installation, drain/upgrade every old publisher and cleanup process as well as incompatible workers. An old COUNT-only publisher cannot safely coexist with new lazy-reader reservations. This is documented and rehearsed locally, not a deployment authorization.

## Journal commands for this stage

Run from the repository root. Choose a unique executor ID for this execution and retain it for resuming the same claim; do not reuse another executor's ID. The commands below use placeholders that the executor substitutes, not text the owner must provide.

```sh
python3 .codex/agents/generated/backtest-npy-input-materialization-v1/ledger_update.py claim --stage S06 --executor UNIQUE_EXECUTOR_ID
```

Only after the claim succeeds, perform the stage. Write the English stage report to `reports/S06.md`, including exact checks, results, owned changes and proof limits. Then, only when all required checks actually passed:

```sh
python3 .codex/agents/generated/backtest-npy-input-materialization-v1/ledger_update.py accept --stage S06 --executor SAME_EXECUTOR_ID --checks-passed --evidence reports/S06.md
```

Evidence paths resolve relative to this pack. Supply additional `--evidence` paths when needed. The report itself may serve as bounded check evidence when it records actual commands/results; a bare pass assertion is insufficient. This command does not run tests or establish their meaning.

Use `resume --stage S06 --executor SAME_EXECUTOR_ID` only for your own existing claim. For an actual new owner decision, `needs-input` requires `--reason`, `--resume-condition` and readable `--evidence`; resume then also requires `--resolution` pointing to the recorded answer. For a concrete hard blocker, `block` requires `--reason` and readable `--evidence`. A hard-blocked instance is terminal and needs a separately authored revision under the canonical lifecycle; do not reset it. Routine implementation problems should be diagnosed and fixed within the current claim, not converted into owner approvals. `status` is read-only. Every failed command leaves the prior ledger intact; an unreferenced immutable receipt after interruption is evidence only and must not be treated as acceptance.
