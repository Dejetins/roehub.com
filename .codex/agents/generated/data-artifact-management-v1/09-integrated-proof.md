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
    "title": "Integrated verification and local operational handoff",
    "prompt_path": "09-integrated-proof.md",
    "report_path": "reports/S09.md",
    "receipt_dir": "receipts/S09",
    "depends_on": [
      "S08"
    ],
    "expected_touches": [
      "../../../../tests/real_infra/backtest",
      "../../../../tests/unit/contexts/backtest",
      "../../../../apps/navigator-web/e2e",
      "../../../../docs/architecture/backtest/backtest-service-artifact-runtime-v1.ru.md",
      "../../../../docs/architecture/market_data/market-data-reference-api-v1.md",
      "../../../../docs/runbooks/offline-release-installation.md",
      "../../../../apps/navigator-web/README.md"
    ],
    "acceptance_criteria": [
      "Every R01–R13 has actual evidence or an explicit failing/unavailable result; no accepted receipt while a required boundary proof is absent.",
      "Disposable end-to-end ingestion -> base all-timeframe NPY -> catalog-selected derivatives -> scheduled/manual update -> Backtest reuse/materialization works with preserved source identity.",
      "Real concurrency/restart/crash/disk/quota/pin/cleanup/auth tests show bounded recovery and preserved last-good artifacts; no real-user data is reseeded or cleaned.",
      "Final browser regression covers existing candle streaming/history plus all artifact/settings/Backtest flows; resource limits and proof scope are reported without invented capacity claims.",
      "Canonical docs and current installation operations reflect implemented behavior; independent production-risk-review has no unresolved release-blocking findings within this local scope; publication remains unauthorized."
    ],
    "proof_boundary": "Integrated local disposable implementation evidence, compatibility and handoff only; no production delivery, full-market capacity or week-long live schedule claim.",
    "validation": {
      "profile": "roehub-focused-gates-and-prompt-pack-artifacts/v1",
      "checks": [
        "Select full affected CI shards/static checks using tools/ci/route_changes.py and current .github/workflows/ci.yml; record actual commands, revisions and failures.",
        "Run real-infra artifact lifecycle and newly added orchestration/authorization tests in disposable stores plus current Navigator browser flows.",
        "Run docs index/project-map checks only as routed; if generators are required, inspect owned changes and preserve unrelated docs work.",
        "Independent production-risk-review for integrated trust/concurrency/recovery; no recursive review. Do not execute release/deployment workflows."
      ],
      "requires_user_acceptance": false
    },
    "entry_inputs": [
      {
        "path": "implementation-plan.md",
        "producer_stage": null
      },
      {
        "path": "../../../../AGENTS.md",
        "producer_stage": null
      },
      {
        "path": "../../../../tests/real_infra/backtest/test_artifact_input_lifecycle.py",
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
        "path": "../../../../docs/runbooks/offline-release-installation.md",
        "producer_stage": null
      },
      {
        "path": "../../../../apps/navigator-web/README.md",
        "producer_stage": null
      },
      {
        "path": "reports/S08.md",
        "producer_stage": "S08"
      }
    ]
  },
  "context": {
    "always_read": [
      "iteration-ledger.md",
      "implementation-plan.md",
      "../../../../AGENTS.md"
    ],
    "task_entrypoints": [
      "../../../../tests/real_infra/backtest/test_artifact_input_lifecycle.py",
      "../../../../tools/ci/route_changes.py",
      "../../../../.github/workflows/ci.yml",
      "../../../../docs/runbooks/offline-release-installation.md",
      "../../../../apps/navigator-web/README.md"
    ],
    "conditional_sources": {
      "API/UI": "../../../../docs/architecture/apps/web/roehub-data-loading-contract-v1.md",
      "NPY consumers": "../../../../docs/architecture/backtest/backtest-service-artifact-runtime-v1.ru.md",
      "cross-context": "../../../../docs/architecture/project-map/AGENT_GUIDE.md",
      "local runtime": "../../../../docs/runbooks/navigator-local-market-data-recovery.md"
    },
    "consult_if_needed": [
      "../../../../docs/architecture/platform/artifact-store-v1.md"
    ]
  }
}
---

# S09 — Integrated verification and local operational handoff

## Task and scope

Execute exactly stage S09 of the linked plan when the user selects this prompt. Read the canonical journal before doing anything that changes product state. Its mutable status/claim/decision fields are authoritative; this prompt does not copy them. The owner approved the complete plan and technical direction on 2026-10-08; no repeated unchanged-plan approval is required. Preserve all accepted requirements, especially mandatory all-timeframe base NPY, common 1/3/7-day cadence and all Backtests indicators selected by default with checkbox exclusions.

Verify the integrated result against the entire requirement matrix, including rejected alternative behaviors. Prefer fixing narrow in-scope defects over weakening tests. Record unperformed infrastructure/provider tests honestly. Controlled clock tests can prove cadence logic but not a week of operations. Final report distinguishes implemented/verified/unavailable and gives local startup/recovery instructions from the now-current implementation, never from the retired runtime. Stop at local implementation handoff; no commit/PR/merge/deploy.

## Context and entry

Read `always_read`, the current stage contract and its listed task entrypoints; open conditional sources only when their boundary is crossed. Read the predecessor report where declared, then inspect actual produced code/contracts before use. A missing future report is a dependency to be produced by its stage, not a request for the owner to author it. Follow AGENTS.md source/foreign-change rules and refresh affected diffs before edits. Reports and new receipt directories are authorized stage outputs in addition to expected product touch zones. Resolve exact new source/test paths inside those zones; do not invent a pre-existing file or overwrite foreign hunks.

Use `staged-plan-runner` for lifecycle/claim/transition. Read the available skill and canonical lifecycle in the current session. Run the bundled validator for this journal and this stage's entry before using the existing directory-local updater. The executable path is `.codex/agents/generated/data-artifact-management-v1/ledger_update.py`; its `claim_capability` binding is in the journal. Its exclusive directory flock, re-read and atomic replacement protect claims; a read/write/read by hand does not. Do not activate or claim another pack.

## Work and proof

Implement only the task and `stage_contract.acceptance_criteria`. The contract front matter carries the exact touch zones, acceptance, entry inputs, report path and validation instructions; those are executable requirements, not illustrative boilerplate. Select current commands from real package/CI sources. No dependency installation, private-provider access, production migration, user-data deletion, full-catalog compute or unrelated runtime restart follows implicitly. Use disposable fixtures for destructive/failure cases. Preserve the selected Navigator client and keep apps/platform-web unchanged unless the owner explicitly expands scope.

Use `backend-quality-gates` when Python changes need verification; `contract-impact-analysis` when a relied-on API/persistence/config/consumer boundary changes; `root-cause-debugging` only for an unexplained failure; `backend-performance-evidence` only for measurement claims. Use browser-qa-evidence for user-flow proof and current selected-browser onboarding; load craft skills only for materially changed layout/accessibility/copy. This is implementation of an accepted existing-screen design, not an automatic Product Design ideation task. Independent review follows AGENTS.md and this stage's checks, not a recursive reviewer chain.

## Decisions and stop conditions

Routine reversible implementation choices stay delegated; do not reopen all-indicator defaults, all-timeframe base coverage or cadence because of personal preference. Report evidence if a current contract makes a requirement incompatible and preserve the accepted intent. A material new owner decision uses the journal's decision packet with a precise resume condition. Missing permissions/current required policy, unresolved foreign ownership, unavailable required proof or unsafe slot/root state blocks only dependent work; continue independent authorized diagnosis. Do not label required missing review/runtime evidence passed, steal claims, reset terminal stages or invent a new execution engine.

## Handoff

Write `reports/S09.md` with requirement IDs addressed; actual created/modified/deleted paths and outside-zone reasons; excluded foreign edits; observed commands/results; review findings/disposition; compatibility (`none`, `compatible-change`, `breaking-change`, `unknown`); proof boundary and residual risks; exact products for the next stage. Keep secrets, credentials, raw provider data and environment dumps out of evidence.

Before acceptance, validate the actual stage results and immutable receipt bindings using the existing updater under the canonical lifecycle. `--checks-passed` is an executor assertion supported by real report/evidence, never a replacement for tests. The updater cannot accept a source-required user-acceptance gate; there is none prescribed by this pack. If newly required, use runner reconciliation rather than bypassing it. Recheck the next stage's entry inputs, current source paths and declared decisions; permission for that row does not authorize automatic continuation. Never edit an accepted plan/prompt/report to make an old receipt pass.

Finish in Russian with scope/results, limits and the exact next prompt link, then stop. On S09 there is no successor: complete the journal only after all required stages are satisfied and return the final local handoff. No publication, release or production delivery claim is authorized.
