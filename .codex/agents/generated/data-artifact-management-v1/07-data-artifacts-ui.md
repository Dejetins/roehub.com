---
{
  "schema_version": "stage-prompt/v1",
  "prompt_pack_execution": {
    "plan_doc": "implementation-plan.md",
    "prompt_pack_dir": ".",
    "stage_ledger": "iteration-ledger.md"
  },
  "stage_contract": {
    "id": "S07",
    "title": "Data artifact management UI",
    "prompt_path": "07-data-artifacts-ui.md",
    "report_path": "reports/S07.md",
    "receipt_dir": "receipts/S07",
    "depends_on": [
      "S06"
    ],
    "expected_touches": [
      "../../../../apps/navigator-web/src",
      "../../../../apps/navigator-web/e2e",
      "../../../../packages/web-contracts/src",
      "../../../../tools/qa/navigator_preview.py"
    ],
    "acceptance_criteria": [
      "Candles / Artifacts switch preserves common URL selection/filter context and existing candle history/stream behavior in the current Navigator geometry.",
      "Artifact rows/inspector show separate measured base and derivative progress, true source coverage, next/last run, profile and bounded history from real APIs.",
      "Indicator catalog starts with all available entries checked, supports search/unchecking/all-none and saved exclusions; no hardcoded catalog.",
      "Single/bulk actions support mixed values, profile/cadence changes, Update now, precompute off retaining base; stale snapshots cannot issue commands.",
      "Real-browser desktop/narrow, keyboard, errors/loading and subject-isolation proof passes on disposable data; build/typecheck/focused tests pass."
    ],
    "proof_boundary": "Navigator candidate UI and real API integration on tested browser/disposable fixture; not accepted-client migration or deployment.",
    "validation": {
      "profile": "roehub-focused-gates-and-prompt-pack-artifacts/v1",
      "checks": [
        "Use the installed candidate scripts from apps/navigator-web: node_modules/.bin/tsc --noEmit; node_modules/.bin/vite build; node_modules/.bin/vitest run with affected Data/API/profile tests.",
        "Apply browser-qa-evidence for actual flows; use in-app browser if selected/available, otherwise playwright-cli per current onboarding. No unnecessary full browser skill for screenshot-only baseline.",
        "Cover all-default/exclusion, 1/3/7 schedule, initial/waiting/unknown progress, last-good plus error, retry, bulk mixed state, URL restore and current candle/stream controls.",
        "Use better-layout, better-accessibility and better-writing only at their affected boundaries; preserve tokens and data-loading contract."
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
        "path": "../../../../apps/navigator-web/src/data-page.tsx",
        "producer_stage": null
      },
      {
        "path": "../../../../apps/navigator-web/src/data-downloads.tsx",
        "producer_stage": null
      },
      {
        "path": "../../../../apps/navigator-web/src/data-workspace-api.ts",
        "producer_stage": null
      },
      {
        "path": "../../../../apps/navigator-web/src/data-workspace.css",
        "producer_stage": null
      },
      {
        "path": "../../../../apps/navigator-web/src/builder-api.ts",
        "producer_stage": null
      },
      {
        "path": "../../../../apps/navigator-web/src/data-page.test.tsx",
        "producer_stage": null
      },
      {
        "path": "reports/S06.md",
        "producer_stage": "S06"
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
      "../../../../apps/navigator-web/src/data-page.tsx",
      "../../../../apps/navigator-web/src/data-downloads.tsx",
      "../../../../apps/navigator-web/src/data-workspace-api.ts",
      "../../../../apps/navigator-web/src/data-workspace.css",
      "../../../../apps/navigator-web/src/builder-api.ts",
      "../../../../apps/navigator-web/src/data-page.test.tsx"
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

# S07 — Data artifact management UI

## Task and scope

Execute exactly stage S07 of the linked plan when the user selects this prompt. Read the canonical journal before doing anything that changes product state. Its mutable status/claim/decision fields are authoritative; this prompt does not copy them. The owner approved the complete plan and technical direction on 2026-10-08; no repeated unchanged-plan approval is required. Preserve all accepted requirements, especially mandatory all-timeframe base NPY, common 1/3/7-day cadence and all Backtests indicators selected by default with checkbox exclusions.

Implement the approved mode-switch design directly on the selected existing Data screen; no unrelated redesign or image-ideation prerequisite. Reuse candle progress styling and compact current controls. Honor snapshot/query cancellation/private cache constraints. Use accessible progressbar labels and unknown-total state. Keep profile editing a draft with save/cancel and restore focus. Inspect all foreign dirty hunks first; restart only a specifically authorized candidate runtime, never a reseeding fixture supervisor.

## Context and entry

Read `always_read`, the current stage contract and its listed task entrypoints; open conditional sources only when their boundary is crossed. Read the predecessor report where declared, then inspect actual produced code/contracts before use. A missing future report is a dependency to be produced by its stage, not a request for the owner to author it. Follow AGENTS.md source/foreign-change rules and refresh affected diffs before edits. Reports and new receipt directories are authorized stage outputs in addition to expected product touch zones. Resolve exact new source/test paths inside those zones; do not invent a pre-existing file or overwrite foreign hunks.

Use `staged-plan-runner` for lifecycle/claim/transition. Read the available skill and canonical lifecycle in the current session. Run the bundled validator for this journal and this stage's entry before using the existing directory-local updater. The executable path is `.codex/agents/generated/data-artifact-management-v1/ledger_update.py`; its `claim_capability` binding is in the journal. Its exclusive directory flock, re-read and atomic replacement protect claims; a read/write/read by hand does not. Do not activate or claim another pack.

## Work and proof

Implement only the task and `stage_contract.acceptance_criteria`. The contract front matter carries the exact touch zones, acceptance, entry inputs, report path and validation instructions; those are executable requirements, not illustrative boilerplate. Select current commands from real package/CI sources. No dependency installation, private-provider access, production migration, user-data deletion, full-catalog compute or unrelated runtime restart follows implicitly. Use disposable fixtures for destructive/failure cases. Preserve the selected Navigator client and keep apps/platform-web unchanged unless the owner explicitly expands scope.

Use `backend-quality-gates` when Python changes need verification; `contract-impact-analysis` when a relied-on API/persistence/config/consumer boundary changes; `root-cause-debugging` only for an unexplained failure; `backend-performance-evidence` only for measurement claims. Use browser-qa-evidence for user-flow proof and current selected-browser onboarding; load craft skills only for materially changed layout/accessibility/copy. This is implementation of an accepted existing-screen design, not an automatic Product Design ideation task. Independent review follows AGENTS.md and this stage's checks, not a recursive reviewer chain.

## Decisions and stop conditions

Routine reversible implementation choices stay delegated; do not reopen all-indicator defaults, all-timeframe base coverage or cadence because of personal preference. Report evidence if a current contract makes a requirement incompatible and preserve the accepted intent. A material new owner decision uses the journal's decision packet with a precise resume condition. Missing permissions/current required policy, unresolved foreign ownership, unavailable required proof or unsafe slot/root state blocks only dependent work; continue independent authorized diagnosis. Do not label required missing review/runtime evidence passed, steal claims, reset terminal stages or invent a new execution engine.

## Handoff

Write `reports/S07.md` with requirement IDs addressed; actual created/modified/deleted paths and outside-zone reasons; excluded foreign edits; observed commands/results; review findings/disposition; compatibility (`none`, `compatible-change`, `breaking-change`, `unknown`); proof boundary and residual risks; exact products for the next stage. Keep secrets, credentials, raw provider data and environment dumps out of evidence.

Before acceptance, validate the actual stage results and immutable receipt bindings using the existing updater under the canonical lifecycle. `--checks-passed` is an executor assertion supported by real report/evidence, never a replacement for tests. The updater cannot accept a source-required user-acceptance gate; there is none prescribed by this pack. If newly required, use runner reconciliation rather than bypassing it. Recheck the next stage's entry inputs, current source paths and declared decisions; permission for that row does not authorize automatic continuation. Never edit an accepted plan/prompt/report to make an old receipt pass.

Finish in Russian with scope/results, limits and the exact next prompt link, then stop. On S09 there is no successor: complete the journal only after all required stages are satisfied and return the final local handoff. No publication, release or production delivery claim is authorized.
