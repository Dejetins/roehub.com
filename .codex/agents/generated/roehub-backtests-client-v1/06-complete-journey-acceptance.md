---
{
  "schema_version": "stage-prompt/v1",
  "prompt_pack_execution": {
    "plan_doc": "../../../../docs/architecture/apps/web/roehub-ui-implementation-plan-v1.md",
    "prompt_pack_dir": ".",
    "stage_ledger": "stage-ledger.md"
  },
  "stage_contract": {
    "id": "S6",
    "title": "Accept the complete local Backtests journey",
    "prompt_path": "06-complete-journey-acceptance.md",
    "report_path": "../../../delivery/evidence/ROEHUB-BACKTESTS-CLIENT-2026-09-08.md",
    "receipt_dir": "../../../delivery/evidence/roehub-backtests-client-v1/receipts/S6",
    "depends_on": [
      "S5"
    ],
    "expected_touches": [
      "zone: focused integration repairs within prior stage scope",
      "zone: end-to-end frontend and Web integration tests",
      "zone: canonical Backtests completion evidence and coverage updates",
      "zone: directly affected canonical Web docs and generated indices",
      "../../../delivery/evidence/ROEHUB-BACKTESTS-CLIENT-2026-09-08.md"
    ],
    "acceptance_criteria": [
      "All eight ticket acceptance criteria have actual linked evidence; an integrated real local configure/preflight/create/progress/terminal/result/trades/CSV path passes against the current code.",
      "RU/EN, 820/1024/1440, 200% zoom, keyboard/focus, accessible chart meaning and browser console/network checks pass for complete shell/list/builder/detail workflows using v23 as the reference.",
      "Duplicate/lost-response recovery, original organization/retention guards, unavailable storage, cancel races, materialization, save strategy, deletion, session navigation and feature-off SSR rollback all pass with real-boundary and controlled-failure evidence correctly distinguished.",
      "The final ticket receipt states local_journey_verified only on full proof and assesses target_role_cutover_ready separately against actual authz dependencies; no default cutover, publication or deployment is inferred.",
      "Only evidence-backed implemented_scope is updated in the functional registry and ticket; unresolved full-target API work, Jobs, Artifacts, ingestion and execution detail remain in scope and unclaimed."
    ],
    "proof_boundary": "Integrated local_journey_verified against real local APIs and browser plus focused failure tests; explicitly separate target_role_cutover_ready and no production delivery claim.",
    "validation": {
      "profile": "roehub-backtests-client-stage/v1",
      "checks": [
        "Run the stage-specific checks in the prompt and record actual commands, results and the real proof boundary.",
        "Run python3 -m tools.docs.generate_docs_index --check and python3 -m tools.docs.generate_project_map --check when their inputs change; regenerate only required generated outputs.",
        "Run git diff --check; inspect owned paths and exclude foreign changes.",
        "Validate the saved new transition receipt with the resolved bundled validator before the atomic ledger transition.",
        "- Run `pnpm --filter @roehub/platform-web typecheck`, `pnpm --filter @roehub/platform-web test`, `pnpm --filter @roehub/platform-web build`, and `pnpm --filter @roehub/platform-web test:e2e` for the integrated scope.\n- Run `.venv/bin/python -m pytest -q tests/unit/apps/api/test_backtests_routes.py tests/unit/apps/api/test_ui_backtests_routes.py tests/unit/apps/web/test_web_v2_1_routes.py` for the final Web/API compatibility boundary.\n- Complete all T1–T8 browser/API checks in the source ticket. Report actual setup, commands, URLs, auth state, redacted fixture identities and evidence. Explicitly state any unavailable criterion and withhold local_journey_verified if required proof is missing.\n- Recheck the v23 SHA-256 and foreign-change preservation; validate current reports/receipt and docs/index consistency. No release/deployment or target-role acceptance without its separate authority/evidence."
      ],
      "requires_user_acceptance": false
    },
    "entry_inputs": [
      {
        "path": "../../../tickets/2026-09-08-roehub-backtests-client.md",
        "producer_stage": null
      },
      {
        "path": "../../../../docs/architecture/apps/web/roehub-ui-functional-contract-v1.md",
        "producer_stage": null
      },
      {
        "path": "../../../delivery/evidence/roehub-ui-agent-governed-pilot/specimens/2026-08-03-linear-black-workbench-v23.html",
        "producer_stage": null
      },
      {
        "path": "../../../delivery/evidence/roehub-backtests-client-v1/S5-results-and-strategy.md",
        "producer_stage": "S5"
      },
      {
        "path": "../../../../apps/platform-web/package.json",
        "producer_stage": "S1"
      },
      {
        "path": "../../../delivery/evidence/roehub-backtests-client-v1/S1-client-foundation.md",
        "producer_stage": "S1"
      },
      {
        "path": "../../../delivery/evidence/roehub-backtests-client-v1/S2-shell-and-library.md",
        "producer_stage": "S2"
      },
      {
        "path": "../../../delivery/evidence/roehub-backtests-client-v1/S3-configure-and-submit.md",
        "producer_stage": "S3"
      },
      {
        "path": "../../../delivery/evidence/roehub-backtests-client-v1/S4-execution-and-cancel.md",
        "producer_stage": "S4"
      }
    ]
  },
  "always_read": [
    ".codex/tickets/2026-09-08-roehub-backtests-client.md",
    "docs/architecture/apps/web/roehub-ui-functional-contract-v1.md"
  ],
  "task_entrypoints": [
    ".codex/tickets/2026-07-20-roehub-authz-backtests.md",
    "tests/unit/apps/api/test_backtests_routes.py",
    "tests/unit/apps/api/test_ui_backtests_routes.py"
  ],
  "conditional_bundles": {
    "visual_ui": [
      ".codex/delivery/evidence/roehub-ui-agent-governed-pilot/specimens/2026-08-03-linear-black-workbench-v23.html"
    ],
    "api_binding": [
      "apps/api/dto/backtests.py",
      "apps/api/dto/ui_backtests.py",
      "docs/architecture/api/api-errors-and-422-payload-v1.md"
    ],
    "security_and_cutover": [
      "docs/architecture/apps/web/roehub-local-platform-access-and-route-contract-v1.json",
      ".codex/tickets/2026-07-20-roehub-authz-backtests.md"
    ],
    "durable_scope_update": [
      "docs/architecture/apps/web/roehub-ui-functional-registry-v1.json",
      "docs/architecture/apps/web/roehub-ui-design-and-delivery-architecture-v1.md"
    ]
  },
  "required_literals": [
    "Idempotency-Key",
    "local_journey_verified",
    "target_role_cutover_ready"
  ],
  "required_keywords": [
    "v23",
    "SSR",
    "recovery"
  ]
}
---

# S6 — Accept the complete local Backtests journey

## Scope and authority

Use `staged-plan-runner` for the named stage only, after an explicit implementation
request. First read its live ledger row, triad and current prerequisite evidence.
Pack authoring is not execution authority. A non-runnable draft stays unclaimed;
resolve its technical entry gap before any stage side effect. Do not activate, claim,
resume or accept through ad-hoc file edits. Use the verified exclusive updater and
canonical lifecycle, with revalidation inside its ownership boundary. Never steal
foreign/stale claims or revive historical packs/tools. Manual mode stops after one stage.

The source Backtests ticket owns fields, APIs, transitions, errors and recovery;
this stage cannot weaken its final acceptance. The accepted plan owns decomposition.
`stage_contract` above is identical to the ledger contract; all mutable status,
permissions, decision packets, claims and receipts live only in the ledger.

Implement only expected touch zones resolved to owned paths within the ticket.
Backend domain/authz/migrations/runtime limits/deployment remain read-only. Minimal
local fixtures and dependency installation within the selected ticket are permitted
only when implementation is requested. Preserve foreign changes and re-read before
replacement; no broad staging, destructive Git, speculative branch/worktree/stash,
external messages, publication or deployment from this prompt. If later explicitly
selected, route publication through `publish-ci-deploy` and the current runbook.

## Context acquisition

`always_read` and `task_entrypoints` in metadata use repository-relative paths;
the triad and stage contract paths use the ledger-relative bundled convention.
Read applicable AGENTS, this prompt/live row, relevant plan sections and the source
ticket sections named by this stage. Read the functional contract's shared behavior
and relevant dependency rows. Open the specified task entrypoints; S1–S5 producer
reports identify the exact future implementation files, so do not invent file paths
or treat those deferred products as missing owner inputs during authoring.

Read conditional bundles only for the touched boundary. Normally stay within eight
source files before implementation apart from control metadata; expand only for a
named ambiguity, failure or direct contract dependency. Stop broad discovery once
scope, sources, contracts, proof and blockers are known. Do not reread historical
programs to reconstruct decisions already consolidated in the current ticket.

## Conditional skills

- Use `browser-qa-evidence` before collecting browser/runtime claims. Use
  `playwright-cli` for terminal browser mechanics unless the user explicitly chooses
  Browser/Chrome; reuse one real browser surface for evidence.
- Use only materially affected `better-layout`, `better-ui`, `better-typography`,
  `better-colors`, `better-writing` and `better-accessibility` domains during UI work;
  do not load all merely because the interface contains controls/text.
- Use `contract-impact-analysis` if routing/session/config/browser/API boundaries
  change, before settling compatibility/rollback; initial backend APIs are read-only.
- Use `backend-quality-gates` when Python Web/integration checks or failures require
  verification; `root-cause-debugging` for unknown concrete failures.
- Use `product-design:index` only for a named unresolved composition/interaction
  requiring a prototype; preserve v23 and avoid a whole-product redesign. Follow
  current source-inspection rules; no live Figma/Penpot or mandatory prototype series.
- Use `architecture-design` only for a material unresolved boundary within authorized
  scope; do not let it expand the ticket. Authoring/revising prompts belongs to
  `prompt-manager`, never edit an active accepted contract under the same stage ID.

## Decisions and stop conditions

Technical names, component factoring, small composition adjustments and test/fixture
choices are delegated within the accepted source. Do not ask the user to approve
bookkeeping or repeat accepted v23 decisions. Record any genuinely missing material
product choice with its affected requirement and exact resume condition; keep useful
independent work moving. A missing preceding output before its producer runs is a
deferred dependency, not an owner question. At execution, missing required API/runtime
proof blocks the corresponding acceptance. Do not simulate grants, service state,
financial results or successful commands to pass it.

Keep credentials/cookies/provider payloads and raw environment dumps out of logs,
DOM captures and durable reports. Use disposable local fixtures and redacted evidence.
Unknown command outcomes follow the ticket's safe reconciliation, never automatic
mutation retry. Stage acceptance does not imply complete-journey acceptance, target
roles, default cutover, provider behavior or production delivery.

## Stage work

1. Re-read all producer reports and receipts plus the entire ticket acceptance section. Build a T1–T8 evidence matrix that distinguishes reused still-current proof from checks rerun after integration changes. Missing proof is not a pass.
2. Run the whole configured local API-backed journey with disposable data, then exercise errors, cancellation, recovery, saved-strategy creation, export and deletion. A mocked end-to-end demo is insufficient. Repair bounded defects in the prior implementation touch zones and recheck affected earlier acceptance; do not use S6 to add new scope or bypass a missing API/authz dependency.
3. Verify v23-based visual composition, RU/EN, 820/1024/1440, 200% zoom, keyboard/focus and automated accessibility smoke, chart alternatives, deep-link reload/back/forward, safe login continuation and logout cleanup. Inspect actual browser console/network evidence. Do not demand another aesthetic owner approval of accepted v23 unless a new material reserved choice actually arises.
4. Verify feature-off restores SSR without data migration or losing created domain objects. Re-read the actual authz ticket/dependency state; local own-resource proof cannot prove target operator/viewer/delegation policies. Do not enable default cutover.
5. Write the canonical final Backtests receipt at the report_path in the contract. Update the ticket and functional registry only with precise implemented behavior and linked proof. Retain requires_ui/requires_api where wider target scope is unfinished; never mark the entire platform done. Preserve historical evidence and authored stage contracts.
6. After all required proof, append the final immutable transition receipt with no next stage and complete the ledger under the current runner ownership mechanism. This does not execute the broader UI backlog.

## Verification and acceptance

The measurable acceptance criteria are in `stage_contract`; all are required.

- Run `pnpm --filter @roehub/platform-web typecheck`, `pnpm --filter @roehub/platform-web test`, `pnpm --filter @roehub/platform-web build`, and `pnpm --filter @roehub/platform-web test:e2e` for the integrated scope.
- Run `.venv/bin/python -m pytest -q tests/unit/apps/api/test_backtests_routes.py tests/unit/apps/api/test_ui_backtests_routes.py tests/unit/apps/web/test_web_v2_1_routes.py` for the final Web/API compatibility boundary.
- Complete all T1–T8 browser/API checks in the source ticket. Report actual setup, commands, URLs, auth state, redacted fixture identities and evidence. Explicitly state any unavailable criterion and withhold local_journey_verified if required proof is missing.
- Recheck the v23 SHA-256 and foreign-change preservation; validate current reports/receipt and docs/index consistency. No release/deployment or target-role acceptance without its separate authority/evidence.

## Adjacent-stage preflight and final handoff

The report path is `../../../delivery/evidence/ROEHUB-BACKTESTS-CLIENT-2026-09-08.md` relative to the ledger. Record exact created,
modified, deleted and outside-expected paths with reasons; excluded foreign changes;
actual commands/results and evidence paths; implementation decisions; residual risks;
and the precise boundary proved. Producer reports must identify new implementation
files and reproducible setup needed by consumers. Keep stage reports stable after
receipt binding; later corrections get new evidence/receipts and preserve history.

This is the final stage: next_stage=null, next_stage_allowed=false. Verify all required obligations and accepted receipt bindings before completing the ledger. Do not fabricate a next stage for cutover or the broader UI backlog.

Create a new immutable receipt under `receipt_dir` using `prompt-pack-receipt/v1`,
with canonical stage-contract SHA-256, live plan/prompt/report hashes, actual passing
validation evidence and exact next-stage binding/reason. No receipt hashes itself or
the mutable ledger. User acceptance is required only if current user authority
actually reserves a material decision; no new ceremony is introduced by the format.
Validate with:

```sh
python3 /Users/daniildegtyarev/.codex/skills/prompt-manager/scripts/validate_pack.py --root /Users/daniildegtyarev/Projects/roehub.com --ledger /Users/daniildegtyarev/Projects/roehub.com/.codex/agents/generated/roehub-backtests-client-v1/stage-ledger.md --check receipt --receipt <new-receipt-path-relative-to-ledger>
```

Require exit 0 and a well-formed JSON pass matching `receipt`; unavailable/failed
validation cannot authorize acceptance. Under the verified exclusive updater,
recheck live bindings and atomically persist the lifecycle/claim/receipt transition
before the final Russian report. Keep prior receipts immutable. Report current
acceptance separately from next permission; do not edit prompts to clear blockers.
