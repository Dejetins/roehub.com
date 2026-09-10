---
{
  "schema_version": "stage-prompt/v1",
  "prompt_pack_execution": {
    "plan_doc": "../../../../docs/architecture/apps/web/roehub-ui-implementation-plan-v1.md",
    "prompt_pack_dir": ".",
    "stage_ledger": "stage-ledger.md"
  },
  "stage_contract": {
    "id": "S3",
    "title": "Configure, preflight and submit with recovery",
    "prompt_path": "03-configure-and-submit.md",
    "report_path": "../../../delivery/evidence/roehub-backtests-client-v1/S3-configure-and-submit.md",
    "receipt_dir": "../../../delivery/evidence/roehub-backtests-client-v1/receipts/S3",
    "depends_on": [
      "S2"
    ],
    "expected_touches": [
      "zone: Backtests builder validation and command adapters",
      "zone: bounded per-tab recovery and session integration",
      "zone: frontend state and API integration tests",
      "zone: directly affected canonical Web docs and generated indices",
      "../../../delivery/evidence/roehub-backtests-client-v1/S3-configure-and-submit.md"
    ],
    "acceptance_criteria": [
      "Every field in the ticket field matrix is either editable or explicitly carried as authoritative effective/default input as specified, with catalog-backed validation, UTC half-open time bounds, correct units and stale-preflight invalidation.",
      "The real local configure to preflight to job-create flow produces a server job identity, blocks preflight body errors even on HTTP success, and handles renewed admission/rejection at create.",
      "One frozen logical request/key prevents double submit; a lost response/reload retains allowed recovery data without automatic mutation replay. Server-bound original organization, retention and fresh authorization gate any same-key replay; unsafe/unknown scope shows unresolved read-only guidance.",
      "422 field paths, 409 normalized/effective hash conflict, metadata-only label semantics, 429, 503, session expiry, storage-unavailable degradation and ordinary unsent-draft discard behavior match the ticket and are tested."
    ],
    "proof_boundary": "Real local preflight/create integration plus controlled command identity, field validation and recovery boundary evidence; full lifecycle/results remain subsequent stages.",
    "validation": {
      "profile": "roehub-backtests-client-stage/v1",
      "checks": [
        "Run the stage-specific checks in the prompt and record actual commands, results and the real proof boundary.",
        "Run python3 -m tools.docs.generate_docs_index --check and python3 -m tools.docs.generate_project_map --check when their inputs change; regenerate only required generated outputs.",
        "Run git diff --check; inspect owned paths and exclude foreign changes.",
        "Validate the saved new transition receipt with the resolved bundled validator before the atomic ledger transition.",
        "- Run client `typecheck`, `test`, `build`, and focused builder/preflight/submit/recovery `test:e2e`.\n- Prove real local preflight and create using disposable data. Add controlled fault injection for response lost after acceptance, double click, 409 request hash versus label-only metadata, stale preflight, 422 path mapping, rate units, market/direction/grid bounds and 429/admission failure.\n- Prove recovery after reload/session re-entry, expired/unknown retention, same subject with changed/unknown organization, session-storage unavailable, dirty-form discard, no automatic mutation retry and logout private-state clearing. Mocks supplement but do not replace the real create proof.\n- Run `.venv/bin/python -m pytest -q tests/unit/apps/api/test_backtests_routes.py tests/unit/apps/api/test_ui_backtests_routes.py tests/unit/apps/web/test_web_v2_1_routes.py` when the corresponding integration/test boundary is touched."
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
        "path": "../../../delivery/evidence/roehub-backtests-client-v1/S2-shell-and-library.md",
        "producer_stage": "S2"
      },
      {
        "path": "../../../../apps/platform-web/package.json",
        "producer_stage": "S1"
      }
    ]
  },
  "always_read": [
    ".codex/tickets/2026-09-08-roehub-backtests-client.md",
    "docs/architecture/apps/web/roehub-ui-functional-contract-v1.md"
  ],
  "task_entrypoints": [
    "apps/api/dto/backtests.py",
    "apps/api/routes/backtests.py",
    "src/trading/contexts/backtest/application/use_cases/backtest_jobs.py"
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

# S3 — Configure, preflight and submit with recovery

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

1. Implement the complete ticket field matrix from authoritative catalog/default APIs: instrument coordinates, timeframe, UTC `[start,end)`, ordered indicators/materialized grid, risk TP/SL levels, direction/market compatibility, cash/rates/sizing, funding/profit_lock/close_on_end, ranking/top N and effective quality constraints. Preserve server defaults and do not invent unsupported editors. Block submission if defaults are unavailable.
2. Implement preflight with field errors and a focusable summary, normalized effective settings, source/funding quality, cost/warnings and invalidation after result-affecting edits. Create must still admit/validate independently. Never silently clamp input or substitute specimen values.
3. Freeze the exact allowlisted request and one `Idempotency-Key` before sending. Bind recovery to authenticated subject and server-derived organization; double-click protection is part of the command lifecycle. Navigate only on an authoritative returned job identity; offer no fake progress.
4. Implement the ticket's full job-create recovery policy now. Verify actual server/configured replay retention; the default 86400 seconds is not proof of the running value. A stored organization is not a server precondition. If original server scope binding is unavailable, keep unresolved/read-only guidance rather than changing API/authz in this pack. Unknown outcomes do not rotate keys or auto-resubmit. Compare normalized effective request identity; a label-only metadata change is not a different computation.
5. Keep ordinary unsent draft in memory with dirty-leave confirmation. Implement only allowlisted per-tab session recovery, no credentials or generic payload cache. Clear on logout/subject change, handle same-subject organization change and unknown identity, remove resolved payloads. If session storage fails, preserve in-memory recovery and show its limitation before submission.
6. Map every applicable ticket error condition, preserving configuration and command identity. Do not defer errors/recovery to S6 or introduce target role controls.

## Verification and acceptance

The measurable acceptance criteria are in `stage_contract`; all are required.

- Run client `typecheck`, `test`, `build`, and focused builder/preflight/submit/recovery `test:e2e`.
- Prove real local preflight and create using disposable data. Add controlled fault injection for response lost after acceptance, double click, 409 request hash versus label-only metadata, stale preflight, 422 path mapping, rate units, market/direction/grid bounds and 429/admission failure.
- Prove recovery after reload/session re-entry, expired/unknown retention, same subject with changed/unknown organization, session-storage unavailable, dirty-form discard, no automatic mutation retry and logout private-state clearing. Mocks supplement but do not replace the real create proof.
- Run `.venv/bin/python -m pytest -q tests/unit/apps/api/test_backtests_routes.py tests/unit/apps/api/test_ui_backtests_routes.py tests/unit/apps/web/test_web_v2_1_routes.py` when the corresponding integration/test boundary is touched.

## Adjacent-stage preflight and final handoff

The report path is `../../../delivery/evidence/roehub-backtests-client-v1/S3-configure-and-submit.md` relative to the ledger. Record exact created,
modified, deleted and outside-expected paths with reasons; excluded foreign changes;
actual commands/results and evidence paths; implementation decisions; residual risks;
and the precise boundary proved. Producer reports must identify new implementation
files and reproducible setup needed by consumers. Keep stage reports stable after
receipt binding; later corrections get new evidence/receipts and preserve history.

Inspect the next prompt S4 and its declared inputs after current validation. Verify actual producer outputs, current dependencies and decisions before proposing next-stage permission. A valid current receipt can accept this stage even if the next is disallowed. Next permission is not authority to run a second stage in manual mode.

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
