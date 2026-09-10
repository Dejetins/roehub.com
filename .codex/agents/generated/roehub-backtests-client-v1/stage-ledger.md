# Backtests client v1 stage ledger

This is the only mutable execution-state journal for the linked pack. The JSON
record is canonical; authoring prose is not a second status table. Current user
selection authorizes authoring. Read `authoring_preflight` before entry: PACK-CLAIM
must be resolved, without inventing a lock or claiming a stage to discover it.
The draft has no executing/accepted stage and all rows are deliberately disallowed.
Future stage products are declared with their producing dependency, not owner gaps.

<!-- prompt-pack-ledger:v1 -->
```json
{
  "schema_version": "prompt-pack-ledger/v1",
  "prompt_pack_execution": {
    "plan_doc": "../../../../docs/architecture/apps/web/roehub-ui-implementation-plan-v1.md",
    "prompt_pack_dir": ".",
    "stage_ledger": "stage-ledger.md"
  },
  "execution_mode": "manual_sequential",
  "ledger_status": "draft",
  "current_stage": null,
  "stages": [
    {
      "contract": {
        "id": "S1",
        "title": "Client foundation and reversible Web integration",
        "prompt_path": "01-client-foundation.md",
        "report_path": "../../../delivery/evidence/roehub-backtests-client-v1/S1-client-foundation.md",
        "receipt_dir": "../../../delivery/evidence/roehub-backtests-client-v1/receipts/S1",
        "depends_on": [],
        "expected_touches": [
          "zone: client workspace and frontend build configuration",
          "zone: minimum shared package boundaries",
          "zone: Web routing settings and asset integration",
          "zone: frontend CI and local proof fixtures",
          "zone: affected Web contract tests",
          "zone: directly affected canonical Web docs and generated indices",
          "../../../delivery/evidence/roehub-backtests-client-v1/S1-client-foundation.md"
        ],
        "acceptance_criteria": [
          "A reproducible pnpm workspace builds apps/platform-web with the accepted stack and scoped typecheck, test, build and test:e2e scripts; dependency versions are resolved/pinned at execution.",
          "The disabled-by-default feature setting serves only selected Backtests client routes after existing authenticated HTML gating; feature off restores SSR; same-origin proxy, safe next, locale and private, no-store remain intact.",
          "A real local browser proves session/route/asset integration and navigation to preserved SSR destinations; meaningful route/proxy tests cover anonymous, authenticated, expired session and unavailable identity.",
          "A disposable local API/auth/data proof recipe is verified and recorded, with actual fixture/tool paths, commands, availability and limitations sufficient for subsequent Backtests integration. No runtime or provider proof is fabricated."
        ],
        "proof_boundary": "Local build plus real browser/Web session-route-proxy integration and verified local proof prerequisites; not a completed Backtests journey.",
        "validation": {
          "profile": "roehub-backtests-client-stage/v1",
          "checks": [
            "Run the stage-specific checks in the prompt and record actual commands, results and the real proof boundary.",
            "Run python3 -m tools.docs.generate_docs_index --check and python3 -m tools.docs.generate_project_map --check when their inputs change; regenerate only required generated outputs.",
            "Run git diff --check; inspect owned paths and exclude foreign changes.",
            "Validate the saved new transition receipt with the resolved bundled validator before the atomic ledger transition.",
            "- Run `pnpm --filter @roehub/platform-web typecheck`, `test`, `build`, and the focused foundation `test:e2e` after creating those scripts; capture their exact invocations/results.\n- For Web integration run `.venv/bin/python -m pytest -q tests/unit/apps/web/test_web_v2_1_routes.py`; add focused assertions for the changed routing/session/proxy seam as necessary.\n- Real browser: authenticated and anonymous route entry, deep-link refresh, locale continuation, protected HTML cache behavior, asset loading, console/network and feature-off SSR rollback. Source inspection/mocked API does not prove these boundaries.\n- Verify local fixture/service prerequisites early, without secrets in the report. Report unavailable runtime explicitly."
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
          }
        ]
      },
      "status": "pending",
      "execution_allowed": false,
      "current_authority": true,
      "executor_claim": null,
      "claimed_at": null,
      "transition_receipt": null,
      "decision_packet": null
    },
    {
      "contract": {
        "id": "S2",
        "title": "v23 shell and API-backed Backtests library",
        "prompt_path": "02-shell-and-library.md",
        "report_path": "../../../delivery/evidence/roehub-backtests-client-v1/S2-shell-and-library.md",
        "receipt_dir": "../../../delivery/evidence/roehub-backtests-client-v1/receipts/S2",
        "depends_on": [
          "S1"
        ],
        "expected_touches": [
          "zone: platform client shell and Backtests library",
          "zone: UI tokens localization and shared accessible controls",
          "zone: list adapters and frontend tests",
          "zone: directly affected canonical Web docs and generated indices",
          "../../../delivery/evidence/roehub-backtests-client-v1/S2-shell-and-library.md"
        ],
        "acceptance_criteria": [
          "The v23-based shell and library render real API-backed records, supported filters/search/cursor states and explicit loading, empty, stale, unavailable, forbidden and error states.",
          "RU/EN, keyboard focus, visible actions and accessible dialogs/table controls work at 820/1024/1440 and 200% zoom; navigation retains real SSR destinations for unrelated features.",
          "List/detail route entry and browser back/forward/refresh preserve valid identity/query state; a known job deep link loads independently of list membership without claiming the later execution/results UI complete.",
          "Real local list and browser visual evidence is recorded; logout/subject changes clear private query state and protected reads stop on expiry. Unimplemented next-stage actions are unavailable rather than fake successful controls."
        ],
        "proof_boundary": "Real API-backed list/read integration and rendered v23 shell/library behavior; configure/execute/result actions are not yet accepted.",
        "validation": {
          "profile": "roehub-backtests-client-stage/v1",
          "checks": [
            "Run the stage-specific checks in the prompt and record actual commands, results and the real proof boundary.",
            "Run python3 -m tools.docs.generate_docs_index --check and python3 -m tools.docs.generate_project_map --check when their inputs change; regenerate only required generated outputs.",
            "Run git diff --check; inspect owned paths and exclude foreign changes.",
            "Validate the saved new transition receipt with the resolved bundled validator before the atomic ledger transition.",
            "- Run client `typecheck`, `test`, `build`, and focused library/navigation `test:e2e` using S1's actual script/setup recipe.\n- Exercise a real local list, empty list and valid deep link; supplement unavailable/403/stale/out-of-order cases with controlled tests.\n- Capture browser evidence for RU/EN, 820/1024/1440, 200% zoom, keyboard and accessibility smoke, console/network. Compare rendered shell/list with v23; this is implementation/browser evidence, not a claim of complete journey fidelity."
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
            "path": "../../../delivery/evidence/roehub-backtests-client-v1/S1-client-foundation.md",
            "producer_stage": "S1"
          },
          {
            "path": "../../../../apps/platform-web/package.json",
            "producer_stage": "S1"
          }
        ]
      },
      "status": "pending",
      "execution_allowed": false,
      "current_authority": true,
      "executor_claim": null,
      "claimed_at": null,
      "transition_receipt": null,
      "decision_packet": null
    },
    {
      "contract": {
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
      "status": "pending",
      "execution_allowed": false,
      "current_authority": true,
      "executor_claim": null,
      "claimed_at": null,
      "transition_receipt": null,
      "decision_packet": null
    },
    {
      "contract": {
        "id": "S4",
        "title": "Observe execution and cancel without losing state",
        "prompt_path": "04-execution-and-cancel.md",
        "report_path": "../../../delivery/evidence/roehub-backtests-client-v1/S4-execution-and-cancel.md",
        "receipt_dir": "../../../delivery/evidence/roehub-backtests-client-v1/receipts/S4",
        "depends_on": [
          "S3"
        ],
        "expected_touches": [
          "zone: Backtests detail progress cancellation and reconciliation",
          "zone: job-read adapters and lifecycle frontend tests",
          "zone: directly affected canonical Web docs and generated indices",
          "../../../delivery/evidence/roehub-backtests-client-v1/S4-execution-and-cancel.md"
        ],
        "acceptance_criteria": [
          "Real job reads drive queued/running/terminal states, measured progress/freshness and unavailable estimates; reload/back/forward restore known job context without recreating a command.",
          "Confirmed cancellation renders pending until authoritative state resolves, terminal completion wins a race, dismissed dialogs send nothing and restore focus, and transport ambiguity triggers read-based reconciliation.",
          "Permission/session errors stop prohibited reads/actions; 409 and 429 preserve identity and eligibility; network/stale conditions never fabricate cancelled or succeeded state.",
          "An observed local job lifecycle and available cancellation path are recorded, supplemented by controlled terminal/race/error tests; generic Jobs/attempt retry semantics remain outside this stage."
        ],
        "proof_boundary": "Real local Backtests lifecycle/read/cancel behavior with controlled race, stale-state and error evidence; not generic Jobs or provider/runtime delivery.",
        "validation": {
          "profile": "roehub-backtests-client-stage/v1",
          "checks": [
            "Run the stage-specific checks in the prompt and record actual commands, results and the real proof boundary.",
            "Run python3 -m tools.docs.generate_docs_index --check and python3 -m tools.docs.generate_project_map --check when their inputs change; regenerate only required generated outputs.",
            "Run git diff --check; inspect owned paths and exclude foreign changes.",
            "Validate the saved new transition receipt with the resolved bundled validator before the atomic ledger transition.",
            "- Run client `typecheck`, `test`, `build`, focused lifecycle/cancel `test:e2e` and regress relevant S3 recovery cases.\n- Use the real local API to observe a submitted job and a valid cancellation interaction with disposable data; record actual terminal outcomes. Deterministic race/error tests supplement the observed local lifecycle.\n- Exercise cancel dismissed, pending, completed concurrently, already terminal, stale progress, 409, 429, transport ambiguity, reload and rapid route changes. Verify keyboard/focus, truthful state text, console/network and no invented ETA."
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
            "path": "../../../delivery/evidence/roehub-backtests-client-v1/S3-configure-and-submit.md",
            "producer_stage": "S3"
          },
          {
            "path": "../../../../apps/platform-web/package.json",
            "producer_stage": "S1"
          }
        ]
      },
      "status": "pending",
      "execution_allowed": false,
      "current_authority": true,
      "executor_claim": null,
      "claimed_at": null,
      "transition_receipt": null,
      "decision_packet": null
    },
    {
      "contract": {
        "id": "S5",
        "title": "Inspect results, export and create a saved strategy",
        "prompt_path": "05-results-and-strategy.md",
        "report_path": "../../../delivery/evidence/roehub-backtests-client-v1/S5-results-and-strategy.md",
        "receipt_dir": "../../../delivery/evidence/roehub-backtests-client-v1/receipts/S5",
        "depends_on": [
          "S4"
        ],
        "expected_touches": [
          "zone: Backtests result variants charts statistics and trades",
          "zone: bounded ECharts adapters and CSV download",
          "zone: saved-strategy and delete-history commands",
          "zone: result frontend and API integration tests",
          "zone: directly affected canonical Web docs and generated indices",
          "../../../delivery/evidence/roehub-backtests-client-v1/S5-results-and-strategy.md"
        ],
        "acceptance_criteria": [
          "Real summary/top/variant/statistics/chart/trade reads preserve variant URL identity and render bounded server data with accessible alternatives; no browser recomputation of financial metrics or invented OHLC.",
          "Materialization 202, empty/degraded detail, 429, export limits/truncation, real CSV versus JSON pending and out-of-order variant reads behave as specified.",
          "Compatibility readiness gates saved-strategy creation through its bodyless source job/variant and independent key/provenance dedupe contract; returned identity links to existing strategy detail and never starts trading.",
          "Delete history honors eligibility/conflicts and removes only after 204 or authoritative resolution; save/delete unknown outcomes preserve context and reconcile safely. Real local results/export/save proof and controlled failures are recorded."
        ],
        "proof_boundary": "Real local result/materialization/export and saved-strategy/delete boundaries, plus visual and failure-state evidence; no live trading or general Artifacts/Models UI.",
        "validation": {
          "profile": "roehub-backtests-client-stage/v1",
          "checks": [
            "Run the stage-specific checks in the prompt and record actual commands, results and the real proof boundary.",
            "Run python3 -m tools.docs.generate_docs_index --check and python3 -m tools.docs.generate_project_map --check when their inputs change; regenerate only required generated outputs.",
            "Run git diff --check; inspect owned paths and exclude foreign changes.",
            "Validate the saved new transition receipt with the resolved bundled validator before the atomic ledger transition.",
            "- Run client `typecheck`, `test`, `build` and focused results/export/save/delete `test:e2e`; regress earlier command/state invariants.\n- Prove actual local result series/stat/trades, CSV and saved-strategy identity with disposable data. Verify no run/start call occurs. Prove an eligible deletion and conflict behavior using disposable history.\n- Controlled tests: lazy 202 materialization, JSON-vs-CSV, truncation, 429, empty/degraded/failed details, rapid variant switch, save duplicate/lost response and separate provenance dedupe, delete conflict/ambiguous outcome.\n- Capture result visual/focus/accessible chart evidence against v23, including required locale/width states; inspect console/network."
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
            "path": "../../../delivery/evidence/roehub-backtests-client-v1/S4-execution-and-cancel.md",
            "producer_stage": "S4"
          },
          {
            "path": "../../../../apps/platform-web/package.json",
            "producer_stage": "S1"
          }
        ]
      },
      "status": "pending",
      "execution_allowed": false,
      "current_authority": true,
      "executor_claim": null,
      "claimed_at": null,
      "transition_receipt": null,
      "decision_packet": null
    },
    {
      "contract": {
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
      "status": "pending",
      "execution_allowed": false,
      "current_authority": true,
      "executor_claim": null,
      "claimed_at": null,
      "transition_receipt": null,
      "decision_packet": null
    }
  ],
  "authoring_preflight": {
    "authority": "User accepted the six-stage decomposition and requested plan/pack authoring on 2026-09-08; implementation and publication are not selected.",
    "entry_candidate": "S1",
    "entry_ready": false,
    "gap_id": "PACK-CLAIM",
    "gap": "No current supported exclusive atomic ledger claim/update implementation was found in current repo guidance/tools or installed staged-plan-runner. Do not infer one from manual mode, Python/filesystem access or the read-only validator.",
    "search_sources": [
      ".codex/AGENTS.md",
      "tools/",
      "/Users/daniildegtyarev/.codex/skills/staged-plan-runner/SKILL.md",
      "/Users/daniildegtyarev/.codex/skills/staged-plan-runner/references/ledger-lifecycle-v1.md"
    ],
    "resolution": "Resolve an existing supported updater with verified atomic mutation and ownership semantics for this ledger, or prepare separately authorized tooling work; bind current root-contained evidence and its SHA-256 as claim_capability. Then enable only S1 and revalidate entry before an implementation handoff.",
    "owner_decisions": []
  },
  "validation_profile": {
    "schema_version": "prompt-pack-ledger/v1",
    "validator_path": "/Users/daniildegtyarev/.codex/skills/prompt-manager/scripts/validate_pack.py",
    "runtime": "Python 3.10+ standard library",
    "draft_command": "python3 /Users/daniildegtyarev/.codex/skills/prompt-manager/scripts/validate_pack.py --root /Users/daniildegtyarev/Projects/roehub.com --ledger /Users/daniildegtyarev/Projects/roehub.com/.codex/agents/generated/roehub-backtests-client-v1/stage-ledger.md --check draft",
    "entry_command": "python3 /Users/daniildegtyarev/.codex/skills/prompt-manager/scripts/validate_pack.py --root /Users/daniildegtyarev/Projects/roehub.com --ledger /Users/daniildegtyarev/Projects/roehub.com/.codex/agents/generated/roehub-backtests-client-v1/stage-ledger.md --check entry --stage S1",
    "result_contract": "Exit 0 + JSON status=pass and matching check; exit 1=fail, exit 2=unavailable. No mutation/claim/execution authority or runtime proof.",
    "receipt_schema": "prompt-pack-receipt/v1"
  },
  "source_traceability": {
    "plan": "All six stages and the retained UI backlog are in plan_doc.",
    "ticket": "../../../tickets/2026-09-08-roehub-backtests-client.md",
    "ticket_acceptance": {
      "T1": [
        "S2",
        "S5",
        "S6"
      ],
      "T2": [
        "S3",
        "S4",
        "S5",
        "S6"
      ],
      "T3": [
        "S3",
        "S6"
      ],
      "T4": [
        "S3",
        "S5",
        "S6"
      ],
      "T5": [
        "S4",
        "S5",
        "S6"
      ],
      "T6": [
        "S2",
        "S3",
        "S4",
        "S5",
        "S6"
      ],
      "T7": [
        "S1",
        "S2",
        "S3",
        "S4",
        "S5",
        "S6"
      ],
      "T8": [
        "S1",
        "S6"
      ]
    },
    "ticket_sections": {
      "Field and validation contract": [
        "S3",
        "S6"
      ],
      "API bindings": [
        "S2",
        "S3",
        "S4",
        "S5",
        "S6"
      ],
      "States and transitions": [
        "S2",
        "S3",
        "S4",
        "S5",
        "S6"
      ],
      "Errors and recovery": [
        "S3",
        "S4",
        "S5",
        "S6"
      ],
      "Integration and rollback": [
        "S1",
        "S6"
      ]
    },
    "retained_outside_first_delivery": [
      "Jobs",
      "Artifacts",
      "data ingestion",
      "execution details",
      "target role cutover",
      "broader backtest modes",
      "measured ETA"
    ]
  },
  "authoring_review": {
    "status": "completed",
    "mode": "independent_subagent",
    "artifact_verdict": "Release",
    "entry_verdict": "Block",
    "unresolved": [
      "PACK-CLAIM"
    ],
    "evidence": "../../../delivery/evidence/ROEHUB-UI-IMPLEMENTATION-PLAN-2026-09-08.md"
  }
}
```
