# S5 — Results, export, saved strategy and history deletion

Date: 2026-09-08. Execution: one executor in the existing local checkout, S5 only.
Status: implementation and required local checks passed. Independent coordinator acceptance is separate.

## Authority and proof boundary

The current sequential workflow supersedes PACK-CLAIM, staged-plan-runner,
updater, ledger transitions, claims and receipts. The existing ledger and v23
specimen remain unchanged. No Goal, subagent, additional task, branch, worktree,
stash, publication, deployment, default cutover, backend domain/authz/migration/
limit changes, AGENTS or global skill edits were made. No dependency was installed.

The real local proof uses the existing own-resource policy, production API/auth,
PostgreSQL, ClickHouse, worker, lazy materializer and production artifacts built
from 4320 explicitly synthetic candles. No success, job state, financial result,
role or compatibility grant is substituted. Controlled browser responses provide
additional failure/race states and are identified separately.

`local_journey_verified=false`; `target_role_cutover_ready=false`. S5 proves its
result/save/export/delete boundary. Integrated ticket T1–T8 acceptance belongs
to S6; provider trading behavior, target roles and production delivery are outside
this proof. The known workstation organization_scope_resolver defect is unchanged.

## Implementation

- `results-api.ts` allowlists summary/top/variant, series, statistics, trades,
  readiness and saved identity/provenance responses. Request paths encode opaque
  variant identity. Returned job/variant, series kind, trade page, saved subject
  and provenance are checked. Raw artifact/cache paths, provider metadata and
  arbitrary response fields do not enter the result cache or recovery storage.
- `results.tsx` integrates succeeded-job results into S4 `JobEntry`. Selected
  variant remains in `?variant=` through deep links, history and reload. Query
  keys include private subject, job, variant, view and page. Unmount cancels
  obsolete reads; late source responses cannot populate the next selection.
- ECharts renders only server equity/drawdown points, requesting 400 against the
  current API maximum of 1500. A single point is shown as a point. No financial
  metrics, aggregate statistics, fake progress or OHLC are computed in JavaScript.
  Every chart offers the same bounded data as a keyboard-accessible table.
  Monthly/symbol statistics and 50-trade pages use server values and pagination,
  bounded to the server's 10000-page route limit.
- One result-read scheduler per selected resource respects 202 materialization
  hints and GET429. Errors take precedence over old cached202: restricted errors
  stop polling, including tab remount; transport/5xx permits explicit recovery
  after its own delay. Pending/failed materialization never changes job state.
  Empty/degraded/truncated detail remains truthful. Reconnect/focus does not
  bypass the deadline. S4's unified job polling/cancel reconciliation is retained.
- CSV is an explicit response-type-aware GET, with a 1–100000-row bound. Only
  HTTP200 `text/csv` becomes a download. HTTP202 JSON remains pending and cannot
  produce a file. Row count/total/max/truncated headers are validated and shown.
  Header and JSON error retry hints are honored. A protected401 closes the private
  surface before reading its body. Source changes abort export and discard late data.
- Saving is a bodyless source-job/variant POST with its own `Idempotency-Key`.
  The command checks session, fresh readiness and source/spec identity, uses a
  synchronous duplicate lock, and never starts a run. Explicit confirmation names
  the source and explains the research-only save and relevant trading limitation.
  The returned strategy UUID links to the existing SSR detail.
- `strategy-recovery.ts` has a separate per-tab allowlist: operation, source job/
  variant, key, creation time, subject, organization (unknown where absent from
  the server contract), and known result ID. No job-create TTL is applied.
  Unknown save retains this record and blocks a fresh command; no automatic or
  client same-key replay is enabled without original-organization binding.
  Known refusals recover only after explicit readiness recheck and Retry-After.
  Identity change/logout clears private recovery. Unavailable storage retains
  validated memory and warns before submitting that leaving loses recovery.
- `delete-history.tsx` confirms only terminal deletion after a fresh session
  check. HTTP204 removes selected history and returns to the library with feedback.
  A conflict requires a new authoritative job read through S4's existing deadline;
  unknown outcome retains the job and never resends DELETE. 404 is absence or
  invisibility, not proof of deletion. A command403 hides selected private results
  and cancels their pending reads. No backend eligibility rule is modified.
- Native dialogs trap keyboard focus, close with Escape and restore their trigger
  on dismissal. Result tabs implement arrows/Home/End and roving tab stops.
  RU/EN labels, v23 surfaces/tokens, wrapping actions and table-contained overflow
  remain. Axis labels now explicitly use the existing `--muted`, not ECharts defaults.

## Readiness interpretation corrected from production evidence

The ticket's source API/key contract selects
`create_strategy_from_backtest_variant.py`, whose immutable save does not require
a live evaluator. In contrast, `compatibility_readiness.py::_compatibility_for_spec`
recognizes only the MA(fast,slow) live evaluator; current artifact ma.ema/window
results truthfully return `not_launchable / unsupported_live_evaluator`.
`launch_blocked` additionally includes market-feed readiness. These are trading
constraints, not a prohibition on saving the existing research variant.

The coordinator independently verified this distinction and instructed the
executor to retain mandatory successful source-validated readiness reads, show
localized trading reasons/warnings and permit the existing bodyless save contract.
The initial UI assumption `compatibility_state === launchable` was corrected;
no backend success, payload, authorization or accepted product scope was changed.
The real proof explicitly exercises not_launchable → 201 saved strategy.

The server use case hashes the request key and scopes provenance to organization
and user; it first dedupes by key, then source job/variant plus spec/request hash.
The provenance lookup has no configured job-create expiry. The current save DTO
returns strategy UUID/user/name/dates/deleted/spec but no organization ID. Client
recovery therefore keeps organization null and does not invent a replay precondition.

## Validation and evidence

All commands run from `/Users/daniildegtyarev/Projects/roehub.com`.

| Command | Actual outcome |
| --- | --- |
| `pnpm --filter @roehub/platform-web typecheck` | Passed |
| `pnpm --filter @roehub/platform-web test` | Passed: 145 tests / 6 files (26 new S5 cases) |
| `pnpm --filter @roehub/platform-web build` | Passed: `main-Dj6go0R_.js`, `main-TfDMVJxy.css`; >500kB Vite advisory |
| `ROEHUB_PROOF_STAGE=S5 pnpm --filter @roehub/platform-web test:e2e` | Foundation/library 5/5, builder 6/6, execution 9/9 passed; in-progress S5 additions exposed the POST-inline assumption and pre-fix delete403 case, corrected and rerun below |
| `ROEHUB_PROOF_STAGE=S5 pnpm --filter @roehub/platform-web test:e2e results.spec.ts` | Passed: 12/12 cases on the final runtime build, 1.3 min |
| `ROEHUB_PROOF_STAGE=S5 pnpm --filter @roehub/platform-web test:e2e execution.spec.ts` | Passed: 9/9 cases on the final runtime build, 2.7 min |
| `ROEHUB_PROOF_STAGE=S5 pnpm --filter @roehub/platform-web test:e2e results.spec.ts --grep 'real results|readiness transport and failed save'` | Passed: 2/2 after test-only additions asserting rendered statistics/trade rows and logout recovery cleanup; 25.9s |
| `.venv/bin/python -m pytest -q tests/unit/apps/api/test_backtests_routes.py tests/unit/apps/api/test_ui_backtests_routes.py tests/unit/apps/web/test_web_v2_1_routes.py` | Passed: 98 tests; pre-existing httpx cookie deprecation warning |
| `.venv/bin/ruff check tools/qa/backtests_client_fixture.py` | Passed |
| `python3 -m tools.docs.generate_project_map --check` | Initial source inventory drift; regenerated with `python3 -m tools.docs.generate_project_map`, check passed |
| `python3 -m tools.docs.generate_docs_index --check` | Passed |
| `git diff --check` | Passed |
| Fixture cleanup | Passed: `.local_artifacts/backtests-client` absent; `docker ps --filter name=roehub-client --format '{{.Names}}'` returned no containers |

No runtime source changed after the final build. The last two test-only assertions
were rerun together, including the real API lifecycle. The aggregate proof comprises 32 browser scenarios (5 + 6 + 9 + 12), with
S4 repeated after the final deletion restriction integration. The failed combined
attempt is not reported as a green full command. S1–S3 evidence is reused because
the subsequent runtime fixes affect only result/delete paths; S4 and S5 are the
corresponding final-build regression boundary.
Evidence lives under `browser/S5/` next to this report. Regression directories
`S5-foundation-regression`, `S5-library-regression`, `S5-builder-regression` and
`S5-execution-regression` preserve accepted S1–S4 evidence.

Real results have one trade under the fixture's monotonic candles: actual CSV is
one row and is **not truncated**. Actual export bounds/headers are checked; the
multi-row truncation response is a separately labelled controlled case. Real
lazy materialization is observed as equity202 before equity200. POST trades is
also exercised against the production API after materialization: it returns200
and the same source identity but an empty inline trades array in this cache-hit
path. The client correctly uses the separate nonempty GET/paginated trade API,
not POST-inline rows, for its table. No backend behavior was altered. Actual save
checks HTTP201, bodylessness, same-key200 `idempotent_replay`, new-key200
`source_variant_exists`, matching strategy API identity and SSR selected row.
Final real source job: `c0eb8779-403b-4697-8063-c7c51022c794`;
saved strategy: `7d8668b3-7ba3-4f56-b33f-f78b9f22a2f8`.
`real-results.json` records an actual equity202 observation before ready detail,
server row counts, statuses and sanitized network/console evidence. A second UI save reaches the real dedupe endpoint and only its response is dropped;
reload keeps unresolved recovery without another POST. A separate queued job
returns actual delete409, is cancelled through its real API/worker, then deleted
through the UI with204. All run/start POST counters must remain zero.

Controlled tests cover queued202→429 cooldown, empty/degraded/failed detail,
late variant response and history/reload, JSON export pending/no download and
truncation, readiness transport recovery, save429→explicit recheck, unknown save
and reload, separate source recovery block, storage failure, mismatched readiness,
identity outage, 202→401/403/404 quiet stop, 202→transport→manual success, deletion
conflict/unknown and delete403 private-data closure. RU/EN × 820/1024/1440 and native
Chromium 200% zoom include screenshots, overflow, axe, tabs/dialog focus checks.
Final native zoom evidence records innerWidth720 / outerWidth1440 / DPR2 with
scrollWidth720 for RU and EN. The final real result screenshot and RU native200%
save dialog were visually inspected; actions, focus ring and axes are readable.
The real console/network assertions allow only the fixture's known OIDC404 probe,
the deliberately dropped save response and the GET404 of just-deleted history;
there are no unexpected page errors. Controlled cases intentionally inject failures.
No screen-reader speech-output claim is made.

## Debugging and review qualifications

- Initial real truncation expectation was disproved by actual CSV headers: the
  existing fixture produces one trade. The test now distinguishes real bounded
  CSV from controlled truncation; no financial data was changed to pass it.
- Initial strict live-readiness save gate blocked the real ma.ema result. The
  confirmed API/use-case distinction above corrected the UI interpretation.
- Initial save schema incorrectly required organization_id. Real HTTP201 plus
  `_strategy_response_mapping` established its absence; the client parser was
  fixed and keeps unknown scope, rather than modifying the server DTO.
- SSR detail initially had no dashboard route in the minimal fixture. Only the
  disposable fixture gained the production strategy and UI strategy dashboard
  composition roots, with unchanged identity dependencies. Its loaded selected
  row is the proof, not merely a destination URL or visible full UUID text.
- Coordinator review found pending202 inheriting restricted-error polling,
  ordinary transport inheriting an infinite manual deadline, and missing explicit
  readiness/known-save-refusal recovery. Focused browser regressions cover each fix.
- A native resize check sampled before ECharts ResizeObserver completed. The
  same overflow assertion now waits for observed layout settlement; final actual
  widths must pass without overflow. Axis contrast uses existing tokens: muted
  #b1b8c0 on the lightest panel #171c21 is 8.56:1; violet #9a78ff is 5.31:1.
- One test used an unsupported Vitest matcher and one focused Playwright command
  anchored grep against the wrong full test title; both were test invocation
  errors and are not counted as passed validation. The final logout assertion initially targeted a button; the existing control is a link, corrected and rerun. The fixture Ruff line length
  failure was fixed narrowly. The exploratory POST trades assertion incorrectly
  assumed nonempty inline rows; actual200 empty-inline is now recorded separately
  from the nonempty paginated GET/CSV. The delete403 regression was observed red
  against the running earlier build and green after final build. Vite retains its >500kB advisory; no performance
  improvement or budget acceptance is claimed.

Cold self-review checks DTO/source identity, privacy, old202/error precedence,
command locks, recovery scope, truthful live/readiness distinction, delete
eligibility and return behavior, accessible alternatives and v23 layout.
Cold self-review verdict: no remaining actionable S5 blocker found. Independent review is performed by the existing coordinator; no extra reviewer
agent or task was created. Its final verdict is separate from executor checks.

## Compatibility assessment

Baseline: accepted S4 client and the existing API/domain contracts. Candidate:
S5 additions in the same checkout. Inspection covered direct Backtests and
strategy routes/DTO/use cases, lazy read-model construction, current SSR strategy
route/dashboard, session/query/recovery, fixture and focused tests. No external
consumer inventory, target-role overlap or provider behavior is inferred.

| Surface | Classification | Evidence and supported direction |
| --- | --- | --- |
| Opt-in browser result, CSV, save and delete interaction | `compatible-change` | Additive controls consume current own-resource APIs; job and variant URLs remain stable |
| Frontend DTO and private cache/recovery | `compatible-change` | Allowlisted new result/source schemas and separate save record; existing S3 job recovery format/replay prohibition unchanged |
| API/domain/authz, persisted schema, runtime limits and feature defaults | `none` | Production implementations unchanged; no migration or new DTO field required |
| Local fixture/API composition | `compatible-change` | Adds existing strategy CRUD/dashboard routers with production identity wiring; previous cold-fixture inputs, worker and cleanup retained |
| S4 selected-job permission presentation | `compatible-change` | Delete403 additionally closes selected private data; polling/cancel timing and terminal precedence retained |

SSR flag-off rollback remains the existing disabled-by-default setting; saved
strategies/jobs are ordinary current-domain records. No deployment or mixed-version
runtime result is claimed. Unverified organization replay and target-role cutover
remain unavailable rather than inferred compatible.

## Owned paths and handoff

Created:
- `apps/platform-web/src/results-api.ts`
- `apps/platform-web/src/results.tsx`
- `apps/platform-web/src/results-i18n.ts`
- `apps/platform-web/src/strategy-recovery.ts`
- `apps/platform-web/src/delete-history.tsx`
- `apps/platform-web/src/results.test.ts`
- `apps/platform-web/e2e/results.spec.ts`
- this S5 report and S5 browser/regression evidence directories.

Modified existing S1–S4 outputs:
- `apps/platform-web/src/execution.tsx` — result/delete integration and deletion restriction closure.
- `apps/platform-web/src/library.tsx` — history-deleted navigation feedback.
- `apps/platform-web/src/app.tsx` — save recovery subject/logout cleanup.
- `apps/platform-web/src/i18n.ts`, `src/style.css` — result copy and v23 responsive presentation.
- `apps/platform-web/e2e/run.mjs` — separate cold results fixture.
- `apps/platform-web/e2e/{foundation,library,builder,execution}.spec.ts` — S5 regression evidence isolation only.
- `tools/qa/backtests_client_fixture.py` — production strategy/detail read composition for proof.
- `docs/architecture/apps/web/roehub-ui-design-and-delivery-architecture-v1.md` — additive S5 implementation boundary.
- `docs/architecture/project-map/{PROJECT_MAP.md,project-map.json}` — generated source/test inventory.

No deletion or out-of-scope implementation change. Fixture composition is explicitly
authorized proof tooling. Preserve/exclude pre-existing dirty AGENTS, ticket, Web
routing/settings/templates, functional contract/index, plan/pack, root workspace,
CI, prior implementation and accepted evidence. No broad staging or commit.
The accepted v23 SHA-256 remains
`3ff799ac5a5872662dda8b67fc1bd4db0c7860b7de9d84e6597465209d5dd2a4`.

S6 input preflight inspected `06-complete-journey-acceptance.md`: ticket, functional
contract, v23, S1–S4 reports, client package and this S5 report are concrete inputs.
Reproduction is `ROEHUB_PROOF_STAGE=S5 pnpm --filter @roehub/platform-web test:e2e`
after build, using the existing isolated fixture runner. S6 was not started.

## Coordinator acceptance

Accepted for the bounded S5 scope after the executor became idle. Independent
review inspected result/source adapters, read scheduling, save/delete commands,
recovery and permission boundaries, production readiness/save semantics, real
network observations, current screenshots and the final validation report.
The reported readiness retry, cached202 error precedence, manual recovery and
chart/copy findings were corrected with focused browser evidence. Real
materialization, CSV, save/provenance dedupe, SSR strategy identity and deletion
are supported; controlled failures are not substituted for those operations.
No blocking S5 finding remains. The v23 hash was independently rechecked and
matches the accepted baseline. Proceed to S6 integrated T1–T8 acceptance;
target-role authorization and default cutover remain unproved and excluded.
