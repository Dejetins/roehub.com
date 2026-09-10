# S4 — Observe execution and cancel without losing state

Date: 2026-09-08. Execution: one executor, local checkout, S4 only.
Status: implementation and required local checks passed; coordinator independent acceptance is separate.

## Authority and proof boundary

The selected sequential workflow supersedes staged-plan-runner, PACK-CLAIM,
updater, claims, receipts and ledger transitions. The old ledger and prompt pack
were not changed. No Goal, agent, additional task, branch, worktree, stash,
publication, deployment, default cutover, backend/domain/authz/limit/migration,
AGENTS or global skill change was made. No dependency was installed.

This stage implements selected-job execution reads and cancellation using the
existing authenticated own-resource contract. The production API, PostgreSQL,
ClickHouse, artifact computation and worker are real; input is the S1 fixture's
4320 synthetic candles. A read-only PostgreSQL observer captures the short
running interval which can fall entirely between UI refresh hints. For the
separate real queued cancellation, the disposable scheduler is paused before
creation and resumed after POST. No job state is seeded or edited and no API
response is substituted in that lifecycle test. Controlled browser responses
supplement the actual lifecycle for prolonged running, pending, races and errors.

`local_journey_verified=false`; `target_role_cutover_ready=false`. S4 is not
results/export/save acceptance, generic Jobs, provider behavior or deployment.

## Delivered behavior

- `execution.tsx` owns selected detail and one bounded polling timer. Job UUID
  validation and exact response identity checks precede use. Cache keys contain
  authenticated subject and job ID. Obsolete reads and in-flight commands are
  aborted on unmount; an aborted command is not claimed to be undone.
- Polling follows `next_allowed_refresh_at` and `retry_after_seconds`, a minimum
  two-second interval, and GET error Retry-After with a five-second error floor.
  Terminal and 401/403/404 stop automatic job reads. Reconnect cannot bypass a
  stop/cooldown; focus and mount do not create hidden refresh paths. Only the
  selected active job polls; the library remains a separately dated snapshot.
- Measured processed/total units, the server percentage and measurement timestamp
  are displayed. Unknown totals do not fabricate progress. Measurement older
  than 60 seconds is visibly stale even if the last GET is fresh. Failed reads
  retain a stale snapshot. ETA is explicitly unavailable. Terminal state hides
  progress and wins over a 100-percent running measurement.
- `reconcileJob` rejects older observations and prevents a late active response
  from rolling back an observed terminal state. Public started/finished times
  and `terminal_summary.top_variants_count` are allowlisted. Raw terminal errors,
  provider metadata and filesystem paths never enter the detail cache. Results
  remain at their job identity; a completed job links to an honest result-context
  section that states S5 result views/exports are not yet available.
- The native cancel dialog identifies the UUID, starts on Keep running, traps
  Tab/Shift+Tab, closes on Escape and restores trigger focus. Completion while
  open closes the dialog and focuses the terminal status when the trigger no
  longer exists. Dismissal sends no POST.
- Confirming takes a synchronous lock, rechecks the authenticated subject through
  the existing session query, then issues exactly one bodyless cancel POST.
  Duplicate confirmation cannot create another command. Identity outage blocks
  the private surface without a false logout; subject change sends no POST.
- Sending, unresolved/cancel-requested and terminal cancelled are different
  states. HTTP success alone never creates cancelled. Terminal succeeded/failed
  wins a concurrent pending response and retains result context/count. Unknown
  transport/5xx outcome permits only read reconciliation and blocks another cancel
  while unresolved. Commands never retry automatically.
- Cancel409/429 preserves job identity and disables another command until a new
  eligible read succeeds. Its GET uses the same polling deadline; GET429 cannot
  trigger a second independent refetch loop. Cooldown expiry enables only an
  explicit eligible command. 403/404 hides private job data and stops job polling.
- A private per-job query-cache marker retains cancellation ambiguity across
  client-side navigation. Logout/subject change clears that cache. Reload reads
  the known route UUID without POST and restores the server's cancel-request
  marker. It does not promise durable storage of a local cancel command payload;
  fresh server state determines eligibility after reload. S3 create body/key
  recovery, disabled replay, dirty forms and subject isolation are unchanged.
- RU/EN and the inherited v23 palette/panels/control vocabulary are preserved.
  No fictional queue position, ETA, role, result data or operator retry was added.

## Validation

Commands run from `/Users/daniildegtyarev/Projects/roehub.com`.

| Command | Actual result |
| --- | --- |
| `pnpm --filter @roehub/platform-web typecheck` | Passed |
| `pnpm --filter @roehub/platform-web test` | Passed: 119 tests, 5 files, including 15 new S4 cases |
| `pnpm --filter @roehub/platform-web build` | Passed: `main-Bd3mqpNq.js`, `main-DnI8MQyG.css`; existing >500 kB Vite advisory |
| `ROEHUB_PROOF_STAGE=S4 pnpm --filter @roehub/platform-web test:e2e` | Passed: 20/20 Chromium scenarios on separate cold fixtures: 5 S1/S2 (47.8 s), 6 S3 (50.9 s), 9 S4 (2.7 min) |
| `pnpm --filter @roehub/platform-web test:e2e execution.spec.ts --grep 'controlled pending cancellation'` | Passed on final test source: 1 case, 9.6 s including cold fixture |
| `.venv/bin/ruff check tools/qa/backtests_client_fixture.py` | Passed |
| `.venv/bin/python -m pytest -q tests/unit/apps/api/test_backtests_routes.py tests/unit/apps/api/test_ui_backtests_routes.py tests/unit/apps/web/test_web_v2_1_routes.py` | Passed: 98 tests; existing httpx cookie deprecation warning |
| `python3 -m tools.docs.generate_project_map --check` | Initial check detected new source/test inventory; regenerated, final check passed |
| `python3 -m tools.docs.generate_docs_index --check` | Passed |
| `git diff --check` | Passed |
| Fixture cleanup | Passed: private `.local_artifacts/backtests-client` absent; `docker ps --filter name=roehub-client --format '{{.Names}}'` returned no containers |

The final full browser run used the fixed build before every fixture startup.
Afterwards one test-only correction made the already passing disabled-control
assertion explicitly awaited; the affected race scenario was rerun separately.
No runtime source or build changed after the full green run.

The S4 browser suite includes the real worker lifecycle, real queued cancellation
and already-terminal cancel; controlled pending/terminal race; cancel409 →
GET429 with **60 seconds of real cooldown**; dropped cancel response and reload;
read/cancel403 and cancel429; stale/network progress; late reads after route
change; keyboard and terminal-during-dialog focus; six RU/EN × 820/1024/1440
screens with axe; identity outage; real session expiry at the command gate with
31 seconds and zero protected requests; and native Chromium 200% zoom/dialogs.

Evidence is under `browser/S4/` next to this report. Final real jobs: `a4b2767f-d918-4f6a-b231-0231a37d51a0` reached
`succeeded` with one result variant; `6bcd72dd-dd8a-4dca-9b59-1ea20eb4aa0c`
reached `cancelled` from queued after one UI POST. An explicit terminal cancel
returned HTTP 200 and the same cancelled state. `real-lifecycle.json` contains
only allowed job IDs, states, counts, observed units and sanitized network status
observations; screenshots show actual active/completed/cancelled views.
`controlled-race.json`, `controlled-cooldown.json`, `controlled-visual.json`,
`session-expiry.json`, `zoom.json` and named screenshots distinguish controlled
responses from real API behavior. Regression outputs are isolated under
`browser/S4-foundation-regression/`, `S4-library-regression/` and
`S4-builder-regression/`; accepted S1–S3 evidence is preserved.

## Debugging and verification qualifications

- The first lifecycle run completed and cancelled real jobs but could not observe
  running in the browser between two permitted refreshes. Read-only PostgreSQL
  observation now records queued → running → succeeded without extra API polling.
  The final cold run also observed running through the browser GET itself.
- A subsequent cancellation naturally lost to successful completion. This is
  a real race, not a failed cancel implementation. The reproducible queued-cancel
  proof now pauses only the fixture scheduler before the second create. Active
  admission is released by the first job's actual terminal state; limits are not
  weakened. Controlled race tests prove truthful UI terminal precedence.
- A provisional independent needsRead effect could bypass GET429. It was removed
  before acceptance. The coordinator independently identified the same risk.
  A real-time 60-second controlled browser regression passed on the unified timer.
  An earlier fake-clock version failed because clock installation happened after
  timers existed; that run is not counted as proof.
- The normal lifecycle console assertion initially included an expected login
  probe: `/api/auth/oidc/status` returns 404 in the minimal local-password fixture.
  Exact known bootstrap probes are now recorded separately; the workflow itself
  must have no console/HTTP errors. No backend OIDC wiring was altered.
- One provisional native-zoom check lost its client asset when a build replaced
  dist under an already running Web manifest. It is not a product failure or
  accepted evidence. Final regression starts each cold fixture after the fixed
  build and does not rebuild while Web runs.
- An exploratory read-only DB query named nonexistent progress columns and failed;
  it changed no data. The final observer selects only the verified `state` column.
- No pre-fix red run was retained for the reconnect guard; its new 403/429 tests
  verify that reconnect sends no protected job read. Browser/axe checks do not
  claim screen-reader speech output or production readiness.

## Compatibility and review

Baseline: the accepted S3 local client and existing API/domain contracts. Candidate:
S4 changes in the same checkout. Search covered direct jobs routes/DTO/use case,
job read-model timestamps/summary/hints, SSR predecessor, current client readers,
query/session/recovery boundaries, fixture and focused tests; no external consumer
inventory or target-role behavior is claimed.

| Surface | Classification | Evidence / direction |
| --- | --- | --- |
| Opt-in browser job detail, status freshness, cancel interaction | `compatible-change` | Existing UUID/filter/variant links and metadata survive; additive controls/polling consume current server API |
| Frontend read DTO/cache | `compatible-change` | Optional public dates/count accepted; old S3 payloads still parse; private data remains stripped; subject/job keys preserved |
| API/domain/authz, persisted schema, limits and feature settings | `none` | Those implementations were not changed; new client calls current GET/bodyless POST; SSR flag-off rollback remains available |
| Create recovery/idempotency | `none` | No recovery body/key/replay changes; S3 regression selected |
| Local fixture CLI | `compatible-change` | Adds pause-runner/resume-runner for its recorded scheduler PID; run/expire behavior preserved; cleanup resumes before terminate |

Cold self-review verdict: no remaining actionable S4 blocker found. It checked
terminal precedence, duplicate commands, stale responses,
identity gating, GET cooldowns, cache privacy and honest result placeholders.
RU 820, real completed/cancelled and native RU 200% dialog screenshots were
visually inspected; actions and focus indicators are visible, and measured native
zoom has innerWidth 720 / outerWidth 1440 / DPR 2 with no horizontal overflow.
All recorded S4 axe checks passed with zero violations.

Independent review: the coordinator reviewed this report, unified polling and
reconciliation, real lifecycle/cooldown evidence and the completed screenshot.
Verdict: no new blocking findings; the cooldown finding is fixed and proved.
No additional reviewer agent or task was created. Formal stage acceptance is
recorded separately by the coordinator after this executor finishes. Rollback is the unchanged disabled-by-default platform-client
flag and SSR; no data migration or deployment is implied.

## Owned paths and handoff

New files:
- `apps/platform-web/src/execution.tsx`
- `apps/platform-web/src/execution.test.tsx`
- `apps/platform-web/e2e/execution.spec.ts`
- this S4 report and `browser/S4/` plus the three S4 regression evidence directories.

Modified existing S1–S3 outputs:
- `apps/platform-web/src/library.tsx` — delegates detail and shares existing read UI helpers.
- `apps/platform-web/src/library-api.ts` — specific cancel adapter and optional public dates/count.
- `apps/platform-web/src/i18n.ts`, `src/style.css` — execution copy and bounded progress styling.
- `apps/platform-web/e2e/run.mjs` — third cold fixture group.
- `apps/platform-web/e2e/{foundation,library,builder}.spec.ts` — only S4 regression output isolation.
- `tools/qa/backtests_client_fixture.py` — recorded scheduler PID, pause/resume, safe cleanup.
- `docs/architecture/apps/web/roehub-ui-design-and-delivery-architecture-v1.md` — additive S4 section.
- `docs/architecture/project-map/PROJECT_MAP.md`, `project-map.json` — required generated source/test inventory; the docs index and component-map required no S4 content change.

No deletions or out-of-scope implementation changes. Fixture pause/resume is
outside the literal product touch zones but is explicitly authorized local-proof
tooling needed to isolate cancellation without weakening admission limits. Pre-existing dirty AGENTS,
ticket, Web routing/settings/templates, functional contract/index, plan/pack,
workspace dependencies and CI outputs are excluded from S4 ownership. Existing
content was preserved; generated inventory changes retain prior S1–S3 sources.
The old ledger and v23 specimen remain unchanged. Verified v23 SHA-256:
`3ff799ac5a5872662dda8b67fc1bd4db0c7860b7de9d84e6597465209d5dd2a4`.

S5 input preflight: `05-results-and-strategy.md` selects the same ticket,
functional contract, v23, this report and existing client package. `JobEntry` in
`execution.tsx`, `Job` in `library-api.ts`, real succeeded-job metadata and the
cold fixture are concrete producer outputs. Results consumers should retain job
identity and integrate the S5 view into the result-context section; variant URL
identity already survives from S2. S5 has not been executed. The existing
workstation organization_scope_resolver dependency remains unresolved and outside
this scope; direct list/status/cancel remain usable. Default cutover and target
roles still require the separate authorization work.

## Coordinator acceptance

Accepted on 2026-09-08 after independent review of polling, cancellation, terminal
reconciliation, response identity and session cleanup, this report, real lifecycle
and cooldown observations, and the rendered completed-job screenshot. The separate
reconciliation timer finding was fixed and the 60-second GET429 regression passed.
No blocking S4 finding remains. Executor checks are reused; coordinator
`git diff --check` passed. Only the local S4 boundary is accepted; results/save/export,
full T1–T8 acceptance and target-role cutover remain separate.
