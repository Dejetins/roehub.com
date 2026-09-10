---
ticket_id: ROEHUB-BACKTESTS-CLIENT-2026-09-08
status: implemented
user_acceptance_status: accepted
user_accepted_at: 2026-09-11
local_journey_verified: true
target_role_cutover_ready: false
owner: unassigned
scope: local_client_implementation_and_integration
depends_on: []
cutover_dependencies:
  - ROEHUB-AUTHZ-BACKTESTS-2026-07-20
functional_contract: docs/architecture/apps/web/roehub-ui-functional-contract-v1.md
evidence:
  - .codex/delivery/evidence/ROEHUB-BACKTESTS-CLIENT-2026-09-08.md
---

# Implement the first Backtests journey in the platform client

## Product-owner acceptance

The user accepted the current implementation on 2026-09-11 and explicitly
authorized publication through a technical branch merged into `main`, then
branch deletion. See the [iteration log](../../docs/architecture/apps/web/backtests-ui-iteration-log.md).
This authority supersedes earlier publication exclusions for this delivery only;
role cutover and runtime deployment remain outside this acceptance.

## Outcome and readiness boundary

A signed-in user can configure an artifact-backed backtest, preflight and
submit it once, observe its real job state, request cancellation, inspect
variants/results and export trades in the new client, following v23 visually.

The historical `ready` status authorized selection for local implementation.
The current `implemented` status records the locally verified artifact-mode
journey and its explicit user-authorized execution; it grants no publication or
cutover authority. The existing authenticated own-resource contract is the
verified integration boundary. Full target-role behavior and
default-route cutover remain dependent on the named authz ticket, whose own
delegation/envelope dependencies must pass. Never simulate target organization
roles or mark that dependency accepted from frontend checks.

The 2026-09-08 documentation task prepared this ticket without executing it.
That preparation is historical: the user subsequently authorized S1–S6 local
implementation, dependencies, checks and in-scope repairs through ordinary
sequential execution. Git/publication authority is not inherited. The standalone
HTML pilot remains immutable reference material.

## Execution decomposition

On 2026-09-08 the user selected the [UI implementation plan](../../docs/architecture/apps/web/roehub-ui-implementation-plan-v1.md)
and a six-stage [Backtests pack](../agents/generated/roehub-backtests-client-v1/stage-ledger.md).
This ticket remains the full behavior and acceptance source; it does not become
six unrelated feature specifications. The pack's unclaimed draft and unresolved
entry prerequisites describe historical preparation only. The user's subsequent
instruction selected one coordinator and one active executor at a time, with
existing S1–S6 prompts used as sequential implementation assignments. For this run,
PACK-CLAIM, staged-plan-runner entry checks, updater, locks, claims, transition
receipts and ledger management were explicitly superseded. The old ledger and
accepted prompts remain unchanged as historical artifacts. Product requirements,
safety, real browser/API proof and stage reports still apply. Default cutover and
publication remain outside the authorized local implementation scope.

## Selected scope

- `screen.backtests.library`, `screen.backtests.builder`, `screen.backtests.detail`;
  only the shell/shared foundations they require.
- Routes `/backtests`, `/backtests/new`, `/backtests/{job_id}`; stable selected
  variant in a query parameter `variant` with safe URL encoding and server validation.
- Existing artifact mode only. Display it truthfully; do not send invented
  `data_mode` values to current APIs. Full `auto`/`direct_db` requirements remain
  API-BACKTEST-MODES in the functional contract.
- Include cancel, delete history, bounded result charts/statistics/trades/CSV,
  and idempotent create-saved-strategy with a link to existing strategy detail.
  Creating a strategy never starts trading.
- Exclude live launch, operator retry-as-new-job, general Jobs, durable strategy
  drafts, new compute/ETA algorithms, role-policy implementation, public site,
  phone UI, deployment and default client cutover.

## Required sources and owned paths

Read the functional contract/registry, product baseline, current IA/access
contract, and the following actual implementation sources:

- `apps/api/routes/backtests.py`, `apps/api/routes/ui_backtests.py`;
- `apps/api/dto/backtests.py`, `apps/api/dto/ui_backtests.py`;
- `src/trading/contexts/backtest/application/services/v2/preflight.py`;
- `src/trading/contexts/backtest/application/dto/runtime_preflight.py`;
- `src/trading/contexts/backtest/application/use_cases/backtest_jobs.py`;
- `src/trading/contexts/strategy/application/use_cases/create_strategy_from_backtest_variant.py`;
- `apps/web/main/app.py`, `apps/web/main/api_client.py`,
  `apps/web/dist/js/core/api.js`, `apps/web/dist/js/pages/backtests.js`,
  `apps/web/templates/pages/backtests.html`;
- `docs/architecture/backtest/backtest-service-artifact-runtime-v1.ru.md`,
  `docs/architecture/api/api-errors-and-422-payload-v1.md`,
  `docs/architecture/strategy/strategy-api-immutable-crud-clone-run-control-v1.md`.

Owned implementation paths when this ticket is selected: `apps/platform-web/`,
minimum `packages/{tokens,ui,charts,localization,web-contracts}/`, root pnpm
workspace/package/lock and frontend config files; narrowly scoped `apps/web`
routing/settings/template asset integration; frontend CI/build integration and
local fixture tooling; corresponding Web/API integration tests, this ticket,
its evidence and directly affected Web architecture docs. Preserve foreign work.
Backend domain, authorization, migrations, runtime limits and deployment files
are read-only: raise a concrete dependency if their change becomes necessary.

Reuse the accepted React/TypeScript/Vite/Router/Query/Table/i18next/Zod/React
Hook Form/Lucide/Vitest/Testing Library/Playwright/axe/ECharts stack. Resolve and
pin compatible versions when implementation starts. Dependency installation is
within that selected implementation task; it is not performed by ticket authoring.

## Field and validation contract

Obtain catalog/defaults before enabling submission. Missing defaults are an
unavailable state, not permission to use specimen values. Server preflight and
create remain authoritative. Re-preflight after any result-affecting edit.

| UI input / request field | Source and validation |
|---|---|
| Optional job label `strategy_name` | Existing job UI metadata, not immutable Strategy identity; follow `_strategy_name_from_payload`, not specimen name rules |
| `coordinates.exchange`, `market_type`, `symbol` | Workstation instrument universe/catalog; one symbol per submitted request; show unavailable selections and preserve dependent-field validity |
| `timeframe` | `runtime-defaults.supported_timeframes`; do not copy a hard-coded list |
| `time_range.start/end` | UTC, nonempty half-open `[start,end)`; validate date bounds from artifact-date-bounds and preflight; do not silently clamp user input |
| `indicators[]` | Ordered nonempty list, bounded by `guardrails.max_indicator_arity`; only supported indicator IDs |
| `indicator_id`, `sources[]`, `window.start/stop/step` | Use supported IDs/source and parameter specs; positive integer grid, start ≤ stop, supported materialized values; source-less indicators omit/empty sources as allowed; no arbitrary parameter axes invented from catalog labels |
| `risk.mode` | `none` or server-supported `tp_sl_grid` |
| `risk.tp/sl.enabled`, `start_pct/stop_pct/step_pct` | Enable/disable each side explicitly; validate selected levels against `hit_times_grid` and preflight; do not invent uncovered levels |
| `execution.direction_mode` | Server-supported directions and `direction_market_compatibility`; short-like directions require the supported futures market, never silently switch exchange/market |
| `execution.initial_cash_quote`, `fee_rate`, `slippage_rate` | Finite values and server limits; display percentage units and convert exactly once to fractions for rates |
| `execution.sizing` | Supported mode; `quote_amount`, `equity_pct`, `min_quote` or `max_quote` only when that mode requires it; use server range validation |
| `execution.funding`, `profit_lock`, `close_on_end` | Preserve server defaults and normalized effective values; show funding/coverage warnings; do not hard-code a different compute policy or invent unsupported controls |
| `ranking.primary_metric/direction`, `top_n` | Supported metrics, ranking default/direction and `guardrails.max_top_n`; positive integral top N |
| `quality_constraints` | Carry authoritative default/effective min-trade policy; no arbitrary optimizer constraint editor in this first task |

Show selected artifact/source freshness, coverage/funding quality, effective
execution settings and preflight cost/warnings. Do not echo storage paths or
secret-bearing metadata. Treat `errors` in a preflight body as blocking even
when the HTTP response is successful. A successful preflight is advisory:
create validates/admit-checks again and may reject changed availability.

## API bindings

All paths below are prefixed with same-origin `/api` in the browser; send
session cookies. Keep existing origin/CSRF and recent-auth behavior. No direct
browser connection to internal services, artifacts filesystem or exchanges.

| Action | Method/path and response |
|---|---|
| Defaults/catalog/workstation | `GET /backtests/runtime-defaults` → `BacktestRuntimeDefaultsResponse`; `GET /ui/backtests/workstation` → `BacktestWorkstationResponse` |
| Artifact date bounds | `GET /ui/backtests/artifact-date-bounds?exchange=…&market_type=…&symbol=…` |
| Validate | `POST /backtests/preflight` → `BacktestPreflightResponse` with normalized request, hashes, metadata, cost, warnings/errors and funding/direction readiness |
| Submit | `POST /backtests/jobs` with `Idempotency-Key` → 201 new or 200 replay `BacktestJobResponse` |
| List/status | `GET /backtests/jobs` (state/risk_mode, cursor, limit 1–250), `GET /backtests/jobs/{job_id}`; UI workstation supplies supported search/date/instrument filters |
| Summary/variants | `GET /backtests/jobs/{job_id}/summary`, `/top`, `/variants/{variant_key}` |
| Charts/statistics | `GET /backtests/jobs/{job_id}/variants/{variant_key}/{equity,drawdown,monthly-stats,symbol-stats}` → bounded detail or 202 materialization |
| Trades | `GET /backtests/jobs/{job_id}/variants/{variant_key}/trades?page=…&page_size=…` (size ≤100); lazy materialization also has a documented POST `/trades` contract |
| CSV | `GET /backtests/jobs/{job_id}/variants/{variant_key}/trades.csv` → CSV or 202 JSON; honor export bounds/truncation headers |
| Cancel | `POST /backtests/jobs/{job_id}/cancel` → current `BacktestJobResponse`; terminal jobs return their current state |
| Delete history | `DELETE /backtests/jobs/{job_id}` → 204; honor domain eligibility/conflict |
| Strategy compatibility/save | `GET …/compatibility-readiness`; `POST /backtests/jobs/{job_id}/variants/{variant_key}/strategies` with `Idempotency-Key` → 201 or 200 duplicate; open returned strategy identity |

Charts use server series/statistics; switching chart presentation does not
recompute financial metrics in JavaScript. Do not offer candlestick data unless
the selected endpoint actually provides OHLC. Bound points to current API limits.
Cancel obsolete result reads and discard late responses after job/variant change.

## States and transitions

| Event | Transition and required result |
|---|---|
| Open list/detail | loading → ready/empty/unavailable/forbidden; valid deep link loads its object even if absent from the current list page |
| New/edit input | configure → validation errors or preflight pending; input changes invalidate old preflight and retain editable values |
| Preflight resolves | valid → review effective settings/warnings; invalid → field errors and focusable summary, no submit |
| Submit | freeze exact request and one key → submission pending → returned job identity; disable duplicate submit while unresolved |
| Job observed | queued → running → server terminal state; render processed/total units and freshness; terminal state wins over progress |
| Progress becomes stale | retain last measurement with stale label; no timer-generated progress or fabricated ETA; missing reliable estimate reads unavailable |
| Cancel dialog dismissed | return without request; restore trigger focus |
| Cancel confirmed | command pending → server current/cancel-requested/terminal state; never translate HTTP success alone into cancelled |
| Job finishes during cancel | show actual succeeded/failed state and preserved results; no false cancellation |
| Result detail requested | pending/materializing (202) → ready/empty/degraded/failed; honor retry hints without changing job status |
| Delete confirmed | pending → remove only after 204 or authoritative resolution; if selected job removed, return to list with explicit feedback |
| Save strategy confirmed | pending → returned saved strategy link; no run/start call; preserve result context on failure |
| Session expires | stop protected polling/commands, safe login continuation; never turn identity outage into logout |
| Unknown command outcome | unresolved → authoritative read or safe same-key recovery; block fresh automatic submission |

Raw job terminal `succeeded` may be localized as Completed. `cancel_requested_at`
is cancellation pending, not terminal `cancelled`. Do not conflate Backtest
`cancelled` with general runtime `canceled` or its different retry rules.

## Errors and recovery

| Condition | UI / next safe action |
|---|---|
| Validation 422, invalid request or uncovered TP/SL | Map `error.details.errors[].path/code/message` to fields; retain input and correct it; no blind retry |
| Cost/admission rejection | Explain server reason/limits; allow parameter changes or later explicit retry as appropriate |
| 401 | Sign in and return through sanitized local `next` |
| 403 / unavailable capability | Explain restriction; no client role override or automatic retry |
| 404 | Object absent/not visible; offer list; do not infer that an unknown create failed |
| 409 idempotency conflict | Same key with a different normalized/effective request hash conflicts; preserve identity, do not rotate key automatically. Metadata-only changes such as `strategy_name` do not establish a new computation |
| 409 cancel/delete conflict | Refresh authoritative job state and explain current eligibility |
| 429 | Show retry delay; schedule only permitted reads; never submit a command automatically |
| Artifact/queue 503 | Keep configuration; show unavailability; if this followed submit, treat outcome as unknown until resolved |
| Timeout/network/5xx after command | Keep request/key/object identity and unresolved state; do not display success or issue a new command |
| 202 detail/export | Show materialization pending; poll the documented status/read after server delay; do not save JSON as CSV |

For idempotent job create, keep the exact submitted body and key for the logical
attempt, bound to server-derived organization and authenticated user identity.
The current backend has a configured replay window (default 86400 s);
verify the actual retention in the local test configuration before relying on
replay. Never replay after the verified window or with modified input. Existing
job reads reconcile known IDs. A list without the expected job is not proof that
submission failed. No automatic mutation retries in the query client. Read-only unresolved recovery is an intentional safe local-delivery outcome when the existing API cannot bind replay to its original organization.

Create-saved-strategy uses a separate contract: no request body, a source
`job_id + variant_key`, request key and durable strategy provenance/deduplication.
Use `create_strategy_from_backtest_variant.py` for that operation's actual key,
scope and retention behavior; do not apply the job-create 86400-second window.

Bounded implementation decisions for this task (not claims about old UI):

- Ordinary unsent configuration stays in memory; validation/read failures do
  not clear it. Leaving a dirty form asks to discard. Reload does not promise
  durable draft restoration; show that limitation before deliberate discard.
- Before sending submit/save, keep a per-tab recovery record containing only
  the operation kind, exact allowlisted research request, key, creation time,
  authenticated subject, server-derived organization and known result identity.
  Use session storage only;
  never store credentials, cookies, provider payloads or generic API responses.
  Remove resolved payloads; clear private state on logout/subject change and do
  not replay it for another actor. This does not grant a durable-draft feature.
- After reload or re-auth by the same subject and organization, re-read known object state;
  safe same-key recovery requires verified unexpired replay semantics and fresh
  authorization. The current job endpoint resolves organization on each request;
  a client-stored organization is not a server precondition. Replay is disabled
  if the server cannot guarantee the original organization binding, or if scope
  changed or is unknown. Do not invent a client organization override. Use
  read-only history/recovery guidance until a separate API contract makes replay
  safe. If identity, retention or outcome cannot be established, show
  unresolved history/recovery guidance and block a fresh automatic attempt.
- If session storage is unavailable, preserve in-memory recovery and warn that
  leaving the page loses its recovery record before submitting; no silent claim
  of reload recovery. This degraded case must be tested.

## Integration and rollback

Introduce the platform client behind an explicitly disabled-by-default local
feature setting in `apps/web`; keep SSR as default. The implementing agent may
choose the setting name and build paths and document them, but must preserve
authenticated HTML gating, same-origin proxy, `private, no-store`, safe `next`,
locale and existing URL identities. No unauthenticated SPA fallback for protected
routes. Existing unrelated pages remain SSR destinations.

Initial local proof may use existing own-resource policy; target operator/viewer
or delegation behavior is not claimed until API-AUTHZ passes. No browser role
switcher or hard-coded permission grant. Disabling the client setting restores
SSR without data migration; all created jobs/strategies use existing contracts.

## Acceptance and proof

The implementation receipt must distinguish `local_journey_verified` from
`target_role_cutover_ready`; only the first is deliverable before the cutover
dependency passes. Do not mark the full target UI shipped from this ticket.

1. v23-based shell/list/result visual checkpoint, usable at 820/1024/1440 and
   200% zoom, RU/EN; no clipped primary actions. Keyboard dialogs/tabs/focus and
   automated accessibility smoke pass; chart alternatives expose meaning.
2. Real local API-backed configure → preflight → create → progress → terminal
   result/variant/trades/CSV journey, using disposable local data. Mocked network
   states supplement this proof but cannot replace it.
3. Field checks cover UTC `[start,end)`, unsupported catalog/grid/direction,
   percentage conversion, resource rejection, stale preflight and funding warning.
4. Double click and lost response after accepted create produce one job; reload
   retains recovery identity/body without automatically submitting again.
   A changed normalized/effective request with the same key conflicts; a label
   change alone does not create a different computation. Expired/unknown replay
   windows and unknown/changed organization scope block replay, including the
   same subject whose sole organization changes after a lost response. Where
   server-bound replay is unavailable, the tested outcome is truthful unresolved
   state/read-only recovery guidance, not a claimed successful recovery. Test
   create-saved-strategy separately using its source-variant/provenance contract.
5. Cancel pending/race with completion, terminal cancel, deletion conflict and
   unknown outcome preserve truthful state and committed evidence.
6. Materialization 202, degraded/empty result, 429 cooldown, stale source, 401/403,
   transport failure and rapid variant switching show the specified outcomes.
7. Deep-link refresh/back/forward, locale and sign-in continuation work; logout
   clears protected query/recovery state. No secrets in DOM/logs/screenshots.
8. Feature setting off restores original SSR; current API/schema and unrelated
   routes remain compatible. Target authorization dependency reported separately.

Create meaningful frontend component/state and API-adapter tests in the new
workspace. Define `pnpm --filter @roehub/platform-web` scripts `typecheck`,
`test`, `build`, `test:e2e`; then run these against the selected implementation.
Run these existing focused Python checks when integration touches their boundary:

```bash
.venv/bin/python -m pytest -q tests/unit/apps/api/test_backtests_routes.py tests/unit/apps/api/test_ui_backtests_routes.py tests/unit/apps/web/test_web_v2_1_routes.py
python3 -m tools.docs.generate_docs_index --check
python3 -m tools.docs.generate_project_map --check
git diff --check
```

At authoring, frontend scripts were planned deliverables and no check was
reported passed. They now exist in `apps/platform-web/package.json`; current
execution evidence is recorded separately below. Test fixture setup uses the
current disposable local configuration, never a retired host. If no usable local
runtime/fixture exists, report the exact missing prerequisite before claiming
API-backed acceptance. Record console/network checks, fixture identity, viewport,
URL/auth state and redacted evidence in the declared receipt.

Compatibility: additive client/build/feature setting `compatible-change`;
runtime APIs, persistence and domain meaning `none`; default route replacement
and target capability enforcement are outside this ticket's initial delivery.

## Local execution outcome (S6, 2026-09-08)

`local_journey_verified=true`; `target_role_cutover_ready=false`.
The [completion report](../delivery/evidence/ROEHUB-BACKTESTS-CLIENT-2026-09-08.md)
links T1–T8 to final real browser/API and explicitly controlled failure evidence.
Final checks passed: typecheck, 145 frontend tests, build, 35 browser scenarios,
98 focused Python tests, docs index/project map and whitespace checks.

The integrated form-to-result/CSV/save journey and flag-off SSR preserve the actual
job, saved strategy and trade without migration. Result429 manual refresh shares
the automatic cooldown, and chart/table units are explicit in RU/EN. The feature
setting remains disabled by default. Independent coordinator acceptance is
separate from this executor's local verification.

Remaining boundaries are explicit: unavailable workstation extended filters,
server-unbound original-organization replay (read-only unresolved recovery),
full auto/direct_db and reliable ETA, target operator/viewer/delegation policy and
broader Jobs/Artifacts/ingestion/execution-detail UI/API work. The authz Backtests
ticket remains `draft`; its delegation/envelope dependencies remain `ready`, not
accepted. Optional fixture OIDC/Overview/account404 and pre-existing missing
Settings assets are documented, not claimed fixed. No default cutover, push,
merge, deployment or later task was performed. The registry retains all wider
`requires_ui` / `requires_api` coverage.

## Approved compact console refinement (2026-09-08)

The owner approved the five-column calculation-console concept and explicitly
required equal field widths and calendar dates only, with no time inputs. The
subsequent implementation request selects that composition adapted to the v23
pilot palette, typography, controls and surfaces. It supersedes the original
timestamp editor and tall two-column builder composition, without reopening S1–S6.

The editor now uses five equal tracks on wide screens, three/two/one as available
width narrows; Market, Signal, Trade, Risk and Ranking share the same axes.
Both dates represent UTC midnight, preserving the existing half-open interval.
Ordered indicators, multiple price sources, TP/SL grids, sizing, policies,
preflight, submission and recovery remain available. Policies and preflight
details use disclosure; a received preflight opens its review automatically.

Verification and compatibility are recorded in
[the compact-console receipt](../delivery/evidence/roehub-backtests-compact-2026-09-08/README.md).
This is a local frontend refinement. The v23 specimen and default client flag,
API and authorization boundaries remain unchanged.

Owner annotation follow-up: short-value fields should no longer fill their
tracks. This supersedes equal width for every control, while retaining shared
alignment axes. Window values, fees, slippage, Top N and timeframe use 98px;
cash uses 140px. Section padding and panel gaps increase, with quieter control
surfaces. The date-only and functional requirements above remain in force.

## Approved persistent Backtests tabs (2026-09-08)

The owner accepted the current form and explicitly authorized moving it into the
main Backtests workspace: a shared title and History / Settings tabs; New
backtest opens Settings; tab changes retain draft, history filters, scroll and
selection; successful submission opens the created job; Close returns to History
and Reset form is a separate action. This supersedes the separate centered
builder page, retaining the accepted form controls and date-only contract.

Implementation uses persistent in-memory panels and compatible deep links
(`/backtests`, `/backtests/new`, `/backtests/:jobId`). Draft lifetime is the current
Backtests document/session, not durable storage. Leaving the document still
warns for dirty drafts. The created job uses the existing progress/results flow.
A hidden result cannot change the active tab when its default variant arrives.

Evidence and remaining browser gate:
[workspace-tabs receipt](../delivery/evidence/roehub-backtests-compact-2026-09-08/workspace-tabs.md).

Owner visual correction: Settings must retain the accepted bounded width and the
same gray panel surface as History. Full-width transparent presentation is
rejected. Restored the shared panel styles and 90rem builder cap; retained the
approved persistent tabs and all draft/history behavior.

Further owner correction: a separately capped Settings card still diverged from
History. Use History's shared workspace panel surface and header; constrain the
form content inside it. The preceding whole-panel 90rem cap is superseded by a
90rem cap on the inner form, preserving compact controls and the existing workflow.

## Approved inline disclosure above Jobs (2026-09-08)

The owner accepts the current Settings form appearance, but selects a collapsible
block above Jobs instead of separate History/Settings tabs. Keep History and the
selected job visible, retain drafts when collapsed, show instrument/period/check
state in the collapsed row, and collapse/open the created job after submission.
Disclosure must animate smoothly with adjustable motion. This supersedes the tab
placement and full-height Settings surface; retain accepted panel tokens and
compact form fields. Implemented speed choices: off/fast/normal/slow (0/180/320/
520ms), persisted as a non-sensitive browser preference; system reduced-motion
wins. Reference proof remains in the workspace-tabs receipt below its amendments.

## Approved disclosure inside Jobs (2026-09-08)

Owner selected placing the existing animated configuration directly inside Jobs,
after its heading/actions and before the history list. This supersedes the
full-width disclosure above both columns. Preserve compact controls, draft,
motion preference and the right results panel. Keep the Jobs/result column ratio
stable when a result is selected; adapt field rows at existing container widths.

Owner acceptance: the disclosure inside Jobs is accepted and fixed as the current
visual/interaction baseline. Subsequent execution troubleshooting must preserve
this layout. Owner reports inability to launch and inspect a backtest in the
current preview. S1–S6 scope remains the complete artifact-mode configure/check/
submit/progress/results journey, not just the editor. Historical S6 proof does
not establish the current preview's end-to-end behavior after fixture restart and
layout changes. Current Web/API/runner endpoints return 200; fixture source only
precomputes 15m ma.ema signals over the three-day synthetic March 2026 interval.
The owner's last screenshot uses 1h/ma.rma and a 2023 start, outside that fixture's
verified configuration. This is a concrete configuration mismatch, not a reproduced
current request failure. Preserve the accepted UI while resolving the execution
and available-data experience.

## Explicit synthetic demonstration preset

Owner requests a prefilled version requiring only an explicit launch click.
Local-host `?preset=synthetic` opts into the fixture configuration: binance spot
BTCUSDT, 15m, ma.ema/close window10/10/1, 2026-03-26 through 2026-03-29 UTC,
long_only, cash10000, fee0.00075, slippage0.0001, all_in, no TP/SL, return descending,
Top10. Other execution/quality policies remain server defaults. The page labels
synthetic data and performs one automatic real preflight after catalog/bounds are
ready. No create is automatic; existing gates and explicit submission remain.
Normal defaults and accepted layout remain unchanged.
