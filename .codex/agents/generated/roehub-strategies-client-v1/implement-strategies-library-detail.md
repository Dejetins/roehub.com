---
doc: implementation-prompt
version: "1.0"
status: ready_for_user_selected_execution
language: en
execution_mode: standalone
plan_doc: docs/architecture/apps/web/roehub-ui-implementation-plan-v1.md
report_path: .codex/delivery/evidence/ROEHUB-STRATEGIES-CLIENT-2026-09-12.md
---

# Implement Strategies library/detail in the accepted Backtests design

## Task and authority

When the user selects this prompt for execution, implement and verify the complete
Strategies library/detail iteration from the plan's **Next iteration — Strategies
library and detail** section. Continue the existing Backtests result → save strategy
→ inspect saved strategy → return flow. Deliver working repository code and real
local browser/API evidence, not a mockup or another implementation plan.

Workspace: `/Users/daniildegtyarev/Projects/roehub.com`.

This is one standalone execution unit with internal implementation steps. There is
no stage ledger, PACK-CLAIM gate, Goal, automatic next task or dependency on the old
Backtests S1–S6 prompts. Their work is complete. Read current repository instructions
and preserve foreign changes, including the existing documentation edits. Do not
stage, commit, push, merge, deploy, create branches/worktrees/stashes, change global
instructions, or start trading under this prompt. Local disposable verification is
in scope; no real exchange credentials or provider operations are needed.

Do not expand this iteration to new-client create/clone/archive, the strategy editor,
launch-profile mutation, run/stop/restart, manual orders, live monitoring, Connections,
Data, Models, Artifacts or general Jobs. Keep existing SSR functionality reachable.
`journey.strategies.create_and_control` remains partially migrated after this task.

## Bounded source acquisition

Read these sources first, selectively:

**always_read**

1. `docs/architecture/apps/web/roehub-ui-implementation-plan-v1.md`: authority,
   D1–D5, next Strategies iteration and its resolved authoring mapping/handoff.
2. `docs/architecture/apps/web/backtests-ui-iteration-log.md`: accepted visual and
   interaction baseline and its limits.
3. `docs/architecture/apps/web/roehub-ui-functional-contract-v1.md`: shared data,
   errors, commands, routes/accessibility and immutable-strategy rules.

**task_entrypoints**

4. `apps/platform-web/src/app.tsx`: current shell, session handling, navigation.
5. `apps/web/main/app.py`: `/strategies*` routes and `_render_protected_page`.
6. `apps/api/routes/strategies.py`: list/detail/compatibility DTOs and routes.
7. `src/trading/contexts/strategy/application/use_cases/compatibility_readiness.py`:
   `check_strategy` and recording behavior; source provenance limitation below.

**conditional bundles**, open only at the corresponding work boundary:

- Visual implementation: `apps/platform-web/src/{style.css,motion.tsx,library.tsx,backtests-page.tsx,i18n.ts}`;
  inspect the rendered accepted Backtests page before changing shared primitives.
- Routing/gate: `apps/web/main/{settings.py,platform_client.py}`,
  `apps/web/templates/pages/{platform_client.html,strategies.html}`,
  `packages/web-contracts/src/index.ts`, `apps/platform-web/src/main.tsx`.
- Reads/status: `apps/platform-web/src/{api.ts,query-client.ts,library-api.ts}`,
  `apps/api/{routes,dto}/ui_strategies_dashboard.py`; inspect only relevant selector,
  refresh and readiness builders in `apps/api/wiring/modules/ui_strategies_dashboard.py`.
- Saved-result navigation: `apps/platform-web/src/{results.tsx,results-api.ts,strategy-recovery.ts,recovery.ts}`.
- Verification: `apps/platform-web/{package.json,playwright.config.ts}`,
  `apps/platform-web/e2e/{run.mjs,journey.spec.ts,foundation.spec.ts}`,
  `tests/unit/apps/web/test_web_v2_1_routes.py`, `tools/qa/backtests_client_fixture.py`.
- Documentation: the plan, functional registry, Web architecture and existing
  Backtests completion/publication evidence. Consult domain/access contracts only
  for a named field, route, authority or compatibility uncertainty.

Do not read every generated prompt, historical prototype or architecture document.
Expand discovery only for a concrete gap; a producer-created file is not a missing
user input. Current source drift requires a narrow recheck, not a restarted program.

## Implementation decisions and source-backed constraints

### 1. Preserve the current design

Visual authority is the refined Backtests accepted on 2026-09-11 and promoted as the
platform baseline on 2026-09-12, source merge
`191a9f8169dab0639d8fb3456eb73b060eb1d2c4`. The old v23 specimen is historical;
do not render it as the reference for this iteration. Reuse current graphite panels,
violet accents, borders/radii, type hierarchy, compact controls and table rows.

Use one Strategies heading and the existing platform shell. Keep a compact library
beside a larger selected-strategy workspace, following current Backtests. Adapt at
narrow widths without horizontally overflowing the document. Reuse the existing
history collapse pattern where helpful; do not rebuild the accepted Backtests form.
Separate sections with the accepted spacing, avoid stretching short fields, and put
technical IDs/timestamps/raw specification behind subordinate disclosures. The useful
summary is name, instrument, timeframe, specification and available status; there is
no API-backed strategy performance report in this slice to fill with invented cards.

Use one existing animation-speed preference and coordinator for both pages, tabs,
disclosures and geometry changes. Preserve the persisted key and Off/reduced-motion
behavior. Switching while an animation runs must not block input, mix selections,
reset scroll, leave an overlay or lose keyboard focus. Extract shared components only
where needed by both consumers; avoid copied styles, a second shell or a new general
design-system project. Charts are not required here; any actually needed chart must
reuse Apache ECharts and current chart conventions.

### 2. Gate routes without changing existing defaults

Currently the Web shell loads platform assets only for `WEB_BACKTESTS_CLIENT_ENABLED`,
and `_render_protected_page` selects the client only for Backtests. `app.tsx` also
always renders Backtests. Extend these explicit seams rather than injecting a second
application into the old Strategies DOM.

Add independent `WEB_STRATEGIES_CLIENT_ENABLED`, default `false`, with strict true/false
parsing matching the existing setting. Preserve the Backtests flag's meaning/default.
Load the shared built assets when either client is enabled. The authenticated Web
entry selects each page only when its own flag is enabled. Missing opted-in assets
must fail clearly, retaining path validation and private/no-store HTML behavior.

Use an allowlisted, presentation-only bootstrap route capability to select client
links versus full server navigation; retain backward compatibility for the current
bootstrap. Client routing must not expose the new Strategies screen through an
internal link while its server flag is off, or expose client Backtests while only
Strategies is on. Route capability is not server authorization. Update titles, active
navigation, footer and focus behavior to describe the active page.

Keep canonical `/strategies` and `/strategies/{strategy_id}`. `/strategies/new`
remains SSR and must be matched before the ID route. Provide explicit authenticated
SSR management continuation with `view=classic` on Strategies list/detail, preserved
through locale/login continuation. This query selects presentation only: it cannot
enable a disabled client, change permissions or change an API command. A client-side
visit to the classic URL must perform server navigation, with no redirect loop.
Keep the existing `strategy_id` query entry compatible and preserve the existing
non-default `mode` entry through SSR when its behavior is outside this slice.
Do not claim the old `/strategies/new` dashboard is a newly implemented editor.

Verify all four combinations of the two flags, direct links, reload, login return,
classic continuation and locale changes. Turning a flag off changes presentation;
it does not delete or recreate any job/strategy.

### 3. Real list and immutable detail first

Use `GET /api/strategies` for the owned immutable snapshot list and
`GET /api/strategies/{strategy_id}` for the selected specification. Use typed,
allowlisted parsers with the existing `requestJson`/query/session infrastructure;
include subject and selected identity in cache keys and cancel obsolete reads.
Never substitute the first strategy when a requested detail is missing/not visible.

The direct list has no server cursor/search/sort parameters and excludes deleted
records. Search/filter the complete returned list on actual name, instrument, market
type and timeframe fields; support clear/reset and distinguish empty library from
no matches. Do not claim filtering across an unloaded dashboard cursor page. The
dashboard route accepts only `strategy_id`, `state`, `cursor` and `refresh`; DTO
filter fields alone do not prove support for other query parameters.

Rows must be compact, clickable and keyboard-operable, with a clear selected state.
Keep URL identity, filters, library scroll and selection across detail changes and
Back/Forward. A filtered-out selected detail may remain open with clear context;
do not silently select another strategy. A direct detail link must work independently
of list loading. Show indicators in their immutable order with inputs/parameters and
signal template; render unknown supported values as text rather than silently omit
them. Keep schema/kind and optional debug JSON subordinate and render as text only.
Name is server-generated; this task does not introduce rename or spec mutation.

### 4. Optional readiness must not block inspection

Use relevant portions of `GET /api/ui/strategies/dashboard?strategy_id=...` for
observed runtime/profile/readiness and source freshness, independently of the direct
spec read. Bind the response to the selected strategy; never display late or mismatched
selected-strategy data. Unsupported/unconfigured services may legitimately be unavailable.

Distinguish saved specification, runtime observation, strategy compatibility, data
readiness and profile readiness. Missing status is unknown/unavailable, not stopped,
ready or zero. `launchable` compatibility alone does not mean trading is enabled.
Honor server refresh hints, `retry_after_seconds`, `next_allowed_refresh_at` and 429.
Avoid per-row readiness requests and unbounded polling; optional read failure must
leave the immutable detail usable. Do not show raw streams/provider/debug payloads
as product status. Reuse localized reason handling and a short expandable explanation.

The direct compatibility GET records a check/event; it is not a pure lookup or a
provenance endpoint. Use it only for an intentional check when needed, with bounded
requests; do not poll it or call it just to discover an originating backtest. No
opening, refresh, compatibility check or navigation may call run/stop/restart,
launch-from-backtest-variant, manual entry/exit or profile mutation.

### 5. Preserve save semantics and truthful return navigation

`results-api.ts` already validates a save response containing strategy ID and
provenance against the selected job/variant and subject. Preserve its endpoint,
idempotency, deduplication, readiness checks and unknown-outcome recovery. Saving
never starts trading. Do not replay a save to reconstruct a missing source link.

Source inspection found `check_strategy` explicitly supplies `source_job_id=None`
and `source_variant_key=None`. The direct strategy DTO has no reverse provenance;
the existing provenance repository exposes no strategy-ID lookup. Therefore this
slice must not promise persisted provenance retrieval through those reads.

From a validated successful save, open exactly the returned strategy and carry the
originating report as bounded navigation context. Preserve the current explicit
Open-strategy interaction; no unexpected automatic redirect after save. Encode only
an allowlisted local job UUID and validated variant key (for example `from_job` and
`from_variant`), never an arbitrary return URL, actor, organization or serialized API
payload. On reload, treat query context as untrusted navigation input: revalidate its
shape, render a same-origin Backtests route and rely on the destination API for access.
Label it as return navigation, not proof that this strategy originated from that job.

An independently opened strategy with no context has no fabricated source-backtest
link. Ignore malformed context without breaking detail; inaccessible/deleted source
shows the destination's truthful state with a way back. Preserve the original variant
when returning. If context cannot be validated, continue to the strategy without that
link. No new persistence API, provenance schema or unbounded browser storage is needed.

### 6. Error and session behavior are part of the result

Provide localized loading, empty, not-found/not-visible, forbidden, unavailable,
invalid-response and stale/refresh-failed states. Retain last valid data on transient
errors, but do not retain private data after 401/logout/subject change. A 403 is not
an invitation to retry with another identity; 404 must not imply deletion succeeded.
Use the existing sanitized login continuation and shared cancellation/cache-clearing
behavior. Search/filter failures must not discard usable detail, or vice versa.
Handle rapid route changes, interrupted reads and browser Back/Forward without
cross-strategy data flashes. No generic mutation retry or client permission override.

## Work, touch zones and conditional skills

Implement the shared route seam, typed Strategies reads, library/detail composition,
optional readiness and saved-result navigation as one cohesive change, then verify
the full journey. Prefer new feature modules under `apps/platform-web/src` to further
enlarging `results.tsx`; exact new module names are delegated. Expected changes are
that client, `packages/web-contracts`, the Web routing/settings/asset seam, focused
tests/fixture configuration and the named docs. Domain API/storage changes are not
required by the resolved mapping. If new evidence makes them necessary, identify the
exact missing contract and continue independent in-scope work before escalating it.

- Use `better-layout` / `better-ui` when composing panels and shared motion;
  `better-accessibility` for route focus, keyboard rows and disclosures.
- Use `contract-impact-analysis` when finalizing the gate/bootstrap/URL change;
  classify affected dimensions, preserve existing consumers and record rollback.
- Use `browser-qa-evidence` for real visual/runtime acceptance; use `playwright-cli`
  only if terminal browser mechanics are selected, honoring any explicit user browser.
- Use `backend-quality-gates` for changed Python gate behavior and
  `root-cause-debugging` only for an unresolved failure.
- Use `production-risk-review` for a concrete security/session/routing concern;
  do not run publication or deployment skills without separate authority.

The design is settled: do not request approval for routine component/spacing choices
or generate alternative images. Inspect Backtests and verify the implemented result.
If a material new product decision is unavoidable, state the exact question and effect
once; do not abandon work independent of the answer. Never weaken acceptance to pass.

## Validation and observable acceptance

Use the repository's installed toolchain. Before the browser commands, redirect
regression evidence as described below. These are executor commands, not checks
already run by the prompt author:

```sh
pnpm --filter @roehub/platform-web typecheck
pnpm --filter @roehub/platform-web test
pnpm --filter @roehub/platform-web build
.venv/bin/python -m pytest tests/unit/apps/web/test_web_v2_1_routes.py tests/unit/apps/web/test_app_routes.py tests/unit/apps/web/test_security.py
pnpm --filter @roehub/platform-web test:e2e foundation.spec.ts
pnpm --filter @roehub/platform-web test:e2e journey.spec.ts
.venv/bin/python -m tools.docs.generate_docs_index --check
git diff --check
```

Add focused Strategies unit/browser tests and execute their exact saved filenames
through the existing test runner. Register the new suite in `e2e/run.mjs` so the
default browser command includes it. Keep fixture lifetimes isolated where tests
depend on empty libraries. Update existing navigation assertions when behavior
changes; do not retain an assertion that new Strategies must render the old DOM.

Before running `foundation.spec.ts` or `journey.spec.ts`, change their evidence-output
selection to support an isolated directory for this iteration and use it for every
new screenshot/observation. Both currently write into accepted historical Backtests
evidence. Do not overwrite those files, and do not restore them afterward to hide an
overwrite. `journey.spec.ts` also begins with a v23 screenshot: replace that obsolete
reference capture with the accepted current Backtests reference for this run while
preserving the real configure/save/rollback assertions. Keep the old v23 specimen and
historical screenshots unchanged. Record the exact output option and invocation in
the implementation report; isolate output directories across fixture groups.

Run focused Python lint/type gates for changed Python files using the selected skill.
Full CI is not required to author/execute this unpublished UI unit; required local
checks and meaningful real-boundary proof must pass.

| ID | Required acceptance and proof |
|---|---|
| A1 | Real owned list, filtering/no matches/empty, keyboard selection, readable immutable detail and deep-link reload; missing requested ID is not substituted |
| A2 | All four client-flag combinations; authenticated client versus SSR/classic/new/mode entries, private HTML, login/locale continuation, no blank route or loop |
| A3 | Real completed Backtests job → one successful existing save → exact strategy → reload → return to original variant; IDs correlate, duplicate-save behavior retained, zero trading/profile commands |
| A4 | Invalid/missing return context is harmless; standalone detail invents no provenance; deleted/inaccessible source handled truthfully; no arbitrary external return URL |
| A5 | Observed optional readiness/freshness remains independent of spec; controlled unavailable/blocked/429/mismatched/late responses never display false ready or another strategy |
| A6 | Loading/stale/error, 401/logout/subject change, 403/404 and rapid navigation tests; no private-data retention or automatic mutation retry |
| A7 | Current Backtests styling retained at matching 820/1024/1440 widths, RU/EN and 200% zoom; no document overflow, usable focus/keyboard, screenshots and accessibility smoke |
| A8 | One animation preference, interrupted switches, Off/reduced motion and shared Backtests regression; actual controls remain responsive with no ghost layer |
| A9 | Default-off presentation rollback preserves persisted strategy/job; docs and registry update only demonstrated library/detail scope; no full Strategies/target-role/production claim |

Real proof must use disposable local API/auth/data, not intercepted success responses.
Reuse `tools/qa/backtests_client_fixture.py` with the new opt-in configured for the
test Web process. The existing runner uses ports 18480–18483 and disposable databases;
inspect ownership before starting and do not replace the user's review session or kill
unrelated processes. Configure isolated ports/state or use a verified dedicated fixture
when there is a collision. Build before startup because the Web process loads its asset
manifest then. Do not assume an old localhost preview is healthy or contains a job.

The existing fixture uses labelled synthetic candles, which are sufficient to prove
this saved-strategy journey; no new market-data download is needed. Keep the accepted
real-price Backtests example unchanged. Do not use sample browser states as evidence
of provider execution, target organization roles or live trading readiness.

Capture fresh Strategies and unchanged-reference Backtests screenshots at normalized
viewport/locale/state. Test real hover/focus/navigation rather than relying on source
inspection. Controlled error responses supplement, not replace, real list/detail/save.
Keep credentials, cookies, environment dumps, session storage, raw private responses
and unsafe traces out of durable evidence. Store only redacted observations and images.

## Documentation and final handoff

Update the existing plan's next-iteration status only to the observed scope, the
functional registry's relevant `implemented_scope` and Web architecture for the new
flag/bootstrap/classic entry. Preserve other requirements and all unfinished status
tags. Do not rewrite Backtests acceptance or its completed prompts/ledger.

Write the implementation report to
`.codex/delivery/evidence/ROEHUB-STRATEGIES-CLIENT-2026-09-12.md`, with new redacted
assets under `.codex/delivery/evidence/roehub-strategies-client-2026-09-12/`.
These are new output paths, not inputs that must exist before implementation. Check
for existing evidence before writing; preserve another execution's content.

Report A1–A9 with actual evidence/commands/results, owned/new/outside-expected paths,
foreign changes excluded, compatibility (`none`/`compatible-change`/`breaking-change`/
`unknown`), unavailable checks and remaining limitations. Distinguish
`local_journey_verified` from `target_role_cutover_ready`; only the first can become
true in this slice. Include a verified preview URL when available, without credentials.
Final user report must be in Russian. Do not launch another task or publish changes.

## Authoring review — 2026-09-12

This record concerns the instruction, not completed implementation:

- Cold-head review: completed.
- Mode: independent subagent; one read-only review, no recursive review.
- Review scope and source: this standalone prompt, the accepted plan and directly
  referenced API/routing/test code; architecture-review's cold-head checklist.
- Verdict: Release after fixes.
- Findings resolved / unresolved: one Medium issue resolved by requiring regression
  output isolation and replacement of the obsolete v23 reference capture before
  running the old suites; no unresolved material findings.
- Local follow-up check: completed; source/test paths and plan links resolve,
  A1–A9 and standalone boundaries are present, docs index and whitespace checks pass.
  No staged-pack validator is applicable; no ledger or pack is created.
- Residual risks: actual local fixture availability, UI behavior and integrated A1–A9
  proof remain execution work. No runtime, implementation or target-role acceptance
  is claimed by this authoring review.
