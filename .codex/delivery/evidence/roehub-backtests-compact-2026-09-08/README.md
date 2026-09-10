# Compact Backtests console — 2026-09-08

## Authority and result

The owner selected the equal-width five-column console, corrected the period
to dates only, and authorized adapting it to the pilot and rebuilding Backtests.
The accepted image was
`/Users/daniildegtyarev/.codex/generated_images/01a07df7-cdd9-7c71-8ad2-2055ba1bbad9/exec-8e1d4f0e-6fcb-41ce-bdb4-9b744e019d63.png`.
Composition follows that image; visual tokens and control shapes follow v23.

Implemented on `/backtests/new` in the existing platform client:

- Shared equal-width tracks for selectors, numbers and dates; five columns on
  wide screens, fewer columns at content breakpoints. No indicator toolbar inset
  changes the first field axis. At 1440px all fields measure approximately
  216.4px, with less than one CSS pixel of rounding spread.
- Market / Signal / Trade / Risk / Ranking bands; compact field labels and
  controls; optional name and transient-draft notice.
- Date-only native calendars; UTC midnight serialization, start inclusive and
  end exclusive. Invalid/missing/reversed/out-of-bounds dates still block checks.
- Multi-source disclosure, ordered indicator add/reorder/remove, window ranges,
  conditional sizing and TP/SL controls remain editable.
- Policies collapsed initially; preflight review below the form opens after a
  response. Effective policies and warnings remain visible in the review.
- Action controls use normal document flow, avoiding overlap on small screens.
  Changed configuration invalidates preflight; submit/recovery gates are retained.

The original v23 file is unchanged: SHA256
`3ff799ac5a5872662dda8b67fc1bd4db0c7860b7de9d84e6597465209d5dd2a4`.
This receipt describes the current refinement, not all foreign/uncommitted
changes already present in the shared checkout.

## Changed paths

- `apps/platform-web/src/builder.tsx`, `builder-model.ts`, `builder-i18n.ts`,
  `style.css`, `builder.test.tsx`.
- `apps/platform-web/e2e/compact-builder.spec.ts` adds direct browser checks.
  `builder.spec.ts`, `execution.spec.ts`, `results.spec.ts`, `journey.spec.ts`
  adapt date-entry interactions; builder evidence can use an explicit directory.
- `.codex/tickets/2026-09-08-roehub-backtests-client.md` records the owner amendment;
  this evidence directory records verification. No backend implementation changed.
- `docs/architecture/project-map/PROJECT_MAP.md` and `project-map.json` were
  regenerated after inspecting their delta: only the new test/evidence inventory,
  counts and digest changed relative to the pre-refinement generated map.

## Compatibility assessment

Baseline: the existing S6 implementation in this shared checkout immediately
before the refinement. Candidate: the local frontend source and rebuilt assets.

| Surface / consumer | Before → after | Classification |
| --- | --- | --- |
| New configuration editor | Arbitrary UTC timestamps → dates only at UTC midnight | `breaking-change`, explicitly selected by owner; intraday entry is removed |
| Editor layout and disclosure | Tall two-column form/review → equal tracks and disclosures; controls retained | `compatible-change` for the remaining configuration functions |
| Research API and stored requests | Same timestamp DTO, half-open semantics, percentage conversion, normalized body, idempotency/recovery code | `none`; midnight requests have unchanged meaning |
| Existing jobs and unresolved recovery | Existing request records are still read as stored; no date migration or replay change | `none` |
| Backend, auth, config/default opt-in flag | No changed implementation | `none` |

Only initial editable catalog dates and date edits become midnight boundaries.
No server timezone, timestamp schema, persisted job or domain computation is
migrated. Rollback is a frontend rebuild; existing research records remain
readable. Browser locale determines native calendar presentation. API/schema
claims are limited to unchanged source and the tested local server interactions.

## Verification

- `pnpm --filter @roehub/platform-web typecheck` — passed.
- `pnpm --filter @roehub/platform-web test` — 147 tests passed, including date-only
  payloads, missing-date rejection, invalid calendar dates and stale preflight.
- `pnpm --filter @roehub/platform-web build` — passed. Existing large-bundle
  warning remains (main JS approximately 1.11 MB before gzip).
- `ROEHUB_PROOF_REUSE=true pnpm --filter @roehub/platform-web test:e2e compact-builder.spec.ts`
  — 3 passed after final layout correction.
- `ROEHUB_PROOF_REUSE=true ROEHUB_PROOF_DIR=.codex/delivery/evidence/roehub-backtests-compact-2026-09-08/regression pnpm --filter @roehub/platform-web test:e2e builder.spec.ts -g 'real configure|body errors|dirty discard|native Chromium'`
  — 4 passed.
- Docs index check passed. Project map initially reported the new test/evidence
  inventory as drift; regenerated it and its check passed. `git diff --check`
  passed.

Browser mechanic: installed `playwright-cli` for inspection and repository
Playwright tests for reproducible checks; Chromium against the existing
disposable local stack at `http://localhost:18480`, authenticated fixture owner.
No production services or user account changes were used. Tests that expire the
shared session or switch organization membership were intentionally excluded.

Direct evidence:

- [Layout measurements](layout.json): RU/EN at 390/820/1024/1440/1672, no horizontal
  overflow, equal field widths, exactly five axes at wide sizes, two date inputs,
  no time inputs, zero axe violations and no recorded console/page errors.
- [Desktop](ru-1440.png), [mobile](ru-390.png), [expanded controls](expanded.png):
  visually inspected after the final correction.
- [Native 200% zoom](regression/zoom.json): RU/EN, no overflow, zero axe violations,
  primary actions remain keyboard reachable.
- [Journey](journey.json): browser timezone America/New_York; actual preflight
  200 and create 201; exact UTC boundaries `2026-03-26T00:00:00Z` through
  `2026-03-29T00:00:00Z`; fee `0.00075`; worker completion and equity chart.
  [Preflight](preflight.png), [completed result](completed.png).
- [Regression](regression/faults.json): blocking body errors, 422 field focus,
  stale-check invalidation, metadata-only edits, renewed admission rejection,
  storage failure and unresolved outcome. [Create checks](regression/real-create.json):
  one create on double-click and preserved rate/idempotency semantics. Its 409
  observation is the explicit controlled conflict probe, not an unexpected fault.
- Manual review of the shared tracks, date conversion, retained source toggles,
  conditional parameters and existing command/recovery guards found no unresolved
  issue in this scope. The focused suite does not repeat all historical S6 proof.

## Local handoff and limits

Local preview remains running at `http://localhost:18480/backtests/new`.
The disposable Web process caches its startup manifest. Generated asset aliases
were refreshed in ignored `dist` to preserve that process, fixture jobs and
sessions; the canonical build manifest points to the new files. A future fresh
process uses the canonical manifest normally. No tracked runtime workaround added.

Local refinement ready. No push, merge, deployment, default cutover or target-role
authorization claim. The earlier `target_role_cutover_ready=false` boundary remains.
This turn proves the new form-to-result journey and focused regression only.

## Annotation follow-up: short inputs and calmer grouping

The owner's later annotations supersede universal equal control widths. Shared
column starts are retained; short inputs now have an explicit common width.

- `src/style.css`: window from/to/step, fee, slippage, Top N and timeframe 98px;
  cash/sizing amounts 140px; dates at most 182px. All shrink to the available
  track on narrow screens. Panel gap 4 → 16px, group padding 14 → 22px on desktop,
  quieter input backgrounds/borders and less rounded outlines. Wide selectors
  retain room for their actual labels.
- `e2e/compact-builder.spec.ts`: replaced the superseded universal-width
  assertion with short-input widths and retained common-axis checks.
- `apps/web/templates/pages/platform_client.html`: optional URL-encoded
  `asset_version` query adds a resource revision to CSS/JS URLs. This allows a
  running preview to load revised assets despite cached startup filenames.
  Without the parameter, existing URLs are unchanged. Auth gating and private
  HTML caching remain unchanged; covered by the existing Web route test.
- `tests/unit/apps/web/test_web_v2_1_routes.py`: checks encoded asset revision
  and unchanged `private, no-store` behavior.

Verification for this follow-up uses the user's in-app Browser via CUA, not the
earlier Chromium test run. [Measurements](refinement/measurements.json) confirm
98px short controls, shared starts, 16px panel gaps and no horizontal overflow at
390, 1024 and 1525px. RU and EN rendered at desktop; Tab moves from fee to slippage,
two date controls and no time controls remain, and no console errors/warnings
were returned. [Desktop EN](refinement/desktop-en.png),
[desktop RU](refinement/desktop-ru.png), [mobile](refinement/mobile-en.png).
Temporary viewport override was reset after checking.

`typecheck` and `build` passed; the existing bundle-size warning remains.
The focused Web route suite passed 10 tests. The adjusted Playwright scenario
was not rerun in a second browser mechanism; current layout proof is the in-app
Browser measurement and visual inspection above. No new backtest was submitted
for these CSS/template changes. Earlier seven-scenario proof remains historical.
All 147 frontend unit/component tests passed again. Focused Ruff, docs-index,
regenerated project-map and whitespace checks passed.

Compatibility: browser presentation and optional asset URL revision are
`compatible-change`; research/API payloads, stored jobs, server authorization and
feature defaults are unchanged (`none`). This follow-up does not change the
previously accepted date-only restriction. Local preview with fresh resources:
`http://localhost:18480/backtests/new?asset_version=compact-98`.
