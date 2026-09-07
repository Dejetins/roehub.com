# UI implementation direction review — 2026-09-08

## Scope and routing

- Classification: review and recommended next steps, not an accepted execution plan.
- Execution unit: inspect accepted UI sources, surviving backlog, implementation and v23.
- Primary skill: `architecture-review`.
- Companion skills: `delivery-orchestrator`, `browser-qa-evidence`, `playwright-cli`.
- Allowed writes: this findings report and its specimen screenshots only.
- Baseline: `c6f6e93e`; `.codex/AGENTS.md` was already modified and was left intact.
- Proof boundary: current source inspection and local specimen browser rendering;
  no authenticated application, backend, authorization or deployment proof.
- No implementation, ticket status change, dependency installation or publication.

## Verdict

`Ready with changes` to begin a bounded implementation task using v23 as the
accepted visual reference. The specimen is not ready to replace the application.
Further standalone whole-pilot redesign is not justified by the observed gaps.
Develop missing states and responsive behavior inside the first real client
journey, then reuse its proven components across subsequent journeys.

The discontinued staged UI workflow must not be revived. The sequence below is
a recommendation for user selection; it creates no new mandatory workflow.

## Observed state

1. `.codex/AGENTS.md` and
   `docs/architecture/apps/web/roehub-ui-design-and-delivery-architecture-v1.md`
   preserve v23 and retire the former workflow. Replacement workflow is undecided.
2. `docs/architecture/platform/roehub-product-transformation-requirements-v1.md`
   already selects pnpm, React/TypeScript, Vite, React Router, TanStack Query/Table,
   i18next, Zod/React Hook Form, Lucide, and the frontend test stack. New product
   charts use Apache ECharts. These are accepted targets, not new proposals.
3. Accepted workspace boundaries are `apps/platform-web`, separate `apps/site`,
   and Roehub-owned tokens/UI/charts/localization/web-contract packages.
   `git ls-files` finds no tracked implementation in these target paths.
4. Current `apps/web` remains FastAPI/Jinja/static JavaScript with a same-origin
   API proxy. Backtest templates and JavaScript wire configuration, preflight,
   job creation, progress, cancellation, deletion and result inspection.
   `apps/api/dto/ui_backtests.py` already models source freshness and
   empty/degraded/unavailable states; reuse their semantics.
5. The accepted local information architecture defines navigation, target
   routes, role boundaries and 12 journeys. The inventory contains 33 local
   surface records, not 33 independent implementation tickets.
6. The authorization kernel ticket is `accepted`; browser mutation envelope
   and delegation core are `ready`; Backtests and other integration tickets
   remain `draft`. These are surviving backlog evidence, not auto-selected work.
   The kernel receipt explicitly does not claim route-level integration.

## Material findings

| Assessment | Fact and source | Consequence and smallest next action | Required proof |
|---|---|---|---|
| OK: visual direction | v23 is explicitly accepted; palette, controls, panel and table treatment are present | Preserve reference; extract only foundations needed by the first journey | Compare implemented screens to v23 |
| Gap / High: responsive implementation | v23 sets body to 1672×941 and `overflow: hidden`; browser measurements retain width 1672 at 1440, 1024 and 820 | Adapt composition during implementation; at 820 the result panel is outside the viewport | Browser captures and usable controls at 820/1024/1440, including zoom and RU/EN |
| Gap / High: journey coverage | v23 shows history/variants/result; `New backtest` click only adds focus, no form or route. Script contains fixture arrays and partial tab/radio interactions | Specify configuration → preflight → submit → queue/progress → result, plus cancel/failure/recovery, in the first Backtests task | Real API-backed journey on disposable local data; reload/deep-link and recovery checks |
| Partial / Medium: navigation fidelity | Specimen menu includes Signals, Reports, Alerts and a Pro-plan identity; accepted IA groups Overview/Research/Operations/System | Reuse visual treatment while composing navigation from accepted IA; do not infer product routes or billing from sample content | Route and navigation mapping with deep-link compatibility |
| Gap / High: target server policy | Current sources and authorization receipts do not establish the full target capability/mutation envelope | Reuse existing authz kernel and relevant tickets; establish policy for the actions included before target-role acceptance | API denial/allow tests for actor, organization, ownership and mutation envelope |
| Partial / Medium: product expansion | Product baseline marks `direct_db` unavailable, dedicated models/first-launch journeys incomplete, ETA calibration unproven | Keep these as explicit backend/product dependencies; do not render simulated availability or delay all visual work until every dependency is solved | Endpoint/runtime evidence for each newly claimed feature |

## Recommended sequence

1. **Bound the first implementation task.** Backtests is the natural first
   journey because the accepted visual source and existing API contracts both
   exist. Record screen/state/action-to-endpoint mapping, URLs, mutation policy,
   responsive acceptance and unresolved data fields. Treat existing artifact
   mode as the initial supported mode; broader modes remain a distinct target
   requiring explicit scope selection. Do not reopen accepted style or stack.
2. **Implement the client foundation within that task.** Introduce the accepted
   platform client and the minimum shared tokens, controls, localization,
   API adapters and ECharts wrapper needed for Backtests. Preserve `apps/web`
   session/proxy authority and SSR fallback. Reuse API semantics and business
   behavior rather than copying the specimen's inline script or old DOM layer.
   First reviewable checkpoint: real history → selected job → variants → result.
3. **Complete and accept the Backtests journey.** Add configuration/preflight,
   submit, progress, cancel, failed/empty/stale/denied/session-loss and recovery
   states. Separate creating a strategy from launching trading. Resolve the
   relevant server-policy prerequisites before exposing target capabilities;
   retain unsupported actions as explicit dependencies. Test against local
   backend fixtures, not only browser mocks. Preserve old URLs and rollback.
4. **Reuse the implementation across adjacent journeys.** Suggested order:
   result → saved strategy/editor; Data and Connections prerequisites before
   strategy execution; then Dashboard/Monitoring and live operations; then
   settings, administration, documentation and dedicated Models as their
   contracts become available. First-launch setup is an explicit prerequisite
   for accepting the complete clean-installation experience, not an implicit
   side effect of redesigning the dashboard.
5. **Handle the public site separately.** It may reuse bounded Roehub packages,
   but has its own identity, responsive, release and deployment boundary.

Per-task acceptance should include the visual checkpoint, focused frontend
tests, applicable API tests, and real browser evidence for the changed journey.
Do not create a complete component catalog before the first journey proves what
is reusable. Do not force settings, editors or administration into the specimen's
three-column composition when the workflow differs.

## Compatibility and residual uncertainty

- This review changes no runtime contract: `none`.
- Future client introduction behind existing URLs is intended as
  `compatible-change`, pending route/session/deep-link proof.
- Target authorization enforcement changes allowed behavior and requires
  explicit compatibility assessment in the implementing ticket; its full impact
  is `unknown` in this review.
- Framework `/docs` relocation is already classified `breaking-change` in the
  accepted IA and should remain a separate migration concern.
- Exact dependency versions, build integration and local API fixture setup must
  be inspected when implementation is selected; none was installed or tested here.
- No current production installation is selected. Local success cannot establish
  deployment or clean-installation readiness.

## Evidence and checks

- Protected specimen SHA-256 verified with `shasum -a 256`:
  `3ff799ac5a5872662dda8b67fc1bd4db0c7860b7de9d84e6597465209d5dd2a4`.
- Browser mechanic: `playwright-cli -s=roehub-ui-review`.
- Direct `file:` navigation was blocked by the browser. Served only the specimen
  directory using `python3 -m http.server 8765 --bind 127.0.0.1 --directory
  .codex/delivery/evidence/roehub-ui-agent-governed-pilot/specimens`.
- Target: `http://127.0.0.1:8765/2026-08-03-linear-black-workbench-v23.html`.
- Commands: `goto`, `resize`, `screenshot`, `eval`, `click`, `console`, `requests`.
- Captured 1672×941 and 820×900; measured body width at 1440/1024/820.
- `New backtest` click: snapshot diff contains focus change only.
- Console: 0 errors, 0 warnings; request log: no dynamic requests and one static
  request. This is specimen evidence, not provider/API proof.
- Screenshots: `ui-next-steps-2026-09-08/v23-1672.png` and `v23-820.png`.
- Task browser and local HTTP server were closed after inspection.
- Application tests, full accessibility audit, authenticated browser flows and
  backend checks were not run: no application code changed.

Cold self-review: `Ready with changes`. Recommendations preserve accepted
visual/architecture decisions, distinguish source facts from runtime proof,
and do not promote historical or draft artifacts to execution authority.
