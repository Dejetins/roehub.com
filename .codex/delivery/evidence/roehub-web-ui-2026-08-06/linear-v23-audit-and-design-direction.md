---
doc: linear-v23-audit-and-design-direction
version: "1.0"
status: complete
language: en
program_id: roehub-local-platform-web-ui-2026-08
prepared_at: 2026-08-06
---

# Roehub Web UI — Linear Reference vs Pilot v23

## Verdict

Pilot v23 has the correct product instinct: a serious, dark, dense Backtests
workbench with jobs, retained variants, and one analytical result. Its main
problem is not missing polish. It distributes visual responsibility across too
many permanent frames, rules, panels, labels, and simultaneous controls.

The Linear references achieve comparable density by making hierarchy legible
before decoration: one dark canvas, one raised workspace, a small number of
nearby surface tones, quiet text, stable rows, and actions that appear at the
object or moment where they matter. Roehub should preserve v23's analytical
content model while replacing its container-first composition with this
surface-and-interaction grammar.

The Custometry pilot candidate v2 confirms that this is practical for a dense
analytical product. Its shell, metric rail, single context toolbar, anchored
popovers, docked inspector, chart/table hierarchy, and narrow-Web navigation
rail are materially closer to the desired Roehub realization than v23.

## Source roles

| Source | Roehub role | Must not become |
|---|---|---|
| Linear screenshots | normative visual and interaction grammar | copied product structure or branded clone |
| Custometry pilot candidate v2 | supporting analytical-application analogue | Roehub screen/product authority |
| Roehub pilot v23 | Backtest domain and jobs → variants → result evidence | fixed shell, three-column layout or token baseline |
| Current `apps/web` | behavior and compatibility evidence | target frontend architecture or visual baseline |

## What the reference interface is actually doing

The reference is not merely black, compact, or rounded. Its durable qualities
are:

1. **One spatial thesis.** The sidebar belongs to the canvas; the workspace is
   the primary raised plane. Every overlay is visibly above that plane.
2. **Hierarchy through tone and position.** Borders explain major structural
   transitions, not every grouping. Local groups use alignment, rhythm, and
   hover.
3. **Object-centred actions.** A row, project, filter, or notification reveals
   its actions nearby. Persistent chrome is reserved for repeated primary work.
4. **Dense but calm typography.** Most text is muted and regular weight. Strong
   contrast and weight identify the current decision, not every heading.
5. **One overlay family.** Menu, cascade, tooltip, property popover, and toast
   share geometry, tone, elevation, keyboard behavior, and motion.
6. **Stable scanning.** Rows and columns line up; secondary information stays
   available without competing with the current object.
7. **Progressive disclosure without disorientation.** Detail may move into an
   inspector or focused page, but the originating selection and location remain
   visible.

## Material differences

| Area | Linear reference | Pilot v23 | Required Roehub direction |
|---|---|---|---|
| Shell | Dark canvas underneath one lifted workspace | Header and three equally framed work columns | Put grouped navigation on the canvas; lift the workspace as one plane |
| Surfaces | A few close dark tones; overlays clearly elevated | Nearly every panel/card/table is outlined | Reserve borders for shell, major split, overlay, table header, focus and risk state |
| Work model | One active object with contextual actions | Jobs, variants, result, KPIs, charts and status all compete | Establish one primary decision per view and progressively disclose subordinate evidence |
| Navigation | Small top-level set with expandable nested groups | Long flat navigation list | Use Overview, Research, Operations and System trees from the accepted IA |
| Header | One breadcrumb line plus local navigation | Brand/title/actions and workbench controls create extra bands | Use one global header, one local subnavigation row and at most one analytical toolbar |
| Controls | Quiet at rest; selection uses fill; local actions appear on hover | Many outlined, clustered and permanently visible controls | Reduce to icon action, text action, segmented choice and primary commitment action |
| Tables | Row rhythm and selection do most grouping | Outer frame, row rules, cell rules and selected rule accumulate | Keep header boundary and selection/hover; remove routine cell boxing |
| Typography | Restrained hierarchy with few strong weights | 8.6–17 px range, many micro-labels and several strong weights | Raise the practical text floor, limit microcopy, use weight and contrast once per hierarchy level |
| Analytical result | Detail is layered and object-specific | Six metrics, main chart, returns, parameters and tools appear at once | Overview first; Variants, Compare, Variant detail and Raw evidence are separate layers |
| Inspector | Persistent property context without becoming a second app | Third fixed column acts as full-time result page | Use inspector for selected-row properties/comparison; use focused detail for deep analysis |
| States | Hover, focus, menus, tooltips and toasts feel related | States exist but use unrelated shapes and emphasis | Define one interaction grammar and timing system |
| Responsive behavior | Density adapts while object identity stays stable | Fixed `1672×941`, rigid `396/361/remaining` grid | Provide explicit 820/1024/1440 transformations and local overflow rules |

## What remains valuable in v23

- The causal sequence `Backtest job → retained variants → selected result`.
- Persistent job identity and measurable progress.
- Variant ranking, selection, and result comparison as first-class work.
- Dense analytical tables and charts suited to expert long sessions.
- A visible operational status layer.
- A dark, technical, non-marketing product character.

These are product and workbench assets, not permission to preserve the current
three-column geometry or border density.

## What the Custometry analogue resolves

- It demonstrates the intended spatial thesis directly: the navigation belongs
  to the black canvas and the rounded workspace visibly sits above it.
- It shows that eight analytical metrics can read as one flat property rail
  without KPI cards.
- It keeps the primary chart visually dominant while the detailed table remains
  immediately available below it.
- It separates transient settings into anchored popovers and durable context
  into a docked inspector.
- It preserves application identity while transforming the sidebar into a
  `58/52 px` rail at `1024/820` with no root horizontal overflow.

Roehub should reuse these composition mechanics while correcting three limits:
result analysis needs five explicit depth layers, the inspector must stop
docking when it starves the analytical canvas, and narrow tables/metric rails
need visible local-overflow or column-priority affordances rather than silent
clipping.

## Target shell and navigation

### Default authenticated shell

- Canvas: near-black, full viewport.
- Primary navigation: visually belongs to the canvas; grouped trees for
  `Overview`, `Research`, `Operations`, and `System`.
- Workspace: raised one level, with a restrained top and inline-start radius,
  one structural outline, independent scrolling, and a stable breadcrumb.
- Local navigation: tabs or view switcher inside the workspace, directly below
  the breadcrumb when the product area needs it.
- Analytical toolbar: optional and singular; holds view, filter, sort, compare,
  refresh, or saved-view controls.
- Inspector: optional overlaying or adjacent region bound to a selected object;
  it is not a permanent empty third column.
- Status: non-intrusive global status in the shell; incident detail appears only
  when expanded.

### Narrow Web transformation

At 820 px, navigation may collapse to a rail/drawer and inspector may become a
drawer or focused subview. Labels, groups, destinations, current selection, and
object identity must remain unchanged. Wide analytical tables retain local
horizontal overflow rather than forcing a phone layout.

## Backtest experience

### Create

Use a compact guided composer with four decision groups: Intent & Data, Search
& Realism, Resources & Preflight, Review & Run. This should feel like one task
with an explicit current step, not four unrelated pages. Completed groups stay
revisitable and summarize their choices in place.

Only consequential assumptions are permanently visible. Advanced settings
open contextually, while execution blockers, data coverage, cost/slippage,
holdout policy, resource estimate, and selected data-mode rationale can never be
hidden behind an innocuous default.

### Run

After submission, navigate to a durable Backtest identity. Separate queue
state from execution progress. Permit safe background navigation. Show ETA only
with confidence/context; preserve cancellation, retry, failure and
unknown-result recovery semantics.

### Results

The information architecture is intentionally layered:

1. **Overview:** run provenance, completion/data quality, no more than six
   decision metrics, equity/drawdown evidence, warnings, shortlist state and
   next safe action.
2. **Variants:** the retained 10–30 candidates in a rankable, filterable table
   with saved column views and explicit ranking criterion.
3. **Compare:** two to four shortlisted candidates with normalized metrics,
   robustness, parameter deltas, distributions and chart overlays.
4. **Variant detail:** Overview, Robustness, Trades, Parameters,
   Data & provenance, and Logs.
5. **Raw/export:** complete trades, diagnostics and machine-readable evidence.

Rank is a navigation aid, not a readiness verdict. Profit, drawdown,
out-of-sample behavior, parameter/cost sensitivity, data quality and
reproducibility must remain distinguishable.

## Backtest to strategy to trading

The transition is a promotion journey, not another creation path:

1. Select a completed reproducible variant.
2. Inspect promotion readiness and unresolved warnings.
3. Invoke contextual `Create strategy` from that variant.
4. Create an immutable draft linked to the exact Backtest snapshot.
5. Verify connection, market, risk profile, resources and capability.
6. Select paper or permitted live mode.
7. Compare simulated and deployment conditions.
8. Confirm the deployment envelope with recent authentication where required.
9. Enter Live with origin, health, safe stop, reconciliation and audit context.

The primary action may be visually prominent only when the variant is eligible.
When it is not, the same location should explain the blocking evidence instead
of merely disabling a button.

## Control and interaction grammar

Use four control primitives:

1. quiet icon action for local, repeated, labelable actions;
2. text action for reversible navigation or secondary commands;
3. segmented/toggle choice for compact mutually exclusive state;
4. primary commitment action for the single consequential step.

Menus, cascades, popovers, tooltips, inspectors, dialogs and toasts reuse a
single overlay family. Destructive, permission-sensitive and mainnet actions
remain exceptional through intent, copy, confirmation and audit behavior—not
through adding permanent red boxes to the entire interface.

## Current code and delivery implications

- `apps/web` remains current-state SSR, same-origin and compatibility evidence;
  it should not be visually reskinned into the target architecture screen by
  screen.
- The ignored compiled `apps/platform-web/dist` spike is not source authority.
- The target tracked client workspace is created from the accepted G6 handoff,
  following the accepted `apps/web`, `apps/platform-web`, `apps/site`, and
  `@roehub/*` boundary.
- Existing backend routes, DTOs, permissions and mutations are evidence and
  compatibility constraints. Target-only capabilities must be labelled and
  cannot be presented as already implemented.
- Design evidence does not authorize or prove runtime, security, persistence,
  deployment or browser behavior.

## Program recommendation

Proceed with one governed product-wide program. G0 fixes authority and the
platform baseline; G1 creates an exact-cover atlas; G2 groups screens into
journeys, families and implementation-sized waves. G3 should visually resolve
the shell plus representative Backtest surfaces at 820, 1024 and 1440 before
family expansion. G4 and G5 then expand by accepted families and end-to-end
waves. G6 validates the critical journeys and hands implementation an exact,
source-bound contract.

The first finished owner review should therefore show the real Roehub shell,
Backtests library, compact create flow, layered results, and the promotion
entry—not a generic component sheet and not a polished copy of v23.
