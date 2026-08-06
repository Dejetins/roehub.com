---
doc: ui-intent-baseline
version: "1.0"
status: accepted_owner_input
language: en
program_id: roehub-local-platform-web-ui-2026-08
platform: web
mobile_scope: unauthorized
---

# Roehub Local Platform Web UI — Intent Baseline

## Purpose and audience

Design the complete Web interface for a self-hosted Roehub installation. It
must let an owner, administrator, operator, trader/researcher, or viewer move
from trusted local setup and data readiness through research, Backtests,
strategy promotion, live operation, monitoring, and recovery without hiding
server authority or operational risk.

The public `roehub.com` site is excluded and remains a separate product,
identity, responsive, release, and trust boundary.

## Owner-defined visual intent

The Linear screenshot set is exemplary in its overall interface quality. The
program must internalize its quiet surface hierarchy, high signal density,
minimal chrome, contextual actions, controlled typography, interaction grammar,
and long-session comfort. It must not copy Linear's product model.

Pilot v23 is a useful Roehub-specific direction, especially its analytical
workbench and jobs → variants → result relationship. It must be redesigned to
remove excessive lines, container noise, strong simultaneous emphasis, rigid
desktop-only geometry, and permanently visible secondary controls.

The owner-selected Custometry pilot candidate v2 is the supporting application
analogue: it demonstrates how the Linear grammar can operate in a dense
analytical Web product through the raised workspace, grouped navigation,
metric rail, contextual popovers, docked inspector, and responsive navigation
rail. It is closer to the intended Roehub realization than v23's current
composition, but it does not replace Linear as visual-language authority or
Roehub sources as product authority.

## Priority journey decisions

### Backtest creation

Backtest creation is a compact guided flow, not one large form and not a long
ceremonial wizard. The recommended structure is:

1. **Intent and data** — strategy source, instrument/market, time range, and
   `auto | direct_db | artifact` mode with a visible selection reason.
2. **Search and realism** — small set of parameter ranges, costs/slippage,
   validation/holdout policy, and only the relevant advanced options.
3. **Resources and preflight** — estimated work, resource profile, coverage,
   data freshness, incompatibilities, and actionable blockers.
4. **Review and run** — compact immutable summary, explicit changes, and one
   primary launch action.

Completed steps remain directly revisitable. Defaults and inherited values are
visible but quiet. Advanced fields use progressive disclosure and never hide a
material simulation assumption.

### Large Backtest results

A completed Backtest may produce hundreds of thousands of evaluated
combinations but retains only a bounded set of top variants, commonly 10–30.
The UI should optimize for triage before exhaustive analysis:

1. **Overview** — run identity, data/provenance, completion quality, six or
   fewer decision metrics, equity/drawdown comparison, warnings, and next safe
   action.
2. **Variants** — rankable, filterable, configurable table; saved views;
   shortlisted candidates; explicit ranking criterion; no permanently visible
   detail for every row.
3. **Compare** — two to four shortlisted variants with normalized metrics,
   equity/drawdown overlays, robustness evidence, parameter differences, and
   trade-distribution comparison.
4. **Variant detail** — one selected result with tabs for Overview,
   Robustness, Trades, Parameters, Data & provenance, and Logs. A contextual
   inspector can remain open when comparing properties.
5. **Raw detail/export** — complete trades and diagnostics remain reachable but
   do not dominate the primary result screen.

Ranking and color must not imply that the top profit is automatically the best
or promotion-ready candidate. Robustness, drawdown, out-of-sample behavior,
cost sensitivity, data quality, and reproducibility remain first-class.

### Backtest variant to strategy and trading

A strategy is not launched from a blank user invention. The primary path is:

1. select a completed, reproducible Backtest variant;
2. review its promotion readiness and unresolved warnings;
3. create an immutable strategy draft from the exact variant snapshot;
4. verify connection, market, risk profile, resource limits, and permissions;
5. select paper or permitted live mode;
6. show the differences between simulated and deployment conditions;
7. confirm the deployment envelope;
8. enter the Live workspace with status, safe stop, reconciliation, and the
   originating Backtest permanently linked.

`Create strategy` is the contextual primary action for a promotion-ready
variant. It opens a pre-populated guided promotion flow; it never opens an
empty strategy editor. Live activation remains a separate recent-auth and
permission-aware decision after the strategy draft exists.

## Interaction and layout requirements

- One dark application canvas and one raised workspace plane.
- Grouped expandable navigation aligned to the accepted Roehub information
  architecture, not one flat list.
- Header + local subnavigation + at most one analytical toolbar.
- Four control primitives: quiet icon action, text action, segmented/toggle
  choice, and primary commitment action. Menus/popovers provide variants.
- Contextual popovers for local actions; inspector for persistent multi-field
  comparison/editing; dialogs only for bounded decisions or confirmation.
- Borders only at shell, major region, overlay, and critical table-header
  boundaries. Prefer surface tone, spacing, alignment, and hover for local
  grouping.
- Calm text hierarchy with limited bold and uppercase use.
- One interaction grammar for hover, focus, tooltip, menu, submenu, popover,
  toast, selection, and reduced-motion transitions.
- Responsive Web at `820`, `1024`, and `1440`; mobile-specific navigation and
  `390` layouts remain unauthorized.

## Product and safety boundaries

The design cannot weaken server-enforced capabilities, same-origin/CSRF,
recent-auth, secret redaction, idempotency, unknown-result reconciliation,
installation-owner authority, or safe-stop semantics. It distinguishes current
behavior from `target_not_implemented` behavior and never presents a design
mock as proof of an available operation.
