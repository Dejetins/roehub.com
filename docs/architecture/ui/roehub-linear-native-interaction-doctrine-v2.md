# Roehub Linear-native interaction doctrine v2

This doctrine translates the product owner's stated Linear concept into a Roehub-owned interaction
and visual direction. It replaces the rejected visual-first doctrine v1. It is a design input for
the Backtests direction checkpoint, not approval of a raster, component, screen, or runtime.

## Status and authority

- Status: `superseded_for_visual_direction`.
- Direction ID: `linear_native_workspace`.
- Product file: `Roehub Authenticated Platform UI` (`nzKVsXuCmoTbHJGckHfK3T`).
- Library file: `Roehub UI Library` (`rgbNUPCuV7q2pARG4Cml8V`).
- Required pilot theme: `graphite`.
- Replaces: `roehub-premium-linear-visual-doctrine-v1.md`, rejected with `DIR-001`.
- Superseded by: `roehub-tahoe-backtests-visual-and-interaction-doctrine-v3.md` after the product
  owner retained the content model but rejected the visual design of `DIR-003`.
- Reuse boundary: functional observations about compact jobs and progressive detail only. Its
  geometry, surface treatment, palette, and component anatomy are prohibited as visual input.
- Owner vocabulary: minimalist, restrained, calm, functional, high-density, structured,
  hierarchical, fast, keyboard-first, developer-focused, dark-first, muted palette, high contrast,
  thin dividers, compact navigation, laconic typography, monochrome icons, contextual actions, low
  visual noise, progressive disclosure, subtle motion, polished microinteractions, command palette,
  desktop-native feel, and information-dense SaaS UI.

## North-star rule

Linear-like quality is an interaction architecture, not a palette. The default state shows only the
information needed to orient, compare, and select. Secondary properties and actions appear in the
context that requests them: selected inspector, hover/focus actions, filter popover, or command
palette. Persistent chrome must earn its space.

`Dark theme + sidebar + thin borders` is not sufficient and must fail review if the interaction
hierarchy still behaves like a dashboard.

## Workspace anatomy

### Stable shell

- Use one continuous inverted-L workspace with quiet shared edges, not a collection of cards.
- At `1440 x 900`, expanded navigation targets the compact end of the accepted `208-320px` range;
  the future collapsed rail remains `44-48px`.
- Navigation labels and monochrome outline icons align to one optical grid. The selected destination
  uses a subtle neutral surface, not a bright filled button.
- Roehub copper is limited to a small brand cue, focus/active identity, or a single high-value
  action. It is never the default icon color or a crypto-terminal motif.

### Context and commands

- A single quiet top/context layer owns Backtests identity, navigation context, and the global
  command trigger with a visible `⌘K` affordance.
- Search and filters are one interaction model, not a row of form fields or permanent chips.
  Default state shows a compact query/filter entry point and a concise active-filter summary.
- Manual refresh, auto-refresh, and degraded freshness occupy one contextual status cluster.
  Degradation never receives a full-width warning band when cached data remains usable.
- Contextual actions appear on selected, hovered, or keyboard-focused rows without shifting layout.

### Worklist and selected-job workspace

- The Backtests worklist is a compact `360-460px` scan-and-select column beside the selected-job
  workspace. This preserves list context without leaving a one-row full-width canvas empty.
- A normal result row targets approximately `44-52px` and may use two tight text baselines, but it
  may not expand into a report card.
- Visible comparison content is limited to: lifecycle cue; strategy/job identity; compact market
  context; the few strongest comparison metrics; and created/freshness context.
- Numeric columns use tabular figures and shared baselines. Labels do not repeat when the column or
  group header already establishes meaning.
- Selection uses a one-step neutral surface change and a restrained leading/focus cue. Hover and
  focus reveal open/more actions in reserved space.

### Selected-job context

- Selecting the row opens a populated detail workspace in the remaining width while preserving
  worklist position and selection. A narrow `300-360px` property rail may sit at its right edge when
  it makes hierarchy clearer.
- The selected context contains only data already required by the pilot: identity, lifecycle,
  market, setup, six result metrics, created time, projection time, and degraded freshness.
- Information is grouped into compact content sections and property rows such as Performance,
  Setup, and Freshness. This is progressive disclosure, not a second dashboard.
- The selected context must not expose an empty placeholder body. The eventual reusable pilot may
  implement only its header first, but the direction specimen may use the brief-required properties
  as non-reusable context.

## Visual system

- Use neutral near-black graphite surfaces with small, legible lightness steps and no blue wash.
- Primary text is crisp off-white; secondary and tertiary text remain readable. Muted does not mean
  low-contrast disappearance.
- Thin separators structure the workspace; elevation is reserved for overlays such as the command
  palette, menus, and popovers.
- Typography uses self-hosted Inter Variable or a faithful system-sans stand-in: `13px` default,
  `12px` metadata, `14-16px` compact titles, medium weights, sentence case, and tabular numerals.
- Icons use one monochrome outline family. No emoji, mixed stroke weights, exchange imagery, neon,
  glow, glass, gradient, decorative card, or oversized radius.

## Interaction-state contract

The visual direction must imply these future states even when the checkpoint shows one static base
state:

| State | Visible evidence in the direction | Runtime proof deferred to implementation |
|---|---|---|
| Command | Global command trigger and `⌘K` cue | Open/search/execute, permissions, focus return |
| Filter | Compact filter summary and explicit entry point | Popover semantics, URL recovery, keyboard editing |
| Selection | Selected row plus populated inspector | Route/history continuity, Escape, resize, persistence |
| Context action | Reserved hover/focus action area | Pointer/keyboard parity and no layout shift |
| Freshness | Cached data stays visible with local degraded text/icon | Refresh request, reconciliation, retry timing |
| Motion | Spatial relationships and stable shell are visually clear | `100-240ms` transitions and reduced-motion behavior |

## Content placement for the Backtests pilot

All accepted brief fields remain present across the selected state, with this hard placement model:

| Layer | Required content |
|---|---|
| Context toolbar | Query/filter entry, active filter summary, manual refresh, auto-refresh, freshness |
| Compact worklist row | State, strategy, short job ID, market/symbol, return, created time |
| Selected-job identity | Strategy, job ID, completed progress, market type, symbol, period |
| Selected-job Performance | Return, Sharpe, drawdown, profit factor, win rate, trades |
| Selected-job Setup | Exchange, market type, symbol, indicator summary, period, direction, combinations |
| Property rail / Freshness | Created time, last projection time, degraded status, manual refresh when permitted |

Duplication is allowed only where it improves orientation between the row and inspector. Missing
content and exposing all content permanently are both failures.

## Hard rejection gate

Reject before owner review if any of these appear:

- a dashboard layout merely recolored to resemble Linear;
- more than two persistent horizontal chrome layers before the list;
- a row taller than `64px` because all fields were exposed at once;
- a full row of removable filter chips in the default state;
- a full-width freshness banner while cached results remain usable;
- an empty full-height inspector or a one-row full-width list dominated by blank canvas;
- permanent row actions that compete with content;
- card grids, boxed metric tiles, spreadsheet cells, glows, gradients, glass, or decorative depth;
- strong orange navigation, multiple orange borders, or crypto-terminal imagery;
- absent command/keyboard affordance, absent contextual action affordance, or absent progressive
  disclosure;
- omitted or invented Backtests data, action, status, or permission semantics;
- Linear branding, copy, entities, icons, or literal screen coordinates.

## Direction checkpoint

The checkpoint receives exactly one named raster specimen: `Roehub Linear-native Backtests`. It is
review evidence only and is never passed to a later executor as a component, layout, or composition
source. If accepted, later work consumes this doctrine plus explicit tokens, component contracts,
and manifests. If rejected, the raster and this doctrine are archived and removed from current
executor context.

Static review may accept only hierarchy and visual language. Keyboard behavior, focus, motion,
accessibility semantics, browser performance, and data behavior require later executable proof.
