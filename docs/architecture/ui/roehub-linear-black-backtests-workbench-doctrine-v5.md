# Roehub Linear-black Backtests Workbench doctrine v5

This doctrine retains the preferred `TAHOE-1` product architecture — Jobs → Variants → Analysis —
and replaces its rejected visual layer with a black, compact, rounded system grounded in Linear's
official 2026 interface refresh.

## Status and authority

- Status: `preferred_direction_refinement_required`.
- Direction ID: `linear_black_backtests_workbench_v5`.
- Product-owner request date: `2026-07-31`.
- Product file after direction selection: `Roehub Authenticated Platform UI`
  (`nzKVsXuCmoTbHJGckHfK3T`).
- Structural input only: `TAHOE-1 Workbench` Jobs → Variants → Analysis hierarchy and its required
  Roehub content inventory.
- Visual inputs: product-owner Linear My Issues screenshot and the official Linear 2026 refresh
  sources listed below.
- The v4 raster's palette, dimensions, surfaces, borders, spacing, row anatomy, and radii are
  explicitly rejected visual inputs.

## Official Linear evidence

- `https://linear.app/now/behind-the-latest-design-refresh`: navigation recedes behind primary
  work; tabs become more compact and rounded; icons are smaller; redundant separators are removed;
  borders are softened and structure is intended to be felt rather than seen; the old cool blue
  palette moves toward a warmer neutral gray.
- `https://linear.app/changelog/2026-03-12-ui-refresh`: headers, navigation, and view controls are
  made consistent; navigation sidebars are dimmer so content receives priority.
- `https://linear.app/now/how-we-redesigned-the-linear-ui`: appearance is generated from base,
  accent, and contrast variables; black/white opacity relationships establish hierarchy;
  structured list, split, and fullscreen layouts are stress-tested together.

Local reference captures from official Linear assets are stored under
`.codex/delivery/evidence/roehub-ui-agent-governed-pilot/references/linear-2026-refresh/`.

## Black neutral palette

The default direction uses no blue-tinted surface:

- app background: near-black neutral `#080809`;
- primary workspace: `#0D0D0F`;
- raised or selected surface: `#151517`;
- hover surface: `#1A1A1D`;
- structural line: white at approximately `6%` opacity;
- stronger selected/focus line: white at approximately `10-12%` opacity;
- primary text: neutral white at approximately `92%` opacity;
- secondary text: neutral white at approximately `60-64%` opacity;
- muted navigation text/icons: neutral white at approximately `38-44%` opacity;
- violet remains a sparse Roehub action/selection accent, never a surface tint;
- green, red, and amber remain exclusively financial/lifecycle semantics.

Only the single primary action may use a filled accent. Selected rows normally use a neutral tonal
surface, an inset indicator, or a small accent glyph instead of a luminous violet outline.

## Radius and surface system

Radii are concentric and visibly softer than v4:

- outer application/workspace shell: `18-20px`;
- principal analysis surface, popover, or drawer: `14-16px`;
- nested chart or table group: `12px`;
- standard button/input: `9-10px`;
- selected list row: `8-10px`;
- tabs, status chips, and compact filters: pill radius (`999px`);
- icon button: circular or `10px` rounded square depending on icon geometry.

Avoid applying the same radius to nested surfaces. With `6px` padding around a `10px` inner
control, the surrounding surface uses approximately `16px`.

## Horizontal and vertical density

The workstation must not create height by wrapping data that fits horizontally:

- global header: `44-48px`;
- compact utility/control height: `28-30px`;
- bottom platform status: `24-28px`;
- job row: `38-44px`, one primary line plus only the minimum inline metadata;
- progress: a `2-3px` inline track or compact percentage, never a tall job card;
- variant row: `32-36px`, one line with parameters and metrics in columns;
- analysis header and mode row should share one compact toolbar where possible;
- KPI values form a single inline metric strip rather than a separate tall card;
- chart receives most remaining height; Monthly and Risk share one compact lower strip;
- no content panel stretches merely to fill its column; unneeded vertical space remains quiet rather
  than turning every item into a card.

## Workbench composition

- A dim compact global navigation rail recedes behind the working area.
- The main workspace is one rounded shell with three internal regions: Jobs, Variants, Analysis.
  Internal regions rely on alignment, tone, and only necessary separators; they are not three
  independently outlined tall cards.
- Jobs are grouped by Running and Completed but represented as compact list rows. At least two
  running jobs retain visible `68%` and `24%` progress and at least three completed jobs remain
  available.
- Variants remain a ranked collection of `12`, with at least ten visible one-line rows. The selected
  row exposes rank, parameters, Return, Sharpe, Drawdown, and Trades without a second text line.
- The selected variant analysis retains all v4 required content and actions: `New backtest`, bell,
  Save, Export CSV, Create strategy, Run strategy, `Overview`, `Equity`, `Drawdown`, `Monthly`,
  `Symbol`, `Trades`, main chart with markers, monthly year/month matrix, risk summary, and bottom
  platform status.
- Trades remain absent from the Equity state and appear only after selecting the Trades mode.

## Automatic rejection gate

Reject before owner review if the specimen:

- contains a visibly blue/navy canvas or blue-tinted panels;
- reproduces the tall cards, two-line variant rows, bordered band stack, or small square radii of
  v4;
- applies a border to every item instead of using spacing and tonal grouping;
- renders jobs as `70px+` cards or variants as `45px+` two-line rows;
- uses sharp `0-6px` corners for the primary shell, panels, buttons, or tabs;
- uses one radius everywhere instead of concentric outer/inner radii;
- makes navigation compete with the chart or selected variant;
- omits any of the accepted v4 functional gates;
- copies Linear branding, issue terminology, data, or exact screen geometry.

## Review boundary

The v5 output is one revised raster direction, not three palette variants. It is reviewed for the
black visual system, radius hierarchy, density, and retained Workbench product architecture. The
raster remains evidence only and is not a component or exact geometry source.

The product owner retained the overall concept and requested the bounded corrections recorded in
`roehub-linear-black-backtests-workbench-doctrine-v6.md`. This v5 doctrine remains visual-system
history and is no longer the current review contract.
