# Roehub Linear-black Backtests Workbench doctrine v7

This doctrine preserves the product-owner-preferred v6 topology and black Linear visual language.
It changes only the control geometry, icon and type proportions, refresh interaction, chart-type
selection, and platform-status information named in the current feedback cycle.

## Status and authority

- Status: `candidate_for_direction_review`.
- Direction ID: `linear_black_backtests_workbench_v7`.
- Product-owner request date: `2026-07-31`.
- Edit target: the v6 raster named by the product owner.
- Visual authority: v6 navigation, jobs/variants/analysis topology, black neutral palette, compact
  desktop density, radii, chart, monthly returns, and Parameters composition.
- Product authority: the current owner feedback overrides v6 only for the bounded surfaces named
  in this doctrine.

## Preserved composition

- Navigation remains exactly `Backtests`, `Strategies`, `Data`, `Signals`, `Reports`, `Alerts`, and
  `Settings` plus user identity.
- Jobs remains `Jobs 6` with `Running 2`, `Queued 1`, and `Completed 3`; Filters and `New backtest`
  remain in the Jobs header.
- Variants retains ten visible compact rows and the selected result.
- Analysis retains `Overview`, `Equity`, `Drawdown`, `Monthly`, and `Trades`; actions remain `Save`,
  `Export CSV`, `Create strategy`, and overflow.
- The selected result retains its one-line KPI summary, Equity chart, Monthly returns, and
  Parameters block.
- `Symbol`, `Run strategy`, `Notebooks`, `Market`, `Risk summary`, and Recent trades remain
  forbidden.

## Control lattice

- Every visible toolbar button, segmented item, compact select, and action control is exactly
  `28px` high. Icon-only controls have a visible `28px × 28px` box.
- Every control intersecting the same toolbar uses the same y-centre, baseline, border weight, and
  optical vertical alignment. Adjacent segmented controls share one `28px` container.
- Labeled controls use `10px` horizontal padding and a `6px` icon/text gap. Control radii are
  `9-10px`; connected segmented controls use matched outer radii and no pill inflation.
- Visible icon glyphs use a fixed `14px × 14px` square box, a consistent `1.5px` stroke, and a
  square `0 0 24 24` source view box. Non-uniform icon scaling is forbidden.
- Compact control text uses `12px` type with `14px` line height, weight `500`, and normal
  `100%` width. Dynamic numbers use tabular figures. Text must never be transformed with vertical
  scaling.
- Visible icon buttons may use a larger non-overlapping implementation hit area, but the raster
  keeps the `28px` compact visual box. The implementation must provide at least a `24px × 24px`
  target and preserve keyboard reachability.

## Refresh interaction

The global header replaces the separate `Refresh` and `Auto 15s` controls with one compact split
control on the shared `28px` lattice:

- left: a `28px × 28px` icon-only immediate-refresh button with accessible name `Refresh now`;
- right: a `42px × 28px` interval trigger showing only `15s` and a chevron;
- the visible words `Refresh` and `Auto` are absent from the resting state;
- the interval trigger has accessible name `Auto-refresh interval: 15 seconds`, exposes menu
  semantics, and reports expanded state;
- its contextual menu contains `Off`, `5s`, `15s` (selected), `30s`, `1m`, a separator, and
  `Refresh now`;
- Enter or Space opens the menu, arrow keys move the active option, Enter selects, and Escape
  closes while restoring focus.

The direction raster shows the menu closed. Current interval remains visible as `15s`.

## Chart-type control

The chart toolbar adds a three-item compact segmented selector before expand, timeframe, and
overflow:

1. line chart, selected;
2. candlestick chart;
3. area chart.

Each item is an aligned `28px × 28px` icon control using the common `14px` glyph box. Runtime
labels/tooltips are `Line chart`, `Candlestick chart`, and `Area chart`. The implementation uses a
single-selection group with an explicit selected state and full keyboard access.

## Platform-status line

- The bottom status is a true edge-to-edge `22px` application-chrome line, not a rounded floating
  card and not a second toolbar.
- All entries use one baseline, `10px` tabular text, compact separators, and no vertical stacking.
- The complete visible fixture is: `API Online`, `Workers 3/4`, `Jobs 2 running`, `Queue 1`,
  `Market data Live`, `Delay 18s`, `Warnings 2`, `Errors 0`, `Critical 0`,
  `Last tick 10:42:18 UTC`, and `Binance connected`.
- Status meaning is always written in text; color is supportive only. `Warnings 2` may use amber.
  Zero error counts remain legible and quiet rather than visually alarming.

## Full interface-craft gate

- Accessibility: icon-only controls have accessible names and tooltips; status never relies on
  color alone; menu and segmented-selection semantics are explicit. Raster review cannot prove
  runtime focus or keyboard behavior.
- Layout: all compact controls follow one height and baseline lattice; the footer is a single
  `22px` line.
- Writing: refresh and interval labels are concise; status labels name both the subsystem and its
  state or count.
- Typography: one compact type scale, normal character proportions, tabular dynamic values, and
  no vertical transforms.
- Colors: preserve v6 neutral black surfaces and existing semantic green, amber, and red accents.
- UI: fixed square icon boxes, consistent stroke, equal button geometry, and explicit chart-type
  selection.

## Automatic rejection gate

Reject before owner review if the specimen:

- changes the v6 topology, black palette, jobs, variants, analysis modes, actions, chart, Monthly
  returns, or Parameters content outside this doctrine;
- shows different button heights or baselines within the same toolbar;
- stretches text or icons vertically, or uses a non-square icon box;
- displays the visible words `Refresh` or `Auto` in the resting global refresh control;
- omits the current `15s` interval or its menu chevron;
- omits any of line, candlestick, or area chart-type controls;
- uses a floating or taller-than-`22px` platform-status band;
- omits market-data state, delay, warning count, error count, or critical count;
- encodes health or severity by color alone;
- reintroduces any element forbidden by the v6 doctrine.

## Review boundary

The v7 output is one bounded raster correction for the same preferred direction. It proves visible
direction intent only. It does not prove exact Figma construction, runtime interaction,
accessibility, responsive behavior, permissions, or production data. Product-owner acceptance is
still required before any canonical Figma work.
