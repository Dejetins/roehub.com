# Roehub Tahoe Backtests workstation doctrine v4

This doctrine defines the fourth direction exploration for the Roehub Backtests workstation. It
combines Apple Tahoe-inspired material craft with Linear-style density and discrete mode controls,
while preserving Roehub's real jobs, top-variants, result-series, result-stats, trades-overlay, and
strategy handoff semantics.

## Status and authority

- Status: `rejected_for_visual_design_structure_preferred`.
- Direction ID: `tahoe_backtests_workstation_v4`.
- Product-owner request date: `2026-07-31`.
- Product file: `Roehub Authenticated Platform UI` (`nzKVsXuCmoTbHJGckHfK3T`).
- Direction page: `01 Direction Review` (`3:2`).
- Material reference: product-owner-supplied Custometry Sales prototype screenshot.
- Control-density reference: product-owner-supplied Linear My Issues screenshot.
- Content authority: Roehub Backtests DTOs and the Backtests Results layout in
  `web-ui-product-redesign-master-plan-v2.md`.
- Every rejected Roehub raster and Figma frame is prohibited as a visual, component, layout, or
  geometry input.

The product owner identified `TAHOE-1 Workbench` as the strongest information architecture while
rejecting the visual system of all v4 specimens. The v4 navy palette, tall job cards, two-line
variant rows, repeated rectangular bands, hard borders, and small nearly square radii are
prohibited as downstream visual inputs. The current successor is
`roehub-linear-black-backtests-workbench-doctrine-v5.md`.

## Shared visual system

All three concepts use one visual system and one palette:

- deep navy-graphite canvas with restrained atmospheric luminosity;
- Tahoe-inspired regular glass only for navigation, toolbars, mode buttons, floating inspectors,
  and transient controls; charts, tables, calendars, and dense data use opaque dark materials;
- thin cool borders, subtle inner highlights, shallow layered shadows, and quiet depth;
- compact desktop-native sizing: `12-13px` body text, `28-30px` controls, `32-36px` table rows,
  `6-10px` compact radii, `12-16px` panel radii;
- a restrained violet selection/chart accent; green, red, and amber only for financial and
  lifecycle semantics;
- tabular numerals, monochrome outline icons, compact spacing, keyboard affordances, and no giant
  cards or decorative glass applied to data rows.

The Tahoe reference governs material quality, depth, borders, radii, typography, and polish. The
Linear reference governs only compact control sizing and the treatment of page modes as separate
rounded buttons. Neither reference transfers branding, product taxonomy, copy, or exact geometry.

## Mandatory workstation state

Every concept represents the same realistic state:

- the Backtests header has a compact prominent `New backtest` action, command/search, filters,
  refresh, `Auto 15s`, and a notification bell with unread count;
- the jobs collection contains several completed jobs and at least two running jobs with visible
  percentage progress;
- the selected completed job exposes `12 variants`, with at least ten compact variant entries or
  rows visible for comparison;
- the selected variant exposes Return, Sharpe, Drawdown, Profit factor, Win rate, Trades, readable
  parameters, market, timeframe, and date range;
- discrete Linear-style mode buttons include `Overview`, `Equity`, `Drawdown`, `Monthly`, `Symbol`,
  and `Trades`; the controls are individual compact pills/buttons, not an underline-only tab strip;
- compact next actions include `Save variant`, `Export CSV`, `Create strategy`, and
  `Run strategy`/`Launch strategy`;
- the analytical canvas shows a price/equity chart with buy/sell markers and an explicit expand or
  focus action;
- the current non-Trades mode does not duplicate a Recent trades table; trades appear only after
  the `Trades` mode is selected;
- the space below the chart contains a monthly returns matrix/calendar organised by year and month,
  plus one additional high-value block such as risk summary, drawdown periods, symbol performance,
  or compatibility/readiness;
- a persistent bottom status bar reports compact platform-wide state such as API, workers, queue,
  market-data freshness, and exchange connection.

Synthetic fixture values are direction-review data, not runtime evidence. The selected
`dema-1h-long-short-a1b2c3` job has `12 variants`; variant `#01` uses `BTCUSDT · Spot · 1h`,
`DEMA 20/50 · RSI 14/55`, Return `+18.42%`, Sharpe `1.37`, Drawdown `−6.80%`, Profit factor `1.61`,
Win rate `54.8%`, and `186 trades`.

## Three concept directions

### TAHOE-1 — Workbench

- Persistent hierarchical columns: grouped Jobs, ranked Variants, and selected-variant Analysis.
- Running and completed jobs remain visible while moving through variants.
- Best for repeated job → variant → analysis switching with maximal context retention.
- Trade-off: analysis canvas is narrower than in the other concepts.

### TAHOE-2 — Compare Desk

- Compact job/run queue across the top, full-width variants comparison matrix in the primary
  workspace, and an expanded selected-variant analysis split below or to the side.
- Best for comparing ten to twenty parameter variants before choosing one for deeper analysis.
- Trade-off: the matrix and analysis region compete for vertical space.

### TAHOE-3 — Focus Studio

- Compact jobs drawer, a horizontal ranked-variant ribbon, large chart-first analysis canvas, and a
  narrow metrics/actions inspector.
- Best for trade-by-trade investigation and strategy handoff after selecting a promising variant.
- Trade-off: fewer comparison columns remain visible simultaneously.

## Automatic rejection gate

Reject a specimen before product-owner review if it:

- omits any mandatory workstation-state item above;
- differs from another direction mainly by colour or decoration;
- uses a rejected Roehub frame or raster as a visual skeleton;
- shows only completed results and no way to create a new Backtest;
- hides running-job progress, the notification bell, or the bottom platform status bar;
- omits the `Drawdown` mode or renders modes as oversized/underline-only tabs;
- omits variant save/export/strategy actions;
- duplicates Recent trades below the chart while also exposing a `Trades` mode;
- fails to show a monthly year/month result view and a second useful analytical block;
- shows fewer than several jobs or fails to communicate at least ten variants for the selected job;
- uses oversized controls, giant cards, heavy glow, excessive blur, neon, low contrast, or glass on
  every content row;
- changes the shared visual system or palette between directions.

## Selection boundary

The product owner receives exactly three separate raster concepts in this order: `TAHOE-1`,
`TAHOE-2`, and `TAHOE-3`. Selection chooses information topology and visual language only. The
selected raster is then translated into a textual layout/component contract; no raster becomes a
component or exact geometry source. Unselected concepts are archived.
