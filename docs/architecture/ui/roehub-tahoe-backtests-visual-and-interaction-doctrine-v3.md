# Roehub Tahoe Backtests visual and interaction doctrine v3

This doctrine defines the next product-owner direction exploration for the Roehub Backtests Results
workspace. It retains only the useful content relationship from rejected `DIR-003` and replaces its
flat visual treatment with an Apple Tahoe / Liquid Glass-inspired Roehub system.

## Status and authority

- Status: `rejected_by_product_owner`.
- Direction ID: `tahoe_backtests_workspace`.
- Product-owner request date: `2026-07-31`.
- Product file: `Roehub Authenticated Platform UI` (`nzKVsXuCmoTbHJGckHfK3T`).
- Direction page: `01 Direction Review` (`3:2`).
- Style reference: product-owner-supplied Custometry Sales prototype screenshot.
- Content authority: Roehub jobs, top variants, variant detail, result series, stats, paginated
  trades, and chart-overlay contracts.
- Rejected Roehub rasters are not visual inputs.

The product owner rejected all v3 visual specimens because the workstation omitted creation,
platform status, running-job progress, event notifications, drawdown analysis, variant actions,
the monthly result view, and sufficiently compact Linear-style controls. This doctrine is retained
only as failed-attempt evidence and is prohibited as a visual or geometry source. Its successor is
`roehub-tahoe-backtests-workstation-doctrine-v4.md`.

## Product model

The direction state represents a realistic research workflow:

1. Several completed Backtest jobs are visible and comparable.
2. The selected job exposes at least ten ranked variants with different parameter combinations.
3. Every variant has a compact summary for scanning and comparison.
4. One selected variant has an expanded analytical view with full metrics, parameters, chart/data
   switching, and trades on the chart.
5. Full trades remain lazy detail. The UI may show materialization/cache state but never imply that
   every job row persisted the complete trade tape.

The default synthetic review fixture uses three completed jobs. The selected
`dema-1h-long-short-a1b2c3` job exposes `12 variants`; at least ten must be visibly represented or
unambiguously counted. The selected `#01` variant uses:

- `BTCUSDT · Spot · 1h`;
- `DEMA 20/50 · RSI 14/55`;
- Return `+18.42%`;
- Sharpe `1.37`;
- Drawdown `−6.80%`;
- Profit factor `1.61`;
- Win rate `54.8%`;
- Trades `186`.

Other variant values are synthetic direction data and must be visibly plausible, ranked, and
clearly subordinate to the selected variant. They are not runtime evidence.

## Tahoe material system

All three concepts use exactly the same visual system:

- deep navy-graphite edge-to-edge background with subtle atmospheric luminosity;
- regular Liquid Glass treatment for navigation, toolbars, tab bars, segmented controls, floating
  inspectors, and transient controls;
- standard dark materials for charts, tables, dense metrics, and long-form content;
- thin luminous borders, soft inner highlights, restrained blur, and shallow layered shadows that
  establish depth without reducing contrast;
- concentric radii: large shell/panel `14-18px`, nested surface `10-14px`, compact control `8-10px`;
- one restrained violet accent for selection/focus/chart identity across all directions;
- green/red/amber only for financial or lifecycle semantics;
- crisp Inter/SF-like typography with strong primary values, readable secondary labels, and tabular
  numerals;
- monochrome outline icons, generous optical alignment, and polished hover/focus silhouettes;
- no glass effect on every content row or every metric cell.

The reference defines material quality, depth, border/radius craft, typography, and control
treatment. It does not transfer Custometry branding, copy, navigation taxonomy, data, or exact
coordinates.

## Shared content requirements

Every concept must visibly provide:

- global Backtests identity, command/search affordance, refresh, and filtering;
- at least three completed jobs with variant counts;
- selected job identity and market/date context;
- at least ten compact variant results or an explicit `12 variants` collection with enough visible
  rows to judge density and comparison;
- rank, short variant identity, parameter summary, Return, Sharpe, Drawdown, and Trades in the
  compact variant representation;
- selected variant KPI summary including Profit factor and Win rate;
- explicit analytical modes: `Overview`, `Chart`, `Data`, and `Trades` or an equivalent hierarchy
  that preserves all four;
- a large price/equity analysis chart with visible buy/sell trade markers;
- a trades table or a clearly visible Trades state adjacent to the chart;
- an explicit expand/focus control that can grow the selected variant to a full detail workspace;
- cache/materialization/freshness state without a warning banner.

## Three concept directions

### TAHOE-A — Nested Explorer

- Persistent jobs column, adjacent ranked-variants column, and expanded selected-variant workspace.
- Optimized for rapid hierarchical movement: job → variant → analysis.
- Compact result is the variant row; expanded result occupies the main canvas with chart, KPI strip,
  tabs, and a small trades table.
- Trade-off: strongest context preservation, least horizontal room for the chart.

### TAHOE-B — Comparison Matrix

- Jobs form a compact selector above a full-width sortable top-variants matrix.
- Selecting a row opens an analytical lower split/drawer while the comparison table remains visible.
- Optimized for comparing ten to twenty variants before committing to deeper analysis.
- Trade-off: best numeric comparison, more vertical competition between matrix and expanded detail.

### TAHOE-C — Analysis Studio

- Compact jobs/variants rail and a chart-first selected-variant canvas with an attached properties
  inspector.
- A small ranked-results strip preserves compact comparison; the selected variant dominates the
  workspace and can enter focus mode for trade-by-trade chart inspection.
- Optimized for deep analysis after a promising variant is chosen.
- Trade-off: strongest chart/trade workflow, weaker simultaneous comparison of all metrics.

## Hard rejection gate

Reject before owner review if a specimen:

- differs from another direction mainly by colour or decorative styling;
- reuses the flat DIR-003 geometry as its visual skeleton;
- shows only one job or one variant;
- omits compact and expanded representations of the selected variant;
- lacks a chart/data/trades switching model or visible trades on the chart;
- treats every surface as translucent glass and harms data readability;
- copies Custometry branding, content, navigation, or exact layout;
- creates unsupported trading actions or implies persisted full trade tapes for every summary row;
- uses dashboard metric-card grids as the primary information architecture;
- introduces neon, heavy glow, excessive blur, low-contrast text, giant radii, or decorative depth;
- changes the shared palette between the three concepts.

## Selection boundary

The owner receives exactly three named raster concepts in the declared order: `TAHOE-A`,
`TAHOE-B`, and `TAHOE-C`. Selection chooses topology and visual language only. After selection, the
chosen concept is translated into a textual layout/component contract; none of the three rasters is
used as a component or geometry source by downstream executors. Unselected concepts move to
`90 Archive`.
