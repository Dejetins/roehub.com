# Linear-black Backtests Workbench v8 — product-owner refinement

## Status and authority

- Date: `2026-07-31`.
- Checkpoint: `direction_selection`.
- Direction ID: `linear_black_backtests_workbench_v8`.
- Status: `implemented_candidate_ready`.
- Preferred base: v7 Repair 2 raster
  `images/2026-07-31-linear-black-workbench-v7-repair-2.png` with SHA-256
  `8def0c403dad0bf599510aade9dd4e192ebfe3c9658a61fe35769f5a67c4ddb6`.
- Owner assessment of that base: `отличный вариант, просто замечательно!` followed by the
  bounded refinements below. This praise selects the v7 Repair 2 raster as the edit base but does
  not accept it as the final direction.
- Header-layout reference: product-owner attachment
  `codex-clipboard-1accbf68-248c-4cbc-bdc7-05ac26eb6513.png`, SHA-256
  `15244c1c8f4f095c60aa4e4baf79d9cd6f2015923a7151c7f4fb425222f8de4a`.

## Required refinement

1. Remove the decorative mark before `roehub`. Place icon-only Search and Notifications directly
   after `roehub` in the upper-left brand group; do not add a chevron.
2. Align the `Backtests` page title exactly to the x-axis of `Jobs`.
3. Jobs owns exactly four compact icon-only controls: Refresh jobs, Filters using a classic
   horizontal-adjustments glyph rather than a funnel, New backtest using only `+`, and overflow.
4. Move `Overview`, `Equity`, `Drawdown`, `Monthly`, and `Trades` below the KPI summary.
5. Remove the visible Equity / Buy-and-hold / Buy / Sell legend block.
6. Order result actions as labeled `Create strategy`, icon-only Save using a diskette, icon-only
   Export CSV using a downward arrow, then overflow.
7. Extend the complete workspace vertically to meet the `22px` platform-status line.
8. Render Return and Drawdown percentages with one decimal place. Monthly returns use one decimal
   place and color only the numbers; positive and negative cells share the same neutral fill.
9. Make every visible variant row the same height as a job row.
10. Add a visible `Failed` job state with textual failure meaning in addition to its red accent.
11. Raise the working-surface lightness from near-black to a neutral dark gray while preserving
    the black Linear-style direction and measured contrast.

## Invariants

- Keep the v7 Repair 2 information architecture, compact `28px` control lattice, normal-width
  typography, `14px` toolbar icon boxes, charts, Monthly returns, Parameters, status inventory,
  navigation inventory, jobs, variants, and selected-result context except where this owner
  refinement explicitly changes them.
- Do not add a new information architecture, gradient/glass treatment, large pills, or a blue
  theme.
- Run deterministic DOM geometry, clipping, contrast, keyboard, visual, repository-validator, and
  focused-test gates before presenting the new raster.
- Product-owner acceptance remains required and must identify the reviewed v8 raster.

## Proof boundary

The v8 artifact is a deterministic local design specimen and review raster. It does not prove
canonical Figma construction, library publication, Roehub runtime behavior, responsive behavior,
permissions, API behavior, production data, or implementation readiness.
