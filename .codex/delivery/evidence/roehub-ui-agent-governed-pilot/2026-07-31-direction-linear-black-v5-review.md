# Linear-black Backtests Workbench v5 review

## Boundary

- Date: `2026-07-31`.
- Checkpoint: `direction_selection`.
- Status: `awaiting_product_owner_decision`.
- Doctrine: `docs/architecture/ui/roehub-linear-black-backtests-workbench-doctrine-v5.md`.
- Candidate: `images/2026-07-31-linear-black-workbench-v5.png` (`1672 × 941`).
- Generated with the built-in image generation tool.
- The raster is direction-review evidence only; it is not a Figma component, exact geometry source,
  runtime implementation, or production proof.

## Visual source map

| Source | Role |
|---|---|
| product-owner Roehub v4 screenshot | Jobs → Variants → Analysis structure and functional inventory only |
| product-owner Linear My Issues screenshot | neutral-black appearance, rounded shell, individual pill controls, restrained selected surfaces |
| `references/linear-2026-refresh/sidebar-before-after.png` | official refreshed sidebar: dim inactive navigation, smaller icons, rounded selected row |
| `references/linear-2026-refresh/tabs-before-after.png` | official refreshed compact separate tab/pill treatment |
| `references/linear-2026-refresh/borders-before-after-rgb.png` | official refreshed border reduction and softer rounded structure |

Official source pages:

- `https://linear.app/now/behind-the-latest-design-refresh`;
- `https://linear.app/changelog/2026-03-12-ui-refresh`;
- `https://linear.app/now/how-we-redesigned-the-linear-ui`.

No rejected Roehub visual treatment was authorized as a palette, surface, spacing, radius, or
component source.

## Bounded generation audit

1. Initial v5 generation established the black palette and improved radii but was automatically
   rejected because Jobs, Variants, and Analysis still appeared as three separately outlined tall
   cards and variant parameters wrapped to a second line.
2. Repair 1 created one unified workspace shell, flat compact job rows, one-line variant rows, and
   reclaimed height for the chart. It was automatically rejected because the analysis modes were
   rendered as one connected segmented bar.
3. Repair 2 changed only the mode controls to separate compact pills. The ticket's maximum of two
   bounded repair attempts is now exhausted.

## Automatic gate observation

| Gate | Result |
|---|---|
| neutral black canvas; no visible navy/blue surface tint | pass |
| dim navigation recedes behind main analysis | pass |
| one rounded workspace shell rather than three framed columns | pass |
| no border/card around every job | pass |
| two running jobs with visible `68%` and `24%` progress | pass |
| at least three completed jobs | pass |
| `12 variants` with at least ten visible rows | pass |
| variant parameters and metrics share one line | pass |
| compact job and variant row anatomy | pass |
| concentric outer, panel, row, control, and pill radii | pass by visual observation |
| separate compact `Overview`, `Equity`, `Drawdown`, `Monthly`, `Symbol`, `Trades` pills | pass |
| `New backtest`, notification bell, variant actions | pass |
| no Recent trades duplication in Equity state | pass |
| chart remains the dominant analytical object | pass |
| Monthly returns plus Risk summary | pass |
| compact bottom platform status | pass |

## Proof boundary and decision

The inspection proves that the named elements are visibly represented in one raster and that the
automatic visual anti-pattern gate found no remaining named violation. It does not prove exact
token values, component structure, interactions, keyboard behavior, accessibility, localization,
responsive behavior, or runtime data. Only the product owner may accept or reject the direction.
