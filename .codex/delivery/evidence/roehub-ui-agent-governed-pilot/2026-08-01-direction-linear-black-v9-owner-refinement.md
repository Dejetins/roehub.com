# Linear-black Backtests Workbench v9 — product-owner refinement

## Status and authority

- Date: `2026-08-01`.
- Checkpoint: `direction_selection`.
- Direction ID: `linear_black_backtests_workbench_v9`.
- Status: `accepted_by_product_owner`.
- Preferred base: v8 raster
  `images/2026-07-31-linear-black-workbench-v8.png` with SHA-256
  `91e095a664f6f4aee1bdfd1bce026f91ac33423611b72d2f8a111ccec42e997d`.
- Owner assessment of that base: `отличная работа!` followed by ten browser annotations. This
  starts a new bounded owner-feedback cycle and does not accept v8 as the final direction.

## Required refinement

1. Render the Variants column headings and the Running, Queued, Completed, and Failed group labels
   with one font family, font size, line height, and weight.
2. Remove `More variant actions` from the selected-result header.
3. Move Create strategy, Save, and Export CSV to the trailing edge so Export CSV aligns vertically
   with the chart-toolbar overflow button below.
4. Separate New backtest from the three trailing Jobs actions and place it beside the Jobs title.
5. Reduce all three panel heights so the top and bottom exterior insets equal the `4px` inter-panel
   gap.
6. Add a functional `%` / `ABS` unit switch to Monthly returns.
7. Increase the KPI-to-analysis-toolbar gap to the shared `12px` content spacing.
8. Make the Monthly returns card bottom a single homogeneous rounded edge without an empty or
   double bottom band.
9. Wrap the Variants table in the same clipped rounded-container treatment used by Jobs lists so
   the first and last rows inherit rounded outer corners.
10. Move the notification control left so its hit box no longer straddles the Jobs/Variants panel
    boundary below.

## Bounded Repair 1

The product owner opened v9 and requested four final corrections in browser annotations:

1. Match the Variants table's `10px` inline inset to the Jobs list inset; widen Variants from
   `310px` to `328px` and reduce the detail panel by the same `18px`.
2. Reduce running progress rings from `36px` to `32px`; render their values at the same `11px`,
   regular weight used by ordinary Sharpe values.
3. Keep `%` as the selected Monthly unit but remove percent signs from individual table cells.
4. Make both Monthly unit buttons exactly `40 × 28px`.

## Bounded Repair 2

The next product-owner annotations require one shared vertical content axis:

1. Move the complete Variants table shell down so its top border aligns with the first Running job
   list top border.
2. Move the KPI metrics block down to that same axis; let the flexible chart region absorb the
   additional height.

The shared offset is `29px` after each `52px` panel header: `4px` Jobs section padding plus the
`25px` Running label height. This places all three borders exactly `81px` below their panel top.

## Invariants

- Preserve v8 information architecture, product content, chart, status inventory, row density,
  neutral-gray surfaces, percentage precision, control height, and all behavior not explicitly
  overridden above.
- Keep compact controls at `28px`, panel spacing at `4px`, and main detail content spacing at
  `12px`.
- The Monthly units switch must use native buttons, a named radiogroup, roving tabindex, arrow-key
  navigation, explicit checked state, and a stable polite live-region announcement.
- Product-owner acceptance identifies the exact reviewed v9 HTML and its SHA-256 below.

## Validation state

- Static implementation and repository validation ran successfully before the product-owner
  decision.
- The product owner opened and reviewed the v9 local page in the in-app Browser, then explicitly
  accepted it.
- Browser safety policy still prevented Codex from programmatically navigating or reloading the
  `file://` target, and the finalized Browser binding exposed no claimable tab after the decision.
  Therefore no new agent-controlled final screenshot, DOM-geometry, interaction, console, or
  clipping capture accompanies this acceptance.

## Proof boundary

The v9 artifact is a deterministic local design specimen accepted by the product owner as the
visual direction. That decision does not claim agent-controlled final DOM geometry, canonical
Figma construction, Roehub runtime behavior, responsive behavior, permissions, API behavior,
production data, or implementation readiness.

## Product-owner decision

- Decision: `accepted`.
- Exact message: `прекрасно, меня полностью устраивает текущий вариант. принято, зафиксируй это.`
- Accepted artifact:
  `specimens/2026-08-01-linear-black-workbench-v9.html`.
- Accepted SHA-256: `fb09994ffa714fffd1b9988758a50ab68246303461007b01ea252d5c5480471c`.
- Acceptance scope: the named v9 topology and visual language at `direction_selection` only.
- This does not accept Figma variables, styles, icons, components, patterns, a composed Figma
  candidate, runtime implementation, or release readiness.
