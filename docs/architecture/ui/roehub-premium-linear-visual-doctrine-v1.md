# Roehub premium Linear-workspace visual doctrine v1

This doctrine translated the product owner's requested premium Linear-style direction into
Roehub-owned visual rules. Its result, `DIR-001`, was rejected completely by the product owner and
this file is retained only as truthful failed-attempt evidence. It is not an input to later design
work.

## Status and authority

- Status: `rejected_by_product_owner`.
- Rejected artifact: `DIR-001 / Roehub Graphite Workspace / v1 / CANDIDATE`, Figma frame `4:2`.
- Product-owner decision: `2026-07-31`, explicit message
  `DIR-001 отклонен полностью. ты не уловил суть дизайна.`
- Superseded by: `docs/architecture/ui/roehub-linear-native-interaction-doctrine-v2.md`.
- Owner direction: `premium_linear_workspace`.
- Product file: `Roehub Authenticated Platform UI`
  (`nzKVsXuCmoTbHJGckHfK3T`).
- Library file: `Roehub UI Library` (`rgbNUPCuV7q2pARG4Cml8V`).
- Required theme for the pilot: `graphite`.
- Visual references define craft, density, hierarchy, and workspace relationships only. Roehub
  owns product meaning, copy, routes, data, accent, and composition.
- Literal Linear screen replication, branding, icons, text, entities, and exact coordinates are
  prohibited.

## Meaning of premium

Premium does not mean more effects. It means fewer visual treatments, executed with greater
precision: stable chrome, exact alignment, restrained colour, high-quality typography, coherent
density, immediate interaction feedback, and calm continuity between list and detail.

The target is a native-feeling professional workspace, not a crypto terminal, exchange dashboard,
admin template, card grid, or marketing mockup.

## Visual grammar

### Workspace before dashboard

- Direction review must include enough workspace context to judge the inverted-L chrome, page
  identity, view header, primary list surface, and contextual right pane together.
- The shell is continuous. Sidebar, header, list, and detail pane use shared edges and quiet
  separators rather than floating cards.
- Ordinary sections are not placed in cards. Elevation is reserved for overlays and temporary
  surfaces.
- The Backtests list preserves its position and selection while the detail pane is open.

### Quiet hierarchy

- The page title and current view are clear without a hero heading.
- Search, filters, display controls, and refresh state form a compact contextual toolbar rather
  than a row of equal-weight labelled form fields.
- Less-frequent filters use filter chips or one explicit filter popover. Visible active filters
  remain removable and URL/state-recoverable in later runtime work.
- A degraded projection is a compact inline status row. It does not become a large amber banner
  or imply that the completed job failed.

### Density and layout

- Base rhythm: `4px`; common gaps: `4`, `8`, `12`, `16`, `20`, and `24px`.
- Desktop control heights: `28-32px`; effective hit areas remain at least `32px`, with `40px`
  preferred where density permits.
- List group headers are approximately `36-40px`; the information-rich pilot Backtest row uses
  three compact bands within approximately `96-116px` and may not become a dashboard card.
- Inter-group spacing is at least twice the corresponding intra-group gap.
- Shared leading edges and numeric column baselines are mandatory. Repeated decorative dividers
  are not.
- The contextual pane is approximately `360-420px` at the `1440px` reference viewport. Its header
  ends after the required freshness information; an empty artificial body is not part of the
  pilot.

### Typography

- Family: self-hosted Inter Variable with system-sans fallback.
- Page/view title: approximately `15-16px`, weight `600`, tight but not compressed.
- Primary row identity: `13px`, weight `500-600`.
- Body and controls: `13px`, weight `400-500`.
- Metadata: `12px`, weight `400-500`; never use thin weights.
- Job IDs and other code-like values may use the approved mono role; prose and labels remain sans.
- Comparable metrics and changing values use tabular numerals.
- Text is natural sentence case. Truncation requires the full value to remain recoverable in the
  later interactive implementation.

### Colour and surfaces

- The `graphite` workspace uses neutral near-black surfaces with small lightness steps; it is not
  pure black and does not use tinted blue chrome as a default.
- Primary text is near-white, secondary text is neutral grey, and tertiary metadata remains
  readable rather than disappearing into the surface.
- Borders are low-contrast structural separators. A selected row uses a quiet neutral surface
  change plus a restrained state cue, not a full orange outline.
- Roehub copper/orange is a scarce brand/action accent. It may mark the single primary action,
  focus/active identity, or a small brand cue; it does not simultaneously colour buttons, row
  borders, toggles, headings, and warnings.
- Completion and degraded freshness retain separate semantic colours plus text/icon cues. Colour
  alone never carries status.
- No decorative gradients, neon glows, glass panels, strong drop shadows, or coloured page wash.

### Components and details

- Icons use one consistent outline family and optical weight; they do not use emoji, mixed icon
  sets, or pictorial crypto imagery.
- Corners are restrained and concentric. Small controls do not inherit oversized card radii.
- Borders communicate structure or state; layered transparent shadows are used only for real
  elevation.
- Hover actions must not shift layout. Focus is a visible semantic ring, not an orange repaint of
  the whole control.
- Motion is optional, fast, interruptible, and absent from this static direction decision.

## Backtests translation

### Direction-review specimen

The visual-direction specimen is a `1440 x 900` workspace-context view containing:

1. a restrained Roehub authenticated sidebar or rail sufficient to judge the workspace grammar;
2. a top context bar and `Backtests` page/view identity;
3. a compact toolbar with search, one filter entry point, visible active filter chips, manual
   refresh, and quiet auto-refresh/freshness status;
4. one completed Backtest job row using a two-level information hierarchy;
5. one compact degraded-freshness status row;
6. one contextual right-pane header for the selected job.

The shell is review context only and does not expand the first reusable library slice. The final
pilot composition manifest still owns only the bounded Backtests toolbar, job row, degraded state,
and detail-pane header.

### Job-row hierarchy

- Primary band: strategy identity, short job ID, completed state, and created time.
- Metric band: return, Sharpe, drawdown, profit factor, win rate, and trades, aligned with tabular
  numerals.
- Context band: exchange, market type, symbol, indicator summary, period, direction, combination
  count, and refresh state.
- All brief-required values remain recoverable. The row may use aligned metric groups and compact
  semantic pills, but it may not become thirteen unrelated boxed cells or omit content for visual
  cleanliness.

### Detail-pane header

- Selected job and strategy lead.
- Symbol, market type, period, completed state, degraded freshness, and last projection time form
  compact property groups.
- The close control has a visible or accessibility-facing name in the future interactive
  implementation.
- No chart, trades, action footer, configuration form, or blank placeholder body is introduced.

## Hard anti-pattern gate

Reject a specimen before owner review if any of these appear:

- generic crypto/exchange terminal styling;
- seven equal-width boxed filters with labels above them;
- a large amber warning banner;
- orange borders around the selected row or detail pane;
- a spreadsheet-like row of tiny unrelated values;
- decorative cards, glows, gradients, glass, or oversized radii;
- giant empty list or pane regions created as visible placeholders;
- omitted required Backtests fields or invented controls;
- Linear names, icons, text, entities, branding, or literal screen geometry.

## Direction acceptance gate

The direction remains unaccepted until the product owner reviews one named raster specimen that:

- passes the six-domain `better-interface` review;
- passes the brief content inventory;
- shows the declared workspace context without adding it to the reusable pilot scope;
- contains no hard anti-pattern;
- records its exact source references and local generated-image identity;
- is explicitly accepted for visual language only, not as a component or screen source.
