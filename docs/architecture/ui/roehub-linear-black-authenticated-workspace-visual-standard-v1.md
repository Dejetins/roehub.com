# Roehub linear-black authenticated workspace visual standard v1

This standard translates the product-owner-accepted Backtests Workbench v9 direction into a
reusable visual grammar for future Roehub authenticated desktop screens. It defines foundations,
component geometry, density, hierarchy, states, and rejection rules. It does not copy a screen
layout into every route and does not make a future screen accepted by analogy.

## Status and authority

- Status: `proposed_for_library_slice_review`.
- Translation date: `2026-08-01`.
- Translation request authority: product owner requested that the accepted visual language be
  expressed as reproducible rules for later screens.
- Artifact-specific acceptance: pending at `library_slice_review`.
- Accepted direction: `linear_black_backtests_workbench_v9`.
- Accepted specimen:
  `.codex/delivery/evidence/roehub-ui-agent-governed-pilot/specimens/2026-08-01-linear-black-workbench-v9.html`.
- Accepted specimen SHA-256:
  `fb09994ffa714fffd1b9988758a50ab68246303461007b01ea252d5c5480471c`.
- Acceptance evidence:
  `.codex/delivery/evidence/roehub-ui-agent-governed-pilot/2026-08-01-direction-linear-black-v9-owner-acceptance.md`.
- Figma delivery authority:
  `docs/architecture/ui/roehub-agent-governed-figma-delivery-standard-v2.md`.

The accepted HTML is visual evidence, not a component source. Reusable implementation comes only
from gated and published assets in `Roehub UI Library`. Product data, actions, permissions, and
states continue to come from current route and runtime contracts.

## Applicability

Use this proposed standard to build and review the Backtests library slice. Only after the product
owner accepts its exact revision at `library_slice_review` may a future ticket select it for an
authenticated desktop workspace such as Strategies, Data, Signals, Reports, Alerts, or Settings.
Even then, carry forward only the shared foundations and component grammar; design each screen
around its own tasks and authoritative content. Do not copy Backtests-specific columns, chart
geometry, labels, or actions into another screen without a product contract.

Responsive behavior, runtime focus behavior, screen-reader output, and route-specific information
architecture require independent implementation evidence. They are not inferred from v9.

## Product posture

1. The interface is an operations workstation, not a marketing dashboard.
2. Hierarchy comes from alignment, density, typography, borders, and restrained surface steps.
3. The canvas is continuous. Panels may be rounded, but they do not float as disconnected cards.
4. Violet identifies selection and the primary action. Semantic colors communicate state and are
   always paired with text or an icon.
5. High information density is intentional. It must remain legible, optically aligned, and calm.
6. Linear may inform functional grouping only. Do not copy Linear geometry, taxonomy, assets, or
   unsupported product concepts.

## Foundation tokens

Token names below are the canonical semantic intent for the first Figma library slice. Exact Figma
variable IDs are recorded by the component registry after creation.

### Color

| Token | Value | Use |
|---|---:|---|
| `color.canvas` | `#0B0E11` | Application canvas |
| `color.chrome` | `#15191D` | Global header chrome |
| `color.surface` | `#151A1F` | Primary panel and card surface |
| `color.surface.subtle` | `#191F25` | Active navigation and quiet raised state |
| `color.surface.raised` | `#1D242B` | Reserved higher neutral step |
| `color.surface.list` | `#14191E` | Row-container background |
| `color.surface.control` | `#1A2026` | Compact control background |
| `color.surface.selected` | `#20262D` | Selected row or active neutral surface |
| `color.border.default` | `#353D45` | Panel and emphasized boundary |
| `color.border.soft` | `#293139` | Row and internal separator |
| `color.text.primary` | `#F1F3F5` | Primary content |
| `color.text.secondary` | `#B1B8C0` | Secondary content and labels |
| `color.text.quiet` | `#858E98` | Tertiary metadata |
| `color.accent.reference` | `#7952F4` | General accepted-v9 violet reference |
| `color.accent.action` | `#6540DF` | Primary action fill |
| `color.accent.action-border` | `#8264EC` | Primary action inner boundary |
| `color.accent.control-selected` | `#5C35D1` | Selected segmented/control fill |
| `color.accent.control-selected-border` | `#7F62E8` | Selected control inner boundary |
| `color.accent.selection-primary` | `#8B5CFF` | Selected job-row leading rule |
| `color.accent.selection-secondary` | `#8158F5` | Selected variant-row leading rule |
| `color.accent.navigation` | `#875DFF` | Active navigation leading rule |
| `color.accent.strong` | `#9A78FF` | Accent detail on dark surfaces |
| `color.focus` | `#A58AFF` | Keyboard focus ring |
| `color.success` | `#49CC54` | Successful state text/icon |
| `color.danger` | `#FF4B39` | Failed state text/icon |
| `color.warning` | `#F5BD22` | Degraded or delayed state text/icon |

Rules:

- Do not use green or red cell fills for performance values. Color only the value or state glyph.
- Do not use accent borders as decoration. Preserve the accepted v9 accent roles above; do not
  collapse them into one normalized color before an explicit `library_slice_review` decision.
- Neutral surfaces may use a subtle `#171C21` to `#13181D` vertical step inside a panel, but no
  glass, blur, glow, translucent card stack, or high-chroma gradient is allowed.
- Status meaning must remain understandable with color removed.

### Geometry and spacing

| Token | Value | Use |
|---|---:|---|
| `size.control.compact` | `28px` | Buttons, selects, segmented items, icon controls |
| `size.icon.compact` | `14px` | Compact control glyph box |
| `size.header.panel` | `52px` | Panel and detail-dock headers |
| `size.row.compact` | `50px` | Jobs and comparable dense result rows |
| `size.progress.compact` | `32px` | Circular progress indicator |
| `size.status.chrome` | `22px` | Edge-to-edge platform status line |
| `radius.container.row` | `7px` | Clipped row/table containers |
| `radius.control` | `8px` | Compact controls and cards |
| `radius.panel` | `10px` | Primary panels |
| `border.hairline` | `1px` | All standard boundaries |
| `space.panel` | `4px` | Exterior and inter-panel gap |
| `space.control.inline` | `8px` | Adjacent controls in one toolbar |
| `space.icon.label` | `6px` | Icon-to-label gap |
| `space.content` | `12px` | Primary internal content rhythm |
| `axis.first-content` | `29px` | Offset after a `52px` panel header |

Rules:

- Peer panels use the same `4px` exterior and inter-panel spacing.
- The first bordered content block in peer columns begins `29px` after the `52px` header, producing
  a shared top border at `81px` from the panel top.
- Use `12px` as the default space between major blocks inside a detail surface. Use `8px` between
  controls and `6px` only for icon/label or tightly related inline content.
- Do not shrink the `28px` control lattice to solve width pressure. Move secondary controls into
  explicit progressive disclosure.
- First and last rows must be clipped by one rounded container. Avoid a second border or empty band
  at the bottom of a table or card.

### Typography

The accepted specimen uses the system stack `-apple-system`, `BlinkMacSystemFont`, `SF Pro Text`,
`Inter`, `Geist`, `Segoe UI`, sans-serif and was reviewed on macOS. Use `SF Pro Text` for the Figma
specimen when the authenticated environment exposes it. If it is unavailable, stop and present the
exact fallback as an explicit `library_slice_review` decision; do not silently normalize to Inter.
Never stretch, condense, or vertically transform text. Use tabular figures for dynamic values.

| Style | Size / line | Weight | Use |
|---|---:|---:|---|
| `type.panel-title` | `16 / 18px` | `500` | Primary panel title |
| `type.detail-title` | `14 / 18px` | `500` | Detail-dock identity; selected fragment may be `600` |
| `type.control` | `12 / 14px` | `500` | Compact controls |
| `type.body` | `12 / 15px` | `400` | General compact content |
| `type.row-title` | `11.5 / 14px` | `500` | Dense row identity |
| `type.numeric` | `11 / 13px` | `400` or `600` | Metrics and table values |
| `type.micro-label` | `10.5 / 13px` | `500` | Section labels and table headings |
| `type.metadata` | `10 / 13px` | `400` | Row metadata and timestamps |
| `type.status` | `10 / 12px` | `400` | Platform and freshness status |
| `type.count` | `11 / 11px` | `600` | Compact count badge |

Section labels and table headings on the same visual level must use exactly the same family, size,
line height, and weight. Do not substitute uppercase labels, letterspaced captions, or oversized
page titles for hierarchy.

## Control and icon grammar

- Every compact control is `28px` high and shares one y-centre and baseline with its toolbar.
- Icon-only controls are visibly `28 × 28px`; labeled controls use `10px` inline padding.
- All compact icons use a square `0 0 24 24` view box, a `14 × 14px` visible box, `1.5px` stroke,
  round caps, and round joins. Use one outline SVG with `currentColor` for its states; do not create
  separate colored icon assets or scale individual icons to compensate for weak geometry.
- Connected segmented items share one `28px` container, matched outer `8px` radii, and a `1px`
  separator. Items representing equivalent choices have equal width even when labels differ.
- Hover strengthens the neutral boundary. Keyboard focus uses a `2px` `color.focus` outline with a
  `2px` offset. Disabled state lowers emphasis without removing the label or state meaning.
- Each icon-only action has an accessibility-facing name and tooltip. Runtime targets must remain
  at least `24 × 24px`; the standard visible control is already `28 × 28px`.
- Do not use a chevron where no menu or disclosure exists. Do not display text beside an icon-only
  action when the accepted composition specifies an icon-only control.

## Shared surface grammar

- Global chrome uses one `56px` top bar and, when present, one `22px` edge-to-edge status line.
- Primary panels use `color.border.default`, `radius.panel`, and a restrained neutral surface.
- Panel headers are `52px` high. Titles align to one axis; actions align to the trailing edge on the
  same `28px` control lattice.
- Related controls form a clear group. A primary create action may sit beside the panel title while
  refresh, filtering, and overflow remain a trailing action group.
- Dense lists and tables live inside one clipped `radius.container.row` container with `1px` soft
  separators and no duplicated outer boundary.
- Selected rows use `color.surface.selected` plus a `2px` violet leading rule. Do not rely on the
  fill alone.
- Major content should occupy the available workspace height. Do not leave an accidental large
  gap above the platform-status line.

The accepted Backtests desktop specimen uses `396px / 328px / flexible detail` columns. That ratio
is Backtests-specific, not a universal screen template. Other routes must preserve the shared
spacing and alignment grammar while deriving their columns from their own tasks.

## Pilot component anatomy

### Backtests toolbar

- Height is governed by the `28px` control lattice and `8px` control gaps.
- Required manifest concepts are text query, job state, exchange, market type, symbol, launched
  date range, manual refresh, auto-refresh preset, and refresh status.
- Less-frequent filters may live behind one explicit filter control, but all fields remain named in
  the component API and composition manifest.
- Manual refresh is an icon-only action. Auto-refresh is a compact select or split control with the
  current interval visible. Refreshing and degraded states must not shift geometry.
- Primary identity or create action is separated from the trailing utility group when both exist.

### Backtests job row

- The row is `50px` high with `13px` leading and `11px` trailing padding and `6px` internal gap.
- Fast-scan identity is stronger than market/setup metadata. Long identity text truncates; required
  values remain recoverable through the detail context or accessibility name.
- A selected row uses the shared selected surface and `2px` leading accent.
- Progress rings are `32px`; their value uses `11px` regular tabular text.
- `Completed`, `Failed`, `Queued`, `Running`, and degraded freshness are distinct concepts. Job
  failure must never be used to represent stale workstation data.
- State uses text plus an icon/progress shape. Green, red, or amber alone is insufficient.

### Detail-dock header

- Header height is `52px`; use `16px` leading and `12px` trailing padding with `7px` identity gaps.
- Identity, symbol, market type, and period read as one compact line. Secondary date/freshness text
  uses the metadata style.
- Actions align to the trailing edge with `8px` gaps on the common control lattice.
- Close is icon-only with an explicit accessibility-facing name. No uncontracted overflow action
  is added.
- The header may show completed status and degraded freshness, but the body remains outside this
  pilot.

### Degraded freshness notice

- Cached data remains visible. The notice says freshness is degraded and never says that the job
  failed unless the job itself is failed.
- Use a neutral surface, warning text/icon, and concise copy. Avoid a saturated amber fill.
- Manual refresh remains visible when permitted. Retry timing appears only when supplied by the
  response.
- The notice preserves surrounding geometry during refreshing, success, and error transitions.

## State and content-extreme requirements

The library gate must exercise:

- default, hover, keyboard focus, disabled, selected, completed, degraded, and refreshing states
  where the component API supports them;
- shortest accepted labels, long English labels, and a longer Russian localization sample;
- null or unavailable metrics without invented zeroes;
- long strategy and symbol identity without clipping required actions;
- state text without relying on semantic color.

Use these deterministic test-only content extremes; they are geometry fixtures, not accepted
product copy:

- long English strategy: `dema-1h-long-short-with-volatility-confirmation-a1b2c3`;
- long Russian strategy: `Стратегия пересечения DEMA с фильтром относительной силы`;
- long symbol: `1000000PEPEUSDT`;
- English degraded notice: `Data freshness is degraded. Showing the latest cached results.`;
- Russian degraded notice:
  `Актуальность данных снижена. Показаны последние сохранённые результаты.`;
- unavailable metric: `—`, never an invented `0`.

Loading is optional for this pilot because `degraded` is the selected required state. Do not add a
loading variant to the product manifest unless it is implemented and gated as a registered library
variant.

## Automatic rejection gate

Reject a library asset or screen candidate before owner review if it:

- uses a historical or rejected Figma file, raster, component, or geometry as a source;
- copies the accepted HTML into Figma as raw one-off nodes instead of library components;
- creates reusable masters in the product file or detached product instances;
- changes the `28px` control lattice, `14px` icon box, icon stroke, toolbar baselines, or equal
  segmented widths;
- mixes font family, size, line height, or weight among peer micro labels;
- uses oversized titles, large empty padding, floating rounded cards, glass, blur, glow, decorative
  gradients, or pill inflation;
- uses semantic fills for performance tables, color-only status, or violet as general decoration;
- creates inconsistent panel insets, doubled bottom borders, unclipped first/last rows, or a large
  unused gap above the status line;
- invents fields, actions, permissions, route behavior, confirmation flows, or runtime states;
- omits accessibility-facing names, focus treatment, content-extreme cases, token bindings, or
  text-style bindings;
- exceeds the ticket's exact library or product mutation boundary.

## Figma translation rules

1. Foundations become variables and text/paint styles on `01 Foundations` only after the gated
   library specimen is accepted.
2. Reusable icons live on `02 Icons`, components on `03 Components`, and composed reusable patterns
   on `04 Patterns`.
3. Before `library_slice_review`, candidate assets and review specimens remain under
   `80 Audit Sandbox` with exact owned node IDs.
4. Component properties use stable semantic names and enumerate every supported state. Hidden
   appearance differences are not separate undocumented components.
5. Every component internal binds approved color/geometry variables and text styles. Placeholder
   geometry, detached instances, duplicate masters, and unregistered variants fail the gate.
6. Publication and product-file enablement remain manual product-owner actions. Only verified
   published component keys may enter a composition manifest.
7. Product candidates contain library instances only; `raw_node_allowlist` remains empty.

## Proof boundary

This proposed standard defines a reproducible visual grammar and review vocabulary for the current
checkpoint. It is not a cross-screen authority until explicitly accepted. Figma gates can prove
node identity, variables, styles, bindings, component reuse, content coverage, and inspected visual
intent. They do not prove runtime semantics, keyboard behavior, screen-reader output, browser
reflow, permissions, API behavior, production data, release, or deployment.
