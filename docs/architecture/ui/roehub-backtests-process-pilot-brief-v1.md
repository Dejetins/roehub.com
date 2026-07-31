# Roehub Backtests agent-governed process pilot brief v1

This brief fixes the product content and review boundary for the first greenfield Figma iteration.
The product owner rejected the early generic dashboard attempts and later selected the product
content model of `DIR-003` as the closest starting point while rejecting its visual design. The
current direction exploration expands the non-reusable context to the real jobs, top-variants,
charts, stats, and lazy-trades model already present in Roehub contracts.

## Status

- Status: `direction_v9_accepted`.
- Ticket: `ROEHUB-UI-AGENT-GOVERNED-PILOT-2026-07-31`.
- Checkpoint: `library_slice_review`.
- This brief does not authorize a complete screen, complete library, or runtime implementation.
- Product-owner brief approval: `2026-07-31`, explicit message `Бриф пилота принимаю.`
- First visual attempt: `rejected_by_product_owner`; it is not a visual, component, or screen
  source.
- Second visual attempt (`DIR-001`): `rejected_by_product_owner`; it is not a visual, component,
  screen, layout, or interaction source.
- Third visual attempt (`DIR-003`): `rejected_by_product_owner_for_visual_design`; its job/detail
  content relationship may inform requirements, but its styling and geometry are prohibited.
- Fourth visual attempt (`tahoe_backtests_workspace` v3): `rejected_by_product_owner`; its rasters
  are prohibited as visual, layout, component, or geometry inputs.
- Fifth visual attempt (`tahoe_backtests_workstation_v4`):
  `rejected_for_visual_design_structure_preferred`; its Workbench hierarchy may inform v5, while
  every v4 visual treatment remains prohibited.
- Fifth visual feedback: `TAHOE-1 Workbench` is the preferred information architecture, while all
  v4 visual treatments remain rejected.
- Linear-black v5: `preferred_direction_refinement_required`; it remains historical evidence for
  the bounded v6 correction cycle.
- Linear-black v6: `preferred_direction_refinement_required`; its topology and visual language are
  the accepted edit target, but its control geometry, refresh interaction, chart-type control, and
  bottom status require correction.
- Seventh owner feedback cycle: preserve the v6 concept and apply only the bounded corrections in
  `roehub-linear-black-backtests-workbench-doctrine-v7.md`.
- Linear-black v7 Repair 2: `preferred_base_refinement_required`; its deterministic geometry,
  typography, content, and visual language are the selected edit base, but the product owner
  requested the explicit layout, action, row-density, Failed-state, percentage, and gray-surface
  changes in the v8 feedback cycle.
- Eighth owner feedback cycle: apply only the refinements in
  `.codex/delivery/evidence/roehub-ui-agent-governed-pilot/2026-07-31-direction-linear-black-v8-owner-refinement.md`.
- Linear-black v8: `preferred_base_refinement_required`; its content, neutral-gray visual
  language, compact control lattice, and deterministic geometry are the selected v9 edit base.
- Ninth owner feedback cycle: apply only the refinements in
  `.codex/delivery/evidence/roehub-ui-agent-governed-pilot/2026-08-01-direction-linear-black-v9-owner-refinement.md`.
- Linear-black v9: `accepted_by_product_owner` at `direction_selection` on `2026-08-01`; exact
  acceptance evidence is
  `.codex/delivery/evidence/roehub-ui-agent-governed-pilot/2026-08-01-direction-linear-black-v9-owner-acceptance.md`.
- Current accepted direction: `linear_black_backtests_workbench_v9`, governed by that acceptance,
  the named owner-refinement evidence, and unchanged v8 invariants not explicitly overridden.
- Proposed reusable visual translation for `library_slice_review`:
  `docs/architecture/ui/roehub-linear-black-authenticated-workspace-visual-standard-v1.md`. It
  translates the accepted visual grammar without copying Backtests-specific content. It becomes a
  future ticket-selectable source only after an explicit product-owner decision naming its exact
  reviewed revision; the current v9 decision does not accept it by analogy.

## User outcome

An authenticated user can scan the current Backtest jobs, locate one completed job, understand
its identity and result quality, notice that the workstation projection is degraded, manually
refresh it, and open the selected job's detail dock without losing list context.

## Authoritative functional sources

- `docs/architecture/apps/web/roehub-local-platform-screen-registry-v1.json`:
  `screen.backtests.library` and `screen.backtests.detail`;
- `docs/architecture/apps/web/roehub-local-platform-access-and-route-contract-v1.json`:
  `backtests.read` and server-filtered visibility;
- `apps/api/dto/ui_backtests.py`: workstation, filters, row, refresh, source, and footer response
  fields;
- `apps/api/routes/ui_backtests.py`: `/ui/backtests/workstation` read contract;
- `apps/api/wiring/modules/ui_backtests.py`: availability, degradation, action booleans, and
  nullable metric semantics.

These sources define function and truthful data semantics only. Current templates, CSS, JavaScript,
the rejected React spike, and all former design files are not visual sources.

## Pilot composition

The candidate is one `1440 x 900` crop containing only:

1. a compact Backtests toolbar;
2. one completed job row;
3. one selected-job detail-dock header;
4. one degraded-refresh message.

The crop may show a small amount of neutral background required to judge spacing and hierarchy. It
does not include global navigation, a full table, charts, trades, configuration forms, or complete
screen chrome.

## Required content

### Toolbar

The composition must represent these filters or controls without inventing new product concepts:

- text query;
- job state;
- exchange;
- market type;
- symbol;
- launched date range;
- manual refresh;
- auto-refresh preset and refresh status.

Progressive disclosure is allowed, but no field may disappear from the manifest. A compact filter
trigger may own less-frequent fields if the review specimen makes that grouping explicit.

### Representative job row

The selected-row state must make these groups recoverable across the compact list row and its
contextual inspector. The list row owns fast comparison; the inspector owns progressively disclosed
properties. Required values may not be silently omitted, but they must not all be forced into the
list row:

- identity: `job_id`, `strategy`;
- market: `exchange`, `market_type`, `symbol`;
- setup: `indicator_summary`, `period`, `direction`, `combinations`;
- result: `best_return_pct`, `best_sharpe`, `avg_drawdown_pct`, `profit_factor`,
  `win_rate_pct`, `trades_count`;
- lifecycle: `state`, `progress_percent`, `created_at`, `refresh_status`;
- actions: open details; other actions remain absent unless a later contract explicitly includes
  their permission and confirmation states.

### Detail-dock header

The header contains:

- selected job identity and strategy;
- `symbol`, `market_type`, and period context;
- completed status;
- last projection time and degraded freshness indication;
- close action with an accessibility-facing name.

The body of the detail dock is outside this pilot.

### Degraded state

The candidate uses a truthful degraded workstation projection:

- cached job data remains visible;
- the message says freshness is degraded rather than claiming the job failed;
- manual refresh remains available when permitted;
- retry timing is shown only when the response supplies it;
- status is not encoded by color alone.

## Synthetic review fixture

The following fixture is representative design data, not runtime or production evidence:

| Field | Value |
|---|---|
| `job_id` | `7f3a…9c1d` |
| `strategy` | `dema-1h-long-short-a1b2c3` |
| `exchange` / `market_type` / `symbol` | `Binance` / `Spot` / `BTCUSDT` |
| `indicator_summary` | `DEMA crossover · RSI filter` |
| `period` | `2024-01-01 → 2025-12-31` |
| `direction` | `Long + short reversal` |
| `combinations` | `512` |
| `best_return_pct` | `18.42%` |
| `best_sharpe` | `1.37` |
| `avg_drawdown_pct` | `−6.80%` |
| `profit_factor` | `1.61` |
| `win_rate_pct` | `54.8%` |
| `trades_count` | `186` |
| `state` / `progress_percent` | `Completed` / `100%` |
| `created_at` | `2026-07-31 10:42 UTC` |
| `refresh_status` | `Degraded` |

The library specimen must also test a longer Russian localization sample in the audit sandbox so
the selected direction is not accepted only for short English labels. The main direction review
uses English product labels; this is a pilot convention, not a final localization decision.

## Visual-direction exploration

The generic `Precision Terminal`, `Analytical Neutral`, and `Layered Operations` directions are
retired after the product-owner rejection. Their rasters are local failed-attempt evidence only and
must not be placed in Figma or used as downstream sources.

The current replacement iteration creates one non-canonical raster under the owner-specified
`linear_black_backtests_workbench_v9` refinement. It preserves the preferred v8 Workbench base
while applying only the named typography, action alignment, panel inset, Monthly units, detail
spacing, table-bottom, rounded-row-container, and notification-position changes. The previously
accepted functional inventory, unchanged v8 invariants, and the v9 refinement list are automatic
pre-review rejection gates.

Unlike the reusable pilot composition, the direction specimen includes enough non-reusable shell
context to judge the Linear-workspace grammar:

- restrained authenticated navigation context;
- top context bar and Backtests page/view identity;
- quiet command/search/filter/refresh toolbar with visible keyboard affordance;
- one completed Backtest row;
- one compact degraded-freshness indication integrated with the relevant context;
- one populated contextual inspector containing only brief-required selected-job data.

The shell context does not automatically expand the first library slice or future composition
manifest. The specimens may show already-contracted jobs, top variants, series, stats, chart
overlay, and paginated trades as non-reusable direction context. They remain raster review material
and are never used as downstream component sources.

Before the specimen reaches the product owner it must:

- pass the hard anti-pattern gate in the current interaction doctrine;
- pass a full six-domain interface-craft review;
- pass the brief content inventory;
- be visually inspected independently from image-generation success;
- record the exact reference and generated-image identities.

## Machine-checkable acceptance for the brief

- every required toolbar field exists in the future manifest;
- every required row field exists in `required_content.fields`;
- reusable first-slice actions remain exactly `open_details`, `manual_refresh`, `set_autorefresh`,
  and `close_detail`; non-reusable direction context may additionally show the already-contracted
  result actions `export_csv`, `create_strategy`, and `launch_strategy` without expanding that
  first-slice manifest;
- required states include `completed` and `degraded`;
- the target is `02 Candidate` in file `nzKVsXuCmoTbHJGckHfK3T`;
- reusable assets come only from `rgbNUPCuV7q2pARG4Cml8V` after publication;
- `raw_node_allowlist` is empty for the composed candidate;
- the historical Figma file is absent from all task inputs;
- no visual direction can add, remove, or reinterpret required content;
- the visual direction passes `roehub-tahoe-backtests-workstation-doctrine-v4.md` and contains none
  of its automatic rejection conditions;
- non-reusable shell context is present for direction judgment but absent from the first reusable
  library-slice scope.

## Product-owner direction decision

The product owner accepted the exact v9 HTML at `direction_selection`. Acceptance applies to its
topology and visual language only; it does not accept a Figma component, screen composition, or
runtime implementation. The next owner checkpoint is `library_slice_review`. All rejected v4
specimens remain archived and absent from later executor context.
