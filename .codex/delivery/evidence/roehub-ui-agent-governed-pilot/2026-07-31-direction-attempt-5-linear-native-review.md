# Direction attempt 5 — Linear-native review

## Scope

- Artifact: `DIR-003 / Roehub Linear-native Backtests / v1 / CANDIDATE`.
- Local raster:
  `.codex/delivery/evidence/roehub-ui-agent-governed-pilot/artifacts/dir-003-roehub-linear-native-backtests.png`.
- Figma read-back:
  `.codex/delivery/evidence/roehub-ui-agent-governed-pilot/artifacts/dir-003-figma-readback.png`.
- Review boundary: static visual direction, content placement, and exact Figma raster placement.

## Visual and interaction-direction audit

| Lens | Observed evidence | Result |
|---|---|---|
| Hierarchy | Stable global chrome, narrow worklist, selected-job content surface, and attached properties/freshness rail | Pass |
| Density | Compact toolbar, one `48-52px`-target row, aligned property rows, no metric tiles or expanded report row | Pass |
| Progressive disclosure | Worklist carries comparison data; full metrics/setup/freshness appear only for the selected job | Pass |
| Keyboard-first direction | Global command trigger has visible `⌘K`; selected row has one explicit open-details affordance | Pass within static boundary |
| Visual craft | Neutral graphite steps, readable muted text, hairline dividers, compact Inter-like type, monochrome icons, scarce semantic colour | Pass |
| Content contract | Required query/filters/refresh state, identity, market, setup, metrics, lifecycle, created time, degradation, and permitted actions are recoverable | Pass |

## Hard-gate verification

Observed none of the v2 rejection conditions:

- no recoloured dashboard or full-width one-row report canvas;
- no more than two persistent chrome layers before the worklist;
- no chips strip, warning banner, empty inspector, metric-card grid, spreadsheet cells, chart, glow,
  gradient, glass, neon, blue wash, or crypto-terminal treatment;
- no strong orange navigation or border treatment;
- no omitted Backtests field and no unsupported row or shell action;
- no Linear branding, entities, text, or literal screen geometry.

## Figma audit

- Product file: `nzKVsXuCmoTbHJGckHfK3T`.
- Active page: `01 Direction Review` (`3:2`) with exactly one child, frame `12:2`.
- Candidate frame: `1504 × 1064`, locked, status
  `awaiting_product_owner_direction_selection`, `reusableSource=false`.
- Raster node: `12:5`, `1440 × 900`, one image fill, `scaleMode=FIT`, image hash
  `c1c7bf0f21d5824208c55f5682a2a292e0174210`.
- Archive page: `90 Archive` (`3:6`) contains rejected `4:2` and gate-failed `10:2`.
- Independent Figma screenshot inspection found no clipping, overlap, unintended scaling, missing
  label, or active-page contamination.

## Static-evidence limits

The raster can establish direction, visible hierarchy, density, and affordances. It cannot prove
actual command-palette behavior, keyboard traversal, focus return, hover parity, motion, reduced
motion, accessible names, screen-reader output, browser performance, API behavior, or runtime
authorization. Those remain later executable gates.

## Verdict

`candidate_ready_for_direction_selection`. This verdict does not accept the direction; only the
product owner can do that for exact frame `12:2`.
