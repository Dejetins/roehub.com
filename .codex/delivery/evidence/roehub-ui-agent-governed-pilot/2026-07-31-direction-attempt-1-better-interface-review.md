# Direction attempt 1 — better-interface review

## Scope and Coverage

- Mode: `full`.
- Scope: the three first-pass Backtests raster specimens generated after pilot-brief approval,
  reviewed together with `roehub-backtests-process-pilot-brief-v1.md`.
- Boundary: static raster and design-contract evidence only. No Figma node was created, no runtime
  UI was inspected, and keyboard, screen-reader, reflow, motion, and exact rendered contrast remain
  unverified.
- Reference viewport requested by the brief: `1440 x 900`.
- Actual raster size: `1586 x 992` for all three specimens.

| Domain | Evidence inspected | Result |
|---|---|---|
| Accessibility | Visible labels, status redundancy, control silhouettes, and static selected/degraded states | 1 finding; runtime semantics not verified |
| Layout | Three rasters, pilot crop, toolbar, row, degraded message, and right pane | 3 findings |
| Writing | Toolbar labels, status labels, degraded copy, and detail header copy | 1 finding |
| Typography | Visible hierarchy, density, labels, IDs, and metric alignment | 1 finding |
| Colors | Neutral surfaces, orange usage, completion/degraded cues; exact contrast not measured | 1 finding |
| UI | Surfaces, borders, selection, icon consistency, elevation, and placeholder treatment | 3 findings |

## Findings

| # | Severity | Domain | Location | Before | After | Why |
|---|---|---|---|---|---|---|
| 1 | HIGH | Layout | Pilot brief `Pilot composition`; all three rasters | Direction is judged from a toolbar/row/dock crop with no workspace chrome or page/view identity | Direction review gets explicit shell, context-bar, list, and right-pane context; the reusable library pilot remains bounded | The crop removes the relationships that make a Linear-style workspace recognizable, so a generic dashboard can pass the written brief |
| 2 | HIGH | UI | All three rasters | Generic crypto-terminal visual language: pure dark canvas, strong orange outlines, warning strip, and terminal-like field chrome | Neutral graphite workspace, continuous panels, quiet separators, scarce Roehub accent, and inline degraded state | The artifact contradicts the requested premium Linear workspace and reads as a cheaper version of the rejected historical direction |
| 3 | HIGH | Layout | Toolbar in all three rasters | Every filter is a separate equal-weight labelled box | Search plus one filter entry point, visible active filter chips, and quiet refresh controls | Equal weight destroys hierarchy and creates a dense form rather than contextual workspace controls |
| 4 | MEDIUM | Colors | Selected row, refresh controls, toggle, warning, and pane status | Orange carries selection, action, toggle, border, and warning meaning simultaneously | Reserve Roehub copper/orange for one brand/action role; keep warning and completion semantic with text/icon redundancy | One colour has too many meanings and becomes visual noise instead of emphasis |
| 5 | MEDIUM | Typography | Toolbar labels and job row | Many tiny labels and values share nearly the same weight; numeric comparison lacks a declared tabular treatment | Inter Variable role scale with `13px` primary UI, `12px` metadata, clear weight hierarchy, and tabular metrics | Weak hierarchy and shifting/proportional values make the dense row harder to scan |
| 6 | MEDIUM | Layout | Job row in attempts 1 and 3 | Required content is flattened into a long spreadsheet row of unrelated cells | Use a two-level row: primary identity/state/key metrics, secondary setup/remaining metrics with shared baselines | The current row is dense without being legible and competes with the detail pane |
| 7 | MEDIUM | UI | Detail pane and lower canvas across attempts | Large blank pane/list regions and repeated horizontal bands appear as unfinished placeholders | End the pilot pane after its required header and use uninterrupted neutral review background | Placeholder geometry makes an otherwise small slice look incomplete and cheap |
| 8 | MEDIUM | Writing | Degraded state in page and pane | The same long degraded sentence is repeated as a dominant banner and pane line | Use one compact page status such as `Live data delayed · Cached result from 10:42 UTC` and one short property value in the pane | Repetition adds alarm without improving recovery or truthfulness |
| 9 | MEDIUM | Accessibility | Compact icon actions and selected/degraded states in static rasters | Icon-only refresh/close controls and focus treatment cannot be verified; some states rely heavily on colour | Direction contract requires named icon controls, visible focus specification, and text/icon status redundancy before component acceptance | Static appearance alone cannot establish operability or an accessible name; the future library must make these requirements explicit |

## Considered but Rejected

| Location | Candidate | Rejected because |
|---|---|---|
| Overall direction | Add brighter gradients and stronger shadows to make it feel premium | Premium is precision and restraint here; decoration would move farther from the accepted workspace grammar |
| Pilot composition | Expand into a full Backtests screen and full design system immediately | It would invalidate the small governance pilot and create more unverified component scope |
| Linear reference | Copy a Linear issue-list screen pixel for pixel and replace labels | Literal replication is prohibited and would import foreign product structure instead of translating the interaction grammar |
| Toolbar | Keep every filter always visible for maximum discoverability | Progressive disclosure and visible active chips preserve discoverability with much less noise |

## Verification

Passed:

- inspected all three source rasters at original size with the same synthetic fixture;
- confirmed all three are `1586 x 992` and therefore directly comparable;
- reviewed the accepted Linear-workspace transition specification, shared workspace standard,
  sanitized geometry evidence, and current pilot brief;
- inspected four current official Linear UI reference images for hierarchy, density, chrome, lists,
  filters, and split-view relationships;
- loaded and applied `better-accessibility`, `better-layout`, `better-writing`,
  `better-typography`, `better-colors`, and `better-ui` through `better-interface`.

Not verified:

- exact contrast ratios and gamut because the rasters are not accepted token output;
- keyboard path, accessible names, screen-reader output, focus return, reflow, localization,
  animation, or browser performance;
- any Figma structure, component, variable, style, binding, or publication state.

## Verdict

Block
