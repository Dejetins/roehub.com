# DIR-001 product-owner rejection audit

## Decision

- Artifact: `DIR-001 / Roehub Graphite Workspace / v1 / CANDIDATE`.
- Figma identity: file `nzKVsXuCmoTbHJGckHfK3T`, page `3:2`, frame `4:2`.
- Decision: `rejected_by_product_owner` on `2026-07-31`.
- Explicit feedback: `DIR-001 отклонен полностью. ты не уловил суть дизайна.`
- Reuse: prohibited for visual language, layout, component anatomy, interaction model, and later
  executor context.

## Observed failure

DIR-001 reproduced a dark palette, thin separators, a sidebar, and a right pane, but retained the
information architecture of a conventional dashboard. It therefore resembled Linear superficially
while contradicting the requested Linear-native behavior.

| Evidence in DIR-001 | Why it fails the requested concept | Required correction |
|---|---|---|
| Four persistent horizontal layers before the primary result | Chrome dominates the task and lowers useful density | One quiet context/header layer and one compact list header |
| Every active filter rendered as a permanent chip | State is visible, but progressive disclosure and speed are absent | Compact filter summary with an explicit popover/command path |
| One result occupies three full-width data bands | The list behaves as an expanded report rather than a scan-and-select surface | Compact comparison row; selected details move into the inspector |
| Full-height inspector contains only a header | The pane creates a large empty placeholder and does not relieve the row | Populate it with the same required result/setup/freshness properties |
| Degraded freshness consumes a dedicated full-width row | A secondary projection state receives primary structural weight | Attach freshness to refresh/status context and selected-job properties |
| Global navigation and brand mark have strong standalone weight | The shell competes with the work surface | Quieter compact navigation, monochrome system icons, scarce accent |
| No visible command or keyboard model | `keyboard-first` and `command palette` exist only in prose | Show a command trigger, shortcut affordances, and contextual row actions |

## Root cause

The rejected v1 doctrine treated Linear mainly as a visual grammar. It did not impose a hard
information-placement contract, so the generator could satisfy content completeness by exposing
everything at once. The automatic interface review then checked the raster against that flawed
doctrine and returned a false-positive `Approve` verdict.

## Process correction

The replacement doctrine makes information placement and progressive disclosure hard gates.
Required content remains complete across the selected list row and inspector, but the list is no
longer required or allowed to carry the entire record. The next owner checkpoint remains
`direction_selection`; no library asset or downstream composition may be created before the owner
selects the replacement direction.

## Static-evidence boundary

This audit is based on the supplied screenshot and exact rejected Figma artifact identity. It can
assess visible hierarchy, density, grouping, and affordances. It cannot prove runtime keyboard
navigation, focus return, motion quality, accessibility semantics, browser behavior, or performance.
