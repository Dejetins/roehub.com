# Linear workspace sanitized reference measurements v1

## Boundary

This document records sanitized derived measurements and explicit waivers for
`LINEAR-WORKSPACE-REFERENCE-2026-07-20`. It is reference evidence only. It does
not prove Roehub runtime behavior, product readiness, Penpot readiness, or
implementation acceptance.

No screenshot, recording, trace, cookie, token, browser profile, storage state,
account export, or raw accessibility snapshot is committed. Workspace content,
member data, and third-party assets are excluded. The live checks used a
separate agent-created tab in an existing authenticated Chrome session and left
the user's original Linear tab open.

## Source identity

| Item | Result |
|---|---|
| Archive | `/Users/daniildegtyarev/Downloads/reference.zip` |
| Archive SHA-256 | `eb7b0ab070f64d553baafacefa90fdb2e87e51bc174c63db9af73bc77f8e41c2` |
| Archive identity | exact match |
| Archive reverified | `2026-07-22T11:45:56Z` (archive plus all listed PNG hashes) |
| PNG count | `16` |
| Individual PNG hashes | all exact matches to `linear-workspace-reference-manifest-v1.json` |
| PNG dimensions | all `2560 x 1440` pixels |
| PNG density metadata | absent; no PNG `pHYs` or EXIF X/Y density tags |
| Browser chrome | included |
| Archive CSS viewport / DPR | unknown; neither value is recoverable from the PNG bytes |

No `72 dpi` field is present in the PNGs. A renderer may display a default
density, but that value is not source metadata and is not treated as browser
zoom, operating-system scale, CSS viewport, or device pixel ratio.

## Historical live observation environment

Observed at `2026-07-20T23:10:40Z` against the authenticated Linear Roehub
workspace in `Google Chrome 150.0.7871.129`. This historical observation was not replayed or refreshed on 2026-07-22.

| Property | Value |
|---|---:|
| CSS viewport | `1512 x 790` |
| Device pixel ratio | `2` |
| `prefers-reduced-motion` | `false` |
| Live screenshots / recordings / traces retained | `0` |
| Cookies / tokens / storage state inspected or retained | `0` |

The live viewport is a separate observation and must not be used to infer the
scale of the archive captures.

## Shell and component geometry

Coordinates and dimensions below are CSS pixels in the live viewport.

| Surface | x | y | width | height | Additional observation |
|---|---:|---:|---:|---:|---|
| Primary navigation | `0` | `0` | `305` | `790` | expanded shell |
| Main surface | `305` | `8` | `1199` | `746` | stable issue detail |
| Sidebar resize target | `303` | `14` | `7` | `736` | computed cursor `col-resize` |
| Issue-options trigger | centered at `854.40` | centered at `30.25` | `28` | `28` | opens the issue popover |
| Issue-options popover | `840` | `48` | `210` | `521` | `12px` radius; `14` materialized options |
| Command surface | `396` | `103` | `720` | `450` | `12px` radius; clipped container |
| Command input | `403` | `144` | `708` | `40` | `role=combobox`, `aria-expanded=true` |

The archive includes static expanded and collapsed sidebar states. Exact CSS
widths cannot be recovered from those full-screen PNGs, so the archive is not
used to freeze sidebar tokens. The live `305px` width and `7px` drag target are
observations of the current reference, not Roehub implementation requirements.

## Keyboard, overlays, and focus

- `Cmd+K` opened the command surface from the issue detail route.
- Focus moved to the active `Command menu` combobox.
- Filling the non-sensitive query `ROE-8` preserved the expanded combobox and
  exposed an accessibility status of `117` items; `18` options were materialized
  in the virtualized list at the observation point.
- No command was executed. The visible command set included actions that could
  mutate the issue, so execution was intentionally waived.
- `Escape` closed the command surface and returned focus to the document body,
  matching the keyboard-open origin.
- Opening `Issue options` moved focus to its filter input.
- `Escape` closed that dialog and returned focus to the `Issue options` button.

This proves the selected open/search/close and popover focus-return paths only.
It does not prove the complete Linear keyboard map or the future Roehub map.

## Sanitized accessibility structure

Only derived role counts are retained; raw snapshots are not committed.

| Role | Issue detail | Command open | Issue options open |
|---|---:|---:|---:|
| `navigation` | `1` | `1` | `1` |
| `main` | `1` | `1` | `1` |
| `link` | `44` | `44` | `44` |
| `button` | `46` | `47` | `46` |
| `textbox` | `2` | `2` | `2` |
| `heading` | `6` | `6` | `6` |
| `switch` | `1` | `1` | `1` |
| `status` | `2` | `3` | `3` |
| `region` | `1` | `1` | `1` |
| `combobox` | `0` | `1` | `0` |
| `dialog` | `0` | `0` | `1` |
| `searchbox` | `0` | `0` | `1` |
| `listbox` | `0` | `1` | `1` |
| `option` | `0` | `18` | `14` |

Counts reflect the materialized accessibility tree at the observation point;
virtualized or off-screen items may not be materialized.

## Route and state observations

- Direct navigation from the issue detail to `/roehub/projects/all` produced a
  truthful `Loading…` state before stable content.
- Browser Back restored the exact ROE-8 issue URL and the issue title in the
  accessibility tree.
- The archive contains a selected Inbox row with an empty detail surface.
- The archive also contains static modal, popover, nested overlay, settings,
  dense list/table, collapsed-sidebar, expanded-sidebar, and right-pane states.

Observed automation round trips were `985ms` for the direct navigation,
`338ms` for Back URL restoration, and `168ms` for the `Cmd+K` action. These
figures include browser-control transport and are diagnostic observations only;
they are not Linear motion timings, client-performance results, or Roehub
budgets.

## Motion observation

The command surface and issue-options popover exposed neither active Web
Animations nor non-zero computed CSS transition/animation durations at the
measurement point. Static captures cannot establish animation duration. No
recording or trace was retained because it could contain account data and raw
browser state.

Roehub implementation must therefore use the accepted timing ranges in
`linear-workspace-ui-transition-standard-v1.md` as provisional targets and
collect its own real-browser motion evidence. This reference does not freeze a
Linear-derived duration token.

## Gap dispositions and waivers

| Required item | Disposition | Impact |
|---|---|---|
| Four-theme representative matrix | Waived | The supplied reference is dark-only. Roehub's `abyss`, `graphite`, `frost`, and `paper` semantics remain project-owned and require later Roehub browser proof. |
| Command palette interaction | Observed with execution waiver | Open, non-sensitive search, and Escape were observed. Executing an action was avoided because the menu contains state-changing commands. |
| Keyboard focus sequence | Partially observed and explicitly waived | Command focus and popover focus return are evidenced. A complete app-wide traversal remains a Roehub implementation acceptance item. |
| Sidebar resize recording and geometry | Geometry observed; recording and mutation waived | Live geometry and `col-resize` are recorded. Dragging could persist a shared presentation preference and interfere with parallel work. |
| Route/drawer/modal/popover recordings | Route and popover observed; recordings waived | Static modal/popover/right-pane states exist, but duration and continuity require Roehub runtime recordings. |
| Loading/empty/error/stale/forbidden/session-expired states | Loading and empty observed; remainder waived | Error/stale sources were unavailable; forbidden/session-expired reproduction would require unsafe authorization or session manipulation. Roehub must prove these with disposable local fixtures. |
| Sanitized accessibility snapshots | Closed | Derived role counts and focus behavior are committed; raw snapshots are excluded. |
| Motion and component geometry | Geometry closed; motion timing waived | Geometry is measured. Motion timing remains governed by the shared standard until Roehub runtime evidence exists. |

## Roehub interpretation

Reuse only density, hierarchy, geometry relationships, focus behavior, and
state-continuity intent. Do not copy Linear branding, text, product entities,
assets, source code, authorization behavior, or account content. All product
semantics, routes, permissions, themes, copy, and runtime acceptance remain
Roehub-owned.
