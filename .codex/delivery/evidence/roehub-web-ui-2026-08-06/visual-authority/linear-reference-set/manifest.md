---
doc: visual-authority-reference-set
version: "1.0"
status: accepted_owner_input
language: en
evidence_mode: native_image
native_viewport: 3024x1964
---

# Linear Reference Set — Roehub Visual-Language Authority

## Authority

The owner supplied this set as the normative visual-language reference for the
self-hosted Roehub platform. Roehub must capture the operating principles, not
copy Linear's information architecture, product concepts, wording, or screen
composition.

The source PNG files are local owner-supplied review inputs and are
intentionally ignored by the public Roehub repository. This tracked manifest
preserves their exact filenames, hashes, roles, and non-copying boundary. An
executor without the local files must report `external_authority` and request
the source evidence through an authorized private channel; it must not fetch,
recreate, or substitute the screenshots.

Primary source visual:

- `01-project-issues-list.png`
- SHA-256: `525c1925d7d8fdd2899df9978d58850ba28dc8c7ff540dff7634caaa3f4887c3`
- Role: application shell, surface hierarchy, list density, grouping, tabs,
  contextual toolbar, and quiet row grammar.

Supporting state sources:

| File | SHA-256 | Evidence coverage |
|---|---|---|
| `02-cascading-object-menu.png` | `2be9b5dfc45fad6bba62b4b05cc818ffe180ed1cc21ec5030140d2914c2b4b97` | object menu, grouping, cascade, shortcuts, destructive-action placement |
| `03-tooltip-and-toast.png` | `a0ee306e1f9a742f357c21052600437c0c685cdd6d14473de1696fa13923e04a` | tooltip, pressed control, toast, transient-state consistency |
| `04-display-properties-popover.png` | `9d97a19f4c80c166d7a3975f9e301c634b50780822a71096aac4a3a9e8ce09d9` | display configuration, segmented mode, switches, property chips |
| `05-select-menu-inside-popover.png` | `787a055efa6f3a35350c6ee07aff56b4651d7af2273f8ee2050d03cf75364386` | nested select, selected item, parent/child layering |
| `06-cascading-filter-menu.png` | `5bf29ef17c3aa980e2d8d8b6c026c053a1ce7c64509ae0630f42d68e5c9152eb` | search-first filtering, domain categories, secondary cascade |
| `07-project-overview-main-column.png` | `b853def1bbf75b596de129d6389279d71494430b9caf86d077fc954b62a5e6cd` | content-first overview, inline properties, restrained empty action |
| `08-project-overview-inspector.png` | `236b410edb060320791f51c7da4edf701c4597c03f185b8c3b24e209d7df27f8` | optional inspector, collapsible sections, activity hierarchy |
| `09-notification-popover-and-toast.png` | `8173d4f54f4d497391faccd9170900a956e96d68ff8e2dcd49ede98bf48570a4` | notification preferences, checkbox grammar, integration action, toast |

## Normative visual-language principles

1. One near-black canvas carries the application; the workspace is a distinct
   raised plane with a rounded top-start corner and one perimeter boundary.
2. Use four or five close dark tones to express hierarchy. Borders separate
   structural levels, not every datum, card, chart, row, or control.
3. High density comes from alignment, typographic rhythm, grouping, and
   progressive disclosure—not from compressed boxes with permanent outlines.
4. Default controls are quiet. Hover, focus, selected, pressed, and open states
   reveal affordance through controlled fill and contrast.
5. A page normally has one breadcrumb/header row, one local subnavigation row,
   and at most one task-specific toolbar row before primary content.
6. Object actions remain close to the object through menus, popovers, tooltips,
   and cascades. A persistent inspector is reserved for multi-property analysis
   or editing that benefits from staying visible.
7. Rows are the primary dense-data primitive. Section headers use a subtle
   surface fill; ordinary rows depend on alignment and hover rather than boxes.
8. Text emphasis is scarce: calm neutral labels, regular body weight, bright
   primary content, and color reserved for status, risk, selection, or action.
9. Tooltip, menu, select, popover, toast, dialog, and inspector share one
   geometry, elevation, spacing, focus, and motion grammar.
10. Empty space is functional. It establishes hierarchy and resting areas
    without lowering data density where analysis is required.

## Explicit non-copying boundary

- Do not reproduce Linear's issue/project information architecture.
- Do not copy its branding, icons, labels, exact colors, or proprietary assets.
- Do not map Roehub entities to issue-tracker metaphors when trading/research
  language is clearer.
- Do not use screenshot fidelity to justify unsupported Roehub behavior.

## Relationship to pilot v23

Pilot v23 remains supporting Roehub-specific composition evidence for a dense
Backtests Workbench. Its permanent borders, fixed three-panel grid, fixed
`1672×941` canvas, micro-typography, always-visible controls, and competing
horizontal bands are not inherited. G3 must reinterpret its useful jobs →
variants → result hierarchy through this reference set's quieter grammar.

## Supporting analytical translation

The hash-pinned Custometry pilot candidate v2 is recorded separately under
`../supporting-custometry-pilot-v2/manifest.md`. It demonstrates one successful
translation of these principles into a dense analytical application. It is
supporting evidence only and does not narrow, replace, or reinterpret this
reference set's normative visual-language role.
