# Roehub Figma design-delivery standard v1

## Status and decision

- Status: `accepted architecture`.
- Decision date: `2026-07-22`.
- Scope: all future Roehub design prototyping, design-system work, visual
  review, and design-to-code handoff.
- Figma is the active Roehub design surface. Penpot artifacts and statements
  remain historical evidence only and must not receive future Roehub writes.
- This Roehub-specific standard supersedes only the Penpot-specific clauses of
  `linear-workspace-ui-transition-standard-v1.md`. It does not change that
  standard for Custometry or any other project.

## Canonical Figma workspace

| Boundary | Canonical identity |
|---|---|
| Account | `dejetins@gmail.com` |
| Plan | `team::831604964356268687` (`pro`, `Full`) |
| Project | `roehub.com` |
| Project ID | `629113387` |
| Authenticated-platform file | `Roehub Authenticated Platform UI` |
| File key | `GBzmB9evtzqnAYNjp9W1sr` |
| File URL | `https://www.figma.com/design/GBzmB9evtzqnAYNjp9W1sr` |

The project is the folder for Roehub design work. The registered file is the
only canonical authenticated-platform design file. Do not create a replacement,
copy, or second canonical file unless a later ticket explicitly changes this
architecture. A future public-site file must live in the same project and have
its exact name and file key registered before the first write; this standard
does not create or name that file.

The external IDs above are routing identifiers, not secrets and not ticket
statuses. The repository remains the authority for task status, dependencies,
accepted evidence, and implementation state.

## Tool and pre-write contract

Every Figma task must:

1. load and follow `figma-use` before every `use_figma` call;
2. add `figma-generate-library` when creating or changing variables, tokens,
   styles, components, or variants;
3. add `figma-generate-design` when creating or changing an application page,
   view, or multi-section layout;
4. use `figma-create-new-file` only when a ticket explicitly authorizes a new
   file; it is not applicable to the registered authenticated-platform file;
5. verify the authenticated account and plan, `projectId`, `fileKey`, editor
   type, exact target pages, and owned nodes immediately before mutation;
6. inspect the current target state and stop on access, identity, source,
   ownership, or concurrent-edit drift that cannot be separated safely.

Design tasks own declared pages and nodes, not the entire project. They must not
write to Custometry, historical Roehub Penpot files, unrelated Figma files, or
unowned regions of the canonical file.

## File structure and design-system ownership

The authenticated-platform file uses this target page order unless the design
ticket records a justified amendment:

```text
00 Cover & decisions
01 Foundations
02 Components
03 Patterns
04 Authenticated shell
05 Backtests golden slice
90 Archive
```

- `01 Foundations` owns color, typography, spacing, sizing, radius, elevation,
  motion, and chart-semantic variables.
- Theme switching uses one semantic variable collection with exactly four
  modes: `abyss`, `graphite`, `frost`, and `paper`; `graphite` is the default.
- `02 Components` owns reusable components and explicit variants. Names follow
  `Roehub/<Family>/<Component>` and properties use stable lower-camel-case
  values such as `state`, `size`, `density`, `appearance`, and `intent`.
- `03 Patterns` owns composed navigation, tables, charts with tabular
  alternatives, panels, overlays, feedback, progress, and system-state patterns.
- Product pages use library instances and semantic variables rather than
  detached copies or raw styling values.
- Figma names are design identities; repository packages and runtime contracts
  remain authoritative only after an accepted implementation handoff.

## Functional translation of references

Linear references are a functional topology, not a visual template. Every
selected Figma surface maps each applicable reference block to a Roehub function
or records a justified omission. The mapping must include:

| Field | Required meaning |
|---|---|
| `reference_block` | The navigation, context, work, property, resource, side-panel, progress, activity, or system-state role being interpreted. |
| `roehub_function` | The Roehub task and domain meaning served by the block. |
| `authoritative_source` | Route, capability, projection, product contract, or accepted architecture that owns the data or action. |
| `required_states` | Loading, empty, stale, degraded, forbidden, failed, and domain-specific states that must be represented. |
| `evidence_or_justified_omission` | Figma node evidence or a reason the block is not meaningful for the journey. |

Do not copy Linear branding, product taxonomy, text, proprietary assets, exact
pixel coordinates, or unsupported entities. Do not invent labels, assignees,
milestones, progress, activity, or collaboration objects for visual similarity.
Roehub owns its composition, themes, permissions, routes, data, terminology,
financial semantics, and interaction priorities.

## Review, evidence, and handoff

A Figma design ticket records:

- account, project, file key, file URL, and editor type;
- target page IDs and stable node IDs;
- before/after page, variable, component, and variant inventories;
- four-theme representative matrix and contrast/accessibility observations;
- functional-reference mapping and justified omissions;
- screenshots or renders of every changed target at useful zoom;
- remaining open design decisions and explicit product-owner review;
- the repeated final read used to detect identity or inventory drift.

The product owner approves the named Figma state before a design ticket becomes
`accepted`. Figma approval does not authorize implementation automatically; the
delivery graph and the implementation ticket remain the execution authority.

## Proof boundary

Figma proves editable design structure and visual intent. It does not prove:

- server authorization, persistence, API behavior, or trading semantics;
- DOM semantics, keyboard behavior, focus restoration, screen-reader output,
  localization, browser reflow, or reduced-motion behavior;
- REST/SSE integration, performance, failure recovery, release, or deployment.

Implementation tickets must re-establish those claims with code, focused tests,
real-browser evidence, measured performance, and rollback proof. Screenshots,
Figma frames, prototypes, and connector success cannot replace runtime evidence.

## Compatibility and rollback

- Runtime, API, route, persistence, and theme identifiers: `none` in this
  migration; only design-tool routing changes.
- Unstarted Penpot vNext work: `compatible replacement` by the registered Figma
  file and renamed future ticket.
- Accepted Penpot evidence: `retain`; no historical assertion is rewritten.
- Custometry design delivery: `none`; its own project instructions remain
  authoritative.
- Repository rollback: revert the Figma routing commit. The external blank file
  may remain in the project but ceases to be authoritative if its repository
  registration is reverted.

## Verification

```bash
python -m tools.delivery.validate_roehub_delivery_model
python -m tools.docs.generate_docs_index --check
python -m tools.docs.generate_project_map --check
git diff --check
```
