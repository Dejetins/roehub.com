# Codex task: complete the Roehub Backtests agent-governed Figma pilot

## Mission

Continue the current ready ticket `ROEHUB-UI-AGENT-GOVERNED-PILOT-2026-07-31` from checkpoint
`library_slice_review` through `pilot_final_approval`. Build the bounded reusable library slice,
gate it, pause for the required product-owner decisions and manual library publication, compose one
isolated product candidate from verified published instances, prove the negative gate, and finish
with structural and independent visual evidence.

This prompt authorizes the ticket-owned repository and canonical Figma writes described below. It
does not authorize Git publication, runtime implementation, deployment, or changes outside the
named boundaries. Do not infer product-owner acceptance from this prompt or from silence.

## Routing decision

- `classification`: one ready delivery ticket.
- `execution_unit`: `.codex/tickets/2026-07-31-roehub-ui-agent-governed-pilot.md`.
- `primary_skills`: `figma:figma-use`, then `figma:figma-generate-library` for the library phase and
  `figma:figma-generate-design` for the product-candidate phase.
- `companion_skills`: `better-ui` for component craft, `better-layout` for geometry/alignment,
  `better-typography` for the accepted type ladder, `better-colors` for token/contrast review,
  `better-accessibility` for state and naming checks, `better-writing` for degraded-state copy, and
  `backend-quality-gates` for repository validators/tests. Load each only immediately before its
  crossed boundary rather than preloading all of them.
- `proof_boundary`: canonical Figma identity, structure, bindings, instance reuse, declared content,
  and inspected visual intent. No runtime or browser claim.
- `external checkpoints`: `library_slice_review`, manual publication/enablement,
  `composed_candidate_review`, `pilot_final_approval`.

Load each named skill completely immediately before using its surface. In particular, invoke
`figma:figma-use` before every `use_figma` operation, `figma:figma-generate-library` before library
mutation, and `figma:figma-generate-design` before product-candidate mutation.

## Common always-read sources

Read these current sources before the first mutation; stop broad discovery after they settle the
scope, exact identities, design rules, and proof boundary:

1. `.codex/AGENTS.md`.
2. `.codex/tickets/2026-07-31-roehub-ui-agent-governed-pilot.md`.
3. `docs/architecture/ui/roehub-agent-governed-figma-delivery-standard-v2.md`.
4. `docs/architecture/ui/roehub-backtests-process-pilot-brief-v1.md`.
5. `docs/architecture/ui/roehub-linear-black-authenticated-workspace-visual-standard-v1.md`.

## Phase-specific context bundles

Keep each active phase at eight repository files or fewer. Load only the named bundle immediately
before that phase and release older phase-only context when it is no longer needed.

### Library preflight and review

- common five sources;
- `.codex/delivery/evidence/roehub-ui-agent-governed-pilot/2026-08-01-direction-linear-black-v9-owner-acceptance.md`;
- `.codex/delivery/evidence/roehub-ui-agent-governed-pilot/specimens/2026-08-01-linear-black-workbench-v9.html`;
- `.codex/delivery/evidence/roehub-ui-agent-governed-pilot/2026-07-31-figma-page-scaffold.json`.

### Publication verification and composition

- common five sources;
- `docs/architecture/ui/roehub-ui-agent-delivery-contract-v1.json`;
- `docs/architecture/ui/roehub-ui-component-registry-schema-v1.json`;
- `docs/architecture/ui/roehub-ui-composition-manifest-schema-v1.json`.

### Final audit

- ticket, Figma delivery standard, pilot brief, proposed visual standard;
- the completed component registry and composition manifest;
- the accepted v9 HTML;
- the current audit JSON or screenshot under review.

Accepted reference SHA-256:
`fb09994ffa714fffd1b9988758a50ab68246303461007b01ea252d5c5480471c`.

Consult the v7/v8 doctrines and reviews only when one of the accepted v9 sources explicitly refers
to an unchanged invariant that cannot be resolved from the reusable visual standard. Never use a
rejected raster, former frame, historical prompt, or historical Figma node as an input.

## Current accepted state

- `direction_selection` is complete.
- Direction ID: `linear_black_backtests_workbench_v9`.
- Product-owner decision: `accepted` on `2026-08-01`.
- Current ticket checkpoint: `library_slice_review`.
- The accepted decision covers topology, density, alignment, typography, compact controls, and the
  neutral-black visual language only. No Figma asset or composition is accepted yet.
- The HTML is visual evidence, not a reusable component source.
- `roehub-linear-black-authenticated-workspace-visual-standard-v1.md` is a
  `proposed_for_library_slice_review` translation. The current v9 decision does not accept that
  derived document or authorize future screens by analogy.

## Canonical Figma boundaries

### Account and project

- Account: `dejetins@gmail.com`.
- Plan: `team::831604964356268687` (`pro`, `Full`).
- Project: `roehub.com`, project ID `629113387`.

### Roehub UI Library

- File key: `rgbNUPCuV7q2pARG4Cml8V`.
- URL: `https://www.figma.com/design/rgbNUPCuV7q2pARG4Cml8V/Roehub-UI-Library`.
- Pages:
  - `00 Governance`: `0:1`;
  - `01 Foundations`: `5:2`;
  - `02 Icons`: `5:3`;
  - `03 Components`: `5:4`;
  - `04 Patterns`: `5:5`;
  - `80 Audit Sandbox`: `5:6`;
  - `90 Archive`: `5:7`.

### Roehub Authenticated Platform UI

- File key: `nzKVsXuCmoTbHJGckHfK3T`.
- URL:
  `https://www.figma.com/design/nzKVsXuCmoTbHJGckHfK3T/Roehub-Authenticated-Platform-UI`.
- Pages:
  - `00 Governance`: `0:1`;
  - `01 Direction Review`: `3:2`;
  - `02 Candidate`: `3:3`;
  - `03 Accepted`: `3:4`;
  - `80 Audit Sandbox`: `3:5`;
  - `90 Archive`: `3:6`.

The historical file `GBzmB9evtzqnAYNjp9W1sr` is forbidden as input. Do not open it for
inspiration, inspect its assets, or copy from it.

Before every write, verify the authenticated identity, plan, file key, page ID/name, exact parent,
current inventory, and owned node IDs. Use one atomic bounded mutation, then read back the result.
Do not scan unrelated pages.

## Reproducible visual rules

Treat `roehub-linear-black-authenticated-workspace-visual-standard-v1.md` as normative. Enforce at
least the following in every created asset and candidate:

- neutral workstation surfaces: canvas `#0B0E11`, panel `#151A1F`, list `#14191E`, selected
  `#20262D`, control `#1A2026`;
- borders `#353D45` / `#293139`, primary text `#F1F3F5`, secondary `#B1B8C0`, quiet `#858E98`;
- preserve the accepted accent roles: primary action `#6540DF` with `#8264EC` boundary, selected
  control `#5C35D1` with `#7F62E8` boundary, selected job rule `#8B5CFF`, selected variant rule
  `#8158F5`, navigation rule `#875DFF`, general reference `#7952F4`, and focus `#A58AFF`;
- success `#49CC54`, danger `#FF4B39`, warning `#F5BD22`;
- `28px` control height, `14px` icon box, `1.5px` icon stroke, `8px` control radius, `10px` panel
  radius, `7px` clipped row-container radius, `1px` borders;
- `4px` panel rhythm, `12px` major content rhythm, `8px` control gaps, `6px` icon/label gaps;
- `52px` panel/detail header, `50px` compact job row, `32px` progress ring;
- peer first-content borders align `29px` after their `52px` headers, at `81px` from panel top;
- the accepted macOS system stack with `SF Pro Text` as the Figma target when available, normal
  width, tabular dynamic values, and the exact style ladder from the proposed visual standard;
- all peer micro labels share one family, size, line height, and weight;
- semantic colors never carry meaning alone; no green/red performance-cell fills;
- no glass, blur, glow, floating dashboard cards, decorative gradients, oversized titles, pill
  inflation, doubled bottom borders, uneven peer insets, inconsistent icon scale, or raw product UI
  nodes;
- screen-specific content comes from the brief, not from the visual reference.

Do not generalize the Backtests-specific `396px / 328px / flexible detail` columns to another
screen. Reuse the visual grammar, tokens, component APIs, and alignment system, then derive each
future screen's structure from its own product tasks.

Do not silently normalize accent roles or replace the accepted system-font rendering with Inter.
If `SF Pro Text` is unavailable in Figma, record the exact fallback as an open owner decision in
`library_slice_review`.

## Deliverable 1: minimal Roehub UI Library slice

Create only the variables, text/paint styles, icons, primitives, and composed components required
for this pilot:

1. Backtests toolbar.
2. Representative Backtests job row.
3. Detail-dock header.
4. Degraded freshness notice and its required refreshing/control state.

The minimum reusable anatomy may include compact icon/labeled controls, search/filter control,
segmented or split refresh control, state glyph, and text styles only when directly consumed by the
four assets. Do not create a complete design system.

Use these stable registry IDs for the four required components:

- `backtests.toolbar`;
- `backtests.job-row`;
- `backtests.detail-dock-header`;
- `feedback.degraded-freshness`.

Their composition slots are exactly `toolbar`, `job-row`, `detail-dock-header`, and
`degraded-freshness`.

### Required component behavior and content

#### Toolbar

- Manifest concepts: text query, job state, exchange, market type, symbol, launched date range,
  manual refresh, auto-refresh preset, and refresh status.
- Progressive disclosure may group less-frequent filters, but no field disappears from the API or
  manifest.
- Manual refresh is icon-only with an accessibility-facing name. The current interval remains
  visible. Refreshing/degraded states do not shift geometry.

#### Job row

- Required content remains recoverable: `job_id`, strategy, exchange, market type, symbol,
  indicator summary, period, direction, combinations, best return, best Sharpe, average drawdown,
  profit factor, win rate, trades count, state, progress, created time, and refresh status.
- The compact row owns fast comparison; the detail context may own progressively disclosed fields.
- Required row state: `completed`; selected presentation must also be demonstrated.
- A degraded workstation projection is not a failed job.

#### Detail-dock header

- Selected job identity and strategy; symbol, market type, and period; completed status; last
  projection time; degraded freshness; icon-only close action with an accessibility-facing name.
- The detail body is out of scope.

#### Degraded freshness

- Cached job data remains visible.
- Copy says freshness is degraded, not that the job failed.
- Manual refresh remains available when permitted.
- Retry timing appears only when supplied by the response.
- State uses text/icon plus color, never color alone.

### State and content-extreme matrix

Exercise default, hover, focus, disabled, selected, completed, degraded, and refreshing where the
component API supports them. Include short English labels, long English labels, and the longer
Russian localization sample required by the pilot brief. Test long strategy/symbol identity and
nullable values without inventing zeroes.

Use these exact test-only geometry fixtures:

- `dema-1h-long-short-with-volatility-confirmation-a1b2c3`;
- `Стратегия пересечения DEMA с фильтром относительной силы`;
- `1000000PEPEUSDT`;
- `Data freshness is degraded. Showing the latest cached results.`;
- `Актуальность данных снижена. Показаны последние сохранённые результаты.`;
- unavailable metric `—`, never `0`.

Create candidate assets and the review specimen under Library `80 Audit Sandbox` (`5:6`) first.
Record every created node ID. Do not publish. After structural and visual gates pass, prepare the
`library_slice_review` packet and stop for the product owner.

## Deliverable 2: library structural and visual gate

Create or update a ticket-owned component registry at:

`docs/architecture/ui/roehub-backtests-pilot-component-registry-v1.json`

Validate it against `docs/architecture/ui/roehub-ui-component-registry-schema-v1.json`. It must
record stable asset IDs/names, Figma file/page/node identity, component properties,
variants, slots, content limits, token/style bindings, accessibility-facing names, lifecycle state,
and published component keys when they later exist.

The library gate fails if any required collection, variable, style, icon, component, set, or
property is missing or duplicated; if internals are not token/style bound; if icons are placeholder
geometry; or if detached instances, duplicate masters, unregistered variants, clipping, missing
states, or content-extreme failures exist.

Run an independent visual review from a screenshot of the exact sandbox specimen. Review hierarchy,
density, typography, alignment, contrast, component states, localization extremes, and visual
fidelity to the accepted v9 grammar. Mutation success is not visual proof.

Prepare evidence under:

`.codex/delivery/evidence/roehub-ui-agent-governed-pilot/`

The `library_slice_review` packet must name exact file/page/node IDs, before/after inventories,
registry revision, screenshots, structural checks, independent visual verdict, repair attempts,
open decisions, and residual risks. A failed candidate gets at most two bounded automatic repair
attempts. On a third failure, stop with the exact failed gates.

### Checkpoint: `library_slice_review`

Stop and ask the product owner to accept or reject the exact named review packet, including the
exact proposed visual-standard revision. Do not promote assets to owner pages, mark the standard
active, or infer acceptance.

After explicit acceptance, promote only the accepted assets to their exact Library owner pages:

- variables and styles -> `01 Foundations` (`5:2`);
- icons -> `02 Icons` (`5:3`);
- components -> `03 Components` (`5:4`);
- composed reusable pattern, if required -> `04 Patterns` (`5:5`).

Read back and gate the promoted assets again. Update the component registry with the final node IDs.
Only the exact accepted visual-standard revision may then become a future ticket-selectable source;
record that artifact-specific decision in ticket evidence before changing its status.

## Deliverable 3: manual publication boundary and key verification

Codex must not publish the Figma library or enable it for the product file.

After accepted assets are promoted and gated, stop and ask the product owner to:

1. publish the named `Roehub UI Library` revision in the Figma UI;
2. enable that library for `Roehub Authenticated Platform UI`.

After the owner confirms completion, perform read-only verification:

- confirm the exact library revision is available to the product file;
- resolve each approved component key;
- confirm every key maps to the expected component name and final node;
- reject missing, duplicate, stale, or unexpected keys;
- update the component registry to `published_and_enabled` only from observed evidence.

Do not proceed to composition while any key is unverified.

## Deliverable 4: manifest-built isolated Backtests candidate

Create:

`docs/architecture/ui/roehub-backtests-pilot-composition-manifest-v1.json`

Validate it against `roehub-ui-composition-manifest-schema-v1.json`. It must declare:

- ticket `ROEHUB-UI-AGENT-GOVERNED-PILOT-2026-07-31`;
- product target file `nzKVsXuCmoTbHJGckHfK3T`;
- target page `02 Candidate` (`3:3`) and an exact parent node ID verified immediately before write;
- viewport `1440 × 900`, state `degraded`;
- library file `rgbNUPCuV7q2pARG4Cml8V` with status `published_and_enabled`;
- the exact verified approved component keys;
- required actions exactly `open_details`, `manual_refresh`, `set_autorefresh`, `close_detail`;
- all required toolbar/job/header fields from the brief;
- required states `completed` and `degraded`;
- one bounded top-level candidate;
- `raw_node_allowlist: []`.

Verify that `02 Candidate` has no existing active candidate or archive the superseded task-owned
candidate by exact ID before creating one. Build exactly one isolated candidate from published
library instances only. Do not create missing masters in the product file, detach instances, or
add undeclared raw UI nodes.

After creation, update `mutation_boundary.owned_node_ids` with the exact candidate root node ID.
Read back the product tree and prove the target parent and created candidate root separately,
component keys, variants, text overrides, content coverage, mutation boundary, exactly one created
top-level candidate, clipping/overflow count, and absence of detached/raw nodes.

## Deliverable 5: negative gate, audits, and final checkpoints

### Deliberate negative gate

Before accepting the valid candidate, create two disposable invalid manifest fixtures under the
ticket evidence directory:

1. one valid-schema fixture with an unknown or unapproved component key;
2. one fixture with a non-empty `raw_node_allowlist`.

Run the validator separately for both and record each expected non-zero result and exact rejection
reason using
`uv run python -m tools.design.validate_roehub_ui_delivery --registry <registry> --manifest <invalid-fixture>`.
Neither invalid fixture may mutate the canonical candidate. Then validate the repaired/valid
manifest successfully with the same registry.

### Structural audit

Create a Figma audit JSON compatible with `io.roehub.ui.figma-audit/v1`. It must carry
`parent_node_id` for the verified target container and `root_node_id` for the newly created
candidate. Run:

`uv run python -m tools.design.validate_roehub_ui_delivery --registry <registry> --manifest <valid-manifest> --audit <audit>`

The audit must prove zero detached instances, raw UI nodes, unknown keys, missing actions/fields/
states, token-binding violations, text-style violations, clipping/overflow, and outside-boundary
changes. It must also prove `top_level_nodes_created == 1` and that the candidate root is a named
owned node distinct from the target parent.

### Independent visual audit

Have one read-only reviewer inspect the exact `1440 × 900` candidate screenshot independently from
the mutation response. Provide the accepted v9 specimen, reusable visual standard, brief, manifest,
and screenshot. The review must explicitly cover hierarchy, density, typography, control axes,
panel/content insets, contrast, state clarity, localization extremes, clipping, and every automatic
rejection condition. Resolve every Blocker/High issue within the maximum two repair attempts.

### Checkpoint: `composed_candidate_review`

Present only a gated packet containing exact file/page/root IDs, manifest revision, component keys,
screenshots, negative-gate evidence, structural audit, independent visual audit, repair history,
open decisions, and residual risks. Stop for explicit product-owner acceptance or rejection.

### Checkpoint: `pilot_final_approval`

After `composed_candidate_review` is explicitly accepted, prepare the final named evidence packet
and stop again for `pilot_final_approval`. Do not collapse the two decisions.

Only after explicit `pilot_final_approval` may the accepted candidate move to Product
`03 Accepted` (`3:4`) by an exact bounded write. Record the accepted root node ID and final
read-back inventory. Do not infer that this pilot authorizes a full screen, runtime implementation,
or future designs by analogy.

## Repository write scope

Allowed repository writes are limited to:

- this ticket and its exact evidence directory;
- `docs/architecture/ui/roehub-backtests-pilot-component-registry-v1.json`;
- `docs/architecture/ui/roehub-backtests-pilot-composition-manifest-v1.json`;
- the smallest validator/test changes required by the existing contracts;
- the architecture index only when changed by the task's architecture documents.

Preserve foreign changes and inspect exact owned hunks. Do not create a branch, worktree, stash,
Goal, prompt pack, ledger, PR, commit, push, release, deployment, or runtime change unless the user
explicitly authorizes that action in the executing chat.

## Required repository validation

Run at the nearest relevant checkpoint:

```bash
uv run python -m tools.design.validate_roehub_ui_delivery --registry <registry>
uv run python -m tools.delivery.validate_roehub_delivery_model
uv run pytest -q tests/unit/tools/test_validate_roehub_ui_delivery.py tests/unit/tools/test_validate_roehub_delivery_model.py
```

For each changed file under `docs/architecture/**`, acquire the global architecture-index lock,
using exact directory `/tmp/roehub-architecture-index.lock` via atomic `mkdir`. If it already
exists, stop rather than bypassing the lock. While holding it, run
`uv run python -m tools.docs.generate_docs_index`, then its `--check` form, and always remove only
that exact lock directory on exit. Run `git diff --check` on the owned diff. These source gates do
not replace Figma read-back and visual evidence.

## Stop conditions

Stop rather than improvise if:

- authenticated Figma identity, plan, file key, page ID/name, or parent node differs;
- the historical file appears in task input;
- a required product field/action/state is unresolved;
- publication or enablement has not been performed manually and verified;
- component keys are missing, stale, duplicate, or unverified;
- the mutation boundary would widen;
- two automatic repair attempts have already failed;
- a required product-owner checkpoint has no explicit artifact-specific decision;
- foreign repository changes cannot be safely separated.

## Final reporting contract

At every checkpoint report in Russian:

- actual scope completed;
- exact repository paths and Figma file/page/node/component identities changed;
- before/after inventories;
- validation commands and results;
- independent visual-review verdict and evidence paths;
- observed proof boundary;
- compatibility class (`none` unless a real API/schema/runtime boundary changes);
- foreign changes excluded;
- residual risks and the one next safe action.

Never claim runtime, browser, accessibility implementation, publication, acceptance, release, or
deployment from Figma structure or local source tests alone.
