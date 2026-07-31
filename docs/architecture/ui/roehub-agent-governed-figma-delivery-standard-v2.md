# Roehub agent-governed Figma delivery standard v2

This standard defines the greenfield, contract-driven process for Roehub authenticated-platform design and replaces frame-led agent iteration.

## Status

- Status: `accepted architecture`.
- Decision date: `2026-07-31`.
- Acceptance authority: product owner.
- Supersedes for future authenticated-platform design:
  - `docs/architecture/ui/roehub-figma-design-delivery-standard-v1.md`;
  - the authenticated-platform design clauses in
    `docs/architecture/apps/web/roehub-ui-design-and-delivery-architecture-v1.md`.
- The superseded files and the former Figma file remain historical evidence only.

## Outcome

Roehub UI work is delivered through a single Codex coordination boundary. Agent output is
untrusted until deterministic structural gates and an independent visual review pass. The
product owner receives only gated candidates and makes product or visual decisions rather
than manually finding routine defects.

## Canonical Figma workspace

| Boundary | Canonical identity |
|---|---|
| Account | `dejetins@gmail.com` |
| Plan | `team::831604964356268687` (`pro`, `Full`) |
| Project | `roehub.com` |
| Project ID | `629113387` |
| Library file | `Roehub UI Library` |
| Library file key | `rgbNUPCuV7q2pARG4Cml8V` |
| Library URL | `https://www.figma.com/design/rgbNUPCuV7q2pARG4Cml8V/Roehub-UI-Library` |
| Product file | `Roehub Authenticated Platform UI` |
| Product file key | `nzKVsXuCmoTbHJGckHfK3T` |
| Product URL | `https://www.figma.com/design/nzKVsXuCmoTbHJGckHfK3T/Roehub-Authenticated-Platform-UI` |

The former file key `GBzmB9evtzqnAYNjp9W1sr` is `historical_only` and is forbidden as
an input to new design tasks, audits, component discovery, visual inference, or acceptance.
No node or style is copied from it into either canonical file.

## Source-of-truth boundaries

| Decision | Owner |
|---|---|
| Product capabilities, routes, permissions, authoritative states | Current Roehub product and runtime contracts |
| Design rules, component API, allowed variants, composition requirements | Repository UI contracts selected by the current ticket |
| Reusable visual implementation | Published assets in `Roehub UI Library` that passed the library gate |
| Candidate and accepted screen composition | Named nodes in `Roehub Authenticated Platform UI` plus their exact composition manifest |
| Runtime behavior | Application code, APIs, tests, and real-browser evidence |
| Product and visual acceptance | Explicit product-owner decision on named artifacts |

Figma remains the canonical editable visual design surface. It does not implicitly own
requirements, status, component semantics, or runtime truth.

## Dependency direction

```text
product/UI blueprint
  -> design contract
    -> approved visual direction
      -> library tokens/styles/components/patterns
        -> composition manifest
          -> isolated Figma candidate
            -> structural audit + visual audit
              -> product-owner decision
                -> accepted Figma composition
                  -> implementation ticket
```

Dependencies never point from an accepted or rejected screen back into the library contract.
Screens are consumers, not component or requirement sources.

## File and page boundaries

`Roehub UI Library` uses exactly this page order:

```text
00 Governance
01 Foundations
02 Icons
03 Components
04 Patterns
80 Audit Sandbox
90 Archive
```

`Roehub Authenticated Platform UI` uses exactly this page order:

```text
00 Governance
01 Direction Review
02 Candidate
03 Accepted
80 Audit Sandbox
90 Archive
```

Invariants:

- exactly one active candidate exists on `02 Candidate`;
- rejected or superseded candidates move to `90 Archive` and are never agent input;
- accepted nodes move to `03 Accepted` only after an explicit owner decision;
- `01 Direction Review` may contain raster exploration or clearly labelled non-canonical
  specimens, never reusable masters;
- reusable masters exist only in the library file;
- `80 Audit Sandbox` is disposable and cannot be accepted directly;
- the product file must consume published library components rather than detached copies;
- a task receives only its exact target page/node, manifest, and allowlisted library assets.

Figma library publication and enabling are explicit connector boundaries. After
`library_slice_review`, the product owner publishes the named library revision in Figma and enables
it for the product file. Codex then verifies availability and component keys before a composition
manifest may declare `published_and_enabled`. No agent infers publication from local components.

## Artifact model

Each non-trivial design iteration selects the smallest current set below:

1. Product/UI blueprint: surfaces, user tasks, authoritative data/actions, and required states.
2. Design contract: tokens, typography, icon policy, component API, density, and accessibility
   requirements.
3. Component registry: stable component keys, variant properties, slots, content limits, and
   lifecycle state, validated against
   `docs/architecture/ui/roehub-ui-component-registry-schema-v1.json`.
4. Composition manifest: the complete allowlist for one candidate.
5. Figma audit report: observed post-write inventory and gate results.
6. Ticket evidence: screenshots, exact IDs, checks, residual gaps, and explicit owner decision.

An accepted frame is not one of these input artifacts.

## Coordinator and executor model

The product owner interacts with Codex only. Codex owns task classification, context selection,
manifest construction, write scope, verification, bounded repair, and the review packet.

The executor may be Figma Plugin API code invoked through MCP or, for non-canonical exploration,
a visual generator. A Figma-native agent is optional and is never trusted with canonical library
or accepted product nodes.

Executor output is always `untrusted_output`. It becomes `candidate_ready` only after all required
gates pass. The executor cannot set `candidate_ready`, `accepted`, or `rejected`.

## State machine

```text
brief_draft
  -> brief_approved
    -> executing
      -> gate_failed -> repair_attempt -> executing
      -> candidate_ready
        -> accepted
        -> rejected -> archived
```

Rules:

- Codex runs all mandatory checks without a separate user request.
- At most two automatic repair attempts are allowed per candidate.
- A third failure stops with exact failed gates; broken output is not presented as complete.
- Silence never advances a state.
- Only the product owner can select a direction or set `accepted`/`rejected`.

## Write contract

Every canonical Figma write requires:

1. current ticket and composition/patch manifest;
2. exact file key, page ID, parent node ID, and owned node IDs;
3. before inventory;
4. one atomic bounded mutation;
5. read-back inventory;
6. structural gate;
7. screenshot and visual audit when appearance changed;
8. rollback or archive path.

The executor must not scan other product pages for inspiration, infer from historical nodes,
create missing library assets inside the product file, detach instances, or widen the mutation
boundary after execution starts.

## Mandatory gates

### Preflight

- selected contract and ticket are current;
- exact Figma account, plan, file, page, and target IDs match;
- manifest validates and contains no unknown component or variant;
- required copy, fields, actions, and states are complete;
- no unresolved product decision is disguised as executor freedom.

### Library

- required collections, variables, modes, styles, components, sets, and properties exist once;
- component internals bind to approved variables and text styles;
- icons contain meaningful vector geometry and accessible names;
- dimensions, radius, content extremes, focus, disabled, loading, error, and density rules pass;
- no detached instances, duplicate masters, placeholder geometry, or unregistered variants exist.

### Composition

- every declared component key and variant matches the manifest;
- no undeclared raw UI nodes exist outside the explicit exploration allowlist;
- every required action, field, column, state, and alternative representation is present;
- layout has no clipping, overlap, accidental overflow, or hidden required content;
- the write changed only the owned candidate boundary.

### Visual

- screenshot is inspected at the declared viewport and state;
- hierarchy, density, typography, alignment, contrast, and content extremes are reviewed;
- visual review is independent of the mutation success response;
- structural success cannot override a visual failure.

### Owner review

- the owner sees only a gated review packet;
- the packet names the candidate, exact Figma IDs, screenshots, passed gates, open decisions, and
  residual risks;
- acceptance applies only to the named revision and never to future variants by analogy.

## Pilot rollout

The first iteration proves the process on a deliberately small Backtests slice:

- one Backtests toolbar;
- one representative job row;
- one detail-dock header;
- one loading or degraded state;
- a small component set sufficient for that composition.

The pilot is not a full design system and cannot unblock production implementation. It succeeds
only if the same manifest can be rendered, audited, deliberately broken, rejected by the gate,
repaired within the bound, and packaged for owner review without asking the owner to perform QA.

## Rollback and recovery

- page scaffolding is additive and idempotent;
- candidate mutations remain isolated until acceptance;
- a failed candidate is archived or removed by exact task-owned node ID;
- library assets are not published until their gate passes;
- publication and product-file enablement are manual Figma-owner actions followed by a Codex
  read-only verification;
- a bad library publication requires a new corrected version; product candidates remain pinned to
  the last accepted component keys until revalidated;
- repository rollback restores prior routing, but the two external files may remain as
  non-authoritative artifacts.

## Proof boundary

This workflow can prove Figma identity, structure, bindings, component reuse, composition coverage,
and inspected visual intent. It cannot prove DOM semantics, keyboard runtime behavior, browser
reflow, screen-reader output, API behavior, authorization, persistence, performance, release, or
deployment. Implementation tickets must prove those boundaries independently.

## Alternatives rejected

### Accepted frames as the next prompt source

Rejected because multiple frames create ambiguous precedence and encourage inference, invention,
and accidental reuse of one-off geometry.

### Variables as a complete UI source

Rejected because variables store scalar values and modes, not component anatomy, patterns, product
requirements, or composition rules. Bindings are opt-in and therefore require audit.

### Unrestricted Figma-agent iteration

Rejected because prompt compliance cannot guarantee mutation isolation, component reuse, or stable
acceptance state.

### Code-first replacement of Figma

Rejected because Roehub requires Figma as the active editable visual design surface. Runtime code
still remains the authority for executable behavior.

## Validation ladder

- Repository contract: JSON/schema validation and focused unit tests.
- Figma boundary: repeated Plugin API inventories plus screenshots of changed targets.
- Product boundary: explicit owner decision on the named review packet.
- Runtime boundary: excluded from this design pilot.
