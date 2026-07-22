---
ticket_id: ROEHUB-LINEAR-REFERENCE-COMPLETION-2026-07-20
status: accepted
owner: unassigned
ticket_graph: .codex/delivery/graphs/roehub-authenticated-platform-delivery-v1.json
depends_on: []
evidence:
  - .codex/delivery/evidence/ROEHUB-LINEAR-REFERENCE-COMPLETION-2026-07-20.md
---

# Complete the Linear-workspace reference evidence

## Outcome

Roehub has a sanitized, reproducible reference pack sufficient to measure the
selected Linear shell, panel, keyboard, motion, state, and perceived-performance
behavior before Penpot or Web implementation begins. The pack also defines
formal functional-block equivalence without requiring literal screen copying.

## Context

- `.codex/delivery/specs/roehub-linear-workspace-ui-transition.md`
- `docs/architecture/ui/linear-workspace-ui-transition-standard-v1.md`
- `docs/architecture/ui/linear-workspace-reference-manifest-v1.json`
- `docs/architecture/ui/roehub-linear-ui-migration-registry-v1.json`
- user-supplied archive `/Users/daniildegtyarev/Downloads/reference.zip`
- three user-supplied supplemental project-overview captures received on
  `2026-07-22`, identified by hash in the manifest and retained outside Git

## Start probe

Before broad reads or writes, confirm that the archive exists and its SHA-256 is
`eb7b0ab070f64d553baafacefa90fdb2e87e51bc174c63db9af73bc77f8e41c2`.
Stop on absence or identity drift.

## Owned paths

- this ticket;
- `.codex/delivery/evidence/ROEHUB-LINEAR-REFERENCE-COMPLETION-2026-07-20.md`;
- `.codex/delivery/specs/roehub-linear-workspace-ui-transition.md`;
- `docs/architecture/ui/linear-workspace-reference-manifest-v1.json`;
- `docs/architecture/ui/linear-workspace-reference-measurements-v1.md`;
- `docs/architecture/ui/roehub-linear-ui-migration-registry-v1.json`.

## Scope

- Verify all archive and supplemental screenshot hashes and record available
  pixel metadata without inferring unavailable CSS scale.
- Close or explicitly waive command-palette, keyboard focus, sidebar resize,
  route/pane/modal/popover motion, loading/error/stale/forbidden/session,
  accessibility snapshot, component geometry, and motion timing gaps.
- Reuse an identical accepted sanitized measurement contract from Custometry
  when available, then add only Roehub-specific interpretation.
- Keep screenshots, recordings, auth storage, cookies, tokens, and browser
  profiles outside Git; commit only hashes, measurements, descriptions, and
  redacted evidence.
- Define formal reference translation: selected Roehub screens preserve
  analogous functional blocks and state relationships, while Roehub owns their
  domain semantics, composition, copy, styling, data sources, and permissions.

## Non-goals

- No product code, Penpot, routes, backend contracts, or product requirements.
- No copied Linear branding, text, assets, source code, or authorization rules.
- No one-to-one page cloning, mandatory pixel positions, or fabricated Roehub
  concepts added only to resemble Linear.

## Proof boundary

Use `browser-qa-evidence` with `playwright` mechanics where the authenticated
reference session permits it. Evidence must record source hashes, browser and
viewport metadata, sanitized ARIA observations, measured geometry/motion,
waivers, redaction, and `git diff --check`. It does not prove Roehub runtime.

## Escalation triggers

- Archive identity drift or unavailable reference session.
- A required capture would expose credentials, account data, or raw browser state.
- Product or code changes are required to close a reference gap.

## Acceptance

All manifest gaps are closed or explicitly waived with impact, terminal
evidence exists, no prohibited capture is tracked, and the ticket becomes
`accepted` only after the recorded checks pass. Downstream UI tickets must map
each selected reference block to a Roehub function or justify its omission.
