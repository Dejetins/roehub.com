---
ticket_id: ROEHUB-LINEAR-REFERENCE-COMPLETION-2026-07-20
status: ready
owner: unassigned
depends_on: []
evidence: []
---

# Complete the Linear-workspace reference evidence

## Outcome

Roehub has a sanitized, reproducible reference pack sufficient to measure the
selected Linear shell, panel, keyboard, motion, state, and perceived-performance
behavior before Penpot or Web implementation begins.

## Context

- `.codex/delivery/specs/roehub-linear-workspace-ui-transition.md`
- `docs/architecture/ui/linear-workspace-ui-transition-standard-v1.md`
- `docs/architecture/ui/linear-workspace-reference-manifest-v1.json`
- `docs/architecture/ui/roehub-linear-ui-migration-registry-v1.json`
- user-supplied archive `/Users/daniildegtyarev/Downloads/reference.zip`

## Start probe

Before broad reads or writes, confirm that the archive exists and its SHA-256 is
`eb7b0ab070f64d553baafacefa90fdb2e87e51bc174c63db9af73bc77f8e41c2`.
Stop on absence or identity drift.

## Owned paths

- this ticket;
- `.codex/delivery/evidence/ROEHUB-LINEAR-REFERENCE-COMPLETION-2026-07-20.md`;
- `docs/architecture/ui/linear-workspace-reference-manifest-v1.json`;
- `docs/architecture/ui/linear-workspace-reference-measurements-v1.md`.

## Scope

- Verify all supplied screenshot hashes and record viewport/scale metadata.
- Close or explicitly waive command-palette, keyboard focus, sidebar resize,
  route/pane/modal/popover motion, loading/error/stale/forbidden/session,
  accessibility snapshot, component geometry, and motion timing gaps.
- Reuse an identical accepted sanitized measurement contract from Custometry
  when available, then add only Roehub-specific interpretation.
- Keep screenshots, recordings, auth storage, cookies, tokens, and browser
  profiles outside Git; commit only hashes, measurements, descriptions, and
  redacted evidence.

## Non-goals

- No product code, Penpot, routes, backend contracts, or product requirements.
- No copied Linear branding, text, assets, source code, or authorization rules.

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
`accepted` only after the recorded checks pass.
