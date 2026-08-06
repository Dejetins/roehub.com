# Roehub Local Platform Web UI Program

This directory is the active draft control plane for the self-hosted Roehub Web
UI program admitted on 2026-08-06.

## Current boundary

- Program status: `draft`; G0 has not been claimed or closed.
- Product scope: self-hosted local platform Web UI only.
- Public `roehub.com`: excluded.
- Mobile-specific UI: unauthorized.
- Current work: source-backed intake, measured platform baseline, and draft
  prompt/ledger bootstrap.
- Product implementation, publication and deployment: not authorized by this
  program state.
- Visual source PNGs are local owner inputs and browser captures intentionally
  excluded from the public repository. Their tracked manifests and hashes are
  authoritative; a checkout without the exact local files cannot claim G0
  visual-source readiness and must use `external_authority` rather than invent
  or download replacements.

## Authoritative entrypoints

- `ui-design-program.json` — program manifest and execution-artifact paths.
- `ui-program-intake.json` — product, journey, screen, state and action
  contracts; currently incomplete.
- `platform-ui-baseline.json` — fixed shell, token, component, interaction and
  responsive contracts; currently incomplete.
- `decisions/visual-authority-owner-decision.json` — canonical accepted visual
  authority decision.
- `g0-intake-questionnaire.md` — machine findings for the G0 executor; it is
  not a request for the owner to fill JSON.

## Accepted supporting evidence

- `../../evidence/roehub-web-ui-2026-08-06/roehub-ui-intent-baseline.md`
- `../../evidence/roehub-web-ui-2026-08-06/linear-v23-audit-and-design-direction.md`
- `../../evidence/roehub-web-ui-2026-08-06/analog-platform-ux-research.md`
- `../../evidence/roehub-web-ui-2026-08-06/visual-authority/linear-reference-set/manifest.md`
- `../../evidence/roehub-web-ui-2026-08-06/visual-authority/supporting-custometry-pilot-v2/manifest.md`

## Next execution unit

Complete G0 from repository sources: exact-cover the 35 registered surfaces and
12 journey entrypoints in intake, measure and bind the fixed responsive platform
baseline to the accepted reference set, create and validate the draft prompt
pack and stage ledger, then produce the G0 → G1 transition receipt. Do not ask
the owner to reproduce repository facts or visual measurements.
