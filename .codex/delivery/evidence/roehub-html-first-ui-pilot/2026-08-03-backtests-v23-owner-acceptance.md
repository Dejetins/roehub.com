# Backtests Workbench v23 — product-owner acceptance

## Decision

- Date: `2026-08-03`.
- Authority: `product_owner`.
- Decision: `accepted`.
- Accepted status: `accepted_visual`.
- Family scope: `Backtests family` only.
- Mobile scope: `unauthorized`.
- Exact product-owner message:
  `Принимаю 2026-08-03-linear-black-workbench-v23.html, SHA-256 3ff799ac5a5872662dda8b67fc1bd4db0c7860b7de9d84e6597465209d5dd2a4, как accepted visual для Backtests family. Mobile scope не разрешаю.`

## Accepted identity

- Source artifact:
  `.codex/delivery/evidence/roehub-ui-agent-governed-pilot/specimens/2026-08-03-linear-black-workbench-v23.html`.
- SHA-256:
  `3ff799ac5a5872662dda8b67fc1bd4db0c7860b7de9d84e6597465209d5dd2a4`.
- Reviewed source canvas: `1672x941`.
- Reviewed state: the initial deterministic document state encoded by the accepted HTML.
- Represented screen identities:
  - `screen.backtests.library`;
  - `screen.backtests.detail`;
  - application-shell fragments visible in the specimen.
- Not represented as an accepted screen: `screen.backtests.builder`.

## G0 source resolution

- This decision makes v23 the current accepted visual authority for the
  Backtests family.
- The exact four current theme IDs remain `abyss`, `graphite`, `frost`, and
  `paper`; `graphite` remains the authenticated-Web default. The older
  `six_themes` target is superseded and cannot be used as current future-design
  authority.
- Mobile-specific information architecture, navigation, manifests, components,
  viewports, or touch-first behavior remain unauthorized.
- The earlier v9 acceptance remains truthful historical evidence. Existing
  tokens, component masters, registries, manifests, generated HTML, screenshots,
  and browser receipts derived from v9 retain that lineage and are not silently
  relabelled as v23 evidence.
- Existing v9-derived artifacts are stale for v23 provenance until a separate,
  explicitly authorized non-G0 unit regenerates and revalidates them.

## Acceptance boundary

This decision accepts the exact v23 visual for the Backtests family. It does not
accept or authorize by analogy:

- a product-wide visual system or another screen family;
- responsive-Web behavior beyond evidence already recorded for the exact
  artifact;
- mobile scope;
- a UI design program manifest, G1 atlas execution, component catalog,
  screen manifest, prompt pack, or ledger;
- production implementation, route, API, permission, runtime, Git publication,
  deployment, or release changes.

## Write authorization

The product owner separately authorized recording this acceptance and updating
only G0 authority sources, with the explicit instruction not to create a program
manifest and not to begin G1.
