# Delivery evidence: ROEHUB-LOCAL-UI-IA-2026-07-20

## Result

- Ticket status: `accepted`.
- Architecture status: `accepted_target_architecture`.
- Product-owner decision: accepted on `2026-07-20`.
- Accepted inventory baseline commit:
  `c6ef2f32464ea681c7582aa8b689aacdc02b5d70`.
- Scope: self-hosted local-platform information architecture only.
- Design-tool access: not performed.
- Product-code changes: none.
- Commit or push for this ticket: not performed.

Ticket acceptance means that this execution unit and its evidence are complete.
The product owner subsequently accepted the target information architecture and
authorization matrix. The former design-system follow-up was retired during the
2026-08-05 cleanup. After the 2026-09-04 workflow retirement, the accepted IA
remains an input to the user's selected UI task and to the server-authorization
ticket graph. It does not authorize visual or Web implementation by itself.

## Delivered artifacts

- `docs/architecture/apps/web/roehub-local-platform-information-architecture-v1.md`
- `docs/architecture/apps/web/roehub-local-platform-screen-registry-v1.json`
- `docs/architecture/apps/web/roehub-local-platform-access-and-route-contract-v1.json`
- `tests/unit/docs/test_roehub_local_platform_information_architecture.py`
- generated architecture index and project-map updates

## Deterministic coverage

- Accepted local surfaces mapped exactly once: `33/33`.
- Accepted local journeys mapped exactly once: `12/12`.
- Canonical screen, non-visual, system, and historical records: `35`.
- Server capabilities: `40`.
- Mutation-surface policies: `24`.
- Canonical target routes: `29`.
- Compatibility migrations: `7`.
- Local-platform widths: `820`, `1024`, and `1440`.
- Excluded local-platform width: `390`.
- Public-site surfaces included: `0`.

## Independent security review

The first independent review returned `Block` for four issues:

1. the operator safe-action subset was not closed and deterministic;
2. delegated capabilities lacked a complete grant and revocation contract;
3. browser mutations lacked one fail-closed server envelope;
4. the `/docs` migration did not cover both Web and proxied core API framework
   documentation.

The corrected review confirmed those four issues were resolved but returned a
second `Block` for two additional issues:

1. `installation.trust.manage` was granted by the authority overlay without a
   corresponding capability contract;
2. an organization operator could appear to restart an installation-shared
   service.

The final correction added a non-delegable, audited
`installation.trust.manage` capability and restricted operator service restart
to an organization-owned service. A shared installation-service restart now
requires a separate `installation_owner` check. The final independent verdict
was `Release`, with no unresolved material finding.

## Proof boundary

This task proves documentation and registry consistency, source traceability,
surface and journey coverage, and the independent review of the target
authorization boundary. It does not prove runtime authorization, browser
behavior, redirects, persistence, visual-program completeness, accessibility,
performance, release, or deployment.

## Verification

The final recorded command set covers:

- both deterministic UI architecture test modules: `6 passed`;
- Ruff lint and formatting: passed for both documentation test modules;
- JSON parsing: passed for both new registries;
- architecture-index check: up to date;
- project-map check: all five artifacts up to date;
- whitespace checks: passed for tracked and ticket-owned untracked artifacts;
- cold architecture self-review: no unresolved hidden product decision,
  public-site leakage, or implementation claim.

The only test warning is the repository-external unknown pytest option
`asyncio_default_fixture_loop_scope`; it does not affect the focused results.
