# ROEHUB-AUTHZ-BROWSER-MUTATION-ENVELOPE-2026-07-20 evidence

## Delivered boundary

- Added a route-facing `BrowserMutationEnvelope` that binds the existing origin
  semantics and double-submit CSRF proof to the application security decision.
- Requests without the configured browser session cookie return
  `applicable=false`; they are not authorized by this envelope and retain their
  existing API-client authentication path.
- Added server-owned action policies that bind an action to one capability and
  its resource/recent-auth requirements. Unknown actions and unknown
  capabilities deny.
- Added an `EffectiveAuthorizer` protocol and a grant-free normalized decision
  carrying only `allowed`, `capability`, `scope`, `authority_source`,
  `delegated_organization_id`, and a stable deny code.
- Added adapters for the direct capability kernel and for
  `DelegatedCapabilityService`. The delegation adapter builds a
  server-owned `DelegationEvaluationRequest` with organization scope and the
  normalized evaluation time, then validates capability, authority source,
  organization, grantee, scope, revocation, and expiry before allowing.
- `MutationSecurityService` now depends on `EffectiveAuthorizer`, validates the
  normalized decision against the action capability and selected organization,
  and fails closed on an exception or inconsistent result.
- Direct kernel authority is recorded as `capability_kernel`; active exact
  delegation is recorded as `delegation` plus only the delegated organization
  id. The persisted grant, delegation id, grantor, expiry, and revoke metadata
  are not exposed in the mutation decision or audit event.
- The permit carries only the validator-produced, recursively immutable payload;
  idempotency hashes the same normalized payload and scopes a key by actor,
  organization, capability, action, and server-resolved resource reference.
- Same-key terminal replay returns the stored terminal reference; changed
  content conflicts; processing and unknown results produce distinct deny
  decisions, and unknown results require reconciliation.
- Audit and idempotency are mandatory for an applicable browser mutation.
  Audit events contain stable action/capability/outcome data and hashes, not raw
  request content, CSRF/session values, or the raw idempotency key.
- Recent authentication defaults to ten minutes and cannot be disabled for the
  capability minimum covering credential management, role management, model
  promotion/rollback, installation trust/resources, and installation recovery.

## Contract impact

| Surface | Classification | Evidence and consequence |
|---|---|---|
| Effective-authorizer port and adapters | `compatible-change` | New opt-in internal boundary composes the accepted capability kernel and delegation core; no repository consumer or product route is wired to the primitive. |
| Cookie-authenticated browser mutation primitive | `compatible-change` | New opt-in facade; direct authority remains valid and active exact delegations can now authorize the same server-owned action policy. Revoked, expired, unavailable, or inconsistent authority denies. |
| API clients without the browser session cookie | `none` | The facade returns `applicable=false` and does not replace existing API authentication or authorization. |
| Existing routes and DTOs | `none` | Strategies, Backtests, Connections, Settings, Operations, and authentication routes were not edited. |
| Delegation core, persistence, schema, configuration | `none` | Existing delegation models/service/repositories, migrations, configuration, and secrets were not edited. |
| Session and CSRF contracts | `none` | Existing `csrf.py`, sessions, origin semantics, and cookie names were not edited. |
| Mutation audit projection | `compatible-change` | The new opt-in event adds normalized authority source and delegated organization id; it never embeds a grant or raw request material. |
| Future route side effects | `unknown` until integration | Each route must execute only `validated_payload`, finalize or reconcile its idempotency reservation, and supply a durable atomic store where crash recovery is required. |

## Checks run

- `uv run pytest -q -ra tests/unit/identity/mutation_security tests/unit/identity/authorization tests/unit/identity/delegation`
  — `38 passed, 1 skipped in 0.70s`; the existing PostgreSQL adapter test was
  skipped because `ROEHUB_TEST_DELEGATION_PG_DSN` is not configured.
- `uv run ruff check src/trading/contexts/identity/adapters/inbound/api/mutation_guard.py src/trading/contexts/identity/application/mutation_security tests/unit/identity/mutation_security`
  — passed (`All checks passed!`).
- `uv run ruff format --check src/trading/contexts/identity/adapters/inbound/api/mutation_guard.py src/trading/contexts/identity/application/mutation_security tests/unit/identity/mutation_security`
  — passed (`8 files already formatted`).
- `uv run pyright src/trading/contexts/identity/adapters/inbound/api/mutation_guard.py src/trading/contexts/identity/application/mutation_security tests/unit/identity/mutation_security`
  — `0 errors, 0 warnings, 0 informations`.
- Additional repository-wide `uv run pyright` was non-zero only in existing
  ignored `local_artifacts/rl_trading/**`: `149 errors, 2 warnings`. The focused
  owned-path type check above is green; foreign artifacts were not changed.
- `git diff --check` passed. Each untracked task file was additionally checked
  with `git diff --no-index --check /dev/null <file>`; all passed.
- On 2026-07-22, the publication-blocker refresh ran
  `python -m tools.docs.generate_project_map` — `OK: project map generated (5 artifacts)` —
  followed by `python -m tools.docs.generate_project_map --check` —
  `OK: project map up-to-date (5 artifacts)`. Only
  `docs/architecture/project-map/PROJECT_MAP.md` and
  `docs/architecture/project-map/project-map.json` changed: inventory
  `3062 -> 3071`, structural digest
  `6993c6a91dbe322593b38b7261c792ffaa784105663da7ed83f65a485a8aa7f2 ->
  10fa1c91af71616846d05550dc937fccfd5545bd08eceb5bafc9f0622aeabe7d`,
  domain `681 -> 687`, quality `433 -> 435`, knowledge `1096 -> 1097`, and
  `context:identity` `81 -> 87`.
- The same architecture-index lock covered
  `python -m tools.docs.generate_docs_index` — unchanged — and
  `python -m tools.docs.generate_docs_index --check` —
  `OK: .../docs/architecture/README.md is up-to-date.` The final generated
  diff passed `git diff --check`.
- Before generation, the exact local-only `/outputs/` pattern was added to
  `.git/info/exclude`; it is not a tracked or staged file. The resulting map
  includes the nine new Browser Mutation Envelope files and no
  `outputs/code-rot-cleaner/*` inventory entries.

Focused tests cover direct authority through the delegation composition,
active delegation, revoked delegation, expired delegation, authorizer
unavailability, grant-free authority metadata, foreign and opaque origins,
missing/mismatched CSRF, non-browser API-client compatibility, terminal replay,
payload conflict, actor/organization idempotency isolation, cross-organization
access, ownership, stale recent authentication, unknown capability, immutable
validated payload, missing mandatory context, audit redaction, and audit-failure
reconciliation.

## CI routing publication-blocker repair

- Before this repair, the command
  `printf '%s\\n' 'tests/unit/identity/mutation_security/test_browser_mutation_envelope.py' | python -m tools.ci.route_changes ci --changed-files /dev/stdin`
  fell through to the full matrix: `tests/unit/identity/` was not classified as
  identity-context work, and the `market-identity-strategy` target did not run
  that test tree.
- The same command now returns only `market-identity-strategy` with target
  `tests/unit/contexts/identity tests/unit/contexts/market_data tests/unit/contexts/strategy tests/unit/identity`.
- `uv run pytest -q tests/unit/tools/test_ci_route_changes.py` — `8 passed in
  0.03s`.
- `uv run pytest -q -ra tests/unit/identity/mutation_security tests/unit/identity/authorization tests/unit/identity/delegation`
  — `38 passed, 1 skipped in 0.68s`; the unchanged PostgreSQL delegation test
  remains skipped because `ROEHUB_TEST_DELEGATION_PG_DSN` is not configured.
- `uv run ruff check tools/ci/route_changes.py tests/unit/tools/test_ci_route_changes.py`
  — `All checks passed!`.
- `uv run ruff format --check tools/ci/route_changes.py tests/unit/tools/test_ci_route_changes.py`
  — `2 files already formatted`.
- `uv run pyright tools/ci/route_changes.py tests/unit/tools/test_ci_route_changes.py`
  — `0 errors, 0 warnings, 0 informations`.

## Security review

- Current cold self-review found no write outside the ticket paths and no route,
  session, persistence, migration, configuration, or delegation-core change.
- A separate read-only production-risk review found no P0/P1 and no actionable
  code finding. It confirmed fail-closed direct/delegated decisions, grant-free
  decision/audit projection, and preservation of the existing transport,
  validation, recent-auth, idempotency, immutable-payload, and audit guarantees.
- The review's only P2 finding was stale evidence. This file corrects the
  delivered boundary, commands, results, compatibility, and residual risks.
- Independent evidence re-review found no remaining finding and returned
  `ready_for_next_gate`.

## Proof boundary and residual risk

- This evidence proves the isolated shared envelope and its compatibility tests
  only. It does not claim that any Strategies, Backtests, Connections,
  Settings, Operations, or authentication route uses the envelope.
- `InMemoryMutationIdempotencyStore` is a thread-safe proof adapter, not durable
  crash-recovery evidence. Product integration must choose its store and
  reconciliation/lease policy without changing this ticket's proof claim.
- Route integration must keep `MutationSecurityRequest.now` server-resolved.
  Revocation between authorization and a later route side effect remains a
  route-level TOCTOU concern requiring integration-specific transaction or
  re-check semantics.
- No push, merge, PR, release, deploy, runtime, or browser-route acceptance was
  performed. `outputs/` remains untouched and is excluded only locally through
  `.git/info/exclude`; that local Git configuration is not staged.
