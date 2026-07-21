# Evidence: `ROEHUB-AUTHZ-DELEGATION-CORE-2026-07-20`

## Delivery status

- Ticket status: `accepted`.
- Baseline: `7314c1febeb4ccc50bba585f513921d27711f59a`.
- Worktree/branch: `codex/roehub-authz-delegation-core`.
- Predecessor `ROEHUB-AUTHZ-CAPABILITY-KERNEL-2026-07-20` was `accepted`; its
  evidence file existed, and `242b0507d2a4fc560bda0645aff44bebb31de05d` was
  confirmed as an ancestor of `HEAD` before the first write.
- No endpoint, route, session, static role, UI, design-system, Penpot, secret,
  release, or deployment change was made.

## Delivered boundary

- `DelegatedCapabilityService` composes the accepted capability kernel rather
  than changing its direct-role policy. Direct kernel allows retain authority
  source `capability_kernel`; persisted exact grants are the only source
  `delegation`.
- The accepted capability-to-grantee matrix is closed: `trader` may receive
  only `data.selection.manage`; `admin` may receive the other twelve accepted
  exact capabilities. An ineligible role is rejected both at grant time and at
  evaluation time, including for a malformed persisted row.
- A grant is bound to one organization, one capability, and the only accepted
  server-owned resource scope, `organization`. Arbitrary scope text is denied
  before persistence or audit. PostgreSQL enforces the same rule with
  `CHECK ((resource_scope = 'organization'::text))`.
- Active owner membership, recent-auth verification, no self-grant, no
  redelegation, active grantee membership, non-delegable rejection, expiry,
  exact organization lookup, and immediate idempotent revocation are enforced.
- An expired but unrevoked grant remains immutable audit history and requires
  explicit owner revocation before a new grant can be created. This avoids a
  silent replacement or an invented revocation actor.
- Grant and revoke events contain only capability id, grantee id, and the
  fixed non-secret scope in the existing administrative audit store.

## Migration and compatibility impact

| Surface | Classification | Evidence |
|---|---|---|
| PostgreSQL schema | `compatible-change` | One additive `identity_delegated_capabilities` table and two indexes; `downgrade()` drops only that table. |
| Authorization semantics | `compatible-change` | New internal delegation boundary composes the accepted kernel; static role policy and installation authority remain unchanged. |
| API, routes, sessions, UI | `none` | No inbound adapter or product/UI path changed. |
| Configuration, secrets | `none` | No configuration or secret storage/read path changed. |
| Operational recovery | `compatible-change` | A disposable explicit PostgreSQL DSN was used for forward/backward/forward verification; no default or shared database was used. |

The pre-write Alembic head was the single `20260711_0043`. The final worktree
head is the single `20260720_0044`, with exact metadata:

```python
revision = "20260720_0044"
down_revision = "20260711_0043"
```

The real disposable PostgreSQL cycle used an explicit temporary DSN on
`127.0.0.1:55444`, never the `alembic.ini` default:

1. `20260711_0043` → `20260720_0044`: table present and scope check present.
2. `20260720_0044` → `20260711_0043`: table absent.
3. `20260711_0043` → `20260720_0044`: table and scope check present again.

Observed final result:

```text
20260720_0044:identity_delegated_capabilities:CHECK ((resource_scope = 'organization'::text))
```

## Verification

| Check | Result |
|---|---|
| `uv run pytest tests/unit/identity/authorization tests/unit/identity/delegation -q` with explicit disposable PostgreSQL DSN | `20 passed` |
| Focused coverage | Grant, expiry, explicit revoke-before-renewal, immediate/idempotent revocation, self/redelegation/non-delegable denial, exact role matrix, malformed-row denial, cross-organization/resource-scope denial, redacted audit, in-memory and PostgreSQL parity, PostgreSQL concurrency/idempotency, and migration metadata. |
| `uv run ruff check` on all changed Python paths | passed |
| `uv run ruff format --check` on all changed Python paths | passed |
| Targeted `uv run pyright` on all changed Python paths | `0 errors, 0 warnings, 0 informations` |
| `git diff --check 7314c1febeb4ccc50bba585f513921d27711f59a` | passed |
| Untracked-file whitespace check | passed |
| `uv run alembic heads --verbose` | one head: `20260720_0044` |

## Exact changed files relative to baseline

```text
.codex/delivery/evidence/ROEHUB-AUTHZ-DELEGATION-CORE-2026-07-20.md
.codex/tickets/2026-07-20-roehub-authz-delegation-core.md
alembic/versions/20260720_0044_identity_delegated_capabilities_v1.py
src/trading/contexts/identity/adapters/outbound/persistence/in_memory/delegation_repository.py
src/trading/contexts/identity/adapters/outbound/persistence/postgres/delegation_repository.py
src/trading/contexts/identity/application/delegation/__init__.py
src/trading/contexts/identity/application/delegation/models.py
src/trading/contexts/identity/application/delegation/service.py
src/trading/contexts/identity/application/ports/delegation_repository.py
tests/unit/identity/delegation/test_delegated_capability_service.py
tests/unit/identity/delegation/test_delegation_migration_contract.py
tests/unit/identity/delegation/test_postgres_delegation_repository.py
```

## Cold review

Cold self-review and an independent read-only security review were completed.
The independent reviewer initially found three issues: arbitrary resource-scope
audit data, a role-matrix expansion, and ambiguous expired-grant renewal. The
implementation and focused coverage were corrected, then independently
re-reviewed with verdict `Release`.

## Residual risk and next boundary

The boundary has no inbound API or browser integration by design. A later,
separately authorized integration ticket must supply server-resolved
recent-auth and organization/resource facts; it must not expose client-supplied
roles or arbitrary scopes. No Penpot or Web UI implementation is authorized by
this acceptance.
