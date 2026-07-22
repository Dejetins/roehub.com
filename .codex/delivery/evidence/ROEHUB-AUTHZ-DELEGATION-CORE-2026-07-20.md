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

## 2026-07-22 — clean-PostgreSQL migration bootstrap repair

### Root cause and decision

CI `29911678454` created an empty PostgreSQL database and called
`apps.migrations.main` directly. That runs Alembic `head`, so published revision
`20260720_0044` attempted to reference `identity_organizations` before SQL
`0011_identity_organizations_rbac_audit_v1.sql` had created it. The same failure
was reproduced locally on a passwordless one-time PostgreSQL 16 container with
an explicit loopback DSN; the process failed with
`psycopg.errors.UndefinedTable: relation "identity_organizations" does not exist`.

The accepted decision is
`docs/architecture/identity/identity-migration-channels-delegation-checkpoint-v1.md`:
SQL `0001..0009` → Alembic `20260711_0043` → SQL `0010..0011` → Alembic `head`
→ SQL `0012..0022`. SQL `0011` exclusively owns `identity_organizations` and
`identity_memberships`; Alembic `20260720_0044` exclusively owns
`identity_delegated_capabilities` and its indexes. The published `0044` source,
`revision`, and `down_revision` remain byte-identical to
`b159bc0e7ac3ac14f53cd6f603574dcbaa08bf52`.

### Changed boundary

- `apps.migrations.main` now accepts `--revision` and defaults to `head`.
- `run_dev_db_bootstrap` executes the two Alembic phases around SQL `0010/0011`
  and then runs the remaining identity SQL files in numeric order.
- CI calls `apps.migrations.bootstrap_main` with both identity and Alembic DSNs
  pointing to its single temporary PostgreSQL service.
- Unit regression coverage asserts the exact phase sequence and explicit
  checkpoint forwarding.

### PostgreSQL proof

1. Clean bootstrap through `apps.migrations.bootstrap_main`: passed. The final
   Alembic revision was `20260720_0044`; present tables were
   `identity_organizations`, `identity_memberships`, and
   `identity_delegated_capabilities`; present indexes were
   `idx_identity_delegated_capabilities_evaluation` and
   `idx_identity_delegated_capabilities_one_unrevoked`.
2. Existing-schema upgrade: passed from SQL through `0011` and Alembic
   `20260711_0043`. `identity_organizations` retained the same PostgreSQL OID;
   one organization and one membership row remained after upgrade to `0044`.
3. Rollback/re-upgrade: passed with the real Alembic sequence
   `20260711_0043 → 20260720_0044 → 20260711_0043 → 20260720_0044`.
   The delegation table was absent after rollback and restored after the final
   upgrade; the existing identity rows remained intact.

### Local verification

| Check | Result |
|---|---|
| `uv run pytest tests/unit/apps/migrations tests/unit/identity/delegation -q` | `85 passed, 1 skipped` (the optional delegation PostgreSQL fixture has no DSN by default). |
| Delegation PostgreSQL fixture with the explicit disposable DSN | Fails before exercising delegation logic because its existing seed inserts `identity_users` without mandatory `created_at`; that test file is outside this repair's allowed paths. |
| `uv run ruff check .` | passed. |
| `uv run ruff format --check .` | repository baseline failure: 554 unrelated files would be reformatted; the four changed Python files pass the same check. |
| Targeted `uv run pyright` | `0 errors, 0 warnings, 0 informations`. |
| `python -m tools.docs.generate_docs_index --check` | passed. |
| `python -m tools.docs.generate_project_map --check` | passed after its deterministic derived refresh. |
| `git diff --check` | passed. |

### Compatibility and residual risk

| Surface | Classification | Result |
|---|---|---|
| Bootstrap and CI migration execution | `compatible-change` | Empty databases now follow the real ordered bootstrap path. |
| Existing SQL-`0011` / Alembic-`0043` installations | `compatible-change` | Upgrade, rollback, and re-upgrade preserve identity schema and data. |
| Published Alembic `0044` and product/API/browser behavior | `none` | No published revision, route, envelope, or product surface changed. |

Residual risk is limited to existing repository formatting drift and the
pre-existing delegation PostgreSQL test seed incompatibility described above.
No push, GitHub Actions rerun, release, deployment, Linear mutation, or other
external publication was performed.

### Independent review

An independent no-working-tree-write review completed after implementation with
verdict `ready_for_next_gate` and no `P0`–`P3` findings. It separately verified
the immutable `0044` SHA-256 and lineage, exact clean-bootstrap order, final
tables/indexes, existing SQL-`0011`/Alembic-`0043` data and OID preservation,
the real downgrade/re-upgrade cycle, CI parity with `bootstrap_main`, and
absence of duplicate table ownership. It independently reproduced the optional
delegation fixture failure before delegation logic at its seed insert without
`created_at`, confirming that it is outside this repair.
