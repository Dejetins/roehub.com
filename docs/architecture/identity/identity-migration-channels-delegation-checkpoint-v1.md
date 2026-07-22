# Identity migration channels and delegation checkpoint v1

Status: accepted repair decision for `ROEHUB-AUTHZ-DELEGATION-CORE-2026-07-20`.

## Decision

Identity SQL migrations own the schema objects introduced in
`migrations/postgres/0001` through `0022`. Alembic owns its independent revision
chain, including the published delegation revision `20260720_0044`. Neither
channel may recreate or redefine tables owned by the other.

The clean-install bootstrap order is fixed:

1. identity SQL `0001` through `0009`;
2. Alembic through checkpoint `20260711_0043`;
3. identity SQL `0010` and `0011`;
4. Alembic through `head` (currently `20260720_0044`);
5. identity SQL `0012` through `0022`, in numeric order.

`0011_identity_organizations_rbac_audit_v1.sql` owns `identity_organizations`
and `identity_memberships`. The delegation migration owns only
`identity_delegated_capabilities` and its two indexes, so it may run only after
the `0011` schema is present. Published Alembic revision metadata and contents,
including `20260720_0044`, remain immutable.

## Compatibility, rollback, and proof

This is a `compatible-change` to the bootstrap orchestration and CI proof path:
it creates no parallel ownership and changes no product/API route. A previously
bootstrapped schema at `20260711_0043` with SQL through `0011` upgrades directly
to `20260720_0044`; its identity tables and rows are preserved. The delegation
revision retains its existing `0043 -> 0044 -> 0043 -> 0044` rollback cycle.

Proof requires the real bootstrap path on a clean disposable PostgreSQL database,
the existing-schema upgrade and rollback cycle, final Alembic revision `0044`,
and presence of the three identity tables plus both delegation indexes.
