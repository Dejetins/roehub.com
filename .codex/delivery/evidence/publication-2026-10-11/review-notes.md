# Publication review notes — 2026-10-11

Scope: all outstanding repository work and local commit d06ace34, delivered through a temporary technical branch. Scratch `.cache/` snapshots and ignored installation data remain local.

Independent release/security review sampled browser-device owner scoping and CSRF, controller request signatures, storage fencing and idempotency, shared provider admission and asynchronous recovery. Fourteen focused reviewer tests passed; no additional confirmed source blocker was found. This is not exhaustive real-infrastructure or browser proof.

## Dispositions

- Sanitized 12 `password=` values in `market-data-resource-efficiency-v1/reports/S03/real-pg-provider-02.txt`. Original SHA-256: `a329f3bcabc73f837b8c192762ae645fffdae5dd8ecc0ef5abb7e58b4109120b`; published sanitized SHA-256: `5d0765c417873f1703a2783e96191b9fddcfe967ed659f492a16b1b06c674f00`. Existing historical checksums describe the original run; they must not be interpreted as hashes of this sanitized publication copy. Original private evidence remains outside Git.
- Storage recent-auth removal was explicitly requested by the owner in thread `01a1222f-5816-72b0-8d04-8c2f27634e94`, turn `01a12286-2712-72c3-9f50-1c82670d8ecb`, following discussion of the ten-minute storage gate. Current runtime contract now reflects that decision. Authenticated session, installation ownership and Origin checks remain required. Other recent-auth boundaries are unchanged. This intentionally relaxes the storage mutation gate (`compatible-change` for API consumers).
- Completed pack updater bytes are retained; historical E501/I001 layout exceptions follow the existing repository convention. The artifact pack test removes one unused local binding while preserving its fixture call. No ledger transition or acceptance state was changed.
- Full Python type checks required explicit assertions in test fixtures, SQL-file bytes and accurate mocks. Runtime inventory now skips known binary image/font assets and has a regression test. Generated documentation/map/inventory files were refreshed.

Navigator typecheck and 369 tests passed locally. Live provider acceptance from the earlier activation remains bounded by intermittent HTTP451 responses; this publication does not assert new provider or production deployment proof. GitHub checks and final merge identity are recorded on the delivery PR.
