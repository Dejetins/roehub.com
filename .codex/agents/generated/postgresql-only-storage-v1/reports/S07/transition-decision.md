# S07 transition decision packet

The S07 executor does not own authoring changes to pending stage contracts or the
exclusive updater. Product cleanup is implemented; do not claim S08.

## Confirmed structural input defect

`08-functional-recovery-verification.md` and the matching S08 ledger contract
contain these entry_inputs:

- `../../../../tests/real_infra/market_data`
- `../../../../tests/real_infra/backtest`

Both are existing directories. The bundled validator `Pack.entry` requires every
entry input to be a file (`Path.is_file`). The canonical updater's accept branch
preflights the successor before creating S07's receipt or updating the ledger.
Therefore accepting S07 fails at `Missing live entry input: S08`. This is unrelated
to S07 product tests and cannot be bypassed with a manual status/receipt edit.

## Narrow candidate and orchestration handoff

Prepare a prompt-pack metadata repair, without executing S08: replace the two
pending S08 entry inputs, in prompt and ledger contract together, with existing
representative files:

- `../../../../tests/real_infra/market_data/test_transactional_ingestion_postgres.py`
- `../../../../tests/real_infra/backtest/test_artifact_dispatch.py`

Keep directory context routes, requirements, current claim, stage statuses,
accepted contracts/receipts and all execution history unchanged. Have the pack
author prepare a reviewed candidate and the exclusive runner apply the authorized
metadata migration under its lock/CAS contract. Do not edit the active JSON by hand,
weaken `is_file`, alter prior receipts, or add an implicit parallel updater.

Resume condition: recorded authorization and a validated, exclusively applied
pending-S08 contract repair, with the S07 claim retained. Then refresh affected
navigation, rerun entry/receipt checks and accept S07 through the canonical updater.
This does not authorize running S08.

The orchestration owner has taken the metadata repair. No new product decision is
requested from the user. Exact candidate is `s08-entry-candidate.json`; the S07
executor leaves active ledger/updater unchanged while this independent repair runs.

## Resolved

The orchestration owner completed the mechanical S08/S09 entry-locator repair.
S07 focused sanity review compared exact reversed prompt bytes and complete
contracts, verified six accepted receipts and unchanged updater, and passed
projected S08 entry. See transition-preflight.log and root-owned
../pending-entry-input-repair-validation.json. No user decision is pending.
