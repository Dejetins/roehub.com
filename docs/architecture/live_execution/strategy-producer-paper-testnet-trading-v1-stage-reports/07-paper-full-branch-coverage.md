# Stage 07: Paper full branch coverage

Статус: `accepted`

## Pre-Start

User required before start: nothing

Stage `06` проверен в `strategy-producer-paper-testnet-trading-v1-stage-ledger.md` до implementation edits: статус `accepted`, `Next stage allowed = yes`, активных blockers нет.

## Scope

Stage `07` должен провести все paper rows из Stage `03` matrix через безопасный producer/API/runtime path с allocation `$50` на strategy, сохранить source event, no-dispatch intent, paper order/fill/accounting и durable coverage result для каждой строки. Paper mode не должен создавать Redis execution dispatch, дергать exchange credential decrypt или открывать mainnet path.

## Concrete Planned File List Before Editing

Ожидаемые broad paths из prompt сужены до конкретных файлов до implementation edits:

| File | Planned action | Reason |
|---|---:|---|
| `src/trading/contexts/live_execution/domain/risk_gate.py` | modify | Добавить явный `paper_no_exchange_submit` no-dispatch risk outcome для paper intents без account/exchange dispatch requirements. |
| `src/trading/contexts/live_execution/domain/paper_accounting.py` | modify | Уточнить paper short accounting shape и зафиксировать funding/fee completeness в snapshot. |
| `src/trading/contexts/live_execution/domain/paper_coverage.py` | create | Domain record for per-matrix-row paper coverage result. |
| `src/trading/contexts/live_execution/application/ports/paper_coverage_repository.py` | create | Repository port for durable paper coverage results. |
| `src/trading/contexts/live_execution/application/use_cases/paper_accounting.py` | modify | Add virtual paper capital reservation path and correct short position accounting. |
| `src/trading/contexts/live_execution/application/use_cases/paper_coverage.py` | create | Application service to record/reload per-row paper coverage results. |
| `src/trading/contexts/live_execution/adapters/outbound/persistence/in_memory/paper_coverage_repository.py` | create | Deterministic unit-test adapter for coverage rows. |
| `src/trading/contexts/live_execution/adapters/outbound/persistence/postgres/paper_coverage_repository.py` | create | Durable Postgres adapter for coverage rows. |
| live_execution package `__init__.py` export files | modify | Export new paper coverage domain/use-case/port/adapters through existing boundaries. |
| `src/trading/contexts/strategy/domain/entities/strategy_signal.py` | modify | Allow a bounded, non-sensitive expected paper order payload for Stage `07` signals. |
| `src/trading/contexts/strategy/application/ports/capital_reservation.py` | modify | Add optional virtual paper reservation contract used by API/runner without exchange credential dependency. |
| `src/trading/contexts/strategy/application/use_cases/run_strategy.py` | modify | Reserve `$50` virtual paper capital for paper profiles that intentionally have no exchange connection. |
| `src/trading/contexts/strategy/application/services/live_runner.py` | modify | Attach bounded expected paper order metadata to persisted paper signals. |
| `src/trading/contexts/strategy/adapters/outbound/acl/live_execution_producer.py` | modify | Make allowlisted paper signals create source events plus rejected no-dispatch intents. |
| `apps/api/dto/ui_execution.py` | modify | Add optional risk-context flag for paper no-dispatch semantics. |
| `alembic/versions/20260617_0033_strategy_paper_scenario_coverage_v1.py` | create | Additive coverage result table for Stage `07`. |
| `tests/unit/contexts/live_execution/test_paper_accounting_service.py` | modify | Cover virtual paper reservation, short accounting and fee/funding completeness. |
| `tests/unit/contexts/live_execution/test_execution_ingress_service.py` | modify | Cover paper no-dispatch risk intent semantics. |
| `tests/unit/contexts/live_execution/test_paper_coverage_service.py` | create | Cover durable coverage result recording/idempotency. |
| `tests/unit/contexts/strategy/application/test_strategy_live_runner.py` | modify | Cover expected paper order payload from live runner. |
| `tests/unit/contexts/strategy/adapters/test_strategy_live_runner_wiring_module.py` | modify | Cover producer ACL paper source-event + intent behavior without dispatch. |
| `tests/unit/apps/migrations/test_strategy_paper_scenario_coverage_sql.py` | create | SQL contract for additive coverage migration. |
| `docs/architecture/live_execution/strategy-producer-paper-testnet-trading-v1-stage-reports/07-paper-full-branch-coverage.md` | modify | Stage report, matrix evidence, file manifest, delivery/runtime proof. |
| `docs/architecture/live_execution/strategy-producer-paper-testnet-trading-v1-stage-reports/strategy-producer-paper-testnet-trading-v1-stage-ledger.md` | modify after validation | Stage status/evidence/handoff. |
| `docs/architecture/README.md` | modify/check if generated index requires it | Docs index after new Stage `07` report; existing unrelated RL Stage `02A` index changes predate this stage and must not be reverted. |

## Initial Contract Impact

| Dimension | Classification | Note |
|---|---|---|
| Public API contract | `compatible-change` | Optional risk-context field only; existing payloads remain valid. |
| Port contract | `compatible-change` | Additive paper coverage repository and virtual paper reservation contract. |
| DTO schema | `compatible-change` | Optional field added to UI execution risk DTO. |
| Persisted schema | `compatible-change` | Additive paper coverage result table; no existing table rewrite. |
| Config schema | `none` | No new required env/config planned. |
| Request hash / cache / identity | `compatible-change` | New coverage-result identity is per `scenario_key`/strategy run; existing launch/request hashes unchanged. |
| Service-call semantics | `compatible-change` | Paper signals produce no-dispatch intents; no exchange SDK/custody call. |
| External side effects | `compatible-change` | Paper rows remain DB-only and Redis execution dispatch must stay absent. |
| Logs / metrics / audit / report | `compatible-change` | Adds no-dispatch risk audit and coverage evidence rows with non-sensitive identifiers. |
| Browser-visible behavior | `compatible-change` | `/strategies` should show paper accounting/outcome data that already exists in the dashboard model. |

## Evidence

Stage `07` accepted after local gates, direct `main` delivery, CI/deploy, Mac Studio sync/smoke, production DB migration proof, controlled paper runtime coverage, browser proof, and cleanup.

### Implementation Summary

- Added bounded `paper_no_exchange_submit` semantics to strategy signal risk context and API DTOs. Paper signals now produce execution source events plus rejected no-dispatch intents, without Redis execution dispatch and without exchange credential decrypt/account checks.
- Added durable `strategy_paper_scenario_coverage` persistence and application/domain/repository adapters so each matrix paper branch can record coverage.
- Added virtual paper reservation/accounting behavior for `$50` strategy allocation, including short negative positions and fee/funding completeness flags.
- Added bounded expected paper-order metadata to strategy signals and runner/ACL paths.
- Repaired two runtime-found constraints before acceptance:
  - Stage `07` provenance launches now include `launch_request_hash` in source-variant uniqueness, allowing distinct paper launch configs for the same source variant.
  - Production `strategy_signals.expected_order_json` SQL check now allows the bounded paper expected-order payload and rejects broader payloads.

### Local Gates

| Gate | Result |
|---|---|
| `uv run ruff check src/trading/contexts/live_execution src/trading/contexts/strategy apps tests` | passed |
| `uv run pyright src/trading/contexts/live_execution src/trading/contexts/strategy apps tests` | passed, `0 errors` |
| `uv run pytest -q tests/unit/contexts/live_execution tests/unit/contexts/strategy tests/unit/apps` | passed, `461 passed, 3 warnings` |
| `python -m tools.docs.generate_docs_index --check` | passed after docs update |

Focused migration/repair tests also passed for paper coverage SQL, provenance launch-hash uniqueness, and bounded paper expected-order SQL.

### Main Delivery, CI, Deploy, Runtime Sync

| Item | Evidence |
|---|---|
| Direct `main` commits | `b530b664` (`Add strategy paper coverage path`), `b5f08677` (`Allow variant launches per config`), `6ff19ac5` (`Allow paper expected order signals`) |
| CI | `27720372309` success for final runtime commit |
| Deploy Backend | `27720477711` success |
| Deploy Web | `27720477721` success |
| Publish App Image | `27720477693` success |
| Mac Studio checkout | synced to `6ff19ac52c9932b9d796f406d764f09dddf2893f` |
| Mac Studio smoke | `/opt/roehub/app/scripts/macos/smoke_prod.sh` exited `0` |
| Production Alembic | `20260617_0035` |
| Production constraint proof | active check is `strategy_signals_expected_order_stage07_chk`; bounded paper expected-order payload is allowed, broader payloads remain blocked |

### Controlled Paper Coverage Runtime Proof

Runtime proof used synthetic subject `codex:stage07-paper-coverage:0120a6cfb935` and scoped producer allowlist only. Producer was disabled before the run, enabled only for the synthetic user during the run, then restored disabled with empty allowlists.

| Source market | Matrix rows | Paper rows | Paper state | Paper capability |
|---|---:|---:|---|---|
| `spot` | 8 | 4 | `launchable` | `paper_only` |
| `futures` | 8 | 4 | `launchable` | `paper_only` |

| Runtime dimension | Evidence |
|---|---|
| Redis strategy stream | emitted 4 controlled spot messages and 4 controlled futures messages |
| Redis execution dispatch streams | unchanged: `execution.requests.v1=15`, `.retry=0`, `.dlq=0` before and after |
| Source events | `74` total rows after proof; accepted branch rows recorded for Stage `07` paper signals |
| No-dispatch intents | `36` total rejected paper intents; `covered_no_dispatch=8` for the accepted run |
| Paper orders/fills/accounting | `36` paper orders, `36` fills, `36` accounting rows total after proof |
| Durable coverage | `coverage=8`, `covered_no_dispatch=8` |
| Stop cleanup | all 8 controlled runs stopped; final `active_runs=0` |
| Session cleanup | temporary browser/API session revoked; final `active_sessions=0` |

The final accepted run covered open/close paper branches for both source market types without exchange dispatch side effects. Earlier failed synthetic probes were cleaned up where possible and treated as validation bugs, not acceptance evidence.

### Browser Proof

| Check | Evidence |
|---|---|
| URL | `https://roehub.com/strategies?strategy_id=fee7a2db-35b9-40c2-b724-6279982ecd86` |
| Page/API responses | page `200`, CSS `200`, JS `200`, dashboard API `200` |
| Console/network | `0` console messages, `0` failed requests |
| DOM checks | `BTCUSDT`, `paper`, and paper outcome text present |
| Dashboard state | `paper_accounting_state=ready` |
| Screenshot | `output/playwright/stage07-paper-coverage-strategies.png` |
| QA JSON | `output/playwright/stage07-paper-coverage-strategies.json` |

Visual inspection showed the selected paper strategy stopped cleanly, `ready: paper_no_exchange_submit`, live profile `paper`, latest open/close signals with expected paper metadata, saved strategy list, and no obvious UI overlap.

## File Manifest

| Action | File | Reason | Contract impact |
|---|---|---|---|
| Modified | `apps/api/dto/ui_execution.py` | Add optional paper no-dispatch risk-context flag. | `compatible-change` DTO additive field |
| Modified | `src/trading/contexts/live_execution/domain/risk_gate.py` | Reject paper no-exchange-submit before account/decrypt checks. | `compatible-change` service-call semantics |
| Modified | `src/trading/contexts/live_execution/domain/paper_accounting.py` | Add short accounting and fee/funding completeness behavior. | `compatible-change` paper accounting semantics |
| Created | `src/trading/contexts/live_execution/domain/paper_coverage.py` | Durable domain record for per-scenario paper coverage. | `compatible-change` additive model |
| Modified | `src/trading/contexts/live_execution/domain/__init__.py` | Export paper coverage domain. | `none` external runtime |
| Modified | `src/trading/contexts/live_execution/application/__init__.py` | Export new application path. | `none` external runtime |
| Modified | `src/trading/contexts/live_execution/application/ports/__init__.py` | Export paper coverage repository port. | `compatible-change` additive port |
| Created | `src/trading/contexts/live_execution/application/ports/paper_coverage_repository.py` | Repository contract for durable coverage rows. | `compatible-change` additive port |
| Modified | `src/trading/contexts/live_execution/application/use_cases/__init__.py` | Export paper coverage use case. | `none` external runtime |
| Modified | `src/trading/contexts/live_execution/application/use_cases/paper_accounting.py` | Add virtual reservation/accounting path for strategy paper runs. | `compatible-change` paper-only behavior |
| Created | `src/trading/contexts/live_execution/application/use_cases/paper_coverage.py` | Record/reload per-row paper coverage results. | `compatible-change` additive use case |
| Modified | `src/trading/contexts/live_execution/adapters/outbound/persistence/__init__.py` | Export coverage adapters. | `none` external runtime |
| Modified | `src/trading/contexts/live_execution/adapters/outbound/persistence/in_memory/__init__.py` | Export in-memory coverage repository. | `none` external runtime |
| Created | `src/trading/contexts/live_execution/adapters/outbound/persistence/in_memory/paper_coverage_repository.py` | Deterministic test adapter. | `none` production runtime |
| Modified | `src/trading/contexts/live_execution/adapters/outbound/persistence/postgres/__init__.py` | Export Postgres coverage repository. | `none` external runtime |
| Created | `src/trading/contexts/live_execution/adapters/outbound/persistence/postgres/paper_coverage_repository.py` | Durable Postgres coverage adapter. | `compatible-change` additive persistence |
| Modified | `src/trading/contexts/strategy/domain/entities/strategy_signal.py` | Allow bounded expected paper order metadata. | `compatible-change` bounded schema |
| Modified | `src/trading/contexts/strategy/application/ports/capital_reservation.py` | Add virtual paper reservation contract. | `compatible-change` additive port behavior |
| Modified | `src/trading/contexts/strategy/application/use_cases/run_strategy.py` | Reserve virtual paper capital for no-exchange paper profiles. | `compatible-change` paper-only behavior |
| Modified | `src/trading/contexts/strategy/application/services/live_runner.py` | Attach expected paper order metadata to persisted paper signals. | `compatible-change` additive signal metadata |
| Modified | `src/trading/contexts/strategy/adapters/outbound/acl/live_execution_producer.py` | Produce source event plus rejected no-dispatch intent for paper signals. | `compatible-change` paper-only side effect |
| Modified | `src/trading/contexts/strategy/application/use_cases/create_strategy_from_backtest_variant.py` | Include launch-request hash in provenance lookup. | `compatible-change` identity disambiguation |
| Modified | `src/trading/contexts/strategy/application/ports/repositories/strategy_backtest_variant_provenance_repository.py` | Extend provenance lookup signature. | `compatible-change` port update |
| Modified | `src/trading/contexts/strategy/adapters/outbound/persistence/in_memory/strategy_backtest_variant_provenance_repository.py` | Match launch-hash provenance semantics in tests. | `compatible-change` adapter behavior |
| Modified | `src/trading/contexts/strategy/adapters/outbound/persistence/postgres/strategy_backtest_variant_provenance_repository.py` | Match launch-hash provenance semantics in Postgres. | `compatible-change` adapter behavior |
| Created | `alembic/versions/20260617_0033_strategy_paper_scenario_coverage_v1.py` | Add `strategy_paper_scenario_coverage` table. | `compatible-change` additive migration |
| Created | `alembic/versions/20260617_0034_strategy_variant_provenance_launch_hash_unique.py` | Make provenance uniqueness include launch request hash. | `compatible-change` identity migration |
| Created | `alembic/versions/20260617_0035_strategy_signals_expected_paper_order_v1.py` | Allow bounded paper expected-order JSON in SQL check. | `compatible-change` bounded schema relaxation |
| Modified | `tests/unit/contexts/live_execution/test_paper_accounting_service.py` | Cover virtual reservation, shorts, fee/funding completeness. | `none` production runtime |
| Modified | `tests/unit/contexts/live_execution/test_execution_ingress_service.py` | Cover paper no-dispatch risk intent. | `none` production runtime |
| Created | `tests/unit/contexts/live_execution/test_paper_coverage_service.py` | Cover coverage recording/idempotency. | `none` production runtime |
| Modified | `tests/unit/contexts/strategy/application/test_strategy_live_runner.py` | Cover expected paper order payload. | `none` production runtime |
| Modified | `tests/unit/contexts/strategy/adapters/test_strategy_live_runner_wiring_module.py` | Cover producer ACL source-event and no-dispatch intent wiring. | `none` production runtime |
| Modified | `tests/unit/contexts/strategy/application/test_strategy_use_cases.py` | Cover distinct source variant launches by launch config. | `none` production runtime |
| Created | `tests/unit/apps/migrations/test_strategy_paper_scenario_coverage_sql.py` | Assert additive coverage SQL contract. | `none` production runtime |
| Created | `tests/unit/apps/migrations/test_strategy_variant_provenance_launch_hash_sql.py` | Assert launch-hash provenance migration contract. | `none` production runtime |
| Created | `tests/unit/apps/migrations/test_strategy_signals_expected_paper_order_sql.py` | Assert bounded expected paper-order SQL contract. | `none` production runtime |
| Modified | `docs/architecture/live_execution/strategy-producer-paper-testnet-trading-v1-stage-reports/07-paper-full-branch-coverage.md` | Stage report and acceptance evidence. | `none` runtime |
| Modified | `docs/architecture/live_execution/strategy-producer-paper-testnet-trading-v1-stage-reports/strategy-producer-paper-testnet-trading-v1-stage-ledger.md` | Ledger status/evidence/handoff. | `none` runtime |
| Modified | `docs/architecture/README.md` | Docs index status for Stage `07`. | `none` runtime |
| Deleted | none | No files deleted. | `none` |

## Blockers

None. Stage `07` is accepted.

Residual risk: this stage proves the paper no-exchange path only. Real testnet order submission, exchange fill/reconciliation behavior, manual entry/exit, UI journal completeness, rate/load behavior, and 24h supervised soak remain later-stage scope.

## Handoff

Stage `08` may start.

Required carry-forward facts:

- Producer defaults must remain fail-closed: disabled by default, `allow_all=false`, empty allowlists unless a bounded runtime proof explicitly scopes them.
- Paper branch coverage is accepted for no-exchange-submit behavior: all accepted proof rows avoided Redis execution dispatch and exchange credential decrypt.
- Runtime Alembic baseline for the accepted path is `20260617_0035`; do not assume Stage `07` is accepted on older DB constraints.
- The browser/API proof used a temporary session that was revoked; do not reuse synthetic sessions or users from this stage.
- Stage `08` still owns manual entry/exit proof; Stage `07` did not submit real exchange orders and did not satisfy Stage `09`.
