# Stage 07: Paper full branch coverage

Статус: `in_progress`

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

TBD.

## File Manifest

TBD after implementation and validation.

## Blockers

TBD.

## Handoff

TBD.
