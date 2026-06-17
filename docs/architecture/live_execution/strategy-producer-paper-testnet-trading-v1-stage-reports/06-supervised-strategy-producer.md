# Stage 06: Supervised Strategy Producer

Статус: `in_progress`

## Pre-Start

User required before start: nothing

Stage `05` проверен в ledger: `accepted`; следующий stage разрешен.

## Scope

Stage `06` переиспользует существующий `apps/worker/strategy_live_runner` как supervised strategy producer. Новый app/process не создается. Цель stage: fail-closed producer control для `paper`/`testnet`, per-user/per-strategy allowlist, admin switch, launchd/Monit supervision и Prometheus/health/readiness evidence.

## Concrete Planned File List Before Editing

| File | Planned action | Reason |
|---|---:|---|
| `src/trading/contexts/strategy/adapters/outbound/config/strategy_runtime_config.py` | modify | Добавить source-of-truth config `strategy.producer` для admin switch, `paper`/`testnet` mode allowlist и user/strategy allowlists. |
| `src/trading/contexts/strategy/adapters/outbound/config/live_runner_runtime_config.py` | modify | Пробросить producer config из `strategy.yaml` и legacy `strategy_live_runner.yaml` loader. |
| `configs/dev/strategy.yaml` | modify | Зафиксировать dev fail-closed producer defaults и отдельный metrics port. |
| `configs/test/strategy.yaml` | modify | Зафиксировать test fail-closed producer defaults и отдельный metrics port. |
| `configs/prod/strategy.yaml` | modify | Зафиксировать prod fail-closed producer defaults и отдельный metrics port, без mainnet/live mode. |
| `apps/worker/strategy_live_runner/wiring/modules/strategy_live_runner.py` | modify | Добавить supervised HTTP `/metrics` + `/health/*`, producer guard metrics, allowlist wrapper and bounded metrics labels. |
| `src/trading/contexts/strategy/application/services/live_runner.py` | modify | Расширить iteration report blocked/source-event counters and wire per-signal producer reason callbacks without direct exchange access. |
| `tests/unit/contexts/strategy/application/test_strategy_live_runner.py` | modify | Unit proof: disabled switch and missing allowlist block producer calls; allowlisted `paper`/`testnet` can call the execution producer; `live` is blocked. |
| `tests/unit/contexts/strategy/adapters/test_strategy_live_runner_runtime_config.py` | modify | Unit proof: runtime config parses producer defaults, env allowlists and rejects `live`/mainnet producer modes. |
| `tests/unit/contexts/strategy/adapters/test_strategy_live_runner_wiring_module.py` | modify | Unit proof: producer guard blocks/allows with bounded reasons and no high-cardinality labels. |
| `tests/unit/apps/test_strategy_live_runner_main.py` | modify | Entrypoint smoke for config-driven metrics port after port change if needed. |
| `infra/macos/launchd/com.roehub.strategy-live-runner.plist` | create | Mac Studio launchd service definition for supervised strategy producer. |
| `infra/scripts/monit/roehub-strategy-live-runner.monitrc` | create | Monit control/check contract for the strategy producer service. |
| `infra/macos/prometheus/prometheus.prod.yml` | modify | Add strategy producer scrape target on dedicated port. |
| `infra/macos/prometheus/prometheus.test.yml` | modify if matching prod topology requires it | Keep test Prometheus topology aligned. |
| `infra/macos/prometheus/rules/strategy-producer.rules.yml` | create | Add bounded producer alert rules for down/disabled/blocker evidence. |
| `docs/runbooks/strategy-live-worker.md` | modify | Replace archived note with Stage 06 operational runbook. |
| `docs/architecture/live_execution/strategy-producer-paper-testnet-trading-v1-stage-reports/06-supervised-strategy-producer.md` | modify | Stage evidence report. |
| `docs/architecture/live_execution/strategy-producer-paper-testnet-trading-v1-stage-reports/strategy-producer-paper-testnet-trading-v1-stage-ledger.md` | modify after validation | Stage ledger handoff. |
| `docs/architecture/README.md` | modify if generated index requires it | Docs index for new stage report/runbook/rules references. |

## File Manifest

| Created | Modified | Deleted | Reason | Contract impact |
|---|---|---|---|---|
| this report | - | none | Record Stage `06` scope, pre-start, file list and evidence. | none: docs/evidence only |
| `infra/macos/launchd/com.roehub.strategy-live-runner.plist`; `infra/scripts/monit/roehub-strategy-live-runner.monitrc`; `infra/macos/prometheus/rules/strategy-producer.rules.yml` | `infra/macos/prometheus/prometheus.prod.yml`; `scripts/macos/bootstrap_native_prod.sh`; `scripts/macos/reload_launchd_services.sh`; `.github/workflows/deploy-backend.yml`; `tests/unit/infra/test_monitoring_assets.py`; `tests/unit/infra/test_native_service_assets.py` | none | Add production launchd/Monit/Prometheus supervision for the reused strategy runner on dedicated port `9207`; ensure deploy installs/reloads/smokes the service. | compatible-change: additive service surface for existing worker; rollback by stopping launchd/Monit service and removing scrape target/rules. |
| none | `configs/dev/strategy.yaml`; `configs/test/strategy.yaml`; `configs/prod/strategy.yaml`; `src/trading/contexts/strategy/adapters/outbound/config/strategy_runtime_config.py`; `src/trading/contexts/strategy/adapters/outbound/config/live_runner_runtime_config.py`; `src/trading/contexts/strategy/adapters/outbound/config/__init__.py`; `src/trading/contexts/strategy/adapters/outbound/__init__.py`; `src/trading/contexts/strategy/adapters/__init__.py`; `tests/unit/contexts/strategy/adapters/test_strategy_runtime_config.py`; `tests/unit/contexts/strategy/adapters/test_strategy_live_runner_runtime_config.py`; `tests/unit/apps/api/test_strategy_wiring_module.py` | none | Add fail-closed `strategy.producer` admin switch, `paper`/`testnet` mode allowlist and per-user/per-strategy allowlists; move strategy producer metrics to non-conflicting port `9207`. | compatible-change: additive config/env schema; default remains disabled and no mainnet/live mode is accepted. |
| none | `apps/worker/strategy_live_runner/wiring/modules/strategy_live_runner.py`; `tests/unit/contexts/strategy/adapters/test_strategy_live_runner_wiring_module.py` | none | Add guarded execution producer wrapper, `/health/live`, `/health/ready`, `/metrics` HTTP surface and bounded producer metrics for source events, blocked skips, last cycle, lag and readiness. | compatible-change: existing execution producer boundary is reused; source-event creation is stricter and fail-closed. |
| none | `docs/runbooks/strategy-live-worker.md` | none | Replace archived note with active Stage `06` service control and alert runbook. | none: docs/runbook only |
| none | `docs/architecture/README.md` | none | Generated docs index entry for this Stage `06` report. | none: docs index only |

Files outside prompt expected paths: `.github/workflows/deploy-backend.yml`, `scripts/macos/bootstrap_native_prod.sh`, `scripts/macos/reload_launchd_services.sh`, `tests/unit/infra/*` are touched because launchd/Monit/Prometheus assets are only effective if the native deploy installs, reloads and locally verifies them. `src/trading/contexts/strategy/adapters/__init__.py` and related export files are touched to keep the new runtime config type available through the existing adapter export pattern. `docs/architecture/README.md` also currently contains unrelated RL Stage `02A` index changes from the shared worktree; those are not Stage `06` scope and must not be included in Stage `06` delivery except for a Stage `06` index hunk.

## Contract Impact

Initial classification before implementation:

| Dimension | Classification | Note |
|---|---|---|
| Public API contract | `none` | No browser/API DTO changes planned. |
| Port contract | `compatible-change` | Strategy execution producer path gains a guard/wrapper before calling the existing port. |
| DTO schema | `none` | No request/response DTO shape change planned. |
| Persisted schema | `none` | No migration planned; evidence uses existing source-event/signal/run tables. |
| Config/env schema | `compatible-change` | Additive `strategy.producer` config and env overrides; default remains fail-closed. |
| Service-call semantics | `compatible-change` | Strategy producer can only call live_execution when admin switch, mode, and allowlist checks pass. |
| External side effects | `compatible-change` | Source-event creation becomes explicitly guarded; no exchange SDK/direct credentials. |
| Metrics/audit/report semantics | `compatible-change` | New bounded producer health/readiness/metrics labels. |
| Ops/runtime | `compatible-change` | Adds launchd/Monit/Prometheus supervision for existing runner process on a dedicated metrics port. |

## Evidence

### Local gates

| Command | Result |
|---|---|
| `plutil -lint infra/macos/launchd/com.roehub.strategy-live-runner.plist` | passed |
| `python -m compileall -q apps/worker/strategy_live_runner src/trading/contexts/strategy` | passed |
| `uv run pytest -q tests/unit/contexts/strategy/adapters/test_strategy_runtime_config.py tests/unit/contexts/strategy/adapters/test_strategy_live_runner_runtime_config.py tests/unit/contexts/strategy/adapters/test_strategy_live_runner_wiring_module.py tests/unit/apps/api/test_strategy_wiring_module.py tests/unit/infra/test_monitoring_assets.py tests/unit/infra/test_native_service_assets.py` | passed, `34 passed` |
| `uv run pytest -q tests/unit/apps/test_strategy_live_runner_main.py tests/unit/contexts/strategy/application/test_strategy_live_runner.py` | passed, `15 passed` |
| `uv run ruff check apps/worker/strategy_live_runner src/trading/contexts/strategy tests` | passed after import-order repair |
| `uv run pyright apps/worker/strategy_live_runner src/trading/contexts/strategy tests` | passed, `0 errors` |
| `uv run pytest -q tests/unit/apps tests/unit/contexts/strategy` | passed, `398 passed, 3 warnings` |
| `python -m tools.docs.generate_docs_index --check` | passed after docs index regeneration |
| `uv run ruff check .` | passed |
| `uv run pyright` | passed, `0 errors` |
| `uv run pytest -q -ra` | passed, `1179 passed, 3 warnings` |

### Local fail-closed proof

| Scenario | Evidence | Result |
|---|---|---|
| Producer admin switch disabled | `tests/unit/contexts/strategy/adapters/test_strategy_live_runner_wiring_module.py::test_guarded_strategy_execution_producer_blocks_disabled_switch` | No delegate/source-event call; bounded reason `producer_disabled`. |
| Producer enabled but no user/strategy allowlist | `test_guarded_strategy_execution_producer_blocks_missing_allowlist` | No delegate/source-event call; bounded reason `producer_allowlist_missing`. |
| Paper profile with user allowlist | `test_guarded_strategy_execution_producer_allows_paper_user_allowlist` | Existing execution producer port is called; no exchange SDK or credential path added. |
| Live/mainnet-like mode | `test_guarded_strategy_execution_producer_blocks_live_mode_even_when_allow_all`; config rejection tests | No source-event call; `strategy.producer.allowed_modes` rejects anything outside `paper`/`testnet`. |
| Metrics label cardinality | `test_strategy_producer_metrics_do_not_use_user_or_strategy_labels` | Producer metrics expose bounded `mode`, `outcome`, `reason`, `scope` labels; no user/strategy UUID labels. |

## Blockers

Pending before acceptance:

- direct main delivery / `origin/main` evidence;
- deploy/backend workflow or equivalent Mac Studio host sync;
- target-runtime launchd loaded/control proof for `com.roehub.strategy-live-runner`;
- Monit proof for `roehub_strategy_live_runner`;
- Prometheus scrape proof for `job="strategy-producer"`;
- target-runtime `/health/live`, `/health/ready`, `/metrics` evidence;
- controlled runtime disabled-switch, missing-allowlist and allowlisted paper/testnet source-event proof.

## Handoff

Do not start Stage `07` until Stage `06` is accepted with main delivery and Mac Studio runtime proof. Keep producer defaults fail-closed: admin switch disabled, `paper`/`testnet` modes only, empty allowlists until a scoped runtime smoke explicitly sets user/strategy UUIDs.
