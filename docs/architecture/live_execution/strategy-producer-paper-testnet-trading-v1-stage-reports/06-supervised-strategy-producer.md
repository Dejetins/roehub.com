# Stage 06: Supervised Strategy Producer

Статус: `accepted`

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

### Main delivery and deploy

| Boundary | Evidence | Result |
|---|---|---|
| Direct main delivery | Commit `d5bac7ec6958d695d095dd3d5365a6822c47bc70` pushed to `origin/main`. | passed |
| CI | GitHub Actions CI run `27716058678` for `d5bac7ec6958d695d095dd3d5365a6822c47bc70`. | completed `success` |
| Deploy Backend | GitHub Actions run `27716344212` for `d5bac7ec6958d695d095dd3d5365a6822c47bc70`. | completed `success` |
| Deploy Web | GitHub Actions run `27716343982` for `d5bac7ec6958d695d095dd3d5365a6822c47bc70`. | completed `success` |
| Publish App Image | GitHub Actions run `27716343745` for `d5bac7ec6958d695d095dd3d5365a6822c47bc70`. | completed `success` |
| Mac Studio checkout sync | `git -C /Users/daniildegtyarev/Projects/roehub.com fetch origin main && git -C /Users/daniildegtyarev/Projects/roehub.com merge --ff-only origin/main && git -C /Users/daniildegtyarev/Projects/roehub.com rev-parse HEAD` | fast-forwarded to `d5bac7ec6958d695d095dd3d5365a6822c47bc70` |

### Mac Studio runtime proof

| Boundary | Evidence | Result |
|---|---|---|
| Runtime files | `grep -R "strategy_producer_admin_enabled\\|GuardedStrategyExecutionProducer\\|9207" /opt/roehub/app/...` | `/opt/roehub/app` contains Stage `06` producer guard and port `9207` runtime files. |
| launchd loaded | `launchctl print gui/$(id -u)/com.roehub.strategy-live-runner` | `state = running`; launchd command uses `/opt/roehub/app/configs/prod/strategy.yaml` and `--metrics-port 9207`. |
| Health live/ready | `curl -fsS http://127.0.0.1:9207/health/live`; `curl -fsS http://127.0.0.1:9207/health/ready` | Both returned JSON; producer defaults are `enabled=false`, `allow_all=false`, `allowed_modes=["paper","testnet"]`, empty allowlists, and `ready=true`. |
| Metrics | `curl -fsS http://127.0.0.1:9207/metrics` | Exposed `strategy_producer_admin_enabled 0.0`, `strategy_producer_allowed_mode{mode="paper"} 1.0`, `strategy_producer_allowed_mode{mode="testnet"} 1.0`, `strategy_producer_ready 1.0`, and runner iteration/error counters. |
| Monit | `/opt/homebrew/bin/monit -c /opt/homebrew/etc/monitrc summary \| grep -E "roehub_strategy_live_runner|Process"` | `roehub_strategy_live_runner OK Process`. |
| Prometheus | `curl -fsS "http://127.0.0.1:9090/api/v1/query?query=up%7Bjob%3D%22strategy-producer%22%7D"` | `up{job="strategy-producer",instance="127.0.0.1:9207"} = 1`. |
| Production smoke | `cd /opt/roehub/app && bash scripts/macos/smoke_prod.sh` | exited `0`; service inventory includes `com.roehub.strategy-live-runner`. |
| Stop/start drill | `/opt/homebrew/etc/monit.d/scripts/launchctl_service_control.sh stop/start com.roehub.strategy-live-runner ...` plus readiness curl | After stop, `/health/ready` was down; after start, `/health/ready` returned `ready=true`; final launchd state `running`, pid `23087`. |

### Controlled runtime source-event proof

| Scenario | Evidence | Result |
|---|---|---|
| Disabled admin switch | Mac Studio `/opt/roehub/app/.venv/bin/python -` probe using `GuardedStrategyExecutionProducer` with `enabled=False`. | `blocked_reasons=producer_disabled`; `blocked_delegate_calls=0`; `disabled_rows=0`. |
| Enabled producer without allowlist | Same probe with `enabled=True`, empty user/strategy allowlists. | `blocked_reasons=producer_allowlist_missing`; `blocked_delegate_calls=0`; `missing_allowlist_rows=0`. |
| Allowlisted paper source event | Same probe with user allowlist and real `LiveExecutionStrategySignalProducer` + `PostgresExecutionIntentRepository`. | One `execution_source_events` row created for synthetic signal `00000000-0000-0000-0000-000006000913`: `source_type=strategy_signal`, `outcome=recorded`, `outcome_reason=source_event_recorded`, `mode=paper`, `action=open`. |
| No exchange/order side effect | Read-only SQL check on the synthetic source event. | `allowed_source_event_intent_id=None`; `allowed_source_event_intent_rows=0`. |

## Blockers

None for Stage `06`.

## Handoff

Stage `07` may start. Keep producer defaults fail-closed: admin switch disabled, `paper`/`testnet` modes only, empty allowlists until a scoped runtime smoke explicitly sets user/strategy UUIDs. Stage `06` created one synthetic source-event-only runtime probe row with signal id `00000000-0000-0000-0000-000006000913`; it has no linked execution intent or order side effect.
