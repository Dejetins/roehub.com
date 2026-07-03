---
doc: rl-trading-agent-platform-v1-stage-13-monitor-only-inference-producer
status: accepted
stage: 13
updated_at: 2026-07-03
---

# Stage 13: Monitor-only Inference Producer

Статус: `accepted`.

User required before start: nothing unless a listed prerequisite is not accepted or a required credential/dataset/runtime source is unavailable; never ask for secrets in chat

## 1. Result and stage status

Stage `13` принят после direct-main delivery и `post_main_production_runtime_proof` на `macstudio`.

Delivery/proof boundary:

- Stage `13` code commit: `9e1ee429212cf22c1ae2e2fa0c4cad6f0b13bc12`;
- GitHub `CI` run `28677744947`: passed;
- GitHub `Deploy Backend` run `28677826766`: passed, including backend source sync, reload and smoke;
- `macstudio` authoritative checkout `/Users/daniildegtyarev/Projects/roehub.com`: `HEAD=9e1ee429212cf22c1ae2e2fa0c4cad6f0b13bc12`;
- runtime tree `/opt/roehub/app` loaded Stage `13` code with `stage08m_features=12`, `policy_feature_count=12`, `uses_intercept=true`;
- `bash scripts/macos/smoke_prod.sh` passed after deploy/sync;
- post-main canary artifact root:
  `/opt/roehub/state/rl_trading/evaluation_runs/stage13_monitor_only_inference_producer_v1/stage13_post_main_20260703T184211Z`.

Этот report использует `post_main_production_runtime_proof` только для evidence, собранного после того, как изменённая revision попала в `main`, GitHub `CI` был зелёным, `Deploy Backend` задеплоил/синхронизировал `/opt/roehub/app`, а post-deploy smoke/runtime/browser checks прошли. Более ранний pre-main canary не является acceptance anchor.

Что доказано:

- живое Redis-окно `md.candles.1m.binance:futures:BTCUSDT` прочитано read-only;
- live Redis feature vector и offline/canonical fixture дали parity `max_abs_diff=0.0`;
- принят Stage `08M` manifest `stage08m_a3823cbd01143878_fd7c614b` загружен по sha256 `9e2767ead0b697d0194e501aa7932b44fc1f5d1b180713f1270c81d1c887a69c`;
- monitor-only inference выполнил decision и записал БД source event `ml_agent_decision -> no_intent`;
- создано `0` `ExecutionIntent` и `0` orders;
- Redis dispatch streams не изменились;
- transient metrics/health endpoint отдал `200` и `rl_trading_inference_*` metrics;
- authenticated browser `/strategies` proof увидел post-main `ml_agent_decision` outcome `no_intent` в RL/ML source-event outcomes.

В ходе Stage `13` найден и исправлен реальный compatibility defect: accepted Stage `08M` manifest хранит `feature_count=12`, `scaler_mean=12`, а `weights` имеют `13` строк, потому что первая строка является ridge-classifier intercept/bias. Stage `13` теперь строит Stage `08M`-совместимый `1 x 12` aggregate feature vector и поддерживает intercept row при inference.

Stage `14` разрешён: `current_stage=14`.

## 2. File manifest

Created:

| Path | Reason | Contract impact |
|---|---|---|
| `src/trading/contexts/rl_trading/domain/monitor_only_inference.py` | Stage `13` feature window parsing, Stage `08M` aggregate feature matrix/hash, preloaded supervised policy inference, source-event payload and latency summary. | `compatible-change` additive Python domain surface |
| `src/trading/contexts/rl_trading/adapters/outbound/acl/__init__.py` | ACL package export. | `compatible-change` internal import surface |
| `src/trading/contexts/rl_trading/adapters/outbound/acl/live_execution_producer.py` | Records `ml_agent_decision` source events and marks terminal `no_intent` without creating `ExecutionIntent`. | `compatible-change` additive live-execution integration through existing source-event port |
| `apps/worker/rl_trading_inference/__init__.py` | Worker package marker. | `compatible-change` optional worker package |
| `apps/worker/rl_trading_inference/main/__init__.py` | Worker main package marker. | `compatible-change` optional worker package |
| `apps/worker/rl_trading_inference/main/main.py` | CLI for `status`, `parity`, `canary-once`, and bounded health/metrics server smoke. | `compatible-change` optional worker entrypoint |
| `apps/worker/rl_trading_inference/wiring/__init__.py` | Worker wiring package marker. | `compatible-change` internal import surface |
| `apps/worker/rl_trading_inference/wiring/modules/__init__.py` | Worker wiring exports. | `compatible-change` internal import surface |
| `apps/worker/rl_trading_inference/wiring/modules/rl_trading_inference.py` | Config loader, Redis read-only window reader, Prometheus metrics and health/readiness HTTP server. | `compatible-change` disabled-by-default runtime surface |
| `tests/unit/contexts/rl_trading/domain/test_monitor_only_inference.py` | Feature parity, fail-closed missing fields, Stage `08M` aggregate feature shape, intercept-row policy loading, source-event payload and latency tests. | `none` test-only |
| `tests/unit/contexts/rl_trading/adapters/test_live_execution_rl_inference_producer.py` | Proves no-intent source-event write and no `ExecutionIntent`. | `none` test-only |
| `tests/unit/apps/worker/test_rl_trading_inference.py` | Worker config, Redis reader, metrics and CLI tests. | `none` test-only |
| `docs/architecture/ml/rl-trading-agent-platform-v1-stage-reports/13-monitor-only-inference-producer.md` | Stage report. | `compatible-change` docs/report |

Modified:

| Path | Reason | Contract impact |
|---|---|---|
| `src/trading/contexts/rl_trading/domain/__init__.py` | Export Stage `13` domain surface, including `STAGE13_STAGE08M_FEATURE_NAMES_V1`. | `compatible-change` additive Python exports |
| `src/trading/contexts/rl_trading/adapters/outbound/__init__.py` | Export Stage `13` ACL adapter. | `compatible-change` additive Python exports |
| `configs/dev/rl_trading_ml_runtime.yaml` | Add disabled-by-default monitor-only inference config, Redis input and latency budget. | `compatible-change` additive fail-closed config defaults |
| `configs/test/rl_trading_ml_runtime.yaml` | Same contract for test profile. | `compatible-change` additive fail-closed config defaults |
| `configs/prod/rl_trading_ml_runtime.yaml` | Same contract for prod/Mac Studio profile. | `compatible-change` additive fail-closed config defaults |
| `tests/unit/contexts/rl_trading/domain/test_ml_runtime_dependency_contract.py` | Assert Stage `13` config remains disabled, monitor-only and source-event fail-closed. | `none` test-only |
| `docs/architecture/ml/rl-trading-agent-platform-v1-stage-reports/rl-trading-agent-platform-v1-stage-ledger.md` | Records accepted Stage `13` state and opens Stage `14`. | `compatible-change` docs/ledger |
| `docs/architecture/README.md` | Mechanical docs index sync after adding/updating the Stage `13` report. | `compatible-change` docs index |

Deleted: none.

Outside expected paths: none for owned Stage `13` work.

## 3. Evidence and quality gates

Runtime artifact root:

`/opt/roehub/state/rl_trading/evaluation_runs/stage13_monitor_only_inference_producer_v1/stage13_post_main_20260703T184211Z`

Artifact hashes:

| Artifact | sha256 |
|---|---|
| `stage13_monitor_only_canary_summary.json` | `faee414b50def6bf0861e36807c2a6a55edd2680b4bc557c161f5d6d5d2d8e03` |
| `parity.json` | `648b1cae9588f21ad9842e975a02ea83b3316d23242ba6967e63b5bcb79c2318` |
| `db_source_event.json` | `f4a7ec3c48f50e1c910a5bdaf0ee65b8f35781e71b9af942fca45811b7aaf6e9` |
| `metrics_scrape.txt` | `fdeebfe648232c7e6b123ceb32cd35239db6d723deb8110434815cd9dcd63015` |
| `browser_strategies_proof.json` | `8a35e7b15387d5b3f72f445165d3ae1581f06bf837008f9ce3bdcf91e81a60bf` |
| `browser_strategies_proof.png` | `3ae13aecc108496a490cf4b056b6168fcf63f634e8a22792ed82b10d6e45698c` |

Runtime canary summary:

| Surface | Evidence |
|---|---|
| Fail-closed prod config before canary | `enabled=false`, `ready=false`, `mode=monitor_only`, `degraded_reasons=["inference_disabled"]`, `source_event_outcome=no_intent` |
| Redis source | `md.candles.1m.binance:futures:BTCUSDT`, `window_size=30`, `ts_open=2026-07-03T18:12:00Z`, `ts_close=2026-07-03T18:42:00Z` |
| Stage `08M` manifest | `/opt/roehub/state/rl_trading/evaluation_runs/stage08m_supervised_warm_start_candidate_scorecard_v1/stage08m_supervised_warm_start_fe2fe3c5257fd9992c55/stage08m_supervised_warm_start_candidate_manifest.json`, sha256 `9e2767ead0b697d0194e501aa7932b44fc1f5d1b180713f1270c81d1c887a69c`, `stage09_allowed=true` |
| Policy load | `policy_feature_count=12`, `policy_uses_intercept=true` |
| Train/live feature parity | `status=accepted`, `max_abs_diff=0.0`, feature hash `694981fce3c81d5a2f204677994cecba864b8f85ac9a3c27713bdd05e36212e3` |
| Decision | `action_name=open_short`, `confidence=0.387767795632`, `model_version_id=stage08m_a3823cbd01143878_fd7c614b`, `decision_id=26cb8c8d76d0a1021831690b40d62e60de4e20169717ac2156b1e6c332e7e4e5` |
| DB source event | `source_event_id=5789d223-2840-42c0-8785-9062f2a79fd4`, `source_type=ml_agent_decision`, `source_event_ref=rl:26cb8c8d76d0a1021831690b40d62e60de4e20169717ac2156b1e6c332e7e4e5`, `outcome=no_intent`, `outcome_reason=monitor_only_no_intent`, `intent_id=null`, count before/after for ref `0 -> 1` |
| No order side effects | `intents_for_source_event=0`, `orders_for_source_event=0` |
| Redis execution dispatch | before/after unchanged: `execution.requests.v1=41`, `execution.requests.retry.v1=1`, `execution.requests.dlq.v1=2` |
| Latency observations | `candle_close_to_feature_ready=0.001644709002`, `feature_to_decision=0.000191541010`, `decision_to_source_event=0.033681417000` seconds |
| Metrics scrape | `/health/live=200`, `/health/ready=200`, `/metrics=200`; included `rl_trading_inference_ready 1.0`, `rl_trading_inference_decisions_total`, `rl_trading_inference_feature_parity_total` |
| Browser `/strategies` proof | authenticated Playwright browser opened `http://127.0.0.1:13013/strategies/ee15e181-309f-478e-8726-04a299f1292f` through an SSH tunnel to temporary web server on `macstudio`; `title="Strategies | Roehub"`, dashboard API status `200`, `page_text_has_logout=true`, `page_text_has_guest=false`, `matching_no_intent_count=2`, first match source event `5789d223-2840-42c0-8785-9062f2a79fd4`, `intent_id=null` |

Quality gates:

| Gate | Result |
|---|---|
| `uv run ruff check src/trading/contexts/rl_trading apps tests` | passed |
| `uv run pyright src/trading/contexts/rl_trading apps tests` | passed: `0 errors, 0 warnings, 0 informations` |
| `uv run pytest -q tests/unit/contexts/rl_trading tests/unit/apps` | passed: `480 passed, 8 skipped, 3 warnings` |
| `uv run python -m apps.worker.rl_trading_inference.main.main status --config configs/prod/rl_trading_ml_runtime.yaml` on `/opt/roehub/app` | passed; fail-closed status reports `enabled=false`, `ready=false`, `degraded_reasons=["inference_disabled"]` |
| `uv run python -m tools.docs.generate_docs_index --check` | passed after report/ledger/docs index update |
| GitHub `CI` | passed for `9e1ee429212cf22c1ae2e2fa0c4cad6f0b13bc12`, run `28677744947` |
| GitHub `Deploy Backend` | passed for `9e1ee429212cf22c1ae2e2fa0c4cad6f0b13bc12`, run `28677826766` |
| `bash scripts/macos/smoke_prod.sh` on `macstudio` | passed after deploy/sync |

## 4. Contract, safety, and delivery state

| Surface | Classification | Reason |
|---|---|---|
| Public API contract | `none` | No API route/DTO was changed for Stage `13`. |
| Live-execution source-event contract | `compatible-change` | Adds a new producer of existing `source_type=ml_agent_decision` events through existing ingress/repository ports. |
| Execution/order intent contract | `none` | Monitor-only path never calls `create_intent`; post-main runtime evidence proves `0` intents and `0` orders for the canary source event. |
| Persisted schema | `none` | No migration or table contract changed. |
| Config schema/defaults | `compatible-change` | Adds disabled-by-default `inference.mode=monitor_only`, Redis read source, metrics port and latency budget. |
| Metrics/logs | `compatible-change` | Adds bounded `rl_trading_inference_*` metrics with no user/strategy identifiers in labels. |
| Browser-visible behavior | `none` | Stage `11` reusable outcome journal already renders `ml_agent_decision`; Stage `13` only populated the existing read model with monitor-only source events. |
| Runtime/deploy | `compatible-change` | `9e1ee429212cf22c1ae2e2fa0c4cad6f0b13bc12` is on `main`, GitHub CI/deploy passed, `/opt/roehub/app` was synced, production smoke passed, and post-main canary/browser proof passed. |

Safety facts:

- No exchange SDK, provider credential, signed request, paper/testnet/live/mainnet order, or Redis execution dispatch path was used.
- Runtime config remains disabled by default: `inference.enabled=false` and `source_events.enabled=false`.
- The browser QA used a temporary DB-backed session for the existing strategy owner, then revoked it; no password or session value is written in this report.
- A temporary web server was started only for browser QA on `127.0.0.1:8013` and was stopped after proof.

Business impact layer:

- Stage `13` adds observable RL/ML monitor-only decisions to the existing `/strategies` journal without enabling trading. Users/operators can inspect model output, but no order, capital allocation, exchange submission or paper/testnet/live behavior changes.

Conditional service-call coverage:

- Exchange/provider calls: `N/A`; monitor-only producer does not import or call exchange SDKs.
- Redis market-data calls: covered read-only via `md.candles.1m.binance:futures:BTCUSDT`; dispatch streams were checked before/after and unchanged.
- Postgres calls: covered by one source-event write/update through existing live-execution repository; `execution_intents` and `execution_orders` counts stayed `0`.
- Browser/API calls: covered by authenticated `/strategies` browser proof and same-origin `/api/ui/strategies/dashboard` request.

Logging/redaction coverage:

- Metrics labels are bounded and contain no user id, strategy id, session id, credential, provider payload or raw order payload.
- Runtime docs record only sanitized ids, hashes, counts and paths. Temporary browser session value was not written to report, screenshot, logs or committed files, and was revoked after proof.

Delivery state: `accepted_delivered_to_main_post_main_production_runtime_proof`.

## 5. Blockers and next-stage handoff

Blockers: none for Stage `13`.

Stage `14` may start from:

- accepted Stage `13` monitor-only producer;
- code commit `9e1ee429212cf22c1ae2e2fa0c4cad6f0b13bc12`;
- source event proof `5789d223-2840-42c0-8785-9062f2a79fd4`;
- artifact bundle `/opt/roehub/state/rl_trading/evaluation_runs/stage13_monitor_only_inference_producer_v1/stage13_post_main_20260703T184211Z`;
- fail-closed runtime defaults (`inference.enabled=false`, `source_events.enabled=false`).

Stage `14` must not infer any paper/testnet/live readiness from this result. Stage `13` proves monitor-only inference and `no_intent` journal visibility only.

## Cold self-review

Mode: `cold self-review fallback`.

Verdict: `accepted`.

Findings:

- `Fixed`: real accepted Stage `08M` manifest required Stage `08M` aggregate feature vector and intercept-row support; focused tests now cover both.
- `Accepted`: post-main runtime proof proves Redis parity, DB `no_intent`, no intents/orders, unchanged dispatch streams, metrics scrape, production smoke and `/strategies` browser visibility.
- `Residual risk`: this stage does not enable paper/testnet/live/mainnet execution; Stage `14+` must keep the money boundary closed until their own acceptance gates pass.
