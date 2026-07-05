---
doc: rl-trading-agent-platform-v1-stage-17-multi-ticker-runtime-load
stage: "17"
status: accepted
mode: infrastructure_only
updated_at: 2026-07-05
---

# Stage 17: Multi-ticker runtime/load gate

Статус: `accepted`.

User required before start: nothing unless a listed prerequisite is not accepted or a required credential/dataset/runtime source is unavailable; never ask for secrets in chat.

Stage `17` проверяет только инфраструктурную способность Mac Studio выполнять
multi-ticker RL inference/load для тарифоподобных лимитов. Этот stage не
доказывает качество модели, торговый edge, product readiness, live execution
readiness, mainnet readiness или user-facing rollout readiness.

Proof boundary: `target_host_main_checkout_runtime_load`. Harness выполнен на
`macstudio` из git checkout `/Users/daniildegtyarev/Projects/roehub.com` на
`main` revision `bf964e9ec89000b1f5fcfdac71d2f83351c7311e`. Runtime tree
`/opt/roehub/app` не синкался, сервисы не перезапускались, production smoke не
использовался, поэтому это не `post_main_production_runtime_proof`.

## Scope

Входит:

- Free/Pro/Premium-like сценарии активных RL live ticker slots: `1`, `5`, `20`;
- read-only чтение live Redis candle streams для `binance:futures`;
- in-memory построение `ml_agent_decision` source-event payload без DB write,
  Redis dispatch write, exchange SDK или provider call;
- измерение segment p95 latency, feed lag, decision throughput, CPU/RSS/MPS,
  Redis execution queue/retry/DLQ growth и overlap с backtest worker;
- handoff для Stage `18` только как `monitor_only_technical_soak`.

Не входит:

- paper/testnet/live/mainnet submit;
- запись source events в production DB;
- изменение tariff/billing/user quota contracts;
- model-quality, profitability, promotion-grade или product suitability claim;
- deploy/sync `/opt/roehub/app`.

## Business Impact

Stage `17` снижает операционную неопределенность перед техническим
monitor-only soak: инфраструктура Mac Studio выдержала плановые
Free/Pro/Premium-like ticker counts без роста Redis execution/retry/DLQ streams
и без заметного RSS pressure. Это не делает кандидата продаваемым, торгуемым
или готовым к mainnet; product/legal/mainnet gates остаются закрытыми Stage
`08N`.

## Prompt And Prerequisites

| Item | Result |
|---|---|
| Prompt path | `.codex/agents/generated/rl-trading-agent-platform-v1/17-multi-ticker-runtime-load.md` |
| Prompt sha256 | `0c6712819d88f498fea37241f6582978fccb5d0b89ea6a3765742e8231a61790` |
| `current_stage` before execution | `17` |
| Stage `16` prerequisite | accepted |
| Stage `08N` prerequisite | accepted |
| Stage `08N` allowance | `stage17_infrastructure_only_allowed=true`; `stage17_full_runtime_allowed=false` |
| User/credential requirement | none; no secrets, browser auth, provider credentials or exchange payloads were read |
| Planned concrete file list before edits | `src/trading/contexts/rl_trading/domain/stage17_runtime_load.py`; `scripts/rl_trading/stage17_multi_ticker_runtime_load.py`; `src/trading/contexts/rl_trading/domain/__init__.py`; `tests/unit/contexts/rl_trading/domain/test_stage17_runtime_load.py`; `tests/unit/apps/worker/test_rl_trading_inference.py`; this report; stage ledger; docs index |

The compact prompt-manager context snapshot was ignored because it referred to
old Stage `02A` and branch/PR delivery state. The current ledger and
`.codex/AGENTS.md` supersede that stale snapshot.

## File Manifest

| Path | State | Reason | Contract impact |
|---|---|---|---|
| `scripts/rl_trading/stage17_multi_ticker_runtime_load.py` | created | Read-only Mac Studio load harness over live Redis candle streams and in-memory `ml_agent_decision` payload generation. | `compatible-change` operator CLI/test harness |
| `src/trading/contexts/rl_trading/domain/stage17_runtime_load.py` | created | Deterministic Stage `17` quota/load summary, p95, Redis delta, resource and Stage `18` handoff logic. | `compatible-change` internal domain/reporting surface |
| `src/trading/contexts/rl_trading/domain/__init__.py` | modified | Export Stage `17` summary types/functions. | `compatible-change` Python package export |
| `tests/unit/contexts/rl_trading/domain/test_stage17_runtime_load.py` | created | Prove accepted and blocked summary behavior, including DLQ-growth blocker. | `none` test-only |
| `tests/unit/apps/worker/test_rl_trading_inference.py` | modified | Prove Stage `17` harness exercises `1/5/20` quota counts with fake Redis and no dispatch/DLQ growth. | `none` test-only |
| `docs/architecture/ml/rl-trading-agent-platform-v1-stage-reports/17-multi-ticker-runtime-load.md` | created | Stage `17` report, evidence and handoff. | `compatible-change` docs/report |
| `docs/architecture/ml/rl-trading-agent-platform-v1-stage-reports/rl-trading-agent-platform-v1-stage-ledger.md` | modified | Mark Stage `17` accepted, set `current_stage=18`, record evidence and Stage `18` limits. | `compatible-change` docs/ledger |
| `docs/architecture/README.md` | modified | Docs index regeneration after adding Stage `17` report. | `compatible-change` docs index |

Deleted: none.

Outside expected paths: none.

Runtime artifact created outside git:

| Path | State |
|---|---|
| `/opt/roehub/state/rl_trading/evaluation_runs/stage17_multi_ticker_runtime_load_v1/stage17_macstudio_20260705t182700z/stage17_multi_ticker_runtime_load_summary.json` | created on `macstudio`; sanitized JSON summary only |

Runtime artifact file sha256:
`3a3baa4f2ca936fec2d89b5d9a17b11ce6c7a4f125f4124c145954a179df4761`.

Runtime artifact internal `summary_hash`:
`5bd5c64f3b29f62b59e548ea374e519c90f7b99c8bbfbb4b9f2d1bc66dd7264d`.

## Evidence

Mac Studio run:

| Field | Value |
|---|---|
| Host | `macstudio` |
| Checkout | `/Users/daniildegtyarev/Projects/roehub.com` |
| Git revision used by harness | `bf964e9ec89000b1f5fcfdac71d2f83351c7311e` |
| Config | `configs/prod/rl_trading_ml_runtime.yaml` |
| Candidate manifest | `/opt/roehub/state/rl_trading/evaluation_runs/stage08m_supervised_warm_start_candidate_scorecard_v1/stage08m_supervised_warm_start_fe2fe3c5257fd9992c55/stage08m_supervised_warm_start_candidate_manifest.json` |
| Output summary | `/opt/roehub/state/rl_trading/evaluation_runs/stage17_multi_ticker_runtime_load_v1/stage17_macstudio_20260705t182700z/stage17_multi_ticker_runtime_load_summary.json` |
| Status | `accepted` |

Quota scenarios:

| Scenario | Paid level | Product label | Requested live tickers | Observed tickers | Quota bypass |
|---|---|---|---:|---:|---|
| Free-like | `free` | `Free` | `1` | `1` | `false` |
| Pro-like | `pro` | `Pro` | `5` | `5` | `false` |
| Premium-like | `ultra` | `Premium` | `20` | `20` | `false` |

Latency p95:

| Segment | p95 ms | Budget ms |
|---|---:|---:|
| `candle_close_to_feature_ready` | `0.07704203` | `250` |
| `feature_to_decision` | `0.165458012` | `100` |
| `decision_to_source_event` | `0.018124992` | `50` |

Feed and throughput:

| Metric | Value |
|---|---:|
| Observations | `26` |
| Decision throughput | `4082.620954970955` decisions/sec |
| Feed lag p95 | `6.738794088364` sec |
| Feed lag max | `6.738996982574` sec |
| Feed lag threshold | `300.0` sec |

Redis execution streams:

| Stream | Before | After | Delta |
|---|---:|---:|---:|
| `execution.requests.v1` | `49` | `49` | `0` |
| `execution.requests.retry.v1` | `1` | `1` | `0` |
| `execution.requests.dlq.v1` | `2` | `2` | `0` |

Resource and contention:

| Metric | Value |
|---|---|
| CPU count | `12` |
| MPS available | `true` |
| RSS before/after | `197.71875 MiB` -> `197.84375 MiB` |
| RSS budget | `4096 MiB` |
| CPU time | `0.00616` sec |
| Wall time | `0.00636845798` sec |
| Disk free under artifact root | `709.573624 GiB` |
| Active overlap | `observed_overlap`: `/opt/roehub/app/.venv/bin/python -m apps.worker.backtest_job_runner.main.main --metrics-port 9204` |
| Trainer config | `trainer.enabled=false` |

Acceptance checks from the sanitized summary:

| Check | Result |
|---|---|
| `all_scenarios_met_requested_ticker_counts` | `true` |
| `monitor_only_source_events_only` | `true` |
| `segment_latency_budget_met` | `true` |
| `redis_execution_stream_growth_zero` | `true` |
| `dlq_growth_zero` | `true` |
| `feed_lag_within_limit` | `true` |
| `rss_budget_met` | `true` |
| `resource_contention_bounded_or_observed` | `true` |

## Quality Gates

| Gate | Result |
|---|---|
| `uv run ruff check src/trading/contexts/rl_trading apps tests` | passed |
| `uv run pyright src/trading/contexts/rl_trading apps tests` | passed |
| `uv run pytest -q tests/unit/contexts/rl_trading tests/unit/apps` | passed: `500 passed, 8 skipped, 3 warnings` |
| `uv run pytest -q tests/unit/contexts/rl_trading/domain/test_stage17_runtime_load.py tests/unit/apps/worker/test_rl_trading_inference.py` | passed: `11 passed` |
| `uv run pytest -q tests/perf_smoke/contexts/rl_trading/test_stage08f_roehub_native_evaluation.py tests/unit/contexts/rl_trading/domain/test_stage17_runtime_load.py` | passed: `2 passed, 1 skipped` (`torch` unavailable locally) |
| `uv run ruff check .` | passed |
| `uv run pyright` | passed |
| `uv run pytest -q -ra` | passed: `1563 passed, 16 skipped, 3 warnings` |
| `python -m tools.docs.generate_docs_index --check` | passed after docs index regeneration |

## Contract Impact

| Surface | Classification | Reason |
|---|---|---|
| Public API contract | `none` | No API route or public payload changed. |
| Port contract | `none` | No application port/interface changed. |
| DTO schema | `none` | No DTO changed. |
| Persisted schema | `none` | No migration/table/storage schema changed. |
| Config schema/defaults | `none` | Existing runtime config was read; no committed defaults changed. |
| Request hash/cache key/persistence identity | `none` | No identity, cache or persistence hash semantics changed. |
| Operator CLI/test harness | `compatible-change` | Adds explicit Stage `17` harness; existing commands unchanged. |
| Logs/metrics/traces/audit/report semantics | `compatible-change` | Adds sanitized runtime-load summary and docs/ledger evidence. |
| External side effects | `none` | Redis live candle streams and execution stream lengths were read; no Redis writes, DB writes, exchange SDK calls or provider submits occurred. |
| Browser-visible behavior | `none` | Browser/UI/auth out of scope. |
| Performance/hot path | `compatible-change` | Adds measurement and summary tooling for monitor-only inference load; production inference behavior remains unchanged. |
| Docs/runbooks | `compatible-change` | Adds Stage `17` report and ledger handoff; no runbook change required. |

## Safety And Redaction

- No secrets, tokens, cookies, passphrases, ciphertext, HMAC, API keys,
  credentials, raw signed requests, raw provider payloads, raw candle payload
  dumps, model checkpoint tensors or datasets were committed.
- The runtime artifact contains instrument keys, hashes, counts and metrics only.
- The ML worker/harness did not call exchange SDKs and did not resolve exchange
  secret custody.
- Mainnet submit remains blocked until Stage `19` readiness review and explicit
  Stage `20` approval.

## Delivery State

| Item | State |
|---|---|
| Code/test harness commit | `bf964e9ec89000b1f5fcfdac71d2f83351c7311e` pushed to `origin/main` |
| Mac Studio checkout | `/Users/daniildegtyarev/Projects/roehub.com` fast-forwarded to `bf964e9ec89000b1f5fcfdac71d2f83351c7311e` before the run |
| Runtime tree `/opt/roehub/app` | not synced/restarted by Stage `17` |
| Stage report/ledger/docs index | updated in this docs handoff |
| GitHub Actions | code commit `CI` run `28750519042` started; final docs commit CI must be checked after publish |

## Next-Stage Handoff

Stage `18` may start only as `monitor_only_technical_soak`, not full
trade-readiness soak. Handoff limits:

- maximum ticker count proven for the technical soak: `20`;
- allowed mode: `monitor_only_technical_soak`;
- forbidden claims: model quality, trading edge, product readiness, mainnet
  readiness;
- keep `stage18_soak_allowed=false`, `stage19_mainnet_readiness_allowed=false`,
  `stage20_mainnet_canary_allowed=false`, `stage21_product_rollout_allowed=false`
  from accepted Stage `08N`;
- if Stage `18` needs production service proof, it must use a later
  `post_main_production_runtime_proof` boundary after green CI/deploy/sync.
