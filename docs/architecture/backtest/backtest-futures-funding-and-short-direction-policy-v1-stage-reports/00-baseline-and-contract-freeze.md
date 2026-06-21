# Backtest Futures Funding And Short Direction Policy v1 - Stage 00 Baseline And Contract Freeze

Docs-only baseline freeze before funding and futures-only short policy implementation.

Date: 2026-06-22

Status: accepted-local, docs-only, not published or deployed.

Branch: `codex/backtest-futures-funding-v1-stage-00`

`User required before start: nothing`

## Scope

Stage 00 froze the current implementation baseline for the funding and futures-only
short policy plan before production code changes.

In scope:

- read `.codex/AGENTS.md`, the source architecture document, the stage ledger,
  listed market-data/backtest docs, and the listed code entrypoints;
- refresh official Binance and Bybit funding-provider facts from primary docs;
- verify the current `market-data-scheduler` topology, enabled-instrument scan
  pattern, and Prometheus metrics baseline;
- record real-boundary availability for ClickHouse, local API, local web/browser
  smoke, and scheduler `/metrics`;
- freeze narrow file manifests for stages `01` through `08`;
- classify contract impacts and update the ledger.

Out of scope:

- no production code changes;
- no ClickHouse migrations;
- no runtime config changes;
- no UI changes;
- no reopening of the closed `backtest-compute-acceleration-v1` stage family.

## Business-Readable Layer

Для продукта этот stage ничего не меняет в поведении Roehub. Он фиксирует
текущую точку отсчета: сейчас funding в backtest не считается, standalone
`short` в backtest runtime еще нет, а browser/default flow все еще может
показывать `spot + long_short_reversal` как обычную конфигурацию. Следующие
stages должны исправлять это постепенно и с доказательствами, чтобы пользователь
не запускал futures-стратегию по gross-only доходности и не воспринимал spot
short-like сценарии как launchable.

Бизнес-результат после всей линии должен быть таким: futures backtest показывает
gross и net-of-funding return рядом, funding freshness видна до запуска и в
результатах, а short-like стратегии создаются только для futures. Stage 00
только проверяет, какие файлы и контракты должны быть затронуты дальше.

## Conditional Service-Call Coverage

Stage 00 service calls: N/A. This stage is documentation and verification only;
it did not add runtime service calls, provider clients, ClickHouse writes,
authenticated browser flows, retries, idempotency behavior or side effects.

Provider/API coverage for future stages:

- Stage `01` owns public unauthenticated Binance and Bybit market-data REST
  calls, ClickHouse writes and `market-data-scheduler` `/metrics` proof.
- Stage `02` owns ClickHouse reads for scheduler-maintained
  `canonical_funding_rates` and artifact filesystem publish/load proof.
- Stage `03`, `06` and `07` own local API route proof.
- Stage `07` owns browser QA and authenticated flow handling if browser auth is
  needed.
- Stage `08` owns broad runtime/browser/Prometheus/delivery proof when delivery
  is in scope.

Secrets/redaction: all stages may report env var names, provider names, route
names, aggregate counts and status codes; they must not report DSNs, API keys,
bearer tokens, ClickHouse passwords, cookies, smoke passwords or raw
secret-bearing provider/runtime payloads.

Startup state:

- `git status --short`: clean before edits.
- No unrelated dirty files were present.
- Current branch was created from local `main` as
  `codex/backtest-futures-funding-v1-stage-00`.

## Baseline Facts

### Official provider facts rechecked

Official docs were rechecked on 2026-06-22. No provider REST smoke was run
because the prompt allowed official docs re-check and the local runtime services
were unavailable.

| Provider surface | Current official fact | Source |
| --- | --- | --- |
| Binance USD-M funding history | `GET /fapi/v1/fundingRate`; `symbol`, `startTime`, `endTime`, `limit` are optional; `limit` default is `100`, max is `1000`; omitting both time bounds returns the most recent `200` records; responses are ascending; rate limit is shared with `GET /fapi/v1/fundingInfo`. | [Binance funding rate history](https://developers.binance.com/docs/derivatives/usds-margined-futures/market-data/rest-api/Get-Funding-Rate-History) |
| Binance USD-M funding info | `GET /fapi/v1/fundingInfo`; response is for symbols with adjusted cap/floor or `fundingIntervalHours`; response fields include adjusted cap, adjusted floor, `fundingIntervalHours`; request weight is `0` but it shares the `500/5min/IP` limit with funding history. | [Binance funding info](https://developers.binance.com/docs/derivatives/usds-margined-futures/market-data/rest-api/Get-Funding-Rate-Info) |
| Binance USD-M futures universe | `GET /fapi/v1/exchangeInfo`; request weight `1`; no request parameters; response contains symbol-level contract metadata including `symbol`, `contractType`, `status`, base/quote/margin assets and trading filters. | [Binance exchange information](https://developers.binance.com/docs/derivatives/usds-margined-futures/market-data/rest-api/Exchange-Information) |
| Bybit funding history | `GET /v5/market/funding/history`; `category` is required and must be `linear` or `inverse`; `symbol` is required uppercase; `limit` range is `1..200`, default `200`; passing only `startTime` errors; passing only `endTime` returns up to `200` records before that end. | [Bybit funding rate history](https://bybit-exchange.github.io/docs/v5/market/history-fund-rate) |
| Bybit instruments info | `GET /v5/market/instruments-info`; category includes `spot`, `linear`, `inverse`, `option`; `linear` defaults to `500` rows and needs `cursor` pagination when the universe is larger; linear/inverse rows include `fundingInterval`, `settleCoin`, `upperFundingRate`, `lowerFundingRate`; default status is trading symbols. | [Bybit instruments info](https://bybit-exchange.github.io/docs/v5/market/instrument) |

Implication for Stage `01`: Roehub internal `market_type=futures` maps to Bybit
external `category=linear` for v1; `futures` must not be sent as a Bybit V5
category.

### Current repository facts

| Area | Observed current fact | Source |
| --- | --- | --- |
| Runtime defaults / preflight | `BACKTEST_DIRECTION_MODES_V1` is only `("long_only", "long_short_reversal")`; standalone `short` is absent; normalized request has no `execution.funding`; request hash is built from the normalized request before funding fields exist. | `src/trading/contexts/backtest/application/services/v2/preflight.py` |
| Backtest coordinates | Current coordinates validate against artifact market ids `1..4`: Binance spot/futures and Bybit spot/futures. | `src/trading/contexts/backtest/application/services/v2/preflight.py`; `src/trading/contexts/backtest_artifacts/application/services/v2/contracts.py` |
| Top result assembly | Top variants are summary-only; `trades_json=None`; `variant_hash` includes canonical params with `execution` and `ranking`, but no funding fields. | `src/trading/contexts/backtest/application/services/v2/top_result_assembly.py` |
| Lazy trades cache | Cache key uses job id, public variant key, variant hash, request hash, engine params hash and `artifact_manifest_hash`; no `funding_manifest_hash` exists. | `src/trading/contexts/backtest/application/services/v2/lazy_trades_detail.py` |
| Lazy TP/SL exit detail | Lazy detail already resolves exact TP/SL exits through `_tp_sl_exit_for_detail`; stop loss wins same-bar ties because `t_sl <= t_tp` returns stop loss. | `src/trading/contexts/backtest/application/services/v2/lazy_trades_detail.py` |
| Artifact families | Root manifest requires `prices`, `mappings`, `signals`, `hit_times`; optional `signal_features` exists; no funding family or funding manifest contract exists. Hit-times include long and short TP/SL arrays. | `src/trading/contexts/backtest_artifacts/application/services/v2/contracts.py` |
| ClickHouse market data DDL | Current DDL has `ref_market`, `ref_instruments`, raw Binance klines, raw Bybit klines, canonical 1m candles and candle stats; there are no funding tables. Fixed market ids are `1` Binance spot, `2` Binance futures, `3` Bybit spot, `4` Bybit futures. | `migrations/clickhouse/market_data_ddl.sql` |
| Reference instrument scope | `ref_instruments` is whitelist-driven; latest rows with `status='ENABLED'` and `is_tradable=1` form the enabled tradable universe. | `docs/architecture/market_data/market-data-reference-data-sync-v2.md`; `src/trading/contexts/market_data/adapters/outbound/persistence/clickhouse/enabled_instrument_reader.py` |
| Scheduler process | `market-data-scheduler` accepts `--metrics-port`, default `9202`, and starts a Prometheus HTTP server on that port. | `apps/scheduler/market_data_scheduler/main/main.py`; `apps/scheduler/market_data_scheduler/wiring/modules/market_data_scheduler.py` |
| Scheduler jobs | Startup runs `sync_whitelist`, `enrich`, `startup_scan`; periodic jobs are `sync_whitelist`, `enrich`, `rest_insurance_catchup`; no `funding_rate_catchup` job exists. | `apps/scheduler/market_data_scheduler/wiring/modules/market_data_scheduler.py` |
| Scheduler all-enabled scan pattern | Startup and REST insurance scans call `ClickHouseEnabledInstrumentReader.list_enabled_tradable()`, so they enumerate only whitelist-enabled tradable rows from `ref_instruments`. | `apps/scheduler/market_data_scheduler/wiring/modules/market_data_scheduler.py`; `src/trading/contexts/market_data/adapters/outbound/persistence/clickhouse/enabled_instrument_reader.py` |
| Scheduler config | Runtime config only parses scheduler jobs `sync_whitelist`, `enrich`, `rest_insurance_catchup`; no funding-specific scheduler config exists. | `src/trading/contexts/market_data/adapters/outbound/config/runtime_config.py` |
| Scheduler metrics baseline | Existing scheduler metrics are `scheduler_job_*`, `scheduler_tasks_*`, `scheduler_startup_scan_instruments_total`, and `scheduler_rest_catchup_*`; there are no `scheduler_funding_catchup_` series. | `apps/scheduler/market_data_scheduler/wiring/modules/market_data_scheduler.py`; `docs/runbooks/market-data-metrics-reference-ru.md` |
| Prometheus scrape baseline | Production Prometheus scrapes `market-data-scheduler` at `127.0.0.1:9202`; no funding alert rule file is included. | `infra/macos/prometheus/prometheus.prod.yml` |
| Backtest API routes | Backtest routes already expose runtime defaults, preflight, jobs, top, variant, compatibility-readiness, scenario matrix, create-strategy, lazy trades, paginated trades and CSV. | `apps/api/routes/backtests.py` |
| Strategy launch validator | `POST /api/strategies/launch-from-backtest-variant` accepts directions `long`, `short`, `long_short_reversal`; it currently blocks only `testnet + spot + short-like`, not paper spot short-like. The current reason is `spot_short_not_supported`, not `short_direction_requires_futures_market`. | `apps/api/routes/strategies.py` |
| Scenario matrix | For `paper + spot + short`, current order capability is `paper_only` with `spot_short_not_real_order_capable`; testnet spot short is blocked. `long_short_reversal` expands to `long` and `short`; standalone backtest `short` is not mapped. | `src/trading/contexts/strategy/application/use_cases/scenario_matrix.py` |
| Browser backtest form | Template and JS default to `market_type="spot"` and `direction="long_short_reversal"`; runtime controls render directions from backend defaults; there is no funding UI state. | `apps/web/templates/pages/backtests.html`; `apps/web/dist/js/pages/backtests.js` |
| Browser launch modal | Launch defaults derive market type from job/state and direction from job/state; client-side launch block applies only to `testnet + spot + short-like`, not paper spot short-like. | `apps/web/templates/pages/backtests.html`; `apps/web/dist/js/pages/backtests.js` |

### Runtime boundary availability in this environment

| Boundary | Probe | Result | Stage 00 consequence |
| --- | --- | --- | --- |
| Scheduler `/metrics` | `curl -fsS --max-time 2 http://127.0.0.1:9202/metrics | rg '^scheduler_' | head -n 20` | Unavailable: connection refused on port `9202`. | Not a blocker for docs-only Stage 00; Stage `01` must prove runtime metrics or block. |
| ClickHouse | `curl -fsS --max-time 2 http://127.0.0.1:8123/ping` | Unavailable: connection refused on port `8123`. | Stage `01` ClickHouse smoke is currently unavailable locally. |
| Local API | `curl -fsS --max-time 2 http://127.0.0.1:8000/openapi.json` | Unavailable: connection refused on port `8000`. | Stage `03`, `06`, `07` local API smokes need a running API. |
| Local web/browser smoke | `curl -fsS --max-time 2 http://127.0.0.1:3000/backtests` | Unavailable: connection refused on port `3000`. | Stage `07` browser QA needs a running web/API environment. |

## Frozen Future Stage File Manifest

Rules for stages `01` through `08`:

- Each stage must begin its report with `User required before start: nothing`.
- Each implementation stage must re-check the previous required stage status in
  the ledger before edits.
- Files outside the manifest below require an explicit pre-edit note in that
  stage report explaining why the boundary changed.
- Deletions are not allowed by default; any deletion requires explicit stage
  report justification and contract classification.
- `docs/architecture/README.md` may change only if
  `python -m tools.docs.generate_docs_index` regenerates it.

| Stage | Primary created files | Primary modified files | Tests / validation files |
| --- | --- | --- | --- |
| `01` Funding storage, automatic catch-up and observability | `migrations/clickhouse/funding_rates_ddl.sql`; `src/trading/contexts/market_data/application/ports/sources/funding_rate_history_source.py`; `src/trading/contexts/market_data/application/ports/sources/funding_instrument_universe_source.py`; `src/trading/contexts/market_data/application/ports/stores/funding_rate_writer.py`; `src/trading/contexts/market_data/application/ports/stores/funding_instrument_universe_store.py`; `src/trading/contexts/market_data/application/use_cases/backfill_funding_rates.py`; `src/trading/contexts/market_data/application/use_cases/sync_futures_funding_universe.py`; `src/trading/contexts/market_data/adapters/outbound/clients/funding_rate_history_source.py`; `src/trading/contexts/market_data/adapters/outbound/persistence/clickhouse/funding_rate_store.py`; `apps/cli/commands/funding_rate_catchup.py`; `infra/macos/prometheus/rules/market-data-funding.rules.yml`; stage report `01-funding-storage-and-catchup.md`. | `apps/cli/main/main.py`; `apps/scheduler/market_data_scheduler/wiring/modules/market_data_scheduler.py`; `src/trading/contexts/market_data/adapters/outbound/config/runtime_config.py`; `configs/dev/market_data.yaml`; `configs/test/market_data.yaml`; `configs/prod/market_data.yaml`; `infra/macos/prometheus/prometheus.prod.yml`; `docs/runbooks/market-data-metrics-reference-ru.md`; optionally `docs/runbooks/market-data-metrics.md`; ledger. | `tests/unit/contexts/market_data/`; `tests/unit/apps/cli/`; `tests/unit/apps/scheduler/`; `tests/unit/infra/test_monitoring_assets.py`; provider REST smoke or blocker; ClickHouse apply/query smoke or blocker; `curl 127.0.0.1:9202/metrics` funding metric proof. |
| `02` Funding artifact family and coverage | Funding artifact contract/loader helpers under `src/trading/contexts/backtest_artifacts/application/services/v2/`; coverage reader port `src/trading/contexts/market_data/application/ports/stores/funding_rate_coverage_reader.py`; stage report `02-funding-artifact-family-and-coverage.md`. | `src/trading/contexts/backtest_artifacts/application/services/v2/contracts.py`; `src/trading/contexts/backtest/application/ports/artifact_arrays.py`; `src/trading/contexts/backtest/adapters/outbound/artifacts_fs/artifact_array_loader.py`; optional `apps/cli/commands/backtest_artifact_publish.py`; ledger. | `tests/unit/contexts/backtest_artifacts/`; `tests/unit/contexts/backtest/`; filesystem publish/load smoke against temp artifact root; ClickHouse-backed coverage smoke when Stage `01` data exists. |
| `03` Runtime defaults, preflight and funding readiness | Stage report `03-preflight-runtime-defaults-funding-readiness.md`; optional DTO modules under `src/trading/contexts/backtest/application/dto/` only if existing DTO files cannot carry additive fields cleanly. | `src/trading/contexts/backtest/application/services/v2/preflight.py`; `apps/api/routes/backtests.py`; `apps/api/dto/backtests.py`; ledger. | `tests/unit/contexts/backtest/`; `tests/unit/apps/api/`; local route smoke for runtime-defaults and preflight when API is available; request-hash tests. |
| `04` No-risk funding adjustment | Reusable funding calculation helper under `src/trading/contexts/backtest/application/services/v2/` if needed; stage report `04-no-risk-funding-adjustment.md`. | `src/trading/contexts/backtest/application/services/v2/top_result_assembly.py`; `src/trading/contexts/backtest/domain/entities/backtest_job_results.py`; `src/trading/contexts/backtest/adapters/outbound/persistence/postgres/backtest_job_repository.py`; ledger. | `tests/unit/contexts/backtest/`; performance evidence on artifact-backed runtime input; formula tests for long/short positive and negative funding. |
| `05` TP/SL funding adjustment | Shared exact exit resolver under `src/trading/contexts/backtest/application/services/v2/` only if extraction is needed to prevent drift; stage report `05-tp-sl-funding-adjustment.md`. | `src/trading/contexts/backtest/application/services/v2/lazy_trades_detail.py`; `src/trading/contexts/backtest/application/services/v2/top_result_assembly.py`; ledger. | `tests/unit/contexts/backtest/`; same-bar TP/SL regression tests; top/detail exit-alignment tests; performance evidence on artifact-backed runtime input. |
| `06` Results API, lazy detail and persistence | Stage report `06-results-api-lazy-detail-and-persistence.md`; migration only if the stage proves Postgres/cache identity cannot remain additive. | `src/trading/contexts/backtest/application/dto/backtest_jobs.py`; `src/trading/contexts/backtest/application/ports/lazy_trades_cache.py`; `src/trading/contexts/backtest/application/services/v2/lazy_trades_detail.py`; `apps/api/dto/backtests.py`; `apps/api/routes/backtests.py`; optionally `src/trading/contexts/backtest/adapters/outbound/persistence/postgres/backtest_job_repository.py`; ledger. | `tests/unit/apps/api/`; `tests/unit/contexts/backtest/`; route smoke for top, variant and lazy detail when API is available; cache-key tests proving `funding_manifest_hash` identity. |
| `07` Futures-only short policy API and CJM | Stage report `07-futures-only-short-policy-api-and-cjm.md`; locale keys under `apps/web/locales/` only for new UI text. | `apps/api/routes/backtests.py`; `apps/api/routes/strategies.py`; `src/trading/contexts/strategy/application/use_cases/scenario_matrix.py`; `src/trading/contexts/strategy/application/use_cases/create_strategy_from_backtest_variant.py`; `apps/web/templates/pages/backtests.html`; `apps/web/dist/js/pages/backtests.js`; ledger. | `tests/unit/apps/`; `tests/unit/contexts/strategy/`; `tests/unit/contexts/backtest/`; `node --check apps/web/dist/js/pages/backtests.js`; real browser QA with console/network checks. |
| `08` Final verification and delivery | Stage report `08-final-verification-and-delivery.md`. | Ledger; `docs/architecture/README.md` if regenerated; no production code unless repairing a verification-only defect with explicit scope. | Broad gates; browser/runtime proof; Prometheus proof for `scheduler_funding_catchup_*`; pre-ship gate; CI/deploy/Mac Studio proof only if delivery is in scope. |

## Tests Likely To Need Updates

| Stage | Likely tests |
| --- | --- |
| `01` | Funding parser/window fixtures; idempotent ClickHouse writer tests; scheduler all-futures-universe tests independent of `EnabledInstrumentReader`; non-due symbol skip tests; Prometheus no-symbol-label tests; monitoring asset tests. |
| `02` | Artifact manifest validation/load tests; funding array dtype and hash tests; coverage status tests; temp-root publish/load smoke. |
| `03` | Preflight normalization and request hash tests; runtime-defaults JSON shape tests; spot short-like rejection tests with `short_direction_requires_futures_market`; API route smoke fixtures. |
| `04` | Funding formula tests; gross/net metric preservation tests; candidate-pool and effective-ranking tests; performance benchmark evidence. |
| `05` | TP/SL exact exit boundary tests; `entry_time < funding_time <= exit_time`; same-bar TP/SL precedence; top/detail alignment; performance benchmark evidence. |
| `06` | DTO/read-model compatibility for missing funding fields; cache key includes `funding_manifest_hash`; route tests for top/variant/lazy-detail funding fields. |
| `07` | API launch rejection for paper and testnet spot short-like; scenario matrix policy tests; browser JS syntax and browser QA; old job readable but launch-blocked fixtures. |
| `08` | Broad regression gates plus runtime/browser/Prometheus evidence, not new feature tests unless final verification finds a narrow defect. |

## Traceability

| Source-plan requirement | Frozen stage |
| --- | --- |
| Funding storage and all-futures automatic catch-up | `01` |
| Dedicated exchange-discovered funding universe, separate from whitelist candle ingestion | `01` |
| Provider docs and `category=linear` mapping | `00`, implementation in `01` |
| Funding artifacts and `funding_manifest_hash` | `02` |
| Runtime defaults, funding readiness and request hash changes | `03` |
| Standalone `short` and futures-only preflight policy | `03`, UI/API completion in `07` |
| No-risk net-of-funding metrics and default net ranking | `04` |
| TP/SL funding using exact exit semantics | `05` |
| Result API, lazy detail, funding event overlay and cache identity | `06` |
| Strategy launch, scenario matrix and browser CJM | `07` |
| Full verification, docs closure and delivery readiness | `08` |

## Contract Impact

Stage 00 itself is docs-only.

| Dimension | Stage 00 impact | Frozen future impact |
| --- | --- | --- |
| Public API contract | `none` | `compatible-change` for additive backtest runtime-defaults/preflight/result fields; `breaking-change` for rejecting new spot short-like backtest/launch attempts. |
| Port contract | `none` | `compatible-change` through new funding source/store/coverage/artifact ports. |
| DTO schema | `none` | `compatible-change` for additive funding readiness/result fields; strict clients may need updates. |
| Persisted schema | `none` | `compatible-change` for new ClickHouse funding tables; `unknown` for Postgres/lazy cache until Stage `06` proves whether migration is needed. |
| Config schema | `none` | `compatible-change` for optional `scheduler.jobs.funding_rate_catchup` and funding catch-up config. |
| Request hash / cache key / persistence identity | `none` | `compatible-change` for new jobs because normalized `execution.funding` and `funding_manifest_hash` must affect identity; existing jobs must remain immutable. |
| Service-call auth/timeout/retry/error semantics | `none` | `compatible-change` through public unauthenticated Binance/Bybit market-data REST calls, bounded retries/backoff, idempotent replay of closed windows. |
| External side-effect idempotency and unknown-state semantics | `none` | `compatible-change`; funding writes must dedupe by stable market/symbol/funding time and rerun bounded windows instead of blind retry after unknown write state. |
| Logs, metrics, traces, audit, ledger, report or redaction semantics | `none` | `compatible-change`; new scheduler funding metrics and report fields must not include secrets and must not use `symbol` as a Prometheus label. |
| Alert or runbook semantics | `none` | `compatible-change`; Stage `01` adds funding freshness alerts and operator actions. |
| Benchmark / rollout gates | `none` | `compatible-change`; Stages `04` and `05` require performance evidence because funding touches backtest ranking/scoring paths. |
| Performance risk on verified hot path | `none` | `unknown` until Stage `04` and `05` benchmark funding post-pool adjustment on artifact-backed runtime input. |
| Browser-visible behavior | `none` | `breaking-change` intentionally for current invalid default/path: browser must no longer present spot long-short as normal and must show gross/net funding returns. |

## Old Docs That Could Mislead Future Stages

- `docs/runbooks/market-data-metrics-reference-ru.md` and
  `docs/runbooks/market-data-metrics.md` are current for candle scheduler
  metrics but do not include funding metrics yet.
- `docs/architecture/backtest/backtest-service-artifact-runtime-v1.ru.md`
  describes current artifact runtime families without funding; future stages
  must treat that as baseline, not as a denial of the new funding family.
- Existing browser docs/code imply `spot + long_short_reversal` is a normal
  default; Stage `07` must correct the UI/CJM after Stage `03` exposes the
  server-side compatibility policy.
- The closed `backtest-compute-acceleration-v1` family must remain historical
  benchmark context only, not an implementation surface for this funding line.

## Validation

Pre-edit validation:

- `python -m tools.docs.generate_docs_index --check`
  - Result: passed.
  - Output: `OK: /Users/daniildegtyarev/Projects/roehub.com/docs/architecture/README.md is up-to-date.`

Post-edit validation:

- `uv run python -m tools.docs.generate_docs_index --check`
  - Result: passed.
  - Output: `OK: /Users/daniildegtyarev/Projects/roehub.com/docs/architecture/README.md is up-to-date.`
- `python -m tools.docs.generate_docs_index --check`
  - Result: passed.
  - Output: `OK: /Users/daniildegtyarev/Projects/roehub.com/docs/architecture/README.md is up-to-date.`
- `git diff --check`
  - Result: passed.

## Cold-Head Self-Review

Subagent availability note: the current multi-agent tool policy does not allow
spawning a subagent unless the user explicitly asks for delegation, so Stage 00
uses the required cold self-review fallback.

Cold-head review: completed
Mode: cold self-review fallback
Review scope: Stage 00 report, stage ledger update, provider-doc refresh, future file manifests and contract classification.
Review instructions: architecture-review/references/cold-head-plan-prompt-pack-review.md
Verdict: Release after fixes
Blockers fixed: Initial draft risked treating local runtime unavailability as a Stage 00 blocker; report now records it as future real-boundary blocker/evidence for implementation stages. Initial manifest risked inheriting broad prompt-pack paths; report now freezes per-stage primary created/modified/test surfaces and requires explicit justification for files outside the manifest.
Local follow-up check: completed
Residual risks: No live provider REST smoke was run; local ClickHouse/API/web/scheduler boundaries were unavailable; future implementation stages must block or collect real-boundary evidence instead of accepting docs-only proof.

## Residual Risks

- Provider docs can drift; each provider-touching implementation stage must
  recheck official docs or run provider smoke before accepting API behavior.
- Local real-boundary services were unavailable in this environment. Stage `01`
  cannot be accepted without ClickHouse/provider/scheduler metrics proof or an
  explicit blocker, and stages `03`, `06`, `07` need local API/browser runtime
  proof.
- Funding performance remains unknown until stages `04` and `05` benchmark the
  post-pool funding adjustment on artifact-backed runtime inputs.
- `StrategySpecV1` direction storage remains a future-stage boundary: Stage
  `07` must prove direction metadata reaches launch/run readiness without a
  breaking spec change.
