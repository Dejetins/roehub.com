# Backtest Futures Funding And Short Direction Policy v1 - Stage Ledger

## Статус

Default execution branch: `main`.

Stage numbers are iteration boundaries in this ledger and in the stage reports;
they are not separate git branch boundaries. Do not create git branches,
worktrees, temporary checkouts, folders, stashes, or auxiliary files for this
prompt pack unless the user explicitly requests that exact workflow. Historical
remote branches `origin/codex/backtest-futures-funding-v1-stage-00`,
`origin/codex/backtest-futures-funding-v1-stage-01`, and
`origin/codex/backtest-futures-funding-v1` exist only as superseded delivery
artifacts from earlier iterations and are not the working model for new stages.

Stage `00` accepted как docs-only baseline freeze; evidence commit
`7dc0e726fc6babe8c101369a40a4119d5d23fd03` is retained in the unified branch
history. Stage `01` implementation is delivered to `main` through
`a77c001c375b101af4ddca51f63c7d6da60e21ea`; local gates, GitHub CI run
`27945620135`, backend deploy run `27945683469`, web deploy run `27945698512`
and app image publish run `27945683522` passed. Stage `01` is accepted after
Mac Studio `post_main_production_runtime_proof`: the deployed runtime wrote live
Binance/Bybit funding rows to ClickHouse and exported
`scheduler_funding_catchup_*` samples from the successful scheduler pass. This
proof was collected only after the target revision was on `main`, GitHub
Actions CI/deploy workflows were green, and `/opt/roehub/app` was synced to the
same main revision.

`User required before start: nothing`.

Source architecture document:
`docs/architecture/backtest/backtest-futures-funding-and-short-direction-policy-v1.md`.

Prompt pack:
`.codex/agents/generated/backtest-futures-funding-and-short-direction-policy-v1/`.

Stage `00` report:
`docs/architecture/backtest/backtest-futures-funding-and-short-direction-policy-v1-stage-reports/00-baseline-and-contract-freeze.md`.

## Правила приемки

- Каждый stage перед кодом перечитывает текущий prompt, этот ledger,
  `.codex/AGENTS.md`, главный архитектурный документ и только нужные code/docs
  entrypoints.
- Перед правками stage записывает narrowed file manifest в свой stage report.
- Stage считается `accepted` только после прохождения заявленных gates,
  обновления stage report, этого ledger и docs index when applicable.
- Runtime/browser-visible stages требуют runtime/browser evidence, а не только
  unit tests.
- Market-data, artifact and API stages require at least one real-boundary proof
  before `accepted`: ClickHouse migration/query, provider REST contract smoke,
  artifact filesystem publish/load, local API route smoke, browser flow, or a
  stage-specific explanation why the real boundary is not applicable yet.
- Delivery stages не смешивают чужие локальные изменения; scope должен быть
  проверен перед staging/commit/push.
- Для этого prompt pack default execution branch is `main`. Stage-итерации
  ведутся через этот ledger, stage reports и commits, а не через отдельные
  ветки, worktree-папки или другие локальные workflow-артефакты без явной
  просьбы пользователя.
- Закрытая линия `backtest-compute-acceleration-v1` не переоткрывается.

## Required Roehub context

- Business impact layer: see the source architecture document. Funding changes
  futures profitability interpretation and short-direction CJM; stages must
  preserve gross `total_return_pct`, add net-of-funding fields, and expose
  degraded funding readiness before strategy launch.
- Stage `00` service-call coverage: N/A for runtime changes. Stage `00` only
  rechecked official provider docs and local boundary availability; it did not
  add provider clients, ClickHouse writes, retries, authenticated browser flows
  or side-effecting runtime behavior.
- Conditional service-call coverage: Stage `01` must cover provider REST,
  ClickHouse and scheduler `/metrics` on Mac Studio target runtime; Stage `02`
  artifact filesystem/ClickHouse reads; Stage `03`, `06` and `07` API/browser
  routes on Mac Studio target runtime when acceptance evidence is recorded;
  Stage `08` delivery/runtime evidence when delivery is in scope. Pure local
  unit-only or Codex-local loopback evidence is not enough where these
  boundaries are touched. `target_host_readiness_pre_main` and
  `read_only_existing_runtime_smoke` can only describe reachability or old
  deployed behavior; `post_main_production_runtime_proof` requires target
  revision on `main`, green GitHub Actions/CI/deploy, deploy/sync into
  `/opt/roehub/app`, then runtime ClickHouse/metrics/log verification.
- Stage `03` service-call coverage: no new Binance, Bybit or ClickHouse calls
  are introduced in the API route. Funding readiness is read from the selected
  artifact manifest summary through the existing trusted filesystem artifact
  resolver. API target-host access is now restored: Mac Studio loopback
  `/health` returned `200`, `/auth/current-user` returned unauthenticated
  `401`, and unauthenticated `/backtests/runtime-defaults` returned Roehub
  error code `auth.required`. This is read-only existing-runtime evidence only;
  authenticated route smoke for the local Stage `03` changes still requires the
  changed code to be delivered and deployed before acceptance.
- Logging/redaction coverage: reports and runtime logs may include env var names,
  provider names, status codes and aggregate counts, but must not include DSNs,
  API keys, bearer tokens, ClickHouse passwords or secret-like values.

## Stage status

| Stage | Prompt | Status | Required evidence | Accepted evidence | Blockers / notes |
| --- | --- | --- | --- | --- | --- |
| `00` | `00-review-baseline-and-freeze-contract.md` | accepted | Baseline review, current code/doc manifest, external API re-check from official docs or provider smoke, contract classification, docs index check if changed. | 2026-06-22 Stage report created at `00-baseline-and-contract-freeze.md`; official Binance/Bybit docs rechecked; current scheduler topology, enabled-instrument scan pattern, Prometheus scrape baseline, runtime boundary availability and frozen Stage `01`-`08` file manifests recorded; evidence commit `7dc0e726fc6babe8c101369a40a4119d5d23fd03` is preserved in the unified branch history; `uv run python -m tools.docs.generate_docs_index --check`, `python -m tools.docs.generate_docs_index --check` and `git diff --check` passed after edits. | Accepted as docs-only baseline evidence. Main merge, Mac Studio deploy and runtime smoke are not applicable for Stage `00`; local ClickHouse/API/web/scheduler metrics boundaries were unavailable and remain future-stage real-boundary requirements. Historical `origin/codex/backtest-futures-funding-v1-stage-00` is superseded. |
| `01` | `01-implement-funding-storage-and-catchup.md` | accepted | ClickHouse funding DDL, dedicated exchange-discovered futures funding universe, funding source/store/use case, CLI dispatcher, automatic `market-data-scheduler` `funding_rate_catchup` job for all tradable Binance/Bybit futures instruments, mandatory interval metadata contract, funding-interval aligned due selection, Prometheus metrics, alert rules, runbook updates, idempotent catch-up tests, Bybit `linear` mapping test, interval metadata fallback/degraded tests, Mac Studio ClickHouse migration/query smoke, provider REST contract smoke or explicit network-unavailable blocker, Mac Studio `/metrics` proof for `scheduler_funding_catchup_*` on `127.0.0.1:9202` inside `ssh macstudio`. | Local implementation and local gates completed; Binance and Bybit provider REST smokes passed; code delivered to `main` at `a77c001c375b101af4ddca51f63c7d6da60e21ea`; CI `27945620135`, Deploy Backend `27945683469`, Deploy Web `27945698512` and Publish App Image `27945683522` passed; Mac Studio runtime proof wrote `rows_written=3661` with `failed=0`, ClickHouse `canonical_count=3663`, `raw_binance_count=1703`, `raw_bybit_count=1960`, and exported `scheduler_funding_catchup_*` rows-written, lag, last-success and universe metrics. | Accepted. Historical pre-fix scheduler errors remain in old log history only; no fresh funding scheduler failure was observed after the successful `2026-06-22 13:31:42 MSK` pass. |
| `02` | `02-implement-funding-artifact-family-and-coverage.md` | accepted | Funding artifact family, manifest hash, coverage reader, artifact publish/load tests, filesystem artifact publish/load smoke against a temp root, ClickHouse-backed coverage smoke against scheduler-maintained `canonical_funding_rates` when ClickHouse is available. | 2026-06-22 implementation added funding root manifest contract, `funding_manifest_hash`, funding array publish/load APIs, ClickHouse-backed canonical coverage reader, futures required/spot `not_applicable` validation, degraded coverage policy and CLI validation summary fields. Focused publish/load/coverage tests passed with `36 passed`; required ruff and pyright gates passed; existing artifact/backtest suite `uv run pytest -q tests/unit/contexts/backtest` passed with `373 passed`; affected app/market-data tests passed with `21 passed`; docs index check passed. Mac Studio `target_host_readiness_pre_main` read-only smoke reached the repo checkout, ClickHouse returned `Ok.`, and `market_data.canonical_funding_rates` returned current rows for multiple futures symbols. The Stage `02` artifact prerequisite scope was later delivered with Stage `03` in main commit `78646c42b08bb02ed9cedae4556e2f2a6d425ce8`. | Accepted. The prompt's exact pytest command includes missing directory `tests/unit/contexts/backtest_artifacts`; current artifact tests live under `tests/unit/contexts/backtest`, and that existing suite passed. |
| `03` | `03-implement-preflight-runtime-defaults-funding-readiness.md` | accepted | Normalized funding request, direction compatibility, preflight readiness fields, request hash tests, local API route smoke for runtime-defaults and preflight with funding-ready/degraded/unavailable/not_applicable fixtures; Mac Studio API evidence when target-runtime acceptance is recorded. | 2026-06-22 implementation added normalized `execution.funding`, standalone `short` in runtime defaults, `direction_market_compatibility`, preflight `funding_readiness`, spot short-like rejection with `short_direction_requires_futures_market`, additive artifact funding metadata, route smoke fixtures and request-hash coverage. Full publish gates passed with `uv run ruff check .`, `uv run pyright`, `uv run pytest -q -ra` (`1304 passed, 3 warnings`) and docs index check. Code was delivered to `main` at `78646c42b08bb02ed9cedae4556e2f2a6d425ce8`; GitHub CI `27963611975`, Deploy Backend `27963927905`, Publish App Image `27963927723`, and Deploy Web `27963927997`/`27963946189` passed. Mac Studio checkout was fast-forwarded to `78646c42`, `/opt/roehub/app` was synced through the backend bundle path, `uv sync --locked`, bootstrap, migrations and launchd reload completed, `smoke_prod.sh` passed, and authenticated route smoke proved runtime-defaults, spot/long-only preflight, futures/short funding readiness and spot/short rejection. | Accepted. Production futures/short preflight currently reports `funding_readiness.status=unavailable`; this is accepted Stage `03` warning metadata and is not a hard blocker. Existing jobs remain readable and immutable. |
| `04` | `04-implement-no-risk-funding-adjustment.md` | implemented locally; acceptance blocked | No-risk funding formula, candidate-pool adjustment, net metrics, focused tests, benchmark/performance evidence on artifact-backed runtime inputs. | 2026-06-22 local implementation added reusable no-risk funding calculation, net/gross summary metrics, bounded candidate-pool adjustment, requested/effective ranking metadata, unavailable-funding annotations, persistence payload fields and focused tests. Required local gates passed: `uv run ruff check src/trading/contexts/backtest tests`; `uv run pyright src/trading/contexts/backtest tests`; `uv run pytest -q tests/unit/contexts/backtest` with `383 passed`. Local micro-benchmark for the isolated funding scan measured median `3.155 ms` and p95 `3.296 ms` over 25 samples. | Not accepted. Mac Studio acceptance performance evidence is blocked: the changed Stage `04` code is local-only in this working tree, and the current Mac Studio active `binance/futures/BTCUSDT` artifact slot has no declared `funding` family (`funding_coverage_status=None`). Direct-main delivery plus a funding-ready artifact-backed benchmark, or an explicitly approved synthetic-funding benchmark boundary, is required before acceptance. |
| `05` | `05-implement-tp-sl-funding-adjustment.md` | planned | Exact TP/SL exit reuse, funding-aware TP/SL metrics, same-bar tests, benchmark/performance evidence on artifact-backed runtime inputs. | TBD | Do not fork divergent exit semantics. |
| `06` | `06-implement-results-api-lazy-detail-and-persistence.md` | planned | DTO/read-model fields, lazy cache identity, chart overlay funding events, API tests, local route smoke for top/variant/lazy-detail showing funding fields and cache identity. | TBD | Top rows remain summary-only; no full trade tape in top rows. |
| `07` | `07-implement-futures-only-short-policy-api-and-cjm.md` | planned | API validation, scenario matrix, UI compatibility/rerun flow, gross/net return UI (`total_return_pct` and `total_return_pct_net_of_funding`), funding degraded warnings, route tests, real browser QA evidence with console/network checks. | TBD | Replace old `testnet spot short`-only blocker with `short_direction_requires_futures_market`. |
| `08` | `08-final-verification-and-delivery.md` | planned | Broad gates, docs index, browser/runtime proof, pre-ship review, delivery evidence if publishing, Mac Studio checkout/runtime smoke when delivery is in scope. | TBD | Do not mark accepted without main/CI/deploy/Mac Studio evidence when delivery is in scope. |

## Delivery ledger

| Stage | Branch | Commit / SHA | PR | Local gates | Remote / runtime evidence | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| `00` | historical `codex/backtest-futures-funding-v1` | `7dc0e726fc6babe8c101369a40a4119d5d23fd03` | N/A | `uv run python -m tools.docs.generate_docs_index --check`, `python -m tools.docs.generate_docs_index --check` and `git diff --check` passed after edits. | Local probes unavailable: `127.0.0.1:9202`, `8123`, `8000`, `3000` refused connections. | Accepted docs-only baseline evidence; not main/Mac runtime deployed because Stage `00` has no runtime behavior. Historical branch artifacts remain superseded and should not be used for new work. |
| `01` | `main` | `a77c001c375b101af4ddca51f63c7d6da60e21ea` on `main` | N/A | Stage-local gates passed; final post-main full gates passed: `uv run ruff check .`, `uv run pyright`, `uv run pytest -q -ra` with `1285 passed`, docs index and `git diff --check`. | Provider REST smokes passed; GitHub CI `27945620135`, Deploy Backend `27945683469`, Deploy Web `27945698512` and Publish App Image `27945683522` passed. Mac Studio `post_main_production_runtime_proof` passed after `main` plus green Actions plus deploy/sync: remote `main` matches `origin/main`, runtime funding files are synced, API auth smoke on `/auth/current-user` returned `401`, ClickHouse ping returned `Ok.`, scheduler `funding_rate_catchup` completed with `instruments_total=1258`, `due=1221`, `ok=1221`, `skipped=37`, `failed=0`, `rows_written=3661`, and `scheduler_funding_catchup_*` metrics are exported. | Accepted. Historical `origin/codex/backtest-futures-funding-v1-stage-01` and `origin/codex/backtest-futures-funding-v1` remain superseded and should not be used for new work. |
| `02` | `main` | `78646c42b08bb02ed9cedae4556e2f2a6d425ce8` as prerequisite scope in the Stage `03` delivery commit | N/A | `uv run ruff check src/trading/contexts/backtest_artifacts src/trading/contexts/backtest tests`; `uv run pyright src/trading/contexts/backtest_artifacts src/trading/contexts/backtest tests`; focused funding tests `36 passed`; replacement existing artifact/backtest suite `uv run pytest -q tests/unit/contexts/backtest` with `373 passed`; affected app/market-data tests `21 passed`; full publish gates on the combined delivery passed. | Mac Studio `target_host_readiness_pre_main`: `ssh macstudio` reached `/Users/daniildegtyarev/Projects/roehub.com`; ClickHouse loopback returned `Ok.`; `market_data.canonical_funding_rates` returned current rows for futures symbols. Combined Stage `02`/`03` delivery reached `main`, green Actions and `/opt/roehub/app` sync at `78646c42`. | Prompt pytest path drift recorded because `tests/unit/contexts/backtest_artifacts` does not exist in this checkout. |
| `03` | `main` | `78646c42b08bb02ed9cedae4556e2f2a6d425ce8` | N/A | `uv run ruff check src/trading/contexts/backtest apps/api tests`; `uv run pyright src/trading/contexts/backtest apps/api tests`; `uv run pytest -q tests/unit/contexts/backtest tests/unit/apps/api` with `588 passed`; focused preflight/API suites; local TestClient route smoke for runtime-defaults and funding readiness statuses; full publish gates `uv run ruff check .`, `uv run pyright`, `uv run pytest -q -ra` with `1304 passed, 3 warnings`, and docs index check. | GitHub CI `27963611975`, Deploy Backend `27963927905`, Publish App Image `27963927723`, and Deploy Web `27963927997`/`27963946189` passed. Mac Studio checkout was fast-forwarded to `78646c42`; `/opt/roehub/app` was synced; `uv sync --locked`, prod bootstrap, migrations, launchd reload and `smoke_prod.sh` passed. Authenticated route smoke: `GET /backtests/runtime-defaults` returned `200` with `short` and funding defaults; spot/long-only `POST /backtests/preflight` returned `200` with funding `off/not_applicable`; futures/short returned `200` with funding `include_when_futures/unavailable`; spot/short returned `422` with `short_direction_requires_futures_market`. | Accepted. Temporary smoke sessions were revoked and no cookies, tokens or DSNs were printed. |
| `04` | `main` local checkout | local uncommitted Stage `04` diff | N/A | `uv run ruff check src/trading/contexts/backtest tests`; `uv run pyright src/trading/contexts/backtest tests`; `uv run pytest -q tests/unit/contexts/backtest` with `383 passed`; `python -m tools.docs.generate_docs_index --check`. | Mac Studio target-host probe reached `/Users/daniildegtyarev/Projects/roehub.com`; `zsh -lc` resolves `/opt/homebrew/bin/uv`; active `binance/futures/BTCUSDT` artifact resolves to `slot_a` manifest `0cd6537e0b5ef70415e99915e93ff8ad46630010033029a0abf7162da43f96a3`, but the slot manifest has no funding family. Current remote Stage `03` checkout can prepare artifact-backed futures input (`rows=[196]`, `trade_T=219072`, `eval_T=219071`) but does not contain the Stage `04` execution timestamp DTO fields. | Implementation is local only. Do not treat this as accepted or production/runtime proof. |
| `05` | TBD | TBD | TBD | TBD | TBD | TBD |
| `06` | TBD | TBD | TBD | TBD | TBD | TBD |
| `07` | TBD | TBD | TBD | TBD | TBD | TBD |
| `08` | TBD | TBD | TBD | TBD | TBD | TBD |

## Decision log

| Date | Decision | Rationale |
| --- | --- | --- |
| 2026-06-21 | Funding is a market-data type, not a backtest-local exchange client. | Keeps ingestion, rate limits, normalization and coverage in the existing market_data context. |
| 2026-06-21 | Funding freshness is owned by the existing `market-data-scheduler`, not by manual CLI alone. | The product needs all Binance/Bybit futures pairs to stay current automatically; the scheduler already owns periodic passes and Prometheus `/metrics` on `9202`. |
| 2026-06-21 | Funding download cadence is funding-interval aligned, not minute-level polling. | Most symbols settle every 8h, but interval can vary by exchange/symbol; scheduler wake-up only checks due work and skips non-due symbols. Interval metadata is mandatory: Bybit uses `fundingInterval`; Binance uses adjusted `fundingInfo` rows or an explicit standard 8h source for symbols absent from that adjusted-only response. |
| 2026-06-21 | Funding universe is exchange-discovered and separate from whitelist-enabled candle ingestion. | Current `ref_instruments` sync is whitelist-driven; funding must cover all tradable futures symbols without enabling candle ingestion for every exchange symbol. |
| 2026-06-21 | Prometheus funding metrics aggregate by exchange/market/status, not by symbol. | Keeps monitoring useful while avoiding high-cardinality series for all futures pairs. |
| 2026-06-21 | Bybit `market_type=futures` maps to external `category=linear` for v1. | Official Bybit v5 API uses `linear`/`inverse`, not `futures`. |
| 2026-06-21 | Standalone `short` must be added to runtime preflight before it is promised in UI/API. | Current runtime defaults expose only `long_only` and `long_short_reversal`. |
| 2026-06-21 | Futures funding jobs use net-of-funding as effective default ranking while preserving gross `total_return_pct`. | Prevents misleading futures top-N after funding is included. |
| 2026-06-21 | Spot short-like jobs are readable but not launchable; new short-like work requires futures. | Aligns backtest, strategy launch and live capability with real order semantics. |
| 2026-06-22 | Stage `00` freezes the current docs/code baseline and future-stage file manifest before implementation. | Implementation agents need current facts, provider-doc dates, real-boundary availability and narrow file boundaries before touching production code. |
| 2026-06-22 | Stage `00` baseline evidence promotes Stage `00` to `accepted` and unblocks the Stage `01` previous-stage gate. | Evidence commit `7dc0e726fc6babe8c101369a40a4119d5d23fd03` is retained in unified branch history; Stage `00` is docs-only, so main/Mac runtime proof is not applicable to this acceptance. |
| 2026-06-22 | Superseded Stage `01` pre-acceptance finding: local implementation alone was not sufficient runtime proof. | Funding storage, provider adapters, scheduler job, CLI, Prometheus rules and runbooks were implemented with local gates passing at `c26cef9e5f7405746566bd1d41da7121507d8709`; Mac Studio ClickHouse and scheduler baseline endpoints were reachable, but Stage `01` DDL/query and `scheduler_funding_catchup_*` export still had to be proven on target runtime before acceptance. Final accepted proof is recorded in the later `a77c001c` decision row. |
| 2026-06-22 | Runtime smoke loopback means Mac Studio loopback for this plan. | Codex-local `127.0.0.1` probes are diagnostics only; acceptance probes for ClickHouse, API/web, scheduler metrics, Prometheus and benchmarks must run through `ssh macstudio` unless a stage explicitly declares local-only evidence. |
| 2026-06-22 | Default future execution for this prompt pack is `main`; do not create branches or worktrees unless the user explicitly requests them. | Stage boundaries are tracked in prompts, reports and this ledger. The earlier `codex/backtest-futures-funding-v1-stage-00`, `codex/backtest-futures-funding-v1-stage-01`, and `codex/backtest-futures-funding-v1` branches are historical/superseded and must not be used as the model for later stages. |
| 2026-06-22 | Superseded Stage `01` branch-head smoke finding: branch sync was not changed-code production proof. | Commit `f94c8fa4a197626d45b3f2190d229d5cd9f9544f` fixed the Bybit non-positive `fundingInterval` crash and was pushed to `origin/codex/backtest-futures-funding-v1`; the then-current `/opt/roehub/app` lacked that parser fix and still had one `funding_rate_catchup` runtime error, so existing runtime evidence was only `read_only_existing_runtime_smoke`, not changed-code production proof. |
| 2026-06-22 | Superseded Stage `01` first-main delivery finding: deploy was green but final runtime proof still had to be collected. | Main commit `d14050b235807d60ae1d8cbf951bb651e40f1f45` fixed the ClickHouse `ILLEGAL_AGGREGATION` alias regression after the first post-main deploy attempt; GitHub CI `27944489784` and backend deploy `27944549900` passed. Mac Studio SSH/Tailscale reachability was unavailable during that proof attempt, so acceptance waited for the later `a77c001c` main revision, green deploy/sync and runtime proof. |
| 2026-06-22 | Stage `01` is accepted after final post-main runtime proof. | Main commit `a77c001c375b101af4ddca51f63c7d6da60e21ea` fixes the empty-history latest timestamp regression after the earlier ClickHouse alias and timezone fixes; GitHub CI/deploy workflows passed, Mac Studio SSH is restored, ClickHouse funding rows are present, `scheduler_funding_catchup_*` metrics are exported, and the deployed scheduler pass completed with `failed=0` and `rows_written=3661`. |
| 2026-06-22 | Stage `02` accepts funding artifacts as an artifact-runtime contract, not as scoring behavior. | Funding arrays are now published from `canonical_funding_rates`, loaded through the filesystem artifact loader, and represented in root manifest identity via `funding_manifest_hash`; scoring, preflight/API/UI warnings and lazy-detail cache behavior remain owned by later stages. |
| 2026-06-22 | Stage `03` is accepted after direct-main delivery and Mac Studio authenticated route smoke. | Runtime defaults and preflight now expose funding readiness and server-side direction compatibility; new spot short-like requests are rejected with `short_direction_requires_futures_market`. Commit `78646c42b08bb02ed9cedae4556e2f2a6d425ce8` is on `main`, GitHub CI/deploy workflows passed, `/opt/roehub/app` was synced, launchd services were reloaded, `smoke_prod.sh` passed, and authenticated Mac Studio route smoke covered `/backtests/runtime-defaults`, spot/long-only preflight, futures/short funding readiness and spot/short rejection. |
| 2026-06-22 | Stage `04` local implementation is not accepted without Mac Studio changed-code funding benchmark evidence. | Gross and net metrics, bounded candidate-pool adjustment, requested/effective ranking metadata and persistence payload fields are implemented locally with required local gates passing, but the active Mac Studio futures artifact slot lacks a `funding` family and the Stage `04` changed code has not been delivered to that checkout. The stage remains `implemented locally; acceptance blocked` until the required target-host benchmark boundary is satisfied. |

## Cold-head receipt

Cold-head review: completed
Mode: cold self-review fallback
Review scope: branch policy, prompt-pack branch metadata, Stage 00/01/02/03/04
reports, stage delivery ledger, previous-stage gates, Stage `04`
gross/net/ranking contract and Mac Studio target-runtime acceptance wording.
Review instructions: architecture-review/references/cold-head-plan-prompt-pack-review.md
Verdict: Release after fixes
Blockers fixed: Bybit category mapping; standalone `short` runtime gap; strategy direction storage gap; spot default + long-short UI contradiction; net ranking ambiguity; missing automatic all-futures funding scheduler mode; missing dedicated exchange-discovered funding universe; mandatory interval metadata contract; missing Prometheus metrics/alerts/runbook coverage; prompt pack no longer uses per-stage branches; Stage `03` separates pre-delivery diagnostics from accepted post-main Mac Studio authenticated route proof; Stage `04` now separates local implementation from blocked target-host performance acceptance and records the exact gross/net ranking contract.
Local follow-up check: completed.
Residual risks: historical remote `*-stage-00` and `*-stage-01` branches still exist as superseded artifacts; they were not deleted without explicit user confirmation. Historical runtime logs still contain pre-fix Stage `01` scheduler failures, but the fresh post-main run is successful. Stage `03` production futures/short preflight currently reports `funding_readiness.status=unavailable`; this is accepted warning metadata and not a hard blocker. Stage `04` target-host performance evidence is blocked until changed code and a funding-ready futures artifact-backed input are available on Mac Studio; provider API behavior must be rechecked by downstream implementation agents when their stages depend on new provider surfaces; Stage `07` must prove direction metadata reaches the live launch boundary.
