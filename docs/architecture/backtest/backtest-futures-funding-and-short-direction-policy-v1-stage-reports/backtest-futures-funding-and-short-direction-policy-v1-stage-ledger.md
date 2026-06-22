# Backtest Futures Funding And Short Direction Policy v1 - Stage Ledger

## Статус

Active prompt-pack branch: `codex/backtest-futures-funding-v1`.

Stage numbers are iteration boundaries in this ledger and in the stage reports;
they are not separate git branch boundaries. Do not create new
`codex/backtest-futures-funding-v1-stage-XX` branches for this prompt pack unless
the user explicitly changes this branch policy. Historical remote branches
`origin/codex/backtest-futures-funding-v1-stage-00` and
`origin/codex/backtest-futures-funding-v1-stage-01` exist only as superseded
delivery artifacts from earlier iterations and are not the working branch for
new stages.

Stage `00` accepted как docs-only baseline freeze; evidence commit
`7dc0e726fc6babe8c101369a40a4119d5d23fd03` is retained in the unified branch
history. Stage `01` implementation is delivered to `main` through
`d14050b235807d60ae1d8cbf951bb651e40f1f45`; local gates, GitHub CI run
`27944489784` and backend deploy run `27944549900` passed. The stage remains
blocked because direct Mac Studio SSH/Tailscale reachability is unavailable from
the operator machine, so `post_main_production_runtime_proof` cannot yet prove
live `canonical_funding_rates` writes plus `scheduler_funding_catchup_*` samples
from the deployed `main` revision.

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
- Для этого prompt pack используется одна git branch:
  `codex/backtest-futures-funding-v1`. Stage-итерации ведутся через этот ledger,
  stage reports и commits внутри этой ветки, а не через отдельные ветки на
  каждый stage.
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
  boundaries are touched.
- Logging/redaction coverage: reports and runtime logs may include env var names,
  provider names, status codes and aggregate counts, but must not include DSNs,
  API keys, bearer tokens, ClickHouse passwords or secret-like values.

## Stage status

| Stage | Prompt | Status | Required evidence | Accepted evidence | Blockers / notes |
| --- | --- | --- | --- | --- | --- |
| `00` | `00-review-baseline-and-freeze-contract.md` | accepted | Baseline review, current code/doc manifest, external API re-check from official docs or provider smoke, contract classification, docs index check if changed. | 2026-06-22 Stage report created at `00-baseline-and-contract-freeze.md`; official Binance/Bybit docs rechecked; current scheduler topology, enabled-instrument scan pattern, Prometheus scrape baseline, runtime boundary availability and frozen Stage `01`-`08` file manifests recorded; evidence commit `7dc0e726fc6babe8c101369a40a4119d5d23fd03` is preserved in the unified branch history; `uv run python -m tools.docs.generate_docs_index --check`, `python -m tools.docs.generate_docs_index --check` and `git diff --check` passed after edits. | Accepted as docs-only baseline evidence. Main merge, Mac Studio deploy and runtime smoke are not applicable for Stage `00`; local ClickHouse/API/web/scheduler metrics boundaries were unavailable and remain future-stage real-boundary requirements. Historical `origin/codex/backtest-futures-funding-v1-stage-00` is superseded. |
| `01` | `01-implement-funding-storage-and-catchup.md` | blocked | ClickHouse funding DDL, dedicated exchange-discovered futures funding universe, funding source/store/use case, CLI dispatcher, automatic `market-data-scheduler` `funding_rate_catchup` job for all tradable Binance/Bybit futures instruments, mandatory interval metadata contract, funding-interval aligned due selection, Prometheus metrics, alert rules, runbook updates, idempotent catch-up tests, Bybit `linear` mapping test, interval metadata fallback/degraded tests, Mac Studio ClickHouse migration/query smoke, provider REST contract smoke or explicit network-unavailable blocker, Mac Studio `/metrics` proof for `scheduler_funding_catchup_*` on `127.0.0.1:9202` inside `ssh macstudio`. | Local implementation and local gates completed; Binance and Bybit provider REST smokes passed; code delivered to `main` at `d14050b235807d60ae1d8cbf951bb651e40f1f45`; CI `27944489784` and backend deploy `27944549900` passed. | Not accepted: direct Mac Studio SSH/Tailscale reachability is currently unavailable from the operator machine, so checkout/runtime parity, ClickHouse canonical rows, scheduler `scheduler_funding_catchup_*` samples and fresh scheduler logs cannot yet be verified for deployed `main`. |
| `02` | `02-implement-funding-artifact-family-and-coverage.md` | planned | Funding artifact family, manifest hash, coverage reader, artifact publish/load tests, filesystem artifact publish/load smoke against a temp root, ClickHouse-backed coverage smoke against scheduler-maintained `canonical_funding_rates` when ClickHouse is available. | TBD | Root manifest and explicit `funding_manifest_hash` must be coherent. |
| `03` | `03-implement-preflight-runtime-defaults-funding-readiness.md` | planned | Normalized funding request, direction compatibility, preflight readiness fields, request hash tests, local API route smoke for runtime-defaults and preflight with funding-ready/degraded fixtures. | TBD | Existing jobs must remain readable and immutable. |
| `04` | `04-implement-no-risk-funding-adjustment.md` | planned | No-risk funding formula, candidate-pool adjustment, net metrics, focused tests, benchmark/performance evidence on artifact-backed runtime inputs. | TBD | Must persist requested/effective ranking metadata. |
| `05` | `05-implement-tp-sl-funding-adjustment.md` | planned | Exact TP/SL exit reuse, funding-aware TP/SL metrics, same-bar tests, benchmark/performance evidence on artifact-backed runtime inputs. | TBD | Do not fork divergent exit semantics. |
| `06` | `06-implement-results-api-lazy-detail-and-persistence.md` | planned | DTO/read-model fields, lazy cache identity, chart overlay funding events, API tests, local route smoke for top/variant/lazy-detail showing funding fields and cache identity. | TBD | Top rows remain summary-only; no full trade tape in top rows. |
| `07` | `07-implement-futures-only-short-policy-api-and-cjm.md` | planned | API validation, scenario matrix, UI compatibility/rerun flow, gross/net return UI (`total_return_pct` and `total_return_pct_net_of_funding`), funding degraded warnings, route tests, real browser QA evidence with console/network checks. | TBD | Replace old `testnet spot short`-only blocker with `short_direction_requires_futures_market`. |
| `08` | `08-final-verification-and-delivery.md` | planned | Broad gates, docs index, browser/runtime proof, pre-ship review, delivery evidence if publishing, Mac Studio checkout/runtime smoke when delivery is in scope. | TBD | Do not mark accepted without main/CI/deploy/Mac Studio evidence when delivery is in scope. |

## Delivery ledger

| Stage | Branch | Commit / SHA | PR | Local gates | Remote / runtime evidence | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| `00` | `codex/backtest-futures-funding-v1` | `7dc0e726fc6babe8c101369a40a4119d5d23fd03` | N/A | `uv run python -m tools.docs.generate_docs_index --check`, `python -m tools.docs.generate_docs_index --check` and `git diff --check` passed after edits. | Local probes unavailable: `127.0.0.1:9202`, `8123`, `8000`, `3000` refused connections. | Accepted docs-only baseline evidence; not main/Mac runtime deployed because Stage `00` has no runtime behavior. Historical `origin/codex/backtest-futures-funding-v1-stage-00` remains superseded and should not be used for new work. |
| `01` | `main` plus prompt-pack branch `codex/backtest-futures-funding-v1` | `d14050b235807d60ae1d8cbf951bb651e40f1f45` on `main`; branch history through `f94c8fa4a197626d45b3f2190d229d5cd9f9544f` | N/A | Stage-local gates passed; post-main full gates passed: `uv run ruff check .`, `uv run pyright`, `uv run pytest -q -ra` with `1283 passed`, docs index and `git diff --check`. | Provider REST smokes passed; GitHub CI `27944489784`, Deploy Backend `27944549900`, Deploy Web `27944549893` and Publish App Image `27944549896` passed. Direct Mac Studio proof is blocked: `ssh -o ConnectTimeout=10 macstudio true` times out to `100.74.213.43`, ICMP has `100%` packet loss and `tailscale ping macstudio-daniil` times out. | Stage `01` remains blocked until post-main production runtime proof can verify checkout/runtime parity, ClickHouse canonical rows, scheduler `scheduler_funding_catchup_*` samples and no fresh scheduler tracebacks. Historical `origin/codex/backtest-futures-funding-v1-stage-01` remains superseded and should not be used for new work. |
| `02` | TBD | TBD | TBD | TBD | TBD | TBD |
| `03` | TBD | TBD | TBD | TBD | TBD | TBD |
| `04` | TBD | TBD | TBD | TBD | TBD | TBD |
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
| 2026-06-22 | Stage `01` remains blocked after local implementation because the unified prompt-pack branch is not delivered to Mac Studio runtime. | Funding storage, provider adapters, scheduler job, CLI, Prometheus rules and runbooks are implemented with local gates passing at `c26cef9e5f7405746566bd1d41da7121507d8709`; Mac Studio ClickHouse and scheduler baseline endpoints are reachable, but Stage `01` DDL/query and `scheduler_funding_catchup_*` export must be proven on target runtime before acceptance. |
| 2026-06-22 | Runtime smoke loopback means Mac Studio loopback for this plan. | Codex-local `127.0.0.1` probes are diagnostics only; acceptance probes for ClickHouse, API/web, scheduler metrics, Prometheus and benchmarks must run through `ssh macstudio` unless a stage explicitly declares local-only evidence. |
| 2026-06-22 | Use one git branch for the whole prompt pack: `codex/backtest-futures-funding-v1`. | Stage boundaries are tracked in prompts, reports and this ledger. The earlier `codex/backtest-futures-funding-v1-stage-00` and `codex/backtest-futures-funding-v1-stage-01` branches are historical/superseded and must not be used as the model for later stages. |
| 2026-06-22 | Stage `01` branch head is synced to Mac Studio, but acceptance remains blocked until post-main production runtime proof. | Commit `f94c8fa4a197626d45b3f2190d229d5cd9f9544f` fixes the Bybit non-positive `fundingInterval` crash and is pushed to `origin/codex/backtest-futures-funding-v1`; Mac Studio branch worktree is clean at that SHA. Current `/opt/roehub/app` lacks that parser fix and still has one `funding_rate_catchup` runtime error, so existing runtime evidence is only `read_only_existing_runtime_smoke`, not changed-code production proof. |
| 2026-06-22 | Stage `01` is now delivered to `main`, but still not accepted. | Main commit `d14050b235807d60ae1d8cbf951bb651e40f1f45` fixes the ClickHouse `ILLEGAL_AGGREGATION` alias regression after the first post-main deploy attempt; GitHub CI `27944489784` and backend deploy `27944549900` passed. Direct Mac Studio SSH/Tailscale reachability is currently unavailable, so the required runtime proof for funding metrics and canonical writes is still blocked. |

## Cold-head receipt

Cold-head review: completed
Mode: cold self-review fallback
Review scope: branch policy, prompt-pack branch metadata, Stage 00/01 reports,
stage delivery ledger, previous-stage gates and Mac Studio target-runtime
acceptance wording.
Review instructions: architecture-review/references/cold-head-plan-prompt-pack-review.md
Verdict: Release after fixes
Blockers fixed: Bybit category mapping; standalone `short` runtime gap; strategy direction storage gap; spot default + long-short UI contradiction; net ranking ambiguity; missing automatic all-futures funding scheduler mode; missing dedicated exchange-discovered funding universe; mandatory interval metadata contract; missing Prometheus metrics/alerts/runbook coverage; prompt pack now uses one active branch instead of per-stage branches.
Local follow-up check: completed.
Residual risks: historical remote `*-stage-00` and `*-stage-01` branches still exist as superseded artifacts; they were not deleted without explicit user confirmation. Stage `01` still requires direct Mac Studio runtime proof for scheduler `/metrics`, canonical funding writes and fresh logs after SSH/Tailscale reachability is restored. Performance evidence still required for funding candidate-pool stages; provider API behavior must be rechecked by implementation agents; Stage `07` must prove direction metadata reaches the live launch boundary.
