# Backtest Futures Funding And Short Direction Policy v1 - Stage 04 No-risk Funding Adjustment

No-risk net-of-funding summary metrics, bounded candidate-pool adjustment and
effective ranking implementation record.

Date: 2026-06-22

Status: accepted.

Execution branch policy: `main` by default; do not create branches, worktrees or
local workflow folders unless the user explicitly requests them.

`User required before start: nothing`

## Scope

Requested scope:

- add funding PnL and net return summary metrics for futures no-risk variants;
- preserve gross `total_return_pct`;
- rank funding-enabled futures no-risk jobs by
  `total_return_pct_net_of_funding` by default;
- apply funding after base scoring on a bounded candidate pool of
  `max(top_n * 5, top_n + 100)`;
- persist `requested_ranking_metric`, `effective_ranking_metric`,
  `funding_adjustment_scope` and
  `funding_adjustment_exact_global_ranking=false`;
- keep top rows summary-only and leave TP/SL funding to Stage `05`.

Out of scope:

- no TP/SL funding;
- no browser/UI changes;
- no full trade tape serialization in top rows.

## Previous-Stage Gate

The Stage `04` prompt requires Stage `03` to be `accepted` in the stage
execution ledger before implementation starts.

Observed ledger state:

- Stage `03` status: `accepted`.
- Delivered main revision: `78646c42b08bb02ed9cedae4556e2f2a6d425ce8`.
- Accepted evidence includes full local publish gates, green GitHub CI/deploy
  workflows, Mac Studio checkout/runtime sync and authenticated route smoke for
  runtime-defaults plus preflight funding readiness.

Decision: previous-stage gate is satisfied. Stage `04` implementation may
proceed.

## File Manifest

Created:

- `docs/architecture/backtest/backtest-futures-funding-and-short-direction-policy-v1-stage-reports/04-no-risk-funding-adjustment.md`
- `src/trading/contexts/backtest/application/services/v2/no_risk_funding.py`

Modified:

- `src/trading/contexts/backtest/application/dto/no_risk_exact.py`
- `src/trading/contexts/backtest/application/dto/prepare_pools.py`
- `src/trading/contexts/backtest/application/ports/staged_runner.py`
- `src/trading/contexts/backtest/application/services/v2/__init__.py`
- `src/trading/contexts/backtest/application/services/v2/job_orchestration.py`
- `src/trading/contexts/backtest/application/services/v2/no_risk_exact.py`
- `src/trading/contexts/backtest/application/services/v2/preflight.py`
- `src/trading/contexts/backtest/application/services/v2/prepare_pools.py`
- `src/trading/contexts/backtest/application/services/v2/top_result_assembly.py`
- `src/trading/contexts/backtest/domain/entities/backtest_job.py`
- `docs/architecture/backtest/backtest-futures-funding-and-short-direction-policy-v1-stage-reports/backtest-futures-funding-and-short-direction-policy-v1-stage-ledger.md`
- `docs/architecture/README.md`
- `tests/unit/contexts/backtest/application/services/v2/test_backtest_preflight_service.py`
- `tests/unit/contexts/backtest/application/services/v2/test_no_risk_exact_scoring_service.py`
- `tests/unit/contexts/backtest/application/services/v2/test_top_result_assembly.py`
- `tests/unit/contexts/backtest/domain/entities/test_backtest_job_entities.py`

Deleted:

- none

## Funding Formula

For each no-risk open-position window, funding events are included when:

- `funding_time > entry_time`;
- `funding_time <= exit_time`.

The Stage `04` post-hoc funding formula is:

```text
funding_pnl_quote += -trade_direction * qty_base * mark_price * funding_rate
funding_return_pct = funding_pnl_quote / initial_cash_quote * 100
total_return_pct_net_of_funding = total_return_pct + funding_return_pct
```

Where:

- `trade_direction=1` for long and `-1` for short;
- positive funding rates make longs pay and shorts receive;
- negative funding rates reverse that direction;
- `qty_base` is derived from the same execution sizing path used by no-risk
  base scoring;
- funding is applied after base scoring, so funding PnL does not resize later
  trades in this stage.

If a funding record has a missing or invalid `mark_price`, Stage `04` falls back
to the latest available 1m execution close at or before the funding event and
adds deterministic warning code `funding_mark_price_fallback_used`. If no valid
artifact or fallback price is available, the event is skipped with
`invalid_funding_event_values`. Funding events whose quality byte is `0` add
`funding_event_data_quality_degraded`. Any fallback or degraded event quality
degrades the per-variant `funding_data_quality`.

## Ranking Contract

- Gross `total_return_pct` remains present and unchanged.
- New no-risk funding metric names:
  - `total_return_pct_net_of_funding`;
  - `funding_return_pct`;
  - `funding_pnl_quote`;
  - `funding_events_count`.
- New funding metadata fields persisted in the top-variant payload:
  - `funding_data_quality`;
  - `funding_warning_codes`;
  - `funding_included`;
  - `funding_adjustment_scope`;
  - `funding_adjustment_exact_global_ranking`;
  - `requested_ranking_metric`;
  - `effective_ranking_metric`;
  - `funding_candidate_pool_size`;
  - `requested_top_n`;
  - `funding_manifest_hash`.
- Funding-enabled futures no-risk requests keep
  `requested_ranking_metric=total_return_pct` when the user requested gross
  default ranking, but normalize
  `effective_ranking_metric=total_return_pct_net_of_funding`.
- Explicit ranking by `total_return_pct_net_of_funding` is accepted only for
  no-risk futures requests with `execution.funding.mode=include_when_futures`.
- Base scoring still uses gross metrics to build a candidate pool; funding is
  applied only to the bounded pool `max(top_n * 5, top_n + 100)`, then the pool
  is re-ranked by the effective metric and trimmed back to requested `top_n`.
- `funding_adjustment_exact_global_ranking=false` is always persisted because
  Stage `04` does not claim exact global net ranking.
- When funding arrays or required execution timestamp arrays are unavailable,
  top rows are annotated with `funding_included=false`,
  `funding_adjustment_scope=unavailable`, net return equal to gross return and
  deterministic warning code `funding_artifacts_unavailable`.

## Contract Impact

- Request normalization: compatible-change for new jobs. Normalized ranking now
  records both requested and effective primary metrics, so new request hashes can
  differ. Existing persisted jobs are not migrated or rehashed.
- Top-variant persistence: compatible-change. Summary metrics and payload JSON
  gain additive funding fields; existing fields and gross `total_return_pct`
  remain unchanged.
- Ranking behavior: compatible-change for futures no-risk funding jobs. The
  effective default ranking changes from gross return to net-of-funding return
  only when futures funding is included.
- Public/browser UI: none in this stage.
- Database schema: none. The existing JSON summary/payload storage carries the
  additive fields.

## Validation

Passed:

- `uv run python -m py_compile src/trading/contexts/backtest/application/services/v2/no_risk_exact.py src/trading/contexts/backtest/application/services/v2/no_risk_funding.py src/trading/contexts/backtest/application/services/v2/preflight.py src/trading/contexts/backtest/application/services/v2/job_orchestration.py src/trading/contexts/backtest/application/services/v2/top_result_assembly.py`
- Focused no-risk funding formula/candidate-pool tests, including missing
  `mark_price` fallback: passed.
- Focused top-result/preflight/entity regression set: `5 passed`.
- `uv run pytest -q tests/unit/contexts/backtest/application/services/v2/test_no_risk_exact_scoring_service.py -q`
  passed.
- `uv run pytest -q tests/unit/contexts/backtest/application/services/v2/test_job_orchestration.py tests/unit/contexts/backtest/application/services/v2/test_top_result_assembly.py tests/unit/contexts/backtest/application/services/v2/test_backtest_preflight_service.py tests/unit/contexts/backtest/domain/entities/test_backtest_job_entities.py`
  passed with `66 passed`.
- `uv run ruff check src/trading/contexts/backtest tests` passed.
- `uv run pyright src/trading/contexts/backtest tests` passed with `0 errors`.
- `uv run pytest -q tests/unit/contexts/backtest` passed with `383 passed`.
- `uv run ruff check .` passed.
- `uv run pyright` passed with `0 errors`.
- `uv run pytest -q -ra` passed with `1309 passed, 3 warnings`.
- `python -m tools.docs.generate_docs_index --check` passed.

## Performance Evidence

Local development micro-benchmark for the isolated funding scan:

- Command: inline `uv run python` micro-benchmark over
  `calculate_no_risk_funding_adjustment`.
- Environment: local development checkout, synthetic funding events, no artifact
  filesystem reads.
- Sample size: 25 measured samples after 5 warmups.
- Workload: 1,000 no-risk trade windows per sample, 7,499 synthetic funding
  events available, 2,500 events applied in the sample result.
- Result: median `3.155 ms`, p95 `3.296 ms`, min `3.079 ms`, max `3.380 ms`.

Mac Studio target-host acceptance evidence:

- Delivery:
  - initial Stage `04` implementation commit
    `75992d426b7219f33709069b0ef00ca43a4effb1`;
  - missing funding `mark_price` fallback commit
    `de3d625fa09ce7b6dc4354c1212262d7c698704a`;
  - GitHub CI `27978360294` and `27979671433` passed;
  - Deploy Backend `27978625433` and `27979954820` passed;
  - Publish App Image `27978625352` passed; later app-image run
    `27979954983` completed successfully with publish skipped by changes
    detector;
  - Deploy Web `27978637862` passed; later web run `27979954239` completed
    successfully with deploy skipped by changes detector.
- Target runtime:
  - Mac Studio checkout
    `/Users/daniildegtyarev/Projects/roehub.com` was fast-forwarded to
    `de3d625fa09ce7b6dc4354c1212262d7c698704a` and verified clean;
  - `/opt/roehub/app` backend deploy completed through the workflow bundle path;
  - `cd /opt/roehub/app && bash scripts/macos/smoke_prod.sh` passed after
    deploy, including service status checks, expected unauthenticated `401`,
    Redis `PONG` and Tailscale backend state `Running`.
- Artifact publish:
  - command boundary:
    `backtest-artifact-publish --config configs/prod/backtest_artifacts.yaml --exchange binance --market-type futures --symbol BTCUSDT --report-format json`;
  - active artifact after publish:
    `binance/futures/BTCUSDT`, `slot_b`, `slot_generation=2`;
  - artifact manifest hash:
    `be8fafdabdd25a74a89de8e3355bb8d4d8df3b01d76ef5ccfcd7ad491963aeb7`;
  - funding manifest hash:
    `5e0cbb9f0c7269fd3956d3b921009161d00df0156eb7a0120cc0929452840f7c`;
  - funding coverage status: `degraded`;
  - funding coverage policy: `degraded_with_warning`;
  - funding rows: `3` of `4` expected, `missing_event_count=1`,
    `reason_codes=["missing_leading_coverage"]`.
- Benchmark command boundary:
  - `ssh macstudio` with
    `SSH_AUTH_SOCK=$(launchctl getenv SSH_AUTH_SOCK)`;
  - `XDG_CACHE_HOME=/tmp/roehub-codex-cache ROEHUB_BACKTEST_ARTIFACTS_CONFIG=configs/prod/backtest_artifacts.yaml uv run python -`;
  - inline harness uses the existing prepare-pools, combo-planning and
    no-risk exact scoring services against the active artifact.
- Benchmark workload:
  - `binance/futures/BTCUSDT`, `15m`;
  - `time_range=[2026-06-20T00:00:00Z, 2026-06-22T19:30:00Z)`;
  - `direction_mode=long_short_reversal`, `risk.mode=none`;
  - arity `2`, `rows_per_indicator=12`, `top_n=20`;
  - `cartesian_combinations=144`;
  - warmups `3`, measured samples `12`;
  - baseline ranking: gross `total_return_pct`;
  - candidate ranking: requested gross `total_return_pct`, effective
    `total_return_pct_net_of_funding`.
- Benchmark result:
  - baseline wall median `1.081 ms`, p95 `1.264 ms`;
  - baseline exact-scoring median `0.210 ms`, p95 `0.316 ms`;
  - funding candidate wall median `26.947 ms`, p95 `32.123 ms`;
  - funding candidate exact-scoring median `0.194 ms`, p95 `0.240 ms`;
  - funding adjustment median `24.553 ms`, p95 `29.192 ms`;
  - heap capacity increased from `20` to bounded candidate pool `120`;
  - top results count stayed `20`;
  - all top-20 results had nonzero funding PnL;
  - top-20 total funding events count was `60`;
  - first result preserved gross `total_return_pct=0.43708173176763127`;
  - first result net return was
    `total_return_pct_net_of_funding=0.4498944891176918`;
  - first result funding adjustment was
    `funding_pnl_quote=1.2812757350060555`,
    `funding_return_pct=0.012812757350060556`,
    `funding_events_count=3`;
  - warning codes were deterministic:
    `funding_mark_price_fallback_used` and `missing_leading_coverage`;
  - `funding_adjustment_scope=bounded_candidate_pool`;
  - `funding_adjustment_exact_global_ranking=false`.

Acceptance performance status: accepted. The current Stage `04` code was on
`main`, CI/deploy/smoke were green, the target checkout was clean at the
delivered revision, and the Mac Studio benchmark used a real artifact-backed
futures input with funding arrays and nonzero funding adjustments.

## Cold-Head Review Receipt

Cold-head review: completed
Mode: cold self-review fallback
Review scope: Stage `04` report, stage ledger update, file manifest, validation
and Mac Studio performance-proof boundary.
Review instructions: architecture-review/references/cold-head-plan-prompt-pack-review.md
Verdict: Release after fixes
Blockers fixed: target-host performance acceptance is now satisfied after
direct-main delivery, green CI/deploy, Mac Studio runtime smoke, active
funding-degraded artifact publish and artifact-backed benchmark with nonzero
funding adjustments; report records exact file manifest, gross/net contract,
bounded candidate-pool ranking scope, mark-price fallback warning and residual
coverage risks.
Local follow-up check: completed
Residual risks: independent subagent review was not used because the available
subagent tool forbids spawning unless the user explicitly asks for delegation;
funding coverage for the accepted benchmark is degraded because the active
BTCUSDT artifact has only `3` of `4` expected funding events and required
mark-price fallback from 1m execution close prices.

## Residual Risks

- Accepted target-host benchmark used degraded funding coverage, not fully
  ready coverage: `3` of `4` expected BTCUSDT funding events were available and
  mark prices were derived through the deterministic 1m close fallback.
- Candidate-pool net ranking is intentionally not exact global net ranking;
  the persisted `funding_adjustment_exact_global_ranking=false` reflects this.
- TP/SL funding remains unimplemented and is owned by Stage `05`.
