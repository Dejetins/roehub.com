# Backtest Futures Funding And Short Direction Policy v1 - Stage 04 No-risk Funding Adjustment

No-risk net-of-funding summary metrics, bounded candidate-pool adjustment and
effective ranking implementation record.

Date: 2026-06-22

Status: implemented locally; acceptance performance evidence is blocked until
the Stage `04` changed code is present in the Mac Studio checkout and a
funding-ready futures artifact-backed benchmark can run there.

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

Invalid funding event values are skipped with deterministic warning code
`invalid_funding_event_values`. Funding events whose quality byte is `0` add
`funding_event_data_quality_degraded` and degrade the per-variant
`funding_data_quality`.

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
- Focused no-risk funding formula/candidate-pool tests: `3 passed`.
- Focused top-result/preflight/entity regression set: `5 passed`.
- `uv run pytest -q tests/unit/contexts/backtest/application/services/v2/test_no_risk_exact_scoring_service.py`
  passed with `53 passed`.
- `uv run pytest -q tests/unit/contexts/backtest/application/services/v2/test_job_orchestration.py tests/unit/contexts/backtest/application/services/v2/test_top_result_assembly.py tests/unit/contexts/backtest/application/services/v2/test_backtest_preflight_service.py tests/unit/contexts/backtest/domain/entities/test_backtest_job_entities.py`
  passed with `66 passed`.
- `uv run ruff check src/trading/contexts/backtest tests` passed.
- `uv run pyright src/trading/contexts/backtest tests` passed with `0 errors`.
- `uv run pytest -q tests/unit/contexts/backtest` passed with `383 passed`.
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

Mac Studio target-host evidence collected before acceptance:

- `ssh macstudio` works when invoked with
  `SSH_AUTH_SOCK=$(launchctl getenv SSH_AUTH_SOCK)`.
- Remote checkout path `/Users/daniildegtyarev/Projects/roehub.com` was clean
  before Stage `04` changed-code delivery.
- Remote non-login shell lacks `uv` on `PATH`; `zsh -lc` resolves
  `/opt/homebrew/bin/uv`.
- Mac Studio active futures artifact probe:
  - `binance/futures/BTCUSDT` resolves to `slot_a`;
  - manifest hash
    `0cd6537e0b5ef70415e99915e93ff8ad46630010033029a0abf7162da43f96a3`;
  - selected metadata reports `funding_coverage_status=None`;
  - the slot manifest does not declare a `funding` family;
  - `binance/futures/ETHUSDT`, `bybit/futures/BTCUSDT` and
    `bybit/futures/ETHUSDT` had no active `current.yaml`.
- Mac Studio current checkout can prepare artifact-backed futures BTCUSDT input
  from the canonical no-risk benchmark request:
  `prepared pools=1`, `rows=[196]`, `trade_T=219072`, `eval_T=219071`.
  That checkout is still at the Stage `03` code shape for this probe, so
  `execution_open_time_1m` and `execution_close_time_1m` are absent there.

Acceptance performance status:

- Not accepted yet. The prompt requires a Mac Studio benchmark for the current
  Stage `04` changed code with baseline before candidate measurement on
  artifact-backed runtime input.
- That benchmark was not run because the current changed code is local-only in
  this working tree and the active Mac Studio futures artifact slot does not
  declare a funding family. Running it as acceptance evidence requires
  direct-main delivery/checkout update and a funding-ready futures artifact
  input, or an explicitly accepted synthetic-funding benchmark boundary.

## Cold-Head Review Receipt

Cold-head review: completed
Mode: cold self-review fallback
Review scope: Stage `04` report, stage ledger update, file manifest, validation
and Mac Studio performance-proof boundary.
Review instructions: architecture-review/references/cold-head-plan-prompt-pack-review.md
Verdict: Release after fixes
Blockers fixed: report now separates local implementation from acceptance
performance evidence, records exact file manifest, preserves gross/net contract,
documents unavailable funding-artifact behavior and keeps Stage `04` status out
of `accepted`.
Local follow-up check: completed
Residual risks: independent subagent review was not used because the available
subagent tool forbids spawning unless the user explicitly asks for delegation;
Mac Studio acceptance benchmark is still blocked as described above.

## Residual Risks

- Stage `04` is implemented locally but not accepted in the ledger because the
  required Mac Studio changed-code performance benchmark has not run.
- Current Mac Studio production artifact state has futures BTCUSDT without a
  declared `funding` family; real net-of-funding runtime acceptance needs a
  funding-ready futures artifact slot.
- Candidate-pool net ranking is intentionally not exact global net ranking;
  the persisted `funding_adjustment_exact_global_ranking=false` reflects this.
- TP/SL funding remains unimplemented and is owned by Stage `05`.
