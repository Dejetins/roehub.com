# Backtests UI iteration log

## Acceptance — 2026-09-11

The product owner accepted the current result as satisfactory and authorized
publication of the complete Backtests implementation through a technical branch
merged into `main`, followed by deletion of that branch.

Accepted baseline:
- Compact date-only configuration embedded as a disclosure inside Jobs; draft
  preservation, separate reset and retained history context.
- Result-focused workspace with compact clickable variant ranking, summary
  cards, metrics, trades and a year-by-month P&L matrix; technical job metadata
  is collapsed below the report. No redundant symbol-statistics tab.
- Apache ECharts equity, drawdown and candlestick price/trade views. Price bars
  come from the job-pinned artifacts, with UTC 1m/5m/15m/30m/1h/4h/1d rollups.
  The verified local example uses actual Binance archives; earlier synthetic
  examples remain explicitly identified as synthetic.
- Overview can expand to the browser viewport while retaining chart switching,
  zoom and timeframe. Escape and the icon control collapse it. Desktop control
  is 28px, aligned with chart switches; narrow-screen touch size is retained.
- One persisted animation-speed preference. Local content transitions update
  immediately; geometry transitions have an opaque surface and no parent ghost
  image. Off and system reduced-motion remain supported.

This acceptance freezes the current UI iteration, not every future Backtests
requirement. `local_journey_verified=true`; `target_role_cutover_ready=false`.
Target organization/role authorization and default cutover remain separate.
No production runtime is selected by the repository adapter; Git publication
must not be described as production deployment.

## Iteration history

| Period | Accepted change | Evidence |
| --- | --- | --- |
| 2026-09-08 | Complete local configure/preflight/submit/execution/results journey | [S6 completion](../../../../.codex/delivery/evidence/ROEHUB-BACKTESTS-CLIENT-2026-09-08.md) |
| 2026-09-08–10 | Compact editor, date-only fields, Jobs disclosure and result workspace | [Iteration evidence](../../../../.codex/delivery/evidence/roehub-backtests-compact-2026-09-08/workspace-tabs.md) |
| 2026-09-10 | Actual Binance candles, artifact-pinned rollups and interactive charts | Same evidence, real-market section |
| 2026-09-10 | Expandable Overview, shared motion, icon-only control and density correction | Same evidence, motion sections |
| 2026-09-11 | Product-owner acceptance and authorized merge publication | This entry; [publication record](../../../../.codex/delivery/evidence/BACKTESTS-UI-PUBLICATION-2026-09-11.md) |

The dates describe recorded local iterations; individual evidence entries are
historical observations, not claims that every later revision was rechecked by
the same run. The publication record identifies the checks for the shipped code.
