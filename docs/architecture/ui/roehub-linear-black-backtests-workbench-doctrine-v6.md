# Roehub Linear-black Backtests Workbench doctrine v6

This doctrine continues the product-owner-preferred Linear-black Workbench direction and applies
the ten bounded product and density corrections supplied against the v5 raster.

## Status and authority

- Status: `preferred_direction_refinement_required`.
- Direction ID: `linear_black_backtests_workbench_v6`.
- Product-owner request date: `2026-07-31`.
- Edit target: generated v5 Workbench raster identified by the product owner.
- Visual authority: the v5 black neutral, radius, surface, and unified-workspace system, except where
  this doctrine explicitly changes anatomy or density.
- Product authority: the ten owner comments in the current review message override prior
  speculative navigation, modes, actions, and summaries.
- Successor: `docs/architecture/ui/roehub-linear-black-backtests-workbench-doctrine-v7.md` applies
  the product-owner-requested control-lattice, refresh, chart-type, and platform-status repair.

## Product corrections

1. The bottom platform row becomes a narrow `24-28px` status strip and uses the reclaimed width for
   useful live context: API, workers, queued jobs, running jobs, data lag/freshness, last sync, and
   exchange connection.
2. Remove the `Symbol` analysis mode.
3. Remove `Run strategy`; this screen can only `Create strategy`.
4. Move Filters from the global header into the Jobs column header.
5. Add a distinct `Queued` job group and one representative queued job. The exact English product
   state label is `Queued`.
6. Replace the duplicative Risk summary with a non-duplicative `Parameters` block containing
   strategy inputs such as DEMA, RSI, direction, fee, and slippage.
7. Compress the KPI block into one narrow single-baseline row. Each item reads `Label Value`
   horizontally; no metric uses a label-above-value stack.
8. Remove `Notebooks` from global navigation.
9. Remove `Market` from global navigation.
10. Move `New backtest` from the global header to the Jobs header, replacing the previous plus-only
    action with a clear compact verb-first label.

## Symmetric density system

- Global header: `44-46px`, containing only page identity, command/search, Refresh, `Auto 15s`, and
  the notification bell.
- Jobs header: `36px`, with `Jobs`, count, `Filters`, `New backtest`, and overflow sharing the same
  vertical centre and `28px` control height.
- Analysis identity/action row: `36-40px`, with identity, date context, Save, Export CSV,
  `Create strategy`, and overflow on one baseline.
- Analysis mode row: `28px` independent pills for `Overview`, `Equity`, `Drawdown`, `Monthly`, and
  `Trades` only.
- KPI row: `34-38px`, with Return, Sharpe, Drawdown, Profit factor, Win rate, and Trades each
  represented inline on one baseline.
- Job group heading: `20-22px`; job row: `34-38px`; variant row: `32-34px`.
- Bottom platform status: `24-28px`.
- Text and icons are optically centred in every row. Top/bottom padding should differ by no more
  than `1px` after optical adjustment.

## Jobs state fixture

- `Running 2`: `trend-follow-4h-btc` at `68%`; `mean-revert-1h-eth` at `24%`.
- `Queued 1`: `momentum-30m-sol` with compact state `Queued · position 1` and no fake progress.
- `Completed 3`: selected `dema-1h-long-short-a1b2c3` plus two completed jobs.
- Jobs count is `6`.
- The bottom status includes `Jobs 2 running` and `Queue 1 waiting`.

## Selected variant composition

- Modes: `Overview`, `Equity`, `Drawdown`, `Monthly`, `Trades`.
- Actions: `Save`, `Export CSV`, `Create strategy`, overflow. No launch or run action exists.
- KPI row remains the authoritative performance summary.
- Lower strip contains Monthly returns and Parameters.
- Parameters contains `DEMA 20 / 50`, `RSI 14 / 55`, `Direction Long + short`, `Fee 0.10%`, and
  `Slippage 0.02%`; it must not repeat Return, Sharpe, Drawdown, Profit factor, Win rate, or Trades.
- Equity view contains no Recent trades block.

## Navigation inventory

The visible global navigation contains only `Backtests`, `Strategies`, `Data`, `Signals`,
`Reports`, `Alerts`, and `Settings` plus user identity. `Notebooks` and `Market` are forbidden.

## Automatic rejection gate

Reject before owner review if the specimen:

- retains `Symbol`, `Run strategy`, `Notebooks`, or `Market`;
- retains Filters or New backtest in the global header;
- omits the `Queued` state or misspells it;
- uses a tall two-line KPI block;
- retains Risk summary or repeats KPI values in the lower-right block;
- makes the bottom platform status taller than a compact single line;
- omits useful queue/running/data status from the bottom line;
- reintroduces stacked label/value anatomy where inline content fits;
- loses any previously accepted chart, monthly, jobs, variants, notification, or status content;
- changes the accepted black neutral Linear visual system.

## Review boundary

The v6 output is one revised raster for the same preferred direction. It remains direction evidence
only and does not accept a Figma component, exact geometry, or runtime implementation.
