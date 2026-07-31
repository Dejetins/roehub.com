# Linear-black Backtests Workbench v6 review

## Boundary

- Date: `2026-07-31`.
- Checkpoint: `direction_selection`.
- Status: `awaiting_product_owner_decision`.
- Doctrine: `docs/architecture/ui/roehub-linear-black-backtests-workbench-doctrine-v6.md`.
- Candidate: `images/2026-07-31-linear-black-workbench-v6.png` (`1672 × 941`).
- Generated with the built-in image generation tool.
- The raster is direction-review evidence only and does not prove Figma structure or runtime.

## Bounded generation audit

1. The initial v6 edit implemented the product changes but was automatically rejected because the
   KPI labels remained above their values and the bottom status remained a tall rounded band.
2. Repair 1 changed only the KPI and bottom status strips. Both now use a compact single baseline.
3. One bounded repair remains available for this feedback cycle if the product owner identifies a
   new local defect; no repair is inferred or run without a named failure.

## Owner-comment gate

| Owner comment | Observed result |
|---|---|
| bottom status is too tall and under-informative | pass: narrow line with API, workers, running jobs, queue, lag, last sync, and exchange |
| remove `Symbol` | pass |
| remove `Run strategy`; retain creation only | pass: `Create strategy` is the sole strategy action |
| move Filters to Jobs | pass |
| add queued state | pass: `Queued 1`, `Queued · position 1` |
| replace duplicative lower-right information | pass: `Parameters` with DEMA, RSI, direction, fee, slippage |
| compress the tall KPI block to one narrow row | pass: six inline label/value pairs |
| remove `Notebooks` | pass |
| remove `Market` | pass |
| move `New backtest` into Jobs in place of plus | pass |

## Retained functional gate

- notification bell with unread count: pass;
- two running jobs with `68%` and `24%`: pass;
- three completed jobs: pass;
- `12 variants` with ten visible compact rows: pass;
- separate `Drawdown` mode: pass;
- Save and Export CSV: pass;
- Equity chart with Buy/Sell markers: pass;
- Monthly returns: pass;
- no Recent trades duplication: pass;
- neutral-black Linear visual system: pass by visual observation.

## Proof boundary

The inspection proves only the visible raster inventory and named visual conditions. It does not
prove exact pixel dimensions, interactive states, permissions, accessibility, localization,
responsive behavior, Figma construction, or production data. Product-owner acceptance remains
required.
