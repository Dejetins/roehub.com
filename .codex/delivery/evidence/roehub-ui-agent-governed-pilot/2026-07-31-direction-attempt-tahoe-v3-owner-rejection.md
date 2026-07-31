# Tahoe Backtests v3 owner rejection

- Date: `2026-07-31`.
- Checkpoint: `direction_selection`.
- Decision authority: product owner.
- Result: `rejected_by_product_owner`.

The fourth direction attempt was rejected before selection. The product owner identified eight
material omissions: no New Backtest action, no platform-wide status bar, no Drawdown chart mode,
no selected-variant next actions, no running-job progress or event notification bell, duplicated
Recent Trades content despite a Trades mode, non-Linear tab treatment, and oversized interface
controls. The lower analytical region also failed to prioritise monthly year/month results.

All v3 rasters are prohibited as visual, layout, component, or geometry inputs. The replacement
iteration is governed by
`docs/architecture/ui/roehub-tahoe-backtests-workstation-doctrine-v4.md`; every omission above is
an automatic rejection gate before product-owner presentation.
