# Backtest chart display menu — 2026-10-07

Navigator uses the same ChartDisplayMenu component as Strategy, replacing the bare Equity trade-exit and Price trade-marker checkboxes. Price menu toggles Trades and historical Entry price segments independently; entry segments use actual trade prices and span each trade's lifetime. Exact SL/TP prices are not supplied by the backtest trade DTO; disabled items explain this via titles rather than inventing levels from percent settings. Full SL/TP parity remains unimplemented pending actual level data.

Shared menu now supports disabled options and excludes them from keyboard navigation. API/security/storage untouched; platform-web untouched. Compatibility: compatible-change for menu UI, API none. Focused self-review.

Validation: typecheck/build passed (existing chunk warning). report-ui16 tests passed, including entry-segment removal independent of markers. Initial new test lacked ResizeObserver fixture; added the same browser stub needed by chart-instance, then reran successfully. IAB: Price menu opens; toggles remove trades and show entry segments; Escape closes/returns to trigger, reopen preserves choices. Error log empty. Screenshot menu.png. Mobile popup geometry unchanged/shared with Strategy; not separately re-exercised this turn. Local web restarted, collector preserved. No publication.

Button-size correction: shared Strategy toolbar rules now also apply to report-chart-ranges, including expanded state. Build passed. IAB computed dimensions match exactly on Strategy and Backtests:28×28px, padding0, SVG width15.3984375px. Screenshot button-size-fixed.png. Local preview updated.
