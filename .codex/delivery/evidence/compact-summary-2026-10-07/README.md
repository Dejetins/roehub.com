# Compact Navigator summary cards — 2026-10-07

Changed only shared Navigator CSS (`apps/navigator-web/src/navigator.css`); no platform-web, data, API or streaming changes. Existing unrelated changes preserved.

Desktop geometry across Overview, Strategy operations and Backtest results:
- Summary row: 88 → 64px, same fonts and colors, two-line label allowance retained.
- Chart panel: 430 → 454px; plot element: 316 → 340px.
- Lower table position unchanged. Tablet grid uses the same summary-height variable.
- Narrow layout retains two rows of 64px cards with 8px gap (136px); chart panel 482px, plot348px. Expanded-chart sizing retains precedence.

Validation: `pnpm --dir apps/navigator-web build` passed, existing large-chunk warning. IAB localhost:20120: Overview Equity/Return/Drawdown, Strategy Price & executions/Equity/Drawdown, Backtests Equity/Price & trades/Drawdown exercised. Final DOM measurements on all three pages: metrics64 / chart454 / plot340. Backtests390×844: metrics136 / chart482 / plot348, no page horizontal overflow or card overflow. Desktop screenshots captured for all pages, mobile Backtests screenshot captured. Browser error log empty. No backend tests needed for CSS-only change. Local preview rebuilt/restarted, existing collector preserved.

Focused self-review; compatibility: compatible-change (visual density), API/storage none. Narrow Overview/Strategy and every data/localization combination were not individually exercised. Browser left on Overview, viewport override reset.
