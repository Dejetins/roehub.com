# Iteration 10 UI integration browser QA

## Version

- Runtime surface: local `apps/web` with controlled same-origin mock API.
- URL: `http://127.0.0.1:8010/backtests`.
- Auth gate URL: `http://127.0.0.1:8011/backtests`.
- Browser tool: Playwright CLI.
- Viewports: `1440x1000`, `390x844`.

## Result

- Overall pass: `yes`.
- Protected route behavior: unauthenticated `/backtests` redirected through the existing login flow to `/api/auth/login?next=%2Fbacktests` in the mock auth surface.
- Runtime defaults loaded from `/api/backtests/runtime-defaults`.
- Preflight posted to `/api/backtests/preflight` and rendered normalized request, warning count, hashes, artifact metadata, and cost estimate.
- Create job posted to `/api/backtests/jobs`; selected job rendered `succeeded` progress with hashes and processed/total units.
- Top-N loaded from `/api/backtests/jobs/{job_id}/top`.
- `show trades` posted to `/api/backtests/jobs/{job_id}/variants/{variant_key}/trades` using the public readable `variant_key`.
- Lazy trades rendered cache/timing metadata, summary metrics, bounded first page `25` of `60` trades, and a nonblank canvas overlay from `backtest_chart_overlay_v1`.
- Mobile document overflow: `false`.

## Artifacts

- Desktop screenshot: `output/playwright/backtest-ui-desktop.png`.
- Mobile screenshot: `output/playwright/backtest-ui-mobile.png`.
- Playwright result: `canvasNonBlankPixels = 504000`, `visibleTradeRows = 25`, `totalTradeLabel = "1-25 of 60"`.

## Console / Network

- Authenticated UI flow console: `0` errors, `0` warnings.
- Authenticated Backtest API network calls: all `2xx`.
- Unauthenticated mock auth flow has the expected blocked mock `/api/auth/login?next=%2Fbacktests` `404` after the page performs the real login redirect handoff.

## Limitation

- QA used a controlled mock API because a real authenticated local or Mac Studio browser session was not available in the local automation context.
- Lazy trades payload does not include OHLC candle arrays; the UI rendered the accepted trade price/time overlay from `chart_overlay.markers` and `chart_overlay.segments`.
