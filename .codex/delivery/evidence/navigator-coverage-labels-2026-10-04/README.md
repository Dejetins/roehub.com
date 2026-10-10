# Navigator coverage and download labels — 2026-10-04

User-authorized changes in `apps/navigator-web`:
- Catalog cells always show explicitly labeled period coverage, including 0%. Active request progress and existing controls render separately below it; failed request controls remain available.
- Completed request inspector replaces the percentage/progress bar with a completion message and the request's exclusive end timestamp in UTC. Counters, range, attempt and events remain available. The download journal keeps its explicitly labeled job progress.
- Disabled streaming explains that new candles are not added automatically and stored history is preserved. Enabled and strategy-pinned states keep appropriate explanations.

Checks from the candidate directory: `node_modules/.bin/tsc --noEmit`, `node_modules/.bin/vitest run src/data-page.test.tsx src/download-progress.test.ts` (46 passed), and `node_modules/.bin/vite build` passed. Existing large-bundle warning remains. `git diff --check` passed. Regression cases verify simultaneous distinct coverage/request progress, completion without a competing 100% bar, retained coverage after completion, and off/on streaming copy. Pre-change failure run was not captured.

IAB on localhost:20120: real ACTUSDT completed request ends 2026-10-04 08:49 UTC; table and Stored history agree at 58.82% (847/1440) for the refreshed 24-hour window ending 18:42 UTC. Completion message, zero progress bars in the completed request section, disabled-stream explanation, readable layout and absence of document horizontal overflow verified. No console errors observed. Final screenshot: `act-coverage.png`. Active-job transition verified with component tests, not a new real download. No download/stream commands were submitted.

Verified local candidate process restarted only to load the built manifest. Prior row-selection changes preserved. Focused self-review: no findings. Compatibility: presentation `compatible-change`; API, persistence, data calculation, authorization and workers `none`. No real-infrastructure fault test required or claimed; no publication.
