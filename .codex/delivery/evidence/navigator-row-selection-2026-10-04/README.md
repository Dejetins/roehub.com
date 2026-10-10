# Data Navigator catalog row selection — 2026-10-04

Scope: candidate `apps/navigator-web`, local `http://localhost:20120`.
Passive catalog row areas now select an instrument using the existing router selection path. Nested links, buttons (including SVG children), checkboxes and other controls retain their actions. Retained catalog/access-denied guards are preserved. Native symbol links remain the keyboard and modified-click navigation path; rows do not introduce additional tab stops. Existing selected/hover styling is preserved, with a pointer cursor on selectable rows.

Validation:
- `node_modules/.bin/vitest run src/data-page.test.tsx`: 37 passed. New cases cover passive cells/background, preserved table identity and URL filters, checkbox batch preservation, download focus without command submission, modified clicks, and retained catalog rejection. Initial new-test failures were test timing (catalog loaded after inspector) and an ambiguous label shared by two forms; corrected tests passed.
- `node_modules/.bin/tsc --noEmit`: passed.
- `node_modules/.bin/vite build`: passed; existing large chunk warning remains.
- `git diff --check`: passed.
- Root pnpm filter did not include this candidate package and ran no tests; all reported checks above were run directly from the candidate directory.

IAB, real local UI: selected ALICE through Market, ALGO through Stream, and ALICE through History. Search remained USDT and catalog scrollTop remained 845.5 through those selections. Both ALGO and ALICE checkboxes remained checked during batch selection. ALICE download icon opened its inspector and focused `data-download-from`; it did not submit a download. Enter on ALGO's native symbol link selected ALGO. Final passive Market click selected ALICE. No console errors observed. Screenshot: `alice-selected.png`.

Only the verified candidate web process was restarted to load its new asset manifest; no API, scheduler, download or streaming commands were issued by the test. Existing server reads were allowed. No provider/storage fault or performance testing is claimed by this UI change.

Focused self-review: no findings. Compatibility: browser selection `compatible-change`; API, persistence, worker and security contracts `none`. Local proof only; no commit or publication.
