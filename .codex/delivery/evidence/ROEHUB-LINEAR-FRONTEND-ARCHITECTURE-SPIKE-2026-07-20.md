---
evidence_id: ROEHUB-LINEAR-FRONTEND-ARCHITECTURE-SPIKE-2026-07-20
ticket_id: ROEHUB-LINEAR-FRONTEND-ARCHITECTURE-SPIKE-2026-07-20
status: accepted
verdict: proceed
observed_at: 2026-07-22T14:41:58Z
visual_design_status: rejected_by_product_owner
visual_source_role: prohibited
---

# React coexistence architecture spike evidence

## Later product-owner design decision

On `2026-07-22`, after inspecting the visible prototype, the product owner
rejected its composition, styling, component anatomy, interface copy, and
screenshots as Roehub UI design. No product owner or designer had approved that
visual result when this technical ticket was marked `accepted`.

The historical technical measurements below remain factual. The visual layer
is `not_a_design_source` and must not be inherited by Figma foundations or the
production React shell. Only route isolation, SSR rollback, state/transport
seams, tests, measurement harness, and dependency observations may be reused.

## Verdict

`proceed` for the separately governed Figma vNext foundations and production
React application-shell tickets.

The bounded `PROTOTYPE` proves that React + TypeScript + Vite can coexist on the
same origin with the unchanged FastAPI/Jinja gateway, that the SSR return path
remains real and reversible, and that the proposed state and transport split
meets the accepted initial client-side performance budgets on the declared
local hardware.

This is not acceptance of a production shell or a real API golden slice. Figma,
backend routes, authorization rules, persistence, trading, the public site,
runtime packaging, release, and deployment were not changed.

## Prototype boundary and run command

One command from the repository root builds and starts the local proof:

```bash
npm run prototype
```

The resulting boundary is:

| Route | Owner in the proof | Observation |
|---|---|---|
| `/__prototype/react/` | built Vite client | HTTP `200`, `X-Roehub-Prototype: true`, visible `PROTOTYPE` marker |
| `/__prototype/api/backtests` | safe local REST fixture | typed read-only projection with bounded deterministic latency |
| `/__prototype/events` | safe local SSE fixture | server event updates the TanStack Query cache |
| `/backtests?from=react-prototype` | unchanged `apps.web.main.app` | current FastAPI/Jinja template rendered with `[data-backtests-root]` and `Backtests` heading |

`apps/platform-web/prototype_gateway.py` declares the prototype routes before
mounting the current SSR application at `/`. The browser followed the normal
anchor from React to `/backtests`, observed the Jinja DOM, then Browser Back
returned to `/__prototype/react/`. No production route was edited or shadowed.
The fixture current-user decision remains in the server-side adapter; the
client has no role, capability, allow, or deny decision.

## Dependency versions

Runtime dependencies are pinned in `package-lock.json`:

| Dependency | Version |
|---|---:|
| React / React DOM | `19.2.8` / `19.2.8` |
| TypeScript | `7.0.2` |
| Vite / React plugin | `8.1.5` / `6.0.4` |
| MobX / mobx-react-lite | `6.16.1` / `4.1.1` |
| TanStack Query | `5.101.4` |
| styled-components | `6.4.4` |
| Vitest / jsdom | `4.1.10` / `29.1.1` |
| Testing Library React / user-event / jest-dom | `16.3.2` / `14.6.1` / `7.0.0` |
| Playwright | `1.61.1` |
| Prettier | `3.9.6` |

Existing Python gateway dependencies used by the isolated wrapper were
FastAPI `0.115.5`, Jinja2 `3.1.4`, HTTPX `0.27.2`, and Uvicorn `0.32.0`.
Toolchain: Node `v24.18.0`, npm `11.16.0`, Python `3.12.2`.

## State, transport, theme, and panel evidence

| Boundary | Evidence |
|---|---|
| MobX local authority | `UiStore` contains only `theme`, `panelWidth`, and `selectedBacktestId`; unit and integration assertions passed. |
| Query remote authority | REST snapshots live under `BACKTESTS_QUERY_KEY`; SSE uses `queryClient.setQueryData`; `UiStore` has no remote row collection. |
| REST cancellation | Query supplies the exact `AbortSignal` to `fetch`; unit test observes identity and abort; browser cancellation increments the real abort observer. |
| SSE update | live `EventSource` event changes revision/source and row status in Query cache; integration and Chrome assertions passed. |
| Authorization | client renders the server projection and never derives an allow/deny result; the SSR fixture principal is resolved server-side. |
| Theme | `abyss`, `graphite`, `frost`, and `paper` change `data-theme` and semantic CSS custom properties without reload. |
| Panel resize | `208-320px` bounds, `240px` reset, pointer drag, double-click reset, ArrowLeft/ArrowRight `8px` steps, and separator ARIA values passed in Chrome. |

No trading operation exists in the fixture. Data and the fixed local session are
disposable and non-production. No production credential was read or entered.

## Browser evidence

- Target: Vite production build served by the isolated FastAPI composition at
  `http://127.0.0.1:4173`.
- Browser binary: system Google Chrome `150.0.7871.181`, headless channel;
  reported UA `HeadlessChrome/150.0.0.0`.
- Viewports: `1440x900` for the full interaction/performance journey and
  `820x900` for the supported narrow-desktop smoke.
- React phase: zero console errors and zero unexpected failed requests; the one
  cancelled REST request is the expected cancellation proof.
- Auth/data: only safe deterministic fixture data and a fixed non-production
  cookie. Output artifacts are ignored and not tracked.
- Final command: `npm run test:browser` -> `2 passed (13.3s)`.

Sanitized local artifacts:

| Artifact | SHA-256 |
|---|---|
| architecture trace `output/playwright/linear-frontend-architecture-spike/test-results/architecture-proves-bounde-e6ecc-the-current-SSR-return-path/trace.zip` | `2923e6dd289bbe92cb735cb253436e0109974d4dcd85603b6dde30ea83f0800b` |
| performance trace `output/playwright/linear-frontend-architecture-spike/test-results/performance-measures-clien-833dd-ong-tasks-and-frame-cadence/trace.zip` | `3582cdff9be463e0d48665fefd088550023be8505e5ef8911d4ef032c09524ed` |
| `theme-abyss-1440x900.png` | `57e64c9ec64f3aae83c08e1aa86806d2ca7c4b3ef8f2aee823bdaa87017c7b4a` |
| `theme-graphite-1440x900.png` | `9406b188712611cea05b3cb338ea548150f0f828137dbf2b6a64d0a92a37adf3` |
| `theme-frost-1440x900.png` | `676ac7b0022a9984bddf6cf8ea187ffb5962dcc18d09c2b32ebc07ba36bfa8c0` |
| `theme-paper-1440x900.png` | `c608e3cb12b421357de5a9994e8f3be9297a0b7943070af41f82084b8519d3f3` |
| `responsive-820x900.png` | `e9c6be18f0a1ff739b702b0eda2739090e2bf2b7ef3b61b271547f43df660c84` |

## Performance method and results

Hardware: MacBook Pro `Mac15,6`, Apple M3 Pro with 11 CPU cores
(`5` performance + `6` efficiency), `18 GB` RAM, macOS `15.7.4`.

Method: one warm local headless-Chrome journey at `1440x900`; `30` sequential
REST refreshes with deterministic `36ms` fixture delay, live `120ms` SSE,
`32` theme clicks, and `32` keyboard resizes. PerformanceObserver collected
event/INP candidates and long tasks; requestAnimationFrame recorded paint and
frame cadence. Network topology was loopback. Backend cache and real API latency
are not applicable. Exact JSON is retained locally at
`output/playwright/linear-frontend-architecture-spike/performance-results.json`.

| Metric (ms) | Samples | p50 | p75 | p95 | Accepted ceiling |
|---|---:|---:|---:|---:|---:|
| interaction acknowledgement-to-paint | 30 | `6.0` | `6.7` | `7.3` | p95 `<=100` |
| interaction-to-REST dispatch | 30 | `0.1` | `0.2` | `0.2` | p95 `<=50` |
| REST response-to-paint | 30 | `6.9` | `7.5` | `8.4` | p95 `<=200` |
| SSE receipt-to-paint | 31 | `5.7` | `7.6` | `8.2` | p95 `<=200` |
| theme change-to-paint | 32 | `6.9` | `7.0` | `8.0` | interaction p95 `<=100` |
| resize-to-paint | 32 | `4.0` | `5.8` | `8.0` | interaction p95 `<=100` |
| event duration / INP observations | 249 | `16` | `16` | `24` | INP `<=200` |

The representative-journey INP candidate (p98 event duration) was `24ms`.
There were `0` long tasks over `50ms`. Frame intervals over `439` samples were
p50 `8.3ms`, p75 `8.4ms`, p95 `9.0ms`; median cadence was `120.48fps`, `100%`
of intervals were `<=20ms`, and `0` intervals exceeded `25ms`.

## Dependency and build cost

`npm run size` measures installed bytes for direct runtime packages and raw/gzip
Vite production assets:

- direct runtime dependency directories: `15,237,160 B` installed;
- lockfile package entries: `186`;
- JavaScript: `324,764 B` raw / `100,297 B` gzip;
- CSS: `3,477 B` raw / `1,332 B` gzip;
- total built JS+CSS: `328,241 B` raw / `101,629 B` gzip.

Installed package bytes describe local dependency cost, not shipped transfer
size. The production-shell ticket should retain a bundle budget and introduce
route splitting only when real shell composition demonstrates the need.

## Verification

| Command | Result |
|---|---|
| `npm run check` | pass: TypeScript, Prettier, `5/5` Vitest unit/integration tests, Vite build, size report |
| `npm run test:browser` | pass: `2/2` real-Chrome architecture/performance journeys with traces |
| `uv run ruff check .` | pass |
| `uv run pyright apps/platform-web/prototype_gateway.py` | pass: `0 errors` |
| `git ls-files -z '*.py' \| xargs -0 uv run pyright apps/platform-web/prototype_gateway.py` | pass: tracked checkout plus new gateway, `0 errors` |
| `python .codex/hooks/tests/run_tests.py` | pass: all `11` active hook regressions |
| `python -m tools.delivery.validate_roehub_delivery_model` | pass |
| `python -m tools.docs.generate_docs_index --check` | pass |
| `python -m tools.docs.generate_project_map --check` | pass: all `5` generated artifacts current |
| `uv run python -m tools.jobs.generate_schemas --check` | pass |
| `python tools/release/oss_metadata.py --check` | pass: `3` artifacts |
| `uv run pytest -q -ra tests/test_smoke.py tests/unit/apps tests/unit/infra tests/unit/platform tests/unit/shared_kernel tests/unit/tools` | pass: `746 passed`, `4` pre-existing HTTPX deprecation warnings |
| `git diff --check` | pass |

A bare local `uv run pyright` additionally traversed ignored
`local_artifacts/rl_trading/**` and reported `149` unrelated existing errors.
Those ignored files are absent from GitHub checkout; the tracked-tree command
above reproduces the published CI boundary without modifying or hiding the
foreign artifacts.

Change routing for the final manifest is `code=true`, `docs=true`,
`run_migrations=false`, with the `apps-platform` test shard. Web-image routing is
`web_image_changed=false`; `Publish App Image` should therefore complete its
classifier and skip image publication. No runtime release or deployment is
authorized or required.

## Cold architecture review

Mode: `cold_self_review` using the read-only `architecture-review` contract.
Verdict: `Release` for this prototype/evidence ticket.

- Dependency direction is one-way: `app:platform-web` imports the current
  `app:web` composition; production Web code does not depend on the prototype.
- The route boundary is explicit, same-origin, and removable. Removing the
  prototype paths and root Node manifests restores the original SSR-only tree;
  no database, API, auth, or persisted state rollback exists.
- Remote data has one client authority (Query); MobX contains presentation state
  only. The server fixture owns the principal decision.
- The prototype exposes only read-only safe fixture routes. No production
  trust, secret, trading, or mutation boundary is crossed.
- Compatibility is `compatible-change` for repository tooling and an isolated
  non-production app; runtime/API/persistence/public-site behavior is `none`.

No release-blocking finding remains inside the ticket scope.

## Limitations and residual risks

1. The deterministic REST/SSE fixtures prove client overhead and cancellation,
   not real API latency, server backpressure, reconnect policy, cache headers,
   session expiry, degraded data, or authorization correctness.
2. SSR proof covers the real current Jinja renderer and route return, but its
   downstream data API is not a production integration proof; that belongs to
   the read-only Backtests golden slice.
3. Measurements cover one high-end Apple M3 Pro and Chrome build, not the lowest
   supported client profile. Production acceptance needs that second profile.
4. Full WCAG contrast, screen-reader, localization, zoom/reflow, reduced-motion,
   chart/table parity, CSP, asset hosting, cache invalidation, and offline/error
   recovery remain production-shell or golden-slice work.
5. The `101,629 B` gzip baseline is acceptable for the spike, but production
   features and ECharts can materially increase it; enforce a route-level
   bundle budget before shell acceptance.
6. Browser traces and screenshots are local ignored artifacts. They contain only
   fixture state and must not be promoted into tracked raw browser/session data.

These are carry-forward acceptance conditions, not reasons to change or stop
the proven baseline. The next ticket is not started by this evidence.
