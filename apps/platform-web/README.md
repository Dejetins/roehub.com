# PROTOTYPE — Roehub frontend architecture spike

This disposable prototype answers one architecture question:

> Can a route-bounded React + TypeScript + Vite client coexist with the current
> FastAPI/Jinja gateway while preserving an immediate SSR rollback path,
> keeping local and server state authorities separate, and meeting the initial
> interaction-performance budgets on safe local fixtures?

It answers these bounded sub-questions:

- Can `/__prototype/react/` remain isolated while `/backtests` is rendered by
  the existing `apps.web.main.app` FastAPI/Jinja application on the same origin?
- Can MobX own only theme, selection, and panel geometry while TanStack Query
  owns REST and SSE projections?
- Does TanStack Query cancellation reach the underlying `fetch` signal?
- Can an SSE event update the Query cache and become visible without copying
  remote state into MobX?
- Do `abyss`, `graphite`, `frost`, and `paper` switch without reload?
- Does panel resize work with pointer, arrow keys, and reset?
- What dependency, bundle, interaction, INP, long-task, and frame-cadence costs
  are observed on the declared local machine?

Run from the repository root with one command:

```bash
npm run prototype
```

Then open `http://127.0.0.1:4173/__prototype/react/`.

The command builds the Vite client and starts a local-only coexistence gateway.
The gateway mounts the unchanged current SSR application at `/` and reserves
only `/__prototype/**` for safe fixture REST, SSE, and React assets. The link
labelled `Return to current SSR /backtests` performs the real rollback path.

The fixture principal, backtest rows, REST latency, and SSE events are local and
disposable. They are not production credentials or API evidence. No trading
operation, backend route, authorization rule, persistence, public-site route,
runtime shell, or Figma artifact is changed by this prototype.
