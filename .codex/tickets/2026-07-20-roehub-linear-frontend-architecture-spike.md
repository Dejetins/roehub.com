---
ticket_id: ROEHUB-LINEAR-FRONTEND-ARCHITECTURE-SPIKE-2026-07-20
status: ready
owner: unassigned
ticket_graph: .codex/delivery/graphs/roehub-authenticated-platform-delivery-v1.json
depends_on:
  - ROEHUB-LINEAR-REFERENCE-COMPLETION-2026-07-20
evidence: []
---

# Prove the Roehub React coexistence architecture

## Outcome

A minimal browser spike proves React/TypeScript/Vite coexistence with the
FastAPI/Jinja gateway, MobX local state, TanStack Query authoritative REST/SSE
state, styled-components plus semantic CSS tokens, four themes, route fallback,
and a repeatable client performance harness.

## Context

- accepted transition specification and shared standard under
  `.codex/delivery/specs/` and `docs/architecture/ui/`;
- accepted local information architecture, screen registry, and access/route
  contract under `docs/architecture/apps/web/`;
- the accepted reference-completion ticket and evidence;
- current `apps/web/` SSR gateway.

## Owned paths

- this ticket and its terminal evidence;
- `apps/platform-web/**`;
- `packages/tokens/**`, `packages/ui/**`, `packages/localization/**`, and
  `packages/web-contracts/**` only as bounded spike code;
- root Node workspace manifests created for the spike;
- ticket-owned browser and performance tests.

## Scope

- Prove a route-bounded SSR/React mount and real rollback path.
- Prove MobX versus Query authority, typed REST cancellation, SSE update,
  immediate four-theme switching, pointer/keyboard panel resize, and no server
  authorization decision in the client.
- Measure dependency/bundle cost and p50/p75/p95 dispatch, acknowledgement,
  response/SSE-to-paint, INP, long tasks, and frame cadence on declared hardware.

## Non-goals

- No production shell, route cluster, Figma design, backend, trading, authorization,
  persistence, public-site, release, or deployment change.

## Proof boundary

Use real-browser and measured-performance evidence. Deterministic mock latency
may prove client overhead but not API latency. Focused Python/Node tests and
`git diff --check` must pass.

## Escalation triggers

- The gateway cannot preserve same-origin auth, route identity, or SSR fallback.
- A backend/API change is required.
- Client performance cannot meet the accepted initial budgets without changing
  the baseline architecture.

## Acceptance

Terminal evidence records dependency versions, route rollback, state-boundary
tests, browser traces, measurement method/results, residual risk, and a clear
proceed/change/stop decision for Figma vNext and the production shell.
