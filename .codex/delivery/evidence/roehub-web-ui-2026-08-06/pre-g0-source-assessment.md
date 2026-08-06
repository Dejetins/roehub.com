---
doc: pre-g0-source-assessment
version: "1.0"
status: admitted
language: en
program_candidate_id: roehub-local-platform-web-ui-2026-08
scope_candidate: self_hosted_local_platform_web
prepared_at: 2026-08-06
---

# Roehub Local Platform Web UI — Pre-G0 Source Assessment

## Delivery routing

- Classification: new product-wide multi-screen Web UI design program.
- Execution unit now: pre-G0 admission and one owner decision packet, not implementation.
- Primary workflow: `ui-design-program`.
- Bootstrap companion: `prompt-manager` after the product path is exact.
- Later execution workflow: `staged-plan-runner`, one ledger-allowed gate/family/wave row at a time.
- Allowed current write scope: program-owned design-control artifacts under `.codex/delivery/` and `.codex/agents/generated/` after admission; no product implementation, publication, deployment, branch, worktree, or stash.
- Proof boundary now: source consistency and repository readiness only. No browser, runtime, implementation, or visual acceptance is claimed.

## Current authority and evidence

| Concern | Current source | Classification |
|---|---|---|
| Web delivery architecture | `docs/architecture/apps/web/roehub-ui-design-and-delivery-architecture-v1.md` | accepted architecture |
| Product outcome and platform constraints | `docs/architecture/platform/roehub-product-transformation-requirements-v1.md` | accepted product requirements baseline |
| Local-platform information architecture | `docs/architecture/apps/web/roehub-local-platform-information-architecture-v1.md` | accepted target architecture |
| Screen and journey entrypoint registry | `docs/architecture/apps/web/roehub-local-platform-screen-registry-v1.json` | accepted target architecture |
| Roles, capabilities, route and mutation constraints | `docs/architecture/apps/web/roehub-local-platform-access-and-route-contract-v1.json`, current API/code | accepted target plus current-state evidence |
| Current browser-visible implementation | `apps/web/` | current SSR implementation evidence |
| Candidate visual direction | `.codex/delivery/evidence/roehub-ui-agent-governed-pilot/specimens/2026-08-03-linear-black-workbench-v23.html` | protected candidate anchor; not yet G0 visual authority |

The protected pilot SHA-256 is
`3ff799ac5a5872662dda8b67fc1bd4db0c7860b7de9d84e6597465209d5dd2a4`.

## Derived product path

The following product meaning is already derivable and should not be asked of
the owner again:

- Product: a self-hosted local Roehub platform, independent from `roehub.com`.
- Primary outcomes: configure data and connections, create and operate
  strategies, configure/run/inspect backtests, inspect models, observe live
  execution and platform health, administer and recover the installation.
- Roles: `owner`, `admin`, `operator`, `trader`, and `viewer`, with the separate
  `installation_owner` authority overlay.
- Accepted platform coverage: 35 registered visual, system, QA, delivery, and
  non-visual contract surfaces with 12 journey entrypoints.
- Responsive Web widths: `820`, `1024`, and `1440`.
- Local-platform phone width `390`, separate mobile navigation, and mobile
  copies of screens remain unauthorized.
- Languages: `ru` and `en`; accessibility, keyboard operation, sufficient
  contrast, zoom, and reduced motion are required.
- Target implementation boundary after G6: tracked `apps/platform-web`, with
  shared `@roehub/*` packages and `apps/web` retaining same-origin/server
  authority until a separately proven journey migration.

## Current implementation assessment

### SSR application

`apps/web` is a substantial current implementation, not the future client
architecture. It currently provides the same-origin shell and API boundary,
session/auth handling, localization, themes, and routed work areas including
Dashboard, Strategies, Backtests, Monitoring, Models, Connections, Admin,
Settings, and legacy runbooks. Strategies and Backtests contain the deepest
current product UI; several other target surfaces remain consolidated,
placeholder-like, absent, or route-incompatible with the accepted registry.

The current flat navigation and consolidated routes do not yet implement the
accepted grouped target structure (`overview`, `research`, `operations`,
`system`) or the complete split-screen model for setup, data, live operations,
settings, administration, and local documentation.

### Target client workspace

There are no tracked `apps/platform-web` sources or tracked JavaScript workspace
manifests yet. A local ignored `apps/platform-web/dist` contains an old compiled
prototype spike only. It is not reproducible source, repository authority, or
a base to edit. The real workspace should be created only by an implementation
unit authorized from the accepted G6 handoff.

### Pilot v23

The pilot establishes a promising dense, dark, analytical workbench direction:
persistent navigation, jobs/variants/detail hierarchy, metrics, analysis tabs,
chart tools, tables, parameters, and a platform status rail.

It does not establish a complete platform baseline:

- it represents one Backtests Workbench screen rather than the whole platform;
- its canvas is fixed at `1672×941` and has no responsive rules;
- it does not cover `820`, `1024`, or intermediate reflow behavior;
- it does not cover the backtest creation/preflight flow, empty/error/permission
  states, role variance, localization stress, or cross-product journeys;
- it is a static specimen and does not prove API, persistence, authorization,
  runtime, or production behavior.

The pilot remains supporting `renderable_html` Roehub workbench evidence. The
accepted Linear reference set is the primary `native_image` visual-language
authority. Novel screens inherit the measured platform baseline established
from that set and the accepted owner intent; they do not claim screen-level
fidelity to either Linear or this one Backtests specimen.

## Resolved owner decisions for G0

### D1 — Program scope: resolved

Recommended default: design the self-hosted local platform only. Keep the
public `roehub.com` site in a separate UI program because it has a different
identity, responsive, release, and trust boundary.

Owner decision: local self-hosted platform only. The public site remains a
separate future program.

### D2 — Visual authority intent: resolved

Recommended default: select pilot v23 as the visual-language anchor that must
be evolved, not copied screen-for-screen. Preserve its serious dense-workbench
character while allowing a new responsive shell, hierarchy, tokens,
typography, interaction states, and accessible controls.

Owner decision: the supplied Linear reference screenshots are the normative
visual-language reference. Pilot v23 remains a Roehub workbench direction that
must be reworked to embody that reference rather than copied unchanged.

### D3 — New UX requirements: resolved for admission

Owner decision: Backtest creation is a compact guided process; large variant
results use ranking, progressive disclosure, comparison, and drill-down; a
strategy can be created and launched only from an existing completed Backtest
variant through an explicit, safe promotion path. Preserve high information
density while removing line noise, excess containers, competing emphasis, and
always-visible secondary controls.

## Proposed development path after owner input

1. Admission: merge D1-D3 into one durable intent baseline without rewriting
   accepted backend/product facts.
2. G0: initialize the program; complete intake and platform baseline; create
   the draft `plan_doc + prompt_pack_dir + stage_ledger` triad; validate it.
3. G1: exact-cover the complete screen/state/action atlas and render it for
   inspection. No routine owner gate.
4. G2: establish journeys, criticality, screen families, workload-bounded
   waves, and exact baseline inheritance. No routine owner gate.
5. G3: design and render the responsive foundations and representative shell
   at the accepted widths. This is the first finished owner visual review.
6. G4: design one complete family per ledger row; present one family board for
   owner acceptance/correction.
7. G5: compose accepted families into bounded end-to-end waves; present one
   compact wave board per row.
8. G6: validate critical journeys and produce the implementation handoff. The
   implementation repository/workspace is not created before this accepted
   handoff unless a separate delivery ticket explicitly authorizes an earlier
   technical spike.

## Readiness and residual risk

The repository is admitted to G0. The active draft is located at
`.codex/delivery/ui-programs/roehub-local-platform-web-ui-2026-08/`.
