---
ticket_id: ROEHUB-DELIVERY-MODEL-CONSOLIDATION-2026-07-22
status: accepted
owner: unassigned
depends_on: []
evidence:
  - .codex/delivery/evidence/ROEHUB-DELIVERY-MODEL-CONSOLIDATION-2026-07-22.md
---

# Consolidate Roehub delivery into one repository queue

## Outcome

Replace the two former authenticated-platform ticket graphs with one repository
queue. Ticket front matter remains the only status authority; the unified graph
holds dependencies and priority; ticket evidence confirms completion; `main` is
the shared accepted base; and GitHub Actions verifies the published result.

## Scope

- Create `.codex/delivery/graphs/roehub-authenticated-platform-delivery-v1.json`
  from the 15 current tickets and their real dependencies.
- Update the 15 ticket front matters, current delivery instructions, the
  transition specification, UI migration registry, historical plan stub, and
  derived documentation/project maps.
- Add a standard-library validator, its focused unit tests, and a GitHub CI
  step that fails on queue-model drift.
- Retain accepted ticket evidence and historical technical identifiers.

## Owned paths

- this ticket and its evidence;
- `.codex/delivery/graphs/roehub-authenticated-platform-delivery-v1.json` and
  removal of the two replaced graph files;
- the 15 graph ticket files under `.codex/tickets/`;
- `.codex/AGENTS.md`, `.codex/PLANS.md`, and the selected transition spec;
- `docs/architecture/ui/roehub-linear-ui-migration-registry-v1.json` and its
  derived `docs/architecture/README.md` entry;
- `tools/delivery/validate_roehub_delivery_model.py`, its unit test, and the
  focused CI workflow step;
- generated `docs/architecture/project-map/PROJECT_MAP.md`,
  `project-map.mmd`, `component-map.mmd`, `project-map.json`, and
  `AGENT_GUIDE.md`.

## Non-goals

- No Linear API, MCP, browser, project, issue, status, label, relation, or
  synchronization.
- No runtime deployment, application-image publication, product implementation,
  or start of the frontend-architecture-spike ticket.
- No branch, worktree, extra repository folder, plan, ledger, or second
  coordination channel.

## Proof boundary

- The validator proves graph membership, front-matter links, status/queue
  invariants, dependencies, absence of replaced graph references, and delivery
  authority rules.
- JSON parsing, hook regressions, generated documentation/project-map checks,
  OSS metadata, CI classification, and `git diff --check` prove repository
  consistency.
- GitHub Actions is observed after publication; no runtime boundary is claimed.

## Acceptance

- The unified graph has the requested 15 tickets, the four factual `accepted`
  states, exactly one `ready` frontend spike, and no `active` graph ticket.
- The former graph files and active references are gone without altering
  accepted evidence.
- Local checks, cold self-review, one scoped commit to `main`, and the required
  GitHub workflow results are recorded in the final handoff.
