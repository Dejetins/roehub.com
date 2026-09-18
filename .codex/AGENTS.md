---
doc: agents
version: "2.3"
status: active
language: en
---

# Roehub Agent Guidance

## Native delivery

Use the platform skill list and global delivery contract directly. Roehub has
no parallel router, role system or workflow engine.

## UI development

The previous staged UI workflow is retired by the user's decision of
2026-09-04. It must not be resumed or recreated as a prerequisite for UI work.
Current UI sequencing is recorded in
`docs/architecture/apps/web/roehub-ui-implementation-plan-v1.md`; its existence
does not authorize execution of every backlog item or resume completed S1–S6.

By the user's decision of 2026-09-12, the refined Backtests implementation
accepted on 2026-09-11 is the visual baseline for subsequent local-platform UI.
Use `docs/architecture/apps/web/backtests-ui-iteration-log.md` and the plan's D4
for its source revision, current styles, compact controls and shared motion.
The old v23 specimen at
`.codex/delivery/evidence/roehub-ui-agent-governed-pilot/specimens/2026-08-03-linear-black-workbench-v23.html`
is historical only; preserve its path and contents but do not use it as the
current visual target. Visual acceptance is not proof of APIs or authorization.
Future work follows the selected scope and current product/security boundaries.

## Repository context

Read the smallest current source set needed for the task:

- `docs/architecture/README.md` for the architecture index;
- `docs/architecture/project-map/AGENT_GUIDE.md` and a narrow project-map slice
  for repository-wide or cross-context work;
- the selected ticket, plan, or ledger only when it is current and relevant;
- affected code, tests, CI configuration, and nearby documentation.

`.codex/PLANS.md` is historical project coordination, not default task context
or execution authority. Existing prompt packs and ledgers are historical unless
their own current state and the selected work both make them authoritative.
Never revive a legacy pack only because it exists. If it names a retired runtime
or proof surface, create or select a new ticket instead.

## Repository proof requirements

- Browser-visible work requires real browser evidence when a suitable browser
  surface is available. Use disposable local test data.
- Classify non-trivial compatibility dimensions as `none`,
  `compatible-change`, `breaking-change`, or `unknown`.

## Publish and runtime

`pre-ship-gate` is readiness-only. `publish-ci-deploy` is the single workflow
for an explicitly authorized "publish changes", push, merge, release, or
deployment request. It must read the relevant CI configuration and repository
runbook, stage only owned changes, and report the actual terminal state.

This repository currently has no configured installation or production
deployment target. Publication may therefore end at `green-pr` or
`shipped-no-runtime`; do not invent a runtime target, access a retired host, or
claim runtime proof without an explicitly authorized runbook.

## Policy artifacts and final reports

There are no repository-specific role TOMLs or prompt templates in the active
workflow. Legacy copies may be retained only as history and never override this
file or the platform skills.

Repository-authored engineering artifacts are English by default. Normative
Russian product documents and localized content remain exceptions.

For changed policy, architecture, or reusable prompt artifacts, perform a cold
self-review. Add one independent review only for shared/global policy, a
security boundary, irreversible migration, release, or a material unresolved
risk. State the review mode, verdict, and residual risk concisely.
