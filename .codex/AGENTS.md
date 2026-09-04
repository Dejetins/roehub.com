---
doc: agents
version: "2.2"
status: active
language: en
---

# Roehub Agent Guidance

## Precedence

Follow platform/system instructions, the user's explicit outcome and authority,
this file, then current code and documentation. Non-waivable safety rules still
apply. Treat chat history, old plans, generated prompts, and folder names as
historical context unless a current source explicitly selects them.

## Native delivery

Use the platform skill list and the global delivery contract directly. Roehub
does not provide a parallel router, role system, or workflow engine.

- For non-trivial or authority-unclear change work, start with
  `delivery-orchestrator`. A bounded, explicit repair may execute directly.
- One ready ticket is one execution unit. Create a plan, ledger, prompt pack,
  or Goal only when the global contract selects it or the user explicitly asks.
- Use `prompt-manager` only for a reusable procedure or an explicitly justified
  prompt artifact. Use `staged-plan-runner` only for a selected active staged
  workflow or a read-only legacy inspection.
- Select only the skill that matches the task. Add browser, contract,
  performance, release, or quality tooling only when that boundary is crossed.

## UI development

The previous staged UI workflow is retired by the user's decision of
2026-09-04. It must not be resumed or recreated as a prerequisite for UI work.
The replacement development workflow has not been selected.

The accepted visual pilot is
`.codex/delivery/evidence/roehub-ui-agent-governed-pilot/specimens/2026-08-03-linear-black-workbench-v23.html`.
Preserve its path and contents during cleanup. It is a visual reference, not
proof of implemented APIs or server authorization. Future UI work follows the
user's selected scope and retains current product and security boundaries.

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

## Scope and evidence

- Preserve foreign changes in the shared checkout. Own exact paths or safely
  separable hunks; never use broad staging, implicit staging, destructive Git,
  speculative branches, worktrees, or stashes as a workaround.
- Use focused repository checks first. Tests are a gate, not universal
  acceptance: collect API, persistence, browser, runtime, performance, CI, or
  recovery evidence only when the changed behavior requires it.
- Browser-visible work requires real browser evidence when a suitable browser
  surface is available. Use disposable local test data; never retain secrets,
  cookies, session state, or raw provider payloads.
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
Russian product documents and localized content remain exceptions. Final
user-facing reports are Russian unless the user asks otherwise; preserve paths,
commands, identifiers, and statuses verbatim.

For changed policy, architecture, or reusable prompt artifacts, perform a cold
self-review. Add one independent review only for shared/global policy, a
security boundary, irreversible migration, release, or a material unresolved
risk. State the review mode, verdict, and residual risk concisely.
