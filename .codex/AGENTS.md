---
doc: agents
schema_version: agents-md/v1
version: "2.0"
status: active
language: en
user_report_language: ru
user_report_language_policy: always
scope: project
role: project_delivery_adapter
delivery_contract: /Users/daniildegtyarev/.codex/skills/delivery-orchestrator/references/delivery-contract-v1.md
prompt_pack_contract: /Users/daniildegtyarev/.codex/skills/prompt-manager/references/prompt-pack-artifacts-v1.md
---

# Roehub Delivery Adapter

## Source Order

Follow platform and system instructions, current user outcome and authority,
root `AGENTS.md`, this adapter, accepted repository architecture and ADRs, the
selected current ticket/plan/prompt/ledger, then current code, tests, CI, and
nearby documentation. Non-waivable safety rules still apply. Treat chat
history, old plans, generated prompts, folder names, and legacy templates as
historical unless a current higher-priority source explicitly selects them.

This adapter narrows the global router for Roehub. It does not provide a
parallel router, role system, workflow engine, or authority source.

## Delivery Contract

Use `delivery_contract` from front matter for non-trivial or authority-unclear
change work. Create a plan, prompt pack, ledger, Goal, branch, worktree, stash,
deployment, or external mutation only when the global contract, repository, or
user explicitly authorizes that exact artifact or action. A bounded explicit
repair may execute directly.

Read the smallest current source set. Use the architecture index only when work
crosses a documented boundary and select only skills for boundaries actually
crossed. Historical presence never makes a plan, pack, ledger, runtime, or
design source current.

## Execution Units

- One ready ticket is one execution unit.
- One allowed stage from a current `plan_doc` + `prompt_pack_dir` +
  `stage_ledger` is one execution unit unless explicitly authorized Goal mode
  permits continuation.
- A tiny explicit repair may execute directly when authority, paths, scope, and
  verification are settled.
- `main` is the shared accepted base. Ticket front matter owns ticket status;
  the current stage ledger owns staged-workflow status.

## Skill Routing

The global table remains the default. These rows narrow Roehub behavior.

| trigger | route | use_when | do_not_use_when |
|---|---|---|---|
| Artifact choice or unclear authority | `delivery-orchestrator` | Delivery topology is non-trivial or no selected current artifact settles execution | A bounded explicit repair already has settled scope and proof |
| DDD, migration, or rollout design | `architecture-design` | A bounded context, dependency, ports/adapters, ADR, migration, or rollout is unresolved | Reviewing an accepted plan without redesign |
| Accepted plan needs executor prompts | `prompt-manager` | A current accepted plan needs a reusable prompt pack and ledger | The plan is historical, unaccepted, or direct ticket execution is sufficient |
| Existing active staged workflow | `staged-plan-runner` | The current triad is consistent and the ledger permits one stage | A pack merely exists or any triad link is missing, stale, or historical |
| Backend gates | `backend-quality-gates` | Focused `uv run ruff`, `uv run pyright`, `uv run pytest`, or failing Python gate triage is needed | Local gates would be used to claim runtime or browser acceptance |
| Authenticated-platform Figma work | `figma:figma-use`; add `figma:figma-generate-library` or `figma:figma-generate-design` for their exact surfaces | The selected ticket authorizes the existing canonical project/file and owned nodes | Creating a replacement canonical file, inferring approval, browser proof, or historical Penpot work |
| Ship-readiness only | `pre-ship-gate` | A readiness verdict is requested without delivery authority | Push, merge, release, deploy, CI watch, or post-deploy proof |
| Publish, push, merge, release, deploy, CI watch, or runtime proof | `publish-ci-deploy` | The user explicitly authorizes the requested publication lifecycle action; without a current runtime runbook the flow ends at `green-pr` or `shipped-no-runtime` | Authority is absent or readiness-only review is requested |

## Project Sources

| path / identity | authority / use |
|---|---|
| `docs/architecture/README.md` | Architecture index; read only for crossed documented boundaries |
| `docs/architecture/project-map/AGENT_GUIDE.md` | Repository-wide or cross-context navigation; pair with one narrow project-map slice |
| `docs/architecture/ui/linear-workspace-ui-transition-standard-v1.md` | Shared authenticated-Web transition standard |
| `docs/architecture/ui/roehub-figma-design-delivery-standard-v1.md` | Roehub Figma delivery standard |
| `.codex/delivery/specs/` | Selected accepted Roehub transition specification |
| `.codex/delivery/graphs/roehub-authenticated-platform-delivery-v1.json` | Authenticated-platform dependencies and priority; ticket front matter remains status authority |
| `.codex/PLANS.md` | Historical coordination only; never default task context or execution authority |

All future Roehub design prototyping and design-to-code handoff use Figma
project `roehub.com` (`projectId` `629113387`) and the existing file
`Roehub Authenticated Platform UI` (`fileKey`
`GBzmB9evtzqnAYNjp9W1sr`). Do not create a replacement or second canonical
file. Before mutation, verify the authenticated plan, project/file identity,
and exact owned pages/nodes.

For authenticated Web transition work, read the two UI standards above, the
selected specification, and the applicable graph node. Accepted 2026-07-20
local design-system artifacts remain historical contract evidence; their
six-theme future target is superseded. Backend and server-authorization tickets
remain independently authoritative.

The visual layer of
`ROEHUB-LINEAR-FRONTEND-ARCHITECTURE-SPIKE-2026-07-20` is
`rejected_by_product_owner` and `not_a_design_source`. Reuse only its route,
state, transport, rollback, test, measurement, and dependency seams; never its
layout, styling, component anatomy, copy, theme values, fixture presentation,
or screenshots.

More generally, architecture and browser spikes prove technical feasibility
only. Their visible output is never a design source unless a design ticket and
an explicit product-owner decision approve named Figma nodes.

Before Figma foundations begin, the selected UI instructions-and-copy review
ticket must be accepted by the product owner. Pause at its named checkpoints;
do not infer approval from silence or mark instructions, copy, or visual design
`accepted` without an explicit decision identifying the reviewed artifact.
Linear is a functional-structure input, not a screen template or tracker. Map
navigation, page context, primary work, and contextual status/progress/activity/
resource blocks to Roehub semantics or record a justified omission. Do not copy
Linear geometry, taxonomy, copy, assets, or unsupported concepts, and do not
rename historical ticket IDs containing `LINEAR`.

For the authenticated-platform queue, ticket evidence confirms completion and
GitHub Actions verifies publication. Linear remains neither a tracker nor a
source of ticket status, priority, or order.

## Prompt Pack And Ledger Contract

A current pack must follow `prompt_pack_contract` and cross-link exactly
`plan_doc`, `prompt_pack_dir`, and `stage_ledger`. Roehub currently has no
active repository prompt or ledger template. `prompt-manager` must use the
minimal shapes and lifecycle contract from `prompt_pack_contract` unless a
future current source explicitly selects a new non-legacy template.
`.codex/agents/legacy/` and its typo copy remain historical evidence only;
never use them to create, format, authorize, or add runtime/secret/Git policy
to current work. Architecture-stage ledgers are a Russian-by-default output
exception; other current ledgers follow the selected plan and repository
documentation language.

Use `required_keywords` for compact domain vocabulary and `required_literals`
for exact strings. Keep `always_read`, task entrypoints, conditional bundles,
and consult-if-needed sources bounded to normally no more than eight files and
roughly 35k-50k tokens, stopping at whichever limit is reached first. Stop when
scope, contracts, docs, proof boundary, and blockers are known. Default to one
`manual_sequential` stage on `main`; do not create `GOAL.md`, a branch,
worktree, stash, or auxiliary workflow artifact without explicit user
authority.

## Safety And Evidence

- Preserve foreign changes in the shared checkout. Own exact paths or safely
  separable hunks; never use broad/implicit staging, destructive Git,
  speculative branches, worktrees, or stashes as a workaround.
- When an architecture source under `docs/architecture/**` changes,
  `docs/architecture/README.md` is a derived companion only for entries caused
  by this task. Acquire the global architecture-index lock, repeat status/diff
  inspection, run `python -m tools.docs.generate_docs_index` and its `--check`
  form, preserve the manual header, and stop on unrelated or non-separable
  index hunks.
- Use focused gates first. Tests do not prove API, persistence, browser,
  runtime, performance, CI, recovery, release, or production behavior.
  Browser-visible work requires real browser evidence when a suitable surface
  exists and disposable local test data without retained secrets or sessions.
- Classify non-trivial compatibility dimensions as `none`,
  `compatible-change`, `breaking-change`, or `unknown`.
- Penpot is historical only. Figma proves design structure and intent, not
  runtime behavior, authorization, accessibility, or performance. Record exact
  file/page/node identity, modes, component inventory, screenshots,
  functional-block mapping, and explicit product-owner review.
- `pre-ship-gate` is readiness-only. `publish-ci-deploy` is the sole Roehub
  publication workflow and requires explicit user authority, relevant CI,
  scoped owned changes, and the actual terminal state. Read a runtime runbook
  only when this adapter explicitly selects a current one and runtime action is
  authorized.
- Roehub currently has no configured installation or production deployment
  target. Publication may end at `green-pr` or `shipped-no-runtime`; never
  invent a target, access a retired host, or claim runtime proof without an
  explicitly authorized current runbook.
- Keep secrets, credentials, cookies, browser state, raw provider payloads, and
  environment dumps out of durable artifacts.

## Output Contract

Repository-authored engineering artifacts are English by default. Normative
Russian product documents, localized content, and architecture-stage ledgers
selected under the prompt-pack contract are exceptions. All final user-facing
reports must always be written in Russian; this rule has no language override.
Preserve exact paths, commands, identifiers, and statuses.

Reports state owned scope, foreign changes excluded, checks and observed proof
boundary, compatibility class, residual risk, and next safe action. Review
architecture artifacts with `architecture-review` when repository or user
authority requires it. Prompt and routing artifacts follow the risk-tier gate in
`prompt-manager`: independent review is mandatory for shared/global, security,
runtime, irreversible, external-write, and staged artifacts; a small local
prompt edit outside those surfaces uses deterministic self-check.
