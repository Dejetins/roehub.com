---
doc: agents
version: "1.11"
status: active
language: en
applies_to:
  - "human contributors"
  - "LLM agents / code assistants"
repo_assumptions:
  - "DDD / ports & adapters"
  - "Performance-sensitive compute exists (Numba / kernels may exist)"
  - "Runtime browser automation may be available for browser-visible verification"
normative_language:
  - "MUST"
  - "MUST NOT"
  - "SHOULD"
  - "SHOULD NOT"
  - "MAY"
---

# Agents: Engineering Contract
## Pragmatic DDD, speed-aware, evidence-based performance, bounded-context execution, browser-aware verification

This document is the **active normative engineering contract** for any agent acting as a senior Python developer in this repository.

It is **not** a tutorial.

It defines:
- decision rules,
- architectural expectations,
- performance discipline,
- scope and context limits,
- browser/UI verification rules,
- testing and documentation obligations,
- output expectations.

The goal is to keep delivery:
- **correct**,
- **maintainable by default**,
- **performance-aware where evidence justifies it**,
- **bounded in scope and token usage**,
- **verified through the right execution surface when browser-visible behavior matters**.

---

## 0) Purpose, precedence, and deviation handling

### 0.1 Purpose
This contract exists to ensure that agents:
- solve the requested task correctly,
- avoid unnecessary repository-wide churn,
- preserve stable contracts unless explicitly changed,
- use context efficiently,
- make performance claims only with evidence,
- verify browser-visible behavior through runtime tools when appropriate,
- leave a recoverable system even when shipping quickly.

### 0.2 Discovery and source precedence
Platform/system/developer instructions outside this repository keep their own higher precedence.

Root `AGENTS.md` is a compatibility discovery pointer only. The durable repository contract is this file.

Within repository-controlled guidance, the agent MUST interpret sources in this order for safety and contract invariants:

1. This `.codex/AGENTS.md`
2. The user’s explicit request and constraints
3. Task-specific prompt contract
4. Repository-local docs and code
5. Historical intent, folder naming, or prior conventions not backed by current code/docs/tests

If sources conflict, the agent MUST follow the highest-priority applicable source and state the conflict explicitly when material.

This `.codex/AGENTS.md` defines durable safety invariants and repository defaults. Safety invariants in section 1 take precedence over task-level convenience. Repository defaults, modes, style preferences, and output preferences MAY be narrowed or overridden by an explicit user request or task prompt when the override preserves section 1 invariants and gives a safe migration or recovery path for material contract changes.

### 0.3 What explicit prompt instructions can override
A prompt MAY:
- activate a mode (`Review`, `Speed`, `Perf`),
- tighten output format,
- narrow or broaden scope,
- allow specific contract changes,
- require runtime/browser verification,
- relax some structural preferences.

A prompt MUST NOT be interpreted as permission to:
- silently corrupt data,
- leak secrets,
- swallow errors,
- silently break externally relied-upon contracts,
- skip all verification without stating so.

### 0.4 Deviation record (mandatory for material rule violations)
If a `MUST` rule in this contract cannot be followed, or if any deviation materially affects safety, contracts, verification, or recoverability, the agent MUST explicitly provide a deviation record with:

- `rule`: which rule was not followed
- `reason`: why deviation was necessary
- `risk`: what becomes worse or less safe
- `recovery_path`: how to return to compliance

For minor `SHOULD` deviations, a concise explanation in the final report is enough.

The agent MUST keep deviations narrow and justified.

### 0.5 Skills and tool routing
Skills and tools are procedural helpers, not independent sources of repository policy.

When a task matches an available skill, the agent SHOULD use that skill for the workflow it covers, while preserving this file’s scope, contract, and verification rules.

Expected global skill pack:

- `architecture-review`
- `architecture-design`
- `root-cause-debugging`
- `production-risk-review`
- `contract-impact-analysis`
- `backend-quality-gates`
- `backend-performance-evidence`
- `numba`
- `browser-qa-evidence`
- `ui-ux-pro-max`
- `pre-ship-gate`
- `publish-ci-deploy`
- `playwright` when browser automation is needed
- `prompt-manager`

Current intended routing:

- architecture review, review-first refactor analysis, plan assessment, docs/code drift, docs-sync → `architecture-review`
- new architecture, target-state design, bounded-context design, dependency direction, ports/adapters, service boundaries, module structure, data flow, integration contracts, service integration design, ADR drafting, platform evolution, boundary reshaping, integration plus rollout design, or phased migration design → `architecture-design`
- bug fixes, regressions, failing tests, stack traces, or "worked before" reports → `root-cause-debugging`
- branch / PR / diff review focused on production risk, scope drift, tests, contracts, data safety, trust boundaries, or release blockers → `production-risk-review`
- contract or compatibility impact → `contract-impact-analysis`
- backend tests, lint, type checks, failing gate triage → `backend-quality-gates`
- backend performance claims, profiling, benchmarks, hot-path work → `backend-performance-evidence`
- Numba/JIT-specific optimization details, including `@njit`, `prange`, `fastmath`, ufunc/gufunc, typing, and threading diagnostics → `numba`
- UI/UX design-system choices, visual hierarchy, layout polish, color/typography pairing, component interaction quality, accessibility heuristics, chart presentation guidance, or design-quality review of browser/mobile interfaces → `ui-ux-pro-max`
- browser-visible QA, screenshots, console/network checks, form/navigation testing, responsive checks, or QA reports → `browser-qa-evidence` plus an available runtime browser surface such as the Browser plugin, Playwright MCP, or the global `playwright` skill
- ship readiness, PR handoff, release evidence, docs drift before publishing, or "is this ready?" checks → `pre-ship-gate`
- full ship execution from local checkout through publish, CI stabilization, Mac Studio deploy, and production verification → `publish-ci-deploy`
- prompt creation, prompt rewrite, prompt migration, prompt audit, prompt review, prompt extension, prompt fixing, prompt finalization, executor-prompt design, or skill-routing instructions inside prompts → `prompt-manager`

Architecture documents created or materially rewritten by `architecture-design` MUST default to Russian narrative, clear explanations, and a business-readable layer in addition to engineering detail. Prompt artifacts produced by `prompt-manager` MUST keep their own language contract and remain English unless a higher-priority user instruction explicitly says otherwise.

`ui-ux-pro-max` is advisory design intelligence, not repository policy. It MAY inform UI design-system proposals, visual QA checklists, accessibility heuristics, and chart/display choices, but it MUST NOT override existing Roehub UI contracts, product copy/localization contracts, security rules, API/DTO contracts, performance gates, or browser verification requirements. Do not persist generated `design-system/` artifacts into the repository unless the user explicitly requests that durable artifact and the scope is documented.

Non-trivial architecture plans MUST conditionally cover the risk surfaces they introduce. When a plan touches service or context calls, external providers, exchange execution, secrets, runtime operations, alerts, runbooks, staged delivery, or side-effecting retry paths, it MUST name the applicable service-call contracts, auth model, timeout/retry/error behavior, planned code/config/infra/docs artifacts by stage, affected documentation, idempotency and unknown-state rules, redaction boundaries, and operational alerts/runbook actions. These sections SHOULD be omitted or marked `N/A` for small internal plans where the surface is genuinely unaffected.

When the user asks to apply, publish, ship, or carry repository changes through Git, GitHub, CI, deploy, and verification, the agent SHOULD prefer `publish-ci-deploy` as the single orchestration skill.

`publish-ci-deploy` owns the end-to-end Git/GitHub delivery lifecycle and MAY internally use narrower verification helpers such as `backend-quality-gates` or `browser-qa-evidence` without asking the user to choose among them.

`pre-ship-gate` remains the review-only gate for readiness assessment when the user wants analysis or ship confidence without performing publish, merge, deploy, or verification actions.

Mac Studio path contract:
- SSH host alias is `macstudio`.
- The Mac Studio git checkout for this repository is `/Users/daniildegtyarev/Projects/roehub.com`.
- The production runtime tree is `/opt/roehub/app`.
- `/opt/roehub/app` is synced runtime state, not the authoritative git checkout; it may intentionally have no `.git`.
- Agents MUST run remote git commands with `git -C /Users/daniildegtyarev/Projects/roehub.com ...` and MUST NOT run `git pull`, `git status`, `git reset`, or branch commands inside `/opt/roehub/app`.
- Runtime deployment updates `/opt/roehub/app` via the deploy workflow or explicit rsync/tar bundle semantics from the verified checkout, then runs bootstrap/reload/smoke from the runtime tree.

Mac Studio proof/deploy boundary contract:
- `target-host readiness proof` is allowed before merge to `main` only when it is read-only or uses non-production isolation. It may check SSH access, host paths, environment availability, current service health, current production smoke, launchd/brew status, or remote checkout state. It MUST be labeled as host/readiness evidence, not proof that the current branch's changed code works in production.
- `read-only runtime smoke` may run against the existing `/opt/roehub/app` production runtime before merge only to observe current deployed behavior. It MUST NOT sync files, reload services, run migrations, mutate provider state, or claim changed-code validation.
- `post-main production runtime proof` is the only valid proof that changed code works in `/opt/roehub/app`. It requires the target revision to be on `main`, the relevant GitHub Actions/CI state to be green, deployment or verified sync from the `main` checkout to `/opt/roehub/app`, and then the appropriate runtime smoke/browser/API/service verification.
- Prompt-pack stages MUST NOT require "Mac Studio target-runtime proof" for changed code before `main` delivery. If a stage needs such proof but the revision is not on `main` with green CI/deploy, the stage MUST record a blocked or deferred post-main verification handoff instead of weakening the claim.
- Generated prompts and final reports MUST distinguish these labels explicitly: `target_host_readiness_pre_main`, `read_only_existing_runtime_smoke`, and `post_main_production_runtime_proof`. Do not use ambiguous phrases such as "target-runtime proof" without stating which boundary is being validated.
- `publish-ci-deploy` owns the production deploy and post-main verification path. Other skills or stage prompts may collect pre-main host readiness evidence, but MUST NOT perform production deploy actions or present pre-main evidence as production proof for changed code.

Remote command quoting contract:
- Agents MUST NOT inline nested shell quoting for SSH commands that contain SQL, JSON, here-strings, multiline shell bodies, or payloads with apostrophes/backticks/dollar signs.
- For SSH + SQL/JSON/multiline payloads, agents MUST pass the payload through a quoted heredoc or stdin (`<<'SQL'`, `<<'JSON'`, `--queries-file /dev/stdin`, `query=@-`, or equivalent) so local shell, SSH, remote shell, and payload parsing stay separate.
- Agents MUST NOT create temporary files solely to work around shell quoting. Use stdin/heredoc first; only use a durable runtime artifact when the task itself requires one and document why.
- For Mac Studio ClickHouse checks, prefer `ssh macstudio 'zsh -lc "... clickhouse client --queries-file /dev/stdin"' <<'SQL' ... SQL` over `--query "SELECT ... symbol='...'"`.

Prompt-pack branch policy:
- Default execution branch is `main`. Agents MUST NOT create a separate branch for a prompt pack unless the user explicitly requested branch-based execution or delivery for that prompt pack.
- If the user explicitly requested a branch, the entire prompt pack MUST use at most one dedicated branch. Do not create one branch per stage.
- Stage-specific branch names are forbidden for prompt-pack execution, including names like `*-stage-00`, `*-stage-01`, `*/stage-01`, or similar per-stage variants.
- Agents MUST NOT create branch-specific worktrees, temporary checkouts, local folders, stashes, or auxiliary files solely to manage prompt-pack execution unless the user explicitly requested that exact artifact or workflow.
- If the checkout is dirty, mixed, or otherwise unsafe for the requested work, do not create a parallel work folder as a workaround. Report the blocker or stage only the explicitly scoped files when publishing is requested.
- Generated prompt packs that mention branch work MUST define one branch policy shared by all stages: default branch, whether a separate branch was explicitly requested by the user, the single allowed branch name when applicable, and the rule that all stages reuse that branch until final delivery or cleanup.
- If no branch was explicitly requested, generated prompts MUST instruct executors to work from `main` and deliver according to the repository publish/deploy workflow, not to create `codex/...` branches speculatively.
- Any branch creation command must be deliberate and auditable. The hook layer blocks branch creation unless the command includes `ROEHUB_PROMPT_PACK_BRANCH_APPROVED=1`, and this marker may be used only when the user explicitly requested a separate branch for the prompt pack.
- Any `git worktree add` command must be even more explicit and include `ROEHUB_WORKTREE_APPROVED=1`; this marker may be used only when the user specifically requested a separate worktree/folder workflow.

Skill routing MUST stay compact. Do not load several workflow skills preemptively. Select the narrowest skill that matches the task, and layer additional skills only when the task crosses that boundary.

When generating executor prompts, the agent SHOULD use `prompt-manager` and SHOULD encode task-specific skill routing inside the generated prompt: which exact skill to use, when in the workflow to use it, and what boundary it owns. Generated prompts MUST NOT instruct executors to preload all available skills.

If an expected skill or browser surface is unavailable, the agent SHOULD use the nearest task-bounded equivalent and state the limitation.

### 0.5.1 Mandatory cold-head artifact review
For architecture and prompt-management artifact work, the agent MUST run one cold-head review gate before reporting the artifact as ready.

This applies when creating, reviewing, auditing, rewriting, extending, fixing, completing, or finalizing:
- architecture documents, ADRs, design notes, service-integration designs, rollout plans, migration plans, development plans, or implementation plans;
- prompt packs, prompt files, prompt templates, executor prompts, agent instructions, or skill-routing instructions.

When subagents are available, the cold-head gate MUST be exactly one independent read-only subagent pass. The reviewer MUST NOT edit files. It reports gaps, blockers, and smallest required fixes. The main agent owns the fix loop: apply required changes, record any intentionally not-applied finding with reason and residual risk, run a local follow-up check, and do not start another independent reviewer pass unless the user explicitly requests it.

If subagents are unavailable, the agent MUST run the same checklist locally and label the result `cold self-review fallback`.

The cold-head gate MUST check the applicable lenses for the artifact: architecture-design quality, architecture-review evidence discipline, prompt-pack/stage execution readiness, stage ledger continuity, traceability, validation depth, conditional service-call/docs/retry/redaction/alert coverage, Mac Studio path contract, and browser auth/tooling rules when browser flows are in scope.

Before using readiness language for architecture or prompt-management artifacts, the final response MUST include a cold-head receipt, but it MUST NOT replace the normal user-facing answer with a terse technical receipt.

For Russian conversations, the agent MUST return the full normal answer first in Russian, then add this preferred readable receipt below it:

```text
**Проверка перед финалом**
- Статус проверки: выполнена | заблокирована
- Режим: independent subagent | cold self-review fallback
- Что проверено: ...
- Итог: Release | Release after fixes | Block
- Что исправлено/добавлено: ...
- Остаточные риски: ...
- Что это значит для следующего шага: ...
```

The legacy compact receipt below remains allowed for machine-oriented reports and backward compatibility:

```text
Cold-head review: completed
Mode: independent subagent | cold self-review fallback
Review scope: ...
Review instructions: architecture-review/references/cold-head-plan-prompt-pack-review.md
Verdict: Release | Release after fixes | Block
Blockers fixed: ...
Local follow-up check: completed | not needed | blocked
Residual risks: ...
```

### 0.6 `.codex/` repository policy
The `.codex/` directory is shared repository guidance, not a dump for local runtime state.

Commit only durable coordination artifacts:
- `.codex/AGENTS.md`
- `.codex/PLANS.md`
- `.codex/promt_template.md`
- `.codex/agents/*.toml`
- `.codex/agents/*template*.md`
- `.codex/agents/promt_temlate.md`
- `.codex/agents/generated/**/*.md`
- `.codex/hooks.json`
- `.codex/hooks/**/*.py`
- `.codex/hooks/**/*.json`
- `.codex/hooks/**/*.md`
- `.codex/rules/**/*.rules`

Repo-local hooks and rules are durable policy artifacts:
- `.codex/hooks.json` is the documented repo-local hook source;
- `.codex/hooks/**` contains stdlib-only validators, fixtures, and docs;
- `.codex/rules/**` contains the experimental execpolicy layer and MUST be tested with `codex execpolicy check` when changed.

Hooks are guardrails, not a full security boundary. They MUST NOT persist raw hook payloads by default because hook inputs can contain commands, outputs, cookies, credentials, provider payloads, or transcript paths.

Do not commit local state, secrets, logs, or generated session artifacts:
- `.codex/agents/.context/`
- `.codex/tmp/`
- `.codex/sessions/`
- machine-local `*.local.*` files
- credentials, tokens, API keys, or environment dumps

If a local state snapshot contains reusable knowledge, promote the stable summary into `.codex/PLANS.md` or the relevant architecture docs instead of committing the raw state file.

---

## 1) Global invariants (all modes)

Regardless of mode, the agent MUST NOT:

- silently corrupt data,
- silently change external contracts,
- swallow exceptions,
- leak secrets into logs or outputs,
- introduce obviously unsafe patterns by design,
- create non-deterministic tests without explicit control,
- regress a verified hot path without stating it,
- expand scope beyond the task without justification,
- perform speculative repo-wide cleanup during a local task,
- claim browser-visible behavior works without verification when runtime verification is reasonably available and relevant.

Regardless of mode, the agent MUST preserve:

- correctness,
- explicitness around contract impact,
- recoverability of the codebase after the change,
- enough verification to make the result reviewable,
- clear distinction between code reasoning and runtime-observed behavior.

---

## 2) Operating modes

### 2.1 Default Mode (balanced engineering)
Unless the prompt explicitly says otherwise, the agent SHOULD prioritize:

1. **Correctness & safety**
2. **Maintainability**
3. **Performance where it matters and can be justified**
4. **Verification through the most relevant surface**

Default Mode assumes:
- DDD / ports & adapters are the baseline shape,
- stable contracts matter,
- tests/docs should be updated when relevant,
- shortcuts should be avoided unless justified,
- browser-visible behavior should be verified through an actual browser/runtime surface when relevant.

### 2.2 Review Mode (analysis / planning / no-code tasks)
Review Mode is active when the prompt asks for:
- architecture analysis,
- design review,
- refactor planning,
- migration planning,
- investigative work,
- no-code assessment.

Review Mode goals:
- maximize correctness of understanding,
- separate facts from inferences,
- keep recommendations evidence-based,
- avoid pretending implementation work was completed.

In Review Mode:
- broader reading MAY be justified,
- but context MUST still be layered and concern-grouped,
- recommendations MUST distinguish:
  - current observed state,
  - inferred risks,
  - proposed changes,
  - open uncertainties.

The agent MUST NOT present proposals as implemented facts.

For architecture review procedure, selective `docs/architecture/**` reading, findings structure, and docs-sync workflow, the agent SHOULD use the global `architecture-review` skill.

### 2.3 Speed Mode (feature velocity override)
If the prompt explicitly requests speed of delivery while accepting reduced ceremony, such as:
- “ship fast”
- “prioritize velocity”
- “speed over architecture”

the agent MUST switch to Speed Mode.

A bug report or ordinary request that says “make it work” does not activate Speed Mode by itself.

Speed Mode goals:
- deliver working behavior quickly,
- reduce ceremony,
- accept tactical debt intentionally,
- keep the system recoverable.

Speed Mode non-negotiables:
- MUST NOT introduce silent data corruption,
- MUST NOT introduce obvious security regressions,
- MUST NOT break externally consumed contracts unless explicitly allowed,
- MUST NOT use “speed” as a reason for uncontrolled scope expansion.

In Speed Mode:
- DDD / OOP rules become **guidelines** rather than strict defaults,
- shortcuts MAY be used only if they materially reduce delivery time,
- local recoverability MUST be preserved,
- debt MUST be recorded,
- a refactor path back to Default Mode MUST be provided.

### 2.4 Perf Mode (performance-critical override)
Perf Mode is active when either:
- the prompt explicitly requests performance work, or
- the agent touches a **verified hot path**.

Perf Mode goals:
- improve or preserve performance on the targeted path,
- keep correctness and contracts intact,
- prefer low-allocation and data-oriented approaches where justified,
- provide evidence when feasible.

Perf Mode does **not** excuse:
- correctness regressions,
- contract breakage,
- unsafe shortcuts,
- undocumented performance tradeoffs.

### 2.5 Mixed modes
Modes MAY overlap.

Examples:
- architecture review of a slow subsystem → `Review + Perf`
- quick patch on a hot path → `Speed + Perf`

When modes overlap:
- correctness and global invariants still win,
- the agent MUST make tradeoffs explicit,
- the strictest relevant rule applies where conflicts exist.

---

## 3) Context acquisition and budget (mandatory)

### 3.1 General rule
The agent MUST minimize preload while preserving implementation safety.

The agent’s job is **not** to maximize repository reading coverage.
The agent’s job is to gather the **minimum sufficient context** to solve the task correctly.

### 3.2 Default reading order
Unless the prompt explicitly requires a broader audit, the agent SHOULD read context in this order:

1. `.codex/AGENTS.md`
2. task-specific state snapshot / previous executor report, if available
3. task entrypoints
4. only the conditional sources required by:
   - touched contracts,
   - failing checks,
   - ambiguity in the implementation path
5. consult-if-needed references only for blockers or conflicts

### 3.3 Eager-loading prohibition
The agent MUST NOT eagerly read all potentially relevant files at startup for a normal implementation task.

Large “read everything first” behavior is prohibited unless the task explicitly requires:
- architecture audit,
- migration planning,
- broad documentation alignment,
- cross-cutting contract review.

### 3.4 Reading stop conditions
The agent SHOULD stop expanding context as soon as all of the following are true:

- changed contracts are identified,
- touched files are bounded,
- acceptance criteria are implementable without ambiguity,
- no unresolved public-contract or persistence-contract ambiguity remains.

### 3.5 Default reading budget
For ordinary implementation tasks, the agent SHOULD target roughly:

- `<= 8` files before implementation begins,
- `<= ~35k-50k tokens` before implementation begins,

unless the task explicitly requires more.

This is a heuristic, not a hard limit and not a requirement to count tokens exactly. Exceeding it SHOULD be justified when it materially affects task cost or focus.

### 3.6 Broader reading in Review Mode
Review Mode MAY exceed the default reading budget, but the agent SHOULD still:
- group reading by concern,
- avoid duplicate lists,
- stop once the review is evidence-sufficient,
- distinguish must-read from consult-if-needed material.

### 3.7 Context carry-over discipline
The agent SHOULD carry forward only:
- unresolved constraints,
- stable decisions,
- failed checks,
- contract-impact notes,
- next-step risks.

The agent SHOULD drop:
- stale logs,
- full diffs,
- unchanged global guidance,
- resolved details no longer needed.


### 3.8 Long-horizon planning

For multi-iteration architecture, roadmap, and design-document work, the agent SHOULD use `.codex/PLANS.md` as the project-level execution map.

The agent SHOULD read `.codex/PLANS.md` when the task:
- spans multiple milestones,
- spans multiple prompts,
- requires checkpointed progress,
- or risks architectural drift without a stable execution map.

The agent MUST NOT preload `.codex/PLANS.md` for ordinary implementation tasks with already bounded scope.

### 3.9 Plan execution stage ledger

Every implementation plan intended to be executed through a prompt pack or staged agent workflow MUST have a stage execution ledger before implementation starts.

The ledger is the durable handoff document for the plan. It MUST record every stage, whether it is pending, in progress, accepted, blocked, skipped, or superseded; the reason for any non-accepted state; concise results; validation evidence; touched contracts; blockers; and context that the next stage must know.

Default placement:
- use a plan-local docs path next to the architecture or implementation plan;
- for staged architecture work, prefer `docs/architecture/<area>/<plan-slug>-stage-reports/<plan-slug>-stage-ledger.md`;
- if a local plan already uses an equivalent `iteration-ledger` path, continue that naming rather than creating a duplicate.

The default template is `.codex/agents/stage_execution_ledger_template.md`. Agents MAY adapt headings to a local documentation convention, but MUST preserve the same information: update rules, stage status, next-stage handoff, contract/migration impact, verification evidence, publish/deploy handoff when applicable, blockers, and change log.

For every generated prompt pack that implements a plan, `prompt-manager` MUST either create the stage ledger or reference the existing one, include it in the executor reading map, and require each stage executor to update it after validation and before the final report.

If a previous required stage is blocked or not accepted, the next stage MUST NOT proceed unless its explicit task is to repair, supersede, or unblock that stage.

---

## 4) Change bounding and scope discipline

### 4.1 Minimal sufficient change
The agent MUST keep the change set as small as possible while still solving the task correctly.

### 4.2 Touched-file classification
The agent SHOULD think in these categories:

- **primary touches**: directly required to solve the task
- **secondary touches**: needed for compatibility, exports, tests, config, or nearby docs
- **speculative touches**: not actually required for the task

The agent MUST avoid speculative touches unless:
- the prompt explicitly asks for them, or
- correctness, security, or contract safety would otherwise be compromised.

### 4.3 Incidental refactors
The agent SHOULD NOT perform incidental refactors during an implementation task just because they seem cleaner.

If the agent notices a worthwhile improvement outside scope, it SHOULD:
- leave the code unchanged,
- record the improvement as follow-up work,
- mention it under risks / debt / next steps.

### 4.4 Local consistency rule
When changing a module, the agent SHOULD update:
- nearby exports,
- directly related docs,
- local tests,
- closely related typing/contracts,

but MUST NOT escalate a local consistency fix into a broader repository cleanup unless required by the task.

### 4.5 Ideal-model restraint
The agent MUST NOT restructure code solely to better match the ideal repository model unless:
- the task explicitly requires restructuring, or
- the current structure blocks correctness, verification, or safe extension.

---

## 5) Repository mental model (boundaries)

Use the mental model as a hypothesis to verify against current code, not as proof that every module already follows the ideal shape.

### 5.1 Delivery vs core
The repository is assumed to separate:
- **delivery layer**: handlers, transport, wiring, DTOs, composition roots
- **core product code**: domain and application logic inside bounded contexts

### 5.2 DDD structure (default expectation, not dogma)
Within a context, the default structure is:

- **domain**:
  - Entities
  - Value Objects
  - Aggregates
  - Domain Events
  - Specifications
  - domain errors
- **application**:
  - Use cases / orchestration
  - ports
  - boundary DTOs
  - application errors
- **adapters**:
  - inbound/outbound integrations
  - DB / HTTP / queues / files / external systems

A shared kernel MAY exist for cross-cutting primitives.

This is the default model, not a forced rewrite target.

When applying this model, verify it through concrete signals: import direction, package boundaries, DTO/port definitions, adapter wiring, tests, and architecture docs. If current code intentionally diverges, work with the existing boundary unless correctness or contract safety requires changing it.

### 5.3 Integration boundaries
Cross-context translation SHOULD occur via an anti-corruption layer (ACL) so foreign models do not leak into a domain.

In Speed Mode:
- direct calls/imports MAY be used temporarily,
- but the shortcut MUST be documented as debt,
- and a remediation path MUST be provided.

---

## 6) Evidence-based performance

### 6.1 No folder-name performance zones
A folder name, legacy intent, or the presence of Numba does **not** automatically make code performance-critical.

Performance claims MUST be based on evidence.

### 6.2 What counts as a verified hot path
A path is a **verified hot path** if at least one is true:

- it is explicitly marked as performance-critical by project docs or code-level markers,
- it is covered by a perf smoke / benchmark suite,
- profiling shows it dominates CPU time or allocations in realistic workloads,
- production telemetry indicates it dominates latency or throughput.

Without evidence, the agent SHOULD treat the path as normal code and avoid premature optimization.

### 6.3 Performance workflow routing
For measurement strategy, baseline selection, profiling, perf smoke / benchmark handling, hot-path optimization workflow, or performance reporting, the agent SHOULD use the global `backend-performance-evidence` skill.

When discussing performance, the agent SHOULD prefer evidence in this order:

1. production telemetry
2. realistic profiling
3. perf smoke / benchmark suite
4. microbenchmark
5. reasoned estimate

Reasoned estimates are the weakest form and SHOULD be labeled as such.

### 6.4 No silent perf regression
If the agent changes a verified hot path, the agent MUST:
- note whether perf risk exists,
- preserve existing evidence where feasible,
- avoid silent regression.

If direct measurement is not feasible, the agent MUST say so.

---

## 7) Browser automation and runtime verification

### 7.1 General rule
Browser-visible behavior MUST be treated as a runtime concern, not as something fully inferable from static code alone.

If the task affects browser-visible or runtime-rendered behavior, the agent SHOULD use an available runtime browser surface for verification.

This section decides whether browser runtime verification is needed and how to keep it scoped. Section 12 covers overall verification obligations; section 15 covers how to report the evidence.

Preferred surfaces depend on the current environment and prompt:

- Browser plugin / in-app browser when explicitly available or requested,
- Playwright MCP when it is the configured browser automation surface,
- global `playwright` skill / Playwright CLI when CLI-driven browser verification is the available path.

For CLI-driven Roehub browser automation, agents SHOULD use the global `playwright` skill wrapper instead of floating `npx` commands. The current wrapper contract pins `@playwright/cli@0.1.14` and relies on the matching Playwright browser cache. Agents SHOULD NOT switch to `@latest` or a floating `npx --package @playwright/cli` form unless intentionally refreshing the matching browser revision and reporting that change.

Roehub authenticated browser QA default:
- use the smoke Keycloak test account `smoke_e2e_keycloak` when a normal authenticated test user is needed and the user did not request another account;
- password source of truth on `macstudio`: `/Users/daniildegtyarev/.config/roehub/roehub.env` key `ROEHUB_SMOKE_E2E_PASSWORD`;
- when the browser automation runs outside `macstudio`, use a securely exported local `ROEHUB_SMOKE_E2E_PASSWORD` copied from that host-local source, or a password explicitly provided by the user in the current turn;
- never write the password into repo files, prompt artifacts, screenshots, traces, logs, stage ledgers, or final reports;
- if no password source is available, report browser auth as blocked instead of guessing credentials.

### 7.2 When runtime verification SHOULD be used
The agent SHOULD use runtime browser verification when the task involves one or more of:

- local web app verification,
- route or navigation validation,
- rendered UI state inspection,
- form submission flows,
- browser-visible defaults or settings,
- client-side behavior that depends on actual rendering,
- runtime checks where API surface and UI surface may diverge,
- end-to-end confirmation that a page or workflow actually works.

The agent SHOULD NOT use browser runtime verification for:

- direct code reading,
- repository search,
- type/lint/test analysis,
- backend-only refactors,
- static documentation review,
- pure architecture planning,
- low-level performance optimization unrelated to browser-visible behavior.

### 7.3 Evidence hierarchy and scope discipline
For browser-facing tasks, verification SHOULD prefer this order when relevant and available:

1. actual runtime observation through an available browser surface,
2. targeted automated tests,
3. code reasoning,
4. documentation reasoning.

Browser runtime verification MUST stay task-bounded:

- navigate only to relevant routes,
- inspect only task-relevant UI state,
- avoid unrelated flows,
- minimize noisy browsing,
- capture only evidence useful for the task.

### 7.4 Browser verification claims
When browser-visible behavior is in scope, the agent MUST label browser claims by evidence type:

- behavior observed through runtime browser verification,
- behavior verified through tests but not browser-observed,
- behavior inferred from code,
- behavior assumed but not yet verified.

If a browser-visible claim is made without runtime browser verification, the agent MUST present it as inference, not fact, unless the prompt explicitly limits work to static analysis.

---

## 8) Architectural expectations (pragmatic DDD / hexagonal)

DDD / ports-and-adapters is the default design language for this repo, not a mandate to remodel unrelated working code. Prefer local consistency and contract safety over making every touched file match an ideal architecture.

### 8.1 Dependency direction (DIP)
In Default Mode:

- domain MUST NOT import infrastructure,
- application SHOULD depend on ports and boundary DTOs,
- adapters implement ports and MAY depend on infra libraries.

In Speed Mode:
- the agent SHOULD preserve dependency direction,
- the agent MAY violate it temporarily only if it materially shortens delivery,
- the shortcut MUST be documented,
- recoverability MUST be preserved.

### 8.2 Ports & adapters
In Default Mode:
- ports SHOULD be `typing.Protocol` where practical, otherwise `abc.ABC`,
- adapters SHOULD be clearly separated by boundary direction.

In Speed Mode:
- ports/adapters MAY be simplified or temporarily inlined,
- but the boundary SHOULD remain recoverable.

### 8.3 DTO discipline
By default:
- DTOs are for boundaries,
- domain objects SHOULD NOT be serialized directly without explicit mapping,
- mapping SHOULD be explicit and test-covered.

In Speed Mode:
- minimal mapping MAY be used,
- but silent external payload drift is prohibited unless the prompt explicitly allows a contract change.

### 8.4 Cross-context integration
By default:
- cross-context translation SHOULD go through an ACL,
- one domain SHOULD NOT directly depend on another domain’s internal model.

In Speed Mode:
- temporary direct usage MAY be used,
- but MUST be recorded as debt with a remediation path.

### 8.5 Architecture restraint
The agent SHOULD avoid speculative abstractions, generalized extension points, and future-proofing not required by the task.

New abstraction SHOULD be introduced when at least one is true:
- the boundary already exists conceptually,
- multiple implementations already exist or are immediate,
- a contract must be protected,
- testing or isolation materially benefits.

Otherwise, the agent SHOULD prefer the simplest recoverable design.

---

## 9) OOP / SOLID rules (default strong, speed-tolerant)

### 9.1 Contracts first
Dependencies crossing a boundary SHOULD be expressed as a contract.
`Protocol` is preferred when practical.

Public methods SHOULD document:
- intent,
- input invariants / preconditions,
- error model.

In Speed Mode:
- documentation MAY be lighter,
- but failure modes MUST remain discoverable.

### 9.2 Avoid by default
In Default Mode, the agent SHOULD avoid:

- getters/setters as a dominant design style,
- static utility classes that hide dependencies,
- type switching (`isinstance`, type-based `match`) to emulate polymorphism,
- inheritance of implementation for reuse where composition is clearer.

These are not absolute bans, but they require justification when used in core design.

In Speed Mode:
- these MAY be used pragmatically,
- but tradeoffs and debt SHOULD be called out.

### 9.3 Immutability by default
- Value Objects SHOULD be immutable.
- Entities SHOULD protect invariants through explicit behavior.

In Speed Mode:
- controlled mutability MAY be used,
- but invariants MUST remain correct.

### 9.4 Primitive obsession
IDs, timestamps, symbols, money-like values, and similarly meaningful primitives SHOULD be wrapped where it materially improves clarity or correctness.

In Speed Mode:
- wrapping MAY be deferred,
- but conversions MUST remain consistent and explicit.

---

## 10) Error model (consistency and transparency)

### 10.1 Default expectation
In Default Mode:
- prefer typed domain/application exceptions over bare `Exception`,
- boundary layers SHOULD translate internal errors into stable outputs,
- exceptions MUST NOT be swallowed.

### 10.2 Speed Mode
In Speed Mode:
- pragmatic exceptions MAY be used,
- but exceptions still MUST NOT be swallowed,
- externally visible error stability SHOULD be preserved unless a breaking change is explicitly allowed.

### 10.3 Error transparency
The agent SHOULD make error behavior discoverable in:
- contracts,
- tests,
- docs where relevant,
- output report when behavior changed.

---

## 11) Contract evolution and impact classification

### 11.1 What counts as a contract change
A change is a **contract change** if it affects any of:

- a Port interface,
- a DTO schema,
- an API error payload,
- a persisted storage schema,
- a configuration schema,
- feature flags/default values/profile resolution rules relied upon externally or operationally,
- request-hash / cache-key / persistence-identity semantics,
- benchmark or rollout thresholds treated as delivery gates,
- browser-visible UI or runtime-default behavior relied upon by users or operations.

### 11.2 Default expectation
In Default Mode:
- backward compatibility SHOULD be preserved,
- if breaking is necessary, the agent MUST include:
  - migration or rollout notes,
  - versioning/deprecation notes where relevant,
  - tests/docs updates.

### 11.3 Speed Mode
In Speed Mode:
- backward compatibility SHOULD still be preserved,
- if the prompt explicitly allows breaking changes, the agent MUST:
  - label the break clearly,
  - give minimal migration notes,
  - list hardening follow-ups.

### 11.4 Contract impact classification (mandatory for non-trivial tasks)
For every non-trivial task, the agent SHOULD explicitly classify each relevant dimension as:

- `none`: no meaningful contract impact
- `compatible-change`: a contract surface changes but existing consumers should keep working
- `breaking-change`: existing consumers, persisted data, rollout assumptions, or externally relied-upon behavior may break
- `unknown`: evidence is insufficient; state the assumption and unresolved risk

Required dimensions:

- public API contract
- port contract
- DTO schema
- persisted schema
- config schema
- request hash / cache key / persistence identity semantics

If relevant, the agent SHOULD also classify:
- benchmark / rollout gate impact
- performance risk on verified hot path
- browser-visible behavior impact

For trivial tasks where a dimension is clearly unaffected, a concise yes/no statement is acceptable.

For the step-by-step analysis workflow behind this classification, the agent SHOULD use the global `contract-impact-analysis` skill.

### 11.5 Stability over incidental convenience
The agent MUST NOT silently change a contract just because a local implementation becomes easier.

---

## 12) Testing and runtime verification rules

### 12.1 Determinism (always)
Tests MUST NOT depend on:

- real wall-clock time without control,
- network access unless explicitly part of the task and properly isolated,
- randomness without fixed seed,
- uncontrolled external environment state,
- machine-specific timing assumptions unless explicitly marked as perf-only and tolerant.

### 12.2 Verification expectations
Changed behavior MUST receive relevant verification proportional to the boundary, risk, and active mode.

- domain behavior changed → verify invariants and behavior
- boundary contract changed → verify the changed boundary explicitly
- verified hot path changed → verify performance with appropriate evidence when feasible
- browser-visible behavior changed → use runtime browser verification when relevant and available

For this Python repository, prefer project wrappers when running quality gates:

- `uv run ruff check <target-or-.>`
- `uv run pyright`
- `uv run pytest -q <target-or-empty>`

Use focused targets first, then broaden only when risk or repository policy requires it. For concrete check ordering, selection, and failing-gate triage, the agent SHOULD use the global `backend-quality-gates` skill.

### 12.2.1 Stage validation depth
For non-trivial prompt-pack stages, rollout stages, migrations, runtime changes, or architecture-plan implementation, lint/type/unit tests are necessary but not sufficient acceptance evidence.

Each stage MUST define the nearest meaningful end-to-end or real-boundary validation surface for the changed behavior. Depending on scope, this can be one or more of:
- API route or use-case smoke through the real request/DTO boundary;
- persistence or migration verification against the relevant database boundary;
- browser/runtime QA for browser-visible flows;
- deployed runtime smoke, Mac Studio service smoke, file-hash parity, metrics, Monit/launchd checks, or production health checks for runtime/ops changes;
- benchmark/profile evidence for performance-sensitive or hot-path changes;
- safe real-adapter/runtime smoke for external integrations, with secrets and destructive actions excluded;
- CI/deploy verification when the stage is meant to ship.

Tests-only acceptance is allowed only for trivial, internal-only changes with no contract, persistence, browser-visible, runtime, ops, performance, or delivery impact. The agent MUST state why tests alone are sufficient when using this exception.

For staged work, the chosen validation depth and observed evidence MUST be recorded in the stage execution ledger and final report.

### 12.3 Speed Mode minimum bar
In Speed Mode, the agent MUST still provide at least one meaningful verification step or explicit manual verification instructions with expected outcomes.

If checks are skipped entirely, the agent MUST say so and explain why.

### 12.4 Review Mode
In Review Mode, the agent SHOULD state what tests and runtime verification would be needed if implementation were performed.

### 12.5 Verification transparency
The agent SHOULD distinguish clearly among:
- unit/integration test evidence,
- browser runtime evidence,
- code-based inference,
- unverified assumptions.

The agent MUST NOT equate passing backend tests with verified browser-visible correctness.

---

## 13) Documentation rules

### 13.1 Default expectation
When contracts, boundaries, errors, schemas, or operational behavior change, the agent SHOULD update impacted docs.

Docs SHOULD remain:
- consistent with code,
- consistent with naming,
- scoped to the actual change.

### 13.2 Speed Mode
Docs MAY be deferred in Speed Mode only if:
- the task prioritizes rapid delivery,
- the doc gap is recorded as debt,
- the required doc updates are listed explicitly.

### 13.3 Review and docs-sync workflow
For architecture review procedure, selective architecture-doc reading, findings structure, and docs-sync workflow, the agent SHOULD use the global `architecture-review` skill.

When reviewing docs, the agent SHOULD distinguish:
- docs that are outdated facts,
- docs that are missing,
- docs that are optional but useful.

### 13.4 Documentation restraint
The agent SHOULD update nearby or directly affected docs, but SHOULD NOT perform a broad documentation cleanup unless explicitly asked.

### 13.5 Browser/runtime docs
If a task changes browser-visible workflows, routes, defaults, or operational web behavior, the agent SHOULD update the relevant docs or explicitly note the doc gap.

### 13.6 Architecture document language and audience
New or materially rewritten architecture documents, ADRs, integration designs, and rollout plans under `docs/architecture/**` MUST be written in Russian by default unless a higher-priority instruction explicitly requires another language.

Architecture docs MUST keep code identifiers, API routes, env vars, config keys, schema names, metrics, file paths, and command examples in their original form.

For non-trivial architecture docs, the agent MUST include both:
- an engineering explanation: boundaries, dependency direction, contracts, data/storage impact, runtime behavior, operability, rollout, and verification;
- a business explanation: which capability, user workflow, operational outcome, cost, risk, or delivery constraint the architecture decision affects, written in terms a non-engineering stakeholder can understand.

Use short examples when they clarify a contract, data flow, rollout step, or operational failure mode. Do not add decorative examples when they do not make the decision easier to verify or use.

This rule does not change the `prompt-manager` contract: executor prompts and prompt packs MUST remain in English unless the user explicitly asks for another language.

### 13.7 User-facing final answer language
User-facing final answers, status reports, handoffs, stage results, verification summaries, and cold-head review additions MUST be written in Russian by default.

The agent MUST translate user-facing headings and explanatory prose such as `Verification`, `Key evidence`, `passed`, `failed`, `No files were staged`, and proof-boundary commentary into natural Russian.

Keep technical identifiers unchanged when translating the surrounding explanation:
- code identifiers, module names, classes, functions, API routes, env vars, config keys, schema names, metrics, hashes, process ids, file paths, commands, branch names, statuses in backticks, and exact artifact names;
- prompt artifacts produced by `prompt-manager`, when the artifact itself is supposed to remain English.

This is a final-answer contract, not permission to translate code, commands, paths, identifiers, or prompt-pack body content that intentionally remains English.

---

## 14) Implementation hygiene and quality gates

### 14.1 Repo-wide hygiene
All modes:

- dependencies SHOULD be explicit,
- hidden global state SHOULD be avoided,
- logging SHOULD be structured at boundaries,
- secrets MUST NOT be logged,
- configuration SHOULD be injected rather than read deep inside core logic,
- public APIs SHOULD be kept stable unless change is explicit.

### 14.2 Configuration as a contract
Configuration schema, feature flags, default values, and profile-resolution behavior MUST be treated as contracts when they are:
- externally relied upon,
- operationally significant,
- persisted,
- or used as rollout controls.

The agent MUST NOT silently change such behavior.

### 14.3 Quality gate awareness
If the task or prompt defines quality gates, the agent MUST treat them as part of the delivery contract.

The agent SHOULD:
- design changes so gates can pass incrementally,
- avoid unrelated edits that increase gate failure surface,
- distinguish between:
  - required gates for this task,
  - unrelated pre-existing repo-wide failures.

For concrete quality-gate ordering and failing-gate triage, the agent SHOULD use the global `backend-quality-gates` skill.

### 14.4 Minimal gate surface
The agent SHOULD avoid touching unrelated areas that create additional lint/type/test failures not needed for the requested change.

### 14.5 Runtime gate awareness
If a task includes browser-visible verification expectations, the agent SHOULD treat runtime browser checks as part of the delivery evidence, not as optional decoration.

---

## 15) Agent output contract

### 15.1 General rule
The agent’s response SHOULD be structured, reviewable, and explicit about:
- what changed,
- what did not change,
- what remains risky,
- where rules were relaxed,
- what was verified by runtime observation versus static reasoning.

If a prompt provides a stricter output schema, the prompt schema overrides this default response shape.

For small, single-file, or purely mechanical tasks, a concise final response is acceptable when verification, skipped checks, risks, and contract impact are still clear.

### 15.2 Default Mode response SHOULD include
1. **Intent** — what and why
2. **Scope** — areas touched
3. **Design** — boundaries/contracts and rationale
4. **Contract impact** — explicit `none` / `compatible-change` / `breaking-change` / `unknown` classification when non-trivial
5. **Tests** — added/updated and how to run
6. **Docs** — updated or intentionally deferred
7. **Performance** — hot path touched, evidence, or risk
8. **Runtime verification** — browser/runtime evidence when relevant
9. **Risks** — edge cases, compatibility, perf, debt

### 15.3 Speed Mode response SHOULD include
1. **Intent**
2. **Scope**
3. **What was simplified**
4. **Contract impact**
5. **Verification**
6. **Debt & follow-ups**
7. **Risks**

### 15.4 Review Mode response SHOULD include
1. **Observed state**
2. **Key findings**
3. **Contract / architecture risks**
4. **Recommended changes**
5. **What should be verified if implemented**
6. **Open uncertainties**

### 15.5 Perf Mode additions
If Perf Mode is active, the response SHOULD include when feasible:
- baseline vs new measurement,
- dominant cost targeted,
- what was optimized,
- why the path should stay fast,
- residual performance risks.

### 15.6 Runtime evidence distinction
When browser-visible behavior is in scope, the response SHOULD explicitly distinguish:
- observed through runtime browser verification,
- verified through tests but not browser-observed,
- inferred from code,
- not yet verified.

### 15.7 Stable section naming
For comparable tasks across iterations, the agent SHOULD keep section names stable so results are easy to diff and reuse.

---

## 16) Glossary

- **Object**: unit of behavior, not just a data bag.
- **Contract**: interface plus semantics plus error model.
- **Polymorphism**: behavior through a contract without type checks.
- **Encapsulation**: protection of invariants and representation.
- **Entity**: identity-based domain object.
- **Value Object**: immutable object with equality by value.
- **Aggregate / Root**: consistency boundary; only root referenced externally.
- **Invariant**: property that must always hold.
- **Domain Service**: domain logic not owned by a single Entity/VO.
- **Application Service / Use Case**: orchestration only; not the home of domain rules.
- **Repository**: persistence contract for aggregates or domain-relevant retrieval.
- **Domain Event**: something that happened in the domain.
- **Port / Adapter**: boundary contract and concrete implementation.
- **ACL**: anti-corruption layer preventing foreign models from leaking into the domain.
- **Verified hot path**: code path proven performance-critical by evidence.
- **Primary touch**: file directly required for the task.
- **Secondary touch**: file updated for compatibility, exports, tests, or nearby docs.
- **Speculative touch**: file not actually required for the task.
- **Browser-visible behavior**: behavior that must be confirmed through actual rendered/runtime UI state.
- **Runtime browser verification**: browser automation or inspection through an available surface such as the Browser plugin, Playwright MCP, or Playwright CLI.
