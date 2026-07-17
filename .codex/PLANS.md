# Roehub Historical Project Coordination Map

This document preserves a **historical project-level coordination map** for
long-horizon Roehub work.

It is **not** a replacement for:
- `.codex/AGENTS.md` — repository rules and durable behavior guidance
- task prompts — per-task delivery contracts
- architecture docs — domain-specific technical source of truth

It does not select current delivery artifacts or execution authority. The
global delivery contract and `.codex/AGENTS.md` do that now. The material below
is retained only to explain prior decisions across:
- architecture documentation,
- roadmap execution,
- milestone-driven refactors,
- large feature series,
- cross-context changes.

This file is **frozen historical evidence**. Do not update it for new work,
reopen a milestone from it, or treat its prompt/ledger references as current.

---

## 1) How to use this file

Do not use `PLANS.md` to route current work, even when it resembles:

- multi-iteration architecture documentation,
- roadmap work spanning multiple prompts,
- milestone-based implementation series,
- broad refactors with checkpoints,
- cross-context coordination,
- work that risks drifting without a stable execution map.

Do **not** use this file as the main control surface for:

- small local fixes,
- ordinary implementation tasks with clear scope,
- code review comments,
- one-off backend changes,
- small test-only updates.

For all current tasks, rely on `.codex/AGENTS.md`, the global delivery
contract, selected current ticket, and local code/docs/tests.

---

## 2) Rules for maintaining this file

This file must remain readable and compact.

### 2.1 Historical maintenance policy
The following was the former maintenance policy:

- active workstreams,
- milestone map,
- current checkpoint per active area,
- current blockers,
- recent decisions,
- recent iteration outcomes,
- next recommended prompts.

Do **not** turn this file into a full historical log.

### 2.2 Former rotation policy
When a section grows too large:

- keep only the current active summary here,
- move detailed history to an archive doc under `docs/` or `.codex/archive/`,
- leave a short note that points to the archive.

### 2.3 Former local-history limit
Keep at most:

- 5 recent decisions
- 5 recent iteration outcomes
- 5 active blockers/follow-ups

Older resolved items should be summarized or archived.

### 2.4 Former update threshold
Update this file when one of the following happens:

- current milestone changes,
- checkpoint scope changes,
- a blocker changes the execution path,
- a major decision affects future prompts,
- a workstream is paused, resumed, or reprioritized,
- acceptance criteria for the current checkpoint materially change.

Do not rewrite this file for minor code-only progress.

---

## 3) Project mission and execution principles

### 3.1 Mission
Roehub is an algorithmic trading platform covering the full cycle:

- self-hosted open-source installation and lifecycle management,
- market data ingestion,
- deterministic indicator computation,
- strategy configuration and live monitoring,
- historical backtesting,
- web-based control and inspection,
- future ML and live execution capabilities.

### 3.2 Core project principles
The execution of this project must preserve:

- **determinism** for backtests and indicator calculations,
- **bounded contexts** and DDD structure,
- **canonical 1m candle truth** in ClickHouse,
- **artifact-backed and scalable computation** where applicable,
- **operational reliability** with guards, leases, timeouts, and safe fallback behavior,
- **vertical slices** that end in working end-to-end value,
- **maintainable contracts** across APIs, configs, persistence, and runtime behavior.

### 3.3 Project-wide delivery priorities
Default priority order:

1. correctness and determinism
2. contract safety
3. maintainability and recoverability
4. performance where measured or strategically required
5. rollout safety and observability
6. polish and secondary cleanup

---

## 4) Global execution rules for long-horizon work

### 4.1 Milestone-first rule
Large work must be broken into milestones small enough to complete, validate, and review without forcing the agent to hold the whole roadmap in active context.

### 4.2 Stop-and-fix rule
If a checkpoint’s required validations fail, repair before proceeding to the next checkpoint.

Do not “continue anyway” across a failed checkpoint unless the prompt explicitly allows research-only progress.

### 4.3 Scope-bounding rule
Each iteration should solve one checkpoint or one tightly related checkpoint cluster.

Do not mix unrelated workstreams in a single implementation prompt unless the coupling is real and unavoidable.

### 4.4 Architectural drift rule
If implementation pressure is causing architectural drift:

- stop expanding scope,
- record the decision or risk in this file,
- create a new prompt for the next step instead of improvising a broad redesign inside the current one.

### 4.5 Documentation discipline
Architecture documentation work should advance checkpoint by checkpoint.
A document should move from:
- open questions,
- to candidate structure,
- to bounded decisions,
- to integrated source-of-truth status.

Avoid endless rewriting without checkpoint closure.

### 4.6 Browser-visible verification rule
If a workstream includes browser-visible behavior, routes, forms, runtime defaults, or web-surface changes, verification should include runtime browser verification when available and relevant.
Static code reasoning alone is not sufficient evidence for browser-visible correctness.

---

## 5) Historical workstream snapshot

This frozen table records the former strategic snapshot. It is not current
status or execution authority.

| Workstream | Goal | Current phase | Status | Current checkpoint |
|---|---|---|---|---|
| Market Data | reliable raw → canonical 1m ingestion across exchanges/markets | stabilizing / extending | active | будущий маршрут, реализация не начата: простой первый запуск и полный постраничный каталог поддерживаемых сегментов бирж |
| Indicators | deterministic tensor computation and registry-driven extensibility | core established, extension ongoing | active | controlled indicator and engine evolution |
| Strategy / Live | immutable strategy specs, live runner, realtime visibility | operational foundation | active | preserve reliable live orchestration and monitoring |
| Backtest | scalable artifact-backed backtesting with staged runtime evolution | active major stream | active | optimize runtime while preserving exact scorer and rollout safety |
| Web UI / Gateway | same-origin browser control surface for backtests and strategies | active | active | keep browser flows consistent with backend/runtime contracts |
| ML | future feature registry and inference path | planned | planned | no active checkpoint |
| Live Execution | future order routing and execution gateway | planned | planned | no active checkpoint |
| Notifications | provider-neutral user/admin notifications, Telegram bot, stats and reports | staged pack closed | active | Stages `00`-`11` closed on `main`; next step is a separate user-approved real Telegram canary/rollout beyond test/smoke recipients |
| Self-Hosted OSS Platform | one release bundle, local-first auth, multi-org RBAC, plugins, admin/control plane, container runtime and migration | historical accepted evidence | superseded | no current checkpoint; future work starts from a separately selected current ticket |
| Cross-cutting Architecture / Docs | roadmap integrity, design docs, milestone coordination | always-on | active | keep architecture docs aligned with delivery |

---

## 6) Milestone map by workstream

This section is strategic, not exhaustive.
It should tell any future agent where the project is headed without forcing them to read every roadmap doc first.

### 6.1 Market Data
High-level milestones:
- M1 delivered core raw → canonical ingestion
- ongoing hardening around:
  - gap detection,
  - REST repair,
  - instrument sync,
  - operational guards,
  - tail-safe dedup behavior

### 6.2 Indicators
High-level milestones:
- M2 delivered registry + compute engine foundation
- ongoing work focuses on:
  - indicator coverage,
  - deterministic compute semantics,
  - guardrails,
  - kernel-group evolution,
  - tensor contract stability

### 6.3 Strategy / Live
High-level milestones:
- M3 delivered identity + strategy v1 + live runner + realtime + Telegram
- ongoing work focuses on:
  - stronger operational behavior,
  - safer strategy lifecycle,
  - realtime observability,
  - eventual bridge to live execution

### 6.4 Backtest
High-level milestones:
- M4 delivered backtest v1
- M5 delivered async jobs / persisted top-K
- M7 active: optimize/grid/pruning, staged runtime, memory/perf guards
- M8 active/planned: backtest v2 capabilities such as intrabar fills, portfolio engine, advanced risk
- current major emphasis:
  - artifact-backed runtime evolution,
  - shortlist/pruning acceleration,
  - exact scorer preservation,
  - rollout-safe approximation,
  - operational scalability

### 6.5 Web UI / Gateway
High-level milestones:
- M6 delivered SSR + HTMX + same-origin gateway + auth flows
- current work focuses on:
  - browser-visible launch and monitoring flows,
  - runtime-default exposure,
  - UX consistency with backend contracts,
  - safe integration with backtest and strategy flows

### 6.6 ML
High-level milestones:
- M9 planned
- intended direction:
  - feature registry,
  - model inference surfaces,
  - deterministic integration boundaries,
  - eventual cooperation with strategy/backtest contexts

### 6.7 Live Execution
High-level milestones:
- M10 planned
- intended direction:
  - execution gateway contracts,
  - order management,
  - exchange-specific adapter isolation,
  - safe operational controls

### 6.8 Notifications
High-level milestones:
- M11 delivered provider-neutral foundation through the Stage `00`-`11` prompt pack
- current rollout boundary:
  - provider-neutral `notifications` bounded context is implemented and documented,
  - Telegram bot binding, commands, delivery queue, stats and reports have staged evidence,
  - user modes for critical-only, signals, trades and reports are additive settings surfaces,
  - admin critical alerts, ops alerts and reports have synthetic/log-only evidence,
  - real Telegram expansion beyond test/smoke recipients still requires a separate user-approved canary/rollout prompt
- execution contract:
  - all stages run in `/Users/daniildegtyarev/Projects/roehub.com`,
  - the only working branch is `main`,
  - future prompts must include `.codex/agents/generated/web-execution-telegram-notifications-v1/00-main-and-stage-execution-contract.md`,
  - branch/worktree/stash workflows are forbidden unless the user explicitly changes this repo contract

### 6.9 Cross-cutting Architecture / Docs
Always-on responsibilities:
- preserve bounded-context clarity,
- keep roadmap docs internally consistent,
- prevent milestone overlap and accidental scope drift,
- maintain a clear “what is canonical now” picture.

### 6.10 Self-Hosted OSS Platform

Historical execution sources (superseded):

- `plan_doc`: `docs/architecture/platform/roehub-self-hosted-oss-platform-v1.md`;
- `prompt_pack_dir`: `.codex/agents/generated/roehub-self-hosted-oss-platform-v1/`;
- `stage_ledger`: `docs/architecture/platform/roehub-self-hosted-oss-platform-v1-stage-reports/roehub-self-hosted-oss-platform-v1-stage-ledger.md`;
- `execution_mode`: `superseded`.

This pack has no autonomous Goal. Its stages are historical evidence and never
authorize an execution, publication, or runtime action.

---

## 7) Current focus

This section is a historical snapshot and is not operational.

### Former primary project focus
- Требования следующей продуктовой трансформации зафиксированы в
  `docs/architecture/platform/roehub-product-transformation-requirements-v1.md`;
  документ не является исполнительным планом, а каждая следующая работа
  выбирается отдельным текущим ticket по действующему delivery contract
- Backtest runtime evolution under staged, rollout-safe architecture
- Notifications final rollout boundary: staged pack closed; real Telegram canary requires user-approved recipient scope
- Web/runtime contract consistency for browser-visible flows
- Architecture-document iteration across active streams
- Cross-stream planning discipline to keep milestones bounded

### Secondary active focus
- Indicators extension without breaking deterministic compute contracts
- Market data reliability and maintenance operations
- Strategy/live operational hardening

### Historically deferred
- Self-Hosted OSS Platform production cutover Stages `24`–`25`; the old stages
  are superseded and cannot be resumed
- Kubernetes, Podman, Colima and multi-node HA for the self-hosted release
- broad cross-repo cleanup
- unmeasured macOS MPS/GPU support claims

---

## 8) Historical checkpoint

This section should describe the single most important current planning checkpoint.

### Current checkpoint
There is no current checkpoint in this frozen file. New work starts only from a
separately selected current ticket under the active delivery contract.

### What “good” looks like right now
- each major change belongs to a named workstream and checkpoint
- architecture docs converge through explicit iterations
- large prompts do not reload the whole project model every time
- each checkpoint has clear validation and explicit non-goals
- browser-visible changes get browser-visible verification
- long-horizon work remains resumable after interruption

### What to avoid right now
- mixing roadmap authoring and broad implementation in one uncontrolled step
- turning `PLANS.md` into a giant historical diary
- letting large prompts become the only memory system
- overloading `.codex/AGENTS.md` with milestone-specific or temporary planning state

---

## 9) Checkpoint acceptance rules

A checkpoint is considered complete only when all relevant items are true:

- scope is still bounded and matches the intended milestone
- required contracts were preserved or explicitly evolved
- tests and validations for the checkpoint were run or explicitly deferred with reason
- docs were updated or a doc debt note was recorded
- relevant runtime/browser behavior was verified when applicable
- next step is clear enough that a new prompt can continue from current state

If one of these is not true, the checkpoint should be treated as incomplete.

---

## 10) Validation philosophy

This section is strategic and stable.

### 10.1 Validation preference order
Prefer evidence in this order:

1. deterministic runtime behavior
2. targeted tests
3. contract-level review
4. code reasoning
5. roadmap/spec alignment

### 10.2 Browser-visible work
For browser-visible work, prefer:

1. runtime browser verification through the available browser surface
2. targeted tests
3. code reasoning

### 10.3 Performance-sensitive work
For performance-sensitive work, prefer:

1. existing perf smoke / benchmarks
2. realistic profiling
3. carefully labeled estimates when measurement is unavailable

### 10.4 Architecture-document work
For architecture-document work, validation means:
- bounded decisions,
- internal consistency,
- alignment with current code/contracts,
- explicit non-goals,
- clear next checkpoint.

---

## 11) Current blockers and risks

Keep this list short and current.

- Large architecture prompts can still drift if they are not checkpoint-bounded.
- Browser-visible behavior may be misjudged if prompts forget to require runtime browser verification through an available browser surface.
- Backtest/runtime work remains the highest-risk source of scope explosion because it crosses performance, contracts, docs, and rollout.
- Notifications real Telegram provider expansion remains deferred until an approved recipient/canary scope and active route readiness are available.
- A single global `PLANS.md` can become noisy unless old decisions and outcomes are rotated aggressively.
- Future ML and live execution work can distort current architecture if introduced too early into active planning.
- The self-hosted migration remains a repository-wide risk; each selected
  current ticket must keep component coverage, runtime facts and contract
  migrations explicit before code changes begin.

---

## 12) Recent decisions

Keep only the most recent, still-relevant decisions here.

1. Roehub target is a self-hosted Apache-2.0 release bundle: one supported command, multiple containers, Linux amd64/arm64 and Docker Desktop macOS.
2. Keep the control plane as a modular monolith; isolate workers, heavy jobs, exchange execution, plugins and host operations at real failure/trust boundaries.
3. Clean install uses local passkey-first auth; generic OIDC is optional and current Keycloak migrates through hybrid mode without changing internal `user_id`.
4. Roehub DB owns organizations/RBAC; `admin` manages roles and plugins while owner/recovery/mainnet invariants remain separately protected.
5. Нормальный путь новой self-hosted-установки не использует файл ticket,
   пользовательские recovery codes или ручной `OpenBao` handoff; полный
   каталог требует snapshot/cursor contract и явно разделяет `bybit:linear` и
   пока неподдерживаемые сегменты. Это будущий маршрут: исторические accepted
   reports и ledger прежнего self-hosted плана остаются источником фактов, а
   реализация начнётся только из отдельно выбранного текущего ticket по
   действующему delivery contract.

---

## 13) Recent iteration outcomes

Keep this list current and short.

1. The self-hosted target architecture was accepted after iterative decisions on packaging, boundaries, auth, plugins, administration and observability.
2. A Russian plan, one execution ledger and an English goal-driven prompt pack for Stages `00`–`25` were created and linked.
3. All 33 current project-map components are mapped to implementation stages; four target components are planned: `context:extensions`, `context:operations`, `app:control_agent`, `app:roehubctl`.
4. Host-specific service-manager and Keycloak-only target documents were marked superseded while current runtime facts remain valid as historical evidence.
5. Notifications Stages `00`-`11` remain closed historical evidence; their provider semantics feed the new plugin migration stage.

---

## 14) Next recommended prompt shapes

Use this as a practical handoff guide.

### For architecture-doc iteration
Prompt should:
- name the workstream,
- name the checkpoint,
- state what decisions must be finalized now,
- state what must remain unresolved or out of scope,
- require bounded reading,
- update the relevant doc plus this `PLANS.md` if milestone state changes.

### For implementation milestone work
Prompt should:
- reference the current workstream and checkpoint,
- include milestone acceptance criteria,
- include quality gates,
- keep touched areas narrow,
- update docs only where directly relevant.

### For browser-visible tasks
Prompt should:
- explicitly say browser-visible behavior is in scope,
- require runtime browser verification through an available browser surface,
- distinguish browser evidence from tests and static reasoning,
- name the specific route/page/flow/default being checked.

### For performance-sensitive tasks
Prompt should:
- identify the verified hot path,
- state what evidence exists,
- define what must not regress,
- keep optimization and contract changes clearly separated.

---

## 15) Archive policy

This file must stay compact.

When any of these happen:
- recent decisions > 5
- recent outcomes > 5
- blockers section becomes stale or long
- a workstream needs detailed milestone history

move detailed material into archive docs such as:

- `.codex/archive/plans-YYYY-MM.md`
- `docs/architecture/_history/...`

Then keep only:
- current summary,
- archive reference,
- active checkpoint state.

---

## 16) Minimal maintenance protocol

When updating this file:

1. update only the sections that materially changed
2. keep wording direct and compact
3. do not copy large prompt text into this file
4. do not turn this into a changelog
5. preserve strategic top sections unless the project model itself changed
6. keep operational sections current enough that work can resume after interruption

---

## 17) Practical rule of thumb

If a future agent asks:
“Do I need `PLANS.md` for this task?”

Use it if the task is:
- milestone-based,
- multi-iteration,
- architecture-heavy,
- cross-context,
- drift-prone,
- or likely to resume across multiple prompts.

Do not use it if the task is:
- small,
- local,
- already fully bounded by one prompt,
- or ordinary implementation with no planning ambiguity.
