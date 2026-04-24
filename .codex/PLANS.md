# Roehub Project Execution Plan

This document is the **project-level execution map** for long-horizon work in Roehub.

It is **not** a replacement for:
- `.codex/AGENTS.md` — repository rules and durable behavior guidance
- task prompts — per-task delivery contracts
- architecture docs — domain-specific technical source of truth

Its purpose is to keep multi-iteration work stable across:
- architecture documentation,
- roadmap execution,
- milestone-driven refactors,
- large feature series,
- cross-context changes.

This file is a **living document**.
It should stay:
- concise,
- current,
- checkpoint-oriented,
- useful after interruptions.

---

## 1) How to use this file

Use `PLANS.md` when work is any of:

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

For ordinary tasks:
- rely on `.codex/AGENTS.md`,
- task prompt,
- local code/docs/tests.

For long-horizon tasks:
- use this file as the stable project map,
- use task prompts as milestone or checkpoint contracts,
- keep this file updated when milestone status or decisions materially change.

---

## 2) Rules for maintaining this file

This file must remain readable and compact.

### 2.1 Keep only active planning state
Keep here only:

- active workstreams,
- milestone map,
- current checkpoint per active area,
- current blockers,
- recent decisions,
- recent iteration outcomes,
- next recommended prompts.

Do **not** turn this file into a full historical log.

### 2.2 Rotation policy
When a section grows too large:

- keep only the current active summary here,
- move detailed history to an archive doc under `docs/` or `.codex/archive/`,
- leave a short note that points to the archive.

### 2.3 Maximum local history
Keep at most:

- 5 recent decisions
- 5 recent iteration outcomes
- 5 active blockers/follow-ups

Older resolved items should be summarized or archived.

### 2.4 Update threshold
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

## 5) Active workstreams

This section tracks the major project streams at a strategic level.

| Workstream | Goal | Current phase | Status | Current checkpoint |
|---|---|---|---|---|
| Market Data | reliable raw → canonical 1m ingestion across exchanges/markets | stabilizing / extending | active | maintain deterministic ingestion and operational reliability |
| Indicators | deterministic tensor computation and registry-driven extensibility | core established, extension ongoing | active | controlled indicator and engine evolution |
| Strategy / Live | immutable strategy specs, live runner, realtime visibility | operational foundation | active | preserve reliable live orchestration and monitoring |
| Backtest | scalable artifact-backed backtesting with staged runtime evolution | active major stream | active | optimize runtime while preserving exact scorer and rollout safety |
| Web UI / Gateway | same-origin browser control surface for backtests and strategies | active | active | keep browser flows consistent with backend/runtime contracts |
| ML | future feature registry and inference path | planned | planned | no active checkpoint |
| Live Execution | future order routing and execution gateway | planned | planned | no active checkpoint |
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

### 6.8 Cross-cutting Architecture / Docs
Always-on responsibilities:
- preserve bounded-context clarity,
- keep roadmap docs internally consistent,
- prevent milestone overlap and accidental scope drift,
- maintain a clear “what is canonical now” picture.

---

## 7) Current focus

This section is operational and should stay current.

### Primary active project focus
- Backtest runtime evolution under staged, rollout-safe architecture
- Web/runtime contract consistency for browser-visible flows
- Architecture-document iteration across active streams
- Cross-stream planning discipline to keep milestones bounded

### Secondary active focus
- Indicators extension without breaking deterministic compute contracts
- Market data reliability and maintenance operations
- Strategy/live operational hardening

### Currently deferred
- ML implementation
- Live execution implementation
- broad cross-repo cleanup
- non-essential platform polish

---

## 8) Active checkpoint

This section should describe the single most important current planning checkpoint.

### Current checkpoint
Maintain and evolve the project through **bounded milestone execution** rather than broad redesign.

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
- A single global `PLANS.md` can become noisy unless old decisions and outcomes are rotated aggressively.
- Future ML and live execution work can distort current architecture if introduced too early into active planning.

---

## 12) Recent decisions

Keep only the most recent, still-relevant decisions here.

1. Use one global `PLANS.md` for all workstreams, not per-subsystem plans.
2. Keep `.codex/AGENTS.md` for durable repo rules; keep `PLANS.md` for long-horizon execution state.
3. Prefer bounded, milestone-shaped prompts over broad “read and redesign everything” prompts.
4. Treat browser-visible verification as a first-class requirement when web/UI behavior is in scope.
5. Keep this file strategic on top and operational below; rotate detail out instead of letting it grow indefinitely.

---

## 13) Recent iteration outcomes

Keep this list current and short.

1. Repo-level agent guidance has been tightened for bounded context loading and anti-legacy prompt shaping.
2. Prompt-generation workflow has been migrated away from flat broad-reading prompt structure.
3. Browser/runtime verification is being incorporated as a first-class concern through available browser automation surfaces.
4. A single project-level execution map is now recognized as necessary for multi-iteration design and roadmap work.

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
