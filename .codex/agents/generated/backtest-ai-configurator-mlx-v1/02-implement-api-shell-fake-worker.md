---
prompt_name: backtest_ai_configurator_mlx_v1_02_api_shell_fake_worker
repo: roehub.com
branch: main
scope: "Iteration 02: implement browser-visible /backtests/ai-config API shell, SSE/status events, feedback route, and deterministic fake worker without MLX."

language:
  implementation: python_fastapi
  agent_report: ru

context_sources:
  always_read:
    - path: .codex/AGENTS.md
      why: "repo contract and safety invariants"
    - path: docs/architecture/backtest/backtest-ai-configurator-mlx-v1.md
      why: "route, DTO and UX contract"
  task_entrypoints:
    - path: apps/api/routes/backtests.py
      why: "nearby backtest route style"
      inspect_symbols:
        - register_backtest_routes
    - path: apps/api/routes/ui_backtests.py
      why: "browser-visible /api vs backend route convention"
      inspect_symbols:
        - register_ui_backtests_routes
    - path: apps/api/wiring/modules/backtest.py
      why: "composition root and dependency wiring"
      inspect_symbols:
        - build_backtest_use_cases
    - path: src/trading/contexts/backtest/application/ai_configurator
      why: "Iteration 01 storage/service contracts"
      inspect_symbols:
        - "*"
  conditional_bundles:
    identity_auth:
      read_when: "when adding auth/current-user dependencies"
      paths:
        - apps/api/routes/identity.py
        - apps/api/dependencies
    sse_patterns:
      read_when: "if existing streaming/event response helpers exist"
      paths:
        - apps/api
        - tests/unit/apps/api
    storage_foundation:
      read_when: "if Iteration 01 artifacts are missing or incomplete"
      paths:
        - src/trading/contexts/backtest/application/ai_configurator
        - src/trading/contexts/backtest/adapters/outbound/persistence/postgres
  consult_if_needed:
    - path: tests/unit/apps/api/test_backtests_routes.py
      read_when: "for API test style or auth fixtures"
    - path: .codex/agents/.context/promt_manager_state.yaml
      read_when: "only to check for a newer executor handoff; ignore if stale/unrelated"

style_references:
  - path: apps/api/routes/backtests.py
    purpose: "route registration and error mapping style"
  - path: tests/unit/apps/api/test_backtests_routes.py
    purpose: "API test fixture style"

hard_requirements:
  depends_on_iteration_01: true
  additive_routes_only: true
  auth_owner_scope_required: true
  fake_worker_only: true
  sse_required: true
  no_mlx_runtime: true
  friendly_capacity_payloads: true
  publish_ci_deploy_required: true
  main_branch_deployment_required: true
  macstudio_sync_required: true

task_toggles:
  implement_api_routes: true
  implement_fake_worker: true
  implement_sse: true
  implement_feedback: true
  implement_catalog_validator: false
  implement_mlx: false
  implement_ui: false

skill_routing:
  - skill: contract-impact-analysis
    use_when: "adding browser-visible routes, DTOs, statuses, errors, feedback contract"
    timing: "before implementation"
    reason: "public API and user-visible error model"
  - skill: backend-quality-gates
    use_when: "running API route, SSE and DTO tests"
    timing: "during verification"
    reason: "backend route quality gates"
  - skill: root-cause-debugging
    use_when: "auth/SSE/repository tests fail unexpectedly"
    timing: "if blocker"
    reason: "reproduce and localize route failures"
  - skill: publish-ci-deploy
    use_when: "after implementation and local gates pass, deliver this iteration to main, sync Mac Studio, and run post-deploy verification"
    timing: "final delivery step"
    reason: "required end-to-end Roehub GitHub CI, main deployment, Mac Studio sync and smoke"

target_envs:
  - local-dev
  - unit-tests
  - github-actions
  - mac-studio-prod

required_literals:
  - "POST /api/backtests/ai-config/jobs"
  - "GET /api/backtests/ai-config/jobs/{job_id}"
  - "GET /api/backtests/ai-config/jobs/{job_id}/events"
  - "POST /api/backtests/ai-config/jobs/{job_id}/feedback"
  - "queued"
  - "capacity_delayed"
  - "Загрузить конфигурацию"

non_goals:
  - "Do not call MLX or build prompt profiles in this iteration."
  - "Do not enable the browser AI panel yet."
  - "Do not change /api/backtests/jobs or existing job create behavior."
  - "Do not expose raw LLM attempts."

final_report_format:
  language: ru
  sections:
    - "Что реализовано"
    - "API контракт"
    - "Проверки"
    - "Доставка и Mac Studio"
    - "Остаточные риски"
    - "Следующая итерация"

quality_gates:
  - cmd: "uv run pytest -q tests/unit/apps/api/test_backtest_ai_config_routes.py tests/unit/apps/api/test_backtests_routes.py"
    expect: "passes"
  - cmd: "uv run ruff check apps/api src/trading/contexts/backtest tests/unit/apps/api"
    expect: "passes"
  - cmd: "uv run pyright"
    expect: "passes"
  - cmd: "git diff --check"
    expect: "passes"

expected_primary_touches:
  - "apps/api/routes/backtest_ai_config.py"
  - "apps/api/dto/backtest_ai_config.py"
  - "apps/api/wiring/modules/backtest_ai_config.py"
  - "src/trading/contexts/backtest/application/ai_configurator/"
  - "tests/unit/apps/api/test_backtest_ai_config_routes.py"

possible_secondary_touches:
  - "apps/api/app.py"
  - "apps/api/wiring/modules/backtest.py"
  - "tests/unit/apps/api/conftest.py"

safety_notes:
  - "All route reads and event streams must enforce owner_user_id."
  - "429 may be used at HTTP level, but response body must contain a friendly message."
  - "SSE events are observable stages only, never model reasoning."
---

# Task

Implement Iteration 02 of the `/backtests` AI Configurator: browser-visible API shell, status read, SSE events, feedback route, admission/quota responses, and a deterministic fake worker path. This iteration proves API shape and UX state transitions without MLX.

Done means:

- `POST /api/backtests/ai-config/jobs` creates or returns an idempotent AI config job;
- capacity/quota/admission failures return friendly payloads with `estimated_wait_seconds` or `retry_after_seconds`;
- `GET /api/backtests/ai-config/jobs/{job_id}` enforces owner scope and returns the current snapshot;
- `GET /api/backtests/ai-config/jobs/{job_id}/events` streams or replays observable status events;
- `POST /api/backtests/ai-config/jobs/{job_id}/feedback` records apply feedback;
- a fake deterministic worker/service can move one job through `queued -> preparing_catalog -> generating -> validating_business -> ready` with a valid placeholder config shape;
- no model runtime is called.

## Context / Current State

Context ledger:

- completed:
  - Iteration 01 should have created AI storage, DTOs, repository ports and quota primitives.
- open_items:
  - no real catalog resolver, prompt policy, MLX adapter or UI integration yet.
- contract_changes:
  - additive `/backtests/ai-config/*` browser-visible routes;
  - existing `/backtests/jobs` route remains unchanged.
- risks:
  - owner scope leaks through status/event routes;
  - duplicate jobs from browser retry if idempotency is ignored;
  - UI-visible overload states degrade into raw HTTP errors.
- next_focus:
  - prove API contract and fake pipeline before adding catalog/validation.

## Requirements (Must)

- Verify Iteration 01 storage contracts exist; if not, stop and report blocker.
- Add routes exactly under backend router path `/backtests/ai-config/*`; browser-visible path remains `/api/backtests/ai-config/*`.
- Require auth/current user on create/read/events/feedback.
- Enforce owner scope on job status, SSE stream and feedback.
- Implement idempotency on create using `(owner_user_id, idempotency_key)`.
- Implement friendly payload for quota/capacity delay; do not return only an error code to UI consumers.
- Implement event names from the plan, with fake worker only producing observable stages.
- Include terminal states `ready`, `needs_clarification`, `blocked_by_policy`, `input_too_large`, `security_review`, `failed`.
- Feedback must be additive and must not mutate validated config.
- Tests must cover auth required, owner forbidden, idempotent retry, capacity response, SSE event smoke, and feedback.

## Requirements (Should)

- Keep route DTOs separate from persistence rows.
- Make fake worker deterministic and easy to remove/replace in later iterations.
- Use small response payloads and avoid returning raw audit attempts.
- If SSE implementation is heavy, implement a minimal compliant event stream backed by stored events and status snapshot.

## Requirements (Nice-to-have)

- Add heartbeat events if local SSE helper patterns make it simple.
- Add route-level docs/comments only where they explain non-obvious idempotency or SSE behavior.

# Context acquisition protocol

Read only in this order and stop once sufficient:

1. `.codex/AGENTS.md`
2. `.codex/agents/.context/promt_manager_state.yaml` or latest state snapshot, if available and relevant
3. latest executor final report, if available
4. task entrypoints
5. only conditional bundles required by auth/SSE/storage ambiguity
6. consult-if-needed references only for blockers

Do not eagerly preload all listed sources.

Pre-implementation reading target:

- `<= 8 files`
- `<= ~35k-50k tokens`

Stop reading once route registration, owner scope, repository calls and DTO shape are clear.

# Reading manifest

Use the front-matter `context_sources` as the canonical reading map.

Read with this intent:

- `always_read`: repo rules and AI contract.
- `task_entrypoints`: route style, wiring style, Iteration 01 contracts.
- `conditional_bundles`: auth and SSE helpers only if needed.
- `consult_if_needed`: test style or state only for blockers.

Do not convert this manifest into a broad mandatory reading list.

# Work plan (agent should follow)

Skill routing for this task:

- `contract-impact-analysis`: use before implementation; owns public route/DTO/error compatibility.
- `backend-quality-gates`: use during verification; owns API tests and lint/type gates.
- `root-cause-debugging`: use only when auth/SSE/repository tests fail.

1. Verify Iteration 01 primitives and route/wiring conventions.
2. Define API DTOs for create/read/events/feedback and friendly capacity/quota responses.
3. Add route registration and wiring without touching existing backtest job routes.
4. Implement fake pipeline/event progression through application service methods.
5. Add focused API tests for auth, owner scope, idempotency, SSE and feedback.
6. Run quality gates and report contract impact.

# Acceptance criteria (Definition of Done)

- API routes exist and are additive.
- Unauthorized users cannot create/read/stream/feedback.
- Users cannot access another user's AI config job or events.
- Duplicate create with same idempotency key returns same job and does not double-charge quota.
- Capacity/quota response includes user-facing message and retry/estimated wait data.
- SSE stream emits observable status events and never chain-of-thought.
- Existing backtest route tests still pass.

- `publish-ci-deploy` terminal state is `deployed`, or `green-pr`/`blocked` is reported with exact blocker evidence.

# Implementation constraints

## Determinism & ordering

- Fake worker statuses must be deterministic and testable.
- SSE tests must avoid sleeps where repository event snapshots can be asserted directly.

## API / contracts

- Browser-visible notation is `/api/backtests/ai-config/*`; backend route path is `/backtests/ai-config/*`.
- Do not add a second `/api` prefix in backend router paths.
- Do not expose raw LLM attempt rows.

## Security

- Treat `current_config` and `message` as untrusted input, even before security gates exist.
- Cap request body/message size at the API boundary or record a clear blocker if a common limiter exists elsewhere.

# Files to indicate (expected touched areas)

Expected primary touches:

- `apps/api/routes/backtest_ai_config.py`
- `apps/api/dto/backtest_ai_config.py`
- `apps/api/wiring/modules/backtest_ai_config.py`
- `src/trading/contexts/backtest/application/ai_configurator/`
- `tests/unit/apps/api/test_backtest_ai_config_routes.py`

Possible secondary touches:

- `apps/api/app.py`
- `apps/api/wiring/modules/backtest.py`
- `tests/unit/apps/api/conftest.py`

# Non-goals

- No MLX adapter.
- No real prompt policy.
- No real catalog resolver.
- No browser UI enablement.
- No production launchd/Monit changes.

# Quality gates (must run and pass)

- `uv run pytest -q tests/unit/apps/api/test_backtest_ai_config_routes.py tests/unit/apps/api/test_backtests_routes.py`
- `uv run ruff check apps/api src/trading/contexts/backtest tests/unit/apps/api`
- `uv run pyright`
- `git diff --check`

If a gate cannot run, classify it as introduced, required-path pre-existing, unrelated pre-existing, environmental, or flaky.

Required delivery step: after the quality gates above pass, invoke `publish-ci-deploy` as the final step. The expected terminal state for this prompt is `deployed`: intended files committed and pushed, GitHub Actions green, revision shipped to `main`, `/opt/roehub/app` on `macstudio` pulled to that revision, the relevant production services reloaded through the repository runbook, and `bash scripts/macos/smoke_prod.sh` passed. If the skill reaches `green-pr` because a human merge/approval is required, or `blocked` because of missing auth, unrelated dirty scope, external CI, Mac Studio access, or production verification failure, report that exact state and do not claim deployment.

# Final output: report format (strict)

Report in Russian with:

- `Что реализовано`: routes, DTOs, fake worker, events, feedback.
- `API контракт`: request/response/status behavior and compatibility classification.
- `Проверки`: exact commands and results.
- `Доставка и Mac Studio`: publish-ci-deploy terminal state, main/PR SHA, CI result, Mac Studio pull/reload/smoke evidence, or exact blocker.
- `Остаточные риски`: missing real catalog/model/security gates.
- `Следующая итерация`: catalog resolver, validation and deterministic security gates.
