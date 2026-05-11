---
prompt_name: backtest_ai_configurator_mlx_v1_04_prompt_profiles_repair_llm_gateway
repo: roehub.com
branch: main
scope: "Iteration 04: implement versioned prompt profiles, structured prompt envelope, LLM gateway port with deterministic test adapter, parse/repair loop, and audit attempts without MLX runtime."

language:
  implementation: python_fastapi
  agent_report: ru

context_sources:
  always_read:
    - path: .codex/AGENTS.md
      why: "repo contract and safety rules"
    - path: docs/architecture/backtest/backtest-ai-configurator-mlx-v1.md
      why: "prompt, repair and gateway contract"
  task_entrypoints:
    - path: src/trading/contexts/backtest/application/ai_configurator
      why: "existing AI pipeline/gates/storage"
      inspect_symbols:
        - "*"
    - path: src/trading/contexts/backtest/application/services/v2/preflight.py
      why: "business validation after model draft"
      inspect_symbols:
        - BacktestPreflightService
    - path: apps/api/routes/backtest_ai_config.py
      why: "API integration for pipeline terminal states"
      inspect_symbols:
        - "*"
    - path: tests/unit/contexts/backtest/application/ai_configurator
      why: "existing AI validation test shape"
      inspect_symbols:
        - "*"
  conditional_bundles:
    prompt_examples:
      read_when: "if prompt file/template location is unclear"
      paths:
        - src
        - configs
    audit_attempts:
      read_when: "when persisting LLM attempts or repair attempts"
      paths:
        - src/trading/contexts/backtest/adapters/outbound/persistence/postgres
        - tests/unit/contexts/backtest/application/ai_configurator
  consult_if_needed:
    - path: docs/architecture/backtest/backtest-ai-configurator-mlx-v1.md
      read_when: "for exact structured prompt envelope and forbidden prompt contents"
    - path: .codex/agents/.context/promt_manager_state.yaml
      read_when: "only to check for a newer executor handoff; ignore if stale/unrelated"

style_references:
  - path: src/trading/contexts/backtest/application/ai_configurator
    purpose: "local service/port style from prior iterations"
  - path: docs/architecture/backtest/backtest-ai-configurator-mlx-v1.md
    purpose: "prompt policy and repair loop source"

hard_requirements:
  depends_on_iteration_03: true
  no_real_mlx_runtime: true
  llm_gateway_port_required: true
  deterministic_test_adapter_required: true
  repair_attempts_max_one: true
  raw_attempt_audit_required: true
  no_secret_prompt_material: true
  publish_ci_deploy_required: true
  main_branch_deployment_required: true
  macstudio_sync_required: true

task_toggles:
  implement_prompt_profiles: true
  implement_llm_gateway_port: true
  implement_repair_loop: true
  implement_audit_attempts: true
  implement_mlx_http_adapter: false
  implement_ui: false

skill_routing:
  - skill: contract-impact-analysis
    use_when: "adding prompt version/hash, audit attempt schema use, or terminal statuses"
    timing: "before implementation"
    reason: "prompt/audit/config/status contracts"
  - skill: backend-quality-gates
    use_when: "running pipeline, repair and route tests"
    timing: "during verification"
    reason: "backend test/lint/type gates"
  - skill: production-risk-review
    use_when: "before final report for LLM trust boundary and audit-data review"
    timing: "before ship"
    reason: "LLM/user-content trust boundary"
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
  - "backtest-ai-configurator-v1"
  - "<TRUSTED_SYSTEM_POLICY>"
  - "<TRUSTED_ALLOWED_CATALOG>"
  - "<UNTRUSTED_USER_REQUEST>"
  - "<UNTRUSTED_CURRENT_CONFIG>"
  - "<OUTPUT_JSON_SCHEMA>"
  - "repair_attempts: 1"

non_goals:
  - "Do not connect to mlx_lm.server yet."
  - "Do not add launchd or Monit."
  - "Do not enable browser AI panel."
  - "Do not use remote OpenAI or non-MLX fallback."

final_report_format:
  language: ru
  sections:
    - "Что реализовано"
    - "Prompt/repair contract"
    - "Безопасность и audit"
    - "Проверки"
    - "Доставка и Mac Studio"
    - "Следующая итерация"

quality_gates:
  - cmd: "uv run pytest -q tests/unit/contexts/backtest/application/ai_configurator tests/unit/apps/api/test_backtest_ai_config_routes.py"
    expect: "passes"
  - cmd: "uv run ruff check apps/api src/trading/contexts/backtest tests/unit/apps/api tests/unit/contexts/backtest"
    expect: "passes"
  - cmd: "uv run pyright"
    expect: "passes"
  - cmd: "git diff --check"
    expect: "passes"

expected_primary_touches:
  - "src/trading/contexts/backtest/application/ai_configurator/ports/llm_gateway.py"
  - "src/trading/contexts/backtest/application/ai_configurator/services/prompt_profiles.py"
  - "src/trading/contexts/backtest/application/ai_configurator/services/pipeline.py"
  - "src/trading/contexts/backtest/application/ai_configurator/services/repair.py"
  - "tests/unit/contexts/backtest/application/ai_configurator/"

possible_secondary_touches:
  - "src/trading/contexts/backtest/adapters/outbound/llm/"
  - "src/trading/contexts/backtest/adapters/outbound/persistence/postgres/"
  - "apps/api/wiring/modules/backtest_ai_config.py"

safety_notes:
  - "Prompt text is not an enforcement layer; deterministic gates from Iteration 03 remain authoritative."
  - "Repair prompt receives untrusted draft/errors only, never secrets/logs/private topology."
  - "Store prompt version/hash and attempts, but do not expose attempts publicly."
---

# Task

Implement Iteration 04 of the `/backtests` AI Configurator: versioned prompt profiles, structured prompt envelope, LLM gateway port, deterministic test adapter, strict JSON parse, one-attempt repair loop, and LLM attempt audit. Do not connect to real MLX yet.

Done means:

- prompt profiles exist for generate, repair and explain/suggest modes;
- prompt assembly uses the structured trusted/untrusted envelope from the architecture doc;
- prompt version and hash are persisted with attempts/jobs;
- `BacktestConfigLLMGateway` port exists and is used by pipeline;
- deterministic test adapter can return valid JSON, invalid JSON, schema-invalid JSON and unsupported config drafts;
- repair loop runs at most once and uses validation errors plus untrusted draft;
- all attempts are audited with latency/token estimates when available;
- tests prove parse failure repair, schema failure repair, business validation failure repair, unrepaired failure, and no public raw attempts.

## Context / Current State

Context ledger:

- completed:
  - Iteration 03 should provide catalog resolver, validators and security gates.
- open_items:
  - real MLX runtime adapter is intentionally not connected yet;
  - UI remains disabled.
- contract_changes:
  - new prompt version/hash and attempt audit behavior;
  - no public route changes except status details already owned by AI routes.
- risks:
  - leaking secrets/private topology into prompts or training rows;
  - treating repair as a second autonomous model instead of orchestrated retry;
  - exposing raw LLM drafts to UI.
- next_focus:
  - model-independent pipeline ready for MLX adapter.

## Requirements (Must)

- Verify Iteration 03 validators/gates exist; if not, stop and report blocker.
- Implement prompt profiles with stable `system_prompt_version=backtest-ai-configurator-v1` and hash.
- Build structured envelope with trusted policy/catalog/schema and untrusted user/current_config blocks.
- Explicitly exclude secrets, env vars, DSNs, model server URL, Tailscale details, raw logs and other users' data from prompts.
- Implement `BacktestConfigLLMGateway` port with generate/repair methods.
- Implement deterministic adapter/fake for tests; do not connect real MLX.
- Parse one strict JSON object; Markdown-wrapped or multi-object responses must fail parse.
- Run output gate, schema validation and business validation before ready state.
- Run repair at most once.
- Persist each generate/repair attempt with profile, prompt version/hash, raw output, parsed draft if available, validation errors, latency/token estimates where available, and success/failure reason.
- Never expose raw draft/attempt publicly.

## Requirements (Should)

- Keep prompt profile text close to code/config where versioning and tests can hash it deterministically.
- Keep token estimates approximate if no tokenizer is available.
- Make repair profile small and path-specific, not a broad "try again" prompt.

## Requirements (Nice-to-have)

- Add golden prompt-envelope tests that assert trusted/untrusted block order.
- Add example safe prompt fixtures for future eval harness.

# Context acquisition protocol

Read only in this order and stop once sufficient:

1. `.codex/AGENTS.md`
2. `.codex/agents/.context/promt_manager_state.yaml` or latest state snapshot, if available and relevant
3. latest executor final report, if available
4. task entrypoints
5. only conditional bundles required by prompt/audit location
6. consult-if-needed references only for blockers

Do not eagerly preload all listed sources.

Pre-implementation reading target:

- `<= 8 files`
- `<= ~35k-50k tokens`

Stop reading once prompt profile location, gateway port, pipeline and audit persistence boundaries are clear.

# Reading manifest

Use the front-matter `context_sources` as the canonical reading map.

Read with this intent:

- `always_read`: repo rules and prompt/repair contract.
- `task_entrypoints`: existing AI pipeline and tests.
- `conditional_bundles`: prompt examples and audit adapter only if needed.
- `consult_if_needed`: architecture doc for exact envelope.

Do not convert this manifest into a broad mandatory reading list.

# Work plan (agent should follow)

Skill routing for this task:

- `contract-impact-analysis`: use before implementation; owns prompt/audit/status contract.
- `backend-quality-gates`: use during verification; owns tests/lint/type checks.
- `production-risk-review`: use before final report; owns LLM trust boundary review.

1. Verify prior iteration validators and pipeline.
2. Add prompt profile objects/config and deterministic hashing.
3. Add LLM gateway port and deterministic test adapter.
4. Integrate generate/parse/output-gate/schema/business validation into pipeline.
5. Add one-attempt repair loop and attempt audit writes.
6. Add tests for generate/repair success and failure modes.
7. Run quality gates and report trust-boundary residual risks.

# Acceptance criteria (Definition of Done)

- Valid deterministic model output reaches `ready` only after all gates.
- Invalid JSON is repaired once or terminates without load button.
- Schema/business validation errors are passed to repair once.
- Security output failures cannot silently become ready.
- Raw attempts are persisted but not exposed through read route.
- Prompt envelope tests prove trusted/untrusted separation.

- `publish-ci-deploy` terminal state is `deployed`, or `green-pr`/`blocked` is reported with exact blocker evidence.

# Implementation constraints

## Determinism & ordering

- Prompt hashes must be stable across process runs.
- Repair attempt count must be deterministic and bounded by config/default.

## API / contracts

- Do not add new public endpoints unless strictly required by tests.
- Existing AI API read shape may include public warnings/status, not raw prompt/draft.

## Security

- Do not include secrets/private paths/private topology in prompt tests.
- Treat all LLM output as untrusted until deterministic gates pass.

# Files to indicate (expected touched areas)

Expected primary touches:

- `src/trading/contexts/backtest/application/ai_configurator/ports/llm_gateway.py`
- `src/trading/contexts/backtest/application/ai_configurator/services/prompt_profiles.py`
- `src/trading/contexts/backtest/application/ai_configurator/services/pipeline.py`
- `src/trading/contexts/backtest/application/ai_configurator/services/repair.py`
- `tests/unit/contexts/backtest/application/ai_configurator/`

Possible secondary touches:

- `src/trading/contexts/backtest/adapters/outbound/llm/`
- `src/trading/contexts/backtest/adapters/outbound/persistence/postgres/`
- `apps/api/wiring/modules/backtest_ai_config.py`

# Non-goals

- No `mlx_lm.server` HTTP calls.
- No standalone MLX/Mac Studio model smoke before delivery; `publish-ci-deploy` production smoke remains required as the final step.
- No browser UI.
- No launchd/Monit/Prometheus.
- No load benchmarking.

# Quality gates (must run and pass)

- `uv run pytest -q tests/unit/contexts/backtest/application/ai_configurator tests/unit/apps/api/test_backtest_ai_config_routes.py`
- `uv run ruff check apps/api src/trading/contexts/backtest tests/unit/apps/api tests/unit/contexts/backtest`
- `uv run pyright`
- `git diff --check`

If a gate cannot run, classify it as introduced, required-path pre-existing, unrelated pre-existing, environmental, or flaky.

Required delivery step: after the quality gates above pass, invoke `publish-ci-deploy` as the final step. The expected terminal state for this prompt is `deployed`: intended files committed and pushed, GitHub Actions green, revision shipped to `main`, `/opt/roehub/app` on `macstudio` pulled to that revision, the relevant production services reloaded through the repository runbook, and `bash scripts/macos/smoke_prod.sh` passed. If the skill reaches `green-pr` because a human merge/approval is required, or `blocked` because of missing auth, unrelated dirty scope, external CI, Mac Studio access, or production verification failure, report that exact state and do not claim deployment.

# Final output: report format (strict)

Report in Russian with:

- `Что реализовано`: prompt profiles, gateway, repair, audit.
- `Prompt/repair contract`: version/hash/envelope/attempt behavior.
- `Безопасность и audit`: what is excluded, what is persisted, what is public.
- `Проверки`: exact commands and results.
- `Доставка и Mac Studio`: publish-ci-deploy terminal state, main/PR SHA, CI result, Mac Studio pull/reload/smoke evidence, or exact blocker.
- `Следующая итерация`: MLX adapter and worker runtime.
