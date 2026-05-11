---
prompt_name: backtest_ai_configurator_mlx_v1_05_mlx_adapter_worker_runtime
repo: roehub.com
branch: main
scope: "Iteration 05: implement MLX/OpenAI-compatible adapter, model registry config, and backtest-ai-configurator worker runtime without enabling production launchd or browser UI."

language:
  implementation: python_fastapi_worker
  agent_report: ru

context_sources:
  always_read:
    - path: .codex/AGENTS.md
      why: "repo contract and skill routing"
    - path: docs/architecture/backtest/backtest-ai-configurator-mlx-v1.md
      why: "MLX runtime and worker contract"
  task_entrypoints:
    - path: src/trading/contexts/backtest/application/ai_configurator
      why: "pipeline, prompt and gateway port"
      inspect_symbols:
        - "*"
    - path: apps/worker
      why: "worker process layout patterns"
      inspect_symbols:
        - "*"
    - path: apps/api/wiring/modules/backtest_ai_config.py
      why: "AI configurator composition root"
      inspect_symbols:
        - "*"
    - path: configs/prod
      why: "production config style"
      inspect_symbols:
        - "*.yaml"
  conditional_bundles:
    market_worker_style:
      read_when: "when implementing worker CLI/main loop or metrics server"
      paths:
        - apps/worker/market_data_ws
        - apps/scheduler
    http_client_style:
      read_when: "when choosing HTTP client/session patterns"
      paths:
        - apps/api
        - src
    local_mlx_docs:
      read_when: "if mlx_lm.server OpenAI-compatible contract is ambiguous"
      paths:
        - docs/architecture/backtest/backtest-ai-configurator-mlx-v1.md
  consult_if_needed:
    - path: docs/runbooks/mac-studio-native-backend-operations.md
      read_when: "only for worker runtime path conventions"
    - path: .codex/agents/.context/promt_manager_state.yaml
      read_when: "only to check for a newer executor handoff; ignore if stale/unrelated"

style_references:
  - path: apps/worker/market_data_ws
    purpose: "worker app structure if present"
  - path: configs/prod
    purpose: "production yaml config conventions"

hard_requirements:
  depends_on_iteration_04: true
  mlx_only: true
  no_remote_openai: true
  loopback_runtime_only: true
  configurable_model_path: true
  active_generations_default_one: true
  no_browser_ui_enablement: true

task_toggles:
  implement_mlx_http_adapter: true
  implement_worker_process: true
  implement_model_registry_config: true
  implement_launchd: false
  implement_monit: false
  implement_ui: false
  run_macstudio_smoke_if_available: true

skill_routing:
  - skill: contract-impact-analysis
    use_when: "adding model config schema, worker runtime config or adapter boundary"
    timing: "before implementation"
    reason: "config/runtime compatibility"
  - skill: backend-quality-gates
    use_when: "running adapter, worker and pipeline tests"
    timing: "during verification"
    reason: "backend worker quality gates"
  - skill: root-cause-debugging
    use_when: "MLX HTTP, worker loop or config smoke fails"
    timing: "if blocker"
    reason: "localize runtime integration failures"

target_envs:
  - local-dev
  - unit-tests
  - mac-studio-optional-smoke

required_literals:
  - "MLXOpenAICompatibleAdapter"
  - "apps.worker.backtest_ai_configurator"
  - "configs/prod/backtest_ai_configurator.yaml"
  - "127.0.0.1"
  - "active_generations: 1"
  - "gemma-4-e2b-it-4bit"

non_goals:
  - "Do not add launchd plist or Monit snippet in this iteration."
  - "Do not expose mlx_lm.server publicly."
  - "Do not use remote OpenAI, llama.cpp, GGUF or non-MLX fallback."
  - "Do not enable the browser AI panel."

final_report_format:
  language: ru
  sections:
    - "Что реализовано"
    - "Runtime/config contract"
    - "Проверки"
    - "Mac Studio smoke"
    - "Следующая итерация"

quality_gates:
  - cmd: "uv run pytest -q tests/unit/contexts/backtest/application/ai_configurator tests/unit/apps/api/test_backtest_ai_config_routes.py tests/unit/apps/worker/test_backtest_ai_configurator_worker.py"
    expect: "passes"
  - cmd: "uv run ruff check apps/worker apps/api src/trading/contexts/backtest tests/unit"
    expect: "passes"
  - cmd: "uv run pyright"
    expect: "passes"
  - cmd: "git diff --check"
    expect: "passes"

expected_primary_touches:
  - "src/trading/contexts/backtest/adapters/outbound/llm/mlx_openai_compatible.py"
  - "apps/worker/backtest_ai_configurator/"
  - "configs/prod/backtest_ai_configurator.yaml"
  - "tests/unit/apps/worker/test_backtest_ai_configurator_worker.py"
  - "tests/unit/contexts/backtest/application/ai_configurator/"

possible_secondary_touches:
  - "apps/api/wiring/modules/backtest_ai_config.py"
  - "pyproject.toml"
  - "docs/architecture/backtest/backtest-ai-configurator-mlx-v1.md"

safety_notes:
  - "MLX runtime base_url must be loopback-only by config."
  - "If no local MLX model exists, unit tests must use a fake HTTP server and Mac Studio smoke is reported as blocked."
  - "Do not log full prompts or raw model output outside restricted AI audit tables."
---

# Task

Implement Iteration 05 of the `/backtests` AI Configurator: MLX OpenAI-compatible adapter, model registry/config, and a standalone `backtest-ai-configurator-worker` process that claims queued jobs and executes the pipeline. Keep production launchd/Monit and browser UI for later iterations.

Done means:

- model registry config supports folder path model selection, context window, max input/output tokens, temperature/top_p, base_url and `active_generations`;
- `MLXOpenAICompatibleAdapter` calls an OpenAI-like `/v1/chat/completions` endpoint on loopback only;
- adapter supports generate and repair gateway operations;
- worker process claims jobs with lease, heartbeats, runs pipeline and marks terminal states;
- concurrency limiter starts with `active_generations=1`;
- unit tests use deterministic fake adapter/server and do not require real MLX;
- optional Mac Studio smoke is documented if a configured MLX model/runtime is available.

## Context / Current State

Context ledger:

- completed:
  - Iteration 04 should provide prompt profiles, repair loop and LLM gateway port.
- open_items:
  - production launchd/Monit/Prometheus not implemented yet;
  - UI remains disabled.
- contract_changes:
  - new config section `backtest_ai_configurator`;
  - new worker process, but no browser-visible behavior change yet.
- risks:
  - accidentally opening model server beyond loopback;
  - worker duplicate processing if lease/concurrency handling is weak;
  - making tests depend on a local model path.
- next_focus:
  - real runtime integration behind existing model-independent pipeline.

## Requirements (Must)

- Verify Iteration 04 gateway/pipeline exists; if not, stop and report blocker.
- Implement only MLX-compatible runtime path; do not add remote/non-MLX fallback.
- Model path must be configurable as a folder path, for example `/Users/daniildegtyarev/.lmstudio/models/mlx-community/gemma-4-e2b-it-4bit`.
- Runtime `base_url` must default to loopback and must not be exposed publicly.
- `active_generations` default is `1`.
- Worker must use repository lease/heartbeat semantics from Iteration 01.
- Worker must not process jobs for other pages; `source_page` remains `backtests`.
- Worker must not expose prompts/model output outside audit tables/log-safe structured summaries.
- Unit tests must not require a real MLX model.

## Requirements (Should)

- Keep worker lifecycle boring: startup config load, readiness checks where applicable, claim loop, graceful shutdown.
- Make HTTP timeouts explicit and bounded.
- Keep model reload as restart/config reload, not hot-swapping inside a request.
- Record model id/path hash on attempts/jobs.

## Requirements (Nice-to-have)

- Add a small local smoke command that can be run on Mac Studio when model/runtime are available.
- Add structured logs for job claimed/completed/failed.

# Context acquisition protocol

Read only in this order and stop once sufficient:

1. `.codex/AGENTS.md`
2. `.codex/agents/.context/promt_manager_state.yaml` or latest state snapshot, if available and relevant
3. latest executor final report, if available
4. task entrypoints
5. only conditional bundles required by worker/config patterns
6. consult-if-needed references only for blockers

Do not eagerly preload all listed sources.

Pre-implementation reading target:

- `<= 8 files`
- `<= ~35k-50k tokens`

Stop reading once adapter port, worker loop and config schema are clear.

# Reading manifest

Use the front-matter `context_sources` as the canonical reading map.

Read with this intent:

- `always_read`: repo rules and runtime contract.
- `task_entrypoints`: AI pipeline, worker patterns and config conventions.
- `conditional_bundles`: worker style or HTTP style only when needed.
- `consult_if_needed`: ops docs only for path conventions.

Do not convert this manifest into a broad mandatory reading list.

# Work plan (agent should follow)

Skill routing for this task:

- `contract-impact-analysis`: use before implementation; owns config and runtime process compatibility.
- `backend-quality-gates`: use during verification; owns tests/lint/type checks.
- `root-cause-debugging`: use only if worker/config/HTTP integration fails.

1. Verify prompt/gateway/pipeline contracts from prior iteration.
2. Add config model/loader for `backtest_ai_configurator`.
3. Implement `MLXOpenAICompatibleAdapter` with loopback-only guard and timeouts.
4. Implement worker main/loop/concurrency limiter/claim-heartbeat-terminal handling.
5. Wire deterministic fake adapter for tests and optional real adapter by config.
6. Add unit tests for adapter request shape, parse, timeouts, worker claim lifecycle and no duplicate active generation.
7. Run gates and optional Mac Studio smoke if available.

# Acceptance criteria (Definition of Done)

- Worker can process a queued job to terminal state using fake adapter in tests.
- Adapter constructs OpenAI-compatible chat completions request and handles response/error/timeout deterministically.
- Config validates model path/base_url/context/max tokens/active generations.
- Non-loopback runtime URL is rejected or requires explicit safe override with documented blocker.
- No tests require real MLX.
- Optional Mac Studio smoke result is reported separately from unit acceptance.

# Implementation constraints

## Determinism & ordering

- Claim loop ordering follows repository lease ordering.
- Concurrency limiter must make max active generations testable.

## API / contracts

- Do not alter AI route shape except wiring worker-backed states if needed.
- Do not change existing backtest job APIs.

## Runtime safety

- Set bounded request timeouts.
- Log model id, not full model path if path reveals private local topology.
- No public network bind for model runtime.

# Files to indicate (expected touched areas)

Expected primary touches:

- `src/trading/contexts/backtest/adapters/outbound/llm/mlx_openai_compatible.py`
- `apps/worker/backtest_ai_configurator/`
- `configs/prod/backtest_ai_configurator.yaml`
- `tests/unit/apps/worker/test_backtest_ai_configurator_worker.py`
- `tests/unit/contexts/backtest/application/ai_configurator/`

Possible secondary touches:

- `apps/api/wiring/modules/backtest_ai_config.py`
- `pyproject.toml`
- `docs/architecture/backtest/backtest-ai-configurator-mlx-v1.md`

# Non-goals

- No launchd plist.
- No Monit snippet.
- No Prometheus target.
- No browser UI enablement.
- No S1/S5/S10/S50/S100 benchmark.

# Quality gates (must run and pass)

- `uv run pytest -q tests/unit/contexts/backtest/application/ai_configurator tests/unit/apps/api/test_backtest_ai_config_routes.py tests/unit/apps/worker/test_backtest_ai_configurator_worker.py`
- `uv run ruff check apps/worker apps/api src/trading/contexts/backtest tests/unit`
- `uv run pyright`
- `git diff --check`

If Mac Studio smoke is attempted, report exact command, model id/path hash, result and whether it is acceptance or optional evidence.

# Final output: report format (strict)

Report in Russian with:

- `Что реализовано`: adapter, config, worker.
- `Runtime/config contract`: model path, loopback, concurrency, timeouts.
- `Проверки`: exact commands and results.
- `Mac Studio smoke`: run/skipped/blocker with reason.
- `Следующая итерация`: Web UI integration.
