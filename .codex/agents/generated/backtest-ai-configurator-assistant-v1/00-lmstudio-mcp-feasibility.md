---
prompt_name: backtest_ai_configurator_assistant_v1_00_lmstudio_mcp_feasibility
repo: /Users/daniildegtyarev/Projects/roehub.com
branch: main
scope: "Stage 00 feasibility proof for LM Studio /api/v1/chat + read-only MCP context lookup before production AI configurator implementation."

language:
  implementation: python
  agent_report: ru

context_sources:
  always_read:
    - path: .codex/AGENTS.md
      why: "repo contract, skill routing, safety rules"
    - path: docs/architecture/backtest/backtest-ai-configurator-assistant-v1.md
      why: "current target plan and Stage 00 acceptance"
  task_entrypoints:
    - path: docs/architecture/backtest/backtest-ai-configurator-assistant-v1.md
      why: "Stage 00 scope, probes, and evidence contract"
      inspect_symbols:
        - "Итерация 00"
        - "Политика доставки"
        - "Канонический system prompt v1"
        - "Критерии приемки"
    - path: configs/prod/backtest_ai_configurator.yaml
      why: "target model/runtime config candidate"
      inspect_symbols:
        - "runtime"
        - "model"
        - "lm_studio"
    - path: apps/web/dist/js/core/sse.js
      why: "existing EventSource helper, only to confirm no UI work is required"
      inspect_symbols:
        - "createSseClient"
  conditional_bundles:
    lm_studio_api_docs:
      read_when: "LM Studio request payload, MCP integration shape, or version behavior is unclear"
      paths:
        - https://lmstudio.ai/docs/developer/rest/chat
        - https://lmstudio.ai/docs/developer/core/mcp
        - https://lmstudio.ai/docs/developer/openai-compat/structured-output
    macstudio_runtime:
      read_when: "Mac Studio host, ports, LM Studio CLI, or model location must be verified"
      paths:
        - docs/architecture/backtest/backtest-ai-configurator-assistant-v1.md
        - configs/prod/backtest_ai_configurator.yaml
    docs_index:
      read_when: "Markdown docs or evidence files are created or updated"
      paths:
        - docs/architecture/README.md
        - tools/docs/generate_docs_index.py
  consult_if_needed:
    - path: .codex/PLANS.md
      read_when: "plan history conflicts with current architecture doc"
    - path: docs/architecture/backtest/
      read_when: "evidence folder naming or benchmark artifact shape is ambiguous"

style_references:
  - .codex/promt_template.md
  - docs/architecture/backtest/backtest-ai-configurator-assistant-v1.md

hard_requirements:
  no_production_code_changes: true
  macstudio_evidence_required: true
  one_executor_prompt_for_stage_00: true
  lm_studio_mcp_api_must_be_proven_not_assumed: true
  write_machine_readable_evidence: true
  direct_main_publish_only_if_accepted: true

task_toggles:
  create_experimental_scripts: true
  create_sample_context_fixture: true
  create_stage00_evidence: true
  run_real_lmstudio_probes_on_macstudio: true
  update_architecture_doc_only_if_api_facts_differ: true
  implement_production_service: false
  implement_ui: false

skill_routing:
  - skill: backend-quality-gates
    use_when: "creating or running Python probe scripts/tests"
    timing: during verification
    reason: "owns ruff/pytest/pyright-style local gates for Python artifacts"
  - skill: backend-performance-evidence
    use_when: "reporting latency or runtime capacity numbers from probes"
    timing: during verification
    reason: "owns measurement method and evidence discipline"
  - skill: publish-ci-deploy
    use_when: "and only when Stage 00 accepted=true after Mac Studio evidence"
    timing: before ship
    reason: "owns direct main publish, CI/deploy watch, and Mac Studio sync verification"

target_envs:
  - local
  - macstudio

required_literals:
  - "/api/v1/chat"
  - "/v1/chat/completions"
  - "response_format"
  - "json_schema"
  - "allowed_tools"
  - "search_backtest_context"
  - "get_backtest_context_item"
  - "structure.percent_rank"
  - "momentum.rsi"
  - "ma.ema"
  - "accepted"
  - "blocking_reason"
  - "next_iteration_allowed"
  - "backtest_ai_configurator_agent_mcp_v1"

non_goals:
  - "Do not implement production AI configurator API, storage, UI, Monit, Prometheus, or migrations."
  - "Do not modify apps/, src/, configs/prod/, infra/, or deployment assets for production behavior."
  - "Do not reintroduce old AI one-shot job endpoints or AI mode buttons."
  - "Do not pass the full context file in the model prompt."
  - "Do not expose shell, arbitrary file, network, write, DB mutation, or run-backtest MCP tools."

final_report_format:
  language: ru
  sections:
    - "Итог"
    - "Evidence"
    - "Проверки"
    - "Измененные файлы"
    - "Следующий шаг"

quality_gates:
  - cmd: "uv run ruff check <created_probe_or_test_paths>"
    expect: "passes for any created Python scripts/tests"
  - cmd: "uv run pytest -q <created_test_paths>"
    expect: "passes if tests are created"
  - cmd: "uv run python -m tools.docs.generate_docs_index --check"
    expect: "passes when Markdown docs/evidence are changed"
  - cmd: "ssh macstudio '<stage00 probe commands>'"
    expect: "real Mac Studio LM Studio + MCP probe evidence is recorded"
  - cmd: "git diff --name-only"
    expect: "no production code/config/infra changes outside allowed Stage 00 artifacts"

expected_primary_touches:
  - "docs/architecture/backtest/benchmark_iterations/<date>_ai_configurator_assistant_v1/iteration_00_lmstudio_mcp_feasibility.md"
  - "docs/architecture/backtest/benchmark_iterations/<date>_ai_configurator_assistant_v1/iteration_00_lmstudio_mcp_feasibility.json"
  - "docs/architecture/backtest/benchmark_iterations/<date>_ai_configurator_assistant_v1/implementation_progress.md"
  - "docs/architecture/backtest/benchmark_iterations/<date>_ai_configurator_assistant_v1/implementation_progress.json"
  - "tools/backtest_ai_configurator/stage00_lmstudio_mcp_probe.py"
  - "tools/backtest_ai_configurator/stage00_backtest_context_mcp.py"
  - "tools/backtest_ai_configurator/stage00_backtest_ai_context_mvp.json"

possible_secondary_touches:
  - "tests/unit/tools/test_stage00_backtest_ai_context_mcp.py"
  - "docs/architecture/backtest/backtest-ai-configurator-assistant-v1.md"
  - "docs/architecture/README.md"

safety_notes:
  - "Stage 00 is a feasibility experiment, not a production implementation."
  - "Mac Studio proof is mandatory; local-only evidence cannot set accepted=true."
  - "Do not print secrets, exchange keys, raw prompt archives, full context dumps, or raw model logs."
  - "If LM Studio /api/v1/chat + MCP cannot be proven, write accepted=false and stop."
---

# Task

Execute Stage 00 for `AI-помощник конфигуратора /backtests v1`: prove or block the target runtime idea before any production implementation.

You must verify on the real Mac Studio host whether LM Studio `/api/v1/chat` can run the selected local MLX model with a read-only MCP context tool and produce either:

- direct parseable Roehub JSON envelopes, or
- a documented and proven fallback where `/api/v1/chat + MCP` performs context lookup and `/v1/chat/completions` with `response_format=json_schema` performs formatting-only JSON normalization.

Done means:

- Stage 00 evidence `.md` and `.json` exist with `accepted`, `blocking_reason`, and `next_iteration_allowed`.
- The evidence was produced from real Mac Studio probes, not only local code inference.
- The model actually called only allowed read-only MCP tools for context lookup.
- Supported prompts return parseable JSON config directly or through the proven formatting-only fallback.
- Unsupported/offtopic/security prompts do not produce config-ready output.
- No production AI configurator service, API, UI, storage, Monit, Prometheus, or deployment code was implemented.

## Context / Current State

Context ledger:

- completed:
  - The current target architecture is documented in `docs/architecture/backtest/backtest-ai-configurator-assistant-v1.md`.
  - The old full-context/backend-selector design is rejected.
  - Stage 00 is explicitly required before production code.
- open_items:
  - Prove LM Studio `/api/v1/chat + MCP` on Mac Studio.
  - Prove model calls only `search_backtest_context` and `get_backtest_context_item`.
  - Prove JSON output behavior and whether formatting-only fallback is required.
- contract_changes:
  - None should be made in production during this stage.
  - Evidence artifacts are allowed.
  - Experimental probe scripts/fixtures are allowed if they are clearly non-production.
- risks:
  - LM Studio API shape may differ from the plan.
  - `/api/v1/chat + MCP` may call tools but may not produce strict JSON reliably.
  - Model may ignore MCP tools and hallucinate context.
  - Mac Studio host or LM Studio local server may not be ready.
- next_focus:
  - Build minimal read-only MCP context fixture.
  - Run staged probes against real LM Studio on Mac Studio.
  - Record accepted/blocking evidence.

Additional context:

- The selected model candidate from the plan is `gemma-4-e2b-it-4bit`.
- The model path candidate is `/Users/daniildegtyarev/.lmstudio/models/mlx-community/gemma-4-e2b-it-4bit`.
- The sample context must include `BTCUSDT`, `momentum.rsi`, `ma.ema`, `structure.percent_rank`, at least one no-window indicator, and `1h`.

## Requirements (Must)

- Read context using the protocol below and stop early once sufficient.
- Implement only Stage 00 experimental artifacts and evidence.
- Do not modify production behavior under `apps/`, `src/`, `configs/prod/`, `infra/`, or deployment service definitions.
- Use real Mac Studio LM Studio runtime for acceptance.
- Verify LM Studio version is `0.4.0+` before acceptance.
- Verify model loaded/readiness using a lightweight `/api/v1/chat` generation, not `/v1/models` alone.
- Create a minimal read-only MCP server with only these tools:
  - `search_backtest_context(query, limit)`
  - `get_backtest_context_item(kind, id)`
- Enforce MCP tool result limits and no arbitrary path/file/network/shell/write access.
- Create a sample context fixture with:
  - `BTCUSDT` availability for `binance/spot`,
  - timeframe `1h`,
  - period bounds,
  - indicators `momentum.rsi`, `ma.ema`, `structure.percent_rank`,
  - one no-window indicator,
  - explicit values for `structure.percent_rank` such as `[10, 14, 20, 28, 42, 56, 84, 126]`,
  - basic risk/execution/ranking limits needed by the prompt.
- Build a probe script that:
  - starts or connects to the read-only MCP server,
  - calls LM Studio `/api/v1/chat` with MCP integration and `allowed_tools`,
  - records tool calls at a high level without dumping full context,
  - parses final JSON,
  - optionally runs a formatting-only fallback through `/v1/chat/completions` with `response_format=json_schema`,
  - writes machine-readable summary evidence.
- Run at least 10-15 fixed prompts covering RU, EN, supported, unsupported, security, off-topic, and auto-run-backtest attempts.
- Never pass the full context file in `system_prompt`, `user` message, or request body except as MCP tool output.
- Reject or mark blocked if the model calls any tool outside `allowed_tools`.
- Write evidence JSON with at least:
  - `schema_version`,
  - `iteration`,
  - `accepted`,
  - `blocking_reason`,
  - `next_iteration_allowed`,
  - `host`,
  - `model_id`,
  - `lm_studio_version`,
  - `direct_json_success_rate`,
  - `fallback_required`,
  - `fallback_success_rate`,
  - `mcp_tool_calls_total`,
  - `invalid_tool_calls_allowed`,
  - `security_leakage`,
  - `unsupported_ready_configs`,
  - `auto_run_ready_configs`,
  - `safe_prompts_blocked`,
  - `context_hash`,
  - `git_commit`.
- Set `accepted=true` only when Mac Studio evidence satisfies the acceptance criteria below.
- If blocked, set `accepted=false`, `next_iteration_allowed=false`, and a concrete `blocking_reason`.

## Requirements (Should)

- Keep scripts deterministic and easy to delete after Stage 00.
- Prefer no new third-party dependencies. If an MCP helper package is already available in the environment, you may use it; otherwise implement the minimal protocol needed for the probe.
- Capture exact LM Studio HTTP error bodies for any `400`/`500` responses, redacted for secrets.
- Keep evidence compact: store summaries, hashes, counts, and short excerpts only.
- Use official LM Studio docs only when API details are unclear.

## Requirements (Nice-to-have)

- Include a short probe timing summary: p50/p95 latency for the fixed prompt set.
- Include a small transcript table in Markdown with prompt id, language, expected outcome, observed outcome, tool calls, parse result, fallback result.

# Context acquisition protocol

Read only in this order and stop once sufficient:

1. `.codex/AGENTS.md`
2. `docs/architecture/backtest/backtest-ai-configurator-assistant-v1.md`
3. `configs/prod/backtest_ai_configurator.yaml`
4. task entrypoints and only the conditional bundle needed for a blocker
5. official LM Studio docs only when request payload or MCP integration details are unclear
6. consult-if-needed references only for blockers, ambiguity, or conflicts

Do not eagerly preload all listed sources.

Pre-implementation reading target:

- `<= 8 files`
- `<= ~35k-50k tokens`

Stop reading once all of the following are true:

- Stage 00 acceptance is clear,
- the probe scripts/fixtures/evidence paths are bounded,
- LM Studio API request shape is known or a docs lookup is planned,
- no unresolved production contract ambiguity remains.

Expand context only for:

- LM Studio API/MCP ambiguity,
- Mac Studio connectivity/runtime blockers,
- failing quality gates,
- evidence schema ambiguity,
- architecture conflicts that affect Stage 00 acceptance.

# Reading manifest

Use the front-matter `context_sources` as the canonical reading map.

Read with this intent:

- `always_read`: repository rules and current target plan.
- `task_entrypoints`: Stage 00 acceptance, model config candidate, and existing SSE helper only to confirm no UI implementation is required.
- `conditional_bundles`: read only when the stated condition applies.
- `consult_if_needed`: read only for blockers, ambiguity, or conflict resolution.

Do not convert this manifest into a broad mandatory reading list.

# Work plan (agent should follow)

Skill routing for this task:

- `backend-quality-gates`: use during verification when creating/running Python probe scripts or tests; owns lint/test gates.
- `backend-performance-evidence`: use during verification only if reporting latency/capacity numbers; owns measurement discipline.
- `publish-ci-deploy`: use before ship only if Stage 00 is accepted on Mac Studio; owns direct main publish, CI/deploy watch, and Mac Studio sync.

1. Read the repository contract and Stage 00 section of the architecture plan.
2. Confirm current git state and avoid mixing unrelated changes.
3. Verify Mac Studio access and LM Studio availability/version/model state.
4. Create the minimal Stage 00 sample context fixture and read-only MCP probe server.
5. Create the LM Studio probe runner with fixed prompt matrix and evidence writer.
6. Run local syntax/unit checks for the probe code.
7. Run the probe on Mac Studio against the real LM Studio model.
8. If `/api/v1/chat + MCP` produces reliable parseable JSON, record direct success.
9. If direct JSON is unreliable but MCP lookup works, test formatting-only fallback via `/v1/chat/completions` with `response_format=json_schema`.
10. Write Markdown and JSON evidence plus implementation progress files.
11. Run docs index check if Markdown evidence/docs changed.
12. Self-check that production code/config/infra was not changed.
13. If and only if `accepted=true`, use `publish-ci-deploy` for direct main publish/sync per repo policy. If blocked, do not publish as accepted.

# Acceptance criteria (Definition of Done)

- Evidence was produced on Mac Studio.
- LM Studio version is verified as `0.4.0+`.
- Target model readiness is proven by lightweight `/api/v1/chat` generation.
- `/api/v1/chat` with MCP integration causes real read-only MCP tool calls.
- Tool calls are limited to `search_backtest_context` and `get_backtest_context_item`.
- The prompt/request does not include the full context file.
- Supported prompts produce parseable config JSON either directly or through proven formatting-only fallback.
- `structure.percent_rank` uses only explicit allowed values.
- The no-window indicator prompt does not invent a `window`.
- Unsupported symbol/indicator prompts do not produce `config_ready`.
- Off-topic and prompt-injection prompts do not produce `config_ready`.
- Auto-run-backtest attempts do not produce `config_ready`.
- Evidence records whether fallback is required.
- `iteration_00_lmstudio_mcp_feasibility.json` contains `accepted`, `blocking_reason`, and `next_iteration_allowed`.
- `implementation_progress.md/json` are created or updated.
- No production service/API/UI/storage/Monit/Prometheus/deployment code was implemented.

# Implementation constraints

## Determinism & ordering

- Keep the fixed prompt matrix deterministic and ordered.
- Keep probe outputs stable enough for review.
- Do not rely on wall-clock ordering for evidence identity; include explicit prompt ids.

## API / contracts

- Do not change public or persisted production contracts.
- Do not add old `/backtests/ai-config/jobs*` endpoints.
- Do not add `mode` to browser-visible request contracts.
- Do not implement conversation storage, UI, or production run/event routes in Stage 00.

## Documentation

- Required new evidence shape:
  - `docs/architecture/backtest/benchmark_iterations/<date>_ai_configurator_assistant_v1/iteration_00_lmstudio_mcp_feasibility.md`
  - `docs/architecture/backtest/benchmark_iterations/<date>_ai_configurator_assistant_v1/iteration_00_lmstudio_mcp_feasibility.json`
  - `docs/architecture/backtest/benchmark_iterations/<date>_ai_configurator_assistant_v1/implementation_progress.md`
  - `docs/architecture/backtest/benchmark_iterations/<date>_ai_configurator_assistant_v1/implementation_progress.json`
- Update `docs/architecture/backtest/backtest-ai-configurator-assistant-v1.md` only if Stage 00 discovers that the current plan is factually wrong.
- Run `uv run python -m tools.docs.generate_docs_index --check` when Markdown docs/evidence are changed.

## Tests

- Add targeted unit coverage if the MCP search/get logic is non-trivial.
- Prefer focused test files under `tests/unit/tools/`.
- Do not broaden to full repo tests unless a focused gate reveals a shared breakage.

## Mac Studio evidence

- Record the exact host, model id, LM Studio version, command/API path, and probe timestamp.
- Do not mark accepted from local-only execution.
- If SSH, LM Studio, model load, port conflict, or API compatibility blocks the probe, write the blocker precisely and stop with `accepted=false`.

## Delivery

- If `accepted=false`, do not claim readiness for the next stage.
- If `accepted=true`, commit scoped Stage 00 artifacts on `main`, push to `origin/main`, wait for CI/deploy path, sync exact commit on Mac Studio, and run a final Stage 00 smoke/evidence verification.
- Use `publish-ci-deploy` for that end-to-end delivery step.

# Files to indicate (expected touched areas)

Primary touches:

- `tools/backtest_ai_configurator/stage00_lmstudio_mcp_probe.py`
- `tools/backtest_ai_configurator/stage00_backtest_context_mcp.py`
- `tools/backtest_ai_configurator/stage00_backtest_ai_context_mvp.json`
- `docs/architecture/backtest/benchmark_iterations/<date>_ai_configurator_assistant_v1/iteration_00_lmstudio_mcp_feasibility.md`
- `docs/architecture/backtest/benchmark_iterations/<date>_ai_configurator_assistant_v1/iteration_00_lmstudio_mcp_feasibility.json`
- `docs/architecture/backtest/benchmark_iterations/<date>_ai_configurator_assistant_v1/implementation_progress.md`
- `docs/architecture/backtest/benchmark_iterations/<date>_ai_configurator_assistant_v1/implementation_progress.json`

Possible secondary touches:

- `tests/unit/tools/test_stage00_backtest_ai_context_mcp.py`
- `docs/architecture/backtest/backtest-ai-configurator-assistant-v1.md`
- `docs/architecture/README.md`

# Non-goals

- Do not implement production AI assistant API.
- Do not implement production MCP service lifecycle.
- Do not implement chat UI.
- Do not add database tables or migrations.
- Do not add Monit, launchd, Prometheus, or Grafana config.
- Do not change `configs/prod/backtest_ai_configurator.yaml` in Stage 00 unless the architecture doc explicitly requires a factual correction; prefer evidence notes instead.
- Do not use LM Studio app RAG/document mode as a production substitute for MCP.
- Do not pass the full context into the model prompt.
- Do not hide failed probes; failed probes are evidence.

# Quality gates (must run and pass)

- `uv run ruff check tools/backtest_ai_configurator tests/unit/tools`
  - Required if those paths exist after your changes.
- `uv run pytest -q tests/unit/tools/test_stage00_backtest_ai_context_mcp.py`
  - Required if a test file is created.
- `uv run python -m tools.docs.generate_docs_index --check`
  - Required after Markdown docs/evidence changes.
- Mac Studio Stage 00 probe command(s)
  - Required. Record exact commands and summarized output in evidence.
- `git diff --name-only`
  - Required. Confirm changed paths are limited to Stage 00 artifacts, evidence, optional docs index, and optional factual doc correction.

# Final output: report format (strict)

Your final message MUST be in Russian and follow exactly:

1. **Итог**
   - State `accepted=true/false`, `blocking_reason`, and `next_iteration_allowed`.

2. **Evidence**
   - List the Markdown and JSON evidence paths.
   - Include Mac Studio host/model/version summary.

3. **Проверки**
   - List commands run and whether they passed.
   - State if any expected gate was not run and why.

4. **Измененные файлы**
   - List touched paths grouped as probes, evidence, docs, tests.
   - Explicitly state whether production code/config/infra changed.

5. **Следующий шаг**
   - If accepted, say Stage 01 may start only after delivery/sync is complete.
   - If blocked, state the smallest next diagnostic or design decision needed.
