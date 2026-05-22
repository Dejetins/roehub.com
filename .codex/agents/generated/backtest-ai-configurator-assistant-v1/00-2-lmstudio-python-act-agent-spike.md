---
prompt_name: backtest_ai_configurator_assistant_v1_00_2_lmstudio_python_act_agent_spike
repo: /Users/daniildegtyarev/Projects/roehub.com
branch: main
scope: "Stage 00.2: prove or block the lower-than-MVP agent runtime using lmstudio-python .act() on Mac Studio with gemma-4-e2b-it-4bit before any production AI configurator implementation."

language:
  implementation: python
  agent_report: ru

context_sources:
  always_read:
    - path: .codex/AGENTS.md
      why: "repo contract, safety, skill routing"
    - path: .codex/agents/.context/promt_manager_state.yaml
      why: "latest compact state; ignore if stale"
    - path: docs/architecture/backtest/backtest-ai-configurator-assistant-v1.md
      why: "current assistant plan and failed Stage 00/00.1 context"
  task_entrypoints:
    - path: .codex/agents/generated/backtest-ai-configurator-assistant-v1/00-1-controlled-agent-verification-mvp.md
      why: "failed predecessor prompt; use only as negative evidence"
      inspect_symbols:
        - "Stage 00.1"
        - "blocked"
        - "MCP"
    - path: docs/architecture/backtest/benchmark_iterations/2026-05-21_ai_configurator_assistant_v1/
      why: "prior failed evidence for gemma and qwen"
      inspect_symbols:
        - "accepted"
        - "blocking_reason"
        - "mcp_tool_calls_total"
    - path: configs/prod/backtest_ai_configurator.yaml
      why: "target model/runtime config candidate"
      inspect_symbols:
        - "lm_studio"
        - "model"
        - "runtime"
    - path: configs/prod/indicators.yaml
      why: "indicator truth source for the sample index"
      inspect_symbols:
        - "momentum.rsi"
        - "ma.ema"
        - "structure.percent_rank"
  conditional_bundles:
    lmstudio_python_act_docs:
      read_when: "lmstudio-python .act() API shape or callback evidence is unclear"
      paths:
        - https://lmstudio.ai/docs/python/agent/act
        - https://lmstudio.ai/docs/python
        - https://lmstudio.ai/docs/developer/rest
    artifact_availability_sources:
      read_when: "building the sample prepared file/index needs real source-shape confirmation"
      paths:
        - docs/architecture/backtest/backtest-ai-configurator-assistant-v1.md
        - /opt/roehub/state/backtest_artifacts/v2
    docs_index:
      read_when: "Markdown docs or evidence files are created or updated"
      paths:
        - docs/architecture/README.md
        - tools/docs/generate_docs_index.py
  consult_if_needed:
    - path: .codex/PLANS.md
      read_when: "plan history conflicts with current assistant doc"
    - path: pyproject.toml
      read_when: "local test dependencies or optional lmstudio dependency handling is unclear"
    - path: docs/architecture/backtest/
      read_when: "evidence folder naming or report shape is ambiguous"

style_references:
  - .codex/promt_template.md
  - docs/architecture/backtest/backtest-ai-configurator-assistant-v1.md

hard_requirements:
  no_production_code_changes: true
  macstudio_evidence_required: true
  use_lmstudio_python_act: true
  target_model_gemma4_required: true
  one_allowed_file_index_access_function_only: true
  model_must_choose_queries_itself: true
  no_full_context_prompt_dump: true
  no_shell_env_repo_db_network_tools_exposed_to_model: true
  write_machine_readable_evidence: true
  do_not_start_next_stage_if_blocked: true
  direct_main_publish_only_if_accepted_and_scope_clean: true

task_toggles:
  create_experimental_scripts: true
  create_sample_prepared_file_index: true
  create_lmstudio_act_probe: true
  create_minimal_validator_fixture: true
  run_real_macstudio_probe: true
  update_architecture_doc_if_facts_change: true
  implement_production_service: false
  implement_ui: false
  implement_storage: false
  implement_monit_or_launchd: false

skill_routing:
  - skill: root-cause-debugging
    use_when: "lmstudio-python .act() cannot call the allowed function or returns unexpected responses"
    timing: during investigation
    reason: "owns evidence-first diagnosis before changing the prompt or probe"
  - skill: backend-quality-gates
    use_when: "creating Python probe scripts, fixture access code, or tests"
    timing: during verification
    reason: "owns focused ruff/pytest gates"
  - skill: backend-performance-evidence
    use_when: "reporting latency, memory, call counts, or acceptance metrics"
    timing: during verification
    reason: "owns measurement discipline and clear thresholds"
  - skill: publish-ci-deploy
    use_when: "only after accepted=true, no unrelated dirty scope, and the user has not forbidden publish"
    timing: before ship
    reason: "owns direct main publish/CI/Mac Studio sync if this evidence artifact is accepted"

target_envs:
  - local
  - macstudio

required_literals:
  - "lmstudio-python"
  - ".act()"
  - "gemma-4-e2b-it-4bit"
  - "/Users/daniildegtyarev/.lmstudio/models/mlx-community/gemma-4-e2b-it-4bit"
  - "prepared file/index"
  - "lookup_prepared_backtest_index"
  - "final JSON"
  - "Roehub validator"
  - "accepted"
  - "blocking_reason"
  - "next_iteration_allowed"
  - "used_context_ids"
  - "structure.percent_rank"
  - "momentum.rsi"
  - "ma.ema"

non_goals:
  - "Do not implement production AI service, UI, DB, Monit, launchd, Prometheus, or Grafana."
  - "Do not restore old /backtests AI mode buttons or old job endpoints."
  - "Do not use LM Studio /api/v1/chat + MCP as the target path in this stage."
  - "Do not expose arbitrary read_file, shell, env, repo, DB, network, or run-backtest capability to the model."
  - "Do not pass the full prepared file/index into the system prompt or user prompt."
  - "Do not treat parseable JSON as semantic success without validator acceptance."

final_report_format:
  language: ru
  sections:
    - "Итог"
    - "Evidence"
    - "Что проверено"
    - "Метрики"
    - "Измененные файлы"
    - "Блокеры / следующий шаг"

quality_gates:
  - cmd: "uv run ruff check tools/backtest_ai_configurator tests/unit/tools"
    expect: "passes if experimental Python files/tests are created"
  - cmd: "uv run pytest -q tests/unit/tools/test_stage00_2_lmstudio_act_agent.py"
    expect: "passes if the test file is created; otherwise explain why not applicable"
  - cmd: "uv run python -m tools.docs.generate_docs_index --check"
    expect: "passes when Markdown docs/evidence are changed"
  - cmd: "ssh macstudio '<Stage 00.2 lmstudio-python .act() probe command>'"
    expect: "real Mac Studio evidence writes accepted/blocking JSON"
  - cmd: "git diff --check"
    expect: "passes"
  - cmd: "git status --short"
    expect: "reviewed; unrelated dirty files are not touched"

expected_primary_touches:
  - "tools/backtest_ai_configurator/stage00_2_lmstudio_act_agent_probe.py"
  - "tools/backtest_ai_configurator/stage00_2_prepared_backtest_index_mvp.json"
  - "tests/unit/tools/test_stage00_2_lmstudio_act_agent.py"
  - "docs/architecture/backtest/benchmark_iterations/2026-05-22_ai_configurator_assistant_v1/iteration_00_2_lmstudio_act_agent_spike.md"
  - "docs/architecture/backtest/benchmark_iterations/2026-05-22_ai_configurator_assistant_v1/iteration_00_2_lmstudio_act_agent_spike.json"
  - "docs/architecture/backtest/benchmark_iterations/2026-05-22_ai_configurator_assistant_v1/implementation_progress.md"
  - "docs/architecture/backtest/benchmark_iterations/2026-05-22_ai_configurator_assistant_v1/implementation_progress.json"

possible_secondary_touches:
  - "tools/backtest_ai_configurator/__init__.py"
  - "docs/architecture/backtest/backtest-ai-configurator-assistant-v1.md"
  - "docs/architecture/README.md"
  - "pyproject.toml"
  - "uv.lock"

safety_notes:
  - "This stage is a disposable feasibility spike below MVP; production code/config/infra changes are prohibited."
  - "If lmstudio-python is not available, use a transient dependency method for the Mac Studio probe before editing pyproject.toml."
  - "Only one model-visible function is allowed: lookup_prepared_backtest_index."
  - "The function may read only the prepared file/index fixture; it must reject arbitrary paths."
  - "The model must choose lookup queries itself; the probe must not parse user prompts into semantic backend queries."
  - "If accepted=false, do not publish and do not start production implementation."
---

# Task

Execute Stage 00.2 for `AI-помощник конфигуратора /backtests v1`: prove or block the lower-than-MVP agent runtime using official `lmstudio-python` `.act()` with the already-tested Mac Studio `gemma-4-e2b-it-4bit` model.

Target concept:

```text
Roehub AI service spike
        ↓
lmstudio-python .act()
        ↓
one allowed function: lookup_prepared_backtest_index
        ↓
the model chooses what to search inside one prepared file/index
        ↓
final JSON
        ↓
minimal Roehub validator fixture
```

Done means:

- Mac Studio evidence proves whether `lmstudio-python .act()` can run the model as an agent for this task.
- The model uses only `lookup_prepared_backtest_index` for supported prompts before returning final JSON.
- The final JSON is parseable and passes a minimal Roehub validator fixture for supported prompts.
- Unsupported/security/off-topic/auto-run prompts do not produce accepted `config_ready`.
- Evidence files include `accepted`, `blocking_reason`, and `next_iteration_allowed`.
- No production service/UI/storage/ops code is implemented in this stage.

## Context / Current State

Context ledger from the previous attempts:

- completed:
  - LM Studio can load local MLX models on Mac Studio.
  - Gemma 4 could call MCP tools in earlier experiments, but the full Stage 00 matrix failed.
  - Qwen2.5-Coder produced parseable JSON but failed the Stage 00.1 MCP path because it did not call MCP tools in that harness.
  - A later manual probe showed Qwen can call tools when the tool-calling protocol is simpler, so the failure is protocol-level, not proof that local models cannot use tools.
- open_items:
  - We have not proven `lmstudio-python .act()` on Mac Studio with `gemma-4-e2b-it-4bit`.
  - We have not proven that a single allowed function over one prepared file/index is enough for final `/backtests` JSON.
  - We have not proven a minimal validator fixture can accept the final JSON produced by the agent.
- contract_changes:
  - Stage 00.2 must not continue the `/api/v1/chat + MCP` path as the active architecture.
  - Stage 00.2 tests the official LM Studio Python agent API instead.
  - Stage 00.2 is not a production contract and must not create public API/storage/UI commitments.
- touched_paths:
  - Create only experimental scripts, fixtures, tests, and evidence under the paths listed in front matter.
  - Update the architecture document only if Stage 00.2 discovers facts that make the current plan materially wrong.
- risks:
  - `gemma-4-e2b-it-4bit` may not reliably call the allowed function through `.act()`.
  - The SDK may not expose enough call transcript evidence; if so, the stage must block rather than assume.
  - Adding `lmstudio-python` as a project dependency too early may pollute production deps.
- next_focus:
  - Verify `.act()` protocol with a tiny matrix before any Roehub service implementation.
  - Record exact Mac Studio commands, LM Studio version, loaded model identifier, and `lmstudio-python` package version.
  - Decide from evidence whether Stage 01 can be redesigned around `.act()` or must choose another agent runtime.

Additional context:

- Use the model `gemma-4-e2b-it-4bit` and path `/Users/daniildegtyarev/.lmstudio/models/mlx-community/gemma-4-e2b-it-4bit`.
- If the model is not loaded, load it through `/Users/daniildegtyarev/.lmstudio/bin/lms load gemma-4-e2b-it --identifier gemma-4-e2b-it-4bit --context-length 8192 --parallel 1 -y` or the nearest verified LM Studio CLI equivalent.
- Do not rely on `/v1/models` as readiness. Readiness for this stage is: model loaded plus a lightweight `.act()` smoke that returns the expected answer.

## Requirements (Must)

- Use `lmstudio-python` `.act()` as the agent runtime. Do not implement a custom multi-round agent loop in this stage.
- Run real checks on Mac Studio. Local tests alone are not acceptance.
- Use only `gemma-4-e2b-it-4bit` for the acceptance run unless it cannot be loaded; if it cannot be loaded, block with exact reason.
- Expose exactly one model-callable function:
  - function name: `lookup_prepared_backtest_index`
  - input: a model-chosen `query: str` and optional small `limit`
  - output: bounded JSON from the prepared file/index
  - allowed data source: only `stage00_2_prepared_backtest_index_mvp.json`
- The function must not accept or follow arbitrary file paths from the model.
- The function must not expose shell, env, repo, DB, network, file listing, arbitrary file read, or backtest execution.
- The model must decide what to look up. Do not add deterministic backend prompt interpretation or semantic selector logic.
- The system prompt must be short and explicit:
  - the assistant prepares `/backtests` configurations only;
  - it must inspect the prepared file/index through the allowed function before `config_ready`;
  - it must not use model memory as source of truth;
  - it must not run backtests;
  - it must return final JSON only after the agent steps.
- The final JSON envelope must include:
  - `schema_version`
  - `status`
  - `assistant_message`
  - `config`
  - `used_context_ids`
  - `unsupported_items`
  - `warnings`
- Build a minimal Roehub validator fixture for Stage 00.2:
  - `status=config_ready` requires non-null `config`;
  - `config.symbol` must be exactly one symbol from the prepared file/index;
  - requested indicator ids must exist in the prepared file/index;
  - indicator params must obey the prepared file/index rules;
  - `structure.percent_rank.window` must use only explicitly allowed values;
  - no-window indicators must not contain synthetic `window`;
  - period/timeframe must fit the prepared file/index.
- Create machine-readable and Markdown evidence:
  - `iteration_00_2_lmstudio_act_agent_spike.json`
  - `iteration_00_2_lmstudio_act_agent_spike.md`
  - `implementation_progress.json`
  - `implementation_progress.md`
- Evidence JSON must include:
  - `accepted`
  - `blocking_reason`
  - `next_iteration_allowed`
  - `host`
  - `lm_studio_version`
  - `lmstudio_python_version`
  - `model_id`
  - `loaded_model_identifier`
  - `prepared_index_hash`
  - `prompt_count`
  - `act_calls_total`
  - `lookup_function_calls_total`
  - `supported_final_valid_rate`
  - `unsupported_ready_configs`
  - `security_ready_configs`
  - `auto_run_ready_configs`
  - `safe_prompts_blocked`
  - `latency_summary_seconds`
  - per-prompt results with status, lookup calls, validator result, and failure reasons.
- If `.act()` transcript/callback evidence is not available, evidence must say so and acceptance must be blocked.
- If Stage 00.2 is blocked, do not publish, do not deploy, and do not start Stage 01.

## Requirements (Should)

- Prefer transient dependency use for `lmstudio-python` in the spike, for example `uv run --with lmstudio ...`, before adding `lmstudio-python` to `pyproject.toml`.
- If transient dependency use is not viable and `pyproject.toml` must be edited, explain why in evidence and keep it clearly optional/experimental.
- Keep the prepared file/index small but representative:
  - `BTCUSDT`
  - `1h`
  - period bounds
  - `momentum.rsi`
  - `ma.ema`
  - `structure.percent_rank`
  - one no-window indicator
  - risk/direction/fees/sizing/ranking basics.
- Record a compact redacted transcript per prompt:
  - user prompt id;
  - function call queries;
  - function result ids;
  - final JSON status;
  - validator result.
- Use deterministic ordering for prepared file/index results and evidence arrays.

## Requirements (Nice-to-have)

- Include a small comparison note explaining why `.act()` is different from the failed `/api/v1/chat + MCP` path.
- Capture `memory_pressure` or a simple process memory snapshot on Mac Studio if it is easy and non-disruptive.

# Context acquisition protocol

Read only in this order and stop once sufficient:

1. `.codex/AGENTS.md`
2. `.codex/agents/.context/promt_manager_state.yaml` or latest state snapshot, if available
3. current architecture doc and prior Stage 00/00.1 evidence summaries
4. task entrypoints
5. only the conditional bundle(s) required by touched contracts or failing checks
6. consult-if-needed references only for blockers, ambiguity, or conflicts

Do not eagerly preload all listed sources.

Pre-implementation reading target:

- `<= 8 files`
- `<= ~35k-50k tokens`

Stop reading once all of the following are true:

- Stage 00.2 scope is clear,
- the `.act()` probe shape is clear,
- the prepared file/index fixture shape is clear,
- evidence schema is implementable,
- no production API/storage/UI/ops contract ambiguity remains.

Expand context only for:

- `.act()` API ambiguity,
- failing Mac Studio probe,
- missing model/runtime,
- unclear docs/evidence path shape,
- dependency conflict.

# Reading manifest

Use the front-matter `context_sources` as the canonical reading map.

Read with this intent:

- `always_read`:
  - repository rules,
  - compact prior state,
  - current assistant plan.
- `task_entrypoints`:
  - failed predecessor context,
  - target runtime/model config,
  - indicator fixture truth source.
- `conditional_bundles`:
  - read only when the stated condition applies.
- `consult_if_needed`:
  - read only for blockers, ambiguity, or conflict resolution.

Do not convert this manifest into a broad mandatory reading list.

# Work plan (agent should follow)

Skill routing for this task:

- `root-cause-debugging`: use during investigation if `.act()` does not call the allowed function or transcript evidence is missing; form a root-cause hypothesis before changing the probe.
- `backend-quality-gates`: use during verification for focused Python tests/lint.
- `backend-performance-evidence`: use during verification for latency/call-count metrics and threshold reporting.
- `publish-ci-deploy`: use before ship only if `accepted=true`, scope is clean, and publishing is still desired; do not use it for blocked evidence.

1. Check current `git status --short`. Preserve unrelated dirty files.
2. Verify Mac Studio access and LM Studio CLI path:
   - `ssh macstudio 'hostname; /Users/daniildegtyarev/.lmstudio/bin/lms --version; /Users/daniildegtyarev/.lmstudio/bin/lms ps'`
3. Verify or load `gemma-4-e2b-it-4bit` with context length 8192 and parallel 1.
4. Verify `lmstudio-python` availability on Mac Studio using a transient dependency method if needed.
5. Create `stage00_2_prepared_backtest_index_mvp.json`.
6. Create `stage00_2_lmstudio_act_agent_probe.py`.
7. Create unit tests for:
   - prepared index loading/hash;
   - `lookup_prepared_backtest_index` bounded results;
   - arbitrary path rejection by construction;
   - minimal validator fixture behavior.
8. Run a Mac Studio smoke:
   - system prompt respected;
   - `.act()` can call `lookup_prepared_backtest_index`;
   - final JSON parseable.
9. Run the fixed Stage 00.2 prompt matrix.
10. Write evidence Markdown/JSON and implementation progress files.
11. Run quality gates.
12. If blocked, stop and report. If accepted, report clearly and only publish if scope is clean and allowed by the current user/request context.

# Acceptance criteria (Definition of Done)

- `accepted=true` only if all required conditions pass.
- Mac Studio host is used for acceptance evidence.
- Target model is `gemma-4-e2b-it-4bit`.
- `lmstudio-python .act()` is the actual runtime used for the acceptance probe.
- Exactly one model-callable function is exposed: `lookup_prepared_backtest_index`.
- Supported prompt matrix:
  - at least 6 supported prompts;
  - every supported prompt calls `lookup_prepared_backtest_index` at least once;
  - every supported prompt returns parseable final JSON;
  - every supported prompt passes the minimal Roehub validator fixture;
  - `supported_final_valid_rate = 1.0`.
- Required supported prompts include:
  - RU: RSI + EMA for BTCUSDT on 1h;
  - EN: RSI + EMA for BTCUSDT on 1h;
  - `structure.percent_rank` with valid explicit value;
  - one no-window indicator case;
  - conservative risk/no-risk case;
  - “which indicators/symbol is available” informational request that does not return `config_ready` unless it also produces a valid config request.
- Negative prompt matrix:
  - at least 4 negative prompts;
  - unsupported symbol;
  - unsupported indicator;
  - prompt injection / prompt extraction;
  - auto-run backtest request.
- Negative acceptance:
  - `unsupported_ready_configs = 0`;
  - `security_ready_configs = 0`;
  - `auto_run_ready_configs = 0`.
- `safe_prompts_blocked = 0`.
- No full prepared file/index is placed into system prompt, user prompt, or final evidence transcript.
- Evidence JSON has `accepted`, `blocking_reason`, and `next_iteration_allowed`.
- `next_iteration_allowed=true` only when `accepted=true`.
- Production code/config/infra/UI/storage are unchanged.

# Implementation constraints

## Determinism & ordering

- Keep prepared file/index arrays deterministically sorted.
- Keep evidence prompt results sorted by prompt id.
- Use stable JSON serialization for hashes.
- Do not rely on incidental dictionary ordering for acceptance decisions.

## API / contracts

- Stage 00.2 does not create public API contracts.
- Stage 00.2 does not modify `/backtests` UI.
- Stage 00.2 does not modify production DB schemas.
- Stage 00.2 does not install Monit/launchd/Prometheus/Grafana artifacts.
- Stage 00.2 does not restore old AI config job endpoints.

## Documentation

- If Stage 00.2 changes current architecture facts, update `docs/architecture/backtest/backtest-ai-configurator-assistant-v1.md` in Russian.
- If Markdown docs/evidence are changed, run the docs index check.
- Do not rewrite historical evidence files from Stage 00 or Stage 00.1.
- New evidence must live under `docs/architecture/backtest/benchmark_iterations/2026-05-22_ai_configurator_assistant_v1/`.

## Safety

- The only model-callable function is `lookup_prepared_backtest_index`.
- The function reads only the prepared file/index loaded by the probe process.
- The function must not accept a path.
- The function must not expose system paths, repo paths, environment variables, secrets, shell output, DB rows, network calls, or raw artifact files.
- The final JSON must not be accepted if the validator fixture rejects it.

## Dependency handling

- Prefer not to edit `pyproject.toml` for Stage 00.2.
- Use transient `lmstudio-python` dependency execution for the Mac Studio probe if possible.
- If a dependency edit is necessary, keep it scoped, justify it in evidence, and run focused dependency/gate checks.

## Model/runtime

- Use the Mac Studio LM Studio installation.
- Use `/Users/daniildegtyarev/.lmstudio/bin/lms` if `lms` is not in PATH.
- Do not treat a loaded model list alone as readiness.
- Readiness requires `.act()` smoke with the target model.

# Files to indicate (expected touched areas)

Expected primary touches:

- `tools/backtest_ai_configurator/stage00_2_lmstudio_act_agent_probe.py`
- `tools/backtest_ai_configurator/stage00_2_prepared_backtest_index_mvp.json`
- `tests/unit/tools/test_stage00_2_lmstudio_act_agent.py`
- `docs/architecture/backtest/benchmark_iterations/2026-05-22_ai_configurator_assistant_v1/iteration_00_2_lmstudio_act_agent_spike.md`
- `docs/architecture/backtest/benchmark_iterations/2026-05-22_ai_configurator_assistant_v1/iteration_00_2_lmstudio_act_agent_spike.json`
- `docs/architecture/backtest/benchmark_iterations/2026-05-22_ai_configurator_assistant_v1/implementation_progress.md`
- `docs/architecture/backtest/benchmark_iterations/2026-05-22_ai_configurator_assistant_v1/implementation_progress.json`

Possible secondary touches:

- `tools/backtest_ai_configurator/__init__.py`
- `docs/architecture/backtest/backtest-ai-configurator-assistant-v1.md`
- `docs/architecture/README.md`
- `pyproject.toml`
- `uv.lock`

Report any additional touched file as either:

- required for Stage 00.2,
- unrelated pre-existing,
- or accidental and reverted.

# Non-goals

- No production AI assistant service.
- No `/backtests` UI changes.
- No conversation storage.
- No migrations.
- No Monit/launchd/service installation.
- No Prometheus/Grafana changes.
- No benchmark for 10 users.
- No old Stage 00/00.1 evidence overwrite.
- No custom agent loop unless `.act()` itself is unavailable; if `.act()` is unavailable, block instead of replacing it silently.

# Quality gates (must run and pass)

Run these locally when corresponding files exist:

```bash
uv run ruff check tools/backtest_ai_configurator tests/unit/tools
uv run pytest -q tests/unit/tools/test_stage00_2_lmstudio_act_agent.py
uv run python -m tools.docs.generate_docs_index --check
git diff --check
git status --short
```

Run the Mac Studio probe and record the exact command in evidence. The command should prove:

- target host;
- LM Studio version;
- `lmstudio-python` version;
- target model loaded;
- `.act()` smoke;
- full prompt matrix;
- final evidence write.

If any gate cannot run, classify it:

- introduced by Stage 00.2,
- pre-existing but on required path,
- unrelated pre-existing,
- environmental,
- or intentionally not applicable.

Do not claim acceptance if the Mac Studio probe did not run.

# Final output: report format (strict)

Write the final report in Russian with these sections:

## Итог

- `accepted=<true|false>`
- `blocking_reason=<reason|null>`
- `next_iteration_allowed=<true|false>`

## Evidence

- Markdown evidence path.
- JSON evidence path.
- Implementation progress paths.

## Что проверено

- LM Studio version.
- `lmstudio-python` version.
- Loaded model identifier.
- `.act()` smoke result.
- Prompt matrix summary.

## Метрики

- `prompt_count`
- `lookup_function_calls_total`
- `supported_final_valid_rate`
- `safe_prompts_blocked`
- `unsupported_ready_configs`
- `security_ready_configs`
- `auto_run_ready_configs`
- latency summary

## Измененные файлы

- Group by probes, tests, evidence, docs, dependency changes.

## Блокеры / следующий шаг

- If accepted: state what Stage 01 may implement next.
- If blocked: state the smallest next diagnostic step.
- If publish was skipped: state why.
