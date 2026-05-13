---
prompt_name: backtest_ai_configurator_lmstudio_v1_10_serving_gate
repo: roehub.com
branch: main
scope: "Iteration 10: establish and document the LM Studio local serving gate on Mac Studio before any further benchmark or runtime code work."

language:
  implementation: docs_ops
  agent_report: ru

context_sources:
  always_read:
    - path: .codex/AGENTS.md
      why: "repo contract and Mac Studio evidence rules"
    - path: docs/architecture/backtest/benchmark_iterations/2026-05-12_iteration_08_ai_configurator_load_security/s1_s5_s10_mlx_benchmark_2026-05-12.md
      why: "failed Iteration 08 evidence"
    - path: docs/architecture/backtest/backtest-ai-configurator-mlx-v1.md
      why: "current stale MLX architecture source"
  task_entrypoints:
    - path: configs/prod/backtest_ai_configurator.yaml
      why: "current model id, path, base URL"
      inspect_symbols:
        - backtest_ai_configurator
    - path: apps/worker/backtest_ai_configurator/wiring/modules.py
      why: "current readiness checks"
      inspect_symbols:
        - _runtime_connection_check
        - build_backtest_ai_configurator_worker_app
    - path: docs/architecture/backtest/benchmark_iterations/2026-05-12_iteration_08_ai_configurator_load_security
      why: "failed benchmark and blocker evidence"
      inspect_symbols:
        - "*"
  conditional_bundles:
    lmstudio_docs:
      read_when: "before deciding exact LM Studio commands"
      paths:
        - "https://lmstudio.ai/docs/developer/core/server"
        - "https://lmstudio.ai/docs/developer/core/headless"
        - "https://lmstudio.ai/docs/cli/serve/server-start"
        - "https://lmstudio.ai/docs/cli/local-models/load"
        - "https://lmstudio.ai/docs/developer/rest/list"
        - "https://lmstudio.ai/docs/developer/openai-compat/structured-output"
    macstudio_ops:
      read_when: "before running Mac Studio checks"
      paths:
        - docs/runbooks/mac-studio-native-backend-operations.md
        - docs/runbooks/mac-studio-monitoring-plan.md
  consult_if_needed:
    - path: .codex/agents/.context/promt_manager_state.yaml
      read_when: "only to check newer executor handoff"

style_references:
  - docs/architecture/backtest/benchmark_iterations/2026-05-12_iteration_08_ai_configurator_load_security/README.md
  - docs/architecture/backtest/benchmark_iterations/2026-05-12_iteration_08_ai_configurator_load_security/macstudio_blocker.md

documentation_continuity:
  old_current_docs:
    - docs/architecture/backtest/backtest-ai-configurator-mlx-v1.md
    - docs/architecture/backtest/benchmark_iterations/2026-05-12_iteration_08_ai_configurator_load_security/README.md
  new_doc_artifact: docs/architecture/backtest/benchmark_iterations/2026-05-13_lmstudio_serving_recovery/lmstudio_serving_gate.md
  canonical_shape: "benchmark evidence folder with README, markdown evidence, JSON evidence"
  docs_gate: "uv run python -m tools.docs.generate_docs_index --check"

hard_requirements:
  macstudio_real_host_required: true
  lmstudio_installed_on_macstudio: true
  no_benchmark_until_serving_gate_passes: true
  no_mlx_lm_server_runtime: true
  loopback_only: true
  publish_ci_deploy_required: true
  macstudio_sync_required: true

task_toggles:
  run_lmstudio_direct_smoke: true
  update_architecture_docs: true
  create_serving_evidence: true
  change_roehub_runtime_code: false
  run_load_benchmark: false

skill_routing:
  - skill: architecture-design
    use_when: "recording target LM Studio serving contract"
    timing: "before implementation"
    reason: "runtime boundary and staged rollout design"
  - skill: backend-performance-evidence
    use_when: "classifying direct serving smoke and benchmark gating"
    timing: "during verification"
    reason: "separate smoke evidence from benchmark acceptance"
  - skill: publish-ci-deploy
    use_when: "after docs/evidence changes and local gates pass"
    timing: "final delivery step"
    reason: "ship docs/evidence and sync Mac Studio"

target_envs:
  - local-dev
  - mac-studio-prod
  - github-actions

required_literals:
  - "/Users/daniildegtyarev/.lmstudio/bin/lms"
  - "lms daemon up"
  - "lms server start --port 8080 --bind 127.0.0.1"
  - "lms load gemma-4-e2b-it --identifier gemma-4-e2b-it-4bit --context-length 8192 --parallel 1"
  - "/api/v1/models"
  - "/v1/chat/completions"
  - "response_format"
  - "json_schema"
  - "accepted: true/false"
  - "blocking_reason"
  - "next_prompt_allowed"

non_goals:
  - "Do not edit old prompt files 01-09."
  - "Do not start S1/S5/S10/S50/S100 benchmark in this iteration."
  - "Do not implement Roehub adapter changes yet."
  - "Do not expose LM Studio beyond loopback."

final_report_format:
  language: ru
  sections:
    - "Serving gate verdict"
    - "Mac Studio evidence"
    - "Документация"
    - "Блокеры"
    - "Доставка и Mac Studio"

quality_gates:
  - cmd: "uv run python -m tools.docs.generate_docs_index --check"
    expect: "passes if docs changed"
  - cmd: "git diff --check"
    expect: "passes"
  - cmd: "ssh macstudio '/Users/daniildegtyarev/.lmstudio/bin/lms daemon status --json; /Users/daniildegtyarev/.lmstudio/bin/lms server status --json --quiet; /Users/daniildegtyarev/.lmstudio/bin/lms ps --json'"
    expect: "recorded in evidence"

expected_primary_touches:
  - docs/architecture/backtest/backtest-ai-configurator-mlx-v1.md
  - docs/architecture/backtest/benchmark_iterations/2026-05-13_lmstudio_serving_recovery/

possible_secondary_touches:
  - docs/architecture/backtest/README.md
  - docs/architecture/README.md

safety_notes:
  - "LM Studio server must bind to 127.0.0.1 only."
  - "Do not include secrets, cookies, tokens, or full private prompt logs in evidence."
  - "Direct serving smoke is prerequisite evidence, not final benchmark acceptance."
---

# Task

Establish the LM Studio local serving gate for `/backtests` AI Configurator on the real Mac Studio host.

This iteration is a recovery and evidence step. It must prove whether LM Studio can reliably serve `gemma-4-e2b-it-4bit` through local loopback APIs before any Roehub adapter changes or benchmark reruns.

Done means:

- LM Studio daemon/server/model state is checked on Mac Studio using absolute `lms` path;
- model `gemma-4-e2b-it` is loaded as identifier `gemma-4-e2b-it-4bit` with context 8192 and parallel 1, or exact blocker is recorded;
- direct `/v1/chat/completions` structured-output smoke runs 10/10 successfully, or exact failure bodies are recorded;
- docs clearly state that `mlx_lm.server` is not the accepted runtime for this checkpoint;
- new evidence exists under `docs/architecture/backtest/benchmark_iterations/2026-05-13_lmstudio_serving_recovery/`;
- no S1/S5/S10/S50/S100 benchmark is run yet.

## Context / Current State

Context ledger:

- completed:
  - Iterations 01-09 produced code, UI, worker, ops scaffolding and failed benchmark evidence.
  - LM Studio is installed on Mac Studio.
  - Model artifact exists under LM Studio model storage.
- open_items:
  - LM Studio server is not accepted as a persistent serving component.
  - Worker readiness is currently too weak because `/v1/models` is not enough.
  - Benchmark acceptance is blocked.
- contract_changes:
  - target runtime should move from generic `mlx_lm.server` assumption to LM Studio local API.
- risks:
  - treating downloaded model or `/v1/models` as loaded-model acceptance;
  - starting benchmark before generation and JSON schema smoke pass;
  - leaking private local paths or prompts in public-facing docs.
- next_focus:
  - prove the serving gate independently from Roehub pipeline.

## Requirements (Must)

- Use official LM Studio docs for server, headless/daemon, model load, list models, chat completions and structured output.
- Use absolute CLI path `/Users/daniildegtyarev/.lmstudio/bin/lms` on Mac Studio.
- Use loopback only: `127.0.0.1:8080`.
- Verify all layers separately: daemon, server, loaded model, native model list, OpenAI-compatible generation, structured output.
- Use `response_format: {"type": "json_schema", ...}` in direct smoke.
- Run exactly 10 direct structured-output generation attempts and record success/failure count.
- Record HTTP status and sanitized response body for failures.
- Update current architecture docs so they do not present `mlx_lm.server` as current accepted path for this model.
- Create a new evidence folder with README, markdown evidence and JSON evidence.
- Markdown and JSON evidence must include explicit machine-readable gate fields: `accepted: true/false`, `blocking_reason: null|string`, and `next_prompt_allowed: true/false`.
- Stop and report blocker if LM Studio cannot load or generate reliably.
- Run `publish-ci-deploy` after docs/evidence changes and gates pass.

## Requirements (Should)

- Prefer `lms ps --json` and `/api/v1/models` over `/v1/models` for loaded-model readiness evidence.
- Include `lms server status --json --quiet` output in evidence.
- Include exact model key, identifier, context length, parallel, LM Studio version if available, and host timestamp.
- Keep benchmark thresholds out of this iteration except as blocked downstream gates.

## Requirements (Nice-to-have)

- Capture `memory_pressure` and `vm_stat` before/after direct smoke if inexpensive.
- Capture `lms load --estimate-only` output if it is available and non-interactive.

# Context acquisition protocol

Read only in this order and stop once sufficient:

1. `.codex/AGENTS.md`
2. latest executor final report or failed benchmark evidence
3. task entrypoints
4. LM Studio docs bundle
5. Mac Studio ops bundle only before remote checks
6. consult-if-needed references only for blockers

Do not eagerly read every old prompt or every benchmark artifact.

Reading budget: max 8 files plus the LM Studio docs pages needed for the commands.

# Reading manifest

- `always_read`: repo contract, failed benchmark, current architecture doc.
- `task_entrypoints`: config, worker readiness, failed evidence folder.
- `conditional_bundles`: LM Studio docs and Mac Studio runbooks only when needed.
- `consult_if_needed`: compact state snapshot only for newer handoff.

Stop reading once the serving gate commands, evidence location and doc update scope are clear.

# Work plan (agent should follow)

1. Confirm current Mac Studio LM Studio state without changing anything.
2. Start daemon/server/load model only if needed for this serving gate.
3. Run 10 direct structured-output chat completion attempts against `127.0.0.1:8080`.
4. Capture loaded-model evidence from `lms ps --json` and `/api/v1/models`.
5. Write the new evidence folder.
6. Update architecture docs to mark LM Studio as target runtime and `mlx_lm.server` as unsupported for this checkpoint unless re-proven.
7. Run docs gates.
8. Use `publish-ci-deploy` to ship the docs/evidence and sync Mac Studio.

# Acceptance criteria (Definition of Done)

- Direct LM Studio serving gate is `accepted` only if 10/10 structured-output attempts return valid JSON conforming to the smoke schema.
- Evidence contains top-level gate markers: `accepted`, `blocking_reason`, and `next_prompt_allowed`; downstream prompts may proceed only when `accepted=true` and `next_prompt_allowed=true`.
- If accepted, evidence includes exact commands and sanitized outputs.
- If not accepted, evidence includes exact blocker and no downstream prompt should proceed.
- Architecture docs no longer imply that `mlx_lm.server` is the current production target for `gemma-4-e2b-it-4bit`.
- No old prompt file is edited.
- `publish-ci-deploy` reaches `deployed`, or exact `green-pr`/`blocked` state is recorded.

# Implementation constraints

## Documentation

- Update old/current docs and create the new evidence artifact in the same benchmark evidence style.
- Do not leave stale statements saying that `/v1/models` alone is sufficient readiness.
- Run docs index check.

## Operations

- Do not bind LM Studio to `0.0.0.0`.
- Do not enable CORS.
- Do not store auth cookies, user session headers, or full private prompts in evidence.

# Files to indicate (expected touched areas)

Expected primary touches:

- `docs/architecture/backtest/backtest-ai-configurator-mlx-v1.md`
- `docs/architecture/backtest/benchmark_iterations/2026-05-13_lmstudio_serving_recovery/README.md`
- `docs/architecture/backtest/benchmark_iterations/2026-05-13_lmstudio_serving_recovery/lmstudio_serving_gate.md`
- `docs/architecture/backtest/benchmark_iterations/2026-05-13_lmstudio_serving_recovery/lmstudio_serving_gate.json`

Possible secondary touches:

- `docs/architecture/backtest/README.md`
- `docs/architecture/README.md`

# Non-goals

- No Roehub adapter implementation.
- No UI changes.
- No benchmark scenarios.
- No public rollout.

# Quality gates (must run and pass)

- `uv run python -m tools.docs.generate_docs_index --check`
- `git diff --check`
- Mac Studio direct LM Studio serving smoke: 10/10 structured JSON responses, or blocker evidence.

If any gate cannot run, classify it as introduced, required-path pre-existing, unrelated pre-existing, environmental, or external.

# Final output: report format (strict)

Report in Russian with:

- `Serving gate verdict`: accepted/blocked and why.
- `Mac Studio evidence`: exact commands, counts, model id, context, parallel, endpoint.
- `Документация`: old docs updated, new evidence path.
- `Блокеры`: exact blocker if not accepted.
- `Доставка и Mac Studio`: publish-ci-deploy state, commit/PR/main SHA, CI, Mac Studio sync/smoke.
