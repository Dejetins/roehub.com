---
prompt_name: backtest_ai_configurator_lmstudio_v1_12_service_lifecycle
repo: roehub.com
branch: main
scope: "Iteration 12: implement Mac Studio LM Studio/llmster service lifecycle, Monit control, loaded-model readiness and post-reboot recovery for the backtest AI configurator."

language:
  implementation: macos_ops_python_shell
  agent_report: ru

context_sources:
  always_read:
    - path: .codex/AGENTS.md
      why: "repo and delivery contract"
    - path: docs/architecture/backtest/benchmark_iterations/2026-05-13_lmstudio_serving_recovery/lmstudio_adapter_acceptance.md
      why: "required adapter acceptance"
    - path: docs/architecture/backtest/backtest-ai-configurator-mlx-v1.md
      why: "ops architecture to update"
  task_entrypoints:
    - path: infra/macos/launchd/com.roehub.backtest-ai-configurator-worker.plist
      why: "existing worker launchd style"
      inspect_symbols:
        - Label
        - ProgramArguments
    - path: infra/scripts/monit/roehub-backtest-ai-configurator.monitrc
      why: "existing Monit worker control"
      inspect_symbols:
        - roehub_backtest_ai_configurator_worker
    - path: scripts/macos/bootstrap_native_prod.sh
      why: "install native prod services"
      inspect_symbols:
        - launchd
        - monit
    - path: scripts/macos/reload_launchd_services.sh
      why: "prod reload path"
      inspect_symbols:
        - prod
  conditional_bundles:
    lmstudio_docs:
      read_when: "before implementing service commands"
      paths:
        - "https://lmstudio.ai/docs/developer/core/headless"
        - "https://lmstudio.ai/docs/cli/daemon/daemon-up"
        - "https://lmstudio.ai/docs/cli/serve/server-start"
        - "https://lmstudio.ai/docs/cli/local-models/ps"
    monitoring_docs:
      read_when: "when updating monitoring docs"
      paths:
        - docs/runbooks/mac-studio-native-backend-operations.md
        - docs/runbooks/mac-studio-monitoring-plan.md
        - infra/macos/prometheus/prometheus.prod.yml
  consult_if_needed:
    - path: docs/architecture/backtest/benchmark_iterations/2026-05-13_lmstudio_serving_recovery/lmstudio_serving_gate.md
      read_when: "if command identity conflicts"

style_references:
  - infra/macos/launchd
  - infra/scripts/monit
  - scripts/macos

documentation_continuity:
  old_current_docs:
    - docs/architecture/backtest/backtest-ai-configurator-mlx-v1.md
    - docs/runbooks/mac-studio-native-backend-operations.md
  new_doc_artifact: docs/architecture/backtest/benchmark_iterations/2026-05-13_lmstudio_serving_recovery/lmstudio_service_lifecycle.md
  canonical_shape: "runbook plus benchmark evidence markdown"
  docs_gate: "uv run python -m tools.docs.generate_docs_index --check"

hard_requirements:
  depends_on_iteration_11_accepted: true
  macstudio_local_deployment_required: true
  autostart_after_reboot_required: true
  monit_control_required: true
  loaded_model_readiness_required: true
  no_dead_mlx_runtime_service: true
  publish_ci_deploy_required: true
  macstudio_sync_required: true

task_toggles:
  implement_lmstudio_ensure_script: true
  implement_launchd: true
  implement_monit: true
  update_bootstrap_reload: true
  run_macstudio_service_verification: true
  run_load_benchmark: false

skill_routing:
  - skill: contract-impact-analysis
    use_when: "adding service labels, env vars or config keys"
    timing: "before implementation"
    reason: "operations contract compatibility"
  - skill: root-cause-debugging
    use_when: "launchd, Monit or LM Studio readiness fails"
    timing: "if blocker"
    reason: "localize service lifecycle failures"
  - skill: backend-quality-gates
    use_when: "running script/unit/doc gates"
    timing: "during verification"
    reason: "local gates"
  - skill: publish-ci-deploy
    use_when: "after local and Mac Studio service checks pass"
    timing: "final delivery step"
    reason: "ship ops files and verify production host"

target_envs:
  - local-dev
  - mac-studio-prod
  - github-actions

required_literals:
  - "com.roehub.lmstudio-backtest-ai-runtime"
  - "roehub_lmstudio_backtest_ai_runtime"
  - "/Users/daniildegtyarev/.lmstudio/bin/lms"
  - "lms daemon up"
  - "lms server start --port 8080 --bind 127.0.0.1"
  - "lms load gemma-4-e2b-it --identifier gemma-4-e2b-it-4bit --context-length 8192 --parallel 1"
  - "lms ps --json"
  - "two stop/start/restart cycles"
  - "/v1/models is not readiness"
  - "accepted: true/false"
  - "blocking_reason"
  - "next_prompt_allowed"

non_goals:
  - "Do not edit old prompt files 01-09."
  - "Do not expose LM Studio on Tailscale or public network."
  - "Do not rely on manual LM Studio GUI state for production readiness."
  - "Do not run benchmark scenarios."

final_report_format:
  language: ru
  sections:
    - "Что реализовано"
    - "Service lifecycle"
    - "Mac Studio verification"
    - "Проверки"
    - "Доставка и Mac Studio"

quality_gates:
  - cmd: "uv run pytest -q tests/unit/apps/worker/test_backtest_ai_configurator_worker.py tests/unit/contexts/backtest/application/ai_configurator"
    expect: "passes"
  - cmd: "uv run ruff check apps/worker src/trading/contexts/backtest scripts tests/unit"
    expect: "passes"
  - cmd: "uv run pyright"
    expect: "passes"
  - cmd: "uv run python -m tools.docs.generate_docs_index --check"
    expect: "passes"
  - cmd: "git diff --check"
    expect: "passes"

expected_primary_touches:
  - infra/macos/launchd/
  - infra/scripts/monit/
  - scripts/macos/
  - apps/worker/backtest_ai_configurator/wiring/modules.py
  - docs/runbooks/mac-studio-native-backend-operations.md
  - docs/architecture/backtest/benchmark_iterations/2026-05-13_lmstudio_serving_recovery/

possible_secondary_touches:
  - infra/macos/prometheus/prometheus.prod.yml
  - docs/runbooks/mac-studio-monitoring-plan.md
  - configs/prod/backtest_ai_configurator.yaml

safety_notes:
  - "launchd labels and Monit names are operational contracts."
  - "Use LM Studio service only on Mac Studio loopback."
  - "Readiness must prove loaded model plus lightweight generation, not only open port."
---

# Task

Implement the Mac Studio service lifecycle for LM Studio serving used by `/backtests` AI Configurator.

This iteration starts only after Iteration 11 adapter acceptance exists and passed. The goal is production-operable local deployment on Mac Studio: automatic startup, model loading, Monit control, and strict readiness evidence.

Done means:

- a durable `ensure LM Studio runtime` script exists and is idempotent;
- launchd and Monit can start/restart the LM Studio runtime path;
- worker readiness or companion readiness checks prove loaded model identity plus lightweight generation;
- reboot/reload-equivalent verification is recorded on Mac Studio;
- old or hypothetical `mlx_lm.server` service paths are not part of current production operations.

## Context / Current State

Context ledger:

- completed:
  - LM Studio direct serving and adapter smoke should be accepted.
  - Worker launchd/Monit scaffolding exists.
- open_items:
  - LM Studio daemon/server/model load is manual or ad hoc.
  - `/health/ready` can be false if runtime connection disappears.
- contract_changes:
  - new local runtime service/control plane on Mac Studio.
- risks:
  - launchd KeepAlive on a one-shot command causing restart storm;
  - Monit checks only port instead of loaded model;
  - relying on LM Studio GUI state after reboot.
- next_focus:
  - production lifecycle before pipeline acceptance.

## Requirements (Must)

- Stop if Iteration 11 adapter acceptance is missing or blocked.
- Implement an idempotent script that ensures:
  - `lms daemon up`;
  - `lms server start --port 8080 --bind 127.0.0.1`;
  - model loaded with identifier `gemma-4-e2b-it-4bit`, context 8192, parallel 1.
- Use absolute `lms` path on Mac Studio.
- Add launchd/Monit integration without exposing LM Studio outside loopback.
- Update bootstrap/reload scripts so Mac Studio can install/reload the runtime predictably.
- Add readiness/smoke command that verifies:
  - server status;
  - `lms ps --json` loaded instance id;
  - `/api/v1/models` loaded instance;
  - lightweight structured generation.
- Do not accept `/v1/models` as readiness. Readiness requires loaded-model identity plus lightweight structured generation.
- Verify two full stop/start/restart cycles through the chosen launchd/Monit/service-control path on Mac Studio.
- Prove there is no restart storm: no repeated Monit restart loop, no launchd crash loop, and no "5 restarts within 10 cycles" equivalent. If the selected safe design is Monit check-program instead of a long-running launchd job, document that explicitly.
- Update runbooks and architecture docs.
- Verify on real Mac Studio through Monit/launchctl/curl/smoke.
- Markdown and JSON evidence must include explicit machine-readable gate fields: `accepted: true/false`, `blocking_reason: null|string`, and `next_prompt_allowed: true/false`.
- Run `publish-ci-deploy` after gates and host verification pass.

## Requirements (Should)

- Avoid a restart storm if `lms server start` exits successfully after starting the service.
- Prefer a check/ensure script that Monit can call for restart.
- Keep logs under existing Roehub log directories if new logs are needed.
- Make service labels explicit and stable.

## Requirements (Nice-to-have)

- Add a machine-readable JSON smoke output for future benchmark harness preflight.
- Add Prometheus-friendly status metric if it fits existing worker metrics.

# Context acquisition protocol

Read only in this order and stop once sufficient:

1. `.codex/AGENTS.md`
2. Iteration 11 adapter acceptance
3. task entrypoints
4. LM Studio docs bundle
5. monitoring docs only when touching Monit/Prometheus
6. consult-if-needed evidence for command identity conflicts

Do not preload all runbooks or old prompts.

Reading budget: max 10 repo files plus 4 LM Studio docs pages.

# Reading manifest

- `always_read`: repo contract, adapter acceptance, architecture doc.
- `task_entrypoints`: worker launchd, Monit, bootstrap, reload.
- `conditional_bundles`: LM Studio docs and monitoring docs only when needed.
- `consult_if_needed`: serving gate evidence for command conflicts.

Stop reading once service labels, scripts, checks and evidence path are clear.

# Work plan (agent should follow)

1. Verify Iteration 11 evidence.
2. Inspect current Mac Studio LM Studio and worker state.
3. Design service lifecycle with contract impact noted.
4. Implement ensure/smoke scripts, launchd, Monit, bootstrap/reload updates.
5. Update docs and evidence.
6. Run local gates.
7. Deploy/sync using `publish-ci-deploy`.
8. On Mac Studio, verify launchd/Monit control, loaded model readiness and smoke.

# Acceptance criteria (Definition of Done)

- `monit summary` or `monit status` includes the LM Studio runtime check, or exact approved alternative is documented.
- `launchctl print` can show the installed service or exact service-control design is documented.
- LM Studio runtime recovers from stop/restart path without manual GUI action.
- Two stop/start/restart cycles pass on Mac Studio without a restart storm.
- `lms ps --json` shows `gemma-4-e2b-it-4bit` loaded.
- A lightweight structured generation smoke passes after service reload.
- Worker `/health/ready` is not accepted unless runtime loaded/generation check passes.
- Evidence contains top-level gate markers: `accepted`, `blocking_reason`, and `next_prompt_allowed`; downstream prompts may proceed only when `accepted=true` and `next_prompt_allowed=true`.
- `publish-ci-deploy` reaches `deployed`, or exact `green-pr`/`blocked` state is recorded.

# Implementation constraints

## Documentation

- Update current runbooks and architecture doc.
- Create `lmstudio_service_lifecycle.md` evidence.
- Run docs index check.

## Operations

- Do not use `0.0.0.0`.
- Do not enable CORS.
- Do not require an interactive LM Studio GUI after reboot.
- If launchd cannot safely keep a one-shot script alive, document and implement the safe Monit check-program pattern instead.

# Files to indicate (expected touched areas)

Expected primary touches:

- `scripts/macos/*lmstudio*`
- `infra/macos/launchd/com.roehub.lmstudio-backtest-ai-runtime.plist`
- `infra/scripts/monit/roehub-lmstudio-backtest-ai-runtime.monitrc`
- `scripts/macos/bootstrap_native_prod.sh`
- `scripts/macos/reload_launchd_services.sh`
- `docs/runbooks/mac-studio-native-backend-operations.md`
- `docs/architecture/backtest/benchmark_iterations/2026-05-13_lmstudio_serving_recovery/lmstudio_service_lifecycle.md`

Possible secondary touches:

- `apps/worker/backtest_ai_configurator/wiring/modules.py`
- `docs/runbooks/mac-studio-monitoring-plan.md`
- `infra/macos/prometheus/prometheus.prod.yml`

# Non-goals

- No load benchmark.
- No paid-user rollout.
- No LM Studio UI embedding.
- No remote serving.

# Quality gates (must run and pass)

- `uv run pytest -q tests/unit/apps/worker/test_backtest_ai_configurator_worker.py tests/unit/contexts/backtest/application/ai_configurator`
- `uv run ruff check apps/worker src/trading/contexts/backtest scripts tests/unit`
- `uv run pyright`
- `uv run python -m tools.docs.generate_docs_index --check`
- `git diff --check`
- Mac Studio launchd/Monit/LM Studio runtime smoke after deploy.

If Mac Studio verification fails, fix root cause or report blocker. Do not proceed to the next prompt.

# Final output: report format (strict)

Report in Russian with:

- `Что реализовано`: scripts, launchd, Monit, docs.
- `Service lifecycle`: labels, commands, restart behavior, readiness.
- `Mac Studio verification`: exact host commands and results.
- `Проверки`: local gates and docs gates.
- `Доставка и Mac Studio`: publish-ci-deploy terminal state and smoke evidence.
