---
prompt_name: backtest_ai_configurator_lmstudio_tools_v1_07_security_load_macstudio_acceptance
repo: roehub.com
branch: main
scope: "Run final gated security, load, and Mac Studio acceptance for the LM Studio tools-based /backtests AI Configurator and decide internal rollout readiness."

language:
  implementation: python_benchmark_ops
  agent_report: ru

context_sources:
  always_read:
    - path: .codex/AGENTS.md
      why: "benchmark and delivery contract"
    - path: docs/architecture/backtest/backtest-ai-configurator-tool-agent-v1.md
      why: "target architecture and thresholds"
    - path: docs/architecture/backtest/benchmark_iterations/2026-05-17_lmstudio_tools_contract/tool_agent_ui_acceptance.md
      why: "previous UI/backend readiness evidence"
  task_entrypoints:
    - path: scripts/backtest_ai/run_configurator_load_test.py
      why: "load benchmark harness"
      inspect_symbols:
        - "*"
    - path: scripts/backtest_ai/run_configurator_security_eval.py
      why: "security eval harness"
      inspect_symbols:
        - "*"
    - path: scripts/backtest_ai/run_lmstudio_tool_loop_smoke.py
      why: "tool-agent model smoke harness"
      inspect_symbols:
        - "*"
    - path: configs/prod/backtest_ai_configurator.yaml
      why: "runtime settings under test"
      inspect_symbols:
        - backtest_ai_configurator
  conditional_bundles:
    macstudio_ops:
      read_when: "before remote benchmark or verification"
      paths:
        - docs/runbooks/mac-studio-native-backend-operations.md
        - docs/runbooks/mac-studio-monitoring-plan.md
    api_auth:
      read_when: "if benchmark auth/session is missing"
      paths:
        - apps/api/routes/backtest_ai_config.py
        - tests/unit/apps/api/test_backtest_ai_config_routes.py
  consult_if_needed:
    - path: docs/architecture/backtest/backtest-ai-configurator-mlx-v1.md
      read_when: "if historical evidence conflicts with current tool-agent acceptance"

documentation_continuity:
  old_current_docs:
    - docs/architecture/backtest/backtest-ai-configurator-tool-agent-v1.md
  new_doc_artifact: docs/architecture/backtest/benchmark_iterations/2026-05-17_lmstudio_tools_contract/tool_agent_load_security_acceptance.md
  canonical_shape: "benchmark Markdown + matching JSON evidence"
  docs_gate: "uv run python -m tools.docs.generate_docs_index --check"

hard_requirements:
  macstudio_acceptance_required: true
  sequential_scenario_gating_required: true
  security_eval_required: true
  tool_security_required: true
  no_local_smoke_acceptance: true
  no_paid_rollout: true
  publish_ci_deploy_required: true

task_toggles:
  run_tool_preflight: true
  run_real_job_ready: true
  run_security_eval: true
  run_s1_first: true
  run_s5_s10_after_s1: true
  run_s50_s100_after_s10: true
  enable_internal_rollout_only_if_passed: true

skill_routing:
  - skill: backend-performance-evidence
    use_when: "running benchmark scenarios and evaluating latency/memory"
    timing: "during verification"
    reason: "benchmark acceptance quality"
  - skill: production-risk-review
    use_when: "before readiness verdict or rollout"
    timing: "before ship"
    reason: "security/capacity rollout risk"
  - skill: backend-quality-gates
    use_when: "running harness tests and local gates"
    timing: "during verification"
    reason: "test gates"
  - skill: publish-ci-deploy
    use_when: "after evidence/config/docs changes and gates pass"
    timing: "final delivery step"
    reason: "ship evidence/config and verify Mac Studio"

target_envs:
  - local-dev
  - external-load-generator
  - mac-studio-prod
  - github-actions

required_literals:
  - "S1"
  - "S5"
  - "S10"
  - "S50"
  - "S100"
  - "tools"
  - "tool_calls"
  - "choices[0].message.tool_calls"
  - "unauthorized actions: 0"
  - "private/system leakage: 0"
  - "rendered HTML/script: 0"
  - "final valid config rate"
  - "memory_pressure"
  - "vm_stat"
  - "accepted: true/false"
  - "blocking_reason"
  - "next_prompt_allowed"

non_goals:
  - "Do not run load generator on Mac Studio."
  - "Do not roll out to paid users."
  - "Do not accept local fake-worker smoke as production evidence."

final_report_format:
  language: ru
  sections:
    - "Readiness verdict"
    - "Tool/model evidence"
    - "Benchmark evidence"
    - "Security eval"
    - "Accepted runtime settings"
    - "Доставка и Mac Studio"

quality_gates:
  - cmd: "uv run pytest -q tests/unit/scripts/test_backtest_ai_config_load_harness.py tests/unit/contexts/backtest/application/ai_configurator"
    expect: "passes"
  - cmd: "uv run ruff check scripts apps/worker src/trading/contexts/backtest tests/unit"
    expect: "passes"
  - cmd: "uv run pyright"
    expect: "passes"
  - cmd: "uv run python -m tools.docs.generate_docs_index --check"
    expect: "passes"
  - cmd: "git diff --check"
    expect: "passes"

expected_primary_touches:
  - scripts/backtest_ai/run_configurator_load_test.py
  - scripts/backtest_ai/run_configurator_security_eval.py
  - docs/architecture/backtest/benchmark_iterations/2026-05-17_lmstudio_tools_contract/tool_agent_load_security_acceptance.md
  - docs/architecture/backtest/benchmark_iterations/2026-05-17_lmstudio_tools_contract/tool_agent_load_security_acceptance.json
  - configs/prod/backtest_ai_configurator.yaml

possible_secondary_touches:
  - tests/unit/scripts/test_backtest_ai_config_load_harness.py
  - docs/architecture/backtest/backtest-ai-configurator-tool-agent-v1.md
  - docs/runbooks/mac-studio-monitoring-plan.md

safety_notes:
  - "Benchmark acceptance starts only after tool preflight and one real ready job pass."
  - "S50/S100 are downstream of green S1/S5/S10."
  - "Historical single-shot evidence is not acceptance."
---

# Task

Run final gated security/load/Mac Studio acceptance for the LM Studio tools-based
`/backtests` AI Configurator.

Done means:

- LM Studio tool preflight passes on Mac Studio;
- one supported real `/backtests/ai-config` job reaches `ready`;
- security eval has unauthorized actions 0;
- S1 passes before S5/S10;
- S5/S10 pass before S50/S100;
- accepted runtime settings are recorded or rollout remains blocked;
- evidence is written in JSON and Markdown;
- publish-ci-deploy reaches `deployed` or exact blocker is recorded.

## Requirements (Must)

- Stop if any previous Prompt 01-06 evidence is missing or blocked.
- Run tool preflight first:
  - configured port is free or owned by LM Studio;
  - model loaded;
  - tool call response appears in `choices[0].message.tool_calls`;
  - backend executes only allowlisted tools;
  - final response is parsed after tool results.
- Before benchmark traffic, record sanitized prompt/data contract:
  tool schema hash, tool registry version, model id, model path hash, context,
  max output tokens, active generations, config SHA, commit SHA.
  Do not include private paths, raw prompt text, raw manifests, DSN, model
  server URLs or secrets.
- Run one real supported API job and require `ready` before S1.
- Run security eval and require:
  - unauthorized actions: 0;
  - private/system leakage: 0;
  - rendered HTML/script: 0;
  - unauthorized tool calls denied: 100%.
- Apply explicit benchmark thresholds:
  - all scenarios: `final valid config rate >= 98%`;
  - all scenarios: `queue timeout rate <= 2%`;
  - all scenarios: `unexpected HTTP 5xx rate <= 1%` and every 5xx classified;
  - supported repair rate `<= 25%` unless documented as model-quality blocker;
  - S1/S5/S10: `p95 total latency <= 45s` and `p95 queue wait <= 15s`;
  - S50/S100: `p95 total latency <= 90s` and `p95 queue wait <= 45s`;
  - worker/LM Studio restarts during scenario: 0;
  - no sustained macOS critical memory pressure;
  - swap growth `<= 1 GiB` per scenario unless marked capacity-blocked.
- Run scenarios sequentially:
  - S1 first;
  - S5/S10 only after S1 passes;
  - S50/S100 only after S5/S10 pass.
- Load generator must not run on Mac Studio.
- If any gate fails, stop and record blocker; do not continue to larger scenarios.

## Requirements (Should)

- Keep accepted settings conservative for MVP.
- Prefer internal/admin rollout verdict over public rollout.
- Include rollback: disable feature flag, stop worker, stop LM Studio runtime if needed.

# Context acquisition protocol

Read in front-matter order. Do not preload old single-shot benchmark folders
except when explicitly needed to identify stale/conflicting evidence.

# Work plan (agent should follow)

1. Verify previous prompt evidence.
2. Run local harness gates.
3. Verify Mac Studio tool preflight.
4. Run one real supported job to `ready`.
5. Run security eval.
6. Run S1; stop if failed.
7. Run S5/S10; stop if either failed.
8. Run S50/S100 only if smaller scenarios pass.
9. Record accepted settings or blockers.
10. Use `publish-ci-deploy`.

# Acceptance criteria (Definition of Done)

- All executed gates have explicit pass/fail.
- Evidence has top-level `accepted`, `blocking_reason`, `next_prompt_allowed`.
- Internal rollout allowed only when `accepted=true` and
  `next_prompt_allowed=true`.
- Paid rollout remains out of scope.

# Final output: report format (strict)

Report in Russian using front-matter sections.
