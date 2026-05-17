---
prompt_name: backtest_ai_configurator_assistant_v1_09_macstudio_s10_benchmark
repo: roehub.com
branch: main
scope: "Run final Mac Studio benchmark for /backtests AI assistant with S1/S5/S10 only."

language:
  implementation: python_benchmark
  agent_report: ru

context_sources:
  always_read:
    - path: .codex/AGENTS.md
      why: "repo contract"
    - path: docs/architecture/backtest/backtest-ai-configurator-assistant-v1.md
      why: "Benchmark thresholds and Iteration 09"
    - path: docs/architecture/backtest/benchmark_iterations/2026-05-17_ai_configurator_assistant_v1/implementation_progress.md
      why: "Iteration 08 human-readable gate"
    - path: docs/architecture/backtest/benchmark_iterations/2026-05-17_ai_configurator_assistant_v1/implementation_progress.json
      why: "Iteration 08 gate"
  task_entrypoints:
    - path: scripts/benchmarks/
      why: "benchmark harness location"
    - path: tests/fixtures/ai_configurator/
      why: "prompt fixture candidates"
    - path: configs/prod/backtest_ai_configurator.yaml
      why: "runtime/concurrency/limits"
    - path: docs/architecture/backtest/benchmark_iterations/2026-05-17_ai_configurator_assistant_v1/
      why: "evidence output location"

hard_requirements:
  previous_iteration_accepted_required: true
  max_load_users_10: true
  s1_s5_s10_only: true
  macstudio_evidence_required: true
  accepted_thresholds_explicit: true
  direct_main_publish_after_acceptance: true

task_toggles:
  implement_benchmark_harness: true
  run_final_benchmark: true

skill_routing:
  - skill: backend-performance-evidence
    use_when: "designing/running benchmark and reporting latency/throughput/memory"
    timing: "during verification"
    reason: "benchmark claims require comparable evidence"
  - skill: backend-quality-gates
    use_when: "running harness tests/lint/type checks"
    timing: "during verification"
    reason: "benchmark harness correctness"
  - skill: publish-ci-deploy
    use_when: "S1/S5/S10 thresholds pass, marker accepted=true"
    timing: "before final report"
    reason: "direct push accepted benchmark evidence to origin/main, wait main CI/deploy, then sync/verify Mac Studio; no PR or feature branch"

target_envs: [mac-studio]

required_literals:
  - "S1"
  - "S5"
  - "S10"
  - "10 users"
  - "1-9 indicators"
  - "valid config rate"
  - "p95 ready latency"
  - "queue timeout rate"

non_goals:
  - "Do not run S50 or S100."
  - "Do not accept local-only benchmark evidence."
  - "Do not change product architecture unless a blocker is found and documented."

final_report_format:
  language: ru
  sections: ["Benchmark result", "Thresholds", "Mac Studio evidence", "Failures/blockers", "Delivery", "Next"]

quality_gates:
  - cmd: "uv run pytest -q tests/unit/contexts/backtest/application/ai_configurator tests/unit/apps/api"
    expect: "regression tests pass before benchmark"
  - cmd: "uv run ruff check scripts tests src/trading/contexts/backtest/application/ai_configurator apps/api"
    expect: "passes"
  - cmd: "uv run pyright"
    expect: "passes or unrelated pre-existing failures classified"
  - cmd: "uv run python -m tools.docs.generate_docs_index --check"
    expect: "passes"

expected_primary_touches:
  - scripts/benchmarks/
  - tests/fixtures/ai_configurator/
  - docs/architecture/backtest/benchmark_iterations/2026-05-17_ai_configurator_assistant_v1/benchmark_report.md
  - docs/architecture/backtest/benchmark_iterations/2026-05-17_ai_configurator_assistant_v1/benchmark_report.json
  - docs/architecture/backtest/benchmark_iterations/2026-05-17_ai_configurator_assistant_v1/iteration_09_benchmark.md
  - docs/architecture/backtest/benchmark_iterations/2026-05-17_ai_configurator_assistant_v1/iteration_09_benchmark.json
  - docs/architecture/backtest/benchmark_iterations/2026-05-17_ai_configurator_assistant_v1/implementation_progress.md
  - docs/architecture/backtest/benchmark_iterations/2026-05-17_ai_configurator_assistant_v1/implementation_progress.json

safety_notes:
  - "Benchmark must record model identifier, config, commit, host, summary_hash, and context snapshot hash."
  - "If S10 fails capacity threshold, accepted=false with blocking_reason."
---

# Task

Implement/run Iteration 09 final benchmark on Mac Studio. The load test is limited to S1/S5/S10, with S10 simulating 10 users.

## Requirements (Must)

- Stop if Iteration 08 is not accepted.
- Also stop if the previous iteration accepted commit is not recorded as pushed to `origin/main` and verified on Mac Studio in its evidence/progress marker.
- Run preflight gates first: direct structured smoke 10/10, adapter generate 10/10, adapter repair 10/10, one API ready job, one UI apply smoke.
- Benchmark categories include supported create RU/EN, 1-9 indicators, edit/explain/list, multiple-symbol first-symbol behavior, unsupported symbol/indicator, safer config, security regressions.
- Load profiles:
  - S1: 1 user, 10 sequential requests;
  - S5: 5 users, think time 20-90 sec;
  - S10: 10 users, think time 20-120 sec.
- Do not run S50/S100.
- Thresholds must match the source doc exactly: valid config rate >= 95%, 1-9 matrix 9/9, invalid load_action 0, security leakage 0, safe prompts blocked 0/10, HTTP 5xx 0, S10 queue timeout <= 1%, S10 p95 ready <= 120s, S10 p95 queue wait <= 90s, normal memory pressure, swap growth < 1GB.
- Create benchmark report/evidence JSON and progress updates.
- After accepted benchmark, use `publish-ci-deploy`; sync/verify accepted commit and benchmark evidence on Mac Studio.
- Delivery contract: use `publish-ci-deploy` in explicit direct-main mode only. Do not create a feature branch, draft PR, or PR branch. Stage only scoped files, commit on `main`, and push to `origin/main` only after all gates pass and evidence has `accepted=true`; wait for relevant main CI/deploy; then pull/sync the exact commit on Mac Studio and run the iteration-specific smoke. If direct main push, CI, or Mac Studio verification cannot be completed, set `accepted=false`, `next_iteration_allowed=false`, and report the blocker.

# Acceptance criteria (Definition of Done)

- Mac Studio benchmark report contains model/config/commit/host/summary_hash/context_snapshot_hash.
- S1/S5/S10 all pass thresholds.
- Evidence JSON has `accepted=true`, `next_iteration_allowed=true`, `pushed_to_main=true`, and `macstudio_verified=true` after delivery.
- If any threshold fails, set `accepted=false`, concrete `blocking_reason`, and do not publish as accepted.

# Final output: report format (strict)

Report in Russian with a compact threshold table, exact evidence paths, Mac Studio host/model/config, delivery status, and any residual risks.
