---
prompt_name: backtest_ai_configurator_assistant_v1_02b_context_snapshot_from_summary
repo: roehub.com
branch: main
scope: "Build AI context snapshot from availability_summary.yaml, indicator catalog, executable support, and form limits."

language:
  implementation: python
  agent_report: ru

context_sources:
  always_read:
    - path: .codex/AGENTS.md
      why: "repo contract"
    - path: docs/architecture/backtest/backtest-ai-configurator-assistant-v1.md
      why: "assistant v1 source of truth, Iteration 02B"
    - path: docs/architecture/backtest/benchmark_iterations/2026-05-17_ai_configurator_assistant_v1/implementation_progress.md
      why: "02A gate state"
    - path: docs/architecture/backtest/benchmark_iterations/2026-05-17_ai_configurator_assistant_v1/implementation_progress.json
      why: "machine-readable 02A gate"
  task_entrypoints:
    - path: configs/prod/backtest_ai_configurator.yaml
      why: "AI limits and snapshot settings"
    - path: configs/prod/indicators.yaml
      why: "indicator parameter source"
    - path: src/trading/contexts/backtest/application/services/signals_from_indicators_v1.py
      why: "executable signal support"
    - path: src/trading/contexts/backtest/application/ai_configurator/
      why: "AI configurator application area"
  conditional_bundles:
    indicator_defs:
      read_when: "building indicator availability intersection"
      paths:
        - src/trading/contexts/indicators/domain/definitions/
        - src/trading/contexts/backtest/application/services/v2/preflight.py
    summary_contract:
      read_when: "availability summary schema or evidence is unclear"
      paths:
        - docs/architecture/backtest/benchmark_iterations/2026-05-17_ai_configurator_assistant_v1/artifact_availability_summary_contract.md
        - docs/architecture/backtest/benchmark_iterations/2026-05-17_ai_configurator_assistant_v1/iteration_02a_artifact_availability_summary.json
    tests:
      read_when: "adding focused snapshot and availability tests"
      paths:
        - tests/unit/contexts/backtest/application/ai_configurator/
        - tests/unit/contexts/backtest/application/services/v2/

style_references:
  - docs/architecture/backtest/benchmark_iterations/2026-05-17_ai_configurator_assistant_v1/implementation_progress.md

hard_requirements:
  previous_iteration_accepted_required: true
  read_symbols_periods_from_availability_summary: true
  one_symbol_prompt_context: true
  indicator_intersection_required: true
  update_progress_artifacts: true
  direct_main_publish_after_acceptance: true

task_toggles:
  implement_context_snapshot: true
  implement_conversation_api: false
  implement_ui: false

skill_routing:
  - skill: architecture-design
    use_when: "defining snapshot DTOs/ports and adapter dependency direction"
    timing: "before implementation"
    reason: "snapshot is a boundary contract"
  - skill: contract-impact-analysis
    use_when: "adding snapshot schema/config or exposing helper endpoint"
    timing: "before final report"
    reason: "schema drives prompt and UI"
  - skill: backend-quality-gates
    use_when: "running tests/lint/type checks"
    timing: "during verification"
    reason: "backend correctness"
  - skill: publish-ci-deploy
    use_when: "all gates pass, Mac Studio snapshot smoke passes, marker accepted=true"
    timing: "before final report"
    reason: "publish accepted changes and verify accepted commit on Mac Studio"

target_envs:
  - local-dev
  - mac-studio

required_literals:
  - "availability_summary.yaml"
  - "allowed_values.symbol"
  - "ignored_symbols"
  - "range"
  - "explicit"
  - "none"
  - "structure.percent_rank"

non_goals:
  - "Do not call LM Studio."
  - "Do not implement conversation storage or UI."
  - "Do not put full symbol universe into model prompt."

final_report_format:
  language: ru
  sections:
    - "Что изменено"
    - "Snapshot schema"
    - "Indicator audit"
    - "Контрактное влияние"
    - "Проверки"
    - "Mac Studio"
    - "Delivery"

quality_gates:
  - cmd: "uv run pytest -q tests/unit/contexts/backtest/application/ai_configurator"
    expect: "passes"
  - cmd: "uv run ruff check src/trading/contexts/backtest/application/ai_configurator tests/unit/contexts/backtest/application/ai_configurator"
    expect: "passes"
  - cmd: "uv run pyright"
    expect: "passes or unrelated pre-existing failures classified"
  - cmd: "uv run python -m tools.docs.generate_docs_index --check"
    expect: "passes"

expected_primary_touches:
  - src/trading/contexts/backtest/application/ai_configurator/context_snapshot.py
  - src/trading/contexts/backtest/application/ai_configurator/dto.py
  - src/trading/contexts/backtest/application/ai_configurator/ports.py
  - src/trading/contexts/backtest/adapters/outbound/ai_configurator_context/
  - configs/prod/backtest_ai_configurator.yaml
  - tests/unit/contexts/backtest/application/ai_configurator/test_context_snapshot.py
  - tests/unit/contexts/backtest/application/ai_configurator/test_indicator_availability_audit.py
  - docs/architecture/backtest/benchmark_iterations/2026-05-17_ai_configurator_assistant_v1/context_snapshot_contract.md
  - docs/architecture/backtest/benchmark_iterations/2026-05-17_ai_configurator_assistant_v1/iteration_02b_context_snapshot.md
  - docs/architecture/backtest/benchmark_iterations/2026-05-17_ai_configurator_assistant_v1/iteration_02b_context_snapshot.json
  - docs/architecture/backtest/benchmark_iterations/2026-05-17_ai_configurator_assistant_v1/implementation_progress.md
  - docs/architecture/backtest/benchmark_iterations/2026-05-17_ai_configurator_assistant_v1/implementation_progress.json

possible_secondary_touches:
  - docs/architecture/backtest/backtest-ai-configurator-assistant-v1.md

safety_notes:
  - "Snapshot can expose sanitized source names/hashes, not local filesystem paths."
  - "If summary is missing/stale/corrupt, readiness must fail closed later."
---

# Task

Implement Iteration 02B: context snapshot builder for AI Configurator using `availability_summary.yaml` as the only source of truth for symbols, exchange/market, timeframes, and periods.

Done means:

- prompt context can be built for exactly one resolved symbol;
- multiple-symbol user requests are represented as first symbol plus warning/ignored symbols;
- indicator availability is YAML + hard definitions + signal registry + defaults/compute + summary coverage;
- explicit/no-window axes are represented correctly;
- context evidence and progress artifacts are accepted and delivery is complete through `publish-ci-deploy`.

## Requirements (Must)

- Stop if Iteration 02A is not accepted.
- Also stop if the previous iteration accepted commit is not recorded as pushed to `origin/main` and verified on Mac Studio in its evidence/progress marker.
- Read symbols/periods from `availability_summary.yaml`, never from market reference.
- Expose `allowed_values.symbol`, not full symbol universe, in model prompt context.
- Preserve `structure.percent_rank` explicit values; do not convert to min/max range.
- Classify all 40 prod indicators as available or excluded with reason.
- Add tests for summary source, multi-symbol handling, explicit axis, no-window axis, and missing summary fail-closed behavior.
- Update `context_snapshot_contract.md`, `iteration_02b_context_snapshot.md/json`, and progress artifacts.
- After accepted evidence, use `publish-ci-deploy`; sync/verify accepted commit on Mac Studio.
- Delivery contract: use `publish-ci-deploy` in explicit direct-main mode only. Do not create a feature branch, draft PR, or PR branch. Stage only scoped files, commit on `main`, and push to `origin/main` only after all gates pass and evidence has `accepted=true`; wait for relevant main CI/deploy; then pull/sync the exact commit on Mac Studio and run the iteration-specific smoke. If direct main push, CI, or Mac Studio verification cannot be completed, set `accepted=false`, `next_iteration_allowed=false`, and report the blocker.

## Requirements (Should)

- Keep snapshot compact and hashable.
- Do not expose local artifact paths to model-facing context.

# Acceptance criteria (Definition of Done)

- Snapshot tests prove symbols/timeframes/periods come from summary.
- 40-indicator audit is complete.
- Mac Studio smoke reads real summary and builds BTCUSDT context.
- Evidence JSON has `accepted=true`, `next_iteration_allowed=true`, `pushed_to_main=true`, and `macstudio_verified=true` after delivery.

# Final output: report format (strict)

Report in Russian. Include paths, gates, Mac Studio snapshot hash/sample, main commit/deploy, and exact next iteration allowance.
