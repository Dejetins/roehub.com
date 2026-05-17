---
prompt_name: backtest_ai_configurator_assistant_v1_02a_artifact_availability_summary
repo: roehub.com
branch: main
scope: "Add artifact publisher availability_summary.yaml as the real source of truth for AI symbol/period availability."

language:
  implementation: python
  agent_report: ru

context_sources:
  always_read:
    - path: .codex/AGENTS.md
      why: "repo contract"
    - path: docs/architecture/backtest/backtest-ai-configurator-assistant-v1.md
      why: "assistant v1 source of truth, Iteration 02A"
    - path: docs/architecture/backtest/benchmark_iterations/2026-05-17_ai_configurator_assistant_v1/implementation_progress.md
      why: "previous iteration gate"
    - path: docs/architecture/backtest/benchmark_iterations/2026-05-17_ai_configurator_assistant_v1/implementation_progress.json
      why: "machine-readable previous iteration gate"
  task_entrypoints:
    - path: configs/prod/backtest_artifacts.yaml
      why: "artifact root and publisher contract"
    - path: apps/scheduler/backtest_artifact_publisher/main/main.py
      why: "publisher scheduler entrypoint"
    - path: apps/scheduler/backtest_artifact_publisher/wiring/modules/backtest_artifact_publisher.py
      why: "publisher wiring"
    - path: src/trading/contexts/backtest_artifacts/application/services/v2/contracts.py
      why: "current.yaml and manifest contract constants"
  conditional_bundles:
    filesystem_adapters:
      read_when: "implementing summary scanner/writer"
      paths:
        - src/trading/contexts/backtest_artifacts/adapters/outbound/artifacts_fs/path_builder.py
        - src/trading/contexts/backtest_artifacts/adapters/outbound/artifacts_fs/current_pointer_writer.py
        - src/trading/contexts/backtest_artifacts/application/services/v2/artifact_manifest_loader.py
    publish_use_cases:
      read_when: "integrating summary generation after successful publish"
      paths:
        - src/trading/contexts/backtest_artifacts/application/use_cases/publish_backtest_artifacts_v2.py
        - src/trading/contexts/backtest_artifacts/application/services/v2/artifact_slot_publisher.py
        - apps/cli/commands/backtest_artifact_publish.py
    tests:
      read_when: "adding focused tests"
      paths:
        - tests/unit/contexts/backtest/application/services/v2/artifact_testkit_v2.py
        - tests/unit/contexts/backtest_artifacts/
  consult_if_needed:
    - path: docs/runbooks/backtest-artifacts-rebuild.md
      read_when: "manual regenerate command or scheduler behavior changes"
    - path: docs/architecture/backtest/backtest-service-artifact-runtime-v1.ru.md
      read_when: "artifact runtime contract ambiguity appears"

style_references:
  - docs/architecture/backtest/benchmark_iterations/2026-05-17_ai_configurator_assistant_v1/implementation_progress.md

hard_requirements:
  previous_iteration_accepted_required: true
  source_of_truth_is_artifact_yaml: true
  atomic_summary_write: true
  macstudio_evidence_required: true
  update_progress_artifacts: true
  direct_main_publish_after_acceptance: true

task_toggles:
  implement_artifact_publisher_summary: true
  implement_ai_context_snapshot: false
  implement_ui: false

skill_routing:
  - skill: architecture-design
    use_when: "choosing where summary scanner/writer belongs inside backtest_artifacts bounded context"
    timing: "before implementation"
    reason: "preserve dependency direction and publisher ownership"
  - skill: contract-impact-analysis
    use_when: "adding availability_summary.yaml schema or CLI/config knobs"
    timing: "before final report"
    reason: "new persisted artifact contract"
  - skill: backend-quality-gates
    use_when: "running unit/lint/type checks"
    timing: "during verification"
    reason: "backend correctness"
  - skill: publish-ci-deploy
    use_when: "all local tests pass, Mac Studio summary evidence passes, iteration marker accepted=true"
    timing: "before final report"
    reason: "direct push accepted commit to origin/main, wait main CI/deploy, then sync/verify Mac Studio; no PR or feature branch"

target_envs:
  - local-dev
  - mac-studio

required_literals:
  - "/opt/roehub/state/backtest_artifacts/v2"
  - "availability_summary.yaml"
  - "exchange/market/symbol"
  - "summary_hash"
  - "current.yaml"
  - "manifest.yaml"

non_goals:
  - "Do not implement AI prompt/context snapshot in this iteration."
  - "Do not use ClickHouse, exchange APIs, market reference, or UI catalog as symbol/period source of truth."
  - "Do not scan artifact root during normal AI request handling."

final_report_format:
  language: ru
  sections:
    - "Что изменено"
    - "Schema summary"
    - "Контрактное влияние"
    - "Проверки"
    - "Mac Studio evidence"
    - "Delivery"
    - "Следующий этап"

quality_gates:
  - cmd: "uv run pytest -q tests/unit/contexts/backtest_artifacts tests/unit/contexts/backtest/application/services/v2"
    expect: "passes, or exact focused tests pass if paths differ"
  - cmd: "uv run ruff check src/trading/contexts/backtest_artifacts apps/scheduler apps/cli tests/unit/contexts/backtest_artifacts"
    expect: "passes or unrelated pre-existing failures classified"
  - cmd: "uv run pyright"
    expect: "passes or unrelated pre-existing failures classified"
  - cmd: "uv run python -m tools.docs.generate_docs_index --check"
    expect: "passes"

expected_primary_touches:
  - src/trading/contexts/backtest_artifacts/
  - apps/scheduler/backtest_artifact_publisher/
  - apps/cli/commands/backtest_artifact_publish.py
  - docs/runbooks/backtest-artifacts-rebuild.md
  - docs/architecture/backtest/benchmark_iterations/2026-05-17_ai_configurator_assistant_v1/artifact_availability_summary_contract.md
  - docs/architecture/backtest/benchmark_iterations/2026-05-17_ai_configurator_assistant_v1/iteration_02a_artifact_availability_summary.md
  - docs/architecture/backtest/benchmark_iterations/2026-05-17_ai_configurator_assistant_v1/iteration_02a_artifact_availability_summary.json
  - docs/architecture/backtest/benchmark_iterations/2026-05-17_ai_configurator_assistant_v1/implementation_progress.md
  - docs/architecture/backtest/benchmark_iterations/2026-05-17_ai_configurator_assistant_v1/implementation_progress.json

possible_secondary_touches:
  - configs/prod/backtest_artifacts.yaml
  - configs/dev/backtest_artifacts.yaml
  - configs/test/backtest_artifacts.yaml
  - docs/architecture/backtest/backtest-ai-configurator-assistant-v1.md

safety_notes:
  - "A missing or corrupt summary must make AI readiness fail closed later."
  - "Do not publish a partial summary; write temp file and atomic rename."
---

# Task

Implement Iteration 02A: make artifact publisher generate `/opt/roehub/state/backtest_artifacts/v2/availability_summary.yaml`.

Done means:

- summary YAML is generated from valid `current.yaml` + active slot `manifest.yaml` only;
- schema contains `exchange/market/symbol`, top-level `start_date/end_date`, per-timeframe coverage, active slot metadata, and `summary_hash`;
- scheduler/manual path can regenerate the summary after publish without rebuilding artifacts;
- Mac Studio evidence proves the summary matches real active artifacts;
- progress/evidence artifacts are updated and accepted changes are published with `publish-ci-deploy`.

## Context / Current State

AI Configurator must not scan full artifact root per request and must not use market reference as availability source of truth. This iteration creates the publisher-owned YAML that later context snapshot will consume.

## Requirements (Must)

- Stop if Iteration 01 is not accepted in `implementation_progress.json`.
- Also stop if the previous iteration accepted commit is not recorded as pushed to `origin/main` and verified on Mac Studio in its evidence/progress marker.
- Build summary only from artifact publisher filesystem/YAML state.
- Exclude instruments with missing/corrupt `current.yaml`, missing active slot, missing active `manifest.yaml`, or identity mismatch.
- Compute deterministic `summary_hash`; repeated generation over identical artifacts must match.
- Add focused fixture tests for valid root, missing current, missing slot, corrupt manifest, hash stability.
- Add Mac Studio evidence using real `/opt/roehub/state/backtest_artifacts/v2`; record instrument count and `binance/spot/BTCUSDT` coverage comparison.
- Create `artifact_availability_summary_contract.md`.
- Create/update `iteration_02a_artifact_availability_summary.md/json` and progress artifacts.
- After acceptance, use `publish-ci-deploy`; then sync/verify accepted commit on Mac Studio and record commit.
- Delivery contract: use `publish-ci-deploy` in explicit direct-main mode only. Do not create a feature branch, draft PR, or PR branch. Stage only scoped files, commit on `main`, and push to `origin/main` only after all gates pass and evidence has `accepted=true`; wait for relevant main CI/deploy; then pull/sync the exact commit on Mac Studio and run the iteration-specific smoke. If direct main push, CI, or Mac Studio verification cannot be completed, set `accepted=false`, `next_iteration_allowed=false`, and report the blocker.

## Requirements (Should)

- Reuse existing artifact manifest loaders/path builders where practical.
- Keep summary generation cheap and deterministic.

# Context acquisition protocol

Read only the always-read files, task entrypoints, then the filesystem/publish/test bundles that apply. Stop once scanner/writer boundaries and tests are clear.

# Acceptance criteria (Definition of Done)

- Local tests/lint/type checks pass or unrelated failures are classified.
- Mac Studio summary exists and is generated from real artifacts.
- Summary instrument count equals valid active current pointer count at run time.
- BTCUSDT summary metadata matches active manifest.
- Evidence JSON has `accepted=true`, `next_iteration_allowed=true`, `pushed_to_main=true`, and `macstudio_verified=true` after delivery.

# Implementation constraints

Use `apply_patch` for manual edits. Do not introduce exchange/API calls. Do not stage unrelated changes.

# Final output: report format (strict)

Report in Russian with the sections listed in front matter. Include evidence paths, exact summary path, Mac Studio command results summary, gates, main push/deploy status.
