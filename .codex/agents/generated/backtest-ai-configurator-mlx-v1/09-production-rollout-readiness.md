---
prompt_name: backtest_ai_configurator_mlx_v1_09_production_rollout_readiness
repo: roehub.com
branch: main
scope: "Iteration 09: perform final production readiness review, gated rollout, required deploy/sync through Mac Studio, and verify /backtests AI configurator end to end."

language:
  implementation: release_ops
  agent_report: ru

context_sources:
  always_read:
    - path: .codex/AGENTS.md
      why: "delivery and safety contract"
    - path: docs/architecture/backtest/backtest-ai-configurator-mlx-v1.md
      why: "MVP production-ready checklist"
  task_entrypoints:
    - path: docs/architecture/backtest/benchmark_iterations
      why: "Mac Studio benchmark/security acceptance evidence"
      inspect_symbols:
        - "*"
    - path: docs/runbooks/mac-studio-native-backend-operations.md
      why: "Mac Studio deploy/reload/smoke path"
      inspect_symbols:
        - "*"
    - path: docs/runbooks/mac-studio-monitoring-plan.md
      why: "Prometheus/Grafana/Monit verification"
      inspect_symbols:
        - "*"
    - path: scripts/macos/smoke_prod.sh
      why: "production smoke entrypoint"
      inspect_symbols:
        - "*"
  conditional_bundles:
    publish_deploy:
      read_when: "only if user explicitly asks to publish/deploy or current prompt includes delivery"
      paths:
        - .github/workflows/deploy-backend.yml
        - scripts/macos/bootstrap_native_prod.sh
        - scripts/macos/reload_launchd_services.sh
    browser_qa:
      read_when: "when verifying /backtests UI after deploy"
      paths:
        - apps/web/templates/pages/backtests.html
        - apps/web/dist/js/pages/backtests.js
    ai_runtime:
      read_when: "when checking worker, Monit, Prometheus or model reload"
      paths:
        - apps/worker/backtest_ai_configurator
        - infra/scripts/monit/roehub-backtest-ai-configurator.monitrc
        - infra/macos/prometheus/prometheus.prod.yml
  consult_if_needed:
    - path: .codex/agents/.context/promt_manager_state.yaml
      read_when: "only to check for a newer executor handoff; ignore if stale/unrelated"
    - path: docs/architecture/backtest/backtest-job-runner-production-plan-v1.md
      read_when: "if /backtests runtime readiness conflicts with AI rollout"

style_references:
  - path: docs/architecture/backtest/backtest-ai-configurator-mlx-v1.md
    purpose: "production-ready checklist"
  - path: docs/runbooks/mac-studio-native-backend-operations.md
    purpose: "Mac Studio verification commands"

hard_requirements:
  all_prior_iterations_required: true
  macstudio_benchmark_required: true
  security_eval_required: true
  feature_flag_rollout_required: true
  rollback_path_required: true
  publish_to_main_required: true
  no_public_rollout_on_unknowns: true
  publish_ci_deploy_required: true
  main_branch_deployment_required: true
  macstudio_sync_required: true

task_toggles:
  perform_readiness_review: true
  verify_local_gates: true
  verify_browser: true
  verify_macstudio: true
  publish_ci_deploy_required: true
  enable_paid_tiers: false

skill_routing:
  - skill: pre-ship-gate
    use_when: "if available, before rollout to assess readiness without publishing"
    timing: "before ship"
    reason: "final readiness gate"
  - skill: production-risk-review
    use_when: "reviewing final diff, contracts, security and ops risks"
    timing: "before ship"
    reason: "production risk assessment"
  - skill: backend-performance-evidence
    use_when: "checking benchmark evidence and accepted runtime settings"
    timing: "before ship"
    reason: "Mac Studio performance acceptance"
  - skill: browser-qa-evidence
    use_when: "verifying /backtests browser-visible behavior"
    timing: "during verification"
    reason: "browser-visible acceptance"
  - skill: publish-ci-deploy
    use_when: "after readiness passes, deliver the final iteration to main, sync Mac Studio, and run post-deploy verification"
    timing: "final delivery step"
    reason: "required end-to-end Roehub GitHub CI, main deployment, Mac Studio sync and smoke"

target_envs:
  - local-dev
  - browser
  - github-actions
  - mac-studio-prod

required_literals:
  - "feature flag default off"
  - "admin/internal users"
  - "Monit"
  - "Prometheus"
  - "S1/S5/S10/S50/S100"
  - "rollback: disable feature flag, stop worker"

non_goals:
  - "Do not roll out to paid tiers unless explicitly approved after evidence."
  - "Do not bypass benchmark/security gates."
  - "Do not use remote/non-MLX fallback."
  - "Do not auto-run backtest jobs from AI."

final_report_format:
  language: ru
  sections:
    - "Readiness verdict"
    - "Evidence checked"
    - "Rollout/deploy status"
    - "Verification"
    - "Доставка и Mac Studio"
    - "Residual risks and rollback"

quality_gates:
  - cmd: "uv run pytest -q tests/unit/contexts/backtest/application/ai_configurator tests/unit/apps/api/test_backtest_ai_config_routes.py tests/unit/apps/web"
    expect: "passes"
  - cmd: "uv run ruff check apps/api apps/web apps/worker src/trading/contexts/backtest scripts tests/unit"
    expect: "passes"
  - cmd: "uv run pyright"
    expect: "passes"
  - cmd: "python -m tools.docs.generate_docs_index --check"
    expect: "passes"
  - cmd: "git diff --check"
    expect: "passes"

expected_primary_touches:
  - "docs/architecture/backtest/benchmark_iterations/"
  - "docs/runbooks/mac-studio-native-backend-operations.md"
  - "docs/runbooks/mac-studio-monitoring-plan.md"

possible_secondary_touches:
  - "configs/prod/backtest_ai_configurator.yaml"
  - "infra/macos/prometheus/prometheus.prod.yml"
  - "infra/scripts/monit/roehub-backtest-ai-configurator.monitrc"
  - "apps/web/dist/js/pages/backtests.js"

safety_notes:
  - "This is a gate/rollout prompt. If evidence is missing, do not ship; report blocker."
  - "Publishing through publish-ci-deploy is required after readiness passes; continue through CI/deploy/Mac Studio smoke."
  - "Mac Studio checkout and /opt/roehub/app deployed runtime are different surfaces."
---

# Task

Execute Iteration 09: final production readiness review and gated rollout for the `/backtests` AI Configurator. Verify all prior iteration evidence, local gates, browser behavior, Mac Studio service health, Prometheus/Monit metrics, and benchmark/security acceptance. After readiness passes, use `publish-ci-deploy` to complete the delivery chain to `main` and Mac Studio.

Done means:

- readiness verdict is explicit: `ready`, `ready for internal/admin only`, or `not ready`;
- S1/S5/S10/S50/S100 Mac Studio evidence is present and accepted, or rollout is blocked;
- security eval has 0 unauthorized actions and 0 leakage, or rollout is blocked;
- `/backtests` browser flow works without auto-running jobs;
- worker is managed by launchd/Monit and metrics are scrapeable;
- rollback path is tested or operationally clear;
- Mac Studio repo/runtime verification and smoke are complete, or exact deploy blocker is recorded.

## Context / Current State

Context ledger:

- completed:
  - Iterations 01-08 should have implemented storage, API, validation/security, prompt/repair, MLX worker, UI, ops and benchmark harness/evidence.
- open_items:
  - paid-tier rollout remains a product decision after internal/admin acceptance.
- contract_changes:
  - public AI endpoints, browser UI, persisted audit data, service/metrics/ops surface.
- risks:
  - missing Mac Studio evidence;
  - security false positives or missed prompt injection;
  - service managed by launchd but not actually reachable by Monit/Prometheus;
  - UI appearing ready while backend is disabled.
- next_focus:
  - internal/admin production rollout or explicit blocker list.

## Requirements (Must)

- Verify every item in the architecture doc `MVP production-ready checklist`.
- Confirm local gates pass.
- Confirm benchmark/security evidence exists and is current for the implementation commit/config.
- Confirm feature flag defaults to off or internal/admin only before public rollout.
- Confirm no non-MLX fallback is configured.
- Confirm no AI output can create/cancel/delete/run a backtest job.
- Confirm Monit can status/restart worker and Prometheus target is up after deploy if deploy is in scope.
- Confirm browser QA for `/backtests`, RU/EN data notice, status events, load config and no auto-run.
- If evidence is missing, report blocker and do not ship.
- When gates pass, use `publish-ci-deploy` and continue through CI, deploy to `main`, Mac Studio sync/reload/smoke.

## Requirements (Should)

- Keep rollout initially admin/internal.
- Include exact rollback: disable feature flag, stop worker via Monit, keep existing `/backtests` form.
- Include exact accepted model/concurrency/context values from benchmark evidence.
- Classify residual risks by severity.

## Requirements (Nice-to-have)

- Capture Grafana/Prometheus target output snippets if accessible.
- Include browser screenshot paths if browser QA tooling produces them.

# Context acquisition protocol

Read only in this order and stop once sufficient:

1. `.codex/AGENTS.md`
2. `.codex/agents/.context/promt_manager_state.yaml` or latest state snapshot, if available and relevant
3. latest executor final report, if available
4. task entrypoints
5. only conditional bundles required by deploy/browser/runtime checks
6. consult-if-needed references only for blockers

Do not eagerly preload all listed sources.

Pre-implementation reading target:

- `<= 8 files`
- `<= ~35k-50k tokens`

Stop reading once readiness checklist, benchmark evidence, deploy state and verification commands are clear.

# Reading manifest

Use the front-matter `context_sources` as the canonical reading map.

Read with this intent:

- `always_read`: delivery rules and production-ready checklist.
- `task_entrypoints`: benchmark evidence and Mac Studio runbooks.
- `conditional_bundles`: publish/deploy, browser QA, runtime only when needed.
- `consult_if_needed`: adjacent backtest runner only if rollout conflicts.

Do not convert this manifest into a broad mandatory reading list.

# Work plan (agent should follow)

Skill routing for this task:

- `pre-ship-gate`: use if available before rollout; owns final readiness.
- `production-risk-review`: use before ship; owns security/contracts/ops risk.
- `backend-performance-evidence`: use before ship; owns benchmark evidence review.
- `browser-qa-evidence`: use during verification; owns browser-visible acceptance.
- `publish-ci-deploy`: required after readiness passes; continue until `deployed`, `green-pr`, or `blocked`.

1. Inspect current diff/scope and verify prior iteration artifacts.
2. Check MVP production-ready checklist item by item.
3. Verify local gates and docs index.
4. Verify benchmark/security evidence identity and accepted runtime settings.
5. Run browser QA locally or against deployed target depending on current stage.
6. If readiness passes, run full publish-ci-deploy flow to `main` and Mac Studio.
7. Verify Mac Studio service: Monit summary/status, launchctl, health/metrics, Prometheus target, smoke prompt if available.
8. Report verdict, evidence and rollback.

# Acceptance criteria (Definition of Done)

- Final verdict is explicit and evidence-backed.
- No missing benchmark/security/ops/browser gate is hidden.
- If not ready, blocker list is actionable and ordered.
- CI/deploy/Mac Studio smoke evidence is included, or exact `green-pr`/`blocked` state is recorded.
- Rollback path is concrete.

- `publish-ci-deploy` terminal state is `deployed`, or `green-pr`/`blocked` is reported with exact blocker evidence.

# Implementation constraints

## Release safety

- This prompt pack explicitly requests publish/deploy; stop only for a concrete `publish-ci-deploy` `green-pr` or `blocked` terminal state.
- Do not enable paid tiers by default.
- Do not paper over missing Mac Studio evidence with local tests.

## Contracts

- Existing `/backtests/jobs` flow remains the manual run path.
- AI routes remain additive and feature-flagged.

## Operations

- Treat `/Users/daniildegtyarev/Projects/roehub.com` checkout and `/opt/roehub/app` runtime bundle as different surfaces.
- Use runbook commands for bootstrap/reload/smoke when deploying.

# Files to indicate (expected touched areas)

Expected primary touches:

- `docs/architecture/backtest/benchmark_iterations/`
- `docs/runbooks/mac-studio-native-backend-operations.md`
- `docs/runbooks/mac-studio-monitoring-plan.md`

Possible secondary touches:

- `configs/prod/backtest_ai_configurator.yaml`
- `infra/macos/prometheus/prometheus.prod.yml`
- `infra/scripts/monit/roehub-backtest-ai-configurator.monitrc`
- `apps/web/dist/js/pages/backtests.js`

# Non-goals

- No new feature work.
- No paid-tier rollout without explicit approval.
- No non-MLX fallback.
- No direct AI job execution.

# Quality gates (must run and pass)

- `uv run pytest -q tests/unit/contexts/backtest/application/ai_configurator tests/unit/apps/api/test_backtest_ai_config_routes.py tests/unit/apps/web`
- `uv run ruff check apps/api apps/web apps/worker src/trading/contexts/backtest scripts tests/unit`
- `uv run pyright`
- `python -m tools.docs.generate_docs_index --check`
- `git diff --check`
- Browser QA on `/backtests`
- Mac Studio service/metrics/smoke checks through `publish-ci-deploy`

If a gate cannot run, classify it and do not mark production readiness complete unless the missing gate is explicitly non-blocking for an internal-only rollout.

Required delivery step: after the quality gates above pass, invoke `publish-ci-deploy` as the final step. The expected terminal state for this prompt is `deployed`: intended files committed and pushed, GitHub Actions green, revision shipped to `main`, `/opt/roehub/app` on `macstudio` pulled to that revision, the relevant production services reloaded through the repository runbook, and `bash scripts/macos/smoke_prod.sh` passed. If the skill reaches `green-pr` because a human merge/approval is required, or `blocked` because of missing auth, unrelated dirty scope, external CI, Mac Studio access, or production verification failure, report that exact state and do not claim deployment.

# Final output: report format (strict)

Report in Russian with:

- `Readiness verdict`: ready/internal-only/not ready and why.
- `Evidence checked`: benchmark/security/browser/ops/local gates.
- `Rollout/deploy status`: deployed/green-pr/blocked with exact evidence.
- `Verification`: commands, targets, results.
- `Доставка и Mac Studio`: publish-ci-deploy terminal state, main/PR SHA, CI result, Mac Studio pull/reload/smoke evidence, or exact blocker.
- `Residual risks and rollback`: ordered risks and exact rollback steps.
