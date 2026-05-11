---
prompt_name: web_ui_backend_v1_stage09_04_final_verification_and_delivery
repo: roehub.com
branch: main
scope: "Stage 09 final verification: backend result endpoints, `/backtests` UI, performance bounds, Mac Studio deploy and smoke."

language:
  implementation: verification_deploy
  agent_report: ru

context_sources:
  always_read:
    - path: .codex/AGENTS.md
      why: "repo contract and final reporting rules"
    - path: docs/architecture/apps/web/web-ui-backend-implementation-plan-v1.md
      why: "Stage 09 acceptance criteria"
    - path: docs/architecture/backtest/backtest-service-artifact-runtime-v1.ru.md
      why: "result/artifact contract"
    - path: docs/architecture/backtest/backtest-job-runner-production-plan-v1.md
      why: "runner/materialization operational dependency"
    - path: .codex/agents/generated/web-ui-backend-v1/09-implement-backtests-results.md
      why: "parent Stage 09 prompt"
  task_entrypoints:
    - path: apps/api/routes/backtests.py
      why: "final endpoint matrix"
    - path: apps/web/templates/pages/backtests.html
      why: "final UI route"
    - path: apps/web/dist/js/pages/backtests.js
      why: "final browser behavior"
    - path: tests/unit/apps/api/test_backtests_routes.py
      why: "backend route tests"
    - path: tests/unit/apps/web/test_app_routes.py
      why: "web tests"
    - path: docs/runbooks/mac-studio-native-backend-operations.md
      why: "Mac Studio deployment/runbook path"

hard_requirements:
  no_uncommitted_unrelated_publish: true
  all_stage09_gates_required: true
  browser_qa_required: true
  result_endpoint_smoke_required: true
  mac_studio_sync_required: true
  deploy_monitoring_required: true

task_toggles:
  implementation_changes_allowed: false
  fix_introduced_failures_allowed: true
  publish_after_success: true

package_contract:
  depends_on:
    - "01 materialization/status contract accepted"
    - "02 result/statistics endpoint hardening accepted"
    - "03 result UI integration accepted"
  owns:
    - "narrow fixes for introduced failures only"
    - "test/evidence files produced by verification if repository tracks them"
  forbidden:
    - "feature expansion"
    - "layout redesign"
    - "unrelated refactors"
    - "destructive git reset/checkout"
  integration_points:
    - "GitHub Actions main"
    - "Mac Studio checkout"
    - "deployed API/web services"
  handoff:
    - "final Stage 09 evidence"
    - "remaining production blockers if any"

skill_routing:
  - skill: prompt-manager
    use_when: "following this prompt pack verification task"
    timing: "startup and final report"
    reason: "prompt-pack discipline"
  - skill: backend-quality-gates
    use_when: "running focused and final gates"
    timing: "verification"
    reason: "backend/web tests"
  - skill: backend-performance-evidence
    use_when: "checking result endpoint bounds and payload/perf risk"
    timing: "verification"
    reason: "result routes can be heavy"
  - skill: browser-qa-evidence
    use_when: "checking `/backtests` in a real browser"
    timing: "verification"
    reason: "browser-visible Stage 09"
  - skill: playwright
    use_when: "capturing final browser evidence"
    timing: "browser QA"
    reason: "Playwright evidence required"
  - skill: publish-ci-deploy
    use_when: "all local verification passes"
    timing: "delivery"
    reason: "direct-main publish/deploy/smoke"

target_envs:
  - local-dev
  - browser
  - github-actions
  - mac-studio

required_literals:
  - "/backtests"
  - "/api/backtests/jobs/{job_id}/summary"
  - "/api/backtests/jobs/{job_id}/variants/{variant_key}/equity"
  - "/api/backtests/jobs/{job_id}/variants/{variant_key}/drawdown"
  - "/api/backtests/jobs/{job_id}/variants/{variant_key}/monthly-stats"
  - "/api/backtests/jobs/{job_id}/variants/{variant_key}/symbol-stats"
  - "/api/backtests/jobs/{job_id}/variants/{variant_key}/trades.csv"
  - "Mac Studio"

non_goals:
  - "Do not implement new feature scope unless fixing a failure introduced by Stage 09."
  - "Do not force-push or discard unrelated local changes."
  - "Do not claim deploy success without production smoke evidence."

quality_gates:
  - cmd: "uv run pytest -q tests/unit/apps/api/test_backtests_routes.py tests/unit/apps/api/test_ui_backtests_routes.py tests/unit/apps/web/test_app_routes.py"
    expect: "passes"
  - cmd: "uv run ruff check apps/api apps/web src/trading/contexts/backtest tests/unit/apps/api tests/unit/apps/web"
    expect: "passes"
  - cmd: "uv run pyright"
    expect: "passes or unrelated pre-existing failures classified"
  - cmd: "python -m tools.docs.generate_docs_index --check"
    expect: "passes if docs changed"
  - cmd: "git diff --check"
    expect: "passes"
---

# Task

Perform final Stage 09 verification and delivery for backtest results.

This is not a feature-expansion prompt. It verifies that Stage 09 implementation is complete, fixes only introduced failures, and then runs the direct-main delivery chain through `publish-ci-deploy`.

## Requirements (Must)

- Verify local worktree status before staging or publishing. Preserve unrelated local changes.
- Run focused API/web tests for backtests result routes and `/backtests` route.
- Run ruff, pyright and docs-index checks according to touched scope.
- Verify browser `/backtests` with real Playwright evidence.
- Verify result endpoints with a known job/variant when available. If no suitable completed job exists, classify this as blocker or use the documented controlled smoke fixture.
- Verify materialization pending/degraded state if detailed data is not yet ready.
- Verify no unexpected API-process sync recompute path remains for cache misses.
- Run `publish-ci-deploy` only after local gates and browser evidence pass.
- After publish, monitor GitHub Actions/deploy to green, sync Mac Studio, restart impacted services if needed, and smoke production.

## Requirements (Should)

- Include payload/bounds evidence for series/trades endpoints.
- Include a short endpoint matrix in final report with observed status.

# Context Acquisition Protocol

Read `.codex/AGENTS.md`, Stage 09 plan, parent prompt, and final changed files. Do not re-open broad unrelated UI pages. Inspect Mac Studio/deploy runbooks only when entering delivery.

Reading budget: default `<= 10 files`, plus command output needed for verification.

# Work Plan

1. Check git status and identify Stage 09 intended diff.
2. Run local backend/web gates.
3. Start local app if needed and run Playwright QA for `/backtests`.
4. Smoke result endpoints locally with a known job/variant or documented fixture.
5. Fix only introduced failures.
6. Re-run gates after fixes.
7. Use `publish-ci-deploy` direct-main delivery.
8. Report final evidence and residual risks.

# Acceptance Criteria

- All required local gates pass or non-blocking failures are classified with evidence.
- Browser evidence exists for `/backtests`.
- Result endpoint smoke is completed or has a concrete documented blocker.
- GitHub Actions/deploy are monitored after push.
- Mac Studio checkout is synchronized.
- Impacted runtime services are restarted/reloaded when required.
- Production smoke confirms deployed `/backtests` and result API behavior.

# Gate Failure Classification

Classify every failure as:

- `introduced`
- `required-path pre-existing`
- `unrelated pre-existing`
- `environmental`
- `flaky`

Do not publish with unresolved `introduced` failures or missing required browser/performance evidence.

# publish-ci-deploy Direct-Main Delivery Contract

When all local criteria pass, run `publish-ci-deploy` in direct-main mode. No branch, no PR. Push directly to `origin/main`, monitor CI/deploy, sync Mac Studio with `git pull --ff-only`, restart impacted services if needed, and smoke production.

Do not report successful publish/deploy while push, CI/deploy monitoring, Mac Studio pull, service restart/reload, or smoke verification remains pending.

# Final Output: Report Format

Report in Russian with these exact sections:

- `Intent`
- `Scope`
- `Endpoint matrix`
- `Tests`
- `Performance`
- `Browser evidence`
- `Production evidence`
- `Contract impact`
- `Risks`
- `Handoff`
- `Publish/deploy`
