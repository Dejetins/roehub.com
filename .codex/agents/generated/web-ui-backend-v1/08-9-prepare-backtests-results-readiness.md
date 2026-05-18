---
prompt_name: web_ui_backend_v1_08_9_backtests_results_readiness
repo: roehub.com
branch: main
scope: "Этап 8.9: подготовить текущую backtests workstation и prompt 09 к безопасной реализации result state."

language:
  implementation: python_fastapi_jinja_css_js_prompt_sync
  agent_report: ru

context_sources:
  always_read:
    - path: .codex/AGENTS.md
      why: "repo contract, DDD, gates, browser and delivery rules"
    - path: .codex/agents/generated/web-ui-backend-v1/09-implement-backtests-results.md
      why: "target downstream Stage 9 prompt contract to prepare"
    - path: .codex/agents/generated/web-ui-backend-v1/08-implement-backtests-history-configurator.md
      why: "Stage 8 workstation boundary and reference-shaped /backtests contract"
    - path: .codex/agents/generated/web-ui-backend-v1/08-5-implement-backtest-runtime-hardening.md
      why: "Stage 8.5 queued runtime boundary and worker-readiness contract"
  task_entrypoints:
    - path: apps/api/routes/backtests.py
      why: "current public backtest jobs/top/variant/lazy trades/cancel routes"
    - path: apps/api/routes/ui_backtests.py
      why: "current /api/ui/backtests/workstation read-model route"
    - path: apps/api/dto/backtests.py
      why: "current public backtest DTOs before result additions"
    - path: apps/api/dto/ui_backtests.py
      why: "current Stage 8 workstation DTOs"
    - path: src/trading/contexts/backtest/application/use_cases/backtest_jobs.py
      why: "queued create, idempotency, public variant lookup and lazy trades use case"
    - path: src/trading/contexts/backtest/application/use_cases/backtest_job_worker.py
      why: "current background worker claim/execute/finish boundary"
    - path: apps/web/templates/pages/backtests.html
      why: "current /backtests workstation template and selected-job alias markers"
    - path: apps/web/dist/js/pages/backtests.js
      why: "current workstation JS, refresh and create flow before result-state work"
  conditional_bundles:
    tests:
      read_when: "when validating or adding readiness assertions"
      paths:
        - tests/unit/apps/api/test_backtests_routes.py
        - tests/unit/apps/api/test_ui_backtests_routes.py
        - tests/unit/apps/web/test_app_routes.py
        - tests/unit/contexts/backtest/application/use_cases/test_backtest_job_worker_use_case.py
    canonical_reference:
      read_when: "if /backtests layout readiness or selected result placement is disputed"
      paths:
        - /Users/daniildegtyarev/Projects/roehub_web_ui/stategy_backtest.png
    docs:
      read_when: "if runtime or result contracts need narrow documentation sync"
      paths:
        - docs/architecture/apps/web/web-ui-backend-implementation-plan-v1.md
        - docs/architecture/backtest/backtest-service-artifact-runtime-v1.ru.md
  consult_if_needed:
    - path: docs/architecture/backtest/benchmark_iterations/2026-05-02_iteration_9_lazy_trades_detail/benchmark_summary.md
      read_when: "if lazy trades cache/public variant-key behavior is ambiguous"

style_references:
  design_manifest:
    path: docs/architecture/apps/web/web-ui-design-manifest-v1.md
    purpose: "visual source of truth if browser-visible readiness fixes are needed"
  external_reference_root:
    path: /Users/daniildegtyarev/Projects/roehub_web_ui
    purpose: "canonical /backtests reference remains stategy_backtest.png"
  canonical_reference:
    route: /backtests
    path: /Users/daniildegtyarev/Projects/roehub_web_ui/stategy_backtest.png
    fidelity: "readiness work must not drift the existing reference-shaped workstation"
  default_palette: terminal-orange
  theme_variants:
    - terminal-orange
    - graphite
    - matrix-green
    - high-contrast
  invariant_financial_colors: true
  default_locale: en
  secondary_locale: ru
  language_switch_required: true

hard_requirements:
  stage9_not_implemented_here: true
  no_result_endpoints_here: true
  no_chart_helpers_here_unless_readiness_breaks: true
  verify_stage8_workstation_prerequisites: true
  verify_stage85_runtime_prerequisites: true
  update_stage9_prompt_gates_if_stale: true
  preserve_public_variant_key_contract: true
  preserve_summary_only_top_rows: true
  preserve_no_full_trades_initial_payload: true
  preserve_request_hash_identity: true
  browser_visible_changes_require_qa: true

task_toggles:
  implement_prompt09_sync: true
  implement_missing_readiness_tests_only: true
  implement_runtime_code_only_if_readiness_regressed: true
  implement_stage9_result_api: false
  implement_stage9_result_ui: false
  publish_after_success: true

package_contract:
  depends_on:
    - "01-shell-auth-register accepted"
    - "02-design-system-js-core accepted"
  owns:
    - ".codex/agents/generated/web-ui-backend-v1/09-implement-backtests-results.md"
    - "tests/unit/apps/api/test_backtests_routes.py readiness assertions only"
    - "tests/unit/apps/api/test_ui_backtests_routes.py readiness assertions only"
    - "tests/unit/apps/web/test_app_routes.py readiness assertions only"
    - "tests/unit/contexts/backtest/application/use_cases/test_backtest_job_worker_use_case.py readiness assertions only"
    - "apps/api/routes/backtests.py readiness fixes only"
    - "apps/api/routes/ui_backtests.py readiness fixes only"
    - "apps/api/dto/backtests.py readiness fixes only"
    - "apps/api/dto/ui_backtests.py readiness fixes only"
    - "src/trading/contexts/backtest/application/use_cases/backtest_jobs.py readiness fixes only"
    - "src/trading/contexts/backtest/application/use_cases/backtest_job_worker.py readiness fixes only"
    - "apps/web/templates/pages/backtests.html readiness markers only"
    - "apps/web/dist/js/pages/backtests.js readiness fixes only"
  forbidden:
    - "new Stage 9 result endpoints such as /summary, /equity, /drawdown, /monthly, /csv"
    - "paginated trades implementation"
    - "CSV export implementation"
    - "chart helper implementation"
    - "separate result page or backtests_result.html"
    - "backtests workstation layout rewrite"
    - "worker runtime redesign beyond fixing broken readiness regressions"
    - "canonical request hash changes"
    - "full trades in top rows or first-render workstation payload"
  integration_points:
    - "Stage 9 prompt quality gates and current-state assumptions"
    - "public backtest jobs API contract"
    - "Stage 8 workstation /api/ui/backtests/workstation"
    - "Stage 8.5 queued create and worker use case"
    - "public variant_key and storage variant_hash split"
    - "lazy trades POST materialization existing boundary"
  handoff:
    - "Stage 9 can start with explicit green readiness evidence and updated prompt gates"

skill_routing:
  - skill: prompt-manager
    use_when: "syncing Stage 9 prompt gates/current-state notes"
    timing: "before any code/test edits"
    reason: "primary deliverable includes prompt readiness correction"
  - skill: architecture-review
    use_when: "checking drift between current code and Stage 9 prompt assumptions"
    timing: "before deciding whether code edits are needed"
    reason: "this is a readiness/drift pass, not Stage 9 implementation"
  - skill: contract-impact-analysis
    use_when: "touching public API, DTOs, request hash, public variant-key semantics, worker state, or browser-visible defaults"
    timing: "before implementation and final report"
    reason: "readiness fixes must not silently change contracts"
  - skill: backend-quality-gates
    use_when: "running focused API/use-case/web tests, ruff, pyright"
    timing: "during verification"
    reason: "readiness evidence is test-driven"
  - skill: browser-qa-evidence
    use_when: "only if readiness fixes change /backtests visible HTML/CSS/JS behavior"
    timing: "after backend/web tests"
    reason: "browser-visible changes require runtime evidence"
  - skill: publish-ci-deploy
    use_when: "all readiness gates pass and changes are complete"
    timing: "after verification"
    reason: "full Roehub delivery chain after complete success"

target_envs:
  - local-dev
  - github-actions
  - macstudio

required_literals:
  - "/backtests"
  - "/backtests?job_id="
  - "/api/ui/backtests/workstation"
  - "/api/backtests/jobs"
  - "/api/backtests/jobs/{job_id}/top"
  - "/api/backtests/jobs/{job_id}/variants/{variant_key}"
  - "/api/backtests/jobs/{job_id}/variants/{variant_key}/trades"
  - "background_auto"
  - "BacktestJobWorkerUseCase"
  - "execution_trigger"
  - "variant_key"
  - "variant_hash"
  - "summary-only"
  - "retry_after_seconds"
  - "refresh_status"
  - "tests/unit/apps/api/test_ui_backtests_routes.py"

non_goals:
  - "Do not implement Stage 9 result APIs."
  - "Do not implement charts, CSV export, or paginated trades."
  - "Do not redesign /backtests workstation."
  - "Do not introduce assistant or experimental panels."
  - "Do not publish if readiness evidence is missing."

final_report_format:
  - "Intent: что подготовлено и почему это разблокирует Stage 9"
  - "Scope: changed files, readiness-only boundaries, owns/forbidden compliance"
  - "Observed readiness: Stage 8 workstation, Stage 8.5 runtime, public variant-key, lazy trades, and prompt 09 status"
  - "Contract impact: public API, DTO, persisted schema, request hash/cache identity, browser-visible behavior, performance risk"
  - "Tests: exact commands, cwd, results, focused gates, lint/type gates"
  - "Docs/prompts: prompt 09 sync, docs changed, docs index result, or explicit no-docs reason"
  - "Runtime evidence: browser evidence if collected, automated test evidence, inference, assumptions"
  - "Risks: what still belongs to Stage 9, worker/deploy blockers, pre-existing/environmental/flaky failures"
  - "Handoff: exact readiness checklist for the Stage 9 executor"
  - "Publish/deploy: direct-main publish-ci-deploy terminal state or exact reason skipped"

quality_gates:
  - cmd: "uv run pytest -q tests/unit/apps/api/test_backtests_routes.py tests/unit/apps/api/test_ui_backtests_routes.py tests/unit/apps/web/test_app_routes.py tests/unit/contexts/backtest/application/use_cases/test_backtest_job_worker_use_case.py"
    expect: "passes"
  - cmd: "uv run ruff check apps/api apps/web src/trading/contexts/backtest tests/unit/apps/api tests/unit/apps/web tests/unit/contexts/backtest"
    expect: "passes if Python/CSS/JS-adjacent code or tests changed; otherwise report not run with reason"
  - cmd: "uv run pyright"
    expect: "passes if Python code changed; otherwise report not run with reason"
  - cmd: "python -m tools.docs.generate_docs_index --check"
    expect: "passes if docs changed"

expected_primary_touches:
  - ".codex/agents/generated/web-ui-backend-v1/09-implement-backtests-results.md"
  - "tests/unit/apps/api/test_ui_backtests_routes.py"
  - "tests/unit/apps/api/test_backtests_routes.py"
  - "tests/unit/apps/web/test_app_routes.py"
  - "tests/unit/contexts/backtest/application/use_cases/test_backtest_job_worker_use_case.py"

possible_secondary_touches:
  - "apps/api/routes/backtests.py"
  - "apps/api/routes/ui_backtests.py"
  - "apps/api/dto/backtests.py"
  - "apps/api/dto/ui_backtests.py"
  - "src/trading/contexts/backtest/application/use_cases/backtest_jobs.py"
  - "src/trading/contexts/backtest/application/use_cases/backtest_job_worker.py"
  - "apps/web/templates/pages/backtests.html"
  - "apps/web/dist/js/pages/backtests.js"
  - "docs/architecture/apps/web/web-ui-backend-implementation-plan-v1.md"

safety_notes:
  - "If readiness checks already pass, prefer prompt 09 gate/current-state sync only."
  - "If any prerequisite is broken, fix only the smallest readiness regression and stop before Stage 9 implementation."
  - "The Stage 9 executor must start from green readiness evidence, including ui_backtests route tests."
  - "A no-op DatabaseBacktestJobExecutionTrigger is acceptable as a queued-table boundary only if the final report explicitly says production worker loop/deploy evidence remains a Stage 8.5/ops prerequisite."
---

# Task

Prepare the repository and Stage 9 prompt for a safe implementation of `09-implement-backtests-results.md`.

Done means:

- Stage 9 prompt is synchronized with the current code reality and includes `tests/unit/apps/api/test_ui_backtests_routes.py` in its mandatory readiness/quality gates;
- current `/backtests` workstation readiness is pinned by tests;
- current queued create/runtime readiness is pinned by tests;
- current public `variant_key` / storage `variant_hash` split and lazy trades POST boundary are pinned by tests;
- no Stage 9 result endpoints, charts, CSV export, or paginated trades are implemented in this prompt;
- final report gives a precise handoff checklist for the Stage 9 executor.

## Context / Current State

- Stage 8 `/backtests` workstation exists in the current checkout.
- Stage 8.5 queued create path exists in the current checkout: new jobs should use `background_auto`, `execution_trigger.enqueue(...)`, and `BacktestJobWorkerUseCase`.
- Existing result-adjacent API currently includes `GET /backtests/jobs/{job_id}/top`, `GET /backtests/jobs/{job_id}/variants/{variant_key}`, and `POST /backtests/jobs/{job_id}/variants/{variant_key}/trades`.
- Stage 9 still owns new result summary/chart/stat endpoints, paginated trades GET, CSV export, and selected-result UI behavior.

## Requirements (Must)

- Read the current Stage 9 prompt and current backtest code before editing.
- Compare Stage 9 assumptions against current code and tests.
- Update `09-implement-backtests-results.md` only where it is stale or missing readiness gates.
- Ensure focused tests cover:
  - `/api/ui/backtests/workstation` bounded payload without full trades;
  - `/backtests`, `/backtests/new`, `/backtests/{job_id}` workstation shell and selected-job markers;
  - `POST /backtests/jobs` returns queued/background semantics and does not run compute inline;
  - worker use case can claim and finish/fail a queued job;
  - public `variant_key` works and raw `variant_hash` is not accepted as public route key;
  - existing lazy trades POST remains separate from top rows.
- If any of those tests already exist and pass, do not duplicate them.
- If a readiness test is missing, add the narrowest assertion in the existing focused test file.
- If a readiness behavior is broken, fix only that readiness behavior.

## Requirements (Should)

- Keep code changes minimal and avoid touching production files if tests already pin the behavior.
- Keep prompt 09 as the downstream implementation prompt, not a historical report.
- Keep Stage 9 quality gates aligned with the current test suite.

## Requirements (Nice-to-have)

- Add a short readiness note to the Stage 9 prompt `Context / Current State` if the current code has materially advanced since the prompt was generated.

# Context acquisition protocol

Read `.codex/AGENTS.md`, `09-implement-backtests-results.md`, then the task entrypoints. Expand into tests only when deciding whether readiness coverage is missing.

Reading budget: keep pre-implementation reading to the smallest sufficient set; default target `<= 10 files`, `<= ~50k tokens`.
Stop reading when the readiness matrix is known and the needed edits are bounded.
Do not eager-load all conditional bundles.
If `.codex/agents/.context/promt_manager_state.yaml` or a latest executor final report exists, read only completed/open_items/risks/handoff summary before task entrypoints; skip if absent.

# Reading manifest

Use front matter `context_sources`.

# Work plan (agent should follow)

1. Build a readiness matrix from current code and tests:
   - Stage 8 workstation shell;
   - Stage 8.5 queued create and worker boundary;
   - public variant-key split;
   - lazy trades POST boundary;
   - Stage 9 prompt gates.
2. Update `09-implement-backtests-results.md` if it is missing the current readiness gates or contains stale current-state assumptions.
3. Add missing readiness tests only when coverage is absent.
4. Fix only readiness regressions if a covered prerequisite is red.
5. Run focused readiness gates.
6. Run lint/type/docs gates only when touched scope requires them.
7. Use `publish-ci-deploy` only after complete success.

# Acceptance criteria (Definition of Done)

- Stage 9 prompt includes `tests/unit/apps/api/test_ui_backtests_routes.py` in the relevant quality gate.
- Stage 9 prompt still forbids implementing result endpoints/charts/CSV inside the readiness stage.
- Focused readiness tests pass.
- No new Stage 9 endpoint route exists after this task.
- No full trades are added to top rows or first-render workstation payload.
- No canonical request hash behavior changes.
- Final handoff says exactly what Stage 9 still must implement.

# Implementation constraints

## Agent package boundaries

- Treat `package_contract.owns` as the write allow-list for this prompt.
- Do not edit `package_contract.forbidden` areas. If an implementation truly needs one, stop and report the required integration point instead of broadening scope silently.
- Keep shared integration edits small and explicit.
- In final report, state whether the diff stayed inside `owns`; list any integration-point edits separately.

## API endpoint specification checklist

Do not add new result endpoints in this prompt. If a readiness fix unexpectedly requires any browser-visible API change, stop and write the local contract first:

- `method/path`;
- `owner scope`;
- `request DTO`;
- `response DTO`;
- `status codes`;
- `error payload`;
- `pagination`;
- `cache identity`;
- `compatibility`.

## Browser runtime evidence checklist

Browser evidence is only required if this prompt changes browser-visible HTML/CSS/JS behavior. If so, collect and report:

- desktop screenshot around `1440x1000`;
- mobile screenshot around `390x844`;
- snapshot after key state;
- console errors absent;
- failed same-origin network requests absent except expected auth redirects;
- auth/protected route behavior verified;
- theme/financial color invariants unchanged.

## Gate failure classification

- Classify every failing gate as `introduced`, `required-path pre-existing`, `unrelated pre-existing`, `environmental`, or `flaky`.
- Do not run `publish-ci-deploy` with unresolved `introduced` failures.
- If a failure is pre-existing or environmental, include exact command, failure summary, and why it blocks or does not block readiness.

## API / contracts

- Public API contract: expected `none`; only readiness tests/prompt sync should change.
- DTO schema: expected `none`.
- Persisted schema: expected `none`.
- Request hash/cache identity: `none`.
- Browser-visible behavior: expected `none` unless fixing a readiness regression.

# Files to indicate (expected touched areas)

Use front matter touched paths.

# Non-goals

- Stage 9 result endpoints.
- Paginated trades.
- CSV export.
- Equity/drawdown/monthly/symbol chart endpoints.
- Chart helpers.
- Browser result state UX.
- Worker deployment/ops rollout.

# Quality gates (must run and pass)

```bash
uv run pytest -q tests/unit/apps/api/test_backtests_routes.py tests/unit/apps/api/test_ui_backtests_routes.py tests/unit/apps/web/test_app_routes.py tests/unit/contexts/backtest/application/use_cases/test_backtest_job_worker_use_case.py
```

Conditional gates:

```bash
uv run ruff check apps/api apps/web src/trading/contexts/backtest tests/unit/apps/api tests/unit/apps/web tests/unit/contexts/backtest
uv run pyright
python -m tools.docs.generate_docs_index --check
```

Run the conditional gates when code/docs changes make them relevant; otherwise state exactly why they were skipped.

# i18n / language contract

The Web UI v1 is multilingual. Preserve this contract:

- default locale is `en`; secondary locale is `ru`;
- any new user-visible copy introduced by this readiness stage must have both `en` and `ru` strings through the shared locale catalog/helper;
- do not localize routes, `/api/*` paths, DTO fields, enum values, market symbols, strategy ids, `job_id`, `variant_key`, config keys, or metric identifiers;
- rendered pages must keep `<html lang>` and root `data-locale` aligned with the selected locale;
- final report must state i18n impact.

# publish-ci-deploy direct-main delivery contract

When all readiness DoD and gates pass, and `publish_after_success` is true, run `publish-ci-deploy` in direct-main mode. Do not create a delivery branch, draft PR, or PR-based merge path.

A successful terminal state means:

- executor is on an up-to-date `main`, or has stopped with an exact blocker explaining why direct-main publish is unsafe;
- only intended scope is staged and committed;
- mandatory local gates pass before push;
- commit is pushed directly to `origin/main`;
- GitHub Actions/deploy workflow for `main` are monitored to green;
- local checkout is synchronized with `origin/main`;
- Mac Studio checkout is synchronized with `origin/main` using `git pull --ff-only` from the actual repo checkout, normally `/Users/daniildegtyarev/Projects/roehub.com`;
- impacted services are restarted only when touched-path impact requires it;
- post-restart smoke verification is completed if runtime code changed;
- final report names exact commands, paths, commit SHA, CI/deploy status, restarted services, smoke result, or exact blocker.

Do not report successful publish/deploy while any of these remain pending.

# Final output: report format (strict)

Report in Russian with these exact sections:

- `Intent`: что подготовлено и почему это разблокирует Stage 9.
- `Scope`: changed files, readiness-only boundaries, and owns/forbidden compliance.
- `Observed readiness`: Stage 8 workstation, Stage 8.5 runtime, public variant-key, lazy trades, and prompt 09 status.
- `Contract impact`: public API, DTO, persisted schema, request hash/cache identity, browser-visible behavior, performance risk.
- `Tests`: exact commands, cwd, result, focused/lint/type/docs gates.
- `Docs/prompts`: prompt 09 sync, docs changed, docs index result, or explicit no-docs reason.
- `Runtime evidence`: browser evidence if collected, automated test evidence, inference, assumptions.
- `Risks`: what still belongs to Stage 9, worker/deploy blockers, pre-existing/environmental/flaky failures.
- `Handoff`: exact readiness checklist for the Stage 9 executor.
- `Publish/deploy`: whether `publish-ci-deploy` ran, terminal state, or exact reason it was skipped.
