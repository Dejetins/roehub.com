---
prompt_name: "Backtest Futures Funding v1 Stage 07 - Futures-only Short Policy API And CJM"
repo: "roehub.com"
branch: "codex/backtest-futures-funding-v1-stage-07"
scope: "API, scenario matrix and browser CJM for short-like directions requiring futures"
language: "en"
context_sources:
  - ".codex/AGENTS.md"
  - "docs/architecture/backtest/backtest-futures-funding-and-short-direction-policy-v1.md"
  - "docs/architecture/backtest/backtest-futures-funding-and-short-direction-policy-v1-stage-reports/backtest-futures-funding-and-short-direction-policy-v1-stage-ledger.md"
hard_requirements:
  - "Record `User required before start: nothing` before edits."
  - "Confirm previous required stage is accepted in the ledger before implementation edits."
  - "Previous-stage ledger gate: confirm Stage 06 is accepted in the stage execution ledger before implementation; if not accepted, stop and record Stage 07 as blocked unless the user explicitly supersedes the gate in the current turn."
  - "Reject spot short-like backtest and strategy launch paths with short_direction_requires_futures_market."
  - "Existing spot short-like jobs remain readable but launch-blocked."
  - "Backtest UI must display gross total_return_pct and net total_return_pct_net_of_funding for funding-enabled futures jobs."
  - "Funding degraded warnings must be visible in results and selected variant detail when present."
  - "Browser policy must consume or mirror server-provided compatibility; do not invent a divergent client rule."
  - "Run real browser QA with console and network checks."
task_toggles:
  implementation: true
  docs_only: false
  browser_qa: true
skill_routing:
  - "contract-impact-analysis"
  - "backend-quality-gates"
  - "browser-qa-evidence"
target_envs:
  - "local API/browser for development smoke"
  - "Mac Studio target-host API/web/browser runtime via ssh macstudio for acceptance evidence"
required_literals:
  - "short_direction_requires_futures_market"
  - "total_return_pct_net_of_funding"
  - "spot"
  - "futures"
non_goals:
  - "No funding ingestion changes."
  - "No scoring changes."
final_report_format:
  - "Scope"
  - "File manifest: created/modified/deleted"
  - "Policy contract"
  - "CJM changes"
  - "Validation"
  - "Browser evidence"
  - "Residual risks"
quality_gates:
  - "node --check apps/web/dist/js/pages/backtests.js"
  - "uv run ruff check apps src/trading/contexts/strategy src/trading/contexts/backtest tests"
  - "uv run pyright apps src/trading/contexts/strategy src/trading/contexts/backtest tests"
  - "uv run pytest -q tests/unit/apps tests/unit/contexts/strategy tests/unit/contexts/backtest"
  - "python -m tools.docs.generate_docs_index --check"
validation_strategy:
  - "API route tests for preflight/create/launch rejection."
  - "Scenario matrix tests for paper and testnet spot short-like rejection."
  - "Browser QA screenshot and console/network checks for direction-market switching, gross/net return display, funding degraded warning and launch-blocked old job path on Mac Studio target runtime."
stage_execution_ledger: "docs/architecture/backtest/backtest-futures-funding-and-short-direction-policy-v1-stage-reports/backtest-futures-funding-and-short-direction-policy-v1-stage-ledger.md"
expected_primary_touches:
  - "docs/architecture/backtest/backtest-futures-funding-and-short-direction-policy-v1-stage-reports/backtest-futures-funding-and-short-direction-policy-v1-stage-ledger.md"
  - "docs/architecture/backtest/backtest-futures-funding-and-short-direction-policy-v1-stage-reports/07-futures-only-short-policy-api-and-cjm.md"
  - "apps/api/routes/backtests.py"
  - "apps/api/routes/strategies.py"
  - "src/trading/contexts/strategy/application/use_cases/scenario_matrix.py"
  - "src/trading/contexts/strategy/application/use_cases/create_strategy_from_backtest_variant.py"
  - "apps/web/templates/pages/backtests.html"
  - "apps/web/dist/js/pages/backtests.js"
possible_secondary_touches:
  - "apps/web/locales/"
  - "tests/unit/apps/"
  - "tests/unit/contexts/strategy/"
  - "tests/unit/contexts/backtest/"
  - "docs/architecture/README.md"
safety_notes:
  - "Do not allow paper spot short as a hidden exception."
  - "Do not break read-only viewing of old jobs."
  - "For acceptance browser/runtime proof, use Mac Studio target runtime; local browser smoke is development evidence only unless explicitly scoped by user."
---

# Task

Enforce the futures-only short policy across API, scenario matrix and browser CJM.

## Stage Gate

Previous-stage ledger gate: before implementation edits, read the stage
execution ledger and verify Stage `06` is accepted. If Stage `06` is not
accepted, do not implement Stage `07`; update the Stage `07` report/ledger as
blocked unless the user explicitly supersedes this gate in the current turn.

## Context / Current State

Current strategy launch validation blocks only `testnet + spot + short-like`. Browser default state can be `spot + long_short_reversal`. Stage `06` exposes funding metrics through result APIs. The target policy rejects all new spot short-like paths, guides users to rerun as futures, and shows funding-adjusted return metrics clearly in the backtest UI.

## Requirements (Must)

- Record `User required before start: nothing` in the stage report before edits.
- Previous-stage gate: Stage `06` must be `accepted` in the stage ledger before Stage `07` implementation starts; otherwise stop and mark this stage blocked unless the user explicitly supersedes Stage `06`.
- Reject `spot + short` and `spot + long_short_reversal` in backtest preflight/create paths.
- Reject strategy launch for any short-like direction unless `market_type=futures`.
- Update scenario matrix so paper spot short-like is rejected, not paper-only.
- Keep old spot short-like jobs readable.
- Add UI behavior: selecting short-like direction switches to futures or blocks spot with server-provided reason.
- Add results UI behavior: show gross `total_return_pct` and net `total_return_pct_net_of_funding` side by side for funding-enabled futures jobs.
- Add selected variant/detail UI behavior: show funding included/degraded status and warning codes when provided by API.
- Add launch modal behavior: old spot short-like job shows rerun-as-futures CTA and cannot launch.
- Run Mac Studio target-runtime browser QA with screenshot plus console/network checks.

## Requirements (Should)

- Use reason code `short_direction_requires_futures_market` consistently.
- Avoid duplicating policy text in several JS branches when a shared helper can be used.
- Keep button labels and warnings concise.

## Requirements (Nice-to-have)

- Add a small UI fixture for degraded funding, gross/net return and launch-blocked state in the same browser flow.

# Context acquisition protocol

Read API validators, scenario matrix, browser template, JS and locale files before editing. Check existing route tests that assert old reason codes.

# Reading manifest

- `apps/api/routes/backtests.py`
- `apps/api/routes/strategies.py`
- `src/trading/contexts/strategy/application/use_cases/scenario_matrix.py`
- `src/trading/contexts/strategy/application/use_cases/create_strategy_from_backtest_variant.py`
- `apps/web/templates/pages/backtests.html`
- `apps/web/dist/js/pages/backtests.js`
- `tests/unit/apps/api/test_strategies_routes.py`
- `tests/unit/contexts/strategy/application/test_strategy_use_cases.py`

# Work plan (agent should follow)

1. Confirm Stage `03` accepted and runtime defaults expose compatibility.
2. Create report and narrowed file manifest.
3. Update API validation and tests.
4. Update scenario matrix and launch use case tests.
5. Update web template/JS/locales.
6. Add gross/net funding return UI tests or route/template assertions.
7. Run focused gates and Mac Studio target-runtime browser QA.
8. Update report and ledger.

# Acceptance criteria (Definition of Done)

- New spot short-like backtest and launch attempts fail with `short_direction_requires_futures_market`.
- Futures short-like paths remain allowed when other readiness checks pass.
- Old spot short-like jobs remain viewable.
- Browser form no longer defaults to an invalid spot long-short combination.
- Backtest results show `total_return_pct` and `total_return_pct_net_of_funding` without replacing or hiding gross return.
- Funding degraded/readiness warnings are visible where result and selected-variant detail data is shown.
- Mac Studio target-runtime browser QA evidence is recorded.

# Implementation constraints

- Do not make client-side validation the only enforcement.
- Do not silently mutate existing job records.

# Files to indicate (expected touched areas)

Use the frontmatter file list as the starting manifest and narrow it before edits.

# Non-goals

- No funding PnL or ingestion changes.

# Quality gates (must run and pass)

```bash
node --check apps/web/dist/js/pages/backtests.js
uv run ruff check apps src/trading/contexts/strategy src/trading/contexts/backtest tests
uv run pyright apps src/trading/contexts/strategy src/trading/contexts/backtest tests
uv run pytest -q tests/unit/apps tests/unit/contexts/strategy tests/unit/contexts/backtest
python -m tools.docs.generate_docs_index --check
```

# Final output: report format (strict)

- Scope
- File manifest: created/modified/deleted
- Policy contract
- CJM changes
- Validation
- Browser evidence
- Cold-head review receipt
- Residual risks
