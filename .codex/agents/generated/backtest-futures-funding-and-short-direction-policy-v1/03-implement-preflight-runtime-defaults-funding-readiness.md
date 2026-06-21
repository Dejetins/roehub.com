---
prompt_name: "Backtest Futures Funding v1 Stage 03 - Preflight Runtime Defaults Funding Readiness"
repo: "roehub.com"
branch: "codex/backtest-futures-funding-v1-stage-03"
scope: "Backtest request normalization, runtime defaults and preflight funding readiness"
language: "en"
context_sources:
  - ".codex/AGENTS.md"
  - "docs/architecture/backtest/backtest-futures-funding-and-short-direction-policy-v1.md"
  - "docs/architecture/backtest/backtest-futures-funding-and-short-direction-policy-v1-stage-reports/backtest-futures-funding-and-short-direction-policy-v1-stage-ledger.md"
hard_requirements:
  - "Record `User required before start: nothing` before edits."
  - "Add standalone short to runtime direction compatibility only when validation and tests are updated."
  - "Spot short-like requests must be invalid for new create/preflight flows."
  - "Existing persisted jobs remain readable and immutable."
  - "Funding readiness is degraded_with_warning, not a hard create blocker."
task_toggles:
  implementation: true
  docs_only: false
  browser_qa: false
skill_routing:
  - "contract-impact-analysis"
  - "backend-quality-gates"
target_envs:
  - "local API"
required_literals:
  - "short_direction_requires_futures_market"
  - "funding_readiness"
  - "degraded_with_warning"
non_goals:
  - "No funding PnL calculation yet."
  - "No browser UI changes."
final_report_format:
  - "Scope"
  - "Files changed"
  - "Request contract"
  - "Preflight contract"
  - "Validation"
  - "Real-boundary evidence"
  - "Residual risks"
quality_gates:
  - "uv run ruff check src/trading/contexts/backtest apps/api tests"
  - "uv run pyright src/trading/contexts/backtest apps/api tests"
  - "uv run pytest -q tests/unit/contexts/backtest tests/unit/apps/api"
  - "python -m tools.docs.generate_docs_index --check"
validation_strategy:
  - "Route smoke for runtime-defaults and preflight with ready, degraded, unavailable and not_applicable fixtures."
  - "Request hash tests for normalized funding config."
stage_execution_ledger: "docs/architecture/backtest/backtest-futures-funding-and-short-direction-policy-v1-stage-reports/backtest-futures-funding-and-short-direction-policy-v1-stage-ledger.md"
expected_primary_touches:
  - "src/trading/contexts/backtest/application/services/v2/preflight.py"
  - "apps/api/routes/backtests.py"
  - "apps/api/dto/backtests.py"
possible_secondary_touches:
  - "src/trading/contexts/backtest/application/dto/"
  - "tests/unit/contexts/backtest/"
  - "tests/unit/apps/api/"
  - "docs/architecture/backtest/backtest-futures-funding-and-short-direction-policy-v1-stage-reports/03-preflight-runtime-defaults-funding-readiness.md"
  - "docs/architecture/README.md"
safety_notes:
  - "Request hash change applies only to new jobs."
  - "Do not mutate old persisted jobs."
---

# Task

Add funding readiness and direction-market compatibility to backtest runtime defaults and preflight.

## Context / Current State

Current preflight exposes `long_only` and `long_short_reversal`, without funding config. The product policy requires standalone `short`, futures-only short-like directions and funding readiness metadata.

## Requirements (Must)

- Record `User required before start: nothing` in the stage report before edits.
- Add normalized `execution.funding.mode` and `coverage_policy`.
- Add funding readiness to preflight responses.
- Expose server-side `direction_market_compatibility` in runtime defaults.
- Add standalone `short` only with complete validation and tests.
- Reject new `spot + short` and `spot + long_short_reversal` requests with `short_direction_requires_futures_market`.
- Preserve existing job readability.
- Prove local API route smoke.

## Requirements (Should)

- Return warning codes that UI can display without string parsing.
- Distinguish `not_applicable` for spot from `unavailable` for futures missing artifacts.

## Requirements (Nice-to-have)

- Add snapshots for runtime-defaults JSON shape.

# Context acquisition protocol

Read current preflight normalization and routes before editing DTOs.

# Reading manifest

- `src/trading/contexts/backtest/application/services/v2/preflight.py`
- `apps/api/routes/backtests.py`
- `apps/api/dto/backtests.py`
- Stage `02` report if available.

# Work plan (agent should follow)

1. Confirm stage `02` status and available funding coverage interfaces.
2. Create the stage report and narrowed file manifest.
3. Implement request normalization and compatibility validation.
4. Add preflight/readiness response fields.
5. Add API route tests and request hash tests.
6. Run route smoke and update report/ledger.

# Acceptance criteria (Definition of Done)

- Runtime defaults expose compatibility policy.
- Preflight reports funding readiness for all relevant market/direction combinations.
- New spot short-like preflight/create flow fails with the new reason code.
- Existing read paths for persisted jobs are not broken.
- Local route smoke is recorded.

# Implementation constraints

- Avoid duplicating compatibility policy in UI code at this stage.
- Keep response additions backward-compatible.

# Files to indicate (expected touched areas)

Use the frontmatter file list as the starting manifest and narrow it before edits.

# Non-goals

- No result scoring.
- No web UI implementation.

# Quality gates (must run and pass)

```bash
uv run ruff check src/trading/contexts/backtest apps/api tests
uv run pyright src/trading/contexts/backtest apps/api tests
uv run pytest -q tests/unit/contexts/backtest tests/unit/apps/api
python -m tools.docs.generate_docs_index --check
```

# Final output: report format (strict)

- Scope
- Files changed
- Request contract
- Preflight contract
- Validation
- Real-boundary evidence
- Cold-head review receipt
- Residual risks
