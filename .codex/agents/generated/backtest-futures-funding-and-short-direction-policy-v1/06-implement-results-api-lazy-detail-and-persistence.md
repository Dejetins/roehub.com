---
prompt_name: "Backtest Futures Funding v1 Stage 06 - Results API Lazy Detail And Persistence"
repo: "roehub.com"
branch: "codex/backtest-futures-funding-v1-stage-06"
scope: "Expose funding metrics through persisted top rows, API read models and lazy detail"
language: "en"
context_sources:
  - ".codex/AGENTS.md"
  - "docs/architecture/backtest/backtest-futures-funding-and-short-direction-policy-v1.md"
  - "docs/architecture/backtest/backtest-futures-funding-and-short-direction-policy-v1-stage-reports/backtest-futures-funding-and-short-direction-policy-v1-stage-ledger.md"
hard_requirements:
  - "Record `User required before start: nothing` before edits."
  - "Top rows remain summary-only."
  - "Lazy cache identity must account for funding_manifest_hash."
  - "API additions must be backward-compatible for existing jobs."
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
  - "funding_manifest_hash"
  - "funding_events"
  - "total_return_pct_net_of_funding"
non_goals:
  - "No browser UI changes."
  - "No launch policy change."
final_report_format:
  - "Scope"
  - "Files changed"
  - "API contract"
  - "Persistence/cache contract"
  - "Validation"
  - "Real-boundary evidence"
  - "Residual risks"
quality_gates:
  - "uv run ruff check apps/api src/trading/contexts/backtest tests"
  - "uv run pyright apps/api src/trading/contexts/backtest tests"
  - "uv run pytest -q tests/unit/apps/api tests/unit/contexts/backtest"
  - "python -m tools.docs.generate_docs_index --check"
validation_strategy:
  - "Route smoke for top, variant and lazy-detail endpoints showing funding fields."
  - "Cache-key tests proving funding_manifest_hash changes identity."
  - "Existing no-funding job read tests."
stage_execution_ledger: "docs/architecture/backtest/backtest-futures-funding-and-short-direction-policy-v1-stage-reports/backtest-futures-funding-and-short-direction-policy-v1-stage-ledger.md"
expected_primary_touches:
  - "src/trading/contexts/backtest/application/dto/backtest_jobs.py"
  - "src/trading/contexts/backtest/application/ports/lazy_trades_cache.py"
  - "src/trading/contexts/backtest/application/services/v2/lazy_trades_detail.py"
  - "apps/api/dto/backtests.py"
  - "apps/api/routes/backtests.py"
possible_secondary_touches:
  - "src/trading/contexts/backtest/adapters/outbound/persistence/postgres/backtest_job_repository.py"
  - "tests/unit/apps/api/"
  - "tests/unit/contexts/backtest/"
  - "docs/architecture/backtest/backtest-futures-funding-and-short-direction-policy-v1-stage-reports/06-results-api-lazy-detail-and-persistence.md"
  - "docs/architecture/README.md"
safety_notes:
  - "If a Postgres migration is required for cache/materialization identity, classify and test it explicitly."
  - "Do not backfill or mutate old rows silently."
---

# Task

Expose funding metrics through result APIs, lazy detail and cache/persistence identity.

## Context / Current State

Top rows persist summary JSON. Lazy detail uses cache keys based on job, variant, request, engine params and artifact manifest. Funding must be visible in read models and cache identity without storing full trades in top rows.

## Requirements (Must)

- Record `User required before start: nothing` in the stage report before edits.
- Add funding fields to top/variant/lazy-detail read models.
- Add funding event overlay data where appropriate.
- Include `funding_manifest_hash` in lazy cache identity or prove root artifact hash makes it redundant and still expose the explicit hash.
- Keep existing jobs readable when funding fields are absent.
- Add local route smoke evidence for funding fields.

## Requirements (Should)

- Keep DTO additions additive.
- Make warning codes stable and machine-readable.

## Requirements (Nice-to-have)

- Add a small API fixture snapshot for a degraded funding job.

# Context acquisition protocol

Read current DTOs, lazy cache ports and route code before changing persistence.

# Reading manifest

- `src/trading/contexts/backtest/application/dto/backtest_jobs.py`
- `src/trading/contexts/backtest/application/ports/lazy_trades_cache.py`
- `src/trading/contexts/backtest/application/services/v2/lazy_trades_detail.py`
- `apps/api/dto/backtests.py`
- `apps/api/routes/backtests.py`
- Stage `04` and `05` reports.

# Work plan (agent should follow)

1. Confirm prior funding scoring stages are accepted.
2. Create report and narrowed file manifest.
3. Add read-model fields and cache identity changes.
4. Update API DTO/route tests.
5. Run local route smoke for top/variant/lazy detail.
6. Update report and ledger.

# Acceptance criteria (Definition of Done)

- Funding metrics are visible in API responses.
- Lazy cache identity changes when funding manifest changes.
- Old jobs without funding fields remain readable.
- Top rows remain summary-only.
- Local route smoke evidence is recorded.

# Implementation constraints

- Do not create a second results API.
- Do not persist full trade tape in top rows.

# Files to indicate (expected touched areas)

Use the frontmatter file list as the starting manifest and narrow it before edits.

# Non-goals

- No browser UI.
- No strategy launch policy changes.

# Quality gates (must run and pass)

```bash
uv run ruff check apps/api src/trading/contexts/backtest tests
uv run pyright apps/api src/trading/contexts/backtest tests
uv run pytest -q tests/unit/apps/api tests/unit/contexts/backtest
python -m tools.docs.generate_docs_index --check
```

# Final output: report format (strict)

- Scope
- Files changed
- API contract
- Persistence/cache contract
- Validation
- Real-boundary evidence
- Cold-head review receipt
- Residual risks
