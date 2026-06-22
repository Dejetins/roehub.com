---
prompt_name: "Backtest Futures Funding v1 Stage 06 - Results API Lazy Detail And Persistence"
repo: "roehub.com"
branch: "codex/backtest-futures-funding-v1"
scope: "Expose funding metrics through persisted top rows, API read models and lazy detail"
language: "en"
context_sources:
  - ".codex/AGENTS.md"
  - "docs/architecture/backtest/backtest-futures-funding-and-short-direction-policy-v1.md"
  - "docs/architecture/backtest/backtest-futures-funding-and-short-direction-policy-v1-stage-reports/backtest-futures-funding-and-short-direction-policy-v1-stage-ledger.md"
hard_requirements:
  - "Record `User required before start: nothing` before edits."
  - "Use the active prompt-pack branch codex/backtest-futures-funding-v1 for every stage; do not create per-stage git branches. Record iteration state in the stage ledger and stage report."
  - "Confirm previous required stage is accepted in the ledger before implementation edits."
  - "Previous-stage ledger gate: confirm Stage 05 is accepted in the stage execution ledger before implementation; if not accepted, stop and record Stage 06 as blocked unless the user explicitly supersedes the gate in the current turn."
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
  - "local API for development smoke"
  - "Mac Studio target-host API/runtime smoke via ssh macstudio for acceptance evidence"
required_literals:
  - "funding_manifest_hash"
  - "funding_events"
  - "total_return_pct_net_of_funding"
non_goals:
  - "No browser UI changes."
  - "No launch policy change."
final_report_format:
  - "Scope"
  - "File manifest: created/modified/deleted"
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
  - "Route smoke for top, variant and lazy-detail endpoints showing funding fields; acceptance runtime smoke runs on Mac Studio when API runtime proof is required."
  - "Cache-key tests proving funding_manifest_hash changes identity."
  - "Existing no-funding job read tests."
stage_execution_ledger: "docs/architecture/backtest/backtest-futures-funding-and-short-direction-policy-v1-stage-reports/backtest-futures-funding-and-short-direction-policy-v1-stage-ledger.md"
expected_primary_touches:
  - "docs/architecture/backtest/backtest-futures-funding-and-short-direction-policy-v1-stage-reports/backtest-futures-funding-and-short-direction-policy-v1-stage-ledger.md"
  - "docs/architecture/backtest/backtest-futures-funding-and-short-direction-policy-v1-stage-reports/06-results-api-lazy-detail-and-persistence.md"
  - "src/trading/contexts/backtest/application/dto/backtest_jobs.py"
  - "src/trading/contexts/backtest/application/ports/lazy_trades_cache.py"
  - "src/trading/contexts/backtest/application/services/v2/lazy_trades_detail.py"
  - "apps/api/dto/backtests.py"
  - "apps/api/routes/backtests.py"
possible_secondary_touches:
  - "src/trading/contexts/backtest/adapters/outbound/persistence/postgres/backtest_job_repository.py"
  - "tests/unit/apps/api/"
  - "tests/unit/contexts/backtest/"
  - "docs/architecture/README.md"
safety_notes:
  - "If a Postgres migration is required for cache/materialization identity, classify and test it explicitly."
  - "Do not backfill or mutate old rows silently."
  - "For acceptance runtime smoke, 127.0.0.1 means Mac Studio loopback after ssh macstudio, not local Codex-host loopback."
---

# Task

Expose funding metrics through result APIs, lazy detail and cache/persistence identity.

## Stage Gate

Previous-stage ledger gate: before implementation edits, read the stage
execution ledger and verify Stage `05` is accepted. If Stage `05` is not
accepted, do not implement Stage `06`; update the Stage `06` report/ledger as
blocked unless the user explicitly supersedes this gate in the current turn.

## Context / Current State

Top rows persist summary JSON. Lazy detail uses cache keys based on job, variant, request, engine params and artifact manifest. Funding must be visible in read models and cache identity without storing full trades in top rows.

## Requirements (Must)

- Record `User required before start: nothing` in the stage report before edits.
- Previous-stage gate: Stage `05` must be `accepted` in the stage ledger before Stage `06` implementation starts; otherwise stop and mark this stage blocked unless the user explicitly supersedes Stage `05`.
- Add funding fields to top/variant/lazy-detail read models.
- Add funding event overlay data where appropriate.
- Include `funding_manifest_hash` in lazy cache identity or prove root artifact hash makes it redundant and still expose the explicit hash.
- Keep existing jobs readable when funding fields are absent.
- Add route smoke evidence for funding fields; local smoke is development evidence, while Mac Studio target-host smoke is required when runtime proof is in scope.

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
5. Run route smoke for top/variant/lazy detail; use Mac Studio target-host smoke for acceptance/runtime evidence when applicable.
6. Update report and ledger.

# Acceptance criteria (Definition of Done)

- Funding metrics are visible in API responses.
- Lazy cache identity changes when funding manifest changes.
- Old jobs without funding fields remain readable.
- Top rows remain summary-only.
- Route smoke evidence is recorded with the environment clearly labeled; Mac Studio target-host smoke is required for acceptance when runtime proof is in scope.

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
- File manifest: created/modified/deleted
- API contract
- Persistence/cache contract
- Validation
- Real-boundary evidence
- Cold-head review receipt
- Residual risks
