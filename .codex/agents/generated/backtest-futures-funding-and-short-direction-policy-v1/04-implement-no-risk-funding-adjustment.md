---
prompt_name: "Backtest Futures Funding v1 Stage 04 - No-risk Funding Adjustment"
repo: "roehub.com"
branch: "codex/backtest-futures-funding-v1-stage-04"
scope: "Net-of-funding metrics for no-risk backtest variants"
language: "en"
context_sources:
  - ".codex/AGENTS.md"
  - "docs/architecture/backtest/backtest-futures-funding-and-short-direction-policy-v1.md"
  - "docs/architecture/backtest/backtest-futures-funding-and-short-direction-policy-v1-stage-reports/backtest-futures-funding-and-short-direction-policy-v1-stage-ledger.md"
  - "docs/architecture/backtest/backtest-service-artifact-runtime-v1.ru.md"
hard_requirements:
  - "Record `User required before start: nothing` before edits."
  - "Confirm previous required stage is accepted in the ledger before implementation edits."
  - "Previous-stage ledger gate: confirm Stage 03 is accepted in the stage execution ledger before implementation; if not accepted, stop and record Stage 04 as blocked unless the user explicitly supersedes the gate in the current turn."
  - "Do not replace gross total_return_pct."
  - "Use total_return_pct_net_of_funding as effective default ranking for funding-enabled futures jobs."
  - "Apply funding after base scoring on a bounded candidate pool."
  - "Persist requested_ranking_metric and effective_ranking_metric."
task_toggles:
  implementation: true
  docs_only: false
  browser_qa: false
skill_routing:
  - "contract-impact-analysis"
  - "backend-quality-gates"
  - "backend-performance-evidence"
target_envs:
  - "local benchmark for development"
  - "Mac Studio target-host benchmark via ssh macstudio for acceptance performance evidence"
required_literals:
  - "total_return_pct_net_of_funding"
  - "funding_adjustment_scope"
  - "funding_adjustment_exact_global_ranking"
non_goals:
  - "No TP/SL funding yet."
  - "No browser UI changes."
final_report_format:
  - "Scope"
  - "File manifest: created/modified/deleted"
  - "Funding formula"
  - "Ranking contract"
  - "Validation"
  - "Performance evidence"
  - "Residual risks"
quality_gates:
  - "uv run ruff check src/trading/contexts/backtest tests"
  - "uv run pyright src/trading/contexts/backtest tests"
  - "uv run pytest -q tests/unit/contexts/backtest"
  - "python -m tools.docs.generate_docs_index --check"
validation_strategy:
  - "Formula tests for long/short positive and negative funding."
  - "Candidate-pool tests proving gross and net metrics are both preserved."
  - "Performance benchmark on artifact-backed runtime input on Mac Studio for acceptance; record baseline before measurement, candidate current measurement, benchmark command, sample size, same environment and comparable results; do not accept tests-only."
stage_execution_ledger: "docs/architecture/backtest/backtest-futures-funding-and-short-direction-policy-v1-stage-reports/backtest-futures-funding-and-short-direction-policy-v1-stage-ledger.md"
expected_primary_touches:
  - "docs/architecture/backtest/backtest-futures-funding-and-short-direction-policy-v1-stage-reports/backtest-futures-funding-and-short-direction-policy-v1-stage-ledger.md"
  - "docs/architecture/backtest/backtest-futures-funding-and-short-direction-policy-v1-stage-reports/04-no-risk-funding-adjustment.md"
  - "src/trading/contexts/backtest/application/services/v2/top_result_assembly.py"
  - "src/trading/contexts/backtest/domain/entities/backtest_job_results.py"
  - "src/trading/contexts/backtest/adapters/outbound/persistence/postgres/backtest_job_repository.py"
possible_secondary_touches:
  - "src/trading/contexts/backtest/application/services/v2/"
  - "tests/unit/contexts/backtest/"
  - "docs/architecture/README.md"
safety_notes:
  - "Do not store full trades in top rows."
  - "Report performance with baseline, command, sample size and environment."
  - "For performance acceptance, run benchmark commands on Mac Studio through ssh macstudio and remote git checkout /Users/daniildegtyarev/Projects/roehub.com."
---

# Task

Implement no-risk net-of-funding summary metrics and effective ranking for futures jobs.

## Stage Gate

Previous-stage ledger gate: before implementation edits, read the stage
execution ledger and verify Stage `03` is accepted. If Stage `03` is not
accepted, do not implement Stage `04`; update the Stage `04` report/ledger as
blocked unless the user explicitly supersedes this gate in the current turn.

## Context / Current State

Top result assembly currently persists summary-only top variants with gross return metrics. Funding must be applied after base scoring to a bounded candidate pool and must not replace gross `total_return_pct`.

## Requirements (Must)

- Record `User required before start: nothing` in the stage report before edits.
- Previous-stage gate: Stage `03` must be `accepted` in the stage ledger before Stage `04` implementation starts; otherwise stop and mark this stage blocked unless the user explicitly supersedes Stage `03`.
- Add funding PnL and net return calculation for no-risk variants.
- Preserve gross `total_return_pct`.
- Add `total_return_pct_net_of_funding`, `funding_return_pct`, `funding_pnl_quote`, `funding_events_count`, `funding_data_quality`, `funding_warning_codes`, `funding_included`, `funding_adjustment_scope`.
- Use bounded candidate pool `max(top_n * 5, top_n + 100)`.
- Persist `funding_adjustment_exact_global_ranking=false`.
- Persist requested and effective ranking metrics.
- Produce Mac Studio performance evidence on artifact-backed runtime input with baseline before measurement and candidate current measurement.

## Requirements (Should)

- Keep the funding formula isolated enough to reuse in Stage `05`.
- Keep warning codes deterministic.

## Requirements (Nice-to-have)

- Add a micro-benchmark for funding merge/scanning.

# Context acquisition protocol

Read top assembly, job result entity, repository serialization and Stage `02` artifact loader outputs before editing.

# Reading manifest

- `src/trading/contexts/backtest/application/services/v2/top_result_assembly.py`
- `src/trading/contexts/backtest/domain/entities/backtest_job_results.py`
- `src/trading/contexts/backtest/adapters/outbound/persistence/postgres/backtest_job_repository.py`
- Stage `02` and `03` reports if available.

# Work plan (agent should follow)

1. Confirm prior stages in the ledger.
2. Create report and narrowed file manifest.
3. Implement reusable funding calculation for no-risk open-position windows.
4. Integrate candidate-pool adjustment and net ranking.
5. Add correctness tests.
6. Run Mac Studio performance evidence and focused gates.
7. Update report and ledger.

# Acceptance criteria (Definition of Done)

- Gross and net metrics are both present.
- Positive funding rate makes longs pay and shorts receive.
- Candidate-pool metadata is persisted.
- Effective ranking for funding futures jobs uses net metric by default.
- Mac Studio performance evidence is recorded with baseline before measurement and candidate current measurement.

# Implementation constraints

- Do not claim exact global net ranking.
- Do not serialize full trade tape in top rows.

# Files to indicate (expected touched areas)

Use the frontmatter file list as the starting manifest and narrow it before edits.

# Non-goals

- No TP/SL funding.
- No web UI work.

# Quality gates (must run and pass)

```bash
uv run ruff check src/trading/contexts/backtest tests
uv run pyright src/trading/contexts/backtest tests
uv run pytest -q tests/unit/contexts/backtest
python -m tools.docs.generate_docs_index --check
```

# Final output: report format (strict)

- Scope
- File manifest: created/modified/deleted
- Funding formula
- Ranking contract
- Validation
- Performance evidence
- Cold-head review receipt
- Residual risks
