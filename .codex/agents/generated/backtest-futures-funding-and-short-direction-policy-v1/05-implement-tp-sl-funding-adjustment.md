---
prompt_name: "Backtest Futures Funding v1 Stage 05 - TP/SL Funding Adjustment"
repo: "roehub.com"
branch: "codex/backtest-futures-funding-v1-stage-05"
scope: "Funding-aware TP/SL variants using exact exit semantics"
language: "en"
context_sources:
  - ".codex/AGENTS.md"
  - "docs/architecture/backtest/backtest-futures-funding-and-short-direction-policy-v1.md"
  - "docs/architecture/backtest/backtest-futures-funding-and-short-direction-policy-v1-stage-reports/backtest-futures-funding-and-short-direction-policy-v1-stage-ledger.md"
  - "docs/architecture/backtest/backtest-service-artifact-runtime-v1.ru.md"
hard_requirements:
  - "Record `User required before start: nothing` before edits."
  - "Reuse exact TP/SL exit semantics; do not fork divergent logic."
  - "Funding event inclusion is entry_time < funding_time <= exit_time."
  - "Preserve gross metrics and add net metrics."
  - "Produce performance evidence on artifact-backed runtime input."
task_toggles:
  implementation: true
  docs_only: false
  browser_qa: false
skill_routing:
  - "root-cause-debugging"
  - "contract-impact-analysis"
  - "backend-quality-gates"
  - "backend-performance-evidence"
target_envs:
  - "local benchmark"
required_literals:
  - "entry_time < funding_time <= exit_time"
  - "total_return_pct_net_of_funding"
  - "same-bar"
non_goals:
  - "No browser UI changes."
  - "No provider ingestion changes."
final_report_format:
  - "Scope"
  - "Files changed"
  - "Exit semantics"
  - "Funding metrics"
  - "Validation"
  - "Performance evidence"
  - "Residual risks"
quality_gates:
  - "uv run ruff check src/trading/contexts/backtest tests"
  - "uv run pyright src/trading/contexts/backtest tests"
  - "uv run pytest -q tests/unit/contexts/backtest"
  - "python -m tools.docs.generate_docs_index --check"
validation_strategy:
  - "Tests for TP/SL exact exit, same-bar precedence and funding timestamp boundary."
  - "Regression test comparing lazy detail exit and top adjustment exit."
  - "Performance benchmark on artifact-backed runtime input; do not accept tests-only."
stage_execution_ledger: "docs/architecture/backtest/backtest-futures-funding-and-short-direction-policy-v1-stage-reports/backtest-futures-funding-and-short-direction-policy-v1-stage-ledger.md"
expected_primary_touches:
  - "src/trading/contexts/backtest/application/services/v2/lazy_trades_detail.py"
  - "src/trading/contexts/backtest/application/services/v2/top_result_assembly.py"
possible_secondary_touches:
  - "src/trading/contexts/backtest/application/services/v2/"
  - "tests/unit/contexts/backtest/"
  - "docs/architecture/backtest/backtest-futures-funding-and-short-direction-policy-v1-stage-reports/05-tp-sl-funding-adjustment.md"
  - "docs/architecture/README.md"
safety_notes:
  - "If exact exit logic cannot be shared safely, stop and report the blocker."
  - "Do not weaken existing TP/SL tests to fit funding."
---

# Task

Extend funding adjustment to TP/SL variants using the exact exit path already trusted by lazy detail.

## Context / Current State

Lazy detail already computes actual TP/SL exits. Funding must use those same actual exits; otherwise top rows and detail rows can disagree.

## Requirements (Must)

- Record `User required before start: nothing` in the stage report before edits.
- Reuse or extract exact exit resolver from current TP/SL detail logic.
- Apply funding events where `entry_time < funding_time <= exit_time`.
- Cover same-bar TP/SL precedence.
- Keep gross and net metrics side by side.
- Produce performance evidence on artifact-backed runtime input.

## Requirements (Should)

- Minimize extra allocations in hot paths.
- Keep no-risk funding behavior from Stage `04` unchanged.

## Requirements (Nice-to-have)

- Add a golden fixture with manually computed TP/SL funding PnL.

# Context acquisition protocol

Read `lazy_trades_detail.py` and top assembly together. If code has drifted, prove the current exit path before refactoring.

# Reading manifest

- `src/trading/contexts/backtest/application/services/v2/lazy_trades_detail.py`
- `src/trading/contexts/backtest/application/services/v2/top_result_assembly.py`
- Stage `04` report.

# Work plan (agent should follow)

1. Confirm Stage `04` accepted.
2. Create report and narrowed file manifest.
3. Localize exact exit semantics with tests before changing behavior.
4. Implement TP/SL funding adjustment.
5. Add regression tests proving top/detail alignment.
6. Run performance evidence and gates.
7. Update report and ledger.

# Acceptance criteria (Definition of Done)

- TP/SL variants include correct net-of-funding metrics.
- Boundary `entry_time < funding_time <= exit_time` is tested.
- Same-bar TP/SL behavior remains unchanged.
- Top and lazy detail use consistent exit times.
- Performance evidence is recorded.

# Implementation constraints

- Do not duplicate exit logic in a way that can drift.
- Do not alter gross TP/SL metrics.

# Files to indicate (expected touched areas)

Use the frontmatter file list as the starting manifest and narrow it before edits.

# Non-goals

- No API/UI exposure work beyond internal data needed for Stage `06`.

# Quality gates (must run and pass)

```bash
uv run ruff check src/trading/contexts/backtest tests
uv run pyright src/trading/contexts/backtest tests
uv run pytest -q tests/unit/contexts/backtest
python -m tools.docs.generate_docs_index --check
```

# Final output: report format (strict)

- Scope
- Files changed
- Exit semantics
- Funding metrics
- Validation
- Performance evidence
- Cold-head review receipt
- Residual risks
