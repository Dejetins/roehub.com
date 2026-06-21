---
prompt_name: "Backtest Futures Funding v1 Stage 02 - Funding Artifact Family And Coverage"
repo: "roehub.com"
branch: "codex/backtest-futures-funding-v1-stage-02"
scope: "Funding artifact publishing, loading, manifest hashing and coverage readiness"
language: "en"
context_sources:
  - ".codex/AGENTS.md"
  - "docs/architecture/backtest/backtest-futures-funding-and-short-direction-policy-v1.md"
  - "docs/architecture/backtest/backtest-futures-funding-and-short-direction-policy-v1-stage-reports/backtest-futures-funding-and-short-direction-policy-v1-stage-ledger.md"
  - "docs/architecture/backtest/backtest-service-artifact-runtime-v1.ru.md"
hard_requirements:
  - "Record `User required before start: nothing` before edits."
  - "Confirm previous required stage is accepted in the ledger before implementation edits."
  - "Previous-stage ledger gate: confirm Stage 01 is accepted in the stage execution ledger before implementation; if not accepted, stop and record Stage 02 as blocked unless the user explicitly scopes this stage to offline fixtures or supersedes the gate in the current turn."
  - "Funding artifacts must become part of manifest identity for futures jobs."
  - "Funding artifacts must read from scheduler-maintained canonical_funding_rates, not from ad-hoc provider calls."
  - "Expose explicit funding_manifest_hash for diagnostics and lazy cache identity."
  - "Partial coverage is degraded_with_warning, not automatic job failure."
task_toggles:
  implementation: true
  docs_only: false
  browser_qa: false
skill_routing:
  - "contract-impact-analysis"
  - "backend-quality-gates"
target_envs:
  - "local"
  - "artifact filesystem"
  - "Mac Studio target host via ssh macstudio for ClickHouse-backed coverage proof when required"
required_literals:
  - "funding_manifest_hash"
  - "coverage_status=degraded"
  - "degraded_with_warning"
non_goals:
  - "No scoring changes."
  - "No UI changes."
final_report_format:
  - "Scope"
  - "File manifest: created/modified/deleted"
  - "Artifact contract"
  - "Coverage contract"
  - "Validation"
  - "Real-boundary evidence"
  - "Residual risks"
quality_gates:
  - "uv run ruff check src/trading/contexts/backtest_artifacts src/trading/contexts/backtest tests"
  - "uv run pyright src/trading/contexts/backtest_artifacts src/trading/contexts/backtest tests"
  - "uv run pytest -q tests/unit/contexts/backtest_artifacts tests/unit/contexts/backtest"
  - "python -m tools.docs.generate_docs_index --check"
validation_strategy:
  - "Filesystem publish/load smoke against a temporary artifact root."
  - "ClickHouse-backed coverage smoke against canonical_funding_rates on Mac Studio when scheduler-maintained funding data is available."
  - "Manifest hash changes when funding arrays change."
stage_execution_ledger: "docs/architecture/backtest/backtest-futures-funding-and-short-direction-policy-v1-stage-reports/backtest-futures-funding-and-short-direction-policy-v1-stage-ledger.md"
expected_primary_touches:
  - "docs/architecture/backtest/backtest-futures-funding-and-short-direction-policy-v1-stage-reports/backtest-futures-funding-and-short-direction-policy-v1-stage-ledger.md"
  - "docs/architecture/backtest/backtest-futures-funding-and-short-direction-policy-v1-stage-reports/02-funding-artifact-family-and-coverage.md"
  - "src/trading/contexts/backtest_artifacts/application/services/v2/contracts.py"
  - "src/trading/contexts/backtest_artifacts/application/services/v2/"
  - "src/trading/contexts/backtest/application/ports/artifact_arrays.py"
  - "src/trading/contexts/backtest/adapters/outbound/artifacts_fs/artifact_array_loader.py"
  - "src/trading/contexts/market_data/application/ports/stores/funding_rate_coverage_reader.py"
possible_secondary_touches:
  - "apps/cli/commands/backtest_artifact_publish.py"
  - "tests/unit/contexts/backtest_artifacts/"
  - "tests/unit/contexts/backtest/"
  - "docs/architecture/README.md"
safety_notes:
  - "Do not remove existing artifact families."
  - "Do not make funding mandatory for spot artifacts."
  - "For runtime ClickHouse proof, 127.0.0.1 means Mac Studio loopback after ssh macstudio, not local Codex-host loopback."
---

# Task

Add funding as a first-class artifact family and coverage source for futures backtests.

## Stage Gate

Previous-stage ledger gate: before implementation edits, read the stage
execution ledger and verify Stage `01` is accepted. If Stage `01` is not
accepted, do not implement Stage `02`; update the Stage `02` report/ledger as
blocked unless the user explicitly scopes this stage to offline fixtures or
supersedes this gate in the current turn.

## Context / Current State

The artifact runtime currently supports prices, signals, signal_features, mappings and hit_times. Funding must be published and loaded under the same root contract, with explicit coverage status and hash identity. Stage `01` owns automatic funding freshness through `market-data-scheduler`; this stage consumes the canonical funding table and must not call exchange providers directly.

## Requirements (Must)

- Record `User required before start: nothing` in the stage report before edits.
- Previous-stage gate: Stage `01` must be `accepted` in the stage ledger before Stage `02` implementation starts; otherwise stop and mark this stage blocked unless the user explicitly supersedes Stage `01` or scopes this stage to offline fixtures.
- Add funding artifact contracts and loader APIs.
- Publish funding arrays from `canonical_funding_rates`; if proving this through ClickHouse, use Mac Studio target-host evidence through `ssh macstudio`.
- Do not call Binance/Bybit REST from artifact publisher or backtest artifact loaders.
- Include funding in root manifest identity for futures jobs.
- Expose explicit `funding_manifest_hash`.
- Add coverage statuses: `ready`, `degraded`, `unavailable`, `not_applicable`.
- Keep spot funding `not_applicable`.
- Prove filesystem artifact publish/load smoke.

## Requirements (Should)

- Include array dtype checks for time, rate, mark price, interval and data quality.
- Make degraded coverage reasons machine-readable.

## Requirements (Nice-to-have)

- Add a small artifact fixture with one full-coverage and one partial-coverage symbol.

# Context acquisition protocol

Read current artifact contracts, publisher command and filesystem loader before adding new abstractions.

# Reading manifest

- `src/trading/contexts/backtest_artifacts/application/services/v2/contracts.py`
- `src/trading/contexts/backtest/application/ports/artifact_arrays.py`
- `src/trading/contexts/backtest/adapters/outbound/artifacts_fs/artifact_array_loader.py`
- `apps/cli/commands/backtest_artifact_publish.py`
- Stage `01` report; if Stage `01` is not accepted, stop unless the user explicitly scopes this stage to offline fixtures only.

# Work plan (agent should follow)

1. Confirm current stage `01` status in the ledger.
2. Create the stage report and narrowed file manifest.
3. Add artifact contract, loader and publisher support.
4. Add coverage reader support and degraded warning codes.
5. Add focused tests and filesystem smoke.
6. Update report and ledger.

# Acceptance criteria (Definition of Done)

- Funding arrays can be published and loaded.
- Manifest hash changes when funding content changes.
- Partial coverage returns degraded readiness metadata.
- Existing price/signal/mapping/hit_times tests still pass.
- Real-boundary artifact filesystem proof is recorded.

# Implementation constraints

- Do not make funding artifacts required for spot.
- Do not break existing artifact root layout.
- Avoid loading full funding arrays unless the selected job/window needs them.

# Files to indicate (expected touched areas)

Use the frontmatter file list as the starting manifest and narrow it before edits.

# Non-goals

- No backtest scoring changes.
- No browser work.

# Quality gates (must run and pass)

```bash
uv run ruff check src/trading/contexts/backtest_artifacts src/trading/contexts/backtest tests
uv run pyright src/trading/contexts/backtest_artifacts src/trading/contexts/backtest tests
uv run pytest -q tests/unit/contexts/backtest_artifacts tests/unit/contexts/backtest
python -m tools.docs.generate_docs_index --check
```

# Final output: report format (strict)

- Scope
- File manifest: created/modified/deleted
- Artifact contract
- Coverage contract
- Validation
- Real-boundary evidence
- Cold-head review receipt
- Residual risks
