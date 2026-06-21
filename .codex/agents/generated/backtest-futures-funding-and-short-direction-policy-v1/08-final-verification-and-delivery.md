---
prompt_name: "Backtest Futures Funding v1 Stage 08 - Final Verification And Delivery"
repo: "roehub.com"
branch: "codex/backtest-futures-funding-v1-stage-08"
scope: "Full verification, docs closure and delivery readiness for the funding/short policy line"
language: "en"
context_sources:
  - ".codex/AGENTS.md"
  - "docs/architecture/backtest/backtest-futures-funding-and-short-direction-policy-v1.md"
  - "docs/architecture/backtest/backtest-futures-funding-and-short-direction-policy-v1-stage-reports/backtest-futures-funding-and-short-direction-policy-v1-stage-ledger.md"
hard_requirements:
  - "Record `User required before start: nothing` before edits."
  - "Do not mark the plan accepted until all prior stages are accepted or explicitly scoped out by user decision."
  - "If publishing is requested, prove exact staged paths before commit."
  - "If delivery is in scope, acceptance requires main/CI/deploy/Mac Studio/browser-runtime evidence."
  - "Final runtime proof must include funding freshness metrics from market-data-scheduler on port 9202."
task_toggles:
  implementation: false
  docs_only: false
  browser_qa: true
skill_routing:
  - "pre-ship-gate"
  - "backend-quality-gates"
  - "browser-qa-evidence"
  - "contract-impact-analysis"
target_envs:
  - "local"
  - "GitHub if publishing"
  - "Mac Studio if delivery is in scope"
required_literals:
  - "Cold-head review: completed"
  - "User required before start: nothing"
non_goals:
  - "No new feature scope."
  - "No unrelated cleanup."
final_report_format:
  - "Scope"
  - "Files changed"
  - "Stage acceptance summary"
  - "Validation"
  - "Delivery evidence"
  - "Residual risks"
quality_gates:
  - "uv run ruff check ."
  - "uv run pyright"
  - "uv run pytest -q -ra"
  - "python -m tools.docs.generate_docs_index --check"
validation_strategy:
  - "Broad local gates."
  - "Browser/runtime proof for the user-facing backtest flow."
  - "Prometheus proof for scheduler_funding_catchup_* series and funding alert rules."
  - "Pre-ship review of exact branch/diff scope."
  - "CI/deploy/Mac Studio smoke only when delivery is requested/in scope."
stage_execution_ledger: "docs/architecture/backtest/backtest-futures-funding-and-short-direction-policy-v1-stage-reports/backtest-futures-funding-and-short-direction-policy-v1-stage-ledger.md"
expected_primary_touches:
  - "docs/architecture/backtest/backtest-futures-funding-and-short-direction-policy-v1-stage-reports/08-final-verification-and-delivery.md"
  - "docs/architecture/backtest/backtest-futures-funding-and-short-direction-policy-v1-stage-reports/backtest-futures-funding-and-short-direction-policy-v1-stage-ledger.md"
  - "docs/architecture/README.md"
possible_secondary_touches:
  - "No production code unless repairing a verification-only defect with explicit scope."
safety_notes:
  - "Do not include unrelated dirty files in staging."
  - "Do not run git commands in /opt/roehub/app on Mac Studio."
  - "Do not print secrets from env files."
---

# Task

Perform final verification, documentation closure and delivery readiness for the full funding/short policy line.

## Context / Current State

This stage is only valid after stages `00` through `07` have either been accepted or explicitly removed from scope by user decision. It closes the plan and verifies that local code completion matches actual runtime/browser behavior.

## Requirements (Must)

- Record `User required before start: nothing` in the stage report before edits.
- Confirm stage ledger status for every prior stage.
- Refuse to mark accepted if any required prior stage is still planned or blocked.
- Run broad local gates.
- Run browser/runtime proof for the backtest funding and futures-only short CJM.
- Run Prometheus/runtime proof for `scheduler_funding_catchup_*` metrics and funding alert rule assets.
- Run pre-ship gate before any publish.
- If delivery is requested, prove exact staged paths, commit/push scope, CI/deploy status, Mac Studio checkout sync and runtime smoke.
- Update final stage report, ledger and docs index.

## Requirements (Should)

- Include a compact risk burn-down table.
- Include exact API/browser URLs used for smoke, without secrets.

## Requirements (Nice-to-have)

- Archive old temporary branches if created during this line and safe to remove.

# Context acquisition protocol

Read the ledger first. Then read each accepted stage report and only the current code/docs needed to verify drift.

# Reading manifest

- `docs/architecture/backtest/backtest-futures-funding-and-short-direction-policy-v1-stage-reports/backtest-futures-funding-and-short-direction-policy-v1-stage-ledger.md`
- Stage reports `00` through `07`
- `.codex/AGENTS.md`
- Current git diff

# Work plan (agent should follow)

1. Confirm no required stage is missing or blocked.
2. Create final stage report and narrowed verification manifest.
3. Run broad local gates and docs index check.
4. Run browser/runtime proof.
5. Run Prometheus/runtime proof for `market-data-scheduler` funding metrics and alert assets.
6. Run pre-ship review.
7. If publishing is requested, stage only intended files, commit/push, watch CI/deploy and collect Mac Studio proof.
8. Update report and ledger with final evidence.

# Acceptance criteria (Definition of Done)

- All prior stages are accepted or explicitly removed from scope.
- Broad gates pass or failures are classified with evidence.
- Browser/runtime proof covers funding visibility and short-like futures-only policy.
- Runtime proof covers automatic funding freshness metrics from `market-data-scheduler` and Prometheus funding alert assets.
- Docs and ledger are current.
- Delivery evidence is complete when delivery is in scope.

# Implementation constraints

- Do not add new feature scope.
- Do not sweep unrelated dirty files into delivery.
- Use Mac Studio git checkout `/Users/daniildegtyarev/Projects/roehub.com` and runtime `/opt/roehub/app` correctly.

# Files to indicate (expected touched areas)

Use the frontmatter file list as the starting manifest and narrow it before edits.

# Non-goals

- No implementation beyond narrowly scoped verification fixes.

# Quality gates (must run and pass)

```bash
uv run ruff check .
uv run pyright
uv run pytest -q -ra
python -m tools.docs.generate_docs_index --check
```

# Final output: report format (strict)

- Scope
- Files changed
- Stage acceptance summary
- Validation
- Browser/runtime evidence
- Delivery evidence
- Cold-head review receipt
- Residual risks
