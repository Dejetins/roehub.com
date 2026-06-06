---
prompt_name: backtest_compute_acceleration_stage_00_baseline
repo: roehub.com
branch: main
scope: "Refresh or verify the current heavy backtest benchmark baseline before compute acceleration stages proceed."

language:
  implementation: python
  agent_report: ru

context_sources:
  always_read:
    - path: .codex/AGENTS.md
      why: "repo contract and benchmark discipline"
    - path: docs/architecture/backtest/backtest-compute-acceleration-plan-v1.md
      why: "source plan and gate rules"
    - path: docs/architecture/backtest/backtest-compute-acceleration-v1-stage-ledger.md
      why: "current stage status and handoff"
  task_entrypoints:
    - path: scripts/backtest/run_api_runner_benchmark_parity.py
      why: "canonical API-runner benchmark harness"
      inspect_symbols:
        - main
        - _write_summary
    - path: scripts/backtest/validate_benchmark_accounting.py
      why: "benchmark accounting governance"
      inspect_symbols:
        - main
  conditional_bundles:
    benchmark_policy:
      read_when: "benchmark comparability or accounting rules are unclear"
      paths:
        - docs/architecture/backtest/benchmark_iterations/README.md
        - docs/architecture/backtest/benchmark_iterations/2026-06-03_matrix_bitset_stage_00_current_baseline/benchmark_results.json
  consult_if_needed:
    - path: docs/architecture/backtest/README.md
      read_when: "artifact runtime scope or trusted boundary is unclear"

style_references:
  - .codex/promt_template.md
  - docs/architecture/backtest/benchmark_iterations/README.md

hard_requirements:
  no_code_changes: true
  mac_studio_acceptance_only: true
  no_next_stage_without_ledger_update: true

task_toggles:
  implementation_allowed: false
  benchmark_required: true
  docs_update_allowed: true

skill_routing:
  - skill: backend-performance-evidence
    use_when: "establishing or comparing benchmark evidence"
    timing: before implementation
    reason: "baseline, comparability, and performance claim discipline"
  - skill: backend-quality-gates
    use_when: "local benchmark accounting or docs gates fail"
    timing: during verification
    reason: "focused Python gate triage"

target_envs:
  - local
  - Mac Studio

runtime_env_sources:
  mac_studio_native_env_file: /Users/daniildegtyarev/.config/roehub/roehub.env
  docker_env_file: /etc/roehub/roehub.env
  benchmark_env_file_arg: "--env-file"
  mac_studio_required_runtime_env:
    ROEHUB_ENV: prod
    ROEHUB_BACKTEST_ARTIFACTS_CONFIG: configs/prod/backtest_artifacts.yaml
  mac_studio_artifact_root: /opt/roehub/state/backtest_artifacts/v2
  benchmark_env_fallback_order:
    - "$ROEHUB_ENV_FILE"
    - /Users/daniildegtyarev/.config/roehub/roehub.env
    - /etc/roehub/roehub.env
  source_references:
    - infra/macos/launchd/com.roehub.api.plist
    - infra/macos/launchd/com.roehub.backtest-job-runner.plist
    - infra/docker/.env.example
    - infra/docker/docker-compose.backend.yml
  required_postgres_env:
    - "STRATEGY_PG_DSN or POSTGRES_DSN or IDENTITY_PG_DSN"
    - "or POSTGRES_DB + POSTGRES_USER + POSTGRES_PASSWORD"
  benchmark_report_contract:
    - "Report env file path, runtime key names, and artifact config path only."
    - "Never print DSN, password, token, API key, or secret values."
  secret_reporting_rule: "Report only key/path presence, never DSN or password values."

required_literals:
  - "2026-06-03_matrix_bitset_stage_00_current_baseline"
  - "run_api_runner_benchmark_parity.py"
  - "validate_benchmark_accounting.py"
  - "next_iteration_allowed"

non_goals:
  - "Implement matrix backend code."
  - "Modify current backtest runtime semantics."
  - "Change artifact publisher, manifests, API, DB schema, or request identity."

final_report_format:
  language: ru
  sections:
    - "Stage status"
    - "Benchmark evidence"
    - "Ledger update"
    - "Checks"
    - "Blockers or next step"

quality_gates:
  - cmd: "uv run python scripts/backtest/validate_benchmark_accounting.py --out docs/architecture/backtest/benchmark_iterations/<stage00_dir>/local_accounting_validation.json"
    expect: "passes for the selected Stage 00 evidence directory"
  - cmd: "python -m tools.docs.generate_docs_index --check"
    expect: "docs index is current if docs changed"
  - cmd: "git diff --check"
    expect: "no whitespace errors"

validation_strategy:
  depth: benchmark
  e2e_required: true
  acceptance_surfaces:
    - "Mac Studio API-runner benchmark"
    - "benchmark accounting validation"
  tests_only_allowed_reason: ""
  evidence_target: docs/architecture/backtest/backtest-compute-acceleration-v1-stage-ledger.md

stage_execution_ledger:
  path: docs/architecture/backtest/backtest-compute-acceleration-v1-stage-ledger.md
  plan_doc: docs/architecture/backtest/backtest-compute-acceleration-plan-v1.md
  current_stage: "00"
  required_update: true
  template: .codex/agents/stage_execution_ledger_template.md

expected_primary_touches:
  - docs/architecture/backtest/backtest-compute-acceleration-v1-stage-ledger.md
  - docs/architecture/backtest/benchmark_iterations/<stage00_dir>/

possible_secondary_touches:
  - docs/architecture/backtest/backtest-compute-acceleration-plan-v1.md
  - docs/architecture/README.md

safety_notes:
  - "Do not edit production runtime code in Stage 00."
  - "Do not mark later stages allowed unless comparable baseline evidence exists."
---

# Task

Refresh or verify Stage 00 current heavy benchmark evidence for backtest compute acceleration. This prompt is a no-code benchmark and governance stage.

Done means:

- Stage 00 evidence is present and comparable on Mac Studio.
- The stage ledger records exact benchmark command, evidence path, correctness, memory, and next-stage decision.
- No implementation or runtime behavior has changed.

## Context / Current State

Context ledger from the previous iteration:

- completed:
  - Stage 00 evidence currently exists at `docs/architecture/backtest/benchmark_iterations/2026-06-03_matrix_bitset_stage_00_current_baseline/`.
  - The plan requires benchmark-gated movement with no production-affecting stage until evidence is accepted.
  - Stage 00 was accepted in the ledger at the time of prompt-pack generation.
- open_items:
  - Verify the baseline is still comparable before starting implementation stages.
  - Refresh Stage 00 only if the existing evidence is stale, missing, or not comparable.
- contract_changes:
  - None allowed for this stage.
- touched_paths:
  - Stage ledger and benchmark evidence only.
- risks:
  - Comparing future stages to stale or non-equivalent evidence.
  - Accidentally treating local-only benchmark results as acceptance evidence.
- next_focus:
  - Keep Stage 01 blocked unless Stage 00 remains accepted.

Additional context:

- Acceptance benchmark host is Mac Studio.
- Local checks are developer evidence only, not performance acceptance.

## Requirements (Must)

- Read context using the protocol below and stop early once sufficient.
- Work from branch `main`; stop and report a blocker if the checkout is not on `main` unless the user explicitly approves another branch for this stage.
- After an `accepted` stage, update ledger/evidence/docs, run required gates, stage only scoped files, commit them to `main`, and report commit SHA and scoped paths. Do not push unless explicitly requested.
- For `accepted_for_learning`, commit scoped shadow/telemetry/docs/evidence only when that record is the durable handoff; keep the production-off limitation explicit.
- For `blocked` or `rejected`, do not commit production runtime changes; commit only ledger/evidence/docs documenting the blocker or rejection when needed, and report residual uncommitted changes.
- Implement only the scoped change described in this prompt.
- Preserve all explicitly protected contracts and invariants.
- Update the stage execution ledger after validation and before the final report.
- Do not modify runtime code, API routes, DB migrations, artifact publisher, canonical manifests, or generated sidecars.
- Run or verify Mac Studio API-runner benchmark evidence.
- Record exact commands, evidence path, artifact/request hashes, `service_wall_clock_s`, `exact_scoring`, `tp_sl_exact_scoring`, memory result, parity result, and docs drift status.
- Keep `next_iteration_allowed: true` only if Stage 00 is accepted and comparable.

## Requirements (Should)

- Reuse existing Stage 00 evidence if still valid and comparable.
- Prefer refreshing the baseline when commit, artifact, request, or Mac Studio runtime differences make comparability doubtful.

## Requirements (Nice-to-have)

- Add a short note in the evidence summary explaining why the chosen baseline is comparable.

# Context acquisition protocol

Read only in this order and stop once sufficient:

1. `.codex/AGENTS.md`
2. latest state snapshot, if available
3. latest executor final report, if available
4. task entrypoints
5. only conditional bundles required by benchmark ambiguity or failing checks
6. consult-if-needed references only for blockers, ambiguity, or conflicts

Pre-implementation reading target:

- `<= 8 files`
- `<= ~35k-50k tokens`

Stop reading once baseline evidence status, comparability, touched docs, and acceptance path are clear.

# Reading manifest

Use the front-matter `context_sources` as the canonical reading map. Do not convert it into a broad mandatory reading list.

# Work plan (agent should follow)

Skill routing for this task:

- `backend-performance-evidence`: use before and during benchmark verification; owns comparable baseline discipline.
- `backend-quality-gates`: use only if local accounting/docs gates fail; owns focused gate triage.

1. Verify Stage 00 ledger status and evidence directory.
2. Decide whether existing evidence is still comparable or must be refreshed on Mac Studio.
3. Run the Mac Studio benchmark if needed, then run accounting validation.
4. Update the stage ledger with status, evidence path, command, checks, and next-stage decision.
5. Run docs and whitespace checks.

# Acceptance criteria (Definition of Done)

- Stage 00 is `accepted` with comparable Mac Studio benchmark evidence, or `blocked` with a precise reason.
- Required heavy rows for `none` and `tp_sl_grid` are recorded.
- `next_iteration_allowed` is true only when baseline evidence is accepted.
- No production code or public contract changed.
- The final report names the evidence directory and commands run.

# Implementation constraints

## Determinism & ordering

- Preserve deterministic evidence naming and ledger ordering.

## API / contracts

- Public API, persistence, request hash, variant identity, artifact manifest, and browser-visible behavior remain unchanged.

## Documentation

- Update only Stage 00 ledger/evidence docs.
- Run docs-index check if Markdown docs change.

## Tests

- No product tests are required unless benchmark accounting code changes, which should not happen in this stage.

## Validation depth

- Benchmark acceptance requires Mac Studio evidence.
- Tests-only acceptance is not allowed.

# Files to indicate (expected touched areas)

Primary touches:

- `docs/architecture/backtest/backtest-compute-acceleration-v1-stage-ledger.md`
- `docs/architecture/backtest/benchmark_iterations/<stage00_dir>/`

Possible secondary touches:

- `docs/architecture/backtest/backtest-compute-acceleration-plan-v1.md`
- `docs/architecture/README.md`

# Non-goals

- Implementing acceleration code.
- Changing benchmark thresholds.
- Changing artifact publisher or canonical artifact layout.

# Quality gates (must run and pass)

- `uv run python scripts/backtest/validate_benchmark_accounting.py --out docs/architecture/backtest/benchmark_iterations/<stage00_dir>/local_accounting_validation.json`
- `python -m tools.docs.generate_docs_index --check`
- `git diff --check`

# Final output: report format (strict)

Your final message MUST be in Russian and follow exactly:

1) **Stage status**

2) **Benchmark evidence**

3) **Ledger update**

4) **Checks**

5) **Blockers or next step**
