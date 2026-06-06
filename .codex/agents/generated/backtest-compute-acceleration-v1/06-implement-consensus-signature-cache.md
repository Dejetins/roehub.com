---
prompt_name: backtest_compute_acceleration_stage_06_signature_cache
repo: roehub.com
branch: main
scope: "Implement exact-safe consensus signature cache for matrix no-risk scoring."

language:
  implementation: python
  agent_report: ru

context_sources:
  always_read:
    - path: .codex/AGENTS.md
      why: "repo contract"
    - path: docs/architecture/backtest/backtest-compute-acceleration-plan-v1.md
      why: "Stage 06 cache and tie rules"
    - path: docs/architecture/backtest/backtest-compute-acceleration-v1-stage-ledger.md
      why: "Stage 05 gate"
  task_entrypoints:
    - path: src/trading/contexts/backtest/application/services/v2/matrix_backend/row_signatures.py
      why: "row signatures and duplicate semantics"
      inspect_symbols:
        - "signature"
    - path: src/trading/contexts/backtest/application/services/v2/matrix_backend/consensus.py
      why: "consensus composition target"
      inspect_symbols:
        - "consensus"
    - path: src/trading/contexts/backtest/application/services/v2/matrix_backend/no_risk_score.py
      why: "cache consumer path"
      inspect_symbols:
        - "score"
    - path: src/trading/contexts/backtest/application/services/v2/top_result_assembly.py
      why: "deterministic top merge"
      inspect_symbols:
        - "assemble"
  conditional_bundles:
    tests:
      read_when: "adding cache and deterministic merge tests"
      paths:
        - tests/unit/contexts/backtest/application/services/v2/test_no_risk_exact_scoring_service.py
        - tests/unit/contexts/backtest/domain/value_objects/test_variant_identity.py
  consult_if_needed:
    - path: docs/architecture/backtest/benchmark_iterations/2026-06-03_matrix_bitset_stage_00_current_baseline/benchmark_results.json
      read_when: "baseline comparison is unclear"

style_references:
  - .codex/promt_template.md

hard_requirements:
  previous_stage_required: "05 accepted"
  exact_safe_cache_only: true
  deterministic_tie_policy_required: true

task_toggles:
  implementation_allowed: true
  benchmark_required: true
  docs_update_allowed: true

skill_routing:
  - skill: contract-impact-analysis
    use_when: "cache keys touch variant identity or request/cache semantics"
    timing: before implementation
    reason: "avoid silent identity changes"
  - skill: backend-performance-evidence
    use_when: "measuring cache hit-rate and speedup"
    timing: during verification
    reason: "performance acceptance discipline"
  - skill: backend-quality-gates
    use_when: "Python gates fail"
    timing: during verification
    reason: "backend gate triage"

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
  - "consensus_signature_count"
  - "cache hit-rate"
  - "deterministic tie policy"
  - "variant_hash"

non_goals:
  - "Approximate dedup or ranking."
  - "Sidecar generation."
  - "TP/SL support."

final_report_format:
  language: ru
  sections:
    - "Stage status"
    - "Implementation"
    - "Cache correctness"
    - "Benchmark"
    - "Checks and next-stage notes"

quality_gates:
  - cmd: "uv run pytest -q tests/unit/contexts/backtest/application/services/v2/test_no_risk_exact_scoring_service.py tests/unit/contexts/backtest/domain/value_objects/test_variant_identity.py"
    expect: "passes or focused equivalent"
  - cmd: "python -m tools.docs.generate_docs_index --check"
    expect: "passes if docs changed"
  - cmd: "git diff --check"
    expect: "passes"

validation_strategy:
  depth: benchmark
  e2e_required: true
  acceptance_surfaces:
    - "focused cache parity tests"
    - "Mac Studio API-runner cache benchmark"
  tests_only_allowed_reason: ""
  evidence_target: docs/architecture/backtest/backtest-compute-acceleration-v1-stage-ledger.md

stage_execution_ledger:
  path: docs/architecture/backtest/backtest-compute-acceleration-v1-stage-ledger.md
  plan_doc: docs/architecture/backtest/backtest-compute-acceleration-plan-v1.md
  current_stage: "06"
  required_update: true
  template: .codex/agents/stage_execution_ledger_template.md

expected_primary_touches:
  - src/trading/contexts/backtest/application/services/v2/matrix_backend/consensus.py
  - src/trading/contexts/backtest/application/services/v2/matrix_backend/topn.py
  - docs/architecture/backtest/backtest-compute-acceleration-v1-stage-ledger.md

possible_secondary_touches:
  - src/trading/contexts/backtest/application/services/v2/matrix_backend/row_signatures.py
  - tests/unit/contexts/backtest/application/services/v2/
  - docs/architecture/backtest/benchmark_iterations/<stage06_dir>/

safety_notes:
  - "Cache keys must be collision-safe or collision-checked."
  - "Do not collapse public variant identity."
---

# Task

Implement Stage 06 consensus signature cache for exact-safe scoring reuse in the matrix no-risk backend.

Done means:

- Cache hit-rate and speedup are measured.
- Public top-N identity and order remain deterministic.
- Cache cannot change scoring, ranking, or variant identity.

## Context / Current State

Context ledger from the previous iteration:

- completed:
  - Stage 05 accepted no-risk reversal and arity 6 heavy rows.
- open_items:
  - Reuse identical consensus/trade tape results safely.
  - Prove deterministic merge/tie policy.
- contract_changes:
  - Internal cache only; no public request/hash changes.
- touched_paths:
  - Consensus, top-N merge, tests, benchmark evidence.
- risks:
  - Cache key collisions.
  - Reordering tied variants.
  - Improving a non-hot sub-timer without service speedup.
- next_focus:
  - Stage 07 sidecar artifacts only after cache stage is accepted or rejected with clear reason.

Additional context:

- If cache does not produce accepted end-to-end speedup, record it as rejected or accepted for learning only.

## Requirements (Must)

- Work from branch `main`; stop and report a blocker if the checkout is not on `main` unless the user explicitly approves another branch for this stage.
- After an `accepted` stage, update ledger/evidence/docs, run required gates, stage only scoped files, commit them to `main`, and report commit SHA and scoped paths. Do not push unless explicitly requested.
- For `accepted_for_learning`, commit scoped shadow/telemetry/docs/evidence only when that record is the durable handoff; keep the production-off limitation explicit.
- For `blocked` or `rejected`, do not commit production runtime changes; commit only ledger/evidence/docs documenting the blocker or rejection when needed, and report residual uncommitted changes.
- Verify Stage 05 accepted.
- Implement only exact-safe cache reuse.
- Use collision-safe keys or validate collisions before reuse.
- Preserve `variant_key`, `variant_hash`, ranking, and top-N shape.
- Record cache hit-rate, timing, memory, service wall, and parity.
- Update deterministic top merge tests if needed.
- Update ledger.

## Requirements (Should)

- Keep cache bounded to child process unless the plan is updated.
- Prefer simple deterministic structures over complex eviction unless measured need exists.

## Requirements (Nice-to-have)

- Report cache effectiveness by arity and direction mode.

# Context acquisition protocol

Read only in this order and stop once sufficient:

1. `.codex/AGENTS.md`
2. latest state snapshot, if available
3. latest executor final report, if available
4. task entrypoints
5. conditional bundles for tests only when needed
6. consult-if-needed references only for blockers

Pre-implementation reading target:

- `<= 8 files`
- `<= ~35k-50k tokens`

Stop reading once cache key, merge policy, tests, and benchmark path are clear.

# Reading manifest

Use the front-matter `context_sources` as the canonical reading map. Do not convert it into a broad mandatory reading list.

# Work plan (agent should follow)

Skill routing for this task:

- `contract-impact-analysis`: use before implementation for identity/cache semantics.
- `backend-performance-evidence`: use during verification for cache hit-rate and speedup.
- `backend-quality-gates`: use if Python gates fail.

1. Verify Stage 05 accepted and identify current matrix consensus path.
2. Design exact-safe signature cache key and collision handling.
3. Implement cache and deterministic merge/tie behavior.
4. Add tests for cache parity, collision handling, and top-N determinism.
5. Run local gates and Mac Studio benchmark.
6. Update ledger with accepted/rejected decision.

# Acceptance criteria (Definition of Done)

- Cache reuse is exact-safe.
- Top-N identity/order does not drift.
- Cache hit-rate and service speedup are recorded.
- Service wall and memory do not regress.
- Stage 07 is allowed only when ledger decision permits it.

# Implementation constraints

## Determinism & ordering

- Stable tie-break must be explicit and tested.

## API / contracts

- No public API, DB, request hash, or persisted identity changes.

## Documentation

- Update ledger and evidence docs.

## Tests

- Add focused cache and deterministic merge tests.

## Validation depth

- Tests plus Mac Studio API-runner benchmark are required.

# Files to indicate (expected touched areas)

Primary touches:

- `src/trading/contexts/backtest/application/services/v2/matrix_backend/consensus.py`
- `src/trading/contexts/backtest/application/services/v2/matrix_backend/topn.py`
- `docs/architecture/backtest/backtest-compute-acceleration-v1-stage-ledger.md`

Possible secondary touches:

- `src/trading/contexts/backtest/application/services/v2/matrix_backend/row_signatures.py`
- `tests/unit/contexts/backtest/application/services/v2/`
- `docs/architecture/backtest/benchmark_iterations/<stage06_dir>/`

# Non-goals

- Approximate cache reuse.
- Sidecar files.
- TP/SL scoring.

# Quality gates (must run and pass)

- `uv run pytest -q tests/unit/contexts/backtest/application/services/v2/test_no_risk_exact_scoring_service.py tests/unit/contexts/backtest/domain/value_objects/test_variant_identity.py`
- `python -m tools.docs.generate_docs_index --check`
- `git diff --check`
- Mac Studio API-runner benchmark with cache hit-rate evidence.

# Final output: report format (strict)

Your final message MUST be in Russian and follow exactly:

1) **Stage status**

2) **Implementation**

3) **Cache correctness**

4) **Benchmark**

5) **Checks and next-stage notes**
