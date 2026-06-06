---
prompt_name: backtest_compute_acceleration_stage_03_runtime_bitset_pack
repo: roehub.com
branch: main
scope: "Implement runtime bitset packing in shadow mode and validate consensus parity."

language:
  implementation: python
  agent_report: ru

context_sources:
  always_read:
    - path: .codex/AGENTS.md
      why: "repo contract"
    - path: docs/architecture/backtest/backtest-compute-acceleration-plan-v1.md
      why: "bitset and fairness rules"
    - path: docs/architecture/backtest/backtest-compute-acceleration-v1-stage-ledger.md
      why: "prior stage gate"
  task_entrypoints:
    - path: src/trading/contexts/backtest/application/services/v2/prepare_pools.py
      why: "source signal matrix and time axis"
      inspect_symbols:
        - "PreparedBacktestPoolsV2"
    - path: src/trading/contexts/backtest/application/services/v2/no_risk_exact.py
      why: "reference consensus semantics"
      inspect_symbols:
        - "score"
    - path: src/trading/contexts/backtest/application/services/v2/benchmark_accounting.py
      why: "pack timing and parity evidence"
      inspect_symbols:
        - "BacktestBenchmarkAccounting"
  conditional_bundles:
    numba:
      read_when: "bitset pack uses Numba kernels"
      paths:
        - src/trading/contexts/backtest/application/services/v2/numba_runtime.py
    tests:
      read_when: "adding bitset parity tests"
      paths:
        - tests/unit/contexts/backtest/application/services/v2/test_no_risk_exact_scoring_service.py
  consult_if_needed:
    - path: src/trading/contexts/backtest/application/dto/no_risk_exact.py
      read_when: "reference DTO shape is unclear"

style_references:
  - .codex/promt_template.md

hard_requirements:
  shadow_only: true
  current_scoring_unchanged: true
  bitset_parity_required: true

task_toggles:
  implementation_allowed: true
  benchmark_required: true
  docs_update_allowed: true

skill_routing:
  - skill: numba
    use_when: "implementing or debugging JIT bitset packing kernels"
    timing: during implementation
    reason: "Numba typing, threading, and memory behavior"
  - skill: backend-performance-evidence
    use_when: "measuring pack cost, memory, and parity"
    timing: during verification
    reason: "hot-path measurement discipline"
  - skill: backend-quality-gates
    use_when: "focused Python checks fail"
    timing: during verification
    reason: "backend gate triage"

target_envs:
  - local
  - Mac Studio

runtime_env_sources:
  mac_studio_native_env_file: /Users/daniildegtyarev/.config/roehub/roehub.env
  docker_env_file: /etc/roehub/roehub.env
  benchmark_env_file_arg: "--env-file"
  mac_studio_artifact_root: /opt/roehub/state/backtest_artifacts/v2
  mac_studio_native:
    env_file: /Users/daniildegtyarev/.config/roehub/roehub.env
    launchd_references:
      - infra/macos/launchd/com.roehub.api.plist
      - infra/macos/launchd/com.roehub.backtest-job-runner.plist
    notes:
      - "The env file is outside the repository and contains the real values."
      - "Do not print DSN or password values; report only key presence."
  docker:
    env_file: /etc/roehub/roehub.env
    template: infra/docker/.env.example
    compose_reference: infra/docker/docker-compose.backend.yml
  benchmark:
    script: scripts/backtest/run_api_runner_benchmark_parity.py
    env_file_arg: "--env-file"
    mac_studio_required_runtime_env:
      ROEHUB_ENV: prod
      ROEHUB_BACKTEST_ARTIFACTS_CONFIG: configs/prod/backtest_artifacts.yaml
    mac_studio_artifact_root: /opt/roehub/state/backtest_artifacts/v2
    fallback_order:
      - "$ROEHUB_ENV_FILE"
      - /Users/daniildegtyarev/.config/roehub/roehub.env
      - /etc/roehub/roehub.env
    required_keys:
      - "STRATEGY_PG_DSN or POSTGRES_DSN or IDENTITY_PG_DSN"
      - "or POSTGRES_DB + POSTGRES_USER + POSTGRES_PASSWORD"
      - "ROEHUB_ENV=prod"
      - "ROEHUB_BACKTEST_ARTIFACTS_CONFIG=configs/prod/backtest_artifacts.yaml"
    benchmark_report_contract:
      - "Report env file path, runtime key names, and artifact config path only."
      - "Never print DSN, password, token, API key, or secret values."

required_literals:
  - "signals_pack_ms"
  - "pos_bits"
  - "neg_bits"
  - "W = ceil(T / 64)"

non_goals:
  - "Use bitsets for production scoring."
  - "Generate sidecar files."
  - "Change top-N or candidate selection."

final_report_format:
  language: ru
  sections:
    - "Stage status"
    - "Implementation"
    - "Parity and performance"
    - "Checks"
    - "Next-stage notes"

quality_gates:
  - cmd: "uv run pytest -q tests/unit/contexts/backtest/application/services/v2/test_no_risk_exact_scoring_service.py"
    expect: "passes or focused equivalent if tests are added elsewhere"
  - cmd: "python -m tools.docs.generate_docs_index --check"
    expect: "passes if docs changed"
  - cmd: "git diff --check"
    expect: "passes"

validation_strategy:
  depth: benchmark
  e2e_required: true
  acceptance_surfaces:
    - "bitset parity tests"
    - "API-runner shadow benchmark"
  tests_only_allowed_reason: ""
  evidence_target: docs/architecture/backtest/backtest-compute-acceleration-v1-stage-ledger.md

stage_execution_ledger:
  path: docs/architecture/backtest/backtest-compute-acceleration-v1-stage-ledger.md
  plan_doc: docs/architecture/backtest/backtest-compute-acceleration-plan-v1.md
  current_stage: "03"
  required_update: true
  template: .codex/agents/stage_execution_ledger_template.md

expected_primary_touches:
  - src/trading/contexts/backtest/application/services/v2/matrix_backend/bitsets.py
  - src/trading/contexts/backtest/application/services/v2/benchmark_accounting.py
  - docs/architecture/backtest/backtest-compute-acceleration-v1-stage-ledger.md

possible_secondary_touches:
  - src/trading/contexts/backtest/application/services/v2/prepare_pools.py
  - tests/unit/contexts/backtest/application/services/v2/

safety_notes:
  - "Shadow bitsets must not feed production top-N."
  - "Padding and word count must be validated before parity claims."
---

# Task

Implement Stage 03 runtime bitset packing in shadow mode. Pack current signal rows into positive and negative `uint64` bitsets and validate sample consensus parity without changing scoring.

Done means:

- Runtime pack timing and memory are recorded.
- Bitset consensus sample parity is proven.
- Current scoring and top-N remain unchanged.

## Context / Current State

Context ledger from the previous iteration:

- completed:
  - Stage 02 measured row/signature potential without pruning.
- open_items:
  - Prove bitset representation matches `+1/0/-1` signal semantics.
  - Measure runtime pack cost before deciding whether sidecar files are useful.
- contract_changes:
  - None beyond additive benchmark telemetry.
- touched_paths:
  - New matrix backend bitset helper, telemetry, tests.
- risks:
  - Incorrect padding can create phantom signals after the last bar.
  - Shadow pack can look fast locally but regress child memory.
- next_focus:
  - Stage 04 no-risk matrix scoring depends on bitset parity.

Additional context:

- Use `W = ceil(T / 64)` and validate padding bits.

## Requirements (Must)

- Work from branch `main`; stop and report a blocker if the checkout is not on `main` unless the user explicitly approves another branch for this stage.
- After an `accepted` stage, update ledger/evidence/docs, run required gates, stage only scoped files, commit them to `main`, and report commit SHA and scoped paths. Do not push unless explicitly requested.
- For `accepted_for_learning`, commit scoped shadow/telemetry/docs/evidence only when that record is the durable handoff; keep the production-off limitation explicit.
- For `blocked` or `rejected`, do not commit production runtime changes; commit only ledger/evidence/docs documenting the blocker or rejection when needed, and report residual uncommitted changes.
- Verify Stage 02 status before implementation.
- Implement runtime bitset pack in a new isolated matrix backend helper.
- Preserve current scoring path and production top-N.
- Prove `+1`, `0`, `-1`, long-only, reversal-relevant masks, padding, and word-count parity.
- Record `signals_pack_ms`, memory peak/cleanup, and sample consensus parity.
- For API-runner benchmark evidence, load the runtime env through
  `--env-file /Users/daniildegtyarev/.config/roehub/roehub.env` on Mac Studio,
  or rely on `ROEHUB_ENV_FILE`; benchmark runtime must include
  `ROEHUB_ENV=prod` and
  `ROEHUB_BACKTEST_ARTIFACTS_CONFIG=configs/prod/backtest_artifacts.yaml`;
  never print secret values.
- Update tests and stage ledger.

## Requirements (Should)

- Keep bitset arrays contiguous and low-allocation.
- Prefer `uint64` representation compatible with later sidecar `.u64.npy`.

## Requirements (Nice-to-have)

- Include small synthetic cases that expose non-multiple-of-64 timeline length.

# Context acquisition protocol

Read only in this order and stop once sufficient:

1. `.codex/AGENTS.md`
2. latest state snapshot, if available
3. latest executor final report, if available
4. task entrypoints
5. conditional bundles for Numba/tests only when needed
6. consult-if-needed references only for blockers

Pre-implementation reading target:

- `<= 8 files`
- `<= ~35k-50k tokens`

Stop reading once signal source, pack helper, parity tests, and benchmark hook are clear.

# Reading manifest

Use the front-matter `context_sources` as the canonical reading map. Do not convert it into a broad mandatory reading list.

# Work plan (agent should follow)

Skill routing for this task:

- `numba`: use during implementation if JIT kernels are introduced or fail.
- `backend-performance-evidence`: use during verification for pack timing/memory.
- `backend-quality-gates`: use if Python gates fail.

1. Check Stage 02 status and current signal matrix shape.
2. Add `matrix_backend/bitsets.py` with runtime pack and validation helpers.
3. Add shadow hook and telemetry without feeding scoring.
4. Add synthetic and current-path parity tests.
5. Run local gates and API-runner shadow benchmark.
6. Update ledger with parity, pack cost, and Stage 04 decision.

# Acceptance criteria (Definition of Done)

- Bitsets reproduce signal masks and consensus samples.
- Padding bits are safe.
- Pack timing and memory are recorded.
- Current result shape/hash does not drift.
- Ledger marks Stage 03 accepted for learning or blocked with reason.

# Implementation constraints

## Determinism & ordering

- Bit order and row order must be documented in code/tests.

## API / contracts

- No public API, DB, request hash, artifact manifest, or top-N contract changes.

## Documentation

- Update ledger and benchmark evidence only unless docs shape changes.

## Tests

- Add focused bitset parity tests.

## Validation depth

- Tests plus API-runner shadow benchmark are required.

# Files to indicate (expected touched areas)

Primary touches:

- `src/trading/contexts/backtest/application/services/v2/matrix_backend/bitsets.py`
- `src/trading/contexts/backtest/application/services/v2/benchmark_accounting.py`
- `docs/architecture/backtest/backtest-compute-acceleration-v1-stage-ledger.md`

Possible secondary touches:

- `src/trading/contexts/backtest/application/services/v2/prepare_pools.py`
- `tests/unit/contexts/backtest/application/services/v2/`

# Non-goals

- Matrix scoring.
- Sidecar generation.
- Publisher changes.

# Quality gates (must run and pass)

- `uv run pytest -q tests/unit/contexts/backtest/application/services/v2/test_no_risk_exact_scoring_service.py`
- `python -m tools.docs.generate_docs_index --check`
- `git diff --check`
- API-runner shadow benchmark with pack timing and no result drift:

```bash
uv run python scripts/backtest/run_api_runner_benchmark_parity.py \
  --env-file /Users/daniildegtyarev/.config/roehub/roehub.env \
  --out-dir docs/architecture/backtest/benchmark_iterations/2026-06-06_matrix_bitset_stage_03_runtime_bitset_pack
```

# Final output: report format (strict)

Your final message MUST be in Russian and follow exactly:

1) **Stage status**

2) **Implementation**

3) **Parity and performance**

4) **Checks**

5) **Next-stage notes**
