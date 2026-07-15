---
prompt_name: 15-isolated-job-and-strategy-runtimes
repo: roehub.com
scope: "Unify isolated execution for backtest, optimize, import, report, ML/RL and custom strategy runtimes with immutable job envelopes and resource controls."
language: {implementation: project_default, agent_report: ru}
prompt_pack_execution: {mode: goal_driven, plan_doc: docs/architecture/platform/roehub-self-hosted-oss-platform-v1.md, prompt_pack_dir: .codex/agents/generated/roehub-self-hosted-oss-platform-v1/, stage_ledger: docs/architecture/platform/roehub-self-hosted-oss-platform-v1-stage-reports/roehub-self-hosted-oss-platform-v1-stage-ledger.md, goal_mode_optional: true, goal_artifact_required: false}
stage: {id: "15", prerequisites: ["09", "10", "12", "14"], previous_stage_gate: "Stages 09, 10, 12 and 14 accepted."}
branch_policy: {default_branch: main, separate_branch_allowed: false, stage_specific_branches_forbidden: true, worktree_allowed: false, stash_allowed: false}
context_sources:
  always_read:
    - {path: .codex/AGENTS.md, why: runtime/performance/side-effect contract}
    - {path: docs/architecture/platform/roehub-self-hosted-oss-platform-v1.md, why: isolated job boundary}
    - {path: docs/architecture/platform/roehub-self-hosted-oss-platform-v1-stage-reports/roehub-self-hosted-oss-platform-v1-stage-ledger.md, why: prerequisites}
  task_entrypoints:
    - {path: apps/worker/backtest_job_runner/, why: current job runner}
    - {path: apps/worker/rl_trading_trainer/, why: training runtime}
    - {path: apps/worker/rl_trading_inference/, why: inference runtime}
    - {path: apps/worker/strategy_live_runner/, why: strategy runtime}
skill_routing:
  - {skill: staged-plan-runner, timing: before actions, reason: ledger gate}
  - {skill: architecture-design, timing: before job/runtime contract, reason: isolation and lifecycle boundaries}
  - {skill: contract-impact-analysis, timing: before job identity changes, reason: attempts, retries, artifacts and external effects}
  - {skill: backend-performance-evidence, timing: verification of compute paths, reason: comparable runtime/resource evidence}
change_ownership:
  parallel_main_expected: true
  owned_change_scope: [src/trading/contexts/backtest/, src/trading/contexts/optimize/, src/trading/contexts/ml/, src/trading/contexts/rl_trading/, src/trading/contexts/strategy/, src/trading/integration/, apps/worker/, apps/scheduler/, migrations/, configs/, tests/, tools/jobs/, docs/architecture/platform/, docs/runbooks/, docs/architecture/README.md]
  foreign_changes_policy: preserve unrelated compute experiments and accepted model artifacts
file_manifest:
  expected_primary_touches: [src/trading/integration/, apps/worker/, apps/scheduler/, migrations/, tools/jobs/, tests/, docs/architecture/platform/roehub-self-hosted-oss-platform-v1-stage-reports/15-isolated-job-and-strategy-runtimes.md, docs/architecture/platform/roehub-self-hosted-oss-platform-v1-stage-reports/roehub-self-hosted-oss-platform-v1-stage-ledger.md]
  possible_secondary_touches: [src/trading/contexts/backtest/, src/trading/contexts/optimize/, src/trading/contexts/ml/, src/trading/contexts/rl_trading/, src/trading/contexts/strategy/, configs/, docs/runbooks/, docs/architecture/README.md]
validation_strategy: {depth: performance, acceptance_surfaces: [real-boundary job container runtime smoke, JobEnvelope, CPU/RAM/PID/time limits, cancel/timeout/attempt semantics, deterministic config snapshot, artifact result, benchmark parity]}
proof_boundary: {label: N/A, exclusions: [native MPS promise, mainnet order submission, production training cost]}
authority: {implementation_write: true, git_publish: false, production_mutation: false, paid_compute: false}
---

# Objective

Run heavy and custom workloads as bounded jobs with durable catalog state and artifact results instead of loading arbitrary code into the control plane.

# Requirements

- Define versioned `JobEnvelope`: job/attempt/org IDs, image digest, plugin/runtime version, immutable config snapshot, input artifact digests, limits, deadline and output manifest.
- Support backtest, optimize, history import, report, artifact transform, ML/RL training/inference and custom strategy runtime capabilities.
- Strategy runtime receives market/config inputs and returns signal/intent only; never exchange keys or direct orders.
- Enforce non-root, read-only root, explicit mounts/egress, CPU/RAM/PID/time limits, cancellation and temporary directory cleanup.
- Durable state is PostgreSQL/artifact store; Redis may coordinate but is not sole truth.
- Retry creates an attempt under the same semantic job and respects idempotency/side-effect rules.

# Validation

Run a real-boundary job container runtime smoke for success, deterministic replay, timeout, cancel, crash, retry, resource exhaustion, restart recovery and artifact publication. Compare representative backtest/compute outputs and benchmarks to the accepted baseline. Prove no exchange secret or Docker socket is visible.

# Stop rules

Block on uncontrolled resources, mutable image tags/config, lost durable state, direct strategy order submit, secret exposure, non-deterministic backtest change or unmeasured verified-hot-path regression. Update ledger after evidence.
