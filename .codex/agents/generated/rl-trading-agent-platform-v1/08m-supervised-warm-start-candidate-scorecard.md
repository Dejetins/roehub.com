---
prompt_name: 08m-supervised-warm-start-candidate-scorecard
repo: roehub.com
branch: main
branch_policy:
  default_branch: main
  separate_branch_allowed: false
  stage_specific_branches_forbidden: true
prompt_pack_execution:
  plan_doc: docs/architecture/ml/rl-trading-agent-platform-v1.md
  prompt_pack_dir: .codex/agents/generated/rl-trading-agent-platform-v1
  stage_ledger: docs/architecture/ml/rl-trading-agent-platform-v1-stage-reports/rl-trading-agent-platform-v1-stage-ledger.md
  mode: manual_sequential
  execution_mode: manual_sequential
  goal_md_policy: "GOAL.md is optional, not required by default"
  goal_driven_mode: "optional only over the same plan_doc/prompt_pack_dir/stage_ledger; no separate GOAL.md required"
  stage_gate: "read ledger before edits; run only when current_stage is 08M and prerequisites match"
  file_manifest_required: true
goal_mode_optional: true
goal_artifact_required: false
proof_boundary:
  label: target_host_readiness_pre_main
  changed_code_production_claim_allowed: false
browser_auth:
  status: "N/A unless this prompt is explicitly expanded into browser-visible UI/auth work"
  smoke_username: smoke_e2e_keycloak
  host_local_password_source: "/Users/daniildegtyarev/.config/roehub/roehub.env key ROEHUB_SMOKE_E2E_PASSWORD"
  redaction_rule: "do not read or print the password unless browser/auth work is explicitly in scope; never write credentials to prompts, docs, logs, traces, reports, screenshots, or ledgers"
change_ownership:
  allowed_files:
    - src/trading/contexts/rl_trading
    - scripts/rl_trading
    - tests/unit/contexts/rl_trading
    - tests/unit/scripts/rl_trading
    - tests/perf_smoke/contexts/rl_trading
    - docs/architecture/ml/rl-trading-agent-platform-v1.md
    - docs/architecture/ml/rl-trading-agent-platform-v1-stage-reports/08m-supervised-warm-start-candidate-scorecard.md
    - docs/architecture/ml/rl-trading-agent-platform-v1-stage-reports/rl-trading-agent-platform-v1-stage-ledger.md
    - docs/architecture/README.md
  forbidden_without_user_approval:
    - exchange execution paths
    - live_execution behavior
    - API/UI behavior
    - branch/worktree/stash/local-folder workflow changes
scope: "Convert accepted 08L supervised warm-start/contextual-bandit research into a bounded final-holdout candidate scorecard."
language:
  implementation: python
  agent_report: ru
context_sources:
  always_read:
    - path: .codex/AGENTS.md
      why: "repo contract"
    - path: docs/architecture/ml/rl-trading-agent-platform-v1.md
      why: "RL plan, reward/action contract and candidate gates"
    - path: docs/architecture/ml/rl-trading-agent-platform-v1-stage-reports/rl-trading-agent-platform-v1-stage-ledger.md
      why: "stage ledger and current stage"
    - path: docs/architecture/ml/rl-trading-agent-platform-v1-stage-reports/08i3-evaluator-action-reward-parity-repair.md
      why: "accepted evaluator/action/reward-reporting repair prerequisite"
    - path: docs/architecture/ml/rl-trading-agent-platform-v1-stage-reports/08i4-post-repair-methodology-recheck.md
      why: "accepted post-repair matrix and row ownership"
    - path: docs/architecture/ml/rl-trading-agent-platform-v1-stage-reports/08j-article-session-extractor-dataset.md
      why: "accepted article-selector dataset"
    - path: docs/architecture/ml/rl-trading-agent-platform-v1-stage-reports/08k-article-demo-profile-training-evaluation.md
      why: "blocked DQN article-demo candidate and strict gate failure"
    - path: docs/architecture/ml/rl-trading-agent-platform-v1-stage-reports/08l-reward-warm-start-research.md
      why: "accepted supervised warm-start/contextual-bandit research handoff"
  task_entrypoints:
    - path: scripts/rl_trading/stage08l_reward_warm_start_research.py
      why: "research artifact builder and accepted metrics"
    - path: scripts/rl_trading/stage08g_cpu_optuna_calibration.py
      why: "strict native final gate and scorecard helpers"
    - path: scripts/rl_trading/stage08f_roehub_native_backtest_evaluation.py
      why: "Roehub-native scorecard and Stage 08J split loading"
    - path: src/trading/contexts/rl_trading/domain
      why: "training/evaluation/reward/action contracts"
hard_requirements:
  update_stage_ledger: true
  stage_report_file_manifest: true
  prompt_path_and_sha256_in_report: true
  no_secret_or_raw_provider_payload_in_artifacts: true
  explicit_delivery_state: true
  real_boundary_evidence_for_non_trivial_stage: true
task_toggles:
  allow_mainnet_submit: false
  allow_exchange_side_effects: false
  allow_browser_runtime_verification: false
  allow_tests_only_acceptance: false
skill_routing:
  - skill: data-analytics-methodology
    use_when: "supervised labels, contextual-bandit proxy, scorecard and baseline interpretation"
    timing: "entire stage"
    reason: "keeps the candidate decision evidence-backed"
  - skill: backend-quality-gates
    use_when: "Python code/tests are changed"
    timing: "verification"
    reason: "owns focused lint/type/test gates"
  - skill: contract-impact-analysis
    use_when: "reporting whether a supervised warm-start scorecard changes reward/action, registry, API or runtime contracts"
    timing: "before final report"
    reason: "prevents silent compatibility drift"
target_envs:
  - "local checkout"
  - "macstudio for bounded candidate scorecard run"
required_literals:
  - "08M"
  - "reward_research_not_contract_replacement"
  - "supervised_oracle_label_warm_start_contextual_bandit"
non_goals:
  - "Do not replace Stage 02C realized-PnL reward in the live/training contract."
  - "Do not register, promote, activate, paper/testnet/live trade, or mainnet submit."
  - "Do not run a broad DQN/Optuna search unless a bounded matrix and stop conditions are recorded first."
quality_gates:
  - cmd: "uv run ruff check src/trading/contexts/rl_trading scripts/rl_trading tests/unit/contexts/rl_trading tests/unit/scripts/rl_trading tests/perf_smoke/contexts/rl_trading"
    expect: "passes for changed Python paths"
  - cmd: "uv run pyright src/trading/contexts/rl_trading scripts/rl_trading tests/unit/contexts/rl_trading tests/unit/scripts/rl_trading tests/perf_smoke/contexts/rl_trading"
    expect: "passes for changed Python paths"
  - cmd: "uv run pytest -q tests/unit/contexts/rl_trading tests/unit/scripts/rl_trading tests/perf_smoke/contexts/rl_trading"
    expect: "focused tests pass"
  - cmd: "uv run python -m tools.docs.generate_docs_index --check"
    expect: "passes after Markdown updates"
validation_strategy:
  depth: research_candidate_scorecard
  e2e_required: true
  proof_boundary: target_host_readiness_pre_main
  evidence_subtype: "Mac Studio non-production training/evaluation artifact under /opt/roehub/state/rl_trading/"
  acceptance_surfaces:
    - "bounded supervised warm-start or contextual-bandit candidate artifact"
    - "untouched final holdout scorecard on accepted Stage 08J backtest split"
    - "same strict baseline/stability/action gate used by Stage 08K"
  evidence_target: docs/architecture/ml/rl-trading-agent-platform-v1-stage-reports/08m-supervised-warm-start-candidate-scorecard.md
stage_execution_ledger:
  path: docs/architecture/ml/rl-trading-agent-platform-v1-stage-reports/rl-trading-agent-platform-v1-stage-ledger.md
  plan_doc: docs/architecture/ml/rl-trading-agent-platform-v1.md
  current_stage: "08M"
  required_update: true
expected_primary_touches:
  - "src/trading/contexts/rl_trading"
  - "scripts/rl_trading"
  - "tests/unit/contexts/rl_trading"
  - "tests/unit/scripts/rl_trading"
  - "tests/perf_smoke/contexts/rl_trading"
  - "docs/architecture/ml/rl-trading-agent-platform-v1-stage-reports/08m-supervised-warm-start-candidate-scorecard.md"
  - "docs/architecture/ml/rl-trading-agent-platform-v1-stage-reports/rl-trading-agent-platform-v1-stage-ledger.md"
possible_secondary_touches:
  - "docs/architecture/ml/rl-trading-agent-platform-v1.md"
  - "docs/architecture/README.md"
safety_notes:
  - "Research artifacts live under /opt/roehub/state/rl_trading/ and are not committed."
  - "A passing proxy is not Stage 09 readiness; only a strict final-holdout scorecard may set stage09_allowed=true."
---

# Task

Implement Stage `08M` supervised warm-start candidate scorecard.

Use accepted Stage `08L` evidence to build a bounded candidate-scorecard path from the
`supervised_oracle_label_warm_start_contextual_bandit` research direction. This stage may
produce an accepted corrective research candidate only if it passes the same strict native
final-holdout gate that blocked `08K`. It must not silently replace the Stage `02C`
reward/action contract, write the model registry, or open paper/testnet/live/mainnet execution.

## Prompt-Pack Execution Anchor

- `plan_doc`: `docs/architecture/ml/rl-trading-agent-platform-v1.md`
- `prompt_pack_dir`: `.codex/agents/generated/rl-trading-agent-platform-v1`
- `stage_ledger`: `docs/architecture/ml/rl-trading-agent-platform-v1-stage-reports/rl-trading-agent-platform-v1-stage-ledger.md`
- `execution_mode`: `manual_sequential`
- `GOAL.md`: optional, not required by default.
- Stage gate: read the ledger before edits; run only when `current_stage=08M`.
- Manifest gate: every created/modified/deleted file and every runtime artifact path must be recorded in the stage report and ledger.

## Requirements (Must)

- Start by stating exactly: `User required before start: nothing unless a listed prerequisite is not accepted or a required credential/dataset/runtime source is unavailable; never ask for secrets in chat`.
- Previous-stage ledger gate: verify `08I3`, `08I4`, `08J`, and `08L` are `accepted`, `08K` is `blocked`, `current_stage=08M`, and `09` is blocked. If not true, write/update `08M` as blocked, update the ledger, and stop.
- Browser/auth anchor: browser QA and authenticated Roehub UI are N/A for `08M`. Do not use username `smoke_e2e_keycloak` and do not read `/Users/daniildegtyarev/.config/roehub/roehub.env` key `ROEHUB_SMOKE_E2E_PASSWORD`; if a browser/auth surface unexpectedly appears, stop and record a scope blocker.
- Compute and record this prompt hash: `shasum -a 256 .codex/agents/generated/rl-trading-agent-platform-v1/08m-supervised-warm-start-candidate-scorecard.md`.
- Verify the accepted `08L` summary artifact before candidate work:
  - `/opt/roehub/state/rl_trading/evaluation_runs/stage08l_reward_warm_start_research_v1/stage08l_reward_warm_start_99a00ffa43c83b9ac553/stage08l_reward_warm_start_research_summary.json`
  - sha256 `5c25cc9d6a99b549f230a506f61a64563c64da61864127ae0c4c30405941b1a1`
  - summary hash `59bdb534baa97bd172266edb4405774ecc12e2005900386ce4d4bae479f28216`
- Preserve Stage `02C` realized-PnL reward as the baseline contract. Dense mark-to-market, realized plus unrealized delta, supervised warm-start, and contextual bandit outputs are research artifacts unless this stage explicitly records a separate accepted contract-change proposal for a later stage.
- Define a bounded candidate matrix before running any heavy work: implementation path, dataset branch, profile, max runtime, metrics, stop conditions, final-holdout gate, and expected artifact paths.
- Use the accepted Stage `08J` article-selector dataset and the `30/10` profile unless the stage records a source-backed reason to block or supersede.
- Keep calibration/training and final holdout isolated. Do not optimize directly on the final backtest split.
- Include technical baselines on the same final surface: hold/no-trade, deterministic random, simple threshold, and the Stage `08K` native candidate scorecard.
- Apply the strict native gate from `08K`: positive final PnL after costs, beats best sanity baseline, sufficient closed trades, no single group dominates, ticker/month stability not obviously broken, and action distribution not pathologically one-sided.
- If the strict native final gate passes, update report and ledger with `stage09_allowed=true`, next stage `09`, and the exact candidate artifact identity. If it does not pass, keep `stage09_allowed=false` and record whether the plan is blocked or whether another explicit corrective prompt is justified.

## Acceptance Criteria

- Stage report records candidate matrix, runtime/resource evidence, dataset/candidate lineage, final holdout scorecards, baselines, stability/action distributions, contract impact, proof boundary, file manifest, and ledger handoff.
- `09` opens only if the strict final holdout gate passes on the accepted evidence surface and the ledger records `stage09_allowed=true`.
- No registry write, activation, paper/testnet/live trading, exchange/provider side effect, browser/auth credential read, or mainnet submit occurs in this stage.

## Final Output

Respond in Russian with result/status, candidate evidence, strict native gate decision, file manifest, quality gates, residual risks, and next-stage handoff.
