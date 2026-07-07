---
prompt_name: 08o-stage08k-dqn-forensic-decomposition
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
  stage_gate: "read ledger before edits; run only when current_stage is 08O and prerequisites match"
  file_manifest_required: true
goal_mode_optional: true
goal_artifact_required: false
proof_boundary:
  label: target_host_non_production_forensic_pre_main
  changed_code_production_claim_allowed: false
  post_main_production_runtime_proof:
    required_for_changed_code_production_claim: true
    requires:
      - target revision delivered to `main`
      - green CI/GitHub Actions for that revision
      - deploy or verified sync into `/opt/roehub/app`
      - production smoke from the synced runtime tree
    status_for_this_stage: "not collected and out of scope"
  production_proof_requires:
    - target revision on `main`
    - green CI/GitHub Actions for that revision
    - deploy or verified sync into `/opt/roehub/app`
    - production runtime smoke after deploy/sync
  pre_main_not_production_proof:
    - target_host_non_production_forensic_pre_main
    - local checkout artifact analysis
    - read-only Mac Studio artifact checks under `/opt/roehub/state/rl_trading/`
browser_auth:
  status: "N/A; this prompt must not read browser credentials or perform authenticated UI work"
  smoke_username: smoke_e2e_keycloak
  host_local_password_source: "/Users/daniildegtyarev/.config/roehub/roehub.env key ROEHUB_SMOKE_E2E_PASSWORD"
  redaction_rule: "do not read or print the password; browser/auth work is out of scope"
change_ownership:
  allowed_files:
    - scripts/rl_trading
    - tests/unit/scripts/rl_trading
    - docs/architecture/ml/rl-trading-agent-platform-v1.md
    - docs/architecture/ml/rl-trading-agent-platform-v1-stage-reports/08o-stage08k-dqn-forensic-decomposition.md
    - docs/architecture/ml/rl-trading-agent-platform-v1-stage-reports/rl-trading-agent-platform-v1-stage-ledger.md
    - docs/architecture/README.md
  forbidden_without_user_approval:
    - exchange execution paths
    - live_execution behavior
    - API/UI behavior
    - model registry state mutation
    - runtime config activation
    - paper/testnet/live/mainnet execution
    - branch/worktree/stash/local-folder workflow changes
scope: "Deep forensic decomposition of the blocked Stage 08K native DQN candidate, focused on explaining aggregate return, concentration, ticker/regime stability and article-comparison evidence."
language:
  implementation: python
  agent_report: ru
context_sources:
  always_read:
    - path: .codex/AGENTS.md
      why: "repo contract, proof boundary, main-branch and staged prompt-pack rules"
    - path: docs/architecture/ml/rl-trading-agent-platform-v1.md
      why: "RL plan and current research/runtime decision chain"
    - path: docs/architecture/ml/rl-trading-agent-platform-v1-stage-reports/rl-trading-agent-platform-v1-stage-ledger.md
      why: "stage ledger and current stage"
    - path: docs/architecture/ml/rl-trading-agent-platform-v1-stage-reports/08j-article-session-extractor-dataset.md
      why: "accepted article-style dataset lineage for 08K"
    - path: docs/architecture/ml/rl-trading-agent-platform-v1-stage-reports/08k-article-demo-profile-training-evaluation.md
      why: "blocked DQN candidate under review"
    - path: docs/architecture/ml/rl-trading-agent-platform-v1-stage-reports/08m-supervised-warm-start-candidate-scorecard.md
      why: "contains the 08K blocked-reference row; use only as comparison context, not as the main candidate"
    - path: docs/architecture/ml/rl-trading-agent-platform-v1-stage-reports/08n-candidate-quality-reclassification.md
      why: "keeps product/mainnet path closed after 08M quality reclassification"
    - path: docs/architecture/ml/rl-trading-agent-platform-v1-stage-reports/18-rl-soak-incident-drills.md
      why: "confirms runtime path stopped at monitor-only technical soak and does not reopen mainnet"
  task_entrypoints:
    - path: /opt/roehub/state/rl_trading/evaluation_runs/stage08k_article_demo_profile_training_evaluation_v1/dual_branch_runs/stage08k_dual_branch_cpu_76f51186c00ecb54255e/stage08k_dual_branch_cpu_run_summary.json
      why: "08K dual-branch summary; expected file sha256 70dfffd378f33da997b2471422491e9defc28f8b361b9c5b645087288129286d"
    - path: /opt/roehub/state/rl_trading/evaluation_runs/stage08k_article_demo_profile_training_evaluation_v1/stage08k_roehub_native_03fd26aa_cc9179e39ded309544c3/final_holdout_b2adb7da3abc/stage08f_evaluation_manifest.json
      why: "08K native final holdout evaluation manifest; expected sha256 c16a08dffa6f49fd8f24d9a1af6737114d54363b4cca3630a4e83c2661880b07"
    - path: /opt/roehub/state/rl_trading/evaluation_runs/stage08k_article_demo_profile_training_evaluation_v1/stage08k_roehub_native_03fd26aa_cc9179e39ded309544c3/stage08k_optuna_summary.json
      why: "08K native Optuna summary; expected file sha256 8585d4342dab24850cd077e5287de5faab251e848f18eb044f70cc410ebf6cec"
    - path: /opt/roehub/state/rl_trading/datasets/stage08j_article_sessionized_dataset_v1/stage08j_article_sessionized_manifest.json
      why: "accepted 08J article dataset manifest; expected sha256 fd7c614b4cc5085cc24cd054143b6bb188283b9cf423122d436e37769fcd639a"
  conditional_bundles:
    - name: source_article_comparison
      read_when: "internet or cached source artifacts are available and article metric comparison is being computed"
      files_or_urls:
        - https://habr.com/ru/articles/934258/
        - https://github.com/YuriyKolesnikov/rl-trading-binance
      note: "If unavailable, do not block artifact decomposition; record article-comparison as incomplete with reason."
    - name: helper_code
      read_when: "a Python helper is needed to parse 08K artifacts or recompute metrics"
      files_or_urls:
        - scripts/rl_trading/stage08g_dual_branch_cpu_training_evaluation.py
        - scripts/rl_trading/stage08g_cpu_optuna_calibration.py
        - scripts/rl_trading/stage08f_roehub_native_backtest_evaluation.py
        - src/trading/contexts/rl_trading/domain/roehub_native_evaluation.py
        - tests/unit/scripts/rl_trading
  consult_if_needed:
    - path: docs/architecture/ml/rl-trading-agent-platform-v1-stage-reports/08i4-post-repair-methodology-recheck.md
      read_when: "checking which methodology rows were assigned to 08K"
    - path: docs/architecture/ml/rl-trading-agent-platform-v1-stage-reports/08l-reward-warm-start-research.md
      read_when: "separating DQN findings from later supervised/contextual-bandit fallback"
    - path: docs/architecture/README.md
      read_when: "Markdown docs are added or changed and docs index must be verified"
hard_requirements:
  update_stage_ledger: true
  stage_report_file_manifest: true
  prompt_path_and_sha256_in_report: true
  no_secret_or_raw_provider_payload_in_artifacts: true
  explicit_delivery_state: true
  no_training_or_tuning: true
  no_runtime_activation: true
  no_mainnet_or_product_readiness_claim: true
  real_boundary_evidence_for_non_trivial_stage: true
task_toggles:
  allow_training: false
  allow_optuna: false
  allow_model_registry_mutation: false
  allow_runtime_config_activation: false
  allow_mainnet_submit: false
  allow_exchange_side_effects: false
  allow_browser_runtime_verification: false
  allow_tests_only_acceptance: false
skill_routing:
  - skill: staged-plan-runner
    use_when: "executing this stage from the existing plan_doc/prompt_pack_dir/stage_ledger"
    timing: "at stage start"
    reason: "derive current stage and prerequisites from the ledger, not chat memory"
  - skill: data-analytics-methodology
    use_when: "decomposing PnL, stability, ticker/regime concentration, scorecard metrics and source-vs-artifact comparisons"
    timing: "during analysis"
    reason: "prevents weak aggregate-return conclusions and keeps metric interpretation evidence-backed"
  - skill: backend-quality-gates
    use_when: "Python helper code/tests are changed"
    timing: "verification"
    reason: "owns focused ruff, pyright and pytest gates"
  - skill: contract-impact-analysis
    use_when: "classifying whether 08K can move to restricted research, calibration, repair or remains blocked"
    timing: "before final report"
    reason: "keeps registry/runtime/UI/mainnet/product contracts fail-closed"
target_envs:
  - "local checkout"
  - "Mac Studio local artifacts under /opt/roehub/state/rl_trading/"
required_literals:
  - "08O"
  - "08K"
  - "12502.65333026"
  - "125.0265333"
  - "316"
  - "single_group_dominates_final_result"
  - "ticker_stability_obviously_broken"
  - "0.954610281973835"
  - "0.24054982817869416"
  - "target_host_non_production_forensic_pre_main"
non_goals:
  - "Do not train, fine-tune or optimize a model."
  - "Do not rerun Optuna or change thresholds as part of this stage."
  - "Do not make 08M the main candidate under analysis; 08M is fallback/context only."
  - "Do not mutate registry state, activation state, runtime config, user entitlements, API/UI behavior, paper/testnet/live execution or exchange state."
  - "Do not open Stage 19, Stage 20 or Stage 21."
quality_gates:
  - cmd: "uv run python -m tools.docs.generate_docs_index --check"
    expect: "passes after Markdown updates"
  - cmd: "git diff --check"
    expect: "passes for changed docs/prompts/code"
  - cmd: "uv run ruff check scripts/rl_trading tests/unit/scripts/rl_trading"
    expect: "passes if Python helper code/tests changed"
  - cmd: "uv run pyright scripts/rl_trading tests/unit/scripts/rl_trading"
    expect: "passes if Python helper code/tests changed"
  - cmd: "uv run pytest -q tests/unit/scripts/rl_trading"
    expect: "passes if Python helper code/tests changed"
validation_strategy:
  depth: candidate_forensic_decomposition
  e2e_required: true
  proof_boundary: target_host_non_production_forensic_pre_main
  acceptance_surfaces:
    - "08K artifact hash validation"
    - "08K math reconciliation: PnL, return percent, starting base, closed trades, final holdout sessions"
    - "PnL decomposition by month, day, ticker, volatility bucket, long/short side, trade/session and best/worst sequences"
    - "root-cause analysis for volatility-bucket dominance and ticker positive group ratio"
    - "article metric comparison beyond return where source and artifact data are available"
    - "allowlist/per-regime feasibility assessment"
    - "scorecard/gate bug audit"
    - "downstream gate decision with mainnet/product path still fail-closed"
  evidence_target: docs/architecture/ml/rl-trading-agent-platform-v1-stage-reports/08o-stage08k-dqn-forensic-decomposition.md
stage_execution_ledger:
  path: docs/architecture/ml/rl-trading-agent-platform-v1-stage-reports/rl-trading-agent-platform-v1-stage-ledger.md
  plan_doc: docs/architecture/ml/rl-trading-agent-platform-v1.md
  current_stage: "08O"
  required_update: true
expected_primary_touches:
  - "docs/architecture/ml/rl-trading-agent-platform-v1-stage-reports/08o-stage08k-dqn-forensic-decomposition.md"
  - "docs/architecture/ml/rl-trading-agent-platform-v1-stage-reports/rl-trading-agent-platform-v1-stage-ledger.md"
  - "docs/architecture/ml/rl-trading-agent-platform-v1.md"
  - "docs/architecture/README.md"
possible_secondary_touches:
  - "scripts/rl_trading"
  - "tests/unit/scripts/rl_trading"
safety_notes:
  - "This is a forensic research stage, not a runtime activation stage."
  - "If evidence is incomplete, fail closed; do not turn aggregate return into product quality."
  - "Stage 19+ remains blocked unless a later explicit product/mainnet quality gate is accepted."
---

# Task

Implement Stage `08O`: deep forensic decomposition of the blocked Stage `08K`
native DQN candidate.

The purpose is to explain, with artifact-backed calculations, whether `08K` is:

- still simply blocked;
- a restricted research candidate usable only for a limited ticker/regime allowlist;
- a per-ticker/per-regime calibration candidate;
- a close article-reproduction candidate but not a Roehub product candidate;
- blocked by a scorecard/gate bug that must be repaired;
- or impossible to classify because evidence is insufficient.

`08K` is the main object of this stage. `08M` is not the main candidate and must
not be used to hide or replace the DQN analysis.

## Prompt-Pack Execution Anchor

- `plan_doc`: `docs/architecture/ml/rl-trading-agent-platform-v1.md`
- `prompt_pack_dir`: `.codex/agents/generated/rl-trading-agent-platform-v1`
- `stage_ledger`: `docs/architecture/ml/rl-trading-agent-platform-v1-stage-reports/rl-trading-agent-platform-v1-stage-ledger.md`
- `execution_mode`: `manual_sequential`
- `GOAL.md`: optional, not required by default.
- Stage gate: read the ledger before edits; run only when `current_stage=08O`.
- Manifest gate: every created/modified/deleted file and every runtime artifact path must be recorded in the stage report and ledger.

## Context / Current State

Stage `08K` completed the full `30/10` article/demo-profile DQN path:

- HF-control branch;
- Roehub-native article-selector branch;
- full training;
- `Optuna` calibration;
- untouched final holdout evaluation.

The Roehub-native DQN final holdout produced:

- PnL after costs: `12502.65333026`;
- return after costs: `125.0265333%`;
- closed trades: `316`;
- best sanity baseline: `0.0`;
- blocker `single_group_dominates_final_result`;
- blocker `ticker_stability_obviously_broken`;
- volatility-bucket dominance: `0.954610281973835`, limit `0.8`;
- ticker positive group ratio: `0.24054982817869416`, minimum `0.25`.

This aggregate return is near the article's published `+144.23%` return by one
coarse metric, but Stage `08K` stayed blocked because aggregate return alone is
not enough to prove a stable Roehub strategy.

Stage `08N`, `17`, and `18` later confirmed that the current runtime/product path
is closed for mainnet/product claims. Stage `18` accepted only
`monitor_only_technical_soak`; Stage `19` remains blocked.

## Requirements (Must)

- Start by stating exactly: `User required before start: nothing unless a listed prerequisite is not accepted or a required credential/dataset/runtime source is unavailable; never ask for secrets in chat`.
- Verify prerequisites before analysis:
  - `08J` is `accepted`;
  - `08K` is `blocked`;
  - `08L` is `accepted`;
  - `08M` is `accepted` but is not the main candidate for this stage;
  - `08N` is `accepted` with `quality_status=accepted_for_research_only`;
  - `17` is `accepted` only as `infrastructure_only`;
  - `18` is `accepted` only as `monitor_only_technical_soak`;
  - `19` is blocked or pending with `stage19_mainnet_readiness_allowed=false`;
  - `current_stage=08O`.
  If not true, write/update the `08O` report as blocked, update the ledger, and stop.
- Compute and record this prompt hash:
  - `shasum -a 256 .codex/agents/generated/rl-trading-agent-platform-v1/08o-stage08k-dqn-forensic-decomposition.md`
- Verify all required `08K` artifact hashes before conclusions:
  - dual-branch summary sha256 `70dfffd378f33da997b2471422491e9defc28f8b361b9c5b645087288129286d`;
  - native final evaluation manifest sha256 `c16a08dffa6f49fd8f24d9a1af6737114d54363b4cca3630a4e83c2661880b07`;
  - native `Optuna` summary sha256 `8585d4342dab24850cd077e5287de5faab251e848f18eb044f70cc410ebf6cec`;
  - Stage `08J` article dataset manifest sha256 `fd7c614b4cc5085cc24cd054143b6bb188283b9cf423122d436e37769fcd639a`.
- Create a sanitized forensic summary artifact under:
  - `/opt/roehub/state/rl_trading/evaluation_runs/stage08o_stage08k_dqn_forensic_decomposition_v1/`
  It must include at least `stage08o_stage08k_forensic_summary.json` and a recorded file sha256.
- Produce the Stage `08O` report at:
  - `docs/architecture/ml/rl-trading-agent-platform-v1-stage-reports/08o-stage08k-dqn-forensic-decomposition.md`
- The report must include:
  - artifact lineage and hashes;
  - math reconciliation for `12502.65333026`, `125.0265333%`, starting base, `316` trades and final holdout scope;
  - PnL decomposition by month, day, ticker, volatility bucket, long/short side, trade/session and best/worst sequences;
  - root-cause analysis for `single_group_dominates_final_result`;
  - root-cause analysis for `ticker_stability_obviously_broken`;
  - allowlist and per-regime feasibility assessment;
  - scorecard/gate bug audit;
  - article comparison beyond aggregate return;
  - exact final classification and downstream gates.
- Compare against the article/source scorecard where evidence is available:
  - return `+144.23%`;
  - drawdown;
  - Sharpe;
  - Sortino;
  - accuracy;
  - profit days;
  - trades/day;
  - average trade size;
  - trade count;
  - equity-curve shape.
  If exact article or source data is unavailable, do not invent values. Record the unavailable fields and why.
- Decide whether the `08K` article similarity is:
  - source-faithful methodology evidence;
  - likely aggregate-return coincidence;
  - incomplete because article/source artifacts are unavailable;
  - or blocked by scorecard/gate artifact limitations.
- Classify Stage `08K` into exactly one `08k_forensic_status`:
  - `08k_remains_blocked`;
  - `restricted_research_candidate`;
  - `article_reproduction_candidate_not_product`;
  - `per_ticker_per_regime_calibration_candidate`;
  - `scorecard_gate_bug_requires_repair`;
  - `insufficient_evidence`.
- Record downstream booleans explicitly:
  - `stage09_for_08k_allowed`;
  - `stage19_mainnet_readiness_allowed`;
  - `stage20_mainnet_canary_allowed`;
  - `stage21_product_rollout_allowed`;
  - `08p_allowed`;
  - `next_prompt`.
- `stage19_mainnet_readiness_allowed`, `stage20_mainnet_canary_allowed`, and `stage21_product_rollout_allowed` must remain `false` in this stage unless a separate explicit product/mainnet quality gate exists. There is no such gate in the current state.
- Update the RL plan and ledger after validation and before final response.
- Keep secrets, raw provider payloads, credentials, cookies, tokens and exchange-private payloads out of logs, reports and ledgers.

## Requirements (Should)

- Prefer reusable parsing helpers over ad hoc one-off shell parsing if artifact structure is non-trivial.
- If adding a helper script, make it deterministic, path-explicit and testable with fixture data.
- Include plain-language interpretation in the report so the product decision is understandable without reading JSON.
- Preserve failed and blocked history; do not rewrite `08K`, `08M`, `08N`, `17`, or `18` as if their original decisions changed.

## Requirements (Nice-to-have)

- Include small tables for top positive/negative tickers, days, volatility buckets and trade sequences.
- Include a compact allowlist candidate table if the evidence supports restricted ticker/regime research.
- Include a clear chart-ready data artifact for future UI or notebook review, but do not build UI in this stage.

# Context Acquisition Protocol

Read `.codex/AGENTS.md`, the RL plan, ledger, Stage `08J`, Stage `08K`,
Stage `08M`, Stage `08N`, and Stage `18` reports first. Then read only the listed
task entrypoints. Expand only for blockers, missing artifact fields, metric
definition ambiguity, failing gates, or source/article comparison.

Do not search the whole repository unless a required entrypoint is missing or a
metric definition cannot be located from the bounded context.

# Reading Manifest

Always read:

- `.codex/AGENTS.md`
- `docs/architecture/ml/rl-trading-agent-platform-v1.md`
- `docs/architecture/ml/rl-trading-agent-platform-v1-stage-reports/rl-trading-agent-platform-v1-stage-ledger.md`
- `docs/architecture/ml/rl-trading-agent-platform-v1-stage-reports/08j-article-session-extractor-dataset.md`
- `docs/architecture/ml/rl-trading-agent-platform-v1-stage-reports/08k-article-demo-profile-training-evaluation.md`
- `docs/architecture/ml/rl-trading-agent-platform-v1-stage-reports/08m-supervised-warm-start-candidate-scorecard.md`
- `docs/architecture/ml/rl-trading-agent-platform-v1-stage-reports/08n-candidate-quality-reclassification.md`
- `docs/architecture/ml/rl-trading-agent-platform-v1-stage-reports/18-rl-soak-incident-drills.md`

Read only if needed:

- `docs/architecture/ml/rl-trading-agent-platform-v1-stage-reports/08i4-post-repair-methodology-recheck.md`
- `docs/architecture/ml/rl-trading-agent-platform-v1-stage-reports/08l-reward-warm-start-research.md`
- `scripts/rl_trading/stage08g_dual_branch_cpu_training_evaluation.py`
- `scripts/rl_trading/stage08g_cpu_optuna_calibration.py`
- `scripts/rl_trading/stage08f_roehub_native_backtest_evaluation.py`
- `src/trading/contexts/rl_trading/domain/roehub_native_evaluation.py`
- relevant tests under `tests/unit/scripts/rl_trading`

# Work Plan

1. Verify the stage ledger and prerequisites. If any prerequisite is wrong, write
   a blocked `08O` report, update the ledger, and stop.
2. Record prompt path/hash and planned file manifest.
3. Verify required `08K` and `08J` artifact hashes on Mac Studio/local artifact paths.
4. Inspect artifact schema and locate the minimum fields needed for:
   - final holdout sessions;
   - per-trade or per-session PnL;
   - ticker;
   - timestamp/day/month;
   - side/action;
   - volatility bucket or enough inputs to reconstruct it;
   - equity or enough fields to reconstruct equity and drawdown.
5. If fields are missing, distinguish:
   - `metric_unavailable_but_non_blocking`;
   - `metric_unavailable_blocks_classification`;
   - `artifact_schema_bug`.
6. Reconcile the `08K` math:
   - starting base;
   - PnL after costs;
   - return percent;
   - closed trades;
   - final holdout session count;
   - baseline comparison.
7. Decompose PnL and stability:
   - month;
   - day;
   - ticker;
   - volatility bucket;
   - long vs short;
   - trade/session;
   - best and worst contiguous sequences.
8. Diagnose `volatility-bucket dominance 0.954610281973835`:
   - confirm numerator/denominator;
   - identify the dominant bucket;
   - decide whether dominance is a real market-regime concentration, a grouping/reporting artifact, or an unavailable-field blocker.
9. Diagnose `ticker positive group ratio 0.24054982817869416`:
   - list positive/negative/flat ticker groups;
   - identify whether a truthful restricted allowlist could exist;
   - estimate whether the result is too concentrated for tariff-based user ticker selection.
10. Compare to article/source metrics beyond aggregate return where data is available.
11. Audit the scorecard/gate itself:
   - verify threshold math;
   - verify group counting;
   - verify no sign inversion, duplicate grouping, missing-session bias, or denominator bug explains the blockers.
12. Decide final `08k_forensic_status` and downstream booleans.
13. Write sanitized JSON summary under `/opt/roehub/state/rl_trading/evaluation_runs/stage08o_stage08k_dqn_forensic_decomposition_v1/`.
14. Write the Stage `08O` report, update the plan, update the ledger, and run quality gates.
15. Stop. Do not run the next stage automatically.

# Acceptance Criteria

Stage `08O` is accepted only if:

- prerequisites and artifact hashes are verified, or missing evidence is recorded as a blocker;
- the `12502.65333026` PnL and `125.0265333%` return are reconciled or flagged as a scorecard bug;
- PnL is decomposed at least by ticker, month/time, volatility bucket and side, unless the artifact schema makes a field unavailable and the report explains why;
- both `single_group_dominates_final_result` and `ticker_stability_obviously_broken` receive a concrete evidence-backed explanation;
- article comparison is completed for available metrics and unavailable metrics are not invented;
- final `08k_forensic_status` is exactly one allowed enum value;
- downstream booleans keep `19+` fail-closed;
- stage report and ledger are updated;
- final report contains a file manifest, evidence paths/hashes, quality gates, contract impact and residual risks.

If these are not met, Stage `08O` must be `blocked`, with exact blocker,
evidence path and the smallest next repair step.

# Implementation Constraints

## Git / Branch

- Work from `main` by default.
- Do not create branches, branch-specific worktrees, temporary checkouts, local folders, stashes or auxiliary workflow artifacts unless the user explicitly requests that exact workflow.
- Preserve foreign dirty changes. Own only files/hunks required by this stage.
- Do not use broad staging or unstaging commands such as `git add .`, `git add -A`, `git add --all`, `git add :/`, `git add -- .`, `git add *`, `git restore --staged .`, `git reset .`, `git commit -a`, or `git commit -am`.
- If publishing is explicitly requested, stage only owned paths/hunks, inspect `git diff --cached --name-status`, and use `ROEHUB_SCOPED_STAGING_REVIEWED=1`.

## Data / Runtime

- Use `/opt/roehub/state/rl_trading/` for large runtime artifacts.
- Do not commit datasets, checkpoints, raw logs, or large decomposition dumps.
- Do not read or request secrets.
- Do not use exchange SDKs or submit orders.
- Do not sync or restart `/opt/roehub/app`.
- Any Mac Studio check is `target_host_non_production_forensic_pre_main`, not `post_main_production_runtime_proof`.

## Documentation

- Use the existing stage-report shape: scope, observed state, evidence, decision matrix, contract impact, file manifest, quality gates, blockers, next-stage handoff.
- Update the ledger after validation and before final response.
- Update `docs/architecture/ml/rl-trading-agent-platform-v1.md` so it no longer says no next prompt is open if `08O` is now the active research stage.
- Run docs-index check when Markdown docs change.

# Files To Indicate

Expected created:

- `docs/architecture/ml/rl-trading-agent-platform-v1-stage-reports/08o-stage08k-dqn-forensic-decomposition.md`
- optional `/opt/roehub/state/rl_trading/evaluation_runs/stage08o_stage08k_dqn_forensic_decomposition_v1/stage08o_stage08k_forensic_summary.json`

Expected modified:

- `docs/architecture/ml/rl-trading-agent-platform-v1-stage-reports/rl-trading-agent-platform-v1-stage-ledger.md`
- `docs/architecture/ml/rl-trading-agent-platform-v1.md`
- `docs/architecture/README.md` if docs index changes

Possible modified/created only if a helper is needed:

- `scripts/rl_trading/stage08o_stage08k_dqn_forensic_decomposition.py`
- `tests/unit/scripts/rl_trading/test_stage08o_stage08k_dqn_forensic_decomposition.py`

Any outside-expected file must be listed in the final manifest with justification.

# Non-Goals

- No new model training.
- No fine-tuning.
- No `Optuna` rerun.
- No threshold tuning as an implementation step.
- No registry mutation.
- No runtime activation.
- No UI/CJM work.
- No paper/testnet/live/mainnet execution.
- No Stage `19`, `20`, or `21` launch.

# Quality Gates

Always run:

```bash
uv run python -m tools.docs.generate_docs_index --check
git diff --check
```

If Python helper code or tests changed, also run focused gates:

```bash
uv run ruff check scripts/rl_trading tests/unit/scripts/rl_trading
uv run pyright scripts/rl_trading tests/unit/scripts/rl_trading
uv run pytest -q tests/unit/scripts/rl_trading
```

# Final Output: Report Format

Return a Russian final report with:

- stage status: `accepted` or `blocked`;
- prompt path and sha256;
- verified artifact paths and hashes;
- `08K` math reconciliation summary;
- decomposition summary;
- blocker explanation for `single_group_dominates_final_result`;
- blocker explanation for `ticker_stability_obviously_broken`;
- article-comparison status;
- final `08k_forensic_status`;
- downstream booleans and next prompt, if any;
- file manifest: created / modified / deleted / outside expected paths;
- checks run and results;
- residual risks.

Do not claim that `08K` is product-ready, live-ready or mainnet-ready unless a
future explicit product/mainnet quality gate is accepted. This prompt cannot
produce that gate.
