---
prompt_name: 07-stage-12-4-rerun-handoff
repo: roehub.com
branch: main
branch_policy:
  default_branch: main
  separate_branch_allowed: false
  stage_specific_branches_forbidden: true
  worktree_allowed: false
  stash_allowed: false
prompt_pack_execution:
  mode: manual_sequential
  plan_doc: docs/architecture/market_data/market-data-live-tail-repair-v1.md
  prompt_pack_dir: .codex/agents/generated/market-data-live-tail-repair-v1/
  stage_ledger: docs/architecture/market_data/market-data-live-tail-repair-v1-stage-reports/market-data-live-tail-repair-v1-stage-ledger.md
  goal_mode_optional: true
  goal_artifact_required: false
scope: "Validate an existing Strategy Producer Stage 12.4 rerun artifact or rerun Stage 12.4 after accepted live-tail repair proof."
language:
  implementation: shell/markdown
  agent_report: ru
context_sources:
  always_read:
    - path: .codex/AGENTS.md
      why: "repo contract"
    - path: docs/architecture/market_data/market-data-live-tail-repair-v1-stage-reports/market-data-live-tail-repair-v1-stage-ledger.md
      why: "repair-cycle accepted proof"
    - path: docs/architecture/live_execution/strategy-producer-paper-testnet-trading-v1-stage-reports/strategy-producer-paper-testnet-trading-v1-stage-ledger.md
      why: "Stage 12.4 current state and 12.5 gate"
    - path: .codex/agents/generated/strategy-producer-paper-testnet-trading-v1/12-4-sustained-6h-soak.md
      why: "canonical 12.4 rerun prompt"
  task_entrypoints:
    - path: docs/architecture/live_execution/strategy-producer-paper-testnet-trading-v1-stage-reports/12-4-sustained-6h-soak.md
      why: "existing blocked report to supersede or append rerun evidence"
    - path: /opt/roehub/state/live_execution/stage12-4-sustained-6h-soak/20260630T012705Z-stage07-rerun-c2138129-a14a-40b3-bcf0-9ff4cf5a5757/latest_status.json
      why: "candidate 6h rerun artifact provided by the user on macstudio runtime"
    - path: /opt/roehub/state/live_execution/stage12-4-sustained-6h-soak/20260630T012705Z-stage07-rerun-c2138129-a14a-40b3-bcf0-9ff4cf5a5757/snapshots.jsonl
      why: "candidate hourly snapshot artifact provided by the user on macstudio runtime"
skill_routing:
  - skill: backend-performance-evidence
    use_when: "summarizing sustained soak latency/resource evidence"
    timing: "during verification"
    reason: "12.4 acceptance depends on comparable latency and resource measurement"
  - skill: browser-qa-evidence
    use_when: "capturing final /strategies state"
    timing: "during verification"
    reason: "browser-visible strategy status must be proven"
  - skill: publish-ci-deploy
    use_when: "accepted report/ledger changes need delivery"
    timing: "before final report"
    reason: "direct-main delivery required"
validation_strategy:
  depth: target_runtime
  e2e_required: true
  acceptance_surfaces: ["6h-runtime", "signal-latency", "signal-dedup", "repair-metrics", "redis", "postgres", "prometheus", "browser"]
  tests_only_allowed_reason: ""
  evidence_target: docs/architecture/live_execution/strategy-producer-paper-testnet-trading-v1-stage-reports/12-4-sustained-6h-soak.md
proof_boundary:
  required_when: "Stage 07 observes deployed runtime after accepted repair proof"
  label: post_main_production_runtime_proof
  changed_code_production_claim_allowed: true
  requires_main_green_ci_deploy_sync: true
  macstudio_git_checkout: /Users/daniildegtyarev/Projects/roehub.com
  macstudio_runtime_tree: /opt/roehub/app
readiness_anchors:
  previous_stage_ledger_gate: "Before validation, read the repair stage ledger and verify Stage 06 is accepted; read the strategy-producer ledger and verify 12.4 is not accepted and 12.5 is closed."
  file_manifest_required: true
  proof_boundary_label_required: post_main_production_runtime_proof
stage_execution_ledger:
  path: docs/architecture/market_data/market-data-live-tail-repair-v1-stage-reports/market-data-live-tail-repair-v1-stage-ledger.md
  plan_doc: docs/architecture/market_data/market-data-live-tail-repair-v1.md
  current_stage: "07"
  required_update: true
expected_primary_touches:
  - docs/architecture/live_execution/strategy-producer-paper-testnet-trading-v1-stage-reports/12-4-sustained-6h-soak.md
  - docs/architecture/live_execution/strategy-producer-paper-testnet-trading-v1-stage-reports/strategy-producer-paper-testnet-trading-v1-stage-ledger.md
  - docs/architecture/market_data/market-data-live-tail-repair-v1-stage-reports/07-stage-12-4-rerun-handoff.md
  - docs/architecture/market_data/market-data-live-tail-repair-v1-stage-reports/market-data-live-tail-repair-v1-stage-ledger.md
possible_secondary_touches:
  - docs/architecture/README.md
safety_notes:
  - "Do not open Strategy Producer Stage 12.5 unless Stage 12.4 is accepted."
  - "Do not reinterpret Stage 06 repair proof as a 6h soak."
---

# Task

Execute Stage `07` by rerunning or explicitly reopening Strategy Producer Stage `12.4` after accepted Market Data live-tail repair proof.

## Requirements (Must)

- Verify repair ledger marks Stage `06 accepted`.
- Verify strategy-producer ledger still marks `12.4 blocked` and `12.5 pending/blocked`.
- Previous stage ledger gate: before validation, read `stage_execution_ledger.path` and verify Stage `06 accepted`; also read `docs/architecture/live_execution/strategy-producer-paper-testnet-trading-v1-stage-reports/strategy-producer-paper-testnet-trading-v1-stage-ledger.md` and verify Strategy Producer Stage `12.4` is not accepted and `12.5` is closed. Record this gate in the Stage `07` report before using any runtime artifact.
- The Stage `07` report must include a file manifest table with `Created / Modified / Deleted / Reason / Contract impact`; justify every touched file outside `expected_primary_touches`.
- `post_main_production_runtime_proof` is valid only when the observed revision is already on `main`, relevant GitHub Actions/CI are green, Mac Studio checkout `/Users/daniildegtyarev/Projects/roehub.com` is synced, `/opt/roehub/app` is deployed/synced from that revision, and the runtime proof is collected after that sync. If any part is missing, label the evidence as blocked or readiness-only; do not claim changed-code production proof.
- Use the existing Strategy Producer prompt `.codex/agents/generated/strategy-producer-paper-testnet-trading-v1/12-4-sustained-6h-soak.md` as the canonical `12.4` acceptance contract.
- If an existing 6h artifact directory is available, validate it first. Do not rerun 6 hours until the artifact is classified against every acceptance surface in the canonical `12.4` prompt.
- Before accepting an existing artifact or starting any 6h timer, prove repair metrics/audit surfaces from Stage `06` are still available.
- Validate or run `12.4` according to its current prompt:
  - active strategy runtime;
  - 6 elapsed hours;
  - signal-path latency/dedup;
  - Redis/DB/Prometheus/Monit/resource snapshots, including non-empty per-process CPU/RSS evidence or an equivalent historical source for the same window;
  - repair metric/audit surface availability after the Market Data repair cycle;
  - final browser/API state.
- Empty `processes=[]` snapshots are not acceptable process-resource evidence by themselves. If the collector produced empty process rows because it parsed a truncated broad `ps` output, either reconstruct the same-window process evidence from a reliable historical source or mark Stage `07` / Strategy Producer `12.4` blocked and require a rerun with a fixed collector.
- If the Stage `07` report/ledger already records that same-window process evidence for an existing candidate is not reconstructible, do not repeat the same artifact-recovery loop; rerun `12.4` with a fixed collector that uses exact `pgrep -f` / `ps -p` collection and fails closed on empty required process rows.
- Final browser/API proof must be collected before acceptance unless the `12.4` report explicitly defers it to `12.5` with complete API evidence and a non-safety reason.
- If a short candle gap appears during the soak, record whether the new repair path recovered it and include audit/metrics evidence.
- If `12.4` passes, update both ledgers: repair Stage `07 accepted` and strategy-producer `12.4 accepted`; only then `12.5` may open.
- If `12.4` blocks again, classify whether blocker is the same live-tail repair issue or a new unrelated issue; update both ledgers.
- Deliver report/ledger docs through direct-main discipline if files changed.

## Non-Goals

- Do not add new repair implementation in this stage unless the rerun finds a narrow blocker and the report marks Stage `07 blocked`.
- Do not start Stage `12.5` in the same execution.

## Acceptance Criteria

- `12.4` accepted rerun has full 6h evidence and repair path remains healthy; or Stage `07` records a clear blocker with source and next action.
- Existing candidate artifact acceptance requires all required `12.4` surfaces, not only `latest_status.json.status=passed`.
- If accepted, strategy-producer ledger allows `12.5`.
- Repair ledger records whether this plan closed the original problem.

## Quality Gates

- `uv run python -m tools.docs.generate_docs_index --check`
- Runtime collector evidence required by `12-4-sustained-6h-soak.md`
- Browser/API proof required by `12-4-sustained-6h-soak.md`

## Final Output

Russian report with `12.4` rerun result, repair-path observations, signal latency/dedup evidence, ledger updates, delivery status, and explicit statement whether `12.5` is allowed.
