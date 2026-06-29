---
prompt_name: 06-macstudio-repair-proof
repo: roehub.com
branch: main
branch_policy:
  default_branch: main
  separate_branch_allowed: false
  stage_specific_branches_forbidden: true
  worktree_allowed: false
  stash_allowed: false
scope: "Deliver the live-tail repair changes and prove on Mac Studio that a missing minute is repaired without ClickHouse."
language:
  implementation: shell/python/markdown
  agent_report: ru
context_sources:
  always_read:
    - path: .codex/AGENTS.md
      why: "Mac Studio, proof-boundary, publish/deploy, and redaction rules"
    - path: docs/architecture/market_data/market-data-live-tail-repair-v1.md
      why: "source plan"
    - path: docs/architecture/market_data/market-data-live-tail-repair-v1-stage-reports/market-data-live-tail-repair-v1-stage-ledger.md
      why: "stage gate"
    - path: docs/architecture/live_execution/strategy-producer-paper-testnet-trading-v1-stage-reports/12-4-sustained-6h-soak.md
      why: "original blocker evidence"
  task_entrypoints:
    - path: apps/worker/strategy_live_runner
      why: "runtime service under proof"
    - path: apps/worker/market_data_ws
      why: "Market Data hot cache writer/runtime"
    - path: infra
      why: "runtime deploy/Monit/Prometheus assets"
skill_routing:
  - skill: publish-ci-deploy
    use_when: "publish, CI, deploy, Mac Studio sync, and production runtime proof are required"
    timing: "main workflow"
    reason: "Stage 06 is a delivery and target runtime proof stage"
  - skill: backend-performance-evidence
    use_when: "summarizing latency/resource evidence from runtime proof"
    timing: "during verification"
    reason: "repair latency and CPU/RAM impact must be measured honestly"
  - skill: root-cause-debugging
    use_when: "controlled missing-minute scenario does not recover"
    timing: "if blocker"
    reason: "must localize whether Redis cache, provider chain, REST tail, or runner ACK policy failed"
validation_strategy:
  depth: ci_deploy
  e2e_required: true
  acceptance_surfaces: ["ci", "deploy", "macstudio-runtime", "redis", "postgres", "prometheus", "strategy-runner"]
  tests_only_allowed_reason: ""
  evidence_target: docs/architecture/market_data/market-data-live-tail-repair-v1-stage-reports/06-macstudio-repair-proof.md
proof_boundary:
  required_when: "Stage 06 validates changed code in production runtime"
  label: post_main_production_runtime_proof
  changed_code_production_claim_allowed: true
stage_execution_ledger:
  path: docs/architecture/market_data/market-data-live-tail-repair-v1-stage-reports/market-data-live-tail-repair-v1-stage-ledger.md
  plan_doc: docs/architecture/market_data/market-data-live-tail-repair-v1.md
  current_stage: "06"
  required_update: true
runtime_env_sources:
  roehub_env_file_order:
    - "$ROEHUB_ENV_FILE"
    - "/Users/daniildegtyarev/.config/roehub/roehub.env"
    - "/etc/roehub/roehub.env"
  report_only_key_presence: true
  forbidden_in_reports: ["raw secrets", "tokens", "credentials", "cookies", "provider payloads"]
remote_command_quoting:
  applies_when: "SSH commands contain SQL, JSON, multiline payloads, apostrophes, backticks, or dollar signs"
  required_pattern: "quoted heredoc or stdin, such as <<'SQL', <<'JSON', --queries-file /dev/stdin, query=@-"
  forbidden_pattern: "nested inline query quoting"
expected_primary_touches:
  - docs/architecture/market_data/market-data-live-tail-repair-v1-stage-reports/06-macstudio-repair-proof.md
  - docs/architecture/market_data/market-data-live-tail-repair-v1-stage-reports/market-data-live-tail-repair-v1-stage-ledger.md
  - docs/architecture/live_execution/strategy-producer-paper-testnet-trading-v1-stage-reports/strategy-producer-paper-testnet-trading-v1-stage-ledger.md
possible_secondary_touches:
  - docs/architecture/README.md
safety_notes:
  - "Do not print host-local env values."
  - "Do not attempt mainnet submit."
  - "Use only controlled paper/testnet strategy subjects."
---

# Task

Implement Stage `06` Mac Studio repair proof.

## Requirements (Must)

- Verify Stage `05 accepted` in the repair ledger.
- Use `publish-ci-deploy` to ensure the target revision is on `origin/main`, CI/deploy is green, and Mac Studio checkout/runtime are synced before changed-code runtime claims.
- Record proof boundary exactly as `post_main_production_runtime_proof`.
- Run a controlled scenario on Mac Studio:
  - selected active paper/testnet strategy;
  - one missing closed minute in `md.candles.1m.<instrument_key>` or an equivalent controlled synthetic stream;
  - ClickHouse repair path unavailable or circuit-open for the repair attempt;
  - REST tail or safe synthetic REST adapter path available;
  - runner restores the missing minute and continues.
- Prove:
  - checkpoint advances across the missing minute;
  - `StrategySignal` continues after the repaired candle;
  - linked `ExecutionSourceEvent` continues where expected;
  - repair audit row records source/status/range;
  - Redis hot cache contains restored candle;
  - metrics show repair source/latency/cache/circuit/checkpoint-stall signals;
  - no mainnet, no secret leak, no raw provider payload.
- Update repair ledger and strategy-producer ledger handoff.
- If accepted and docs changed, deliver final report/ledger through direct-main discipline.

## Non-Goals

- Do not run the full 6h Stage `12.4` in this stage.
- Do not use real mainnet accounts.
- Do not mutate exchange account config.

## Acceptance Criteria

- Mac Studio runtime proof demonstrates missing-minute recovery when ClickHouse is unavailable.
- DB evidence shows new `StrategySignal` and expected `ExecutionSourceEvent` after the repaired gap.
- Audit and metrics evidence are recorded.
- Strategy-producer ledger records that the original blocker is repaired and `12.4` may be rerun.
- Ledger marks `06 accepted`.

## Quality Gates

- `uv run ruff check .`
- `uv run pyright`
- `uv run pytest -q -ra`
- `uv run python -m tools.docs.generate_docs_index --check`
- CI/deploy evidence from `publish-ci-deploy`
- Mac Studio runtime smoke and controlled repair proof

## Final Output

Russian report with main SHA, CI/deploy status, Mac Studio checkout/runtime status, controlled missing-minute proof, repair metrics/audit evidence, strategy-producer handoff, residual risks, and Stage `07` handoff.
