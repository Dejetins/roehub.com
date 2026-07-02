---
doc: rl-trading-agent-platform-v1-stage-10a-retraining-promotion-lifecycle
status: accepted
stage: 10A
updated_at: 2026-07-02
---

# Stage 10A: retraining and promotion lifecycle

Статус: `accepted`.

User required before start: nothing unless a listed prerequisite is not accepted or a required credential/dataset/runtime source is unavailable; never ask for secrets in chat

## Pre-Edit Gate

| Field | Value |
|---|---|
| Prompt path | `.codex/agents/generated/rl-trading-agent-platform-v1/10a-retraining-promotion-lifecycle.md` |
| Prompt sha256 | `afaa90a5b8c603afd4ae8d30cad603ec4605804a3441761437e515af033a279a` |
| Ledger state observed before work | `current_stage=10A`; Stage `10A` pending/current |
| Prerequisite verdict | accepted Stage `09B`; accepted Stage `10`; accepted Stage `08I3`; accepted Stage `08I4`; accepted corrective Stage `08M` candidate with `stage09_allowed=true`; Stage `10A` may proceed |
| `.codex/agents/.context/promt_manager_state.yaml` | read; treated as stale prompt-generation state where it conflicts with current `.codex/AGENTS.md`, prompt and ledger |
| Browser/auth | `N/A`; username `smoke_e2e_keycloak` was not used and `ROEHUB_SMOKE_E2E_PASSWORD` was not read |
| Exchange/provider effects | `N/A`; no exchange SDK, order submit, paper/testnet/live/mainnet path, provider credential, or raw provider payload surface is in scope |

## Planned Concrete File List Before Edit

Created:

| Path | Reason | Contract impact |
|---|---|---|
| `src/trading/contexts/rl_trading/domain/retraining_promotion_lifecycle.py` | Stage `10A` retraining task, drift task, promotion profile, approval gate and rollback manifest domain surface. | `compatible-change` additive internal Python domain surface |
| `scripts/rl_trading/stage10a_retraining_promotion_lifecycle.py` | Host-local operator CLI for manual/scheduled lifecycle planning, promotion checks and rollback dry-run. | `compatible-change` host-local CLI |
| `tests/unit/contexts/rl_trading/domain/test_retraining_promotion_lifecycle.py` | Focused tests for deterministic retrain tasks, schedule disabled-by-default, no auto-promotion and rollback manifest. | `none` test-only |
| `tests/unit/scripts/rl_trading/test_stage10a_retraining_promotion_lifecycle.py` | CLI smoke tests for host-local lifecycle commands. | `none` test-only |
| `docs/architecture/ml/rl-trading-agent-platform-v1-stage-reports/10a-retraining-promotion-lifecycle.md` | Stage report, file manifest, evidence and next-stage handoff. | `compatible-change` docs/report |

Modified:

| Path | Reason | Contract impact |
|---|---|---|
| `src/trading/contexts/rl_trading/domain/__init__.py` | Export Stage `10A` lifecycle domain surface. | `compatible-change` additive internal Python exports |
| `configs/dev/rl_trading_ml_runtime.yaml` | Add disabled-by-default retraining scheduler and promotion defaults. | `compatible-change` additive fail-closed config/default contract |
| `configs/test/rl_trading_ml_runtime.yaml` | Add disabled-by-default retraining scheduler and promotion defaults. | `compatible-change` additive fail-closed config/default contract |
| `configs/prod/rl_trading_ml_runtime.yaml` | Add disabled-by-default retraining scheduler and promotion defaults. | `compatible-change` additive fail-closed config/default contract |
| `tests/unit/contexts/rl_trading/domain/test_ml_runtime_dependency_contract.py` | Assert schedule and auto-promotion stay fail-closed across profiles. | `none` test-only |
| `docs/runbooks/mac-studio-native-backend-operations.md` | Add host-local Stage `10A` operator commands and rollback notes. | `compatible-change` runbook/operator semantics |
| `docs/architecture/ml/rl-trading-agent-platform-v1-stage-reports/rl-trading-agent-platform-v1-stage-ledger.md` | Mark Stage `10A` status/evidence and next-stage handoff after validation. | `compatible-change` docs/ledger |
| `docs/architecture/README.md` | Docs index sync after adding this report. | `compatible-change` docs index |

Deleted: none planned.

Outside expected paths: none planned.

Runtime artifacts planned outside git:

| Artifact root | Reason |
|---|---|
| `/opt/roehub/state/rl_trading/lifecycle_runs/stage10a_retraining_promotion_lifecycle_v1/<run_id>/` | Host-local lifecycle summaries, retrain task manifests, promotion profile/check reports and rollback manifests. |

## Scope

Implemented Stage `10A` as an additive host-local lifecycle surface:

- retrain task planning supports `full_retrain` and `fine_tune`;
- manual, scheduled and drift triggers are represented explicitly;
- scheduled trigger is disabled by default and blocks until explicitly enabled;
- drift creates candidate retraining tasks but never mutates champion state;
- promotion scorecard profile has numeric hard/warn thresholds for PnL after fees/funding/slippage, max drawdown, trades count, ticker stability, out-of-sample period, overfit, latency/resource budget and integrity gates;
- promotion check requires hard gates plus operator/admin approval hashes;
- candidate never auto-promotes: promotion check writes artifacts only and records `registry_write_performed=false`, `activation_mutation=false`, `auto_promote=false`;
- rollback dry-run writes a host-local rollback manifest and command with `no_artifact_deletion=true`.

Not in scope:

- production DB migration or registry write;
- `/opt/roehub/app` deployment or production runtime smoke;
- browser/UI/API routes;
- exchange SDK, paper/testnet/live/mainnet side effects;
- actual long-running model training.

## Business Impact

Stage `10A` turns the accepted research/calibration artifacts into an operator-governed lifecycle. The operational outcome is that retraining can be requested as a candidate workflow, but the platform cannot silently replace the active model because a schedule fired, a drift signal appeared, or a candidate scorecard looked good.

This reduces product and financial risk before user-visible controls: Stage `11` can expose operator/admin UI actions against a defined backend lifecycle instead of inventing promotion semantics in the UI.

В бизнес-терминах этот stage закрывает риск "модель сама обновилась и начала влиять на торговые решения". После Stage `10A` retraining создает только candidate task, promotion требует operator/admin approval, а rollback оформлен как отдельная host-local операция. Для пользователей это пока не новая UI-функция, но для продукта это обязательная защита перед тем, как в Stage `11` появятся видимые controls.

## Observed State

| Area | Evidence summary |
|---|---|
| Stage prerequisites | Ledger had `current_stage=10A`; Stage `09B`, Stage `10`, Stage `08I3`, Stage `08I4` and accepted Stage `08M` corrective candidate with `stage09_allowed=true` are accepted. |
| Config defaults | `configs/dev|test|prod/rl_trading_ml_runtime.yaml` keep `trainer.enabled=false`, add `retraining.enabled=false`, `scheduled_trigger.enabled=false`, `promotion.auto_promote=false`, required operator/admin approval, and non-destructive rollback defaults. |
| Manual retrain path | Host-local Mac Studio command wrote accepted manual `full_retrain` candidate-task artifacts. |
| Scheduled trigger | Host-local Mac Studio command returned blocked status with `schedule_disabled_by_default`. |
| Drift trigger | Unit tests verify drift creates a candidate task and `mutates_champion=false`. |
| Promotion gate | Host-local Mac Studio `promotion-check` accepted only with metric gates plus operator/admin approval hashes and did not perform registry/runtime mutation. |
| Rollback | Host-local Mac Studio `rollback-dry-run` wrote a rollback manifest with no artifact deletion and no registry write. |

## Runtime Artifact Evidence

Proof boundary: `target_host_readiness_pre_main`. These artifacts were created on `macstudio` under `/opt/roehub/state/rl_trading/` from the synced non-production checkout. This is not `post_main_production_runtime_proof` and not `/opt/roehub/app` deploy evidence. A later `post_main_production_runtime_proof` for changed code requires the target revision on `main`, green CI/GitHub Actions, deployment or verified sync into `/opt/roehub/app`, and then production runtime smoke.

| Surface | Path | sha256 / hash |
|---|---|---|
| Manual retrain summary | `/opt/roehub/state/rl_trading/lifecycle_runs/stage10a_retraining_promotion_lifecycle_v1/stage10a_macstudio_manual_20260702t223000z/stage10a_retrain_lifecycle_summary.json` | sha256 `5d836e9ce5ea693462262d87d0fa39d15bc3ac9e885b0594a787144d0911585b`; summary hash `02061a976e7cfd6eaaa3f8151175e5a8081916f6990fbdf276c05d3ce010e2eb` |
| Manual retrain task | `/opt/roehub/state/rl_trading/lifecycle_runs/stage10a_retraining_promotion_lifecycle_v1/stage10a_macstudio_manual_20260702t223000z/stage10a_retrain_task_manifest.json` | sha256 `2ffb1952ca4411cc98c2e7dc49e558b9a8d8e6cbb1dc0043475ec9aea6c16911` |
| Promotion threshold profile | `/opt/roehub/state/rl_trading/lifecycle_runs/stage10a_retraining_promotion_lifecycle_v1/stage10a_macstudio_manual_20260702t223000z/stage10a_promotion_threshold_profile.json` | sha256 `9f2c7485adcfc4efb49d7fa30c34eb5818c13e4525f6aa7fc6a7fa5d4a733a7a` |
| Scheduled-disabled summary | `/opt/roehub/state/rl_trading/lifecycle_runs/stage10a_retraining_promotion_lifecycle_v1/stage10a_macstudio_scheduled_disabled_20260702t223000z/stage10a_retrain_lifecycle_summary.json` | sha256 `d387b4b2124b6d6c7c81138fdc5c8979e31856adadabc0f4cf24d5c00209b57f`; status `blocked` |
| Scheduled-disabled task | `/opt/roehub/state/rl_trading/lifecycle_runs/stage10a_retraining_promotion_lifecycle_v1/stage10a_macstudio_scheduled_disabled_20260702t223000z/stage10a_retrain_task_manifest.json` | sha256 `595fdc1affa74cff4e84d166c845c10a09271ede9418da5b9d74fc778e220e78`; blocker `schedule_disabled_by_default` |
| Promotion check summary | `/opt/roehub/state/rl_trading/lifecycle_runs/stage10a_retraining_promotion_lifecycle_v1/stage10a_macstudio_promotion_check_20260702t223000z/stage10a_promotion_check_summary.json` | sha256 `ff7acf87d3b2e88ce55862ecd15c02da7ed395dc67b7c352df2da723f04a0313`; summary hash `2ce362fd461e47ef821a7aa0852aa769a9f6127dd9984ae26f7e7cc55db5cc06` |
| Promotion check | `/opt/roehub/state/rl_trading/lifecycle_runs/stage10a_retraining_promotion_lifecycle_v1/stage10a_macstudio_promotion_check_20260702t223000z/stage10a_promotion_check.json` | sha256 `02c9f915f069bea2854bb5f314873be49a79700d5608753b07b01beaf04dc74d` |
| Rollback summary | `/opt/roehub/state/rl_trading/lifecycle_runs/stage10a_retraining_promotion_lifecycle_v1/stage10a_macstudio_rollback_20260702t223000z/stage10a_rollback_summary.json` | sha256 `edb0cddbc34579a4d327d4b762e1b619f257deefb01943f6de22ab8e6a8943c1`; summary hash `a20c481af953643b7bc19238185b2017ac1477318319aa31bc2d007d2643897d` |
| Rollback manifest | `/opt/roehub/state/rl_trading/lifecycle_runs/stage10a_retraining_promotion_lifecycle_v1/stage10a_macstudio_rollback_20260702t223000z/stage10a_rollback_manifest.json` | sha256 `b8455c0350ddf38c0ddbbc423c48562e09403126e217600bf97e685c0a73ea00` |

The Stage `10` calibration pack consumed by the Mac Studio smoke was `/opt/roehub/state/rl_trading/calibration_packs/stage10_per_ticker_calibration_v1/stage10_macstudio_20260702t213000z/stage10_per_ticker_calibration_pack.json`, sha256 `7ee51c9f58d8054be97ba2c444a585a99aabbf50ba3ca2e47a78f0d7dbae4219`.

## Contract Impact

| Dimension | Classification | Notes |
|---|---|---|
| Public API contract | `none` | No HTTP route, UI payload, public DTO or browser behavior changed. |
| Port/internal application contract | `compatible-change` | Adds internal domain/use-case functions and host-local CLI commands; no existing port signature changed. |
| DTO schema | `none` | No wire DTO changed. |
| Persisted schema | `none` | No DB migration/table/storage schema changed. |
| Config schema/defaults | `compatible-change` | Adds fail-closed `retraining`, `promotion`, and `rollback` YAML sections; existing defaults are preserved and schedule/auto-promotion remain disabled. |
| Request hash/cache key/persistence identity | `compatible-change` | Adds deterministic lifecycle artifact hashes and ids under `/opt/roehub/state/rl_trading/`; existing cache/request identities unchanged. |
| Artifact/report semantics | `compatible-change` | Adds versioned retrain task, promotion profile/check and rollback manifests. |
| Service-call auth/timeout/retry/error semantics | `none` | No service call, queue, Redis, DB write, external adapter or browser auth path changed. |
| External side-effect/idempotency/unknown-state semantics | `none` | No exchange/provider submit, paper/testnet/live/mainnet action, registry write, runtime activation or production mutation. |
| Logs/metrics/traces/audit/redaction | `compatible-change` | Adds sanitized artifact summaries with hashes and approval-reference hashes only; no secrets, raw provider payloads, signed requests or checkpoint tensors. |
| Alert/runbook semantics | `compatible-change` | Adds host-local operator commands; no automated alert route or service restart behavior changed. |
| Browser-visible behavior | `none` | Browser/auth surface is `N/A` for Stage `10A`. |
| Performance hot path | `none` | Offline artifact planning/checking only; no inference or execution hot path is wired. |
| Docs/runbooks | `compatible-change` | Adds stage report and Mac Studio runbook section. |

## Conditional Operational Coverage

| Surface | Coverage |
|---|---|
| Service calls | `N/A`; no Roehub API, worker, queue, Redis, ClickHouse write, external provider, exchange SDK or browser service call was added or changed. |
| Timeout/retry/idempotency | `N/A`; no retry loop or side-effecting operation was introduced. |
| Unknown external side-effect state | `N/A`; no provider, money-moving, paper/testnet/live/mainnet or production mutation occurred. |
| Secrets and redaction | No secrets, tokens, cookies, credentials, raw provider payloads, signed requests, raw checkpoint tensors, plaintext operator identifiers or API keys were written. Approval inputs are hashes only. |
| Logging/audit | Lifecycle artifacts record stage, hashes, blockers, approval-reference hashes and non-mutation flags; no raw logs or provider payloads are embedded. |
| Alerts/monitoring/runbook | Runbook commands were added; no launchd/Monit/Prometheus alert behavior changed. |
| Browser/auth | `N/A`; `smoke_e2e_keycloak` and `ROEHUB_SMOKE_E2E_PASSWORD` were not used. |
| Mac Studio path contract | Git/sync work used `/Users/daniildegtyarev/Projects/roehub.com`; runtime artifacts are under `/opt/roehub/state/rl_trading/`; no git command or smoke was run under `/opt/roehub/app`. |

## File Manifest

Created:

| Path | Reason | Contract impact |
|---|---|---|
| `src/trading/contexts/rl_trading/domain/retraining_promotion_lifecycle.py` | Stage `10A` retraining task, drift task, promotion profile, approval gate and rollback manifest domain surface. | `compatible-change` additive internal Python domain surface |
| `scripts/rl_trading/stage10a_retraining_promotion_lifecycle.py` | Host-local operator CLI for manual/scheduled lifecycle planning, promotion checks and rollback dry-run. | `compatible-change` host-local CLI |
| `tests/unit/contexts/rl_trading/domain/test_retraining_promotion_lifecycle.py` | Focused tests for deterministic retrain tasks, schedule disabled-by-default, no auto-promotion and rollback manifest. | `none` test-only |
| `tests/unit/scripts/rl_trading/test_stage10a_retraining_promotion_lifecycle.py` | CLI smoke tests for host-local lifecycle commands. | `none` test-only |
| `docs/architecture/ml/rl-trading-agent-platform-v1-stage-reports/10a-retraining-promotion-lifecycle.md` | Stage report, evidence and handoff. | `compatible-change` docs/report |

Modified:

| Path | Reason | Contract impact |
|---|---|---|
| `src/trading/contexts/rl_trading/domain/__init__.py` | Export Stage `10A` lifecycle domain surface. | `compatible-change` additive internal Python exports |
| `configs/dev/rl_trading_ml_runtime.yaml` | Add disabled-by-default retraining scheduler and promotion defaults. | `compatible-change` additive fail-closed config/default contract |
| `configs/test/rl_trading_ml_runtime.yaml` | Add disabled-by-default retraining scheduler and promotion defaults. | `compatible-change` additive fail-closed config/default contract |
| `configs/prod/rl_trading_ml_runtime.yaml` | Add disabled-by-default retraining scheduler and promotion defaults. | `compatible-change` additive fail-closed config/default contract |
| `tests/unit/contexts/rl_trading/domain/test_ml_runtime_dependency_contract.py` | Assert schedule and auto-promotion stay fail-closed across profiles. | `none` test-only |
| `docs/runbooks/mac-studio-native-backend-operations.md` | Add host-local Stage `10A` operator commands and rollback notes. | `compatible-change` runbook/operator semantics |
| `docs/architecture/ml/rl-trading-agent-platform-v1-stage-reports/rl-trading-agent-platform-v1-stage-ledger.md` | Mark Stage `10A` accepted and open Stage `11`. | `compatible-change` docs/ledger |
| `docs/architecture/README.md` | Docs index sync after adding this report. | `compatible-change` docs index |

Deleted: none.

Outside expected paths: none.

Foreign dirty files observed before Stage `10A` and not owned by this stage: none.

## Quality Gates And Evidence

| Gate | Result |
|---|---|
| Focused ruff | passed: `uv run ruff check src/trading/contexts/rl_trading/domain/retraining_promotion_lifecycle.py scripts/rl_trading/stage10a_retraining_promotion_lifecycle.py tests/unit/contexts/rl_trading/domain/test_retraining_promotion_lifecycle.py tests/unit/scripts/rl_trading/test_stage10a_retraining_promotion_lifecycle.py tests/unit/contexts/rl_trading/domain/test_ml_runtime_dependency_contract.py src/trading/contexts/rl_trading/domain/__init__.py` |
| Focused pyright | passed: `uv run pyright src/trading/contexts/rl_trading/domain/retraining_promotion_lifecycle.py scripts/rl_trading/stage10a_retraining_promotion_lifecycle.py tests/unit/contexts/rl_trading/domain/test_retraining_promotion_lifecycle.py tests/unit/scripts/rl_trading/test_stage10a_retraining_promotion_lifecycle.py tests/unit/contexts/rl_trading/domain/test_ml_runtime_dependency_contract.py` -> `0 errors` |
| Focused pytest | passed: `uv run pytest -q tests/unit/contexts/rl_trading/domain/test_retraining_promotion_lifecycle.py tests/unit/scripts/rl_trading/test_stage10a_retraining_promotion_lifecycle.py tests/unit/contexts/rl_trading/domain/test_ml_runtime_dependency_contract.py` -> `9 passed` |
| Prompt-level ruff | passed: `uv run ruff check src/trading/contexts/rl_trading apps tests` |
| Prompt-level pyright | passed: `uv run pyright src/trading/contexts/rl_trading apps tests` -> `0 errors` |
| Prompt-level pytest | passed: `uv run pytest -q tests/unit/contexts/rl_trading tests/unit/apps` -> `462 passed, 3 warnings` |
| Mac Studio focused pytest | passed: `uv run pytest -q tests/unit/contexts/rl_trading/domain/test_retraining_promotion_lifecycle.py tests/unit/scripts/rl_trading/test_stage10a_retraining_promotion_lifecycle.py tests/unit/contexts/rl_trading/domain/test_ml_runtime_dependency_contract.py` -> `9 passed` |
| Mac Studio artifact smoke | passed under `target_host_readiness_pre_main`: manual retrain accepted, scheduled trigger blocked as disabled-by-default, promotion check accepted only with approval hashes, rollback dry-run accepted. |
| Docs index | passed after final report/ledger update: `uv run python -m tools.docs.generate_docs_index --check` |

Deterministic rerun evidence is covered by `test_stage10a_retrain_plan_is_deterministic_and_never_auto_promotes`, which runs the same config twice and asserts stable summary hash and file sha256.

## Residual Risks

- The Mac Studio promotion smoke used a sanitized fixture candidate manifest to validate the lifecycle contract. It is not a new trained candidate and not production promotion evidence.
- Stage `10A` does not write production registry rows, does not activate runtime inference, and does not prove `/opt/roehub/app` changed code in production; that proof remains post-main only after green CI/GitHub Actions and deploy/sync.
- Later stages must preserve the Stage `10` fail-closed per-ticker rows and the `10A` approval/no-auto-promotion contract when adding UI/API controls.

## Cold Self-Review

Mode: `cold self-review fallback`.

Reason: a subagent tool is technically available, but its tool contract forbids spawning subagents unless the user explicitly requested subagents. No such request was made, so independent subagent review was not used.

Verdict: `Release`.

Checked:

- stage continuity: Stage `09B`, Stage `10`, Stage `08I3`, Stage `08I4` and accepted Stage `08M` candidate prerequisites are satisfied; next stage is `11`;
- proof boundary: Mac Studio evidence is `target_host_readiness_pre_main`, not production validation for changed code; `post_main_production_runtime_proof` still requires `main`, green CI/GitHub Actions, deploy/sync and production smoke;
- contracts: config additions are fail-closed; API/UI/exchange/persisted DB contracts unchanged;
- safety: no secrets, provider payloads, raw checkpoint tensors, browser auth or exchange side effects;
- validation: focused and prompt-level local gates passed; Mac Studio focused tests and artifact smoke passed.

## Blockers

None for Stage `10A` acceptance.

## Next-Stage Handoff

Stage `10A` is accepted. Ledger `current_stage` moves to `11`.

Next allowed prompt: `.codex/agents/generated/rl-trading-agent-platform-v1/11-rl-tab-ui-skeleton.md`.

Stage `11` may expose browser/API controls only by reusing the Stage `10A` lifecycle semantics: no auto-promotion, schedule disabled by default, operator/admin approval for promotion, no exchange side effects, and rollback as an explicit guarded action. Browser/auth QA becomes in scope only in Stage `11`.
