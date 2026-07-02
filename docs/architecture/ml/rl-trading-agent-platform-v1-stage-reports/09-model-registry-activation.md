---
doc: rl-trading-agent-platform-v1-stage-09-model-registry-activation
status: accepted
stage: 09
updated_at: 2026-07-02
---

# Stage 09: model registry and activation gates

Статус: `accepted`.

User required before start: nothing unless a listed prerequisite is not accepted or a required credential/dataset/runtime source is unavailable; never ask for secrets in chat

## Pre-Edit Gate

| Field | Value |
|---|---|
| Prompt path | `.codex/agents/generated/rl-trading-agent-platform-v1/09-model-registry-activation.md` |
| Prompt sha256 | `c62526fce0b484fa53c84b3fe901cd021d13fb4b856241f5edd6157c51bd5a43` |
| Ledger state observed before work | `current_stage=09`; Stage `09` pending; `09B` blocked until `09` accepted |
| Prerequisite verdict | accepted `08I3`, accepted `08I4`, accepted `08J`, blocked `08K` superseded by accepted `08M` corrective candidate with `stage09_allowed=true` |
| Accepted Stage `09` input candidate | `stage08m_a3823cbd01143878_fd7c614b` |
| Accepted candidate manifest sha256 | `9e2767ead0b697d0194e501aa7932b44fc1f5d1b180713f1270c81d1c887a69c` |
| Browser/auth | `N/A`; username `smoke_e2e_keycloak` was not used and `ROEHUB_SMOKE_E2E_PASSWORD` was not read |
| Exchange/provider effects | `N/A`; no exchange SDK, order submit, paper/testnet/live/mainnet path, or provider credential surface is in scope |
| `.codex/agents/.context/promt_manager_state.yaml` | read but treated as stale prompt-generation state where it conflicts with current repo/prompt direct-main/local-only delivery policy |

## Planned Concrete File List Before Edit

| Path | Planned state | Reason |
|---|---|---|
| `src/trading/contexts/rl_trading/domain/model_registry.py` | create | Stage `09` registry state machine, activation gates, lifecycle cleanup planning, accepted-hash checkpoint loader. |
| `src/trading/contexts/rl_trading/domain/__init__.py` | modify | Export the Stage `09` registry domain surface for local tests and later application use cases. |
| `alembic/versions/20260702_0040_rl_trading_model_registry_v1.py` | create | Additive Postgres metadata schema for datasets, training runs, model versions, calibration packs, activation audit and lifecycle policy. |
| `tests/unit/contexts/rl_trading/domain/test_model_registry.py` | create | State-machine, activation invariant, lifecycle cleanup and checkpoint hash/path tests. |
| `tests/unit/apps/migrations/test_rl_trading_model_registry_sql.py` | create | Migration shape, additive safety and redaction assertions. |
| `docs/architecture/ml/rl-trading-agent-platform-v1-stage-reports/09-model-registry-activation.md` | create/modify | Stage report, prompt hash, file manifest, evidence, contract impact and handoff. |
| `docs/architecture/ml/rl-trading-agent-platform-v1-stage-reports/rl-trading-agent-platform-v1-stage-ledger.md` | modify | Record Stage `09` status/evidence and next-stage allowance after validation. |
| `docs/architecture/README.md` | modify only if docs index check requires regeneration | Docs index sync after adding the Stage `09` report. |

Initial blockers: none observed. Real-boundary evidence required before acceptance: migration shape checks, registry state-machine tests, checkpoint hash/path validation and docs index.

## Scope

Stage `09` реализовал локальные registry/activation gates вокруг accepted Stage `08M` candidate `stage08m_a3823cbd01143878_fd7c614b`. Входит:

- additive Postgres metadata schema для dataset versions, training runs, model versions, calibration packs, activations, audit events и lifecycle policy;
- domain/use-case helpers для registry state machine, candidate handoff guard, activation readiness, activation audit, lifecycle cleanup planning и missing-artifact marking;
- trusted local checkpoint loader: canonical path under `/opt/roehub/state/rl_trading/`, file existence, sha256 validation before `torch.load`, and `weights_only=True` when supported;
- deterministic tests for invalid transitions, corrupt/missing checkpoint hash, rollback explicit selection, lifecycle cleanup retention and migration SQL guardrails.

Не входит:

- production DB migration apply;
- `/opt/roehub/app` deploy or runtime smoke;
- registry write against production data;
- active model load in production;
- backup/restore drill;
- paper/testnet/live/mainnet readiness.

## Business Impact

Stage `09` переводит accepted research candidate из "можно передать в следующий gate" в безопасный registry contract: модель и calibration не могут быть активированы без matching hashes, accepted states, audit payload and local artifact checks. Для пользователя это пока не новая trading-функция, а защита от неаудируемой загрузки checkpoint, silent artifact deletion и auto-activation candidate модели.

## Registry Contract Evidence

| Field | Value |
|---|---|
| Registry contract hash | `ec7b6bcbf6c6eecb7a0597c66594ec46669f4e6a9735455a0368526a184b29f1` |
| Accepted input candidate | `stage08m_a3823cbd01143878_fd7c614b` |
| Accepted input manifest sha256 | `9e2767ead0b697d0194e501aa7932b44fc1f5d1b180713f1270c81d1c887a69c` |
| Runtime artifact root | `/opt/roehub/state/rl_trading/` |
| New runtime artifacts | none |
| Proof boundary | local schema/domain/checkpoint-boundary evidence only; not `post_main_production_runtime_proof` |

Proof-boundary separation:

| Boundary label | Stage `09` status | Meaning |
|---|---|---|
| `target_host_readiness_pre_main` | not collected | No Mac Studio host-readiness command was required for this local schema/domain stage. Existing Stage `08M` candidate artifacts remain the accepted input evidence, but Stage `09` did not mutate `/opt/roehub/state/rl_trading/`. |
| `read_only_existing_runtime_smoke` | not collected | Existing `/opt/roehub/app` production runtime was not smoked or used as evidence. |
| `post_main_production_runtime_proof` | not collected | Changed-code production proof requires the target revision on `main`, green CI/GitHub Actions, deploy or verified sync from the `main` checkout into `/opt/roehub/app`, and then the relevant runtime smoke from that production runtime tree. Stage `09` claims none of that. |

## File Manifest

| Path | State | Reason | Contract impact |
|---|---|---|---|
| `src/trading/contexts/rl_trading/domain/model_registry.py` | created | Stage `09` registry state machine, accepted candidate guard, activation gate/audit helpers, cleanup planner and trusted checkpoint loader. | `compatible-change` additive internal Python domain surface |
| `src/trading/contexts/rl_trading/domain/__init__.py` | modified | Export the Stage `09` registry domain surface. | `compatible-change` additive internal Python exports |
| `alembic/versions/20260702_0040_rl_trading_model_registry_v1.py` | created | Additive Postgres metadata schema for RL datasets, runs, models, calibrations, activations, audit and lifecycle policy. | `compatible-change` additive persisted schema |
| `tests/unit/contexts/rl_trading/domain/test_model_registry.py` | created | Focused coverage for state transitions, activation invariants, checkpoint hash/path gate and lifecycle cleanup plan. | `none` test-only |
| `tests/unit/apps/migrations/test_rl_trading_model_registry_sql.py` | created | SQL migration guardrails: additive tables, checks, indexes and redaction vocabulary. | `none` test-only |
| `docs/architecture/ml/rl-trading-agent-platform-v1-stage-reports/09-model-registry-activation.md` | created/modified | Stage report, file manifest, evidence, contract impact and handoff. | `compatible-change` docs/report |
| `docs/architecture/ml/rl-trading-agent-platform-v1-stage-reports/rl-trading-agent-platform-v1-stage-ledger.md` | modified | Mark `09` accepted locally, advance `current_stage=09B`, record next-stage blocker boundary. | `compatible-change` docs/ledger |
| `docs/architecture/ml/rl-trading-agent-platform-v1.md` | modified | Sync plan narrative and stage table with accepted Stage `09`. | `compatible-change` docs/plan |
| `docs/architecture/README.md` | modified | Docs index regenerated after adding this report. | `compatible-change` docs index |

Deleted: none.

Outside expected paths: none.

Runtime artifact manifest: none created by Stage `09`; Stage `09` only references existing Stage `08M` non-production candidate artifacts already recorded in the `08M` report and ledger.

## Contract Impact

| Dimension | Classification | Notes |
|---|---|---|
| Public API contract | `none` | No API route, HTTP payload or UI read model changed. |
| Port contract | `compatible-change` | Adds internal domain/use-case helpers; no existing port signature changed. |
| DTO schema | `none` | No wire DTO changed. |
| Persisted schema | `compatible-change` | Adds new `rl_trading_*` registry metadata tables and indexes; no existing table or constraint changed. |
| Config schema/defaults | `none` | No env/YAML/default resolution changed; lifecycle policy is persisted metadata for future startup guard, not a config default change. |
| Request hash/cache key/persistence identity | `compatible-change` | Introduces model/calibration/dataset/activation identity hashes for new registry records only. Existing request/cache identities are unchanged. |
| Service-call auth/timeout/retry/error semantics | `none` | No service call or external adapter changed. |
| External side-effect/idempotency/unknown-state semantics | `none` | No exchange/provider submit, no Redis publish, no DB write execution, no production mutation. |
| Logs/metrics/traces/audit/reports/redaction | `compatible-change` | Adds sanitized activation audit payload shape and stage report/ledger entries; operator reference is hash-only. |
| Alerts/runbook semantics | `none` | No alert route, monitoring config, incident workflow or runbook action changed in this stage. |
| Benchmark/rollout gates | `compatible-change` | Advances ledger from `09` to `09B`; does not open runtime activation. |
| Browser-visible behavior | `none` | Browser/auth surface is `N/A`. |
| Performance hot path | `none` | Offline registry/checkpoint guards only; no inference hot path is wired yet. |

## Conditional Operational Coverage

| Surface | Coverage |
|---|---|
| Service calls | `N/A`; no Roehub API, worker, Redis, ClickHouse, external provider, exchange SDK or browser service call was added or changed. |
| Timeout/retry/idempotency | `N/A`; no retry loop or side-effecting operation was introduced. New activation identity is deterministic metadata for later use. |
| Unknown external side-effect state | `N/A`; no provider or money-moving call occurred. |
| Secrets and redaction | No secrets, tokens, cookies, credentials, raw provider payloads, signed requests, raw checkpoint tensors or plaintext operator identifiers were written. |
| Alerts/monitoring/runbook | `N/A` for runtime wiring; lifecycle policy and audit schema are ready for later stages, but no alert/Monit/Prometheus config changed. |
| Browser/auth | `N/A`; username `smoke_e2e_keycloak` was not used and `ROEHUB_SMOKE_E2E_PASSWORD` was not read. |
| Mac Studio path contract | New code enforces `/opt/roehub/state/rl_trading/` as default artifact root; no git command or smoke was run under `/opt/roehub/app`. |

## Quality Gates And Evidence

| Gate | Result |
|---|---|
| Focused pytest | passed: `uv run pytest -q tests/unit/contexts/rl_trading/domain/test_model_registry.py tests/unit/apps/migrations/test_rl_trading_model_registry_sql.py` -> `11 passed` |
| Focused ruff | passed: `uv run ruff check src/trading/contexts/rl_trading/domain/model_registry.py tests/unit/contexts/rl_trading/domain/test_model_registry.py tests/unit/apps/migrations/test_rl_trading_model_registry_sql.py alembic/versions/20260702_0040_rl_trading_model_registry_v1.py` |
| Focused pyright | passed: `uv run pyright src/trading/contexts/rl_trading/domain/model_registry.py tests/unit/contexts/rl_trading/domain/test_model_registry.py tests/unit/apps/migrations/test_rl_trading_model_registry_sql.py` -> `0 errors` |
| Prompt-level ruff | passed: `uv run ruff check src/trading/contexts/rl_trading apps tests` |
| Prompt-level pyright | passed: `uv run pyright src/trading/contexts/rl_trading apps tests` -> `0 errors` |
| Prompt-level pytest | passed: `uv run pytest -q tests/unit/contexts/rl_trading tests/unit/apps` -> `452 passed, 3 warnings` |
| Alembic offline SQL render | passed: `uv run alembic upgrade 20260702_0039:20260702_0040 --sql` rendered the new registry migration without DB credentials |
| Checkpoint boundary | covered by focused test: path must stay under artifact root, file must exist, sha256 must match before `torch.load`, candidate status is not loadable, rollback candidate needs explicit selection |
| Docs index | passed after regeneration: `uv run python -m tools.docs.generate_docs_index`; `uv run python -m tools.docs.generate_docs_index --check` -> OK |

## Residual Risks

- Stage `09` did not apply migration to a live Postgres database; that remains delivery/deploy evidence for a later publish/deploy path.
- Stage `09B` still must prove local backup/restore against accepted registry metadata and artifacts before calibration/runtime activation can proceed.
- `weights_only=True` is used when supported; fallback is documented in code path and only occurs after local path/hash validation.
- The accepted `08M` candidate is still a research candidate wrapped by registry gates, not a production champion, paper/testnet/live model or mainnet model.

## Cold Self-Review

Mode: `cold self-review fallback`. Independent subagent review was not used because the available multi-agent tool policy requires an explicit user request before spawning subagents.

Verdict: `Release`.

Checked:

- stage continuity: ledger now has `current_stage=09B`, Stage `09` accepted and Stage `09B` runnable;
- proof boundaries: local evidence only, no `read_only_existing_runtime_smoke`, no `post_main_production_runtime_proof`, and post-main proof conditions explicitly require `main`, green CI/GitHub Actions, deploy or verified sync, then runtime smoke;
- contract impact: persisted schema is additive, API/UI/execution/exchange/browser surfaces are unchanged;
- secrets/redaction: no raw provider payloads, credentials, raw checkpoint tensors or plaintext operator identifiers are written;
- validation: focused and prompt-level backend gates passed, Alembic offline SQL render passed, docs index passed after regeneration.

No blocking issue remains. Residual risks are listed above and are handed to Stage `09B`/publish-deploy paths rather than hidden in Stage `09`.

## Next-Stage Handoff

Next executable prompt: `.codex/agents/generated/rl-trading-agent-platform-v1/09b-local-artifact-backup-restore.md`.

Stage `09B` may run now. It must use the Stage `09` registry metadata contract and prove backup/restore for accepted metadata/artifacts. It must not treat Stage `09` as production DB apply, active model load, runtime activation, paper/testnet/live readiness or mainnet readiness.
