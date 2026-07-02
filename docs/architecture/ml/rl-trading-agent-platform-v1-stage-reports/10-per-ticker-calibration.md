---
doc: rl-trading-agent-platform-v1-stage-10-per-ticker-calibration
status: accepted
stage: 10
updated_at: 2026-07-02
---

# Stage 10: per-ticker calibration

Статус: `accepted`.

User required before start: nothing unless a listed prerequisite is not accepted or a required credential/dataset/runtime source is unavailable; never ask for secrets in chat

Stage `10` добавил fail-closed per-ticker/per-market calibration pack для accepted Stage `08M` candidate `stage08m_a3823cbd01143878_fd7c614b` на `binance:futures`. Pack не активирует global-only threshold, не промотит champion, не включает paper/testnet/live/mainnet и не меняет production runtime.

Доказательная граница: `target_host_readiness_pre_main`. Это Mac Studio non-production artifact evidence under `/opt/roehub/state/rl_trading/`, а не `post_main_production_runtime_proof` для changed code.

## Pre-Edit Gate

| Field | Value |
|---|---|
| Prompt path | `.codex/agents/generated/rl-trading-agent-platform-v1/10-per-ticker-calibration.md` |
| Prompt sha256 | `8b9314d49672634fdaecb0119747899e0ee78bd3ee2d3d45e1fad6605da64839` |
| Ledger state observed before work | `current_stage=10`; Stage `10` pending/current |
| Prerequisite verdict | accepted `09B`; accepted `08I3`; accepted `08I4`; accepted corrective `08M` candidate with `stage09_allowed=true`; Stage `10` may proceed |
| `.codex/agents/.context/promt_manager_state.yaml` | read; treated as stale prompt-generation state where it conflicts with current `.codex/AGENTS.md`, prompt and ledger |
| Browser/auth | `N/A`; username `smoke_e2e_keycloak` was not used and `ROEHUB_SMOKE_E2E_PASSWORD` was not read |
| Exchange/provider effects | `N/A`; no exchange SDK, order submit, paper/testnet/live/mainnet path, provider credential, or raw provider payload surface is in scope |

## Scope

Implemented Stage `10` as an additive local calibration artifact surface:

- `per_ticker_calibration.py` builds a versioned calibration pack from accepted `08M` scorecard and candidate manifest lineage;
- `stage10_per_ticker_calibration.py` is an operator-facing CLI that writes pack, registry record and summary under `/opt/roehub/state/rl_trading/`;
- ticker rows are actionable only when final-holdout evidence passes `min_ticker_sessions=10`, `min_ticker_positive_ratio=0.50`, and positive PnL after costs;
- blocked ticker rows record skip reasons and `max_position_fraction_multiplier=0.0`;
- accepted ticker rows expose calibrated confidence, minimum edge after costs, and risk/sizing multipliers;
- normalization reference is hash-only (`raw_values_embedded=false`); raw checkpoint tensors and provider payloads are not embedded.

Not in scope:

- champion promotion or Stage `10A` lifecycle decisions;
- production DB write/apply, `/opt/roehub/app` deploy, production runtime smoke;
- browser/UI/API route changes;
- exchange/paper/testnet/live/mainnet side effects.

## Business Impact

Stage `10` moves the accepted research candidate from one global scorecard into ticker-specific operating constraints. Practically, this means later runtime stages can skip symbols whose own final-holdout evidence is sparse or unstable instead of applying a global threshold to every ticker.

This is still not a user-visible trading feature. Product behavior, billing, entitlements, UI, execution modes and exchange submission remain unchanged. The business value is risk reduction before promotion: weak ticker branches are made explicit and fail closed.

## Calibration Artifact Evidence

| Field | Value |
|---|---|
| Run id | `stage10_macstudio_20260702t213000z` |
| Run dir | `/opt/roehub/state/rl_trading/calibration_packs/stage10_per_ticker_calibration_v1/stage10_macstudio_20260702t213000z` |
| Status | `accepted` |
| Model version id | `stage08m_a3823cbd01143878_fd7c614b` |
| Market scope | `binance:futures` |
| Ticker rows | `323` |
| Accepted/actionable rows | `65` |
| Blocked/fail-closed rows | `258` |
| Calibration pack id | `stage10_stage08m_a3823cbd01143878_fd7c614b_fd7c614b_per_ticker` |
| Calibration pack path | `/opt/roehub/state/rl_trading/calibration_packs/stage10_per_ticker_calibration_v1/stage10_macstudio_20260702t213000z/stage10_per_ticker_calibration_pack.json` |
| Calibration pack sha256 | `7ee51c9f58d8054be97ba2c444a585a99aabbf50ba3ca2e47a78f0d7dbae4219` |
| Calibration pack hash | `7650c16337cb7ea8d95882ca0942c97c5846f65827d573733294e43ce3d19f42` |
| Registry record path | `/opt/roehub/state/rl_trading/calibration_packs/stage10_per_ticker_calibration_v1/stage10_macstudio_20260702t213000z/stage10_calibration_registry_record.json` |
| Registry record sha256 | `c0cb139c4a585fcce2a16d6a17098ddd379655de1bd8bb9f42ffb7b7c5eaa5fd` |
| Summary path | `/opt/roehub/state/rl_trading/calibration_packs/stage10_per_ticker_calibration_v1/stage10_macstudio_20260702t213000z/stage10_per_ticker_calibration_summary.json` |
| Summary sha256 | `d4bfe8aaeb337e5941ba976de5d0fe043cb16469f298820737e03758af401ad6` |
| Summary hash | `31f5607c3e1bcec3543aa1f76d1e0484cec9cb0c841f85576a9b8c4e36e98f15` |
| Global-only threshold active | `false` |
| Raw checkpoint tensors embedded | `false` |
| Normalization raw values embedded | `false` |

Blocked reason counts are overlapping because one ticker can fail more than one gate:

| Reason | Count |
|---|---:|
| `insufficient_ticker_sessions` | `198` |
| `non_positive_ticker_pnl_after_costs` | `112` |
| `ticker_positive_ratio_below_minimum` | `73` |

## Calibration Semantics

| Surface | Stage `10` behavior |
|---|---|
| PnL | Used as `pnl_after_costs` score component; non-positive ticker PnL blocks actionable calibration. |
| Drawdown | `0.0` weight in Stage `10` because accepted `08M` scorecard does not expose drawdown; no hidden proxy is substituted. |
| Turnover | Used as final-holdout session-count evidence proxy; full confidence starts at `30` sessions. |
| Risk | Uses ticker absolute-PnL concentration proxy and fail-closed blocked rows. |
| Action thresholds | Accepted rows set ticker/market `minimum_confidence_to_open` and `minimum_edge_after_costs`; blocked rows set threshold mode `blocked_fail_closed`. |
| Confidence | Accepted rows get score-derived `confidence_multiplier`; blocked rows get `0.0`. |
| Skipped action reasons | Blocked rows include `ticker_calibration_not_accepted` plus concrete blockers such as `insufficient_ticker_sessions`. |
| Risk/sizing | Accepted rows expose `max_position_fraction_multiplier`; blocked rows use `0.0`. |

## Contract Impact

| Dimension | Classification | Notes |
|---|---|---|
| Public API contract | `none` | No API route, HTTP payload, UI read model or browser behavior changed. |
| Port contract | `none` | No existing port/protocol signature changed. |
| DTO schema | `none` | No wire DTO changed. |
| Persisted schema | `none` | No migration/table/storage schema changed in Stage `10`. |
| Config schema/defaults | `none` | No env/YAML/default resolution changed. CLI thresholds are explicit command arguments only. |
| Request hash/cache key/persistence identity | `compatible-change` | Adds new calibration pack identity/hash and registry-record identity under `/opt/roehub/state/rl_trading/`; existing request/cache identities unchanged. |
| Artifact/report semantics | `compatible-change` | Adds versioned calibration pack, registry record and summary artifacts. |
| Service-call auth/timeout/retry/error semantics | `none` | No service call or external adapter changed. |
| External side-effect/idempotency/unknown-state semantics | `none` | No exchange/provider submit, Redis publish, DB write execution or production mutation. |
| Logs/metrics/traces/audit/redaction | `compatible-change` | Adds sanitized calibration artifact summaries and stage report/ledger entries; no secrets, raw provider payloads or raw checkpoint tensors. |
| Alert/runbook semantics | `none` | No alert route, monitoring config, incident workflow or runbook action changed. |
| Benchmark/rollout gates | `compatible-change` | Stage `10A` is now allowed; runtime activation remains blocked by later stages. |
| Browser-visible behavior | `none` | Browser/auth surface is `N/A`. |
| Performance hot path | `none` | Offline artifact generation only; no inference hot path is wired. |

## Conditional Operational Coverage

| Surface | Coverage |
|---|---|
| Service calls | `N/A`; no Roehub API, worker, queue, Redis, ClickHouse write, external provider, exchange SDK or browser service call was added or changed. |
| Timeout/retry/idempotency | `N/A`; no retry loop or side-effecting operation was introduced. |
| Unknown external side-effect state | `N/A`; no provider, money-moving, paper/testnet/live/mainnet or production mutation occurred. |
| Secrets and redaction | No secrets, tokens, cookies, credentials, raw provider payloads, signed requests, raw checkpoint tensors, plaintext operator identifiers or API keys were written. |
| Alerts/monitoring/runbook | `N/A`; no production runtime, scheduler, alert route, notification provider, incident workflow or runbook action changed. |
| Browser/auth | `N/A`; browser-visible behavior and authenticated UI were out of scope. |
| Mac Studio path contract | Runtime artifacts are under `/opt/roehub/state/rl_trading/`; no git command or smoke was run under `/opt/roehub/app`. |

## File Manifest

Created:

| Path | Reason | Contract impact |
|---|---|---|
| `src/trading/contexts/rl_trading/domain/per_ticker_calibration.py` | Stage `10` calibration pack builder, per-ticker fail-closed gates, registry-linked pack metadata, action-threshold/confidence/risk-sizing payload. | `compatible-change` additive internal Python domain surface |
| `scripts/rl_trading/stage10_per_ticker_calibration.py` | Operator-facing calibration pack command using accepted `08M`/`09B` lineage and runtime artifact paths. | `compatible-change` host-local CLI |
| `tests/unit/contexts/rl_trading/domain/test_per_ticker_calibration.py` | Focused tests for accepted/blocked ticker rows, lineage/hash validation, fail-closed semantics and registry record output. | `none` test-only |
| `tests/unit/scripts/rl_trading/test_stage10_per_ticker_calibration.py` | CLI smoke around deterministic Stage `10` fixture inputs. | `none` test-only |
| `docs/architecture/ml/rl-trading-agent-platform-v1-stage-reports/10-per-ticker-calibration.md` | Stage report, evidence and handoff. | `compatible-change` docs/report |

Modified:

| Path | Reason | Contract impact |
|---|---|---|
| `src/trading/contexts/rl_trading/domain/__init__.py` | Export Stage `10` calibration domain surface. | `compatible-change` additive internal Python exports |
| `docs/architecture/ml/rl-trading-agent-platform-v1-stage-reports/rl-trading-agent-platform-v1-stage-ledger.md` | Mark Stage `10` accepted and open Stage `10A`. | `compatible-change` docs/ledger |
| `docs/architecture/README.md` | Docs index sync after adding this report. | `compatible-change` docs index |

Deleted: none.

Outside expected paths: none.

Runtime artifacts created on `macstudio` under `/opt/roehub/state/rl_trading/`:

| Artifact | sha256 |
|---|---|
| `/opt/roehub/state/rl_trading/calibration_packs/stage10_per_ticker_calibration_v1/stage10_macstudio_20260702t213000z/stage10_per_ticker_calibration_pack.json` | `7ee51c9f58d8054be97ba2c444a585a99aabbf50ba3ca2e47a78f0d7dbae4219` |
| `/opt/roehub/state/rl_trading/calibration_packs/stage10_per_ticker_calibration_v1/stage10_macstudio_20260702t213000z/stage10_calibration_registry_record.json` | `c0cb139c4a585fcce2a16d6a17098ddd379655de1bd8bb9f42ffb7b7c5eaa5fd` |
| `/opt/roehub/state/rl_trading/calibration_packs/stage10_per_ticker_calibration_v1/stage10_macstudio_20260702t213000z/stage10_per_ticker_calibration_summary.json` | `d4bfe8aaeb337e5941ba976de5d0fe043cb16469f298820737e03758af401ad6` |

Foreign dirty files observed before Stage `10` and not owned by this stage: Stage `09B` files/report/runbook and pre-existing docs/ledger/README changes. They were preserved and not reverted.

## Quality Gates And Evidence

| Gate | Result |
|---|---|
| Focused pytest | passed: `uv run pytest -q tests/unit/contexts/rl_trading/domain/test_per_ticker_calibration.py tests/unit/scripts/rl_trading/test_stage10_per_ticker_calibration.py` -> `4 passed` |
| Focused ruff | passed: `uv run ruff check src/trading/contexts/rl_trading/domain/per_ticker_calibration.py src/trading/contexts/rl_trading/domain/__init__.py scripts/rl_trading/stage10_per_ticker_calibration.py tests/unit/contexts/rl_trading/domain/test_per_ticker_calibration.py tests/unit/scripts/rl_trading/test_stage10_per_ticker_calibration.py` |
| Focused pyright | passed: `uv run pyright src/trading/contexts/rl_trading/domain/per_ticker_calibration.py src/trading/contexts/rl_trading/domain/__init__.py scripts/rl_trading/stage10_per_ticker_calibration.py tests/unit/contexts/rl_trading/domain/test_per_ticker_calibration.py tests/unit/scripts/rl_trading/test_stage10_per_ticker_calibration.py` -> `0 errors` |
| Prompt-level ruff | passed: `uv run ruff check src/trading/contexts/rl_trading apps tests` |
| Prompt-level pyright | passed: `uv run pyright src/trading/contexts/rl_trading apps tests` -> `0 errors` |
| Prompt-level pytest | passed: `uv run pytest -q tests/unit/contexts/rl_trading tests/unit/apps` -> `458 passed, 3 warnings` |
| Mac Studio focused pytest | passed: `uv run pytest -q tests/unit/contexts/rl_trading/domain/test_per_ticker_calibration.py tests/unit/scripts/rl_trading/test_stage10_per_ticker_calibration.py` -> `4 passed` |
| Mac Studio calibration run | passed: `uv run python scripts/rl_trading/stage10_per_ticker_calibration.py --run-id stage10_macstudio_20260702t213000z --generated-at-utc 2026-07-02T21:30:00Z` -> `status=accepted`, `65` accepted rows, `258` blocked rows |
| Mac Studio artifact sanity check | passed: registry hash matches pack hash, `global_only_threshold_activated=false`, `raw_checkpoint_tensors_embedded=false`, `normalization_raw_values_embedded=false` |
| Docs index | passed after final report/ledger update: `uv run python -m tools.docs.generate_docs_index --check` |

## Residual Risks

- Stage `10` accepts a calibration pack, not production activation. Stage `10A`, runtime inference, paper/testnet/live and mainnet gates remain closed.
- Drawdown is not available in the accepted `08M` scorecard. Stage `10` records drawdown weight `0.0` rather than inventing a proxy.
- `258/323` ticker rows are fail-closed; later stages must preserve these skip reasons instead of falling back to a global threshold.
- Delivery state is `local-only` plus synced Mac Studio checkout snapshot for `target_host_readiness_pre_main` artifact evidence. No `origin/main` commit, CI/deploy, `/opt/roehub/app` deploy or production runtime validation is claimed by this stage.

## Cold Self-Review

Mode: `cold self-review fallback`. Independent subagent review was not used because subagent spawning requires an explicit user request.

Verdict: `Release`.

Checked:

- stage continuity: Stage `09B` accepted, Stage `10` accepted, and Stage `10A` is now the next executable stage;
- proof boundary: Mac Studio evidence is `target_host_readiness_pre_main`, not production validation for the target revision; `post_main_production_runtime_proof` would require `main`, green CI/GitHub Actions, deploy or verified sync into `/opt/roehub/app`, and production runtime smoke;
- contract impact: new surface is additive local CLI/artifact metadata; API/UI/exchange/persisted DB contracts unchanged;
- secrets/redaction: implementation records paths/hashes/counts and does not embed raw checkpoint tensors, provider payloads or credentials;
- validation: focused and prompt-level local gates passed; Mac Studio focused tests, calibration run and artifact sanity check passed.

## Next-Stage Handoff

Stage `10` is accepted. Ledger `current_stage` moves to `10A`.

Next allowed prompt: `.codex/agents/generated/rl-trading-agent-platform-v1/10a-retraining-promotion-lifecycle.md`.

Stage `10A` must consume the calibration pack above, preserve fail-closed ticker rows, and decide candidate/champion promotion only through its own gates. It must not treat Stage `10` as production runtime proof, paper/testnet/live readiness or mainnet readiness.
