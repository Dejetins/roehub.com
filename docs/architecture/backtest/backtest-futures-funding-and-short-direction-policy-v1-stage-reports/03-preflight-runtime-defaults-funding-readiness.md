# Backtest Futures Funding And Short Direction Policy v1 - Stage 03 Preflight Runtime Defaults Funding Readiness

Runtime defaults, request normalization and preflight funding-readiness
implementation record.

Date: 2026-06-22

Status: local implementation complete; Mac Studio
`target_host_readiness_pre_main` is now collected and
`read_only_existing_runtime_smoke` observed the already deployed runtime health
and auth boundary. Authenticated Stage `03` route smoke for the changed code is
not claimed because the local changes have not been committed, delivered to
`main`, deployed or synced into `/opt/roehub/app`. No
`post_main_production_runtime_proof` is claimed.

Execution branch policy: `main` by default; do not create branches, worktrees or
local workflow folders unless the user explicitly requests them.

`User required before start: nothing`

## Scope

Requested scope:

- add server-side direction/market compatibility to backtest runtime defaults;
- add standalone `short` to supported preflight direction modes;
- normalize `execution.funding.mode` and `execution.funding.coverage_policy`;
- reject new spot short-like create/preflight requests with
  `short_direction_requires_futures_market`;
- expose preflight `funding_readiness` without hard-blocking degraded or
  unavailable futures coverage;
- keep existing persisted job read paths additive and immutable.

## Business Impact Layer

Stage `03` делает backtest create/preflight честнее для пользователя: spot больше
не может молча запускать short-like режим, который не соответствует реальному
рынку. Futures-запросы получают машинно-читаемую готовность funding, поэтому UI
и следующие stages смогут показывать предупреждение о degraded/unavailable
funding без парсинга текста и без блокировки создания backtest job.

Existing persisted jobs are not mutated. Historical spot short-like jobs remain
readable through existing read models; this stage only changes new preflight and
create normalization.

## Previous-Stage Gate

The Stage `03` prompt requires Stage `02` to be `accepted` in the stage
execution ledger before implementation starts.

Observed ledger state:

- Stage `02` status: `accepted`.
- Accepted evidence: local funding artifact implementation, coverage reader,
  artifact publish/load tests, backtest/artifact test suite, affected
  app/market-data tests, docs index check and Mac Studio
  `target_host_readiness_pre_main` funding-table smoke.

Decision: previous-stage gate is satisfied. Stage `03` implementation may
proceed.

## File Manifest

Created:

- `docs/architecture/backtest/backtest-futures-funding-and-short-direction-policy-v1-stage-reports/03-preflight-runtime-defaults-funding-readiness.md`

Modified:

- `apps/api/dto/backtests.py`
- `src/trading/contexts/backtest/adapters/outbound/artifacts_fs/artifact_context_resolver.py`
- `src/trading/contexts/backtest/application/dto/runtime_preflight.py`
- `src/trading/contexts/backtest/application/services/v2/__init__.py`
- `src/trading/contexts/backtest/application/services/v2/preflight.py`
- `tests/unit/contexts/backtest/application/services/v2/test_backtest_preflight_service.py`
- `tests/unit/apps/api/test_api_torch_dependency_isolation.py`
- `tests/unit/apps/api/test_app_strategy_router_toggle.py`
- `tests/unit/apps/api/test_backtests_routes.py`
- `tests/unit/apps/api/test_ui_backtests_routes.py`
- `scripts/backtest/run_iteration_5_tp_sl_hit_times_benchmark.py`
- `.codex/hooks/validators/branch_workflow_guard.py`
- `.codex/hooks/validators/prompt_pack_branch_policy_guard.py`
- `docs/architecture/backtest/backtest-futures-funding-and-short-direction-policy-v1-stage-reports/backtest-futures-funding-and-short-direction-policy-v1-stage-ledger.md`
- `docs/architecture/README.md`

Deleted:

- none

## Working Tree Note

The checkout already contained uncommitted Stage `02` files before Stage `03`
implementation began. Stage `03` changes are scoped to the manifest above and
do not revert or sweep unrelated local changes.

## Request Contract

- `direction_modes` now includes `short` in addition to `long_only` and
  `long_short_reversal`.
- New `spot + short` and `spot + long_short_reversal` preflight/create requests
  are rejected with issue code `short_direction_requires_futures_market` under
  the existing `backtest.invalid_request` error envelope.
- `execution.funding` is normalized into every new request:
  - spot requests normalize to `{"mode": "off", "coverage_policy": "degraded_with_warning"}`;
  - futures requests default to
    `{"mode": "include_when_futures", "coverage_policy": "degraded_with_warning"}`;
  - explicit futures `mode=off` remains valid and changes the normalized
    request hash for new jobs.
- The request hash changes for new jobs because normalized funding config and
  updated runtime-default compatibility metadata are now part of the normalized
  request/config identity.
- Existing persisted jobs are not rehashed or migrated.

## Preflight Contract

- `GET /backtests/runtime-defaults` now returns
  `direction_market_compatibility`, including server-side spot/futures direction
  policy and per-market funding defaults.
- `POST /backtests/preflight` now returns:
  - `funding_readiness.status`;
  - `funding_readiness.warning_codes`;
  - `funding_readiness.coverage_ratio`;
  - `funding_readiness.window`;
  - `funding_readiness.funding_manifest_hash` when the selected artifact
    manifest declares one;
  - selected-state `direction_market_compatibility`.
- Funding readiness is derived from the selected artifact root manifest summary
  populated by `FilesystemBacktestArtifactContextResolver`; the API route does
  not call ClickHouse directly.
- Spot and `funding.mode=off` return `funding_readiness.status=not_applicable`.
- Futures manifests with ready/degraded/unavailable funding are reported as
  ready/degraded/unavailable. Degraded and unavailable statuses add preflight
  warnings, but creation proceeds under `degraded_with_warning`.

## Conditional Service-Call Coverage

- Service calls: N/A for new provider calls. Stage `03` does not call Binance,
  Bybit or ClickHouse from the API route.
- Artifact filesystem reads: applicable through the existing trusted artifact
  context resolver. The resolver now copies the funding manifest summary into
  additive preflight metadata.
- API route boundary: covered by local TestClient route smoke and unit route
  tests. Mac Studio `target_host_readiness_pre_main` now proves host access,
  checkout visibility and loopback API availability. Mac Studio
  `read_only_existing_runtime_smoke` observed the already deployed runtime only;
  it did not authenticate into `/backtests/runtime-defaults` and does not prove
  the local Stage `03` code in `/opt/roehub/app`.
- Browser flow: N/A. The prompt explicitly marks browser QA as false and no web
  UI files were changed.

## Logging And Redaction Coverage

Stage `03` adds no new logs, metrics labels, traces or raw provider payload
emission. Reports may include status codes, reason-code literals and aggregate
test results only. No DSNs, Keycloak secrets, cookies, bearer tokens,
ClickHouse passwords or provider payloads are recorded.

## Alerts Monitoring Runbook Coverage

N/A for this stage. Funding freshness metrics and alerts remain owned by Stage
`01`; Stage `03` only exposes readiness metadata from artifacts through the API
contract. No Prometheus rules or runbooks are changed here.

## Validation

Passed:

- `uv run pytest -q tests/unit/contexts/backtest/application/services/v2/test_backtest_preflight_service.py`
  passed with `33 passed`.
- `uv run pytest -q tests/unit/apps/api/test_backtests_routes.py` passed with
  `80 passed`.
- `uv run pytest -q tests/unit/apps/api/test_ui_backtests_routes.py` passed with
  `6 passed`.
- Focused combined route/service suite passed with `119 passed`.
- `uv run ruff check src/trading/contexts/backtest apps/api tests` passed.
- `uv run pyright src/trading/contexts/backtest apps/api tests` passed with
  `0 errors`.
- Initial exact broad pytest command reached completion but failed three API
  app-import tests because identity fail-fast leaked into tests that are not
  about identity configuration. The test isolation was fixed by forcing
  `IDENTITY_FAIL_FAST=false` in those app-import tests.
- Final exact broad gate
  `uv run pytest -q tests/unit/contexts/backtest tests/unit/apps/api` passed
  with `588 passed`.
- `python -m tools.docs.generate_docs_index --check` passed.
- Publish readiness full gates passed before direct-main delivery:
  - `uv run ruff check .`;
  - `uv run pyright`;
  - `uv run pytest -q -ra` with `1304 passed, 3 warnings`;
  - `uv run python -m tools.docs.generate_docs_index --check`.

Delivery gate fixes made before publish:

- `scripts/backtest/run_iteration_5_tp_sl_hit_times_benchmark.py` now delegates
  `load_funding_arrays` in its failing-loader probe so it satisfies the expanded
  `BacktestArtifactArrayLoader` protocol.
- `.codex/hooks/validators/branch_workflow_guard.py` and
  `.codex/hooks/validators/prompt_pack_branch_policy_guard.py` received
  formatting-only fixes required by full `ruff check .`; hook behavior is
  unchanged.

## Real-Boundary Evidence

Proof-boundary rule for this report:

- `target_host_readiness_pre_main` can only prove host reachability or current
  service availability before delivery.
- `read_only_existing_runtime_smoke` can only observe the already deployed
  runtime before delivery.
- `post_main_production_runtime_proof` requires `main`, green
  `CI/GitHub Actions`, deploy/sync into `/opt/roehub/app`, and then API/runtime
  smoke from the Mac Studio runtime. This stage has not met those prerequisites.

Local API development smoke:

- `GET /backtests/runtime-defaults` through FastAPI TestClient returned status
  `200`, directions `['long_only', 'short', 'long_short_reversal']`, and spot
  rejection codes for `short` and `long_short_reversal` equal to
  `short_direction_requires_futures_market`.
- `POST /backtests/preflight` through FastAPI TestClient returned status `200`
  for funding readiness statuses `not_applicable`, `ready`, `degraded` and
  `unavailable`; degraded returned `['missing_trailing_coverage']`, unavailable
  returned `['no_funding_rows']`.

Target host:

- First `target_host_readiness_pre_main` attempt failed with
  `ssh: connect to host 100.74.213.43 port 22: Operation timed out`.
- Retry after access restoration succeeded on `2026-06-22 18:11:36 MSK`:
  `ssh macstudio` reached host `MacStudioDaniil`; the Mac Studio checkout at
  `/Users/daniildegtyarev/Projects/roehub.com` was on branch `main` with
  `HEAD=bf2723dc` and remote-tracking `origin/main=bf2723dc`;
  `/opt/roehub/app` existed.
- `read_only_existing_runtime_smoke` against Mac Studio loopback observed the
  already deployed runtime: `GET http://127.0.0.1:8000/health` returned `200`
  with `{"status":"ok"}`; `GET /auth/current-user` returned `401` with
  `missing_session_id`; unauthenticated `GET /backtests/runtime-defaults`
  returned `401` with Roehub error code `auth.required`.
- No authenticated Mac Studio `/backtests/runtime-defaults` or
  `/backtests/preflight` smoke is claimed for the local Stage `03` changes
  because those changes have not been committed, delivered, deployed or synced
  into `/opt/roehub/app`.
- `post_main_production_runtime_proof` is explicitly not claimed. For this stage
  family it would require the target revision to be on `main`, relevant GitHub
  Actions/CI and deploy workflows to be green, the Mac Studio checkout/runtime
  to be deployed or synced from that `main` revision into `/opt/roehub/app`,
  and only then API/runtime smoke against Mac Studio loopback.

## Contract Impact

- public API contract: `compatible-change` for additive response fields;
  intentional validation change for new spot short-like create/preflight
  requests.
- DTO schema: `compatible-change`; runtime defaults, preflight result and API
  DTOs add optional/additive funding and compatibility mappings.
- persisted schema: `none`; no database migration and no persisted job rewrite.
- config schema: `none`; no runtime config file changed.
- request hash / persistence identity: `compatible-change`; new jobs include
  normalized funding config in the request hash, while old jobs remain
  immutable.
- port contract: `compatible-change`; artifact context metadata is additive.
- service-call auth/timeout/retry/error semantics: `none`; no new service call.
- logs, metrics, traces, audit, ledger, report, redaction semantics:
  `compatible-change` for stage report/ledger wording only; no runtime log or
  metric contract changed.
- alert/runbook semantics: `none`.
- browser-visible behavior: `none` in this stage; server-provided compatibility
  metadata is ready for later UI work.

## Cold-Head Review Receipt

Cold-head review: completed
Mode: cold self-review fallback
Review scope: Stage `03` report, ledger update plan, runtime-defaults/preflight
contract, request hash impact, `target_host_readiness_pre_main`,
`read_only_existing_runtime_smoke`, `post_main_production_runtime_proof`
boundary labels and validation evidence.
Review instructions:
`architecture-review/references/cold-head-plan-prompt-pack-review.md`
Verdict: Release after fixes
Blockers fixed: added business-readable impact layer; added conditional
service-call, logging/redaction and alerts/monitoring sections; separated local
API smoke from Mac Studio `target_host_readiness_pre_main`,
`read_only_existing_runtime_smoke` and unavailable authenticated route proof for
the changed code; documented request hash and persisted-job compatibility.
Local follow-up check: completed
Residual risks: Mac Studio host access and unauthenticated runtime health/auth
boundary are proven, but authenticated Mac Studio route smoke for the changed
Stage `03` code is still not proven because the changes are local and not
deployed; standalone `short` is accepted at preflight for futures, while
funding PnL and UI/browser CJM remain later-stage work.

## Residual Risks

- Mac Studio `target_host_readiness_pre_main` and unauthenticated
  `read_only_existing_runtime_smoke` are now collected, but authenticated
  `/backtests/runtime-defaults` and `/backtests/preflight` route smoke for the
  changed Stage `03` code is still unavailable until the changes are delivered
  and deployed or another safe authenticated target-smoke path is provided.
- No `post_main_production_runtime_proof` is claimed because this local change
  has not been committed, pushed, deployed or synced into `/opt/roehub/app`.
- `short` is now a normalized futures direction in preflight; downstream funding
  PnL, result fields, UI/browser warnings and launch CJM remain owned by later
  stages.
