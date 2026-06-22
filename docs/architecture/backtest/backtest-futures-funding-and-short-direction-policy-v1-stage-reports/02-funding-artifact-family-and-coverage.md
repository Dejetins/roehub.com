# Backtest Futures Funding And Short Direction Policy v1 - Stage 02 Funding Artifact Family And Coverage

Funding artifact publishing, loading, manifest identity and coverage-readiness
implementation record.

Date: 2026-06-22

Status: accepted for local Stage `02` implementation evidence. This is not
`post_main_production_runtime_proof`: this turn did not commit, push, deploy or
sync changed code into `/opt/roehub/app`.

Execution branch policy: `main` by default; do not create branches, worktrees or
local workflow folders unless the user explicitly requests them.

`User required before start: nothing`

## Scope

Requested scope:

- add funding as a first-class artifact family for futures artifact roots;
- publish and load funding arrays from scheduler-maintained
  `canonical_funding_rates`;
- include funding in root manifest identity and expose `funding_manifest_hash`;
- add explicit coverage statuses: `ready`, `degraded`, `unavailable`,
  `not_applicable`;
- keep spot funding `not_applicable`;
- prove filesystem artifact publish/load smoke.

## Previous-Stage Gate

The Stage `02` prompt requires Stage `01` to be `accepted` in the stage
execution ledger before implementation starts.

Observed ledger state:

- Stage `01` status: `accepted`.
- Delivered main revision: `a77c001c375b101af4ddca51f63c7d6da60e21ea`.
- Accepted evidence includes green GitHub CI/deploy workflows and Mac Studio
  `post_main_production_runtime_proof` for live ClickHouse funding writes plus
  exported `scheduler_funding_catchup_*` metrics.

Decision: previous-stage gate is satisfied. Stage `02` implementation may
proceed.

## File Manifest

Created:

- `docs/architecture/backtest/backtest-futures-funding-and-short-direction-policy-v1-stage-reports/02-funding-artifact-family-and-coverage.md`
- `src/trading/contexts/market_data/application/ports/stores/funding_rate_coverage_reader.py`

Modified:

- `apps/cli/commands/backtest_artifact_publish.py`
- `src/trading/contexts/backtest/application/ports/artifact_arrays.py`
- `src/trading/contexts/backtest/adapters/outbound/artifacts_fs/artifact_array_loader.py`
- `src/trading/contexts/backtest/adapters/outbound/artifacts_fs/path_builder.py`
- `src/trading/contexts/backtest_artifacts/adapters/outbound/artifacts_fs/path_builder.py`
- `src/trading/contexts/backtest_artifacts/application/services/v2/artifact_manifest_loader.py`
- `src/trading/contexts/backtest_artifacts/application/services/v2/artifact_manifest_validator.py`
- `src/trading/contexts/backtest_artifacts/application/services/v2/artifact_precompute_runner.py`
- `src/trading/contexts/backtest_artifacts/application/services/v2/contracts.py`
- `src/trading/contexts/backtest_artifacts/application/use_cases/publish_backtest_artifacts_v2.py`
- `src/trading/contexts/market_data/adapters/outbound/persistence/clickhouse/funding_rate_store.py`
- `src/trading/contexts/market_data/application/ports/stores/__init__.py`
- `docs/architecture/backtest/backtest-futures-funding-and-short-direction-policy-v1-stage-reports/backtest-futures-funding-and-short-direction-policy-v1-stage-ledger.md`
- `docs/architecture/README.md`
- `tests/unit/contexts/backtest/adapters/outbound/artifacts_fs/test_artifact_array_loader.py`
- `tests/unit/contexts/backtest/application/services/v2/artifact_testkit_v2.py`
- `tests/unit/contexts/backtest/application/services/v2/test_artifact_manifest_validator_v2.py`
- `tests/unit/contexts/backtest/application/services/v2/test_tp_sl_hit_times_service.py`
- `tests/unit/contexts/backtest/application/services/v2/test_yaml_backtest_artifact_loader_v2.py`
- `tests/unit/contexts/backtest/application/use_cases/test_publish_backtest_artifacts_v2.py`
- `tests/unit/contexts/market_data/adapters/test_clickhouse_funding_rate_store.py`
- `tests/unit/apps/cli/test_backtest_artifact_publish_cli.py`
- `tests/unit/apps/scheduler/test_backtest_artifact_publisher_app.py`
- `tests/unit/apps/scheduler/test_backtest_artifact_publisher_metrics.py`

Deleted:

- none

## Artifact Contract

- Funding is a first-class optional root manifest section named `funding`.
- Futures root manifests require a funding section. Spot root manifests may omit
  funding for legacy compatibility, and generated spot artifacts write
  `coverage_status=not_applicable`.
- The slot-relative funding directory is `funding/`.
- Funding arrays and dtypes:
  - `funding/funding_time.i64.npy`: `int64`, axis `funding_event`.
  - `funding/funding_rate.f64.npy`: `float64`, axis `funding_event`.
  - `funding/mark_price.f64.npy`: `float64`, axis `funding_event`.
  - `funding/funding_interval_minutes.u16.npy`: `uint16`, axis
    `funding_event`.
  - `funding/data_quality.u8.npy`: `uint8`, axis `funding_event`.
- `funding_manifest_hash` is explicit in the funding section. The root
  manifest provenance/input identity includes the funding hash and coverage
  status so futures root `artifact_manifest_hash` changes when funding content
  or funding coverage identity changes.
- `FilesystemBacktestArtifactArrayLoader.load_funding_arrays()` mmap-loads
  funding arrays only for `ready` and `degraded` funding manifests.

## Coverage Contract

- Funding artifact publishing consumes the market-data port
  `FundingRateCoverageReader`.
- The ClickHouse adapter implementation reads
  `market_data.canonical_funding_rates`; artifact publishing and artifact
  loading do not call Binance or Bybit REST providers.
- Coverage statuses are:
  - `ready`: expected and observed funding events match the requested window.
  - `degraded`: partial coverage; `coverage_policy` must be
    `degraded_with_warning`, and machine-readable `reason_codes` explain the
    gap.
  - `unavailable`: no canonical funding rows for the requested futures window;
    no funding arrays are written.
  - `not_applicable`: spot market artifacts.
- Partial coverage is not a publish failure. It writes available funding rows
  and records `coverage_status=degraded` with `degraded_with_warning`.

## Validation

- `uv run pytest -q tests/unit/contexts/backtest/adapters/outbound/artifacts_fs/test_artifact_array_loader.py tests/unit/contexts/backtest/application/services/v2/test_yaml_backtest_artifact_loader_v2.py tests/unit/contexts/backtest/application/services/v2/test_artifact_manifest_validator_v2.py tests/unit/contexts/backtest/application/use_cases/test_publish_backtest_artifacts_v2.py tests/unit/contexts/market_data/adapters/test_clickhouse_funding_rate_store.py`
  passed with `36 passed`.
- `uv run ruff check src/trading/contexts/backtest_artifacts src/trading/contexts/backtest tests`
  passed.
- `uv run pyright src/trading/contexts/backtest_artifacts src/trading/contexts/backtest tests`
  passed with `0 errors`.
- The prompt's exact pytest command,
  `uv run pytest -q tests/unit/contexts/backtest_artifacts tests/unit/contexts/backtest`,
  cannot run as written in this checkout because
  `tests/unit/contexts/backtest_artifacts` does not exist. The artifact tests
  are under `tests/unit/contexts/backtest`.
- Replacement existing suite:
  `uv run pytest -q tests/unit/contexts/backtest` passed with `373 passed`.
- Changed app/market-data tests:
  `uv run pytest -q tests/unit/contexts/market_data/adapters/test_clickhouse_funding_rate_store.py tests/unit/apps/cli/test_backtest_artifact_publish_cli.py tests/unit/apps/scheduler/test_backtest_artifact_publisher_metrics.py tests/unit/apps/scheduler/test_backtest_artifact_publisher_app.py`
  passed with `21 passed`.
- `python -m tools.docs.generate_docs_index --check` passed.

## Real-Boundary Evidence

- `artifact filesystem`: focused publish/load tests wrote funding arrays under a
  pytest temporary artifact root, parsed the root funding manifest, mmap-loaded
  arrays through `FilesystemBacktestArtifactArrayLoader`, and proved
  `funding_manifest_hash` changes when funding rates change.
- `target_host_readiness_pre_main`: via `ssh macstudio`, the Mac Studio
  checkout was reachable at `/Users/daniildegtyarev/Projects/roehub.com`,
  ClickHouse loopback returned `Ok.`, and
  `market_data.canonical_funding_rates` returned current rows for multiple
  futures symbols, including `HOMEUSDT`, `ESPORTSUSDT` and `HUSDT` windows from
  `2026-06-22 08:00:00.000` to `2026-06-22 13:00:00.000`.
- No provider REST calls were used for the artifact publisher, loader or
  coverage smoke in this stage.
- No `post_main_production_runtime_proof` is claimed for changed code in this
  turn.

## Contract Impact

- Affected dimensions:
  - artifact manifest contract: `compatible-change` for root parsing because
    `funding` is optional in the root schema; futures validation now requires
    funding for newly validated futures artifacts.
  - backtest artifact array loader port: `compatible-change` for runtime
    adapters in this repository after test doubles were updated; external
    implementers must add `load_funding_arrays`.
  - market-data store port: `compatible-change`; new read-only funding coverage
    port consumes existing canonical funding rows.
  - CLI validation summary output: `compatible-change`; it adds
    `funding_coverage_status` and `funding_manifest_hash`.
- No database schema migration is introduced by Stage `02`.

## Cold-Head Review Receipt

Cold-head review: completed
Mode: cold self-review fallback
Review scope: Stage `02` report, ledger entry, funding artifact manifest
contract, coverage reader contract and validation evidence.
Review instructions:
`architecture-review/references/cold-head-plan-prompt-pack-review.md`
Verdict: Release
Blockers fixed: none found in the fallback pass.
Local follow-up check: completed
Residual risks: exact prompt pytest path contains a nonexistent directory in
this checkout; changed code is not committed or deployed, so no post-main
production runtime proof is claimed.

## Residual Risks

- The exact prompt pytest path includes
  `tests/unit/contexts/backtest_artifacts`, which is not present; the existing
  artifact tests live under `tests/unit/contexts/backtest`.
- The Mac Studio proof is read-only target-host evidence for canonical funding
  table availability, not evidence that this uncommitted changed code runs in
  production.
- Downstream stages still own preflight/API/UI behavior, scoring adjustments and
  lazy-detail cache use of `funding_manifest_hash`.
