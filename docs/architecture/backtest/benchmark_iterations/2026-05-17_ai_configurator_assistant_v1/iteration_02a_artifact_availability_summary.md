# Iteration 02A Artifact availability summary

Дата: 2026-05-17.

Статус: accepted, published to `origin/main`, deployed and verified on Mac Studio.

## Цель

Сделать publisher-owned `availability_summary.yaml` реальным source of truth для AI Configurator
по доступным `exchange/market/symbol`, периодам и timeframe coverage.

Файл:

```text
/opt/roehub/state/backtest_artifacts/v2/availability_summary.yaml
```

## Что изменено

- Добавлен `BacktestArtifactAvailabilitySummaryGeneratorV2` в bounded context
  `backtest_artifacts`.
- Добавлен atomic writer `AtomicArtifactAvailabilitySummaryWriterV2`.
- Summary строится только из valid `current.yaml` и active slot `manifest.yaml`.
- Instruments с missing/corrupt current, missing active slot, missing/corrupt active manifest,
  identity/hash mismatch или пустым backtest coverage исключаются.
- Manual CLI `backtest-artifact-publish` после успешного publish regenerates summary.
- Добавлен manual recovery mode без rebuild:

```bash
uv run python -m apps.cli.main.main backtest-artifact-publish \
  --config configs/prod/backtest_artifacts.yaml \
  --regenerate-summary-only
```

- Scheduler `backtest-artifact-publisher` regenerates summary after a successful cycle with at
  least one successful symbol, and marks the run `summary_failed` if summary write fails.
- Runbook updated with operational summary regeneration procedure.
- Contract document added:
  `artifact_availability_summary_contract.md`.

## Schema summary

Top-level:

- `schema_version: 1`
- `generated_at_utc`
- `artifact_root`
- `artifact_root_schema_version: 2`
- `summary_hash`
- `source: artifact_publisher_active_slot_scan`
- `instruments`

Instrument key:

```text
exchange/market/symbol
```

Instrument payload:

- `exchange`, `market`, `symbol`
- `active_slot`, `slot_generation`, `asof_date`, `published_at_utc`, `manifest_sha256`
- top-level conservative `start_date` / `end_date`
- `backtest_timeframes`
- `timeframes.<tf>.start_date/end_date/bars/price_available/signals_available/mappings_available/indicator_ids`
- `hit_times.timeframe` and `hit_times.available`

`summary_hash` is deterministic over canonical payload without `summary_hash` and
`generated_at_utc`; repeated generation over identical artifacts keeps the same hash.

## Mac Studio evidence

Pre-delivery evidence ran current 02A code from temporary workspace
`/tmp/roehub-iteration-02a` against real artifact root:

```bash
cd /tmp/roehub-iteration-02a
PYTHONPATH=/tmp/roehub-iteration-02a/src:/tmp/roehub-iteration-02a \
  /opt/roehub/app/.venv/bin/python -m apps.cli.main.main backtest-artifact-publish \
  --config /tmp/roehub-iteration-02a/configs/prod/backtest_artifacts.yaml \
  --regenerate-summary-only
```

Result:

- `summary_path`: `/opt/roehub/state/backtest_artifacts/v2/availability_summary.yaml`
- `summary_hash`: `e2615a75818937f79ef7bf5c955b492af383dcadd15c106f4379a07926721df7`
- `generated_at_utc`: `2026-05-17T21:05:11Z`
- `instrument_count`: 2
- `skipped_count`: 0
- `skipped_reasons`: `{}`

Independent Mac Studio scan result:

- `summary_instrument_count`: 2
- `valid_active_current_count`: 2
- `counts_match`: true
- `keys_match`: true
- `binance/spot/BTCUSDT` `backtest_timeframes` match active manifest: true
- `binance/spot/BTCUSDT` slot generation match: true
- `binance/spot/BTCUSDT` manifest SHA match: true
- `binance/spot/BTCUSDT` `1h` coverage match:
  - `start_date`: `2017-08-17`
  - `end_date`: `2026-05-02`
  - `bars`: 76155
  - `indicator_ids`: 40 ids, identical between summary and active manifest

Post-deploy evidence ran the published code from `/opt/roehub/app`:

```bash
cd /opt/roehub/app
PYTHONPATH=/opt/roehub/app/src:/opt/roehub/app \
  /opt/roehub/app/.venv/bin/python -m apps.cli.main.main backtest-artifact-publish \
  --config /opt/roehub/app/configs/prod/backtest_artifacts.yaml \
  --regenerate-summary-only
```

Result:

- `summary_path`: `/opt/roehub/state/backtest_artifacts/v2/availability_summary.yaml`
- `summary_hash`: `e2615a75818937f79ef7bf5c955b492af383dcadd15c106f4379a07926721df7`
- `generated_at_utc`: `2026-05-17T21:15:44Z`
- `instrument_count`: 2
- `skipped_count`: 0
- `skipped_reasons`: `{}`

Independent post-deploy scan:

- `summary_hash_matches_content`: true
- `summary_instrument_count`: 2
- `valid_active_current_count`: 2
- `counts_match`: true
- `keys_match`: true
- `binance/spot/BTCUSDT` active slot, slot generation, manifest SHA and backtest timeframes match active manifest: true
- `binance/spot/BTCUSDT` `1h` coverage match: `2017-08-17` to `2026-05-02`, 76155 bars, 40 indicator ids

Deployed runtime file hashes in `/opt/roehub/app` match the local commit for:

- `apps/cli/commands/backtest_artifact_publish.py`
- `apps/scheduler/backtest_artifact_publisher/wiring/modules/backtest_artifact_publisher.py`
- `src/trading/contexts/backtest_artifacts/application/services/v2/artifact_availability_summary.py`
- `src/trading/contexts/backtest_artifacts/adapters/outbound/artifacts_fs/availability_summary_writer.py`

`bash scripts/macos/smoke_prod.sh` completed with exit code 0.

## Проверки

Completed:

```text
uv run pytest -q tests/unit/contexts/backtest/application/services/v2/test_artifact_availability_summary_v2.py tests/unit/apps/cli/test_backtest_artifact_publish_cli.py tests/unit/apps/scheduler/test_backtest_artifact_publisher_app.py tests/unit/apps/scheduler/test_backtest_artifact_publisher_metrics.py
```

Result: `20 passed`.

```text
uv run pytest -q tests/unit/contexts/backtest/application/services/v2
```

Result: `222 passed`.

```text
uv run ruff check src/trading/contexts/backtest_artifacts apps/scheduler apps/cli tests/unit/contexts/backtest/application/services/v2/test_artifact_availability_summary_v2.py tests/unit/apps/cli/test_backtest_artifact_publish_cli.py tests/unit/apps/scheduler/test_backtest_artifact_publisher_app.py tests/unit/apps/scheduler/test_backtest_artifact_publisher_metrics.py
```

Result: passed.

```text
uv run pyright
```

Result: passed, `0 errors`.

```text
uv run python -m tools.docs.generate_docs_index --check
```

Result: passed.

## Contract impact

| Surface | Classification | Notes |
| --- | --- | --- |
| Public API contract | none | No HTTP/API route or browser payload changes. |
| Persisted artifact contract | compatible-change | Additive root-level `availability_summary.yaml`; existing slots/current/manifest unchanged. |
| CLI contract | compatible-change | Additive `--regenerate-summary-only`; existing publish args still work. |
| Scheduler workflow | compatible-change | Adds post-success summary regeneration and `summary_failed` run status. |
| Config schema | none | No config key/default change. |
| Request hash/cache identity | none | Backtest job request identity unchanged. |
| AI request path | none in this iteration | No AI context snapshot or UI implementation. |

## Delivery

Direct-main delivery completed.

- local branch: `main`
- implementation commit: `fe63cb1cd87f35da7b418edeafb85c49107a2de5`
- pushed to `origin/main`: true
- `CI` run `26002792343`: success
- `Deploy Backend` run `26002792309`: success
- `Publish App Image` run `26002792328`: success
- `Deploy Web` run `26002811146`: success
- Mac Studio runtime path: `/opt/roehub/app`
- Mac Studio runtime checkout: deployed copy, no `.git` directory
- Mac Studio verified commit: `fe63cb1cd87f35da7b418edeafb85c49107a2de5` via workflow success, deployed file hashes, summary smoke and prod smoke
- `pushed_to_main`: true
- `macstudio_verified`: true

## Acceptance marker

Accepted: true.

Blocking reason: none.

Next iteration allowed: true.
