# Iteration 11 API runner compute memory parity

## Intent

Проверить на Mac Studio, что public API-created jobs проходят через runner and disposable child process path, выполняют тот же canonical compute, что accepted May 2 reference, и отдельно доказать scheduler behavior, child memory release, lazy cache miss/release and bounded API cache-hit reads.

## Benchmark fixture

- Host: `MacStudioDaniil`
- Git commit under test: `ffe6c50923b8ddafb04169098949612ea5368517`
- Reference iteration: `2026-05-02_iteration_8_execution_sizing_completion`
- Canonical JSON: `docs/architecture/backtest/benchmark_iterations/2026-04-26_engine_test_btcusdt_15m/benchmark_results.json`
- Reference JSON: `docs/architecture/backtest/benchmark_iterations/2026-05-02_iteration_8_execution_sizing_completion/benchmark_results.json`
- Request: `BTCUSDT`, `15m`, `REQUEST_TOP_N = 100`, `BENCHMARK_TOP_K = 5`
- Sample policy: `rows_per_indicator=6`, `warmup_rows_per_indicator=2`, `historical_prefix_compatible`
- Artifact manifest hash: `595be8b41c7d50d9d7456f4cc49ec625b7b9f0de557ace7cbdf3c066f091d0af`
- Hit-times manifest hash: `09143b055123190d6c919ca758817fa89915245d70719437f037340cdc4196fc`
- Request hash: `22d1a64757a3461507481fabea6d1434de1997f3fd063a180b289a524692c9f1`
- Excluded reference job: `tp_sl_grid/arity_7/long_only`, observed May 2 runtime `147.41507504200035s`; rationale `exclude_heaviest_140s_job`.

## API-runner path

- Required reference jobs: `27`
- Passed reference jobs: `27/27`
- Public path exercised: API create, API status, API top variants, runner claim, disposable full-job child, persisted result read.
- Required state path: `queued -> running -> succeeded`
- Result shape evidence: each job returned top variants through the public API; `api_runner_path.jobs[]` records `api_top_count`, child evidence, telemetry, and top-result sample.
- Backlog before/after: no queued or running full jobs and no queued or running lazy-detail jobs remained after the run.

## Mac Studio results

- Overall pass: `yes`
- Python: `3.12.13`
- Mac Studio acceptance: `yes`
- API base used by benchmark: recorded in `benchmark_results.json` as `api_base`.
- Primary command:

```bash
PYTHONPATH=/tmp/roehub-api-runner-benchmark/src:/tmp/roehub-api-runner-benchmark \
ROEHUB_BENCHMARK_GIT_COMMIT=ffe6c50923b8ddafb04169098949612ea5368517 \
/Users/daniildegtyarev/Projects/roehub.com/.venv/bin/python \
  scripts/backtest/run_api_runner_benchmark_parity.py \
  --api-base http://127.0.0.1:18081 \
  --out-dir /tmp/roehub-api-runner-benchmark-output \
  --timeout-seconds 21600 \
  --poll-interval-seconds 0.2
```

## Parity

- Parity pass: `yes`
- Failed jobs: `[]`
- Comparable proof: exact telemetry, sampled metrics, top-result samples, `top_results_count`, and public API top variants are compared against the accepted May 2 reference for every required job.
- Service-only overhead is not folded into canonical notebook-compatible stage ratios.

## Performance

- Stage timing jobs: `27`
- Canonical stage policy: stage timings from child compute are recorded separately from API/runner wall time and service-only overhead.
- CPU/thread evidence: each job evidence records process metrics, child process evidence, and exact telemetry including `numba_num_threads`.
- Active API responsiveness samples were captured during light and heavy scheduler phases; max observed latency was `287.37462501158006ms` during the light phase and `206.87641699623782ms` during the heavy phase.

## Memory release

- Full-job memory pass: `yes`
- Checked full-job children: `27`
- Failed full-job memory checks: `[]`
- Evidence fields: child pid, start time, exit time, exit code, peak RSS, parent RSS before/after, retained RSS delta, `vmmap`, and physical footprint where available.
- Parent retained RSS delta evidence: `yes`
- `vmmap` / physical footprint evidence: `yes`

## Lazy cache-hit memory

- Lazy cache memory pass: `yes`
- Target job: `add2c15b-0464-4dc6-a48b-044f1c8c5e5a`
- Public variant key: `job_add2c15b5e5a__dema_close_w10__risk_none__vh_ac456135`
- Lazy cache miss path: `queued -> running -> completed`; disposable lazy child exited with code `0`.
- Lazy child peak RSS: `442728448` bytes; peak physical footprint: `285212672` bytes.
- Lazy parent retained RSS delta after child exit: `0` bytes.
- API cache-hit retained RSS delta: `49152` bytes against `67108864` byte limit.
- Cache bundle: `trades.jsonl` exists with `30056` rows and `18177058` bytes.
- Bounded reader audit: API page, series, monthly stats, symbol stats, and CSV paths use cache reader methods instead of full-detail payload loading.

## Scheduler smoke

- Scheduler smoke pass: `yes`
- Configured caps: `ROEHUB_BACKTEST_LIGHT_CONCURRENCY=2`, `ROEHUB_BACKTEST_HEAVY_CONCURRENCY=1`
- Light/heavy overlap policy: `disabled_v1`
- Light phase: two `light_candidate` jobs ran concurrently; max active light `2`, max active heavy `1`, light concurrency cap pass `yes`.
- Heavy phase: heavy jobs ran FIFO by `created_at ASC, job_id ASC`; max active heavy `1`; heavy no-overlap pass `yes`.
- Heavy preflight evidence: arity `3` no-risk requests were classified as `heavy` with `estimated_combinations_upper_bound=7529536`.
- Light candidate evidence: arity `1` no-risk requests were classified as `light_candidate` at preflight and succeeded after runner refinement.
- Promotion evidence: `light_candidate_promotion` requeued after post-prepare refinement to `heavy`, then finished with state path `queued -> running -> queued -> running -> succeeded`.
- No starvation evidence: older queued heavy jobs were processed before the later light job in the heavy phase.

## Legacy path absence

- Pass: `yes`
- Runner parent does not construct the full compute graph; production runner uses `BacktestChildProcessExecutor`.
- Public API cache-hit paths use bounded cache readers and do not call the legacy full-detail read model.
- Large-grid production path uses ordinal streaming chunks and does not use Python `itertools.product`.

## Dead code audit

- Pass: `yes`
- Retained helpers: `full_job_compute.py` is child-only; `result_series.py` is reference-only for legacy in-memory builders; the research notebook is reference-only semantic baseline.
- Removed/replaced paths: API create path replaced sync inline compute with background queued jobs; lazy cache hit replaced monolithic full-detail JSON reads with metadata and JSONL readers; large-grid production path uses ordinal streaming chunks instead of Cartesian materialization.

## Docs drift audit

- Pass: `yes`
- Active docs checked: `docs/architecture/backtest/backtest-job-runner-production-plan-v1.md`, `docs/architecture/backtest/backtest-service-artifact-runtime-v1.ru.md`, `docs/architecture/backtest/benchmark_iterations/README.md`
- Active blockers: `[]`
- Remaining historical references are allowed only when labeled historical.

## Artifacts

- `benchmark_results.json`
- `benchmark_summary.md`
- `local_accounting_validation.json`

## Decision

Pass. Mac Studio API-runner evidence is comparable to the accepted May 2 fixture after excluding only `tp_sl_grid/arity_7/long_only`, and memory/scheduler/lazy-cache evidence passes as separate acceptance surfaces.

## Operator Commands

```bash
uv run python scripts/backtest/run_api_runner_benchmark_parity.py --api-base http://127.0.0.1:18081 --out-dir /tmp/roehub-api-runner-benchmark-output --timeout-seconds 21600 --poll-interval-seconds 0.2
uv run python scripts/backtest/validate_benchmark_accounting.py --out docs/architecture/backtest/benchmark_iterations/2026-05-14_iteration_11_api_runner_compute_memory_parity/local_accounting_validation.json
```
