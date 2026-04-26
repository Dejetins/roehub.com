# Backtest Benchmark Iteration 0 — Docs Benchmark Harness

Iteration 0 закрывает documentation/benchmark harness gate для Backtest Service Artifact Runtime v1.

## Охват

- Сделано: source-of-truth навигация, benchmark template contract и локальная
  проверка canonical benchmark evidence.
- Не входит: service runtime code, scoring kernels, notebooks, Mac Studio
  acceptance benchmark run.

## Источник истины

- Канонический runtime contract:
  `docs/architecture/backtest/backtest-service-artifact-runtime-v1.ru.md`
- Companion/reference copy:
  `docs/architecture/backtest/backtest-service-artifact-runtime-v1.md`
- Prompt pack:
  `docs/architecture/backtest/backtest-service-implementation-prompt-pack-iteration-0-1.md`
- Benchmark template:
  `docs/architecture/backtest/benchmark_iterations/README.md`

## Каноническое evidence

- Notebook baseline:
  `tests/notebook_tests/engine_test/btcusdt_15m_research_engine.ipynb`
- Числовые целевые значения:
  `docs/architecture/backtest/benchmark_iterations/2026-04-26_engine_test_btcusdt_15m/benchmark_results.json`
- Человекочитаемое summary:
  `docs/architecture/backtest/benchmark_iterations/2026-04-26_engine_test_btcusdt_15m/benchmark_summary.md`
- Request hash:
  `22d1a64757a3461507481fabea6d1434de1997f3fd063a180b289a524692c9f1`
- Artifact manifest hash:
  `a76ccba27c8fabb3d5a6ad14c7d8f121839a5e22c107d038223261159367b259`
- Hit-times manifest hash:
  `2366cc2f5a44ccc7faf716ed65a4f37bcbb91150471eec177d7f633a615dbaba`
- Run matrix: `28` runs (`7 arities x 2 risk modes x 2 direction modes`)
- Target hit-times path: `hit_times/15m`

## Контрактные решения

- Public create endpoint остается `POST /backtests/jobs`; old `POST /backtests`
  wording помечен как superseded для v1.
- Public request vocabulary использует `risk.mode`, а не public `execution profile`.
- Public `variant_key` остается readable и job-scoped; stable storage/cache
  identity представлен через `variant_hash`.
- Benchmark records должны использовать current notebook-compatible timer names:
  `service_warmup`, `numba_warmup`, `sample_warmup`,
  `total_without_warmup`, `load_hit_times`, `tp_sl_grid_validation`,
  `prepare_pools`, `build_exact_context`, `build_proxy_context`,
  `combo_iteration`, `proxy_filter`, `self_check`, `exact_scoring`,
  `tp_sl_exact_scoring`, `heap_update`, `top_result_proxy_fill`.

## Локальное evidence

- Canonical benchmark JSON shape: required keys присутствуют и `runs` length is
  `28`.
- Docs index generation/check является локальным deterministic gate для этой
  docs-only iteration.
- Mac Studio acceptance benchmark не запускался в Iteration 0; он остается
  обязательным для implementation iterations, которые заявляют runtime
  performance acceptance.

## Quality Gates

- `python -m tools.docs.generate_docs_index`: pass, docs index обновлен и затем
  unchanged on rerun.
- `python -m tools.docs.generate_docs_index --check`: pass.
- Canonical benchmark JSON shape check: pass, required keys присутствуют и
  `runs` length is `28`.
- Stale vocabulary scan: pass with classified matches. Backtest matches находятся
  в superseded/compatibility wording или active `POST /backtests/jobs` /
  `POST /backtests/preflight` routes. Generated `docs/architecture/README.md`
  также содержит одну unrelated Strategy `runs/events` index line.
- `git diff --check`: pass.

## Открытые вопросы

- Benchmark command contract пока не зафиксирован, потому что в current tree нет
  stable repository command для Mac Studio service benchmark execution.
