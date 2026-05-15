# Итерация 16: quality gate и ranking-only exact, benchmark arity 6

Benchmark на Mac Studio для экспериментальных изменений фильтра качества по числу закрытых сделок и облегченного no-risk exact scoring.

Сравнение выполнено не с May 2, а с последним чистым benchmark текущей API-runner архитектуры:
`docs/architecture/backtest/benchmark_iterations/2026-05-14_iteration_15_api_runner_clean_arity6_cpu_memory`.

## Короткий вывод

Текущий вариант изменений не проходит acceptance и не должен оставаться в production runtime.

Причины:

- `none/arity_6/long_short_reversal` вернул `top_count=0`, потому что `min_closed_trades=300` отфильтровал все exact-кандидаты.
- `confirm_prefilter` оказался вредным для скорости: он добавил отдельный полный проход на `14.870..15.221s` перед exact.
- Для no-risk `exact_scoring` местами стал быстрее, но end-to-end child/service wall стал хуже: около `29s` вместо прежних `~16s` exact-stage baseline.
- Фильтр качества пока не является глобальным source of truth: в `tp_sl_grid` telemetry нет `min_closed_trades`, `below_min_*`, `eligible`.
- Memory cleanup по full jobs прошел, но это не компенсирует regression по корректности и wall-clock.

Экспериментальные runtime-файлы после benchmark были возвращены на Mac Studio к текущему `HEAD`; локальная рабочая копия оставлена с изменениями и артефактами для разбора.

## Условия замера

- Host: `MacStudioDaniil`
- Дата запуска: `2026-05-15` MSK, timestamps в артефактах записаны в UTC.
- Symbol/timeframe: `BTCUSDT` / `15m`
- Период: `2020-01-11T20:08:00Z` .. `2026-04-11T20:08:00Z`
- Jobs: только arity 6
- `top_n=50`
- `benchmark_top_k=5`
- Threads: `ROEHUB_BACKTEST_NUMBA_NUM_THREADS=12`, `ROEHUB_BACKTEST_HEAVY_NUMBA_NUM_THREADS=12`
- Политика runner во время benchmark: один heavy child process за раз.
- CPU sampler: постоянные `ps` samples по child process и `--job-id`.
- Memory gate: retained delta по system anonymous/wired memory, лимит `512 MiB`.

## Что было реализовано в эксперименте

Фильтр качества:

- Auto formula policy: `timeframe_sqrt_v1`
- For this benchmark fixture: `15m` over about `6.25` years produced `min_closed_trades=300`.
- Row prefilter: row `nonzero < min_closed_trades` rejected early.
- No-risk combo prefilter: candidate `confirm_count < min_closed_trades` rejected before exact.
- No-risk exact heap: candidate accepted into top-N only when `trade_count >= min_closed_trades`.

Изменения по аллокациям и ранжированию:

- `filtered_eval_T` changed to a sliced view instead of an unconditional contiguous copy.
- `segment_pos_workspace` reused through scratch object instead of hot-path allocation per dispatch.
- Default no-risk ranking path computes `total_return_pct + trade_count` first, then hydrates full metrics only for shortlist.

## Результаты относительно последнего чистого benchmark текущей архитектуры

| Job | Предыдущий exact s | Новый exact s | Изменение | Новый service wall s | CPU mean/p50/max % | Memory delta MiB | Результат | Top | Quality evidence |
| --- | ---: | ---: | ---: | ---: | --- | ---: | --- | ---: | --- |
| `none/arity_6/long_only` | 15.968 | 6.754 | -57.7% | 28.719 | 830.5 / 1165.6 / 1186.4 | 302.9 | fail | 50 | `min=300`, `confirm_drop=27216`, `trade_drop=0`, `eligible=19440` |
| `none/arity_6/long_short_reversal` | 15.810 | 14.283 | -9.7% | 29.419 | 1160.4 / 1178.5 / 1186.8 | 0.2 | fail | 0 | `min=300`, `confirm_drop=2592`, `trade_drop=44064`, `eligible=0` |
| `tp_sl_grid/arity_6/long_only` | 16.566 | 17.223 | +4.0% | 38.545 | 524.0 / 100.0 / 1184.3 | 423.2 | pass | 50 | telemetry фильтра качества отсутствует |
| `tp_sl_grid/arity_6/long_short_reversal` | 15.504 | 15.366 | -0.9% | 16.726 | 1086.9 / 1172.2 / 1191.2 | 50.7 | pass | 50 | telemetry фильтра качества отсутствует |

## Диагностика no-risk stage

| Job | Confirm prefilter s | Exact scoring s | Второй проход полных метрик s | Heap eligible | Top count |
| --- | ---: | ---: | ---: | ---: | ---: |
| `none/arity_6/long_only` | 15.221 | 6.754 | 0.025 | 19440 | 50 |
| `none/arity_6/long_short_reversal` | 14.870 | 14.283 | 0.000 | 0 | 0 |

Интерпретация:

- Ranking-only exact может снижать стоимость no-risk scoring kernel.
- Отдельный `confirm_prefilter` дублирует большую часть exact-работы и съедает выигрыш.
- Текущая формула слишком агрессивна для этого fixture: `min_closed_trades=300` меняет форму результата и может дать пустой top-N.

## Память

Full-job system memory cleanup passed for all four arity-6 jobs.

| Job | System memory gate | Retained delta MiB |
| --- | --- | ---: |
| `none/arity_6/long_only` | pass | 302.9 |
| `none/arity_6/long_short_reversal` | pass | 0.2 |
| `tp_sl_grid/arity_6/long_only` | pass | 423.2 |
| `tp_sl_grid/arity_6/long_short_reversal` | pass | 50.7 |

Lazy cache-hit memory check still failed with `queued -> running -> failed`; the same failure existed in the previous clean benchmark and is not evidence of this patch improving or regressing lazy trades.

## Решение по результату

Эту реализацию нельзя принимать как есть.

Что можно сохранить как полезные находки:

- `eval_T` view cleanup is low-risk and likely worth preserving in a revised patch.
- `segment_pos_workspace` reuse is low-risk and likely worth preserving in a revised patch.
- Ranking-only exact is directionally useful, but only if the quality gate is fused into exact or made cheap enough not to double-scan the same candidates.

Что нужно переделать перед acceptance:

- Recalibrate `min_closed_trades`; `300` is too strict for the arity-6 `BTCUSDT/15m` fixture.
- Remove the separate expensive `confirm_prefilter`, or replace it with a cheap safe upper-bound prefilter.
- Apply final `trade_count >= min_closed_trades` consistently to `tp_sl_grid` exact too.
- Repeat this same arity-6 benchmark after rework and require no `top_count=0` regression, no wall-clock regression, and memory cleanup pass.

## Артефакты

- `benchmark_results.json`
- `benchmark_summary.md`
- `child_process_evidence/*.json`
