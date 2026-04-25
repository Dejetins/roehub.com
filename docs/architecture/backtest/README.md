# Backtest Refactor Docs

Статус: runtime-compute reset. Старое runtime-ядро backtest, API/UI запуска и
`backtest-job-runner` удалены из активной кодовой базы.

## Канонические документы

- `docs/architecture/backtest/backtest-service-artifact-runtime-v1.md` — целевая архитектура нового artifact-backed backtest service.
- `docs/architecture/backtest/benchmark_iterations/README.md` — рабочая папка для benchmark evidence по итерациям.

## Текущая граница доверия

- Доверенный scope: `backtest_artifacts` publisher/precompute и `.npy` артефакты.
- Сохранены только зависимости, которые нужны publisher/precompute: manifest/publish services,
  signal rules, hit-times precompute, artifact config, path/current-pointer adapters и job
  storage для publish guard.
- Не доверенный legacy runtime compute удален: runtime kernels/scorers/shortlists,
  old backtest use-cases, API routes, web pages/assets и `backtest-job-runner`.
- Новый backtest service проектируется поверх artifact publisher/precompute как входного слоя
  данных, без восстановления legacy runtime path.

## Индексы

После изменения docs запускать:

- `python -m tools.docs.generate_docs_index`
- `python -m tools.docs.generate_docs_index --check`
