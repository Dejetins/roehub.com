# Backtest Refactor Docs

Статус: runtime-compute reset. Старое runtime-ядро backtest, API/UI запуска и
`backtest-job-runner` удалены из активной кодовой базы.

## Канонические документы

- `docs/architecture/backtest/deep-research-report.md` — исследовательский аудит и карта текущих проблем/рисков.
- `docs/architecture/backtest/backtest-core-refactor-prompt-pack-v1.md` — исполняемый prompt-pack по шагам рефакторинга.

## Текущая граница доверия

- Доверенный scope: `backtest_artifacts` publisher/precompute и `.npy` артефакты.
- Сохранены только зависимости, которые нужны publisher/precompute: manifest/publish services,
  signal rules, hit-times precompute, artifact config, path/current-pointer adapters и job
  storage для publish guard.
- Не доверенный legacy runtime compute удален: runtime kernels/scorers/shortlists,
  old backtest use-cases, API routes, web pages/assets и `backtest-job-runner`.
- Новый backtest engine пока не проектируется в этой итерации. Следующий дизайн должен
  стартовать с нуля поверх artifact publisher/precompute как входного слоя данных.

## Индексы

После изменения docs запускать:

- `python -m tools.docs.generate_docs_index`
- `python -m tools.docs.generate_docs_index --check`
