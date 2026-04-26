# Backtest Refactor Docs

Статус: runtime-compute reset. Старое runtime-ядро backtest, API/UI запуска и
`backtest-job-runner` выведены из доверенного active runtime path.

## Канонические документы

- `docs/architecture/backtest/backtest-service-artifact-runtime-v1.md` — каноническая целевая архитектура нового artifact-backed backtest service.
- `docs/architecture/backtest/benchmark_iterations/README.md` — рабочая папка для benchmark evidence по итерациям.

## Текущая граница доверия

- Доверенный scope: `backtest_artifacts` publisher/precompute и `.npy` артефакты.
- Сохранены только зависимости, которые нужны publisher/precompute: manifest/publish services,
  signal rules, hit-times precompute, artifact config, path/current-pointer adapters и job
  storage для publish guard.
- Не доверенный legacy runtime compute не является source of truth: runtime
  kernels/scorers/shortlists, old backtest use-cases, API routes, web pages/assets и
  `backtest-job-runner` могут оставаться в репозитории только как residual,
  compatibility-only или obsolete code до явной классификации.
- Новый backtest service проектируется поверх artifact publisher/precompute как входного слоя
  данных, без восстановления legacy runtime path.

## Superseded vocabulary

Для нового runtime v1 source of truth находится в
`backtest-service-artifact-runtime-v1.md`. Старые формулировки в roadmap/doc/code
не переопределяют v1 contract, если говорят про:

- `POST /backtests` как основной create endpoint вместо `POST /backtests/jobs`;
- `runs` вместо `jobs`;
- любые hit-times формулировки, которые противоречат target `hit_times/15m`;
- public `execution profile` вместо `risk.mode`;
- SHA-only public `variant_key` вместо readable public `variant_key` +
  stable `variant_hash`.

## Индексы

После изменения docs запускать:

- `python -m tools.docs.generate_docs_index`
- `python -m tools.docs.generate_docs_index --check`
