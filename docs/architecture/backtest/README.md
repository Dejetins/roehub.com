# Backtest Refactor Docs

Статус: runtime-compute reset. Старое runtime-ядро backtest, API/UI запуска и
`backtest-job-runner` выведены из доверенного active runtime path.

## Канонические документы

- `docs/architecture/backtest/backtest-service-artifact-runtime-v1.ru.md` — канонический implementation source и целевой contract нового artifact-backed backtest service.
- `docs/architecture/backtest/backtest-job-runner-production-plan-v1.md` — целевой production-план для повторного ввода `backtest-job-runner`, lazy trades materialization, tier limits, Mac Studio service, metrics и smoke evidence.
- `docs/architecture/backtest/backtest-ai-configurator-mlx-v1.md` — текущий reset-документ AI configurator для `/backtests`: single-shot LM Studio prompt/blob contract выведен из active runtime path, сохранены storage/API/validator/security foundations, следующий целевой контракт — LM Studio tools.
- `docs/architecture/backtest/backtest-ai-configurator-tool-agent-v1.md` — канонический target-state контракт для LM Studio OpenAI-compatible `tools` / `tool_calls`, backend-owned tool executor, stage gates и acceptance matrix Prompts 03-07.
- `docs/architecture/backtest/backtest-service-artifact-runtime-v1.md` — companion/reference copy; если расходится с русским документом, для реализации побеждает `.ru.md`.
- `docs/architecture/backtest/backtest-service-implementation-prompt-pack-iteration-0-1.md` — executable prompt pack для Iteration 0/1 реализации.
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
- Новый `backtest-job-runner` может вернуться в trusted active runtime path только
  через `backtest-job-runner-production-plan-v1.md`: standalone Mac Studio process,
  queued full jobs, async lazy trades materialization, tier quotas и production smoke
  на реальных artifacts.
- Новый backtest service проектируется поверх artifact publisher/precompute как входного слоя
  данных, без восстановления legacy runtime path.

## Superseded vocabulary

Для нового runtime v1 source of truth находится в
`backtest-service-artifact-runtime-v1.ru.md`. Старые формулировки в roadmap/doc/code
не переопределяют v1 contract, если говорят про:

- `POST /backtests` как основной create endpoint вместо `POST /backtests/jobs`;
- `runs` вместо `jobs`;
- любые hit-times формулировки, которые противоречат target `hit_times/15m`;
- public `execution profile` вместо `risk.mode`;
- SHA-only public `variant_key` вместо readable public `variant_key` +
  stable `variant_hash`.

Deep-research reports под этой папкой являются background/review evidence, а не
реализационным source of truth для v1.

## Индексы

После изменения docs запускать:

- `python -m tools.docs.generate_docs_index`
- `python -m tools.docs.generate_docs_index --check`
