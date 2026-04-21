# Backtest Refactor Docs

Статус: active transition к изолированному Backtest Gateway и новой документации ядра.

## Канонические документы

- `docs/architecture/backtest/deep-research-report.md` — исследовательский аудит и карта текущих проблем/рисков.
- `docs/architecture/backtest/backtest-core-refactor-prompt-pack-v1.md` — исполняемый prompt-pack по шагам рефакторинга.

## Текущее направление

- API слой `backtest` работает как gateway (transport + DB/artifacts integration), без локального compute-ядра в API процессе.
- Precompute/publish поток вынесен в отдельный контекст `backtest_artifacts`.
- Старые документы по legacy backtest ядру удалены; новые решения фиксируются только в двух канонических документах выше.

## Индексы

После изменения docs запускать:

- `python -m tools.docs.generate_docs_index`
- `python -m tools.docs.generate_docs_index --check`
