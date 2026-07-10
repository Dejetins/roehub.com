# Roehub v2 interactive design prototype

Изолированный интерактивный прототип концепции **Roehub Workbench — Institutional Native Research**.

## Запуск

```bash
npm install
npm run dev -- --host 0.0.0.0 --port 4173 --strictPort
```

Открыть: `http://localhost:4173`.

## Что входит

- Login;
- Overview;
- Strategies: Library, Analytics, Runtime control, RL / ML;
- Backtests: Configure, Queue, Results;
- Live operations;
- Model registry;
- Connections;
- Settings и шесть тем;
- Program map;
- UI state gallery;
- desktop, tablet и mobile shell.

Prototype использует только локальные детерминированные данные. Он не подключается к production API, не содержит секретов и не является решением о переносе production-приложения на React.

Подробности:

- `design-qa.md` — визуальное сравнение и история исправлений;
- `browser-qa.md` — browser evidence и граница доказательств;
- `../../docs/architecture/apps/web/web-ui-product-redesign-master-plan-v2.md` — карта программы и план production-миграции.
