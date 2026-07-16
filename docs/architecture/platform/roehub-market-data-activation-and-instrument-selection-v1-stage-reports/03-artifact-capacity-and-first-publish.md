# Этап 03 — ёмкость артефактов и первая публикация

## Результат

Этап принят. Для первого реального контура публикации оставлен ровно один
ручной bounded publish `binance:futures:BTCUSDT`. Контейнер publisher имеет
лимит `1 GiB`; его внутренняя конфигурация согласована с ним: один signal
worker и бюджет `768 MiB`. Плановое исполнение намеренно остаётся выключенным
с причиной `capacity_gate`.

Первичное заполнение scheduler ограничено последними семью днями (`10080`
минутных свечей). Периодическая историческая достройка отключена до отдельного
решения о расширении. Поэтому добавление выбора пользователем не запускает
неконтролируемый historical backfill всего каталога.

## Доказательство на реальной границе (`runtime smoke`)

`runtime smoke` выполнен в изолированном Docker Desktop проекте
`roehub-stage02`, без production-данных и торговых операций. После bounded
заполнения вручную запущена публикация только для `BTCUSDT` с
`--max-source-bars 10080`:

- источник: `10080` закрытых свечей от `2026-07-08T22:51:00Z` до
  `2026-07-15T22:51:00Z`;
- результат: `succeeded`, активный `slot_a`, generation `1`;
- SHA-256 манифеста:
  `933973047920870ef45c958c658fa1d4e8b106b22cc68def0e36846e7b1a0154`;
- размер текущего артефакта: `26764848` bytes, `1667` файлов;
- funding coverage: `ready`, `21/21` событий, пропусков `0`;
- diagnostics: `0`;
- пиковая память cgroup: `715964416` bytes при лимите `1073741824` bytes;
  `oom=0`, `oom_kill=0`.

Полная проверка находится в
[`03-artifact-capacity-proof.json`](evidence/03-artifact-capacity-proof.json).
Она не содержит secrets, provider payloads или пользовательских данных.

## Поведение и границы

- `backtest-artifact-publish` получил явный `--max-source-bars`; ограничение
  применяется к временному диапазону источника до чтения свечей.
- Scheduler передаёт funding catch-up только текущий effective набор
  инструментов, а не весь reference catalog.
- `rest_insurance_catchup.enabled: false` во всех runtime-профилях. Его
  последующее включение требует отдельного решения о ёмкости и повторного
  `runtime smoke`.
- Для временных рамок артефакта до трёх суток необходимо не меньше `10080`
  минутных свечей; меньший bounded диапазон корректно отклоняется, а не создаёт
  неполный артефакт.

## Проверки

- описанный выше `runtime smoke`: bounded scheduler fill, ручной publish,
  pointer switch, cgroup peak и отсутствие OOM подтверждены в evidence;
- focused `pytest` для planner, bounded REST CLI, funding catch-up и publisher
  — `30 passed`;
- focused Ruff — успешно;
- focused Pyright — `0 errors, 0 warnings`;
- `uv run python -m tools.release.generate_runtime_topology --check` — успешно.

Следующий разрешённый этап: `04`. Он выполняет только disposable proof
защищённой инициализации OpenBao; долговременное владение PGP-материалом,
unseal shares и административными credential остаётся явным действием
владельца.

## Остаточный риск

Доказательство получено локально на `linux/arm64` в Docker Desktop. Исходная
обязательная проверка native `linux/amd64` не заменена и остаётся внешним
blocker исходного self-hosted плана. Производственная среда не читалась и не
изменялась.
