# Этап 09 — организационная изоляция исследовательского контура

## Статус

- Этап: `09`.
- Статус: `accepted` после обязательных исправлений единственной независимой
  холодной проверки и локальной повторной проверки.
- Дата: `2026-07-13`.
- Режим: `goal_driven`.
- Proof boundary: `N/A`; одноразовые PostgreSQL и ClickHouse с искусственными
  идентификаторами и 120 искусственными свечами. Production базы, пользователи,
  секреты, артефакты и конфигурация не читались и не изменялись.
- Blocker: отсутствует.
- Следующий разрешённый этап: `10`.

## Результат

Research scope теперь выводится на сервере из единственного активного
membership аутентифицированного `user_id`. Backtest API не принимает
`organization_id` как authority из payload, path или заголовка. Нулевое число
membership закрывает операцию кодом `research.organization_scope_forbidden`,
несколько membership — `research.organization_scope_ambiguous`.

`BacktestJob` и его DTO получили обязательный `organization_id`. Создание,
получение, списки, отмена, удаление, квоты, claim/worker, top variants, Stage A
shortlist и lazy materialization проходят через organization-scoped ports и
PostgreSQL queries. Межорганизационное чтение job возвращает отсутствие строки;
same-org обращение другого пользователя сохраняет прежний отказ `403`.

Миграция `0014_research_organization_isolation_v1.sql`:

- прекращает работу, если четыре mutable research-таблицы уже содержат строки;
- не выполняет `UPDATE`, backfill или legacy import;
- добавляет composite organization/job и organization/user/job foreign keys;
- добавляет organization-scoped индексы чтения, claim, idempotency и lazy queue;
- включена в единый Stage `04` bootstrap/manifest/status lifecycle как
  `research-tenancy-0014`.

Request identity разделена по смыслу. Детерминированный `request_hash` и
content hashes остаются organization-neutral, когда описывают одинаковые байты
или вычислительное содержимое. Idempotency и cache keys используют новые
versioned namespaces `research-request/v1` и `research-cache/v1` вместе с
`organization_id`; legacy alias и dual-read отсутствуют.

После независимой проверки устранены две ранее пропущенные runtime-границы.
Strategy launch-from-backtest-variant теперь разрешает организацию через
`PostgresResearchOrganizationScopeResolver` и читает job/variant только через
обязательный `organization_id`. Маршруты indicators compute/estimate/registry и
market-data reference сначала выполняют server-side organization resolution;
при недоступном или неоднозначном scope ClickHouse, registry и compute не
вызываются. Lazy materialization worker и дочерний процесс также передают scope
во все job/variant reads.

## Классификация ресурсов

| Ресурс | Владение | Инвариант |
|---|---|---|
| `canonical_candles_1m` | `installation-shared` | Одна каноническая строка доступна после server-side authorization; копии по организациям запрещены. |
| Indicator definitions/kernels | `installation-shared`, immutable compute | Входные массивы и output semantics не менялись. |
| Backtest jobs/results/top variants/shortlist | `organization-owned`, user внутри организации | Organization scope обязателен в entity, DTO, repository и composite FK. |
| Lazy materialization tasks/cache namespace | `organization-owned` | Parent job/user и cache identity обязаны принадлежать одной организации. |
| Immutable artifact bytes/manifests | `installation-shared` по digest | Байты не дублируются; org ownership несёт job/materialization metadata. |
| Optimize definitions | immutable code/config | Mutable jobs/results сейчас отсутствуют; первая persistence обязана наследовать org scope. |

## Реальная граница проверки

Команда `uv run python -m apps.migrations.verify_storage_runtime` полностью
подняла пустые PostgreSQL `16.14`, ClickHouse `24.8.14.39` и Redis `7.2.14` на
Docker CLI `29.6.1` / Engine `29.5.2`. Она повторила fresh bootstrap,
interruption recovery, idempotent rerun, restart volumes и external readiness,
затем выполнила новый research probe с двумя организациями. Сырые SQL-команды
использовались только для создания искусственных fixtures и отрицательных
foreign-key writes; разрешение scope и все проверяемые чтения выполнялись
production-классами.

Безопасный результат research-части:

```json
{"ambiguous_scope":"rejected","authorization_overhead":"passed","authorization_p95_budget_ms":15.0,"authorization_p95_overhead_ms":3.849,"cross_organization_repository_read":"rejected","database_constraints":{"job_membership":"passed","lazy_materialization_parent":"passed","top_variant_parent":"passed"},"direct_read_ms":{"max":1.908,"min":1.116,"p50":1.215,"p95":1.449,"samples":50.0,"stddev":0.121,"warmups":10.0},"missing_scope":"rejected","organization_cache_namespace":"passed","organization_idempotency_namespace":"passed","production_candle_reader":"OrganizationScopedCanonicalCandleReader","production_repository_adapter":"PostgresBacktestJobRepository","production_scope_resolver":"PostgresResearchOrganizationScopeResolver","request_hash_parity":"passed","request_hash_pipeline":"build_research_content_hash","schema":"io.roehub.research-tenancy-runtime-proof/v1alpha1","scoped_read_ms":{"max":6.537,"min":4.752,"p50":5.038,"p95":5.298,"samples":50.0,"stddev":0.248,"warmups":10.0},"server_derived_scope":"passed","shared_candle_parity":"passed","shared_canonical_rows":120}
```

Доказательство подтвердило:

1. два независимых server-derived scopes и fail-closed отсутствие/неоднозначность;
2. owned reads и пустые cross-org/cross-user reads через
   `PostgresBacktestJobRepository`;
3. отказ трёх cross-owner writes на реальных foreign keys;
4. 120 общих canonical candles для обеих организаций без дублирования;
5. одинаковый digest columnar batch для прямого и обоих авторизованных чтений
   через `OrganizationScopedCanonicalCandleReader`;
6. разные production idempotency/cache hashes при одинаковом публичном ключе и
   одинаковый organization-neutral content hash через
   `build_research_content_hash`;
7. полную очистку контейнеров и volumes после проверки.

## Детерминизм и производительность

Compute kernels, signal rules, ClickHouse candle mapping и artifact byte
formats Stage `09` не изменял. Полный suite включает существующие golden/parity
проверки этих контуров и прошёл. Новый тест дополнительно сохраняет нейтральность
`request_hash`, а реальный probe сравнивает точный digest общей строки для двух
организаций.

Изменённая горячая граница — только authorization перед ClickHouse read.
Сопоставимый benchmark выполнен в одном контейнерном прогоне на одном batch из
120 свечей и одном ClickHouse-клиенте. Для каждого пути выполнено 10 прогревов
и 50 измерений; порядок baseline/candidate чередовался, состояние ClickHouse
cache было одинаково прогретым:

| Путь | p50 | p95 |
|---|---:|---:|
| прямое ClickHouse чтение | `1.215 ms` | `1.449 ms` |
| PostgreSQL scope + то же ClickHouse чтение | `5.038 ms` | `5.298 ms` |
| абсолютная добавка | `3.823 ms` | `3.849 ms` |

Fail-closed бюджет `p95 <= direct p95 + 15 ms` пройден с запасом `11.151 ms`.
Бюджет ограничивает один дополнительный индексированный PostgreSQL lookup на
локальной self-hosted сети; он не маскирует изменение ClickHouse workload.
Это доказательство накладных расходов новой authorization boundary, но не
утверждение о производительности ML/RL kernels или macOS M3 Pro; они не
менялись и остаются в границе Stage `24`.

## Проверки качества

- `uv run ruff check .` — `passed`.
- Целевой `pyright` по backtest/backtest_artifacts/API/migrations/tests/scripts/
  release tools — `0 errors, 0 warnings`.
- Общий `uv run pyright` — проанализировано `1263` файла; `155` существующих
  несвязанных ошибок только в восьми файлах (`149` в `local_artifacts/rl_trading`,
  по `2` в старом secrets transport test и двух exchange cleanup tools).
  Файлов Stage `09` среди них нет; чужие файлы не изменялись.
- Отрицательные API/production-reader tests доказали отказ для другой
  организации, другого пользователя, отсутствующего и неоднозначного scope до
  repository/variant/registry/ClickHouse вызова.
- Полный `uv run pytest -q` после синхронизации generated artifacts —
  `1786 passed`, `4` существующих `httpx` deprecation warnings. Первый прогон
  один раз поймал времязависимый unrelated RL Stage `17` load-test; изолированный
  повтор прошёл, повторный полный прогон также прошёл.
- OSS metadata write/check — `passed`; в direct dependency policy добавлены
  подтверждённые wheel-лицензии `argon2-cffi=MIT` и
  `webauthn=BSD-3-Clause`, появившиеся в Stage `06`.
- Runtime input inventory write/check — `passed`, `139` имён без значений.
- Installation/release golden matrix — `21 passed`; generation-manifest hashes
  остались детерминированными после пересборки metadata.
- `git diff --check` — `passed` после документационной финализации.

## Контракты и совместимость

| Поверхность | Классификация | Обоснование |
|---|---|---|
| Backtest API/DTO | `breaking-change` | Ответ job получил обязательный `organization_id`; authority не принимается от клиента. |
| Domain entities и ports | `breaking-change` | `organization_id` и resolver стали обязательными; repository/trigger/lazy signatures scoped. |
| Persistence | `breaking-change` | NOT NULL ownership, composite foreign keys, unique/index identity и clean-only migration. |
| Request/idempotency/cache identity | `breaking-change` | Org-neutral content hash сохранён, но write/dedupe/cache namespace стал versioned и organization-scoped. |
| Shared market-data access | `breaking-change` | Ранее доступные authenticated routes теперь требуют единственный server-derived organization scope и fail closed до reader. Канонические строки не копируются. |
| Indicator/compute outputs | `none` | Kernels и байтовые форматы не изменялись; parity suite прошёл. |
| Конфигурация | `compatible-change` | Добавлен одноразовый proof flag и узкому storage-migrations образу — закреплённые runtime dependencies/отключение неиспользуемого JIT. |
| Межсервисные вызовы | `none` | Новых сетевых зависимостей нет; PostgreSQL scope precedes существующий ClickHouse reader. |
| Внешние эффекты | `none` для production | Только disposable containers и artificial rows; cleanup прошёл. |
| Browser/UI defaults | `none` | Browser surfaces не менялись. |

Основная классификация этапа — `breaking-change`, допустимая для greenfield v1
без backfill и dual-read.

## Файлы этапа

Созданы migration `0014`, research scope port/PostgreSQL adapter, organization-
scoped market-data service, versioned research identity helper, runtime probe,
focused tests и этот отчёт.

Изменены Backtest domain/DTO/ports/use cases/worker/queue, оба PostgreSQL job
repositories, lazy materialization/cache, API DTO/errors/wiring/routes tests,
storage bootstrap/status/manifest/verifier и узкий storage-migrations image,
benchmark fixtures, platform plan, runtime inventory, OSS metadata и
installation golden hashes.

Вне основного prompt manifest изменены `apps/migrations`,
`configs/installation/runtime-input-inventory.json`, `tools/release` и
installation golden: реальная fresh-schema проверка обязана встраиваться в
принятый Stage `04` lifecycle, Stage `03` обязан инвентаризировать новый proof
input, а полный test gate выявил рассогласование ранее добавленных Stage `06`
dependencies с Stage `01` license artifacts. Значения зависимостей, секреты и
production inputs не читались.

Удалённых tracked-файлов нет. `.codex/PLANS.md`, `local_artifacts`, старые
exchange cleanup tools, production data/secrets и прочие чужие изменения
сохранены. Commit, staging, push, deploy и production mutation не выполнялись.

## Холодная проверка

- Режим: единственная проверка `independent subagent`, затем холодная локальная
  повторная проверка без второго независимого ревью.
- Первоначальный вердикт: `Block`.
- Исправленные блокирующие и высокие замечания:
  1. production Strategy variant launch теперь разрешает организацию и передаёт
     её в обязательные repository reads;
  2. indicators и market-data reference routes получили реальную fail-closed
     authorization boundary, а organization-scoped candle reader включён в
     production-class runtime proof;
  3. raw-SQL closures оставлены только для fixtures/FK-denials; positive и
     negative reads выполняют production resolver/repository/reader;
  4. benchmark заменён на 120 свечей, 10 прогревов, 50 чередующихся измерений,
     min/p50/p95/max/stddev и обоснованный `15 ms` p95 budget;
  5. lazy worker/child используют `organization_id`, content parity вызывает
     production `build_research_content_hash`, shared-data классификация
     исправлена на `breaking-change`.
- Локальная повторная проверка: `completed`.
- Результат после исправлений: `Release after fixes`.
- Доказательства повторной проверки: real Docker proof, отрицательные
  route/reader tests, `1786 passed`, scoped `pyright` `0`, ruff, OSS/inventory,
  installation golden, docs/project-map и `git diff --check` прошли.
- Остаточные риски: multi-organization session selection пока fail-closed как
  ambiguous; full container profile и macOS ML/RL performance принадлежат
  последующим этапам. Production data и API process deployment не проверялись и
  не входят в proof boundary `N/A`.

## Передача Stage 10

После принятия `09` Stage `10` получает server-derived
organization scope, fresh-schema composite ownership pattern, organization-
scoped versioned identities и доказанную shared immutable data boundary.
Production databases и legacy ownership не являются входом следующего этапа.
