# Stage 20 — наблюдаемость и разрешённые эксплуатационные действия

## Статус

- Этап: `20`.
- Состояние: `accepted`.
- Режим: `goal_driven`.
- Граница доказательств: `N/A` — изолированный локальный профиль `trading`,
  синтетическая административная учётная запись и управляемые инъекции
  отказов. Production incident mutation, текущие данные, внешние уведомления,
  реальные поставщики, реальные ордера, staging, deploy и публикация
  исключены.
- Реальная граница: настоящий Docker Engine и Compose подняли продуктовые и
  наблюдающие контейнеры; настоящий Chromium открыл работающий FastAPI/Jinja
  интерфейс. Это не принятие только по тестам.

## Результат кандидата

Наблюдаемость вынесена в независимый контейнерный слой. Prometheus,
Alertmanager, blackbox exporter, Grafana, Loki и сервис
`operational-health` имеют фиксированные multi-architecture digest, отдельные
readiness-проверки и остаются работающими при остановке Web, API, рабочего
процесса, PostgreSQL, ClickHouse, Redis, OpenBao и plugin gateway. Prometheus,
Alertmanager, Grafana и Loki получили отдельные постоянные тома.

Контракт `io.roehub.operational-health/v1alpha1` переводит ограниченные
технические сигналы в состояния `ready`, `degraded`, `stopped`, `unknown`,
затронутую возможность, инструкцию и действие из разрешённого списка. Метрики,
журналы, снимок состояния и административный DTO не содержат идентификаторы
организации, аккаунта, секреты, значения provider payload или произвольные
метки. Тип probe и ограниченные поля ответа задаются отдельно для каждого
сервиса: HTTP `200` с доменным `degraded` не превращается в `ready`, а простая
достижимость без readiness-семантики остаётся `unknown`. При недоступности
сервиса состояния fail safe переходят в `stopped` или `unknown` в зависимости
от подтверждённой причины.

Снимок хранит время последнего завершённого refresh, состояние фонового
процесса и freshness SLA. Просроченный снимок становится `unknown`, собственная
readiness возвращает HTTP `503`, а отдельный alert фиксирует и разрешает
нарушение свежести.

Административная страница получила отдельный раздел эксплуатационного
состояния. Глубокая диагностика не использует анонимное встраивание Grafana:
анонимный API Grafana запрещён, `iframe` отсутствует. Перезапуск связывается с
существующей типизированной операцией Stage `18`, требует
`operations.execute`, свежую аутентификацию и точную фразу подтверждения;
наблюдатель видит состояние и инструкции, но действие отключено. Серверная
матрица разрешает restart только для безопасного остановленного подмножества;
stateful, secret и monitoring services получают только диагностику. Control
agent ждёт здорового post-condition после разрешённого restart.

## Топология и поставка

- Prometheus:
  `prom/prometheus:v3.5.0@sha256:63805ebb8d2b3920190daf1cb14a60871b16fd38bed42b857a3182bc621f4996`.
- Alertmanager:
  `prom/alertmanager:v0.28.1@sha256:27c475db5fb156cab31d5c18a4251ac7ed567746a2483ff264516437a39b15ba`.
- Blackbox exporter:
  `prom/blackbox-exporter:v0.27.0@sha256:a50c4c0eda297baa1678cd4dc4712a67fdea713b832d43ce7fcc5f9bea05094d`.
- Grafana:
  `grafana/grafana:12.0.2@sha256:b5b59bfc7561634c2d7b136c4543d702ebcc94a3da477f21ff26f89ffd4214fa`.
- Loki:
  `grafana/loki:3.5.1@sha256:a74594532eec4cc313401beedc4dd2708c43674c032084b1aeb87c14a5be1745`.
- Конфигурации Prometheus, Alertmanager, blackbox exporter, Grafana и Loki,
  правила, панели, источники данных и operational manifest генерируются из
  канонической release/runtime topology, а не редактируются внутри
  контейнеров.
- Контейнеры наблюдаемости работают без root, с read-only root filesystem,
  `cap_drop: ALL` и `no-new-privileges`; пароль Grafana поступает через
  secret file после `secret-init`.
- Обязательства для распространяемых AGPL-артефактов Grafana и Loki записаны
  в `tools/release/oss_policy.json`; release metadata, SPDX и notices
  регенерированы.

## Эксплуатационные контракты и инструкции

- `io.roehub.operational-manifest/v1alpha1` описывает конечный набор сервисов,
  возможностей, probes, инструкций и разрешённых действий.
- `io.roehub.operational-health/v1alpha1` описывает снимок состояния без
  высококардинальных или чувствительных значений.
- Добавлены инструкции `runtime.service-degraded` и
  `runtime.observability-unavailable`; генератор проверяет существование всех
  ссылок `/runbooks/...` в правилах и динамического шаблона инструкции.
- Сгенерированы 9 русских документов и индекс; 20 старых инструкций остаются
  явно неперенесёнными и не выдаются за канонические.
- Alertmanager использует только локальный получатель `local-audit-only`;
  внешний канал уведомлений в доказательстве не включался.

## Реальная проверка среды исполнения

Доказательство:
`docs/architecture/platform/roehub-self-hosted-oss-platform-v1-stage-reports/evidence/20-observability-runtime-proof.json`.

- Compose-проект `roehub-stage20-fix2` поднял сгенерированный профиль
  `trading`. В базовой выборке `api`, `web` и `strategy-live-runner` были
  `ready`; PostgreSQL, ClickHouse и Redis — `unknown`, потому что их TCP
  достижимость не выдаётся за readiness; новый запечатанный OpenBao и plugin
  gateway — `degraded`.
- Доказаны все четыре состояния: `ready`, `degraded`, `stopped`, `unknown`.
  В частности, plugin gateway вернул HTTP `200` с доменным состоянием
  `degraded` и остался `degraded` в operational snapshot.
- Пауза API привела к `unknown` с `probe.timeout`, после восстановления — к
  `ready`. Отдельная инъекция просроченного refresh дала HTTP `503`,
  `snapshot_stale`, диагностику без restart и полный цикл alert
  `fired-and-resolved`.
- Последовательно остановлены и восстановлены `web`, `api`,
  `strategy-live-runner`, `postgresql`, `clickhouse`, `redis`, `openbao` и
  `plugin-gateway`.
- Все восемь отказов перешли в `stopped`; после восстановления HTTP-сервисы
  вернулись к своим доменным состояниям, stateful TCP services — к `unknown`,
  а свежий запечатанный OpenBao — к `degraded`.
- Во всех восьми сценариях `alertmanager`, `blackbox`, `grafana`, `loki`,
  `operational-health` и `prometheus` оставались запущены.
- Алерт API появился и разрешился в Prometheus и Alertmanager.
- После перезапуска Loki число сохранённых переходов осталось `6` до и после;
  ряд отказа Prometheus также сохранился после перезапуска.
- Проверено наличие четырёх постоянных томов наблюдаемости и запрет
  анонимного Grafana API.
- Verifier проверил `docker compose down`, его exit status и отсутствие
  остаточных ресурсов до финальной записи доказательства. Контейнеры, сеть и
  тома проекта `roehub-stage20-fix2` полностью удалены.

## Реальная браузерная проверка

Доказательство:
`docs/architecture/platform/roehub-self-hosted-oss-platform-v1-stage-reports/evidence/20-admin-operational-health-browser-proof.json`.

- Настоящий Chromium прошёл цепочку Web HTTP proxy → настоящий admin API route
  → `operational-health` HTTP service → сгенерированная Docker topology.
  Локальная identity была синтетической, а typed control backend —
  безвредным и не делегирующим Docker-effect.
- При реальной остановке контейнера `api` UI показал `api=Stopped`, после
  восстановления — `api=Ready`. Одновременно отображены настоящий
  `plugin-gateway=Degraded`, `openbao=Degraded` и диагностические
  `postgresql=Unknown`, `redis=Unknown`.
- Ссылка `/runbooks/web.api-health-degraded` открыла сгенерированную русскую
  инструкцию с HTTP `200` через защищённый Web renderer.
- Для владельца restart активировался только после точной фразы. Запрос прошёл
  настоящий typed HTTP-контракт с HTTP `202`, а безвредный backend записал
  ровно `action=restart`, `profile=base`, `services=["api"]`.
- Принудительная отправка от `viewer` получила HTTP `403` с
  `organization_permission_denied`; отправка с просроченной recent-auth — HTTP
  `403` с `recent_auth_required`. Действия диагностики для ready/degraded/
  unknown-состояний в UI отключены как не-restart.
- Русская локаль показывает `Операционное здоровье`, `Остановлен` и
  `Перезапустить`.
- В разрешённом сценарии были только HTTP `200` и `202`, неожиданных ответов
  `>=400` не было. Два HTTP `403` в отрицательных сценариях ожидаемы и записаны
  как доказательство server-side запретов.
- Ширины `375`, `768`, `1440` не создают document overflow; на `375` таблица
  прокручивается только внутри собственного контейнера.
- Проверены снимки `375x812` и `1440x900`; синтетическая организация не
  является production data, а эксплуатационная таблица не содержит полей
  организации, аккаунта или секретов.
- SHA-256 снимка `375x812`:
  `3f9b3609486ad4546b31a5844593c10013827ea8fdc55199d8df7181698c6ed5`;
  снимка `1440x900`:
  `f0caf4bc4e5d0f71c7c86ecde8845c83f0911b7854b5fa713ef15fef778d2e81`.
- Проект `roehub-stage20-browser1` после проверки удалён; остаточных
  контейнеров, сетей и томов нет.

## Исправления по независимой и реальным проверкам

1. Generic HTTP `2xx=ready` заменён типизированными probes и ограниченным
   разбором domain payload. Release manifest использует настоящие readiness
   endpoints; `/health/live`, `/metrics` и TCP reachability не объявляются
   доменной готовностью.
2. Blanket restart удалён. Server-side action matrix, API и control policy
   разрешают restart только для безопасных остановленных сервисов; stateful,
   secret и monitoring services получают diagnostics-only. Control agent ждёт
   здорового post-condition.
3. Синтетический browser-only `MockTransport` заменён связанной реальной
   цепочкой Web → API → operational-health → Docker topology. Положительный
   restart прошёл безвредный typed backend, а `viewer` и stale recent-auth
   получили server-side deny.
4. `operational-health` больше не получает общий product environment,
   PostgreSQL DSN, token/password paths или общий `runtime-secrets` volume.
   Grafana secret вынесен в отдельный `grafana-secrets`; отрицательные topology
   tests фиксируют least privilege.
5. Добавлены last-completed refresh timestamp, freshness SLA, worker state,
   fail-closed HTTP `503`, alert и управляемая инъекция зависшего refresh.
6. Cleanup выполняется и проверяется до записи evidence; project id, exit status
   и нулевые списки контейнеров, сетей и томов входят в артефакт.
7. Добавлены alert на потерю отправки переходов в Loki и OpenBao coverage;
   машинные runbook links и `monitoring_gaps` повторно сгенерированы и
   проверены.
8. Реальная browser-цепочка обнаружила, что Web proxy удалял namespace `/api`
   и отправлял `/v1/admin/...`, получая HTTP `404`. Proxy сохраняет
   `/api/v1/...`; regression test закрывает дефект.
9. API при одном только общем `ROEHUB_STORAGE_POSTGRES_DSN` больше не включает
   частично настроенный control-agent client; неполный управляющий набор
   по-прежнему завершается fail closed.
10. Сетевой bind, повторное получение host ports после restart и ленивые
    monitoring exports приведены к проверяемому runtime-контракту без изменения
    безопасных локальных значений по умолчанию.

## Контрактное влияние

| Измерение | Классификация | Обоснование |
|---|---|---|
| API и DTO | `compatible-change` | Admin snapshot получил добавочное поле `operational_health`; недоступный health backend отображается как `unknown`, существующие поля сохранены. |
| Порты и adapters | `compatible-change` | Добавлен ограниченный HTTP-клиент operational health и ленивые monitoring exports; существующие обязательные application ports не расширены. |
| Хранение | `compatible-change` | Добавлены четыре независимых named volumes наблюдаемости; таблицы, миграции и persisted product defaults не изменены. |
| Runtime topology и конфигурация | `breaking-change` | В Compose и release manifest добавлены шесть сервисов, fixed digests, probes, тома, secret file и новые URL/bind inputs; требуется согласованная регенерация профиля. |
| Идентичность и RBAC | `none` | Новые роли и permissions не вводились; restart использует существующие `operations.execute` и `recent-auth`. |
| Идемпотентность, request и cache identity | `none` | Новых request/cache keys нет; restart наследует organization-bound `Idempotency-Key` Stage `19`. |
| Generation hashes | `breaking-change` | Runtime input inventory и generation manifests изменились; golden SHA-256 обновлены, старые release bundle hashes несовместимы. |
| Межсервисные вызовы | `compatible-change` | Добавлены bounded probes и API → operational-health с timeout; автоматические provider retries или произвольные команды не появились. |
| Внешние эффекты | `breaking-change` | Admin surface теперь может отправить типизированный restart безопасного остановленного сервиса; policy сужает допустимое множество и требует healthy post-condition. В доказательстве использован безвредный backend, внешние уведомления, provider writes и реальные ордера отсутствовали. |
| Аудит и журналы | `compatible-change` | Добавлены редактированные bounded transition events в Loki без organization/account/secret labels. |
| Инструкции | `compatible-change` | Добавлены две канонические машинно-читаемые инструкции и строгая проверка alert links. |
| Browser defaults | `breaking-change` | Добавлен новый operational-health раздел и подтверждаемое restart-действие; темы, shell, локали и отсутствие анонимного Grafana embedding сохранены, но видимые возможности admin UI изменились. |

## Проверки принятого этапа

- `real-boundary runtime proof`: `passed` — настоящий Docker Engine/Compose,
  восемь управляемых отказов, timeout/stale-инъекции, alerts, сохранность
  Prometheus/Loki и проверенный cleanup без остаточных ресурсов.
- `real-boundary browser proof`: `passed` — настоящий Chromium и цепочка Web →
  API → operational-health → Docker topology; HTTP `202` через безвредный typed
  backend и два server-side HTTP `403` для запрещённых сценариев.
- Сфокусированный набор после последних исправлений: `67 passed`,
  `3 warnings`.
- Полный `pytest`: `1966 passed`, `4 warnings` за `79.83s`; предупреждения —
  существующая deprecation-диагностика тестового `httpx cookies=`.
- Полный `ruff`: `passed`.
- Targeted `pyright`: `0 errors, 0 warnings`.
- Полный `pyright` остаётся красным из-за `153` ошибок в чужих
  `local_artifacts` и несвязанных legacy tools; затронутые Stage `20` пути
  проверены отдельно и чисты.
- `node --check apps/web/dist/js/pages/admin.js`: `passed`.
- `docker compose config` для `base`, `trading`, `ml`: `passed`.
- `promtool`: конфигурация и 2 правила `passed`; `amtool`, blackbox
  `--config.check` и Loki `-verify-config=true`: `passed`.
- `oss_metadata.py --check`, `generate_runtime_topology.py --check`,
  `runtime_input_inventory.py --check` и `generate_runbooks.py --check`:
  `passed`.
- Project map, docs index и `git diff --check`: `passed` после финального текста
  отчёта и журнала.

## Независимая проверка

- Режим: единственный `independent subagent`, read-only.
- Вердикт: `Block`: 3 `Blocker`, 3 `High`, 3 `Medium`.
- Исправления: закрыты доменная readiness-семантика, безопасная матрица
  действий и post-condition, реальная browser/runtime связь, least-privilege
  monitoring, freshness, воспроизводимый cleanup, два alert/runbook gap и
  заниженная классификация контрактов.
- Локальная холодная повторная проверка после исправлений: `Release after
  fixes`. Второй независимый review не запускался.

## Остаточные границы и запрещённые действия

- 20 старых инструкций ещё не перенесены; принятые 9 полностью покрывают
  текущие машинные ссылки, а остальные не объявлены готовыми.
- Локальный audit receiver доказывает lifecycle алерта, но не внешний канал;
  real notification canary требует отдельного разрешения и не нужен для
  Stage `20`.
- Фактический доступ к corresponding source для распространяемых AGPL-образов
  Grafana/Loki остаётся обязательным воротом release stage; локальный Stage
  `20` проверяет policy/notices/SPDX, но не объявляет публикацию выполненной.
- Текущая production authentication, production incidents и real Grafana SSO
  не проверялись в границе `N/A`.
- Production/current data, реальные учётные данные, provider payloads и ордера
  не читались и не изменялись. Commit, push, deploy, staging и production
  mutation не выполнялись.
