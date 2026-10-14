# Инициализация и миграции хранилищ

Все профили `base`, `trading`, `ml` используют PostgreSQL 16 и Redis 7.2.
`roehub.yaml`, generated runtime config и storage status имеют схему `v2alpha1`.
Старые схемы и удалённые store keys отвергаются до bootstrap. Действующие
установки автоматически не переключаются.

## Bootstrap

```bash
python -m apps.migrations.storage_main bootstrap --service-config /etc/roehub/service-config.json
python -m apps.migrations.storage_main readiness --service-config /etc/roehub/service-config.json
```

Порядок: validate config и endpoints, проверить возможности PostgreSQL/Redis,
применить identity migrations, Alembic и оставшиеся PostgreSQL phases из
`migrations/postgres/manifest.json`, выдать application DML и maintenance grants,
проверить итоговый статус. Applied SQL/checksums не изменяются. Незавершённая SQL
phase откатывается транзакционно; повторный bootstrap сверяет completed checksums.
Checksum drift требует новой миграции, а не изменения истории.

DSN передаётся через `ROEHUB_STORAGE_POSTGRES_DSN`, Redis URL через
`ROEHUB_STORAGE_REDIS_URL`. Пароли остаются в secret files; DSN, credentials и raw
provider payloads не включаются в status или доказательства. External PostgreSQL
должен заранее иметь application/monitor/maintenance роли и корректный TLS CA.
Один explicit endpoint сохраняет исходный host для TLS verification; libpq
multihost, service overrides и encryption fallback в counted installation запрещены.

## Роли и партиции

Migration principal применяет DDL и grants. `roehub_app` использует DML;
`roehub_maintenance` вызывает bounded `market_data.ensure_partitions` через
maintenance lifecycle. Observer выделяет current/next month и ограниченное число
исторических месяцев из work requests и stream_history_repair receipts
(pending/failed). Keyset-проход ограничен 64 requests и 12 отсутствующими
месяцами за цикл; повторный цикл не создаёт существующие партиции. Отсутствие
market-партиции даёт retryable storage_partition_pending до её подготовки.
После исправления прежние failed receipts явно rearm через operator action.
API/worker admission не выдаёт application
ролью partition DDL. Перед исторической загрузкой нужные партиции должны существовать.

## Общий бюджет соединений

Installation clients используют один защищённый POSIX volume на одном Docker host:
`market=16`, `control=24`, `admin=8`. Reservation создаётся до handshake и остаётся
занятой до подтверждённого завершения backend. Admin observer/reaper входит в 8.
Каждая replica обязана видеть тот же volume; независимые локальные registries не
дают installation-wide лимит. `max_connections=48` — дополнительный server cap.

Освобождение требует raw TCP EOF и Linux `TCP_INFO=CLOSE_WAIT` до локального
shutdown: PG16 удерживает серверный socket до завершения backend-процесса. RST,
timeout и отсутствие строки в `pg_stat_activity` не освобождают слот. Для idle TLS
клиент завершает свой backend через bounded rollback/self-termination, дожидается
того же FIN witness и только затем закрывает libpq. ACTIVE/COPY/pipeline и неизвестные
исходы остаются fail-closed. CancelRequest не создаёт дополнительного сокета.

При crash владельца duplicate socket теряется, reservation остаётся quarantined.
Автоматического reaper нет. Оператор восстанавливает бюджет через проверенную
остановку именно enrolled PostgreSQL, с блокировкой всех 48 слотов. Серверу не нужен
общий filesystem с клиентами; административный канал — доверенный Docker Engine
UNIX socket, в том числе через SSH forwarding. Он не монтируется в application containers.

### Enrollment и operator recovery

Выполнять из доверенного Linux operator environment с Python/psycopg и доступом
к **тому же клиентскому registry volume**. Значения `PG_CONTAINER` и `PG_DOCKER_SOCKET`
выбирает оператор для конкретной установки; DSN/`PGPASSFILE`/TLS CA берутся из её
admin secret configuration. Не передавать пароль в CLI и не сохранять environment.
Для external PG нужны выделенный Docker container с private PID namespace и прямой
TCP endpoint. Другой control plane требует отдельного проверенного recovery adapter.

Перед enrollment остановить application owners и внешние supervisors. Endpoint
IP:port должен неизменно принадлежать выбранному PostgreSQL весь срок enrollment:
TCP proxies, transparent failover и переназначение того же IP/NAT mapping не входят
в эту процедуру. До изменения адреса/маршрута нужно завершить старые соединения,
выполнить recovery и отдельно переоформить enrollment. Смена DNS/IP fail-closed;
не подменять mapping вручную, чтобы обойти незавершённые попытки.

```bash
# Root в operator environment; ROEHUB_PG_BUDGET_ROOT указывает на client registry.
python -m tools.installation.postgres_budget_recovery enroll \
  --registry "$ROEHUB_PG_BUDGET_ROOT" --docker-socket "$PG_DOCKER_SOCKET" \
  --container "$PG_CONTAINER"
```

Enrollment связывает counted SQL connection с backend-owned TCP socket в выбранном
container, проверяет system identifier, epoch, daemon/container ID и data mount
identity. Root-owned endpoint mapping доступен клиентам только для чтения; каждая
попытка записывает resolved endpoint до открытия сокета. Runtime secrets не входят
в mapping. Процедура отказывает при активных owners, неизвестных/чужих endpoint
records или уже существующем enrollment.

```bash
python -m tools.installation.postgres_budget_recovery recover \
  --registry "$ROEHUB_PG_BUDGET_ROOT" --docker-socket "$PG_DOCKER_SOCKET" \
  --container "$PG_CONTAINER"
python -m tools.installation.postgres_budget_recovery resume \
  --registry "$ROEHUB_PG_BUDGET_ROOT" --docker-socket "$PG_DOCKER_SOCKET" \
  --container "$PG_CONTAINER"
```

`recover` устанавливает durable pending/stopping intent, отключает Docker restart,
останавливает enrolled server и требует `Running=false`, `Restarting=false`, `Pid=0`.
Только после этого очищает существующие slot inodes. Потеря ответа на stop допускает
повтор `recover`; неизвестный исход не даёт права ручного удаления registry.
`resume` запускает тот же target, ждёт строку 8 `postmaster.pid=ready`, проверяет
counted admin connection и публикует новый epoch. Обычный server restart также требует
этого явного восстановления epoch. Оба шага ограничены deadline; ошибка оставляет
admission закрытым. Restart policy остаётся `no`: оператор возвращает согласованную
политику только после успешного resume, затем запускает clients/supervisors.

### Нативные PostgreSQL 16 utilities

В том же доверенном Linux environment должны быть `pg_dump`, `pg_restore`, `psql`
версии 16, Python/psycopg и исходники Roehub. Использовать counted wrapper:

```bash
python -m tools.installation.postgres_native pg_dump \
  --format=custom --no-owner --no-privileges > /protected/roehub.dump
python -m tools.installation.postgres_native pg_restore \
  --exit-on-error --no-owner --no-privileges /protected/roehub.dump
python -m tools.installation.postgres_native psql -c 'SELECT 1'
```

Wrapper резервирует один admin slot и допускает один upstream session. Параллельные
jobs, reconnect, сокращённые/clustered connection flags и psql meta-commands запрещены.
`pgpass` сопоставляется с исходным remote host/port до назначения loopback port;
пароль передаётся дочернему процессу через environment, не argv. CLI timeout — 300 s,
явный максимум — 3600 s. Dump/restore должны дополнительно соблюдать backup quiesce,
digest/signature и owner contracts; wrapper сам их не заменяет.

Поддержаны `sslmode=disable` и `verify-full` с явным `sslrootcert`. Для TLS Python
relay проверяет CA и исходный hostname, а native utility общается с ним по plaintext
loopback на доверенном operator host. Этот локальный участок — отдельная граница
доверия; не использовать shared/untrusted network namespace. Дополнительные libpq TLS
options и обязательный channel binding отвергаются, а не игнорируются. Relay не
посылает TLS close_notify после Terminate и освобождает слот только по серверному FIN.

## Backup и disposable proof

Согласованный PostgreSQL dump должен включать public и market_data schemas,
partition data, actual/absence index, generation/revisions и verified frontier.
Redis checkpoint не заменяет PostgreSQL. OpenBao и остальные owners сохраняют
отдельные recovery/security contracts из backup runbook.

```bash
uv run python -m apps.migrations.verify_storage_runtime
```

Verifier использует только disposable volumes, проверяет interruption/rollback,
повторный bootstrap и external addressing. Результат одного профиля не доказывает
полную matrix или текущую установленную версию. SQL0001–0039 сохраняют свои bytes.

## Явное принятие совместимой legacy-схемы

Если существующая установка имеет таблицы, но не имеет части SQL-маркеров,
обычный bootstrap может отказать на greenfield guard. Не исполнять повторно DDL
над пользовательскими данными и не записывать фиктивную историю выполнения.
Сначала сохранить и проверить восстановление полного PostgreSQL dump, private
configuration и artifact root; остановить владельцев данных.

`tools.installation.adopt_legacy_postgres` поддерживает только проверенный gap
исторических фаз 0011–0024 и 0026 из явного allowlist.
Остальные фазы, включая 0036–0039, должны иметь реальные applied markers с текущими
checksums. Если PG-only фазы ещё отсутствуют, оператор применяет именно новые
аддитивные фазы по immutable manifest под migration lock до принятия legacy-схемы.
Нельзя выводить исполнение data migrations из одного наличия таблиц.

На отдельном пустом PG16 применить **весь** обычный bootstrap той же ревизии,
затем создать witness. DSN выбирается из private environment, не из CLI:

```bash
python -m tools.installation.adopt_legacy_postgres witness --reference /protected/reference.json
# Переключить private DSN на quiesced target, сохраняя installation budget.
python -m tools.installation.adopt_legacy_postgres adopt --reference /protected/reference.json
```

Witness связывает immutable phase checksums и структуру public: columns/defaults,
constraints, indexes, functions, triggers и RLS. Проверяются обязательные provider
seed identities, audit cursor и наличие событий существующих work requests;
market schema и current/next partitions также обязательны. Единственная
нормализация — точная redundant single-column credential FK при наличии её
validated organization-scoped composite FK; физически обе сохраняются. Иные
расхождения требуют расследования, а не расширения исключений на месте.

Принятие сохраняет отдельные `postgres-sql-adopted` и
`postgres-adoption-reference` записи. Это подтверждение текущей совместимости,
**не** утверждение исторического исполнения SQL. Повтор процедуры идемпотентен.
После принятия обязательно выполнить обычный bootstrap/readiness повторно;
на последующих запусках он использует проверенные checksums и provenance.

## Замена enrolled Docker container

Обычные application restart/start не меняют enrollment PostgreSQL. Перед первым
enrollment создать PostgreSQL через тот же canonical generated Compose, который
использует controller: разные Compose image resolution/hash могут пересоздать
контейнер при следующем `up`. Stateful PostgreSQL restart через обычный controller
запрещён; planned maintenance использует `recover`/`resume` выше.

Если enrolled контейнер уже удалён Docker и создан новый с тем же data volume,
использовать отдельную процедуру, а не удалять authority/slots вручную. Остановить
всех clients и точный controller, отключить его restart policy; требуется
эксклюзивный операторский доступ к Docker. Endpoint и daemon должны остаться теми
же. Новый контейнер должен работать непосредственно с прежним PGDATA/cluster.

```bash
python -m tools.installation.postgres_budget_recovery replace-recover \
  --registry "$ROEHUB_PG_BUDGET_ROOT" --docker-socket "$PG_DOCKER_SOCKET" \
  --container "$PG_CONTAINER" --supervisor "$PG_CONTROLLER_CONTAINER"
python -m tools.installation.postgres_budget_recovery resume \
  --registry "$ROEHUB_PG_BUDGET_ROOT" --docker-socket "$PG_DOCKER_SOCKET" \
  --container "$PG_CONTAINER"
```

Под блокировкой всех 48 slots adapter требует точный Docker 404 старого ID,
остановленный controller, отсутствие других running owners тома, совпадающие
mount/cluster/endpoint и backend-owned listener нового PostgreSQL. HTTP 500,
timeout, чужой volume/target или активный owner ничего не разрешают. Старая pin
сохраняется как root-only retired authority; durable replacement intent связывает
старый и новый IDs. Только подтверждённый stop нового сервера очищает reservations.
Потерянный stop response и повтор уже завершённого stop безопасно повторяются с
тем же intent. `resume` проверяет новый epoch counted admin connection; затем
оператор возвращает согласованный supervisor и запускает clients. Смена адреса,
тома, кластера или неизвестный старый owner требует отдельного расследования.
