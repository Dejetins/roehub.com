# Этап 17 — контейнеризация всех сервисов и профилей

## Статус

- Этап: `17`.
- Статус: `accepted`; единственная независимая проверка дала `Block`, оба
  `Blocker` и оба `High` исправлены, локальный follow-up и полный повторный
  build+runtime proof прошли.
- Дата: `2026-07-13`.
- Режим: `goal_driven`.
- Граница доказательств: `N/A`; локальный Docker Engine, одноразовые сети и
  volumes, clean-install конфигурация `base`, `trading` и `ml`.
- Исключены: production deployment, импорт текущих production-данных, финальная
  multi-architecture матрица, реальные credentials, реальные заявки, commit,
  push и deploy.
- Следующий разрешённый этап: `18`.

## Результат

Добавлен канонический `io.roehub.runtime-service-manifest/v1alpha1`, из которого
генерируются внутренние Compose-файлы и service configs для `base`, `trading`
и `ml`. Пользователь не редактирует внутренний `.env` или Compose: topology
строится из release/config manifests и проверяется на детерминированность.

Inventory сопоставлен с актуальной project map. В контейнерный контракт входят
все runnable приложения, workers и schedulers:

- приложения `api`, `cli`, `common`, `exchange_control`,
  `exchange_execution`, `migrations`, `monitoring`, `plugin_gateway`,
  `scheduler` и `web`;
- workers `backtest_job_runner`, `job_runtime`, `market_data_ws`,
  `notification_dispatcher`, `notification_report_scheduler`,
  `rl_trading_inference`, `rl_trading_trainer`, `strategy_live_runner` и
  `telegram_bot_worker`;
- `roehubctl`, `job-runtime` и `rl-trainer` запускаются как изолированные
  on-demand роли, а не как постоянно работающие сервисы;
- `runtime_probe` является только проверочной ролью и не подменяет ни один
  продуктовый компонент.

Runtime images имеют версии `roehub/runtime:0.1.0-stage17` и
`roehub/runtime-ml:0.1.0-stage17`. Базы и инфраструктурные образы закреплены
точными версиями или digest; `latest` в генерируемой topology отсутствует.
Продуктовые runtime-процессы работают как `65532:65532`, с read-only root
filesystem, удалёнными capabilities, временными `tmpfs`, health endpoints и
явными ограничениями CPU и памяти. Одноразовый root `secret-init` имеет только
`CAP_CHOWN`, read-only rootfs и не имеет сети. Официальные init-entrypoints
PostgreSQL, Redis и ClickHouse получают root только для владения чистым томом,
после чего PID 1 работает как UID `999`, `999` и `101`; OpenBao — UID `100`.

PostgreSQL, Redis, ClickHouse, OpenBao data/audit, artifact state и ML state
вынесены в named volumes. Redis получает автоматически созданный file-backed пароль, включает
authentication, AOF и `noeviction`. ClickHouse использует официальный
`CLICKHOUSE_PASSWORD_FILE`. Значения секретов не попадают в Compose, отчёт или
evidence. Runtime clients получают file-backed authentication только внутри
процесса. Межконтейнерные зависимости используют service DNS; `127.0.0.1` и
`localhost` не являются адресами зависимостей.

Safe defaults едины для всех профилей:

- notifications работают в `log_only`, Telegram выключен;
- отдельные `notification-report-scheduler` и `telegram-bot-worker` запущены и
  готовы в безопасном режиме `disabled`; они больше не засчитываются через
  чужой dispatcher process;
- strategy producer и exchange consumer выключены;
- exchange adapter имеет режим `disabled`, поэтому `trading` не означает
  `mainnet` и не разрешает submit;
- RL inference в `ml` выключен до явной owner-конфигурации модели;
- OpenBao стартует неинициализированным и не получает автоматически созданных
  production-полномочий.

## Генерируемая topology

`base` запускает управляющий Web/API-контур, PostgreSQL, Redis и безопасные
сервисы notifications/artifacts. `trading` добавляет ClickHouse, market-data,
strategy, exchange control/execution и связанные workers. `ml` добавляет
RL inference и on-demand trainer. Optional сервис не завершает аварийно весь
control plane: liveness отделён от readiness, а зависимость сообщает
типизированное состояние деградации.

Генератор создаёт для каждого профиля Compose, redacted runtime config,
`market_data.yaml`, `indicators.yaml` и ML config. В release inventory включены
все входы, которые реально копируются в runtime image или монтируются в
контейнеры. Project map и OSS metadata учитывают новые незакоммиченные файлы,
поэтому текущая implementation не может быть пропущена только из-за состояния
Git index.

## Реальная граница проверки

Итоговая versioned запись находится в
[`evidence/17-runtime-topology-proof.json`](evidence/17-runtime-topology-proof.json)
и имеет `schema=io.roehub.runtime-topology-proof/v1alpha1`, `status=passed`.
Она создана полным прогоном без `--skip-build`; `compose_build=passed`. Image
IDs: `roehub/runtime:0.1.0-stage17` —
`sha256:a11a4211a4e7f68fc174c346ffc0f06eae2e21b6af2d07895b30d522857ea286`,
`roehub/runtime-ml:0.1.0-stage17` —
`sha256:f4af0919c88c48f5a9eb991a7bace5f3440b2b7f38fc5251899ae1e31501375d`.

Доказаны:

- real `docker compose config`, запуск и ожидание health для каждого профиля;
- девять запущенных контейнеров `base`, восемнадцать `trading` и девятнадцать
  `ml`;
- readiness PostgreSQL/Redis в `base` и PostgreSQL/Redis/ClickHouse в
  `trading`/`ml`;
- imports фактических entrypoints для 9/17/19 объявленных ролей;
- actual команды `roehubctl`, `job-runtime` и `rl-trainer` для применимых
  профилей;
- реальный Stage `15` `JobAttemptExecutor`/`OciJobRunner` lifecycle: OCI launch,
  signed artifact result, timeout/cancel cleanup, Docker socket denial и restart
  recovery;
- service DNS/TCP connectivity, non-root/read-only runtime и resource limits;
- перезапуск API с ростом boot counter `1 -> 2`;
- сохранение sentinel через полный `down`/`up`: PostgreSQL, Redis AOF, OpenBao
  data/audit и artifact volume во всех профилях; ClickHouse в `trading`/`ml`;
  ML state в `ml`;
- фактический startup Web/API и прикладная readiness всех 6/14/15 runtime
  сервисов; для metrics-based workers проверен их собственный operational
  marker, а ClickHouse exporter завершил успешный scrape;
- generated `artifact_root=/var/lib/roehub/artifacts/backtest/v2` совпадает с
  persistent mount и проверяется golden-тестом; build context исключает
  `.env`, private keys и certificate files;
- teardown и cleanup всех одноразовых контейнеров, сетей и volumes.

В `trading` и `ml` readiness `exchange-control` возвращает `503` до owner-init
OpenBao Transit, а readiness `exchange-execution` — до проверки PITR restore и
включения адаптера. В `ml` readiness `rl-inference` возвращает `503`, пока
inference и source events выключены. Это проверенная fail-closed деградация, а
не startup failure: liveness этих процессов, storage и control plane остаются
работоспособными, внешние эффекты невозможны.

Runtime smoke может выполнять публичное read-only получение market data в
`trading`/`ml`. Он не выполняет external provider writes, не создаёт реальные
заявки и не изменяет production state; evidence фиксирует
`external_provider_writes=false`, `real_order_effects=false` и
`production_mutation=false`.

## Проверки качества

- Целевой pytest — `102 passed`.
- Расширенный pytest — `309 passed`.
- Полный pytest — `1905 passed`, четыре прежних предупреждения `httpx`.
- Целевые `ruff` и `pyright` — `passed` и `0 errors`.
- Runtime generator write/check и Compose config для трёх профилей — `passed`.
- Runtime input inventory (`152`), OSS metadata/SPDX/notices и project map
  generation/check — `passed`.
- Полный build+runtime smoke без `--skip-build` — `passed`; все одноразовые
  Stage `17` и job-runtime ресурсы удалены.
- Docs index/project map generation/check и финальный `git diff --check` —
  `passed`.

## Контракты и совместимость

| Поверхность | Классификация | Обоснование |
|---|---|---|
| API, DTO и ports | `compatible-change` | Добавлены liveness endpoints и container-safe bind; существующие бизнес-DTO не удалены. |
| Persistence | `breaking-change` | Greenfield topology требует именованные volumes, Redis authentication/AOF и file-backed ClickHouse authentication. |
| Config/defaults | `breaking-change` | Новый manifest становится обязательным источником внутренних Compose/config; safe-disabled defaults и profile membership заданы явно. |
| Identity/hash | `breaking-change` | Runtime image tags, manifest/schema identity и generated profile content становятся versioned release identity. |
| Service calls | `breaking-change` | Все внутренние зависимости переходят на service DNS и file-backed runtime authentication. |
| External effects | `compatible-change` | По умолчанию provider writes, Telegram, strategy producer, RL inference и exchange submit выключены. |
| Secrets/trust | `breaking-change` | Redis/ClickHouse secrets создаются в file-backed volume; OpenBao остаётся fail-closed до owner-init. |
| Audit/runbook | `compatible-change` | Доказательство фиксирует readiness, restart, persistence, cleanup и отсутствие внешних write effects. |
| Browser defaults | `compatible-change` | Web/API доступны в `base`; опасные торговые и ML возможности остаются выключенными. |

Основная классификация — `breaking-change`, ожидаемая для новой greenfield
installation topology. По решению `A07` legacy import, backfill и dual-read не
добавляются.

## Независимая проверка

- Режим: одна cold independent review замороженного кандидата; после
  исправлений выполнен локальный follow-up без второй независимой проверки.
- Первоначальный вердикт: `Block` — два `Blocker` и два `High`.
- Исправлен первый `Blocker`: backtest `artifact_root` перенесён под реальный
  persistent mount, совпадение пути и сохранность тома доказаны.
- Исправлен второй `Blocker`: report scheduler и Telegram worker получили
  отдельные container entrypoints; `job-runtime` получил executable doctor, а
  verifier повторяет реальный `JobAttemptExecutor`/OCI lifecycle.
- Исправлен первый `High`: persistence теперь проверяет каждого владельца
  состояния, а не только PostgreSQL.
- Исправлен второй `High`: evidence содержит отдельный application readiness
  результат для каждого объявленного сервиса, а не принимает один только факт
  доступности произвольного `/metrics`.
- Дополнительно закрыты остаточные риски review: `.dockerignore` исключает
  secret-shaped build inputs; runtime PID и resources инфраструктуры доказаны.
- Локальный итог после исправлений: `Release after fixes`; `Blocker` и `High`
  отсутствуют.
- Остаточные риски: owner-init и Transit policy OpenBao относятся к lifecycle
  следующих этапов; PITR restore доказывается Stage `21`; реальная ML model и
  активные Telegram/report schedules требуют owner-конфигурации; финальная
  Linux/macOS и multi-architecture матрица относится к Stage `24`.

## Файлы и ограничения выполнения

Добавлены runtime manifest/schema, topology generator/verifier, Dockerfile и
entrypoint, runtime probe, генерируемые профили, health endpoints, storage auth
propagation, тесты, runtime evidence и обновлённые project map/OSS artifacts.

Чужие dirty изменения сохранены. Staging, commit, push, deploy, production
mutation и реальные order submit не выполнялись. Одноразовые контейнеры, сети,
volumes и job-runtime OCI resources после proof удалены.

## Передача Stage 18

Stage `18` разрешён после исправления всех замечаний единственной независимой
проверки, полного build+runtime proof, полного локального gate и перевода Stage
`17` в `accepted`. Он получает generated topology и безопасно деградировавшие
сервисы, но не получает право включать `mainnet`, Telegram или production
effects.

## Повторная проверка финальных образов для этапа 22

Runtime topology повторно проверена без пересборки на финальных
multi-architecture images:

- `runtime` —
  `sha256:e3303b08b337c24e451045985355047d0c383eefc39aa1965400ecc1e4a9d0ae`;
- `ml_runtime` —
  `sha256:2fa36fe8f01798dd139dc70d60356b03e3e4453bca4ad2d39f0f69a4b0758531`;
- OpenBao —
  `sha256:610395fc927391e2cfa4e082ba9cb520a8359b2c14591a9ff63378bf0c52225b`.

Профили `base`, `trading` и `ml`, actual application entrypoints,
job-runtime OCI lifecycle, persistence, restart, safe-degraded exchange/RL
состояния, non-root/resource boundaries и cleanup прошли. Production state и
provider writes не использовались. Эта повторная проверка заменяет только
digest-bound runtime evidence; исходные решения и контрактные классификации
этапа `17` не меняются.

Финальная повторная проверка выполнена после обнаруженных этапом `23`
greenfield-дефектов bootstrap и server-side Web BFF routing, синхронизации
checksum `0012_identity_local_auth_v1.sql` и повторного выпуска runtime/ML
образов. Поэтому это доказательство заменяет предыдущие runtime/ML digest и
подтверждает уже исправленный release-кандидат.
