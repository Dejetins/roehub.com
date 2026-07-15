---
validation_depth: runtime
tests_only_acceptance: false
real_boundary_status: blocked
real_boundary_evidence:
  - docs/architecture/platform/roehub-self-hosted-oss-platform-v1-stage-reports/evidence/22-signed-offline-release-bundle-runtime-proof.json
  - docs/architecture/platform/roehub-self-hosted-oss-platform-v1-stage-reports/evidence/24-macos-docker-desktop-greenfield-lifecycle-proof.json
  - docs/architecture/platform/roehub-self-hosted-oss-platform-v1-stage-reports/evidence/24-platform-matrix-readiness-proof.json
---

# Этап 24 — платформенная матрица кандидата релиза

## Статус

- Этап: `24`.
- Состояние: `blocked`.
- Режим: `goal_driven`.
- Локальная часть на Docker Desktop: `passed`.
- Итог этапа: `Block`: обязательный нативный исполнитель `linux/amd64`
  отсутствует, поэтому полную межплатформенную матрицу закрыть нельзя.
- Следующий разрешённый этап: `none`.
- Этап `25`: не запускался и требует отдельного явного разрешения даже после
  принятия Stage `24`.
- Граница доказательств: `N/A` — одноразовые локальные контейнеры и сгенерированные
  fixture-данные на MacBook Pro M3 Pro. Production/current state, персональные
  данные, реальные provider credentials, заявки, staging, публикация и deploy
  исключены.

## Проверенный кандидат

Stage `22` полностью воспроизведён после удаления прежнего локального кэша.
Сохранённый подписанный кандидат:

`/Users/daniildegtyarev/.cache/roehub/stage22-offline-release/candidates/roehub-0.1.0`.

- версия: `0.1.0`;
- manifest SHA-256:
  `569f24ee203bc7e27a7cdef6fa4cdc0fa58cbb072fb840d86687b4005c968add`;
- tree SHA-256:
  `a9ca3b3cbefc7ea6329b97f60e39f9d186256186569a7cf02f6dd4b24c4d3e0d`;
- размер: `5196194970` байт;
- подпись и защита от изменения проверены;
- wheel, OpenBao, runtime, ML runtime и полный автономный комплект побайтово
  воспроизводимы;
- `250` исходных `NOASSERTION` классифицированы, `unresolved_count=0`;
- автономная базовая установка подняла `15` сервисов, workload egress — `0`.

Новые обязательные image digest:

- OpenBao:
  `sha256:8492e2c1a523aac5da44e41c86e84eac992479fb7c4a79c2e1a07b8b24bcec4a`;
- runtime:
  `sha256:35ad7fdd77e6aa6cbbcf5fc20b29184277b4400d0e8c97c24f46876a061b352f`;
- ML runtime:
  `sha256:32ea812a7450a6949a14d333ced2bf420858144ed935887b5a159301b1e43ad8`.

## Локальная матрица Docker Desktop

Проверенная среда:

- MacBook Pro `Mac15,6`, Apple M3 Pro, `11` CPU cores, `18 GB` RAM;
- macOS `15.7.4`, host architecture `arm64`;
- Docker Desktop `4.82.0`, context `desktop-linux`;
- Docker client/server `29.6.1`, API `1.55`;
- Docker server: `linux/arm64`;
- Colima отсутствует.

Полный последовательный greenfield lifecycle завершился за `471.74 s`:

- подписанный автономный комплект установлен в пустое состояние;
- исходная, восстановленная и повторная установки подняли по `24` постоянных
  контейнера профиля `trading`;
- PostgreSQL identity state, ClickHouse, Redis, Artifact Store и OpenBao
  согласованы после восстановления;
- первоначальная настройка и повторная настройка создали ожидаемую структуру
  installation/owner/organization/user/membership;
- браузерная консоль: `0` ошибок;
- межорганизационный административный запрос: `404`;
- вход по passkey после восстановления: `passed`;
- вложенный Stage `21` backup/restore/update/rollback: `passed` с первой попытки;
- отмена и возобновление backup/restore, отказ до commit, повтор обновления,
  rollback и irreversible-migration guard доказаны;
- после выполнения контейнеры, сети, тома и локальные image tags удалены.

Сохранены:

- `evidence/24-macos-docker-desktop-greenfield-lifecycle-proof.json`;
- `evidence/24-macos-docker-desktop-admin.png`, PNG `1280 x 720`, SHA-256
  `04bb8ff4efee0c20da5eb9356231c7f4000bd9addf7e19ae63a6a7036f5f7dc8`.

## Исправления проверяющего контура

Первый полный проход выявил, что вложенная проверка Stage `21` использовала
исходные registry-имена образов вместо локальных идентификаторов, созданных
автономным установщиком. При `--pull never` `secret-init` получил
`No such image` для `alpine:3.22@sha256:...`.

Исправление передаёт подписанный `compose.trading.offline.yaml` в storage и
monitoring Compose-команды Stage `21`. Для отдельного OpenBao proof из него
создаётся минимальный одно-сервисный override с локальным image ID и
`pull_policy: never`. Сначала отдельно доказаны storage/monitoring и полный
OpenBao backup/restore, затем исходный lifecycle повторён целиком.

Продуктовые API, DTO, ports, persistence, identity и runtime semantics не
изменились. Изменение внутреннего proof API обратно совместимо; container/hash
identity Stage `22` остаётся `breaking-change` до первого опубликованного
релиза и уже отражена в согласованном наборе digest.

## Интерактивный локальный запуск

По явному запросу владельца тот же подписанный кандидат повторно импортирован
в Docker Desktop и профиль `trading` оставлен работающим для ручного просмотра:

- подпись и `263` файла повторно проверены внешним trusted public key;
- импортированы `12` уникальных `linux/arm64` image без обращения к registry;
- `24` постоянных сервиса работают, `secret-init` и `storage-migrations`
  штатно завершились;
- Web/API/storage/OpenBao/monitoring/trading health checks прошли;
- `http://127.0.0.1:8080/` отвечает `200`, форма первоначального владельца
  открыта на русском языке, browser console errors — `0`;
- одноразовый bootstrap ticket сохранён только локально с mode `0600`, его
  значение не выводилось в журналы, доказательства или чат.

Первый запрос к `127.0.0.1:8080` выявил отдельный Docker Desktop ingress defect:
исходный Compose содержит host binding, но web подключён только к сети
`internal: true` без gateway. Docker сохранил `HostConfig.PortBindings`, однако
не активировал `NetworkSettings.Ports`. После диагностического подключения
только web-контейнера к локальной сети `bridge` тот же binding начал отвечать
`200`. Это допустимый временный обход для локального просмотра, но не
acceptance fix: после пересоздания контейнера он может потребоваться снова, а
канонический ingress/network contract должен быть исправлен и повторно доказан
новым подписанным кандидатом.

## Память, диск и очистка

- тяжёлые операции выполнялись с `GOMAXPROCS=1`, `GOMEMLIMIT=768MiB`,
  `SYFT_PARALLELISM=1` и `COMPOSE_PARALLEL_LIMIT=1`;
- Buildx builders останавливались после сборки и были удалены до lifecycle;
- наблюдавшийся системный резерв памяти не опускался ниже `35%`;
- после lifecycle cleanup было свободно `52%` системной памяти и Docker
  содержал `0` images, containers, volumes и build cache;
- во время разрешённого интерактивного запуска `24` постоянных сервиса
  потребляют около `2788.6 MiB`, свободно около `45%` системной памяти;
- после импорта образов и запуска на диске свободно около `162 GiB`;
- retained candidate и необходимые Stage `22` OCI/source caches занимают около
  `12 GiB`, импортированные Docker images — около `9.784 GB`; они сохранены для
  текущего просмотра и нативной матрицы `linux/amd64`.

## Блокирующее условие

Prompt Stage `24` требует нативное исполнение на `linux/amd64`, `linux/arm64`
и Docker Desktop macOS. Текущий M3 Pro предоставляет нативный `linux/arm64`.
Rosetta/QEMU могут использоваться только для диагностики и не являются
принимаемым доказательством `linux/amd64`.

Настроенного непроизводственного Docker context или SSH executor с нативной
архитектурой `linux/amd64` нет. Поэтому Stage `24` нельзя принять, а Stage `25`
запускать нельзя.

Дополнительный blocker кандидата: штатная публикация web-порта на Docker
Desktop не работает при единственной сети `internal: true`; текущий ручной
`bridge` workaround не является доказательством принимаемого ingress-контракта.

Для возобновления требуется отдельный нативный Linux-хост на Intel/AMD с Docker
Engine и Docker Compose v2, доступный через заранее настроенный Docker context
или SSH alias. Секреты и ключи в чат передавать не нужно. На нём необходимо
повторить проверку того же manifest/tree candidate без эмуляции, затем закрыть
оставшиеся поверхности Stage `24`: сопоставимые backtest/ML/runtime benchmarks,
итоговую сверку всех компонентов project map и сервисов release manifest, а
также отдельную responsive/accessibility браузерную матрицу.

## Проверка перед остановкой

- Режим проверки артефакта: холодная самостоятельная перепроверка; независимый
  субагент запрещён текущим режимом выполнения.
- Браузерная готовность локального объёма: `ready_with_caveats` — реальный
  Chromium, авторизация, API-изоляция, консоль и восстановленный passkey
  проверены; отдельные mobile/responsive/accessibility проверки не входят в
  Stage `24` lifecycle evidence.
- Текущий стоп-сигнал: отсутствующая нативная платформа `linux/amd64`; после её
  появления остаются перечисленные выше проверки полной матрицы Stage `24` и
  выпуск нового кандидата с доказанной штатной публикацией web-порта.
- Production/current state не читался и не менялся. Commit, push, публикация,
  staging и deploy не выполнялись.
