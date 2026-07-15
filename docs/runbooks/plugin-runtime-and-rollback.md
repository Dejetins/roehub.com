# Plugin runtime: диагностика, изоляция и rollback

## Назначение

Инструкция относится к `Plugin API v1alpha1`. Она помогает проверить
подписанный bundle, состояние async operation и изолированного контейнера,
после чего безопасно выполнить rollback. Marketplace и production install пока
не входят в поддерживаемую границу.

## Правила безопасности

- Не копировать в чат, ticket, логи или evidence session cookie, signing key,
  OpenBao value, provider payload и raw plugin response.
- Не добавлять контейнеру Docker socket, host path, platform database network
  или произвольные environment variables.
- Не включать unsigned mode в `mainnet`. Не обходить signature/publisher
  verification даже для локального rollback.
- Не менять operation/event rows вручную. `unknown` сначала сверяется, а не
  повторяется вслепую.

## Предварительная проверка bundle

```bash
roehubctl plugins validate ./plugin-bundle \
  --publisher-keys /etc/roehub/plugin-publisher-keys.json \
  --trading-mode testnet
```

Ожидаются `status=passed`, package/image digest, version и ограниченный список
permissions. Команда не запускает контейнер и не печатает значения секретов.

Для локальной разработки unsigned bundle допускается только явной командой:

```bash
roehubctl plugins validate ./plugin-bundle \
  --allow-unsigned-development \
  --trading-mode paper
```

Если `trading-mode=mainnet`, это должно завершиться отказом до запуска.

## Диагностика операции

1. Найти внутренние `organization_id`, `operation_id`, `plugin_id` и
   `instance_id`; не использовать клиентский organization override.
2. Получить `PluginOperation/v1alpha1` через organization-scoped API.
3. Проверить `status`, request hash, package digest и audit events.
   Executor не должен принимать отдельный bundle/config: проверяется только
   сохранённый request snapshot, а claim `pending → running` является CAS.
4. Для `running` проверить gateway/container health и срок lease worker.
5. Для `unknown` определить, мог ли container принять mutating request.
   Повтор разрешён только с предметной idempotency identity и после сверки.

Повтор install/update с тем же `Idempotency-Key` и тем же payload обязан вернуть
исходный `operation_id`. Другой payload с тем же ключом — конфликт, а не новая
операция.

## Проверка изоляции контейнера

Проверить runtime policy по container inspection:

- user — non-root uid из manifest;
- root filesystem — read-only;
- capabilities — `ALL` dropped;
- `no-new-privileges` включён;
- memory/CPU/PID соответствуют manifest;
- mounts пусты, Docker socket отсутствует;
- подключена только выделенная internal plugin network;
- platform database hostname и запрещённый внешний адрес недоступны.
- container `Image` точно равен подписанному `image.digest`; tag/reference не
  используется как execution identity.

Health и metrics должны отвечать с заголовком
`X-Roehub-Plugin-Protocol: roehub.plugin.rpc/v1alpha1`. Неверная версия должна
получить отказ. Проверка другой capability должна получить `403`, если право не
выдано установке.

## Обновление и rollback

До update:

1. Проверить подпись нового immutable package и совместимость Roehub/API/arch.
2. Сравнить permissions с текущими. Любое расширение требует recent-auth и
   отдельного audit event.
3. Сохранить текущий package как `previous_package_id`.
4. Не удалять старый image/package до проверки health, metrics и одного
   безопасного capability request.

Rollback запускается versioned API/CLI operation:

```bash
roehubctl plugins rollback <plugin-id> \
  --api-url https://roehub.example \
  --organization-id <organization-id> \
  --session-file /run/user/$(id -u)/roehub-session \
  --idempotency-key <new-operator-key>
```

После `succeeded` подтвердить, что active package digest равен предыдущему,
instance identity не сменилась, config revision не откатилась неявно, health и
metrics готовы, а `plugin.rollback.completed` присутствует в неизменяемом
аудите.

До claim rollback обязан повторно проверить, что target package всё ещё
разрешён текущим operator trust file, его mirrored key status не `revoked`, а
unsigned development package не возвращается в `mainnet`. Любое расхождение
фиксируется отказом в аудите.

## Когда остановиться

Остановить автоматическое продолжение и пометить этап/операцию `blocked`, если:

- подпись, publisher key, license, SBOM, image digest или compatibility не
  доказаны;
- контейнерная граница недоступна либо policy inspection не подтверждает
  ограничения;
- есть Docker socket, host mount, database connectivity или незаявленный
  egress;
- permission expansion прошёл без recent-auth/audit;
- mutating RPC завершился timeout и результат нельзя сверить;
- rollback target не является ранее проверенным подписанным пакетом.
