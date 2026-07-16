# Этап 05 — локальная пересборка и передача на повторную сертификацию

## Результат

Этап принят на границе `local_macos_docker_desktop_arm64`. Локальный образ
`roehub-dev/runtime:market-data-stage05` пересобран для `linux/arm64` и все
сервисы профиля `trading` последовательно пересозданы без удаления volume.
Стандартный ingress работает через объявленный binding `127.0.0.1:8080:8010`;
ручное подключение `web` к `bridge` не использовалось.

В ходе свежего прогона выявлен и исправлен runtime-дефект ручной публикации:
`ClickHouseSettingsLoader` не читал `ROEHUB_CLICKHOUSE_PASSWORD_FILE`.
Загрузчик теперь использует файл учётных данных, не копируя значение в
окружение. После пересборки ручной bounded publish
`binance:futures:BTCUSDT --max-source-bars 10080` успешно переключил
`current.yaml`.

Локальный контур оставлен работающим для просмотра на
`http://127.0.0.1:8080/`. Контрольная выборка `docker stats` показывает около
`3.1 GiB` памяти; ресурсы Stage `05` не очищены намеренно. Удаление
контейнеров и volume выполняется только по отдельному запросу владельца.

## Доказательство на реальной границе (`runtime smoke`)

В Docker Desktop подтверждены:

- все `24` постоянных сервиса профиля `trading` запущены, а их healthchecks
  успешны; оба одноразовых service (`secret-init`, `storage-migrations`)
  штатно завершились;
- `market-data-scheduler` и `market-data-ws` остаются единственными
  workload-сервисами с `market-data-egress`, а внутренняя сеть `roehub`
  сохранена;
- readiness после пересоздания: одно WebSocket-подключение, `390` сообщений,
  `2` вставленные строки, ошибки WebSocket/insert/REST/scheduler — `0`, свеча
  не старше `55.48 s`;
- `10080` минутных свечей использованы для одного bounded artifact publish;
  `current.yaml` существует в единственном экземпляре, его SHA-256
  `d9e83f4bc65c44faa9f5cf6b637f086a8705f958a91fb4630398129ac2d40165`;
- текущий артефакт содержит `1667` файлов и `31367168` bytes; peak cgroup
  publisher — `705806336` bytes при внутреннем budget `805306368` bytes и
  контейнерном лимите `1073741824` bytes;
- реальный браузер отобразил форму первичной настройки на `127.0.0.1:8080`
  без ошибок или предупреждений консоли. Полный browser proof выбора
  инструмента остаётся в Stage `02`; в Stage `05` настройка пользователя
  намеренно не создавалась и не изменялась.

Подробности находятся в
[`05-local-runtime-proof.json`](evidence/05-local-runtime-proof.json) и
[`05-local-runtime-readiness-proof.json`](evidence/05-local-runtime-readiness-proof.json).
Оба артефакта не содержат секретов, ticket, cookie или пользовательских
значений.

## Контрактное влияние

| Поверхность | Класс | Переход и откат |
|---|---|---|
| Public API / browser DTO | `none` | В Stage `05` маршруты, payload и UI выбора не менялись. |
| Port contract | `none` | Изменено только чтение конфигурации CLI-adapter. |
| Persisted schema | `none` | Существующие volume и миграции сохранены. |
| Config schema | `compatible-change` | Поддержан уже объявленный `ROEHUB_CLICKHOUSE_PASSWORD_FILE`; строковые `CH_PASSWORD` и `CLICKHOUSE_PASSWORD` сохраняют прежний приоритет. |
| Request hash / cache / persistence identity | `none` | Артефактный identity и pointer format не менялись. |
| Service-call credentials | `compatible-change` | Publisher впервые корректно использует file-backed credential; при unreadable или empty файле он fail-fast. |
| Browser-visible behavior | `none` | Подтверждён штатный onboarding; полный authenticated settings flow не повторялся без user-owned credentials. |
| Rollout gate | `breaking-change` | Изменённый runtime invalidates исходные подписанные evidence Stage `22`/`23`; требуется новая multi-arch signed сборка и повторная сертификация. |

## Проверки

- `124 passed` focused pytest, включая market data, selection, publisher,
  OpenBao и runtime topology;
- `8 passed` для file-backed ClickHouse CLI adapter;
- Ruff — успешно; Pyright — `0 errors, 0 warnings`; `node --check` — успешно;
- `docker compose ... config --quiet` и полный последовательный
  `up -d --force-recreate --wait` — успешно;
- readiness verifier — успешно после пересборки;
- ручной publish, pointer и cgroup memory evidence — успешно;
- runtime browser onboarding — наблюдался; console errors/warnings отсутствуют.

## Передача в следующую сертификацию

Локальный `linux/arm64` образ не является новым подписанным кандидатом Stage
`22` и не заменяет `linux/amd64` acceptance evidence. Перед продолжением
исходного self-hosted маршрута необходимо:

1. собрать и подписать новый multi-arch Stage `22` candidate с текущими
   исходниками;
2. повторить Stage `23` greenfield lifecycle;
3. повторить Stage `24`, включая native `linux/amd64` executor,
   component/service reconciliation и responsive/accessibility matrix;
4. выполнить durable owner PGP custody для OpenBao отдельным действием
   владельца.

Production, реальные provider credentials, торговые операции, commit/push и
deploy этим runtime proof не затрагивались.

## Холодная самостоятельная проверка

`cold self-review fallback`: повторно проверены Compose graph, normal ingress,
путь file-backed credential, readiness JSON, bounded artifact publish, cgroup
memory, browser onboarding и границы handoff. Вердикт: `Release after fixes`.
Остаточные риски: local `linux/arm64` не доказывает native `linux/amd64`, а
durable OpenBao owner custody и новая signed release certification требуют
следующих отдельных действий.
