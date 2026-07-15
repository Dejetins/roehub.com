---
validation_depth: runtime
tests_only_acceptance: false
real_boundary_status: passed
real_boundary_evidence:
  - docs/architecture/platform/roehub-self-hosted-oss-platform-v1-stage-reports/evidence/22-signed-offline-release-bundle-runtime-proof.json
---

# Этап 22 — подписанный автономный комплект релиза

## Статус

- Этап: `22`.
- Состояние: `accepted`.
- Режим: `goal_driven`.
- Глубина принятия: `runtime`; тесты не использовались вместо реальной
  автономной установки.
- Граница доказательств: `N/A` — локальные OCI-архивы, сохранённый кандидат и
  одноразовый Docker. Текущая рабочая среда, production-данные, реальные
  учётные данные, provider effects, staging, commit, push, публикация и deploy
  исключены.
- Следующий разрешённый этап: `23`.

## Снятие лицензионной блокировки OpenBao

Владелец разрешил вариант `1`: собственную закреплённую сборку OpenBao `2.5.4`
с сохранением JWT QR-функции. Нелицензированный
`github.com/yeqown/reedsolomon@v1.0.0` удалён из module graph и ELF; QR-код
строится через MIT-лицензированный
`github.com/skip2/go-qrcode@v0.0.0-20200617195104-da1b6568686e`.

Итоговый multi-architecture image:

- reference:
  `ghcr.io/dejetins/roehub-openbao:2.5.4-roehub-licensed-qr.1@sha256:8492e2c1a523aac5da44e41c86e84eac992479fb7c4a79c2e1a07b8b24bcec4a`;
- полный соответствующий исходный архив:
  `openbao-v2.5.4-roehub-licensed-qr.1.tar.gz`;
- SHA-256 исходного архива:
  `e1cc071b4666312de84e4bdf32e7e25be04f95738a7dbc5adff4c357c3a24f07`;
- исходный архив включает MPL-2.0, MIT, Roehub notice и точный patch;
- две сборки OCI-архива побайтово идентичны;
- `go test ./builtin/credential/jwt -run '^TestPrintQR$'` и повторные реальные
  проверки этапов `08` и `17` прошли.

## Строгий аудит `NOASSERTION`

Необработанные SPDX не переписываются. Отдельный fail-closed overlay
`io.roehub.runtime-license-audit/v1alpha1` связывает каждое разрешение с точным
образом, платформой, компонентом, исходным URL, путём и SHA-256 доказательства.

Проверены четыре документа:

| Образ | Платформа | Исходные `NOASSERTION` |
|---|---|---:|
| `runtime` | `linux/amd64` | 59 |
| `runtime` | `linux/arm64` | 65 |
| `ml_runtime` | `linux/amd64` | 60 |
| `ml_runtime` | `linux/arm64` | 66 |

Всего классифицировано `250`, неразрешённых записей — `0`:

- `112` — `embedded-non-component`: тестовые данные JupyterLab и Windows-only
  DLL/EXE, которые не являются исполняемыми Linux-компонентами образа;
- `4` — `first-party-image`;
- `102` — `policy-license-file`: точные Python/Debian license и copyright
  artifacts проверены в OCI-слоях;
- `32` — `scanner-concluded`: Syft нашёл проверяемую лицензию, но оставил
  `licenseDeclared=NOASSERTION`.

Для `runtime` доказаны `27/28`, а для `ml_runtime` — `28/29` уникальных
license artifacts на `linux/amd64`/`linux/arm64`. Любая запись, которая не
совпадает с точной политикой или фактическим файлом слоя, останавливает сборку.

## Подписанный автономный кандидат

После явной очистки локального диска прежний retained candidate и OCI-кэши
были удалены. По новому разрешению владельца Stage `22` полностью повторён на
Docker Desktop `linux/arm64`: OpenBao, runtime и ML runtime собраны дважды,
каждая пара OCI-архивов побайтово совпала, а новый согласованный набор digest
до неопубликованного релиза `0.1.0` атомарно внесён в release/runtime metadata.
Полный license/source/signature/no-egress proof повторён с нуля.

Во время первой репетиции этапа `23` реальная PostgreSQL-граница обнаружила,
что bootstrap challenge ссылался на ещё не созданного owner. Исправление
сохранило HTTP/DTO-контракт: будущий owner UUID теперь хранится в контексте
одноразового challenge, а nullable `user_id` получает FK только после создания
пользователя. Checksum миграции `0012_identity_local_auth_v1.sql` синхронизирован
с `migrations/postgres/manifest.json`. После этого runtime/ML образы, профили,
Stage `17` и весь Stage `22` были пересобраны и повторены с нуля.

Следующая репетиция обнаружила второй greenfield-дефект: server-side auth gate
Web обращался к `/api/auth/current-user` через прямой адрес API, где маршрут
имеет вид `/auth/current-user`. `WEB_API_BASE_URL` направлен на Web BFF
`http://web:8010`, а отдельный `WEB_API_UPSTREAM_URL` по-прежнему указывает на
`http://api:8000`. После исправления runtime/ML образы, Stage `17`, лицензионный
аудит и весь автономный proof Stage `22` повторены ещё раз.

Сохранён устанавливаемый кандидат:
`/Users/daniildegtyarev/.cache/roehub/stage22-offline-release/candidates/roehub-0.1.0`.

- manifest SHA-256:
  `569f24ee203bc7e27a7cdef6fa4cdc0fa58cbb072fb840d86687b4005c968add`;
- tree SHA-256:
  `a9ca3b3cbefc7ea6329b97f60e39f9d186256186569a7cf02f6dd4b24c4d3e0d`;
- размер: `5196194970` байт;
- `263` подписанных assets, `13` image records и соответствующие исходники
  `grafana`, `loki`, `openbao`;
- подпись проверена внешним trusted public key; изменение `NOTICE` отвергнуто;
- wheel, runtime, ML runtime, OpenBao и весь offline bundle побайтово
  воспроизводимы;
- wheel SHA-256 обеих сборок:
  `36d262f8c35913263bbcd7b2afa52365b7afce10815da8c9f4b744ecd7fb9ffd`;
- runtime index digest:
  `sha256:35ad7fdd77e6aa6cbbcf5fc20b29184277b4400d0e8c97c24f46876a061b352f`;
- ML runtime index digest:
  `sha256:32ea812a7450a6949a14d333ced2bf420858144ed935887b5a159301b1e43ad8`.

Во время реального запуска обнаружены и исправлены две ошибки fail-closed
проверки комплекта:

1. безопасные внутренние symlink Grafana с `..` ошибочно отвергались без
   нормализации итогового пути; выход ссылки за корень по-прежнему запрещён и
   покрыт регрессионным тестом;
2. `infra/openbao/config/openbao.hcl` отсутствовал в подписанном payload, из-за
   чего Docker создавал каталог вместо bind-source и OpenBao завершался;
   runtime-конфигурация теперь является обязательным подписанным asset.

## Реальная автономная граница

`tools/release/install-offline.sh` установил кандидат в пустое состояние без
registry access, импортировал `12` уникальных локальных образов и выполнил
runtime smoke. Затем `base` Compose был запущен в сети `internal: true` с
packet capture на Docker bridge.

- `15` постоянных сервисов стали здоровыми, включая OpenBao, API, Web,
  PostgreSQL, Redis, Grafana, Loki и workers;
- внешний workload-трафик: `0` пакетов;
- Grafana reporting/update/plugin checks и Loki analytics выключены;
- bridge gateway `172.18.0.1` исключён только как источник служебного IGMP
  Docker; адреса workload-контейнеров остаются под полной проверкой;
- cold verification сохранённого кандидата совпала с исходной;
- временные контейнеры, сети, volumes и `12` owned image tags удалены.

## Контракты и совместимость

| Поверхность | Классификация | Обоснование |
|---|---|---|
| API и DTO | `none` | Прикладные HTTP/DTO контракты не менялись. |
| OpenBao JWT QR | `compatible-change` | Функция и CLI-контракт сохранены; заменена только внутренняя библиотека рендеринга. |
| Container identity | `breaking-change` | OpenBao, runtime и ML runtime получили новые обязательные digest до первого опубликованного релиза. |
| Формат автономного комплекта | `breaking-change` | Добавлены обязательный signed license audit, corresponding source и полный asset inventory. |
| Runtime/config | `breaking-change` | Установка требует internal network, локальные digest-pinned images и подписанный `openbao.hcl`. |
| Web BFF routing | `compatible-change` | Server-side Web использует собственный `/api/*` proxy, а upstream identity API остаётся отдельным внутренним адресом; публичные URL и DTO не меняются. |
| Хранение | `compatible-change` | Greenfield-миграция разрешает bootstrap challenge без преждевременного FK и связывает будущий owner UUID через проверенный контекст; опубликованных данных и миграции существующей установки нет. |
| Внешние эффекты | `compatible-change` | Registry/provider writes отсутствуют; phone-home выключен и проверен. |
| Инструкции и аудит | `compatible-change` | Добавлены точные source/license/provenance записи и автономный installer proof. |

Основная классификация этапа — `breaking-change`, ожидаемая для ещё не
опубликованного greenfield release contract.

## Проверки и ресурсы

- Полный `io.roehub.stage22-runtime-proof/v1alpha1`: `status=passed`.
- Подпись, tamper rejection, exact inventory и cold verification: `passed`.
- OCI config/layer integrity и оба release platform: `passed`.
- Runtime license audit: `250` классифицировано, `0` unresolved.
- OpenBao/Runtime/ML OCI и wheel reproducibility: `passed`.
- Автономная установка, Compose health и workload no-egress: `passed`.
- Сфокусированные регрессионные тесты после последних исправлений:
  `25 passed`; Ruff: `passed`; Pyright: `0 errors`.

Тяжёлые операции выполнялись последовательно с `GOMAXPROCS=1`,
`GOMEMLIMIT=768MiB` и `SYFT_PARALLELISM=1`. Во время финальной активации было
свободно не менее `35%` системной памяти; OpenBao использовал `26.82 MiB`, API —
`299 MiB`. После завершения runtime-ресурсы очищены.

## Независимая проверка и остаточные границы

- Режим: ранее выполненный `independent subagent`, затем холодная локальная
  перепроверка всех исправлений и полный runtime proof.
- Первоначальный вердикт: `Block` по no-egress, corresponding source,
  artifact-level licenses и отсутствующему retained candidate.
- Все исходные blocker закрыты; локальный итог: `Release after fixes`.
- Этап `23` повторно установил сохранённый кандидат, доказал полный greenfield
  lifecycle `install/bootstrap/backup/restore/rollback` и не нашёл дрейфа
  подписанного комплекта. Оставшаяся граница — платформенная матрица этапа
  `24`.
- Production/current state, реальные учётные данные и персональные данные не
  читались. Commit, push, публикация, deploy и staging не выполнялись.
