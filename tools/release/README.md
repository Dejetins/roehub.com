# Управление выпуском и открытыми лицензиями

## Единый источник версии

Единственный редактируемый номер версии Roehub находится в
`pyproject.toml#project.version`. Значение записывается без префикса `v` и
соответствует SemVer 2.0.0. Git tag и имя выпуска при публикации используют
форму `vX.Y.Z`, но не становятся вторым источником версии.

До `1.0.0` несовместимое изменение публичного API, DTO, схемы хранения,
конфигурации, идентичности или release manifest повышает `MINOR`; совместимое
исправление повышает `PATCH`. После `1.0.0` несовместимое изменение повышает
`MAJOR`. Любое изменение всё равно проходит отдельную контрактную
классификацию и миграционный барьер.

Release manifest имеет схему `io.roehub.release/v1alpha1`. Читатель может
игнорировать неизвестные необязательные поля. Удаление обязательного поля или
изменение его смысла требует новой версии схемы manifest. Версия manifest не
заменяет версию продукта.

## Лицензионный реестр

`oss_policy.json` — проверяемый реестр прямых Python-зависимостей, исходного
JavaScript-прототипа, контейнерных образов, встроенных ресурсов и первых
графических ресурсов. Статусы имеют следующий смысл:

- `compatible`: компонент можно распространять при сохранении обычных notices;
- `conditional`: перед выпуском нужно выполнить записанное обязательство;
- `excluded`: компонент существует в репозитории, но не входит в комплект;
- любое другое значение блокирует проверку.

`AGPL-3.0-only` Grafana не применяется к Roehub: Grafana допускается только как
отдельный неизменённый сервис агрегированного Compose-комплекта с сохранением
лицензии и доступом к соответствующему исходному коду. `LGPL-3.0-only` Psycopg
остаётся отделимой библиотекой; конкретные binary wheels и связанные библиотеки
проверяются для каждого выпуска. Эти условия не являются общим юридическим
заключением и должны быть повторно подтверждены на digest-уровне.

## Воспроизводимые артефакты

Обновление после осознанного изменения политики:

```bash
uv run python tools/release/oss_metadata.py --write
```

Проверка без сетевых вызовов и публикации:

```bash
uv run python tools/release/oss_metadata.py --check
```

Команда проверяет единственный источник версии, хеш официального `LICENSE`,
полноту прямых зависимостей, образы Compose/Dockerfile, встроенные Web-ресурсы,
шрифты, бинарные и графические ресурсы. Затем она сверяет:

- `preliminary-sbom.spdx.json` в формате SPDX 2.3;
- `THIRD_PARTY_NOTICES.md`;
- `release-metadata.json`.

Эти файлы предварительные: транзитивные Python-пакеты, слои образов и
platform-specific wheels требуют artifact-level проверки в последующих этапах.
Образы с `latest` запрещены в выпуске до фиксации digest.

Подписанный offline bundle дополняет предварительный реестр реальными
platform-specific SPDX 2.3 SBOM для `linux/amd64` и `linux/arm64`. Bundle не
ослабляет `oss_policy.json`: он связывает точные OCI archive, notices,
соответствующие исходные архивы Grafana/Loki и provenance одним подписанным
манифестом.

Проверка не публикует пакеты, образы, tags или releases, не меняет внешние
системы и не отправляет телеметрию. Любые будущие проверки обновлений и
телеметрия продукта могут быть только явно включаемыми пользователем.

## Установочная конфигурация

`configs/installation/roehub.yaml` — единственный пользовательский вход для
новой self-hosted установки. Он проверяется схемой
`schemas/config/roehub.schema.json`; значения секретов запрещены, а допустимые
поля `*_ref` содержат только ссылки `openbao://...`. Пользовательские `.env` и
Compose-файлы текущей установки не являются совместимым контрактом и не
преобразуются.

Генерация профилей `base`, `trading` и `ml`:

```bash
uv run python tools/release/generate_installation_config.py \
  --output /path/visible/to/docker \
  --write
uv run python tools/release/generate_installation_config.py \
  --output /path/visible/to/docker \
  --check
```

Для каждого профиля создаются Compose-фрагмент, внутренняя конфигурация
сервисов, входы OIDC/OpenBao/Prometheus, скрытое от секретных ссылок эффективное
представление и манифест хешей. Один и тот же `roehub.yaml` и
`release-metadata.json` дают побайтово одинаковый результат. Compose-фрагмент
использует только закреплённый по digest образ потребителя конфигурации,
`network_mode: none`, файловую систему только для чтения и удалённые Linux
capabilities.

Реестр старых переменных среды и файловых входов не читает их значения:

```bash
uv run python tools/release/runtime_input_inventory.py --check
```

`configs/installation/runtime-input-inventory.json` делает появление новых
скрытых зависимостей проверяемым, но не превращает их в пользовательский
контракт v1. Их владельцы переноса указаны по будущим этапам.

Обязательная реальная проверка требует доступный Docker Engine и выполняет
разбор и одноразовый контейнер для каждого профиля:

```bash
uv run python tools/release/verify_installation_runtime.py
```

Проверка создаёт два временных вывода под `~/.cache`, сравнивает их побайтово,
выполняет `docker compose config` и запускает потребителя с отключённой сетью.
Каталог под домашним путём выбран потому, что Colima по умолчанию не передаёт
macOS-каталог `/tmp` внутрь виртуальной машины Docker.

## Подписанный автономный комплект

`offline_bundle.py` создаёт `io.roehub.offline-release-bundle/v1alpha1` только
из полного набора образов, закреплённых digest. Для каждого уникального OCI
index он сохраняет обе поддерживаемые архитектуры и два SPDX SBOM. Манифест
перечисляет каждый байт комплекта и подписывается `SSHSIG-Ed25519` в namespace
`roehub.offline-release-manifest.v1`.

Проверка до активации:

```bash
python3 tools/release/offline_bundle.py verify \
  --bundle /path/to/unpacked-bundle \
  --trusted-public-key /secure/path/roehub-release-signing-key.pub
```

Подготовка локальных образов и immutable Compose override без registry:

```bash
./tools/release/install-offline.sh \
  --trusted-public-key /secure/path/roehub-release-signing-key.pub \
  --state-directory "$HOME/.local/share/roehub/offline" \
  --profile base \
  --runtime-smoke
```

Публичный ключ внутри комплекта является только идентификатором подписанта и не
задаёт доверие. Путь `--trusted-public-key` должен указывать на ключ, полученный
по независимому доверенному каналу. Installer требует Python 3.9 или новее,
OpenSSH `ssh-keygen`, `skopeo`, Docker Engine и Compose v2; PyPI, Git и registry
во время проверки и импорта не используются.
