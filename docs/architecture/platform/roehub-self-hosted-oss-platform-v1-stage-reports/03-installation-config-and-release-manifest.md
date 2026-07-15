# Этап 03 — конфигурация установки и манифест выпуска

## Результат

- Дата: `2026-07-13`.
- Этап: `03`.
- Режим: `goal_driven`.
- Статус: `accepted`.
- Граница доказательства: `N/A`; локальная schema/generation/runtime boundary,
  без запуска полной платформы и без импорта текущей рабочей конфигурации.
- `real-boundary evidence`: `passed`; настоящий Docker Engine разобрал все
  профили, после чего три одноразовых контейнера прочитали generated config и
  завершились с `config-consumer-ok`.
- Следующий разрешённый этап: `04`.

Этап возобновлён с начала после явного разрешения пользователя установить
системные компоненты. `roehub.yaml` теперь является единственным редактируемым
пользователем входом новой установки. Старые `.env`, Compose-файлы и текущие
значения среды не преобразуются и не становятся совместимым контрактом.

## Установленная контейнерная среда

Через Homebrew установлены и реально запущены:

| Компонент | Версия / конфигурация |
|---|---|
| `colima` | `0.10.3`, `vz`, `virtiofs`, 8 CPU, 12 GiB RAM, 60 GiB disk |
| Docker client | `29.6.1` |
| Docker Engine | `29.5.2` |
| Docker Compose | `5.3.1` |
| Docker Buildx | `0.35.0` |
| Lima | `2.1.4` |

Homebrew при первоначальной установке автоматически удалил ранее
установленный `glib` как неиспользуемую зависимость. Чтобы не оставлять это
постороннее удаление, `glib` восстановлен до доступной версии `2.88.2`; вместе
с ним Homebrew установил `json-c 0.19` и обновил необходимые зависимости
`libunistring 1.4.2` и `gettext 1.0`.

Контекст `colima` активен. Проверки версий:

```text
docker client|server: 29.6.1|29.5.2
docker compose: 5.3.1
docker buildx: 0.35.0
```

Buildx подтвердил платформы закреплённого образа потребителя:
`linux/amd64` и `linux/arm64/v8`. Текущий ARM64 digest также подтверждён
локальным pull и `RepoDigests`.

## Реализованный контракт

### Единственный вход установки

- `configs/installation/roehub.yaml` использует
  `io.roehub.installation/v1alpha1`.
- JSON Schema запрещает неизвестные поля и поддерживает только архитектуры
  `linux/amd64` и `linux/arm64`.
- Вход включает домен, порты, каталоги, профили, встроенные или внешние
  хранилища, ограничения ресурсов, TLS, proxy, явно включаемую проверку
  обновлений, OIDC, OpenBao, уведомления и безопасный режим торговли.
- Профиль `base` содержит notifications/Telegram capability и локальное
  хранилище артефактов по умолчанию.
- `mainnet`, raw secret-shaped keys/values, опасные Compose overrides,
  неподдерживаемые архитектуры и повторяющиеся порты отклоняются fail closed.
- Секреты допускаются только как `openbao://...` ссылки под настроенным
  `secret_root`; effective view скрывает даже эти ссылки.

### Манифест выпуска

`tools/release/release-metadata.json` сохраняет
`io.roehub.release/v1alpha1` и дополнен:

- обязательными `supported_architectures`;
- `images.config_consumer` с digest-pinned ссылкой;
- platform matrix для `linux/amd64` и `linux/arm64`.

`oss_metadata.py` проверяет digest, запрет `latest`, полноту платформ и включает
образ выпуска в SPDX/notices. Лицензии пакетов и слоёв Alpine остаются
условным обязательством будущих этапов комплекта выпуска; это записано как
`LicenseRef-Alpine-Base-Image`, а не выдано за завершённый layer audit.

### Детерминированные выходы

Для каждого профиля `base`, `trading`, `ml` генератор создаёт:

- `compose.yaml`;
- `service-config.json`;
- `oidc.json`;
- `openbao.json`;
- `prometheus.yml`;
- `effective-config.redacted.json`;
- `generation-manifest.json` с SHA-256 входов и выходов.

Compose-потребитель использует digest-pinned образ, `network_mode: none`,
read-only root filesystem, `cap_drop: ALL`, `no-new-privileges`, непривилегированного
пользователя и только read-only mounts. Генератор проверяет, что переданные
байты `roehub.yaml` и release manifest совпадают с разобранными объектами, чтобы
hash lineage нельзя было подменить.

## Реестр скрытых runtime inputs

`runtime_input_inventory.py` статически анализирует Git-visible Python,
Dockerfile, shell и YAML в `apps/`, `src/`, `infra/`. Значения переменных среды
не читаются и не записываются.

| Категория | Количество |
|---|---:|
| `installation_generated_runtime` | 48 |
| `openbao_secret_reference` | 17 |
| `product_config_postgresql` | 16 |
| `explicit_legacy_runtime_handoff` | 35 |
| Всего env keys | 116 |
| Явные файловые config inputs | 20 |

Команда `--check` завершится ошибкой при появлении нового или исчезновении
старого входа. Реестр является migration/handoff inventory, но не вторым
пользовательским контрактом.

## Реальная граница Docker

Выполнено:

```bash
uv run python tools/release/verify_installation_runtime.py
```

Скрипт:

1. проверил Docker client/server и Compose;
2. дважды сгенерировал все три профиля;
3. побайтово сравнил полный набор выходов;
4. выполнил `docker compose config --quiet` для `base`, `trading`, `ml`;
5. проверил отсутствие `latest` и `mainnet` в нормализованном Compose;
6. запустил отдельный config-consumer для каждого профиля;
7. получил `config-consumer-ok` во всех трёх случаях.

Итог:

```text
installation runtime verification passed: docker=29.6.1|29.5.2, compose=5.3.1, profiles=base,trading,ml
```

Первая попытка из macOS `/tmp` доказательно провалилась: Colima не передаёт
этот путь в VM, поэтому bind sources становились пустыми каталогами. Вторая
попытка из `~/.cache` прошла для `base` и `trading`, но `ml` выявил лимит 4 CPU.
После увеличения Colima до 8 CPU и 12 GiB вся матрица прошла. Воспроизводимый
проверочный скрипт сразу использует Docker-visible `~/.cache`.

## Проверки качества

| Проверка | Результат |
|---|---|
| JSON Schema Draft 2020-12 для installation/release schemas | `passed` |
| Schema/property/golden/CLI/inventory/OSS unit tests | `21 passed` |
| Golden SHA-256 для 7 выходов × 3 профиля | `passed` |
| Двойная генерация и `--check` | `passed` |
| `ruff` по затронутым Python-файлам | `passed` |
| `pyright` по затронутым Python-файлам | `0 errors` |
| `oss_metadata.py --check` | `passed`, 3 artifacts |
| `runtime_input_inventory.py --check` | `passed`, 116 keys |
| Official SPDX 2.3 JSON Schema | `valid` |
| `docker compose config` + config-consumer, 3 профиля | `passed` |
| Project map `--check` | `passed`, 5 artifacts |
| `git diff --check` | `passed` |

## Контрактное влияние

| Поверхность | Старый контракт | Новый контракт | Потребители / доказательство | Классификация | Миграция и откат |
|---|---|---|---|---|---|
| Public API | не менялся | не менялся | API-код не затронут | `none` | нет |
| Ports | не менялись | не менялись | platform config module не меняет domain ports | `none` | нет |
| DTO | не менялись | не менялись | DTO-код не затронут | `none` | нет |
| Persisted schema | не менялась | не менялась | migrations/DB не запускались | `none` | нет |
| Installation config | `.env`/ручной Compose и разрозненные YAML текущей системы | единственный `roehub.yaml` с версионированной схемой | schema, generator, negative tests | `breaking-change` | Только greenfield; converter, dual-read и aliases не добавляются. Откат удаляет новые локальные config artifacts. |
| Release manifest | базовые release/license/SBOM metadata | additive images/platform fields | `oss_metadata`, release schema, generator | `compatible-change` | Старые readers игнорируют новые optional fields; новый generator требует текущий manifest. |
| Request/cache/persistence identity | не менялась | не менялась | hash lineage относится только к generated files | `none` | нет |
| Service-call auth/timeout/retry/error | не менялись | только generated OIDC/OpenBao inputs, без runtime calls | контейнер читает файлы без сети | `none` | реализация вызовов остаётся будущим stages |
| External side effects | отсутствовали | отсутствуют | consumer использует `network_mode: none` | `none` | нет |
| Logs/metrics/audit/redaction | ручные runtime inputs | generated Prometheus input и redacted effective view | golden/runtime proof | `compatible-change` | additive; raw refs не попадают в effective view |
| Alerts/runbooks | не менялись | не менялись | Stage `02` artifacts сохранены | `none` | нет |
| Browser defaults | не менялись | не менялись | Web не затронут | `none` | нет |

## Файловый манифест и полномочия

Созданы:

- `schemas/config/{roehub.schema.json,release-manifest.schema.json}`;
- `configs/installation/{roehub.yaml,runtime-input-inventory.json}`;
- `src/trading/platform/config/installation.py`;
- `tools/release/{generate_installation_config.py,runtime_input_inventory.py,verify_installation_runtime.py}`;
- `tests/golden/installation/profile-output-sha256.json`;
- `tests/unit/platform/config/test_installation.py`;
- `tests/unit/tools/{test_generate_installation_config.py,test_runtime_input_inventory.py}`;
- этот отчёт.

Изменены:

- `src/trading/platform/config/__init__.py`;
- `tools/release/{README.md,oss_policy.json,oss_metadata.py,preliminary-sbom.spdx.json,THIRD_PARTY_NOTICES.md,release-metadata.json}`;
- `tests/unit/tools/test_oss_metadata.py`;
- stage ledger и generated docs index.

Не менялись:

- текущие `configs/{dev,test,prod}` и `infra/docker/*`;
- production DB, secrets, users, artifacts и runtime configuration;
- product configuration в PostgreSQL;
- foreign изменения `.codex/PLANS.md`, supersession docs и project-map.

Staging, commit, push, release, deploy и production mutation не выполнялись.

## Передача этапу 04

Этап `04` получает:

- проверенный installation schema и три профиля;
- embedded/external store selection без raw credentials;
- OpenBao references и output hash lineage;
- реальный Docker Engine/Compose boundary;
- явный запрет импорта текущей базы;
- список legacy runtime inputs только как migration inventory.

Этап `04` обязан создавать чистые PostgreSQL/ClickHouse stores и доказывать
идемпотентный bootstrap, повторный запуск и восстановление после прерывания.
