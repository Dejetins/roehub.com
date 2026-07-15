# Stage 01 — открытая лицензия и управление выпусками

## Результат

- Дата: `2026-07-13`.
- Stage: `01`.
- Режим: `goal_driven`.
- Статус: `accepted`.
- Граница доказательства: `N/A` — выполнены локальные проверки source/package
  artifacts; package, image, tag и release не публиковались.
- Версия продукта: `0.1.0` из единственного редактируемого источника
  `pyproject.toml#project.version`.
- Лицензия продукта: `Apache-2.0`; локальный `LICENSE` byte-for-byte совпадает
  с официальным текстом и имеет SHA-256
  `cfc7749b96f63bd31c3c42b5c471bf756814053e847c10f3eb003417bc523d30`.
- Инвентарь: 50 прямых Python-зависимостей, 11 образов, 3 встроенных Web-файла,
  6 прямых зависимостей исключённого прототипа и 1 first-party asset.
- Лицензионный итог: 59 `compatible`, 5 `conditional`, 6 `excluded`,
  0 `incompatible`.
- Следующий разрешённый этап после обновления ledger: `02`.

Это техническая проверка состава и воспроизводимости комплекта, а не замена
юридического заключения для конкретной юрисдикции или способа распространения.

Runtime-boundary к Stage `01` неприменима: этап не меняет application code,
runtime configuration, API, persistence, browser flow или external side
effects. Ближайшая содержательная реальная граница проверена не тестом: Hatch
собрал настоящий wheel во временный каталог, затем из ZIP/`METADATA` прочитаны
фактические version, license expression и четыре вложенных license/notice files.
SPDX artifact отдельно прошёл официальный JSON Schema 2.3, а vendor files —
byte-for-byte сверку с upstream. Production runtime, registry и publish pipeline
не затрагивались и не могут служить допустимым proof для docs/package-only
authority этого этапа.

## Проверка трёх источников исполнения

| Поле | Доказательство | Итог |
|---|---|---|
| `plan_doc` | `docs/architecture/platform/roehub-self-hosted-oss-platform-v1.md` связывает Stage `01` с лицензией и выпуском | `passed` |
| `prompt_pack_dir` | Stage prompt существует, имеет `stage.id=01` и prerequisite `00` | `passed` |
| `stage_ledger` | До начала: `00=accepted`, `current_stage=01`; Stage `01` переведён в `in_progress` до implementation writes | `passed` |
| Authority | Разрешены локальные implementation writes; Git publication и production mutation запрещены | `passed` |

## Проектная лицензия и управление вкладом

- `LICENSE` содержит официальный Apache License 2.0 без локальных изменений.
- `NOTICE` содержит только project notice и требуемую атрибуцию Lightweight
  Charts; полный технический реестр не изменяет условия Apache-2.0.
- `CONTRIBUTING.md` фиксирует узкие изменения, contract classification,
  проверки, запрет секретов и лицензию намеренно отправленных вкладов.
- `SECURITY.md` не обещает несуществующий приватный адрес: GitHub private
  vulnerability reporting используется только если кнопка реально доступна;
  иначе создаётся минимальный issue без эксплуатационных деталей.
- Секреты, contacts, credentials, DSN, provider payloads и production data в
  созданные артефакты не записывались.

Первичные источники:

- `https://www.apache.org/licenses/LICENSE-2.0.txt`;
- `https://www.apache.org/legal/apply-license`;
- `https://spdx.org/licenses/Apache-2.0.html`.

## Версия и совместимость release manifest

| Контракт | Решение |
|---|---|
| Source of truth | Только `pyproject.toml#project.version`; `uv.lock` является синхронным lock artifact, а не редактируемым конкурентом |
| Формат | SemVer 2.0.0 без `v`; tag/release name в будущем использует `vX.Y.Z` |
| Начальная версия | `0.1.0`; прежний `0.0.0` был техническим placeholder и не считается опубликованным стабильным контрактом |
| До `1.0.0` | Несовместимый публичный контракт повышает `MINOR`, совместимое исправление — `PATCH` |
| После `1.0.0` | Несовместимый публичный контракт повышает `MAJOR` |
| Manifest schema | `io.roehub.release/v1alpha1`; неизвестные optional fields игнорируются, removal/semantic change required field требует новой schema version |

`prototypes/roehub-v2` имеет собственный `private` package metadata и исключён
из source/binary release set. Его `0.0.0` не является версией Roehub и не
участвует в release manifest.

## Сторонние компоненты и обязательства

Детерминированный policy находится в `tools/release/oss_policy.json`. Проверка
fail closed при новом прямом package, image, vendor file, font/binary/asset,
неизвестной лицензии, изменившемся vendor hash, отсутствии обязательства для
`conditional` или несовместимом status.

Условия, которые остаются барьерами конкретного комплекта, но не являются
несовместимым объединением в текущем source tree:

| Компонент | Лицензия | Текущий вывод | Следующий барьер |
|---|---|---|---|
| `psycopg[binary]` | `LGPL-3.0-only` | Отделимая библиотека; Roehub не перелицензируется | Для каждого wheel сохранить license/source notice и проверить `psycopg-binary`, `libpq`, OpenSSL contents |
| `grafana/grafana:latest` | `AGPL-3.0-only` | Только отдельный неизменённый Compose service как aggregate | Заменить `latest` на audited digest, сохранить notices и corresponding-source access; объединение/модификация блокирует выпуск без новой проверки |
| `python:3.12-slim` | `PSF-2.0` + layers | Python license совместима, транзитивные Debian layers ещё не закрыты | Digest-level SBOM и notices |
| `prom/prometheus:latest`, `prom/blackbox-exporter:latest` | `Apache-2.0` | Прямые лицензии совместимы | `latest` запрещён в release manifest до audited digest |

Lightweight Charts `5.2.0` и его `NOTICE` byte-for-byte совпали с upstream.
Требование пользовательской атрибуции выполнено через
`attributionLogo: true` в `apps/web/dist/js/charts/backtest_series.js` и теперь
проверяется policy tool. `htmx 1.9.12` byte-for-byte совпал с upstream и
классифицирован как `0BSD`. Fonts и bundled native binaries не обнаружены.

Известные non-blocking unknowns записаны в policy и generated metadata:

1. транзитивные пакеты `uv.lock` ещё не проверены по одному;
2. container layers и OS packages требуют digest-specific audit;
3. platform-specific Torch и Psycopg wheels требуют artifact-level audit;
4. образы с `latest` являются inventory evidence, а не release inputs.

## Воспроизводимые артефакты и CI

`tools/release/oss_metadata.py` использует только Python stdlib, локальные файлы
и `git ls-files`; update checks, telemetry и другие runtime network calls не
добавлялись. Он создаёт и проверяет:

- `tools/release/preliminary-sbom.spdx.json` — валидный SPDX 2.3 document;
- `tools/release/THIRD_PARTY_NOTICES.md`;
- `tools/release/release-metadata.json`.

`.github/workflows/ci.yml` получил обязательный read-only job `oss-metadata`.
Job не имеет write permissions, не логинится в registry и не вызывает publish
workflow. Существующий `publish-app-image.yml` не запускался и не изменялся.

Реальный wheel `roehub-0.1.0-py3-none-any.whl` собран только во временном
каталоге `/tmp`. Его metadata содержит `Version: 0.1.0`,
`License-Expression: Apache-2.0` и четыре license files: project `LICENSE`,
top-level `NOTICE`, generated third-party notices и upstream Lightweight Charts
notice.

## Матрица влияния на контракты

| Измерение | Классификация | Причина / совместимость |
|---|---|---|
| Public API и DTO | `none` | Runtime routes, payloads и ports не менялись |
| Persistence | `none` | Schema, migrations и данные не менялись |
| Конфигурация продукта | `none` | Runtime config/defaults не менялись |
| Package/release metadata | `compatible-change` | Placeholder `0.0.0` заменён первым SemVer `0.1.0`; public stable release ранее не существовал |
| Release manifest/hash identity | `compatible-change` | Создан новый `v1alpha1` metadata contract и deterministic artifact hashes |
| Build dependency | `compatible-change` | `hatchling` закреплён `1.31.0`; wheel доказан |
| CI gate | `compatible-change` | Новый fail-closed read-only license/SBOM check; публикации нет |
| Identity, cache/request keys | `none` | Пользовательская и runtime identity не менялись |
| Service calls / external effects | `none` | Сетевое поведение продукта не менялось; были только read-only evidence requests |
| Audit, инструкции, security flow | `compatible-change` | Добавлены project governance и безопасный reporting flow |
| Browser defaults | `none` | Текущая attribution setting только проверена, не изменена |

Rollback до публикации: вернуть metadata/docs/tooling одним scoped revert. После
первого выпуска version rollback запрещён: следующий исправленный комплект
получает новую SemVer version и новый manifest.

## Проверки

| Проверка | Результат |
|---|---|
| Официальный Apache text vs `LICENSE` SHA-256 | `match=yes`, `cfc7749b96f63bd31c3c42b5c471bf756814053e847c10f3eb003417bc523d30` |
| Upstream vendor hashes | htmx, Lightweight Charts и `NOTICE` совпали byte-for-byte |
| `uv run python tools/release/oss_metadata.py --write` | `passed`, 3 artifacts |
| `uv run python tools/release/oss_metadata.py --check` | `passed`, no drift |
| SPDX 2.3 official JSON Schema | `valid` |
| `uv run ruff check tools/release/oss_metadata.py tests/unit/tools/test_oss_metadata.py` | `passed` |
| `uv run pytest -q tests/unit/tools/test_oss_metadata.py` | `3 passed` |
| `uv lock --check` | `passed`, 207 packages resolved |
| Wheel build and metadata inspection | `passed`; version/license/four license files present |
| CI YAML parse and required dependency | `passed`; `oss_metadata` входит в final `ci.needs` |

После добавления этого отчёта дополнительно выполнены project-map/docs-index
generation/check и scoped `git diff --check`; результаты отражены в ledger.

## Файловый манифест и authority

- Созданы: `CONTRIBUTING.md`, `NOTICE`, `SECURITY.md`,
  `tools/release/{README.md,oss_policy.json,oss_metadata.py,preliminary-sbom.spdx.json,THIRD_PARTY_NOTICES.md,release-metadata.json}`,
  `tests/unit/tools/test_oss_metadata.py` и этот отчёт.
- Изменены: `LICENSE`, `pyproject.toml`, `uv.lock`, `.github/workflows/ci.yml`,
  stage ledger и generated docs index.
- Удалённых файлов нет.
- `uv.lock` синхронизирован с canonical version, а test path добавлен как
  необходимое focused evidence для нового tooling; оба пути записаны как
  обоснованные secondary touches.
- Foreign changes в `.codex/PLANS.md`, supersession docs, project map и
  pre-existing generated docs hunks сохранены.
- Commit, staging, push, release, deploy, registry mutation и production
  mutation не выполнялись.

## Передача Stage 02

Stage `02` может использовать `0.1.0` и `io.roehub.release/v1alpha1` как
release metadata baseline. Он не вправе публиковать артефакты. Unknown
transitive license risks передаются Stages `17`, `18`, `20` и `22`, где
конкретные digest-pinned bundles должны закрыть условия `conditional`.
