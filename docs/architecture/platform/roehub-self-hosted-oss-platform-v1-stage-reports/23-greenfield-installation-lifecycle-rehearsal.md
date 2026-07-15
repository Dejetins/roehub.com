---
validation_depth: runtime
tests_only_acceptance: false
real_boundary_status: passed
real_boundary_evidence:
  - docs/architecture/platform/roehub-self-hosted-oss-platform-v1-stage-reports/evidence/23-greenfield-installation-lifecycle-proof.json
  - docs/architecture/platform/roehub-self-hosted-oss-platform-v1-stage-reports/evidence/23-greenfield-admin.png
---

# Этап 23 — репетиция жизненного цикла чистой установки

## Статус

- Этап: `23`.
- Состояние: `accepted`.
- Режим: `goal_driven`.
- Глубина принятия: `runtime`; тесты не заменяли автономную установку,
  браузерную проверку, резервное копирование, восстановление и rollback.
- Граница доказательств: `N/A` — сохранённый подписанный комплект Stage `22`,
  одноразовые Docker-проекты и только синтетические данные.
- Исключены: current/production state, реальные credentials, персональные
  данные, реальные orders, provider writes, staging, commit, push, публикация
  и deploy.
- Следующий разрешённый этап: `24`.

## Проверенный кандидат

Установлен сохранённый комплект
`/Users/daniildegtyarev/.cache/roehub/stage22-offline-release/candidates/roehub-0.1.0`.
Его подпись проверена доверенным публичным ключом Stage `22` до запуска.

- версия: `0.1.0`;
- manifest SHA-256:
  `4f4a34e070724b5e997b3a5cbd6526212515b4db683ebd5ba53d4ddd197bab7f`;
- tree SHA-256:
  `b5fff5abd04995ddf64321520a620e768614f787aa937232e059da3315edb21a`;
- runtime index digest:
  `sha256:e3303b08b337c24e451045985355047d0c383eefc39aa1965400ecc1e4a9d0ae`;
- ML runtime index digest:
  `sha256:2fa36fe8f01798dd139dc70d60356b03e3e4453bca4ad2d39f0f69a4b0758531`;
- OpenBao index digest:
  `sha256:610395fc927391e2cfa4e082ba9cb520a8359b2c14591a9ff63378bf0c52225b`.

## Реальная репетиция

`io.roehub.greenfield-installation-lifecycle-proof/v1alpha1` завершился со
`status=passed` за `415.539 s`.

1. В пустые volumes установлена конфигурация `trading`; bootstrap ticket
   создал единственного `installation_owner`, восемь recovery codes не
   сохранялись в evidence.
2. Через публичные/use-case границы созданы две организации, три пользователя,
   четыре membership, две invitation и две plugin permission. Запрос
   межорганизационного administrative snapshot получил `404`.
3. Chromium подтвердил authenticated admin UI, `0` console errors и bounded
   screenshot без session/cookie/recovery material. SHA-256 PNG:
   `04bb8ff4efee0c20da5eb9356231c7f4000bd9addf7e19ae63a6a7036f5f7dc8`.
4. PostgreSQL, ClickHouse, Redis checkpoint, OpenBao metadata и artifact digest
   скопированы только через новый encrypted backup. Вторая пустая установка
   получила точные counts/digests и восстановленный passkey login.
5. После полного teardown третья пустая установка повторила bootstrap с
   ожидаемыми структурными counts.
6. Runtime Stage `21` повторён с первой попытки: отмена/resume backup/restore,
   injected update failure, безопасное resume, rollback `0.1.0 → 0.0.0` и
   запрет irreversible update без trusted forward plan прошли.
7. Все source/target/repeat/recovery containers, networks, volumes, browser
   session и owned offline image tags очищены.

Current production database, identity store, OpenBao, Redis checkpoints,
artifact paths и secrets не читались и не монтировались. Evidence фиксирует
`current_production_access=false`, `personal_data_present=false`,
`external_provider_writes=false` и `real_orders=false`.

## Дефекты, найденные реальной границей

Репетиция не была принята по одним тестам и до успешного полного повтора нашла
и закрыла два продуктовых дефекта.

1. Bootstrap challenge преждевременно ссылался FK на ещё не созданного owner.
   Будущий UUID теперь хранится в одноразовом challenge context, nullable
   `user_id` связывается после создания пользователя, checksum миграции
   `0012_identity_local_auth_v1.sql` синхронизирован с manifest.
2. Server-side Web auth gate обращался к `/api/auth/current-user` через прямой
   API address. `WEB_API_BASE_URL` теперь указывает на Web BFF
   `http://web:8010`, а `WEB_API_UPSTREAM_URL` сохраняет внутренний upstream
   `http://api:8000`.

Оба изменения вошли в заново выпущенные runtime/ML images и после этого были
повторно проверены Stage `17`, полным лицензионным/автономным Stage `22` и этим
Stage `23`.

Проверочный контур также был усилен без изменения продукта: canonical audit
table, строгий admin marker, isolated CDP screenshot, точный подсчёт console
errors, HTTP address OpenBao, bounded force-click только animated passkey
button и in-container readiness для monitoring на `internal: true` сети.

## Память и ресурсы

Тяжёлые фазы выполнялись последовательно с `GOMAXPROCS=2`,
`GOMEMLIMIT=1GiB`, `SYFT_PARALLELISM=1` и `UV_CONCURRENT_DOWNLOADS=2`.
Одновременно работала только одна полная установка из `24` постоянных
контейнеров. Во время наблюдения свободная системная память оставалась в
диапазоне примерно `32–51%`; крупнейшие процессы приложения использовали около
`299 MiB` для API и `282.2 MiB` для artifact publisher, ClickHouse — до
`246.9 MiB`. После cleanup Docker не удерживал контейнеры Stage `23`.

## Контракты и совместимость

| Поверхность | Классификация | Обоснование |
|---|---|---|
| Публичные API/DTO | `none` | Существующие маршруты и payload не менялись. |
| Persistence | `compatible-change` | Greenfield FK создаётся только после появления owner; опубликованных установок и legacy migration нет. |
| Config/defaults | `compatible-change` | Server-side Web использует предназначенный BFF URL; upstream API остаётся отдельным внутренним адресом. |
| Identity | `compatible-change` | Одноразовый challenge сохраняет будущую identity без ослабления ticket/recent-auth/passkey правил. |
| Container/hash identity | `breaking-change` | Runtime/ML digest и подписанный кандидат перевыпущены до первого публичного релиза. |
| Межсервисные вызовы | `compatible-change` | Исправлен Web BFF hop; внешний контракт не изменён. |
| Внешние эффекты | `none` | Registry/provider/production writes и реальные orders отсутствовали. |
| Browser defaults | `none` | Пользовательский login/admin flow сохранён и доказан после restore. |
| Proof harness | `none` | Изменены только изолированные verifier/fixture границы и sanitization. |

Основная классификация — `breaking-change` только по container/release identity
ещё не опубликованного greenfield кандидата; функциональные контракты
сохранены.

## Проверки

- Полный автономный lifecycle: `passed`; attempt count Stage `21` — `1`.
- Runtime evidence SHA-256:
  `0c76f1311586e50f85633fcaacacfe792197a58ed194cd5353e9f7023bdacb3c`.
- Сфокусированный Python gate: `66 passed`, три прежних предупреждения `httpx`.
- Ruff: `passed`; Pyright: `0 errors, 0 warnings`.
- Runtime topology generation/check и OSS metadata check: `passed`.
- Docs index/project map generation/check и `git diff --check`: `passed`.

## Проверка перед передачей

- Режим: холодная самостоятельная перепроверка; новый независимый subagent не
  запускался.
- Вердикт: `Release after fixes`.
- Исправлены: оба продуктовых greenfield-дефекта, неточный браузерный proof и
  несовместимая с internal network host-port проверка monitoring readiness.
- Остаточные риски: `0.0.0` остаётся versioned unpublished fixture, а Linux
  amd64/arm64 и macOS M3 Pro матрица ещё должна быть доказана Stage `24`.

Stage `24` получает сохранённый подписанный кандидат, точные digest и полный
greenfield lifecycle proof. Stage `25` не разрешён автоматически даже после
принятия Stage `24`.
