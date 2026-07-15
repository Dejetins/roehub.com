---
validation_depth: runtime
tests_only_acceptance: false
real_boundary_evidence:
  - docs/architecture/platform/roehub-self-hosted-oss-platform-v1-stage-reports/evidence/21-backup-restore-upgrade-runtime-proof.json
---

# Stage 21 — резервное копирование, восстановление, обновление и откат

## Статус

- Этап: `21`.
- Состояние: `accepted`.
- Режим: `goal_driven`.
- Глубина принятия: `runtime`; тесты служат только gate и не являются
  основанием принятия без `real-boundary` Docker/Unix-socket репетиции.
- Граница доказательств: `N/A` — одноразовые Docker-проекты, только
  сгенерированные данные, временные operator-owned keys и versioned unpublished
  fixture `0.0.0 → 0.1.0`. Production/current state, реальные учётные данные,
  provider effects, реальные ордера, staging, commit, push и deploy исключены.

## Принятый контракт

`io.roehub.installation-backup/v1alpha1` покрывает ровно восемь владельцев:
release/config, PostgreSQL, ClickHouse, Redis checkpoint, OpenBao, artifacts,
plugin/operation/audit и bounded observability history. Для каждого владельца
manifest фиксирует точный consistency mode, source schema, capture timestamp,
ограничения, размеры и SHA-256 plaintext/ciphertext. Canonical manifest имеет
detached Ed25519 signature и installation fingerprint.

Capture coordinator создаёт состояние внутри эффекта control agent, фиксирует
реальное quiesce window и digest каждого файла. `source_root` обязан быть
owner-only `0700`, файлы — `0600`. Один защищённый file descriptor используется
для чтения и передачи bytes в `age`; permanent bundle содержит только
ciphertext и подписанные metadata. После проверенной публикации plaintext
staging удаляется и каталог синхронизируется. При отмене или частичном отказе
staging сохраняется только для digest-bound resume и удаляется после успешного
завершения.

До `latest-verified.json` проверяются configured public key, installation
fingerprint, signature, ciphertext digests и расшифровываемость всех восьми
элементов operator identity. Symlink ancestors, небезопасные права, source
swap, вложенные roots, неизвестные файлы и dirty restore resume завершаются
fail closed.

`restore` публикует terminal success только после typed state coordinator:
PostgreSQL, ClickHouse и Redis импортированы, artifacts и release/config
развёрнуты, exact OpenBao snapshot связан с уже выполненным fresh-volume force
restore, все digests и product rows сверены, новая установка отвечает
`status=ready`. RTO охватывает decrypt, импорт, сверку и ready-result.

`update` и `rollback` исполняются через тот же control backend. Переходы берутся
из owner-protected `io.roehub.installation-release-policy/v1alpha1`, связанного
с installation fingerprint. Rollback разрешён только для reversible
transition; irreversible update без trusted forward recovery plan digest
отклоняется. Post-effect receipt failure переводится в `unknown`, а signed
manifest/result и `operation_id` позволяют crash-safe reconciliation без
повтора эффекта.

## Проверка на реальной границе

Этап не принят по unit tests. Обязательный артефакт runtime-проверки:
`docs/architecture/platform/roehub-self-hosted-oss-platform-v1-stage-reports/evidence/21-backup-restore-upgrade-runtime-proof.json`.

- `roehub-stage21-proof13` поднял отдельные PostgreSQL `16`, ClickHouse,
  Redis, OpenBao и шесть сервисов наблюдаемости. Web/API не запускались.
- Через настоящий authenticated Unix socket и `roehubctl` выполняющийся backup
  был отменён вторым запросом, завершился `backup.cancelled`, не опубликовал
  manifest и затем успешно возобновился. Аналогично отменён и возобновлён
  выполняющийся restore.
- Подписанный manifest содержит восемь encrypted entries. Source plaintext
  staging автоматически очищен после полного verify.
- Fresh target получил пять PostgreSQL rows (`user`, `config`, `plugin`,
  `operation`, `audit`), три ClickHouse rows с тем же временным диапазоном,
  Redis checkpoint, 32 MiB content-addressed artifact, три точных файла
  release/config и bounded observability snapshot.
- Извлечённый OpenBao ciphertext совпал с metadata и operator-bound digest,
  расшифровался тем же operator identity и является тем самым snapshot, который
  в этой репетиции прошёл настоящий fresh-volume force restore и fresh-storage
  guard. Recovery/unseal values в bundle и evidence отсутствуют.
- Через `roehubctl update` доказаны `0.0.0 → 0.1.0`, инъекция
  `upgrade.injected_failure_before_commit` и безопасное возобновление. Через
  `roehubctl rollback` восстановлена `0.0.0` во вторую свежую цель. Попытка
  irreversible `0.0.0 → 0.2.0` без forward plan отклонена кодом
  `upgrade.forward_recovery_plan_required`.
- Измерено `observed_rpo_seconds=37.535141` и
  `observed_rto_seconds=6.386634541209787`. Это результаты одной локальной
  репетиции, а не SLA.
- Cleanup всех пяти Compose-проектов завершился с кодом `0`; остаточные
  containers, networks и volumes равны `0`.

## Независимая проверка и исправления

Единственная independent cold review дала `Block`: шесть `Blocker`, три `High`
и три замечания меньшего уровня. Второй independent review не запускался.
Локальная холодная перепроверка после исправлений дала `Release after fixes`.

Закрыты искусственные RPO/RTO, plaintext staging, decrypt-only restore, обход
control agent для update/rollback, непроверенный latest pointer, path/TOCTOU и
dirty-resume gaps, номинальная отмена, post-effect crash reconciliation и
semantic drift JSON Schema. `infra/openbao/verify_runtime.py` изменён узко в
рамках required OpenBao recovery evidence: он принимает внешний operator age
key pair и экспортирует metadata sidecar для exact snapshot binding.

## Контрактное влияние

| Измерение | Классификация | Обоснование |
|---|---|---|
| API и DTO | `breaking-change` | В operation protocol добавлены исполняемые backup/restore cancel actions и release recovery semantics. |
| Порты и adapters | `breaking-change` | Restore/update/rollback требуют typed installation state coordinator и crash reconciliation. |
| Хранение | `breaking-change` | Добавлены encrypted bundle, signed manifest, capture/release policies, progress, result, receipt и latest pointer. |
| Runtime/config | `breaking-change` | Control agent получает owner-protected backup/release policy и operator key references. |
| Identity/RBAC | `none` | Authority остаётся у существующего `installation_owner`; новые роли не вводятся. |
| Request identity | `breaking-change` | Новые action values и release/backup subjects входят в request digest и durable receipt. |
| Generation hashes | `breaking-change` | Добавлены пять deterministic JSON Schema и обновлены docs/project-map outputs. |
| Межсервисные вызовы | `compatible-change` | Добавлена локальная цепочка `roehubctl → Unix socket → control-agent → state owners`; Web/API не нужны. |
| Внешние эффекты | `breaking-change` | Backup пишет ciphertext, restore/update/rollback создают отдельные installation targets. Production effects не выполнялись. |
| Аудит | `compatible-change` | Hash-chain journal получает redacted recovery actions и может reconcile `unknown` по signed effect receipt. |
| Инструкции | `compatible-change` | Обновлены backup/recovery и emergency runbooks. |
| Browser defaults | `none` | Web UI на этом этапе не изменялся. |

## Проверки

- Сфокусированные recovery/control/schema/OpenBao/runbook tests: `44 passed`.
- Полный repository gate: `1985 passed`, `4 warnings` только от устаревающего
  per-request cookies API в `httpx`.
- Stage `21` ruff: `passed`; targeted pyright: `0 errors, 0 warnings`.
- `python -m tools.backup.generate_schemas --check`: `passed`.
- Реальный runtime proof: `passed`; residual Docker resources: `0`.
- Docs, runbook и project-map generators/checks, full ruff и `git diff --check`
  выполнены перед переводом журнала на Stage `22`.

## Остаточные границы

В рамках Stage `23` этот runtime lifecycle повторён ещё раз с первой попытки.
После обновления Docker Compose до `5.3.1` проверочная сеть сохранила
`internal: true`: host ports не публикуются, а readiness
`operational-health` проверяется настоящим HTTP-запросом внутри контейнера.
Grafana получает только одноразовый тестовый пароль в versioned override;
production secret/config не читается. Все шесть monitoring services оставались
работоспособными во время recovery, Web/API не запускались, cleanup пяти
Compose-проектов снова завершился без остаточных ресурсов.

- `0.0.0` — versioned unpublished fixture первого релиза, не опубликованный
  `N-1`. Автономный release bundle относится к Stages `22`–`23`.
- Одна локальная репетиция RPO/RTO не задаёт SLA.
- Runtime proof использует одноразовые state-owner adapters; packaging полного
  host unit и автономного комплекта проверяется на Stages `22`–`23`.
- Production/current state, реальные ключи/credentials, provider effects и
  реальные ордера не читались и не изменялись. Commit, push, deploy и staging
  не выполнялись.
