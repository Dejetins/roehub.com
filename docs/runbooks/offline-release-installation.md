# Автономная установка подписанного выпуска

## Область применения

Инструкция относится к распакованному комплекту
`io.roehub.offline-release-bundle/v1alpha1`. Она не переносит текущую базу,
пользователей, секреты или артефакты и предназначена только для новой установки.

## Предварительные условия

- Docker Engine и Docker Compose v2 доступны локально;
- установлены Python 3.9 или новее, OpenSSH `ssh-keygen` и `skopeo`;
- комплект перенесён на целевой хост целиком;
- доверенный `ssh-ed25519` public key получен отдельно от комплекта.

Не используйте `trust/release-signing-key.pub` из комплекта как единственный
источник доверия. Встроенный ключ позволяет сверить identity, но не доказывает,
кто передал комплект.

## Проверка без активации

Из корня распакованного комплекта выполните:

```bash
python3 tools/release/offline_bundle.py verify \
  --bundle "$PWD" \
  --trusted-public-key /secure/path/roehub-release-signing-key.pub
```

Команда до любых изменений Docker проверяет:

- `SSHSIG-Ed25519` подпись манифеста;
- identity внешнего доверенного ключа;
- точный список, размер, режим и SHA-256 каждого файла;
- digest OCI index и child manifests для `linux/amd64`/`linux/arm64`;
- два непустых SPDX 2.3 SBOM для каждого образа;
- соответствующие исходники Grafana/Loki и запрет phone-home по умолчанию.

Любая ошибка является fail-closed: не переходите к импорту или запуску.

## Импорт и подготовка профиля

```bash
./tools/release/install-offline.sh \
  --trusted-public-key /secure/path/roehub-release-signing-key.pub \
  --state-directory "$HOME/.local/share/roehub/offline" \
  --profile base \
  --runtime-smoke
```

Допустимые профили: `base`, `trading`, `ml`. Installer повторяет полную
проверку, извлекает из локальных OCI archive только архитектуру текущего хоста,
загружает образы через Docker без registry и создаёт:

- `offline-image-lock.json` с immutable Docker image ID;
- `compose.<profile>.offline.yaml` с `pull_policy: never`;
- результат `docker compose config` для исходного и offline override.

Опция `--runtime-smoke` запускает Roehub с `--network none`, read-only rootfs и
writable tmpfs. Она не поднимает хранилища и не меняет пользовательские данные.

## Явная активация

После успешной проверки оператор может отдельно запустить выбранный профиль:

```bash
docker compose \
  -f "configs/installation/generated/base/compose.yaml" \
  -f "$HOME/.local/share/roehub/offline/compose.base.offline.yaml" \
  up -d
```

Перед `up -d` настройте секретные ссылки OpenBao и persistent paths согласно
`configs/installation/roehub.yaml`. Не помещайте значения секретов в YAML или
командную строку.

## Остановка и диагностика

Остановка не удаляет данные:

```bash
docker compose \
  -f "configs/installation/generated/base/compose.yaml" \
  -f "$HOME/.local/share/roehub/offline/compose.base.offline.yaml" \
  down
```

При ошибке сохраните stderr, `offline-image-lock.json`, версию Docker/Compose и
SHA-256 `offline-release-manifest.json`. Не заменяйте образы тегами и не
редактируйте подписанные файлы: получите новый полный комплект.

## Backtest NPY policy, rollout and recovery

В выбранной установке `backtest_artifacts.retention_policy` версии 1 может
задавать per-coordinate `signals: on_demand` и `hit_times: on_demand`.
Отсутствие policy сохраняет precompute. Candles/mappings и нужный funding
остаются опубликованными; имеющиеся пригодные derivatives переиспользуются.
Policy не меняет queued request, не удаляет старые файлы и не включает proactive
расчёт всех координат. Installation configs создаются штатным генератором;
`configs/installation/runtime-input-inventory.json` — value-free inventory
потребителей, не пользовательский интерфейс ввода environment.

Worker composition использует `ROEHUB_BACKTEST_SCRATCH_ROOT` (по умолчанию
`roehub-backtest-attempts` в системном temp). Границы в байтах:

| Input | Default | Область |
| --- | ---: | --- |
| `ROEHUB_BACKTEST_ATTEMPT_DISK_BYTES` | 2147483648 | Generated NPY + derivative manifest на attempt |
| `ROEHUB_BACKTEST_WORKER_DISK_BYTES` | 8589934592 | Суммарно зарезервированный scratch |
| `ROEHUB_BACKTEST_DISK_RESERVE_BYTES` | 1073741824 | Свободное место после admission |
| `ROEHUB_BACKTEST_ATTEMPT_COMPUTE_BYTES` | 536870912 | Conservative preparation working-set budget |

Total reservation включает generated allowance плюс 64 MiB overhead даже для
native/legacy attempt. Каждый input/output IPC и prepared manifest ограничен
8 MiB до записи; derivative manifest имеет отдельный 8 MiB слот внутри generated
allowance. Stdout/stderr непрерывно дренируются через pipes с ограниченным tail,
без растущих файлов. Под namespace flock резервируется вся сумма; освобождение
происходит после reap и удаления принадлежащей attempt директории. В v2 ownership
marker `reserved_bytes` означает total; старые v1 markers читаются с дополнительным
консервативным overhead и сохраняются до точной reconciliation. Эти guards не
являются filesystem quota против посторонних процессов: внешний writer может
исчерпать диск, тогда ENOSPC завершает attempt без готового результата.

Перед payload attestation для native/mixed/generated проверяется консервативный
working set всего объявленного inventory; builder получает оставшийся бюджет.
Это не ограничение OS RSS, включающего Python/JIT/allocator. Существующие более
строгие signal chunk/cell budgets сохраняются. Настройка disk не разрешает
выделение такого же объёма RAM. Cache между attempts не создаётся.

Порядок обновления: additive migration `0028`, совместимые readers, затем полный
worker/loader/lazy/publisher protocol; drain старых workers, publishers и
COUNT-only previous-slot cleanup до включения новых writes/partial policy.
Также нельзя смешивать старый scratch admission с новым v2 accounting.
Неравномерные persisted TP/SL `levels_pct` требуют нового reader до producer.
До новых writes возможен возврат старого кода. После schema-2 partial roots,
recipe jobs или sparse payload старый binary небезопасен: сохранить new readers,
остановить новые submissions/policy changes, drain или сохранить новые jobs и
подготовить полные совместимые artifacts перед отдельно разрешённым rollback.
Nullable provenance columns и историю не удалять; потерянный sparse intent из
уже expanded historical range не восстанавливать предположением.

Для диагностики сверить physical `(exchange, market_type, symbol, slot)`, reader
`owner_token`, kind/organization/id/attempt, parent incarnation, generation/hash
и epoch с реальным процессом. Reader primary key добавляет `owner_token` к physical
key. Publisher берёт `slot_a` затем `slot_b`, повторно проверяет pointer/target и
держит incremental-source reader до конца чтения. Previous-slot deletion также
требует writer admission и повторной проверки, что target не current.

Lease timeout, PID или возраст directory не разрешают удаление. Неизвестная
liveness/DB loss оставляет quarantine. Worker recovery требует inherited lifetime
lock и подтверждённого reap/смерти точного owner; writer recovery дополнительно
сверяет pointer и staged/final manifests. Epoch ограждает DB commit, но не
останавливает filesystem writes. У синхронного API candle-read после crash может
остаться pin без worker attempt directory: нужна ручная сверка точного owner и
подтверждение смерти процесса/закрытия mmap перед repository reconciliation.
TTL reclaim запрещён. Удалять можно только подтверждённые owned incomplete files.

Это локально проверенный протокол, не команда обновления существующей production
установки. Runbook `backtest-artifacts-rebuild.md` остаётся retired. Конкретный
хост и production rollout этим документом не выбираются.

Accounting bound использует максимум одновременно существующих byte payloads:
`ownership.json` (owner ≤4 MiB), namespace creation intent (тот же owner + bounded
identity), full `preflight.json` (≤8 MiB), `attempt.json`/lazy `input-context.json`
(≤8 MiB), `result.json` (≤8 MiB), `inputs/prepared-inputs.yaml` (≤8 MiB).
Prepared temporary и final hard link делят inode; derivative temporary/final
rename не удваивает payload. Производные NPY и `inputs/manifest.yaml` входят в
отдельный generated reserve, включая предварительный 8 MiB manifest slot.
64 MiB overhead оставляет запас сверх этих максимумов; stdout/stderr не занимают
scratch. Diagnostic evidence при явно настроенном внешнем evidence directory и
persisted lazy cache — отдельные хранилища, не worker attempt scratch.

Full `result.json` содержит top summaries и агрегированную telemetry, без trades;
production admission допускает top_n≤50, test profile≤100. Lazy IPC передаёт
только cache status/path после записи detail в существующий cache. Поэтому лимит
8 MiB не ограничивает число lazy trades. Проверен roundtrip 100 assembled summary
rows. Для произвольных custom расширений telemetry/top_n это новый явный предел;
oversized output отклоняется, а не усекается. Более строгие memory/disk admission
и v2 marker — `breaking-change` для старого worker и ранее пропускаемых overbudget
attempts; financial schema/precision не изменены. Drain относится и к этому
изменению, не только к schema-2 publication.

## Управление Backtest artifacts: неактивный opt-in API

После additive migrations до `0031_backtest_artifact_storage_v1.sql` новая API
композиция читает `ROEHUB_ARTIFACT_MANAGEMENT_INSTALLATION_ID`,
`ROEHUB_BACKTEST_ARTIFACTS_CONFIG`, `ROEHUB_INDICATORS_CONFIG` и существующий
`STRATEGY_PG_DSN`. Это не активирует capability, scheduler или legacy handoff.
Сначала нужны совместимые readers, импорт старых policy, остановка/дренирование
старого writer и отдельная явная owner-активация по принятому management contract.

Storage capability работает через существующий подписанный Unix control-agent
и его API service identity. На стороне control-agent используется тот же trusted
конфиг артефактов и `ROEHUB_STORAGE_POSTGRES_DSN` (или `STRATEGY_PG_DSN`).
Registry содержит `backtest-primary`, соответствующий `artifact_root` в конфиге.
Браузер передаёт только этот ID, settings version, budget/schedule и opaque token;
произвольные пути, owner credentials и worker receipts не принимаются.

Перед установочной активацией control-agent должен видеть именно worker mount:
одинаковая строка пути на host и в контейнере не доказывает общий volume. Применить
и проверить общий mount — отдельная установочная операция; текущие generated
Compose файлы этой функцией не изменены. Отсутствующий/недоступный root или control
service даёт unavailable, а не успешную проверку. Локальные disposable проверки
S06 не являются доказательством соответствия production mounts.

Installation owner открывает privileged storage GET, получает текущую version,
выполняет validate (узкая запись/fsync/unlink), затем configure с validation ID,
version и новым Idempotency-Key. Token ограничен пятью минутами, actor, root inode
и version; configure повторно проверяет root. Первая пустая настройка сохраняется
без включения schedules. Изменение root с файлами, pins или зарегистрированными
работами возвращает `relocation_required`; автоматического перемещения нет.
Изменение budget на прежнем root не удаляет файлы и не отменяет pins.

Cleanup всегда отдельный явный preview/confirm одного inactive slot. Проверяются
строгий current pointer, hash/generation manifests, существующий registered owner,
queued/active/quarantined pins, writer reservation и актуальная settings version.
TTL не разрешает удаление. Partial/unknown IO сохраняет reservation в quarantine;
повтор команды с тем же ключом возвращает `unknown`, не повторяет удаление.
Нужна отдельная owner reconciliation; не создавать новый cleanup key, пытаясь
обойти quarantine. Автоматическая cleanup policy не включается.

### S09 local verification and current recovery procedure

Before any separately authorized installation activation, run the current disposable
proof with the existing Python environment. `ROEHUB_BACKTEST_TEST_DSN` must point to a
new disposable PostgreSQL service; `ROEHUB_BASE_TEST_CLICKHOUSE_PORT` must point to a
new loopback disposable ClickHouse service. Fixtures create/drop their own databases
and schemas. Never point these variables at user/preview databases.

```sh
.venv/bin/pytest -q -ra tests/real_infra/backtest/test_artifact_integrated.py tests/real_infra/backtest/test_automatic_base_lifecycle.py tests/real_infra/backtest/test_artifact_dispatch.py tests/real_infra/backtest/test_artifact_http_storage.py tests/real_infra/backtest/test_artifact_input_lifecycle.py tests/real_infra/backtest/test_managed_derivative_publication.py tests/real_infra/backtest/test_artifact_management.py
```

The existing scheduler entrypoint is
`python -m apps.scheduler.backtest_artifact_publisher.main.main`. Managed composition
requires explicit `ROEHUB_ARTIFACT_MANAGEMENT_INSTALLATION_ID`, migrated settings,
trusted root and disabled legacy scheduling. Starting the process does not grant
activation authority. Retain the existing deployment's account/config sources;
this document provides no secrets or selected installation. Do not run the retired
rebuild runtime. A normal owner Storage save also does not activate scheduling.

For recovery, first read request/build state and retained published coverage. Do not
confuse a source watermark with published coverage. Restore the same root revision
and process configuration, then allow the existing managed cycle to reconcile owned
attempts under the host/lifetime locks. A matching fsynced receipt and verified
pointer/manifest can finish an interrupted PG completion. Missing/corrupt receipts,
unknown liveness or ambiguous publication remain quarantined: preserve files, pins
and reservations; no TTL takeover, new-key cleanup retry or manual SQL state reset.
Revoked actors are rejected on retry. Correct the cause before a new explicit command.

Budget edits are read by a running managed worker at its next admission, including
retained bytes and reserve; they do not cancel work already admitted. Root relocation
with data remains a separate operation. Cleanup still requires a fresh preview and
confirmation, and refuses current/pinned/active targets. Disable submissions and drain
owned work for rollback; retain schema-aware readers after sparse schema-2 publication.
Local tests do not certify shared production mounts, S3, providers or full-market load.


### Managed local ClickHouse budgets (S02, 2026-10-10)

Для существующей managed local установки `tools/installation/local_platform.py`
имеет явные режимы `clickhouse_workload_budgets`: `legacy`, `provisioned`, `active`.
Это отдельный локальный installation lifecycle, не изменение подписанного
комплекта или автоматическое включение generic offline topology. Включение требует
совместимого image-bound candidate и сохранения текущего trust binding.

Сначала `tools.installation.clickhouse_workloads` готовит additive private
credentials (0700 directory / 0600 files), не заменяя существующие. Публичные
outputs содержат profiles, paths и bootstrap, но не пароли. Bootstrap передаёт
private password-hash XML ClickHouse с корректным UID/mode, затем запускает
официальный entrypoint. После проверки `provisioned` server следует drain и
переключение `active` clients. Rollback возвращает prior application image и
`provisioned`, оставляя identities и данные; cleanup/rotation не подразумеваются.
Таблица caps, grants/MV, исключения admin, диагностика отказов и воспроизводимые
проверки находятся в разделе S02 local market-data recovery runbook. Существующие
storage migration manifests не меняются: candle/index/PG schemas сохранены.

### REST provider admission rollout (S03, 2026-10-10)

Новый HTTP composition требует common PostgreSQL и explicit egress scope. До
запуска новых scheduler/collector/CLI/strategy клиентов подтвердить остановку
старых участников и их I/O, применить additive phase
`market-data-provider-admission-0034`, затем активировать matching image/config.
Смешанные старые и новые REST клиенты не обеспечивают общий лимит.

Источник signed-topology конфигурации — `configs/prod/market_data.yaml`;
`tools/release/generate_runtime_topology.py --write` обновляет оба имени
market-data config и generation manifests, `--check` проверяет drift. Не править
сгенерированные файлы вручную. Scope `local-installation-egress` — явный пример
для одной установки, не определение NAT/IP. Сохраняются prod cap 3/factor 0.7,
selections, credentials и S02 ClickHouse resource contract; новая выдача секретов
для provider admission не требуется.

Rollback требует drain новых клиентов и восстановления предыдущих image **и**
config (legacy parser не понимает `shared_conservative`). Additive таблицы/поля,
расход квот, cooldown и неопределённые permits сохраняются. Удаление точного owner
возможно только после подтверждённой остановки всех его I/O; см. раздел S03
`docs/runbooks/navigator-local-market-data-recovery.md`. Подготовка артефактов и
локальная проверка не означают установку или production delivery.
