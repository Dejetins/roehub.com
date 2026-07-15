# Stage 02 — машинно-читаемые эксплуатационные инструкции

## Результат

- Дата: `2026-07-13`.
- Stage: `02`.
- Режим: `goal_driven`.
- Статус: `accepted`.
- Контракт: `ops.roehub.io/v1`.
- Граница доказательства: `N/A` для runtime — application code, конфигурация
  сервисов, alert rules и production state не менялись.
- Реальная граница артефактов: генератор обработал 25 фактических legacy
  runbooks и 58 фактических Prometheus alerts, проверил все 35 существующих
  alert → runbook links вместе с anchors и построил русские документы и JSON
  index из schema-valid YAML.
- Репрезентативный перенос: 5 runbooks, 10 problem IDs, 9 связанных alerts,
  7 allowlisted action capabilities и 1 явный monitoring gap.
- Неперенесённый остаток: 20 legacy runbooks явно записаны в JSON index для
  Stages `17` и `20`.
- Следующий разрешённый этап после ledger update: `03`.

Unit tests не подменяют доказательство границы: кроме отрицательных fixture
tests, validation прочитала реальный текущий corpus alert rules и Markdown
anchors, реальные legacy sources и generated outputs. Runtime incident
resolution и Web UI rendering исключены Stage authority и не заявляются.

## Проверка трёх источников исполнения

| Поле | Доказательство | Итог |
|---|---|---|
| `plan_doc` | План фиксирует `ops.roehub.io/v1`, русский render и typed actions | `passed` |
| `prompt_pack_dir` | Stage prompt существует, `stage.id=02`, prerequisite `00` | `passed` |
| `stage_ledger` | До начала: `00=accepted`, `01=accepted`, `current_stage=02`; затем `02=in_progress` | `passed` |
| Authority | Разрешены локальные schema/docs/tooling writes; publication и production mutation запрещены | `passed` |

## Канонический контракт

Созданы две Draft 2020-12 JSON Schema:

- `schemas/ops/runbook.schema.json` — английский канонический runbook;
- `schemas/ops/runbook-locale.schema.json` — полное русское locale-покрытие.

`schemas/ops/action-capabilities.json` является закрытым каталогом
возможностей. Runbook не может содержать `command`, `shell`, `script`, `argv`
или environment payload. Diagnostic capability всегда read-only; mutating
action проходит точное сопоставление effect и не может ослабить minimum
approval каталога.

Обязательные поля охватывают:

- стабильные identity/revision и legacy source;
- title, summary, severity и component IDs;
- prerequisites и symptoms;
- typed diagnostics с expected evidence;
- allowlisted actions с `approval` и `effect`;
- rollback только через ссылку на объявленное действие;
- evidence success/failure contracts;
- forbidden evidence keys и redaction rules;
- owner/escalation, related alerts и явные monitoring gaps;
- warnings и stop conditions.

Stable runbook IDs соответствуют
`^[a-z][a-z0-9]*(?:[.-][a-z0-9]+)*$`. Problem ID формируется как
`<runbook-id>/<symptom-id>` и всегда указывает ровно на один runbook.

## Репрезентативные инструкции

| Runbook ID | Поверхность | Source | Related alerts / gap |
|---|---|---|---|
| `database.clickhouse-degraded` | database | `docs/runbooks/clickhouse-memory-profiles.md` | `ClickHouseHttpPingDown`, `ClickHouseServiceDegraded` |
| `worker.market-data-live-tail-gap` | worker | `docs/runbooks/market-data-live-tail-repair.md` | 3 market-data repair alerts |
| `auth.openbao-unavailable` | auth/OpenBao | `docs/runbooks/exchange-secret-management.md` | Явный gap `missing_openbao_alert`, передан Stage `20` |
| `execution.provider-state-unknown` | exchange execution | `docs/runbooks/exchange-execution.md` | `LiveExecutionReconciliationPending`, `LiveExecutionUnknownState` |
| `web.api-health-degraded` | Web/API | `docs/runbooks/web-ui-gateway-same-origin.md` | `ApiHealthDown`, `ApiMetricsDown` |

Все canonical narratives в `docs/runbooks/ops/*.yaml` являются английскими;
генератор отклоняет кириллицу в этой части. Русские переводы находятся отдельно
в `docs/runbooks/locales/ru/*.yaml`; exact ID coverage проверяется для каждой
narrative section. Потерянный warning, stop condition, instruction или expected
evidence блокирует generation.

Generated user artifacts:

- `docs/runbooks/generated/ru/*.md` — 5 русских документов, safety warnings
  выводятся до diagnostics/actions;
- `docs/runbooks/runbooks.json` — deterministic runbook, problem, alert и
  legacy-unmigrated index.

## Безопасность и полномочия действий

Каталог Stage `02` допускает только:

- read-only `observe.audit.read`, `observe.config.read`,
  `observe.database.read`, `observe.health.read`, `observe.logs.read`,
  `observe.metrics.query`, `observe.network.probe`;
- `control.service.restart`, `control.worker.pause/resume`,
  `control.execution.pause/resume`, `control.reconciliation.trigger` и
  `control.release.rollback` с зафиксированными effect/minimum approval.

Это описание capabilities, а не authority на исполнение. Generator не вызывает
эти действия. `operator` или `installation_owner` остаётся обязательным runtime
gate будущего control-agent.

Recursive scan блокирует secret-shaped values, private-key markers, bearer/API
token patterns и DSN с password. Schema не имеет поля для raw secret value.
Каждая русская инструкция выводит forbidden evidence keys и redaction rules.
Для unknown exchange state первым warning остаётся запрет blind retry, а resume
требует `installation_owner`.

## Alert и link integrity

`generate_runbooks.py` читает alert rules из:

- `infra/macos/prometheus/rules/`;
- `infra/monitoring/monitoring/prometheus/rules/`.

Проверены 58 уникальных alert IDs. Для 35 alerts с существующим annotation
`runbook` подтверждены path и GitHub-style Markdown anchor. Canonical
`related_alerts` обязаны существовать в этом corpus. Runbook без related alert
допустим только с непустым explicit monitoring gap; это правило выявило и
сохранило отсутствие OpenBao alert, а не замаскировало его.

Alert rules на этом этапе не переписывались. Поэтому runtime alert routing
остаётся совместимым с legacy Markdown, а новый JSON index добавляет canonical
mapping без преждевременной смены operational route. Перевод оставшихся 20
legacy documents и синхронное переключение annotations принадлежат Stages `17`
и `20`.

## Матрица влияния на контракты

| Измерение | Классификация | Причина / совместимость |
|---|---|---|
| Public API и DTO | `none` | Application endpoints и payloads не менялись |
| Persistence | `none` | Schema/data/migrations не менялись |
| Runtime config/defaults | `none` | Service wiring и defaults не менялись |
| Runbook schema/index | `compatible-change` | Добавлен новый `v1`; legacy Markdown сохранён и явно индексирован |
| Alert semantics | `compatible-change` | Rules/annotations не менялись; canonical alert index добавлен |
| Action authority | `compatible-change` | Добавлен fail-closed typed catalog; executor ещё отсутствует |
| Secret/evidence contract | `compatible-change` | Добавлены обязательные redaction и stop rules без runtime data capture |
| Identity/hash/cache keys | `none` | Runtime identities не менялись; появились только stable docs IDs |
| Service calls/external effects | `none` | Generator выполняет локальное чтение и запись generated docs |
| Browser values | `none` | Web UI rendering исключён и не изменён |

Future breaking boundary: когда Stage `18` control-agent начнёт исполнять typed
actions, capability ID, minimum approval, idempotency и audit semantics станут
runtime contract и потребуют versioned rollout. Stage `02` этого не утверждает.

## Проверки

| Проверка | Результат |
|---|---|
| Draft 2020-12 schema self-check | `passed` для canonical и locale schemas |
| `uv run python -m tools.docs.generate_runbooks` | `passed`, 5 Markdown + JSON index |
| `uv run python -m tools.docs.generate_runbooks --check` | `passed`, deterministic/no drift |
| Реальный alert/runbook corpus | 58 unique alerts; 35/35 существующих links/anchors valid |
| Locale coverage | 5/5 runbooks, все narrative IDs и safety warnings покрыты |
| Capability authority | 7 diagnostic + 7 action capabilities allowlisted; weakened approval rejected |
| Secret/arbitrary operation negative fixtures | secret-shaped value и `command` field rejected |
| `uv run ruff check tools/docs/generate_runbooks.py tests/unit/tools/test_generate_runbooks.py` | `passed` |
| `uv run pyright tools/docs/generate_runbooks.py tests/unit/tools/test_generate_runbooks.py` | `0 errors` |
| `uv run pytest -q tests/unit/tools/test_generate_runbooks.py` | `5 passed` |

После добавления отчёта повторно выполнены runbook generation/check,
docs/project-map generation/check и scoped `git diff --check`; итог записан в
ledger.

## Файловый манифест и authority

- Созданы: `schemas/ops/`, 5 canonical YAML, 5 Russian locale YAML, 5 generated
  Russian Markdown, `docs/runbooks/runbooks.json`,
  `tools/docs/generate_runbooks.py`,
  `tests/unit/tools/test_generate_runbooks.py` и этот report.
- Изменены: stage ledger, generated docs index и generated project-map outputs.
- Удалённых файлов нет.
- Existing 25 top-level legacy Markdown и Prometheus rules не изменялись.
- Foreign changes в `.codex/PLANS.md`, supersession docs, platform artifacts и
  mixed generated outputs сохранены.
- Commit, staging, push, deploy, alert mutation и production mutation не
  выполнялись.

## Передача Stage 03

Stage `03` может ссылаться на `ops.roehub.io/v1` и
`io.roehub.release/v1alpha1`. Он обязан использовать только action capability
IDs, не shell strings, а любые будущие config diagnostics должны проходить
secret-redaction contract. Неперенесённые 20 runbooks и OpenBao monitoring gap
не блокируют Stage `03`, но остаются обязательным входом Stages `17` и `20`.
