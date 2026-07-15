# Этап 16 — шлюз исполнения и безопасность mainnet

## Статус

- Этап: `16`.
- Статус: `accepted`; три последовательных независимых вердикта `Block`
  исправлены, повторные локальные доказательства собраны, итоговый follow-up
  того же reviewer дал `Release after fixes`.
- Дата: `2026-07-13`.
- Режим: `goal_driven`.
- Граница доказательств: `N/A`; одноразовый PostgreSQL `16`, локальный
  `core:exchange-emulator`, синтетические организации и полное выполнение
  greenfield migration lifecycle.
- Исключены: нативный `mainnet`, реальные или тестовые биржевые заявки,
  production state, production credentials, deploy и пользовательское
  включение торговли.
- Следующий разрешённый этап: `17`.

## Результат

`ExecutionIntent` теперь имеет каноническую идентичность, включающую
организацию, владельца, exchange account, инструмент, сторону, тип и размер
заявки, нормализованные ограничения и idempotency identity. Ограничения v1
содержат `time_in_force`, `reduce_only` и необязательный UTC `expires_at`.
Повтор того же ключа с другим payload отклоняется до side effect как
`idempotency_payload_mismatch`.

Новая доменная политика хранит:

- allowlist поставщиков исполнения с точными `provider_id`, версией, видом,
  exchange, capability и SHA-256 revision hash исполняемого адаптера;
- состояние счёта с режимом `research|paper|testnet|mainnet`, редакциями риска
  и счёта, ссылкой на активную версию учётных данных, сроком свежести риска,
  лимитами заявки/дня/account exposure;
- installation/organization/account `kill-switch`, где `active=true` всегда
  означает запрет;
- ограниченное сроком разрешение `mainnet`, выданное текущим `owner` из
  persisted Identity membership после `recent-auth` той же активной session,
  привязанное к организации, владельцу, счёту, exchange, market type,
  risk/account/provider revisions и отдельному audit event;
- неизменяемый append-only audit всех изменений политики, разрешений и
  решений submit guard.

Обычные плагины не могут зарегистрировать order-submit capability. Допустимы
только `core` и `verified` providers. Нативные Binance и Bybit adapters остаются
`testnet`-only; их точные provider identities проверяются перед отправкой.
Стратегия, API и plugin не получают exchange credential и не вызывают submit.

`ExchangeControlCredentialResolver` формирует два безопасных fingerprint без
включения decrypted material: ссылку на текущую credential version и редакцию
материального состояния account. Resolver вызывается повторно непосредственно
перед pre-submit guard и вызовом адаптера; gateway проверяет именно свежий
результат. Смена версии учётных данных, connection state, риска, account
revision или provider revision закрывает ранее выданный допуск. Отсутствие
policy или durable audit также закрывает отправку.

Каждый PostgreSQL guard сначала получает session-level advisory lock по
organization/account, а затем читает policy, Identity authority, audit и risk
snapshot в одной `REPEATABLE READ` transaction. Lock берётся до создания MVCC
snapshot, поэтому параллельные intent одного счёта видят уже принятые
reservation. Accepted pre-submit audit атомарно становится durable reservation
текущего order и продолжает учитываться в дневном объёме и account exposure
после истечения worker claim lease, пока типизированная сверка или terminal
order state не докажет безопасное освобождение. Текущий snapshot также включает свежую account projection,
проверенную exchange configuration и рассчитанный денежный размер заявки.
Изменение policy и соответствующий audit event атомарны; client-supplied role
или recent-auth timestamp не являются источником полномочий.

Перед submit процесс получает атомарный PostgreSQL claim с TTL и audit/approval
references и продлевает его непосредственно перед внешним эффектом.
Конкурирующий worker не может получить второй claim. Любая финальная запись
требует совпадающий `submit_claim_id` и неистёкший lease, поэтому устаревший
worker не может завершить order или записать post-claim guard rejection после
передачи claim. Живой claim после restart оставляет Redis delivery pending и
не вызывает submit. После истечения claim или adapter timeout состояние сначала
сверяется по `client_order_id`:

- найденный provider order фиксируется как `reconciled` без повторной отправки;
- подтверждённое отсутствие фиксируется отдельно и разрешает новую попытку
  только при следующей отдельной доставке;
- ошибка или неопределённый результат сверки остаётся `unknown`/pending и не
  допускает blind retry; lookup имеет типизированный результат
  `found|confirmed_absent|unknown`.

Тайм-аут, `OSError`, любой HTTP error во время submit, включая Binance
`-2013`, любой Bybit `retCode != 0`, невалидный JSON/успешный ответ без
provider order id и иная двусмысленная native-ошибка считаются `unknown`, а
Redis delivery в этом состоянии не подтверждается. Binance `-2013` означает
`confirmed_absent` только для отдельного status lookup, не для POST submit.
Revision hash Binance, Bybit и emulator вычислен
из bytes реально загруженного Python module с namespace поставщика, поэтому
изменение исполняемого кода меняет policy identity.

Две проверки gateway выполняются вокруг claim/private-stream boundary. Если
между ними включается `kill-switch`, submit блокируется, создаются audit,
order event и critical notification, а adapter не вызывается.

## Хранение и миграция

Миграция `0020_execution_gateway_mainnet_safety_v1.sql` является только
greenfield-миграцией и закрывается, если execution tables уже содержат строки.
Она добавляет canonical intent fields, provider allowlist, account safety
state, scoped kill switches, mainnet approvals, immutable gateway audit и
submit-claim/guard/approval references в order ledger.

Database constraints запрещают mainnet order row без guard audit и approval,
кроме явного `guard_rejected`; provider kind ограничен `core|verified`.
`recent_auth_session_id` ссылается на Identity session, а `approved_at` может
отстоять от `recent_auth_at` не более чем на десять минут. Удалять policy и
approval, обновлять audit events или выполнять policy update без нового
совпадающего audit event нельзя; mainnet approval допускает только первый
переход в revoked state. Миграция включена в manifest/bootstrap/readiness как
`execution-gateway-safety-0020`.

## Реальная граница проверки

`tests/fixtures/execution_gateway/runtime_proof.py` поднимает одноразовый
PostgreSQL `16`, применяет полный manifest до `0020`, создаёт свежую
organization/owner/account fixture и использует только
`core:exchange-emulator`. Emulator не открывает сеть и не создаёт provider или
денежных side effects.

Последняя versioned запись сохранена в
[`evidence/16-execution-gateway-runtime-proof.json`](evidence/16-execution-gateway-runtime-proof.json).
Она имеет `schema=io.roehub.execution-gateway-proof/v1`, `status=passed`,
`mainnet_external_effects=false`, `approval_revoked=true`, `41` audit events,
один approval, четырнадцать orders и шесть reconciliation runs.

Доказаны:

- persisted owner approval, session-bound recent-auth, expiry, authority loss
  и revocation;
- canonical intent/idempotency и duplicate submit без второго adapter effect;
- preflight и повторный immediate pre-submit guard;
- повторное получение текущей credential version перед submit;
- свежесть account/risk snapshot и server-derived notional/daily/exposure;
- сериализация двух разных intent одного account для дневного лимита и
  exposure: в каждом race ровно одна reservation принята, вторая закрыта;
- сохранение accepted reservation после успешного emulator submit, падения
  worker до записи результата и истечения claim lease: следующий intent
  закрывается лимитом, а исходный order восстанавливается сверкой без resubmit;
- timeout до и после provider acceptance с обязательной сверкой;
- отдельный replay только после подтверждённого отсутствия;
- kill-switch во время исполнения;
- invalidation при смене risk revision;
- отказ general plugin;
- запрет удаления/неаудированного обновления policy, неизменяемость gateway
  audit и rollback всей policy transaction при duplicate audit event;
- конкурентный PostgreSQL claim, restart и fenced finalization устаревшего
  worker, включая post-claim guard rejection;
- связь adapter revision с bytes загруженного module artifact.

Полный `uv run python -m apps.migrations.verify_storage_runtime` после `0020`
также прошёл: fresh bootstrap, interrupted recovery, idempotent rerun,
persistent-volume restart, external readiness, все прежние organization/auth/
OIDC/research/trading/notification probes и cleanup. Stage `10` trading probe
теперь использует настоящий persisted gateway policy и audit, а не тестовый
обход.

Approved host-local testnet credential scope не обнаружен: среди имён
переменных среды нет Binance/Bybit/testnet credential markers. Поэтому
условный bounded testnet smoke не запускался и не заменялся чтением секретов.
Это не блокирует этап: prompt требует его только при наличии заранее
разрешённых host-local credentials. Production и текущие exchange connections
не читались.

## Проверки качества

- Целевой `pyright` для live execution, exchange execution, migrations и
  runtime proof — `0 errors, 0 warnings`.
- Целевой `ruff` — `passed`.
- Расширенный целевой pytest live execution/exchange execution/migrations —
  `258 passed`.
- Реальный PostgreSQL/emulator proof — `passed` с cleanup.
- Полный storage lifecycle — `passed` с cleanup.
- Полный `uv run ruff check .` — `passed`.
- Полный pytest — `1893 passed`, четыре прежних `httpx` warnings.
- Docs index/project map generation и `--check`, runtime input inventory
  (`146`) и `git diff --check` — `passed`.
- Дополнительный repository-wide `pyright` не является gate этапа и остаётся
  красным на `153` существующих ошибках в `local_artifacts/` и старых tools;
  Stage `16` targeted scope имеет `0 errors, 0 warnings`.
- Первая независимая проверка и два последовательных follow-up дали `Block`;
  итоговый follow-up того же reviewer дал `Release after fixes` и разрешил
  принять Stage `16`.

## Контракты и совместимость

| Поверхность | Классификация | Обоснование |
|---|---|---|
| API, DTO и ports | `breaking-change` | Intent получает constraints/canonical hash; submit проходит через новый policy port и claim. Публичный mainnet endpoint не добавлен. |
| Persistence | `breaking-change` | Greenfield schema получает `0020`, новые policy/audit tables и order references. |
| Config/defaults | `compatible-change` | `mainnet` не добавлен в `roehub.yaml`, env или Compose; production config остаётся `testnet`. Внутренний `emulator` добавлен только как закрытый enum для локального доказательства. |
| Identity/hash | `breaking-change` | Canonical intent, account revision и credential-version reference становятся обязательными submit identities. |
| Service calls | `breaking-change` | `exchange-execution` обязан выполнить persisted preflight, claim, private-stream boundary и повторный pre-submit guard. |
| External effects | `none` | Только disposable PostgreSQL и no-network emulator; exchange submit не выполнялся. |
| Secrets/trust | `compatible-change` | Decrypt остаётся внутри exchange execution; gateway получает только fingerprint ссылки, а plugin execution запрещён. |
| Audit/runbook | `compatible-change` | Добавлены reason codes, immutable events и безопасные diagnostic queries. |
| Browser defaults | `none` | UI и browser-visible mainnet controls не добавлялись; это относится к Stage `19`. |

Основная классификация — `breaking-change` с fail-closed default, ожидаемая для
greenfield v1. Legacy backfill, aliases и dual-read отсутствуют по `A07`.

## Независимая проверка

- Режим: одна cold independent review и последовательные follow-up того же
  reviewer.
- Вердикты: первый `Block`; первый follow-up `Block`; второй follow-up `Block`;
  итоговый follow-up `Release after fixes`.
- Исправлены blockers: current credential re-resolution; единая transaction
  guard snapshot; persisted Identity session/membership authority; типизированная
  сверка и полный ambiguous-failure handling; запрет policy DELETE; текущие
  notional/daily/exposure/account/config проверки; adapter revision digest;
  claim-fenced finalization и restart race proof; атомарные policy/audit writes;
  DB-bound recent-auth freshness; синхронизация plan/report/ledger semantics.
- После первого follow-up дополнительно исправлены: cross-intent race дневного
  лимита/exposure через pre-snapshot account serialization и durable
  reservation; claim-fenced guard rejection; Bybit/HTTP submit ambiguity;
  module-artifact revision binding; duplicate-audit rollback и audited/unaudited
  PostgreSQL UPDATE proof.
- После второго follow-up accepted reservation отделена от срока жизни worker
  claim и сохраняется до typed reconciliation/terminal state; Binance `-2013`
  ограничен подтверждённым отсутствием только в status lookup, а на POST submit
  остаётся `unknown` без Redis ACK. Оба случая добавлены в PostgreSQL/process
  proofs.
- Итоговый follow-up независимо повторил восемь ближайших тестов, подтвердил
  versioned evidence и не нашёл `Blocker` или `High`.
- Остаточные риски до review: production topology Stage `17`, lifecycle/GC
  Stage `18`, browser recent-auth controls Stage `19`, notifications/alerts
  Stage `20`, backup/restore Stage `21` и observability Stage `22`.

## Файлы и ограничения выполнения

Созданы canonical intent/gateway domain и ports, in-memory/PostgreSQL policy
repositories, policy service, emulator, migration `0020`, runtime proof и
focused tests. Обновлены process/order repositories, native adapter identities,
credential resolver, migration lifecycle, архитектурные документы и runbook.

Чужие dirty изменения сохранены. Staging, commit, push, deploy, production
mutation и реальные order submit не выполнялись. Одноразовые Stage `16`
containers удалены; установленные Docker images сохранены для повторяемых
proof.

## Передача Stage 17

Stage `17` разрешён после independent verdict, исправления всех blockers,
повторных gates и перевода этого этапа в `accepted`. Он получает
строгую execution policy и не получает право включать нативный `mainnet` или
создавать реальные заявки.
