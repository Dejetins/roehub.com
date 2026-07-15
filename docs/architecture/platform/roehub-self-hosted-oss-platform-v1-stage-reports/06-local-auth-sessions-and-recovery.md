# Stage 06 — локальная аутентификация, сессии и восстановление

## Статус

- Этап: `06`.
- Статус: `accepted`.
- Дата: `2026-07-13`.
- Режим: `goal_driven`.
- Proof boundary: `N/A`; disposable local API/PostgreSQL/browser, без production
  authentication и внешнего OIDC.
- Blocker: отсутствует.
- Следующий разрешённый этап: `07`.

## Результат

Base profile больше не требует Keycloak. Реализованы passkey-first local auth,
закрытая регистрация, одноразовый `roehubctl` bootstrap, опциональный Argon2id
password fallback, восемь одноразовых recovery codes, opaque server-side
sessions, CSRF, expiry/revocation, `recent-auth` и безопасный аудит. TOTP не
включён: Stage `06` выбрал один опциональный fallback и не оставил частично
настроенной TOTP-поверхности.

Первый bootstrap создаёт локального пользователя, `installation_owner`, первую
организацию и membership `owner` в одной persistence boundary. Существующие
production users, Keycloak subjects, secrets, базы и сессии не читались и не
переносились.

Архитектурный контракт: `docs/architecture/identity/local-auth-sessions-recovery-v1.md`.

## Реализованные границы

- `LocalAuthService` владеет WebAuthn ceremonies, fallback, recovery, rate
  limit, session issuance и fail-closed rotation.
- In-memory/PostgreSQL adapters реализуют одноразовые ticket/challenge/recovery,
  passkey counter, auth events и lockout.
- `local-auth-0012` добавлен в единый Stage `04` migration lifecycle.
- `GET /auth/current-user` отделён от OIDC и работает при local-only base.
- `/auth/local/*` реализует bootstrap, passkey, password, recovery, passkey
  management, `recent-auth` и logout.
- `roehubctl local-auth-bootstrap --output-file ...` создаёт эксклюзивный файл
  mode `0600`, не печатая значение.
- Browser загружает bootstrap-файл вместо ручного ввода значения.
- `/login` ставит passkey первым; `/register` сообщает, что регистрация закрыта.
- Production config fail-fast требует explicit HTTPS origin/RP; dev использует
  допустимый WebAuthn RP ID `localhost`.

## Persistence evidence

Phase manifest:

- `local-auth-0012` SHA-256:
  `80655fe744ec74df03816d7a7f74ae8ec5e910b5f0c2569d88d2c13a626f2a44`;
- `0012_identity_local_auth_v1.sql` SHA-256:
  `89710347fdc39ed2cf7075c11336319ac1b74f81a1395d0e2bccdc309c5650e4`.

Финальный `uv run python -m apps.migrations.verify_storage_runtime` на реальных
PostgreSQL `16.14`, ClickHouse `24.8.14.39` и Redis `7.2.14` вернул:

- `fresh_bootstrap=passed`;
- `idempotent_rerun=passed`;
- `interrupted_recovery=passed`;
- `persistent_volume_restart=passed`;
- `external_readiness=passed`;
- `local_auth.bootstrap_hash_only=passed`;
- `local_auth.single_active_bootstrap=passed`;
- `local_auth.passkey_counter=passed`;
- `local_auth.recovery_replay=rejected`;
- `local_auth.rate_limit=passed`;
- `local_auth.audit_immutable=passed`;
- Stage `05` `organization_isolation=passed` и все семь organization constraints
  остались зелёными;
- `cleanup=passed`.

CLI в контейнере получил DSN только через env reference, записал ticket в
одноразовый mounted file и не вывел значение. Raw credentials, UUID и DSN из
proof не извлекались.

## Реальная браузерная проверка

Режим: реальный headless Chromium через pinned Playwright CLI, виртуальный
CTAP2 internal authenticator с resident key и user verification. Disposable
origin — `http://localhost:8000`; production authentication не затрагивалась.

Проверено:

1. Bootstrap file имел mode `0600`, загружался через file input и не попадал в
   команды, snapshots, screenshots или отчёт.
2. Owner/passkey bootstrap завершился; browser получил ровно восемь recovery
   codes один раз. До следующего snapshot codes были удалены из DOM.
3. POST logout без double-submit CSRF вернул `403`; с CSRF — `204`.
4. Опциональный password fallback вернул `200`, а `/auth/current-user` — `200`.
5. Recovery code вернул `200`, повтор того же code — generic `401`.
6. WebAuthn `recent-auth` вернул `200`; HttpOnly session cookie изменилась без
   публикации значения.
7. Детерминированно истёкшая server-side session получила `401`.
8. Основная кнопка passkey выполнила options/complete `200` и привела на
   `/dashboard` с `/auth/current-user=200`.
9. `/register` подтвердил `Registration is closed`.
10. Desktop `1440x900` и mobile `390x844` визуально проверены; fallback fields
    пусты, overlap/cutoff нет.

Sanitized visual artifacts находятся в ignored output:

- `output/playwright/stage06-local-auth/login-desktop.png`;
- `output/playwright/stage06-local-auth/login-mobile-fallbacks-full.png`.

На auth proof console были только два ожидаемых negative-test resource signals:
`403` для отсутствующего CSRF и `401` для recovery replay. После запуска чистой
visual session `/register` дал `0` errors / `0` warnings. Dashboard business API
не входил в identity-only fixture; его `404` не использовался как Stage `06`
доказательство.

## Проверки

- `uv run pytest -q tests/unit/apps/api tests/unit/apps/web tests/unit/apps/migrations tests/unit/contexts/identity tests/unit/platform/config tests/unit/tools/test_runtime_input_inventory.py tests/unit/tools/test_generate_installation_config.py` — `386 passed`, четыре существующих `httpx` deprecation warnings;
- focused local-auth rotation regression — `4 passed`;
- scoped `ruff` для Stage `06` — `passed`;
- scoped `pyright` для Stage `06` — `0 errors, 0 warnings`;
- полный `uv run pyright` — не является зелёным baseline: 149 ошибок и два
  warning в foreign `local_artifacts/rl_trading/...`; Stage `06` эти файлы не
  менял, scoped gate прошёл;
- runtime input inventory generation/check — `passed`, `126` имён без значений;
- Docker storage/local-auth runtime proof — `passed`;
- docs index generation/check — `passed`;
- project-map generation/check — `passed`, пять generated artifacts;
- `git diff --check` — `passed` после journal update.

## Контракты и совместимость

| Поверхность | Классификация | Обоснование |
|---|---|---|
| Local-auth API/DTO | `compatible-change` | Добавлены `/auth/local/*`; OIDC routes не удалены. |
| Base identity semantics | `breaking-change` | Local passkey становится базовым способом вместо обязательного Keycloak. |
| Application ports | `compatible-change` | Добавлены local-auth и user/session lifecycle операции. |
| PostgreSQL persistence | `breaking-change` | Greenfield lifecycle получает phase `local-auth-0012`. |
| Runtime config | `breaking-change` | Добавлены три explicit production RP/origin input; dev RP изменён на `localhost`. |
| Session/cookie/CSRF | `breaking-change` | Opaque server sessions, rotation, revocation, expiry и double-submit обязательны. |
| Organization/RBAC | `compatible-change` | Bootstrap реализует Stage `05` owner semantics без обхода membership. |
| Request hash/cache/resource identity | `none` | Product resource namespace не менялся. |
| Межсервисные вызовы | `compatible-change` | OIDC остаётся optional; local base не вызывает provider. |
| Внешние эффекты | `none` | Email, provider, exchange, production mutation отсутствуют. |
| Audit | `compatible-change` | Добавлены append-only redacted auth events. |
| Browser defaults | `breaking-change` | Passkey first, closed registration, file-based bootstrap. |

Основная классификация Stage `06` — `breaking-change`, запланированная только
для новой greenfield installation. Legacy identity/session migration отсутствует.

## Холодная проверка

- Режим: `cold self-review fallback`; независимое делегирование в этом запуске
  не разрешено.
- Первоначальный вердикт: `Release after fixes`.
- Исправлено:
  1. provider-neutral `/auth/current-user` отделён от условного OIDC router;
  2. bootstrap browser UX переведён с ручного ввода на file upload;
  3. недопустимый WebAuthn RP ID `127.0.0.1` заменён на `localhost`;
  4. `recent-auth` компенсирующе отзывает новую сессию, если отзыв старой не
     подтверждён;
  5. browser evidence не сохраняет recovery/password/cookie values и снимается
     только после очистки.
- Повторная проверка: `4 passed`, `386 passed`, scoped ruff/pyright, реальный
  PostgreSQL proof и browser flow завершились успешно.
- Итоговый вердикт: `Release` для Stage `06`.
- Остаточные риски: OIDC linking/degradation относится к Stage `07`, OpenBao
  secret references — к Stage `08`, invitation/admin browser UX — к Stage `19`;
  TOTP намеренно не выбран как второй optional fallback.

## Файлы этапа

Созданы:

- local-auth application port/use case и in-memory/PostgreSQL adapters;
- local/current-user API routes;
- `apps/cli/commands/local_auth_bootstrap.py`;
- `apps/migrations/local_auth_runtime_probe.py`;
- `migrations/postgres/0012_identity_local_auth_v1.sql`;
- Stage `06` unit/migration/browser fixture;
- `tools/qa/local_auth_browser_app.py`;
- архитектурный документ и этот отчёт.

Изменены:

- identity user/session ports, repositories и exports;
- API identity routes/wiring;
- CLI entrypoint и `roehubctl` project script;
- migration manifest/lifecycle/verifier/Dockerfile;
- web login/register templates, auth JavaScript, CSS и RU/EN locale;
- `pyproject.toml`, `uv.lock`;
- runtime input inventory;
- Stage `06` tests, журнал и generated docs/project-map outputs.

Удалённых файлов нет. Foreign `.codex/PLANS.md`, supersession docs,
`.github/workflows/ci.yml`, license/governance artifacts, unrelated
`local_artifacts` и смешанные generated hunks сохранены. Commit, push, deploy и
production mutation не выполнялись.

## Передача Stage 07

Stage `07` должен добавить универсальный OIDC provider contract, linking к
существующему local user, timeout/degradation/revocation и real browser proof,
не превращая provider availability в условие входа локального владельца.
Local passkey/recovery остаются независимым fail-safe способом.
