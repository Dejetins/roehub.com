# Web UI local same-origin (WEB-EPIC-02)

Статус:

- `gateway` удален из репозитория;
- production same-origin делает `Caddy` на `VPS`;
- local/dev same-origin теперь обеспечивает сам `apps/web` через встроенный `/api/*` proxy.

## Обязательный файл окружения

Используйте тот же шаблон env-файла, что и в деплое:

- `/etc/roehub/roehub.env` на серверах
- локальный эквивалентный путь (пример: `./infra/docker/.env.local`)

Минимальные ключи для UI-профиля:

- `POSTGRES_PASSWORD`
- `WEB_API_BASE_URL`
- `WEB_API_UPSTREAM_URL`
- `TELEGRAM_BOT_TOKEN`

DSN-ключи `IDENTITY_PG_DSN`, `POSTGRES_DSN`, `STRATEGY_PG_DSN` можно не задавать:
`docker-compose.yml` в UI-профиле собирает их в формате conninfo автоматически
из `POSTGRES_DB`, `POSTGRES_USER`, `POSTGRES_PASSWORD`.

Референс с плейсхолдерами:

- `infra/docker/.env.example`

## Запуск dev одной командой

```bash
docker compose -f infra/docker/docker-compose.yml \
  --env-file /etc/roehub/roehub.env \
  --profile ui up -d --build api web db-bootstrap
```

Ожидаемый адрес:

- `http://127.0.0.1:8010`

Быстрые проверки:

```bash
docker compose -f infra/docker/docker-compose.yml \
  --env-file /etc/roehub/roehub.env \
  --profile ui ps

curl -i http://127.0.0.1:8010/api/auth/current-user
curl -i http://127.0.0.1:8010/assets/site.css
```

## Поведение bootstrap БД

`db-bootstrap` запускается перед `api` в UI-профиле и выполняет:

1. `python -m apps.migrations.bootstrap_main`
   - `IDENTITY_PG_DSN`/`POSTGRES_DSN` по умолчанию передаются как conninfo:
     `host=postgres port=5432 dbname=<POSTGRES_DB> user=<POSTGRES_USER> password=<POSTGRES_PASSWORD>`
2. Базовую SQL-миграцию Identity в `IDENTITY_PG_DSN`:
   - применяет `0001_identity_v1.sql`
   - применяет `0002_identity_2fa_totp_v1.sql`
   - применяет `0003_identity_exchange_keys_v1.sql`
3. Защищённая миграция `0004_identity_exchange_keys_v2.sql`:
   - пропускает, если колонки v2 уже существуют
   - применяет только если layout v1 существует и таблица пустая
   - завершает запуск с ошибкой, если в layout v1 уже есть строки (небезопасный путь миграции)
4. Alembic head в `POSTGRES_DSN` через существующий runner:
   - `python -m apps.migrations.main --dsn "$POSTGRES_DSN"`

Сервис одноразовый (`restart: "no"`). Если bootstrap падает, `api` не стартует.

## Домен Telegram Login Widget

Прод:

1. Откройте `@BotFather`.
2. Выполните `/setdomain`.
3. Установите домен `roehub.com`.

Примечание:

- production login widget работает через `VPS` edge на `https://roehub.com`;
- local/dev использует тот же `/api/*` browser contract, но без отдельного gateway-контейнера.

Разработка:

1. Пробросьте `127.0.0.1:8010` через туннель (`cloudflared` или `ngrok`).
2. Установите домен туннеля в `@BotFather /setdomain`.
3. Откройте страницу логина через URL туннеля.

Ограничение:

- у одного бота может быть только один активный домен, поэтому использование production-бота
  для dev-туннеля может сломать login widget в проде
- рекомендация: используйте отдельного staging/dev-бота для локального тестирования через туннель

## Диагностика: "bot domain invalid"

- Убедитесь, что host в браузере точно совпадает с доменом из BotFather (без лишнего поддомена или порта).
- Убедитесь, что страница логина открыта через `https`-URL туннеля.
- Повторите `/setdomain` и подождите до нескольких минут, пока изменения распространятся на стороне Telegram.
- Проверьте, что widget использует ожидаемый username бота.

## Примечание по same-origin маршрутизации

`apps/web` принимает browser-side `/api/*` запросы и проксирует их в upstream API без `/api` префикса:

- `/api/<path>` на web -> `/<path>` на API upstream.

В production эту же семантику на публичном edge реализует `VPS Caddy`.
