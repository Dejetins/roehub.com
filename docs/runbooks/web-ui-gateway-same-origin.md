# Шлюз Web UI с same-origin (WEB-EPIC-02)

Ранбук для локального и серверного запуска same-origin стека `web + api + gateway`.

## Обязательный файл окружения

Используйте тот же шаблон env-файла, что и в деплое:

- `/etc/roehub/roehub.env` на серверах
- локальный эквивалентный путь (пример: `./infra/docker/.env.local`)

Минимальные ключи для UI-профиля:

- `POSTGRES_PASSWORD`
- `IDENTITY_PG_DSN`
- `POSTGRES_DSN`
- `STRATEGY_PG_DSN`
- `WEB_API_BASE_URL`
- `TELEGRAM_BOT_TOKEN`

Референс с плейсхолдерами:

- `infra/docker/.env.example`

## Запуск dev одной командой

```bash
docker compose -f infra/docker/docker-compose.yml \
  --env-file /etc/roehub/roehub.env \
  --profile ui up -d --build
```

Ожидаемый адрес:

- `http://127.0.0.1:8080`

Быстрые проверки:

```bash
docker compose -f infra/docker/docker-compose.yml \
  --env-file /etc/roehub/roehub.env \
  --profile ui ps

curl -i http://127.0.0.1:8080/api/auth/current-user
curl -i http://127.0.0.1:8080/assets/site.css
```

## Поведение bootstrap БД

`db-bootstrap` запускается перед `api` в UI-профиле и выполняет:

1. `python -m apps.migrations.bootstrap_main`
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

Разработка:

1. Пробросьте `127.0.0.1:8080` через туннель (`cloudflared` или `ngrok`).
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

## Примечание по health-маршрутизации

Gateway отрезает префикс `/api` и проксирует запрос в API upstream:

- `/api/<path>` на gateway -> `/<path>` на API.

Если позже в API появится `/health`, через gateway он будет доступен как `/api/health`.
