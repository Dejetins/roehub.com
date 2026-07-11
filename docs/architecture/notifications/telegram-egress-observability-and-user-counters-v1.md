# Telegram egress, observability and user counters v1

Дата: `2026-07-11`

## Назначение

Контракт закрепляет production-доступ Roehub к Telegram Bot API через закрытый
HTTPS-прокси и SSH-туннель, не публикуя инфраструктурные адреса, ключи, токены
или Telegram `chat_id` в открытом репозитории. Он также определяет проверку
доступности API и пользовательский счетчик реально отправленных сообщений.

## Границы доверия

| Данные | Источник истины | Допустимое раскрытие |
|---|---|---|
| Адрес шлюза и закрытый SSH-ключ | host-local `~/.ssh/config` на `macstudio` | только наличие SSH-псевдонима `roehub-telegram-egress` |
| Токен Telegram-бота | host-local env | только булево наличие |
| Маскированная ссылка чата | `chat_id_ref` | API и UI могут показывать маскированную форму |
| Текст сообщения и ответ провайдера | delivery ledger и хеши попытки | не выводятся в системный журнал и метрики |

Репозиторий содержит только пример
`infra/macos/ssh/roehub-telegram-egress.conf.example`. Заполненная SSH-конфигурация,
ключи и значения env не являются проектными артефактами.

## Проверка Telegram API

`notification-dispatcher` выполняет `getMe` через ту же переменную прокси и тот
же токен, которые используются адаптером отправки. Проверка не отправляет
сообщение и имеет отдельные интервал и тайм-аут.

Метрики:

- `notifications_telegram_api_up`;
- `notifications_telegram_api_probe_latency_seconds`;
- `notifications_telegram_api_last_success_unixtime`;
- `notifications_telegram_api_probe_total{result="success|failure"}`.

Метрики не содержат пользователя, маршрут, чат, токен, текст или URL с токеном.
Системный журнал содержит только результат, задержку и нормализованный код
ошибки. Алерт `NotificationsTelegramApiDown` срабатывает после двух минут
неуспешных проверок.

## Граница привязки аккаунта

Канонический источник существующей Telegram-привязки остается в bounded context
Identity. Этот контракт не создает вторую таблицу привязок и не включает
незавершенный `telegram_bot_worker`. Полная цепочка
`binding code -> /start -> identity_telegram_channels -> NotificationRoute`
требует отдельного изменения с единым источником истины, SQL-транзакцией,
production worker и end-to-end проверкой.

Аргумент одноразового `/start`-кода не сохраняется в
`notification_telegram_updates.command_args_json`: вместо значения записывается
`<redacted>`.

## Пользовательский счетчик

Источник истины: `notification_deliveries` в соединении с
`notification_routes`. В счетчик входят только строки, для которых:

- маршрут имеет `recipient_kind='user'`;
- `owner_user_id` совпадает с текущей аутентифицированной сессией;
- доставка имеет `status='sent'`;
- `provider_key='telegram_bot_api'`.

Административные маршруты, `log_only`, `fake`, ожидающие, подавленные и
неопределенные доставки не учитываются. API возвращает
`telegram_sent_total`, `telegram_sent_last_24h` и `last_telegram_sent_at` внутри
`/ui/account/notifications/scoped`. Prometheus не публикует счетчики с меткой
пользователя из-за приватности и неограниченной кардинальности.

## Журналирование

Для каждой обработанной доставки системный журнал содержит только
`delivery_id`, `route_id`, `provider`, `status`, `attempt_count` и `error_code`.
Адрес, пользователь, текст сообщения, токен и тело ответа Telegram исключены.
Повторные health probe без смены состояния пишутся только на уровне `DEBUG`.
Файлы dispatcher и egress tunnel ротируются управляемой `launchd`-службой
`com.roehub.notification-log-rotation`.

## Совместимость и развертывание

| Поверхность | Изменение | Классификация |
|---|---|---|
| API | добавлено поле `delivery_counters` | `compatible-change` |
| Порты и адаптеры | добавлены reader счетчика и health probe | `compatible-change` |
| PostgreSQL | добавлен индекс sent-deliveries; binding schema не меняется | `compatible-change` |
| Конфигурация | добавлены параметры healthcheck; production включает только `getMe` | `compatible-change` |
| Отправка | `provider_mode='log_only'` и Telegram provider остаются выключенными | `none` |
| Журналы и метрики | добавлены обезличенные записи, метрики и алерт | `compatible-change` |

Откат: выключить `healthcheck_enabled`, вернуть предыдущий `plist`, отключить
новое отображение счетчика. Delivery ledger при рабочем откате не удаляется.
