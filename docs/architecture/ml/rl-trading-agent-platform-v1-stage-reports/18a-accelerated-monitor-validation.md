---
doc: stage-report
stage: 18A-accelerated
status: accepted
language: ru
---

# Ускоренная проверка постоянного монитора Stage 18A

## Итог

Ускоренная изолированная проверка принята: она подтверждает механику trusted DQN `08K` и политики `long_only + hold_1m`, но не заменяет продолжающееся окно `five_ticker_24h` и не является оценкой торгового качества.

Хешированный итоговый артефакт:

`/opt/roehub/state/rl_trading/evaluation_runs/stage18a_accelerated_monitor_validation_v1/stage18a_accelerated_20260711T152018Z/summary.json`

- `status=accepted`;
- `summary_hash=5a28ce83ce716a51102feb179919e6dd1d4a2bda13998cf7497845499ca73e6d`;
- file sha256 `71227f273ef0c82ecaed4df44738751747544cc03ec8d7730a917b8ccdfd5cd4`;
- commit с harness: `e3da67368cd27f0b38d17229389af785dd247ac2`;
- GitHub CI: `29157588359`, успешно.

## Исторический прогон

Полный backtest split Stage `08J` содержит `4162` article-style сессии и `165`
естественных решений `open_long` текущей monitor-policy. Для быстрого
механического доказательства выбраны `100` сессий:

- `20` с естественным `open_long`;
- `80` детерминированных `hold` controls;
- решения модели не изменялись и не принуждались;
- selection является `event_enriched_diagnostic_not_performance_sample`.

Результат:

- `20` открытий и `20` корректных виртуальных закрытий через одну минуту;
- `40` source events с `outcome=no_intent`;
- `0` intents и `0` orders;
- повторный replay добавил `0` событий;
- суммарный виртуальный PnL `-1194.1594759411992`;
- p95 решения модели `0.5992500809952617 ms`;
- p95 записи in-memory source event `0.0373331131413579 ms`.

PnL не используется как quality claim: выборка намеренно обогащена
`open_long`, не является случайной или временной OOS-выборкой и служит только
для проверки открытия, закрытия, учета и идемпотентности.

## Граница закрытия свечи

В отдельные Redis streams было опубликовано `20` финальных сообщений за
`48.769-489.963 ms` до заявленного закрытия свечи. Использованы отдельные:

- stream prefix;
- consumer group;
- consumer name;
- state-файл;
- run id;
- in-memory execution repository.

Все `20` сообщений обработаны только после закрытия свечи. Зафиксировано:

- `close_boundary_retries_total=1`;
- `errors_total=0`;
- `safety_breaches_total=0`;
- Redis pending после обработки `0`;
- повторный replay добавил `0` событий;
- p95 обработки `13.006708002649248 ms`;
- временные Redis streams удалены после фиксации evidence.

## Безопасность и границы вывода

Ускоренный harness не создавал PostgreSQL adapter, exchange adapter или
execution dispatch. Рост `execution.requests.v1`,
`execution.requests.retry.v1` и `execution.requests.dlq.v1` равен `0/0/0`.
Baseline активного `five_ticker_24h` не изменился; его фоновый сборщик продолжает
работать отдельно.

Этот результат:

- подтверждает исправленную механику close-boundary retry;
- подтверждает естественные `08K` решения, минутные закрытия и идемпотентность;
- не принимает `five_ticker_24h`;
- не открывает `twenty_ticker_7d`, Stage `19+`, paper, testnet или mainnet;
- не доказывает прибыльность, устойчивость или продуктовую готовность модели.
