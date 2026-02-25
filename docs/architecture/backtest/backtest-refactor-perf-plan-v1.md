---
title: План рефакторинга и оптимизации бэктеста (Sync + Jobs)
version: 1
status: draft
owner: backtest
---

# План рефакторинга и оптимизации бэктеста (Sync + Jobs) v1

Этот документ предлагает план рефакторинга, чтобы большие индикаторные гриды бэктеста работали эффективно и безопасно в обоих режимах:

- синхронный режим API (`POST /api/backtests`)
- асинхронный режим через job runner

Ограничения:

- Без функциональных изменений: результаты, таблицы, и особенно ранжирование по `total_return_pct` должны остаться эквивалентными.
- Без миграций БД.
- Стратегия оптимизации Numba-first (в первой фазе без новых native-зависимостей).

## Статус реализации (обновлено 2026-02-25)

Фазы 1-3 и ключевые элементы фазы 4/7 из этого плана реализованы в рабочем коде:

- Stage A/Stage B используют общий `BacktestStagedCoreRunnerV1` для sync и jobs.
- `CloseFillBacktestStagedScorerV1` готовит batched/scoped indicator tensors на run (`prepare_for_grid_context(...)`) и избегает per-variant `IndicatorCompute.compute(...)` в hot path.
- Подготовленные indicator tensors запрашиваются с `GridSpec.layout_preference = Layout.VARIANT_MAJOR`,
  чтобы извлечение series по варианту могло идти view-based в scoring hot path.
- Sync `POST /backtests` работает через async route + `asyncio.to_thread(...)` и кооперативную отмену:
  - по `request.is_disconnected()`;
  - по hard deadline (`BacktestRunControlV1`).
- Hard deadline sync route вынесен в runtime config: `backtest.sync.sync_deadline_seconds`
  (`configs/<env>/backtest.yaml`) и пробрасывается wiring-слоем в router builder.
- Job runner использует тот же shared core scoring path и проверяет cancel/lease на checkpoint-границах стадий.
- Sync guards применяются как half-budget от runtime config:
  - `max_variants_per_compute = floor(full / 2)`;
  - `max_compute_bytes_total = floor(full / 2)`.
- Добавлен runtime CPU knob `backtest.cpu.max_numba_threads`, применяемый через `numba.set_num_threads(...)` в sync и jobs.

Актуальные implementation точки:

- `src/trading/contexts/backtest/application/services/staged_core_runner_v1.py`
- `src/trading/contexts/backtest/application/services/run_control_v1.py`
- `src/trading/contexts/backtest/application/services/close_fill_scorer_v1.py`
- `apps/api/routes/backtests.py`
- `apps/api/wiring/modules/backtest.py`
- `src/trading/contexts/backtest/application/use_cases/run_backtest_job_runner_v1.py`

## Зачем этот рефакторинг

Наблюдаемые проблемы текущего пайплайна:

- Синхронные запросы могут получать 504 на proxy/UI, пока сервер продолжает вычисления (нет поддержки отмены/прерывания).
- При больших гридах растут CPU и память, а нагрузка сохраняется даже после падения запроса.
- Скоринг Stage A и Stage B не использует дизайн grid/tensor compute; Stage A фактически считает варианты по одному.
- Материализация сигналов использует unicode-массивы numpy (например, `dtype='U7'`) и кэширование по каждому варианту, что может взрывать память.
- Поддержание top-K в job runner не оптимально для потоковой обработки (пересортировка списков при каждой вставке).
- Оценка preflight в UI может сильно недооценивать реальный пик памяти, потому что оценивает только тензоры индикаторов и свечи, но не промежуточные буферы исполнения бэктеста.

## Инварианты (нельзя менять)

- Перебор вариантов и их порядок детерминированы.
- Семантика `variant_key` / `indicator_variant_key` остаётся неизменной.
- `total_return_pct` эквивалентен для каждого варианта (в рамках текущей float-семантики), что сохраняет стабильность ранжирования.
- Семантика стадий:
  - Stage A ранжирует базовые варианты при фактически отключённых SL/TP.
  - Stage B расширяет shortlist до полного набора вариантов по risk-параметрам и возвращает top-K.

## Текущие узкие места (конкретно)

Основные hotspots:

1. Покомпонентный (per-variant) расчёт индикаторов в Stage A
   - `BacktestStagedRunnerV1` вызывает `scorer.score_variant(...)` для каждого варианта.
   - `CloseFillBacktestStagedScorerV1` собирает `GridSpec` с явными одиночными значениями и вызывает `IndicatorCompute.compute` для каждого варианта.
   - Это убивает смысл Numba grid compute.

2. Представление сигналов и кэширование
   - Сигналы хранятся как unicode-массивы numpy и кэшируются по variant key в `_signals_cache`.
   - При большом числе вариантов это становится доминирующим потребителем памяти.

3. Не-потоковое управление top-K / shortlist
   - Stage A строит полный список скоренных строк, а затем сортирует.
   - Буфер top-K в job runner пересортировывает список при каждой вставке.

4. Отсутствие поддержки отмены/таймаутов в sync-режиме
   - Когда клиент/proxy получает таймаут, вычисления продолжаются.
   - Нет кооперативного сигнала отмены, который пробрасывается в staged-пайплайн.

## Целевая архитектура

Ключевая идея: вычислять тензоры индикаторов батчами, оценивать сигналы в компактной числовой форме и выполнять скоринг + отбор top-K потоком, не материализуя огромные промежуточные списки.

### 1) Общий core runner

Ввести единый core staged-движок исполнения, который используют оба режима:

- `RunBacktestUseCase` (sync)
- `RunBacktestJobRunner...` (async)

Этот core должен принимать:

- поток базовых вариантов Stage A
- scorer, который умеет считать батч (а не только одиночный вариант)
- cancellation token / deadline
- memory + variant guards

Sync-режим и job-режим должны отличаться только:

- способом генерации / сохранения вариантов
- тем, куда репортится прогресс
- жёсткими runtime-лимитами и поведением при отмене

### 2) Пакетный (batched) расчёт индикаторов

Заменить per-variant вызовы `IndicatorCompute.compute` на пакетные вычисления:

- Stage A: вычислять тензор грида индикатора (или его часть) один раз на батч.
- Оценивать сигналы для всех вариантов внутри батча.
- Считать метрики для всех вариантов батча через векторизованное/пакетное исполнение, где возможно, либо через минимальный цикл с компактными массивами.

Правило выбора размера батча:

- ограничено `max_compute_bytes_total` и backtest-специфическим бюджетом (см. Модель памяти).

### 3) Компактное кодирование сигналов

Заменить unicode-сигналы на целочисленное кодирование, сохраняя функциональную эквивалентность:

- `NEUTRAL = 0`, `LONG = 1`, `SHORT = -1` (например, `int8`)

Хранить сигналы как:

- плотный вектор `np.int8` на вариант, или
- 2D-массив `signals[variants, t]` в variant-major layout для пакетного скоринга

Важно: преобразование в текущий публичный/API формат (если нужно) делать только для top-вариантов.

### 4) Потоковый top-K и shortlist

Использовать heap-based (куча) потоковый отбор вместо сортировки полного списка:

- Stage A: поддерживать min-heap фиксированного размера для кандидатов `preselect`.
- Stage B: поддерживать min-heap фиксированного размера для кандидатов `top_k`.

Детерминированное разрешение равенств:

- Сохранить текущий детерминированный ranking key (первичный: `total_return_pct`, вторичный: стабильная идентичность варианта).
- Реализовать порядок в heap так, чтобы результат совпадал с текущей полной сортировкой.

### 5) Кооперативная отмена и дедлайны

Добавить интерфейс отмены, который можно проверять в длинных циклах:

- `CancelToken.is_cancelled()` и/или `Deadline.check()`

Sync-режим должен использовать:

- отмену из контекста запроса (client disconnect), если доступно
- плюс настраиваемый жёсткий дедлайн / максимум wall time

Job runner должен использовать:

- отмену по lease/heartbeat (останавливаемся, если lease потерян)

### 6) Явные sync-guards (половина текущих)

Sync-запуски должны ограничиваться агрессивнее, чем job-запуски.

Реализовать:

- `sync_max_variants_per_compute = floor(max_variants_per_compute / 2)`
- `sync_max_compute_bytes_total = floor(max_compute_bytes_total / 2)`

Поведение:

- Sync должен применять либо variants guard, либо memory guard (что сработает первым), возвращая понятную ошибку уровня validation.
- Jobs могут сохранять полные бюджеты.

Где прокинуть:

- Sync: `RunBacktestUseCase(... max_variants_per_compute=..., max_compute_bytes_total=...)`
- Jobs: use-case job runner с полными значениями.

### 7) Настраиваемое CPU на пользователя (Numba-first)

Вынести в конфиг настройку, которая контролирует CPU-потребление на пользователя/запуск.

Фаза 1 (Numba-first):

- В начале запуска выставлять число Numba threads в конфигурируемое значение.
- Избегать дополнительных Python thread pools для per-variant скоринга, когда используется batch compute.

Фаза 2 (опционально):

- добавить общий limiter конкурентности между запросами/пользователями

## Модель памяти (контроль реального пика)

Добавить backtest-специфическую оценку памяти, учитывающую:

- свечи (уже оцениваются)
- тензоры индикаторов (уже оцениваются)
- буферы сигналов (int8)
- промежуточные буферы исполнения (positions/equity arrays, trades buffers)

Подход к реализации:

- определить `BacktestBatchBudget`, который вычисляет максимум вариантов на батч для текущего запуска:
  - учитывает `t`, байты тензора индикатора на вариант и per-variant рабочий набор бэктеста
  - возвращает `variants_per_batch`

Staged-пайплайн не должен аллоцировать больше, чем этот per-batch бюджет.

## План реализации (по фазам)

### Фаза 0: Базовая линия + Notebook-тесты (без изменения поведения)

- Добавить notebook-регрессии + perf/memory smoke тесты в `notebooks/`.
- Добавить golden-механизм для порядка `total_return_pct` на небольшом синтетическом датасете.

Файлы:

- `notebooks/01_backtest_regression_smoke.ipynb`
- `notebooks/02_backtest_perf_memory_smoke.ipynb`
- `notebooks/README.md`

### Фаза 1: Потоковый отбор (пока без изменений compute)

- Заменить полную сортировку Stage A на потоковый heap shortlist.
- Заменить полную сортировку Stage B на потоковый heap top-K.
- Заменить repeated sort в top-K буфере job runner на heap-структуру.

Затронутые файлы (вероятно):

- `src/trading/contexts/backtest/application/services/staged_runner_v1.py`
- `src/trading/contexts/backtest/application/services/job_runner_streaming_v1.py`

### Фаза 2: API пакетного скоринга

Добавить интерфейс scorer, который умеет считать батчи:

- `score_variants_batch(stage, candles, selections_batch, risk_params_batch, ...) -> metrics[]`

Сохранить существующий single-variant scorer как обёртку для совместимости.

Затронутые файлы (вероятно):

- `src/trading/contexts/backtest/application/ports/staged_runner.py`
- `src/trading/contexts/backtest/application/services/staged_runner_v1.py`
- `src/trading/contexts/backtest/application/services/close_fill_scorer_v1.py`

### Фаза 2.1: Compact signals в текущем single-variant scorer (perf milestone)

Уточнение к плану: до полного batch scorer вводим компактное кодирование сигналов уже в текущем v1 pipeline.

- Каноническое кодирование сигналов: `NEUTRAL = 0`, `LONG = 1`, `SHORT = -1` (`np.int8`).
- `signals_from_indicators_v1` вычисляет compact-сигналы для hot path без materialize `dtype='U7'`.
- `CloseFillBacktestStagedScorerV1` кэширует только compact-сигналы и использует bounded LRU (лимиты по entries и bytes).
- `BacktestExecutionEngineV1` принимает compact-сигналы напрямую; legacy `LONG|SHORT|NEUTRAL` остаются совместимыми через pre-normalization.
- Тесты обязаны проверять эквивалентность `Total Return [%]` между legacy и compact path.

Затронутые файлы:

- `src/trading/contexts/backtest/application/services/signals_from_indicators_v1.py`
- `src/trading/contexts/backtest/application/services/close_fill_scorer_v1.py`
- `src/trading/contexts/backtest/application/services/execution_engine_v1.py`

### Фаза 3: Пакетный расчёт индикаторов + компактные сигналы

- Вычислять тензоры индикаторов батчами, используя естественную форму грида.
- Оценивать сигналы для всего батча.
- Хранить сигналы в компактной форме (int8) и избегать per-variant unicode-массивов.
- Ограничить кэширование до: (a) reuse внутри батча, или (b) bounded LRU по стабильной идентичности, со строгими лимитами памяти.

Затронутые файлы (вероятно):

- `src/trading/contexts/backtest/application/services/close_fill_scorer_v1.py`
- `src/trading/contexts/backtest/application/services/signals_from_indicators_v1.py`
- `src/trading/contexts/indicators/adapters/outbound/compute_numba/engine.py` (только если для batching нужен новый API; предпочтительно не трогать)

### Фаза 4: Кооперативная отмена

- Добавить `CancelToken` / проверку deadline внутри циклов стадий.
- Привязать sync-отмену к lifecycle запроса.
- Привязать job-отмену к lease/heartbeat.

Затронутые файлы (вероятно):

- `src/trading/contexts/backtest/application/services/staged_runner_v1.py`
- `src/trading/contexts/backtest/application/use_cases/run_backtest.py`
- `src/trading/contexts/backtest/application/use_cases/run_backtest_job_runner_v1.py`

### Фаза 5: Sync-guards (половина бюджетов)

- Убедиться, что sync использует половину от настроенных `max_variants_per_compute` и `max_compute_bytes_total`.
- Обновить сообщения об ошибках так, чтобы было явно: sync ограничен, для больших поисков рекомендуется jobs.

Затронутые файлы (вероятно):

- `apps/api/wiring/modules/backtest.py`
- `src/trading/contexts/backtest/application/use_cases/run_backtest.py`
- `src/trading/contexts/backtest/adapters/outbound/config/backtest_runtime_config.py` (только если добавляем новые knobs)

## Стратегия валидации

Набор notebook-проверок должен включать:

- Determinism: одинаковые входы дают идентичный упорядоченный результат два раза подряд.
- Golden: синтетический датасет совпадает с закоммиченными golden-значениями (variant keys + `total_return_pct`).
- Scale smoke: умеренный грид (например, 500-5,000 вариантов) завершается за разумное время на ноутбуке; фиксируем тайминги.
- Memory smoke: нет неограниченного роста Python-level кэшей во время запуска.

## Документы, которые нужно обновить после реализации

- `docs/architecture/backtest/backtest-grid-builder-staged-runner-guards-v1.md`
- `docs/architecture/backtest/backtest-execution-engine-close-fill-v1.md`
- `docs/architecture/backtest/backtest-job-runner-worker-v1.md`
- `docs/architecture/apps/web/web-backtest-sync-ui-preflight-save-variant-v1.md`
- `docs/architecture/indicators/indicators-grid-builder-estimate-guards-v1.md` (добавить примечание, что оценка только по индикаторам)

## Индекс по файлам (вероятные точки изменений)

- Sync orchestration: `src/trading/contexts/backtest/application/use_cases/run_backtest.py`
- Staged runner core: `src/trading/contexts/backtest/application/services/staged_runner_v1.py`
- Scorer and signal cache: `src/trading/contexts/backtest/application/services/close_fill_scorer_v1.py`
- Signal evaluation: `src/trading/contexts/backtest/application/services/signals_from_indicators_v1.py`
- Job runner streaming: `src/trading/contexts/backtest/application/services/job_runner_streaming_v1.py`
- API wiring/config: `apps/api/wiring/modules/backtest.py`
