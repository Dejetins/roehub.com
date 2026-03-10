# Алгоритм расчета из `tests/notebook_tests/06_backtest_compute.ipynb` (v2, подробная спецификация)

Документ фиксирует алгоритм ноутбука `tests/notebook_tests/06_backtest_compute.ipynb` как переиспользуемую, воспроизводимую спецификацию.
Цель ноутбука: быстро оценить большое число стратегий (пары SMA/EMA сигналов), подобрать лучшую TP/SL ячейку для каждой стратегии и посчитать финальные метрики.

Ключевая идея ускорения: вместо полного «бар-за-баром» re-play на каждой ячейке TP/SL строится компактный trade list, а grid-search делается через монотонные hit-time таблицы и разложение вкладов в difference-буферы (Numba).

---

## 1. Общие принципы

1) Детерминизм + fail-fast.
Алгоритм максимально рано валидирует входы: наличие обязательных OHLC полей, монотонность времени, согласованность размеров, соответствие TP/SL сетки precompute-таблицам, диапазон сигналов `{-1,0,1}`.

2) Разделение «подбор лучшей ячейки» и «метрики».
Сначала выбирается лучшая TP/SL ячейка по критерию максимального итогового equity (return), и только потом для выбранной ячейки считаются метрики (Sharpe/DD/winrate/exposure и т.д.).

3) Сужение пространства кандидатов (staged pipeline).
Полный перебор всех пар SMA/EMA слишком дорог. Поэтому используется каскад:
- E: отбор лучших одиночных SMA и EMA по скору на ретёрнах.
- A/B/D: быстрый попарный proxy-score и фильтр по числу подтверждений (confirmations).
- Полный TP/SL grid-search запускается только для top-k пар.

4) Trade-list вместо эмуляции состояния на каждом баре.
Для каждой пары SMA/EMA список подтверждений на signal timeline «сжимается» в список сделок: `(entry_exec_idx, direction, signal_exit_exec_idx)`.
Повторные подтверждения в той же стороне, пока позиция уже открыта, игнорируются.

5) Двухтаймфреймовость как контракт, но допускаются разные режимы.
Алгоритм концептуально различает:
- signal timeline (где формируются подтверждения стратегии),
- execution timeline (где исполняются входы/выходы и живут hit-time таблицы TP/SL).

Важно: в ветке `USE_PRECOMPUTED_SIGNALS=1` signal timeline фактически совпадает с execution timeline (сигналы берутся из тех же строк `arr`). В fallback ветке (recompute) signal timeline строится как 1h агрегация, а затем маппится в execution через `searchsorted`.

6) Ускорение через Numba и линейную алгебру.
Критичные места реализованы в `@njit` (в т.ч. `parallel=True`) и/или через матричные операции (`@`) по чанкам.

7) Fast grid-search опирается на монотонность hit-time таблиц.
Для фиксированного `start_exec` времена hit-time должны быть неубывающими по оси уровней. Это позволяет для каждого трейда быстро определить границы регионов «выход по сигналу» vs «выход по TP/SL» и применить range-add в diff-буферах.

8) Правила приоритета выхода заданы жестко.
- TP/SL lookup стартует с `entry_exec + 1` (чтобы не было lookahead на бар входа).
- Если TP и SL приходят в один и тот же execution индекс, побеждает SL (консервативно).
- Если signal-exit совпадает по бару с TP/SL, побеждает signal-exit.

---

## 2. Контракт входов/выходов

### 2.1. Входные артефакты

1) `prices_and_signals_5m.npy` (далее `arr`) — `np.ndarray`, shape `(T_exec, C)`.

Минимально требуемые колонки (по именам в sidecar):
- `open_time` (ms, int-like)
- `close_time` (ms, int-like)
- `open`, `high`, `low`, `close` (float-like)
- `volume` (опционально)

2) Sidecar/metadata с именами колонок (tuple/list длины `C`).
В ноутбуке реализован поиск columns файла в `BASE_DIR` по подстрокам `column/columns/cols` и расширениям `.npy/.pkl/.pickle/.json`.

3) Hit-time таблицы (`uint32`, mmap-friendly):
- `*.long_tp.u32.npy`: shape `(n_tp, T_exec)`
- `*.long_sl.u32.npy`: shape `(n_sl, T_exec)`
- `*.short_tp.u32.npy`: shape `(n_tp, T_exec)`
- `*.short_sl.u32.npy`: shape `(n_sl, T_exec)`

Семантика `hit_table[level_i, start_exec]`:
- возвращает `exit_exec_idx` (индекс execution бара), где впервые «сработал» уровень,
- или `T_exec` как sentinel «никогда».

4) TP/SL сетка runtime обязана совпадать с precompute по числу уровней:
- `hit_long_tp.shape[0] == len(tp_grid)`
- `hit_long_sl.shape[0] == len(sl_grid)`

### 2.2. Выход

1) `res: pd.DataFrame` — метрики и служебные поля по каждой выбранной стратегии (пара SMA/EMA).
2) `top: pd.DataFrame` — TOP-N (по умолчанию `TOP_N=50`) после сортировки.

---

## 3. Конфигурация и runtime флаги

### 3.1. Базовые параметры (как в ноутбуке)

```python
signal_tf = "1h"          # intent
exec_tf = "5m"            # intent

fee_rate = 0.0004         # комиссия за сторону
close_on_end = 1          # форс-закрытие в конце, если нет signal exit и TP/SL

# Staged prefilters
top_frac_side = math.sqrt(0.10)
min_confirm = 30
top_frac_pairs = 0.10
time_chunk = 4096
min_nonzero_single = 200

# TP/SL grid (должна совпасть с precompute)
tp_start_pct, tp_stop_pct, tp_step_pct = 4.0, 50.0, 0.5
sl_start_pct, sl_stop_pct, sl_step_pct = 2.0, 25.0, 0.5
```

### 3.2. Переменные окружения

```python
BT_USE_PRECOMPUTED_SIGNALS=1  # 1: использовать сигналы из arr, 0: fallback recompute

BT_GRID_SELF_CHECK=0          # 1: сверять fast kernel vs slow reference
BT_GRID_SELF_CHECK_N=50       # размер self-check подмножества
BT_GRID_BENCH_SLOW=0          # 1: прогонять slow reference на всем наборе (очень долго)
BT_GRID_DIFF_F32=0            # 1: использовать float32 diff буферы в fast kernel
```

---

## 4. Высокоуровневый пайплайн

1) Загрузить `arr` и колонки; извлечь индексы OHLC; найти `sma_cols/ema_cols`.
2) Загрузить hit-time таблицы; проверить dtype/shape; валидировать монотонность по уровням.
3) Сконструировать TP/SL сетку; проверить совпадение с precompute; подготовить fee/log-факторы.
4) Подготовить сигналы (precomputed или fallback recompute).
5) Посчитать returns на signal timeline и выровнять матрицы для scoring.
6) Stage E: выбрать top-fraction SMA и top-fraction EMA по скору.
7) Stage A/B/D: по top SMA/EMA посчитать матрицы confirmations и proxy-score; выбрать top pairs.
8) Смаппить signal bars -> execution entry indices.
9) Посчитать trade_counts и построить compact trade list per strategy.
10) Запустить fast grid-search для выбора лучшей TP/SL ячейки на каждую стратегию.
11) Посчитать метрики для выбранной TP/SL ячейки.
12) Собрать `res`, отсортировать, сделать sanity checks, сформировать `top`.

---

## 5. Подробно по этапам и функциям

### Этап 0. `njit_cached`: безопасный декоратор Numba

Задача: в notebook окружении `cache=True` может падать (нет file locator). Декоратор пробует `cache=True`, и только если ошибка про locator — переключается на `cache=False`.

Пример:

```python
@njit_cached(parallel=True, fastmath=True)
def kernel(...):
    ...
```

Функция: `njit_cached(parallel=False, fastmath=False, inline="never")`.

---

### Этап 1. Загрузка `arr` и имён колонок

1) `arr = np.load(NPY_PATH, mmap_mode="r")` (если `USE_MMAP=True`).
2) Имена колонок загружаются `_load_columns_or_fail(BASE_DIR, expected_len=arr.shape[1])`.
3) Проверяются обязательные колонки OHLC/time.
4) Выделяются `sma_cols` и `ema_cols` по префиксам:

```python
sma_cols = [c for c in cols if isinstance(c, str) and c.startswith("signal|ma.sma|")]
ema_cols = [c for c in cols if isinstance(c, str) and c.startswith("signal|ma.ema|")]
```

Функция: `_load_columns_or_fail(base_dir: str, expected_len: int) -> tuple[str, ...]`.

Поведение:
- если в `globals()` есть `prices_and_signals_np_columns` нужной длины — используется он,
- иначе сканирует `base_dir` и пытается загрузить кандидатов.

---

### Этап 2. Загрузка hit-time таблиц и их валидация

1) Загружаются `hit_long_tp/sl`, `hit_short_tp/sl`.
2) Проверки:
- dtype строго `np.uint32`.
- `hit.shape[1] == T_exec`.
- монотонность по оси уровней: `hit[level+1, :] >= hit[level, :]`.

Зачем нужна монотонность:
- fast kernel использует `lower_bound_ge_hit` и `first_equal_hit`, которые корректны только при монотонной оси уровней.

Код-паттерн проверки:

```python
if hit_table.shape[0] > 1 and not np.all(hit_table[1:, :] >= hit_table[:-1, :]):
    raise RuntimeError("... not monotone ...")
```

---

### Этап 3. Нормализация фактических таймфреймов

Execution cadence детектируется по `open_time`:

- `positive_dt_stats_ms(open_time_ms)` возвращает `(median_dt, min_dt, max_dt, non_pos_count)`.
- `effective_exec_tf = timeframe_label_from_dt_ms(median_dt)`.
- `bars_per_year_exec = bars_per_year_from_dt_ms(median_dt)` используется для annualization в Sharpe.

Почему так:
- входные данные могут иметь «неидеальную» дискретность; для метрик используется медианный положительный шаг.
- intent (`signal_tf`, `exec_tf`) сохраняется только как справочное поле в отчете.

Функции:
- `positive_dt_stats_ms(open_time_ms: np.ndarray) -> tuple[int, int, int, int]`
- `timeframe_label_from_dt_ms(dt_ms: int) -> str`
- `bars_per_year_from_dt_ms(dt_ms: int) -> float`

---

### Этап 4. Построение TP/SL сетки и факторов

1) Строятся сетки в процентах:

```python
tp_grid = pct_grid(tp_start_pct, tp_stop_pct, tp_step_pct)  # float32
sl_grid = pct_grid(sl_start_pct, sl_stop_pct, sl_step_pct)  # float32
```

2) Проверяется совпадение числа уровней с precompute таблицами.

3) Из процентов строятся множители equity:

```python
tp = (tp_grid / 100.0).astype(np.float32)
sl = (sl_grid / 100.0).astype(np.float32)

long_tp_eq = 1.0 + tp
long_sl_eq = 1.0 - sl
short_tp_eq = 1.0 + tp
short_sl_eq = 1.0 - sl
```

Важно:
- TP/SL факторы определены в «equity space», одинаково для long/short (знак направления учитывается hit-time таблицами).

4) Комиссия:

```python
fee_two_sides = (1.0 - fee_rate) * (1.0 - fee_rate)
```

5) Для fast kernel заранее готовятся log-факторы (включая fee):

```python
log_fac_tp_long = np.log(long_tp_eq.astype(np.float64) * fee_two_sides)
log_fac_sl_long = np.log(long_sl_eq.astype(np.float64) * fee_two_sides)
...
log_fee_two_sides = math.log(fee_two_sides)
```

Функция: `pct_grid(start_pct, stop_pct, step_pct) -> np.ndarray`.

---

### Этап 5. Подготовка сигналов (precomputed vs fallback recompute)

#### 5.1. Precomputed ветка (default)

Условие:

```python
use_precomputed_signals = USE_PRECOMPUTED_SIGNALS and signal_cols_available
```

Алгоритм:
1) Вычислить индексы SMA/EMA колонок `sma_col_idx/ema_col_idx`.
2) Оптимизация: если индексы образуют contiguous block, брать срезом `arr[:, start:stop]`.
3) Валидация:
- `isfinite`.
- значения «почти целые»: `np.max(abs(x - rint(x))) <= eps`.
- в диапазоне `[-1, 1]`.
4) Приведение к `int8` и транспонирование:

```python
sma_sig_full = np.ascontiguousarray(np.rint(sma_sig_native).T.astype(np.int8, copy=False))
ema_sig_full = np.ascontiguousarray(np.rint(ema_sig_native).T.astype(np.int8, copy=False))
```

5) Signal timeline в этой ветке совпадает с `arr`:

```python
signal_open_time_ms = open_time_ms_raw
signal_close_time_ms = close_time_ms_raw
close_signal = arr[:, close_idx]
```

Практический смысл:
- вы получаете сигнал на каждом execution баре (даже если он меняется редко).

#### 5.2. Fallback recompute ветка

Применяется, если:
- `BT_USE_PRECOMPUTED_SIGNALS=0`, или
- в `arr` нет SMA/EMA колонок.

Шаги:
1) Подготовить OHLC на signal timeline:
- если execution уже 1h (`is_native_1h_exec`), то использовать raw OHLC,
- иначе агрегировать 5m -> 1h через `aggregate_5m_to_1h`.

2) Построить источники для индикаторов:

```python
source_arrays_1h = build_source_arrays_from_ohlc(ohlc_1h)
```

3) Вытащить `(source, window)` из имен колонок:

```python
sma_sources, sma_windows = parse_signal_column_specs(sma_cols)
ema_sources, ema_windows = parse_signal_column_specs(ema_cols)
```

4) Посчитать матрицы сигналов и собрать их в исходном порядке колонок:

```python
sma_sig_full = build_signal_matrix_for_specs(..., kind="sma")
ema_sig_full = build_signal_matrix_for_specs(..., kind="ema")
```

5) Signal timeline в этой ветке — 1h:

```python
signal_open_time_ms = ohlc_1h["open_time_ms"]
signal_close_time_ms = ohlc_1h["close_time_ms"]
close_signal = ohlc_1h["close"]
```

Функции и их смысл:

1) `parse_signal_column_specs(signal_cols) -> (sources, windows)`
- парсит tokens `source=` и `window=` из имени колонки `signal|ma.<kind>|...|source=...|window=...`.

Пример имени:

```text
signal|ma.sma|v1|source=hlc3|window=20
```

2) `aggregate_5m_to_1h(...) -> dict[str, np.ndarray]`
- детерминированная агрегация на основе `open_time_ms // 1h_ms`.
- `open` берется с первого 5m бара часа, `close` с последнего, `high/low` через reduceat.

3) `build_source_arrays_from_ohlc(ohlc_1h)`
- возвращает источники `open/high/low/close/hlc3/ohlc4`.

4) `compute_sma_signal_matrix(source, windows)` / `compute_ema_signal_matrix(source, windows)`
- возвращают `int8` сигналы `{-1,0,1}`: сравнение `v` и `ma` (SMA/EMA) на каждом баре.

5) `build_signal_matrix_for_specs(...)`
- оптимизирует вычисления: считает матрицу по `unique_windows` на каждый source, затем раскладывает в исходный порядок.

---

### Этап 6. Returns и выравнивание матриц для scoring

После подготовки сигналов (в любой ветке) строятся returns на signal timeline:

```python
ret = (close_signal[1:] / close_signal[:-1]) - 1.0
ret = ret.astype(np.float32, copy=False)
n_int = ret.shape[0]

# для E/A/B scoring используем только бары, которые «покрываются» ret
sma_eval_T = np.ascontiguousarray(sma_sig_full[:, :n_int])
ema_eval_T = np.ascontiguousarray(ema_sig_full[:, :n_int])
```

Замечание:
- переменная в ноутбуке названа `ret_1h`, но фактически это returns на signal timeline (которая может быть 5m в precomputed ветке или 1h в fallback).

---

### Этап 7. E-предфильтр одиночных SMA/EMA

Цель: резко сократить число рядов SMA/EMA до top-fraction по скору.

1) Считается число ненулевых сигналов:

```python
sma_nz = (sma_eval_T != 0).sum(axis=1)
ema_nz = (ema_eval_T != 0).sum(axis=1)
```

2) Отбрасываются ряды с `nz < min_nonzero_single`.

3) Score: chunked dot-product для экономии памяти и лучшего CPU utilization:

```python
sma_score = single_score_chunked(sma_eval_T, ret, chunk=time_chunk)
ema_score = single_score_chunked(ema_eval_T, ret, chunk=time_chunk)
```

4) Score с комиссионным штрафом:

```python
score_adj = score - fee_rate * nz
```

5) Выбираются top `top_frac_side` отдельно для SMA и EMA:

```python
sma_keep = np.sort(topk_fraction_idx(sma_score_adj, top_frac_side))
ema_keep = np.sort(topk_fraction_idx(ema_score_adj, top_frac_side))
```

Функции:
- `single_score_chunked(sig_T_i8, ret_f32, chunk) -> score_f32`
- `topk_fraction_idx(score, frac) -> idx`

---

### Этап 8. A/B/D-предфильтр пар (chunked GEMM)

На этом этапе вычисляется proxy-score по confirmations, чтобы выбрать топовые пары SMA/EMA.

Данные:
- `sma_eval_T`: shape `(n_sma, n_int)`
- `ema_eval_T`: shape `(n_ema, n_int)`

На каждом time chunk:

```python
l_sma = (sma_chunk == 1).astype(np.float32)
l_ema = (ema_chunk == 1).astype(np.float32)
n_long += l_sma @ l_ema.T
proxy_score += l_sma @ (l_ema * r[None, :]).T

s_sma = (sma_chunk == -1).astype(np.float32)
s_ema = (ema_chunk == -1).astype(np.float32)
n_short += s_sma @ s_ema.T
proxy_score -= s_sma @ (s_ema * r[None, :]).T
```

Далее:

```python
n_confirm = n_long + n_short
valid = n_confirm >= min_confirm

proxy_adj = proxy_score - (1.5 * fee_rate) * n_confirm
proxy_adj_masked = np.where(valid, proxy_adj, NEG_INF)
```

Отбор top пар:
- берется доля `top_frac_pairs` от числа valid пар по `proxy_adj_masked`.

Практический смысл:
- `n_confirm` ограничивает стратегии с редкими входами,
- `proxy_score` дает дешевую оценку «насколько подтверждения совпадают с направлением доходностей».

---

### Этап 9. Маппинг signal bars -> execution entry indices

Цель: обеспечить вход на следующем доступном execution баре после закрытия signal бара.

Варианты:

1) Если execution фактически 1h (`is_native_1h_exec`):

```python
sig_entry_exec_idx = np.arange(n_sig_trade, dtype=np.int32) + 1
sig_entry_exec_idx = np.where(sig_entry_exec_idx >= T_exec, T_exec, sig_entry_exec_idx)
```

2) Иначе (обычно 5m exec, 1h signal):

```python
entry_time_ms = signal_close_time_ms + 1
sig_entry_exec_idx = np.searchsorted(exec_open_ms, entry_time_ms, side="left")
sig_entry_exec_idx = np.where(sig_entry_exec_idx >= T_exec, T_exec, sig_entry_exec_idx)
```

Замечание:
- `+1ms` гарантирует «строго после close».

---

### Этап 10. Построение compact trade list

#### 10.1. `build_trade_list_for_pair`

Сигнал подтверждения сделки:
- long: `sma[t] + ema[t] == 2`
- short: `sma[t] + ema[t] == -2`

Сжатие:
- если уже в позиции и подтверждение снова в ту же сторону — игнор.
- если пришло подтверждение в противоположную сторону — закрыть текущую сделку «по сигналу» на `entry_exec_idx` этого подтверждения и открыть новую.

Выход:
- `out_entry_exec_idx[tr]`
- `out_dir[tr]` (`+1` или `-1`)
- `out_sig_exit_exec_idx[tr]` (execution индекс «закрыть по сигналу», либо `T_exec` как sentinel «нет выхода по сигналу»)

Минимальный пример (псевдокод):

```python
current_dir = 0
for t in range(n_sig):
    dirn = +1 if sma[t] + ema[t] == 2 else -1 if sma[t] + ema[t] == -2 else 0
    if dirn == 0:
        continue
    entry_exec = sig_entry_exec_idx[t]
    if entry_exec >= T_exec:
        break

    if current_dir == 0:
        current_dir = dirn
        current_entry = entry_exec
        continue

    if dirn == current_dir:
        continue

    # direction changed => close by signal and open new
    trades.append((current_entry, current_dir, entry_exec))
    current_dir = dirn
    current_entry = entry_exec

if current_dir != 0:
    trades.append((current_entry, current_dir, T_exec))
```

#### 10.2. `count_trades_for_pairs`

Функция нужна для двух вещей:
- оценить `trade_counts[k]` для каждой стратегии,
- в fast kernel аллоцировать буферы `entry_arr/dir_arr/sig_exit_arr` ровно нужного размера (без переполнения).

Важно:
- внутри используется временная аллокация массивов размера `n_sig` на каждый `k`.

---

### Этап 11. Единая логика выхода одной сделки: `evaluate_trade_factor`

Сигнатура (упрощенно):

```python
pf, exit_exec, closed = evaluate_trade_factor(
    dirn, entry_exec, sig_exit_exec, tp_i, sl_i,
    exec_open, last_close, T_exec, close_on_end,
    hit_long_tp, hit_long_sl, hit_short_tp, hit_short_sl,
    long_tp_eq, long_sl_eq, short_tp_eq, short_sl_eq,
)
```

Ключевые правила:

1) Lookup стартует с `lookup_exec = entry_exec + 1`.
2) TP/SL кандидат выбирается по раннему времени (tie-break SL):

```python
if tsl <= ttp:
    tp_sl_exec = tsl
    tp_sl_pf = sl_pf
else:
    tp_sl_exec = ttp
    tp_sl_pf = tp_pf
```

3) Signal-exit выигрывает при равенстве бара:

```python
if sig_exit_exec < T_exec and sig_exit_exec <= tp_sl_exec:
    exit_open = exec_open[sig_exit_exec]
    ...
    return pf, sig_exit_exec, closed=1
```

4) Если TP/SL случился раньше сигнала — возврат по фиксированному фактору уровня:

```python
if tp_sl_exec < T_exec:
    return tp_sl_pf, tp_sl_exec, closed=1
```

5) Если ничего не сработало:
- при `close_on_end=1` закрываем по последнему `close`.
- иначе считаем, что сделка не закрыта (`closed=0`) и фактор `pf=1`.

Long pf на signal/end exit:
- `pf = exit_open / entry_open`.

Short pf на signal/end exit (x1 USDT ROI модель):
- `pf = max(0, 2 - exit_open / entry_open)`.

Комиссия:
- `evaluate_trade_factor` возвращает gross `pf`.
- комиссия применяется снаружи: `fac = fee_two_sides * pf`.

---

### Этап 12. Grid-search: выбор лучшей TP/SL ячейки

#### 12.1. Slow reference kernel (oracle)

`evaluate_best_tp_sl_trade_list_slow`:
- для каждой стратегии строит trade list,
- для каждой ячейки `(tp_i, sl_i)` прогоняет все сделки через `evaluate_trade_factor`,
- копит equity (с fees), берет максимум.

Используется для self-check и регрессионных тестов.

#### 12.2. Fast kernel: монотонное разложение вкладов в log-space

`evaluate_best_tp_sl_trade_list_fast_monotone` и вариант `_f32` делают:

1) Для каждой стратегии выделяются diff-буферы:
- `row_diff`: shape `(n_tp, n_sl + 1)`
- `col_diff`: shape `(n_tp + 1, n_sl)`
- `rect_diff`: shape `(n_tp + 1, n_sl + 1)`

2) Каждый трейд добавляет вклад в log-equity пространство на все TP/SL ячейки разом.
Идея: для фиксированного `start = entry_exec + 1` времена hit-time монотонны по уровню, поэтому границы регионов можно найти бинарным поиском.

3) После всех трейдов diff-буферы превращаются в настоящие вкладовые поля префиксными суммами.

4) Лучшая ячейка — максимальная сумма:

```python
v = row_diff[tp_i, sl_i] + col_diff[tp_i, sl_i] + rect_diff[tp_i, sl_i]
```

5) Чтобы получить «точный» `best_ret`, после нахождения индекса ячейки выполняется короткий replay только в этой ячейке (чтобы избежать артефактов от `NEG_LARGE` для log(0) в short-модели).

##### 12.2.1. Difference-примитивы

1) `add_row_range(row_diff, row_i, col_start, col_stop, value)`
- добавляет `value` на сегмент строки `[col_start:col_stop)` через diff:
  - `row_diff[row_i, col_start] += value`
  - `row_diff[row_i, col_stop] -= value`

2) `add_col_range(col_diff, row_start, row_stop, col_j, value)`
- добавляет `value` на сегмент колонки `[row_start:row_stop)`.

3) `add_rect(rect_diff, row_start, col_start, row_stop, col_stop, value)`
- добавляет `value` на прямоугольник `[row_start:row_stop) x [col_start:col_stop)`.

##### 12.2.2. Разложение по регионам (для одного трейда)

Обозначения:
- `t_sig` — execution индекс выхода по сигналу (`sig_exit_exec`) или `T_exec` (если нет выхода по сигналу).
- `t_tp[i] = hit_tp[i, start]` для TP уровней.
- `t_sl[j] = hit_sl[j, start]` для SL уровней.

Случай A: `sig_exit_exec < T_exec` (есть выход по сигналу)

1) Находим границы уровней, которые «не успевают» до сигнала:

```python
i_sig = lower_bound_ge_hit(hit_tp, start, n_tp, t_sig)  # первый tp с hit >= t_sig
j_sig = lower_bound_ge_hit(hit_sl, start, n_sl, t_sig)  # первый sl с hit >= t_sig
```

2) Регион выхода по сигналу:
- это все ячейки `(i, j)` где `i >= i_sig` и `j >= j_sig`.
- для них вклад одинаковый: `log_fee_two_sides + log(pf_signal)`.

В ноутбуке это:

```python
add_rect(rect_diff, i_sig, j_sig, n_tp, n_sl, contrib)
```

3) Для оставшихся ячеек выход происходит по TP или SL до сигнала.

TP вклад (по строкам):
- для каждого `i < i_sig` найдите первый `j_ptr`, где SL перестает быть раньше/вровень TP:

```python
while j_ptr < j_sig and hit_sl[j_ptr, start] <= hit_tp[i, start]:
    j_ptr += 1
```

- затем TP действует для `j >= j_ptr`:

```python
add_row_range(row_diff, i, j_ptr, n_sl, log_fac_tp[i])
```

SL вклад (по колонкам):
- для каждого `j < j_sig` найдите первый `i_ptr`, где TP уже не раньше SL:

```python
while i_ptr < i_sig and hit_tp[i_ptr, start] < hit_sl[j, start]:
    i_ptr += 1
```

- затем SL действует для `i >= i_ptr`:

```python
add_col_range(col_diff, i_ptr, n_tp, j, log_fac_sl[j])
```

Tie-break SL выигрывает TP/SL tie реализуется неравенствами:
- в TP-цикле используется `<=` для SL,
- в SL-цикле используется `<` для TP.

Случай B: `sig_exit_exec == T_exec` (выхода по сигналу нет)

1) Находим границу «никогда не сработает»:

```python
i_never = first_equal_hit(hit_tp, start, n_tp, T_exec)
j_never = first_equal_hit(hit_sl, start, n_sl, T_exec)
```

2) Если `close_on_end=1`, регион `(i >= i_never, j >= j_never)` закрывается по last_close.
Добавляется прямоугольник с вкладом `log_fee_two_sides + log(pf_end)`.

3) Для `i < i_never` добавляются TP-вклады через `add_row_range`.
4) Для `j < j_never` добавляются SL-вклады через `add_col_range`.

##### 12.2.3. Префиксные суммы

После того как все трейды добавили diff-обновления:
- `row_diff` интегрируется по `j` (внутри каждой строки),
- `col_diff` интегрируется по `i` (внутри каждой колонки),
- `rect_diff` интегрируется 2D префиксом (сначала по строке, затем плюс верхняя строка).

В ноутбуке это реализовано явными двойными циклами (Numba friendly).

---

### Этап 13. Метрики для выбранной ячейки: `metrics_for_best_cell_trade_list`

Функция берет `best_tp_idx[k]` и `best_sl_idx[k]` для каждой стратегии и считает:

- `total_ret = eq - 1`
- `max_dd` по equity curve (peak tracking)
- `sharpe` по сделочным доходностям с annualization
- `trades`, `winrate`, `avg_trade_ret`, `avg_trade_exec_bars`, `exposure`

Расчет Sharpe (как в ноутбуке):

```python
mean_tr = sum_tr / trade_cnt
var_tr = (sum_tr2 / trade_cnt) - mean_tr**2
trades_per_year = trade_cnt / years
sharpe = (mean_tr / sqrt(var_tr)) * sqrt(trades_per_year)
```

где:

```python
years = T_exec / bars_per_year_exec
```

Замечания:
- Sharpe считается по сделкам (не по барам), поэтому annualization использует `trades_per_year`, а не `bars_per_year_exec` напрямую.
- `exposure` считается как доля execution баров, проведенных в позиции: `exposure_bars / T_exec`.

---

### Этап 14. Финальная сборка DataFrame и sanity checks

Сборка:

```python
res = pd.DataFrame({
    "return_pct": out_total * 100,
    "max_dd_pct": out_dd * 100,
    "sharpe": out_sh,
    ...
    "best_tp_pct": tp_grid[best_tp_idx],
    "best_sl_pct": sl_grid[best_sl_idx],
    "confirm_cnt": n_confirm[top_si, top_ej],
    "trade_list_cnt": grid_trade_counts,
    "proxy_adj": proxy_adj[top_si, top_ej],
})
res = res.sort_values(["return_pct", "sharpe"], ascending=[False, False], kind="mergesort")
```

Sanity checks:
- best_tp/sl в диапазонах.
- best_tp/sl кратны шагу:

```python
def aligned_to_step(values, start, step, tol=1e-6):
    scaled = (values - start) / step
    return np.abs(scaled - np.round(scaled)) <= tol
```

- нет NaN в ключевых колонках.

Top-N:
- выбирается `TOP_N=50` и добавляются человекочитаемые `sma_col/ema_col`.

---

## 6. Каталог функций (что обязательно повторить при переносе)

Воспроизводимость алгоритма зависит от сохранения следующих функций/контрактов:

1) Timeframe/валидации:
- `_load_columns_or_fail`
- `positive_dt_stats_ms`, `timeframe_label_from_dt_ms`, `bars_per_year_from_dt_ms`
- `pct_grid`, `aligned_to_step`

2) Сигналы:
- `parse_signal_column_specs`
- `aggregate_5m_to_1h`
- `build_source_arrays_from_ohlc`
- `compute_sma_signal_matrix`, `compute_ema_signal_matrix`
- `build_signal_matrix_for_specs`

3) Ranking/prefilter:
- `single_score_chunked`
- `topk_fraction_idx`

4) Trade list:
- `build_trade_list_for_pair`
- `count_trades_for_pairs`

5) Выход из сделки:
- `evaluate_trade_factor` (включая tie-break и short ROI формулу)

6) Grid-search:
- `evaluate_best_tp_sl_trade_list_slow` (oracle)
- `evaluate_best_tp_sl_trade_list_fast_monotone` (prod)
- `evaluate_best_tp_sl_trade_list_fast_monotone_f32` (опционально)
- `add_row_range`, `add_col_range`, `add_rect`
- `lower_bound_ge_hit`, `first_equal_hit`

7) Метрики:
- `metrics_for_best_cell_trade_list`

---

## 7. Минимальный воспроизводимый скелет (как переносить в код)

Ниже не точная реализация, а структура, которую удобно вынести в production модуль.

```python
from dataclasses import dataclass
import numpy as np


@dataclass
class BacktestComputeCfg:
    fee_rate: float
    close_on_end: int
    top_frac_side: float
    min_nonzero_single: int
    min_confirm: int
    top_frac_pairs: float
    time_chunk: int
    tp_start_pct: float
    tp_stop_pct: float
    tp_step_pct: float
    sl_start_pct: float
    sl_stop_pct: float
    sl_step_pct: float
    use_precomputed_signals: bool


def run_backtest_compute(arr: np.ndarray, cols: tuple[str, ...], hit_tables: dict[str, np.ndarray], cfg: BacktestComputeCfg):
    # 1) validate inputs (columns, shapes, monotonic time)
    # 2) build tp/sl grids; validate against hit_tables
    # 3) prepare signals (precomputed or recompute)
    # 4) compute returns; build eval/trade matrices
    # 5) stage E: pick sma_keep/ema_keep
    # 6) stage A/B/D: select top pairs (top_si/top_ej)
    # 7) map signal bars -> exec entries (sig_entry_exec_idx)
    # 8) count trades and run fast grid-search -> best_tp_idx/best_sl_idx
    # 9) compute metrics at best cell
    # 10) build DataFrame + sanity checks
    raise NotImplementedError
```

---

## 8. Инварианты и «тонкие места», которые нельзя сломать

1) `open_time` (и на execution, и на signal timeline) строго возрастает.

2) Hit-time таблицы монотонны по уровню для каждого `start_exec`:

```python
hit[level+1, start_exec] >= hit[level, start_exec]
```

3) TP/SL сетка runtime совпадает с precompute по числу уровней.

4) Сигналы в precomputed ветке должны быть строго/почти `{-1,0,1}`.

5) Приоритеты выхода фиксированы:
- TP/SL lookup стартует с `entry+1`.
- signal-exit выигрывает при равенстве бара с TP/SL.
- SL выигрывает при равенстве времени TP и SL.

6) Комиссия применяется на каждую закрытую сделку как множитель:

```python
fee_two_sides = (1 - fee_rate)^2
fac = fee_two_sides * pf
```

7) Модель short для signal/end exits:

```python
pf = max(0, 2 - exit_open / entry_open)
```

Это поведение влияет на grid-search и метрики; менять его без пересчета hit-time таблиц нельзя.

8) Fast kernel использует `NEG_LARGE` как замену `log(0)`.
Поэтому обязательно пересчитывать `best_ret` для выбранной ячейки через точный replay (как в ноутбуке), иначе возможны артефакты.

---

## 9. Ссылка на источник

- Источник алгоритма: `tests/notebook_tests/06_backtest_compute.ipynb`.
