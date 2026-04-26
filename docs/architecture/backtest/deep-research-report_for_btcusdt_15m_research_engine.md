# Техническое исследование прототипа artifact-backed backtest engine BTCUSDT 15m

## Executive Summary

Исследование основано на статическом разборе приложенного notebook и на предоставленных тобой benchmark-замерах с `macstudio`. Главный вывод такой: **для текущего 2-indicator workload “быстрых 10x” без изменения алгоритма уже не осталось**. В notebook compute-core уже достаточно неплохо организован: hot path работает на `ndarray` + `Numba`, Python-heap/top-k занимает считанные десятки миллисекунд, а не секунды. Поэтому дальнейшее ускорение будет приходить не из “косметики”, а из трех конкретных направлений: **снижение лишнего движения данных**, **правильный выбор exact/proxy path**, и **замена полного O(K·T)-сканирования на событийно-компрессированное exact-ядро для будущих 3–5 индикаторов**.

Самые сильные уже подтвержденные улучшения в вашем коде — это не микро-оптимизации, а именно **структурные изменения**. Первое: переход от boolean-mask extraction к contiguous slice в `extract_signal_rows()` дал радикальный выигрыш; это полностью согласуется с моделью NumPy, где basic slicing дает view, а integer/boolean advanced indexing всегда создает copy. Второе: замена старого exact path `count_trades -> build_trade_list -> score_trade_list` на односкановый `evaluate_no_risk_streaming_two()` убирает целый проход по сигналам и устраняет per-combo временные массивы. Эти изменения реальны, а не косметические. citeturn0search2turn0search3turn0search5

Если говорить о том, что делать **сразу**, то я бы выделил пять приоритетов. Во-первых, **убрать лишнюю материализацию 1m-данных**: сейчас notebook создает `float64`-копии всех пяти полей `price_ohlcv_1m`, хотя в exact scoring используются только `open` и `close`, а исходный artifact и так `float32`. Во-вторых, **перестать безусловно строить `eval_T`**, когда combo-prefilter фактически отключен. В-третьих, **закрепить streaming exact scorer как production exact path** и оставить старый двухпроходный путь только как parity reference. В-четвертых, **использовать GEMM-based combo proxy только тогда, когда proxy-stage реально включен**, потому что матричный вариант в ваших замерах уже быстрее, но берет заметно больше временной памяти и должен жить под контролем thread settings. В-пятых, **для будущих 3+ индикаторов переходить на event-compressed/hybrid engine**, иначе декартово произведение кандидатов сделает current architecture не медленной, а просто непригодной. NumPy `dot`/`matmul` действительно может использовать optimized BLAS, а BLAS-бэкенды часто многопоточны, поэтому смешение BLAS и Numba требует явного контроля потоков. citeturn1search0turn2search0turn2search11

Ожидаемый эффект по порядку важности выглядит так. Если где-то вне notebook все еще жив старый `count+trade_list` exact path, то перевод его на streaming path — **высокий выигрыш**, потому что он атакует примерно `0.637s + 0.775s` из вашего старого baseline. Удаление лишних 1m-копий и lazy-materialization `eval_T` даст скорее **низкий или средний выигрыш по wall-clock**, но **хороший выигрыш по RSS и cache pressure**. Перевод combo proxy на матричную форму дает **средний или высокий stage-level выигрыш**, если proxy-stage реально активен. А вот переход к change-point/event-compressed exact scorer — это уже **гипотеза с самым большим upside**, потенциально 3x–20x на exact stage, но только если среднее число реальных переключений сигнала на строку существенно меньше длины ряда. Это надо доказать измерением change density, а не предположением.

Краткий управленческий вывод: **Rust/C++ сейчас не первый ход**. Сначала нужно добрать дешевое и безопасное: убрать лишние копии, формализовать streaming path, правильно профилировать, и проверить event-compressed exact model на реальных строках DEMA/HMA. Если после этого warm-run все еще не удовлетворяет target, тогда уже будет смысл обсуждать compiled extension. Это соответствует и рекомендациям Numba: сначала профилировать на реальных данных и только потом выбирать технику оптимизации; `fastmath` стоит оставлять только там, где оно проходит строгий parity fixture. citeturn0search5turn2search12

## Архитектура текущего алгоритма

Текущий prototype — это **artifact-backed search/evaluation layer**, а не “полный backtest engine с нуля”. Upstream-слой уже сделал тяжелую работу: построил артефакты цен, time-mapping и full signal matrices для `ma.dema` и `ma.hma`. Notebook дальше решает более узкую задачу: **выбрать подмножество строк сигналов, сделать предварительный pruning, прогнать exact no-risk simulation по combinations и вернуть global top-k**.

Фактический стек простой и правильный для такого workload: **Python 3.12, NumPy 2.0, Numba 0.60, memmap-backed `.npy` artifacts**. Это важно: в горячем compute-path здесь нет pandas, groupby, merge, resample и прочих типичных “убийц” throughput. Для текущего слоя это плюс: ядро уже работает на массивах, и потому большая часть оставшегося performance work — это не “перевести pandas в NumPy”, а **сделать текущие массивные сканы более точными по data movement и по algorithmic shape**.

Архитектурно notebook делится на четыре слоя. Первый слой — **artifact loading and run slicing**: манифесты, memmap цен, memmap сигналов, выбор активного интервала `2020-01-11 -> 2026-04-11`, построение `sig_entry_exec_idx_15m` и `T_exec_limit_1m`. Второй слой — **indicator pool preparation**: `row_ids_for_sources()`, `extract_signal_rows()`, `single_score_chunked()`, `prefilter_indicator_rows()`, `prepare_indicator_pool()`. Третий слой — **combo search**: генерация комбинаций, optional combo proxy pruning, self-check. Четвертый слой — **exact no-risk scoring**: либо старый двухпроходный путь через `count_trades_for_two_combos()` + `evaluate_no_risk_trade_list_fast_two()`, либо новый односкановый `evaluate_no_risk_streaming_two()`.

С инженерной точки зрения **computational core** находится не “везде”, а в очень узком наборе функций. Для current notebook это прежде всего `extract_signal_rows()`, `single_score_chunked()`, `build_combo_proxy_cache_two()` или `proxy_prefilter_combos_chunk_two()`, и особенно `evaluate_no_risk_streaming_two()`. Все остальное — orchestration, metadata packaging, top-k maintenance и self-check harness. Это согласуется и с вашими baseline-цифрами: Python heap/top-k almost free, а время сидит в data prep, proxy и exact kernels.

Есть один важный архитектурный нюанс, который надо зафиксировать явно. **Семантика текущего “exact” path уже не глобально exact относительно всех 588×588 requested rows**, потому что до exact evaluation существует heuristic Stage A pruning на уровне отдельного индикатора. То есть точность `trade list / trade count / top-k` уже должна проверяться относительно **текущей конфигурации search space**, а не относительно полного декартова произведения без pruning. Это не проблема само по себе, но это нужно держать в голове, когда команда будет обсуждать “exactness”, future scaling и сравнение старых/новых движков.

## Карта выполнения и compute-core

Pipeline текущего notebook выглядит так.

Сначала загружаются manifests и memmap-артефакты: 15m price arrays, 1m price arrays, mapping 15m→1m и две full signal matrices `(1176, 303995)` для `ma.dema` и `ma.hma`. Затем строится активный временной диапазон, который у вас оказался contiguous по 15m, что и делает possible быстрый `slice` path. После этого формируется `sig_entry_exec_idx_15m`: на каждом 15m signal bar entry выполняется по `open` следующего 1m бара. Далее на основе `row_pools` выбираются нужные блоки строк по источникам `close/high/hlc3`, сигнал-строки копируются в bounded contiguous matrices, выполняется single-indicator prefilter, и только потом запускается combo/evaluation layer.

Следующий шаг — combo stage. Для 2-indicator case это еще терпимо: после row-prefilter остается `142 × 142 = 20,164` exact candidate combos. Но уже здесь видна future problem: orchestration организована как явный Cartesian product по filtered rows, chunked по `256`. Для 2 indicators это нормально, для 3 indicators это уже `142^3 = 2,863,288`, а для 4 — `406,586,896`. На этом месте архитектура перестает быть “медленной” и становится **комбинаторно несостоятельной**.

После combo stage идет exact evaluation. В текущем notebook hot path — `evaluate_no_risk_streaming_two()`. Он по сути совмещает в одном kernel три вещи: построение trade-state transitions, no-risk accounting и расчет агрегатных метрик. Это правильное направление: exact simulation здесь stateful, ветвящаяся, чувствительная к порядку закрытий позиций, fees, slippage, optional profit lock и final close policy. Такой код **не стоит насильно переводить во “fully vectorized pandas-style formulation”**, потому что цена за иллюзию vectorization будет либо рост семантического риска, либо еще большее давление на память.

Ниже — карта ключевых compute-методов.

| Метод / функция | Назначение | Частота вызова | Потенциальная проблема | Тип bottleneck | Сложность | Потенциал ускорения | Риск изменения |
|---|---|---:|---|---|---|---|---|
| `extract_signal_rows` | Копирует bounded rows из memmap signal matrix в contiguous RAM | 1 раз на индикатор | copy-heavy, зависит от indexing path | memory / I/O / copy | O(R·T) | Средний | Низкий |
| `single_score_chunked` | Прокси-скор single-indicator rows | 1 раз на индикатор | chunk cast `int8 -> float32`, повторные проходы | memory-bound / temp alloc | O(R·T) | Средний | Низкий |
| `prefilter_indicator_rows` | nonzero-count + proxy + top-fraction prune | 1 раз на индикатор | строит `eval_T` всегда, даже когда не нужен | copy / memory | O(R·T) | Средний | Низкий |
| `build_trade_list_for_two_rows` | Build compact trade list для одной пары строк | per combo в old path | полный скан `n_sig`, per-combo arrays | CPU-bound | O(T) | Средний | Низкий |
| `count_trades_for_two_combos` | Count-only pass для chunk combos | per combo chunk в old path | лишний полный проход перед exact scoring | CPU-bound | O(K·T) | Высокий | Низкий |
| `score_trade_list_no_risk` | Exact scoring одной trade list | per combo в old path | sequential accounting, allocations уже вне функции | CPU-bound | O(trades) | Низкий/средний | Средний |
| `evaluate_no_risk_trade_list_fast_two` | Старый exact path: allocate -> build list -> score | per combo chunk | two-pass logic + per-combo temp arrays | CPU + alloc | O(K·T + Σtrades) | Высокий | Низкий |
| `apply_no_risk_trade_to_state` | Обновляет агрегированные метрики по одному trade | per closed trade в streaming path | tight inner helper; potentially inline-sensitive | CPU-bound | O(1) per trade | Низкий | Средний |
| `evaluate_no_risk_streaming_two` | Текущий exact core | per combo chunk | все еще full scan по всем 15m барам | CPU-bound / algorithmic | O(K·T) | Высокий | Средний |
| `proxy_prefilter_combos_chunk_two` | Numba combo proxy по chunk | только если proxy-stage активен | full scan по интервалам для каждой combo | CPU-bound | O(K·T) | Средний | Низкий |
| `build_combo_proxy_cache_two` | GEMM-based proxy/confirm cache | только если proxy-stage активен | большие float32 temporaries, BLAS threads | CPU + memory | O(Rd·Rh·T) | Средний/высокий | Средний |
| `iter_combo_chunks_two` | Python chunk generator для Cartesian product | per run | Python orchestration scale problem for 3+ indicators | Python overhead / combinatorial | O(K) generation | Низкий сейчас, критический в future | Низкий |
| `search_topk_two_indicator_no_risk` | Главный orchestration method | per run | смешивает orchestration и policy branches | orchestration | O(total pipeline) | Низкий/средний | Низкий |

Есть одна деталь, которую не нужно “оптимизировать”, пока не доказано обратное. `topk_fraction_idx()` уже использует `np.argpartition`, а это именно тот примитив, который и должен стоять в partial top-k selection вместо полного sort. У `argpartition` порядок внутри партиций не определен, поэтому ваша дополнительная детерминизация через tuple-based heap вполне оправдана. Время на этот участок уже микроскопическое; туда не стоит тратить инженерные часы. citeturn1search1

## Узкие места и алгоритмическая сложность

Первое узкое место — **выборка и материализация signal rows**. Здесь самый важный факт уже подтвержден вашими замерами: `extract_signal_rows()` становится дорогим не потому, что memmap “медленный”, а потому что indexing path меняет характер доступа к данным. Когда используется boolean mask/advanced indexing, NumPy гарантированно переходит к copy semantics; когда используется contiguous slice, он может работать как view-like slicing по базовому буферу. Именно поэтому ваш выигрыш с `~0.406s` до `~0.0066–0.0077s` выглядит правдоподобно и ожидаемо, а не “случайно удачно померенным”. citeturn0search2turn0search3

Второе узкое место — **лишняя материализация 1m-данных**. По коду видно, что notebook строит `price_fields_1m` как пять отдельных `float64` массивов из исходного `ohlcv.f32.npy`, хотя в exact scoring реально используются только `open` и `close`. Это не главный источник wall-clock в уже warmed-up search loop, но это плохой data hygiene: лишний memory footprint, лишняя нагрузка на cache hierarchy и бессмысленное дублирование данных, которые уже и так были `float32`. Для текущего workload это скорее **memory bottleneck с косвенным влиянием на CPU**, чем прямой time bottleneck. Здесь не нужно спорить с математикой: если источник `float32`, то перекладывание всей колонки в `float64` не создает новой рыночной информации, а только расходует память.

Третье узкое место — **старый двухпроходный exact path**. Если смотреть на старый baseline, там `count trades` и `exact scoring` вместе съедали примерно `1.412s` из `3.012s`. Это главный признак того, что старый exact path был не просто “неидеален”, а с алгоритмической точки зрения дублировал значимый объем работы. `count_trades_for_two_combos()` и последующий `build_trade_list_for_two_rows()` оба сканируют один и тот же временной ряд по тем же комбинациям. Новый `evaluate_no_risk_streaming_two()` правильно атакует именно эту проблему: он не убирает statefulness exact simulation, но убирает **повторный проход** и **пер-комбо временные массивы**.

Четвертое узкое место — **combo proxy stage**, но здесь картина зависит от режима. В текущей notebook-конфигурации combo prefilter фактически выключен, потому что `COMBO_PREFILTER_TOP_FRAC=1.0` и `COMBO_MIN_CONFIRM=1`, а значит proxy-stage сейчас не должен быть в hot path. Но как только он включается, старый `proxy_prefilter_combos_chunk_two()` имеет ту же фундаментальную форму, что и exact scorer: O(K·T). Ваш GEMM-based prototype, напротив, перекладывает работу в матричные умножения, а NumPy прямо указывает, что `dot`/`matmul` используют optimized BLAS, когда это возможно. Именно этим и объясняется, почему матричный вариант у вас уже заметно быстрее. Но внимание: BLAS-библиотеки часто многопоточны, а thread counts там надо регулировать отдельно. citeturn1search0turn2search0

Пятое и главное узкое место — **полное сканирование всех 15m-баров для каждой candidate combo**. Для текущего случая это еще проходит: `20,164 × 218,913 ≈ 4.41 млрд` bar-decisions. Но уже для трех индикаторов при тех же `142` filtered rows на индикатор получаем `2,863,288` combos и порядка `6.27e11` bar-decisions. Для четырех — уже `8.90e13`. Это не вопрос “нужно чуть-чуть ускорить Numba”, это вопрос **изменить форму алгоритма**. Именно здесь проходит граница между “оптимизацией” и “архитектурой”.

Шестое узкое место — **memory shape combo proxy cache**. `build_combo_proxy_cache_two()` на filtered pools `142 × 218,912` вынужденно создает большие float32 temporaries: `dema_pos`, `dema_neg`, `hma_pos`, `hma_neg`. Это примерно `474 MiB` только на четыре основных mask-массива, без учета промежуточных выражений вроде `(hma_pos * ret)` и внутренних BLAS-temporaries. При текущем `~1.31 GiB` RSS это еще укладывается, но цена ускорения здесь платится именно памятью. Такой обмен “memory for speed” на Mac Studio уместен, пока proxy-stage действительно экономит exact search time. Но если combo prefilter отключен, платить эту цену просто не за что.

С точки зрения Big O текущий notebook выглядит так:

- `extract_signal_rows`: **O(R_req · T_run)** по времени и памяти на bounded copy.
- `single_score_chunked`: **O(R_req · T_int)** по времени, память — `O(R_req)` плюс chunk temp.
- `prefilter_indicator_rows`: фактически **две полных проходки по матрице**: nonzero-count и proxy-score.
- `proxy_prefilter_combos_chunk_two`: **O(K_chunk · T_int)**.
- `build_combo_proxy_cache_two`: **O(R_d · R_h · T_int)**, но с хорошим constant factor за счет BLAS.
- старый exact path: примерно **O(K · T_sig) + O(K · T_sig) + O(Σ trades)**.
- текущий streaming exact path: **O(K · T_sig) + O(Σ trades)**.
- top-k heap: **O(K_exact · log top_k)**, практически несущественно.

Есть и один важный отрицательный вывод. **Полная vectorization exact accounting path здесь не является естественным следующим шагом**. Этот код stateful: `available_quote`, `safe_quote`, `equity`, `peak_equity`, drawdown, `use_profit_lock`, closing policy и position direction завязаны на последовательность сделок и на порядок их закрытия. Поэтому всё, что выглядит как “давайте просто применим больше NumPy”, здесь очень быстро упирается либо в массивы-посредники с чудовищным footprint, либо в риск тихо сломать семантику. Для такого участка Numba-loop — не компромисс, а правильный инструмент; сама документация Numba подчеркивает, что nopython mode и loop-based kernels являются нормальным способом получения высокой производительности, а `fastmath` надо включать только там, где допустима ослабленная IEEE-семантика. citeturn0search5

## Рекомендации и приоритетная матрица

Ниже — рекомендации, разделенные на quick wins, medium refactoring и deep architecture changes. Я разделяю **факт**, **гипотезу** и **риск** там, где это важно.

### Quick wins без смены архитектуры

Первое, что я бы сделал, — **жестко минимизировал materialization 1m-price arrays**. Сейчас в notebook копируются все пять колонок `ohlcv` в `float64`, хотя exact scoring использует только `open` и `close`. Правильная версия для текущего scope — либо вообще работать по исходному `float32` memmap, либо материализовать только `open` и `close`, причем без повышения dtype. Это почти наверняка даст небольшой выигрыш по времени, но заметный выигрыш по RSS и cache pressure. Из текущих объемов это сокращает отдельный 1m working set примерно со `~174 MiB` до `~35 MiB`, если оставить только `open/close` как `float32`.

Второй safe quick win — **не строить `eval_T` всегда**. В текущем notebook `eval_T` строится в `prefilter_indicator_rows()` даже тогда, когда combo prefilter отключен. В таком режиме `eval_T` нужен только для post-hoc proxy recalculation на финальном `top_k`, а это максимум сотня комбинаций, а не весь filtered pool. Значит, `eval_T` надо делать lazy: либо по флагу `need_eval`, либо отдельно в момент, когда он реально нужен. На текущем fixture это не перевернет wall-clock, но уберет лишнюю 60 MiB materialization и часть prep time.

Третий quick win — **разгрузить notebook-init от заведомо неиспользуемых артефактов**. `price_open_time_1m`, `price_close_time_1m`, `last_close_1m` в текущем scope не участвуют в hot path. Это не тот случай, где надо охотиться за миллисекундами, но это тот случай, где code path должен отражать фактическую зависимость. Чем меньше “случайных” массивов материализовано или даже просто протащено через init, тем чище benchmark и точнее понимание RSS.

Четвертый quick win — **микробенчнуть `combo_chunk_size` заново именно для streaming exact path**. Исторически `256` было разумным chunk size, когда на combo chunk приходились count kernel, trade-list allocations и exact stage. Но теперь структура работы поменялась. При `evaluate_no_risk_streaming_two()` стоит прогнать хотя бы `256 / 512 / 1024 / 2048` и посмотреть, где баланс между Numba launch overhead, output-buffer size и thread scheduling лучше на M2 Max. Это low-risk benchmark, который легко отыграть назад.

Пятый quick win — **оставить `topk_fraction_idx()` в покое**. Он уже использует правильный partial-selection primitive, а не полный sort. Тратить время на переписывание этого участка бессмысленно до тех пор, пока stage-level timings показывают `0.026s` на весь Python heap/top-k work. `argpartition` именно для такого top-fraction selection и предназначен, с оговоркой про неустойчивый порядок, который у вас и так детерминируется позже. citeturn1search1

### Medium refactoring с частичной сменой кода

Первый medium-level шаг — **закрепить `evaluate_no_risk_streaming_two()` как canonical exact engine**, а старые `count_trades_for_two_combos()` и `evaluate_no_risk_trade_list_fast_two()` оставить только как reference / parity path. По твоим данным parity уже выглядит хорошей: trade counts совпадают, а максимальные отклонения по return — на уровне `1e-12`. Это очень сильный сигнал, что streaming path можно не просто “использовать в notebook”, а делать production default exact kernel.

Второй medium refactoring — **решить single-indicator prefilter как fused kernel**, а не как “nonzero pass + chunked proxy pass + unconditional eval copy”. Сейчас на один и тот же `trade_T[:, :-1]` делается несколько проходов. Здесь есть честная гипотеза на ускорение: fused Numba-kernel, который в одном `prange` по строкам одновременно считает `nonzero`, `proxy` и, если надо, change-count для будущей event compression, может уменьшить memory traffic и упростить pool prep. Это не guaranteed slam dunk, поэтому сначала benchmark, потом merge.

Третий шаг — **если combo proxy stage включен, двигаться в сторону GEMM-based proxy path**, а не в сторону дальнейшего тюнинга `proxy_prefilter_combos_chunk_two()`. Ваши же замеры уже показали `~0.225s` против `~0.594s`. Это логично: NumPy матричные операции умеют использовать optimized BLAS, а Numba-loop тут ограничен memory bandwidth и dispatch cost. Но этот выигрыш надо брать аккуратно: BLAS бывает многопоточным, а у вас Numba thread layer — `workqueue`. В production benchmark harness нужно фиксировать и логировать both `NUMBA_NUM_THREADS` и BLAS thread count, иначе можно получить “быстрее на одной машине, хуже на другой”. NumPy прямо предупреждает, что BLAS/LAPACK backends могут быть multithreaded, а Numba дает отдельные parallel diagnostics и threading-layer behavior. citeturn2search0turn2search12turn0search0

Четвертый medium-level шаг — **сделать compute-core API более “service-ready”**. Сейчас orchestration еще notebook-centric: globals, policy constants, инициализация рядом с кодом, `cache=False` helper для JIT. Для будущего production engine это надо вынести в импортируемый модуль, где exact/proxy kernels принимают уже готовые `ndarray` и explicit config struct. Самое практичное последствие такого выноса — возможность включить `cache=True` в Numba и снижать cold-start overhead. При этом надо помнить ограничение Numba-caching: globals рассматриваются как compile-time constants, и кэш надо делать вокруг функций с явными параметрами, а не вокруг неявно захваченных значений. citeturn6search2turn6search3

### Глубокие архитектурные изменения

Главная глубокая рекомендация — **перестать мыслить exact scoring как full scan по каждому бару**. Для MA-family signals натуральное следующее представление — это **change-point / segment representation**. Текущий exact kernel тратит время на миллиарды повторных чтений одинакового направления, хотя реальная trade logic меняется только в моменты, когда меняется хотя бы один из двух входных сигналов. Если для каждой строки заранее хранить массив change points и значения по сегментам, exact pair scoring можно переписать как merge двух event streams. Тогда сложность будет не `O(T)` на combo, а `O(E_dema + E_hma)`, где `E` — число смен состояния, а не число всех 15m баров.

Это важно не только для exact стадии, но и для proxy logic. Если есть префикс-суммы `ret_15m`, то вклад каждого постоянного сегмента в proxy score можно считать за O(1) на сегмент, а не за O(length_of_segment). Это дает шанс ускорить и Stage A prefilter, и future pairwise/N-wise proxy evaluation. Но я специально помечаю это как **гипотезу с высоким upside**, а не как “гарантированный рефакторинг”: сначала надо замерить среднее число sign changes на строку для DEMA/HMA, иначе можно попасть в сложный код без реальной выгоды.

Псевдокод exact engine в таком представлении выглядит так:

```python
# precompute per row once
change_idx, values = compress_signal_row(sig_row)   # segment boundaries and values
entry_idx = sig_entry_exec_idx_15m

# exact pair evaluation
i = j = 0
state = empty_trade_state()

while i < len(dema_segments) and j < len(hma_segments):
    t0 = max(dema_seg_start[i], hma_seg_start[j])
    t1 = min(dema_seg_end[i], hma_seg_end[j])

    dirn = consensus_dir2(dema_values[i], hma_values[j])
    if dirn changes position state:
        close/open trade at entry_idx[t0]
        update_accounting_state(state)

    advance segment that ends at t1
```

Если средняя change density действительно низкая, это главный кандидат на следующий большой скачок производительности.

Вторая глубокая рекомендация — **для 3–5 индикаторов отказаться от полного Cartesian enumeration как архитектурного базиса**. На текущих `142` filtered rows на индикатор 3-indicator case уже дает `2.86M` combos, а 4-indicator — `406.6M`. Тут нужен либо **hierarchical pruning**, либо **beam search**, либо **pairwise upper-bound cache + staged join**, где exact path запускается только на узком пучке кандидатов после нескольких этапов cheap bounds. Иначе никакая оптимизация одной функции не спасет общую систему.

Третья глубокая рекомендация — **не идти в Rust/C++ до тех пор, пока не реализованы event-compressed exact scorer и staged pruning**. Если сделать compiled extension прямо сейчас, она ускорит текущий O(K·T) цикл, но не решит комбинаторный рост. После algorithmic refactor compiled core может иметь смысл, но как второй or third move, а не как first move.

Ниже — приоритетная матрица.

| Рекомендация | Ожидаемое ускорение | Сложность внедрения | Риск | Приоритет | Что измерить | Как проверить корректность |
|---|---|---:|---:|---|---|---|
| Убрать `float64`-копии 1m `ohlcv`, оставить только `open/close` | Низкий по wall, средний по RSS | Низкая | Низкий | Высокий | RSS, init time, total wall | parity exact metrics на artifact fixture |
| Lazy-build `eval_T` только когда нужен | Низкий/средний | Низкая | Низкий | Высокий | prep time, RSS | equality filtered rows, same top-k |
| Зафиксировать streaming exact path как production default | Средний/высокий | Средняя | Низкий/средний | Очень высокий | exact stage wall, CPU%, peak RSS | same trade count, same top-k, max abs metric diff |
| Переоценить `combo_chunk_size` для streaming path | Низкий | Низкая | Низкий | Средний | exact stage wall vs chunk size | identical outputs |
| Fused single-indicator prefilter kernel | Средний | Средняя | Средний | Средний | prepare pools time, memory traffic | same filtered rows, same scores within tol |
| GEMM combo proxy при включенном proxy-stage | Средний/высокий по proxy stage | Средняя | Средний | Высокий | proxy stage wall, peak RSS, thread usage | same confirm counts, proxy diff tolerance |
| Event-compressed exact scorer | Высокий, потенциально 3x–20x на exact stage | Высокая | Высокий | Очень высокий для v2 | change density, exact stage wall | same trade list, trade count, top-k |
| Hierarchical pruning для 3+ indicators | 10x+ / existential | Высокая | Высокий | Критический для roadmap 3+ | combos surviving at each stage | same results on small exhaustive fixtures |
| Перенос kernels в модуль + `cache=True` | Низкий по warm-run, средний по cold-start | Средняя | Низкий | Средний | first-run latency | parity + stable cache behavior |

## План профилирования и roadmap

Профилировать этот engine нужно не “вообще”, а по строго разнесенным фазам. Для wall-clock брать `time.perf_counter_ns()`, потому что это high-resolution performance counter; для process CPU — `time.process_time_ns()`. Python orchestration мерить через `cProfile`, потому что это стандартный deterministic profiler с приемлемым overhead для long-running code. Python-level allocations смотреть через `tracemalloc` snapshots, но помнить, что `tracemalloc` отражает именно Python allocator side, а не всю нативную память BLAS/NumPy. Для Numba loops обязательно включать `parallel_diagnostics()` или `NUMBA_PARALLEL_DIAGNOSTICS`, чтобы проверить, что нужные loops реально lowered в parallel form, а не просто помечены `parallel=True` для красоты. citeturn4search2turn4search4turn3search0turn2search12

Baseline harness я бы построил так. Каждая benchmark-сессия начинается с warmup-run, который не входит в финальные цифры. После этого идут не меньше 10 warm runs на одной и той же fixture, с логированием:
- wall time total,
- stage times,
- process CPU time,
- current RSS / peak RSS,
- Numba thread count,
- BLAS thread count,
- combo count,
- exact candidate count,
- median / p95 / best / worst.

Набор benchmark-сценариев должен быть не один, а минимум четыре. Первый — **current real fixture** на полном диапазоне. Второй — **exact-only mode** с выключенным combo proxy, чтобы видеть pure exact core. Третий — **proxy-enabled mode** с `combo_top_frac < 1`, чтобы сравнить `proxy_prefilter_combos_chunk_two()` и `build_combo_proxy_cache_two()`. Четвертый — **dense-vs-sparse synthetic signals**, специально созданные для проверки будущего event-compressed scorer: один датасет с frequent sign alternation, другой — с длинными постоянными участками.

Размеры данных я бы тестировал на нескольких scale points: примерно `25%`, `50%`, `100%` активного run range, плюс хотя бы один synthetic stress case с artificially increased filtered row count. Это нужно, чтобы отделить **constant-factor wins** от реальных algorithmic wins. Если improvement хорош только на одном fixed size, но не меняет slope, его нельзя считать решением для future 3+ indicator roadmap.

Проверка корректности после каждой оптимизации должна быть бинарно жесткой там, где это критично. Для current exact path нужен набор invariant checks:
- одинаковые `trade_count`,
- одинаковые row ids в top-k,
- одинаковый порядок top-k при одинаковом tie-break policy,
- одинаковый `confirm_count`, если proxy-stage включен,
- max abs diff по floating metrics в пределах договоренного tolerance,
- отдельные synthetic edge cases: no trade, one trade, alternating every bar, close-on-end, `use_fixed_quote`, `use_profit_lock`, zero/negative PnL streaks.

Дальше — реалистичный roadmap.

**Неделя первая** должна быть чисто диагностической и безопасной. Выделить benchmark harness, внедрить phase timers, добавить RSS/CPU logging, зафиксировать thread settings, измерить `combo_chunk_size`, убрать лишние 1m copies, сделать lazy `eval_T`, и зацементировать parity fixtures для streaming exact path. Это safe work, которое почти не трогает семантику и должно быстро превратиться в PR-ы.

**Недели вторая и третья** — это compute-core. Здесь имеет смысл делать fused prefilter kernel, productionize `evaluate_no_risk_streaming_two()` вне notebook, сравнить old exact vs streaming exact на полном fixture, и если proxy-stage нужен — выбрать между Numba-proxy и GEMM-proxy на основании реальных stage benchmarks, а не вкуса команды.

**Неделя четвертая и дальше** — только algorithmic redesign. Сначала измерить change density для строк DEMA/HMA. Если она низкая, строить prototype event-compressed exact scorer и сравнивать его не с Python reference, а с уже validated streaming exact path. Параллельно проектировать staged pruning для 3-indicator case. Если этих шагов не сделать, roadmap “3–5 индикаторов” технически не защищен, независимо от того, насколько хорошо отполированы текущие Numba loops.

## Варианты новой архитектуры и финальные выводы

Первый вариант — **минимальный refactor**. Он подходит, если задача на ближайший месяц — просто сделать current 2-indicator artifact-backed run быстрее и чище. В этом варианте сохраняется существующая логика notebook, но ядро выносится в модуль, exact path стандартизуется вокруг `evaluate_no_risk_streaming_two()`, убираются лишние copies, включается proper benchmark harness, а combo proxy выбирается policy-driven. Это самый безопасный путь и, по моему мнению, именно его надо делать первым.

Второй вариант — **hybrid vectorized + event-driven architecture**. Это уже путь для v2. Идея здесь не в том, чтобы “всё завекторизовать”, а в том, чтобы разделить движок на два слоя. Первый слой — vectorized/artifact layer: row preparation, prefilter, prefix sums, pairwise bounds, staged pruning. Второй слой — event-driven exact layer: stateful no-risk accounting по сжатому event stream. В такой архитектуре vectorization работает там, где уместна, а state machine остается там, где порядок событий критичен. Для вашего предметного домена это наиболее естественный long-term design.

Третий вариант — **compiled compute core**. Он имеет смысл лишь после того, как вы исчерпали algorithmic и data-movement wins на NumPy/Numba. В этой версии orchestration остается на Python, а exact scorer и, возможно, event-compressed merge kernel уходят в более низкоуровневый compiled layer. Это может понадобиться, если после event compression warm-run всё еще не проходит target или если Numba окажется недостаточно предсказуемой в deployment. Но делать это до redesign exact algorithm — рано.

Финальный вывод предельно конкретный. **Сначала** нужно закрепить уже найденные реальные выигрыши и убрать оставшийся data-movement waste: streaming exact path как production default, lazy `eval_T`, только нужные `1m open/close` без `float64` copies, и clean benchmark harness на Mac Studio. **Потом** — решить, нужен ли combo proxy в активном режиме и какой из двух вариантов победит на реальных stage benchmarks. **После этого** — переходить к единственной большой ставке, которая действительно меняет future scalability: event-compressed exact scoring и staged pruning для 3+ индикаторов.

Если резюмировать в одной фразе: **текущий prototype уже достаточно хорош, чтобы перестать шлифовать Python-мелочи и начать менять форму exact алгоритма**. Именно это даст следующий серьезный выигрыш.