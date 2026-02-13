# Runbook — Numba warmup / JIT

## Назначение

Документ помогает диагностировать задержки первого расчёта индикаторов и отличать ожидаемый JIT warmup от неисправности.

Связанный контекст:
- `docs/architecture/indicators/indicators-overview.md`
- `docs/architecture/indicators/indicators-compute-engine-core.md`
- `docs/runbooks/indicators-numba-cache-and-threads.md`

## Когда использовать

- Первый `POST /indicators/compute` заметно медленнее последующих.
- После рестарта API время первого запроса снова высокое.
- В логах нет подтверждения, что warmup завершился.

## Что проверить

### 1) Runtime config и env overrides

```bash
rg -n "numba_num_threads|numba_cache_dir|max_compute_bytes_total" configs/prod/indicators.yaml
printenv | rg 'NUMBA_NUM_THREADS|NUMBA_CACHE_DIR|ROEHUB_NUMBA_NUM_THREADS|ROEHUB_NUMBA_CACHE_DIR'
```

Ожидание:
- значения присутствуют и не конфликтуют между YAML и environment.
- если заданы env-переменные, они должны быть осознанными override.

### 2) Наличие warmup-лога на старте

```bash
docker logs --tail 300 roehub-api | rg 'compute_numba warmup complete|warmup_seconds|numba_num_threads_effective|numba_cache_dir'
```

Ожидание:
- есть запись `compute_numba warmup complete`.
- в логе видны `warmup_seconds`, эффективные threads и путь к cache.

### 3) Кэш Numba реально используется

```bash
ls -la .cache/numba/prod
```

Ожидание:
- директория существует и пополняется файлами после старта/прогрева.

## Что делать

1. Если warmup-лог отсутствует, проверить wiring:
   - `apps/api/wiring/modules/indicators.py`
   - `apps/api/main/app.py`
2. Если прогрев слишком долгий, снизить стартовую конкуренцию и размер warmup-контекста:
   - временно уменьшить `numba_num_threads`.
3. Если после рестарта снова "холодный" старт, проверить персистентность cache dir:
   - `numba_cache_dir` должен быть mounted volume в контейнере.
4. Для smoke-проверки после изменений запустить:

```bash
uv run pytest -q tests/unit/contexts/indicators/adapters/outbound/compute_numba/test_runtime_wiring.py
uv run pytest -q tests/perf_smoke/contexts/indicators/test_compute_numba_perf_smoke.py
```

Ожидание:
- warmup проходит без ошибок.
- первый вычислительный вызов после старта не блокируется аномально долго.

## Что считать нормальным

- Первый запуск после деплоя может быть медленнее из-за JIT.
- После успешного warmup повторные вызовы должны быть стабильно быстрее.

## Эскалация

Если warmup падает на старте:
- приложить конфиг (`compute.numba`), env overrides и фрагмент лога с ошибкой.
- проверить writable-доступ к `numba_cache_dir` (см. отдельный runbook).
