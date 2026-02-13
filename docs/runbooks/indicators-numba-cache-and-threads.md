# Runbook — Numba cache dir + threads

## Назначение

Документ описывает, как управлять `numba_cache_dir` и количеством потоков (`NUMBA_NUM_THREADS`) для `indicators` compute.

Связанный контекст:
- `docs/architecture/indicators/indicators-overview.md`
- `docs/architecture/indicators/indicators-compute-engine-core.md`
- `docs/runbooks/indicators-numba-warmup-jit.md`

## Источники настройки

1. YAML defaults в `configs/prod/indicators.yaml` (`compute.numba.numba_num_threads`, `compute.numba.numba_cache_dir`).
2. Env overrides:
   - `ROEHUB_NUMBA_NUM_THREADS`, `NUMBA_NUM_THREADS`
   - `ROEHUB_NUMBA_CACHE_DIR`, `NUMBA_CACHE_DIR`
3. Итог применяет warmup/runtime config loader.

## Быстрая проверка

### 1) Проверить значения в YAML

```bash
rg -n "numba_num_threads|numba_cache_dir" configs/prod/indicators.yaml
```

Ожидание:
- ключи присутствуют и соответствуют целевой среде.

### 2) Проверить активные env overrides

```bash
printenv | rg 'ROEHUB_NUMBA_NUM_THREADS|NUMBA_NUM_THREADS|ROEHUB_NUMBA_CACHE_DIR|NUMBA_CACHE_DIR'
```

Ожидание:
- либо переменные отсутствуют (используется YAML), либо заданы осознанно.

### 3) Проверить writable cache dir

```bash
CACHE_DIR=".cache/numba/prod"
mkdir -p "$CACHE_DIR"
touch "$CACHE_DIR/.probe" && rm "$CACHE_DIR/.probe"
```

Ожидание:
- нет ошибок прав записи.

## Что делать

1. Подобрать потоковость под CPU quota контейнера:
   - начать с `NUMBA_NUM_THREADS=<кол-во vCPU>`.
   - не завышать значение относительно выделенных CPU.
2. Для стабильного cold-start использовать постоянный кэш:
   - `numba_cache_dir` должен указывать на volume, переживающий рестарты.
3. Если есть конфликт env и YAML, выбрать один источник истины и убрать лишние overrides.

## Примеры конфигурации

### Bash

```bash
export ROEHUB_ENV=prod
export NUMBA_NUM_THREADS=8
export NUMBA_CACHE_DIR=/var/lib/roehub/numba-cache
```

### PowerShell

```powershell
$env:ROEHUB_ENV = "prod"
$env:NUMBA_NUM_THREADS = "8"
$env:NUMBA_CACHE_DIR = "C:\\roehub\\numba-cache"
```

## Container notes

- Смонтировать cache directory как persistent volume.
- Проверить права пользователя процесса API на запись в `numba_cache_dir`.
- После изменения потоков пересмотреть latency/CPU в perf-smoke.

## Команды валидации

```bash
uv run pytest -q tests/unit/contexts/indicators/adapters/outbound/compute_numba/test_runtime_wiring.py
uv run pytest -q tests/perf_smoke/contexts/indicators/test_compute_numba_perf_smoke.py
```

Ожидание:
- warmup и compute проходят, ошибок по cache dir/threads нет.
