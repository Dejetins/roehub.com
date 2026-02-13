# Indicators Architecture

Документы архитектуры bounded context `indicators`.

## Индекс

### Entry point

- `docs/architecture/indicators/indicators-overview.md` — единая точка входа (архитектурный поток, инварианты, checklist "How to add a new indicator").

### Group docs (deterministic order)

- `docs/architecture/indicators/indicators-ma.md`
- `docs/architecture/indicators/indicators-volatility.md`
- `docs/architecture/indicators/indicators-momentum.md`
- `docs/architecture/indicators/indicators-trend.md`
- `docs/architecture/indicators/indicators-volume.md`
- `docs/architecture/indicators/indicators-structure.md`

### Runbooks

- `docs/runbooks/indicators-numba-warmup-jit.md` — Numba warmup / JIT.
- `docs/runbooks/indicators-numba-cache-and-threads.md` — `numba_cache_dir`, `NUMBA_NUM_THREADS`, container notes.
- `docs/runbooks/indicators-why-nan.md` — Troubleshooting: why NaN?.

### Deep implementation docs (v1)

- `docs/architecture/indicators/indicators-application-ports-walking-skeleton-v1.md` — доменные DTO/ports и контракты.
- `docs/architecture/indicators/indicators-registry-yaml-defaults-v1.md` — hard defs + YAML defaults + deterministic merge.
- `docs/architecture/indicators/indicators-grid-builder-estimate-guards-v1.md` — Grid Builder + Batch Estimate + Guards.
- `docs/architecture/indicators/indicators-candlefeed-acl-dense-timeline-v1.md` — CandleFeed ACL (`market_data_acl`) с dense `[start, end)` 1m timeline и NaN holes.
- `docs/architecture/indicators/indicators-compute-engine-core.md` — CPU/Numba compute engine core, warmup, memory guard, runtime config.
- `docs/architecture/indicators/indicators-ma-compute-numba-v1.md` — MA kernels (Numba) + NumPy oracle.
- `docs/architecture/indicators/indicators-volatility-momentum-compute-numba-v1.md` — Volatility + Momentum kernels (Numba) + NumPy oracle.
- `docs/architecture/indicators/indicators-trend-volume-compute-numba-v1.md` — Trend + Volume kernels (Numba) + NumPy oracle.
- `docs/architecture/indicators/indicators-structure-normalization-compute-numba-v1.md` — Structure/Normalization kernels (Numba) + NumPy oracle.

### Specs and config

- `docs/architecture/indicators/indicators_formula.yaml` — source of truth для формул и output mapping.
- `configs/prod/indicators.yaml` — боевые defaults и compute runtime knobs.
