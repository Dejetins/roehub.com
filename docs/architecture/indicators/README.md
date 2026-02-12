# Indicators Architecture

Документы архитектуры bounded context `indicators`.

## Индекс

- `docs/architecture/indicators/indicators-overview.md` — обзор слоя indicators и границы ответственности.
- `docs/architecture/indicators/indicators-application-ports-walking-skeleton-v1.md` — доменные DTO/ports и контракты.
- `docs/architecture/indicators/indicators-registry-yaml-defaults-v1.md` — hard defs + YAML defaults + deterministic merge.
- `docs/architecture/indicators/indicators-grid-builder-estimate-guards-v1.md` — Indicators — Grid Builder + Batch Estimate + Guards (v1).
- `docs/architecture/indicators/indicators-candlefeed-acl-dense-timeline-v1.md` — CandleFeed ACL (`market_data_acl`) with dense `[start, end)` 1m timeline, `NaN` holes, `last-wins` duplicates, and `ignore` out-of-range policy.
- `docs/architecture/indicators/indicators-compute-engine-core.md` — CPU/Numba compute engine core, warmup, memory guard, runtime config.
- `docs/architecture/indicators/indicators-ma-compute-numba-v1.md` — MA kernels (Numba) + Numpy oracle + `POST /indicators/compute` v1.
