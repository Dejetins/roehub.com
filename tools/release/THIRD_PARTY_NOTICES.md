# Реестр сторонних компонентов Roehub

Файл сгенерирован `tools/release/oss_metadata.py`; ручные изменения будут отклонены.
Статус `conditional` означает обязательства, которые должны быть выполнены для
конкретного комплекта выпуска. Статус `excluded` означает, что компонент не входит
в исходный или бинарный комплект Roehub.
Собственные образы `roehub/runtime*` не являются сторонними компонентами и намеренно
не включаются в этот файл: это исключает циклическую зависимость image digest от notice.

## Прямые зависимости Python

| Компонент | Версия | Лицензия | Статус |
|---|---:|---|---|
| `hatchling` | `1.31.0` | `MIT` | `compatible` |
| `pandas` | `2.2.3` | `BSD-3-Clause` | `compatible` |
| `pyarrow` | `18.0.0` | `Apache-2.0` | `compatible` |
| `mypy` | `1.13.0` | `MIT` | `compatible` |
| `pyright` | `1.1.408` | `MIT` | `compatible` |
| `pytest` | `8.3.3` | `MIT` | `compatible` |
| `pytest-asyncio` | `0.24.0` | `Apache-2.0` | `compatible` |
| `pytest-cov` | `5.0.0` | `MIT` | `compatible` |
| `ruff` | `0.7.4` | `MIT` | `compatible` |
| `types-python-dateutil` | `2.9.0.20241003` | `Apache-2.0` | `compatible` |
| `types-pyyaml` | `6.0.12.20240917` | `Apache-2.0` | `compatible` |
| `optuna` | `4.9.0` | `MIT` | `compatible` |
| `torch` | `2.7.1+cpu` | `BSD-3-Clause` | `compatible` |
| `alembic` | `1.14.0` | `MIT` | `compatible` |
| `argon2-cffi` | `25.1.0` | `MIT` | `compatible` |
| `beautifulsoup4` | `4.14.3` | `MIT` | `compatible` |
| `binance-historical-data` | `0.1.14` | `MIT` | `compatible` |
| `clickhouse-connect` | `0.8.10` | `Apache-2.0` | `compatible` |
| `confluent-kafka` | `2.6.1` | `Apache-2.0` | `compatible` |
| `cryptography` | `43.0.3` | `Apache-2.0 OR BSD-3-Clause` | `compatible` |
| `fastapi` | `0.115.5` | `MIT` | `compatible` |
| `httpx` | `0.27.2` | `BSD-3-Clause` | `compatible` |
| `itsdangerous` | `2.2.0` | `BSD-3-Clause` | `compatible` |
| `jinja2` | `3.1.4` | `BSD-3-Clause` | `compatible` |
| `jsonschema` | `4.23.0` | `MIT` | `compatible` |
| `lxml` | `6.0.2` | `BSD-3-Clause` | `compatible` |
| `numba` | `0.60.0` | `BSD-2-Clause` | `compatible` |
| `numpy` | `2.0.2` | `BSD-3-Clause` | `compatible` |
| `opentelemetry-api` | `1.28.2` | `Apache-2.0` | `compatible` |
| `opentelemetry-exporter-otlp` | `1.28.2` | `Apache-2.0` | `compatible` |
| `opentelemetry-instrumentation` | `0.49b2` | `Apache-2.0` | `compatible` |
| `opentelemetry-instrumentation-fastapi` | `0.49b2` | `Apache-2.0` | `compatible` |
| `opentelemetry-instrumentation-requests` | `0.49b2` | `Apache-2.0` | `compatible` |
| `opentelemetry-sdk` | `1.28.2` | `Apache-2.0` | `compatible` |
| `orjson` | `3.10.12` | `Apache-2.0 OR MIT` | `compatible` |
| `prometheus-client` | `0.21.0` | `Apache-2.0` | `compatible` |
| `psycopg` | `3.2.4` | `LGPL-3.0-only` | `conditional` |
| ↳ обязательство |  |  | Keep Psycopg separable; ship its license and source-location notice. The pinned runtime artifacts are covered by per-platform SPDX and the digest-bound runtime license audit; changing a wheel or image digest requires re-audit. |
| `pybit` | `5.13.0` | `MIT` | `compatible` |
| `pydantic` | `2.9.2` | `MIT` | `compatible` |
| `pydantic-settings` | `2.6.1` | `MIT` | `compatible` |
| `pyotp` | `2.9.0` | `MIT` | `compatible` |
| `python-dateutil` | `2.9.0.post0` | `Apache-2.0 OR BSD-3-Clause` | `compatible` |
| `python-dotenv` | `1.0.1` | `BSD-3-Clause` | `compatible` |
| `python-multipart` | `0.0.9` | `Apache-2.0` | `compatible` |
| `pyyaml` | `6.0.2` | `MIT` | `compatible` |
| `redis` | `5.2.0` | `MIT` | `compatible` |
| `sqlalchemy` | `2.0.36` | `MIT` | `compatible` |
| `structlog` | `24.4.0` | `Apache-2.0 OR MIT` | `compatible` |
| `tenacity` | `9.0.0` | `Apache-2.0` | `compatible` |
| `ujson` | `5.10.0` | `BSD-3-Clause` | `compatible` |
| `uvicorn` | `0.32.0` | `BSD-3-Clause` | `compatible` |
| `webauthn` | `2.5.1` | `BSD-3-Clause` | `compatible` |

## Контейнерные образы

| Компонент | Версия | Лицензия | Статус |
|---|---:|---|---|
| `${ROEHUB_APP_IMAGE:?ROEHUB_APP_IMAGE is required}` | `?ROEHUB_APP_IMAGE is required}` | `Apache-2.0` | `compatible` |
| ↳ обязательство |  |  | First-party image; release manifest must replace the variable with a digest-pinned Roehub image. |
| `alpine:3.22@sha256:14358309a308569c32bdc37e2e0e9694be33a9d99e68afb0f5ff33cc1f695dce` | `14358309a308569c32bdc37e2e0e9694be33a9d99e68afb0f5ff33cc1f695dce` | `LicenseRef-Alpine-Base-Image` | `conditional` |
| ↳ обязательство |  |  | Audit packages and layers for the final release digest and preserve all required notices and corresponding-source offers. |
| `clickhouse/clickhouse-server:24.8` | `24.8` | `Apache-2.0` | `compatible` |
| `clickhouse/clickhouse-server:24.8@sha256:1ffa82edee000a42c09313bd9f1293d94c570aee74babc1b3ca9983a35fa597b` | `1ffa82edee000a42c09313bd9f1293d94c570aee74babc1b3ca9983a35fa597b` | `Apache-2.0` | `compatible` |
| `docker.io/library/alpine@sha256:5b10f432ef3da1b8d4c7eb6c487f2f5a8f096bc91145e68878dd4a5019afde11` | `5b10f432ef3da1b8d4c7eb6c487f2f5a8f096bc91145e68878dd4a5019afde11` | `LicenseRef-Alpine-Base-Image` | `conditional` |
| ↳ обязательство |  |  | Build-only and runtime base for the OpenBao derivative; preserve exact package notices and audit the final digest. |
| `docker.io/library/golang@sha256:8d22e29d960bc50cd025d93d5b7c7d220b1ee9aa7a239b3c8f55a57e987e8d45` | `8d22e29d960bc50cd025d93d5b7c7d220b1ee9aa7a239b3c8f55a57e987e8d45` | `BSD-3-Clause AND LicenseRef-Alpine-Base-Image` | `conditional` |
| ↳ обязательство |  |  | Build-only image for the OpenBao derivative; preserve toolchain and base-image notices in provenance. |
| `docker.io/library/node@sha256:ba36e9b2705008e63e354214f0e3011c528af9df2ca13ac2bd2c0114650302e6` | `ba36e9b2705008e63e354214f0e3011c528af9df2ca13ac2bd2c0114650302e6` | `MIT AND LicenseRef-Debian-Base-Image` | `conditional` |
| ↳ обязательство |  |  | Build-only image for the OpenBao UI; preserve toolchain and base-image notices in provenance. |
| `ghcr.io/astral-sh/uv:0.8.4@sha256:40775a79214294fb51d097c9117592f193bcfdfc634f4daa0e169ee965b10ef0` | `40775a79214294fb51d097c9117592f193bcfdfc634f4daa0e169ee965b10ef0` | `Apache-2.0 OR MIT` | `compatible` |
| `ghcr.io/dejetins/roehub-openbao:2.5.4-roehub-licensed-qr.1@sha256:8492e2c1a523aac5da44e41c86e84eac992479fb7c4a79c2e1a07b8b24bcec4a` | `8492e2c1a523aac5da44e41c86e84eac992479fb7c4a79c2e1a07b8b24bcec4a` | `MPL-2.0 AND MIT` | `conditional` |
| ↳ обязательство |  |  | Preserve the exact modified OpenBao corresponding source, Roehub patch, MPL-2.0 and MIT license texts, and derivative notice for this digest. |
| `ghcr.io/google/cadvisor:v0.56.2` | `v0.56.2` | `Apache-2.0` | `compatible` |
| `grafana/grafana:12.0.2@sha256:b5b59bfc7561634c2d7b136c4543d702ebcc94a3da477f21ff26f89ffd4214fa` | `b5b59bfc7561634c2d7b136c4543d702ebcc94a3da477f21ff26f89ffd4214fa` | `AGPL-3.0-only` | `conditional` |
| ↳ обязательство |  |  | Distribute only as a separate unmodified aggregate service; preserve notices and provide corresponding-source access for this exact digest. |
| `grafana/loki:3.5.1@sha256:a74594532eec4cc313401beedc4dd2708c43674c032084b1aeb87c14a5be1745` | `a74594532eec4cc313401beedc4dd2708c43674c032084b1aeb87c14a5be1745` | `AGPL-3.0-only` | `conditional` |
| ↳ обязательство |  |  | Distribute only as a separate unmodified aggregate service; preserve notices and provide corresponding-source access for this exact digest. |
| `oliver006/redis_exporter:v1.80.1` | `v1.80.1` | `MIT` | `compatible` |
| `postgres:16` | `16` | `PostgreSQL` | `compatible` |
| `postgres:16@sha256:be01cf82fc7dbba824acf0a82e150b4b360f3ff93c6631d7844af431e841a95c` | `be01cf82fc7dbba824acf0a82e150b4b360f3ff93c6631d7844af431e841a95c` | `PostgreSQL` | `compatible` |
| `prom/alertmanager:v0.28.1@sha256:27c475db5fb156cab31d5c18a4251ac7ed567746a2483ff264516437a39b15ba` | `27c475db5fb156cab31d5c18a4251ac7ed567746a2483ff264516437a39b15ba` | `Apache-2.0` | `compatible` |
| `prom/blackbox-exporter:v0.27.0@sha256:a50c4c0eda297baa1678cd4dc4712a67fdea713b832d43ce7fcc5f9bea05094d` | `a50c4c0eda297baa1678cd4dc4712a67fdea713b832d43ce7fcc5f9bea05094d` | `Apache-2.0` | `compatible` |
| `prom/prometheus:v3.5.0@sha256:63805ebb8d2b3920190daf1cb14a60871b16fd38bed42b857a3182bc621f4996` | `63805ebb8d2b3920190daf1cb14a60871b16fd38bed42b857a3182bc621f4996` | `Apache-2.0` | `compatible` |
| `python:3.12-slim` | `3.12-slim` | `PSF-2.0` | `conditional` |
| ↳ обязательство |  |  | Audit and notice Debian packages and all transitive image layers for the final digest. |
| `python:3.12-slim-bookworm@sha256:8a7e7cc04fd3e2bd787f7f24e22d5d119aa590d429b50c95dfe12b3abe52f48b` | `8a7e7cc04fd3e2bd787f7f24e22d5d119aa590d429b50c95dfe12b3abe52f48b` | `PSF-2.0` | `conditional` |
| ↳ обязательство |  |  | Audit and notice Debian packages and all transitive image layers for the final digest. |
| `python:3.12.11-slim-bookworm@sha256:519591d6871b7bc437060736b9f7456b8731f1499a57e22e6c285135ae657bf7` | `519591d6871b7bc437060736b9f7456b8731f1499a57e22e6c285135ae657bf7` | `PSF-2.0` | `conditional` |
| ↳ обязательство |  |  | Audit and notice Debian packages and all transitive image layers for the final digest. |
| `quay.io/prometheuscommunity/postgres-exporter:v0.18.1` | `v0.18.1` | `Apache-2.0` | `compatible` |
| `redis:7.2-bookworm` | `7.2-bookworm` | `BSD-3-Clause` | `compatible` |
| `redis:7.2-bookworm@sha256:e51cbc16f94b2426e80b9516db174a07d55e882217a1ec1d729b137b32e24e42` | `e51cbc16f94b2426e80b9516db174a07d55e882217a1ec1d729b137b32e24e42` | `BSD-3-Clause` | `compatible` |

## Образы комплекта выпуска

| Компонент | Версия | Лицензия | Статус |
|---|---:|---|---|
| `prom/alertmanager:v0.28.1@sha256:27c475db5fb156cab31d5c18a4251ac7ed567746a2483ff264516437a39b15ba` | `27c475db5fb156cab31d5c18a4251ac7ed567746a2483ff264516437a39b15ba` | `Apache-2.0` | `compatible` |
| `prom/blackbox-exporter:v0.27.0@sha256:a50c4c0eda297baa1678cd4dc4712a67fdea713b832d43ce7fcc5f9bea05094d` | `a50c4c0eda297baa1678cd4dc4712a67fdea713b832d43ce7fcc5f9bea05094d` | `Apache-2.0` | `compatible` |
| `clickhouse/clickhouse-server:24.8@sha256:1ffa82edee000a42c09313bd9f1293d94c570aee74babc1b3ca9983a35fa597b` | `1ffa82edee000a42c09313bd9f1293d94c570aee74babc1b3ca9983a35fa597b` | `Apache-2.0` | `compatible` |
| `alpine:3.22@sha256:14358309a308569c32bdc37e2e0e9694be33a9d99e68afb0f5ff33cc1f695dce` | `14358309a308569c32bdc37e2e0e9694be33a9d99e68afb0f5ff33cc1f695dce` | `LicenseRef-Alpine-Base-Image` | `conditional` |
| ↳ обязательство |  |  | Audit every package and layer for the final release digest; preserve required notices and corresponding-source offers before distribution. |
| `grafana/grafana:12.0.2@sha256:b5b59bfc7561634c2d7b136c4543d702ebcc94a3da477f21ff26f89ffd4214fa` | `b5b59bfc7561634c2d7b136c4543d702ebcc94a3da477f21ff26f89ffd4214fa` | `AGPL-3.0-only` | `conditional` |
| ↳ обязательство |  |  | Distribute only as a separate unmodified aggregate service; preserve notices and provide corresponding-source access for this exact digest. |
| `grafana/loki:3.5.1@sha256:a74594532eec4cc313401beedc4dd2708c43674c032084b1aeb87c14a5be1745` | `a74594532eec4cc313401beedc4dd2708c43674c032084b1aeb87c14a5be1745` | `AGPL-3.0-only` | `conditional` |
| ↳ обязательство |  |  | Distribute only as a separate unmodified aggregate service; preserve notices and provide corresponding-source access for this exact digest. |
| `ghcr.io/dejetins/roehub-openbao:2.5.4-roehub-licensed-qr.1@sha256:8492e2c1a523aac5da44e41c86e84eac992479fb7c4a79c2e1a07b8b24bcec4a` | `8492e2c1a523aac5da44e41c86e84eac992479fb7c4a79c2e1a07b8b24bcec4a` | `MPL-2.0 AND MIT` | `conditional` |
| ↳ обязательство |  |  | Preserve the exact modified OpenBao corresponding source, Roehub patch, MPL-2.0 and MIT license texts, and derivative notice for this digest. |
| `postgres:16@sha256:be01cf82fc7dbba824acf0a82e150b4b360f3ff93c6631d7844af431e841a95c` | `be01cf82fc7dbba824acf0a82e150b4b360f3ff93c6631d7844af431e841a95c` | `PostgreSQL` | `compatible` |
| `prom/prometheus:v3.5.0@sha256:63805ebb8d2b3920190daf1cb14a60871b16fd38bed42b857a3182bc621f4996` | `63805ebb8d2b3920190daf1cb14a60871b16fd38bed42b857a3182bc621f4996` | `Apache-2.0` | `compatible` |
| `redis:7.2-bookworm@sha256:e51cbc16f94b2426e80b9516db174a07d55e882217a1ec1d729b137b32e24e42` | `e51cbc16f94b2426e80b9516db174a07d55e882217a1ec1d729b137b32e24e42` | `BSD-3-Clause` | `compatible` |
| `alpine:3.22@sha256:14358309a308569c32bdc37e2e0e9694be33a9d99e68afb0f5ff33cc1f695dce` | `14358309a308569c32bdc37e2e0e9694be33a9d99e68afb0f5ff33cc1f695dce` | `LicenseRef-Alpine-Base-Image` | `conditional` |
| ↳ обязательство |  |  | Audit every package and layer for the final release digest; preserve required notices and corresponding-source offers before distribution. |

## Встроенные Web-ресурсы

| Компонент | Версия | Лицензия | Статус |
|---|---:|---|---|
| `htmx` | `1.9.12` | `0BSD` | `compatible` |
| `lightweight-charts-notice` | `5.2.0` | `Apache-2.0 AND 0BSD` | `compatible` |
| `lightweight-charts` | `5.2.0` | `Apache-2.0 AND 0BSD` | `compatible` |
| ↳ обязательство |  |  | Preserve the upstream NOTICE and user-visible TradingView attribution/link requirement. |

## Известные риски транзитивных лицензий

- Raw Syft NOASSERTION records in first-party runtime images must be resolved by the digest-bound runtime license audit; any new or unmatched record blocks release.
- External image package and layer obligations remain bound to their per-platform SPDX and corresponding-source records; changing an image digest requires re-audit.
- The pinned psycopg-binary and torch artifacts are covered by per-platform runtime SPDX and embedded license evidence; changing a wheel or image digest requires re-audit.
- Images using latest tags are inventory evidence only and are forbidden in a release manifest until digest-pinned.
