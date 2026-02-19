---
# =========================
# Prompt Front-Matter (YAML)
# =========================
prompt_name: codex-str-epic-06-strategy-runtime-config-v1-implementation
repo: Dejetins/roehub.com
branch: main
scope: "Implement STR-EPIC-06 end-to-end: Strategy runtime discipline via configs/*/strategy.yaml (source of truth), shim strategy_live_runner.yaml, enable toggles, metrics port, and env overrides"
language:
  implementation: python
  agent_report: ru
primary_docs:
  - docs/architecture/strategy/strategy-runtime-config-v1.md
  - docs/runbooks/strategy-live-worker.md
  - docs/architecture/strategy/strategy-live-runner-redis-streams-v1.md
  - docs/architecture/strategy/strategy-realtime-output-redis-streams-v1.md
  - docs/architecture/strategy/strategy-telegram-notifier-best-effort-policy-v1.md
  - docs/architecture/identity/identity-telegram-login-user-model-v1.md
  - docs/architecture/roadmap/milestone-3-epics-v1.md
style_references:
  - src/trading/contexts/market_data/application/ports/stores/canonical_candle_reader.py
  - src/trading/contexts/market_data/application/ports/stores/enabled_instrument_reader.py
  - src/trading/contexts/market_data/application/ports/stores/raw_kline_writer.py
hard_requirements:
  keep_fail_fast_on_startup: true
  do_not_break_public_imports: true
  deterministic_ordering_everywhere: true
  stable_exports_via_init: true
  add_or_update_unit_tests_as_needed: true
  docstrings_must_link_to_docs_and_related_files: true
  maintain_existing_variant_key_v1_semantics: true
task_toggles:
  read_codex_agents_first: true
  strategy_yaml_is_source_of_truth: true
  keep_strategy_live_runner_yaml_as_shim: true
  support_legacy_strategy_live_runner_yaml_payload: true
  support_config_path_resolution_cli_or_env_or_roehub_env: true
  add_env_overrides_for_scalar_toggles_and_metrics_port: true
  api_router_must_respect_strategy_api_enabled: true
  live_worker_must_exit_0_when_disabled: true
  cli_metrics_port_override_must_win: true
  update_docs_index_if_any_md_changed: true
target_envs:
  - dev
  - test
  - prod
required_literals:
  - ROEHUB_ENV
  - ROEHUB_STRATEGY_CONFIG
  - strategy.api.enabled
  - strategy.live_worker.enabled
  - strategy.realtime_output.redis_streams.enabled
  - strategy.telegram.enabled
  - strategy.metrics.port
  - STRATEGY_PG_DSN
  - TELEGRAM_BOT_TOKEN
  - ROEHUB_REDIS_PASSWORD
  - CH_HOST
  - CH_PORT
  - identity_telegram_channels
non_goals:
  - "Do not implement hot reload for any strategy config"
  - "Do not rename existing environment variables or migrate to new DSN keys in v1"
  - "Do not add Prometheus metrics endpoint to the API process in this epic"
final_report_format:
  language: ru
  sections:
    - "Результат"
    - "Изменённые файлы"
    - "Ключевые решения"
    - "Как проверить"
    - "Риски / что дальше"
quality_gates:
  - cmd: "uv run ruff check ."
    expect: "exit code 0"
  - cmd: "uv run pyright"
    expect: "exit code 0"
  - cmd: "uv run pytest -q"
    expect: "exit code 0"
  - cmd: "uv run python -m tools.docs.generate_docs_index --check"
    expect:
      - "exit code 0"
      - "run only if any .md file was added/changed"
expected_touched_paths:
  - configs/dev/strategy.yaml
  - configs/test/strategy.yaml
  - configs/prod/strategy.yaml
  - configs/*/strategy_live_runner.yaml
  - src/trading/contexts/strategy/adapters/outbound/config/*.py
  - src/trading/contexts/strategy/adapters/outbound/config/__init__.py
  - apps/worker/strategy_live_runner/main/main.py
  - apps/worker/strategy_live_runner/wiring/modules/strategy_live_runner.py
  - apps/api/main/app.py
  - apps/api/wiring/modules/strategy.py
  - tests/unit/contexts/strategy/**/*.py
  - tests/unit/apps/api/**/*.py
  - docs/**
safety_notes:
  - "Read .codex/AGENTS.md first. If it conflicts with this prompt, this prompt has priority."
  - "Keep backward compatibility: existing invocations using configs/*/strategy_live_runner.yaml must keep working."
---

# Task

Implement the architecture document `docs/architecture/strategy/strategy-runtime-config-v1.md` fully and consistently.

Done means the repository has Strategy-level runtime discipline comparable to market_data/indicators:

- `configs/<env>/strategy.yaml` exists for `dev`, `test`, and `prod` and is the **source of truth**.
- `configs/<env>/strategy_live_runner.yaml` remains supported but becomes a **shim/alias** pointing to `strategy.yaml` (while still supporting legacy full payload for compatibility).
- A fail-fast loader/validator for `strategy.yaml` exists and is used by Strategy API wiring and Strategy live worker wiring.
- Enable toggles and metrics port are implemented exactly as documented.
- Env overrides exist for the whitelisted scalar keys (toggles + metrics port + config path).
- Unit tests are added/updated to cover the new loader, shim behavior, and toggle semantics.
- Ruff, Pyright, and Pytest pass.

IMPORTANT: The user-provided path "strategy-api-docs/architecture/strategy/strategy-runtime-config-v1.md" is a typo; the canonical doc path is `docs/architecture/strategy/strategy-runtime-config-v1.md`. Follow the canonical path.

## Context / Current State

- Strategy live worker currently loads `configs/*/strategy_live_runner.yaml` via `load_strategy_live_runner_runtime_config`.
- Strategy API wiring does not use YAML toggles yet; Strategy router is always included.
- Worker metrics port is currently a CLI flag (`--metrics-port`) in several entrypoints.
- There is an existing pattern for env->YAML->default scalar override resolution (e.g. `src/trading/platform/config/indicators_compute_numba.py`).

## Requirements (Must)

- Read `.codex/AGENTS.md` before any implementation work and mention this in the final report.
- Implement `configs/dev/strategy.yaml`, `configs/test/strategy.yaml`, `configs/prod/strategy.yaml` (version 1).
- Implement a fail-fast loader + validator for `strategy.yaml`.
- Implement config path selection with the following precedence:
  1) CLI `--config` (where applicable)
  2) `ROEHUB_STRATEGY_CONFIG`
  3) fallback `configs/<ROEHUB_ENV>/strategy.yaml` (default env: `dev`)
- Implement enable toggles with the documented semantics:
  - `strategy.api.enabled=false` => do not include Strategy router in API process.
  - `strategy.live_worker.enabled=false` => live worker process logs "disabled" and exits 0.
  - `strategy.realtime_output.redis_streams.enabled=false` => live worker uses NoOp realtime output publisher.
  - `strategy.telegram.enabled=false` => live worker uses NoOp telegram notifier.
- Implement `strategy.metrics.port` with precedence: CLI `--metrics-port` override wins over YAML.
- Implement env overrides for scalar keys (whitelist only):
  - toggles and metrics port (per doc recommended env keys).
- Keep secrets out of YAML; YAML may reference env var names only (e.g. `TELEGRAM_BOT_TOKEN`, `ROEHUB_REDIS_PASSWORD`).
- Keep backward compatibility:
  - legacy full `strategy_live_runner.yaml` payload still loads correctly.
  - shim `strategy_live_runner.yaml` loads via reference to `strategy.yaml`.
- Update `__init__.py` exports where new public config types/loaders are introduced.
- Add/update unit tests as needed.
- Ensure docstrings for new/changed config classes/protocols include `Docs:` and `Related:` references to concrete files.
- Run and pass quality gates.
- If any `.md` file is changed/added during implementation, run `python -m tools.docs.generate_docs_index` and keep index consistent.

## Requirements (Should)

- Minimize churn: reuse existing config dataclasses where appropriate and do not break existing imports.
- Keep deterministic error messages (stable setting paths in error strings) to avoid flaky tests and improve operator UX.
- Use strict, predictable parsing for boolean env overrides (1/0/true/false/yes/no/on/off).

## Requirements (Nice-to-have)

- Provide a small helper module for scalar env override resolution that can be reused across contexts.
- Add focused tests for precedence rules (CLI vs env vs YAML).

# Required reading (do first)

1) `.codex/AGENTS.md`
2) `docs/architecture/strategy/strategy-runtime-config-v1.md`
3) `docs/runbooks/strategy-live-worker.md`
4) `src/trading/contexts/strategy/adapters/outbound/config/live_runner_runtime_config.py`
5) `apps/worker/strategy_live_runner/main/main.py`
6) `apps/worker/strategy_live_runner/wiring/modules/strategy_live_runner.py`
7) `apps/api/main/app.py`
8) `apps/api/wiring/modules/strategy.py`
9) `src/trading/platform/config/indicators_compute_numba.py`
10) `tests/unit/contexts/strategy/adapters/test_strategy_live_runner_runtime_config.py`
11) `src/trading/contexts/market_data/application/ports/stores/*.py`

# Work plan (agent should follow)

1) Read required docs/files and map current Strategy config/wiring behavior.
2) Define `strategy.yaml` schema v1 and implement config dataclasses + fail-fast loader.
3) Implement config path resolution (`ROEHUB_STRATEGY_CONFIG` + `ROEHUB_ENV` fallback) and scalar env overrides (whitelist).
4) Add new `configs/<env>/strategy.yaml` files and convert `configs/<env>/strategy_live_runner.yaml` into shim format.
5) Update live worker wiring to:
   - exit 0 when disabled,
   - read metrics port from config (CLI override wins),
   - respect realtime_output and telegram toggles.
6) Update Strategy API app wiring to respect `strategy.api.enabled`.
7) Update `load_strategy_live_runner_runtime_config` to support shim `config_ref.path` while preserving legacy payload support.
8) Update `__init__.py` exports and any related docs references if needed.
9) Add/update unit tests covering:
   - `strategy.yaml` loader parsing + validation,
   - shim behavior,
   - toggle semantics for API/router inclusion and worker exit behavior,
   - precedence: CLI vs env vs YAML.
10) Run quality gates and fix issues. Run docs index tool only if `.md` changed.

# Acceptance criteria (Definition of Done)

- `configs/dev/strategy.yaml`, `configs/test/strategy.yaml`, `configs/prod/strategy.yaml` exist and validate.
- `configs/*/strategy_live_runner.yaml` works as shim/alias and legacy payload still works.
- Strategy API does not expose Strategy routes when `strategy.api.enabled=false`.
- Strategy live worker exits 0 when `strategy.live_worker.enabled=false`.
- `strategy.metrics.port` exists; CLI `--metrics-port` override still works.
- Env overrides work for the whitelisted scalar keys and do not allow arbitrary nested overrides.
- Unit tests cover the loader, shim, and toggle semantics.
- Quality gates pass:
  - `uv run ruff check .`
  - `uv run pyright`
  - `uv run pytest -q`

# Implementation constraints

## Determinism & ordering

- Ensure deterministic config parsing, defaults, and error messages.
- Keep deterministic ordering of any generated dicts/lists used in error reporting.

## API / contracts (if relevant)

- Do not break existing public imports.
- Do not change Strategy v1 runtime semantics outside what is required by STR-EPIC-06.

## Documentation

- New/changed config classes must include docstrings with `Docs:` and `Related:` links.
- If any `.md` is changed/added, run `uv run python -m tools.docs.generate_docs_index`.

## Tests

- Tests must be deterministic and not depend on real time or network.
- Prefer unit tests; integration tests are optional and must be explicit.

# Files to indicate (expected touched areas)

- `configs/*/strategy.yaml`
- `configs/*/strategy_live_runner.yaml`
- `src/trading/contexts/strategy/adapters/outbound/config/*.py`
- `apps/worker/strategy_live_runner/main/main.py`
- `apps/worker/strategy_live_runner/wiring/modules/strategy_live_runner.py`
- `apps/api/main/app.py`
- `apps/api/wiring/modules/strategy.py`
- `tests/unit/**`

# Non-goals

- Hot reload for configs.
- Renaming/migrating env vars to new DSN keys.
- Adding Prometheus metrics endpoint to the API process.

# Quality gates (must run and pass)

- `uv run ruff check .`
- `uv run pyright`
- `uv run pytest -q`
- `uv run python -m tools.docs.generate_docs_index --check` (only when any `.md` changed)

# Final output: report format (strict)

Your final message MUST be in `ru` and follow exactly:

1) **Результат**

2) **Изменённые файлы**

3) **Ключевые решения**

4) **Как проверить**

5) **Риски / что дальше**

Additionally, explicitly mention:

- Whether `.codex/AGENTS.md` was found and read.
- Whether docs index regeneration was needed and executed.
