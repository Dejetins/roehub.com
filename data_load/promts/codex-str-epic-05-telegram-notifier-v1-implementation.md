---
# =========================
# Prompt Front-Matter (YAML)
# =========================
prompt_name: codex-str-epic-05-telegram-notifier-v1-implementation
repo: Dejetins/roehub.com
branch: main
scope: "Implement STR-EPIC-05 end-to-end: Strategy Telegram notifier v1 with best-effort delivery, identity chat binding ACL, and notification policy"
language:
  implementation: python
  agent_report: ru
primary_docs:
  - docs/architecture/strategy/strategy-telegram-notifier-best-effort-policy-v1.md
  - docs/architecture/strategy/strategy-live-runner-redis-streams-v1.md
  - docs/architecture/strategy/strategy-realtime-output-redis-streams-v1.md
  - docs/architecture/roadmap/milestone-3-epics-v1.md
  - docs/_templates/architecture-doc-template.md
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
  prefer_log_only_in_dev_test: true
  require_telegram_mode_in_prod_when_enabled: true
  implement_identity_acl_for_confirmed_chat_binding: true
  implement_failed_event_debounce_seconds: 600
  send_plain_text_english_messages: true
  update_docs_index_if_md_changed: true
target_envs:
  - dev
  - test
  - prod
required_literals:
  - TelegramNotifier
  - signal
  - trade_open
  - trade_close
  - failed
  - log_only
  - telegram
  - TELEGRAM_BOT_TOKEN
  - identity_telegram_channels
non_goals:
  - "Do not implement guaranteed delivery or exactly-once notifications"
  - "Do not add localization or advanced templating"
  - "Do not introduce distributed debounce/rate-limiting in v1"
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
  - src/trading/contexts/strategy/application/ports/*.py
  - src/trading/contexts/strategy/application/services/*.py
  - src/trading/contexts/strategy/adapters/outbound/**/telegram/*.py
  - src/trading/contexts/strategy/adapters/outbound/**/identity/*.py
  - src/trading/contexts/strategy/adapters/outbound/config/live_runner_runtime_config.py
  - apps/worker/strategy_live_runner/wiring/modules/strategy_live_runner.py
  - configs/dev/strategy_live_runner.yaml
  - configs/test/strategy_live_runner.yaml
  - configs/prod/strategy_live_runner.yaml
  - tests/unit/contexts/strategy/**/*.py
  - docs/architecture/**/*.md
  - src/trading/contexts/strategy/**/__init__.py
safety_notes:
  - "If .codex/AGENTS.md conflicts with this prompt, this prompt has priority."
  - "Do not remove or silently change existing Strategy v1 contracts unrelated to STR-EPIC-05."
---

# Task

Implement the architecture document `docs/architecture/strategy/strategy-telegram-notifier-best-effort-policy-v1.md` fully and consistently across Strategy application ports, outbound adapters, runtime config, worker wiring, and tests.

Done means:
- Telegram notifications are emitted for Strategy events `signal`, `trade_open`, `trade_close`, and `failed`.
- Delivery is best-effort: notification failures never break live-runner processing.
- Chat ID is resolved through an identity ACL/port and only confirmed identity binding is used.
- Debounce is applied to repeated `failed` notifications (default 600s).
- `log_only` mode works in dev/test; real Telegram mode works in prod when identity chat binding is confirmed.
- Code quality gates pass (ruff, pyright, pytest).

## Context / Current State

- Strategy live runner already exists and currently publishes realtime output to Redis Streams.
- Identity already has Telegram channel binding data (`identity_telegram_channels`) including confirmation status.
- There is no complete Strategy-side Telegram notifier v1 implementation yet (port/policy/adapters/wiring/tests).
- Runtime config and wiring patterns already exist for Redis stream adapters and should be mirrored for notifier integration.
- Existing codebase uses strict docstring style with links to docs and related files in classes/protocols.

## Requirements (Must)

- Read `.codex/AGENTS.md` before any implementation work.
- If `.codex/AGENTS.md` conflicts with this prompt, follow this prompt.
- Implement Strategy port `TelegramNotifier` and any required supporting protocol(s)/DTO(s).
- Implement identity ACL/port to resolve confirmed `chat_id` by `user_id`.
- Implement notification policy for event filtering and failed-error debounce.
- Integrate notifier into Strategy live-runner flow as best-effort side effect.
- Ensure best-effort semantics (exceptions never break pipeline iteration).
- Add/update runtime config model and YAML configs for dev/test/prod notifier modes.
- Keep fail-fast startup behavior for invalid required production notifier configuration.
- Keep deterministic behavior and stable ordering where ordering matters.
- Update `__init__.py` exports for any new public types/adapters.
- Add/update unit tests for policy, adapters, wiring, and live-runner integration.
- Add docstring references (`Docs:` and `Related:`) in new/changed classes and protocols.
- Ensure `ruff`, `pyright`, and `pytest` pass.
- If any `.md` docs are changed/added, run `python -m tools.docs.generate_docs_index` and keep index consistent.

## Requirements (Should)

- Follow existing naming and structure conventions used in Strategy and Market Data contexts.
- Keep messages plain text (English) and concise.
- Add observability hooks/metrics counters where natural and consistent with existing patterns.
- Make debounce key deterministic and explicit.

## Requirements (Nice-to-have)

- Reuse existing best-effort adapter patterns (e.g., no-op/log-only fallback shape).
- Keep adapters test-friendly via dependency injection (clock/http client/repositories).
- Provide clear inline validation errors for misconfiguration.

# Required reading (do first)

1) `.codex/AGENTS.md`
2) `docs/architecture/strategy/strategy-telegram-notifier-best-effort-policy-v1.md`
3) `docs/architecture/strategy/strategy-live-runner-redis-streams-v1.md`
4) `docs/architecture/strategy/strategy-realtime-output-redis-streams-v1.md`
5) `docs/architecture/roadmap/milestone-3-epics-v1.md`
6) `src/trading/contexts/strategy/application/services/live_runner.py`
7) `apps/worker/strategy_live_runner/wiring/modules/strategy_live_runner.py`
8) `src/trading/contexts/strategy/adapters/outbound/config/live_runner_runtime_config.py`
9) `migrations/postgres/0001_identity_v1.sql`
10) `src/trading/contexts/market_data/application/ports/stores/*.py`

# Work plan (agent should follow)

1) Read required docs/files and map existing Strategy runtime integration points.
2) Design and add Strategy notifier port(s), identity ACL contract, and notification DTOs.
3) Implement notification policy (event filtering + failed debounce).
4) Implement adapters:
   - log-only notifier,
   - Telegram Bot API notifier,
   - confirmed chat binding resolver via identity ACL.
5) Wire notifier into live-runner (best-effort side effect only).
6) Extend runtime config models + YAML configs (dev/test/prod).
7) Update module exports (`__init__.py`) and related docs references where necessary.
8) Add/update unit tests for:
   - policy,
   - adapters,
   - wiring/config validation,
   - live-runner integration semantics.
9) Run quality gates and fix all issues.
10) If any markdown docs changed, regenerate docs index and include resulting updates.

# Acceptance criteria (Definition of Done)

- Strategy emits notifier calls only for `signal`, `trade_open`, `trade_close`, `failed`.
- `failed` notifications are debounced for 600 seconds by deterministic key.
- Missing/unconfirmed chat binding never crashes pipeline; behavior is skip + warning/metric.
- In dev/test, notifier can run in `log_only` mode.
- In prod with notifier mode `telegram`, missing required config fails fast at startup.
- All new/changed classes/protocols include docstrings with docs/related file links.
- Public imports remain stable or are intentionally extended without breaking existing imports.
- Unit tests for new behavior exist and pass.
- `uv run ruff check .`, `uv run pyright`, and `uv run pytest -q` pass.
- If `.md` changed, docs index is updated and `uv run python -m tools.docs.generate_docs_index --check` passes.

# Implementation constraints

## Determinism & ordering

- Keep deterministic event handling and debounce key computation.
- Preserve any existing deterministic ordering in live-runner processing and adapter behavior.

## API / contracts (if relevant)

- Do not break current Strategy v1 contracts and public imports.
- New notifier/ACL contracts must be explicit, minimal, and version-appropriate for v1.

## Documentation

- New/changed classes and protocols must include `Docs:` and `Related:` references to concrete files.
- If new architecture docs are added/changed, run `python -m tools.docs.generate_docs_index`.

## Tests

- Add focused unit tests for each new component and integration point.
- Keep tests deterministic, isolated, and readable.

# Files to indicate (expected touched areas)

- `src/trading/contexts/strategy/application/ports/*.py`
- `src/trading/contexts/strategy/application/services/*.py`
- `src/trading/contexts/strategy/adapters/outbound/**/*.py`
- `apps/worker/strategy_live_runner/wiring/modules/strategy_live_runner.py`
- `configs/dev/strategy_live_runner.yaml`
- `configs/test/strategy_live_runner.yaml`
- `configs/prod/strategy_live_runner.yaml`
- `src/trading/contexts/strategy/**/__init__.py`
- `tests/unit/contexts/strategy/**/*.py`
- `docs/architecture/**/*.md`

# Non-goals

- Implementing delivery guarantees, retry queues, or exactly-once semantics.
- Implementing localization or rich message templating.
- Implementing distributed/global rate limiting for notifier events.

# Quality gates (must run and pass)

- `uv run ruff check .`
- `uv run pyright`
- `uv run pytest -q`
- `uv run python -m tools.docs.generate_docs_index --check` (only when any `.md` file changed)

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
