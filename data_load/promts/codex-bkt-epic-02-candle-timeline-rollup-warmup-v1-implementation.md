---
# =========================
# Prompt Front-Matter (YAML)
# =========================
prompt_name: codex-bkt-epic-02-candle-timeline-rollup-warmup-v1-implementation
repo: Dejetins/roehub.com
branch: main
scope: "Implement BKT-EPIC-02 end-to-end: backtest candle timeline builder (minute normalization + best-effort rollup + carry-forward + warmup lookback) with deterministic behavior and unit tests"
language:
  implementation: python
  agent_report: ru
primary_docs:
  - docs/architecture/backtest/backtest-candle-timeline-rollup-warmup-v1.md
  - docs/architecture/backtest/backtest-bounded-context-domain-use-case-skeleton-v1.md
  - docs/architecture/roadmap/milestone-4-epics-v1.md
  - docs/architecture/shared-kernel-primitives.md
  - docs/architecture/indicators/indicators-candlefeed-acl-dense-timeline-v1.md
  - docs/architecture/api/api-errors-and-422-payload-v1.md
  - .codex/AGENTS.md
style_references:
  - src/trading/contexts/market_data/application/ports/stores/canonical_candle_reader.py
  - src/trading/contexts/market_data/application/ports/stores/enabled_instrument_reader.py
  - src/trading/contexts/market_data/application/ports/stores/raw_kline_writer.py
  - src/trading/shared_kernel/primitives/timeframe.py
  - src/trading/contexts/indicators/adapters/outbound/feeds/market_data_acl/market_data_candle_feed.py
  - src/trading/contexts/strategy/application/services/timeframe_rollup.py
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
  implement_backtest_candle_timeline_service: true
  normalize_user_time_range_to_minute_for_1m_load: true
  rollup_is_best_effort_v1: true
  empty_bucket_policy_is_carry_forward: true
  derived_tf_outputs_must_not_contain_nan: true
  warmup_bars_is_in_target_timeframe: true
  target_slice_is_by_bar_close_ts: true
  use_indicators_candle_feed_for_1m_dense: true
  do_not_change_indicators_candle_feed_contract: true
  update_docs_index_if_any_md_changed: true
target_envs:
  - dev
  - test
  - prod
required_literals:
  - warmup_bars_default
  - "200"
non_goals:
  - "Do not implement staged grid builder (BKT-EPIC-04)."
  - "Do not implement execution engine / trades / positions (BKT-EPIC-05)."
  - "Do not implement reporting metrics (BKT-EPIC-06)."
  - "Do not add API routes or wiring for /backtests (BKT-EPIC-07)."
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
  - cmd: "python -m tools.docs.generate_docs_index --check"
    expect:
      - "exit code 0"
      - "run only if any .md file was added/changed"
expected_touched_paths:
  - src/trading/contexts/backtest/application/services/**
  - src/trading/contexts/backtest/application/use_cases/run_backtest.py
  - src/trading/contexts/backtest/domain/errors/**
  - tests/unit/contexts/backtest/**
  - docs/architecture/roadmap/milestone-4-epics-v1.md
  - docs/architecture/backtest/backtest-candle-timeline-rollup-warmup-v1.md
safety_notes:
  - "Read .codex/AGENTS.md first. If it conflicts with this prompt, this prompt has priority."
  - "Keep behavior deterministic: no random, no time.now() in pure builders, stable sorting only."
---

# Task

Implement the architecture spec `docs/architecture/backtest/backtest-candle-timeline-rollup-warmup-v1.md` fully.

Done means the repository contains a deterministic, unit-tested candle timeline builder for backtest v1 that:

1) Accepts user-provided `Start/End` without requiring timeframe alignment.
2) Normalizes the internal 1m load range to minute boundaries, including warmup lookback.
3) Loads dense 1m candles via indicators `CandleFeed.load_1m_dense(...)` (canonical source-of-truth is still `CanonicalCandleReader` via ACL).
4) Builds derived timeframe candles (`5m/15m/1h/4h/1d`) using epoch-aligned buckets.
5) Applies best-effort rollup (missing minutes do not drop the bucket) and ensures derived outputs contain NO NaN.
6) Implements empty-bucket carry-forward (`OHLC=prev_close`, `volume=0`) and raises a deterministic validation error when there is no market data at all.
7) Exposes helpers to compute the target report slice `[Start, End)` using `bar_close_ts` semantics.

Your final report MUST be written in Russian.

## Context / Current State

- Backtest bounded context exists from BKT-EPIC-01 (`src/trading/contexts/backtest/**`).
- Backtest use-case skeleton currently loads 1m candles directly using the request time_range.
- Indicators CandleFeed contract is strict about alignment to 1m duration and produces dense 1m arrays with NaN holes.
- Shared kernel Timeframe supports epoch-aligned bucket_open/bucket_close.

## Requirements (Must)

- Follow `.codex/AGENTS.md` and the primary doc `docs/architecture/backtest/backtest-candle-timeline-rollup-warmup-v1.md` exactly.
- Implement a backtest application service (or pure functions) under `src/trading/contexts/backtest/application/services/` responsible for:
  - minute normalization (floor/ceil) for internal 1m loading
  - best-effort rollup
  - carry-forward empty bucket policy
  - target slice computation by bar close timestamps
- Do NOT modify indicators CandleFeed behavior/contract; consume it.
- Derived candle arrays MUST NOT contain NaN.
- Use `Timeframe.bucket_open/bucket_close` (epoch-aligned) for bucket boundaries.
- Warmup lookback uses `warmup_bars` in units of the target timeframe.
- Ensure errors are mapped to deterministic `RoehubError` (422 `validation_error`) using existing backtest error mapping pattern.
- Add unit tests covering:
  - start/end minute normalization
  - best-effort rollup with missing minutes inside a bucket
  - carry-forward behavior for empty buckets
  - deterministic error when no market data exists (no prev_close in entire range)
  - target slice boundaries by `bar_close_ts`
- Keep stable exports via `__init__.py` if you introduce new services.
- Docstrings in new classes/functions MUST link to the primary doc and related files.

## Requirements (Should)

- Update `RunBacktestUseCase` to use the new candle timeline builder for its candle loading path.
- Keep rollup math vectorized where reasonable but prioritize clarity and determinism.
- Ensure datetime and epoch-millis conversions never use floating point rounding.

## Requirements (Nice-to-have)

- Provide small deterministic stubs for CandleFeed in unit tests to avoid heavyweight mocks.

# Required reading (do first)

1) `.codex/AGENTS.md`
2) `docs/architecture/backtest/backtest-candle-timeline-rollup-warmup-v1.md`
3) `docs/architecture/indicators/indicators-candlefeed-acl-dense-timeline-v1.md`
4) `docs/architecture/shared-kernel-primitives.md`
5) `src/trading/shared_kernel/primitives/timeframe.py`
6) `src/trading/contexts/indicators/adapters/outbound/feeds/market_data_acl/market_data_candle_feed.py`
7) `docs/architecture/api/api-errors-and-422-payload-v1.md`

# Work plan (agent should follow)

1) Inspect the current backtest use-case and existing CandleFeed/Timeframe contracts.
2) Implement backtest candle timeline builder service (minute normalization + best-effort rollup + carry-forward).
3) Add target-slice helper by `bar_close_ts` and document its semantics in docstrings.
4) Wire the builder into `RunBacktestUseCase` candle loading path (minimal change, still a skeleton use-case).
5) Add/adjust domain errors if needed and ensure they map to deterministic RoehubError.
6) Add unit tests for each deterministic rule.
7) Run ruff/pyright/pytest; fix all failures.
8) If any `.md` was changed, run `python -m tools.docs.generate_docs_index` and ensure `--check` passes.

# Acceptance criteria (Definition of Done)

- A backtest candle timeline builder exists under `src/trading/contexts/backtest/application/services/` and matches the doc.
- Derived timeframe outputs are deterministic and contain no NaN.
- Empty bucket carry-forward behavior is implemented and tested.
- When no market data exists at all, a deterministic 422 validation error is produced.
- `RunBacktestUseCase` uses the builder for candle loading.
- `uv run ruff check .`, `uv run pyright`, `uv run pytest -q` pass.

# Implementation constraints

## Determinism & ordering

- Use stable sorts only where needed.
- Canonical epoch-millis conversions MUST avoid floats.

## API / contracts (if relevant)

- Do NOT add any FastAPI routes or wiring changes.
- Errors must use existing RoehubError codes (422 validation_error for timeline/rollup issues).

## Documentation

- Do not change docs semantics. If you must edit any `.md`, run docs index tool.

## Tests

- Unit tests only (no ClickHouse/Postgres/Redis).

# Files to indicate (expected touched areas)

- `src/trading/contexts/backtest/application/services/**`
- `src/trading/contexts/backtest/application/use_cases/run_backtest.py`
- `tests/unit/contexts/backtest/**`

# Non-goals

- No staged runner, no execution engine, no reporting metrics, no API routes.

# Quality gates (must run and pass)

- `uv run ruff check .`
- `uv run pyright`
- `uv run pytest -q`
- `python -m tools.docs.generate_docs_index --check`

# Final output: report format (strict)

Your final message MUST be in Russian and follow exactly:

1) **Результат**

2) **Изменённые файлы**

3) **Ключевые решения**

4) **Как проверить**

5) **Риски / что дальше**
