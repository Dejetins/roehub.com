# Stage 05: Stats Query Service Day Week Month

Дата: `2026-06-29`

Статус: `accepted`

Acceptance boundary: Stage `05` добавляет provider-neutral `NotificationStatsQueryService` для portfolio, strategy и exchange stats snapshots за periods `today`, `week`, `month`. Stage accepted after implementation commit `275be8702f19b06fe03e057cd719eafff25cbaf3` was published to `main`, GitHub CI/deploy passed, `macstudio` checkout synchronized to the same commit and production smoke passed.

## User Required Before Start

Nothing.

No Telegram token, chat id, admin route, password, cookie or provider payload was required or printed.

## Scope

Implemented:

- `NotificationStatsQueryService` with portfolio, strategy and exchange scopes;
- explicit period metadata: `period`, `period_start`, `period_end`, `timezone`, `generated_at`;
- explicit quality statuses: `complete`, `partial`, `unavailable`;
- explicit `missing_sources`, `latest_source_at` and `freshness_seconds`;
- source counters for signals, fills, orders, balances, positions and open orders;
- PnL fields only when paper accounting source declares `pnl_complete=True`;
- `NotificationStatsSourceReader` ACL port and in-memory seeded source reader for fixture/database-style tests;
- Telegram command integration so `/stats`, `/strategy` and `/exchange` can render stats snapshots when a stats service is configured.

Not implemented in this stage:

- direct SQL reader over production ledgers;
- scheduled report lifecycle;
- browser/API stats endpoints;
- production Telegram canary.

## Stats Fields

| Field group | Fields |
|---|---|
| Scope | `owner_user_id`, `scope_kind`, `scope_ref` |
| Period | `period`, `timezone`, `period_start`, `period_end`, `generated_at` |
| Quality | `quality_status`, `missing_sources`, `latest_source_at`, `freshness_seconds` |
| Activity | `signal_count`, `fill_count`, `order_count` |
| Exchange projection | `balance_count`, `position_count`, `open_order_count` |
| Money fields | `realized_pnl`, `unrealized_pnl`, `fee_total`, `funding_total`, `equity` |

## Quality Behavior

| Scenario | Result |
|---|---|
| All required sources present and accounting is `pnl_complete=True` | `complete` |
| Some sources are missing or unavailable | `partial` |
| No source rows after owner/scope/period filtering | `unavailable` |
| Accounting exists but `pnl_complete=False` | PnL fields are `None`; quality includes `pnl_complete_accounting` as missing |
| Strategy/exchange belongs to another owner | filtered out; returns `unavailable`, not leaked counts |

## Real Boundary Evidence

Local seeded ACL-reader smoke executed with no external provider:

| Evidence | Result |
|---|---|
| Portfolio day stats | `quality=complete`, day window starts `2026-06-29` |
| Strategy week stats | `quality=complete`, week window starts `2026-06-29` |
| Foreign strategy month stats | `quality=unavailable`, owner rows filtered out |
| Smoke line | `stage05_stats_smoke=ok portfolio_quality=complete strategy_quality=complete foreign_quality=unavailable day_start=2026-06-29 week_start=2026-06-29 month_start=2026-06-01 owner_filtered=True` |

## Validation

| Check | Result |
|---|---|
| `uv run pytest -q tests/unit/contexts/notifications` | passed: `37 passed` |
| `uv run ruff check src/trading/contexts/notifications tests/unit/contexts/notifications` | passed |
| `uv run pyright src/trading/contexts/notifications tests/unit/contexts/notifications` | passed |
| Seeded stats ACL smoke | passed: `stage05_stats_smoke=ok ... owner_filtered=True` |
| `uv run python -m tools.docs.generate_docs_index --check` | local check failed because the dirty checkout contains unrelated untracked `market-data-live-tail-repair-v1` docs; generated diff inspection showed the Stage `05` README entry matches the generator and only those unrelated market-data entries remain missing |
| GitHub CI for implementation commit `275be8702f19b06fe03e057cd719eafff25cbaf3` | passed: run `28395374117` |
| GitHub deploy/image for implementation commit `275be8702f19b06fe03e057cd719eafff25cbaf3` | passed: Backend `28395593695`, Web `28395593653`, App Image `28395593623` |
| `macstudio` checkout sync | passed: `/Users/daniildegtyarev/Projects/roehub.com` fast-forwarded to `275be8702f19b06fe03e057cd719eafff25cbaf3` |
| `macstudio` production smoke | passed: `cd /opt/roehub/app && bash scripts/macos/smoke_prod.sh` exited `0` |

## Contract Impact

| Surface | Classification | Notes |
|---|---|---|
| Public API | `none` | No HTTP route changed. |
| DTO schema | `none` | No API DTO changed. |
| Ports | `compatible-change` | Adds stats source reader ACL port and query service. |
| Persisted schema | `none` | No migration changed. |
| Config/defaults | `none` | No runtime config changed. |
| External service calls | `none` | No network/provider call added. |
| External side effects | `compatible-change` | Telegram command handler can render stats if injected with a service; without it previous unavailable behavior remains. |
| Browser-visible behavior | `none` | No UI changed. |
| Performance | `unknown` | Source reader is fixture/in-memory in this stage; production SQL query cost is not benchmarked. |

## File Manifest

| Path | Action | Reason | Contract impact |
|---|---|---|---|
| `src/trading/contexts/notifications/application/stats_query.py` | created | Add stats snapshot query service, source reader port and renderer. | `compatible-change` application surface |
| `src/trading/contexts/notifications/adapters/outbound/acl/in_memory_stats_source_reader.py` | created | Add seeded ACL reader for tests and smoke. | `none` production |
| `src/trading/contexts/notifications/adapters/outbound/acl/__init__.py` | created | Export ACL adapter package. | `none` |
| `src/trading/contexts/notifications/adapters/outbound/__init__.py` | modified | Export seeded stats ACL reader. | `compatible-change` adapter export |
| `src/trading/contexts/notifications/adapters/__init__.py` | modified | Export seeded stats ACL reader. | `compatible-change` adapter export |
| `src/trading/contexts/notifications/application/__init__.py` | modified | Export stats query service/types. | `compatible-change` application export |
| `src/trading/contexts/notifications/__init__.py` | modified | Export stats query service/types. | `compatible-change` context export |
| `src/trading/contexts/notifications/application/telegram_commands.py` | modified | Render stats snapshots for bound commands when stats service is configured. | `compatible-change` command behavior |
| `tests/unit/contexts/notifications/test_stats_query.py` | created | Cover day/week/month, quality behavior, owner filters and rendering. | `none` |
| `tests/unit/contexts/notifications/test_telegram_commands.py` | modified | Cover command integration with stats service. | `none` |
| `docs/architecture/notifications/web-execution-telegram-notifications-v1-stage-reports/05-stats-query-service-day-week-month.md` | created | Stage `05` report. | `none` |
| `docs/architecture/notifications/web-execution-telegram-notifications-v1-stage-reports/web-execution-telegram-notifications-v1-stage-ledger.md` | modified | Record Stage `05` local implementation and evidence. | `none` |
| `docs/architecture/README.md` | modified | Add Stage `05` report to docs index. | `none` |

## Residual Risks

- Production SQL-backed stats reader remains future work before real user stats can be enabled outside injected fixtures.
- Stage `05` does not claim complete PnL from incomplete ledgers; quality falls back to `partial` or `unavailable`.
- Scheduled reports still require Stage `06`.
