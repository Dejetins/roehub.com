# Local preview session lifetime — 2026-09-08

## Authority, cause and scope

The owner requested removing the short session restriction on the current local
Backtests service. The local API inherited identity defaults: 1800-second idle
expiry and 43200-second absolute expiry. PostgreSQL showed exactly those durations
for all four existing owner sessions (all expired, none revoked). The current-user
resolver reads `is_active_at` without updating the expiry, so even continuous work
cannot extend the initial 30-minute deadline. Cause confirmed from persisted
metadata and the resolver/repository path, not inferred from a browser screenshot.

Changed only the disposable preview factory in
`tools/qa/backtests_client_fixture.py`: default idle and absolute TTL are now
31536000 seconds (365 days). Explicit test TTLs still override these defaults;
the helper refuses production environment use. Production identity settings,
DTOs, persisted schema, cookie flags, CSRF and logout/revocation are unchanged.
This removes the short timeout, not all possible future expiry. Previously
expired sessions were not resurrected; the owner must sign in once again.

Compatibility: local session/cookie lifetime is `compatible-change`, explicitly
selected by the owner; production defaults and auth/revocation contracts are
`none`. Longer local session validity is the selected tradeoff. No external or
production runtime target is involved.

## Verification

- `.venv/bin/python -m pytest -q tests/unit/tools/test_backtests_preview_sessions.py tests/unit/apps/api/test_identity_wiring_module.py tests/unit/apps/api/test_identity_current_user_dependency.py`: **26 passed**.
- `.venv/bin/ruff check tools/qa/backtests_client_fixture.py tests/unit/tools/test_backtests_preview_sessions.py`: passed.
- New deterministic test: session remains active after 31 minutes, 13 hours and
  30 days; explicit revoke still ends it. Also verifies explicit test TTLs and
  production rejection. No pre-fix red test run; pre-fix database TTLs establish
  the baseline.
- Real local API/Web proxy verification with a separate disposable login:
  login 200; both auth/CSRF cookie Max-Age values 31536000; persisted idle and
  absolute durations 31536000; the test session aged by two days still returns
  current-user 200; logout 204; subsequent current-user 401.
- Web 18480, API 18481, SSR 18482 and worker 18483 health/metrics all 200.
  Real HTTP checks prove API/proxy/cookie behavior, not browser UI interaction.

No credential, session identifier, token, cookie value or DSN is stored here.
The separate verification session was logged out. The user's expired sessions
were not changed and no browser credentials were entered.

## Runtime application

API composition was checked on temporary loopback port 18484 first, then replaced
on 18481. Existing PostgreSQL, ClickHouse, Web, SSR, runner and filesystem artifacts
were retained. Backtest job count stayed identical (0 before/after).

The original disposable supervisor would delete the database containers whenever
any child exited. It was stopped and retired without invoking that teardown.
Services now run independently; their PID/container inventory is in private local
`.local_artifacts/backtests-preview-processes.json`, with API PID in
`.local_artifacts/backtests-client/api.pid`. The former supervisor PID file was
removed. Do not use the old supervisor lifecycle to restart or clean this retained
stack. Source `run_stack` continues to create fresh disposable stacks normally.

Focused self-review: preview-only composition scope, explicit override behavior,
unchanged production defaults, retained revoke/CSRF and preserved data verified.
