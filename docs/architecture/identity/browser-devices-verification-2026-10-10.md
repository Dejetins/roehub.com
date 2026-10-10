# Browser-device sessions — local verification, 2026-10-10

Scope: one account/browser-profile device row across repeated authentication,
measured session activity, owner-only revocation, explicit legacy grouping and
Navigator Security UI. The normative behavior and compatibility classification are
in [local auth sessions](local-auth-sessions-recovery-v1.md#устройства-и-управление-входами-2026-10-10).

## Checks

- Python unit checks: **74 passed** across `test_browser_devices.py`, identity
  route/current-user tests, `tests/unit/contexts/identity`, and storage lifecycle
  tests. Includes same-browser repeated password login, owner isolation, CSRF,
  bearer omission, current/legacy revocation, and in-memory touch/revoke race.
- Frontend: **11 passed** in `session-devices.test.tsx` and
  `settings-page.test.tsx`; typecheck and production build passed. Vite reports
  the existing large bundle warning; this is not a bundle-size acceptance claim.
- Focused Ruff and Pyright passed for changed production Python sources; changed
  tests passed Ruff. Relevant tracked diff passed whitespace checks.
- Real PostgreSQL: **3 passed**, using a disposable schema created and removed by
  `tests/runtime/test_browser_devices_postgres.py`. Eight concurrent registrations
  leave one device and one active credential. Also checked owner isolation,
  touch without TTL extension, revoked-session protection, legacy revocation,
  pagination, and non-destructive migration reapplication.
- Browser at `http://localhost:20120/settings/security`: observed one current
  `Chrome · macOS` device, explicit current marker, measured UTC activity, and one
  group of ten unknown earlier sign-ins. Manual refresh retained the list;
  ArrowRight/Home switched Activity/Devices with focus and correct content.
  Existing user sessions were not revoked for browser QA. Revocation behavior is
  covered by API/repository tests, not a destructive action on the user's account.
- Independent security review found and then rechecked two fixes: all in-memory
  writers now share the same lock; mutation denials clear private snapshots/cache.
  Final review found no further blocking findings in this scope.

## Local migration limitation

The populated preview database lacked complete historical storage-lifecycle
markers. The full bootstrap stopped at the existing greenfield guard in phase
`organization-0011` (`strategy_strategies` is nonempty). Earlier idempotent
baseline phases completed before that guard; the database was not reset.

Only the new additive `0032` migration was then applied through
`apply_identity_browser_devices_sql`, under the existing PostgreSQL advisory
lock, with its manifest checksum recorded. No unapplied historical phase was
marked as applied. API and web were restarted through the local control-agent
lifecycle, after the schema change. The new screen was verified against that
runtime. This does **not** certify full bootstrap/installation readiness or fix
historical marker drift. That drift needs a separate evidenced reconciliation,
not fabricated migration receipts. No production release or Git publication was
performed.
