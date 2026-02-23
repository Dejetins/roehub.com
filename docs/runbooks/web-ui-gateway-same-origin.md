# Web UI Gateway Same-Origin (WEB-EPIC-02)

Runbook for local and server startup of the same-origin `web + api + gateway` stack.

## Required environment file

Use the same env-file pattern as deployment:

- `/etc/roehub/roehub.env` on servers
- local equivalent path (example: `./infra/docker/.env.local`)

Minimum keys for UI profile:

- `POSTGRES_PASSWORD`
- `IDENTITY_PG_DSN`
- `POSTGRES_DSN`
- `STRATEGY_PG_DSN`
- `WEB_API_BASE_URL`
- `TELEGRAM_BOT_TOKEN`

Reference placeholders:

- `infra/docker/.env.example`

## One-command dev start

```bash
docker compose -f infra/docker/docker-compose.yml \
  --env-file /etc/roehub/roehub.env \
  --profile ui up -d --build
```

Expected endpoint:

- `http://127.0.0.1:8080`

Quick checks:

```bash
docker compose -f infra/docker/docker-compose.yml \
  --env-file /etc/roehub/roehub.env \
  --profile ui ps

curl -i http://127.0.0.1:8080/api/auth/current-user
curl -i http://127.0.0.1:8080/assets/site.css
```

## DB bootstrap behavior

`db-bootstrap` runs before `api` in UI profile and executes:

1. `python -m apps.migrations.bootstrap_main`
2. Identity SQL baseline in `IDENTITY_PG_DSN`:
   - apply `0001_identity_v1.sql`
   - apply `0002_identity_2fa_totp_v1.sql`
   - apply `0003_identity_exchange_keys_v1.sql`
3. Guarded `0004_identity_exchange_keys_v2.sql`:
   - skip if v2 columns already exist
   - apply only if v1 layout exists and table is empty
   - fail fast if v1 layout has rows (unsafe migration path)
4. Alembic head in `POSTGRES_DSN` via existing runner:
   - `python -m apps.migrations.main --dsn "$POSTGRES_DSN"`

The service is one-shot (`restart: "no"`). If bootstrap fails, `api` does not start.

## Telegram Login Widget domain

Production:

1. Open `@BotFather`.
2. Run `/setdomain`.
3. Set domain to `roehub.com`.

Development:

1. Expose gateway `127.0.0.1:8080` through a tunnel (`cloudflared` or `ngrok`).
2. Set tunnel domain in `@BotFather /setdomain`.
3. Open login page via tunnel URL.

Trade-off:

- one bot can have only one active domain, so using the production bot for dev tunnel can break prod
  login widget
- recommended: separate staging/dev bot for local tunnel testing

## Troubleshooting: "bot domain invalid"

- Ensure browser host exactly matches BotFather domain (no extra subdomain or port).
- Ensure you open login page via `https` URL from tunnel.
- Re-run `/setdomain` and wait up to a few minutes for Telegram-side propagation.
- Confirm widget uses the expected bot username.

## Health routing note

Gateway strips `/api` and proxies to API upstream:

- `/api/<path>` at gateway -> `/<path>` at API.

If API later adds `/health`, it will be reachable as `/api/health` through gateway.
