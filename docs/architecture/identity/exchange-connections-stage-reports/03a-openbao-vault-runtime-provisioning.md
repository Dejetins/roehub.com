# Stage 3A: OpenBao/Vault Runtime Provisioning

Дата проверки: 2026-05-24.

Статус: accepted for runtime provisioning. OpenBao Transit-compatible runtime,
Transit key, ACL policies/tokens, Monit, Prometheus and recovery runbook are
provisioned on Mac Studio before application Transit integration.

## Scope

Stage 3A provisions the secret backend only. It does not add
`ExchangeSecretCipher` application code, does not migrate `identity_exchange_keys`,
does not add Binance/Bybit validation and does not expose plaintext credentials
to `apps/api`.

## Runtime Evidence

| Runtime component | Command / call | Expected result | Observed result | Blocker |
|---|---|---|---|---|
| Branch gate | `test "$(git branch --show-current)" = main` | Work starts on `main`. | Passed: current branch is `main`. | None |
| Fast-forward gate | `git pull --ff-only origin main` | Local checkout can fast-forward from `origin/main`. | Passed before changes: `Already up to date.` | None |
| GitHub CLI | `gh --version && gh auth status` | CLI installed and authenticated for CI/deploy inspection. | Passed: `gh version 2.85.0`; authenticated as `Dejetins`. | None |
| OpenBao install | `brew info openbao`; `brew install openbao` | OpenBao is installed on target runtime. | Passed on Mac Studio: OpenBao `2.5.4` installed by Homebrew. | None |
| launchd runtime | `launchctl print gui/$(id -u)/com.roehub.openbao` | `com.roehub.openbao` is running. | Passed: `state = running`, PID `89539`, last exit code never exited. | None |
| Health | `curl -fsS "$OPENBAO_ADDR/v1/sys/health"` | Runtime is initialized and unsealed. | Passed on Mac Studio: `initialized=true`, `sealed=false`, version `2.5.4`. | None |
| Metrics endpoint | `curl -fsS "$OPENBAO_ADDR/v1/sys/metrics?format=prometheus"` | Prometheus-format metrics are accessible locally. | Passed: endpoint returned Go/OpenBao process metrics. | None |

## Transit ACL Evidence

| Runtime component | Command / call | Expected result | Observed result | Blocker |
|---|---|---|---|---|
| Transit mount | `bao secrets enable -path=transit transit` | `transit` secret engine exists. | Provisioning reported `transit_mount=enabled`. | None |
| Transit key | `bao write -f transit/keys/roehub-exchange-credentials` | Key exists without requiring metadata-read permission for app tokens. | Provisioning reported `transit_key=roehub-exchange-credentials`; encrypt call below proves usable key. | None |
| `exchange-control` policy | `bao policy write roehub-exchange-control-transit ...` | Service identity has scoped Transit policy. | Policy installed from `infra/macos/openbao/policies/roehub-exchange-control-transit.hcl`. | None |
| `apps/api` policy | `bao policy write roehub-api-transit-deny-decrypt ...` | API token exists but has no decrypt path. | Policy installed from `infra/macos/openbao/policies/roehub-api-transit-deny-decrypt.hcl`. | None |
| Encrypt allowed | `curl -fsS -X POST "$OPENBAO_ADDR/v1/transit/encrypt/roehub-exchange-credentials" -H "X-Vault-Token: $ROEHUB_EXCHANGE_CONTROL_TRANSIT_TOKEN" --data '{"plaintext":"U1RBR0UzQV9TTU9LRQ=="}'` | `exchange-control` token can encrypt. | Passed via sanitized smoke: `exchange_control_encrypt=ok`; response body discarded. | None |
| Decrypt denied | `curl -i -X POST "$OPENBAO_ADDR/v1/transit/decrypt/roehub-exchange-credentials" -H "X-Vault-Token: $ROEHUB_API_TRANSIT_TOKEN" --data '{"ciphertext":"vault:v1:stage3a-placeholder"}'` | `apps/api` token is denied decrypt. | Passed on Mac Studio: HTTP `403`; sanitized smoke returned `apps_api_decrypt_denied=403`. | None |

## Monitoring Evidence

| Runtime component | Command / call | Expected result | Observed result | Blocker |
|---|---|---|---|---|
| Monit config | `/opt/homebrew/opt/monit/bin/monit -c /opt/homebrew/etc/monitrc reload` | Monit loads `roehub_openbao`. | Passed: Monit reinitialized. | None |
| Monit summary | `/opt/homebrew/opt/monit/bin/monit -c /opt/homebrew/etc/monitrc summary \| grep -Ei "openbao|vault|transit|roehub"` | OpenBao supervision is visible. | Passed: `roehub_openbao OK`; existing Roehub services still listed. | None |
| Monit status | `monit status roehub_openbao` | Service is monitored and starts on reboot. | Passed: `status OK`, `monitoring status Monitored`, `on reboot start`, PID `89539`. | None |
| Prometheus config | `/api/v1/status/config` | Config contains `job_name: openbao`. | Passed: Prometheus loaded `openbao` job targeting `127.0.0.1:8200`. | None |
| Prometheus query | `curl -fsS 'http://127.0.0.1:9090/api/v1/query?query=up{job="openbao"}'` | OpenBao target is up. | Passed: `up{job="openbao",instance="127.0.0.1:8200"} = 1`. | None |

## Recovery And Rollback

| Runtime component | Command / call | Expected result | Observed result | Blocker |
|---|---|---|---|---|
| Runbook | `docs/runbooks/exchange-secret-management.md` | Recovery, backup/restore, token rotation, key rotation and emergency disable are documented. | Added. | None |
| Recovery material | Host-local `/Users/daniildegtyarev/.config/roehub/openbao/init-stage3a.json` | Stored outside repo with restrictive permissions. | Provisioning wrote operator material only to host-local secret directory. | None |
| Env injection | Host-local `/Users/daniildegtyarev/.config/roehub/roehub.env` | `OPENBAO_ADDR`, `ROEHUB_EXCHANGE_CONTROL_TRANSIT_TOKEN`, `ROEHUB_API_TRANSIT_TOKEN` are injected without committing values. | Provisioning updated `roehub.env`; token values are not in repo. | None |
| Backup path | `/opt/roehub/state/openbao/backups/<timestamp>/data` | Storage backup path is documented; backup contents treated as secret-bearing. | Documented in runbook. | None |
| Rollback | Revoke service tokens, stop `roehub_openbao`, remove app env vars if needed. | No Roehub database rollback required. | Documented; Stage 3A changed only operational runtime/config. | None |

## Contract Impact Classification

| Dimension | Classification | Reason |
|---|---|---|
| Public API contract | `none` | Existing `apps/api` HTTP routes and DTOs are unchanged. |
| Port / application contract | `none` | No application `ExchangeSecretCipher` port or adapter is introduced in Stage 3A. |
| DTO schema | `none` | No DTOs are changed. |
| Persisted schema | `none` | No database migration or app table shape is changed. |
| Config schema | `compatible-change` | New operational env vars are required on target runtime: `OPENBAO_ADDR`, `ROEHUB_EXCHANGE_CONTROL_TRANSIT_TOKEN`, `ROEHUB_API_TRANSIT_TOKEN`. |
| Ops / runtime | `compatible-change` | New local-only OpenBao runtime, launchd service, Monit check, Prometheus job, policies and runbook are added. |
| Request hash / cache / persistence identity | `none` | No request hash, cache key or persistence identity semantics are changed. |

## Stage 3B Handoff Facts

- Stage 3A is accepted only for runtime provisioning; application integration is
  still Stage 3B.
- `OPENBAO_ADDR` is `http://127.0.0.1:8200` on Mac Studio.
- Transit mount is `transit`.
- Transit key is `roehub-exchange-credentials`.
- `exchange-control` uses policy `roehub-exchange-control-transit` and env var
  `ROEHUB_EXCHANGE_CONTROL_TRANSIT_TOKEN`.
- `apps/api` uses policy `roehub-api-transit-deny-decrypt` and env var
  `ROEHUB_API_TRANSIT_TOKEN`; decrypt is proven denied with HTTP `403`.
- OpenBao file storage requires operator unseal after process restart.
- Stage 3C, Stage 4 and Stage 5 remain blocked until their own prerequisites
  are accepted.
