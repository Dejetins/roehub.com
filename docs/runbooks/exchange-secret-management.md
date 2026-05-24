# Exchange Secret Management Runbook

Статус: Stage 3A runtime accepted on Mac Studio, 2026-05-24.

Этот runbook описывает OpenBao Transit boundary для custody exchange
credentials. Он не содержит root tokens, unseal keys, service tokens,
ciphertexts от реальных credentials или provider responses.

## Runtime Contract

| Component | Value | Notes |
|---|---|---|
| Secret backend | OpenBao 2.5.4 | Installed by Homebrew on Mac Studio. |
| Bind address | `127.0.0.1:8200` | Local-only listener; no public network exposure. |
| Runtime address | `OPENBAO_ADDR=http://127.0.0.1:8200` | Injected through host-local env, not committed. |
| Storage | `/opt/roehub/state/openbao/data` | File storage on Mac Studio. |
| Config | `/opt/roehub/config/openbao/openbao.prod.hcl` | Installed from `infra/macos/openbao/openbao.prod.hcl`. |
| launchd label | `com.roehub.openbao` | Installed from `infra/macos/launchd/com.roehub.openbao.plist`. |
| Monit service | `roehub_openbao` | Installed from `infra/scripts/monit/roehub-openbao.monitrc`. |
| Metrics | `/v1/sys/metrics?format=prometheus` | Scraped by Prometheus job `openbao`. |

## Transit ACL

| Principal | Policy | Runtime env var | Transit capabilities |
|---|---|---|---|
| `exchange-control` | `roehub-exchange-control-transit` | `ROEHUB_EXCHANGE_CONTROL_TRANSIT_TOKEN` | `update` on `transit/encrypt/roehub-exchange-credentials`, `transit/decrypt/roehub-exchange-credentials`, `transit/hmac/roehub-exchange-credentials/*`. |
| `apps/api` | `roehub-api-transit-deny-decrypt` | `ROEHUB_API_TRANSIT_TOKEN` | No decrypt capability; token exists only to prove deny behavior in Stage 3A. |

Policy source files:

- `infra/macos/openbao/policies/roehub-exchange-control-transit.hcl`
- `infra/macos/openbao/policies/roehub-api-transit-deny-decrypt.hcl`

Transit key:

- mount: `transit`
- key: `roehub-exchange-credentials`

## Provisioning

Use this only from the Mac Studio target runtime after the OpenBao launchd
service is running.

```bash
bash /opt/roehub/bin/provision_openbao_transit_stage3a.sh
```

The script writes sensitive material only under:

- `/Users/daniildegtyarev/.config/roehub/openbao/`
- `/Users/daniildegtyarev/.config/roehub/roehub.env`

Both locations must remain host-local and mode `0600`/`0700`. Do not copy their
contents into the repository, reports, screenshots or shell transcripts.

## Smoke Checks

```bash
set -a
source /Users/daniildegtyarev/.config/roehub/roehub.env
set +a

curl -fsS "$OPENBAO_ADDR/v1/sys/health"
bash /opt/roehub/bin/smoke_openbao_transit_acl.sh
/opt/homebrew/opt/monit/bin/monit -c /opt/homebrew/etc/monitrc summary | grep -Ei "openbao|exchange_control"
curl -fsS 'http://127.0.0.1:9090/api/v1/query?query=up{job="openbao"}'
```

Expected results:

| Check | Expected result |
|---|---|
| Health | `initialized=true`, `sealed=false`. |
| Encrypt | `exchange_control_encrypt=ok`; response body is discarded. |
| API decrypt | `apps_api_decrypt_denied=403`. |
| Monit | `roehub_openbao OK`. |
| Prometheus | `up{job="openbao",instance="127.0.0.1:8200"} = 1`. |

## Backup And Restore

| Operation | Procedure | Safety notes |
|---|---|---|
| Config backup | Back up `/opt/roehub/config/openbao/openbao.prod.hcl` and policy HCL files from the repository source of truth. | These files are non-secret. |
| Storage backup | Stop writes, then copy `/opt/roehub/state/openbao/data` to `/opt/roehub/state/openbao/backups/<timestamp>/data`. | Backup contents are secret-bearing because they contain encrypted key material and tokens. Store only on trusted encrypted media. |
| Recovery material backup | Keep `/Users/daniildegtyarev/.config/roehub/openbao/init-stage3a.json` outside the repository. | Contains unseal and root material; never print or commit. |
| Restore | Stop `com.roehub.openbao`, restore config/storage, start service, unseal with operator-held material, run smoke checks. | Do not restore from unknown or partially copied storage snapshots. |

## Restart And Unseal

OpenBao file storage starts sealed after process restart. launchd/Monit can
restart the process, but an operator must unseal before the service becomes
healthy.

```bash
/opt/homebrew/opt/monit/bin/monit -c /opt/homebrew/etc/monitrc restart roehub_openbao
# operator unseal step uses host-local recovery material and must not be logged
curl -fsS "$OPENBAO_ADDR/v1/sys/health"
```

If `/v1/sys/health` reports `sealed=true`, do not start Stage 3B/3C/4/5 work.

## Token Rotation

1. Create a replacement periodic token from an operator shell using the same
   policy and display name.
2. Update only `/Users/daniildegtyarev/.config/roehub/roehub.env`.
3. Restart the consuming service.
4. Run `smoke_openbao_transit_acl.sh`.
5. Revoke the old token after the replacement is proven.

Never commit token values or token lookup output.

## Transit Key Rotation

```bash
bao write -f transit/keys/roehub-exchange-credentials/rotate
```

Rotation adds a new key version without exposing plaintext. Stage 3B+ must keep
stored ciphertexts as Transit ciphertext and use Transit rewrap/rotation flows
instead of decrypting through `apps/api`.

## Emergency Disable

| Emergency action | Command shape | Rollback impact |
|---|---|---|
| Disable application access | Revoke the relevant service token. | No database rollback; app flows fail closed. |
| Stop runtime | `monit stop roehub_openbao` or `launchctl bootout gui/$(id -u) /Users/daniildegtyarev/Library/LaunchAgents/com.roehub.openbao.plist`. | Stage 3B+ secret operations unavailable until restart/unseal. |
| Disable Transit mount | Operator-only `bao secrets disable transit`. | Breaking for ciphertext operations; use only as incident response. |

## Stage Handoff

Stage 3B may start only while all of these are true:

- Stage 3A report is accepted.
- `OPENBAO_ADDR` points to the healthy target runtime.
- `ROEHUB_EXCHANGE_CONTROL_TRANSIT_TOKEN` can encrypt with
  `roehub-exchange-credentials`.
- `ROEHUB_API_TRANSIT_TOKEN` cannot decrypt with
  `roehub-exchange-credentials`.
- Monit and Prometheus both show the OpenBao runtime healthy.
