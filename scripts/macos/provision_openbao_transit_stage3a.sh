#!/usr/bin/env bash
set -Eeuo pipefail

PATH="/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin:${PATH:-}"
OPENBAO_ADDR="${OPENBAO_ADDR:-http://127.0.0.1:8200}"
BAO_ADDR="$OPENBAO_ADDR"
export OPENBAO_ADDR BAO_ADDR

SECRET_DIR="${ROEHUB_OPENBAO_SECRET_DIR:-/Users/daniildegtyarev/.config/roehub/openbao}"
ENV_FILE="${ROEHUB_ENV_FILE:-/Users/daniildegtyarev/.config/roehub/roehub.env}"
POLICY_DIR="${ROEHUB_OPENBAO_POLICY_DIR:-/opt/roehub/config/openbao/policies}"
INIT_FILE="$SECRET_DIR/init-stage3a.json"
EXCHANGE_TOKEN_FILE="$SECRET_DIR/exchange-control-token-stage3a.json"
API_TOKEN_FILE="$SECRET_DIR/apps-api-token-stage3a.json"

mkdir -p "$SECRET_DIR"
chmod 700 "$SECRET_DIR"

status_json="$(curl -sS "$OPENBAO_ADDR/v1/sys/health" || true)"
if printf "%s" "$status_json" | grep -q '"initialized":false'; then
  /opt/homebrew/opt/openbao/bin/bao operator init -key-shares=1 -key-threshold=1 -format=json > "$INIT_FILE"
  chmod 600 "$INIT_FILE"
  echo "openbao_initialized=created"
else
  echo "openbao_initialized=already"
fi

UNSEAL_KEY="$(
  python3 - "$INIT_FILE" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as f:
    data = json.load(f)
print(data["unseal_keys_b64"][0])
PY
)"
if curl -sS "$OPENBAO_ADDR/v1/sys/health" | grep -q '"sealed":true'; then
  /opt/homebrew/opt/openbao/bin/bao operator unseal "$UNSEAL_KEY" >/dev/null
  echo "openbao_unsealed=ok"
else
  echo "openbao_unsealed=already"
fi

ROOT_TOKEN="$(
  python3 - "$INIT_FILE" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as f:
    data = json.load(f)
print(data["root_token"])
PY
)"
export BAO_TOKEN="$ROOT_TOKEN"

if /opt/homebrew/opt/openbao/bin/bao secrets list -format=json \
  | python3 -c 'import json, sys; data = json.load(sys.stdin); raise SystemExit(0 if "transit/" in data else 1)'
then
  echo "transit_mount=already"
else
  /opt/homebrew/opt/openbao/bin/bao secrets enable -path=transit transit >/dev/null
  echo "transit_mount=enabled"
fi

/opt/homebrew/opt/openbao/bin/bao write -f transit/keys/roehub-exchange-credentials >/dev/null
/opt/homebrew/opt/openbao/bin/bao policy write roehub-exchange-control-transit "$POLICY_DIR/roehub-exchange-control-transit.hcl" >/dev/null
/opt/homebrew/opt/openbao/bin/bao policy write roehub-api-transit-deny-decrypt "$POLICY_DIR/roehub-api-transit-deny-decrypt.hcl" >/dev/null

ensure_token_file() {
  local token_file="$1"
  local policy="$2"
  local display_name="$3"
  local existing_accessor=""

  if [[ -f "$token_file" ]]; then
    existing_accessor="$(
      python3 - "$token_file" <<'PY'
import json
import sys

try:
    with open(sys.argv[1], encoding="utf-8") as f:
        print(json.load(f)["auth"]["accessor"])
except Exception:
    raise SystemExit(1)
PY
    )"
    if [[ -n "$existing_accessor" ]] && /opt/homebrew/opt/openbao/bin/bao token lookup -accessor "$existing_accessor" >/dev/null 2>&1; then
      echo "token_${display_name}=reused"
      return 0
    fi
  fi

  /opt/homebrew/opt/openbao/bin/bao token create -policy="$policy" -display-name="$display_name" -orphan -period=720h -format=json > "$token_file"
  echo "token_${display_name}=created"
}

ensure_token_file "$EXCHANGE_TOKEN_FILE" roehub-exchange-control-transit exchange-control
ensure_token_file "$API_TOKEN_FILE" roehub-api-transit-deny-decrypt apps-api
chmod 600 "$EXCHANGE_TOKEN_FILE" "$API_TOKEN_FILE"

python3 - "$EXCHANGE_TOKEN_FILE" "$API_TOKEN_FILE" "$ENV_FILE" <<'PY'
import json
import pathlib
import sys

exchange_file, api_file, env_file = map(pathlib.Path, sys.argv[1:])
exchange_token = json.loads(exchange_file.read_text())["auth"]["client_token"]
api_token = json.loads(api_file.read_text())["auth"]["client_token"]
updates = {
    "OPENBAO_ADDR": "http://127.0.0.1:8200",
    "ROEHUB_EXCHANGE_CONTROL_TRANSIT_TOKEN": exchange_token,
    "ROEHUB_API_TRANSIT_TOKEN": api_token,
}
lines = env_file.read_text().splitlines() if env_file.exists() else []
seen: set[str] = set()
out: list[str] = []
for line in lines:
    key = line.split("=", 1)[0] if "=" in line and not line.lstrip().startswith("#") else None
    if key in updates:
        out.append(f"{key}={updates[key]}")
        seen.add(key)
    else:
        out.append(line)
for key, value in updates.items():
    if key not in seen:
        out.append(f"{key}={value}")
env_file.write_text("\n".join(out) + "\n")
env_file.chmod(0o600)
PY

unset ROOT_TOKEN UNSEAL_KEY BAO_TOKEN

curl -fsS "$OPENBAO_ADDR/v1/sys/health" >/dev/null
echo "openbao_health=ok"
echo "transit_key=roehub-exchange-credentials"
echo "policy_exchange_control=roehub-exchange-control-transit"
echo "policy_apps_api=roehub-api-transit-deny-decrypt"
echo "runtime_env=/Users/daniildegtyarev/.config/roehub/roehub.env"
