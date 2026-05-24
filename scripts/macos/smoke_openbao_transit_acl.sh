#!/usr/bin/env bash
set -Eeuo pipefail

require_env() {
  local name="$1"
  if [[ -z "${!name:-}" ]]; then
    echo "missing required env var: ${name}" >&2
    exit 64
  fi
}

require_env OPENBAO_ADDR
require_env ROEHUB_EXCHANGE_CONTROL_TRANSIT_TOKEN
require_env ROEHUB_API_TRANSIT_TOKEN

encrypt_out="$(mktemp)"
decrypt_out="$(mktemp)"
trap 'rm -f "$encrypt_out" "$decrypt_out"' EXIT

curl -fsS "${OPENBAO_ADDR}/v1/sys/health" >/dev/null
echo "openbao_health=ok"

curl -fsS \
  -X POST "${OPENBAO_ADDR}/v1/transit/encrypt/roehub-exchange-credentials" \
  -H "X-Vault-Token: ${ROEHUB_EXCHANGE_CONTROL_TRANSIT_TOKEN}" \
  --data '{"plaintext":"U1RBR0UzQV9TTU9LRQ=="}' \
  -o "$encrypt_out"
if ! grep -q '"ciphertext"' "$encrypt_out"; then
  echo "exchange_control_encrypt=missing_ciphertext" >&2
  exit 1
fi
echo "exchange_control_encrypt=ok"

decrypt_status="$(
  curl -sS \
    -o "$decrypt_out" \
    -w "%{http_code}" \
    -X POST "${OPENBAO_ADDR}/v1/transit/decrypt/roehub-exchange-credentials" \
    -H "X-Vault-Token: ${ROEHUB_API_TRANSIT_TOKEN}" \
    --data '{"ciphertext":"vault:v1:stage3a-placeholder"}'
)"
if [[ "$decrypt_status" != "403" ]]; then
  echo "apps_api_decrypt_denied=unexpected_status_${decrypt_status}" >&2
  exit 1
fi
echo "apps_api_decrypt_denied=403"
