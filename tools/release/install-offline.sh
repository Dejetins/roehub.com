#!/bin/sh
set -eu

usage() {
  echo "usage: $0 --trusted-public-key PATH [--state-directory PATH] [--profile base|trading|ml] [--runtime-smoke]" >&2
  exit 2
}

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
BUNDLE_ROOT=$(CDPATH= cd -- "$SCRIPT_DIR/../.." && pwd)
STATE_DIRECTORY=${ROEHUB_OFFLINE_STATE_DIRECTORY:-"$HOME/.local/share/roehub/offline"}
PROFILE=base
TRUSTED_PUBLIC_KEY=
RUNTIME_SMOKE=

while [ "$#" -gt 0 ]; do
  case "$1" in
    --trusted-public-key)
      [ "$#" -ge 2 ] || usage
      TRUSTED_PUBLIC_KEY=$2
      shift 2
      ;;
    --state-directory)
      [ "$#" -ge 2 ] || usage
      STATE_DIRECTORY=$2
      shift 2
      ;;
    --profile)
      [ "$#" -ge 2 ] || usage
      PROFILE=$2
      shift 2
      ;;
    --runtime-smoke)
      RUNTIME_SMOKE=--runtime-smoke
      shift
      ;;
    *)
      usage
      ;;
  esac
done

[ -n "$TRUSTED_PUBLIC_KEY" ] || usage
command -v python3 >/dev/null 2>&1 || {
  echo "Python 3.9 or newer is required for signature and digest verification" >&2
  exit 1
}
python3 -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 9) else 1)' || {
  echo "Python 3.9 or newer is required for signature and digest verification" >&2
  exit 1
}
command -v docker >/dev/null 2>&1 || {
  echo "Docker Engine with Compose v2 is required" >&2
  exit 1
}
command -v skopeo >/dev/null 2>&1 || {
  echo "skopeo is required to import the verified OCI archives" >&2
  exit 1
}
command -v ssh-keygen >/dev/null 2>&1 || {
  echo "OpenSSH ssh-keygen is required for SSHSIG-Ed25519 verification" >&2
  exit 1
}

exec python3 "$SCRIPT_DIR/offline_bundle.py" install \
  --bundle "$BUNDLE_ROOT" \
  --trusted-public-key "$TRUSTED_PUBLIC_KEY" \
  --state-directory "$STATE_DIRECTORY" \
  --profile "$PROFILE" \
  ${RUNTIME_SMOKE}
