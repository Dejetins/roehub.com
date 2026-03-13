#!/usr/bin/env bash
set -euo pipefail

# Docs: docs/runbooks/mac-studio-monitoring-plan.md
# Related: docs/runbooks/mac-studio-backend-operations.md

BREW_BIN="${BREW_BIN:-/opt/homebrew/bin/brew}"

if [[ ! -x "${BREW_BIN}" ]]; then
  echo "brew not found at ${BREW_BIN}" >&2
  exit 1
fi

"${BREW_BIN}" list node_exporter >/dev/null 2>&1 || "${BREW_BIN}" install node_exporter
"${BREW_BIN}" services start node_exporter
