#!/usr/bin/env bash
# Shared environment for launchers. Run these scripts with Bash on Linux/WSL.
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -f "${PROJECT_ROOT}/.env" ]]; then
  set -a
  source "${PROJECT_ROOT}/.env"
  set +a
fi
export PYTHONPATH="${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
cd "${PROJECT_ROOT}"

require_file() {
  [[ -f "$1" ]] || { echo "Missing file: $1" >&2; exit 1; }
}
