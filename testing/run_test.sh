#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TEST_ROOT="${REPO_ROOT}/testing"

mode="${1:-all}"
if [[ "${mode}" == "ci_smoke" ]]; then
  shift
  cd "${REPO_ROOT}"
  python3 -m pytest "${TEST_ROOT}" -m ci_smoke "$@"
else
  cd "${REPO_ROOT}"
  python3 -m pytest "${TEST_ROOT}" "$@"
fi
