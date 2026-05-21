#!/usr/bin/env bash
# Copy a ParFlow install prefix into the Xcode app Resources folder.
#
# Usage: ./packaging/macos/stage-app-payload.sh <install-prefix>
set -euo pipefail

INSTALL_PREFIX="${1:?install prefix required}"
REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
DEST="${REPO_ROOT}/packaging/macos/ParFlow/Resources/parflow"

rm -rf "${DEST}"
mkdir -p "$(dirname "${DEST}")"
cp -R "${INSTALL_PREFIX}/" "${DEST}/"

if [[ -f "${REPO_ROOT}/README-BINARY.md" ]]; then
  cp "${REPO_ROOT}/README-BINARY.md" "${DEST}/"
fi

echo "Staged ParFlow payload at ${DEST}"
