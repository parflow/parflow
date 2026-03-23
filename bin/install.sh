#!/usr/bin/env bash
#
# ParFlow Installer
# -----------------
# Downloads and installs a pre-built ParFlow binary bundle.
#
# Usage:
#   curl -sSf https://raw.githubusercontent.com/parflow/parflow/master/bin/install.sh | bash
#
#   # With options:
#   curl -sSf https://raw.githubusercontent.com/parflow/parflow/master/bin/install.sh | bash -s -- --dir ~/parflow --version 3.15.0
#
set -euo pipefail

# =============================================================================
# Defaults
# =============================================================================
INSTALL_DIR="${HOME}/parflow"
VERSION="latest"
REPO="parflow/parflow"
GITHUB="https://github.com/${REPO}"

# =============================================================================
# Parse arguments
# =============================================================================
while [[ $# -gt 0 ]]; do
    case $1 in
        --dir)      INSTALL_DIR="$2"; shift 2 ;;
        --version)  VERSION="$2"; shift 2 ;;
        --help|-h)
            echo "ParFlow Installer"
            echo ""
            echo "Usage: install.sh [--dir PATH] [--version VERSION]"
            echo ""
            echo "  --dir PATH       Install location (default: ~/parflow)"
            echo "  --version VER    ParFlow version, e.g. 3.15.0 (default: latest)"
            echo ""
            exit 0 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# =============================================================================
# Detect platform
# =============================================================================
OS="$(uname -s)"
ARCH="$(uname -m)"

case "${OS}" in
    Darwin)
        case "${ARCH}" in
            arm64)  PLATFORM="macos-arm64" ;;
            x86_64) PLATFORM="macos-x86_64" ;;
            *)      echo "Error: Unsupported macOS architecture: ${ARCH}"; exit 1 ;;
        esac
        ;;
    Linux)
        case "${ARCH}" in
            x86_64) PLATFORM="linux-x86_64" ;;
            *)      echo "Error: Unsupported Linux architecture: ${ARCH}"; exit 1 ;;
        esac
        if grep -qi microsoft /proc/version 2>/dev/null; then
            echo "Detected WSL — using Linux x86_64 bundle"
        fi
        ;;
    MINGW*|MSYS*|CYGWIN*)
        echo ""
        echo "Windows detected. ParFlow requires WSL2 (Windows Subsystem for Linux)."
        echo ""
        echo "Install WSL2 (open PowerShell as Administrator):"
        echo "  wsl --install"
        echo ""
        echo "Then open Ubuntu from the Start menu and re-run this installer."
        exit 1
        ;;
    *)
        echo "Error: Unsupported operating system: ${OS}"
        exit 1
        ;;
esac

# =============================================================================
# Resolve version
# =============================================================================
echo ""
echo "ParFlow Installer"
echo "  Platform:  ${PLATFORM}"
echo "  Install:   ${INSTALL_DIR}"

if [[ "${VERSION}" == "latest" ]]; then
    echo "  Version:   latest (resolving...)"
    LATEST_TAG=$(curl -sSf "https://api.github.com/repos/${REPO}/releases/latest" \
        | grep '"tag_name"' | head -1 | cut -d'"' -f4)

    if [[ -z "${LATEST_TAG}" ]]; then
        echo "Error: Could not determine latest release."
        echo "Check: ${GITHUB}/releases"
        exit 1
    fi

    VERSION="${LATEST_TAG#v}"
    TAG="${LATEST_TAG}"
    echo "  Resolved:  ${VERSION}"
else
    TAG="v${VERSION}"
fi

TARBALL="parflow-${VERSION}-${PLATFORM}.tar.gz"
URL="${GITHUB}/releases/download/${TAG}/${TARBALL}"

# =============================================================================
# Download and install
# =============================================================================
echo ""
echo "Downloading ${TARBALL}..."

TMPDIR=$(mktemp -d)
trap "rm -rf ${TMPDIR}" EXIT

HTTP_CODE=$(curl -fSL -w '%{http_code}' -o "${TMPDIR}/${TARBALL}" "${URL}" 2>/dev/null) || true

if [[ ! -f "${TMPDIR}/${TARBALL}" ]] || [[ "${HTTP_CODE}" != "200" ]]; then
    echo ""
    echo "Error: Download failed (HTTP ${HTTP_CODE:-???})"
    echo "  URL: ${URL}"
    echo "  Check: ${GITHUB}/releases"
    exit 1
fi

echo "Downloaded $(du -sh "${TMPDIR}/${TARBALL}" | cut -f1)"
echo ""
echo "Installing to ${INSTALL_DIR}..."

mkdir -p "${INSTALL_DIR}"
tar xzf "${TMPDIR}/${TARBALL}" -C "${INSTALL_DIR}" --strip-components=1

if [[ ! -x "${INSTALL_DIR}/bin/parflow" ]]; then
    echo "Error: Installation incomplete — parflow binary not found"
    exit 1
fi

# =============================================================================
# Done
# =============================================================================
SHELL_NAME="$(basename ${SHELL})"
RC_FILE="${HOME}/.${SHELL_NAME}rc"

echo ""
echo "============================================="
echo "  ParFlow ${VERSION} installed"
echo "============================================="
echo ""
echo "Add this to your shell (or ${RC_FILE}):"
echo ""
echo "  export PARFLOW_DIR=${INSTALL_DIR}"
echo "  export PATH=\${PARFLOW_DIR}/bin:\${PATH}"
echo ""
echo "Then run your simulations as usual:"
echo ""
echo "  python my_simulation.py"
echo ""

# Offer to add to shell rc if running interactively
if [[ -t 0 ]]; then
    echo -n "Add to ${RC_FILE} now? [y/N] "
    read -r REPLY
    if [[ "${REPLY}" =~ ^[Yy] ]]; then
        echo "" >> "${RC_FILE}"
        echo "# ParFlow ${VERSION}" >> "${RC_FILE}"
        echo "export PARFLOW_DIR=${INSTALL_DIR}" >> "${RC_FILE}"
        echo 'export PATH=${PARFLOW_DIR}/bin:${PATH}' >> "${RC_FILE}"
        echo "Added. Open a new terminal or: source ${RC_FILE}"
    fi
fi
