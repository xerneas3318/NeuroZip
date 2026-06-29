#!/usr/bin/env bash
# NeuroZip installer.
#   curl -fsSL https://github.com/xerneas3318/NeuroZip/releases/download/v1.0.0/install.sh | bash
#
# Installs the precompiled, self-contained `neurozip` CLI: a zipapp that runs on
# any system python3, with no pip or brew build step. The PyTorch inference stack
# is set up automatically on first inference use; `neurozip download` fetches the
# trained model bundle.
set -euo pipefail

VERSION="${NEUROZIP_VERSION:-v1.0.0}"
REL="https://github.com/xerneas3318/NeuroZip/releases/download/${VERSION}"
BIN_DIR="${NEUROZIP_BIN:-$HOME/.local/bin}"

say() { printf '\033[1;34m[neurozip]\033[0m %s\n' "$1"; }
die() { printf '\033[1;31m[neurozip] error:\033[0m %s\n' "$1" >&2; exit 1; }

command -v python3 >/dev/null 2>&1 || die "python3 not found (need >= 3.9)."
command -v curl    >/dev/null 2>&1 || die "curl not found."

mkdir -p "$BIN_DIR"
say "downloading precompiled neurozip ${VERSION} ..."
curl -fsSL "${REL}/neurozip.pyz" -o "${BIN_DIR}/neurozip" || die "download failed."
chmod +x "${BIN_DIR}/neurozip"
say "installed -> ${BIN_DIR}/neurozip"

case ":$PATH:" in
  *":$BIN_DIR:"*) ;;
  *) say "add it to PATH:  export PATH=\"${BIN_DIR}:\$PATH\"" ;;
esac

say "next:  neurozip download   then   neurozip serve"
