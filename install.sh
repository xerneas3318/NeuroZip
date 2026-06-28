#!/usr/bin/env bash
# NeuroZip curl installer (beta).
#   curl -fsSL https://raw.githubusercontent.com/xerneas3318/NeuroZip/main/install.sh | bash
#
# Installs the `neurozip` CLI (stdlib-only core; the ML extra is optional).
set -euo pipefail

REPO="xerneas3318/NeuroZip"
REF="${NEUROZIP_REF:-main}"

say() { printf '\033[1;34m[neurozip]\033[0m %s\n' "$1"; }
die() { printf '\033[1;31m[neurozip] error:\033[0m %s\n' "$1" >&2; exit 1; }

command -v python3 >/dev/null 2>&1 || die "python3 not found (need >=3.9)."

PIP=(python3 -m pip)
"${PIP[@]}" --version >/dev/null 2>&1 || die "pip not available for python3."

say "installing neurozip@${REF} from github.com/${REPO} ..."
if ! "${PIP[@]}" install --user --upgrade "git+https://github.com/${REPO}.git@${REF}"; then
  die "install failed (is the repo public / are you authenticated?)."
fi

say "done. Try:  neurozip --version   and   neurozip serve"
say "for the neural layer:  python3 -m pip install --user 'neurozip[ml]'"
