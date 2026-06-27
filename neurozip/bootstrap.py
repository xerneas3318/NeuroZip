"""Self-bootstrapping ML runtime.

The brew/core install is stdlib-only (fast). The first time an inference command runs,
this builds an isolated virtualenv at ~/.neurozip/venv with neurozip[ml] (PyTorch etc.),
shows a one-time status, and the CLI re-execs into it. Subsequent runs reuse it.

Override the venv location with $NEUROZIP_VENV (handy for testing / shared envs).
"""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
import venv
from pathlib import Path


def venv_dir() -> Path:
    if os.environ.get("NEUROZIP_VENV"):
        return Path(os.environ["NEUROZIP_VENV"])
    return Path.home() / ".neurozip" / "venv"


def _venv_python(d: Path) -> Path:
    sub = "Scripts" if os.name == "nt" else "bin"
    exe = "python.exe" if os.name == "nt" else "python"
    return d / sub / exe


def has_torch() -> bool:
    return importlib.util.find_spec("torch") is not None


def _python_has_torch(py: Path) -> bool:
    try:
        subprocess.check_call([str(py), "-c", "import torch"],
                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True
    except Exception:
        return False


def ensure_runtime() -> str | None:
    """Ensure an interpreter with neurozip[ml] exists.

    Returns the path to that interpreter if the CURRENT one lacks torch (caller should
    re-exec into it), or None if the current interpreter is already ML-capable.
    """
    if has_torch():
        return None

    d = venv_dir()
    py = _venv_python(d)
    if py.exists() and _python_has_torch(py):
        return str(py)

    from . import runtime as rt
    print("[neurozip] First-time setup: building the inference environment (one-time).")
    print("[neurozip] Installing PyTorch + the model stack. This can take a few minutes...")
    d.parent.mkdir(parents=True, exist_ok=True)
    venv.create(d, with_pip=True, clear=True)
    spec = f"neurozip[ml] @ {rt.SOURCE_URL}"
    try:
        subprocess.check_call([str(py), "-m", "pip", "install", "--quiet", "--upgrade", "pip"])
        print("[neurozip] downloading and installing PyTorch + deps (large, please wait)...")
        subprocess.check_call([str(py), "-m", "pip", "install", spec])
    except subprocess.CalledProcessError as e:
        sys.exit(f"[neurozip] setup failed ({e}). You can retry, or install manually:\n"
                 f"  pip install 'neurozip[ml]'")
    if not _python_has_torch(py):
        sys.exit("[neurozip] setup completed but torch is still missing; please report this.")
    print("[neurozip] inference environment ready.")
    return str(py)


def reexec_if_needed(command: str, ml_commands: set) -> None:
    """For ML commands, build+re-exec into the ML venv when the current python lacks torch."""
    if command not in ml_commands:
        return
    if has_torch() or os.environ.get("NEUROZIP_BOOTSTRAPPED"):
        return
    py = ensure_runtime()
    if py:
        env = dict(os.environ, NEUROZIP_BOOTSTRAPPED="1")
        os.execve(py, [py, "-m", "neurozip", *sys.argv[1:]], env)
