# core.py
# ---------------------------------------------------------------------------
# Core Utilities for VisoMaster Fusion Launcher
# ---------------------------------------------------------------------------
# Provides:
#   • Centralized filesystem path resolution via PATHS
#   • Basic runtime validation (must_exist)
#   • Theme application (True-Dark QSS)
#   • Portable subprocess helpers for Python and UV operations
# ---------------------------------------------------------------------------

from pathlib import Path
import os
import sys
import subprocess
from PySide6 import QtWidgets


# ---------- Path Resolution ----------


def resolve_paths():
    """Return all filesystem paths used by the launcher."""
    script_path = Path(__file__).resolve()
    ui_dir = script_path.parent.parent  # .../app/ui
    app_dir = ui_dir.parent  # .../app
    repo_dir = app_dir.parent  # .../VisoMaster-Fusion
    base_dir = repo_dir.parent  # .../VisoMaster
    portable_dir = base_dir / "portable-files"

    return {
        "BASE_DIR": base_dir,
        "PORTABLE_DIR": portable_dir,
        "APP_DIR": repo_dir,
        # The interpreter every dependency lives in is the venv one, not the bare runtime
        # under portable-files/python that Windows_Start_Portable.bat only uses as a base.
        "PYTHON_EXE": portable_dir / "venv" / "Scripts" / "python.exe",
        "UV_EXE": portable_dir / "uv" / "uv.exe",
        "GIT_EXE": portable_dir / "git" / "bin" / "git.exe",
        "STYLES_DIR": app_dir / "ui" / "styles",
        "LOGO_PNG": app_dir / "ui" / "core" / "media" / "visomaster_logo.png",
        "SMALL_ICON": app_dir / "ui" / "core" / "media" / "visomaster_small.png",
        # Both installers (Windows_Install_or_Update.bat and Windows_Start_Portable.bat) drive the
        # requirements file and the model downloader that sit next to them in the package root.
        "REQ_FILE": base_dir / "requirements.txt",
        "MAIN_PY": repo_dir / "main.py",
        "DOWNLOAD_PY": base_dir / "HF_model_downloader.py",
        "OPTIMIZE_PY": app_dir / "tools" / "optimize_models.py",
        "PORTABLE_CFG": base_dir / "portable.cfg",
    }


PATHS = resolve_paths()


# ---------- Validation ----------


def must_exist(p: Path, what: str):
    """Exit early with a clear message if a required path is missing."""
    if not Path(p).exists():
        print(f"[Launcher] ERROR: Missing {what}: {p}")
        sys.exit(1)


# ---------- Theme Handling ----------


def apply_theme_to_app(app: QtWidgets.QApplication):
    """Apply the True-Dark QSS theme to the launcher."""
    qss_path = PATHS["STYLES_DIR"] / "true_dark.qss"
    if not qss_path.exists():
        print(f"[Launcher] Warning: true_dark.qss not found in {PATHS['STYLES_DIR']}")
        return
    try:
        app.setStyleSheet(qss_path.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"[Launcher] Error applying theme: {e}")


# ---------- Subprocess Helpers ----------


def run_python(script_path: Path, args: list | None = None, cwd: Path | None = None):
    """Run a Python script using the portable Python interpreter.

    The working directory defaults to APP_DIR (the repo root) so that
    `python -m app.ui.launcher` and similar module invocations find the `app/`
    package regardless of where the caller's cwd is. Scripts that live in the
    package root and resolve paths relative to it - HF_model_downloader.py builds
    `VisoMaster-Fusion/model_assets` from the cwd - must pass cwd=BASE_DIR instead.
    """
    cmd = [str(PATHS["PYTHON_EXE"]), str(script_path)] + (args or [])
    subprocess.run(cmd, cwd=str(cwd or PATHS["APP_DIR"]), shell=False)


# UV network tuning defaults.
# These can be overridden by setting environment variables before launching.
# UV_HTTP_TIMEOUT: per-request HTTP read timeout in seconds (default 120 s).
#   Increase if large wheels (torch, onnxruntime) time out on slow connections.
# UV_HTTP_RETRIES: number of retry attempts on transient network errors (default 5).
# UV_CONCURRENT_DOWNLOADS: parallel download slots (default 4).
#   Reduce to 1 on very slow / metered connections to avoid overloading the pipe.
_UV_HTTP_TIMEOUT = "120"
_UV_HTTP_RETRIES = "5"
_UV_CONCURRENT_DOWNLOADS = "4"


def uv_pip_install():
    """Run dependency installation using the portable uv executable.

    Passes uv network tuning via environment variables so that slow or unstable
    connections (common in portable installs) do not abort mid-install with a
    cryptic timeout error. Existing user-provided uv environment values win.
    """
    env = os.environ.copy()
    env.setdefault("UV_HTTP_TIMEOUT", _UV_HTTP_TIMEOUT)
    env.setdefault("UV_HTTP_RETRIES", _UV_HTTP_RETRIES)
    env.setdefault("UV_CONCURRENT_DOWNLOADS", _UV_CONCURRENT_DOWNLOADS)
    # requirements.txt mixes a PyTorch extra index with pinned direct-URL wheels whose
    # version tags are non-standard; both installers set these, so match them exactly.
    env.setdefault("UV_SKIP_WHEEL_FILENAME_CHECK", "1")
    env.setdefault("UV_LINK_MODE", "copy")

    return subprocess.run(
        [
            str(PATHS["UV_EXE"]),
            "pip",
            "install",
            "-r",
            str(PATHS["REQ_FILE"]),
            "--python",
            str(PATHS["PYTHON_EXE"]),
            "--index-strategy",
            "unsafe-best-match",
        ],
        cwd=str(PATHS["APP_DIR"]),
        env=env,
        check=True,
        shell=False,
    )
