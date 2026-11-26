"""Pre-flight check. Run: uv run python scripts/check_env.py  (exit code 1 = something missing)."""

from __future__ import annotations

import importlib
import platform
import sys

REQUIRED = (
    "numpy", "pandas", "pyarrow", "scipy", "sklearn", "statsmodels", "matplotlib", "seaborn",
    "jupyterlab", "ipykernel", "ipywidgets", "attrs", "rich", "tqdm", "gdown", "optuna", "arena",
)
OPTIONAL = ("torch", "transformers", "lightgbm")


def _report(ok: bool, label: str, detail: str = "") -> bool:
    sys.stdout.write(f"  [{'ok' if ok else '!!'}] {label:<14} {detail}\n")
    return ok


def check_packages() -> bool:
    ok = True
    for name in REQUIRED:
        try:
            mod = importlib.import_module(name)
            _report(True, name, getattr(mod, "__version__", ""))
        except ImportError as exc:
            ok = _report(False, name, f"MISSING ({exc}); run: uv sync") and ok
    for name in OPTIONAL:
        try:
            importlib.import_module(name)
            _report(True, name, "(optional)")
        except ImportError:
            _report(True, name, "(optional, not installed)")
    return ok


def check_data() -> None:
    from arena.paths import default_data_dir

    try:
        _report(True, "data", str(default_data_dir()))
    except FileNotFoundError as exc:
        _report(True, "data", f"none yet ({exc})")


def main() -> int:
    sys.stdout.write(f"Python {platform.python_version()} on {platform.platform()}\n")
    ok = _report(sys.version_info >= (3, 11), "python>=3.11", sys.executable)
    ok = check_packages() and ok
    check_data()
    sys.stdout.write("\nALL GOOD\n" if ok else "\nFIX THE [!!] LINES ABOVE\n")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
