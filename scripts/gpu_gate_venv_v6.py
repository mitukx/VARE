"""Fail-closed validation and provenance capture for the isolated v6 gate venv."""

from __future__ import annotations

import importlib
import site
import sys
from pathlib import Path


EXTERNAL_MODULES = (
    "torch",
    "transformers",
    "trl",
    "accelerate",
    "datasets",
    "math_verify",
    "latex2sympy2_extended",
    "antlr4",
    "pyarrow",
    "packaging",
)


def _inside(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def validate_isolated_environment(
    *,
    prefix: str,
    base_prefix: str,
    sys_path: list[str],
    module_paths: dict[str, str],
) -> dict:
    """Validate venv isolation and return stable, JSON-serializable provenance."""
    venv_root = Path(prefix).resolve()
    base_root = Path(base_prefix).resolve()
    if venv_root == base_root:
        raise SystemExit("NO-GO: Python is not running inside a virtual environment")

    resolved_sys_path = []
    for item in sys_path:
        if not item:
            continue
        resolved = Path(item).resolve()
        resolved_sys_path.append(str(resolved))
        if any(part in {"site-packages", "dist-packages"} for part in resolved.parts):
            if not _inside(resolved, venv_root):
                raise SystemExit(f"NO-GO: global package path is visible on sys.path: {resolved}")

    resolved_modules = {}
    for name in EXTERNAL_MODULES:
        value = module_paths.get(name)
        if not value:
            raise SystemExit(f"NO-GO: required module has no file provenance: {name}")
        path = Path(value).resolve()
        if not _inside(path, venv_root):
            raise SystemExit(f"NO-GO: {name} imported outside the isolated venv: {path}")
        resolved_modules[name] = str(path)

    return {
        "sys_prefix": str(venv_root),
        "sys_base_prefix": str(base_root),
        "sys_path": resolved_sys_path,
        "module_files": resolved_modules,
        "system_site_packages": False,
    }


def current_environment_provenance() -> dict:
    modules = {name: importlib.import_module(name) for name in EXTERNAL_MODULES}
    module_paths = {
        name: str(Path(module.__file__).resolve())
        for name, module in modules.items()
        if getattr(module, "__file__", None)
    }
    record = validate_isolated_environment(
        prefix=sys.prefix,
        base_prefix=sys.base_prefix,
        sys_path=sys.path,
        module_paths=module_paths,
    )
    if site.ENABLE_USER_SITE:
        raise SystemExit("NO-GO: Python user site packages are enabled")
    record["python_executable"] = str(Path(sys.executable).resolve())
    record["python_version"] = sys.version
    record["user_site_enabled"] = False
    return record
