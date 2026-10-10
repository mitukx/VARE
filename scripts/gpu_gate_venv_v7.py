"""Fail-closed validation and provenance capture for the isolated v7 gate venv."""

from __future__ import annotations

import importlib
import importlib.metadata
import os
import site
import sys
from pathlib import Path

from packaging.version import Version


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
    "pip",
)

EXPECTED_TARGET_PIP_VERSION = "26.2.1"
MINIMUM_HOST_PIP_VERSION = Version("22.3")
PIP_BOOTSTRAP_METHOD = "python -m pip --python <venv> install pip"


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
    host_pip_version = os.environ.get("VARE_HOST_PIP_VERSION")
    if not host_pip_version:
        raise SystemExit("NO-GO: VARE_HOST_PIP_VERSION is required for v7 bootstrap provenance")
    bootstrap = validate_pip_bootstrap_identity(
        prefix=sys.prefix,
        host_pip_version=host_pip_version,
        target_pip_version=importlib.metadata.version("pip"),
        target_pip_module=module_paths["pip"],
    )
    record["python_executable"] = str(Path(sys.executable).resolve())
    record["python_version"] = sys.version
    record["user_site_enabled"] = False
    record["pip_bootstrap"] = bootstrap
    return record


def validate_pip_bootstrap_identity(
    *, prefix: str, host_pip_version: str, target_pip_version: str, target_pip_module: str
) -> dict:
    """Bind the official host-pip --python bootstrap to the isolated target venv."""
    root = Path(prefix).resolve()
    try:
        host = Version(host_pip_version)
        target = Version(target_pip_version)
    except Exception as error:
        raise SystemExit(f"NO-GO: invalid pip bootstrap version: {error}") from error
    if host < MINIMUM_HOST_PIP_VERSION:
        raise SystemExit(
            f"NO-GO: host pip {host} lacks the required --python option (pip >= 22.3)"
        )
    if str(target) != EXPECTED_TARGET_PIP_VERSION:
        raise SystemExit(
            f"NO-GO: target venv pip must be {EXPECTED_TARGET_PIP_VERSION}; found {target}"
        )
    target_module = Path(target_pip_module).resolve()
    if not _inside(target_module, root):
        raise SystemExit(f"NO-GO: pip imported outside the isolated venv: {target_module}")
    return {
        "method": PIP_BOOTSTRAP_METHOD,
        "host_pip_version": str(host),
        "minimum_host_pip_version": str(MINIMUM_HOST_PIP_VERSION),
        "target_pip_version": str(target),
        "expected_target_pip_version": EXPECTED_TARGET_PIP_VERSION,
        "target_pip_module": str(target_module),
    }
