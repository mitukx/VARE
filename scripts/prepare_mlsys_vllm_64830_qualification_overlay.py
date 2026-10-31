#!/usr/bin/env python3
"""Freeze the exact vLLM #64830 regression and isolated method extracts."""
from __future__ import annotations

import argparse
import ast
import difflib
import hashlib
import json
import subprocess
from pathlib import Path

HISPARSE_BLOCK = '''        if self.attention_config.hisparse_config is not None:
            unsupported.append("HiSparse")

'''
WATERMARK_BLOCK = '''        if self.watermark_config is not None:
            unsupported.append("watermarking")

'''

SCAFFOLD = '''from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace

import pytest


class HiSparseConfig:
    pass


class AttentionConfig:
    def __init__(self, hisparse_config=None):
        self.hisparse_config = hisparse_config


class WatermarkConfig:
    def __init__(self, key):
        self.key = key


class ParallelConfig:
    def __init__(self, distributed_executor_backend=None):
        self.distributed_executor_backend = distributed_executor_backend
        self.prefill_context_parallel_size = 1
        self.pipeline_parallel_size = 1
        self.enable_batch_sharded_sampling = False


class SchedulerConfig:
    @staticmethod
    def default_factory(async_scheduling=False):
        return SimpleNamespace(async_scheduling=async_scheduling)


def _load_candidate_vllm_config():
    method_path = Path(os.environ["VARE_METHOD_FILE"])
    namespace = {"__name__": "vllm_64830_qualification_candidate"}
    source = method_path.read_text(encoding="utf-8")
    exec(compile(source, str(method_path), "exec"), namespace)
    return namespace["VllmConfig"]


VllmConfig = _load_candidate_vllm_config()

'''


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def git_show(repository: Path, revision: str, path: str) -> bytes:
    return subprocess.check_output(
        ["git", "-C", str(repository), "show", f"{revision}:{path}"]
    )


def extract_class_method(source: str, class_name: str, method_name: str) -> str:
    tree = ast.parse(source)
    classes = [
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == class_name
    ]
    if len(classes) != 1:
        raise ValueError(f"expected one {class_name!r} class")
    methods = [
        node
        for node in classes[0].body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == method_name
    ]
    if len(methods) != 1:
        raise ValueError(f"expected one {method_name!r} method")
    lines = source.splitlines()
    node = methods[0]
    start = min([node.lineno, *(d.lineno for d in node.decorator_list)])
    method = "\n".join(lines[start - 1 : node.end_lineno])
    return f"class {class_name}:\n{method}\n"


def extract_test(source: str, test_name: str) -> str:
    tree = ast.parse(source)
    matches = [
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == test_name
    ]
    if len(matches) != 1:
        raise ValueError(f"expected one top-level {test_name!r} test")
    lines = source.splitlines()
    node = matches[0]
    start = min([node.lineno, *(d.lineno for d in node.decorator_list)])
    return "\n".join(lines[start - 1 : node.end_lineno]) + "\n"


def prepare(source_repository: Path, protocol_path: Path, output_dir: Path) -> dict:
    source_repository = source_repository.expanduser().resolve()
    protocol_path = protocol_path.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {output_dir}")

    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    candidate = protocol["candidate"]
    revisions = {
        "baseline": candidate["base_revision"],
        "fixed": candidate["fixed_merge_revision"],
    }
    expected_remote = "https://github.com/vllm-project/vllm.git"
    actual_remote = subprocess.check_output(
        ["git", "-C", str(source_repository), "remote", "get-url", "origin"],
        text=True,
    ).strip()
    if actual_remote != expected_remote:
        raise ValueError(f"unexpected source repository remote: {actual_remote}")

    source_path = "vllm/config/vllm.py"
    test_path = "tests/test_config.py"
    method_sources = {}
    full_source_hashes = {}
    for arm in ("baseline", "fixed"):
        data = git_show(source_repository, revisions[arm], source_path)
        source = data.decode("utf-8")
        method_sources[arm] = extract_class_method(
            source, "VllmConfig", "_get_v1_model_runner_unsupported_features"
        )
        full_source_hashes[arm] = sha256_bytes(data)

    fixed_method = method_sources["fixed"]
    if fixed_method.count(HISPARSE_BLOCK) != 1:
        raise ValueError("expected exactly one frozen HiSparse behavior block")
    if fixed_method.count(WATERMARK_BLOCK) != 1:
        raise ValueError("expected exactly one frozen watermark behavior block")
    mutant_method = fixed_method.replace(HISPARSE_BLOCK, "", 1).replace(
        WATERMARK_BLOCK, "", 1
    )
    if mutant_method == method_sources["baseline"]:
        raise ValueError("mutant unexpectedly equals the complete baseline method")

    test_data = git_show(
        source_repository, candidate["fixed_merge_revision"], test_path
    )
    test_source = test_data.decode("utf-8")
    test_function = extract_test(
        test_source, "test_v1_model_runner_rejects_v2_only_config"
    )
    test_file_hash = sha256_bytes(test_data)

    trusted_dir = output_dir / "trusted"
    trusted_dir.mkdir(parents=True)
    (trusted_dir / "test_vllm_config_unsupported_features.py").write_text(
        SCAFFOLD + test_function, encoding="utf-8"
    )
    (trusted_dir / "upstream_regression_test.py").write_text(
        test_function, encoding="utf-8"
    )
    method_names = {
        "baseline": "vllm_config_baseline_method.py",
        "fixed": "vllm_config_fixed_method.py",
        "wrong_mutant": "vllm_config_wrong_mutant_method.py",
    }
    all_methods = {
        "baseline": method_sources["baseline"],
        "fixed": method_sources["fixed"],
        "wrong_mutant": mutant_method,
    }
    method_hashes = {}
    for arm, filename in method_names.items():
        method_bytes = all_methods[arm].encode("utf-8")
        (trusted_dir / filename).write_bytes(method_bytes)
        method_hashes[arm] = sha256_bytes(method_bytes)

    patch_lines = difflib.unified_diff(
        fixed_method.splitlines(keepends=True),
        mutant_method.splitlines(keepends=True),
        fromfile="vllm/config/vllm.py@fixed",
        tofile="vllm/config/vllm.py@wrong-mutant",
        n=0,
    )
    mutant_patch = "".join(patch_lines).encode("utf-8")
    (output_dir / "wrong-mutant.patch").write_bytes(mutant_patch)

    test_file = trusted_dir / "test_vllm_config_unsupported_features.py"
    upstream_test_file = trusted_dir / "upstream_regression_test.py"
    pytest_config = trusted_dir / "pytest.ini"
    pytest_config.write_text("[pytest]\n", encoding="utf-8")
    manifest = {
        "schema_version": 1,
        "protocol_id": protocol["protocol_id"],
        "protocol_sha256": sha256_bytes(protocol_path.read_bytes()),
        "source_repository": actual_remote,
        "source_path": source_path,
        "test_path": test_path,
        "source_revisions": revisions,
        "source_file_sha256": full_source_hashes,
        "extracted_method_sha256": method_hashes,
        "mutant_patch_sha256": sha256_bytes(mutant_patch),
        "upstream_test_file_sha256": test_file_hash,
        "extracted_test_function_sha256": sha256_bytes(
            test_function.encode("utf-8")
        ),
        "test_overlay_sha256": sha256_bytes(test_file.read_bytes()),
        "method_artifact_sha256": method_hashes,
        "selected_test": "tests/test_config.py::test_v1_model_runner_rejects_v2_only_config",
        "parametrization": ["HiSparse", "watermarking"],
        "files": {},
    }
    for path in (
        test_file,
        upstream_test_file,
        pytest_config,
        *(trusted_dir / name for name in method_names.values()),
        output_dir / "wrong-mutant.patch",
    ):
        manifest["files"][str(path.relative_to(output_dir))] = {
            "bytes": path.stat().st_size,
            "sha256": sha256_bytes(path.read_bytes()),
        }
    manifest_path = output_dir / "overlay-manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return {
        "output_dir": str(output_dir),
        "overlay_sha256": manifest["test_overlay_sha256"],
        "baseline_method_sha256": method_hashes["baseline"],
        "fixed_method_sha256": method_hashes["fixed"],
        "wrong_mutant_method_sha256": method_hashes["wrong_mutant"],
        "wrong_mutant_patch_sha256": manifest["mutant_patch_sha256"],
        "manifest_sha256": sha256_bytes(manifest_path.read_bytes()),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-repository", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            prepare(args.source_repository, args.protocol, args.output_dir),
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
