#!/usr/bin/env python3
"""Exercise Transformers #49412 registration behavior against pinned source bytes.

This harness imports the upstream fusion_mapping.py module while replacing model
discovery and its registry dependencies with small in-memory test doubles. It
does not execute candidate code, run upstream tests, load a model, or access a
hidden evaluator.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import resource
import sys
import time
import tracemalloc
import types
from pathlib import Path


BASE_REVISION = "cb33194ad6152bd9fad6305378d92db385dd7b32"
FIXED_REVISION = "01269bc99768ad45f8aeaeb701e1315371e25239"
MODULE_PATH = Path("src/transformers/fusion_mapping.py")

NO_DEDUPE_BLOCK = """        existing_converters = [
            existing for existing in existing_converters if tuple(existing.source_patterns) not in previous_sources
        ]"""
CLEAR_ALL_BLOCK = """        existing_converters = [
            existing for existing in existing_converters if tuple(existing.source_patterns) not in previous_sources
        ]"""
CONFLICT_CHECK = """        for converter in converters:
            source_patterns = tuple(converter.source_patterns)
            existing_converter = existing_converter_sources.get(source_patterns)
            if existing_converter is not None:
                raise ValueError(
                    f"Fusion {fusion_name} for model type {model_type} conflicts with an existing conversion mapping "
                    f"for source patterns {source_patterns}."
                )"""


class ProbeError(RuntimeError):
    pass


class Transform:
    def __init__(self, source_patterns: tuple[str, ...], target_patterns: tuple[str, ...]):
        self.source_patterns = source_patterns
        self.target_patterns = target_patterns


def load_module(source_file: Path, mutation: str):
    source = source_file.read_text(encoding="utf-8")
    if mutation == "no-deduplication":
        if source.count(NO_DEDUPE_BLOCK) != 1:
            raise ProbeError("no-deduplication mutation anchor count was not exactly one")
        source = source.replace(NO_DEDUPE_BLOCK, "        existing_converters = list(existing_converters)", 1)
    elif mutation == "clear-unrelated-mappings":
        if source.count(CLEAR_ALL_BLOCK) != 1:
            raise ProbeError("clear-unrelated-mappings mutation anchor count was not exactly one")
        source = source.replace(CLEAR_ALL_BLOCK, "        existing_converters = []", 1)
    elif mutation == "allow-conflicting-mapping":
        if source.count("            if existing_converter is not None:") != 1:
            raise ProbeError("allow-conflicting-mapping mutation anchor count was not exactly one")
        source = source.replace("            if existing_converter is not None:", "            if False:", 1)

    # Keep the upstream source module itself under test. Its imports are replaced
    # with minimal registry/model-discovery doubles so this probe is deterministic
    # and does not initialize a model or depend on optional runtime packages.
    torch = types.ModuleType("torch")
    nn = types.ModuleType("torch.nn")
    nn.Module = type("Module", (), {})
    nn.Conv3d = type("Conv3d", (), {})
    nn.Linear = type("Linear", (), {})
    torch.Tensor = type("Tensor", (), {})
    torch.nn = nn

    registry: dict[str, list[Transform]] = {}
    conversion_mapping = types.ModuleType("transformers.conversion_mapping")
    conversion_mapping.get_checkpoint_conversion_mapping = lambda key: list(registry[key]) if key in registry else None

    def register_mapping(key, mapping, overwrite=False):
        if key in registry and not overwrite:
            raise ValueError(f"mapping already exists: {key}")
        registry[key] = list(mapping)

    conversion_mapping.register_checkpoint_conversion_mapping = register_mapping

    core_model_loading = types.ModuleType("transformers.core_model_loading")
    core_model_loading.Conv3dToLinear = type("Conv3dToLinear", (), {})
    core_model_loading.WeightConverter = Transform
    core_model_loading.WeightRenaming = Transform
    core_model_loading.WeightTransform = Transform

    monkey_patching = types.ModuleType("transformers.monkey_patching")
    monkey_patching.register_patch_mapping = lambda mapping, overwrite=False: None
    utils = types.ModuleType("transformers.utils")
    logging = types.ModuleType("transformers.utils.logging")
    logging.get_logger = lambda name: types.SimpleNamespace(info=lambda *args, **kwargs: None)
    utils.logging = logging

    modules = {
        "torch": torch,
        "torch.nn": nn,
        "transformers": types.ModuleType("transformers"),
        "transformers.conversion_mapping": conversion_mapping,
        "transformers.core_model_loading": core_model_loading,
        "transformers.monkey_patching": monkey_patching,
        "transformers.utils": utils,
        "transformers.utils.logging": logging,
    }
    previous = {name: sys.modules.get(name) for name in modules}
    sys.modules.update(modules)
    try:
        module_name = "transformers.fusion_mapping"
        spec = importlib.util.spec_from_loader(module_name, loader=None, origin=str(source_file))
        module = importlib.util.module_from_spec(spec)
        module.__package__ = "transformers"
        exec(compile(source, str(source_file), "exec"), module.__dict__)
        sys.modules[module_name] = module
        module._probe_registry = registry
        module._probe_executed_source_sha256 = hashlib.sha256(source.encode("utf-8")).hexdigest()
        return module, previous, modules
    except BaseException:
        for name, old in previous.items():
            if old is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = old
        raise


def unload_module(previous, modules):
    sys.modules.pop("transformers.fusion_mapping", None)
    for name, old in previous.items():
        if old is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = old


class FakeConfig:
    model_type = "vare_49412_probe"


class FakeModel:
    config_class = FakeConfig


class FakeSpec:
    def __init__(self, source="patch_embed.proj.weight", target="patch_embed.linear_proj.weight"):
        self.source = source
        self.target = target

    def make_transforms(self, config):
        return [Transform((self.source,), (self.target,))]


def install_discovery_stub(module):
    module._discover_fusable_modules = lambda *args, **kwargs: {"ProbeModule": object}


def register(module, spec, fusion_name="probe_fusion"):
    module._register_module_fusion(FakeModel, object(), fusion_name, spec)


def same_fusion_is_idempotent(module):
    install_discovery_stub(module)
    register(module, FakeSpec())
    register(module, FakeSpec())
    values = module._probe_registry[FakeConfig.model_type]
    assert len(values) == 1, f"expected one converter after repeat, found {len(values)}"
    assert values[0].source_patterns == ("patch_embed.proj.weight",)


def unrelated_mapping_survives(module):
    install_discovery_stub(module)
    registry = module._probe_registry
    unrelated = Transform(("unrelated.weight",), ("unrelated.target",))
    registry[FakeConfig.model_type] = [unrelated]
    register(module, FakeSpec())
    register(module, FakeSpec())
    values = registry[FakeConfig.model_type]
    assert len(values) == 2, f"expected unrelated and fusion converters, found {len(values)}"
    assert any(item is unrelated for item in values), "unrelated converter was removed"
    assert sum(item.source_patterns == ("patch_embed.proj.weight",) for item in values) == 1


def unrelated_conflict_is_rejected(module):
    install_discovery_stub(module)
    conflicting = Transform(("patch_embed.proj.weight",), ("external.target",))
    module._probe_registry[FakeConfig.model_type] = [conflicting]
    try:
        register(module, FakeSpec())
    except ValueError as exc:
        assert "conflicts with an existing conversion mapping" in str(exc)
        assert module._probe_registry[FakeConfig.model_type] == [conflicting]
        return
    raise AssertionError("conflicting converter was silently accepted")


def run_case(case: str, source_file: Path):
    mutation = {
        "fixed": "none",
        "mutant-no-deduplication": "no-deduplication",
        "mutant-clear-unrelated-mappings": "clear-unrelated-mappings",
        "mutant-allow-conflict": "allow-conflicting-mapping",
        "baseline": "none",
    }[case]
    source = source_file.read_bytes()
    start = time.perf_counter()
    tracemalloc.start()
    module, previous, modules = load_module(source_file, mutation)
    assertions = {
        "fixed": [same_fusion_is_idempotent, unrelated_mapping_survives, unrelated_conflict_is_rejected],
        "baseline": [same_fusion_is_idempotent],
        "mutant-no-deduplication": [same_fusion_is_idempotent],
        "mutant-clear-unrelated-mappings": [unrelated_mapping_survives],
        "mutant-allow-conflict": [unrelated_conflict_is_rejected],
    }[case]
    passed = []
    failures = []
    for assertion in assertions:
        # Each property starts with isolated in-memory registries and module state.
        if hasattr(module, "_probe_registry"):
            module._probe_registry.clear()
        for attr in ("_FUSION_DISCOVERY_CACHE", "_REGISTERED_FUSION_SOURCE_PATTERNS"):
            if hasattr(module, attr):
                value = getattr(module, attr)
                if hasattr(value, "clear"):
                    value.clear()
        try:
            assertion(module)
            passed.append(assertion.__name__)
        except (AssertionError, ValueError) as exc:
            failures.append({"property": assertion.__name__, "error": f"{type(exc).__name__}: {exc}"})
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    unload_module(previous, modules)

    if case == "fixed":
        result = "PASS" if len(passed) == len(assertions) else "FAIL"
    elif case == "baseline":
        result = "BASELINE_FAIL_REPRODUCED" if len(failures) == 1 else "FAIL"
    else:
        result = "MUTANT_KILLED" if len(failures) == 1 else "MUTANT_SURVIVED"

    record = {
        "case": case,
        "source_sha256": hashlib.sha256(source).hexdigest(),
        "executed_module_sha256": module._probe_executed_source_sha256,
        "mutation": mutation,
        "result": result,
        "properties_passed": passed,
        "property_failures": failures,
        "wall_seconds": round(time.perf_counter() - start, 6),
        "process_max_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        "python_tracemalloc_peak_bytes": peak,
    }
    print(json.dumps(record, sort_keys=True))
    return 0 if result in {"PASS", "BASELINE_FAIL_REPRODUCED", "MUTANT_KILLED"} else 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", required=True, choices=[
        "baseline", "fixed", "mutant-no-deduplication",
        "mutant-clear-unrelated-mappings", "mutant-allow-conflict",
    ])
    parser.add_argument("--source", required=True, type=Path, help="Path to the pinned fusion_mapping.py bytes")
    parser.add_argument("--expected-source-sha256", required=True)
    args = parser.parse_args()
    if not args.source.is_file():
        parser.error(f"source module not found: {args.source}")
    actual = hashlib.sha256(args.source.read_bytes()).hexdigest()
    if actual != args.expected_source_sha256:
        parser.error(f"source hash mismatch: expected {args.expected_source_sha256}, got {actual}")
    return run_case(args.case, args.source)


if __name__ == "__main__":
    raise SystemExit(main())
