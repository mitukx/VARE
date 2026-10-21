from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from copy import deepcopy
from unittest.mock import patch

sys.path.insert(0, "/workspace/src")


def load_upstream_fixture():
    fixture = Path("/workspace/tests/utils/test_fusion_mapping.py")
    spec = importlib.util.spec_from_file_location("vare_public_fusion_fixture", fixture)
    if spec is None or spec.loader is None:
        raise AssertionError("upstream fusion fixture is unavailable")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    module = load_upstream_fixture()

    class IdempotenceCase(module.FusionMappingTest):
        def setUp(self):
            self.patch_mapping_patcher = patch.object(
                module.monkey_patching, "_monkey_patch_mapping_cache", {}
            )
            self.patch_mapping_patcher.start()
            self.discovery_cache_patcher = patch.object(
                module.fusion_mapping, "_FUSION_DISCOVERY_CACHE", {}
            )
            self.discovery_cache_patcher.start()
            if hasattr(module.fusion_mapping, "_REGISTERED_FUSION_SOURCE_PATTERNS"):
                self.registered_sources_patcher = patch.object(
                    module.fusion_mapping, "_REGISTERED_FUSION_SOURCE_PATTERNS", {}
                )
                self.registered_sources_patcher.start()
            self.checkpoint_conversion_mapping_cache = deepcopy(
                module.conversion_mapping._checkpoint_conversion_mapping_cache
            )

        def tearDown(self):
            self.patch_mapping_patcher.stop()
            self.discovery_cache_patcher.stop()
            if hasattr(self, "registered_sources_patcher"):
                self.registered_sources_patcher.stop()
            module.conversion_mapping._checkpoint_conversion_mapping_cache = deepcopy(
                self.checkpoint_conversion_mapping_cache
            )

        def test_repeat_registration(self):
            module.DummyFusionConfig.model_type = "vare_public_fusion_idempotence"
            config = module.DummyFusionConfig()
            module.register_fusion_patches(
                module.DummyFusionModel, config, fusion_config={"patch_embeddings": True}
            )
            before = [
                converter.source_patterns
                for converter in module.get_checkpoint_conversion_mapping(config.model_type)
            ]
            module.register_fusion_patches(
                module.DummyFusionModel, config, fusion_config={"patch_embeddings": True}
            )
            after = [
                converter.source_patterns
                for converter in module.get_checkpoint_conversion_mapping(config.model_type)
            ]
            assert before == after

    case = IdempotenceCase("test_repeat_registration")
    result = case.defaultTestResult()
    case.run(result)
    if result.errors or result.failures:
        details = "\n".join(message for _, message in [*result.errors, *result.failures])
        raise AssertionError(details)


if __name__ == "__main__":
    main()
