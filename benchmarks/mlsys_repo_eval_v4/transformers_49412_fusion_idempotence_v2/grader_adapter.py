"""Candidate-side probe; expected idempotence and conflict outcomes stay in the parent."""
from __future__ import annotations

import sys
from typing import Any

sys.path.insert(0, "/subject/src")


def _fixture(model_type: str):
    import torch.nn as nn
    from transformers import PretrainedConfig
    from transformers.modeling_utils import PreTrainedModel

    class VisionConfig(PretrainedConfig):
        base_config_key = "vision_config"

        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.in_channels = 3
            self.patch_size = 2
            self.temporal_patch_size = 2
            self.patch_embed_stride = (2, 2, 2)

    VisionConfig.model_type = f"{model_type}_vision"

    class FusionConfig(PretrainedConfig):
        sub_configs = {"vision_config": VisionConfig}

        def __init__(self, vision_config=None, **kwargs):
            super().__init__(**kwargs)
            self.vision_config = VisionConfig() if vision_config is None else vision_config

    # Class bodies do not close over a variable assigned in the same scope.
    FusionConfig.model_type = model_type

    class PatchEmbedding(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed_dim = 8
            self.proj = nn.Conv3d(3, 8, kernel_size=(2, 2, 2), stride=(2, 2, 2), bias=True)

    class Model(PreTrainedModel):
        config_class = FusionConfig

        def __init__(self, config):
            super().__init__(config)
            self.patch_embed = PatchEmbedding()
            self.post_init()

    return FusionConfig, Model


def repeated_registration(request: dict[str, Any]) -> dict[str, Any]:
    import transformers.conversion_mapping as conversion_mapping
    import transformers.fusion_mapping as fusion_mapping
    import transformers.monkey_patching as monkey_patching

    model_type = str(request["model_type"])
    monkey_patching._monkey_patch_mapping_cache.clear()
    fusion_mapping._FUSION_DISCOVERY_CACHE.clear()
    config_cls, model_cls = _fixture(model_type)
    config = config_cls()
    fusion_mapping.register_fusion_patches(model_cls, config, fusion_config={"patch_embeddings": True})
    before = [
        item.source_patterns
        for item in conversion_mapping.get_checkpoint_conversion_mapping(model_type)
    ]
    fusion_mapping.register_fusion_patches(model_cls, config, fusion_config={"patch_embeddings": True})
    after = [
        item.source_patterns
        for item in conversion_mapping.get_checkpoint_conversion_mapping(model_type)
    ]
    return {"repeat_accepted": True, "mapping_unchanged": before == after}
