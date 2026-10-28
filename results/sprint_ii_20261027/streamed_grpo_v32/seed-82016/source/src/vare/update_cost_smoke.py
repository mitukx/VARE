"""One-update save/reload smoke for the exact-integer v2 checkpoint.

The acceptance function checks a record. It does not score the held-out pack
and it does not compare policies.
"""

from __future__ import annotations

import math
from typing import Mapping

from vare.heldout_exact_integer import evaluate_expression, load_task_pack
from vare.integer_seed_split import derive_seeds, task_from_seed

PROTOCOL_ID = "heldout-exact-integer-update-cost-smoke-v1"
PROTOCOL_SHA256 = "0d6959238b63b3b9a08b8db49c52a0905bbc51f617cd4b1ba8d5241da3c940d0"
V2_MANIFEST_SHA256 = "ee582b61fc2ca21d45bacb829c728bf47118d987f302b6f0f2eb8aeb82571278"
HELDOUT_PACK_SHA256 = "189b5e482b0f8c2b40ff36e0e1a84827699f348c11981a0ff912ce67323fd7fc"
LORA_KEYS = ("self_attn.q_proj", "self_attn.v_proj")


def train_example() -> dict[str, object]:
    seed = derive_seeds("train")[0]
    task = task_from_seed(seed, "train")
    return {
        "completion": f"#### {task.value}",
        "expression": task.expression,
        "prompt": task.prompt,
        "seed": task.seed,
        "split": "train",
        "value": task.value,
    }


def assert_train_example_is_outside_the_heldout_pack(pack_text: str) -> dict[str, object]:
    example = train_example()
    if evaluate_expression(str(example["expression"])) != example["value"]:
        raise ValueError("train example does not match its expression")
    rows = load_task_pack(pack_text)
    heldout_expressions = {row["expression"] for row in rows}
    heldout_prompts = {row["prompt"] for row in rows}
    if example["expression"] in heldout_expressions or example["prompt"] in heldout_prompts:
        raise ValueError("train example collides with the held-out pack")
    return example


def validate_protocol(protocol: Mapping[str, object]) -> None:
    if protocol.get("protocol_id") != PROTOCOL_ID:
        raise ValueError("unexpected smoke protocol id")
    if protocol.get("optimizer_steps") != 1:
        raise ValueError("the smoke allows one optimizer update")
    if protocol.get("batch_size") != 1 or protocol.get("grad_accumulation_steps") != 1:
        raise ValueError("the smoke allows one example and one update")
    if protocol.get("optimizer") != "adam" or protocol.get("learning_rate") != 1e-5:
        raise ValueError("optimizer pin drifted")
    if protocol.get("num_layers") != 1 or protocol.get("seed") != 20261015:
        raise ValueError("layer or seed pin drifted")
    if protocol.get("mask_prompt") is not True:
        raise ValueError("the smoke masks the prompt")
    if protocol.get("backends_are_replicates") is not False:
        raise ValueError("the two backends are separate smokes")
    lora = protocol.get("lora")
    if not isinstance(lora, Mapping):
        raise ValueError("missing lora pin")
    if (
        lora.get("rank") != 8
        or lora.get("dropout") != 0.0
        or lora.get("scale") != 20.0
        or list(lora.get("keys", [])) != list(LORA_KEYS)
    ):
        raise ValueError("lora pin drifted")
    if protocol.get("v2_manifest_sha256") != V2_MANIFEST_SHA256:
        raise ValueError("v2 result pin drifted")
    if protocol.get("heldout_pack_sha256") != HELDOUT_PACK_SHA256:
        raise ValueError("held-out pack pin drifted")
    example = train_example()
    if protocol.get("train_seed") != example["seed"]:
        raise ValueError("train seed drifted")
    if protocol.get("train_expression") != example["expression"]:
        raise ValueError("train expression drifted")
    if protocol.get("train_completion") != example["completion"]:
        raise ValueError("train completion drifted")
    if protocol.get("scores_heldout_pack") is not False:
        raise ValueError("the smoke must not score the held-out pack")
    if protocol.get("frozen_before_smoke_output") is not True:
        raise ValueError("the smoke protocol was not frozen before a run")
    models = protocol.get("models")
    if not isinstance(models, Mapping):
        raise ValueError("missing model pins")
    mlx = models.get("mlx")
    transformers = models.get("transformers")
    if not isinstance(mlx, Mapping) or not isinstance(transformers, Mapping):
        raise ValueError("missing model pins")
    if mlx.get("revision") != "4f83f8f146fdf28b512a06562b671d7af4fab457":
        raise ValueError("mlx revision drifted")
    if transformers.get("revision") != "aa8e72537993ba99e69dfaafa59ed015b17504d1":
        raise ValueError("transformers revision drifted")


def _finite_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def accept_smoke_record(record: Mapping[str, object]) -> dict[str, object]:
    """Fail closed unless the record is one finite update with a matching reload."""
    reasons: list[str] = []
    if record.get("optimizer_steps") != 1:
        reasons.append("not_one_update")
    if not _finite_number(record.get("loss")):
        reasons.append("loss_not_finite")
    if record.get("trainable_parameters_changed") is not True:
        reasons.append("parameters_unchanged")
    if record.get("reload_matches_saved") is not True:
        reasons.append("reload_mismatch")
    if record.get("heldout_pack_used_for_update") is not False:
        reasons.append("heldout_pack_used")
    resources = record.get("resources")
    if not isinstance(resources, Mapping):
        reasons.append("missing_resources")
    else:
        wall = resources.get("wall_seconds")
        peak = resources.get("peak_memory_bytes")
        if not _finite_number(wall) or float(wall) < 0:  # type: ignore[arg-type]
            reasons.append("missing_wall_seconds")
        if not _finite_number(peak) or float(peak) <= 0:  # type: ignore[arg-type]
            reasons.append("missing_peak_memory")
    return {
        "accepted": not reasons,
        "establishes": "one update, save/reload, and a resource record only",
        "protocol_id": PROTOCOL_ID,
        "reasons": reasons,
    }
