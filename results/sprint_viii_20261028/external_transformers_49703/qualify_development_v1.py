from __future__ import annotations

import ast
import hashlib
import importlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[3]
TASK_BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
BASE_SOURCE = TASK_BASE / "source/base/modeling_mask2former.py"
FIXED_SOURCE = TASK_BASE / "source/fixed/modeling_mask2former.py"
WORKSPACE = TASK_BASE / "candidate-template"
TASK_ROOT = TASK_BASE / "task"
GRADER_PATH = ROOT / "scripts/sprint_viii_transformers_49703_grader_v1.py"
SOURCE_PATH = "src/transformers/models/mask2former/modeling_mask2former.py"


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def load_trusted_class(path: Path, package_globals: dict):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    classes = [node for node in tree.body
               if isinstance(node, ast.ClassDef) and node.name == "Mask2FormerTransformerModule"]
    if len(classes) != 1:
        raise RuntimeError(f"expected one target class in {path}")
    namespace = dict(package_globals)
    code = compile(ast.Module(body=classes, type_ignores=[]), str(path), "exec")
    exec(code, namespace)
    return namespace["Mask2FormerTransformerModule"]


def model_registration_control(source: Path, package_globals: dict, config_type) -> dict:
    import torch

    cls = load_trusted_class(source, package_globals)

    def config(enforce: bool):
        return config_type(
            hidden_dim=32,
            feature_size=32,
            mask_feature_size=8,
            encoder_layers=1,
            decoder_layers=1,
            num_attention_heads=4,
            dim_feedforward=32,
            num_queries=4,
            train_num_points=16,
            enforce_input_projection=enforce,
        )

    projected = cls(in_features=16, config=config(True))
    state_keys = [key for key in projected.state_dict() if key.startswith("input_projections.")]
    parameter_names = [name for name, _ in projected.named_parameters()
                       if name.startswith("input_projections.")]
    identity = cls(in_features=32, config=config(False))
    identity_keys = [key for key in identity.state_dict() if key.startswith("input_projections.")]
    return {
        "trusted_upstream_class_source_sha256": sha256(source.read_bytes()),
        "projected_container_type": type(projected.input_projections).__name__,
        "projected_container_length": len(projected.input_projections),
        "projected_state_dict_parameter_count": len(state_keys),
        "projected_named_parameter_count": len(parameter_names),
        "projected_total_model_parameter_count": sum(p.numel() for p in projected.parameters()),
        "identity_container_type": type(identity.input_projections).__name__,
        "identity_container_length": len(identity.input_projections),
        "identity_projection_parameter_count": len(identity_keys),
        "torch_version": torch.__version__,
    }


def main() -> int:
    from scripts.sprint_viii_transformers_49703_grader_v1 import grade_bytes

    base_bytes = BASE_SOURCE.read_bytes()
    fixed_bytes = FIXED_SOURCE.read_bytes()
    old = b"self.input_projections = []"
    new = b"self.input_projections = nn.ModuleList()"
    if base_bytes.count(old) != 1 or fixed_bytes.count(new) != 1:
        raise RuntimeError("upstream base/fix source does not match the pinned one-line repair")
    wrong_bytes = fixed_bytes.replace(new, b"self.input_projections = nn.ModuleDict()", 1)
    if wrong_bytes == fixed_bytes:
        raise RuntimeError("wrong mutant was not constructed")

    controls = {
        "baseline": grade_bytes(base_bytes),
        "known_fix": grade_bytes(fixed_bytes),
        "incorrect_mutant_module_dict": grade_bytes(wrong_bytes),
    }
    if (controls["baseline"]["status"] != "FAIL"
            or controls["known_fix"]["status"] != "PASS"
            or controls["incorrect_mutant_module_dict"]["status"] != "FAIL"):
        raise RuntimeError(f"independent grader calibration failed: {controls}")

    package = importlib.import_module("transformers.models.mask2former.modeling_mask2former")
    config_module = importlib.import_module("transformers.models.mask2former.configuration_mask2former")
    behavior = {
        "baseline": model_registration_control(
            BASE_SOURCE, vars(package), config_module.Mask2FormerConfig,
        ),
        "known_fix": model_registration_control(
            FIXED_SOURCE, vars(package), config_module.Mask2FormerConfig,
        ),
    }
    if behavior["baseline"]["projected_state_dict_parameter_count"] != 0:
        raise RuntimeError(f"baseline did not reproduce the defect: {behavior['baseline']}")
    if (behavior["known_fix"]["projected_state_dict_parameter_count"] != 6
            or behavior["known_fix"]["projected_named_parameter_count"] != 6
            or behavior["known_fix"]["identity_projection_parameter_count"] != 0):
        raise RuntimeError(f"known fix did not preserve registration and identity paths: {behavior['known_fix']}")

    tracked = subprocess.run(
        ["git", "-C", str(WORKSPACE), "ls-tree", "-r", "--name-only", "HEAD"],
        check=True, capture_output=True, text=True,
    ).stdout.splitlines()
    source_sha = sha256((WORKSPACE / SOURCE_PATH).read_bytes())
    if tracked != [SOURCE_PATH] or source_sha != sha256(base_bytes):
        raise RuntimeError("candidate template is not isolated at the pinned upstream baseline")
    candidate_paths = [path.relative_to(WORKSPACE).as_posix()
                       for path in WORKSPACE.rglob("*")
                       if path.is_file() and ".git" not in path.parts]
    if candidate_paths != [SOURCE_PATH]:
        raise RuntimeError(f"candidate workspace contains unexpected files: {candidate_paths}")

    result = {
        "record_id": "transformers-49703-development-task-qualification-v1",
        "status": "QUALIFIED_DEVELOPMENT_NOT_HELDOUT",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "task_id": "transformers-49703-mask2former-projection-registration-development-v1",
        "task_family": "transformers-49698-mask2former-module-registration",
        "split": "development",
        "upstream": {
            "repository": "huggingface/transformers",
            "issue": "https://github.com/huggingface/transformers/issues/49698",
            "pull_request": "https://github.com/huggingface/transformers/pull/49703",
            "baseline_revision": "8e40ad7bc7019ba19ae8b0e7d98e60376d0fac2a",
            "fix_revision": "094669a133c54bafd2975c11a325140bc93b3459",
            "merge_commit": "f348d3dcfa9317bed603e01b8ca22b11fe59615c",
            "target_path": SOURCE_PATH,
            "baseline_source_sha256": sha256(base_bytes),
            "fix_source_sha256": sha256(fixed_bytes),
            "fix_diff": "self.input_projections = [] -> nn.ModuleList()",
        },
        "candidate_isolation": {
            "candidate_visible_source_files": candidate_paths,
            "candidate_local_base_commit": subprocess.run(
                ["git", "-C", str(WORKSPACE), "rev-parse", "HEAD"],
                check=True, capture_output=True, text=True,
            ).stdout.strip(),
            "candidate_template_source_sha256": source_sha,
            "known_fix_in_candidate_workspace": False,
            "grader_or_hidden_test_in_candidate_workspace": False,
            "git_upstream_history_in_candidate_workspace": False,
            "unexpected_tracked_paths": [],
        },
        "independent_grader": {
            "entrypoint": str(GRADER_PATH.relative_to(ROOT)),
            "source_sha256": sha256(GRADER_PATH.read_bytes()),
            "execution_boundary": "AST-only; candidate source is never imported or executed by grader",
            "calibration": controls,
            "actual_upstream_class_runtime_controls": behavior,
        },
        "exposure": {
            "prior_vare_task_family_overlap": False,
            "pretraining_exposure": "UNKNOWN",
            "heldout_allocation": False,
            "development_consumed_at": "not yet; exposure is recorded only when the model attempt starts",
            "limitation": "A recently merged public upstream fix may still be present in model pretraining; no absence claim is made.",
        },
        "runtime": {
            "python": sys.version,
            "platform": platform.platform(),
            "torch": importlib.metadata.version("torch"),
            "transformers": importlib.metadata.version("transformers"),
            "model_backend": "MLX Metal; model generation not run during qualification",
        },
        "qualification_decision": "GO_FOR_ONE_PRE-REGISTERED_DEVELOPMENT_CAMPAIGN",
        "qualification_rule": "baseline FAIL, exact upstream fix PASS, wrong ModuleDict mutant FAIL; actual base class loses all projection parameters, fixed class registers 6 projected state/parameter keys and preserves the 3-level identity path.",
    }
    out = TASK_BASE / "qualification/qualification-v1.json"
    payload = (json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode()
    fd = os.open(out, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    print(json.dumps({"status": result["status"], "baseline": controls["baseline"]["status"],
                      "fix": controls["known_fix"]["status"],
                      "wrong_mutant": controls["incorrect_mutant_module_dict"]["status"],
                      "runtime_base_projection_parameters": behavior["baseline"]["projected_state_dict_parameter_count"],
                      "runtime_fixed_projection_parameters": behavior["known_fix"]["projected_state_dict_parameter_count"],
                      "qualification_record": str(out.relative_to(ROOT))}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
