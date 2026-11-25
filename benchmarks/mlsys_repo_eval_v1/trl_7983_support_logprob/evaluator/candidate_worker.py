"""Run a pinned CPU behavior probe against one candidate module."""

from __future__ import annotations

import contextlib
import importlib.util
import json
import os
import sys
import types
from pathlib import Path


def _install_dependency_stubs() -> None:
    transformers = types.ModuleType("transformers")
    transformers.__path__ = []  # type: ignore[attr-defined]
    transformer_utils = types.ModuleType("transformers.utils")
    transformer_utils.is_triton_available = lambda: False  # type: ignore[attr-defined]
    sys.modules["transformers"] = transformers
    sys.modules["transformers.utils"] = transformer_utils

    trl = types.ModuleType("trl")
    trl.__path__ = []  # type: ignore[attr-defined]
    kernels = types.ModuleType("trl.kernels")
    kernels.__path__ = []  # type: ignore[attr-defined]
    trainer = types.ModuleType("trl.trainer")
    trainer.__path__ = []  # type: ignore[attr-defined]
    trainer_utils = types.ModuleType("trl.trainer.utils")
    setattr(
        trainer_utils,
        "maybe_gather_lm_head_ctx",
        lambda _weight, _bias: contextlib.nullcontext(),
    )
    for name, module in (
        ("trl", trl),
        ("trl.kernels", kernels),
        ("trl.trainer", trainer),
        ("trl.trainer.utils", trainer_utils),
    ):
        sys.modules[name] = module


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit("usage: candidate_worker.py RUNTIME_ROOT CANDIDATE_SOURCE")

    runtime_root = Path(sys.argv[1]).resolve(strict=True)
    candidate_source = Path(sys.argv[2]).resolve(strict=True)
    sys.path.insert(0, str(runtime_root))
    _install_dependency_stubs()

    import torch

    torch.set_num_threads(1)
    spec = importlib.util.spec_from_file_location(
        "trl.kernels.chunked_logprob", candidate_source
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("could not construct the candidate package module")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    payload = json.loads(sys.stdin.buffer.read())
    rows = []
    for case in payload["cases"]:
        try:
            weight = (
                torch.tensor(case["logits"], dtype=torch.float32)
                .reshape(-1, 1)
                .requires_grad_()
            )
            hidden = torch.ones((1, 1), dtype=torch.float32)
            targets = torch.tensor([case["target"]], dtype=torch.long)
            support_ids = torch.tensor([case["support"]], dtype=torch.long)
            outputs = module.ChunkedLogProbFunction.apply(
                hidden,
                weight,
                None,
                targets,
                1.0,
                case["chunk_size"],
                None,
                1.0,
                (),
                support_ids,
                None,
            )
            logprobs = outputs[0]
            (-logprobs.sum()).backward()
            if weight.grad is None:
                raise RuntimeError(
                    "candidate backward did not produce a weight gradient"
                )
            rows.append(
                {
                    "case": case["name"],
                    "status": "EXECUTED",
                    "logprobs": logprobs.detach().cpu().tolist(),
                    "weight_grad": weight.grad.detach().cpu().reshape(-1).tolist(),
                    "finite": bool(
                        torch.isfinite(logprobs).all()
                        and torch.isfinite(weight.grad).all()
                    ),
                }
            )
        except Exception as exc:
            rows.append(
                {
                    "case": case["name"],
                    "status": "TEST_ERROR",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )
    report = json.dumps(
        {
            "python_version": sys.version.split()[0],
            "torch_version": torch.__version__,
            "rows": rows,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    os.write(1, report.encode("utf-8") + b"\n")


if __name__ == "__main__":
    main()
