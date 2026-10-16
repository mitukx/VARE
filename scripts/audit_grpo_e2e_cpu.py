#!/usr/bin/env python3
"""Read-only checks of retained GRPO bytes, numerical vectors and raw metrics."""

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import patch


def require(value, message):
    if not value:
        raise ValueError(message)


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def model_digest(state):
    h = hashlib.sha256()
    for name, value in sorted(state.items()):
        h.update(name.encode())
        h.update(str(value.dtype).encode())
        h.update(str(tuple(value.shape)).encode())
        h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def audit_unclipped_reference(root, model_dir, rvl_source):
    """Replay the existing oracle, detecting scaling that clipping can hide.

    Actual preclip vectors are reconstructed from retained clipped vectors and
    the recorded norm using pinned Torch's coefficient, not directly captured.
    No new rollout, production update, promotion or artifact mutation occurs.
    """
    import torch
    from validate_grpo_e2e_cpu import reference_update

    protocol = json.loads((root / "protocol.json").read_text())
    require(
        digest(Path(__file__).with_name("validate_grpo_e2e_cpu.py"))
        == protocol["runner_sha256"],
        "reference oracle source mismatch",
    )
    for name, expected in protocol["model_files"].items():
        require(digest(model_dir / name) == expected, "reference model file mismatch")
    for stem in ["hf_backend", "hf_trainer"]:
        require(
            digest(rvl_source / f"src/rvl_systems/{stem}.py")
            == digest(root / f"{stem}.source.py"),
            "reference backend source mismatch",
        )
    sys.path.insert(0, str(rvl_source / "src"))
    from rvl_systems.hf_backend import HFLocalBackend
    from rvl_systems.hf_trainer import HFCausalLMGRPOTrainer, HFTTrainerConfig

    torch.set_num_threads(1)
    backend = HFLocalBackend(str(model_dir), device="cpu", precision="fp32")
    model = backend.model
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(
            name.startswith("transformer.wpe.") or name.startswith("transformer.ln_f.")
        )
    config = HFTTrainerConfig(learning_rate=1e-4)
    HFCausalLMGRPOTrainer(model, config=config)  # Reuse its dropout configuration.
    original_clip = torch.nn.utils.clip_grad_norm_
    rows = []
    for record in json.loads((root / "rounds.json").read_text()):
        batch = [
            SimpleNamespace(
                attempt=SimpleNamespace(**exp["attempt"]),
                verification=SimpleNamespace(**exp["verification"]),
            )
            for exp in record["batch"]
        ]
        before = torch.load(
            root / (batch[0].attempt.policy_id + ".pt"), weights_only=True
        )
        captured = {}

        def capture(parameters, *a, **kw):
            captured["vectors"] = {
                name: p.grad.detach().clone()
                for name, p in model.named_parameters()
                if p.grad is not None
            }
            norm = original_clip(parameters, *a, **kw)
            captured["norm"] = float(norm)
            return norm

        with patch.object(torch.nn.utils, "clip_grad_norm_", capture):
            reference = reference_update(model, before, batch, config)
        numerical = torch.load(
            root / (record["candidate"] + "-numerics.pt"), weights_only=True
        )
        norm = record["metrics"]["grad_norm"]
        require(
            math.isclose(norm, captured["norm"], rel_tol=2e-6, abs_tol=2e-6),
            "unclipped gradient norm mismatch",
        )
        coefficient = (
            config.max_grad_norm / (torch.tensor(norm, dtype=torch.float32) + 1e-6)
        ).clamp(max=1)
        reconstructed = {
            name: value / coefficient
            for name, value in numerical["actual_clipped_grads"].items()
        }
        error = max(
            float((reconstructed[name] - value).abs().max())
            for name, value in captured["vectors"].items()
        )
        require(
            math.isfinite(error) and error < 2e-6,
            "unclipped gradient reconstruction mismatch",
        )
        require(
            abs(reference["loss"] - record["metrics"]["loss"]) < 2e-6,
            "replayed loss mismatch",
        )
        state = torch.load(root / (record["candidate"] + ".pt"), weights_only=True)
        require(
            max(
                float((reference["model"][name] - value).abs().max())
                for name, value in state["model"].items()
            )
            < 2e-6,
            "replayed parameter mismatch",
        )
        bad_error, bad_postclip_error = None, None
        if norm > 0:
            bad_preclip = {
                name: value * len(batch) for name, value in reconstructed.items()
            }
            bad_coefficient = (
                config.max_grad_norm
                / (torch.tensor(norm * len(batch), dtype=torch.float32) + 1e-6)
            ).clamp(max=1)
            bad_error = max(
                float((bad_preclip[name] - value).abs().max())
                for name, value in captured["vectors"].items()
            )
            bad_postclip_error = max(
                float((bad_preclip[name] * bad_coefficient - value).abs().max())
                for name, value in numerical["actual_clipped_grads"].items()
            )
            require(bad_error > 2e-6, "batch-scaling negative control is uninformative")
        rows.append(
            {
                "round": record["round"],
                "actual_preclip_norm": norm,
                "reference_preclip_norm": captured["norm"],
                "reconstructed_preclip_max_error": error,
                "incorrect_batch_scaling_error": bad_error,
                "incorrect_batch_scaling_postclip_error": bad_postclip_error,
            }
        )
    return {
        "pass": True,
        "rounds": rows,
        "nonzero_batch_scaling_controls": sum(
            r["incorrect_batch_scaling_error"] is not None for r in rows
        ),
        "max_reconstructed_preclip_error": max(
            r["reconstructed_preclip_max_error"] for r in rows
        ),
        "auditor_sha256": digest(Path(__file__)),
        "scope": "Read-only replay of the existing full-graph oracle; actual preclip vectors reconstructed from stored clipped vectors and norm. No new rollout or native-trainer run; primary measurements unchanged.",
    }


def audit(root):
    import torch

    manifest = json.loads((root / "SHA256.json").read_text())
    require(
        set(manifest) == {p.name for p in root.iterdir()} - {"SHA256.json"},
        "artifact inventory differs",
    )
    for name, expected in manifest.items():
        require(
            Path(name).name == name and digest(root / name) == expected,
            f"artifact hash mismatch: {name}",
        )
    summary = json.loads((root / "summary.json").read_text())
    protocol = json.loads((root / "protocol.json").read_text())
    require(
        summary["protocol_sha256"] == digest(root / "protocol.json"),
        "protocol mismatch",
    )
    require(
        digest(root / "validate_grpo_e2e_cpu.source.py") == protocol["runner_sha256"],
        "runner mismatch",
    )
    for name, expected in summary["source_sha256"].items():
        require(digest(root / name) == expected, "source snapshot mismatch")
    require(summary["status"] == "PASS", "run did not pass")
    n = summary["rounds"]
    records = json.loads((root / "rounds.json").read_text())
    require(
        len(records) == n == summary["candidate_updates"], "missing candidate update"
    )
    require([r["round"] for r in records] == list(range(n)), "round identity mismatch")
    require(
        summary["actual_optimizer_calls_including_fault_and_resume"] == n + 3,
        "optimizer call count mismatch",
    )
    stages = json.loads((root / "stage_samples_s.json").read_text())
    counts = {
        "rollout": 8 * n,
        "verification": 8 * n,
        "training": n,
        "validation": n,
        "evaluation": 2 * n,
        "freshness_prune": n,
        "round": n,
    }
    require(stages.keys() == counts.keys(), "stage coverage mismatch")
    for name, values in stages.items():
        require(
            len(values) == counts[name]
            and all(math.isfinite(x) and x >= 0 for x in values),
            "invalid latency samples",
        )
        metric = summary["stages"][name]
        expected = {
            "n": len(values),
            "total_s": sum(values),
            "p50_s": sorted(values)[len(values) // 2],
            "p95_s": sorted(values)[math.ceil(len(values) * 0.95) - 1],
        }
        require(metric == expected, f"latency summary mismatch: {name}")
    rollouts = json.loads((root / "rollouts.json").read_text())[: 8 * n]
    require(
        len(rollouts) == 8 * n
        and [r["created_step"] for r in rollouts] == list(range(8 * n)),
        "rollout coverage mismatch",
    )
    wall = summary["wall_s_with_reference_checkpoint_audits"]
    require(wall > 0 and math.isfinite(wall), "invalid wall time")
    require(
        math.isclose(
            8 * n / wall, summary["throughput_tasks_s_including_audits"], rel_tol=1e-12
        ),
        "task throughput mismatch",
    )
    tokens = sum(r["metadata"]["token_count"] for r in rollouts)
    require(
        math.isclose(
            tokens / wall,
            summary["throughput_tokens_s_including_audits"],
            rel_tol=1e-12,
        ),
        "token throughput mismatch",
    )
    evaluations = json.loads((root / "evaluations.json").read_text())
    require(len(evaluations) == 2 * n, "missing checkpoint evaluation pair")
    expected_eval_ids = {f"grpo-independent-{summary['seed']}-{i}" for i in range(4)}
    for report in evaluations:
        scores = report["metadata"]["per_task_scores"]
        require(
            report["n"] == 4 and scores.keys() == expected_eval_ids,
            "evaluation task coverage mismatch",
        )
        require(
            all(math.isfinite(x) and 0 <= x <= 1 for x in scores.values()),
            "invalid evaluation score",
        )
        require(
            math.isclose(sum(scores.values()) / 4, report["primary"], abs_tol=1e-12),
            "evaluation aggregate mismatch",
        )
    active, version = "policy-0", 0
    checkpoint_hashes = {
        active: model_digest(
            torch.load(root / (active + ".pt"), weights_only=True)["model"]
        )
    }
    for record in records:
        batch = record["batch"]
        require(
            batch and len(batch) == len(record["advantages"]),
            "empty or misaligned batch",
        )
        groups = {}
        for exp in batch:
            attempt = exp["attempt"]
            require(
                attempt["policy_id"] == active and attempt["policy_version"] == version,
                "stale training provenance",
            )
            require(
                attempt["output"] == attempt["metadata"]["rvl_generation"]["response"],
                "response binding mismatch",
            )
            groups.setdefault(attempt["metadata"]["vare_rollout_group"], []).append(exp)
        require(all(len(group) == 4 for group in groups.values()), "partial GRPO group")
        numerical = torch.load(
            root / (record["candidate"] + "-numerics.pt"), weights_only=True
        )
        state = torch.load(root / (record["candidate"] + ".pt"), weights_only=True)
        checkpoint_hashes[record["candidate"]] = model_digest(state["model"])
        require(
            checkpoint_hashes[record["candidate"]] == record["candidate_model_sha256"],
            "checkpoint tensor digest mismatch",
        )
        require(
            {
                r["policy_id"]
                for r in evaluations[2 * record["round"] : 2 * record["round"] + 2]
            }
            == {active, record["candidate"]},
            "evaluation policy pair mismatch",
        )
        for left_key, right_key, error_key in (
            ("actual_clipped_grads", "reference_clipped_grads", "gradient_error"),
            (
                "actual_trainable_parameters",
                "reference_trainable_parameters",
                "parameter_error",
            ),
        ):
            left, right = numerical[left_key], numerical[right_key]
            require(
                left and left.keys() == right.keys(),
                "numerical vector coverage mismatch",
            )
            error = max(float((left[k] - right[k]).abs().max()) for k in left)
            require(
                math.isfinite(error) and error < 2e-6 and error == record[error_key],
                "numerical comparison mismatch",
            )
        require(
            len(state["optimizer"]["state"]) == len(numerical["actual_clipped_grads"]),
            "optimizer state coverage mismatch",
        )
        require(
            all(
                torch.equal(v, state["model"][k])
                for k, v in numerical["actual_trainable_parameters"].items()
            ),
            "numerics belong to another checkpoint",
        )
        require(
            all(
                float(v["step"]) == version + 1
                for v in state["optimizer"]["state"].values()
            ),
            "optimizer counter mismatch",
        )
        incumbent = torch.load(root / (active + ".pt"), weights_only=True)
        require(
            torch.equal(state["cpu_rng"], incumbent["cpu_rng"]),
            "checkpoint RNG mismatch",
        )
        if record["decision"]["accepted"]:
            active, version = record["candidate"], version + 1
        require(
            record["active_policy"] == [active, version], "promotion lineage mismatch"
        )
    acknowledgments = json.loads((root / "worker-acknowledgments.json").read_text())
    require(
        len(acknowledgments) == 8 * n + 4, "rollout acknowledgment coverage mismatch"
    )
    for ack, attempt in zip(acknowledgments[: 8 * n], rollouts, strict=True):
        require(
            all(
                ack[key] == attempt[field]
                for key, field in (
                    ("policy_id", "policy_id"),
                    ("policy_version", "policy_version"),
                    ("step", "created_step"),
                )
            ),
            "rollout acknowledgment identity mismatch",
        )
        require(
            ack["model_sha256"] == checkpoint_hashes[ack["policy_id"]],
            "rollout checkpoint hash mismatch",
        )
    faults = summary["faults"]
    require(
        {f["fault"] for f in faults}
        == {
            "rollout_policy_version",
            "response_binding",
            "partial_group",
            "mixed_policy",
            "stale",
            "logprob_length",
            "nonfinite_logprob",
            "after_optimizer",
        },
        "fault coverage mismatch",
    )
    require(
        all(f["rejected"] is True and f["state_unchanged"] is True for f in faults),
        "fault control failed",
    )
    require(summary["resume_next_update_exact"] is True, "resume comparison failed")
    return {
        "pass": True,
        "files_hashed": len(manifest),
        "candidate_updates": n,
        "latency_samples_verified": sum(map(len, stages.values())),
        "fault_controls": len(faults),
        "scope": "Default mode checks saved bytes/vectors/metrics and lineage. Optional model-backed mode replays the existing full-graph oracle. Resumed-output equality remains an online runner check.",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--rvl-source", type=Path)
    args = parser.parse_args()
    if bool(args.model_dir) != bool(args.rvl_source):
        parser.error("--model-dir and --rvl-source must be supplied together")
    result = audit(args.output)
    if args.model_dir:
        result["unclipped_reference_replay"] = audit_unclipped_reference(
            args.output, args.model_dir, args.rvl_source
        )
    print(json.dumps(result, indent=2))
