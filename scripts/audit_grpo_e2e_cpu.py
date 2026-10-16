#!/usr/bin/env python3
"""Read-only checks of retained GRPO bytes, numerical vectors and raw metrics."""

import argparse
import hashlib
import json
import math
from pathlib import Path


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
        "scope": "Saved bytes/vectors/metrics and lineage; loss/optimizer reference and resumed-output equality are runner checks, not independently rerun by this auditor.",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    print(json.dumps(audit(parser.parse_args().output), indent=2))
