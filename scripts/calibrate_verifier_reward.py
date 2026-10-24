"""Sample a frozen four-task reward-only calibration with TinyStories-1M."""
from __future__ import annotations

import argparse
import asyncio
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import re
import resource
import subprocess
import sys
import time
import uuid

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "benchmarks/verifier_consistent_async_grpo_v7/reward-calibration-v1.json"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def atomic_new(path: Path, payload: bytes) -> None:
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"refusing to replace frozen calibration artifact {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    descriptor = os.open(temp, os.O_CREAT | os.O_EXCL | os.O_WRONLY
                         | getattr(os, "O_NOFOLLOW", 0), 0o600)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temp, path, follow_symlinks=False)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        try:
            temp.unlink()
        except FileNotFoundError:
            pass


def write_new(path: Path, value: object) -> None:
    atomic_new(path, json.dumps(value, indent=2, sort_keys=True,
                                ensure_ascii=False, allow_nan=False).encode() + b"\n")


def atomic_replace(path: Path, value: object) -> None:
    temp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    data = json.dumps(value, indent=2, sort_keys=True,
                      ensure_ascii=False, allow_nan=False).encode() + b"\n"
    descriptor = os.open(temp, os.O_CREAT | os.O_EXCL | os.O_WRONLY
                         | getattr(os, "O_NOFOLLOW", 0), 0o600)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        try:
            temp.unlink()
        except FileNotFoundError:
            pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def first_surface(text: str, surface_classes: dict[str, list[str]]) -> str | None:
    matches = []
    for category, aliases in surface_classes.items():
        for alias in aliases:
            match = re.search(rf"(?<![A-Za-z0-9_]){re.escape(alias)}(?![A-Za-z0-9_])",
                              text, flags=re.IGNORECASE)
            if match is not None:
                matches.append((match.start(), -len(alias), category))
    return min(matches)[2] if matches else None


def rewards(prompt: str, target: str, output: str, spec: dict) -> dict[str, float]:
    category = first_surface(output, spec["surface_classes"])
    if category is None:
        return {"v0": float(spec["v0"]["unknown_score"]),
                "v1": float(spec["v1"]["unknown_score"]),
                "surface_class": "unknown"}
    if category == "air":
        v1 = 0.0
    elif category == target:
        v1 = float(spec["v1"]["exact_score"])
    elif category in spec["v1"]["related_classes"][target]:
        v1 = float(spec["v1"]["related_score"])
    else:
        v1 = float(spec["v1"]["other_support_score"])
    return {"v0": float(spec["v0"]["class_scores"][category]),
            "v1": v1, "surface_class": category}


def peak_rss_bytes() -> int:
    raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(raw if sys.platform == "darwin" else raw * 1024)


def inventory(output: Path) -> dict[str, str]:
    return {str(path.relative_to(output)): sha(path)
            for path in sorted(output.rglob("*")) if path.is_file()
            and path.name != "SHA256.json" and not path.name.endswith(("-wal", "-shm"))}


async def sample(args, protocol: dict) -> dict:
    historical_protocol = json.loads((ROOT / "benchmarks/verifier_consistent_async_grpo_v7/protocol.json").read_text())
    reserved = set(historical_protocol["workload"]["training_prompts"])
    reserved.update(historical_protocol["workload"]["evaluation"]["prompts"])
    calibration_prompts = [task["prompt"] for task in protocol["sampling"]["tasks"]]
    require(len(calibration_prompts) == len(set(calibration_prompts)),
            "calibration prompts must be unique")
    require(not reserved.intersection(calibration_prompts),
            "calibration prompt overlaps a consumed or held-out v7 prompt")
    target_classes = set(protocol["reward_definition"]["v1"]["related_classes"])
    require(all(task["target_class"] in target_classes
                for task in protocol["sampling"]["tasks"]),
            "calibration target lacks a frozen target-conditioned rule")
    require(sys.version.split()[0] == protocol["runtime"]["python"],
            "Python version differs from frozen calibration")
    require(platform.machine() == "arm64", "calibration requires Apple Silicon")
    memsize = int(subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True).strip())
    require(memsize == 32 * 1024**3, "calibration requires the preflighted 32 GB host")
    for name, digest in protocol["model"]["files"].items():
        require(sha(args.model_dir / name) == digest, f"model file hash mismatch: {name}")
    require(subprocess.check_output(["git", "-C", str(args.rvl_source), "rev-parse", "HEAD"],
                                    text=True).strip() == protocol["runtime"]["rvl_revision"],
            "pinned RVL revision differs")
    require(subprocess.check_output(["git", "-C", str(args.rvl_source), "status", "--porcelain"],
                                    text=True).strip() == "M src/rvl_systems/hf_backend.py",
            "pinned RVL checkout state differs")
    backend_path = args.rvl_source / "src/rvl_systems/hf_backend.py"
    require(sha(backend_path) == protocol["runtime"]["backend_sha256"],
            "pinned RVL backend source hash differs")
    require(sha(ROOT / "scripts/benchmark_verifier_consistent_async_grpo.py")
            == protocol["source_identity"]["helper_sha256"],
            "pinned local generation helper identity differs")

    sys.path.insert(0, str(ROOT / "scripts"))
    import benchmark_verifier_consistent_async_grpo as helper
    helper.prepare_pinned_imports(args.rvl_source)
    helper.verify_import_origins(args.rvl_source)
    import torch
    import transformers
    from rvl_systems.hf_backend import HFLocalBackend

    require(torch.__version__ == protocol["runtime"]["torch"], "Torch version differs")
    require(transformers.__version__ == protocol["runtime"]["transformers"],
            "Transformers version differs")
    torch.set_num_threads(protocol["runtime"]["threads"])
    swap_before = helper.swap_used_bytes()
    started = time.perf_counter()
    cpu_started = time.process_time()
    backend = HFLocalBackend(str(args.model_dir),
                             max_new_tokens=protocol["sampling"]["max_new_tokens"],
                             device="cpu", precision="fp32")
    scoring = protocol["reward_definition"]
    groups = []
    for task_index, task in enumerate(protocol["sampling"]["tasks"]):
        seed = int(protocol["sampling"]["seed"] + 100 * task_index)
        samples = await backend.generate(
            task["id"], task["prompt"], n=protocol["sampling"]["group_size"],
            temperature=protocol["sampling"]["temperature"], seed=seed)
        require(len(samples) == protocol["sampling"]["group_size"],
                f"incomplete sample group: {task['id']}")
        rows = []
        for member, generated in enumerate(samples):
            grade = rewards(task["prompt"], task["target_class"], generated.response,
                            scoring)
            require(math.isfinite(grade["v0"]) and math.isfinite(grade["v1"]),
                    "non-finite dense reward")
            rows.append({
                "member": member,
                "prompt": generated.prompt,
                "output_text": generated.response,
                "token_count": generated.token_count,
                "behavior_logprob": generated.logprob,
                "generation_latency_s": generated.latency_s,
                "generation_metadata": generated.metadata,
                **grade,
            })
        group = {
            "task_id": task["id"], "target_class": task["target_class"],
            "prompt": task["prompt"], "seed": seed,
            "verifier_identity": {
                "protocol_sha256": sha(PROTOCOL),
                "v0_name": scoring["v0"]["name"],
                "v1_name": scoring["v1"]["name"],
            },
            "members": rows,
            "reward_summary": {
                version: {
                    "mean": sum(row[version] for row in rows) / len(rows),
                    "unique_values": sorted({row[version] for row in rows}),
                    "positive_count": sum(row[version] > 0 for row in rows),
                } for version in ("v0", "v1")
            },
        }
        path = args.output / "groups" / f"{task['id']}.json"
        write_new(path, group)
        groups.append(group)
        elapsed = time.perf_counter() - started
        rss = peak_rss_bytes()
        swap_now = helper.swap_used_bytes()
        swap_growth = (max(0, swap_now - swap_before)
                       if swap_now is not None and swap_before is not None else None)
        atomic_replace(args.output / "progress.json", {
            "status": "RUNNING", "completed_groups": len(groups),
            "last_task_id": task["id"], "elapsed_s": elapsed,
            "peak_rss_bytes": rss, "swap_growth_bytes": swap_growth,
        })
        limits = protocol["resource_limits"]
        require(rss <= limits["max_rss_bytes"], "peak RSS exceeded the frozen limit")
        require(elapsed <= limits["max_wall_seconds"], "wall time exceeded the frozen limit")
        require(swap_growth is None or swap_growth <= limits["max_positive_swap_growth_bytes"],
                "positive swap growth exceeded the frozen limit")

    members = [row for group in groups for row in group["members"]]
    shifts = [abs(row["v0"] - row["v1"]) for row in members]
    mean_shift = sum(shifts) / len(shifts)
    shifted_fraction = sum(value >= 0.25 for value in shifts) / len(shifts)
    nonconstant_groups = sum(len(group["reward_summary"]["v1"]["unique_values"]) > 1
                             for group in groups)
    gates = {
        "all_groups_nonconstant_v1": nonconstant_groups == len(groups),
        "mean_absolute_shift": mean_shift >= protocol["decision_rules"][
            "minimum_mean_absolute_reward_shift_v0_to_v1"],
        "fraction_shift_at_least_0_25": shifted_fraction >= protocol["decision_rules"][
            "minimum_fraction_with_absolute_reward_shift_at_least_0_25"],
    }
    summary = {
        "status": "CALIBRATION_PASS" if all(gates.values()) else "NO_GO_REWARD_CALIBRATION",
        "group_count": len(groups), "members": len(members),
        "v1_nonconstant_groups": nonconstant_groups,
        "v1_positive_rewards": sum(row["v1"] > 0 for row in members),
        "v0_positive_rewards": sum(row["v0"] > 0 for row in members),
        "unique_outputs": len({row["output_text"] for row in members}),
        "verifier_version_mean_absolute_reward_shift": mean_shift,
        "fraction_abs_shift_at_least_0_25": shifted_fraction,
        "gates": gates,
        "groups": [{"task_id": group["task_id"],
                    "v0": group["reward_summary"]["v0"],
                    "v1": group["reward_summary"]["v1"]} for group in groups],
        "model_revision": protocol["model"]["revision"],
        "protocol_sha256": sha(PROTOCOL),
        "source_sha256": {
            "sampler": sha(Path(__file__).resolve()),
            "auditor": protocol["source_identity"]["auditor_sha256"],
            "helper": protocol["source_identity"]["helper_sha256"],
            "backend": protocol["runtime"]["backend_sha256"],
        },
        "environment": {
            "python": sys.version, "torch": torch.__version__,
            "transformers": transformers.__version__, "device": "cpu fp32",
            "threads": protocol["runtime"]["threads"], "host_memory_bytes": memsize,
            "wall_s": time.perf_counter() - started,
            "cpu_s": time.process_time() - cpu_started,
            "peak_rss_bytes": peak_rss_bytes(),
            "swap_before_bytes": swap_before, "swap_after_bytes": helper.swap_used_bytes(),
        },
        "optimizer_updates": 0,
        "claim_limit": "Reward-only output-diversity and verifier-drift calibration; no GRPO update, asynchronous comparison, coding capability, or held-out quality result.",
    }
    return summary


async def main(args) -> None:
    require(not args.output.exists() and not args.output.is_symlink(),
            "calibration output path already exists")
    protocol = json.loads(PROTOCOL.read_text())
    require(protocol["schema"] == "vare.dense-verifier-reward-calibration.v1",
            "wrong calibration protocol")
    require(protocol["source_identity"]["sampler_sha256"] == sha(Path(__file__).resolve()),
            "sampler source differs from frozen calibration")
    require(protocol["source_identity"]["auditor_sha256"] == sha(
        ROOT / "scripts/audit_verifier_reward_calibration.py"),
        "independent auditor source differs from frozen calibration")
    args.output.mkdir(parents=True, exist_ok=False)
    write_new(args.output / "protocol.json", protocol)
    atomic_replace(args.output / "progress.json", {"status": "RUNNING", "completed_groups": 0})
    try:
        result = await sample(args, protocol)
        write_new(args.output / "summary.json", result)
        atomic_replace(args.output / "progress.json", {
            "status": result["status"], "completed_groups": result["group_count"],
            "elapsed_s": result["environment"]["wall_s"],
        })
    except BaseException as error:
        write_new(args.output / "failure.json", {
            "type": type(error).__name__, "message": str(error),
            "time_utc": datetime.now(timezone.utc).isoformat(),
            "completed_group_files": sorted(path.name for path in
                                             (args.output / "groups").glob("*.json")),
        })
        raise
    finally:
        write_new(args.output / "SHA256.json", inventory(args.output))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--rvl-source", type=Path, required=True)
    options = parser.parse_args()
    asyncio.run(main(options))
