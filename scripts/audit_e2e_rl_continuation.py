"""Independent byte/tensor/continuity audit for the CPU v2 experiment."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
from pathlib import Path
import random
import sqlite3
import statistics
import subprocess

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]

LEGACY_EXPECTED_SOURCES = frozenset({
    "benchmarks/e2e_rl_continuation_v2/protocol.json",
    "requirements/e2e-rl-real-model-v1.txt",
    "scripts/audit_e2e_rl_continuation.py",
    "scripts/benchmark_e2e_rl_continuation.py",
    "scripts/benchmark_e2e_rl_real_model.py",
    "src/vare/attestation.py",
    "src/vare/checkpoint_artifacts.py",
    "src/vare/engine.py",
    "src/vare/lag.py",
    "src/vare/promotion.py",
    "src/vare/promotion_journal.py",
    "src/vare/replay.py",
    "src/vare/reward_evidence.py",
    "src/vare/runtime.py",
    "tests/test_e2e_rl_continuation.py",
})
RUNTIME_PACKAGE_SOURCES = frozenset({
    "src/vare/__init__.py",
    "src/vare/__main__.py",
    "src/vare/_grader_guard.py",
    "src/vare/_public.py",
    "src/vare/answer_scoring.py",
    "src/vare/agent_edit_format.py",
    "src/vare/agent_task_contract.py",
    "src/vare/attestation.py",
    "src/vare/blob_budget.py",
    "src/vare/checkpoint_artifacts.py",
    "src/vare/cli.py",
    "src/vare/config.py",
    "src/vare/container_isolation.py",
    "src/vare/controls.py",
    "src/vare/curriculum.py",
    "src/vare/demo.py",
    "src/vare/durable.py",
    "src/vare/engine.py",
    "src/vare/entrypoint.py",
    "src/vare/environments/__init__.py",
    "src/vare/environments/campaign.py",
    "src/vare/environments/catalog.py",
    "src/vare/environments/factory.py",
    "src/vare/environments/runner.py",
    "src/vare/environments/spec.py",
    "src/vare/environments/verifier.py",
    "src/vare/eval_stats.py",
    "src/vare/evidence.py",
    "src/vare/failure.py",
    "src/vare/failure_triage.py",
    "src/vare/format_vs_correctness.py",
    "src/vare/format_vs_correctness_seeded_summary_v3.py",
    "src/vare/format_vs_correctness_v2.py",
    "src/vare/heldout_exact_integer.py",
    "src/vare/integer_seed_split.py",
    "src/vare/integrations/__init__.py",
    "src/vare/integrations/chat_completions.py",
    "src/vare/integrations/rvl.py",
    "src/vare/integrations/rvl_grpo.py",
    "src/vare/integrations/rvl_trajectory.py",
    "src/vare/interfaces.py",
    "src/vare/isolated_candidate.py",
    "src/vare/isolated_hooks.py",
    "src/vare/lag.py",
    "src/vare/lifecycle.py",
    "src/vare/meta_improvement.py",
    "src/vare/mlsys_coding_actions.py",
    "src/vare/mlsys_coding_comparison.py",
    "src/vare/mlsys_repo_task.py",
    "src/vare/mlxlm_1912_repo_task.py",
    "src/vare/promotion.py",
    "src/vare/promotion_journal.py",
    "src/vare/protocol.py",
    "src/vare/replay.py",
    "src/vare/reward_evidence.py",
    "src/vare/runner.py",
    "src/vare/runtime.py",
    "src/vare/safety.py",
    "src/vare/sandboxed_repository.py",
    "src/vare/separated_candidate_harness.py",
    "src/vare/separated_grading.py",
    "src/vare/telemetry.py",
    "src/vare/trajectory_admission.py",
    "src/vare/trajectory_dispatch.py",
    "src/vare/trajectory_learner.py",
    "src/vare/trajectory_promotion.py",
    "src/vare/trajectory_provenance.py",
    "src/vare/trajectory_reference.py",
    "src/vare/trajectory_safety.py",
    "src/vare/trajectory_store.py",
    "src/vare/trajectory_workers.py",
    "src/vare/trusted_evaluation.py",
    "src/vare/trusted_scoring.py",
    "src/vare/types.py",
    "src/vare/transformers_repo_task.py",
    "src/vare/update_cost_smoke.py",
    "src/vare/verifiers.py",
})
# This historical 80-file inventory is intentionally explicit. Do not derive
# it from the current runtime inventory: new runtime files must not invalidate
# the immediately preceding fully snapshotted campaign.
PREVIOUS_EXPECTED_SOURCES = frozenset({
    "benchmarks/e2e_rl_continuation_v2/protocol.json",
    "requirements/e2e-rl-real-model-v1.txt",
    "scripts/audit_e2e_rl_continuation.py",
    "scripts/benchmark_e2e_rl_continuation.py",
    "scripts/benchmark_e2e_rl_real_model.py",
    "src/vare/__init__.py",
    "src/vare/__main__.py",
    "src/vare/_grader_guard.py",
    "src/vare/_public.py",
    "src/vare/answer_scoring.py",
    "src/vare/attestation.py",
    "src/vare/blob_budget.py",
    "src/vare/checkpoint_artifacts.py",
    "src/vare/cli.py",
    "src/vare/config.py",
    "src/vare/container_isolation.py",
    "src/vare/controls.py",
    "src/vare/curriculum.py",
    "src/vare/demo.py",
    "src/vare/durable.py",
    "src/vare/engine.py",
    "src/vare/entrypoint.py",
    "src/vare/environments/__init__.py",
    "src/vare/environments/campaign.py",
    "src/vare/environments/catalog.py",
    "src/vare/environments/factory.py",
    "src/vare/environments/runner.py",
    "src/vare/environments/spec.py",
    "src/vare/environments/verifier.py",
    "src/vare/eval_stats.py",
    "src/vare/evidence.py",
    "src/vare/failure.py",
    "src/vare/failure_triage.py",
    "src/vare/format_vs_correctness.py",
    "src/vare/format_vs_correctness_seeded_summary_v3.py",
    "src/vare/format_vs_correctness_v2.py",
    "src/vare/heldout_exact_integer.py",
    "src/vare/integer_seed_split.py",
    "src/vare/integrations/__init__.py",
    "src/vare/integrations/rvl.py",
    "src/vare/integrations/rvl_grpo.py",
    "src/vare/integrations/rvl_trajectory.py",
    "src/vare/interfaces.py",
    "src/vare/isolated_hooks.py",
    "src/vare/lag.py",
    "src/vare/lifecycle.py",
    "src/vare/meta_improvement.py",
    "src/vare/mlsys_repo_task.py",
    "src/vare/promotion.py",
    "src/vare/promotion_journal.py",
    "src/vare/protocol.py",
    "src/vare/replay.py",
    "src/vare/reward_evidence.py",
    "src/vare/runner.py",
    "src/vare/runtime.py",
    "src/vare/safety.py",
    "src/vare/sandboxed_repository.py",
    "src/vare/telemetry.py",
    "src/vare/trajectory_admission.py",
    "src/vare/trajectory_dispatch.py",
    "src/vare/trajectory_learner.py",
    "src/vare/trajectory_promotion.py",
    "src/vare/trajectory_provenance.py",
    "src/vare/trajectory_reference.py",
    "src/vare/trajectory_safety.py",
    "src/vare/trajectory_store.py",
    "src/vare/trajectory_workers.py",
    "src/vare/trusted_evaluation.py",
    "src/vare/trusted_scoring.py",
    "src/vare/types.py",
    "src/vare/update_cost_smoke.py",
    "src/vare/verifiers.py",
    "tests/test_e2e_rl_continuation.py",
})
# Campaign source inventories are versioned. A published version is frozen:
# a genuinely new source tree gets a new version instead of an edit to an
# older one, so earlier fully snapshotted campaigns keep auditing exactly.
# v3 is the 83-file inventory in use until the learning pipeline package.
INVENTORY_V3_SOURCES = LEGACY_EXPECTED_SOURCES | RUNTIME_PACKAGE_SOURCES
# Runtime modules of src/vare/learning_pipeline. Each is reached from the
# package entry points (pipeline imports, the lora_worker subprocess) and is
# hash-frozen by protocols/coding_policy_learning_mlx_4b_v1.json.
LEARNING_PIPELINE_SOURCES = frozenset({
    "src/vare/learning_pipeline/__init__.py",
    "src/vare/learning_pipeline/comparison.py",
    "src/vare/learning_pipeline/curriculum.py",
    "src/vare/learning_pipeline/edit_format.py",
    "src/vare/learning_pipeline/exclusions.py",
    "src/vare/learning_pipeline/grpo.py",
    "src/vare/learning_pipeline/lora_worker.py",
    "src/vare/learning_pipeline/pipeline.py",
    "src/vare/learning_pipeline/policy.py",
    "src/vare/learning_pipeline/prompting.py",
    "src/vare/learning_pipeline/repo_env.py",
    "src/vare/learning_pipeline/resources.py",
    "src/vare/learning_pipeline/scoring.py",
    "src/vare/learning_pipeline/sources.py",
    "src/vare/learning_pipeline/training.py",
})
INVENTORY_V4_SOURCES = INVENTORY_V3_SOURCES | LEARNING_PIPELINE_SOURCES
# v5 adds the MLX GRPO trainer. It is not hash-frozen by the v1 learning
# protocol (v1 never ran GRPO) and is reached from scripts/grpo_vertical_slice.py.
GRPO_TRAINER_SOURCES = frozenset({
    "src/vare/learning_pipeline/grpo_mlx.py",
})
INVENTORY_V5_SOURCES = INVENTORY_V4_SOURCES | GRPO_TRAINER_SOURCES
# Exact-match registry; a campaign is identified by its complete source set.
SOURCE_INVENTORIES = {
    "v1-pinned": LEGACY_EXPECTED_SOURCES,
    "v2": PREVIOUS_EXPECTED_SOURCES,
    "v3": INVENTORY_V3_SOURCES,
    "v4": INVENTORY_V4_SOURCES,
    "v5": INVENTORY_V5_SOURCES,
}
CURRENT_SOURCE_INVENTORY = "v5"
EXPECTED_SOURCES = SOURCE_INVENTORIES[CURRENT_SOURCE_INVENTORY]
# Versions whose campaigns record the recovery endpoint at the resumed-round end.
EXACT_RECOVERY_ENDPOINT_INVENTORIES = frozenset({"v3", "v4", "v5"})

# The frozen v2 campaign is tied to the commit and source tree that produced it.
# This independent pin prevents a re-sealed archive from choosing its own code.
PINNED_LOCKED_BASE_COMMIT = "56ba2fefc69a8851f0457646b7031fe8bf2f2c80"
PINNED_LOCKED_SOURCE_SHA256 = {
    "benchmarks/e2e_rl_continuation_v2/protocol.json": "668b0c9c8ce3959fccc4eabe6926cc797e55262abfb4e91ad08b73dee4fdff34",
    "requirements/e2e-rl-real-model-v1.txt": "690eda84362435cdbf8906f37f5f929866161e36a1b8308096a90d511677cfe8",
    "scripts/audit_e2e_rl_continuation.py": "077cd96d26a8d899481f31dadf6ceac2bd1eef1d0a43bb9ebd4156ea37e701b1",
    "scripts/benchmark_e2e_rl_continuation.py": "d9c7006dd208b541b95550d07c05e6527069cda0c7328190f81d44df64e15182",
    "scripts/benchmark_e2e_rl_real_model.py": "669c6a892e41f95728f505d0c724f2570492c0d7cfeae80e29e9abbed414bae2",
    "src/vare/attestation.py": "785e74d7d0cd6d41a73cf75cc65c093a0afb8d5082ad3f11dd828770616e93a3",
    "src/vare/checkpoint_artifacts.py": "98f96cf1bde57011a7e787c0bf61820df6a89ae37f1c043e3dfa73288c052b5c",
    "src/vare/engine.py": "bcd914e14de291073a6d4c93cc3c3fbac2874b810f108ec47debb8691ae25f66",
    "src/vare/lag.py": "1a7185c0b17187c13e6369f3d10584eabe2ac697d5afbd8bfd867bef07c92a57",
    "src/vare/promotion.py": "73f2fcbd3358d1f9153a53012f49ebe41ca0d65f555ed6457c8b98ccc708474b",
    "src/vare/promotion_journal.py": "2e7b98951427aa63cff61ba948a3e4eb2e480b1378ed3c891e9df20225876ee7",
    "src/vare/replay.py": "2ff8258d3fdb43d7b34180f1a1e11a4a5cbd0a795cec3dddb990f41aaecb0a93",
    "src/vare/reward_evidence.py": "a5b5979b8f647a61d0d411acf032a60608fac2045a4d96299cec1e7471206292",
    "src/vare/runtime.py": "d1df9c0680b4fb4fd602694ae79eb5937dc391f53596d05a56f521ce79ecebb5",
    "tests/test_e2e_rl_continuation.py": "f56d8251f02e2c67b217321d51ae00c0b2fa60ec195baafebff46470c13f053e",
}


def source_inventory_version(source_names):
    """Name the registered inventory that exactly equals the snapshotted sources."""
    matches=[version for version,names in SOURCE_INVENTORIES.items() if source_names==names]
    need(len(matches)==1,"pinned source inventory differs")
    return matches[0]


def validate_source_inventory(manifest, source_names, locked):
    """Keep the frozen archive pinned while accepting fully snapshotted reruns."""
    version=source_inventory_version(source_names)
    if version == "v1-pinned":
        need(locked and manifest.get("base_commit")==PINNED_LOCKED_BASE_COMMIT,
             "locked campaign commit differs from the independently pinned revision")
        need(manifest.get("sources")==PINNED_LOCKED_SOURCE_SHA256,
             "locked campaign source snapshots differ from the independently pinned revision")
        return
    commit=manifest.get("base_commit")
    need(type(commit) is str and len(commit)==40 and
         all(char in "0123456789abcdef" for char in commit),
         "campaign base commit is not a full SHA")
    committed_sources=source_hashes_from_commit(commit,source_names)
    need(manifest.get("sources")==committed_sources,
         "source snapshot differs from recorded base commit")


def source_hashes_from_commit(commit, source_names):
    """Hash the named Git blobs at a commit without extracting untrusted paths."""
    git_env={**os.environ,"GIT_NO_REPLACE_OBJECTS":"1"}
    try:
        tree=subprocess.run(
            ["git","-C",str(REPOSITORY_ROOT),"ls-tree","-r","-z","--full-tree",
             commit,"--",*sorted(source_names)],
            check=True,capture_output=True,env=git_env,
        )
        entries={}
        for record in tree.stdout.split(b"\0"):
            if not record:
                continue
            metadata,path=record.split(b"\t",1)
            mode,object_type,oid=metadata.decode("ascii").split()
            name=os.fsdecode(path)
            need(object_type=="blob" and mode in {"100644","100755"},
                 "pinned source is not a regular Git blob")
            need(name not in entries,"Git tree returned a duplicate source path")
            entries[name]=oid
        need(set(entries)==source_names,
             "source inventory differs from recorded base commit")
        blobs=subprocess.run(
            ["git","-C",str(REPOSITORY_ROOT),"cat-file","--batch"],
            input="".join(f"{oid}\n" for oid in entries.values()).encode("ascii"),
            check=True,capture_output=True,env=git_env,
        )
    except (OSError,subprocess.CalledProcessError) as exc:
        raise ValueError("campaign base commit is unavailable for source verification") from exc
    committed_sources={}
    output=blobs.stdout
    offset=0
    for name,expected_oid in entries.items():
        header_end=output.find(b"\n",offset)
        need(header_end>=0,"Git blob response header is incomplete")
        response_oid,object_type,size_text=output[offset:header_end].decode("ascii").split()
        need(response_oid==expected_oid and object_type=="blob",
             "Git blob response does not match the source tree")
        size=int(size_text)
        data_start=header_end+1
        data_end=data_start+size
        need(data_end<len(output) and output[data_end:data_end+1]==b"\n",
             "Git blob response body is incomplete")
        committed_sources[name]=hashlib.sha256(output[data_start:data_end]).hexdigest()
        offset=data_end+1
    need(offset==len(output),"Git blob response contains unexpected data")
    return committed_sources


def need(ok, message):
    if not ok:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def json_sha(value):
    raw=json.dumps(value,sort_keys=True,separators=(",",":"),allow_nan=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def promotion_expected(incumbent, candidate):
    """The retained v2 protocol promotes on a nondecreasing independent mean."""
    return candidate["primary"]>=incumbent["primary"]


def expected_tasks(seed, round_index, count, evaluation=False):
    """Reconstruct the benchmark's deterministic cohort without trusting its artifacts."""
    rng = random.Random(seed * 100000 + round_index * 100 + int(evaluation))
    kind = "audit" if evaluation else "train"
    return [{"family": "transport", "id": f"v2-{kind}-s{seed}-r{round_index}-i{i}",
             "metadata": {},
             "prompt": f"v2 {kind} {rng.getrandbits(64):016x}: return A. Answer:"}
            for i in range(count)]


def resource_sample_metrics(directory):
    samples = [json.loads(line) for line in (Path(directory)/"resources.jsonl").read_text().splitlines()]
    need(len(samples)>=2,"resource samples are incomplete")
    times=[sample["time"] for sample in samples]
    rss=[sample["rss_bytes"] for sample in samples]
    cpu=[sample["cpu_delta_s"] for sample in samples]
    need(all(type(value) in (int,float) and math.isfinite(value) for value in (*times,*cpu)) and
         all(type(value) is int and value>=0 for value in rss) and
         all(right>left for left,right in zip(times,times[1:])),
         "resource samples contain invalid time, CPU, or RSS values")
    window=times[-1]-times[0]
    need(window>0 and all(value>=0 for value in cpu),"resource sample window is invalid")
    return {"sample_count":len(samples),"sample_start_monotonic_s":times[0],
            "sample_end_monotonic_s":times[-1],"sample_window_s":window,
            "sampled_cpu_core_percent":100*sum(cpu)/window,
            "sampled_tree_peak_rss_bytes":max(rss)}


def fingerprint(value):
    """Independent encoding oracle, including dtype/shape, Adam and RNG bytes."""
    import torch
    digest = hashlib.sha256()
    def walk(item):
        if isinstance(item, torch.Tensor):
            digest.update(str((str(item.dtype),list(item.shape))).encode())
            digest.update(item.detach().contiguous().numpy().tobytes())
        elif isinstance(item, dict):
            for key in sorted(item,key=str):
                digest.update(str(key).encode())
                walk(item[key])
        elif isinstance(item,(tuple,list)):
            for member in item:
                walk(member)
        else:
            digest.update(repr(item).encode())
    walk(value)
    return digest.hexdigest()


def checkpoint(case, policy_id, mode):
    if mode == "minimal":
        path = case / f"{policy_id}.pt"
        return path.read_bytes(), sha(path)
    uri = (case / "checkpoints/artifacts.sqlite").resolve().as_uri()+"?mode=ro"
    # The checkpoint-store schema is inspected independently rather than
    # trusting a supplied checkpoint digest or the benchmark's read helper.
    with sqlite3.connect(uri,uri=True) as db:
        row = db.execute("SELECT digest,size FROM artifacts WHERE policy_id=?",(policy_id,)).fetchone()
    need(row is not None,"checkpoint missing")
    raw = (case / "checkpoints/objects" / row[0]).read_bytes()
    need(len(raw)==row[1] and hashlib.sha256(raw).hexdigest()==row[0],"checkpoint integrity mismatch")
    return raw,row[0]


def audit(root, check_tensors=False, model_dir=None):
    root = Path(root)
    sums = read(root / "SHA256SUMS.json")
    actual={str(p.relative_to(root)) for p in root.rglob('*') if p.is_file() and p.name!="SHA256SUMS.json"}
    need(actual==set(sums),"artifact inventory differs")
    for name,digest in sums.items():
        need(sha(root/name)==digest,f"changed artifact: {name}")
    manifest,summary,rows=read(root/"manifest.json"),read(root/"summary.json"),read(root/"progress.json")
    source_names=set(manifest.get("sources",{}))
    source_inventory=source_inventory_version(source_names)
    need(manifest.get("source_sha256")==manifest.get("sources"),
         "pinned source inventory differs")
    for name,digest in manifest["sources"].items():
        need(sha(root/"sources"/name)==digest,"source snapshot differs")
    protocol_path=root/"sources/benchmarks/e2e_rl_continuation_v2/protocol.json"
    need(manifest.get("protocol_sha256")==sha(protocol_path),
         "declared protocol snapshot digest differs")
    protocol=read(protocol_path)
    args=manifest["arguments"]
    locked=(args["workers"]==protocol["workers"] and args["seeds"]==protocol["seeds"] and
            args["rounds"]==protocol["rounds"] and args["tasks"]==protocol["rollouts"] and
            args["eval_tasks"]==protocol["evaluation_tasks"] and not args["skip_faults"])
    validate_source_inventory(manifest,source_names,locked)
    need(manifest["protocol_conformant"]==summary["protocol_conformant"]==locked,"reduced run claims locked protocol")
    expected_id=protocol["schema"] if locked else protocol["schema"]+".diagnostic"
    need(manifest["protocol_id"]==summary["protocol_id"]==expected_id,"protocol identity differs")
    expected_cases={f"normal-w{w}-s{s}-{m}" for w in args["workers"] for s in args["seeds"] for m in ("minimal","guarded")}
    if not args["skip_faults"]:
        expected_cases.update(f"kill-{cut}-s{s}-guarded" for s in args["seeds"] for cut in protocol["cutpoints"])
    need({r["directory"] for r in rows}==expected_cases and len(rows)==len(expected_cases),"campaign case budget differs")
    if check_tensors or model_dir:
        import torch
        torch.set_num_threads(1)
    if model_dir:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        for name,digest in protocol["model_files"].items():
            need(sha(Path(model_dir)/name)==digest,"model revision differs")
        model=AutoModelForCausalLM.from_pretrained(model_dir,local_files_only=True).eval()
        tokenizer=AutoTokenizer.from_pretrained(model_dir,local_files_only=True)
        choices=[tokenizer.encode(" A")[0],tokenizer.encode(" B")[0]]
    references,round_records={},{}
    tensor_states=evaluation_pairs=evaluation_reports=replayed_scores=0
    for row in rows:
        case=root/row["directory"]
        measured_resources=resource_sample_metrics(case)
        config=read(case/"config.json")
        need(config.get("source_sha256")==manifest["sources"],
             "case source inventory differs from the pinned source set")
        need((row["seed"],row["workers"],row["mode"])==
             (config["seed"],config["workers"],config["mode"]),
             "progress case metadata differs from retained configuration")
        if row["cutpoint"]=="none":
            expected_directory=f"normal-w{row['workers']}-s{row['seed']}-{row['mode']}"
        else:
            need(row["cutpoint"] in protocol["cutpoints"] and row["mode"]=="guarded",
                 "invalid fault case metadata")
            expected_directory=f"kill-{row['cutpoint']}-s{row['seed']}-guarded"
        need(row["directory"]==expected_directory,"case directory identity differs from progress metadata")
        case_summary=read(case/"summary.json")
        need(row.get("case_wall_s")==case_summary.get("case_wall_s"),
             "progress case wall differs from retained case summary")
        timing_path=case/"case_timing.json"
        case_timing=None
        if timing_path.exists():
            timing=read(timing_path)
            case_timing=timing
            start,end=timing.get("start_monotonic_s"),timing.get("end_monotonic_s")
            need(type(start) in (int,float) and math.isfinite(start) and
                 type(end) in (int,float) and math.isfinite(end) and end>start,
                 "case timing endpoints are invalid")
            need(start<=measured_resources["sample_start_monotonic_s"]<
                 measured_resources["sample_end_monotonic_s"]<=end,
                 "resource sample timestamps fall outside retained case timing endpoints")
            elapsed=end-start
            need(case_summary.get("case_wall_s")==elapsed and row.get("case_wall_s")==elapsed,
                 "case wall summary differs from retained timing endpoints")
        else:
            need(locked and manifest.get("base_commit")==PINNED_LOCKED_BASE_COMMIT,
                 "case timing endpoints are missing outside the pinned legacy campaign")
        need(row["protocol_id"]==config["protocol_id"]==expected_id,"case protocol identity differs")
        need(row["protocol_conformant"]==config["protocol_conformant"]==locked,"case conformance differs")
        commits=[read(case/"rounds"/f"r{i:04d}"/"committed.json") for i in range(args["rounds"])]
        need([r["completed_round"] for r in commits]==list(range(args["rounds"])),"round gap or duplicate")
        cursor_path=case/"cursor.json"
        need(cursor_path.is_file(),"final durable cursor is missing")
        cursor=read(cursor_path)
        final_commit=commits[-1]
        need(type(cursor.get("completed_round")) is int and
             cursor["completed_round"]==final_commit["completed_round"] and
             cursor.get("active_policy")==final_commit["active_policy"] and
             cursor.get("active_checkpoint_sha256")==final_commit["active_checkpoint_sha256"] and
             cursor.get("active_state_sha256")==final_commit["active_state_sha256"] and
             type(cursor.get("accepted_updates")) is int and
             cursor["accepted_updates"]==final_commit["accepted_updates"],
             "final durable cursor differs from the final committed round")
        need(row.get("stale_replay_rejected")==sum(commit["stale_replay_rejected"] for commit in commits),
             "progress stale-replay total differs from committed rounds")
        need(row["completed_rounds"]==row["rounds"]==len(commits),
             "completed round count differs from committed records")
        need(commits[-1]["active_state_sha256"]==row["final_state_sha256"],"final state differs")
        round_records[row["directory"]]=commits
        if row["cutpoint"]=="none":
            references[(row["workers"],row["seed"],row["mode"])]=commits
        canary=read(case/"canary.json")
        canary_attempt=canary.get("attempt",{})
        need(type(canary_attempt.get("created_step")) is int and
             canary_attempt.get("policy_id")=="policy-0" and
             type(canary_attempt.get("policy_version")) is int and canary_attempt["policy_version"]==0,
             "stale replay canary identity is invalid")
        attempts=[json.loads(line) for line in (case/"optimizer_attempts.jsonl").read_text().splitlines()]
        expected_attempts=args["rounds"]+int(row["cutpoint"] in ("post_optimizer","candidate_saved","journal_prepared"))
        need(len(attempts)==row["physical_optimizer_attempts"]==expected_attempts,"physical attempt accounting differs")
        attempts_by_round={}
        attempt_identities=set()
        for attempt in attempts:
            round_index=attempt.get("round_index")
            need(type(round_index) is int and 0<=round_index<len(commits),
                 "invalid optimizer attempt round identity")
            learner_attempt=attempt.get("learner_attempt")
            need(type(learner_attempt) is int and learner_attempt>=0,
                 "invalid optimizer attempt identity")
            identity=(round_index,learner_attempt)
            need(identity not in attempt_identities,"duplicate optimizer attempt identity")
            attempt_identities.add(identity)
            committed=commits[round_index]
            need(attempt["candidate_state_sha256"]==committed["candidate_state_sha256"] and
                 attempt["candidate_model_sha256"]==committed["update"]["parameters_after_sha256"],
                 "physical optimizer attempt differs from committed candidate")
            attempts_by_round.setdefault(round_index,set()).add(learner_attempt)
        kill_round=protocol["kill_round"]
        prepublication_cutpoints=("post_optimizer","candidate_saved","journal_prepared")
        for index in range(len(commits)):
            if row["cutpoint"]=="none":
                expected_attempt_ids={0}
            elif row["cutpoint"] in prepublication_cutpoints:
                expected_attempt_ids=({0} if index<kill_round else
                                      {0,1} if index==kill_round else {1})
            else:
                expected_attempt_ids={0} if index<=kill_round else {1}
            need(attempts_by_round.get(index,set())==expected_attempt_ids,
                 "optimizer attempt identities do not match the kill-round schedule")
        expected_setup_attempts={attempt for identities in attempts_by_round.values() for attempt in identities}
        setup_paths=list(case.glob("setup-attempt*.json"))
        setup_ids={int(path.stem.removeprefix("setup-attempt")) for path in setup_paths}
        need(setup_ids==expected_setup_attempts,"setup attempts differ from learner process attempts")
        for setup_path in setup_paths:
            setup=read(setup_path)
            endpoint_pairs=(
                ("rollout_pool_start_s","rollout_pool_started_monotonic_s","rollout_pool_ready_monotonic_s"),
                ("evaluation_pool_start_s","evaluation_pool_started_monotonic_s","evaluation_pool_ready_monotonic_s"),
            )
            for duration_key,start_key,end_key in endpoint_pairs:
                if start_key not in setup or end_key not in setup:
                    need(locked and manifest.get("base_commit")==PINNED_LOCKED_BASE_COMMIT,
                         "pool startup timing endpoints are missing outside the pinned legacy campaign")
                    continue
                duration,start,end=(setup.get(key) for key in (duration_key,start_key,end_key))
                need(type(duration) in (int,float) and math.isfinite(duration) and duration>=0 and
                     type(start) in (int,float) and math.isfinite(start) and
                     type(end) in (int,float) and math.isfinite(end) and end>start and
                     math.isclose(duration,end-start,rel_tol=0,abs_tol=0.01),
                     "pool startup summary differs from retained timing endpoints")
                need(case_timing is not None and
                     case_timing["start_monotonic_s"]<=start<end<=case_timing["end_monotonic_s"],
                     "pool startup endpoints fall outside retained case timing")
        trace=[json.loads(line) for line in (case/"trace.jsonl").read_text().splitlines()]
        trace_stages={}
        known_stages={"checkpoint_hash","checkpoint_save","checkpoint_recovery","evaluation",
                      "freshness","promotion","replay","rollback","rollout",
                      "rollout_verified","round","training","verification"}
        for span in trace:
            stage=span.get("stage")
            round_index=span.get("round_index")
            learner_attempt=span.get("learner_attempt")
            start,end,duration=(span.get(key) for key in ("start","end","duration_s"))
            need(stage in known_stages,"unknown trace stage")
            need(type(round_index) is int and -1<=round_index<len(commits) and
                 type(learner_attempt) is int and learner_attempt>=0 and
                 type(start) in (int,float) and math.isfinite(start) and
                 type(end) in (int,float) and math.isfinite(end) and end>=start and
                 type(duration) in (int,float) and math.isfinite(duration) and
                 duration==end-start,
                 "trace span timestamps or duration are invalid")
            if round_index>=0:
                need(learner_attempt in attempts_by_round.get(round_index,set()),
                     "trace span references an unknown learner attempt")
            trace_stages.setdefault((stage,round_index,learner_attempt),[]).append(span)
        spans=[r for r in trace if r["stage"]=="round"]
        expected_timed_rounds=[i for i,commit in enumerate(commits)
                               if not commit.get("recovered_publication")]
        span_indices=[span.get("round_index") for span in spans]
        need(sorted(span_indices)==expected_timed_rounds and len(set(span_indices))==len(spans),
             "round trace spans do not cover each timed committed round exactly once")
        for span in spans:
            index=span["round_index"]
            need(type(index) is int and type(span.get("learner_attempt")) is int and
                 span["learner_attempt"]==max(attempts_by_round[index]) and
                 type(span.get("duration_s")) in (int,float) and
                 math.isfinite(span["duration_s"]) and span["duration_s"]>0,
                 "round trace span identity or duration differs from committed round")
        spans=[span["duration_s"] for span in spans]
        need(row["round_wall_s"]==sum(spans),"round wall differs")
        need(row["tasks_s"]==row["tokens_s"]==args["tasks"]*len(spans)/sum(spans),"throughput differs")
        for round_index,attempt_ids in attempts_by_round.items():
            for attempt_id in attempt_ids:
                def stage_count(stage):
                    return len(trace_stages.get((stage,round_index,attempt_id),[]))
                expected_batch_size=args["tasks"]
                for stage in ("rollout","rollout_verified","verification"):
                    need(stage_count(stage)==expected_batch_size,
                         f"{stage} trace coverage differs from rollout batch")
                canary_stale=int(round_index>0 and canary_attempt["policy_version"]<
                                 commits[round_index-1]["accepted_updates"])
                expected_freshness=2*expected_batch_size+(expected_batch_size if row["mode"]=="guarded" else 0)+canary_stale
                need(stage_count("freshness")==expected_freshness,
                     "freshness trace coverage differs from executed admission checks")
                need(stage_count("replay")==1,
                     "replay latency span does not cover each learner attempt")
                interrupted_train=(row["cutpoint"] in ("post_optimizer","candidate_saved") and
                                   round_index==kill_round and attempt_id==0)
                need(stage_count("training")==int(not interrupted_train),
                     "training trace coverage differs from the optimizer cutpoint")
                interrupted_promotion=(row["cutpoint"] in ("journal_prepared","pointer_published") and
                                       round_index==kill_round and attempt_id==0)
                need(stage_count("promotion")==int(not interrupted_promotion and not interrupted_train),
                     "promotion trace coverage differs from the publication cutpoint")
                expected_evaluations=0 if interrupted_train else 2
                evaluation_spans=trace_stages.get(("evaluation",round_index,attempt_id),[])
                need(len(evaluation_spans)==expected_evaluations,
                     "evaluation trace coverage differs from retained evaluation pairs")
                interrupted_before_save=(row["cutpoint"]=="post_optimizer" and
                                         round_index==kill_round and attempt_id==0)
                need(stage_count("checkpoint_save")==int(not interrupted_before_save),
                     "checkpoint-save trace coverage differs from optimizer cutpoint")
        need(len(trace_stages.get(("checkpoint_save",-1,0),[]))==1 and
             not trace_stages.get(("checkpoint_save",-1,1)),
             "initial checkpoint-save trace coverage differs")
        need(sum(len(v) for (stage,_,_),v in trace_stages.items() if stage=="checkpoint_save")==
             1+sum(len(ids) for ids in attempts_by_round.values())-
             int(row["cutpoint"]=="post_optimizer"),
             "checkpoint-save trace total differs from saved candidates")
        need(row["accepted_updates"]>=2,"no actual continuation")
        resources=row["resources"]
        need(resources["sample_count"]==measured_resources["sample_count"] and
             resources["sample_interval_s"]==0.02 and
             resources["sampled_tree_peak_rss_bytes"]==measured_resources["sampled_tree_peak_rss_bytes"],
             "resource sample count, interval, or peak RSS differs from retained samples")
        need(math.isclose(resources["sampled_cpu_core_percent"],
                           measured_resources["sampled_cpu_core_percent"],rel_tol=0.01,abs_tol=0.01),
             "resource CPU summary differs from retained samples")
        previous_pid,previous_step="policy-0",0
        for index,commit in enumerate(commits):
            round_dir=case/"rounds"/f"r{index:04d}"
            train,heldout=read(round_dir/"training_tasks.json"),read(round_dir/"evaluation_tasks.json")
            need(len(train)==args["tasks"] and len(heldout)==args["eval_tasks"],"task budget differs")
            need(train==expected_tasks(row["seed"],index,args["tasks"]),
                 "retained training task pool provenance differs")
            need(heldout==expected_tasks(row["seed"],index,args["eval_tasks"],True),
                 "independent evaluation cohort provenance differs")
            train_by_id={task["id"]:task for task in train}
            need(len(train_by_id)==len(train) and len({task["prompt"] for task in train})==len(train),
                 "training task pool contains duplicate identities")
            need(not ({r["prompt"] for r in train}&{r["prompt"] for r in heldout}),"optimization/evaluation overlap")
            raw,old_sha=checkpoint(case,previous_pid,row["mode"])
            state=torch.load(io.BytesIO(raw),map_location="cpu",weights_only=True) if check_tensors or model_dir else None
            if state:
                need(state["step"]==previous_step,"incumbent Adam step differs")
                incumbent_model=fingerprint(state["model"])
            rollout_files={}
            for file in round_dir.glob("rollouts-attempt*.jsonl"):
                suffix=file.stem.removeprefix("rollouts-attempt")
                need(suffix.isdigit() and file.name==f"rollouts-attempt{int(suffix)}.jsonl",
                     "invalid rollout attempt identity")
                attempt_id=int(suffix)
                need(attempt_id not in rollout_files,"duplicate rollout attempt identity")
                rollout_files[attempt_id]=file
            need(set(rollout_files)==attempts_by_round.get(index,set()),
                 "rollout evidence does not cover every optimizer attempt")
            rollouts_by_attempt={}
            for attempt_id,file in rollout_files.items():
                rollouts=[json.loads(line) for line in file.read_text().splitlines()]
                need(len(rollouts)==args["tasks"],"incomplete rollout attempt")
                rollouts_by_step={}
                for rollout in rollouts:
                    task=rollout["task"]
                    declared=train_by_id.get(task["id"])
                    need(declared is not None and
                         all(task.get(field)==declared.get(field) for field in ("id","family","prompt")),
                         "rollout task does not match the retained training task pool")
                    created_step=rollout["created_step"]
                    need(type(created_step) is int and created_step not in rollouts_by_step,
                         "rollout attempt has duplicate or invalid step identity")
                    rollouts_by_step[created_step]=rollout
                    worker=rollout["metadata"]["worker"]
                    generation=rollout["metadata"]["rvl_generation"]
                    generation_metadata=generation["metadata"]
                    token_logprobs=generation_metadata["response_token_logprobs"]
                    need(rollout["logprob"]==worker["logprob"]==generation["logprob"] and
                         token_logprobs and math.isclose(generation["logprob"],math.fsum(token_logprobs),
                                                        rel_tol=0.0,abs_tol=1e-7),
                         "rollout logprob differs from worker/RVL evidence")
                    need(generation["prompt_id"]==task["id"] and generation["prompt"]==task["prompt"] and
                         generation["response"]==rollout["output"] and
                         generation_metadata["prompt_token_ids"]==worker["prompt_ids"] and
                         generation_metadata["response_token_ids"]==[worker["token"]],
                         "rollout tokens differ from worker/RVL generation evidence")
                    semantic_generation=dict(generation)
                    semantic_generation.pop("latency_s",None)
                    need(rollout["metadata"].get("rvl_generation_behavior_sha256")==json_sha(semantic_generation) and
                         rollout["metadata"].get("rvl_group_behavior_sha256")==json_sha([generation]),
                         "rollout behavior digest differs from retained RVL generation")
                    need(worker["loaded_checkpoint_sha256"]==old_sha and worker["loaded_step"]==previous_step,"stale worker served a new policy label")
                    need(rollout["policy_id"]==previous_pid and rollout["policy_version"]==previous_step,"policy provenance differs")
                    if state:
                        need(worker["loaded_model_sha256"]==incumbent_model,"worker tensor acknowledgment differs")
                rollouts_by_attempt[attempt_id]=rollouts_by_step
                worker_rows=[rollout["metadata"]["worker"] for rollout in rollouts]
                rollout_spans=trace_stages.get(("rollout",index,attempt_id),[])
                projected_workers=[{key:span.get(key) for key in worker_rows[0]}
                                   for span in rollout_spans]
                def canonical(value):
                    return json.dumps(value,sort_keys=True,separators=(",",":"))
                need(sorted(map(canonical,projected_workers))==sorted(map(canonical,worker_rows)),
                     "rollout latency spans differ from retained worker results")
                for span in rollout_spans:
                    service_start,service_end,service_s=(span.get(key) for key in
                                                         ("service_start","service_end","service_s"))
                    need(type(service_start) in (int,float) and math.isfinite(service_start) and
                         type(service_end) in (int,float) and math.isfinite(service_end) and
                         type(service_s) in (int,float) and math.isfinite(service_s) and service_s>=0 and
                         span["start"]<=service_start<=service_end<=span["end"],
                         "rollout service interval is outside its trace span")
                expected_policy_lag=max(0,previous_step-rollouts[0]["policy_version"])
                need(all(type(span.get("policy_lag")) is int and
                         span["policy_lag"]==expected_policy_lag for span in rollout_spans),
                     "rollout trace policy lag differs from rollout provenance")
                verified=trace_stages.get(("rollout_verified",index,attempt_id),[])
                need(sorted(span.get("loaded_checkpoint_sha256") for span in verified)==
                     sorted(worker["loaded_checkpoint_sha256"] for worker in worker_rows),
                     "verified rollout spans differ from worker checkpoint acknowledgments")
                verification_steps=trace_stages.get(("verification",index,attempt_id),[])
                need(sorted(span.get("step") for span in verification_steps)==
                     sorted(rollout["created_step"] for rollout in rollouts),
                     "verification spans differ from rollout attempt identities")
            if index==0:
                canary_rollout=rollouts_by_attempt.get(0,{}).get(canary_attempt["created_step"])
                verification=canary.get("verification",{})
                reward=float(canary_attempt.get("output","").strip()=="A")
                need(canary_rollout==canary_attempt and verification.get("trusted") is True and
                     verification.get("score")==reward and verification.get("passed")==bool(reward) and
                     verification.get("verifier_name")=="return-A-fixture" and
                     verification.get("verifier_version")==1 and
                     verification.get("metadata",{}).get("rvl_ready_reward_snapshot")==reward,
                     "stale replay canary differs from its authenticated round-zero rollout")
            expected_stale_rejections=int(canary_attempt["policy_version"]<previous_step)
            engine_files={}
            screening_counts_by_attempt={}
            for path in round_dir.glob("engine-attempt*.jsonl"):
                suffix=path.stem.removeprefix("engine-attempt")
                need(suffix.isdigit() and path.name==f"engine-attempt{int(suffix)}.jsonl",
                     "invalid engine attempt identity")
                attempt_id=int(suffix)
                need(attempt_id not in engine_files,"duplicate engine attempt identity")
                engine_files[attempt_id]=path
                events=[json.loads(line) for line in path.read_text().splitlines()]
                screened=[event for event in events if event.get("event")=="replay_experiences_screened"]
                need(len(screened)==expected_stale_rejections and
                     all(type(event.get("count")) is int and event["count"]==1 and
                         type(event.get("ts")) in (int,float) and math.isfinite(event["ts"])
                         for event in screened),
                     "stale replay screening events do not prove canary rejection")
                screening_counts_by_attempt[attempt_id]=sum(event["count"] for event in screened)
            need(set(engine_files)==attempts_by_round.get(index,set()),
                 "engine events do not cover every optimizer attempt")
            candidate=commit["candidate_id"]
            candidate_raw,candidate_sha=checkpoint(case,candidate,row["mode"])
            update=commit["update"]
            need(update["batch_size"]==args["tasks"] and update["parameters_before_sha256"]!=update["parameters_after_sha256"],"mock/no-op/wrong-budget update")
            batch_files={}
            for path in round_dir.glob("batch-attempt*.json"):
                suffix=path.stem.removeprefix("batch-attempt")
                need(suffix.isdigit() and path.name==f"batch-attempt{int(suffix)}.json",
                     "invalid batch attempt identity")
                attempt_id=int(suffix)
                need(attempt_id not in batch_files,"duplicate batch attempt identity")
                batch_files[attempt_id]=path
            need(set(batch_files)==attempts_by_round.get(index,set()),
                 "batch evidence does not cover every optimizer attempt")
            for attempt_id,path in batch_files.items():
                batch=read(path)
                need(batch["incumbent_id"]==previous_pid and batch["checkpoint_sha256"]==old_sha and batch["step"]==previous_step,"training batch incumbent differs")
                need(len(batch["members"])==args["tasks"] and len({m["created_step"] for m in batch["members"]})==args["tasks"],"duplicate or incomplete optimizer batch")
                need(all(m["policy_id"]==previous_pid and m["policy_version"]==previous_step and m["loaded_checkpoint_sha256"]==old_sha for m in batch["members"]),"stale evidence reached optimizer")
                rollout_by_step=rollouts_by_attempt[attempt_id]
                need({member["created_step"] for member in batch["members"]}==set(rollout_by_step),
                     "optimizer batch members differ from the recorded rollout attempt")
                for member in batch["members"]:
                    rollout=rollout_by_step[member["created_step"]]
                    worker=rollout["metadata"]["worker"]
                    need(member["policy_id"]==rollout["policy_id"] and
                         member["policy_version"]==rollout["policy_version"] and
                         member["loaded_checkpoint_sha256"]==worker["loaded_checkpoint_sha256"] and
                         member["logprob"]==rollout["logprob"] and
                         member["reward"]==float(rollout["output"].strip()=="A"),
                         "optimizer batch signal differs from its rollout and verification")
                if state:
                    need(batch["model_sha256"]==incumbent_model,"training batch tensor hash differs")
            if state:
                candidate_state=torch.load(io.BytesIO(candidate_raw),map_location="cpu",weights_only=True)
                need(fingerprint(candidate_state)==commit["candidate_state_sha256"],"candidate model/Adam/RNG fingerprint differs")
                need(candidate_state["step"]==previous_step+1,"candidate step not exactly next")
                need(fingerprint(candidate_state["model"])==update["parameters_after_sha256"] and incumbent_model==update["parameters_before_sha256"],"update tensors do not match retained checkpoint")
                need(fingerprint(candidate_state["optimizer"])==update["optimizer_state_sha256"],"Adam moments differ")
                tensor_states+=1
            expected_active=candidate if commit["accepted"] else previous_pid
            expected_step=previous_step+int(commit["accepted"])
            need(commit["active_policy"]==expected_active and commit["accepted_updates"]==expected_step,"double-applied or missing committed update")
            if state:
                expected_state=candidate_state if commit["accepted"] else state
                need(fingerprint(expected_state)==commit["active_state_sha256"],"active state differs after promotion/reject")
            need(commit["stale_replay_rejected"]==expected_stale_rejections,
                 "committed stale replay count differs from stale-canary eligibility")
            expected_policies={previous_pid,candidate}
            records,by_attempt=[],{}
            for path in sorted(round_dir.glob('evaluation-*.json')):
                record=read(path)
                policy_id=record["policy_id"]
                _,separator,attempt=path.stem.rpartition('-attempt')
                need(separator and attempt.isdigit() and policy_id in expected_policies and
                     path.name==f"evaluation-{policy_id}-attempt{int(attempt)}.json" and
                     record["report"]["policy_id"]==policy_id,"independent evaluation artifact identity differs")
                need(int(attempt) in attempts_by_round.get(index,set()),
                     "independent evaluation does not correspond to a physical optimizer attempt")
                policy_records=by_attempt.setdefault(int(attempt),{})
                need(policy_id not in policy_records,"duplicate evaluation policy in learner attempt")
                need(record["scores"]==record["report"].get("metadata",{}).get("per_task_scores"),
                     "evaluation scores differ from report metadata")
                if row["mode"]=="guarded":
                    need(record["report"]["metadata"].get("vare_evaluation_set_sha256") ==
                         sha(round_dir/"evaluation_tasks.json"),
                         "guarded evaluation set digest differs from retained cohort")
                policy_records[policy_id]=record
                records.append(record)
            complete_attempts={attempt for attempt,policies in by_attempt.items() if set(policies)==expected_policies}
            expected_evaluation_attempts=set(attempts_by_round[index])
            if row["cutpoint"] in ("post_optimizer","candidate_saved") and index==kill_round:
                expected_evaluation_attempts.discard(0)
            need(set(by_attempt)==complete_attempts==expected_evaluation_attempts,
                 "independent evaluation pairs do not cover every optimizer attempt that reached the gate")
            for attempt_id in complete_attempts:
                evaluation_spans=trace_stages.get(("evaluation",index,attempt_id),[])
                need(sorted(span.get("policy_id") for span in evaluation_spans)==sorted(expected_policies),
                     "evaluation trace identities differ from retained checkpoint evaluations")
            result=commit.get("result")
            if result is not None:
                need(not commit.get("recovered_publication") and result["round_index"]==index and
                     result["incumbent_id"]==previous_pid and result["candidate_id"]==candidate and
                     result["decision"]["accepted"] is commit["accepted"],
                     "committed round result identity differs")
                completed_attempts=[]
                for event_file in round_dir.glob("engine-attempt*.jsonl"):
                    suffix=event_file.stem.removeprefix("engine-attempt")
                    need(suffix.isdigit(),"invalid engine attempt identity")
                    attempt_id=int(suffix)
                    for line in event_file.read_text().splitlines():
                        event=json.loads(line)
                        if event.get("event")=="round_complete" and event.get("round_index")==index:
                            completed_attempts.append(attempt_id)
                need(len(completed_attempts)==1 and
                     completed_attempts[0] in attempts_by_round.get(index,set()),
                     "completed round has no unique optimizer attempt")
                committed_attempt=completed_attempts[0]
                need(commit["stale_replay_rejected"]==screening_counts_by_attempt[committed_attempt],
                     "committed stale replay count differs from its screening event")
                need(committed_attempt in complete_attempts and
                     by_attempt[committed_attempt][previous_pid]["report"]==result["incumbent_eval"] and
                     by_attempt[committed_attempt][candidate]["report"]==result["candidate_eval"],
                     "evaluation pair differs from committed round result")
            else:
                need(commit.get("recovered_publication") is True and row["mode"]=="guarded",
                     "committed round result is missing")
                need(row["cutpoint"] in ("pointer_published","journal_committed") and index==kill_round,
                     "recovered publication is outside the configured SIGKILL publication boundary")
                with sqlite3.connect((case/"promotion.sqlite").resolve().as_uri()+"?mode=ro",uri=True) as db:
                    transactions=db.execute(
                        "SELECT incumbent_id,incumbent_sha256,candidate_id,candidate_sha256,evaluation_sha256,state "
                        "FROM transactions WHERE candidate_id=? AND candidate_sha256=?",
                        (candidate,commit["active_checkpoint_sha256"]),
                    ).fetchall()
                need(len(transactions)==1,"recovered promotion transaction is missing or ambiguous")
                incumbent_id,incumbent_sha,candidate_id,transaction_candidate_sha,evaluation_digest=transactions[0][:5]
                transaction_state=transactions[0][5]
                need(incumbent_id==previous_pid and incumbent_sha==old_sha and candidate_id==candidate and
                     transaction_candidate_sha==candidate_sha and transaction_state=="committed",
                     "recovered promotion journal binding differs")
                committed_attempts=[]
                for attempt_id in complete_attempts:
                    evaluation_metadata={policy:by_attempt[attempt_id][policy]["report"]["metadata"]
                                         for policy in (previous_pid,candidate)}
                    if json_sha({"incumbent":evaluation_metadata[previous_pid],
                                 "candidate":evaluation_metadata[candidate]})==evaluation_digest:
                        committed_attempts.append(attempt_id)
                need(len(committed_attempts)==1,
                     "recovered evaluation pair differs from promotion evaluation digest")
                committed_attempt=committed_attempts[0]
                need(commit["stale_replay_rejected"]==screening_counts_by_attempt[committed_attempt],
                     "recovered stale replay count differs from its screening event")
                incumbent_report=by_attempt[committed_attempt][previous_pid]["report"]
                candidate_report=by_attempt[committed_attempt][candidate]["report"]
                expected_accepted=promotion_expected(incumbent_report,candidate_report)
                need(commit["accepted"] is expected_accepted,
                     "recovered promotion decision differs from its independent evaluation pair")
            if result is not None:
                incumbent_report=by_attempt[committed_attempt][previous_pid]["report"]
                candidate_report=by_attempt[committed_attempt][candidate]["report"]
                expected_accepted=promotion_expected(incumbent_report,candidate_report)
                decision=result["decision"]
                need(commit["accepted"] is expected_accepted and
                     decision["accepted"] is expected_accepted and
                     result["promoted_id"]==(candidate if expected_accepted else previous_pid) and
                     decision["primary_gain"]==candidate_report["primary"]-incumbent_report["primary"],
                     "promotion decision differs from its independent evaluation pair")
            evaluation_pairs+=len(complete_attempts)
            for record in records:
                need(set(record["scores"])=={task["id"] for task in heldout} and
                     record["report"]["n"]==args["eval_tasks"],"independent evaluation task coverage differs")
                need(len(record["scores"])==args["eval_tasks"] and statistics.fmean(record["scores"].values())==record["report"]["primary"],"independent evaluation arithmetic differs")
                evaluated_raw,evaluated_sha=checkpoint(case,record["policy_id"],row["mode"])
                need(record["checkpoint_sha256"]==evaluated_sha,"evaluation checkpoint differs")
                evaluation_reports+=1
                if model_dir:
                    evaluated_state=torch.load(io.BytesIO(evaluated_raw),map_location="cpu",weights_only=True)
                    model.load_state_dict(evaluated_state["model"])
                    with torch.inference_mode():
                        for task in heldout:
                            score=float(model(torch.tensor([tokenizer.encode(task["prompt"])] )).logits[0,-1,choices].softmax(-1)[0])
                            need(abs(score-record["scores"][task["id"]])<1e-7,"independent checkpoint score does not replay")
                            replayed_scores+=1
            previous_pid,previous_step=expected_active,expected_step
        if row["mode"]=="guarded":
            need(row["final_journal"]["pending_tx_id"] is None and row["final_journal"]["round_owner_id"] is None,"journal unresolved")
            with sqlite3.connect((case/"promotion.sqlite").resolve().as_uri()+"?mode=ro",uri=True) as db:
                active=db.execute("SELECT policy_id,checkpoint_sha256 FROM active").fetchone()
                duplicate=db.execute("SELECT candidate_id FROM transactions WHERE state='committed' GROUP BY candidate_id HAVING COUNT(*)>1").fetchall()
                pending=db.execute("SELECT COUNT(*) FROM transactions WHERE state='prepared'").fetchone()[0]
                owners=db.execute("SELECT COUNT(*) FROM runtime_round").fetchone()[0]
            need(not duplicate and pending==owners==0,"duplicate commit or unresolved durable owner")
            need(active==(commits[-1]["active_policy"],commits[-1]["active_checkpoint_sha256"]),"actual journal pointer differs")
        if row["cutpoint"]!="none":
            killed,progress=read(case/"kill_record.json"),read(case/"post_restart_update.json")
            marker=read(case/"cutpoint.json")
            marker_time=marker.get("time")
            killed_at=killed.get("killed_at")
            need(marker.get("cutpoint")==row["cutpoint"]==killed.get("cutpoint") and
                 marker.get("round_index")==protocol["kill_round"],
                 "cutpoint marker differs from the configured recovery case")
            need(type(marker.get("pid")) is int and marker["pid"]==killed.get("learner_pid") and
                 type(marker.get("pgid")) is int and marker["pgid"]==marker["pid"],
                 "cutpoint marker differs from the killed learner process")
            need(type(marker_time) in (int,float) and math.isfinite(marker_time) and
                 type(killed_at) in (int,float) and math.isfinite(killed_at) and killed_at>=marker_time,
                 "cutpoint marker was not observed before the learner was killed")
            need(killed["returncode"]==-9 and killed["old_descendants"] and killed["live_old_descendants"]==0,"learner/descendant SIGKILL not established")
            need(row.get("live_old_descendants")==killed["live_old_descendants"],
                 "progress live descendant count differs from kill record")
            killed_attempts=[attempt for attempt in attempts if attempt["round_index"]==protocol["kill_round"] and attempt["learner_attempt"]==0]
            need(len(killed_attempts)==1 and
                 marker.get("volatile_model_sha256")==killed_attempts[0]["candidate_model_sha256"],
                 "cutpoint marker does not identify the killed optimizer state")
            if row["cutpoint"] in prepublication_cutpoints:
                expected_marker_policy=commits[protocol["kill_round"]-1]["active_policy"]
            else:
                expected_marker_policy=commits[protocol["kill_round"]]["active_policy"]
            need(marker.get("active_policy")==expected_marker_policy,
                 "cutpoint marker active policy differs from the recovery boundary")
            expected_round=protocol["kill_round"] if row["cutpoint"] in ("post_optimizer","candidate_saved","journal_prepared") else protocol["kill_round"]+1
            need(progress["completed_round"]==expected_round,"recovery did not execute the next logical optimizer round")
            resumed_spans=trace_stages.get(("round",expected_round,1),[])
            need(len(resumed_spans)==1,"recovery endpoint lacks one resumed-round trace")
            resumed_end=resumed_spans[0]["end"]
            if source_inventory in EXACT_RECOVERY_ENDPOINT_INVENTORIES:
                need(progress.get("time")==resumed_end,
                     "recovery endpoint differs from resumed-round trace")
            else:
                next_spans=trace_stages.get(("round",expected_round+1,1),[])
                need(len(next_spans)==1 and resumed_end<=progress.get("time")<=next_spans[0]["start"],
                     "legacy recovery endpoint falls outside the resumed-round trace interval")
            need(row["recovery_to_next_update_s"]==progress["time"]-killed["killed_at"]>0,
                 "recovery clock differs")
    for row in rows:
        records=round_records[row["directory"]]
        if row["cutpoint"]=="none":
            other="minimal" if row["mode"]=="guarded" else "guarded"
            reference=references[(row["workers"],row["seed"],other)]
        else:
            reference=references[(2,row["seed"],"guarded")]
        for observed,expected in zip(records,reference,strict=True):
            need(observed["active_state_sha256"]==expected["active_state_sha256"] and observed["candidate_state_sha256"]==expected["candidate_state_sha256"] and observed["accepted"]==expected["accepted"],"continuity reference mismatch")
            def update(r):
                return {k:v for k,v in r["update"].items() if k!="optimizer_steps"}
            need(update(observed)==update(expected),"next numerical optimizer update differs")
        if row["cutpoint"]!="none":
            need(row.get("exact_reference_match") is True,
                 "recovery exact-reference flag differs from retained round comparisons")
    committed_rounds=sum(len(round_records[row["directory"]]) for row in rows)
    need(summary["cases"]==len(rows) and summary["completed_rounds"]==committed_rounds,
         "summary budget differs from committed records")
    need(summary["physical_optimizer_attempts"]==sum(r["physical_optimizer_attempts"] for r in rows),"summary attempts differ")
    expected_normal_groups={(r["workers"],r["mode"]) for r in rows if r["cutpoint"]=="none"}
    normal_groups=summary.get("normal")
    need(isinstance(normal_groups,list) and
         all(isinstance(group,dict) and type(group.get("workers")) is int and
             group.get("mode") in ("minimal","guarded")
             for group in normal_groups),
         "normal summary groups are invalid")
    actual_normal_groups=[(group["workers"],group["mode"]) for group in normal_groups]
    need(len(actual_normal_groups)==len(expected_normal_groups) and
         set(actual_normal_groups)==expected_normal_groups,
         "normal summary groups do not cover every normal case aggregate")
    for group in normal_groups:
        values=[r["tasks_s"] for r in rows if r["cutpoint"]=="none" and r["workers"]==group["workers"] and r["mode"]==group["mode"]]
        need(group["tasks_s_mean"]==statistics.fmean(values) and group["n"]==len(values),"mean differs")
        if len(values)>1:
            need(group["tasks_s_sd"]==statistics.stdev(values),"seed variance differs")
        else:
            need(group["tasks_s_sd"] is None,"single-case seed variance must be null")
    return {"pass":True,"cases":len(rows),"protocol_conformant":locked,"artifacts_hashed":len(sums),
            "candidate_tensor_states_verified":tensor_states,"evaluation_pairs_verified":evaluation_pairs,
            "evaluation_reports_verified":evaluation_reports,"checkpoint_score_count_replayed":replayed_scores,
            "checkpoint_scores_replayed":bool(replayed_scores) and replayed_scores==evaluation_reports*args["eval_tasks"]}


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory",type=Path)
    parser.add_argument("--check-tensors",action="store_true")
    parser.add_argument("--model-dir",type=Path)
    args=parser.parse_args()
    print(json.dumps(audit(args.directory,args.check_tensors,args.model_dir),sort_keys=True))
