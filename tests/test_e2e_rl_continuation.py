"""Actual CPU weight-sync and full learner SIGKILL/restart falsifiers."""
import os
import csv
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import json
import importlib.util

import pytest

from scripts import benchmark_e2e_rl_continuation as bench
from scripts import audit_e2e_rl_continuation as auditor

_derive_spec=importlib.util.spec_from_file_location(
    "continuation_derive_tables",bench.ROOT/"results/e2e-rl-continuation-v2/derive_tables.py")
derive_tables=importlib.util.module_from_spec(_derive_spec)
_derive_spec.loader.exec_module(derive_tables)


def test_v2_task_namespaces_are_round_and_evaluation_disjoint():
    train=bench.fresh_tasks(3901,0,8)
    future=bench.fresh_tasks(3901,1,8)
    heldout=bench.fresh_tasks(3901,0,8,True)
    assert train==bench.fresh_tasks(3901,0,8)
    assert len({t.prompt for t in [*train,*future,*heldout]})==24


def test_auditor_inventory_covers_the_runtime_package():
    runtime_sources={str(path.relative_to(bench.ROOT))
                     for path in (bench.ROOT/"src/vare").rglob("*.py")}
    pinned_runtime_sources={path for path in auditor.RUNTIME_PACKAGE_SOURCES
                            if path.startswith("src/vare/")}
    assert runtime_sources==pinned_runtime_sources


def test_restart_refuses_changed_learner_source_before_loading_model(tmp_path):
    config=tmp_path/"config.json"
    bench.atomic_json(config,{"source_sha256":{"scripts/benchmark_e2e_rl_continuation.py":"0"*64}})
    with pytest.raises(RuntimeError,match="source differs"):
        bench.learner(config,1,"none")


@pytest.fixture(scope="module")
def campaign(tmp_path_factory):
    model=os.environ.get("VARE_E2E_MODEL_DIR")
    if not model:
        pytest.skip("actual continuation cases require VARE_E2E_MODEL_DIR")
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    pytest.importorskip("psutil")
    target=tmp_path_factory.mktemp("continuation")/"run"
    process=subprocess.run([sys.executable,str(bench.ROOT/"scripts/benchmark_e2e_rl_continuation.py"),
                            "--output",str(target),"--model-dir",str(Path(model).resolve()),
                            "--workers","2","--seeds","3901","--rounds","6","--tasks","4","--eval-tasks","4"],
                           capture_output=True,text=True,timeout=240,
                           env={**os.environ,"OMP_NUM_THREADS":"1","MKL_NUM_THREADS":"1","TOKENIZERS_PARALLELISM":"false","PYTHONHASHSEED":"0"})
    assert process.returncode==0,process.stdout+process.stderr
    return target,Path(model).resolve()


def test_actual_checkpoint_state_and_scores_replay(campaign):
    root,model=campaign
    result=auditor.audit(root,True,model)
    assert result["cases"]==8 and result["candidate_tensor_states_verified"]==48
    assert result["checkpoint_scores_replayed"]
    assert result["evaluation_pairs_verified"]==49 and result["evaluation_reports_verified"]==98
    assert result["checkpoint_score_count_replayed"]==392
    # Reduced regression runs must never masquerade as the frozen 3-seed study.
    assert not result["protocol_conformant"]
    assert auditor.read(root/"summary.json")["protocol_id"].endswith(".diagnostic")


def test_campaign_snapshots_the_full_runtime_package(campaign,tmp_path):
    root,_=campaign
    manifest=auditor.read(root/"manifest.json")
    runtime_sources={str(path.relative_to(bench.ROOT)) for path in (bench.ROOT/"src/vare").rglob("*.py")}
    assert len(runtime_sources)==67
    assert runtime_sources <= set(manifest["sources"])
    assert runtime_sources <= set(manifest["source_sha256"])
    assert runtime_sources <= auditor.EXPECTED_SOURCES
    changed=tmp_path/"changed-runtime-dependency"
    shutil.copytree(root,changed)
    source="src/vare/curriculum.py"
    path=changed/"sources"/source
    path.write_bytes(path.read_bytes()+b"\n# changed after campaign\n")
    manifest=auditor.read(changed/"manifest.json")
    digest=auditor.sha(path)
    manifest["sources"][source]=manifest["source_sha256"][source]=digest
    bench.atomic_json(changed/"manifest.json",manifest)
    for row in auditor.read(changed/"progress.json"):
        config_path=changed/row["directory"]/"config.json"
        config=auditor.read(config_path)
        config["source_sha256"][source]=digest
        bench.atomic_json(config_path,config)
    reseal_records(changed)
    with pytest.raises(ValueError,match="source snapshot differs from recorded base commit"):
        auditor.audit(changed)


@pytest.mark.parametrize("cut",["post_optimizer","candidate_saved","journal_prepared",
                                 "pointer_published","journal_committed","round_committed"])
def test_actual_learner_sigkill_resumes_without_double_committed_update(campaign,cut):
    root,_=campaign
    case=root/f"kill-{cut}-s3901-guarded"
    proof=auditor.read(case/"kill_record.json")
    assert proof["returncode"]==-9 and proof["old_descendants"] and proof["live_old_descendants"]==0
    summary=auditor.read(case/"summary.json")
    assert summary["accepted_updates"]>=2 and summary["completed_rounds"]==6
    assert summary["final_state_sha256"]==auditor.read(root/"normal-w2-s3901-guarded/summary.json")["final_state_sha256"]
    assert summary["physical_optimizer_attempts"]==6+int(cut in ("post_optimizer","candidate_saved","journal_prepared"))
    cursor=auditor.read(case/"post_restart_update.json")
    assert cursor["completed_round"]==(4 if cut in ("post_optimizer","candidate_saved","journal_prepared") else 5)


def test_real_worker_stale_weights_fail_then_updated_weights_pass(campaign):
    import torch
    import io
    root,model=campaign
    case=root/"normal-w2-s3901-guarded"
    store=bench.TrustedCheckpointStore(case/"checkpoints",read_only=True)
    pool,_=bench.base.pool_start(model,1)
    def request(pid,skip=False):
        raw,digest=store.read_verified(pid)
        state=torch.load(io.BytesIO(raw),weights_only=True,map_location="cpu")
        model_sha=bench.base.tensor_digest(state["model"])
        data=pool.submit(bench.generate_synced,"v2 regression: return A. Answer:",1,
                         str(store.objects/digest),digest,model_sha,state["step"],skip).result(timeout=120)
        return data,digest,model_sha,state["step"]
    try:
        request("policy-0")
        stale,sha,model_sha,step=request("policy-0001",True)
        with pytest.raises(RuntimeError,match="wrong checkpoint"):
            bench.check_worker_identity(stale,sha,model_sha,step)
        current,sha,model_sha,step=request("policy-0001")
        bench.check_worker_identity(current,sha,model_sha,step)
        assert current["loaded_model_sha256"]!=stale["loaded_model_sha256"]
    finally:
        pool.shutdown(wait=True,cancel_futures=True)


def test_rehashed_false_attempt_count_is_rejected(campaign,tmp_path):
    root,_=campaign
    changed=tmp_path/"tampered"
    shutil.copytree(root,changed)
    progress=auditor.read(changed/"progress.json")
    progress[0]["physical_optimizer_attempts"]+=1
    bench.atomic_json(changed/"progress.json",progress)
    hashes=auditor.read(changed/"SHA256SUMS.json")
    hashes["progress.json"]=auditor.sha(changed/"progress.json")
    bench.atomic_json(changed/"SHA256SUMS.json",hashes)
    with pytest.raises(ValueError,match="attempt accounting"):
        auditor.audit(changed)


@pytest.fixture(scope="module")
def audit_records(request):
    retained=os.environ.get("VARE_CONTINUATION_AUDIT_RAW_DIR")
    # Review falsification may reuse raw records without starting a learner.
    return Path(retained).resolve() if retained else request.getfixturevalue("campaign")[0]


def reseal_records(root):
    bench.atomic_json(root/"SHA256SUMS.json",{str(p.relative_to(root)):auditor.sha(p)
                      for p in root.rglob('*') if p.is_file() and p.name!="SHA256SUMS.json"})


@pytest.mark.parametrize("mode",["minimal","guarded"])
@pytest.mark.parametrize("corruption",["incumbent","candidate","both","split_attempt","duplicate_policy"])
def test_rehashed_incomplete_independent_evaluation_is_rejected(audit_records,tmp_path,mode,corruption):
    changed=tmp_path/"incomplete"
    shutil.copytree(audit_records,changed)
    row=next(r for r in auditor.read(changed/"progress.json") if r["mode"]==mode and r["cutpoint"]=="none")
    directory=changed/row["directory"]/"rounds/r0000"
    candidate=auditor.read(directory/"committed.json")["candidate_id"]
    incumbent_file=directory/"evaluation-policy-0-attempt0.json"
    candidate_file=directory/f"evaluation-{candidate}-attempt0.json"
    if corruption in ("incumbent","both"):
        incumbent_file.unlink()
    if corruption in ("candidate","both"):
        candidate_file.unlink()
    if corruption=="split_attempt":
        candidate_file.rename(directory/f"evaluation-{candidate}-attempt1.json")
    if corruption=="duplicate_policy":
        candidate_file.write_bytes(incumbent_file.read_bytes())
    reseal_records(changed)
    with pytest.raises(ValueError,match="independent evaluation"):
        auditor.audit(changed)


def test_rehashed_absent_independent_evaluation_is_rejected(audit_records,tmp_path):
    changed=tmp_path/"absent"
    shutil.copytree(audit_records,changed)
    for path in changed.glob('*/rounds/r*/evaluation-*.json'):
        path.unlink()
    reseal_records(changed)
    with pytest.raises(ValueError,match="independent evaluation"):
        auditor.audit(changed)


def test_rehashed_evaluation_attempt_must_exist_in_optimizer_history(audit_records,tmp_path):
    changed=tmp_path/"unlaunched-attempt"
    shutil.copytree(audit_records,changed)
    row=next(r for r in auditor.read(changed/"progress.json")
             if r["mode"]=="minimal" and r["cutpoint"]=="none")
    round_dir=changed/row["directory"]/"rounds/r0000"
    for path in round_dir.glob("evaluation-*-attempt0.json"):
        path.rename(path.with_name(path.stem.rsplit("-attempt",1)[0]+"-attempt99.json"))
    reseal_records(changed)
    with pytest.raises(ValueError,match="optimizer attempt"):
        auditor.audit(changed)


def test_rehashed_missing_rollout_attempt_is_rejected(audit_records,tmp_path):
    changed=tmp_path/"missing-rollout-evidence"
    shutil.copytree(audit_records,changed)
    row=next(r for r in auditor.read(changed/"progress.json")
             if r["mode"]=="minimal" and r["cutpoint"]=="none")
    for path in (changed/row["directory"]/"rounds").glob("r*/rollouts-attempt*.jsonl"):
        path.unlink()
    reseal_records(changed)
    with pytest.raises(ValueError,match="rollout evidence"):
        auditor.audit(changed)


def test_rehashed_progress_seed_must_match_case_configuration(audit_records,tmp_path):
    changed=tmp_path/"forged-progress-seed"
    shutil.copytree(audit_records,changed)
    progress=auditor.read(changed/"progress.json")
    original_seed=progress[0]["seed"]
    for row in progress:
        if row["seed"]==original_seed:
            row["seed"]=original_seed+9999
    bench.atomic_json(changed/"progress.json",progress)
    reseal_records(changed)
    with pytest.raises(ValueError,match="retained configuration"):
        auditor.audit(changed)


def test_rehashed_missing_batch_attempt_is_rejected(audit_records,tmp_path):
    changed=tmp_path/"missing-batch-evidence"
    shutil.copytree(audit_records,changed)
    row=next(r for r in auditor.read(changed/"progress.json")
             if r["mode"]=="minimal" and r["cutpoint"]=="none")
    for path in (changed/row["directory"]/"rounds/r0000").glob("batch-attempt*.json"):
        path.unlink()
    reseal_records(changed)
    with pytest.raises(ValueError,match="batch evidence"):
        auditor.audit(changed)


def test_rehashed_batch_signal_must_match_its_rollout(audit_records,tmp_path):
    changed=tmp_path/"misbound-batch-signal"
    shutil.copytree(audit_records,changed)
    row=next(r for r in auditor.read(changed/"progress.json")
             if r["mode"]=="minimal" and r["cutpoint"]=="none")
    path=changed/row["directory"]/"rounds/r0000/batch-attempt0.json"
    batch=auditor.read(path)
    batch["members"][0]["reward"]=1.0-batch["members"][0]["reward"]
    bench.atomic_json(path,batch)
    reseal_records(changed)
    with pytest.raises(ValueError,match="batch signal differs"):
        auditor.audit(changed)


def test_rehashed_round_task_pool_must_match_rollouts(audit_records,tmp_path):
    changed=tmp_path/"swapped-training-task-pool"
    shutil.copytree(audit_records,changed)
    row=next(r for r in auditor.read(changed/"progress.json")
             if r["mode"]=="minimal" and r["cutpoint"]=="none")
    rounds=changed/row["directory"]/"rounds"
    first,second=rounds/"r0000/training_tasks.json",rounds/"r0001/training_tasks.json"
    first_tasks,second_tasks=auditor.read(first),auditor.read(second)
    bench.atomic_json(first,second_tasks)
    bench.atomic_json(second,first_tasks)
    reseal_records(changed)
    with pytest.raises(ValueError,match="retained training task pool"):
        auditor.audit(changed)


def test_rehashed_completed_round_total_must_match_commits(audit_records,tmp_path):
    changed=tmp_path/"false-completed-round-total"
    shutil.copytree(audit_records,changed)
    progress=auditor.read(changed/"progress.json")
    for row in progress:
        row["completed_rounds"]-=1
    bench.atomic_json(changed/"progress.json",progress)
    summary=auditor.read(changed/"summary.json")
    summary["completed_rounds"]=sum(row["completed_rounds"] for row in progress)
    bench.atomic_json(changed/"summary.json",summary)
    reseal_records(changed)
    with pytest.raises(ValueError,match="completed round count"):
        auditor.audit(changed)


def test_rehashed_evaluation_pair_must_match_committed_round_result(audit_records,tmp_path):
    changed=tmp_path/"misbound-result"
    shutil.copytree(audit_records,changed)
    row=next(r for r in auditor.read(changed/"progress.json")
             if r["mode"]=="minimal" and r["cutpoint"]=="none")
    round_dir=changed/row["directory"]/"rounds/r0000"
    path=round_dir/"evaluation-policy-0-attempt0.json"
    record=auditor.read(path)
    record["report"]["metadata"]["uncommitted_claim"]=True
    path.write_text(json.dumps(record,sort_keys=True)+"\n")
    reseal_records(changed)
    with pytest.raises(ValueError,match="committed round result"):
        auditor.audit(changed)


def test_rehashed_recovered_evaluation_pair_must_match_promotion_journal(audit_records,tmp_path):
    changed=tmp_path/"misbound-recovery"
    shutil.copytree(audit_records,changed)
    row=next(r for r in auditor.read(changed/"progress.json")
             if r["mode"]=="guarded" and r["cutpoint"]=="journal_committed")
    case=changed/row["directory"]
    commit=next(auditor.read(path) for path in sorted((case/"rounds").glob("r*/committed.json"))
                if auditor.read(path).get("recovered_publication"))
    round_dir=case/"rounds"/f"r{commit['completed_round']:04d}"
    path=next(round_dir.glob(f"evaluation-{commit['candidate_id']}-attempt*.json"))
    record=auditor.read(path)
    record["report"]["metadata"]["uncommitted_claim"]=True
    path.write_text(json.dumps(record,sort_keys=True)+"\n")
    reseal_records(changed)
    with pytest.raises(ValueError,match="promotion evaluation digest"):
        auditor.audit(changed)


def test_rehashed_protocol_snapshot_must_match_declared_digest(audit_records,tmp_path):
    changed=tmp_path/"protocol-digest-conflict"
    shutil.copytree(audit_records,changed)
    protocol_path=changed/"sources/benchmarks/e2e_rl_continuation_v2/protocol.json"
    protocol=auditor.read(protocol_path)
    protocol["objective"]+=" tampered after run"
    bench.atomic_json(protocol_path,protocol)
    manifest=auditor.read(changed/"manifest.json")
    source_name="benchmarks/e2e_rl_continuation_v2/protocol.json"
    manifest["sources"][source_name]=manifest["source_sha256"][source_name]=auditor.sha(protocol_path)
    bench.atomic_json(changed/"manifest.json",manifest)
    reseal_records(changed)
    with pytest.raises(ValueError,match="declared protocol snapshot digest"):
        auditor.audit(changed)


def test_rehashed_missing_pinned_source_is_rejected(audit_records,tmp_path):
    changed=tmp_path/"missing-pinned-source"
    shutil.copytree(audit_records,changed)
    source="src/vare/engine.py"
    manifest=auditor.read(changed/"manifest.json")
    del manifest["sources"][source]
    del manifest["source_sha256"][source]
    bench.atomic_json(changed/"manifest.json",manifest)
    for row in auditor.read(changed/"progress.json"):
        config_path=changed/row["directory"]/"config.json"
        config=auditor.read(config_path)
        del config["source_sha256"][source]
        bench.atomic_json(config_path,config)
    (changed/"sources"/source).unlink()
    reseal_records(changed)
    with pytest.raises(ValueError,match="pinned source inventory differs"):
        auditor.audit(changed)


def test_frozen_source_inventory_requires_exact_commit_and_digests():
    manifest={"base_commit":auditor.PINNED_LOCKED_BASE_COMMIT,
              "sources":dict(auditor.PINNED_LOCKED_SOURCE_SHA256)}
    auditor.validate_source_inventory(manifest,auditor.LEGACY_EXPECTED_SOURCES,True)
    wrong_commit={**manifest,"base_commit":"f"*40}
    with pytest.raises(ValueError,match="independently pinned revision"):
        auditor.validate_source_inventory(wrong_commit,auditor.LEGACY_EXPECTED_SOURCES,True)
    wrong_sources={**manifest,"sources":{**manifest["sources"],"src/vare/engine.py":"0"*64}}
    with pytest.raises(ValueError,match="independently pinned revision"):
        auditor.validate_source_inventory(wrong_sources,auditor.LEGACY_EXPECTED_SOURCES,True)


def test_fresh_conformant_campaign_accepts_its_complete_source_inventory():
    commit=bench.subprocess.check_output(["git","rev-parse","HEAD"],cwd=bench.ROOT,text=True).strip()
    source_hashes=auditor.source_hashes_from_commit(commit,auditor.EXPECTED_SOURCES)
    manifest={"base_commit":commit,"sources":source_hashes}
    auditor.validate_source_inventory(manifest,auditor.EXPECTED_SOURCES,True)


def test_source_hashes_from_commit_use_raw_git_blobs_with_export_subst(tmp_path,monkeypatch):
    import hashlib
    import io
    import tarfile

    repo=tmp_path/"export-subst"
    repo.mkdir()
    subprocess.run(["git","init","-q",str(repo)],check=True)
    subprocess.run(["git","-C",str(repo),"config","user.name","VARE test"],check=True)
    subprocess.run(["git","-C",str(repo),"config","user.email","vare-test@example.invalid"],check=True)
    source=b'commit = "$Format:%H$"\n'
    (repo/".gitattributes").write_text("source.py export-subst\n")
    (repo/"source.py").write_bytes(source)
    subprocess.run(["git","-C",str(repo),"add",".gitattributes","source.py"],check=True)
    subprocess.run(["git","-C",str(repo),"commit","-qm","fixture"],check=True)
    commit=subprocess.check_output(["git","-C",str(repo),"rev-parse","HEAD"],text=True).strip()
    monkeypatch.setattr(auditor,"REPOSITORY_ROOT",repo)

    actual=auditor.source_hashes_from_commit(commit,{"source.py"})["source.py"]
    assert actual==hashlib.sha256(source).hexdigest()
    archived=subprocess.run(["git","-C",str(repo),"archive","--format=tar",commit,"--","source.py"],
                            check=True,capture_output=True)
    with tarfile.open(fileobj=io.BytesIO(archived.stdout),mode="r:") as archive:
        expanded=archive.extractfile("source.py").read()
    assert expanded!=source
    assert actual!=hashlib.sha256(expanded).hexdigest()


def test_new_source_inventory_requires_a_full_base_commit():
    manifest={"base_commit":"not-a-commit",
              "sources":{name:"0"*64 for name in auditor.EXPECTED_SOURCES}}
    with pytest.raises(ValueError,match="base commit is not a full SHA"):
        auditor.validate_source_inventory(manifest,auditor.EXPECTED_SOURCES,True)


def test_new_source_inventory_requires_a_locally_verifiable_commit():
    manifest={"base_commit":"f"*40,
              "sources":{name:"0"*64 for name in auditor.EXPECTED_SOURCES}}
    with pytest.raises(ValueError,match="base commit is unavailable"):
        auditor.validate_source_inventory(manifest,auditor.EXPECTED_SOURCES,True)


def test_missing_final_durable_cursor_is_rejected(audit_records,tmp_path):
    changed=tmp_path/"missing-final-cursor"
    shutil.copytree(audit_records,changed)
    row=next(item for item in auditor.read(changed/"progress.json") if item["cutpoint"]=="none")
    (changed/row["directory"]/"cursor.json").unlink()
    reseal_records(changed)
    with pytest.raises(ValueError,match="final durable cursor is missing"):
        auditor.audit(changed)


@pytest.mark.parametrize("field",["completed_round","active_policy","active_checkpoint_sha256",
                                   "active_state_sha256","accepted_updates"])
def test_final_durable_cursor_must_match_final_commit(audit_records,tmp_path,field):
    changed=tmp_path/f"forged-final-cursor-{field}"
    shutil.copytree(audit_records,changed)
    row=next(item for item in auditor.read(changed/"progress.json") if item["cutpoint"]=="none")
    cursor_path=changed/row["directory"]/"cursor.json"
    cursor=auditor.read(cursor_path)
    cursor[field]=cursor[field]+1 if type(cursor[field]) is int else "forged"
    bench.atomic_json(cursor_path,cursor)
    reseal_records(changed)
    with pytest.raises(ValueError,match="final durable cursor differs"):
        auditor.audit(changed)


def test_recovery_latency_must_match_resumed_round_trace(audit_records,tmp_path):
    changed=tmp_path/"forged-recovery-endpoint"
    shutil.copytree(audit_records,changed)
    rows=auditor.read(changed/"progress.json")
    row=next(item for item in rows if item["cutpoint"]=="post_optimizer")
    case=changed/row["directory"]
    progress_path=case/"post_restart_update.json"
    progress=auditor.read(progress_path)
    progress["time"]+=60.0
    row["recovery_to_next_update_s"]+=60.0
    bench.atomic_json(progress_path,progress)
    bench.atomic_json(changed/"progress.json",rows)
    reseal_records(changed)
    with pytest.raises(ValueError,match="recovery endpoint"):
        auditor.audit(changed)


def test_rehashed_duplicate_optimizer_attempt_identity_is_rejected(audit_records,tmp_path):
    changed=tmp_path/"duplicate-optimizer-attempt"
    shutil.copytree(audit_records,changed)
    row=next(r for r in auditor.read(changed/"progress.json")
             if r["cutpoint"]=="post_optimizer")
    case=changed/row["directory"]
    attempt_path=case/"optimizer_attempts.jsonl"
    attempts=[json.loads(line) for line in attempt_path.read_text().splitlines()]
    interrupted=next(a for a in attempts if a["round_index"]==4 and a["learner_attempt"]==0)
    completed=next(a for a in attempts if a["round_index"]==4 and a["learner_attempt"]==1)
    attempts[attempts.index(interrupted)]=completed
    attempt_path.write_text("".join(json.dumps(a,sort_keys=True)+"\n" for a in attempts))
    (case/"rounds/r0004/rollouts-attempt0.jsonl").unlink()
    (case/"rounds/r0004/batch-attempt0.json").unlink()
    reseal_records(changed)
    with pytest.raises(ValueError,match="duplicate optimizer attempt identity"):
        auditor.audit(changed)


@pytest.mark.parametrize("mode",["minimal","guarded"])
def test_rehashed_independent_evaluation_cohort_must_match_seeded_tasks(
        audit_records,tmp_path,mode):
    changed=tmp_path/f"mutated-heldout-{mode}"
    shutil.copytree(audit_records,changed)
    row=next(r for r in auditor.read(changed/"progress.json")
             if r["mode"]==mode and r["cutpoint"]=="none")
    path=changed/row["directory"]/"rounds/r0000/evaluation_tasks.json"
    tasks=auditor.read(path)
    tasks[0]["prompt"]+=" altered"
    bench.atomic_json(path,tasks)
    reseal_records(changed)
    with pytest.raises(ValueError,match="independent evaluation cohort provenance"):
        auditor.audit(changed)


def test_guarded_evaluation_attestation_must_match_retained_cohort_bytes(audit_records,tmp_path):
    changed=tmp_path/"guarded-heldout-digest-mismatch"
    shutil.copytree(audit_records,changed)
    row=next(r for r in auditor.read(changed/"progress.json")
             if r["mode"]=="guarded" and r["cutpoint"]=="none")
    path=changed/row["directory"]/"rounds/r0000/evaluation_tasks.json"
    tasks=auditor.read(path)
    original_digest=auditor.sha(path)
    path.write_text(json.dumps(tasks,indent=2,sort_keys=True)+"\n")
    assert auditor.read(path)==tasks and auditor.sha(path)!=original_digest
    reseal_records(changed)
    with pytest.raises(ValueError,match="guarded evaluation set digest"):
        auditor.audit(changed)


def test_rehashed_round_trace_cannot_drop_a_timed_committed_round(audit_records,tmp_path):
    changed=tmp_path/"missing-round-trace-span"
    shutil.copytree(audit_records,changed)
    progress=auditor.read(changed/"progress.json")
    row=next(r for r in progress if r["mode"]=="minimal" and r["cutpoint"]=="none")
    case=changed/row["directory"]
    trace_path=case/"trace.jsonl"
    trace=[json.loads(line) for line in trace_path.read_text().splitlines()]
    trace=[span for span in trace if not (span["stage"]=="round" and span["round_index"]==4)]
    trace_path.write_text("".join(json.dumps(span,sort_keys=True)+"\n" for span in trace))
    round_spans=[span["duration_s"] for span in trace if span["stage"]=="round"]
    row["round_wall_s"]=sum(round_spans)
    row["tasks_s"]=row["tokens_s"]=auditor.read(changed/"manifest.json")["arguments"]["tasks"]*len(round_spans)/sum(round_spans)
    bench.atomic_json(changed/"progress.json",progress)
    summary=auditor.read(changed/"summary.json")
    group=next(g for g in summary["normal"] if g["workers"]==row["workers"] and g["mode"]==row["mode"])
    values=[r["tasks_s"] for r in progress if r["cutpoint"]=="none" and
            r["workers"]==row["workers"] and r["mode"]==row["mode"]]
    group["tasks_s_mean"]=statistics.fmean(values)
    group["tasks_s_sd"]=statistics.stdev(values) if len(values)>1 else None
    bench.atomic_json(changed/"summary.json",summary)
    reseal_records(changed)
    with pytest.raises(ValueError,match="round trace spans do not cover"):
        auditor.audit(changed)


def test_normal_commit_cannot_claim_recovered_publication_to_skip_trace(audit_records,tmp_path):
    changed=tmp_path/"forged-recovered-publication"
    shutil.copytree(audit_records,changed)
    progress=auditor.read(changed/"progress.json")
    row=next(r for r in progress if r["mode"]=="guarded" and r["cutpoint"]=="none")
    case=changed/row["directory"]
    commit_path=case/"rounds/r0004/committed.json"
    commit=auditor.read(commit_path)
    commit["recovered_publication"]=True
    commit["result"]=None
    bench.atomic_json(commit_path,commit)
    trace_path=case/"trace.jsonl"
    trace=[json.loads(line) for line in trace_path.read_text().splitlines()]
    trace=[span for span in trace if not (span["stage"]=="round" and span["round_index"]==4)]
    trace_path.write_text("".join(json.dumps(span,sort_keys=True)+"\n" for span in trace))
    round_spans=[span["duration_s"] for span in trace if span["stage"]=="round"]
    row["round_wall_s"]=sum(round_spans)
    row["tasks_s"]=row["tokens_s"]=auditor.read(changed/"manifest.json")["arguments"]["tasks"]*len(round_spans)/sum(round_spans)
    bench.atomic_json(changed/"progress.json",progress)
    summary=auditor.read(changed/"summary.json")
    group=next(g for g in summary["normal"] if g["workers"]==row["workers"] and g["mode"]==row["mode"])
    values=[r["tasks_s"] for r in progress if r["cutpoint"]=="none" and
            r["workers"]==row["workers"] and r["mode"]==row["mode"]]
    group["tasks_s_mean"]=statistics.fmean(values)
    group["tasks_s_sd"]=statistics.stdev(values) if len(values)>1 else None
    bench.atomic_json(changed/"summary.json",summary)
    reseal_records(changed)
    with pytest.raises(ValueError,match="recovered publication is outside"):
        auditor.audit(changed)


def test_rollout_logprob_must_match_worker_and_rvl_generation(audit_records,tmp_path):
    changed=tmp_path/"misbound-behavior-logprob"
    shutil.copytree(audit_records,changed)
    row=next(r for r in auditor.read(changed/"progress.json")
             if r["mode"]=="minimal" and r["cutpoint"]=="none")
    round_dir=changed/row["directory"]/"rounds/r0000"
    rollout_path=round_dir/"rollouts-attempt0.jsonl"
    rollouts=[json.loads(line) for line in rollout_path.read_text().splitlines()]
    changed_rollout=rollouts[0]
    changed_rollout["logprob"]+=0.25
    rollout_path.write_text("".join(json.dumps(item,sort_keys=True)+"\n" for item in rollouts))
    batch_path=round_dir/"batch-attempt0.json"
    batch=auditor.read(batch_path)
    member=next(m for m in batch["members"] if m["created_step"]==changed_rollout["created_step"])
    member["logprob"]=changed_rollout["logprob"]
    bench.atomic_json(batch_path,batch)
    reseal_records(changed)
    with pytest.raises(ValueError,match="rollout logprob differs from worker/RVL evidence"):
        auditor.audit(changed)


def test_rehashed_evaluations_cannot_disappear_after_gate_entry(audit_records,tmp_path):
    changed=tmp_path/"missing-evaluation-pair-for-reached-gate"
    shutil.copytree(audit_records,changed)
    row=next(r for r in auditor.read(changed/"progress.json")
             if r["cutpoint"]=="journal_prepared")
    round_dir=changed/row["directory"]/"rounds/r0004"
    files=list(round_dir.glob("evaluation-*-attempt0.json"))
    assert len(files)==2
    for path in files:
        path.unlink()
    reseal_records(changed)
    with pytest.raises(ValueError,match="evaluation pairs do not cover every optimizer attempt"):
        auditor.audit(changed)


def test_rehashed_stale_replay_screening_event_cannot_disappear(audit_records,tmp_path):
    changed=tmp_path/"missing-stale-replay-screening-event"
    shutil.copytree(audit_records,changed)
    row=next(r for r in auditor.read(changed/"progress.json")
             if r["mode"]=="guarded" and r["cutpoint"]=="none")
    path=changed/row["directory"]/"rounds/r0001/engine-attempt0.jsonl"
    events=[json.loads(line) for line in path.read_text().splitlines()]
    events=[event for event in events if event.get("event")!="replay_experiences_screened"]
    path.write_text("".join(json.dumps(event,sort_keys=True)+"\n" for event in events))
    reseal_records(changed)
    with pytest.raises(ValueError,match="stale replay screening events"):
        auditor.audit(changed)


def test_stale_replay_canary_must_match_round_zero_rollout(audit_records,tmp_path):
    changed=tmp_path/"misbound-stale-replay-canary"
    shutil.copytree(audit_records,changed)
    row=next(r for r in auditor.read(changed/"progress.json")
             if r["mode"]=="guarded" and r["cutpoint"]=="none")
    path=changed/row["directory"]/"canary.json"
    canary=auditor.read(path)
    canary["attempt"]["output"]="forged"
    bench.atomic_json(path,canary)
    reseal_records(changed)
    with pytest.raises(ValueError,match="stale replay canary differs from its authenticated"):
        auditor.audit(changed)


def test_rehashed_normal_summary_cannot_omit_all_groups(audit_records,tmp_path):
    changed=tmp_path/"missing-normal-summary-groups"
    shutil.copytree(audit_records,changed)
    summary=auditor.read(changed/"summary.json")
    summary["normal"]=[]
    bench.atomic_json(changed/"summary.json",summary)
    reseal_records(changed)
    with pytest.raises(ValueError,match="normal summary groups do not cover"):
        auditor.audit(changed)


def test_rehashed_extra_optimizer_attempt_must_belong_to_killed_round(audit_records,tmp_path):
    changed=tmp_path/"optimizer-attempt-at-wrong-round"
    shutil.copytree(audit_records,changed)
    row=next(r for r in auditor.read(changed/"progress.json")
             if r["cutpoint"]=="post_optimizer")
    case=changed/row["directory"]
    attempt_path=case/"optimizer_attempts.jsonl"
    attempts=[json.loads(line) for line in attempt_path.read_text().splitlines()]
    target=next(a for a in attempts if a["round_index"]==4 and a["learner_attempt"]==0)
    source=next(a for a in attempts if a["round_index"]==3 and a["learner_attempt"]==0)
    moved={**source,"learner_attempt":1}
    attempts[attempts.index(target)]=moved
    attempt_path.write_text("".join(json.dumps(a,sort_keys=True)+"\n" for a in attempts))
    round3=case/"rounds/r0003"
    rollout0=round3/"rollouts-attempt0.jsonl"
    (round3/"rollouts-attempt1.jsonl").write_bytes(rollout0.read_bytes())
    batch0=round3/"batch-attempt0.json"
    bench.atomic_json(round3/"batch-attempt1.json",auditor.read(batch0))
    (case/"rounds/r0004/rollouts-attempt0.jsonl").unlink()
    (case/"rounds/r0004/batch-attempt0.json").unlink()
    for evaluation in list(round3.glob("evaluation-*-attempt0.json")):
        copied=auditor.read(evaluation)
        bench.atomic_json(round3/evaluation.name.replace("-attempt0", "-attempt1"),copied)
    reseal_records(changed)
    with pytest.raises(ValueError,match="optimizer attempt identities do not match the kill-round schedule"):
        auditor.audit(changed)


@pytest.mark.parametrize("field",["sampled_cpu_core_percent","sampled_tree_peak_rss_bytes"])
def test_resource_summaries_must_match_retained_samples(audit_records,tmp_path,field):
    changed=tmp_path/f"tampered-resource-summary-{field}"
    shutil.copytree(audit_records,changed)
    progress=auditor.read(changed/"progress.json")
    progress[0]["resources"][field]+=10
    bench.atomic_json(changed/"progress.json",progress)
    reseal_records(changed)
    expected="resource CPU summary" if field=="sampled_cpu_core_percent" else "peak RSS"
    with pytest.raises(ValueError,match=expected):
        auditor.audit(changed)


def test_derived_resource_metrics_are_read_from_samples(tmp_path):
    (tmp_path/"resources.jsonl").write_text(
        '{"time":10.0,"rss_bytes":100,"cpu_delta_s":0.2}\n'
        '{"time":11.0,"rss_bytes":200,"cpu_delta_s":0.8}\n')
    assert derive_tables.resource_metrics(tmp_path)=={
        "sampled_resource_window_s":1.0,
        "sampled_cpu_core_percent":100.0,"sampled_tree_peak_rss_bytes":200}


def test_derived_comparison_ignores_resealed_case_wall_summaries(audit_records,tmp_path):
    changed=tmp_path/"mutable-case-wall"
    shutil.copytree(audit_records,changed)
    rows=auditor.read(changed/"progress.json")
    row=next(item for item in rows if item["cutpoint"]=="none")
    case_summary=auditor.read(changed/row["directory"]/"summary.json")
    timing_path=changed/row["directory"]/"case_timing.json"
    if timing_path.exists():
        timing=auditor.read(timing_path)
        timing["end_monotonic_s"]+=1000
        bench.atomic_json(timing_path,timing)
        # Derive the published value from the mutated endpoints. Adding 1000 to
        # the old duration can differ by a few ulps from (end + 1000) - start.
        case_wall_s=timing["end_monotonic_s"]-timing["start_monotonic_s"]
    else:
        case_wall_s=row["case_wall_s"]+1000
    row["case_wall_s"]=case_wall_s
    case_summary["case_wall_s"]=case_wall_s
    bench.atomic_json(changed/"progress.json",rows)
    bench.atomic_json(changed/row["directory"]/"summary.json",case_summary)
    reseal_records(changed)
    assert auditor.audit(changed)["pass"]
    original_output=tmp_path/"original-derived"
    changed_output=tmp_path/"changed-derived"
    derive_tables.derive(audit_records,original_output)
    derive_tables.derive(changed,changed_output)
    with (original_output/"comparison.csv").open() as handle:
        original_rows=list(csv.DictReader(handle))
    with (changed_output/"comparison.csv").open() as handle:
        changed_rows=list(csv.DictReader(handle))
    assert original_rows==changed_rows
    assert "case_wall_s_mean" not in original_rows[0]
    assert "sampled_resource_window_s_mean" in original_rows[0]


def test_progress_case_wall_must_match_retained_case_summary(audit_records,tmp_path):
    changed=tmp_path/"progress-case-wall-mismatch"
    shutil.copytree(audit_records,changed)
    rows=auditor.read(changed/"progress.json")
    rows[0]["case_wall_s"]+=1
    bench.atomic_json(changed/"progress.json",rows)
    reseal_records(changed)
    with pytest.raises(ValueError,match="progress case wall differs"):
        auditor.audit(changed)


def test_case_timing_endpoints_must_match_published_case_wall(campaign,tmp_path):
    root,_=campaign
    changed=tmp_path/"tampered-case-endpoint"
    shutil.copytree(root,changed)
    row=auditor.read(changed/"progress.json")[0]
    timing_path=changed/row["directory"]/"case_timing.json"
    timing=auditor.read(timing_path)
    timing["end_monotonic_s"]+=10
    bench.atomic_json(timing_path,timing)
    reseal_records(changed)
    with pytest.raises(ValueError,match="case wall summary differs"):
        auditor.audit(changed)


def test_setup_startup_summary_must_match_retained_endpoints(campaign,tmp_path):
    root,_=campaign
    changed=tmp_path/"forged-startup-summary"
    shutil.copytree(root,changed)
    setup_path=changed/"normal-w2-s3901-guarded"/"setup-attempt0.json"
    setup=auditor.read(setup_path)
    assert "rollout_pool_started_monotonic_s" in setup
    assert "rollout_pool_ready_monotonic_s" in setup
    setup["rollout_pool_start_s"]=999.0
    bench.atomic_json(setup_path,setup)
    reseal_records(changed)
    with pytest.raises(ValueError,match="pool startup summary differs from retained timing endpoints"):
        auditor.audit(changed)


def test_derived_table_omits_unverified_legacy_pool_startups(audit_records,tmp_path):
    changed=tmp_path/"mutable-legacy-pool-startup"
    shutil.copytree(audit_records,changed)
    row=next(item for item in auditor.read(changed/"progress.json") if item["cutpoint"]=="none")
    setup_path=changed/row["directory"]/"setup-attempt0.json"
    setup=auditor.read(setup_path)
    setup["rollout_pool_start_s"]=999.0
    setup["evaluation_pool_start_s"]=999.0
    bench.atomic_json(setup_path,setup)
    reseal_records(changed)
    original_output=tmp_path/"original-derived"
    changed_output=tmp_path/"changed-derived"
    derive_tables.derive(audit_records,original_output)
    derive_tables.derive(changed,changed_output)
    with (original_output/"comparison.csv").open() as handle:
        original_rows=list(csv.DictReader(handle))
    with (changed_output/"comparison.csv").open() as handle:
        changed_rows=list(csv.DictReader(handle))
    assert original_rows==changed_rows
    assert "rollout_pool_start_s_mean" not in original_rows[0]
    assert "evaluation_pool_start_s_mean" not in original_rows[0]


def test_rehashed_promotion_must_match_independent_evaluation_pair(audit_records,tmp_path):
    changed=tmp_path/"promotion-disagrees-with-evaluation"
    shutil.copytree(audit_records,changed)
    row=next(r for r in auditor.read(changed/"progress.json")
             if r["mode"]=="guarded" and r["cutpoint"]=="none")
    round_dir=changed/row["directory"]/"rounds/r0000"
    commit_path=round_dir/"committed.json"
    commit=auditor.read(commit_path)
    evaluation_path=round_dir/f"evaluation-{commit['candidate_id']}-attempt0.json"
    evaluation=auditor.read(evaluation_path)
    scores={task_id:0.0 for task_id in evaluation["scores"]}
    evaluation["scores"]=scores
    evaluation["report"]["primary"]=0.0
    evaluation["report"]["metadata"]["per_task_scores"]=scores
    commit["result"]["candidate_eval"]=evaluation["report"]
    bench.atomic_json(evaluation_path,evaluation)
    bench.atomic_json(commit_path,commit)
    reseal_records(changed)
    with pytest.raises(ValueError,match="promotion decision differs from its independent evaluation pair"):
        auditor.audit(changed)


def test_rehashed_promoted_policy_must_match_the_gate_decision(audit_records,tmp_path):
    changed=tmp_path/"promoted-policy-disagrees-with-gate"
    shutil.copytree(audit_records,changed)
    row=next(r for r in auditor.read(changed/"progress.json")
             if r["mode"]=="guarded" and r["cutpoint"]=="none")
    commit_path=changed/row["directory"]/"rounds/r0000/committed.json"
    commit=auditor.read(commit_path)
    commit["result"]["promoted_id"]="policy-0"
    bench.atomic_json(commit_path,commit)
    reseal_records(changed)
    with pytest.raises(ValueError,match="promotion decision differs from its independent evaluation pair"):
        auditor.audit(changed)


@pytest.mark.parametrize("corruption",["marker_cutpoint","kill_record_cutpoint","marker_pid"])
def test_rehashed_cutpoint_must_match_the_killed_process(audit_records,tmp_path,corruption):
    changed=tmp_path/f"misbound-cutpoint-{corruption}"
    shutil.copytree(audit_records,changed)
    row=next(r for r in auditor.read(changed/"progress.json")
             if r["cutpoint"]=="post_optimizer")
    case=changed/row["directory"]
    marker_path=case/"cutpoint.json"
    killed_path=case/"kill_record.json"
    marker=auditor.read(marker_path)
    killed=auditor.read(killed_path)
    if corruption=="marker_cutpoint":
        marker["cutpoint"]="candidate_saved"
        bench.atomic_json(marker_path,marker)
    elif corruption=="kill_record_cutpoint":
        killed["cutpoint"]="candidate_saved"
        bench.atomic_json(killed_path,killed)
    else:
        marker["pid"]+=1
        bench.atomic_json(marker_path,marker)
    reseal_records(changed)
    with pytest.raises(ValueError,match="cutpoint marker differs"):
        auditor.audit(changed)


def test_rehashed_exact_recovery_summary_is_derived_from_round_comparisons(audit_records,tmp_path):
    changed=tmp_path/"forged-exact-recovery-flag"
    shutil.copytree(audit_records,changed)
    progress=auditor.read(changed/"progress.json")
    row=next(r for r in progress if r["cutpoint"]=="post_optimizer")
    row["exact_reference_match"]=False
    bench.atomic_json(changed/"progress.json",progress)
    reseal_records(changed)
    derived=derive_tables.derive(changed,tmp_path/"derived")
    recovery=next(r for r in derived["recovery"] if r["cutpoint"]=="post_optimizer")
    assert recovery["exact_reference_cases"]==recovery["n"]
    with pytest.raises(ValueError,match="recovery exact-reference flag differs"):
        auditor.audit(changed)


@pytest.mark.parametrize("corruption",["duration","missing_verification"])
def test_rehashed_non_round_trace_cannot_change_published_latency(audit_records,tmp_path,corruption):
    changed=tmp_path/f"trace-{corruption}"
    shutil.copytree(audit_records,changed)
    progress=auditor.read(changed/"progress.json")
    row=next(r for r in progress if r["cutpoint"]=="none" and r["mode"]=="guarded")
    case=changed/row["directory"]
    trace_path=case/"trace.jsonl"
    trace=[json.loads(line) for line in trace_path.read_text().splitlines()]
    if corruption=="duration":
        span=next(span for span in trace if span["stage"]=="training" and span["round_index"]==0)
        span["duration_s"]+=0.01
    else:
        trace.remove(next(span for span in trace if span["stage"]=="verification" and span["round_index"]==0))
    trace_path.write_text("".join(json.dumps(span,sort_keys=True)+"\n" for span in trace))
    reseal_records(changed)
    with pytest.raises(ValueError,match="trace span timestamps|verification trace coverage"):
        auditor.audit(changed)
    with pytest.raises(ValueError,match="trace span timestamps|verification trace coverage"):
        derive_tables.derive(changed,tmp_path/"derived")


def test_stale_rejection_table_uses_committed_round_evidence(audit_records,tmp_path):
    changed=tmp_path/"forged-stale-total"
    shutil.copytree(audit_records,changed)
    progress=auditor.read(changed/"progress.json")
    row=next(r for r in progress if r["cutpoint"]=="none" and r["mode"]=="guarded")
    committed=sum(auditor.read(changed/row["directory"]/"rounds"/f"r{i:04d}"/"committed.json")
                  ["stale_replay_rejected"] for i in range(row["rounds"]))
    row["stale_replay_rejected"]=committed+1
    bench.atomic_json(changed/"progress.json",progress)
    reseal_records(changed)
    derived=derive_tables.derive(changed,tmp_path/"derived")
    table_row=next(item for item in derived["comparison"]
                   if item["workers"]==row["workers"] and item["mode"]=="guarded")
    expected=sum(sum(auditor.read(changed/normal["directory"]/"rounds"/f"r{i:04d}"/"committed.json")
                     ["stale_replay_rejected"] for i in range(normal["rounds"]))
                 for normal in progress if normal["cutpoint"]=="none" and
                 normal["workers"]==row["workers"] and normal["mode"]=="guarded")
    assert table_row["stale_canary_rejected"]==expected
    with pytest.raises(ValueError,match="progress stale-replay total"):
        auditor.audit(changed)


def test_rehashed_policy_lag_trace_must_match_rollout_provenance(audit_records,tmp_path):
    changed=tmp_path/"forged-policy-lag"
    shutil.copytree(audit_records,changed)
    row=next(r for r in auditor.read(changed/"progress.json")
             if r["cutpoint"]=="none" and r["mode"]=="guarded")
    trace_path=changed/row["directory"]/"trace.jsonl"
    trace=[json.loads(line) for line in trace_path.read_text().splitlines()]
    span=next(item for item in trace if item["stage"]=="rollout" and item["round_index"]==0)
    span["policy_lag"]=999
    trace_path.write_text("".join(json.dumps(item,sort_keys=True)+"\n" for item in trace))
    reseal_records(changed)
    with pytest.raises(ValueError,match="rollout trace policy lag differs from rollout provenance"):
        auditor.audit(changed)
    with pytest.raises(ValueError,match="rollout trace policy lag differs from rollout provenance"):
        derive_tables.derive(changed,tmp_path/"derived")


def test_rehashed_live_descendant_total_comes_from_kill_record(audit_records,tmp_path):
    changed=tmp_path/"forged-live-descendants"
    shutil.copytree(audit_records,changed)
    progress=auditor.read(changed/"progress.json")
    row=next(item for item in progress if item["cutpoint"]=="post_optimizer")
    row["live_old_descendants"]=9
    bench.atomic_json(changed/"progress.json",progress)
    reseal_records(changed)
    derived=derive_tables.derive(changed,tmp_path/"derived")
    recovery=next(item for item in derived["recovery"] if item["cutpoint"]=="post_optimizer")
    assert recovery["live_old_descendants"]==0
    with pytest.raises(ValueError,match="progress live descendant count differs from kill record"):
        auditor.audit(changed)
