"""Actual CPU weight-sync and full learner SIGKILL/restart falsifiers."""
import os
from pathlib import Path
import shutil
import subprocess
import sys
import json

import pytest

from scripts import benchmark_e2e_rl_continuation as bench
from scripts import audit_e2e_rl_continuation as auditor


def test_v2_task_namespaces_are_round_and_evaluation_disjoint():
    train=bench.fresh_tasks(3901,0,8)
    future=bench.fresh_tasks(3901,1,8)
    heldout=bench.fresh_tasks(3901,0,8,True)
    assert train==bench.fresh_tasks(3901,0,8)
    assert len({t.prompt for t in [*train,*future,*heldout]})==24


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
