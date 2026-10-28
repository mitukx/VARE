"""Independent retained-real-trajectory recovery audit; no model is loaded."""
from pathlib import Path
import hashlib
import json
import os
import shutil
import subprocess
import sys

BASE = Path(__file__).resolve().parents[1]
ROOT = BASE / "work/VARE"
CASE = "cpu-qwen25-3b-manifest-nonempty"
ORIGINAL = BASE / "outputs/cpu-model-campaign" / CASE
PROTOCOL = ROOT / "benchmarks/vare_development_repair_v1/campaign-cpu-v2-protocol.json"
OUTPUT = BASE / "outputs/retained-trajectory-recovery-audit"
OUTPUT.mkdir()

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def write(p, value):
    with p.open('x') as stream:
        json.dump(value, stream, sort_keys=True, indent=2)
        stream.flush()
        os.fsync(stream.fileno())

pending = json.loads((ORIGINAL / 'pending-grade.json').read_text())
write(OUTPUT / 'protocol.json', {
    'id': 'retained-real-trajectory-recovery-audit-v1',
    'inputs': pending['files_sha256'], 'original_pending_sha256': sha(ORIGINAL/'pending-grade.json'),
    'campaign_protocol_sha256': sha(PROTOCOL),
    'probe_source_sha256': sha(Path(__file__)),
    'acceptance': 'A subprocess durably records the pending boundary and exits by actual SIGKILL (-9). Existing pinned recovery then grades the exact retained candidate, without model loading. Corrupting retained source must fail before grading; a completed cell must refuse another execution.',
    'claim_boundary': 'Independent recovery audit using retained real-model artifacts, not another model attempt, model quality result, within-generation recovery test, or long-horizon RL run.',
})

def stage(label):
    base = OUTPUT / label
    cell = base / CASE
    cell.mkdir(parents=True)
    for name in [*pending['files_sha256'], 'pending-grade.json']:
        target = cell / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ORIGINAL/name, target)
    return base, cell

base, cell = stage('sigkill-recovery')
child = """from pathlib import Path
import json,os,signal,sys
p=Path(sys.argv[1])/'fault-observed-boundary.json'
with p.open('x') as f:
 json.dump({'boundary':'durable retained rollout before grading','pid':os.getpid()},f);f.flush();os.fsync(f.fileno())
fd=os.open(p.parent,os.O_RDONLY);os.fsync(fd);os.close(fd)
os.kill(os.getpid(),signal.SIGKILL)
"""
fault = subprocess.run([sys.executable, '-c', child, str(cell)], capture_output=True, text=True)
write(OUTPUT/'observed-sigkill.json', {'returncode':fault.returncode,'expected_returncode':-9,'stderr':fault.stderr})
assert fault.returncode == -9
command = [sys.executable, str(ROOT/'scripts/run_vare_development_campaign_v1.py'),
           '--protocol', str(PROTOCOL), '--case', CASE, '--output', str(base)]
recovery = subprocess.run(command, capture_output=True, text=True, timeout=45)
write(OUTPUT/'recovery-process.json', {'command':command,'returncode':recovery.returncode,'stdout':recovery.stdout,'stderr':recovery.stderr})
assert recovery.returncode == 0
result = json.loads((cell/'result.json').read_text())
assert result['recovered_after_sigkill'] and result['model_invocations'] == 1
assert not (cell/'inference').exists()
repeat = subprocess.run(command,capture_output=True,text=True,timeout=10)
assert repeat.returncode != 0 and 'completed cell cannot be rerun' in repeat.stderr
write(OUTPUT/'repeat-rejection.json',{'returncode':repeat.returncode,'stderr':repeat.stderr})

badbase,badcell = stage('tampered-source')
source = badcell/'workspace/scripts/verify_coding_agent_evidence_manifest_v1.py'
source.write_text(source.read_text()+'\n# changed after durable boundary\n')
badcommand = command[:-1]+[str(badbase)]
tampered = subprocess.run(badcommand,capture_output=True,text=True,timeout=10)
assert tampered.returncode != 0 and 'recovery artifact mismatch' in tampered.stderr
assert not (badcell/'grading').exists() and not (badcell/'inference').exists()
write(OUTPUT/'tamper-rejection.json',{'returncode':tampered.returncode,'stderr':tampered.stderr})
write(OUTPUT/'summary.json',{'status':'AUDIT_PASS','observed_fault_returncode':fault.returncode,
                           'recovery_completed':True,'new_model_invocations':0,
                           'historical_model_invocations':result['model_invocations'],
                           'candidate_sha256_preserved':result['candidate_sha256'],
                           'semantic_result_unchanged':result['grade']['success'] is False,
                           'tamper_rejected_before_grading':True,'repeat_rejected':True})
print((OUTPUT/'summary.json').read_text())
