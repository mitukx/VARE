#!/usr/bin/env python3
"""Execute the frozen SFT run with RSS, swap, and wall-time aborts."""
import hashlib, json, os, signal, subprocess, sys, time
from pathlib import Path
HERE=Path(__file__).resolve().parent
protocol=json.loads((HERE/'execution-protocol-v2.json').read_text())
cmd=protocol['execution_command']
env=os.environ.copy(); env.update(protocol['environment'])
def free_swap_mib():
    out=subprocess.check_output(['sysctl','vm.swapusage'],text=True)
    return float(out.split('free = ',1)[1].split('M',1)[0])
def rss_mib(pid):
    try:
        out=subprocess.check_output(['ps','-o','rss=','-p',str(pid)],text=True).strip()
        return float(out)/1024.0 if out else 0.0
    except subprocess.CalledProcessError:
        return 0.0
start=time.monotonic(); initial_swap=free_swap_mib()
if initial_swap < protocol['compute_limits']['free_swap_mib_min']:
    raise SystemExit(f'preflight_abort_free_swap:{initial_swap:.2f}MiB')
for name in ('resources.jsonl','train.stdout.log','train.stderr.log'):
    (HERE/name).write_text('')
with (HERE/'train.stdout.log').open('w') as stdout, (HERE/'train.stderr.log').open('w') as stderr:
    proc=subprocess.Popen(cmd,stdout=stdout,stderr=stderr,env=env)
    peak=0.0; reason='completed'; samples=0
    with (HERE/'resources.jsonl').open('a') as resource:
        while proc.poll() is None:
            rss=rss_mib(proc.pid); free=free_swap_mib(); elapsed=time.monotonic()-start
            peak=max(peak,rss); samples+=1
            resource.write(json.dumps({'elapsed_s':round(elapsed,3),'pid':proc.pid,'rss_mib':round(rss,2),'free_swap_mib':round(free,2)})+'\n'); resource.flush()
            if rss > protocol['compute_limits']['process_rss_mib_max']:
                reason='abort_process_rss_cap'; proc.send_signal(signal.SIGTERM)
            elif free < protocol['compute_limits']['free_swap_mib_min']:
                reason='abort_free_swap_floor'; proc.send_signal(signal.SIGTERM)
            elif elapsed > protocol['compute_limits']['wall_seconds']:
                reason='abort_wall_cap'; proc.send_signal(signal.SIGTERM)
            if proc.poll() is not None: break
            time.sleep(1)
        try: rc=proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill(); rc=proc.wait()
result={'protocol_sha256':hashlib.sha256((HERE/'execution-protocol-v2.json').read_bytes()).hexdigest(),'command':cmd,'pid':proc.pid,'exit_code':rc,'abort_reason':reason if reason!='completed' else None,'elapsed_seconds':round(time.monotonic()-start,3),'peak_rss_mib':round(peak,2),'initial_free_swap_mib':initial_swap,'final_free_swap_mib':free_swap_mib(),'resource_samples':samples,'adapter_exists':(HERE/'adapter').exists()}
(HERE/'training-result.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
sys.exit(0 if rc==0 and reason=='completed' else 1)
