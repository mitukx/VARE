"""Matched synthetic refresh controls over the existing task-family TV monitor.

No model, optimizer or learned refresh algorithm. Independent audit labels never
enter the refresh decision/fitting interface. All policy shifts are exogenous.
"""
from __future__ import annotations
import argparse
import gzip
import json
from pathlib import Path
import random
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from vare.evidence import sha256_file, sha256_json  # noqa: E402
from vare.lag import DistributionShiftMonitor  # noqa: E402
from vare.config import PromotionConfig  # noqa: E402
from vare.promotion import PromotionGate  # noqa: E402
from vare.types import EvaluationReport  # noqa: E402


def cohort(seed, scenario, phase, split, size, *, baseline=False):
    rng = random.Random(f"{seed}:{scenario}:{phase}:{split}")
    rows = []
    attack_rate = 0 if baseline or phase < 4 else (.4 if phase < 8 else .15)
    quality = .7 if baseline or phase < 4 else (.72 if phase < 8 else .9)
    for i in range(size):
        attack = rng.random() < attack_rate
        truth = False if attack else rng.random() < quality
        rows.append({"id":f"{seed}:{scenario}:{phase}:{split}:{i}",
                     "family":"B" if attack and scenario=="visible" else "A",
                     "feature":"proxy-exploit" if attack else "ordinary",
                     "proxy":True if attack else truth,"truth":truth})
    return rows


class RefreshControl:
    """Fixture of existing lifecycle contract; max two 64-label refreshes."""
    def __init__(self, strategy):
        self.strategy = strategy
        self.version = 0
        self.labels = 0
        self.table = {}
        self.monitor = DistributionShiftMonitor(window=256)
        self.monitor.fit_reference(["A"]*256)

    def observe(self, training):
        for row in training:
            self.monitor.observe(row["family"])
        return self.monitor.score()

    def decide_and_fit(self, phase, training, calibration):
        # Only observations and explicitly budgeted calibration labels enter.
        # Held-out labels are never an argument.
        score = self.observe(training)
        trigger = self.strategy=="periodic" and phase in (4,8)
        trigger |= self.strategy=="shift" and score>.25
        refreshed = trigger and self.version < 2
        if refreshed:
            grouped = {}
            for row in calibration:
                grouped.setdefault((row["feature"],row["proxy"]),[]).append(row["truth"])
            self.table.update({key:sum(values)/len(values)>=.5 for key,values in grouped.items()})
            self.version += 1
            self.labels += len(calibration)
            self.monitor.fit_reference(row["family"] for row in training)
        return {"shift_tv":score,"refreshed":bool(refreshed),"verifier_version":self.version,"labels_spent":self.labels}

    def verify(self, rows):
        return [self.table.get((row["feature"],row["proxy"]),row["proxy"]) for row in rows]


def validate_mechanics(protocol, root=ROOT):
    for name, expected in protocol["mechanics_sha256"].items():
        if sha256_file(root/name) != expected:
            raise ValueError("frozen mechanics mismatch: " + name)


def run(args):
    args.output.mkdir(parents=True,exist_ok=False)
    args.output_created = True
    protocol_path = ROOT/'benchmarks/verifier_refresh_controls_v2/protocol.json'
    protocol = json.loads(protocol_path.read_text())
    if args.seed not in protocol['seeds'] or args.scenario not in protocol['scenarios'] or sha256_file(__file__) != protocol['runner_sha256']:
        raise ValueError('condition outside fixed protocol')
    validate_mechanics(protocol)
    (args.output/'protocol.json').write_bytes(protocol_path.read_bytes())
    for name in protocol['mechanics_sha256']:
        target=args.output/'source'/name
        target.parent.mkdir(parents=True,exist_ok=True)
        target.write_bytes((ROOT/name).read_bytes())
    (args.output/'runner.py').write_bytes(Path(__file__).read_bytes())
    (args.output/'lag.py').write_bytes((ROOT/'src/vare/lag.py').read_bytes())
    controls = {name:RefreshControl(name) for name in ('fixed','periodic','shift')}
    gate = PromotionGate(PromotionConfig(min_primary_gain=.03,min_eval_examples=256))
    metrics, curves = {}, []
    for name in controls:
        metrics[name] = {'errors':0,'audit_n':0,'disagreements':0,'false_proxy_promotions':0,'missed_proxy_improvements':0,'proxy_promotions':0,'independent_promotions':0}
    start = time.perf_counter()
    with gzip.GzipFile(filename=str(args.output/'cohorts.jsonl.gz'),mode='wb',mtime=0) as raw:
        for phase in range(12):
            training = cohort(args.seed,args.scenario,phase,'training',256)
            calibration = cohort(args.seed,args.scenario,phase,'calibration',64)
            audit = cohort(args.seed,args.scenario,phase,'independent',256)
            baseline = cohort(args.seed,args.scenario,phase,'baseline-independent',256,baseline=True)
            for split, rows in [('training',training),('calibration',calibration),('independent',audit),('baseline-independent',baseline)]:
                for row in rows:
                    raw.write((json.dumps({'phase':phase,'split':split,**row},sort_keys=True)+'\n').encode())
            baseline_report = EvaluationReport('incumbent',sum(r['truth'] for r in baseline)/256,n=256)
            truth_report = EvaluationReport('candidate',sum(r['truth'] for r in audit)/256,n=256)
            truth_promote = gate.decide(baseline_report,truth_report).accepted
            for name, control in controls.items():
                refresh = control.decide_and_fit(phase,training,calibration)
                outputs = control.verify(audit)
                proxy_report = EvaluationReport('candidate',sum(outputs)/256,n=256)
                proxy_promote = gate.decide(baseline_report,proxy_report).accepted
                error = sum(v != row['truth'] for v,row in zip(outputs,audit,strict=True))
                disagreement = sum(v != row['proxy'] for v,row in zip(outputs,audit,strict=True))
                metrics[name]['errors'] += error
                metrics[name]['audit_n'] += 256
                metrics[name]['disagreements'] += disagreement
                metrics[name]['false_proxy_promotions'] += int(proxy_promote and not truth_promote)
                metrics[name]['missed_proxy_improvements'] += int(truth_promote and not proxy_promote)
                metrics[name]['proxy_promotions'] += int(proxy_promote)
                metrics[name]['independent_promotions'] += int(truth_promote)
                curves.append({'phase':phase,'strategy':name,**refresh,'audit_errors':error,'audit_n':256,
                    'proxy_primary':proxy_report.primary,'independent_primary':truth_report.primary,
                    'baseline_independent_primary':baseline_report.primary,'proxy_promote':proxy_promote,'independent_promote':truth_promote})
    for name, control in controls.items():
        metrics[name].update({'error_rate':metrics[name]['errors']/metrics[name]['audit_n'],
                             'refreshes':control.version,'calibration_labels':control.labels})
    summary={'pass':True,'seed':args.seed,'scenario':args.scenario,'metrics':metrics,'wall_s':time.perf_counter()-start,
             'revision':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
             'protocol_sha256':sha256_json(protocol),'source_sha256':sha256_file(__file__),
             'scope':'Synthetic exogenous policy distribution shift, no model training; false/missed decisions are proxy-only counterfactuals. Independent gate decisions use separate audit labels.'}
    for name,data in [('curves.json',curves),('summary.json',summary)]:
        (args.output/name).write_text(json.dumps(data,indent=2)+'\n')
    (args.output/'SHA256.json').write_text(json.dumps({str(p.relative_to(args.output)):sha256_file(p) for p in args.output.rglob('*') if p.is_file()},indent=2)+'\n')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--seed',type=int,required=True)
    parser.add_argument('--scenario',choices=('visible','hidden'),required=True)
    args=parser.parse_args()
    try:
        run(args)
    except BaseException as exc:
        if getattr(args,'output_created',False):
            (args.output/'failure.json').write_text(json.dumps({'type':type(exc).__name__,'message':str(exc)})+'\n')
        raise
