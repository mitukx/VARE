"""Post-hoc statistics for learning_pipeline_v1 run-001 (stdlib only)."""
import json, math, sys
from pathlib import Path
from collections import Counter
root = Path(sys.argv[1])
ARMS = ["no_update", "sft_reference", "rft_self"]
TRUSTED = "trusted_full_v1"
def load(arm):
    rows = [json.loads(l) for l in open(root / "eval" / arm / "graded.jsonl")]
    return rows
def wilson(k, n, z=1.959963984540054):
    p = k / n; d = 1 + z*z/n
    c = (p + z*z/(2*n)) / d; h = z*math.sqrt(p*(1-p)/n + z*z/(4*n*n)) / d
    return max(0.0, c-h), min(1.0, c+h)
def binom_cdf(k, n):
    return sum(math.comb(n, i) for i in range(k+1)) / 2**n
def mcnemar_exact(b, c):
    n = b + c
    if n == 0: return 1.0, 1.0
    two = min(1.0, 2*binom_cdf(min(b, c), n))
    mid = min(1.0, two - math.comb(n, min(b, c)) / 2**n)
    return two, mid
def newcombe_paired(a, b, c, d):
    # Newcombe (1998) method 10 CI for paired difference p_arm - p_base; a=both pass,b=arm only,c=base only,d=neither
    n = a+b+c+d
    p1 = (a+b)/n; p2 = (a+c)/n  # arm, base
    l1, u1 = wilson(a+b, n); l2, u2 = wilson(a+c, n)
    num = (a*d - b*c)
    den = math.sqrt((a+b)*(c+d)*(a+c)*(b+d)) if (a+b)*(c+d)*(a+c)*(b+d) else 0
    phi = 0.0
    if den:
        if num > n/2: phi = (num - n/2)/den
        elif num >= 0: phi = 0.0
        else: phi = num/den
    diff = p1 - p2
    dl = math.sqrt((p1-l1)**2 - 2*phi*(p1-l1)*(u2-p2) + (u2-p2)**2)
    du = math.sqrt((u1-p1)**2 - 2*phi*(u1-p1)*(p2-l2) + (p2-l2)**2)
    return diff - dl, diff + du
data = {}
out = {"arms": {}, "paired_vs_no_update": {}}
for arm in ARMS:
    rows = load(arm)
    ids = [r["instance_id"] for r in rows]
    assert len(ids) == len(set(ids)) == 42, arm
    data[arm] = {r["instance_id"]: int(bool(r["verifiers"][TRUSTED]["passed"])) for r in rows}
    k = sum(data[arm].values()); n = len(rows)
    cheap = sum(int(bool(r["verifiers"]["cheap_f2p_v1"]["passed"])) for r in rows)
    lo, hi = wilson(k, n)
    out["arms"][arm] = {"n": n, "trusted_pass": k, "rate": k/n, "wilson95": [lo, hi],
                        "cheap_pass": cheap, "cheap_false_positive": sum(int(bool(r.get("cheap_false_positive", False))) for r in rows),
                        "outcomes": dict(Counter(r["outcome"] for r in rows)),
                        "checkpoint_ids": sorted({r["checkpoint_id"] for r in rows})}
assert set(data["no_update"]) == set(data["sft_reference"]) == set(data["rft_self"])
ids = sorted(data["no_update"])
raw = {}
for arm in ARMS[1:]:
    a = sum(1 for t in ids if data[arm][t] and data["no_update"][t])
    b = sum(1 for t in ids if data[arm][t] and not data["no_update"][t])
    c = sum(1 for t in ids if not data[arm][t] and data["no_update"][t])
    d = len(ids) - a - b - c
    two, mid = mcnemar_exact(b, c)
    one = binom_cdf(min(b, c), b + c) if b + c else 1.0
    raw[arm] = two
    out["paired_vs_no_update"][arm] = {"both_pass": a, "arm_only": b, "base_only": c, "neither": d,
        "diff": (b - c)/len(ids), "newcombe95_diff": list(newcombe_paired(a, b, c, d)),
        "mcnemar_exact_two_sided_p": two, "mcnemar_mid_p_two_sided": mid,
        "exploratory_one_sided_p_arm_better": one if b > c else 1 - binom_cdf(b - 1, b + c) if b+c else 1.0,
        "arm_only_tasks": [t for t in ids if data[arm][t] and not data["no_update"][t]],
        "base_only_tasks": [t for t in ids if not data[arm][t] and data["no_update"][t]]}
# Holm over the two preregistered comparisons
order = sorted(raw, key=raw.get); m = len(order); running = 0
for i, arm in enumerate(order):
    running = max(running, min(1.0, (m - i) * raw[arm]))
    out["paired_vs_no_update"][arm]["holm_p"] = running
# arm-vs-arm (exploratory)
b = sum(1 for t in ids if data["rft_self"][t] and not data["sft_reference"][t])
c = sum(1 for t in ids if not data["rft_self"][t] and data["sft_reference"][t])
out["exploratory_rft_vs_sft"] = {"rft_only": b, "sft_only": c, "mcnemar_exact_two_sided_p": mcnemar_exact(b, c)[0]}
# rollouts
rpath = root / "rollouts" / "graded.jsonl"  # 2.7 MB, not committed; see SHA256SUMS.txt
rr = [json.loads(l) for l in open(rpath)] if rpath.exists() else []
tp = sum(int(bool(r["verifiers"][TRUSTED]["passed"])) for r in rr)
tasks = {}
for r in rr: tasks.setdefault(r["instance_id"], []).append(int(bool(r["verifiers"][TRUSTED]["passed"])))
if rr: out["train_rollouts"] = {"samples": len(rr), "trusted_pass": tp, "wilson95": list(wilson(tp, len(rr))) if rr else None,
    "tasks": len(tasks), "tasks_with_pass": sum(1 for v in tasks.values() if any(v)),
    "tasks_all_pass": sum(1 for v in tasks.values() if all(v)),
    "mixed_groups": sum(1 for v in tasks.values() if 0 < sum(v) < len(v)),
    "cheap_false_positive": sum(int(bool(r.get("cheap_false_positive", False))) for r in rr)}
out["method"] = ("Pass = trusted_full_v1 (F2P+P2P) verifier passed. Exact McNemar: two-sided binomial test on discordant pairs, "
    "p=min(1,2*P(X<=min(b,c))), X~Bin(b+c,0.5). Holm over the two preregistered comparisons. Wilson score 95% CIs; "
    "Newcombe method-10 95% CI for paired difference. One-sided and arm-vs-arm values are exploratory, not preregistered.")
print(json.dumps(out, indent=2))
