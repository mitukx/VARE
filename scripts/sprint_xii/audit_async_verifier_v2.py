#!/usr/bin/env python3
"""Strengthened append-only audit v2; v1 and its receipts remain historical."""

import argparse
import gzip
import hashlib
import io
import json
import math
from pathlib import Path
import torch


def read(p):
    return json.loads(p.read_text())


def digest(b):
    return hashlib.sha256(b).hexdigest()


def canon(x):
    return json.dumps(x, sort_keys=True, separators=(",", ":")).encode()


def state(path):
    raw = path.read_bytes()
    assert digest(raw) == read(Path(str(path) + ".json"))["sha256"]
    return torch.load(
        io.BytesIO(gzip.decompress(raw)), weights_only=True, map_location="cpu"
    )


def audit(protocol, output):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    p = read(protocol)
    torch.set_num_threads(1)
    model = AutoModelForCausalLM.from_pretrained(
        p["model"]["local_snapshot"],
        local_files_only=True,
        dtype=torch.float32,
        attn_implementation="eager",
    ).eval()
    tokenizer = AutoTokenizer.from_pretrained(
        p["model"]["local_snapshot"], local_files_only=True
    )
    checks = []
    details = []

    def check(label, condition):
        checks.append({"check": label, "pass": bool(condition)})
        if not condition:
            raise AssertionError(label)

    for seed in p["seeds"]:
        reference_groups = [
            read(output / f"seed-{seed}/A/group-{i}.json") for i in range(p["groups"])
        ]
        for g in reference_groups:
            with torch.inference_mode():
                logits = model(torch.tensor([g["prompt_ids"]])).logits[
                    0, -1, g["action_ids"]
                ]
            check(
                f"{seed}/{g['index']}/behavior-model-logits",
                torch.allclose(
                    logits, torch.tensor(g["base_logits"]), atol=2e-5, rtol=1e-6
                ),
            )
            probs = torch.log_softmax(logits / p["temperature"], dim=0)
            check(
                f"{seed}/{g['index']}/behavior-logprobs",
                all(
                    abs(
                        row["behavior_logprob"]
                        - float(probs["ABCD".index(row["response"])])
                    )
                    < 2e-5
                    for row in g["generations"]
                ),
            )
        final = {}
        for arm in p["arm_order"]:
            root = output / f"seed-{seed}" / arm
            rows = [
                json.loads(x) for x in (root / "learner.jsonl").read_text().splitlines()
            ]
            prev = "0" * 64
            for index, row in enumerate(rows):
                copy = dict(row)
                got = copy.pop("sha")
                assert (
                    row["seq"] == index
                    and row["previous"] == prev
                    and digest(canon(copy)) == got
                )
                prev = got
            check(f"{seed}/{arm}/journal-chain", True)
            receipts = [read(root / f"done-{i}.json") for i in range(p["groups"])]
            commits = [x["group"] for x in rows if x["kind"] == "commit"]
            check(
                f"{seed}/{arm}/unique-effective-commits",
                len(commits)
                == len(set(commits))
                == sum(x["accepted"] for x in receipts),
            )
            param = torch.nn.Parameter(torch.zeros(4))
            opt = torch.optim.AdamW([param], lr=p["learning_rate"])
            last = None
            for i, receipt in enumerate(receipts):
                g = read(root / f"group-{i}.json")
                ref = reference_groups[i]
                check(
                    f"{seed}/{arm}/{i}/matched-generated-actions",
                    g["generations"] == ref["generations"]
                    and g["prompt_ids"] == ref["prompt_ids"],
                )
                check(
                    f"{seed}/{arm}/{i}/artifact-hash",
                    digest((root / f"group-{i}.json").read_bytes())
                    == read(root / f"group-{i}.sha.json")["sha256"],
                )
                original = [
                    float(
                        x["response"] in ("AB" if g["verifier_version"] == 0 else "CD")
                    )
                    for x in g["generations"]
                ]
                check(
                    f"{seed}/{arm}/{i}/raw-reward-identity",
                    original == g["rewards"] and len(set(original)) == 2,
                )
                expected = (
                    original
                    if arm == "B"
                    else [
                        float(x["response"] in ("AB" if i == 0 else "CD"))
                        for x in g["generations"]
                    ]
                )
                check(
                    f"{seed}/{arm}/{i}/use-time-admission",
                    receipt["accepted"] == (not (arm == "C" and i in (1, 2))),
                )
                if not receipt["accepted"]:
                    continue
                check(
                    f"{seed}/{arm}/{i}/reward-identity", receipt["rewards"] == expected
                )
                events = [
                    x
                    for x in rows
                    if x["kind"] == "optimizer_applied" and x["group"] == i
                ]
                check(f"{seed}/{arm}/{i}/real-update-record", len(events) >= 1)
                for event in events:
                    check(
                        f"{seed}/{arm}/{i}/gradient-origin-state",
                        torch.allclose(
                            param.detach(),
                            torch.tensor(event["bias_before4"]),
                            atol=1e-7,
                            rtol=1e-5,
                        ),
                    )
                    bias = param.detach().clone().requires_grad_(True)
                    current = torch.log_softmax(
                        (torch.tensor(g["base_logits"]) + bias) / p["temperature"], 0
                    )
                    r = receipt["rewards"]
                    mean = sum(r) / len(r)
                    scale = math.sqrt(sum((x - mean) ** 2 for x in r) / len(r) + 1e-6)
                    advantages = torch.tensor([(x - mean) / scale for x in r])
                    indexes = torch.tensor(
                        ["ABCD".index(x["response"]) for x in g["generations"]]
                    )
                    old = torch.tensor(
                        [x["behavior_logprob"] for x in g["generations"]]
                    )
                    ratio = torch.exp((current[indexes] - old).clamp(-20, 20))
                    loss = -torch.minimum(
                        ratio * advantages, ratio.clamp(0.8, 1.2) * advantages
                    ).mean()
                    loss.backward()
                    grad = bias.grad
                    grad = grad * min(1.0, 1.0 / (float(grad.norm()) + 1e-6))
                    check(
                        f"{seed}/{arm}/{i}/independent-gradient",
                        torch.allclose(
                            grad, torch.tensor(event["gradient4"]), atol=2e-5, rtol=1e-4
                        ),
                    )
                opt.zero_grad()
                param.grad = torch.tensor(events[-1]["gradient4"])
                opt.step()
                for event in events:
                    check(
                        f"{seed}/{arm}/{i}/post-update-state",
                        torch.allclose(
                            param.detach(),
                            torch.tensor(event["bias_after4"]),
                            atol=1e-7,
                            rtol=1e-5,
                        ),
                    )
                cp = state(root / f"checkpoint-{i}.pt.gz")
                ids = g["action_ids"]
                check(
                    f"{seed}/{arm}/{i}/Adam-parameter-replay",
                    torch.allclose(param, cp["bias"][ids], atol=1e-7, rtol=1e-5),
                )
                actual = list(cp["optimizer"]["state"].values())[0]
                expected_opt = opt.state[param]
                check(
                    f"{seed}/{arm}/{i}/Adam-state-replay",
                    all(
                        torch.allclose(
                            actual[k][ids] if k != "step" else actual[k],
                            expected_opt[k],
                            atol=1e-7,
                            rtol=1e-5,
                        )
                        for k in ["step", "exp_avg", "exp_avg_sq"]
                    ),
                )
                check(
                    f"{seed}/{arm}/{i}/RNG-state",
                    torch.equal(
                        cp["rng"], torch.Generator().manual_seed(seed).get_state()
                    ),
                )
                outside = cp["bias"].clone()
                outside[ids] = 0
                check(
                    f"{seed}/{arm}/{i}/only-declared-adapter-changed",
                    not bool(torch.count_nonzero(outside)),
                )
                last = cp
            active = read(root / "active.json")
            final_group = max(x["group"] for x in receipts if x["accepted"])
            expected_path = f"checkpoint-{final_group}.pt.gz"
            expected_manifest = read(root / (expected_path + ".json"))
            final_commit = next(
                x for x in rows if x["kind"] == "commit" and x["group"] == final_group
            )
            check(
                f"{seed}/{arm}/active-final-commit-path",
                active["path"] == expected_path,
            )
            check(
                f"{seed}/{arm}/active-final-commit-hash",
                active["sha256"]
                == expected_manifest["sha256"]
                == final_commit["checkpoint_sha"]
                == digest((root / expected_path).read_bytes()),
            )
            check(
                f"{seed}/{arm}/active-final-policy-version",
                active["policy_version"]
                == expected_manifest["metadata"]["policy_version"]
                == len(commits),
            )
            active_state = state(root / active["path"])
            check(
                f"{seed}/{arm}/active-is-validated-final-state",
                torch.equal(active_state["bias"], last["bias"]),
            )
            final[arm] = active_state
            recorded_eval = read(root / "evaluation.json")
            evaluated = []
            for evaluation_index in range(3):
                text = tokenizer.apply_chat_template(
                    [
                        {"role": "system", "content": "Follow the output format."},
                        {
                            "role": "user",
                            "content": f"Synthetic evaluation case {100 + evaluation_index}. Select one route label from A, B, C, D. Reply with one label.",
                        },
                    ],
                    tokenize=False,
                    add_generation_prompt=True,
                )
                expected_ids = tokenizer.encode(text, add_special_tokens=False)
                row = recorded_eval["tasks"][evaluation_index]
                check(
                    f"{seed}/{arm}/evaluation-prompt-{evaluation_index}",
                    row["prompt_ids"] == expected_ids,
                )
                with torch.inference_mode():
                    base = model(torch.tensor([expected_ids])).logits[0, -1, ids]
                    probabilities = torch.softmax(
                        base + active_state["bias"][ids], dim=0
                    )
                check(
                    f"{seed}/{arm}/evaluation-probabilities-{evaluation_index}",
                    torch.allclose(
                        probabilities,
                        torch.tensor(row["probabilities"]),
                        atol=2e-6,
                        rtol=1e-5,
                    ),
                )
                score = float(probabilities[2:].sum())
                check(
                    f"{seed}/{arm}/evaluation-score-{evaluation_index}",
                    abs(score - row["v1_expected_reward"]) < 2e-6,
                )
                evaluated.append(score)
            check(
                f"{seed}/{arm}/evaluation-aggregation",
                abs(sum(evaluated) / len(evaluated) - recorded_eval["mean_v1_reward"])
                < 2e-6,
            )
            processes = read(root / "processes.json")
            expected_kills = 4 if arm == "E" else 0
            check(
                f"{seed}/{arm}/actual-SIGKILL-count",
                sum(x["returncode"] == -9 for x in processes) == expected_kills,
            )
            if arm == "E":
                check(
                    f"{seed}/{arm}/fresh-process-restarts",
                    len(set(x["pid"] for x in processes)) == 6,
                )
                # Six processes: actor initial+restart; learner initial+3 restarts.
            metrics = read(root / "case-summary.json")
            details.append(metrics)
        for other in ("D", "E"):
            check(
                f"{seed}/A-{other}/exact-adapter",
                torch.equal(final["A"]["bias"], final[other]["bias"]),
            )
            check(
                f"{seed}/A-{other}/exact-RNG",
                torch.equal(final["A"]["rng"], final[other]["rng"]),
            )
            oa = list(final["A"]["optimizer"]["state"].values())[0]
            ob = list(final[other]["optimizer"]["state"].values())[0]
            check(
                f"{seed}/A-{other}/exact-Adam",
                all(torch.equal(oa[k], ob[k]) for k in oa),
            )
        divergence = float((final["A"]["bias"] - final["B"]["bias"]).norm())
        details.append(
            {
                "seed": seed,
                "naive_reference_parameter_l2": divergence,
                "naive_reference_evaluation_delta": read(
                    output / f"seed-{seed}/B/evaluation.json"
                )["mean_v1_reward"]
                - read(output / f"seed-{seed}/A/evaluation.json")["mean_v1_reward"],
            }
        )
    return {
        "status": "PASS",
        "checks": checks,
        "checks_passed": len(checks),
        "details": details,
        "scope": "synthetic categorical routing objective reversal; one-token bias-adapter systems experiment",
    }


if __name__ == "__main__":
    arg = argparse.ArgumentParser()
    arg.add_argument("--protocol", type=Path, required=True)
    arg.add_argument("--output", type=Path, required=True)
    arg.add_argument("--receipt", type=Path, required=True)
    a = arg.parse_args()
    if a.receipt.exists() or a.receipt.with_suffix(".failure.json").exists():
        raise FileExistsError("append-only audit receipt already exists")
    try:
        result = audit(a.protocol, a.output)
    except BaseException as exc:
        a.receipt.with_suffix(".failure.json").write_text(
            json.dumps({"type": type(exc).__name__, "message": str(exc)}, indent=2)
        )
        raise
    a.receipt.write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {"status": result["status"], "checks_passed": result["checks_passed"]}
        )
    )
